# Removing the fleet-wide commit gate

Decision record, 2026-08-27. **The `tessera-global-commits` Prefect global concurrency limit and
all of its plumbing are removed.** Commits to the global embeddings store are now ungated.

Decided by Robert on the reasoning that the gate guards a minuscule possibility, that if it does
occur the slowdown is manageable and easily overcome, and that an extra gate is complexity the
campaign does not need. This document records the measurement that supports it, the evidence the
gate was built on (not retracted), and the scale at which that evidence would matter again.

## What the gate was

A Prefect global concurrency limit, one slot held around `session.commit`, forwarded from
`run_global_campaign` to every fill as `commit_limit_name` and upserted at preflight to
`min(max_parallel_clusters, MAX_SIMULTANEOUS_COMMITTERS=8)`. It existed because ADR-008 run 1
measured rebase retries scaling with the number of racing writers — commit contention is on the
branch-tip compare-and-swap, so every commit re-serialises the repo-global snapshot.

**The run-1 curve, kept because it is the reopen criterion:**

| simultaneous committers | rebase retries | commit duration |
|---:|---:|---:|
| 2 | ~0.5 | 0.5 s |
| 8 | ~3.5 | 1.3 s |
| **16** | **~7.5** | **2.2 s** — breached that experiment's own ≤2×-serial acceptance bar |
| 120 | ~58 | 15 s |

Cross-group conflict-freedom held at every N: **zero unresolvable conflicts.**

## Why it goes

**1. It could not bind, and that was already written down.** ADR-008 carries a note from
2026-07-30: the gate wraps `session.commit` only, so at the shipped 8 clusters "the limit equals
the number of possible committers, so **it cannot bind at all**". It was kept then for
configurations we do not run. That is the definition of speculative complexity in a live system.

**2. Measured live, 2026-08-27.** One completed assembly (`steady-otter`, 19:03:20Z — the
publication that took the store from 77 to 78 zone-years):

| field | seconds |
|---|---:|
| `commit_s` | **1.0** |
| `attrs_commit_s` | **0.3** |
| `merge_s` | 0.0 |
| `fill_wall_s` (fork-to-merge) | **21,447** (5.96 h) |
| `total_s` | 21,455 |

The commit is **1.3 s out of a six-hour assembly** — 0.006% of it. And because `commit_s` is
measured *around* the gated call, **the gate wait is inside that 1.0 s**, which makes it direct
evidence of zero queueing rather than an inference.

**3. ~~There is structurally almost nothing to contend for.~~ WITHDRAWN 2026-08-27 — this was
false, and it was the load-bearing half of the argument.**

I wrote that the embeddings store is "per ROI — `{bucket}/embeddings/{roi_name}.zarr` — so
concurrent commits from different zones land in different Icechunk repositories and cannot share a
branch tip at all." **That describes `BucketPaths.store_for()`, the SINGLE-ROI path. The global
campaign uses `BucketPaths.global_store()`, whose own docstring says the opposite:** "the **single**
global-embeddings Icechunk repo… all 120 UTM-zone groups into one repo (ADR-008 D5), addressed by
zone group name — **unlike `store_for`**, which is one `.zarr` per (roi, kind)". The docstring
contrasts the two functions explicitly. I read the wrong one and did not check.

Two reviewers found this independently, which is the correct outcome and the reason it is recorded
here rather than quietly patched.

**What is actually true.** All 120 zone groups live in ONE repo on ONE `main` branch tip. Commits
touch **disjoint groups**, so there are no DATA conflicts — that part of the original design note
holds and run 1 measured zero unresolvable conflicts at every N. But every commit re-serialises the
repo-global snapshot through a branch-tip compare-and-swap, so **rebase retries scale with the
number of racing writers regardless of how disjoint their data is.** Disjoint groups and a shared
CAS are different things, and conflating them is the whole error.

**The exposure at the configured width.** `max_parallel_clusters` defaults to **10**, and each
chained cluster can have a feeder committing a terminal plan (`mark_zone_year_empty`) while its
trailing-assembly thread commits a cell — so the ceiling is roughly **2N = 20 concurrent
committers**. Run 1 breached its own <=2x-serial acceptance criterion at **N>=16**. The default
configuration's ceiling therefore sits ABOVE the measured threshold, and supported wider
configurations sit far above it.

The registry point stands and is unaffected: one Parquet part per cell, write-once objects rather
than a shared mutable dataset.

**4. The observability cost was real.** The gate's `active_slots` was read as a progress signal on
the night of 2026-08-27 and gave a wrong answer twice — first as "four cells stuck in hours-long
commits", then as evidence about which cluster was assembling. Both were wrong, because a commit
lasts ~1 s while only ONE cluster had an assembly in flight (verified from per-worker shard counts
in CloudWatch, and from all ten clusters' inference progress).

**Corrected in place: I first attributed this to a lagging counter**, citing the sibling ingest gate
as reading "51 held against 45 actually live". That comparison was wrong — it used a monitoring
query with the wrong flow name. Queried correctly, the ingest gate read **56 held against 56
genuinely RUNNING cells**, i.e. exactly right. So the gates' counters are accurate, and what the
commit gate's `active=4` was holding is **unexplained**, not stale: with one assembly in flight, the
candidates are Prefect lease-held slots from a holder that died without releasing, or the terminal
cells a feeder commits through `mark_zone_year_empty`. Either way the point stands for removal — a
1 s commit with zero measured queueing cannot be bound by a limit — but the "counter lags" reason is
withdrawn and should not be repeated.

## Where this leaves the change — NOT MERGEABLE AS IT STANDS

**The measurement survives; the argument built on it does not.** `commit_s` of 1.0 s with the gate
wait measured inside it is a real observation of **zero queueing** — but it was taken with exactly
**ONE assembly in flight fleet-wide** (verified from per-worker shard counts and from all ten
clusters' inference progress). That is a measurement at LOW concurrency. It says nothing about the
N=16-20 case the gate exists for, and I presented it as though it did.

So the honest position: **removing the gate is not justified by anything measured here.** Three ways
forward, for a human to choose:

1. **Keep the gate.** Costs a ~1 s slot acquisition per commit, measured, with zero observed queueing
   at current load. This is the conservative option and the evidence does not argue against it.
2. **Keep a bound but simplify it.** The gate's real defect was never that it was a bound; it was
   that its value derived from `min(max_parallel_clusters, 8)` and its `active_slots` counter misled
   me twice as a progress signal. A fixed bound with no telemetry pretensions addresses both.
3. **Remove it AND cap campaign width** so that `2 * max_parallel_clusters < 16`, i.e.
   `max_parallel_clusters <= 7`. That trades committer safety for cluster count, which is a
   throughput decision, not a correctness one.

`icechunk`'s own rebase loop (`ConflictDetector`, `rebase_tries=1000`) remains the backstop under
every option, and a genuine chunk conflict still raises `RebaseFailedError` rather than being
masked — so the failure mode of getting this wrong is a slower commit, not a torn one.

## When to put it back

Reopen if simultaneous committers can approach **16**, since that is where run 1 breached its own
acceptance bar. Concretely, that means any of:

* `max_parallel_clusters` above ~12 (the fleet's true committer ceiling is ~2N, because a cluster's
  feeder can also commit a terminal plan inside `plan()`);
* more than one cluster owning the same zone, which would put two committers on one store;
* a change making assembly commit per-shard or per-tile rather than once per zone-year.

The cheap detector is already in the telemetry: **`commit_s` in `ASSEMBLY_SUMMARY`.** It includes
gate wait by construction, so a commit time drifting above a couple of seconds is the signal, and it
needs no new instrumentation.

## What was removed

* `MAX_SIMULTANEOUS_COMMITTERS`, `commit_limit_name` (campaign, `fill_zone_year`,
  `fill_zones_sequential`), and the preflight `_upsert_limit` call for it.
* `_PrefectCommitGate`, and the `CommitGate` type alias.
* The `gate` parameter through `write_year_shards`, `commit_year_attrs`, `commit_with_rebase`,
  `assemble_global`, `mark_zone_year_empty` and the four `zone_fill` entry points.
* `DEFAULT_COMMIT_CAP` (the in-process semaphore default), which had no remaining users.

**`FleetGate` itself stays.** The ingest gate is the campaign's pause lever and uses the same class;
only the commit gate's subclass and wiring are gone. The two thread-safety tests written against
`_PrefectCommitGate` moved to `tests/unit/test_fleet_gate.py` and now target `FleetGate` directly,
because the per-thread context stack they exercise is still load-bearing for ingest.

## Cost of being wrong

Bounded and visible. If commits do contend, they rebase — `rebase_tries=1000` — and the cost shows
up as a longer `commit_s` against a six-hour fill. The failure mode is a slower commit, not a lost
or torn one: cross-group conflict-freedom was measured at every N up to 120, and a genuine chunk
conflict raises rather than corrupting.
