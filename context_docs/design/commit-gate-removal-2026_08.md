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

**1. What it actually bound was queueing, not risk.** ADR-008 carries a note from 2026-07-30
saying that at the shipped 8 clusters "the limit equals the number of possible committers, so **it
cannot bind at all**". **That note is wrong, and this document supplies the evidence against it:**
the ceiling is `2N`, because a cluster's feeder can commit a terminal plan while its trailing
assembly commits a cell — so 8 clusters reach 16 committers against a limit of 8, and the gate
could queue about half of them.

Correcting it does not weaken the case; it sharpens what was removed. **A queued committer waits
for a commit, and a commit is the 0.5-2.2 s measured above.** So the protection given up is a short
wait on a sub-second-to-seconds operation that happens once or twice per zone-year, inside an
assembly measured in hours.

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

## The decision: remove it

The per-ROI premise this was first argued from is withdrawn above and stays withdrawn. **The
argument that survives is stronger, and it is about COST rather than isolation.**

Grant the reviewers everything: all 120 zone groups share one repo and one branch tip, every commit
re-serialises the repo-global snapshot, and at `max_parallel_clusters=10` the `2N` ceiling is 20
committers. Then read what run 1 measured at that concurrency and far beyond:

| simultaneous committers | rebase retries | commit duration |
|---:|---:|---:|
| 2 | 0.5 | 0.5 s |
| 8 | 3.5 | 1.3 s |
| **16** | 7.5 | **2.2 s** |
| **120** | 58 | **15 s** |

**"Cross-group conflict-freedom held: zero unresolvable conflicts at every N"** — including 120, six
times the campaign's theoretical ceiling.

So N>=16 is a real threshold, and what it thresholds is a **<=2x-serial acceptance criterion**: a
commit taking **2.2 s instead of ~1 s**. That is the whole consequence. There is no correctness
cliff at 16, or anywhere up to 120 — a `RebaseFailedError` needs a genuine chunk conflict, and
commits touch disjoint zone GROUPS even while sharing a branch tip. The same-zone case that could
conflict is serialised elsewhere: the driver partitions zones across clusters, a cluster's trailing
assembly is single-threaded, and `commit_year_attrs` carries its own `tries=8` retry for exactly
that collision.

**Weighed against that, the gate costs more than it saves.** It bounds a slowdown measured in
seconds, and it misled its own operator twice in one night when read as a progress signal (20:35Z).
A control that cannot fail dangerously but can be misread is not free.

One correction to the "collisions are rare" intuition, recorded so nobody rebuilds the estimate
without it: **the commit RATE is far higher than one per assembly.** Terminal cells — all-ocean or
no-optical — commit through `mark_zone_year_empty`, and **72 of the first 78 completions were
terminal**. Collisions are likelier than an assembly-only model suggests. It does not change the
conclusion, because the cost of one is the seconds above, but it is the part of the reasoning most
easily got wrong.

## The threshold, in cluster counts

Run 1 breached its own <=2x-serial acceptance bar at **16 simultaneous committers**. The fleet's
committer ceiling is **2N** in `max_parallel_clusters`, because a cluster's feeder can commit a
terminal plan inside `plan()` while its trailing-assembly thread commits a cell. So:

| `max_parallel_clusters` | committer ceiling (2N) | versus the N>=16 breach |
|---:|---:|---|
| 7 | 14 | below — safe |
| **8** | **16** | **AT the breach** |
| 10 (the default) | 20 | above |
| 16 | 32 | far above |

**So the boundary is `max_parallel_clusters` >= 8, and the safe cap is <= 7.**

**Corrected in place: this section first said "`max_parallel_clusters` above ~12", which is
arithmetically wrong** — 2N reaches 16 at N=8, not N=12, so it would have left every configuration
from 8 to 12 at or above the breach with no gate. Caught in review. It also contradicted the `<= 7`
bound stated in the options above, inside this same document: **I swept the withdrawn premise but
did not re-check the arithmetic that depended on it**, which is exactly the failure mode a sweep is
supposed to prevent.

Two other conditions would also put committers back in contention regardless of N:

* more than one cluster owning the same zone, which would put two committers on one zone group;
* a change making assembly commit per-shard or per-tile rather than once per zone-year.

**The detector is the `COMMIT <secs>: <message>` line in `commit_with_rebase`.** Every commit goes
through that function — the assembly path and the terminal path both — so it is the only place that
sees all of them. `commit_s` in `ASSEMBLY_SUMMARY` was the obvious candidate and is **not
sufficient**: a terminal cell marks itself through `mark_zone_year_empty` and returns without ever
reaching `assemble_global`, so the summary never fires for it. Since terminal cells were 72 of the
first 78 completions, that detector would have missed the dominant source of concurrent commits and
stayed silent exactly when the gate should be restored.

A commit time drifting above a couple of seconds is the signal.

**One line per COMMIT, which is not one per zone-year.** An assembled cell emits **two**:
`write_year_shards` commits the shard data, then `commit_year_attrs` commits the completion attrs.
A terminal cell emits **one**, since it reaches `commit_year_attrs` directly through
`mark_zone_year_empty`. So the line is a reliable signal of commit LATENCY, which is what the reopen
criterion needs, but raw line counts must not be read as cell counts — the ratio moves with how much
of the workload is terminal.

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
