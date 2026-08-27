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

**3. There is structurally almost nothing to contend for.** The embeddings store is **per ROI** —
`{bucket}/embeddings/{roi_name}.zarr` — so concurrent commits from different zones land in
different Icechunk repositories and cannot share a branch tip at all. The registry is **one Parquet
part per cell**, write-once objects rather than a shared mutable dataset. The residual same-store
case is same-zone/different-year, and that is serialised by other means: the campaign driver
partitions zones across clusters, and within a cluster the trailing assembly is a single thread.

**4. The observability cost was real.** The gate's `active_slots` was read as a progress signal on
the night of 2026-08-27 and gave a wrong answer twice — first as "four cells stuck in hours-long
commits", then as evidence about which cluster was assembling. Both were wrong, because a commit
lasts ~1 s and the counter lags (the sibling ingest gate read 51 held against 45 actually live). A
control that cannot bind but can still mislead is worse than no control.

## What replaces it

Nothing. Commits proceed directly, with `icechunk`'s own rebase loop
(`ConflictDetector`, `rebase_tries=1000`) as the only protection — which is what actually resolved
the run-1 contention. A real chunk conflict still surfaces as `RebaseFailedError` rather than being
masked.

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
