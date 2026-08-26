# Ingest, inference and assembly run ungated — why, and what it costs

**Status: implemented** (`orchestration/runners/sequential_fill.py`, PR #149, 2026-08-26).
This is the record behind that change. The runner's module docstring is the canonical
statement of the *current* design; this document holds the measurements and the history,
which do not belong in source.

## The incident

The 2026-08 global campaign asked for 60 concurrent ingests and ran **7**. Its own dispatch
log shows the configuration was never wrong:

```
chained-clusters: 10 cluster(s) x 6 zone(s) at a time = 60 simultaneous ingest(s),
hard-capped fleet-wide at max_parallel_ingest=60 by 'tessera-global-ingests'
```

7 is far below 60, so the fleet-wide Prefect gate never engaged. The cause was two
in-process semaphores in the runner, and **both released their slot only after a cell's
ASSEMBLY landed**:

| gate | capacity | what it admitted | consequence |
|---|---|---|---|
| `_MosaicBudget` | `look_ahead + 2` | ingest *starts* | a new ingest waited on an assembly several cells back, so ingest throughput was set by assembly throughput |
| `zone_slots` | `look_ahead + 2` | cells into the inference stream | inference stalled once that many cells awaited assembly; the same bound also capped how many zones' tiles were dispatchable at once |

The second bound was documented in the runner as the "small-zone fleet fill" KNOWN
LIMITATION — a cluster of zones each far smaller than the fleet could not fill it. Both
gates therefore idled the GPU fleet, which is the most expensive resource in the stack.

## Why the gates bound on the wrong stage

Both were sized as a STORAGE budget, on the assumption that assembly is a cheap trailing
task. The runner's docstring put it at **10–15% of a cell's inference wall time**.

Measured on this campaign's first three assemblies: **~1,380 tiles/hour**, specifically
1,369 / 1,354 / 1,424 — within 5% of each other across three different cells. On dense
cells that is a far larger fraction of inference wall time than assumed, which makes
assembly a plausible per-cell critical path rather than a trailing detail. A gate released
by assembly is therefore a gate on the slowest stage.

The storage rationale had already lapsed independently: the flow ingests a whole cluster
before requesting GPUs, so peak storage is a cluster's mosaics by design (ADR-011).

## The design now

Nothing gates the three stages against each other. The ordering principle is that
inference never waits for anything but its own input.

| stage | concurrency | paced by |
|---|---|---|
| ingest | `1 + look_ahead` per cluster (6 at the campaign's shape) | its own thread pool; an ingest is started for every pending cell up front, so the next begins the moment one finishes |
| inference | unbounded admission | `inputs.wait` on the chosen cell's own mosaic |
| assembly | 1 per cluster | the single trailing thread |

`look_ahead` now sizes ingest width only. Assembly may lag arbitrarily behind inference,
including past the end of it.

## Two accepted costs

**The retained-failure cap is looser.** It still stops admission deterministically for
failures the feeder sees itself — an ingest failure raises inside `inputs.wait` on the
feeder's own thread and is counted before the loop's next cap check. An *inference* failure
is recorded on the trailing finalizer thread, so the feeder can admit further cells before
the count catches up. In production that run-ahead is bounded by ingest rather than by a
gate: `inputs.wait` blocks on a real ingest, only `1 + look_ahead` run at a time, and each
takes tens of minutes to hours. Two tests name which half is which.

**The assembly drain at the end can be long.** It runs inside the caller's Ray context, but
the session has already retired its actors by then, so the cluster's GPU workers idle down
on `idle_timeout_minutes` and only the head node remains. Moving the drain outside the Ray
context would remove even that, at the cost of handing pending assemblies back to the flow —
a change to the runner's contract, deliberately not made. **If the drain proves to be the
campaign's critical path, that is the change to make**, and the measurement to take first is
assembly wall time against inference wall time on the same cell.

## Verification

Every behavioural test fails against the pre-change source and passes against the new one.
Run by pointing `PYTHONPATH` at a checkout that still has both semaphores — no patching
needed, which is what made this cheap:

| test | on pre-change source |
|---|---|
| `test_inference_does_not_wait_for_assembly` | `RuntimeError: 8/8 cell(s) failed` — the feeder stalled at 2 of 8, the assembly hold timed out |
| `test_ingest_starts_for_every_cell_and_is_not_paced_by_the_fill` | an ingest started only after a cell was finalized |
| `test_the_readiest_cell_is_taken_wherever_it_sits` | took `01N` instead of the landed `05N` |
| `test_systematic_failure_stops_feeder_at_retained_cap` | ingest was gated by admission |
| `test_inference_failures_are_paced_by_ingest_not_by_the_cap` | admitted 6 of 12 |

**Not covered:** reaching 60 concurrent is only observable in prod at full width. The
mechanism is the same code path at any divisor, and the dispatch log above already shows the
per-cluster share being computed correctly.
