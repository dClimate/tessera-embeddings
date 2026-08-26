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

**The retained-failure cap is slightly looser, but not blind.** This was the one real
defect the first version of the change introduced, found in review.

A cell's outcome has two parts: RECORDING it (bookkeeping, instant) and ASSEMBLING it
(I/O, hours). `_finalize` did both on the single trailing thread, so a failed tally — which
needs no assembly at all — queued behind the entire assembly backlog before being counted.
With admission unbounded the feeder could then admit the whole cluster before the first
failure registered, leaving the cap blind for hours in exactly the systematic-failure case
it exists for. Measured: with the assembly thread held, **zero** cells reached the cap.

The fix makes the finalizer an ASSEMBLY queue only. `_submit_assembly` accounts a failed
tally on the caller's thread and never queues it. With the assembly thread held, failures
now reach the cap immediately.

What remains: accounting is still asynchronous relative to the feeder — an inference
failure is counted on the scheduler's thread, not the feeder's — so the feeder can admit a
few more cells before the count catches up. That residual is thread-scheduling latency
rather than an unbounded I/O wait, and in production the feeder is paced by `inputs.wait`
on a real ingest anyway. An ingest failure is exact, because it raises on the feeder's own
thread before the loop's next cap check.

**Do not test this by counting admissions.** With instant fakes the feeder outruns the
scheduler and the admitted count ranged 4 to 12 across runs. The stable axis is which
thread does the accounting; the test holds one assembly and asserts failures are counted
meanwhile.

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
| `test_failed_cells_are_counted_while_an_assembly_is_stuck` | 0 cells reached the cap while one assembly was held |

**Not covered:** reaching 60 concurrent is only observable in prod at full width. The
mechanism is the same code path at any divisor, and the dispatch log above already shows the
per-cluster share being computed correctly.
