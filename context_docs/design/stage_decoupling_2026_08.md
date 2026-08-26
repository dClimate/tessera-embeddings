# Ingest, inference and assembly run ungated — why, and what it costs

**Status: implemented** (`orchestration/runners/sequential_fill.py`, PR #149, 2026-08-26).
The runner's module docstring states the design; this note holds the measurements, the
accepted costs, and the recovery procedure.

## Why the gates went

The 2026-08 global campaign asked for 60 concurrent ingests and ran **7**, with the
configuration correct all along:

```
chained-clusters: 10 cluster(s) x 6 zone(s) at a time = 60 simultaneous ingest(s),
hard-capped fleet-wide at max_parallel_ingest=60 by 'tessera-global-ingests'
```

7 is far below 60, so the fleet-wide Prefect gate never engaged. Two in-process semaphores
were the cause, and **both released their slot only after a cell's ASSEMBLY landed**:
`_MosaicBudget` admitted ingest *starts*, so ingest throughput was set by assembly
throughput; `zone_slots` admitted cells to the inference stream, so inference stalled once
`look_ahead + 2` cells were awaiting assembly — and that same bound capped how many zones'
tiles were dispatchable, the documented "small-zone fleet fill" limitation.

Both were sized as a storage budget on the assumption that assembly is cheap — the docstring
said 10–15% of inference wall time. **Measured on this campaign's first three assemblies:
~1,380 tiles/hour** (1,369 / 1,354 / 1,424, within 5% across three cells). On dense cells
that is a plausible per-cell critical path, so a gate released by assembly gates the slowest
stage. The storage rationale had lapsed independently: the flow ingests a whole cluster
before requesting GPUs, so peak storage is a cluster's mosaics by design (ADR-011).

## The design now

GPUs are the most expensive resource, so inference waits for nothing but its own input.

| stage | concurrency | paced by |
|---|---|---|
| ingest | `1 + look_ahead` per cluster (6 at campaign shape) | its own pool; every pending cell is started up front, so the next begins as one finishes |
| inference | unbounded admission | `inputs.wait` on the chosen cell's own mosaic |
| assembly | 1 per cluster | the single trailing thread, free to lag past the end of inference |

`look_ahead` now has three explicit jobs and no implicit ones: the ingest pool width
(`1 + look_ahead`), the priming abort quorum (the pool's first wave), and the
retained-failure cap (`look_ahead + 2`). It previously also carried the inference stream
depth, the storage bound and the fleet-fill parallelism.

**Mosaics are deleted on exactly two occasions: after a successful assembly, or by hand.**
No failure path deletes one — ingests are idempotent and a retained mosaic is what lets the
retry resume. The three delete sites are all success branches (`_finalize`'s `else`, a
terminal `plan()`, and a recovered retry); failure paths only record and retain. Pinned by
tests asserting no cleanup on a failing run.

## Accepted costs

**The retained-failure cap is slightly looser.** An inference failure is counted on the
scheduler's thread, not the feeder's, so the feeder can admit a few more cells before the
count catches up. That is thread-scheduling latency, and in production the feeder is paced by
`inputs.wait` on a real ingest anyway. An ingest failure is exact — it raises on the feeder's
own thread. *Do not test this by counting admissions:* with instant fakes the feeder outruns
the scheduler and the count ranged 4 to 12 across runs. The stable axis is which thread does
the accounting.

**The assembly drain can be long**, and it runs inside the caller's Ray context. It is cheap
anyway because the session has retired its actors by then, so GPU workers idle down on
`idle_timeout_minutes` and only the head node remains. Draining outside the Ray context would
need pending assemblies handed back to the flow — a change to the runner's contract, not made.

**A systematic ASSEMBLY failure escapes the retained-mosaic cap.** If the error manifests
slowly, inference outruns the assembly thread and the whole cluster queues, each cell then
failing and retaining its mosaic. This is intended, not a defect: *"If assembly systematically
fails, inference should just continue and infer everything... fix assembly, re-run, and wipe
the mosaics afterwards. If we incur storage costs so be it."* (owner, 2026-08-26). The
inference is not wasted — staged tiles persist and `scan_existing_staged_artifacts` treats a
staged tile as "do not re-infer".

## Recovery from a systematic assembly failure, and its one trap

Fix assembly, re-run, assemble from staging, then sweep the mosaics.

**The trap:** `_staging_run_id` hashes `_STAGED_OUTPUT_SOURCES`, which is
`("inference", "config/inference.py")` — the whole `inference` package, and `assembly.py` is
in it. Fixing assembly therefore **changes the staging fingerprint and orphans every staged
tile**. The re-run must pass `staging_code_identity` pinned to the pre-fix value
(`staging-identity-and-resume.md`). Note also that assembly reads `mosaic_base` for projected
coordinates and CRS, so a retained mosaic is *required* for an assembly retry.

## Declined in review

**`ready()` treats a failed ingest as landed**, so `_take_next` can pick failures ahead of a
landed cell. Deliberate: the alternative is blocking forever on a mosaic that will never
arrive, a failed pick costs microseconds, and a cap trip after N genuine ingest failures is
the cap working. Making readiness success-aware needs a new `CellInputs` method to buy an
ordering preference nothing has demanded. **Reopen if** cells with good mosaics sit unstarted
behind failed ones, or a cap trips on a cluster with plenty of landed work.

## Verification

Every behavioural test fails against the pre-change source and passes against the new one —
run by pointing `PYTHONPATH` at a checkout that still has both semaphores, so no patching was
needed. Fixes made *during* review were checked against the rejected fix instead.

| test | on the source it discriminates against |
|---|---|
| `test_inference_does_not_wait_for_assembly` | `RuntimeError: 8/8 cell(s) failed` — feeder stalled at 2 of 8, the assembly hold timed out |
| `test_ingest_starts_for_every_cell_and_is_not_paced_by_the_fill` | an ingest started only after a cell was finalized |
| `test_the_readiest_cell_is_taken_wherever_it_sits` | took `01N` instead of the landed `05N` |
| `test_systematic_failure_stops_feeder_at_retained_cap` | ingest was gated by admission |
| `test_failed_cells_are_counted_while_an_assembly_is_stuck` | 0 cells reached the cap while one assembly was held |
| `test_wait_first_aborts_on_a_pool_of_failures_not_the_whole_list` | `DID NOT RAISE` — waited on four still-pending cells |
| `test_wait_first_does_not_abort_while_a_first_wave_cell_can_still_land` | aborted on a second-wave failure, cancelling a good ingest |
| `test_the_queued_ingests_are_cancelled_before_the_retry_pass` | the retry queued behind the unattempted ingests |

Whole unit suite: **3,449 passed, 1 skipped**; ruff and mypy clean.

**Not covered:** reaching 60 concurrent is only observable in prod at full width. The
mechanism is the same code path at any divisor, and the dispatch log above already shows the
per-cluster share computed correctly.
