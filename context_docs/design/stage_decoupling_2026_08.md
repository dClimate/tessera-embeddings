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

**The FIRST WAVE is the failure quorum — not the whole list, and not a running count.**
Same shape as the defect above, in the priming wait, and it took two passes to get right.

`wait_first` originally raised only once EVERY future it was given had failed, and this
change hands it every live cell, so a doomed cluster would have run wave after wave before
anything surfaced. **The first fix counted failures cumulatively against `max_parallel`,
which was worse than the bug.** The executor starts a queued cell the moment a worker frees,
so two fast failures from successive waves reach that quorum while a slow first-wave cell is
still running and about to succeed — priming aborts, the caller's `finally` cancels that
ingest, and the cluster makes no progress despite a real mosaic being produced.

The quorum is now the named first wave: the first `max_parallel` of the supplied cells,
which are the ones the pool starts. "Every one of them failed" cannot be true while any of
them might still succeed. It is also exactly the quorum that held when the caller passed a
look-ahead window, and the adapter already knew its own pool width, so it needed no new
parameter.

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

## Declined in review, and what would change our mind

**`ready()` treats a failed ingest as landed.** `_take_next` now scans the whole pending
list, so a failed ingest anywhere in it becomes pickable immediately — where before only a
look-ahead window was reachable. Review asked for readiness to be success-aware.

Declined. `ready()` returning true for a failure is deliberate and documented: the
alternative is blocking forever on a mosaic that will never arrive. A failed pick costs
microseconds — `inputs.wait` re-raises, the cell is recorded and retained for the in-child
retry, the feeder continues — so it does not idle the fleet. And the cap tripping after
`look_ahead + 2` genuine ingest failures is the cap working, not a misfire. Making
readiness success-aware needs a new method on the `CellInputs` protocol to buy an ordering
preference we have no evidence we need.

**What would change our mind:** a run where cells with good mosaics sat unstarted behind
failed ones, or a cap trip on a cluster that had plenty of landed work available. Neither
has been observed.

**A systematic ASSEMBLY failure escapes the retained-mosaic cap.** Found independently by
two reviewers. The mechanism is real: assembly failures are only discovered when `_finalize`
runs them, so if the error manifests SLOWLY, inference outruns the single assembly thread,
the backlog fills with the whole cluster, and every queued cell then fails and retains its
mosaic — `retained_failed` passes `look_ahead + 2` with no admission left to stop it. A
fast-failing assembly error is fine: the finalizer keeps up and the cap trips after about
seven cells.

**Declined, because this is the DESIRED behaviour, not a defect.** If assembly is
systematically broken, inference should run to completion and leave a pile of assembly work.
The owner's ruling, 2026-08-26: *"If assembly systematically fails, inference should just
continue and infer everything, so that we end up with a ton of assembly tasks to do and
nothing else... we would expect to fix assembly, re-run, and wipe the mosaics afterwards. If
we incur storage costs so be it."*

The suggested fix — reserve admission against assemblies that have not yet succeeded — is an
admission bound released by assembly, i.e. the exact coupling this change removes.

**A claim made in review and WITHDRAWN.** The first reply on those threads said the GPU cost
of finishing a doomed run is real and that a circuit breaker draining the prepared queue
would save fleet time. **That is wrong, and the opposite is true.** Staged tiles persist, and
`ZarrWriter.scan_existing_staged_artifacts` treats a staged tile as "do not re-infer" — so
the inference is fully reusable and a re-run assembles without re-inferring anything. A
circuit breaker would forfeit exactly the work worth keeping. There is no open item here.

**The recovery procedure, and its one trap.** Fix assembly, re-run, assemble from staging,
then sweep the mosaics. The trap: `_staging_run_id` hashes `_STAGED_OUTPUT_SOURCES`, which is
`("inference", "config/inference.py")` — the whole `inference` package, and `assembly.py` is
in it. **Fixing assembly therefore changes the staging fingerprint and orphans every staged
tile under the old `run_id`.** The re-run must pass `staging_code_identity` pinned to the
pre-fix value so it resolves the same staging prefix. That parameter exists for this class of
situation (`staging-identity-and-resume.md`).

Note also that assembly reads `mosaic_base` for projected coordinates and CRS
(`assembly.py:339`), so a retained mosaic is REQUIRED for an assembly retry, not incidental.

**Retries jump the ingest queue.** Every pending cell's ingest is submitted up front, so
after the feeder stops — out of cells, or the retained-failure cap tripped — the pool can
still hold most of a cluster queued. The in-child retry re-starts the ingest for a cell whose
INPUTS failed, and that fresh `start` went to the back of the FIFO queue: the retry that
exists to be prompt, on a cluster that is still provisioned and still billing, would have
waited hours.

`cancel_unstarted()` now runs before the retry pass. `Future.cancel()` succeeds only for a
task still in the queue, so it drops queued work and never interrupts a running ingest. The
cancelled cells are unattempted either way and stay pending for the next campaign pass, so
this also saves the ingest compute their abandoned mosaics would have cost. Raised in review
as P2 and approved by the owner.

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
| `test_wait_first_aborts_on_a_pool_of_failures_not_the_whole_list` | `DID NOT RAISE` — it waited on four still-pending cells |
| `test_the_queued_ingests_are_cancelled_before_the_retry_pass` | (with the call removed) the retry queued behind the unattempted ingests |

The last of the review fixes needed a different baseline, because the bug was in the fix
rather than in the original. Restoring the cumulative-count logic makes
`test_wait_first_does_not_abort_while_a_first_wave_cell_can_still_land` fail with
`2 ingest(s) failed with none landing (01N-2024, 03N-2024)` — 03N being the second-wave cell
that should never have counted.

Whole unit suite on the final state: **3,447 passed, 1 skipped**; ruff and mypy clean.

**Not covered:** reaching 60 concurrent is only observable in prod at full width. The
mechanism is the same code path at any divisor, and the dispatch log above already shows the
per-cluster share being computed correctly.
