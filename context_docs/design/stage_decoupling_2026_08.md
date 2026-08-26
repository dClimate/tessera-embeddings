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

**A pool is the failure quorum, not the whole list.** Same shape as the defect above, in
the priming wait. `wait_first` raised only once EVERY future it was given had failed, and
the caller now hands it every live cell — so a doomed cluster would have run wave after wave
through the ingest pool before anything surfaced. The quorum is now `max_parallel`: one
pool's worth of failures with nothing landing aborts the wait, which is exactly the quorum
that held when the caller passed a look-ahead window. The adapter already knew its own pool
width, so this needed no new parameter.

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
