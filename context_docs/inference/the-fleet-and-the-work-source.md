# The fleet and the work source

*Written 2026-09-09, from two days of the global campaign. It explains one thing: the GPU actor
fleet exists for exactly as long as one function call, and every question about when a cluster
holds GPUs and when it releases them follows from that.*

## The fact everything else follows from

`run_inference` (`src/tessera_embeddings/inference/runner.py`) creates the Ray actors near the
top of its body and kills them in its own `finally`. There is no exit from that function which
leaves an actor alive, and nothing above it holds a reference to one. So:

> **A GPU fleet's lifetime is one `run_inference` call.** Two calls means two fleets, built and
> destroyed in sequence, however close together they are.

A cluster's main pass over its cell list is ONE such call. Cells reach it through a `more_work`
callback — the runner's feeder thread prepares a cell and hands its tiles over, the scheduler
polls for more whenever its queue drops to the live actor count. That is why a healthy pass
shows a single `Killing N actors to release resource reservations` line for a whole roster of
zones, and why actors are never re-created between zones.

## What went wrong: a retry through a second call

The in-child retry — "one more go at a failed cell on the cluster that is still standing" — did
not use that source. It looped over the failed cells after the stream had finished and called
`infer_single`, a per-cell path that reaches `run_inference` again. One call per retried cell.

Each of those built a fleet from nothing, and "from nothing" is literal. By the time the retry
pass ran, the main pass's `finally` had killed every actor and retracted the fleet-mix demand,
and the cluster had spent the assembly backlog drain with idle GPU nodes, which the autoscaler
reclaims after `idle_timeout_minutes` (10 for this flow). So each retry cell started with zero
placed GPU slots and had to re-provision under the launch-rate bound.

What that cost, from two clusters:

- **graceful-caracal, 2026-09-09.** Main pass finished 07:05Z, retry line at 07:35Z. Seven
  retried cells, seven `Creating 250 inference actors`, 240 s to the first two usable actors and
  then increments of +3/+4/+6. Actors ever requested per round: 173, then 102, 92, 71 — each
  round smaller than the last.
- **mellow-walrus, same day.** Three 58N cells (2021, 2023, 2024) whose ingests had genuinely
  failed. Each resumed its interrupted store, completed the ingest in 2 to 2.6 hours, reached
  inference, and rebuilt: fleets at 09:03:59Z and 14:11:31Z reaching about 46 placed slots of
  250, then roughly four hours at ~230 tiles/hour against ~1,700 on a healthy fleet. Six to
  seven hours per cell. That cluster spent a whole day on three cells.

### Reading those actor counts

173 / 102 / 92 / 71 are not live fleet sizes and they are not evidence of a capacity shortage.
The kill line prints the length of the pool's actor list, which `add_actors` appends to all run
long and from which retired slots are never removed — so each number is "actors ever requested
during that call". Two ceilings hold every one of them below the 250 asked for:

- **placement headroom.** A request may not exceed the GPU slots the fleet actually holds plus
  `ACTOR_REQUEST_HEADROOM` (25). It tracks reality rather than climbing toward a target. Full
  derivation in `gpu-fleet-launch-throttling.md`.
- **outstanding work.** A request stops once the pool is as large as the work left. A retry cell
  resuming most of its staged tiles caps its own fleet at the tiles it has left to do.

## The fix, and the two properties it turns on

A failed cell now goes to the **back of the feeder's work queue** and is served by the session
that is already running. `infer_single` is deleted, so the runner is handed no way to start
inference twice.

**An ingest failure gets a new ingest immediately and without blocking.** `discard` drops the
memoised failed attempt (`start` is idempotent, so without this the retry re-reads the same
failure), then `start` submits. Neither waits. The adapter's own pool and the fleet-wide ingest
gate decide when it runs, so the retry queues behind the roster's other ingests while inference
and assembly carry on. The feeder is not held either: it skips a cell whose mosaic has not
landed, and only takes this one once its ingest is done or nothing else is left.

This reverses an older decision. The post-stream retry cancelled every queued ingest first,
because its own fresh `start` would otherwise have waited behind the whole roster on a cluster
that was already billing. Cancelling now would discard cells this same run is still going to
fill, so `cancel_unstarted` survives for the crashed-session unwind alone.

**An assembly failure is the one exception**, and it needs no fleet. Its failure is discovered on
the trailing assembly thread, potentially long after the last tile inferred and after the feeder
has been joined, so there is no live stream to re-admit it to. The pass that remains re-runs
`assemble` over the tally the cell already holds — no plan, no prepare, no inference — so it
cannot provision anything however it is edited later. Note that even the OLD path provisioned
nothing for such a cell: every tile was staged, so `run_inference`'s resume scan emptied the
chunk list and it returned before creating an actor.

## The termination question, and why it is a counter

Three things can now put work in the feeder's queue: the feeder itself, the scheduler thread
(a cell whose tiles failed), and — in the version first written — the trailing assembly thread.
The feeder decides when no more inference work can arrive, and the two ways of getting that
wrong are asymmetric:

- **finish early** and a cell about to be re-admitted is dropped silently;
- **finish late** and a GPU cluster stays open with nothing to do.

So the runner keeps `work_queue` (cells to admit, with attempt numbers) and `undecided` (cells
taken from it whose inference outcome is still open), both under one lock, with the invariant
that every take is matched by exactly one settle. The feeder ends when the queue is empty AND
nothing is undecided.

**`undecided` counts inference, not the whole cell, and that distinction cost a debugging
session.** Settling a cell when its `assemble` returned kept the work source unexhausted for the
whole assembly backlog, so the session and its actors stayed up through hours of single-threaded
work that needs no GPU at all. A test caught it: the operator-pause test's source never reached
exhaustion within 200 polls because one assembly was still running. A cell is settled where it is
handed to the assembly queue. That is also the reason assembly failures are not re-admitted —
holding the stream open for them would reintroduce exactly this.

Exhaustion is read from that shared state rather than from the feeder thread having exited, so a
feeder wedged inside a caller-supplied readiness or wait probe cannot hold the session, the
actors and the cluster open with no bound. The price is that a feeder which dies owes the stream
an answer: its `finally` clears the queue, abandons the count, records the abandoned cells as
failures and notifies.

## Winding the fleet down

Retirement used to be gated on "the source is exhausted". But a source answering `[]` is not
exhausted — it has nothing to give *right now*, because its next cell is still ingesting or a
failed cell is being re-ingested. Suppressing retirement for the whole of that wait held a full
GPU fleet idle against a multi-hour ingest, which is the most expensive way this system can wait
for anything. The in-stream retry makes it more common by design.

So: when there is no inference work to be had — exhausted, or merely waiting — the fleet winds
down while ingest and the assembly backlog drain carry on. The session stays alive, so the pool
re-grows through the ordinary batch machinery when work arrives. Nothing is rebuilt: no teardown,
no second `ray up`, no re-created `ProgressTracker`, no fresh `wait_for_actors`, no re-published
fleet demand, and the resume state, accumulated results and per-chunk attempt counts all persist.

### Two things had to be true for that, and one of them was not

**A wound-down pool must be able to come back.** The batch request is bounded by
`placed_actor_slots + headroom - requested`, and retired slots stay in the pool's actor list
forever — the indices are the pool's identity, shared by its pending, reserved, initializing and
instance-id structures, so the list cannot be compacted. Passing every slot ever created as
`requested` left a wound-down pool of 170-odd slots permanently unable to ask for anything: the
wind-down would have been one-way. It reads live slots instead, which also gives the target its
natural meaning — how *wide* the fleet should be, not how many were ever asked for.

**The pool must not empty.** The first version of this let the fleet retire to zero and degated
the dispatch loop's `live_count > 0` clause on the source being active. The reasoning was sound
and checks out against the code: the retired set is added to at exactly one site, nothing
discards from it, a dead actor's slot is *reused* by `replace` rather than removed, and the
systemic-death detector only logs — so zero live slots can only mean a deliberate wind-down,
never a fleet that died. The change was still wrong. An empty pool has nothing to dispatch to,
so the loop kept polling a source it could not serve, and if the fleet could not be
re-provisioned it made no progress and never exited. A mutation run hung instead of failing,
which is how it surfaced.

Retirement keeps a **liveness floor of one actor** while the source is unexhausted instead. Work
that arrives is dispatched at once and the fleet grows from there; `live_count > 0` holds, so the
loop condition needs no knowledge of the wind-down and is left exactly as it was; and there is no
second timeout for anyone to tune. One GPU against the ~170 released is a rounding error.

### Oscillation

No hysteresis, and none is needed. Retirement requires an actor to have been seen idle on a
*previous* call and then to exceed the idle grace period (120 s), so a cell that becomes ready
inside two minutes retires nothing at all; the floor and the outstanding-work check bound it from
below. A spurious retirement costs one instance launch, while not retiring costs GPU-hours for
every minute of the wait, so the asymmetry favours the existing number.

## The gate that caused most of it

Of 71 cell failures across the campaign in ten days — 62 ingest, 7 inference, 2 assembly — **42
were not failures at all.** They were a Prefect `503` on `/api/v2/concurrency_limits/decrement`,
which is the *release* of the fleet-wide ingest concurrency slot, not its acquisition
(acquisition uses `increment-with-lease`). The ingest adapter holds that slot across the whole
ingest, so by the time it is released the child run has already been polled to a terminal state
and popped. The exception was raised after the work had succeeded, described nothing about it,
propagated out of the pool future, and was booked as an `inputs/prepare` failure — sending a cell
with a complete, correctly-marked mosaic to the retry pass, which rebuilt a fleet for it.

`FleetGate.__exit__` now suppresses and logs a release failure. Safe because a slot is leased
rather than owned: measured against the dev server, a slot whose holder was killed without
decrementing came back at 1.22 lease periods, so at the ingest gate's 900 s lease that is one
idle slot for about 15 minutes, temporary rather than cumulative. Residual, against the
production limit of 5: 42 events in ten days at ~15.5 min each is 10.9 slot-hours out of 1,200,
about 0.9% of ingest capacity. The gate's saturation behaviour is to HOLD rather than fail, so
even a burst is a short self-healing ingest pause.

A second gate bug was found in the same place. `gate_is_holding` treated any `422` in the cause
chain as "the limit is too small", which is the campaign's pause lever — but FastAPI answers a
malformed request with the same status, and the wording check that might have separated them sat
behind an `or` after an `isinstance` that matches every Prefect HTTP error. A parameter-validation
`422` therefore parked the gate forever, polling every 30 s and inviting an operator to raise a
limit that was never the problem. The two are told apart by body shape: the concurrency router
raises `HTTPException(422, detail="...")` with `detail` a **string**, while FastAPI's own
rejection carries `detail` as a **list** of per-field objects each with a `loc`. Discriminating on
the wording instead was rejected deliberately — it would make a deliberate pause depend on a
server-side message staying phrased the same way, which is a worse failure than the latent one.

## What the rebuild population actually is

The cells that rebuilt a fleet are the 7 inference failures plus every genuine ingest failure
whose re-ingest succeeded. At least 3 of the 20 genuine ingest failures did succeed — the
mellow-walrus 58N sequence above — so the population over ten days is at least 10.

An earlier reading of those 20 held that they clustered by zone (five consecutive years of 25N,
four of 23N, three each of 58N and 08N) and were therefore deterministic data problems that never
reached inference. **That inference was wrong**, and the zone clustering turned out to be
consistent with interrupted-then-resumed ingests rather than with data defects: an interrupted
mosaic is resumed rather than rebuilt, so a zone whose ingest is interrupted repeatedly looks
exactly like a zone with a data problem until you check whether the retry completed.
