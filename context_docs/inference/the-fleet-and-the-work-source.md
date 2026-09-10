# The fleet and the work source

*2026-09-09. One fact explains the rest: the GPU actor fleet exists for exactly as long as one
function call.*

## The fact everything follows from

`run_inference` creates the Ray actors near the top of its body and kills them in its own
`finally`. No exit leaves an actor alive, and nothing above it holds a reference.

> **A GPU fleet's lifetime is one `run_inference` call.** Two calls means two fleets, built and
> destroyed in sequence, however close together.

A cluster's main pass over its cell list is ONE such call, fed through a `more_work` callback.
That is why a healthy pass logs a single `Killing N actors to release resource reservations` for
a whole roster of zones, and why actors are never re-created between zones.

## What went wrong: a retry through a second call

The in-child retry looped over the failed cells after the stream had finished and called
`infer_single`, a per-cell path reaching `run_inference` again — one call per retried cell. By
then the main pass's `finally` had killed every actor and retracted the fleet demand, and the
cluster had spent the assembly backlog drain with idle GPU nodes, which the autoscaler reclaims
after `idle_timeout_minutes` (10 for this flow). So each retry started from zero placed slots.

- **graceful-caracal.** Main pass finished 07:05Z, retry line 07:35Z. Seven retried cells, seven
  `Creating 250 inference actors`, 240 s to the first two usable actors, then +3/+4/+6. Actors
  ever requested per round: 173, 102, 92, 71.
- **mellow-walrus.** Three 58N cells whose ingests failed, resumed their interrupted stores,
  completed in 2 to 2.6 h, reached inference and rebuilt — fleets at 09:03:59Z and 14:11:31Z
  reaching ~46 placed slots of 250, then ~4 h at ~230 tiles/hour against ~1,700. Six to seven
  hours per cell; a whole day of one cluster on three cells.

**Those actor counts are not a capacity shortage.** The kill line prints the length of the pool's
actor list, which `add_actors` appends to all run long and from which retired slots are never
removed, so each number is "actors ever requested during that call". Two ceilings hold them below
250: placement headroom (a request may not exceed placed GPU slots plus `ACTOR_REQUEST_HEADROOM`,
25 — see `gpu-fleet-launch-throttling.md`) and outstanding work.

## The fix

A failed cell goes to the **back of the feeder's work queue** and is served by the session
already running. `infer_single` is deleted, so nothing can start inference twice.

An ingest failure gets a new ingest dispatched immediately and without blocking: `discard` drops
the memoised failed attempt (`start` is idempotent, so without it the retry re-reads the same
failure), then `start` submits. Neither waits — the adapter's pool and the fleet-wide ingest gate
decide when it runs, so the retry queues behind the roster's other ingests while inference and
assembly carry on, and `_take_next` skips the cell until its mosaic lands.

This reverses an older decision: the post-stream retry cancelled every queued ingest first,
because its own `start` would otherwise have waited behind the whole roster on a billing cluster.
Cancelling now would discard cells this run will still fill, so `cancel_unstarted` survives for
the crashed-session unwind alone.

## Why the fleet is NOT held through the assembly backlog drain

*This is the paragraph that will be re-litigated, so it is stated in full.*

An assembly failure is the one phase not re-admitted to the live stream. The temptation is to
treat it like every other failure and keep the work source unexhausted until every cell is
finished, assembly included. That is wrong, and the reason is arithmetic rather than taste.

Assembly runs single-threaded on one trailing worker, needs no GPU at all, and may lag arbitrarily
far behind inference — a cluster's backlog can be most of its cells and take hours. An unexhausted
work source keeps the GPU fleet standing. So counting a cell as outstanding until its `assemble`
returned held ~170 GPUs idle through the entire backlog: at the campaign's on-demand basis of
\$1.861/GPU-hour that is roughly \$316 for every hour of assembly, buying nothing, because no part
of assembly can use a GPU. The first version of this change did exactly that, and a test caught
it: the operator-pause test's source never reached exhaustion within 200 polls because one
assembly was still running.

The cost of the alternative is nil. A cell whose inference succeeded and whose assembly failed has
every tile staged, so the retry needs no fleet at all — `run_inference`'s resume scan empties the
chunk list and returns before creating an actor. Even the OLD per-cell path provisioned nothing
for such a cell. So the retry is left to a pass after the backlog drains, and that pass re-runs
`assemble` over the tally the cell already holds: no plan, no prepare, no inference, and therefore
no way for a later edit to build a fleet there.

A cell is consequently settled where it is handed to the assembly queue, not where its assembly
returns. `undecided` counts inference, not whole cells.

**That retry has to ask whether the cell already published.** `assemble` commits the year and tags
it before it returns, and it can raise after doing both — a failed deferred-cleanup submission does
exactly that. Retrying straight from the tally then writes a second snapshot, and a zone-year tag
is write-once (icechunk refuses to reuse a tag name, even a deleted one), so the tag will not move:
every remaining attempt fails and a cell that is published and correct is reported as failed, its
staging possibly already deleted. The retry therefore re-runs `plan` first, which is the check the
deleted per-cell path made before every assembly: a complete year comes back terminal with nothing
to infer or assemble, and it tags the tip first if the failure landed between commit and tag.

## The termination question

Two threads add to the feeder's queue — the feeder itself and the scheduler thread — and the
feeder decides when no more inference work can arrive. The failure modes are asymmetric: finish
early and a cell about to be re-admitted is dropped silently; finish late and a GPU cluster stays
open with nothing to do. Hence `work_queue` and `undecided` under one lock, with the invariant
that every take is matched by exactly one settle, and the feeder ending only when the queue is
empty and nothing is undecided.

Exhaustion is read from that shared state rather than from the feeder thread having exited, so a
feeder wedged inside a caller-supplied probe cannot hold the session, the actors and the cluster
open with no bound. A feeder that dies therefore owes the stream an answer: it clears the queue,
abandons the count, records the abandoned cells as failures and notifies.

## Winding the fleet down

Retirement used to be gated on "the source is exhausted". A source answering `[]` is not
exhausted — it has nothing to give right now, because its next cell is still ingesting or a failed
cell is being re-ingested — so a cluster waiting on an ingest held a full idle fleet for the whole
wait. Now, when there is no inference work to be had, the fleet winds down while ingest and the
assembly backlog carry on. The session stays alive, so the pool re-grows through the ordinary
batch machinery: no teardown, no second `ray up`, no re-created `ProgressTracker`, no fresh
`wait_for_actors`, and the resume state, results and attempt counts all persist.

Two things had to be true, and one was not.

- **A wound-down pool must come back.** A request is bounded by
  `placed_actor_slots + headroom - requested`, and retired slots stay in the pool's actor list
  forever (the indices are the pool's identity, shared by its pending, reserved, initializing and
  instance-id structures, so the list cannot be compacted). Counting every slot ever created left
  a wound-down pool of 170-odd slots permanently unable to ask for anything. It reads live slots.
- **The pool must not empty.** The first version let it retire to zero and degated the dispatch
  loop's `live_count > 0` clause on the source being active. The reasoning checked out — the
  retired set is added to at one site, nothing discards from it, `replace` reuses a dead actor's
  slot, and the systemic-death detector only logs, so zero live slots can only mean a deliberate
  wind-down. It was still wrong: an empty pool has nothing to dispatch to, so the loop kept
  polling a source it could not serve and never exited if capacity never arrived. A mutation run
  hung rather than failed, which is how it surfaced. Retirement keeps a **liveness floor of one
  ready actor** instead, so the loop condition needs no knowledge of the wind-down.

No hysteresis. Retirement requires an actor to have been seen idle on a *previous* call and then
to exceed the 120 s idle grace, so a cell that becomes ready inside two minutes retires nothing.

### Retiring a slot is not terminating its machine

`_retire_slot` fires the EC2 terminator only when the slot it just killed was the last live one on
its instance. That matters only under `config.num_gpus < 1`, where several actors share a GPU
machine; the campaign's whole-GPU actors are alone on theirs, so the first retirement is also the
last and the check is a no-op.

**The subtlety is that co-residency is decided by instance ID, and a slot that is still placing
does not have one.** An initializing actor carries the `pending-init` placeholder until its
constructor and its `get_instance_id` round-trip finish, and a placeholder never equals an `i-` ID
— so the comparison on its own would read a booting sibling's machine as empty and terminate it
mid-boot. Under packing, the pending IDs are resolved first (non-blocking, so an actor still in
`__init__` stays unresolved), and any slot left unresolved defers termination: co-residency cannot
be *disproved*, and the two errors do not cost the same. Deferring leaves a machine up until the
fleet's own teardown reclaims it. Terminating kills a live actor, and if that actor was the
liveness floor, leaves the session holding a handle to nothing.

### Available means deliverable

The wind-down is suppressed while the source says inference work is available *right now*, which
exists so that an operator pause — work withheld, not absent — does not read as a drought. The
predicate counted a prepared zone, and also a still-queued cell whose mosaic had landed, on the
argument that the feeder plans such a cell within seconds.

That argument assumes the feeder is free to act, and it is one thread. When nothing has landed the
feeder is handed the queue *head* and blocks on that cell's ingest, which at the opening of a
cluster's window is 4-10 h. Another cell landing during that block is work the feeder cannot
reach until the head returns, so the predicate answered "available" while the queue was empty and
the whole fleet was billed idle for the head's remaining ingest — the exact outcome the wind-down
was built to prevent. The feeder now publishes whether it is parked, and a queued cell counts only
when it is not. When the park ends the feeder enqueues and the pool re-grows.

**The first version of that flag covered the ingest wait alone, and that was not enough.** The
feeder parks in other adapter calls for as long or nearly so: `cleanup` deletes a multi-terabyte
mosaic, and `discard` — the re-ingest of a failed cell — has to confirm the previous attempt is
dead before a replacement starts. Naming the calls one at a time is only ever as complete as the
list someone thought of, so the flag is no longer set at call sites at all. Every call the runner
makes into `CellInputs` goes through one wrapper (`_FeederInputs`), which publishes the park for
the duration of any method, and a method added to the protocol later cannot be forwarded unmarked
— the wrapper is what the runner passes as its `CellInputs`, so the type checker refuses it until
the new method is wrapped too. Only the feeder's own calls count; the same adapter is called from
the scheduler thread, the cleanup pool and the main thread, and none of those stop the feeder
admitting work.

One of those calls should not have been on the feeder at all. A cell that plans out TERMINAL
(already complete, all ocean) is committed and tagged inside `plan`, so deleting its mosaic is
housekeeping on work that has already landed — and it is now handed to the cleanup pool exactly as
a landed cell's delete is. Run inline it stopped the rest of the roster being admitted for the
length of the delete.

What the flag deliberately does not cover is the feeder's own bounded work on the cell it holds:
`prepare`, `plan`, and the staged-resume scan. Those are seconds and they end with that cell's
tiles on the hand-over queue, where the predicate sees them anyway. The 120 s idle grace is what
absorbs a brief answer of "nothing right now"; the predicate's job is to be honest, not brief.

### A wind-down needs a way back

Retirement mid-run is only safe where the pool can grow back, and `_maybe_request_next_batch` is
the whole of what recreates a retired slot. `actor_request_batch_size = 0` is a documented mode
meaning "request the whole fleet at once", and it disables batching — so one ingest gap would have
cut the fleet to the liveness floor for the rest of the run, with every later cell inferred on a
single actor. The mid-run wind-down is now skipped outright in that mode rather than taught to
regrow in it: refusing a bad combination is smaller than supporting one. The TAIL wind-down is
unaffected, since an exhausted source has nothing to grow back for. The campaign passes 25 and the
default is 50, so this was never the configuration we run.

### Slots that never placed

`retire_idle` cannot see a slot that is still initializing — dispatch skips it, so it is never
observed idle — but such a slot is not free. A Ray actor waiting on placement holds a GPU resource
request the autoscaler forwards to AWS, and it counts in the pool's `requested`, which is what the
fleet publisher advertises. Through an ingest drought that kept asking for machines the run had no
work for, and if the capacity arrived the instances booted, loaded the checkpoint and billed a
whole idle grace before the wind-down could reach them. `retire_initializing` cancels them after
the same grace, on its own clock, and never while any work is outstanding or while no READY actor
is standing — an unplaced slot is otherwise the session's only route to a live actor, and
cancelling the last one leaves a live source with nothing that can ever run its chunks.

### One instance, several actors

Retirement fires the EC2 terminator per retired ACTOR, which is right only while an actor is alone
on its instance. Under `num_gpus < 1` the scheduler packs several onto one, and terminating on the
first retirement kills a machine still hosting a busy sibling — or the one actor the liveness floor
just kept, leaving the session believing it holds a live slot whose handle is dead. The actor is
still killed; the instance is now terminated only when no live slot is left on it, so the last
actor off a packed instance releases it and whole-GPU fleets (the default, and what the campaign
runs) behave exactly as before.

### The fleet the thresholds are judged against

Two systemic-failure guards were still scaled by the length of the pool's actor list: the death
count that reports a fleet as dead, and the number of simultaneously-stalled chunks that *aborts*
the run. Retired slots never leave that list, so it is a history of every generation the run has
held — a 40-actor fleet reads 79 entries after one drought and 118 after two. The death threshold
is log-only, but the stall threshold aborts, and taking the larger of the list and the target made
it strictly less sensitive with every cycle: 11 simultaneous stalls demanded of a 40-actor fleet
instead of 4.

The live count is not the answer either, because it collapses to the wind-down floor and would
make the first death after a drought read as the whole fleet dying. Both now read one
`ActorPool.fleet_size`, which is the run's actor *target* — the only fleet quantity stable across
a session's lifetime — falling back to the slot count for a caller that supplies its whole fleet
up front and never regrows.

### A refusal has to stop the ingests it refuses

The flow submits an ingest for every live cell before the runner is entered, so the retained-failure
cap's "left unattempted" cells still have queued ingests. Emptying the work queue does not stop
them: the adapter would work through the rest of the roster and write every one of those
multi-terabyte mosaics off-budget, for cells the run has said it will not attempt, while inference
and the assembly drain carry on for hours. The cap now names those cells to
`CellInputs.cancel_unstarted`. Only the never-attempted ones, and only the ones not yet started — a
running ingest is inside the concurrency the campaign already budgeted, and cancelling it needs a
confirmation wait the feeder cannot afford while it still owes an answer to every streaming cell.

## The gate that caused most of it

Of 71 cell failures in ten days — 62 ingest, 7 inference, 2 assembly — **42 were not failures.**
They were a Prefect `503` on `/api/v2/concurrency_limits/decrement`, the *release* of the ingest
concurrency slot rather than its acquisition (which uses `increment-with-lease`). The adapter holds
that slot across the whole ingest, so by release time the child run has already been polled to a
terminal state and popped. The exception was raised after the work succeeded and was booked as an
`inputs/prepare` failure, sending a cell with a complete mosaic to the fleet-rebuilding retry pass.

`FleetGate.__exit__` now suppresses and logs a release failure. Safe because a slot is leased: a
holder killed without decrementing had its slot back at 1.22 lease periods on dev, so at the
900 s ingest lease that is one idle slot for ~15 minutes, temporary rather than cumulative.
Residual against the production limit of 5: 10.9 slot-hours out of 1,200, about 0.9%.

A second gate bug sat in the same place. `gate_is_holding` treated any `422` as "the limit is too
small" — the campaign's pause lever — but FastAPI answers a malformed request with the same status,
and the wording check that might have separated them sat behind an `or` after an `isinstance`
matching every Prefect HTTP error, so a parameter-validation `422` parked the gate forever. The two
are told apart by body shape: the router's own `422` carries `detail` as a string, FastAPI's as a
list of per-field objects with `loc`. Not by wording — that would make a deliberate pause brittle
to upstream rephrasing.

## The rebuild population

The cells that rebuilt a fleet are the 7 inference failures plus every genuine ingest failure whose
re-ingest succeeded. At least 3 of the 20 did (mellow-walrus 58N above), so the population over ten
days is at least 10.

An earlier reading held that those 20 clustered by zone (five consecutive years of 25N, four of
23N, three each of 58N and 08N) and were therefore deterministic data problems that never reached
inference. **That was wrong.** The clustering is equally consistent with interrupted-then-resumed
ingests: an interrupted mosaic is resumed rather than rebuilt, so a zone interrupted repeatedly
looks exactly like a zone with a data problem until you check whether the retry completed.
