# Replacing a settled fill without waiting for the dispatch round

**Date:** 2026-08-25
**Status:** implemented, behind `immediate_refill` (default off)
**Scope:** `src/tessera_embeddings/orchestration/prefect/flows/run_global_campaign.py`,
`src/tessera_embeddings/orchestration/prefect/flows/_live_claims.py`,
`tests/unit/test_run_global_campaign.py`, `tests/unit/test_live_claims.py`

Read this before changing anything about the driver's re-dispatch loop, and especially
before widening which terminal states the immediate path accepts.

## What happened

The global campaign driver dispatched ten chained fills, each owning a roster of cells.
One died with:

```
Only 0 / 50 actors initialized within 600s (minimum required: 1)
```

The GPU instance type was exhausted across every availability zone of the region, so the
fill could not get a single actor. It was carrying **81 cells across 9 zones**, had reached
only the first, and aborted. Nothing about the fill was wrong; it was told to stop waiting
while AWS was still refusing.

Two separate defects fell out of it. The first — waiting too short — was fixed separately
by raising the first-actor wait, from 600 s to 1800 s in PR #140 and then to 21600 s (six
hours) in PR #142. This document is about the second.

**Recovery was not prompt.** The driver's re-dispatch loop is a barrier: it gathers every
subflow's outcome before re-reading the store. With `overlap_years=true` there is a single
round for the whole campaign, so those 81 cells waited for the other nine fills to finish
— days, not minutes.

### How this composes with the longer first-actor wait

Two changes now address the same failure from opposite ends, so it is worth recording that
they were meant to coexist and that neither retires the other.

`ACTOR_INIT_TIMEOUT_SEC` (`inference/lifecycle.py`) governs how long a fill waits for its
first GPU actor. Raising it to six hours means a fill in a real drought **stops dying** —
it outlasts the drought instead. `immediate_refill` governs what a death **costs** when one
happens anyway, for a drought longer than six hours or for any of the reasons that have
nothing to do with capacity.

They also compose in a second, less obvious direction: **the longer wait makes the barrier
more expensive, not less.** A dispatch round closes only when every fill returns, so a fill
sitting in a six-hour actor wait now holds its round open for up to six hours. That wait is
precisely what `immediate_refill` takes off the critical path of any OTHER fill in the round
that has already died. The timeout constant's own docstring names the same coupling from the
other side — an unbounded wait "would also stop the driver's dispatch round from ever
closing, since a round closes only when every fill returns, which is the very recovery a
starved fill needs."

So: the wait decides how often a roster is stranded; this decides how long a stranded roster
waits. Removing either one puts back a distinct cost.

## The reframing that shaped the fix

The loss was originally described as "the fill took its ingest with it." It is worth being
precise, because the precise version is much better news.

**The ingest BYTES survive.** Checked against the code rather than assumed:

| mechanism | file | effect |
|---|---|---|
| every failing path retains its cell's mosaic | `orchestration/runners/sequential_fill.py::_retain_failed_mosaic` | nothing is deleted on failure |
| `cleanup` only runs for a cell that LANDED | `orchestration/prefect/flows/fill_zones_sequential.py::_DeploymentCellInputs.cleanup` | a failed cell's mosaic is not swept |
| `sweep_orphan_mosaics` deletes only cells that are complete AND tagged | `run_global_campaign.py` | cannot reach a failed cell |
| an interrupted mosaic is RESUMED date by date | `ingest_zone_year`, and `context_docs/crash-recovery/orphaned-fleet-teardown.md` | the retry continues rather than rebuilding |

So a dying fill costs the **wall clock of the wait**, not the data. That is both the
strongest argument for shortening the wait and the strongest reassurance about doing it:
the change recovers the expensive loss without loosening any teardown rule.

## The hard part: never two writers on one zone

Nothing locks a zone. The partition does: the driver partitions by ZONE, so a zone's every
year lands in one cluster and its assemblies serialise on that cluster's single trailing
thread. An immediate retry that dispatches a failed fill's cells while anything else might
still be working them breaks that guarantee, and the guarantee has no other enforcement.

### Why the store is not the answer

Checked rather than assumed, because "the commits are atomic" is the tempting shortcut:

- The campaign store commits through `storage/shard_writer.py::commit_with_rebase` with a
  bare `icechunk.ConflictDetector()` — **no** chunk-conflict solver — so a genuine
  same-chunk race raises `RebaseFailedError` rather than merging.
- Same-zone, different-year attribute collisions are handled by `commit_year_attrs`, which
  commits attributes separately and, on conflict, re-reads and re-applies. That is
  order-independent because each writer only ever inserts its own year's key.
- The mosaic store commits with **no** `rebase_with` and never retries `ConflictError`
  (`storage/zarr_store.py::CONCURRENT_WRITER_ERRORS`) — deliberately, so a second writer
  fails instead of interleaving dates on one axis.

Observed in the field: `ingest_optimization_campaign_2026_07.md` §4.16 records 47S/2021
dispatched four times, generation 4 landing 17 minutes into a healthy generation 3, and
concludes **"the guard is a consistency guard, not an admission control."** It finds out at
commit time, after both fleets have been paid for.

Net: a two-writer race costs a fleet and a cell, not silent corruption. That makes a
mistake affordable. **It is not a condition, and it must not drift into the safety
argument.**

### The two conditions that are the answer

**Condition one — the predecessor's terminal state proves its own writers stopped.**

This is a chain of `finally` blocks, not an inference from a state name. Tracing the
incident's exception: `wait_for_actors` (`inference/lifecycle.py`) raises inside
`run_inference`, inside `_session`, inside `fill_zones_sequential`, inside the flow's
`with ray_cluster(...)`. On the way out:

1. `orchestration/runners/sequential_fill.py`, `finally`: `feeder.join(...)` then
   `finalizer.shutdown(wait=True)`. The trailing assembly thread is the ONLY thing that
   commits to the embeddings store, and it is joined here.
2. `fill_zones_sequential.py`, `finally`: `inputs.shutdown()` — which requests cancellation
   of every in-flight child ingest and **waits up to `_CANCEL_CONFIRM_S` (300 s) for each to
   confirm terminal** — then `deactivate()`; and `ray_cluster.__exit__` tears the fleet down.

All of that completes before the exception leaves the flow function, and therefore before
the state becomes `FAILED`. `CRASHED` carries none of it: the process that would run the
`finally` is the process that died, and `orphaned-fleet-teardown.md` records that a crash
verdict can be reached from missed heartbeats (300 s timeout against a 30 s cadence) while
the run is still writing. `CANCELLED` is a request that has been acted on, but its
descendants are swept asynchronously and without waiting.

So `_QUIESCENT_TERMINAL_STATES = {FAILED, COMPLETED}`, and nothing else.

**Condition two — enough time has passed for its descendants to have stopped.**

Condition one covers the fill's own writers and, because `inputs.shutdown()` cancels its
direct children and waits up to `CANCELLATION_CONFIRM_S` for each to confirm terminal before
the state is set, its direct children too. What nothing waited for is *their* grandchildren —
the S1/S2 ROI ingests — whose cancellation is requested by an `on_cancellation` hook that does
not block.

So the driver waits `_SETTLE_DELAY_S` before dispatching a replacement. **The figure is
derived, not chosen:** it is `CANCELLATION_CONFIRM_S`, the same budget the level above already
spent, applied once more to the one level that did not get it. Measured from the cancellation
request rather than from the terminal state the two come to twice that — which is the interval
`orphaned-fleet-teardown.md` already recommends between a run's death and re-dispatching its
cells. Arriving at the recorded number from the mechanism rather than adopting it is the
point: if the confirmation budget moves, this follows, because it is the same constant.

### A live-run census was built, and then deleted

The first two versions of this change asked the orchestrator which cells a live run claimed,
and withheld those. It is recorded here because the reasoning generalises.

The census attracted review findings faster than they could be fixed: offset pagination
skipping a claim as the result set mutated; runs with a null state excluded by the filter; the
same null-state hole surviving the stability loop added to fix the first two; and the bounding
of that loop. Each fix added a mechanism — negate the terminal states rather than list the live
ones, repeat until two passes agree, union across passes, cap the passes, decline if unsettled
— and every one was individually defensible. Together they were a great deal of machinery
guarding one window.

**The deeper problem is that a census is the wrong instrument.** It can only report what was
true a moment ago, and it cannot make a lingering child stop. An asynchronous cancellation is
settled by time, not by observation. Two mechanisms in this system already ACT rather than
observe, and what they need is for someone to let them finish:

| mechanism | what it does |
|---|---|
| the fill's own teardown | cancels its child ingests and WAITS for each to confirm terminal, before its state is set |
| the orphan sweep flows | independently find and stop runs and mosaics that outlived a teardown, on their own schedule |

The census was a third guard layered over two that already existed, and it was the only one of
the three that could not actually stop anything. Deleting it removed a module, its tests, and
every finding it had produced or would produce.

**What is lost.** The census could catch a child that ignored its cancellation for longer than
the delay, and a writer dispatched from outside the campaign. Neither is eliminated now — but
neither was *guaranteed* before either, since a point-in-time query cannot exclude a writer
that starts after it, which is what the last review round said about it in its own terms. The
residual is bounded by what the store does with a second writer: mosaic commits pass no
`rebase_with` and never retry `ConflictError`, so the loser fails loudly. The cost is a wasted
fleet and a cell that waits for the round, not silent corruption.

**What is not claimed.** Nothing here reserves a cell. Closing that properly means fencing at
the write — a lease or generation token the committer checks — which is the same prerequisite
the crashed and cancelled cases are waiting on, and it is a larger change than belongs in a
mid-campaign merge.

**And a structural third.** The replacement's cells are intersected with the roster it
inherited, as well as scoped by it in the work-list query. The scope is the work list's
contract; the intersection makes "a replacement can never reach a zone a live sibling
holds" a property of the function rather than one delegated to another. This was added
after the disjointness property test caught a widening replacement — via a test stub that
ignored `expected_zones`, but the guard is worth having regardless.

## What was deliberately left alone

**The teardown of a dying fill's ingest children.** Letting them outlive the fill re-opens
exactly the hazard `_DeploymentCellInputs.shutdown()` and `.discard()` were written to
close, in their own words: the retry that would race an orphaned child is a whole new
parent run, deriving its child tag from its own flow-run id, so it can neither find that
child nor be told about it — and mosaic commits do not rebase. Since the bytes survive
anyway, sparing the children buys nothing and costs the invariant.

**The barrier.** It is also where the store re-read and the no-progress guard live. A fully
streamed loop needs a per-cell attempt ledger before `set(remaining) == before` can be
replaced. Not with a campaign running.

**Crashed and cancelled fills.** `orphaned-fleet-teardown.md` names the prerequisite:
"If the campaign ever re-dispatches crashed cells **automatically**, the bound stops being
good enough and the fix is **fencing at the write** — a lease or generation token the
ingest checks before each commit." That is the open follow-up, and it is an OSS change.

**Failure classification.** `ingest_read_failure_causes_2026_08.md` records the remaining
amplification across the cell and round budgets. Untouched.

## Alternatives rejected

| option | why not |
|---|---|
| teach the sweeper to spare a dying fill's ingest children | breaks one-writer-per-mosaic-prefix and produces an orphan the retry structurally cannot see; and the bytes already survive, so it buys nothing |
| remove the barrier entirely | the store re-read and the no-progress guard live there; needs a per-cell attempt ledger first |
| retry harder inside the child | already exists (`attempts_per_cell_in_cluster`, plus the runner's in-child retry pass, whose comment gives the same rationale). Cannot help when the session never came up, which is this failure |
| wait out the drought | PRs #140 and #142, already merged. Reduces how often this fires, and cannot help a fill that failed for any other reason. It also LENGTHENS the barrier it leaves in place — see the composition note above |
| give each fill fewer zones | `max_parallel_clusters`, `max_parallel_ingest`, `num_actors` and `overlap_years` are one decision that moves together (§3). Changes cost and fleet shape to shrink a blast radius without speeding recovery |
| move ingest dispatch up into the driver | the child's look-ahead is what keeps GPUs fed; hoisting it reintroduces the backpressure ADR-011 removed and creates a second owner for the mosaic prefix |
| a global scan for orphaned cells | a per-run recovery that scans globally amplifies. This is one paged live-runs query, made in-process, at most once per settled slot, dispatching at most one replacement |
| an integer knob for the generation bound | a fourth number is a fourth thing to get wrong under pressure, and the value is not a judgement an operator is placed to make mid-incident |

## What the first review found, and what it changed

Two independent reviewers produced eleven threads on the first version. Tabulated by root
cause rather than by file, they were **six distinct findings, five of them raised by both
reviewers** — which is the signal worth recording: two careful readers walking into the same
five things is a structural problem, not eleven edge cases.

| Finding | Root cause |
|---|---|
| Parameters built lazily; tasks created outside the cancellation `try` | A |
| The sibling-liveness gate is stale by the time the dispatch happens | A |
| The store re-read escapes instead of declining | B |
| Offset paging can silently under-report a claim | B |
| Null-state runs excluded by the live-state filter | B |
| The refill bound resets every round | C |

**Root cause A — a hand-rolled `gather` lost its atomicity.** `gather(*(...))` consumes its
whole generator before scheduling anything, so a `_chained_params` that raised dispatched
nothing at all. Interleaving build and schedule meant a later cluster's failing land-mask
probe left earlier fills running while the campaign failed — and an ordinary `FAILED` state
does not fire the child-cancel hook. The rule now stated in the source: **nothing that can
await or raise may sit between deciding to dispatch and dispatching.** Every parameter set is
built before any task exists; every task is created inside the `try`; and an await-free commit
gate re-checks liveness immediately before the dispatch, because planning awaits and a sibling
can finish across that await while its task sits unharvested in the map.

**Root cause B — uncertainty was not uniformly fail-closed.** `_refill`'s own docstring
promised it "refuses on anything it cannot establish, including its own inability to ask", and
the store re-read two lines below it was unguarded. One guard now covers every read the
decision makes. The claim query stops enumerating LIVE states and negates Prefect's own
`TERMINAL_STATES`, so the dangerous verdict is no longer the fallback, and it repeats until
two consecutive passes agree, unioning across them — which fixes the paging skip and the
null-state window with one mechanism, bounded at `_STABILITY_PASSES` and declining, never
proceeding, when it runs out.

**Root cause C — a bound implemented per round against a promise made per cell.** The marker
lived in the round's task map, so each round handed out fresh eligibility: four attempts at the
default two rounds, against three documented in the flag's docstring, this file, the README and
the pull request. A documented bound that is not true is worse than an undocumented one,
because the next person sizing a fleet believes it. Eligibility is now tracked per cell at
campaign scope.

**A defect neither reviewer caught, found by the structural pass.** The replacement's own
`_chained_params` call does an S3 land-mask probe and an SSM read, and it sat outside the
guard — so a transient read failure there ended the campaign rather than declining a refill.
It is the clearest argument for tabulating by cause instead of answering eleven comments in
order: nothing in the list pointed at it, and the rule that fixes root cause B fixes it too.

**Two remedies were declined, with reasons.** Finding 5 suggested including null-state runs in
the query or classifying locally: the first is impossible, since the server compiles both the
inclusive and the exclusive state filter to SQL that drops NULL, and the second means paging
the whole run history per refill decision against a server this campaign already knows goes
quiet at fleet width. Finding 6 offered "count the refill against the cell's dispatch budget"
as an alternative: that would give a refilled cell FEWER round attempts than an un-refilled
one, silently weakening the recovery path `max_dispatch_rounds` exists for.

## Verification

Green: 3,390 unit tests, `tests/architecture`, the module-rule runner, ruff and mypy.
The 97 pre-existing `test_run_global_campaign.py` tests pass unchanged, which is the
flag-off claim.

Every new test was checked against a **mutation**, not only against a full revert — a full
revert makes them all fail on an unknown parameter, which proves nothing about what they
measure. The table was REBUILT from scratch after the review restructure rather than ported,
which is how the one genuine gap in it was found: the pre-plan liveness check had no test of
its own, because the commit gate that followed it subsumed its effect. It is a cost filter,
not the safety gate, and it is now pinned as one.

| mutation | tests that failed |
|---|---|
| decision neutralised (never refill) | flag pair (on), replacement scope, raised-dispatch reason, one-generation, campaign-wide bound, settling wait, wait-not-paid |
| every terminal state accepted | crashed, cancelled |
| pre-plan liveness gate dropped | round-already-over |
| replacements made eligible for their own replacement | 7 tests |
| store ignored, whole roster re-dispatched | replacement scope, campaign-wide bound |
| roster intersection AND zone scope dropped | 7 tests, incl. **disjointness** |
| raised-dispatch reason folded into the state check | raised-dispatch |
| build and schedule interleaved again | 15 tests |
| commit gate removed | round-ended-while-planning |
| liveness judged by truthiness, not `done()` | round-ended-while-planning |
| planning guard narrowed | store-read decline, replacement-build decline |
| refill eligibility scoped to the round | campaign-wide bound |
| settling wait removed | settling wait, round-ended-while-planning, re-read ordering |
| wait no longer derived from the cancellation budget | derivation |
| wait paid even when nothing is missing | wait-not-paid |
| the wait moved after the re-read | re-read ordering |

Two notes on test design, recorded because both were nearly got wrong:

- **The flag-off assertion is parametrised against the flag-on case.** On its own it passes
  whether or not the change is present, which makes it worse than no test.
- **The disjointness violation is RECORDED, not raised.** The driver catches a dispatch's
  exception and records it as a cluster failure, so an assertion thrown inside the fake
  dispatcher is swallowed and the test passes. It is collected on the harness and asserted
  after the run.

A third, found by the property test itself: `_shrinking`, the shared work-list stub,
ignored `expected_zones`. A stub that ignores a scope hands a zone-scoped caller every
zone, which is the shape of the bug a zone-scoped caller exists to avoid. Fixed.

## Watching the first run with it on

- The replacement's zone list must be a strict subset of the dead fill's, and must never
  overlap a live sibling's.
- Simultaneous fills must never exceed `max_parallel_clusters`. The replacement takes the
  vacated slot; if the count rises, the slot accounting is wrong.
- `campaign-monitoring-plan.md`'s alert model is keyed to a five-minute poll and a round
  cadence, and `notify-fill-run-lost` is written on the assumption that the driver retries
  later rather than at once. Watch for flapping.
- The decline reasons are all logged with their cause — "the dispatch itself raised", "the
  fill ended <state>", "a live run still claims them", "could not read live run claims". A
  refill that never fires should say which of those it was.
