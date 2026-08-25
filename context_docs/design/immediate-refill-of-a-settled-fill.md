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

**Condition two — nothing live claims the cells.**

Condition one covers the fill's own writers, not everything downstream. Its grandchild ROI
ingests are cancelled one level further out and asynchronously, and `shutdown()` says of
itself that it is best effort — it logs by name any child it could not confirm.

So `_live_claims.zone_years_claimed_by_live_runs` asks the orchestrator which cells a live
run claims. This is **lifted from `yield-embeddings/scripts/dispatch_pending_fills.py`**,
which has been corrected once by an incident and whose three claim forms encode that
knowledge:

1. a `zone` plus `year` parameter,
2. an entry in a `cells` list,
3. **any parameter value containing a `mosaics/<zone>/<year>` path.**

Form 3 is the one that cost a run: a one-month optical top-up (`topup-59S-2021-dec`) was
still committing December when 59S read as "complete (3 legs)", so a fill was dispatched
against a mosaic that was still growing. It would have inferred from a short year, tagged
it complete, and nothing would have flagged it.

Two polarity choices are load-bearing and are pinned by tests:

- **No allow-list of known writers.** An unrecognised flow reads as a claim. An allow-list
  fails silently in the one direction that matters, because a writer missing from it reads
  as an absence of claims. `claims_in` takes a run's parameters and nothing else, so there
  is no identity to consult.
- **An unanswerable query is not an empty answer.** It raises, and the driver declines.
  Same rule `orphaned-fleet-teardown.md` gives for a state that could not be read.

Refusal is at **zone** granularity. Dropping one claimed year while keeping its zone's
others would put that zone's years on two clusters — the split the partition exists to
prevent.

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

## Verification

Green: 3,390 unit tests, `tests/architecture`, the module-rule runner, ruff and mypy.
The 97 pre-existing `test_run_global_campaign.py` tests pass unchanged, which is the
flag-off claim.

Every new test was checked against a **mutation**, not only against a full revert — a full
revert makes them all fail on an unknown parameter, which proves nothing about what they
measure.

| mutation | tests that failed |
|---|---|
| decision neutralised (never refill) | flag pair (on), replacement scope, live-claim withholding, raised-dispatch reason, one-generation |
| every terminal state accepted | crashed, cancelled |
| live claims ignored | live-claim withholding |
| unreadable claim query reads as "nothing claimed" | unreadable-query decline |
| "a sibling is still running" gate dropped | flag pair (on), last-cluster |
| replacements made eligible for their own replacement | 5 tests, incl. one-generation and disjointness |
| store ignored, whole roster re-dispatched | replacement scope |
| roster intersection AND zone scope dropped | 5 tests, incl. **disjointness** |
| raised-dispatch reason folded into the state check | raised-dispatch |
| mosaic-path claim form dropped | mosaic-path, unpadded-zone |
| zone canonicalisation dropped | unpadded-zone |
| CANCELLING treated as finished | cancelling-is-live |
| paging stops after the first page | paging |
| malformed cell entry not tolerated | malformed-cell |
| unparseable zone dropped instead of recorded | unparseable-zone |

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
