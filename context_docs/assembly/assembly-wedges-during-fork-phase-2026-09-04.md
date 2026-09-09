# The 2026-09-04 assembly wedges: the hang moved into the fork phase, where no safeguard reaches

**Five clusters stopped assembling on 2026-09-04 and never resumed.** By 2026-09-08 they held 177
completed-but-unassembled cells between them while four sibling clusters ran normally. This is the
same underlying icechunk hang as the 2026-08-29 / 08-31 stalls (see
`../storage/writing-to-the-global-store.md` §3, and the incident record in yield-embeddings PR #71,
`context_docs/crash-recovery/assembly-stalls-before-commit-2026-08-29.md`), but it
manifests at a **different point in the assembly** — during the fork write phase rather than at the
commit or the post-write catch-up — and that point is protected by none of the safeguards built for
the earlier incidents. The recovery machinery (`rehome_after_a_wedged_catch_up`) is structurally
unreachable here, and the commit-stall alarm is structurally blind to it.

Store integrity was never at risk: the wedged process writes ~16 bytes/45 s, so there is no active
writer, no torn commit, no second writer. Staged tiles persist, so no cell is lost.

## The signature, and what it proves

Four of the five wedged with an identical, clean signature. Each cluster's `Assembly progress` line
stream ends at **exactly "1/16 tile partitions outstanding"** and then goes silent forever:

| cluster | zone-year | last progress line (UTC) | shards at stop |
|---|---|---|---|
| gorgeous-rat | 19S | 2026-09-04 21:47:27 | 7940/7951, 1/16 outstanding |
| light-macaque | 30N | 2026-09-04 22:29:53 | 7191/7208, 1/16 outstanding |
| fictional-gaur | 51N-2020 | 2026-09-04 23:22:16 | 5575/5590, 1/16 outstanding |
| vociferous-earthworm | (fill) | 2026-09-04 23:23:24 | 6572/6575, 1/16 outstanding |

(The fifth, mystic-gopher, last emitted a large-zone `Assembly progress` on 2026-09-03 and then only
queued tiny 4–17-tile southern zones; its wedge point is less cleanly dated and is treated as
probable-same rather than confirmed-same.)

**"1/16 outstanding" is the normal LAST progress line of a healthy assembly**, not a special one: it
is emitted from `_await_forks`'s loop `while pending: wait(..., FIRST_COMPLETED)`, which prints only
while a fork is still outstanding, so when the sixteenth finishes the loop exits with no closing
"0/16" line and the next thing a healthy fill prints is `ASSEMBLY_SUMMARY`. A healthy cluster
(mellow-walrus, honest-pudu) shows exactly this: "…1/16 outstanding" immediately followed by
`ASSEMBLY_SUMMARY`. So the wedged clusters reached the very end of the shard write and then produced
**neither** the summary nor any failure — the assembly thread stopped between the last worker and the
completion mark.

> **ROOT CAUSE CONFIRMED 2026-09-09 (supersedes the "cannot be pinned from logs" paragraph below).**
> The hang is not a mystery tail wait. It is an icechunk deadlock triggered by opening the store with
> `max_concurrent_requests = 1` — the value our S3-budget split produces on a wide campaign
> (`per_worker_s3_cap: 1` in the live assembly summary). Reproduced on real S3 with native stacks
> (`scripts/scoping/wedge_repro/`, run `repro-main-05`): under concurrent writers, `Session.commit`
> AND `Repository.diff`/`rebase` park forever in icechunk's tokio runtime; a control at cap 2, load
> identical, publishes cleanly. Bisection: cap 1 deadlocks, cap ≥ 2 clean. See
> `icechunk-max-concurrent-requests-1-deadlock.md`. Two consequences correct the analysis below:
> (1) the **commit is NOT excluded** — one captured stack is parked in `commit`, and the earlier runs'
> stall alarm never fired only because those runs used icechunk's default cap of 256, not the
> production cap of 1; (2) `rehome_after_a_wedged_catch_up` does not merely fail to reach the wedge —
> its `diff` call parks in the **same** deadlock, which is why the fallback re-commit never recovered
> the 09-04 clusters. The real cure is to never open at cap 1 (floor the per-worker cap at 2 in
> `_s3_budget_split`); publication spacing and the rehome recovery do not address the cause.

**[Original hypothesis, retained for the record.] Where in that tail cannot be pinned from logs, and
the honest envelope is what matters.** After the
last progress line the assembly thread still has several *unbounded, silent* waits ahead of it, any
one of which fits the evidence: the worker-pool shutdown join (`ex.shutdown()`, wait=True, no
timeout — hangs if a worker process will not exit), the post-write catch-up handling at
`ticking.__exit__`, `rehome_after_a_wedged_catch_up`, and `session.merge(...)`. The **commit** is
excluded: `traced_commit`'s stall alarm wraps it and never fired. This is the crucial difference from
2026-08-29/31, where the hang was localised to the commit itself. Here it is upstream of the commit,
in a stretch of code that emits nothing and has no timeout.

## The live wedged process, measured 2026-09-08 via ECS Exec (`/proc`, no ptrace)

`vociferous-earthworm`'s flow-runner task (`5dc8e2f7…`, Fargate, 16 vCPU) finished all its inference
today at 14:02:55 ("Inference complete — draining 67 trailing assembly/assemblies, one at a time")
and has been silent since. Its trailing-assembly thread has been wedged since 2026-09-04, so the
`finalizer.shutdown(wait=True)` that runs after inference completes now blocks forever joining it.

| probe | result |
|---|---|
| CPU across all 144 threads | 21 ticks / 45 s ≈ 0.5% of one core |
| bytes read / written / 45 s | 16 / 16 |
| worker child processes | none (only the flow process + a `multiprocessing.resource_tracker`) |
| icechunk tokio-runtime threads (16) | all parked in `futex_wait_queue`, zero CPU |
| threads Running or in uninterruptible I/O | none — every thread parked |
| commit-stall alarm (`ASSEMBLY COMMIT STALLED`) | never fired, on any cluster, in 7 days |

No thread is in a live S3 read. Everything is parked on a lock/futex. This is a deadlock, not a slow
operation.

## Why the same wedge, at a new site

The trigger conditions match 2026-08-31 exactly: icechunk 2.1.1, the one shared `main` branch, and
concurrent publications raising a coordinator's catch-up depth. gorgeous-rat and light-macaque both
logged `Caught up … to branch tip WAFP5J8EKE5VYQSHJY00` at 21:15:2x — the same tip, i.e. a shared
publication moment — shortly before wedging. The catch-up runs on a timer *throughout* the fork
phase (`CATCH_UP_INTERVAL_S = 5 s`), so a tick that hits the depth-4 hang while the workers are still
writing wedges the fill **inside** the `with ticking(...)` block, not at its exit.

**The distinguishing factor between the recoverable 08-31 case and the unrecoverable 09-04 case is
timing.** On 08-31 the wedging tick was noticed at the block's *exit* — the workers had finished, the
`with` body returned, `ticking.__exit__` ran `ticker.join(timeout=30)`, saw it still alive, and
raised `CatchUpDidNotStopError`, which `run_forked` catches and recovers with
`rehome_after_a_wedged_catch_up`. On 09-04 the tick wedged *while a worker was still writing*, so the
`with` body (`_await_forks`) never returned, `ticking.__exit__` never ran, and the recovery was never
reached.

## Q&A

**1. Why did the assembly workers wedge?** An icechunk operation in the shard-write tail hung — the
same `session.rebase` → `fetch_snapshot` family that hung seven of nine commits on 08-29 and two of
two catch-ups on 08-31. The exact operation is not pinned (see the envelope above and the open
question below), but the trigger is the same catch-up-depth condition and the store, version and
concurrency are identical.

**2. Is it the same wedge as before?** The underlying icechunk hang is the same family. The
**manifestation** is new: the fork phase stalls at "1/16", versus the commit/post-write stall of the
earlier incidents.

**3. Are the causes the same as we understood?** The *trigger* is the same (catch-up depth from
concurrent publications). What we had not accounted for is that a catch-up tick can wedge **during**
the writes, not only after them — and that this location defeats every recovery we built. So: not a
misdiagnosis of the earlier incidents, but an incomplete model of where the hang can land.

**4. Why did the safeguards not work?**
- The **commit-stall alarm** (TE #162, faulthandler dump + repeated shout) wraps only
  `traced_commit`. The fork-phase wedge is before any commit, so the alarm never armed. This is why
  seven days produced zero alarms and zero stacks.
- The **abort escape** in `_await_forks` (`if abort.is_set(): raise CatchUpAbortedTheWaitError`)
  fires only when a catch-up tick *raises*. A tick that *hangs* never raises, so `abort` is never
  set and the wait is never broken.

**5. Why did the fallback restart / re-commit not work?** `rehome_after_a_wedged_catch_up` (TE #165,
wired on the campaign path at `shard_writer.py:827`) is reached only through
`except CatchUpDidNotStopError`, which is raised only from `ticking.__exit__`, which runs only after
the `with ticking(...)` body returns. If the wedge is *inside* that body — in `_await_forks` or the
worker-pool `ex.shutdown()` join — the body never returns, `__exit__` never runs, and the re-home is
never reached. If instead the wedge is at or after `__exit__` (in `rehome` itself, or `session.merge`)
then re-home either ran into the same hang or was passed. Either way the recovery did not fire, and
no failure was recorded — the cell neither published nor failed, it simply stopped, and its cluster's
single trailing-assembly thread stopped with it.

## The one thing not yet directly observed

Why the *coordinator* thread stops (rather than only the sixteenth worker) cannot be determined from
logs, and no Python stack is obtainable from an already-wedged process — Fargate denies ptrace, and
faulthandler is only armed around `traced_commit`, which this wedge never reaches. The leading
hypothesis is that the hung catch-up holds a lock (icechunk session/repo, or the GIL) that the
coordinator or the last worker then needs; a competing detail is that on 08-31 a hung tick evidently
released the GIL (the `join` timed out rather than hanging), which a GIL-holding hypothesis has to
reconcile. **This does not gate the fix**: every candidate mechanism implies the same requirement —
the fork phase must be bounded and watched — and the dev reproduction (faulthandler armed in advance)
is the instrument that will settle the micro-mechanism.

## What was built (tessera-embeddings, branch `fix/bound-the-assembly-fork-phase`)

Three bounds, each aimed at a different one of the silent waits above, plus the stack capture that
was missing.

1. **A fork-phase watchdog** (`shard_writer._fork_stall_watchdog`, `FORK_STALL_TIMEOUT_S = 30 min`).
   A daemon thread watches the shard counters the workers write into shared memory — they advance
   whether or not the coordinator thread is running. If the total stops moving for thirty minutes
   (about six times the longest gap a healthy dense write has shown) it dumps every thread's stack
   with `faulthandler`, sets the abort flag, and terminates the worker pool, so a coordinator parked
   in `_await_forks` or in the pool's `shutdown(wait=True)` join is freed and the fill fails as an
   ordinary assembly failure: cell retained, staged tiles kept, re-dispatched. Log line:
   `ASSEMBLY FORK PHASE STALLED`. What it cannot do is unwind a coordinator thread parked inside
   icechunk itself; the dump still fires, so that case is at least diagnosed.

2. **A bounded drain** (`sequential_fill.drain_trailing_assemblies`, `TRAILING_ASSEMBLY_CEILING_S =
   6 h`). `finalizer.shutdown(wait=True)` — the end-of-inference drain, and where vociferous-earthworm
   sits today — is replaced by a wait on the queued assemblies' futures with a *per-assembly* ceiling
   that resets on each completion, so a sixty-cell backlog drains however long it takes but one
   assembly that never returns is abandoned after six hours (the longest healthy global assembly is
   ~3.5 h, and the watchdog above already fails a stalled write inside thirty minutes, so only a hang
   the watchdog cannot unwind reaches this). It dumps stacks, cancels the assemblies still queued,
   and raises `TrailingAssemblyWedgedError`. Log line: `TRAILING ASSEMBLY WEDGED`.

3. **A wedged drain ends the process instead of reporting FAILED**
   (`fill_zones_sequential_flow`, `_end_process_after_wedged_drain`, exit status 75). The campaign
   driver reads `FAILED` as "this fill stopped writing" and may hand its zones to a replacement
   cluster five minutes later — the README's admission rule rests on "a fill that returned or raised
   has, by then, joined the trailing assembly thread". After a wedged drain that thread has NOT been
   joined; it is parked inside icechunk holding a writable session. So the flow tears down (fleet,
   ingests, housekeeping) and then `os._exit`s. Prefect records CRASHED from the missed heartbeats,
   which the driver already treats as "cannot infer who stopped": the cells wait for the next round,
   exactly as any crash's do. The thread dies with the process, so nothing can commit later.

4. **The write is recovered, not discarded — #165's re-home, generalised and made killable**
   (`shard_writer.run_forks`, `_run_pool`, `publish_forks_in_child`). Robert's point, and the right
   one: a wedge should cost a retry of the commit, never the hours of shard writes or a cluster
   restart. Three changes deliver it. A partition whose worker stalls is re-run once on a fresh pool
   with every finished fork kept; a second stall on the same partition is a deterministic fault and
   fails the cell. Finished workers are terminated and never joined — their forks are already the
   parent's — so a worker process that will not exit costs nothing. And the publish (fresh session at
   the tip, conflict check over `base..tip`, merge, shard commit, completion mark) runs in a child
   process under a per-step timeout; a child that wedges is killed and the step retried once on a new
   session, mark-only if the shards already landed. The forks never leave the parent, so the write is
   never at risk. Why a child: the 09-04 coordinators were parked inside icechunk where no Python
   mechanism unwinds a thread, and a coordinator already wedged cannot start its own retry. The child
   arms `faulthandler.dump_traceback_later` inside the parent's timeout, so the next wedge prints the
   stack this incident could not.

Every claim above was checked by breaking the source and watching the tests go red (five mutations
of the watchdog, each caught; a hang is turned into a red test by running the subject under a
deadline). The tests inject a wedge deterministically — a worker future that never resolves, a pool
shutdown that never returns, an assembly that never returns — rather than reproducing the icechunk
hang itself, which ten prior attempts and the 08-31 harness never managed at small scale.

**The root cure, built on top of this (`feat/space-publications`).** None of the three bounds above prevents the icechunk hang; they bound its cost
and capture its stack. Preventing it means keeping catch-up depth below 4, i.e. spacing publications
apart, which needs the fleet-wide lock TE #151 removed. `../storage/writing-to-the-global-store.md`
§3 and §5 had deferred that twice; this incident is its justification. `storage/publication_spacing.py`
now holds a limit-ONE fleet-wide slot around every publication and then for three catch-up intervals,
so two publications can never land inside one interval — see the §3 and §5 addenda there.

## Dev evidence, 2026-09-08

`scripts/scoping/wedge_repro/publication_density.py` against real S3 in the dev account, 120 zone
groups at production layout, hundreds of pre-seeded snapshots, terminal marks every 3 s:

| arm | outcome |
|---|---|
| unspaced control, 10 coordinators | catch-up depth reached **4** — the hang's precondition — with publications 0.8 s apart; the hang itself did not appear (it never has at this scale) |
| spaced, 10 coordinators | max depth **1** across 110 ticks; all published; data intact |
| wedged catch-up | `Re-homing` fired, the cell published; neighbours untouched |
| wedged worker | `ASSEMBLY FORK PHASE STALLED`, four thread stacks captured, the cell failed as `ForkPhaseStalledError`, five neighbours published |

Two defects the run surfaced were fixed in the same PRs and confirmed by a re-run: the spacing hold
was measured from slot acquisition (a 3 s commit left a 12.1 s gap against 15) and is now a full
spacing after the last commit — every gap by the store's clock then read ≥ 15.9 s over 110 ticks with
max depth 2; and `rehomed` never reached `ASSEMBLY_SUMMARY` — it now does (1 / 1 on the wedged arm).
Full tables in the harness README.

### Dev evidence for the recovery, 2026-09-08

| arm | outcome |
|---|---|
| wedged publish (child killed at 60 s) | retried on a fresh session, cell **published**, `publish_retries` 1, the killed child's stack captured; 6/6 published |
| wedged worker, once | watchdog at 90 s, partition re-run, cell **published**, `partitions_rerun` [1]; 6/6 published |
| wedged worker, always | re-run stalled too; that cell failed as `ForkPhaseStalledError` after one re-run, 5/6 published |

All three: data intact, spacing held (≥ 16 s by the store's clock), max catch-up depth 2, nothing
left in the dev bucket. A wedge now costs a kill and a retry, not the write.
