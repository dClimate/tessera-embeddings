# Keeping the assembly session current (2026-08-30)

**The change.** While an assembly's forks are writing, the coordinator now brings its own
session up to the branch tip on the same timer it already reports progress on — but only past
commits that did not touch its own zone group. One new function, one new hook, one call site.

**Why.** On 2026-08-29 seven of nine assemblies finished hours of writing and then never
returned from their commit. The trace named the call: `icechunk::asset_manager::fetch_snapshot`,
reached from `session::rebase` via `session::list_nodes`, and it is only reached when the
session has fallen behind the branch. The full incident record — traces, thread stacks, the
socket evidence and the eliminations — is in yield-embeddings,
`context_docs/crash-recovery/assembly-stalls-before-commit-2026-08-29.md`.

## A third confirmation, from production, unprompted

At 05:04Z on 2026-08-30 — while this fix was being written, with seven assemblies still stalled
from the 22:30 batch — a NEW cell published: 50N/2017. Its trace shows why:

```
05:04:43.496  INFO icechunk::session: Commit started, old_snapshot_id: GH5GAHTPF2TS1QFRGHZ0
05:04:44.086  INFO icechunk::session: Commit done
```

`GH5GAHTPF2TS1QFRGHZ0` was the branch TIP — 32N/2018's completion mark from 01:39Z. So this
session was **zero snapshots behind**: no "Branch tip has changed", no rebase, no `list_nodes`,
no `fetch_snapshot`. Both its commits landed in 0.6 seconds total.

Three independent confirmations now, none of them arranged: depth 0 and 2 publish in about a
second, depth 4 stalls indefinitely. This one is the most valuable because nobody set it up —
the campaign produced it on its own, on the same store, in the same hour that seven cells sat
wedged.

## And a fifth: a NEW stall, ten hours later, predicted in mechanism first

At 11:20Z on 2026-08-30 a fresh cell stalled — 50N/2018, on a different fill, ten hours after
the batch that produced the original seven. It matters because the discriminator below (three
transaction logs rather than two, with the third's fetch overlapping the rebase loop) was
derived from the earlier stalls and had not yet been tested against a new one. It reproduced
every element:

```
11:20:27.140  Commit started, old_snapshot_id: NXDTHZQNYV577DK4BN0G   <- the tip at 05:35Z
11:20:27.739  Branch tip has changed, rebase needed
11:20:27.749  Downloading transaction log: KCS9T67R0SDDDWD2N8YG
11:20:27.749  Downloading transaction log: AX6YD47KB374KDMWDZP0
11:20:27.796  Rebasing snapshot KCS9T67R0SDDDWD2N8YG
11:20:27.797  Downloading transaction log: JQB6ZB6FM6N76FQT6C8G      <- THE THIRD, mid-loop
11:20:27.819  Rebasing snapshot AX6YD47KB374KDMWDZP0
11:20:27.819  Downloading snapshot: KCS9T67R0SDDDWD2N8YG             <- and nothing after
```

Two cells had published since its session opened (32N/2019, 13N/2018), so **depth 4** — the
level that failed seven for seven — and it failed the same way, at the same call.

**The tally across eleven hours and eight cells is now without exception:** every commit at or
near the tip published in about a second; every commit four snapshots behind stalled.

## The trigger, in one line

A fill opens its session, forks it, writes for about three hours, then commits. Every cell that
publishes during those hours puts another snapshot between the session's base and the tip, and
the commit has to walk all of them. **Zero and two never failed; four failed seven times out of
seven.** Each published cell writes two commits, so in a batch of N simultaneous assemblies the
k-th to finish is `2(k-1)` behind: the first two publish, the third onward stalls. Nine started,
two published, seven stalled.

## What was tried, and what the experiment showed

A harness reproducing the production shape — 120 root groups, one forked session pickled to
eight spawned workers, a concurrent publisher landing two commits per cell, icechunk 2.1.1 on
real S3 in the dev account — ran four arms. `depth` is snapshots to catch up on at commit.

| arm | what it does | depth at commit | commit | conflict case | data |
|---|---|---:|---:|---|---|
| `control` | nothing else commits | 0 | 0.57 s | — | PASS |
| `current` | today's code | 4 / 8 / 16 | 1.58 / 2.31 / 2.77 s | **raises** | PASS |
| `b1` | rebase on a timer, unguarded | **0** | 0.66 s | **SILENTLY OVERWROTE** | PASS |
| `b3` | merge into a fresh session | **0** | 0.59 s | **SILENTLY OVERWROTE** | PASS |
| **`b1g`** | **rebase on a timer, guarded** | **0** | **0.56 s** | **raises** | **PASS** |

**Both naive fixes work and both are unsafe.** With a genuine collision — a second writer
touching the same group and year — today's code correctly refuses with `RebaseFailedError`.
Under `b1` and `b3` the commit succeeded and overwrote the other writer without a word. The
reason is structural: a conflict is only ever detected against commits made *after* the
session's base, so moving the base past a commit is also a decision to stop checking it.

**The guard is what makes it safe.** Before advancing, diff base→tip and refuse if anything
inside our own group changed. Then the base stays put, the commit's own rebase runs exactly as
it does today, and the collision raises. Measured: `b1g` at depth 16 reached the tip with seven
catch-ups and committed in 0.56 s; `b1g` against a collision blocked five times, arrived at the
commit one snapshot behind, and raised `RebaseFailedError` — identical to `current`.

**Why on a timer rather than once at the end.** A single catch-up just before the commit would
walk the same distance, through the same call, that the commit would have. The value is in
never letting the gap grow, not in closing it once.

**Why before the merge and not after.** Before the merge the session holds nothing — the forks
are still outstanding — so catching up is a pointer move. After it the session holds the whole
write, and catching up would replay it against every intervening snapshot: the expensive path
this exists to avoid.

## Was the oracle able to fail?

Yes, and it was checked rather than assumed. `--drop-fork K` omits one worker's fork from the
merge while still claiming its writes; the verifier reported exactly that worker's six chunks
as missing, under both `b1` and `b3`. An earlier version of the harness could not have caught
this — eight workers wrote the same four chunk positions with values that did not encode the
worker, so an entire lost fork was invisible and every arm "passed". Workers now write disjoint
chunks carrying their own index.

## What this does not claim

**The harness never reproduced the stall.** Ten standalone attempts before it and this one all
committed normally at every depth, on a laptop against dev S3. So the efficacy argument is not
"the fix made the stall stop"; it is "the stall has only ever been observed above a depth this
fix holds at zero". The safety argument is the one the experiment settles directly.

**`assemble` is deliberately untouched.** It writes a per-ROI store where it is the only
writer, so its session never falls behind. `catch_up` defaults to `None` and that path is
unchanged.

**The upstream bug is still a bug.** This removes the precondition, not the fault. The report
belongs upstream regardless.

## Review findings folded in (2026-08-30)

An adversarial review against a real repository found three things the first version got wrong.
All are fixed; all are recorded because each is a way this class of change fails.

**The guard could be bypassed, and the result was the silent overwrite it exists to prevent.**
`Session.rebase` takes no target snapshot — it advances to whatever the tip is when it runs —
so a commit landing between the diff check and the rebase is skipped without being vetted.
Reproduced end to end. It cannot be undone, because an empty session's rebase never raises
whatever it walks over. So the catch-up now re-checks the range it *actually* crossed and
raises `CaughtUpPastAConflict`, turning a silent overwrite into a loud abort.

**A failed catch-up destroyed the fill, including after the work had succeeded.** The first
version let exceptions propagate, on the stated reasoning that best-effort belonged at the call
site — but the call site did not implement it. The final catch-up runs *after* every worker has
returned its fork, so an exception there discarded a finished multi-hour write. The triggers
are real and not all transient: a node rename anywhere in the store makes `rebase` raise even
on an empty session, and a reset branch makes `diff` raise. `catch_up_best_effort` now logs and
tallies those as `failed`, while still letting `CaughtUpPastAConflict` through.

**The guard read seven of `Diff`'s eight members.** A rename populates `moved_nodes` and
nothing else, so a commit that renamed our own array read as untouched. Both ends of each move
now count, and all eight members are read by direct attribute access — the previous
`getattr(diff, name, None) or set()` would have let a future field rename silently disable a
whole class of check.

**One behaviour worth knowing rather than fixing.** A block latches: the checked range only
grows once the base stops moving, so one same-zone commit early in a fill makes every later
tick report `blocked`. That is safe — it is exactly today's behaviour — but it means the
mechanism switches off for the rest of that fill. Advancing to the last snapshot before the
offending commit would keep most of the benefit and is not possible today, because `rebase`
accepts no target snapshot. That is a second reason to want the upstream fix.

## A second review, of the tests rather than the code

The tests were then reviewed the same way — by mutating the source thirty times and checking
which tests noticed. Four of them noticed nothing, and the pattern is worth keeping.

**The mechanism was not connected to anything a test looked at.** Deleting the `catch_up=`
argument from the single production call site passed the entire suite. Every test of the
catch-up exercised a function that, as far as the suite was concerned, need never be called.
There is now a test that drives `write_year_shards` and asserts it asks for a catch-up with
its own repository, session and group.

**The ordering test could not detect the inversion it forbade.** It asserted
`has_uncommitted_changes is False` at catch-up time — but that is `False` under BOTH orderings
when the worker hands back an untouched fork, so moving the final tick after the merge left it
green. It now records the order of the two calls directly.

**No test had more than one commit in the gap.** Every catch-up test created exactly one
intervening snapshot, so a guard inspecting only the newest commit looked correct. Production
is explicitly multi-commit — the depths measured above are 4, 8 and 16 — so there is now a
test where the offending commit is the older of two.

**Four of the six enumerated `Diff` fields were unpinned**, and the conflict solver was not
pinned at all: swapping `ConflictDetector` for `BasicConflictSolver` — which would resolve a
collision silently, the exact failure being guarded against — passed everything. Both are now
asserted, the field list by equality against a real `Diff` so that shortening it fails too.

**The lesson worth carrying.** Every one of these tests passed, was readable, and named the
right property in its docstring. What they lacked was an oracle that could move: an assertion
whose value differs between the correct and the broken implementation. Mutating the source and
watching which tests go red is cheap, and it is the only thing that establishes that.

## Final check: the shipped function, not the design

Every arm above tested the harness's own restatement of the mechanism. The last run imports
`catch_up_best_effort` from this branch and calls it, so what was measured is the code that
will run in production, against a real repository on dev S3:

| run | catch-ups | depth at commit | commit | outcome |
|---|---|---:|---:|---|
| 16 snapshots behind, no collision | 6 `advanced`, 6 `current` | **0** | 0.59 s | all 48 chunks correct, neighbours intact |
| 2 behind, deliberate collision | 5 `blocked`, 1 `current` | 1 | 0.61 s | **`RebaseFailedError`** — refused, and the other writer's value survives untouched |

Both are the intended behaviour: the gap closes to nothing when it is safe to close it, and
the fill refuses rather than overwrite when it is not.

**What was NOT run:** a full assembly on the dev CLUSTER, with sixteen spawned forks and a
three-hour write. The mechanism is exercised at production shape but not at production scale
or duration, and the first campaign fill is the first time it runs for hours. Watch the
`catch_ups` tally on the first few `ASSEMBLY_SUMMARY` records.

## Eight coordinators catching up at once — the configuration production actually runs

Every arm above ran ONE coordinator catching up, with a scripted publisher supplying the
cross-traffic. That left the real question untested, and it is the one that matters most for
this fix: **the catch-up multiplies calls to the very path that stalls** — `rebase` ->
`list_nodes` -> `fetch_snapshot` — from about one per assembly to one per tick per assembly.
If the fault were sensitive to concurrent traffic on that path rather than to depth, the fix
would make things worse.

So: eight coordinators, all doing catch-ups, all writing forks, all publishing into one store,
each other's commits the only cross-traffic. Staggered starts and different write lengths so
they reach their commits at different moments. Node hierarchy matched to production — 120
groups of eight arrays each.

| | catch-ups | depth at commit | commit | data |
|---|---|---:|---:|---|
| z000 | 5 current | 0 | 1.40 s | PASS |
| z001 | 5 current, 1 advanced | 0 | 1.59 s | PASS |
| z002 | 5 current, 2 advanced | 0 | 1.44 s | PASS |
| z003 | 6 current, 2 advanced | 0 | 1.77 s | PASS |
| z004 | 6 current, 3 advanced | 0 | 1.43 s | PASS |
| z005 | 7 current, 3 advanced | 0 | 1.42 s | PASS |
| z006 | 8 current, 3 advanced | 0 | 1.48 s | PASS |
| z007 | 5 current, 7 advanced | 0 | 1.38 s | PASS |

**All eight published. All at depth 0. 384 chunks verified, none wrong, no cross-clobbering,
and all sixteen commits on the branch.** About 65 catch-up calls went through the stalling path
inside two minutes — a call DENSITY roughly sixteen times production's, since production
spreads its ticks over three hours — and none of them hung.

**And the same fleet WITHOUT the fix reaches the dangerous depths**, which is what makes the
result mean something rather than describing a configuration that was never at risk:

| arm | depths reached at commit |
|---|---|
| catch-ups off | 0, 2, **4, 4, 4, 6, 6, 7** — six of eight at or above the level that fails in production |
| catch-ups on | 0, 0, 0, 0, 0, 0, 0, 0 |

## Is any of this scale-dependent? Partly — and it matters which part

**What is matched to production:** the node hierarchy that `list_nodes(/)` walks (120 groups x
8 arrays), the number of concurrent coordinators, two commits per cell, real S3, icechunk
2.1.1, and forks pickled across spawned processes.

**What is NOT matched, measured rather than assumed:** the snapshot object — the thing the
stalling call downloads — is **33 KB here against production's 159 KB**, because production's
also carries manifest references accumulated across ~25 published cells. And the fork phase is
seconds rather than three hours.

**The safety conclusion carries across that gap; the efficacy conclusion never rested on the
harness at all.**

* *Safety* is semantic, not dimensional. Whether a fork created before a rebase still merges
  correctly, whether the guard refuses on a collision, whether anything clobbers a neighbour —
  these are path-and-coordinate addressing and set membership on path strings. None has a
  branch on object size, so a bigger snapshot cannot change the answer.
* *Efficacy* comes from PRODUCTION, at full scale, on the real store: depth 0 or 2 published in
  about a second on five occasions; depth 4 stalled on eight, across eleven hours and two
  separate batches ten hours apart. The harness's job was only to show the fix drives depth to
  zero and corrupts nothing, and it shows both at fleet width.

**One correction worth stating plainly: the error has NOT been regenerated at small scale.**
The control fleet reached depth 7 and still committed normally, as did every single-coordinator
arm up to depth 16. The harness reproduces the *condition*, never the *failure*. Anyone reading
this as "reproduced in miniature, therefore proven" would be overstating it — what is proven in
miniature is that the fix removes the condition without breaking anything.


## Where the code lives, and why it is its own module

`storage/session_catch_up.py`. The feature was written inside `shard_writer.py` and taken out
once the review rounds made its shape clear: a constant, an error type, a predicate, the
operation, a safety wrapper and a timer — 220 lines with one invariant between them, sitting in
a 1,035-line module about writing shards.

**The invariant, stated once in that module's docstring:** a session's base may only move past
commits that have been checked against the group being written, and it must never be left moved
without that check having happened.

Every review finding so far has been an attack on exactly that sentence — the diff/rebase race
crosses commits without checking them; the type-keyed error handler left the base moved after a
failed check; the missing `moved_nodes` field made a check incomplete. Putting them in one file
under one stated invariant is what makes the next such finding land somewhere obvious.

`shard_writer` keeps only the wiring: build the callback, wrap the fork phase in the timer, tick
once more before the merge, and report the tally. `_await_forks` went back to being a waiting
helper with no housekeeping hook at all.


## What the fix does NOT close, stated with the arithmetic

A coordinator's base is current as of its last tick, not as of its commit. Peers can publish in
between, so assemblies finishing together can still stack: **if three commit inside one
`CATCH_UP_INTERVAL_S`, the third arrives four snapshots behind** — the depth that fails.

It takes three inside one interval, and that is further than the observed spacing. In the
2026-08-29 incident the nine commits fell at 01:02, 01:39, 01:46, 01:58:24, 01:58:36, 02:00,
02:07, 02:09 and 02:11 — never more than two inside any 60-second window. Replaying those
timings against a 60 s interval puts every one of them at depth 0.

**Why not close it completely.** The obvious move is one more catch-up immediately before the
commit, and that is exactly the unbounded call removed above: it sits outside the timer, so a
stall there hangs forever and silently. Serialising commits was measured earlier and does not
address the cause, because the gap opens during the write rather than during the commit.

**What would show it mattering, and the lever.** The `catch_ups` tally. Ticks reporting
`advanced` late in a fill mean peers are landing inside the interval; the response is to
shorten `CATCH_UP_INTERVAL_S`, which trades linearly against how much traffic the catch-up puts
through the same path that stalls. It is deliberately left at 60 s until the tally says
otherwise, rather than multiplied on a guess.

---

# Production, 2026-08-31: the mechanism held, and the residual risk fired twice

The campaign restarted onto this code at 22:45Z on 2026-08-30 with ten coordinators. What
follows corrects the arithmetic in the section above, which understated the exposure by exactly
one publication.

## The result the fix was built for

Eight cells published between 01:39Z and 02:39Z, and their own `catch_ups` tallies say how far
each would otherwise have had to rebase:

| cell | published | advances | `commit_s` | depth without the fix |
|---|---|---|---|---|
| 50N-2018 | 01:39:05Z | 0 | 0.689 | 0 |
| 44N-2020 | 02:13:01Z | 1 | 0.502 | 2 |
| 48N-2019 | 02:23:22Z | 2 | 0.764 | **4** |
| 34N-2018 | 02:27:12Z | 3 | 0.744 | 6 |
| 37N-2018 | 02:30:24Z | 4 | 0.634 | 8 |
| 36N-2019 | 02:34:20Z | 5 | 0.672 | 10 |
| 38N-2019 | 02:38:42Z | 6 | 0.621 | 12 |
| 47N-2018 | 02:38:53Z | 7 | 0.620 | 14 |

`commit_s` does not rise with the number of advances. The catch-up is not spreading the cost of
a deep rebase across many shallow ones; it is avoiding it. 38N-2019 and 47N-2018 committed
**eleven seconds apart** — the tightest simultaneous pair yet observed, and both at depth 0.

## The arithmetic above is wrong: TWO publications inside an interval, not three

The earlier section says *"if three commit inside one `CATCH_UP_INTERVAL_S`, the third arrives
four snapshots behind"*, and then reassures with *"never more than two inside any 60-second
window"*. **Two is already enough, and that error is why the risk looked further away than it
was.** A publication is TWO snapshots — the `Run …: fill` commit and the `mark … complete`
commit. Depth is counted from a coordinator's last successful advance, so:

    one publication inside the interval   -> depth 2   (safe, 36 for 36)
    two publications inside the interval  -> depth 4   (the depth that fails)

The replay in the earlier section counted publications as one snapshot each. Corrected, its own
2026-08-29 data — two commits inside a window, repeatedly — predicts the failure that then
happened.

## What actually fired, twice, and the census that makes it unambiguous

At 02:38:42Z and 02:38:53Z two cells published eleven seconds apart. Only two coordinators were
still in a fork phase and therefore exposed: `prompt-magpie` (35N-2018) and `blue-coati`
(32N-2021). **Both wedged. Two for two.**

| depth attempted | attempts | wedged |
|---|---|---|
| 2 (one publication) | **36** | 0 |
| 4 (two publications) | **2** | **2** |

The 36 successes are every `Caught up` line in the run, across eight zones and six rounds; each
crossed exactly one publication. This reproduces the original 8-for-8 failure at depth 4 exactly,
in a different call site, and settles that the fix relocated the wedge without changing it.

`prompt-magpie` aborted at 02:40:37Z with `CatchUpDidNotStopError` when its forks finished.
`blue-coati` was still writing shards at 41% and is expected to abort the same way when it
finishes — a live prediction, recorded here before the fact.

## The wedge is unbounded, so the stop timeout is not the lever

The obvious cheap fix — raise `CATCH_UP_STOP_TIMEOUT_S` above 30 s — is dead. **The wedged
thread never returned.** No further `Caught up 35N`, no `Catch-up for 35N failed`, nothing, in
the 27+ minutes the run stayed alive afterwards. That matches the 14,396 s the original stall
sat. The thread is blocked inside a pyo3 call into Rust, where no Python mechanism — not
`PyThreadState_SetAsyncExc`, not a signal, not a timeout — can unwind it. **It cannot be killed
and it cannot be waited out.**

## Therefore: stop sharing state with it, rather than trying to stop it

The abort is raised from `ticking.__exit__`, which runs **after** the `with` body has completed.
At that moment `results` already holds every worker's `(fork, stats)` — the entire multi-hour
shard write is in hand and undamaged. Only the coordinator session is unusable, because a thread
we cannot kill may still mutate it.

**And merging a fork into a session that has moved is already what this code does.** `run_forked`
forks at base X, the catch-up rebases the session X → tip while the workers run, and the forks
are merged into the moved session at the end. Each of the eight publications above did that,
across one to seven intervening rebases. So a fork's changes are not bound to the session object
that produced them.

That makes a recovery path available that costs no re-work:

1. Abandon the poisoned session. The wedged daemon thread keeps its reference; that is a leak of
   one session per occurrence, and bounded.
2. Open a **fresh** session on the branch.
3. Run the same `_diff_touches` check between the abandoned session's base and the new one, so
   the conflict detection the catch-up would have done is not skipped. Read-only diffs stayed at
   0.1–0.4 s throughout every stall, so this call is not on the wedging path.
4. Merge the forks already in `results` into the fresh session and commit.

This is what "retry" should mean here: not re-running the write, and not killing a thread that
cannot be killed, but **re-homing finished work onto a session no one else holds.**

## Two mitigations that reduce how often it is needed

Both attack the measured trigger — two publications inside one interval — rather than the wedge.

* **Shorten `CATCH_UP_INTERVAL_S`.** The window in which a second publication is fatal is the
  interval itself. 60 s → 10 s narrows it six-fold. Cost is a `snapshot_id`-versus-branch-tip ref
  read per tick, which returns `current` without touching the wedging path.
* **Space publications apart.** The commit path already takes the `tessera-global-commits`
  global concurrency limit. Holding it for a fixed period after each commit makes a minimum gap
  between publications; set that gap above the tick interval and depth 4 becomes unreachable
  rather than merely unlikely. This is the only one of the three that is a guarantee.

Note this contradicts the earlier claim that *"serialising commits … does not address the cause,
because the gap opens during the write rather than during the commit."* That is right about
serialising and wrong about **spacing**: the gap does open during the write, but its depth is set
by how many publications land in it, which is a property of the commit side.

## The fix: re-home the finished forks, and halve the interval

Two changes, and neither tries to stop the wedged thread, because it cannot be stopped.

**1. `rehome_after_a_wedged_catch_up`.** `run_forked` raises from the timer's exit, which runs
after its body, so every worker's fork is already in hand when the abort fires. Only the
coordinator session is unusable. So: abandon it, open a fresh session, walk the skipped range
for anything that touched this group, merge the forks we already have, commit. No shard is
rewritten and no inference is repeated.

`run_forked` now returns `(telemetry, session)` rather than a bare dict. That is deliberate
friction: after a re-home the session the caller must commit is not the one it passed in, and a
dict key would let a caller ignore that silently — which is the same defect class as `catch_ups`
never reaching `ASSEMBLY_SUMMARY`.

**Guarded on the workers having finished.** `ticking` raises from its `finally`, which runs on
the failure path too, and an exception raised there REPLACES the body's. So a worker that died
while the ticker happened to be wedged arrives at the handler looking exactly like a clean run.
Re-homing that would commit a partial write and report success. `workers_finished` is the only
thing that separates them, and `test_a_dead_worker_is_never_rehomed` is the test that keeps it.

**2. `CATCH_UP_INTERVAL_S` 60 s -> 30 s.** Halves the window a second publication can land in.
Not a guarantee, and not claimed as one.

### Verified against icechunk 2.1.1 before any of it was built

The design rests on one assumption — that a fork can be merged into a session it was not created
from. Five arms on a real repository, because if that is false there is no recovery:

| arm | question | result |
|---|---|---|
| A | merge a fork into its own session | works |
| B | rebase the session, then merge (what production already does) | works |
| C | **merge into a fresh session** | works; data lands |
| D | **does re-homing clobber the writers it skipped?** | **PRESERVED** — their chunks and their attrs |
| E | does `_diff_touches` see a same-group commit in the skipped range? | True for the group, False for an untouched one |

D is the one that could have vetoed the design. A recovery that resurrected a stale view of a
neighbouring zone would be far worse than the stall it cures.

### Dev proof: ten coordinators, one store, real S3, two deliberately wedged

The unit tests pin the mechanism; this pins it under the shape production actually runs. One
icechunk repo on S3 in the dev account, 120 zone groups of 8 arrays each so the snapshot objects
are production-sized, **ten coordinators writing concurrently**, staggered starts and different
write lengths so their publications cluster. Coordinators 3 and 7 have their catch-up wedge
after two successful ticks — the 2026-08-31 failure, made deterministic. Everything runs the
REAL `run_forked`, `catch_up_best_effort`, `rehome_after_a_wedged_catch_up` and
`commit_with_rebase`, so the wiring is under test with the mechanism.

| arm | published | failed |
|---|---|---|
| `shipped` (main) | 8 / 10 | 04N, 08N — `CatchUpDidNotStopError` |
| `fix` | **10 / 10** | none — 04N and 08N report `rehomed: true` |

Identical fleet, identical wedges. The two cells main loses, the fix publishes, at commit times
of 1.447 s and 1.472 s against 0.748-2.058 s for the eight that never wedged — re-homing is not
a slow path.

**The oracle can fail.** Every worker writes disjoint chunks carrying its own
`(coordinator, worker)` identity, so a lost fork leaves a visible hole rather than being masked.
Run against a store missing four of the cells it claims, the same verification returns FAIL with
20 specific problems. And the verifier checks every published cell, not only the re-homed ones,
so a re-home that damaged a neighbour would show up: none did.

**Two defects in the harness, both caught before they were believed.** It first committed with
`session.commit` instead of `commit_with_rebase`, so four of ten coordinators failed with
`ConflictError` on the harness's own omission — which reads exactly like the fix breaking them.
And a mid-run SSO expiry killed four coordinators outright, writing no result at all. Neither
was in the subject.

## What is still open: spacing publications apart

Shortening the interval buys odds. The guarantee needs a minimum spacing between publications
set strictly above the tick interval, so at most one publication can land between consecutive
ticks — Robert's numbers are a 30 s interval with publications at least 31 s apart, and the
arithmetic holds provided a tick's own work stays inside the one-second margin.

**It is not built here, and the reason is a prior decision rather than a technical one.** A
minimum spacing needs a fleet-wide lock, and this repo removed exactly that in 2026-08 —
`commit-gate-removal-2026_08.md`, with `_PrefectCommitGate` deleted, `commit_limit_name` gone
from `run_global_campaign`, and a test asserting no knob remains to reintroduce one. That
removal was argued on cost: the gate bound queueing on a 0.5-2.2 s commit and had misled its
own operator when read as a progress signal.

Those reasons do not carry over cleanly, because spacing is a different intervention from a
concurrency cap, and its cost is bounded — ten coordinators finishing at once would delay the
last publication by about five minutes against assemblies measured in hours. But reversing a
documented decision guarded by a test belongs in its own change, with its own review, not folded
into this one.

Two things also worth stating before that is built. A store-side implementation cannot work: two
coordinators reading the same branch tip would both wait the same amount and then both commit
together, so mutual exclusion is genuinely required. And the one-second margin is thin — the
spacing should be derived from `CATCH_UP_INTERVAL_S` rather than written as a second literal, so
the two cannot drift apart.
