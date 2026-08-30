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
