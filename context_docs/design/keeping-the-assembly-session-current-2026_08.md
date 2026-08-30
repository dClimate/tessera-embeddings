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
