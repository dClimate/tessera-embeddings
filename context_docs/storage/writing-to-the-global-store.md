# Writing to the global store: assembly, sessions, credentials and commits

**Assembly is the campaign's last stage and its longest.** It reads a cell's staged inference tiles
out of S3 and writes them into the published Icechunk store as whole shards, then marks the cell
complete. On a dense zone-year that is **4.97 TB across 2.34 M objects**, about three and a half
hours, sixteen worker processes, and — at fleet width — up to ten coordinators doing it at once into
**one repository on one branch**.

Five things went wrong in that stage during August 2026, and they are one document because they are
one machine. A fix to the first raised the write concurrency the second happened under; the third is
what the second's mitigation turned into; the fourth is the gate that used to bound all of them and
no longer exists. The fifth is the index published beside the store, which the same code writes.

| § | what | when |
|---|---|---|
| 2 | assembly ran five of its sixteen forks, on every cell, for weeks | 2026-08-28 |
| 3 | seven of nine assemblies hung before committing, and the fix relocated the wedge | 2026-08-29 → 31 |
| 4 | 227 writes refused as incorrectly signed, in two accounts at once | 2026-08-28 |
| 5 | why nothing bounds committers any more, and what would reopen it | 2026-08-27 |
| 6 | the per-tile registry published beside the store | 2026-08-19 |

---

## 1. The shape of the write

Four things about the write path carry every argument below.

**One repository, one branch tip.** All 120 UTM zone groups live in a single Icechunk repository on
one `main` branch. Commits touch **disjoint groups**, so there are no data conflicts — but every
commit re-serialises the repo-global snapshot through a branch-tip compare-and-swap. **Disjoint
groups and a shared compare-and-swap are different things**, and conflating them is an error this
document has made once already (§5).

**A coordinator forks its session to sixteen workers.** `run_forked` opens a writable session, forks
it, hands one fork to each of sixteen spawned worker processes, and merges their forks back at the
end. The forks pickle across process boundaries; a fork can be merged into a session it was not
created from, which is what makes §3's recovery possible.

**The write is almost all shard writing.** Measured to completion on a dense zone-year:

| | measured |
|---|---|
| shard-write phase | **195.9 min** (3.27 h) |
| merge + commit | **37 s — 0.32% of assembly** |
| total | **196.6 min** (3.28 h) |
| effective read rate | **423 MB/s** |

**For planning, assembly IS the shard write**, with no second phase to model. That also means the
commit is negligible, which is the fact §5 turns on.

**Assembly must finish before the next cell's inference does.** A cluster runs one trailing assembly
thread behind a rolling inference fleet. At the campaign's 250 actors per cluster the margin is
**1.10×** and scale-invariant, because both terms are linear in tiles. The two cross at **275
actors**, which is what caps actors per cluster and therefore why the fleet is 10 clusters rather
than 8. Assembly stays off the critical path, but by 10% rather than comfortably, and a cluster runs
~126 cells in sequence so a persistent deficit compounds rather than averaging out.

---

## 2. The S3 budget clamp cost assembly two thirds of its fork pool

**Status: fixed.** `_s3_budget_split` reduced the assembly fork count to fit a per-fill S3 request
budget. On the global campaign that budget is always far below the requested worker count, so the
clamp bound on *every* assembly.

### What was measured

Every global assembly emits one `ASSEMBLY_SUMMARY` line. Seven existed in the campaign's history to
2026-08-28. All seven:

| run | workers_requested | workers_used | per_worker_s3_cap | fill_wall (h) |
|---|---:|---:|---:|---:|
| 48N-2017 | 16 | 5 | 1 | 5.78 |
| 32N-2017 | 16 | 5 | 1 | 5.74 |
| 47N-2017 | 16 | 5 | 1 | 6.16 |
| 38N-2017 | 16 | 5 | 1 | 5.09 |
| 33N-2017 | 16 | 5 | 1 | 6.29 |
| 43N-2018 | 16 | 5 | 1 | 5.96 |
| 38N-2018 | 16 | 5 | 1 | 5.11 |

Dead flat across six zones and three weeks. **A fixed number that never responds to job size is the
signature of a clamp, not of a resource limit.** Two live assemblies observed at the same time wrote
shards at 264 and 270 per hour despite being different zones with different data volumes — the same
finding seen from outside.

### The arithmetic, and why it was invisible

`run_global_campaign.py` passes `s3_concurrency = TARGET_AGGREGATE_S3_CONCURRENCY // (2 *
n_clusters)`. With the target at 100 and ten clusters that is **5**. `_s3_budget_split` then did
`workers = min(n_workers, budget)`, turning a 16-worker assembly into a 5-worker one and setting each
fork's request cap to 1. The clamp was deliberate and documented; what was not anticipated is that
**the campaign's own divisor makes it bind unconditionally at fleet width.**

Every place a human configures the worker count still said 16. `AssemblyConfig.max_workers` is 16;
`compute_n_workers` returns 16 for these zones; the flow requests 16. Only the emitted `workers_used`
disagreed — and that field pair exists precisely to expose this case, its docstring saying "a fill
quietly running below its requested width is exactly what this pair exposes." **Nothing read it.**

The clamp also silently reverted a deliberate change. `max_workers` was raised from 8 to 16 on
2026-08-06 from a measurement: at 8 the box was half idle, with processor use peaking at 7,443 of
16,384 allocated units and memory at 20 of 64 GiB. The clamp then held the effective count at **5 —
below the 8 that had already been found too small.**

### The fix, and the invariant it gives up

`workers = n_workers`; the budget divides only the per-worker cap. Because the cap floors at 1,
aggregate concurrency becomes `max(budget, n_workers)` rather than `<= budget`. **The floor and the
ceiling cannot both hold.** The old code held the ceiling by dropping forks; the new code holds the
fork count and lets the ceiling give way.

| | per cluster | fleet (10 clusters) |
|---|---:|---:|
| before | 5 × 1 = 5 | 50 |
| after | 16 × 1 = 16 | 160 |

The nominal fleet target is 100, so this runs at **1.6× the target**, bounded by
`AssemblyConfig.max_workers × n_clusters`.

**The justification is asymmetry of cost.** Overshooting the target risks a 503 `SlowDown`, which
retries. Holding the target by dropping forks costs wall-clock unconditionally, on every cell,
whether or not the service was ever going to complain. The repo's own recorded observation puts
`SlowDown` at **800 concurrent PUTs**, so 160 sits at about one fifth of the level where the problem
was actually seen.

### What was not verified

- **No assembly was run with the change**; the expected 2.5–3× speedup is arithmetic over recorded
  processor and wall-clock fractions, not a measurement of the changed code.
- **Resident memory at 16 workers has never been observed.** Each worker holds at most one staged-tile
  slice, about 1–1.5 GB, so 16 workers is *estimated* at roughly 24 GB against a 16 vCPU / 64 GiB
  runner sized explicitly for `n_workers=16` at about 19 GiB. The only *measured* figure is 20 GiB at
  a pool of 8.
- **The `2 *` factor in the campaign's budget divisor is left alone.** Whether the staging and
  published buckets carry independent request budgets was not established.
- **The 800-PUT `SlowDown` figure is the repo's own code comment** and was not traced to a primary
  record. The `t7_ramp.py` PUT-ramp harness exists in `scripts/scale_tests/` but no recorded results
  exist in this repository.

> **Read §4 next, not in isolation.** This change tripled the fork count on every assembly, which
> tripled the number of processes each fetching a storage credential. The credential incident
> happened the same week.

---

## 3. Keeping the assembly session current

**The change.** While an assembly's forks are writing, the coordinator brings its own session up to
the branch tip on the same timer it already reports progress on — but only past commits that did not
touch its own zone group.

**Why.** On 2026-08-29 seven of nine assemblies finished hours of writing and then never returned
from their commit. The trace named the call: `icechunk::asset_manager::fetch_snapshot`, reached from
`session::rebase` via `session::list_nodes`, and it is only reached when the session has fallen
behind the branch. The full incident record — traces, thread stacks, socket evidence and the
eliminations — is in `yield-embeddings`,
`context_docs/crash-recovery/assembly-stalls-before-commit-2026-08-29.md`.

### The trigger, in one line

A fill opens its session, forks it, writes for about three hours, then commits. Every cell that
publishes during those hours puts another snapshot between the session's base and the tip, and the
commit has to walk all of them. **Zero and two never failed; four failed seven times out of seven.**
Each published cell writes two commits, so in a batch of N simultaneous assemblies the k-th to finish
is `2(k−1)` behind: the first two publish, the third onward stalls. Nine started, two published,
seven stalled.

**Three independent confirmations, none of them arranged.** At 05:04Z on 2026-08-30 — while the fix
was being written, with seven assemblies still stalled — a NEW cell published, 50N/2017, and its
trace shows why:

```
05:04:43.496  INFO icechunk::session: Commit started, old_snapshot_id: GH5GAHTPF2TS1QFRGHZ0
05:04:44.086  INFO icechunk::session: Commit done
```

`GH5GAHTPF2TS1QFRGHZ0` was the branch TIP. That session was **zero snapshots behind**: no "Branch tip
has changed", no rebase, no `list_nodes`, no `fetch_snapshot`. Both its commits landed in 0.6 seconds
total. And at 11:20Z a fresh cell stalled — 50N/2018, on a different fill, ten hours after the batch
that produced the original seven — reproducing every element of the discriminator derived from the
earlier stalls, including a third transaction log fetched mid-rebase-loop:

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

**The tally across eleven hours and eight cells is without exception:** every commit at or near the
tip published in about a second; every commit four snapshots behind stalled.

### What was tried, and what the experiment showed

A harness reproducing the production shape — 120 root groups, one forked session pickled to eight
spawned workers, a concurrent publisher landing two commits per cell, icechunk 2.1.1 on real S3 in
the dev account — ran four arms. `depth` is snapshots to catch up on at commit.

| arm | what it does | depth at commit | commit | conflict case | data |
|---|---|---:|---:|---|---|
| `control` | nothing else commits | 0 | 0.57 s | — | PASS |
| `current` | today's code | 4 / 8 / 16 | 1.58 / 2.31 / 2.77 s | **raises** | PASS |
| `b1` | rebase on a timer, unguarded | **0** | 0.66 s | **SILENTLY OVERWROTE** | PASS |
| `b3` | merge into a fresh session | **0** | 0.59 s | **SILENTLY OVERWROTE** | PASS |
| **`b1g`** | **rebase on a timer, guarded** | **0** | **0.56 s** | **raises** | **PASS** |

**Both naive fixes work and both are unsafe.** With a genuine collision — a second writer touching
the same group and year — today's code correctly refuses with `RebaseFailedError`. Under `b1` and
`b3` the commit succeeded and overwrote the other writer without a word. The reason is structural: a
conflict is only ever detected against commits made *after* the session's base, so **moving the base
past a commit is also a decision to stop checking it.**

**The guard is what makes it safe.** Before advancing, diff base→tip and refuse if anything inside
our own group changed. Then the base stays put, the commit's own rebase runs exactly as it does
today, and the collision raises. Measured: `b1g` at depth 16 reached the tip with seven catch-ups and
committed in 0.56 s; `b1g` against a collision blocked five times, arrived at the commit one snapshot
behind, and raised `RebaseFailedError` — identical to `current`.

**Why on a timer rather than once at the end.** A single catch-up just before the commit would walk
the same distance, through the same call, that the commit would have. The value is in never letting
the gap grow, not in closing it once.

**Why before the merge and not after.** Before the merge the session holds nothing — the forks are
still outstanding — so catching up is a pointer move. After it the session holds the whole write, and
catching up would replay it against every intervening snapshot: the expensive path this exists to
avoid.

**Was the oracle able to fail?** Yes, and it was checked rather than assumed. `--drop-fork K` omits
one worker's fork from the merge while still claiming its writes; the verifier reported exactly that
worker's six chunks as missing, under both `b1` and `b3`. An earlier version of the harness could not
have caught this — eight workers wrote the same four chunk positions with values that did not encode
the worker, so an entire lost fork was invisible and every arm "passed". Workers now write disjoint
chunks carrying their own index.

**What this does not claim.** The harness never reproduced the stall: ten standalone attempts before
it and this one all committed normally at every depth, on a laptop against dev S3. So the efficacy
argument is not "the fix made the stall stop"; it is "the stall has only ever been observed above a
depth this fix holds at zero". The safety argument is the one the experiment settles directly. And
**`assemble` is deliberately untouched** — it writes a per-ROI store where it is the only writer, so
its session never falls behind. The upstream bug is still a bug; this removes the precondition, not
the fault.

### Review findings folded in

An adversarial review against a real repository found three things the first version got wrong. All
are fixed; all are recorded because each is a way this class of change fails.

**The guard could be bypassed, and the result was the silent overwrite it exists to prevent.**
`Session.rebase` takes no target snapshot — it advances to whatever the tip is when it runs — so a
commit landing between the diff check and the rebase is skipped without being vetted. Reproduced end
to end. It cannot be undone, because an empty session's rebase never raises whatever it walks over.
So the catch-up now re-checks the range it *actually* crossed and raises `CaughtUpPastAConflict`,
turning a silent overwrite into a loud abort.

**A failed catch-up destroyed the fill, including after the work had succeeded.** The first version
let exceptions propagate, on the stated reasoning that best-effort belonged at the call site — but the
call site did not implement it. The final catch-up runs *after* every worker has returned its fork, so
an exception there discarded a finished multi-hour write. The triggers are real and not all
transient: a node rename anywhere in the store makes `rebase` raise even on an empty session, and a
reset branch makes `diff` raise. `catch_up_best_effort` now logs and tallies those as `failed`, while
still letting `CaughtUpPastAConflict` through.

**The guard read seven of `Diff`'s eight members.** A rename populates `moved_nodes` and nothing else,
so a commit that renamed our own array read as untouched. Both ends of each move now count, and all
eight members are read by direct attribute access — the previous `getattr(diff, name, None) or set()`
would have let a future field rename silently disable a whole class of check.

**One behaviour worth knowing rather than fixing.** A block latches: the checked range only grows once
the base stops moving, so one same-zone commit early in a fill makes every later tick report
`blocked`. That is safe — it is exactly today's behaviour — but it means the mechanism switches off
for the rest of that fill. Advancing to the last snapshot before the offending commit would keep most
of the benefit and is not possible today, because `rebase` accepts no target snapshot. That is a
second reason to want the upstream fix.

### A second review, of the tests rather than the code

The tests were then reviewed the same way — by mutating the source thirty times and checking which
tests noticed. **Four of them noticed nothing**, and the pattern is worth keeping.

**The mechanism was not connected to anything a test looked at.** Deleting the `catch_up=` argument
from the single production call site passed the entire suite. Every test of the catch-up exercised a
function that, as far as the suite was concerned, need never be called.

**The ordering test could not detect the inversion it forbade.** It asserted
`has_uncommitted_changes is False` at catch-up time — but that is `False` under BOTH orderings when
the worker hands back an untouched fork, so moving the final tick after the merge left it green.

**No test had more than one commit in the gap.** Every catch-up test created exactly one intervening
snapshot, so a guard inspecting only the newest commit looked correct. Production is explicitly
multi-commit — the depths measured above are 4, 8 and 16.

**Four of the six enumerated `Diff` fields were unpinned**, and the conflict solver was not pinned at
all: swapping `ConflictDetector` for `BasicConflictSolver` — which would resolve a collision silently,
the exact failure being guarded against — passed everything.

**The lesson worth carrying.** Every one of these tests passed, was readable, and named the right
property in its docstring. What they lacked was **an oracle that could move**: an assertion whose
value differs between the correct and the broken implementation. Mutating the source and watching
which tests go red is cheap, and it is the only thing that establishes that.

### Verified at the shape production actually runs

Every arm above tested the harness's own restatement of the mechanism, with ONE coordinator catching
up. Two further runs closed both gaps.

**The shipped function, not the design.** The last single-coordinator run imports
`catch_up_best_effort` from the branch and calls it, against a real repository on dev S3:

| run | catch-ups | depth at commit | commit | outcome |
|---|---|---:|---:|---|
| 16 snapshots behind, no collision | 6 `advanced`, 6 `current` | **0** | 0.59 s | all 48 chunks correct, neighbours intact |
| 2 behind, deliberate collision | 5 `blocked`, 1 `current` | 1 | 0.61 s | **`RebaseFailedError`** — refused, other writer's value survives |

**Eight coordinators catching up at once**, which is the question that matters most: **the catch-up
multiplies calls to the very path that stalls** — `rebase` → `list_nodes` → `fetch_snapshot` — from
about one per assembly to one per tick per assembly. If the fault were sensitive to concurrent
traffic on that path rather than to depth, the fix would make things worse.

Eight coordinators, all doing catch-ups, all writing forks, all publishing into one store, staggered
starts, node hierarchy matched to production at 120 groups of eight arrays each. **All eight
published. All at depth 0. 384 chunks verified, none wrong, no cross-clobbering, and all sixteen
commits on the branch.** About 65 catch-up calls went through the stalling path inside two minutes —
a call DENSITY roughly sixteen times production's, since production spreads its ticks over three
hours — and none of them hung. And **the same fleet WITHOUT the fix reaches the dangerous depths**,
which is what makes the result mean something rather than describing a configuration that was never
at risk:

| arm | depths reached at commit |
|---|---|
| catch-ups off | 0, 2, **4, 4, 4, 6, 6, 7** — six of eight at or above the level that fails in production |
| catch-ups on | 0, 0, 0, 0, 0, 0, 0, 0 |

**Is any of this scale-dependent? Partly — and it matters which part.** Matched to production: the
node hierarchy that `list_nodes(/)` walks, the number of concurrent coordinators, two commits per
cell, real S3, icechunk 2.1.1, and forks pickled across spawned processes. **Not** matched, measured
rather than assumed: the snapshot object — the thing the stalling call downloads — is **33 KB here
against production's 159 KB**, because production's also carries manifest references accumulated
across ~25 published cells. And the fork phase is seconds rather than three hours.

*Safety* is semantic, not dimensional: whether a fork created before a rebase still merges correctly,
whether the guard refuses on a collision, whether anything clobbers a neighbour — these are
path-and-coordinate addressing and set membership on path strings, and none has a branch on object
size. *Efficacy* comes from PRODUCTION, at full scale: depth 0 or 2 published in about a second on
five occasions; depth 4 stalled on eight, across eleven hours and two separate batches ten hours
apart. **One correction worth stating plainly: the error has NOT been regenerated at small scale.**
The control fleet reached depth 7 and still committed normally, as did every single-coordinator arm
up to depth 16. The harness reproduces the *condition*, never the *failure*.

### Where the code lives, and why it is its own module

`storage/session_catch_up.py`. The feature was written inside `shard_writer.py` and taken out once
the review rounds made its shape clear: a constant, an error type, a predicate, the operation, a
safety wrapper and a timer — 220 lines with one invariant between them, sitting in a 1,035-line
module about writing shards.

**The invariant, stated once in that module's docstring:** a session's base may only move past
commits that have been checked against the group being written, and it must never be left moved
without that check having happened.

Every review finding so far has been an attack on exactly that sentence — the diff/rebase race
crosses commits without checking them; the type-keyed error handler left the base moved after a
failed check; the missing `moved_nodes` field made a check incomplete. Putting them in one file under
one stated invariant is what makes the next such finding land somewhere obvious. `shard_writer` keeps
only the wiring.

---

### Production, 2026-08-31: the mechanism held, and the residual risk fired twice

The campaign restarted onto this code at 22:45Z on 2026-08-30 with ten coordinators. Eight cells
published between 01:39Z and 02:39Z, and their own `catch_ups` tallies say how far each would
otherwise have had to rebase:

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

`commit_s` does not rise with the number of advances. The catch-up is not spreading the cost of a
deep rebase across many shallow ones; **it is avoiding it.**

#### The earlier arithmetic was wrong: TWO publications inside an interval, not three

An earlier version of this document said *"if three commit inside one `CATCH_UP_INTERVAL_S`, the
third arrives four snapshots behind"*, and reassured with *"never more than two inside any 60-second
window"*. **Two is already enough, and that error is why the risk looked further away than it was.**
A publication is TWO snapshots — the `Run …: fill` commit and the `mark … complete` commit. Depth is
counted from a coordinator's last successful advance, so:

    one publication inside the interval   -> depth 2   (safe, 36 for 36)
    two publications inside the interval  -> depth 4   (the depth that fails)

Corrected, the 2026-08-29 data — two commits inside a window, repeatedly — predicts the failure that
then happened.

#### What actually fired, twice, and the census that makes it unambiguous

At 02:38:42Z and 02:38:53Z two cells published eleven seconds apart. Only two coordinators were still
in a fork phase and therefore exposed. **Both wedged. Two for two.**

| depth attempted | attempts | wedged |
|---|---|---|
| 2 (one publication) | **36** | 0 |
| 4 (two publications) | **2** | **2** |

The 36 successes are every `Caught up` line in the run, across eight zones and six rounds; each
crossed exactly one publication. This reproduces the original 8-for-8 failure at depth 4 exactly, in
a different call site, and settles that **the fix relocated the wedge without changing it.**

#### The wedge is unbounded, so the stop timeout is not the lever

The obvious cheap fix — raise `CATCH_UP_STOP_TIMEOUT_S` above 30 s — is dead. **The wedged thread
never returned.** No further `Caught up`, no failure line, nothing, in the 27+ minutes the run stayed
alive afterwards. That matches the 14,396 s the original stall sat. The thread is blocked inside a
pyo3 call into Rust, where no Python mechanism — not `PyThreadState_SetAsyncExc`, not a signal, not a
timeout — can unwind it. **It cannot be killed and it cannot be waited out.**

#### Therefore: stop sharing state with it, rather than trying to stop it

The abort is raised from `ticking.__exit__`, which runs **after** the `with` body has completed. At
that moment `results` already holds every worker's `(fork, stats)` — the entire multi-hour shard write
is in hand and undamaged. Only the coordinator session is unusable, because a thread we cannot kill
may still mutate it.

**And merging a fork into a session that has moved is already what this code does.** `run_forked`
forks at base X, the catch-up rebases the session X → tip while the workers run, and the forks are
merged into the moved session at the end. So a fork's changes are not bound to the session object
that produced them. That makes a recovery available that costs no re-work: abandon the poisoned
session (the wedged daemon thread keeps its reference — a leak of one session per occurrence,
bounded); open a **fresh** session on the branch; run the same `_diff_touches` check between the
abandoned session's base and the new one, so the conflict detection the catch-up would have done is
not skipped (read-only diffs stayed at 0.1–0.4 s throughout every stall, so this call is not on the
wedging path); merge the forks already in `results` and commit.

**This is what "retry" should mean here:** not re-running the write, and not killing a thread that
cannot be killed, but **re-homing finished work onto a session no one else holds.**

#### The fix: re-home the finished forks, and halve the interval

**1. `rehome_after_a_wedged_catch_up`.** `run_forked` now returns `(telemetry, session)` rather than a
bare dict. That is deliberate friction: after a re-home the session the caller must commit is not the
one it passed in, and a dict key would let a caller ignore that silently — which is the same defect
class as `catch_ups` never reaching `ASSEMBLY_SUMMARY`.

**Guarded on the workers having finished.** `ticking` raises from its `finally`, which runs on the
failure path too, and an exception raised there REPLACES the body's. So a worker that died while the
ticker happened to be wedged arrives at the handler looking exactly like a clean run. Re-homing that
would commit a partial write and report success. `workers_finished` is the only thing that separates
them, and `test_a_dead_worker_is_never_rehomed` is the test that keeps it.

**2. `CATCH_UP_INTERVAL_S` 60 s → 30 s.** Halves the window a second publication can land in. Not a
guarantee, and not claimed as one.

**Verified against icechunk 2.1.1 before any of it was built.** The design rests on one assumption —
that a fork can be merged into a session it was not created from. Five arms on a real repository,
because if that is false there is no recovery:

| arm | question | result |
|---|---|---|
| A | merge a fork into its own session | works |
| B | rebase the session, then merge (what production already does) | works |
| C | **merge into a fresh session** | works; data lands |
| D | **does re-homing clobber the writers it skipped?** | **PRESERVED** — their chunks and their attrs |
| E | does `_diff_touches` see a same-group commit in the skipped range? | True for the group, False for an untouched one |

D is the one that could have vetoed the design. A recovery that resurrected a stale view of a
neighbouring zone would be far worse than the stall it cures.

**Dev proof: ten coordinators, one store, real S3, two deliberately wedged.** One icechunk repo on S3
in dev, 120 zone groups of 8 arrays each so the snapshot objects are production-sized, ten
coordinators writing concurrently, staggered starts. Coordinators 3 and 7 have their catch-up wedge
after two successful ticks — the 2026-08-31 failure, made deterministic. Everything runs the REAL
`run_forked`, `catch_up_best_effort`, `rehome_after_a_wedged_catch_up` and `commit_with_rebase`:

| arm | published | failed |
|---|---|---|
| `shipped` (main) | 8 / 10 | 04N, 08N — `CatchUpDidNotStopError` |
| `fix` | **10 / 10** | none — 04N and 08N report `rehomed: true` |

Identical fleet, identical wedges. The two cells main loses, the fix publishes, at commit times of
1.447 s and 1.472 s against 0.748–2.058 s for the eight that never wedged — re-homing is not a slow
path. **The oracle can fail:** every worker writes disjoint chunks carrying its own
`(coordinator, worker)` identity, so a lost fork leaves a visible hole; run against a store missing
four of the cells it claims, the same verification returns FAIL with 20 specific problems. And the
verifier checks every published cell, not only the re-homed ones, so a re-home that damaged a
neighbour would show up: none did.

**Two defects in the harness, both caught before they were believed.** It first committed with
`session.commit` instead of `commit_with_rebase`, so four of ten coordinators failed with
`ConflictError` on the harness's own omission — which reads exactly like the fix breaking them. And a
mid-run SSO expiry killed four coordinators outright, writing no result at all. Neither was in the
subject.

#### What is still open: spacing publications apart

Shortening the interval buys odds. **The guarantee needs a minimum spacing between publications set
strictly above the tick interval**, so at most one publication can land between consecutive ticks —
a 30 s interval with publications at least 31 s apart, and the arithmetic holds provided a tick's own
work stays inside the one-second margin.

**It is not built, and the reason is a prior decision rather than a technical one.** A minimum spacing
needs a fleet-wide lock, and this repo removed exactly that (§5): `_PrefectCommitGate` is deleted,
`commit_limit_name` is gone from `run_global_campaign`, and a test asserts no knob remains to
reintroduce one. That removal was argued on cost — the gate bound queueing on a 0.5–2.2 s commit and
had misled its own operator when read as a progress signal.

Those reasons do not carry over cleanly, because **spacing is a different intervention from a
concurrency cap**, and its cost is bounded: ten coordinators finishing at once would delay the last
publication by about five minutes against assemblies measured in hours. But reversing a documented
decision guarded by a test belongs in its own change, with its own review.

Two things also worth stating before that is built. A store-side implementation cannot work: two
coordinators reading the same branch tip would both wait the same amount and then both commit
together, so mutual exclusion is genuinely required. And the one-second margin is thin — the spacing
should be derived from `CATCH_UP_INTERVAL_S` rather than written as a second literal, so the two
cannot drift apart.

Note this contradicts an earlier claim in this document that *"serialising commits does not address
the cause, because the gap opens during the write rather than during the commit."* That is right
about serialising and wrong about **spacing**: the gap does open during the write, but its depth is
set by how many publications land in it, which is a property of the commit side.

---

## 4. The 28 August storage-credential incident

**Audience: anyone, including people who have never worked on this pipeline.** Terms are defined
where they first appear.

**One-paragraph summary.** For about an hour and three quarters on the morning of 28 August 2026,
Amazon's storage service rejected a large number of our write requests, saying the requests were not
correctly signed. Four units of work lost their output and had to be redone; no data was corrupted
and nothing was permanently lost, so the cost was time rather than data. We can prove the problem was
confined to one of the several software libraries we use to talk to storage, which rules out a
general fault at Amazon. **We have not established the root cause.** We did find and fix a related
configuration mistake of our own, and we added logging that will identify the cause if it recurs.

### Background: what has to go right for one write to succeed

**Credentials.** Nothing writes to S3 without proving who it is. The proof is three short strings: an
**access key** (a public identifier, safe to write in a log), a **secret key**, and a **session
token**. They are issued together, they expire after a set period, and they must be used as a
matching set.

**Request signing.** Every request is signed: the sender combines the request with its secret key to
produce a signature, and Amazon recomputes that signature to check it. If the two do not match,
Amazon rejects the request with **`SignatureDoesNotMatch`**. That error does not mean "you lack
permission" — it means only that the two signatures did not agree. **It does not say why**, and the
possible causes are genuinely different in kind: the three strings may not have been a matching set,
or the request may have been assembled or signed in a way the service computed differently.

**Several libraries, one destination.** Different parts of the pipeline reach S3 through different
libraries. Four are relevant: **Icechunk** (which stores the dataset), and **boto3**, **s3fs** and
**GDAL** (used for reading source imagery and supporting files). They all talk to the same service,
in the same processes, at the same time — which turns out to be the key to interpreting what
happened.

### What happened

Between 07:04 and 08:49 UTC, the fleet took **227 refused writes**, each reported as
`SignatureDoesNotMatch`. Four zone-year units lost their output and were redone automatically. Their
inputs were untouched, so the loss was compute time.

**The decisive observation: every one came from Icechunk, and specifically from Icechunk writing
data.** None came from the other three libraries, which were all working hard throughout the same
period.

| library | rejections in the window |
|---|---:|
| **Icechunk** (dataset writes) | **227** |
| boto3 | 0 |
| s3fs | 0 |
| GDAL | 0 |

This is what rules out a general fault at Amazon. A signing or validation problem in Amazon's storage
service would not single out one library's writes while sparing three others running in the same
processes against the same service.

> **How the count was reached, and two numbers withdrawn.** 227 is the number of log lines naming a
> refused chunk write (`write_chunk`, and `store::set`, which agree exactly), and those came from
> **210 separate machines**. Counting more broadly, 805 lines mention the error at all — each failure
> emits several lines of context, and this log group sometimes writes a line twice — and those span
> 272 machines. **The two figures pair with different populations**, so quoting 227 against 272, as
> an earlier version did, is arithmetically impossible. **An earlier version said 516 rejections.
> That figure was the non-event remainder — the context lines — not a count of failures.** It also
> credited GDAL with one incidental rejection. **There were none:** the line matched a search for
> "gdal" only because that substring appears by chance inside an S3 extended request id
> (`...HoEH7GtwggdalSnjlC4Q...`). The line was itself an Icechunk chunk write.

Four other explanations were checked directly and all came back at zero for the same period:
expired credentials (`ExpiredToken`, `TokenRefreshRequired`), unrecognised credentials
(`InvalidAccessKeyId`, `InvalidClientTokenId`), clock drift (`RequestTimeTooSkewed`), and
rate-limiting (`SlowDown`, `ThrottlingException`). So: valid, unexpired credentials; identifiers
Amazon recognised; no complaint about our clocks; no rate-limiting; a single error type; refused
writes on 210 separate machines; and it stopped on its own.

### Why two separate accounts failed at the same moment

This was the most puzzling feature of the incident, and the part most likely to mislead. The failures
spanned **two storage buckets in two different AWS accounts** — one of ours, and one belonging to a
delivery partner. Those two destinations use *completely different credentials*, from different
sources. Two independent credential systems failing in the same window would be a remarkable
coincidence.

It is not a coincidence, because **both destinations are written through the same layer.** They are
reached by different application code — the standalone store by one assembly entry point, the
published store by another — but both hand the work to the same storage library, and both obtain
their credentials through the same single function in our code. The credential *source* differs per
destination; the credential *delivery mechanism* does not. **That means a defect in the shared
mechanism explains simultaneous failures in two accounts with no Amazon-side event required.** This
is the strongest structural conclusion the incident supports.

### What we found in the storage library

Icechunk lets an application supply a small function that hands over a fresh credential whenever one
is needed, and is supposed to hold on to it until it expires. There is an open bug report against
Icechunk — [issue #2077][2077], filed 15 April 2026 — showing that this holding-on works **only
once**: because of how the caching code is written, the stored credential can never be replaced after
the first one expires. The report describes Icechunk then asking the application for a credential on
**every storage request** for the remaining life of the process, and its author measured **301,000
credential requests in four hours** from a single workload. An Icechunk maintainer replied that this
"sounds like intended behavior," so it is disputed as a bug and any mitigation has to be ours.

> **We have not verified that frequency for our own workload, and it is disputed.** Review argued
> that Icechunk's S3 client caches the credential between requests, which would make the re-fetching
> happen per *deserialisation* of a session — often, in a workload that ships a session per task, but
> not once per request inside one long-lived worker like ours. Settling it needs a reading from
> inside a running assembly, which the instrumentation shipped alongside this change can now provide.
> **What the mitigating flag does is not in doubt** — it removes the credential fetch a freshly
> deserialised session would otherwise make — only how much that saves.

[2077]: https://github.com/earth-mover/icechunk/issues/2077

### What was wrong on our side, and what we changed

Assembly splits its work across sixteen worker processes, each of which receives a copy of the
storage connection. **We have two code paths that do this**, one for each of two kinds of
destination. They should have been configured identically. They were not:

| code path | destination | setting before | setting after |
|---|---|---|---|
| assembling into a standalone store | an interim dataset | **on** | on |
| assembling into the main dataset | **the published dataset** | **off** | **on** |

The second path is the one the production campaign actually uses, and it was the one with the setting
switched off — so each of its sixteen worker processes started with no cached credential and had to
fetch one before it could write. The setting is now switched on for both paths. It is switched on
**explicitly at each of these two places rather than made the default**, because it has a real cost
and about ten other places in the code open the same dataset without ever copying the connection to a
worker.

> **A correction, in case you see the earlier version quoted.** While investigating I stated that
> *no* part of our code switched this setting on. That was wrong — the first of the two paths always
> had. I had checked the callers of one function and missed that a sibling path reaches the same
> setting through a differently-named one. The real defect was an inconsistency between two nearly
> identical paths, which is a sharper finding than the one I reported.

**The cost of the setting, and why it is applied per location.** Icechunk's documentation warns that
with this setting on, the credential is stored inside the connection object — and "they can be sent
over the network if you pickle the session/repo." In other words, switching this on puts a live
secret key and session token inside something that gets copied elsewhere. Where that copy travels is
the whole question:

| how the copy travels | exposure |
|---|---|
| **to a worker process on the same machine** (what both changed paths do) | The copy goes through an operating-system pipe to a child of the same process, on the same machine, which already holds the credential in memory. **No new exposure.** |
| to a worker on another machine, over the network | A live secret in transit — and those systems can also spill data to disk under memory pressure, leaving a credential written down. |

Making the setting a global default is the tempting shortcut and is the one that would put secrets on
the network for paths that gain nothing from it.

**What this change does and does not do.** It does not risk credentials going stale: Icechunk's
documentation is explicit that once the stored credential expires, the stored copy is no longer used
and the live fetch takes over. It only helps for one credential lifetime, and that is shorter than it
sounds — the task-role path promises fifteen minutes; **the published-store path promises five**
(its interval is deliberately short so a callback lands inside the credential library's
mandatory-refresh window). Against an assembly of about three and a half hours, that is roughly 2% of
the run. **This is a consistency fix and a reduction in wasted work — it is not a cure for a
long-running job**, and I described it as one before reading the documentation carefully. And it is
not the established cause.

### Why the "too many credential requests" theory is not the leading explanation

Ranking it first was an overreach, and the reason is worth recording. The 301,000 figure came from an
application whose credential function made a network call every time it was asked. **Ours usually
does not** — it reads an already-fetched credential out of a small in-memory cache, with refreshes
happening quietly in the background, so the cost of being asked repeatedly is normally acquiring a
lock inside the process rather than a network round trip. Two exceptions matter and an earlier
version missed them: the very first call in a process performs the role assumption, and a call
landing in the credential library's mandatory-refresh window blocks on a real token-service request.
Those are a handful of calls per process, not one per storage operation. **Lock contention makes
things slow; it does not produce incorrectly signed requests**, which is the reason this is not the
leading explanation.

Three candidate explanations remain, and none is proven:

1. **A defect in Icechunk's own request-signing code under heavy parallel use.** Icechunk allows a
   large number of storage operations in flight at once, multiplied by sixteen worker processes,
   multiplied by ten simultaneous regions. This campaign puts far more pressure on that code than the
   workload in the bug report did.
2. **A race in the handoff of credentials** between the part that refreshes them and the part that
   attaches them to requests. The new logging tests this one directly.
3. **A transient fault on Amazon's side.** Weakest on the library-isolation evidence — but a later
   reading strengthens it. At 16:20 UTC the same day, the campaign was running **five** simultaneous
   assemblies rather than four, and — because of §2's fix — each was using sixteen worker processes
   rather than five. **Storage write concurrency was therefore higher than during the incident.**
   Across the following hour and 1,041,266 log lines there were **zero** `SignatureDoesNotMatch`
   rejections. That is real evidence against explanation 1 and for explanation 3. One hour is only
   one hour, but it is the first independent measurement that separates the candidates.

### What would settle it, and one thing not to do

Every process now records **which credential it is using**: the access key (a public identifier, and
the one Amazon's own audit log indexes on), the expiry, the process id, and a sequence number. The
first credential a process uses and every subsequent change are recorded at normal log level; routine
repeats at debug level. The secret key and session token are **never** logged. If this recurs, that
record answers the one question the current evidence cannot — whether the credential in use changed
at the moment of failure. **One unchanged credential across the failure** rules out the handoff and
means no retry could have helped; it does not choose between the other two. **A different credential
appearing moments before the first rejection** implicates the handoff — and only then does adding a
retry make sense, with randomised delays, because sixteen workers sharing a fixed delay would retry
in synchronised bursts.

**Until one of those is observed, do not add retries here.** A retry was drafted during this
investigation on the belief that these writes had none, and **withdrawn**: they already retry up to
ten times with increasing delays, and these rejections exhausted that allowance rather than arriving
unprotected. A second layer on top would have added delay and hidden the signal.

### A note on the new logging's own concurrency

Worth recording because the same mistake was made three times over, in a small piece of code, and
review caught every one. The pattern is a useful one to recognise: **each version tried to say
something about the order of events using information that did not record it.**

The logging keeps a small note of which credentials a process has already reported, so a credential
is announced once rather than on every request. The first version compared the incoming credential
against the last one recorded. The second compared expiry times, on the theory that a later expiry
meant a newer credential. **Both were wrong, for the same reason: nothing available at that point in
the code records when a credential was obtained.** A thread can pick up the old credential, be paused
while another thread fetches and records a newer one, and then arrive carrying the old credential
*with a later expiry stamped on it* — because the expiry is calculated after the credential is picked
up. Either version would report a credential change backwards and another forwards: three change
records for one real change, corrupting exactly the signal the logging exists to provide.

A third version kept the "have I mentioned this one yet" test but still reported each new credential
as a change *from* the previous one. Review found that this breaks in the one case the first two did
not cover: at the very start of a process there is no history to match against, so a straggler
carrying the old credential is treated as new, the record reads "changed from the new credential to
the old one", and the superseded credential is left standing as the apparent current one. This case
is not hypothetical on the path we changed: a worker process that inherits a credential only calls
the credential function once that credential expires, which is exactly the moment a refresh can
interleave.

**The fix was to stop making ordering claims of any kind.** Each record now says only "this process
began using credential X", never "X replaced Y". Read in sequence order for one process, those
records *are* its credential history — which is what can honestly be established. Ordering of the log
lines themselves is handled separately by a counter stamped at the moment the note is updated; that
counter orders the moments the code *decided* to hand over a credential, and is not proof of the
order the credentials were *used*. The reviewer's alternative — hold a lock across the credential
fetch — was declined: that fetch can block on a network call, so holding a lock across it would put
every storage request in the process behind a network round trip.

**Versions.** We run Icechunk **2.1.1**. The only later release is **2.1.2** (29 July 2026), and its
release notes contain no change to the credential-caching code, so upgrading does not address issue
#2077. That issue has no linked fix.

---

## 5. Commits are ungated, and what would reopen that

**Decided 2026-08-27 by Robert**, on the reasoning that the gate guarded a minuscule possibility, that
if it does occur the slowdown is manageable and easily overcome, and that an extra gate is complexity
the campaign does not need. **The `tessera-global-commits` Prefect global concurrency limit and all
of its plumbing are removed.** This section records the measurement that supports it, the evidence
the gate was built on — which is *not* retracted — and the scale at which that evidence would matter
again.

### What the gate was

A Prefect global concurrency limit, one slot held around `session.commit`, forwarded from
`run_global_campaign` to every fill as `commit_limit_name` and upserted at preflight to
`min(max_parallel_clusters, 8)`. It existed because ADR-008's run 1 measured rebase retries scaling
with the number of racing writers.

**The run-1 curve, kept because it is the reopen criterion:**

| simultaneous committers | rebase retries | commit duration |
|---:|---:|---:|
| 2 | ~0.5 | 0.5 s |
| 8 | ~3.5 | 1.3 s |
| **16** | **~7.5** | **2.2 s** — breached that experiment's own ≤2×-serial acceptance bar |
| 120 | ~58 | 15 s |

Cross-group conflict-freedom held at every N: **zero unresolvable conflicts**, including at 120, six
times the campaign's theoretical ceiling.

### Why it goes

**1. What it actually bound was queueing, not risk.** ADR-008 carried a note from 2026-07-30 saying
that at 8 clusters "the limit equals the number of possible committers, so it cannot bind at all".
**That note is wrong**, and correcting it sharpens rather than weakens the case: the ceiling is `2N`,
because a cluster's feeder can commit a terminal plan while its trailing assembly commits a cell — so
8 clusters reach 16 committers against a limit of 8, and the gate could queue about half of them. **A
queued committer waits for a commit, and a commit is the 0.5–2.2 s measured above.** So the
protection given up is a short wait on a sub-second-to-seconds operation that happens once or twice
per zone-year, inside an assembly measured in hours.

**2. Measured live.** One completed assembly (`steady-otter`, 19:03:20Z — the publication that took
the store from 77 to 78 zone-years): `commit_s` **1.0 s**, `attrs_commit_s` **0.3 s**, `merge_s`
0.0 s, against `fill_wall_s` of **21,447 s (5.96 h)**. The commit is **1.3 s out of a six-hour
assembly** — 0.006% of it. And because `commit_s` is measured *around* the gated call, **the gate
wait is inside that 1.0 s**, which makes it direct evidence of zero queueing rather than an
inference.

**3. ~~There is structurally almost nothing to contend for.~~ WITHDRAWN — this was false, and it was
the load-bearing half of the argument.** The first version of this reasoning said the embeddings
store is "per ROI, so concurrent commits from different zones land in different Icechunk repositories
and cannot share a branch tip at all." **That describes `BucketPaths.store_for()`, the SINGLE-ROI
path.** The global campaign uses `BucketPaths.global_store()`, whose own docstring says the opposite
and contrasts the two functions explicitly. I read the wrong one and did not check. Two reviewers
found this independently, which is the correct outcome and the reason it is recorded rather than
quietly patched. What is actually true is §1's first paragraph.

**4. The observability cost was real.** The gate's `active_slots` was read as a progress signal on the
night of 2026-08-27 and gave a wrong answer twice — first as "four cells stuck in hours-long
commits", then as evidence about which cluster was assembling. Both were wrong, because a commit
lasts ~1 s while only ONE cluster had an assembly in flight. **Corrected in place: I first attributed
this to a lagging counter**, citing the sibling ingest gate as reading "51 held against 45 actually
live". That comparison used a monitoring query with the wrong flow name; queried correctly, the
ingest gate read **56 held against 56 genuinely RUNNING cells**. So the gates' counters are accurate,
and what the commit gate's `active=4` was holding is **unexplained, not stale**. The point stands for
removal — a 1 s commit with zero measured queueing cannot be bound by a limit — but the "counter
lags" reason is withdrawn.

### The decision

Grant the reviewers everything: all 120 zone groups share one repo and one branch tip, every commit
re-serialises the repo-global snapshot, and at ten clusters the `2N` ceiling is 20 committers. Then
read what run 1 measured at that concurrency and far beyond. **N ≥ 16 is a real threshold, and what
it thresholds is a ≤2×-serial acceptance criterion: a commit taking 2.2 s instead of ~1 s.** That is
the whole consequence. There is no correctness cliff at 16, or anywhere up to 120 — a
`RebaseFailedError` needs a genuine chunk conflict, and commits touch disjoint zone GROUPS even while
sharing a branch tip. The same-zone case that could conflict is serialised elsewhere: the driver
partitions zones across clusters, a cluster's trailing assembly is single-threaded, and
`commit_year_attrs` carries its own `tries=8` retry for exactly that collision.

**Weighed against that, the gate costs more than it saves.** It bounds a slowdown measured in
seconds, and it misled its own operator twice in one night. A control that cannot fail dangerously
but can be misread is not free.

One correction to the "collisions are rare" intuition, recorded so nobody rebuilds the estimate
without it: **the commit RATE is far higher than one per assembly.** Terminal cells — all-ocean or
no-optical — commit through `mark_zone_year_empty`, and **72 of the first 78 completions were
terminal**. Collisions are likelier than an assembly-only model suggests. It does not change the
conclusion, because the cost of one is seconds, but it is the part of the reasoning most easily got
wrong.

### The threshold, in cluster counts, and the detector

| `max_parallel_clusters` | committer ceiling (2N) | versus the N≥16 breach |
|---:|---:|---|
| 7 | 14 | below — safe |
| **8** | **16** | **AT the breach** |
| 10 (the campaign) | 20 | above |
| 16 | 32 | far above |

**So the boundary is `max_parallel_clusters` ≥ 8, and the safe cap is ≤ 7.** (An earlier version of
this section said "above ~12", which is arithmetically wrong — 2N reaches 16 at N=8 — and would have
left every configuration from 8 to 12 at or above the breach with no gate. Caught in review. It also
contradicted the ≤7 bound stated elsewhere in the same document: **I swept the withdrawn premise but
did not re-check the arithmetic that depended on it**, which is exactly the failure mode a sweep is
supposed to prevent.)

Two other conditions would put committers back in contention regardless of N: more than one cluster
owning the same zone, which would put two committers on one zone group; and a change making assembly
commit per-shard or per-tile rather than once per zone-year.

**The detector is the `COMMIT <secs>: <message>` line in `commit_with_rebase`.** Every commit goes
through that function — the assembly path and the terminal path both — so it is the only place that
sees all of them. `commit_s` in `ASSEMBLY_SUMMARY` was the obvious candidate and is **not
sufficient**: a terminal cell marks itself through `mark_zone_year_empty` and returns without ever
reaching `assemble_global`, so the summary never fires for it. Since terminal cells were 72 of the
first 78 completions, that detector would have missed the dominant source of concurrent commits and
stayed silent exactly when the gate should be restored. **A commit time drifting above a couple of
seconds is the signal.**

One line per COMMIT is not one per zone-year: an assembled cell emits **two** (shard data, then
completion attrs), a terminal cell **one**. So the line is a reliable signal of commit LATENCY, but
raw line counts must not be read as cell counts.

**What was removed:** `MAX_SIMULTANEOUS_COMMITTERS`, `commit_limit_name` and its preflight upsert;
`_PrefectCommitGate` and the `CommitGate` type alias; the `gate` parameter through
`write_year_shards`, `commit_year_attrs`, `commit_with_rebase`, `assemble_global`,
`mark_zone_year_empty` and the four `zone_fill` entry points; and `DEFAULT_COMMIT_CAP`, which had no
remaining users. **`FleetGate` itself stays** — the ingest gate is the campaign's pause lever and
uses the same class.

**Cost of being wrong: bounded and visible.** If commits do contend, they rebase (`rebase_tries=1000`)
and the cost shows up as a longer `commit_s` against a six-hour fill. The failure mode is a slower
commit, not a lost or torn one.

---

## 6. The registry published beside the store

The registry is a Parquet dataset beside the embeddings store, **one row per 2048-pixel tile per
year**, written by the same assembly as each cell lands. It exists so two questions can be answered
without opening a petabyte-scale store: a user's *"is my area covered, and how well"*, and a later
infill campaign's *"which tiles were short of imagery, and would more of it help"*.

Location and layout: `<store-stem>.registry/parts/zone=<Z>/year=<Y>/<run_id>.parquet`, a **sibling**
of the store rather than a path inside it, because Icechunk owns every key under its own prefix and
its garbage collection reconciles that prefix against its own manifests.

**Status: the schema and the producer are built and exercised against live campaign runs. The
consumer side — a pipeline that can act on the work list this produces — is not built.**

### What the first live run found

Dispatched 2026-08-19 against `tessera-radar` in `global-tessera-dev`: zone 16S, years 2022 then
2021, 17 live tiles each. Both cells landed and both published a part. The run exposed four defects,
and the most serious was invisible in every log except the one naming the path it wrote to.

**The registry was published beside the wrong store.** Parts landed at
`.../global/tessera.registry/...` while the run was filling `tessera-radar.icechunk`.
`BucketPaths.optical_registry()` called `global_store()` with no argument, so it derived from the
*default* repo basename whatever store the run was writing. The part was valid, 17 rows, and the log
said it had been published — it simply described a zone-year that the store beside it does not
contain. **That is worse than an overwrite:** two stores' parts merge into one dataset under the same
`zone=`/`year=` keys, and no column can tell them apart, so the error is unfalsifiable from the
artifact a consumer was told to read *instead of* the store. Fixed by making the method take the
resolved store URI as a **required** argument — a default would reintroduce the same silent failure
the first time a caller forgot it.

**The depth rule never reached the rows.** Every published row carried `optical_min_obs = None` while
the store root declares 15. `fill_zones_sequential` is a second entry into `assemble_zone_year`, and
it never forwarded the value. The columns that exist to say *how close a refusal came* — `obs_max`,
`median_obs_where_any` — are distances from that line, so without it they cannot be read at all.

**Nothing was recorded for tiles that were only partly refused.** All 17 rows read `embedded = true`
with every measurement null. The actor accumulates refusal reasons over a shard's strips on the
success path as well as the refused one, so a shard that lost part of its land to the depth gate has
always known how much — but only the wholly-refused branch wrote a record. A tile 60% refused
therefore published as covered ground. **Those partial refusals are the bulk of an infill work list**,
not a refinement of it.

**No row said where its tile was.** The access request promises a bounding box per row and none was
produced.

### What the same run confirmed works

- The dataset opens and reads **across parts** with hive partitioning; `zone` (string) and `year`
  (int32) come from the path with no type collision.
- One row per live tile, 17 of 17, with `run_id` and `assembled_at` present on every row.
- Every consistency invariant held: reasons summing to `refused_px`, `eligible_px <= chunk_px`,
  `px_with_any_optical <= eligible_px`.
- `tag_year_complete` correctly **refused** to mint `year-2021-complete` from a 9-zone subset. No
  globally-named completion tag exists.

**Why 16S could not exercise the refusal columns.** 16S/2022 refused **nothing**: all 67,728,022
embedded pixels met the depth-15 rule, and `s1_free_px` was 0 despite the ascending orbit carrying
only 4 of 12 months, because the descending orbit covered all 12. An empty infill work list is the
*correct* answer for such a cell — but it means the run could not distinguish a registry that
measures and finds zero from one that measures nothing. That distinction is now explicit: **a
measured zero is written as `0`, and only a tile whose coverage record never arrived stays null.**

### The second run: verified, and it found one more thing

09S/2021 (10 tiles, western Amazon) and 03S/2024 (4 tiles) landed against the fixed code, chosen as
small fresh cells over persistently cloudy land. The dataset now reads as one table across both parts
— 14 rows, two zones, two years, partition keys typed from the path, and **zero consistency
violations**. Four of the fourteen tiles carry refused pixels while being embedded — the case that
previously recorded nothing — and the worst is 03S/2024 `chunk_59_11`: 42,212 refused of 4,194,304
eligible, 40,852 of them imaged but thin and 1,360 never imaged at all. Both zones are wholly
radar-free, which is correct.

**That run also showed why `median_obs_where_any` had to be replaced as the ranking key.**
`chunk_59_11`'s footprint median is 37 against a line of 15, because the median describes the pixels
that PASSED. Its `median_obs_where_thin` is 5.0 — the refused pixels are about ten observations
short. Ranked by footprint median it would have sat below tiles with no refusals at all; ranked by
thin median it is correctly first.

**And it confirmed the schema-evolution hazard on live data.** The two parts were written by
different builds — 09S by `10f31a1`, 03S by `22b2352` — and `median_obs_where_thin` exists in only
the newer one. The merged read shows the column *because* `zone=03S` sorts before `zone=09S`, so the
newer part's schema is the one inferred. **Had the older zone sorted first the column would have
vanished from the read with no error.** That is why `dataset_schema()` exists and why any
whole-dataset read must pass it. The `code_commit` column is what made the diagnosis possible, which
is its first real use.

**One failure that was not a registry defect.** 03S/2024's first attempt lost all four tiles to
`TypeError: _coverage_record() missing 1 required keyword-only argument`. No commit on the branch
contains that inconsistency — the function and both its call sites move together in every one — and
the tarball on S3 afterwards was correct. **A pin bump pushed while that cell was mid-ingest had CI
replace the Ray code tarball between its ingest and its fill.** Re-dispatched against a stable
tarball, the same cell completed. Two things worth keeping: the failure direction was right, a
required keyword-only argument with no default produced four loud tile failures instead of a registry
of rows measured against a line nobody recorded. And **a green test suite cannot see this class at
all** — the tree was internally consistent; the inconsistency was created by deployment.

### Decisions worth keeping

**Null means "not measured", never zero.** A resumed tile is a synthetic success carrying no coverage
record, and an unreadable marker leaves none. Writing zero there would assert a measurement nobody
took, which is the misreading the whole artifact exists to prevent.

**`embedded` is not coverage on its own.** It says the tile holds embeddings *at all*. "Is my area
covered" is `embedded` together with `refused_px`, and a reader using the first alone will overstate
coverage.

**The schema is declared, never inferred**, and `zone`/`year` are partition keys only — both because
the failures live in the merge, not in any one part.

**One `_coverage_record`, two call sites.** A refused shard's marker and an embedded shard's result
are built by the same function, because comparing the two rows is the entire point and two copies of
those expressions is how they stop meaning the same thing.

**The bounding box is not derived in the registry module.** It comes from
`zone_grid.tile_range_bbox_wgs84`, whose densified 64-sample perimeter is *measured* to contain the
true envelope to within one pixel and which handles the antimeridian. Rows for zones 01 and 60 carry
`west > east` per GeoJSON and STAC — a consumer filtering `west <= lon <= east` silently drops them.

**A part is published only after its cell commits**, and the write is quiet on failure. Before the
commit, failing loudly costs nothing; after it, raising would fail complete work over its index. Every
column is derivable from the store, which is what makes a lost part recoverable and that trade correct
rather than lazy.

### What is still missing

- **The pipeline cannot consume the work list.** Threading an optional set of tile labels through a
  fill is a real change, not a small one: the year record is replaced wholesale and the skipped set is
  derived as live-minus-staged.
- **Multi-contribution year provenance.** A refilled cell should append a contribution — run id,
  timestamp, code identity, input coverage, and the tile labels it wrote — rather than replace the
  original. Sketched and agreed; not built. Safe to do while no campaign has published. See
  [`../inference/minimum-optical-depth.md`](../inference/minimum-optical-depth.md) §10 for the update-cycle
  model this belongs to.
- **Compaction.** Parts are keyed by run so a refill adds one rather than replacing the original.
  Reading the raw parts double-counts a twice-filled cell unless the reader keeps the latest
  `assembled_at` per `(zone, year, tile)`.

---

## Appendix: where this lives in the code

| what | where |
|---|---|
| the sixteen-worker fork split | `storage/shard_writer.py`, `run_forked` and `write_year_shards` |
| the fork-count clamp, and the budget it divides | `inference/assembly.py`, `_s3_budget_split`; `config/assembly.py`, `AssemblyConfig.max_workers` |
| the session catch-up: constant, error type, predicate, operation, wrapper, timer | `storage/session_catch_up.py` |
| re-homing after a wedged catch-up | `storage/session_catch_up.py`, `rehome_after_a_wedged_catch_up` |
| the commit itself, and the `COMMIT <secs>` detector line | `storage/zarr_store.py`, `commit_with_rebase` |
| the two assembly paths that copy a connection to workers | `inference/assembly.py`, `assemble` and `assemble_global` |
| the single place every storage credential is built and logged | `providers/aws/credentials.py`, `_serve_icechunk_credential` |
| the two credential sources (our account; the partner's) | same file: `iam_icechunk_credentials`, `AssumedRoleIcechunkCredentials` |
| the scatter setting discussed in §4 | `scatter_initial_credentials`, threaded through `storage/zarr_store.py` |
| the existing storage retry allowance | `storage/zarr_store.py`, `StorageRetriesSettings` |
| the registry's path, and its required store argument | `config/paths.py`, `optical_registry()` |
| the registry writer and its one record builder | `inference/assembly.py`, `_coverage_record` |
