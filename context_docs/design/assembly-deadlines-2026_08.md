# Bounding a fill's assembly — two budgets, one of them derived from the work

**Status: implemented** (`storage/shard_writer.py`,
`orchestration/runners/sequential_fill.py`, 2026-08-29).
**This is containment for a stall whose root cause is NOT known.** Nothing here
explains why an assembly stops; it stops a stalled one from taking the rest of a fill
down with it. When the cause is found, revisit whether these budgets are still the right
shape.

## What happened

On the night of 2026-08-28/29 the global campaign was running ten concurrent
`fill-zones-sequential` runs, one per ECS Fargate task. **Seven of eleven trailing
assemblies stopped.** The signature was identical on every stuck task:

- All sixteen forked shard-writer worker processes had finished and exited. The
  `ProcessPoolExecutor` was fully torn down — the driver had no worker children, only
  `multiprocessing.resource_tracker`, and only four pipes.
- **No exception was raised.** The runner catches `Exception` around the assembly and
  records a failure; no failure was recorded for any of them.
- No socket had a pending request, so it was not blocked on S3.
- It never reached `commit_with_rebase`, which logs `COMMIT %.2fs` only *after*
  `session.commit()` returns.

So the stall sits somewhere between "every fork has returned its result" and "the shard
commit has landed": the pool teardown, `session.merge()`, the `years_complete` read, or
inside `commit_with_rebase` before its log line. Which of those, and why, is unknown.

**The amplifier.** `_finalize` runs on
`ThreadPoolExecutor(max_workers=1, thread_name_prefix="trailing-assembly")`. One slot, so
a single stalled assembly blocks that fill's assemblies for the rest of its run. Six
fills published nothing for seven hours while still inferring at full rate — the GPUs
were busy, the store was not moving.

## The measurements the budget comes from

The campaign-side forensic record of this incident is
`yield-embeddings/context_docs/crash-recovery/assembly-stalls-before-commit-2026-08-29.md`.
It carries the per-assembly timeline, the ruled-out causes and the reproduction attempts.
This section takes only what the budget needs.

Every row is one zone-year's fork-write phase: how many shards it wrote, and the
wall-clock minutes from the session opening to the last fork returning.

| cell | shards | fork-phase min | shards/min | outcome |
|---|---:|---:|---:|---|
| 44N / 2017 | 7,907 | 147 | 53.8 | stalled after |
| 35N / 2017 | 9,132 | 192 | 47.6 | committed |
| 13N / 2017 | 6,575 | 140 | 47.0 | committed |
| 36N / 2017 | 8,738 | 190 | 46.0 | committed |
| 48N / 2019 | 8,803 | 193 | 45.6 | stalled after |
| 35N / 2018 | 9,132 | 211 | 43.3 | stalled after |
| 33N / 2018 | 8,593 | 199 | 43.2 | stalled after |
| 34N / 2018 | 8,734 | 203 | 43.0 | stalled after |
| 38N / 2019 | 9,100 | 212 | 42.9 | stalled after |
| 47N / 2018 | 9,030 | 213 | 42.4 | stalled after |
| 36N / 2018 | 8,738 | 207 | 42.2 | committed |

**Observed range: 42.2 to 53.8 shards per minute — a spread of only 1.27x.** Measured
consistently, the fork-write phase is a remarkably steady rate. The slowest wall clock in
the table is **213 minutes**.

### A discrepancy worth recording, because the budget was first derived from the other reading

An earlier statement of this table gave two of these cells as far slower — 36N/2018 at
~392 minutes and 35N/2018 at ~400 — which produced a "22 to 46 shards/min" range and a
2.4x spread. Those two figures do not measure this phase. 392 is 15:33 minus 09:01 and 400
is 15:43 minus 09:03: the gap from the *previous* assembly's completion to this one's, which
on a single-slot finalizer includes however long the cell queued. The other nine rows agree
between the two readings to within a few minutes; only these two, and only because they are
the two cells that waited behind another assembly.

The correction does not change the constants below, and it changes their justification in
the safe direction: what was believed to be 3.4x headroom over the slowest fill on record
is really **6.4x**. Both readings are recorded here rather than one quietly replacing the
other, because the budget has to be defensible under either.

The correction *does* retire one claim. The "2.4x spread, driven by contention and per-tile
density" reasoning came entirely from those two rows. At a fixed measurement definition the
observed spread is 1.27x, which is too narrow to attribute to anything. The safety factor
below is therefore padding against variation this table **cannot see** — a denser cell, a
busier campaign shape, a fleet width nobody has run yet — not against a measured spread.
Per-tile cost genuinely does vary between cells (it shows up in the A10G actor
out-of-memory failures clustering in particular fills; see
dClimate/tessera-embeddings#160), but this table does not measure it.

### The other half

**The phase after the forks, on the four cells that committed, took between 0.60 s and
1.61 s** to commit, and 0.2–1.8 s end to end counting the merge and the attrs commit
(`merge_s` + `commit_s` + `attrs_commit_s` from their `ASSEMBLY_SUMMARY` records). That is
the whole of it: one merge and two commits.

## Why two budgets and not one

The two phases have nothing in common.

| | fork-write phase | everything after it |
|---|---|---|
| work | scales with the shard count | fixed: one teardown, one merge, two commits |
| observed duration | 140–213 min | 0.2–1.8 s |
| is this where fills stopped? | no — every fork returned | **yes** |

A single cutoff across both would have to clear three and a half hours, and a cutoff that
long can never notice a phase that should take two seconds. Splitting them is the entire
point of this change; the exact constants matter much less.

### The fork-write budget

```
fork_phase_budget_s(n) = max(FORK_PHASE_FLOOR_S,
                             60 * n / FORK_PHASE_SHARDS_PER_MINUTE * FORK_PHASE_SAFETY_FACTOR)

FORK_PHASE_SHARDS_PER_MINUTE = 20     # half the slowest measured rate (42.2)
FORK_PHASE_SAFETY_FACTOR     = 3
FORK_PHASE_FLOOR_S           = 3600
```

Derived from the work, not the clock, because the phase's duration tracks its shard
count. The divisor is **half the slowest rate measured** (42.2/min), so the safety factor
is padding on top of an already-halved worst case rather than padding towards it.

At the densest cell in the table, 9,132 shards, that is 9,132 / 20 x 60 x 3 = **82,188 s
(22.8 h)**, against a worst observed 213 min (3.6 h) — **6.4x the slowest fill on record**,
or 3.4x under the superseded reading of the table above. Below ~400 shards the floor binds,
so a sparse zone-year still gets an hour.

The factor is deliberately generous, and the reason is what the table cannot show rather
than what it does. Eleven cells at one campaign shape, on two days, span 1.27x. That is far
too narrow a base to predict a denser cell, a busier fleet, or a campaign width nobody has
run. A budget this loose costs nothing when the work is healthy and buys the one thing that
matters here: it will not kill a slow assembly.

This budget is a **backstop against a permanently stuck fork phase, not a detector.** It
is nearly a day long. Nothing is expected to hit it, and if something does, that is a
finding in its own right.

It is also the cheap one to enforce **when the fill uses a process pool**. It fires
inside `_await_forks`, and `run_forked`'s existing `except BaseException` handler already
terminates the worker processes. Nothing a worker wrote is in the store until its fork is
merged, so the cost of the kill is unreferenced objects that garbage collection reclaims.
No thread is leaked on that path.

**A single-payload fill is the exception.** `partition_round_robin` yields one payload
for a sparse cell whatever `n_workers` says, and one payload runs in-process — there is
no worker process to terminate. That case is bounded the same way the post-fork phase is,
by abandoning a thread, because the alternative is a sparse cell blocking its fill's
finalizer exactly as the stall this exists for does. It is the cheaper leak of the two:
the abandoned worker writes into a fork that is never merged, so nothing it produced can
reach the store. What it keeps that a killed process would not is CPU and object writes,
against objects nothing references.

### The post-fork budget

```
POST_FORK_BUDGET_S = 600     # 10 minutes
```

Flat, not work-derived, because the work is fixed however large the fill was. Ten minutes
is:

- **~333x the worst observed** for the phase end to end (1.8 s; the slowest commit alone
  was 1.61 s), and
- **40x the worst commit latency measured at full campaign width** — 15 s at 120
  simultaneous committers (`commit-gate-removal-2026_08.md`).

So a contention storm cannot reach it, and the false-positive risk is close to nil. **This
is the bound that can actually catch the stall.**

It spans two functions — `run_forked` does the teardown and the merge,
`write_year_shards` the group read and both commits — so the two share **one** clock
(`_PostForkDeadline`), started at the single moment the phase begins: the last fork's
result in hand. Two independent ceilings would have let a fill sit for twenty minutes
instead of ten.

## How the fill survives it

Python cannot interrupt a thread blocked inside a Rust or C call, and every call in the
post-fork phase descends into icechunk's Rust extension. A bare timeout returns control
to the caller but leaves the wedged thread holding whatever it holds — which, on the
single-slot finalizer, would have fixed nothing.

So the post-fork work runs on a **single-use daemon thread** and the caller waits on it
with a deadline. On overrun the caller raises `AssemblyDeadlineError` and the worker
thread is **deliberately leaked**.

What that buys, and what it costs:

- **Buys:** the finalizer thread comes back. It is never the wedged thread, so it needs
  no replacing and the fill goes straight on to the next cell's assembly. The cell is
  recorded through the existing failure path, `_record_failure(cell, "assembly", exc)`,
  which retains its mosaic — so a retry resumes from its staged tiles.
- **Costs, in the order they are likely:** *a leaked thread still holds an icechunk
  session.* On the evidence so far it never unblocks — the stalled assemblies were still
  stalled hours later — so the expected outcome is that the thread's merged index dies
  with the process at the end of the fill, leaving its chunk objects in S3 with nothing
  pointing at them. Those objects are not recoverable after the fact: a chunk object is
  anonymous, and the mapping that says which array and position it belongs to exists only
  in the index built at commit time. They are garbage for an orphan sweep, and the cell is
  re-run from its staged tiles.

  That loss is not caused by the deadline. It is the same loss as cancelling the fill, and
  the same loss as waiting forever — the difference is that this way the fill's other cells
  still publish.

  The less likely branch is worth stating too: if the thread ever *does* unblock it will
  commit, possibly after a retry of the same cell already has. Both write the same year's
  shards from the same staged tiles and `commit_year_attrs` inserts one year key
  idempotently, so that costs a duplicate write and a second provenance entry rather than a
  damaged year. Either way it is a live writer nobody is waiting on, which is the main
  reason the budgets above are generous rather than tight.
- **Affordable because:** the runner is 16 vCPU / 64 GiB and sits near 1.2 GB resident,
  against tens of cells per fill. The threads are daemons, so a leaked one cannot hold
  up interpreter exit.
- **In-child retry interaction:** `attempts_per_cell_in_cluster` defaults to 2, so a
  deterministically stalling cell leaks one thread per attempt — two, not an unbounded
  number. Both attempts announce. The retry runs under a broad `except Exception` that
  logs an ordinary error line, so the announcement is a shared helper rather than a
  clause at one site: otherwise the first attempt would alert and the second — the one
  that proves the stall is deterministic — would be swallowed.

The runner announces it at CRITICAL under `ASSEMBLY DEADLINE EXCEEDED`, a greppable
prefix in the same style as `FAILURE CAP EXCEEDED`, because this repo has no alerting
transport and monitoring matches on text. It is the one assembly failure that says
nothing about the cell — every other names something about the data or the store; this
one only says a phase stopped returning.

## What is deliberately NOT covered

- **`repo.writable_session("main")` and `source.live_shards()`**, at the top of
  `write_year_shards`, are before the fork phase and outside both budgets. The observed
  signature places the stall after the forks returned, so bounding them would be guessing.
- **The process-pool construction and the `submit()` calls.** The fork budget starts before
  `session.fork()` — an icechunk call of the same family as the ones that stalled — but the
  pool that follows is not inside it. Bounding it would put the executor itself on a thread
  the caller abandons, where its processes could no longer be reached to terminate, trading
  one leaked thread for a leaked pool of sixteen. **Cost of leaving it:** a stall in
  `ProcessPoolExecutor` startup still holds the caller's thread indefinitely. Nothing has
  been seen to stall there, and unlike the phases that have, it would happen before the
  first progress line — so it shows as a fill that never starts assembling rather than one
  that never finishes, which is the easier of the two to spot.
- **The single-ROI assembly path** (`inference/assembly.py`'s other `run_forked` caller)
  passes no budgets and is unchanged. The incident is on the global campaign path.
- **The mosaic delete** (`inputs.cleanup`) that follows a landed assembly also runs on
  the finalizer thread and is not bounded. It is a pre-existing exposure of the same
  shape, not part of this incident, and no fill has been seen to stop there.
- **The stall itself.** Nothing here diagnoses it. `PR 1` on the infrastructure side
  (`SYS_PTRACE` on the flow-runner task definitions) is what makes the driver's stack
  readable while it is still stuck, which is the evidence that would settle it.

## What would retire this

A root cause. The strongest clue is in the campaign-side record and is not a timing one:
counting the commits that landed between an assembly opening its session and trying to
commit, **every one of the seven that faced four intervening commits stalled, and every one
of the four that faced two or fewer committed in about a second — eleven out of eleven.**
One case rules out simple simultaneity: 44N/2017 finished its long half 94 minutes after
any other assembly was active, with nothing else committing, and still stalled. So the
suspect is how stale a session had become, not contention at the instant of the commit.
Reproduction at that shape has so far been negative — rebase depth adds about 30% to commit
cost in a synthetic harness and never stalls — so something about the real environment is
still missing.

If it turns out to be, say, a lost wakeup in the rebase or a retry loop with no ceiling,
the fix belongs there and this becomes a cheap guard rather than the mechanism keeping fills
alive. Until then, **the post-fork budget firing is a signal to go and look, not a routine
outcome** — one `ASSEMBLY DEADLINE EXCEEDED` line means a fill hit the thing nobody has
explained yet.
