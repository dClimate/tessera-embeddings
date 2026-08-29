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

Every row is one zone-year's fork-write phase: how many shards it wrote, and the
wall-clock minutes from fork creation to the last fork returning.

| cell | shards | fork-phase minutes | shards/min | outcome |
|---|---:|---:|---:|---|
| 48N / 2019 | 8,803 | 189 | 46.6 | stalled after |
| 33N / 2018 | 8,593 | 194 | 44.3 | stalled after |
| 34N / 2018 | 8,734 | 199 | 43.9 | stalled after |
| 38N / 2019 | 9,100 | 207 | 44.0 | stalled after |
| 47N / 2018 | 9,030 | 208 | 43.4 | stalled after |
| 44N / 2017 | 7,907 | 147 | 53.8 | stalled after |
| 13N / 2017 | 6,575 | 134 | 49.1 | committed |
| 36N / 2017 | 8,738 | ~185 | 47.2 | committed |
| 35N / 2017 | 9,132 | ~187 | 48.8 | committed |
| 36N / 2018 | 8,738 | ~392 | 22.3 | committed |
| 35N / 2018 | 9,132 | ~400 | 22.8 | stalled after |

**Observed range: 22.3 to 53.8 shards per minute — a 2.4x spread.**

The table carries two natural controls: 36N and 35N each appear twice, same zone and
IDENTICAL shard counts (8,738 and 9,132), one year fast and the other 2.1x slower. Same
geometry, same partition width — so the spread is not the shard count.

What it *is* has two plausible drivers, and this table cannot separate them. The slow
member of each pair ran while the shared Ray cluster was inferring hard, which is the
obvious candidate. But per-tile cost also differs materially between cells: a denser tile
carries more valid pixels and a larger working set, which shows up independently in the
A10G actor out-of-memory failures clustering in particular fills rather than spreading
across the fleet — see the separate work in dClimate/tessera-embeddings#160, which scales
the inference batch to the card the actor landed on. A year holding more valid data is
both heavier to embed and heavier to write.

**The budget does not need to separate them.** Whichever mix produced 22.3 shards/min,
that is the slow end actually observed, and the divisor is set below it. What matters is
that both drivers are things the campaign varies rather than constants, so the safety
factor is padding against a future cell denser than any in this table, or a future hour
busier than the one it was measured in.

**Nothing here is keyed to a fleet width.** The formula's only input is the fill's own
shard count. Ten concurrent fills is what happened to be running when the table was
measured, not a parameter — and two of those ten were cancelled the next day, which
changes the contention the rest see without changing any budget they are held to.

*Correction.* An earlier reading of this table quoted the range as "22 to 46". The
maximum is **53.8** (44N/2017), not 46. It does not change the budget, which is set by
the slow end.

**The phase after the forks, on the four cells that committed, took between 0.2 s and
1.8 s end to end** — `merge_s` + `commit_s` + `attrs_commit_s` from their
`ASSEMBLY_SUMMARY` records. That is the whole of it: one merge and two commits.

## Why two budgets and not one

The two phases have nothing in common.

| | fork-write phase | everything after it |
|---|---|---|
| work | scales with the shard count | fixed: one teardown, one merge, two commits |
| observed duration | 134–400 min | 0.2–1.8 s |
| what varies it | contention and per-tile density (2.4x) | branch-tip CAS contention |
| is this where fills stopped? | no — every fork returned | **yes** |

A single cutoff across both would have to clear 400 minutes, and a 400-minute cutoff can
never notice a phase that should take two seconds. Splitting them is the entire point of
this change; the exact constants matter much less.

### The fork-write budget

```
fork_phase_budget_s(n) = max(FORK_PHASE_FLOOR_S,
                             60 * n / FORK_PHASE_SHARDS_PER_MINUTE * FORK_PHASE_SAFETY_FACTOR)

FORK_PHASE_SHARDS_PER_MINUTE = 20     # the slow end (22.3), rounded down
FORK_PHASE_SAFETY_FACTOR     = 3
FORK_PHASE_FLOOR_S           = 3600
```

Derived from the work, not the clock, because the phase's duration tracks its shard
count. The divisor is already the slowest rate ever measured, rounded down, so the
safety factor is padding **on top of** the worst case rather than padding towards it.

At the densest cell in the table, 9,132 shards, that is 9,132 / 20 x 60 x 3 = **82,188 s
(22.8 h)**, against a worst observed 400 min (6.7 h) — **3.4x the slowest fill on
record**. The factor is deliberately generous because the things that vary here —
cluster contention and per-tile density — are both set by the campaign rather than fixed,
and either can go past what this table saw. Below ~400 shards the floor binds, so a
sparse zone-year still gets an hour.

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

- **~333x the worst observed** for the phase (1.8 s), and
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
- **Costs:** *a leaked thread still holds an icechunk session.* It holds its merged
  forks and their memory, and it is still a writer. If it ever unblocks it will commit,
  possibly after a retry of the same cell has already committed. Both write the same
  year's shards from the same staged tiles, and `commit_year_attrs` inserts one year key
  idempotently, so the visible cost is a duplicate write and a second provenance entry
  rather than a damaged year. It is still a live writer nobody is waiting on. This is
  the main reason the budgets above are generous rather than tight.
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
- **The single-ROI assembly path** (`inference/assembly.py`'s other `run_forked` caller)
  passes no budgets and is unchanged. The incident is on the global campaign path.
- **The mosaic delete** (`inputs.cleanup`) that follows a landed assembly also runs on
  the finalizer thread and is not bounded. It is a pre-existing exposure of the same
  shape, not part of this incident, and no fill has been seen to stop there.
- **The stall itself.** Nothing here diagnoses it. `PR 1` on the infrastructure side
  (`SYS_PTRACE` on the flow-runner task definitions) is what makes the driver's stack
  readable while it is still stuck, which is the evidence that would settle it.

## What would retire this

A root cause. If the stall turns out to be, say, a lost wakeup in the merge or a
retry loop with no ceiling, the fix belongs there and this becomes a cheap guard rather
than the mechanism keeping fills alive. Until then, **the post-fork budget firing is a
signal to go and look, not a routine outcome** — one `ASSEMBLY DEADLINE EXCEEDED` line
means a fill hit the thing nobody has explained yet.
