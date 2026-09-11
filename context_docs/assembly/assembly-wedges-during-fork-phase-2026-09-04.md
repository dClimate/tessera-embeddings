# The 2026-09-04 assembly wedges

**Five clusters stopped assembling on 2026-09-04 and never resumed**, holding 177
completed-but-unassembled cells between them by 09-08 while four siblings ran normally. Store
integrity was never at risk: a wedged process wrote ~16 bytes / 45 s, so there was no active
writer, no torn commit, no second writer, and staged tiles persisted.

**Root cause, confirmed 2026-09-09.** The assembly opened its icechunk repository with
`max_concurrent_requests = 1`, at which icechunk deadlocks `Session.commit` and
`Repository.diff`/`rebase` under concurrent writers. Reproduced on real S3 with native stacks on
2.1.1 and 2.2.0; cap 2 is clean under identical load. See
[`icechunk-max-concurrent-requests-1-deadlock.md`](icechunk-max-concurrent-requests-1-deadlock.md).

**Nobody chose that value.** `_s3_budget_split` computed `max(1, budget // n_workers)`, and the
campaign's budget of 5 floored it at 1 on every assembly — `per_worker_s3_cap: 1` in all seven
recorded `ASSEMBLY_SUMMARY` lines.

## The signature

Four of the five ended their `Assembly progress` stream at exactly "1/16 tile partitions
outstanding" and then went silent:

| cluster | zone-year | last line (UTC) | shards at stop |
|---|---|---|---|
| gorgeous-rat | 19S | 2026-09-04 21:47:27 | 7940/7951 |
| light-macaque | 30N | 2026-09-04 22:29:53 | 7191/7208 |
| fictional-gaur | 51N-2020 | 2026-09-04 23:22:16 | 5575/5590 |
| vociferous-earthworm | (fill) | 2026-09-04 23:23:24 | 6572/6575 |

(The fifth, mystic-gopher, is probable-same rather than confirmed: its wedge point cannot be
dated cleanly.) **"1/16 outstanding" is the normal LAST line of a healthy assembly** — `_await_forks`
prints only while a fork is outstanding, so a healthy fill's next line is `ASSEMBLY_SUMMARY`. These
reached the end of the shard write and produced neither the summary nor a failure.

Measured on a live wedged process 09-08 via ECS Exec (`/proc`, no ptrace):

| probe | result |
|---|---|
| CPU across 144 threads | 21 ticks / 45 s ≈ 0.5% of one core |
| bytes read / written per 45 s | 16 / 16 |
| worker child processes | none |
| icechunk tokio threads (16) | all parked in `futex_wait_queue` |
| commit-stall alarm, 7 days | never fired |

A deadlock, not a slow operation.

## Why every safeguard missed it

- **The commit-stall alarm** wraps only `traced_commit`. This wedge is upstream of the commit, so
  it never armed — hence seven days with zero alarms and zero stacks.
- **The abort escape** in `_await_forks` fires when a catch-up tick *raises*. A tick that *hangs*
  never raises.
- **`rehome_after_a_wedged_catch_up`** is reached only via `CatchUpDidNotStopError`, raised only
  from `ticking.__exit__`, which runs only after its block returns — and the wedge was inside that
  block. It would not have helped anyway: its own `diff` parks in the same deadlock.

The 08-31 case was recoverable and this one was not purely because of *when* the tick wedged:
after the workers finished, versus while one was still writing.

## The decision

Remove the cap and take icechunk's default; remove the machinery built for a hang that can no
longer happen; keep only what converts an unbounded hang into a diagnosable crash. Shipped in
tessera-embeddings #186, superseding #181, #182 and #183.

**Removed.** The whole request-cap plumbing (`TARGET_AGGREGATE_S3_CONCURRENCY`,
`_s3_budget_split`, `per_worker_s3_cap`, the `s3_concurrency` flow parameter) — uncapped at 32
workers per fill across the fleet measured zero throttling, and the 800-PUT `SlowDown` figure the
cap rested on was never traced to a primary record. `rehome_after_a_wedged_catch_up` and its
`run_forked` hook, for the reason above. Publication spacing, which prevents the catch-up *depth*
that was the hang's precondition rather than its cause. The recovery half of #181 — killable-child
publish, per-step publish timeout, partition re-run, salvage-finished-forks.

**Kept, each earning its place.**

1. **Storage timeouts and retries** (`zarr_store._default_repo_config`). A different, separately
   observed failure: a socket wedged mid-response, diagnosed with a worker stuck in `sk_wait_data`.
2. **The fork-phase watchdog** (`shard_writer._fork_stall_watchdog`, 30 min). A daemon thread
   watches the shard counters the workers write into shared memory; on a stall it raises the
   failure flag and terminates the pool, and only THEN dumps every thread's stack, after which the
   fill fails as an ordinary assembly failure with its cell retained. Log line
   `ASSEMBLY FORK PHASE STALLED`. That order matters: the critical log line and the stack dump are
   both writes to stderr, which BLOCK rather than raise when the container's log pipe fills, and
   `suppress` covers only the raising case — diagnosing first would let the one component whose
   purpose is to end a hang be ended by one. Not a fix for this incident — the net under the next
   unknown cause, and the only way to get a stack where `CAP_SYS_PTRACE` is denied. It cannot
   unwind a coordinator parked inside icechunk; the dump still fires.
3. **The bounded assembly backlog drain** (`sequential_fill.drain_trailing_assemblies`, 6 h
   per assembly, resetting on each completion). `finalizer.shutdown(wait=True)` had no bound, which
   is what parked vociferous-earthworm behind one wedged cell after its inference finished. The
   watchdog covers the fork phase only. Same ordering rule as the watchdog, in the shape the
   drain's position forces: the watchdog signals through a flag another thread reads, so it can
   diagnose in line; the drain signals by RAISING on the caller's own thread, so a diagnostic in
   front of that raise can swallow it. It therefore cancels, raises, and leaves the critical line
   and the stack dump to a daemon thread (`_diagnose_wedge_off_thread`) that may park forever.
4. **Crash, not FAILED, after a wedged drain** (`_end_process_after_wedged_drain`, exit 75). The
   driver treats FAILED as quiescent and may admit a replacement cluster; the README's admission
   rule rests on the wedged thread having been joined, which it has not. Two things sit between
   detecting the wedge and this exit and are deliberately not allowed to hold it up. The flow's
   `finally` SKIPS the housekeeping join on the wedged path only (`shutdown(wait=False,
   cancel_futures=True)`): that join has three unbounded legs — the `s5cmd` subprocess, the
   `fsspec` recursive `rm` fallback, and the read-back that lists objects — so bounding any one of
   them would read as a fix and bound nothing. Abandoning an in-flight delete leaks a storage
   prefix, which is recoverable and costs storage; not exiting leaves a cluster the driver may
   read as quiescent. And `hard_exit_after_flush` ARMS the exit on a 5 s daemon timer before it
   announces, because its `finally` guaranteed the exit only against a flush that raises, not one
   that blocks on a handler lock, socket or full pipe.
5. **`CatchUpDidNotStopError`** and `ticking`'s bounded join. Nothing recovers from it; it makes a
   hung catch-up a crash rather than an indefinite park, in a window the watchdog has already left.

**Also: 32 assembly workers**, as a dispatch parameter on the chained fill rather than a library
default — 32 needs the 244 GiB `assembly_large` runner and would oversubscribe every other caller.
It was the cap of 1, not the worker count, that deadlocked.

## Dev evidence, 2026-09-08

`publication_density.py` from the `wedge_repro` harness (unmerged branch
`dev/publication-density-harness`) against real S3: 120 zone groups at production layout, hundreds
of pre-seeded snapshots, terminal marks every 3 s.

| arm | outcome | shipped |
|---|---|---|
| unspaced control, 10 coordinators | catch-up depth reached 4 — the hang's precondition; the hang itself never appears at this scale | — |
| spaced, 10 coordinators | max depth 1 across 110 ticks, all published | no |
| wedged catch-up | `Re-homing` fired, the cell published | no |
| wedged worker | `ASSEMBLY FORK PHASE STALLED`, four stacks captured, cell failed as `ForkPhaseStalledError`, five neighbours published | **yes** |
| wedged publish (child killed at 60 s) | retried on a fresh session, cell published | no |
| wedged worker, re-run once | partition re-run, cell published | no |

The spacing and recovery arms were built and measured before the root cause was found. They are
kept here so a future wedge need not rediscover that a fork can be merged into a session it was not
created from, or that a publish can be driven from a killable child.

Every watchdog and drain claim was checked by breaking the source and watching the tests go red; a
hang is made red by running the subject under a deadline. The tests inject a wedge deterministically
rather than reproducing the icechunk hang, which no small-scale attempt ever managed.
