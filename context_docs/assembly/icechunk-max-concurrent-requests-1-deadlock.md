# icechunk: `max_concurrent_requests = 1` deadlocks commit and rebase under concurrent writers

**Status 2026-09-09: root cause of the 2026-09-04 assembly wedges, confirmed on real S3 with native
stacks, on our pinned 2.1.1 AND the current release 2.2.0.** Python 3.12. This is our record of the
fault and our mitigation; the upstream issue drafted from it is not yet filed. **A standalone
minimal reproducer exists** and fired on its first attempt — see Reproduction.

## Summary

A `Repository` opened with `RepositoryConfig.max_concurrent_requests = 1` deadlocks: a
`Session.commit` or a `Repository.diff` / `Session.rebase` parks inside icechunk's tokio runtime and
never returns. The process stays alive with no runnable thread, writes nothing further, and does not
time out.

**The necessary ingredient is ordinary object-store writes happening at the same time as a
``rebase`` that has real commits to cross, through one request permit.** A lone sequential writer at
cap 1 is fine. A fork pool is NOT required (``--no-forks`` deadlocks too); forks only change which
call is caught holding the permit. Raising the value to 2 clears it completely under identical load,
which is the cleanest evidence that the value itself is the trigger.

It stranded five compute clusters for four days and cost roughly 177 units of work, because the
operations that never return are the same ones our recovery path used.

## Environment

| | |
|---|---|
| icechunk | **2.1.1 and 2.2.0** (Python) — reproduced on both; 2.2.0 is the current release |
| Python | 3.12 |
| Store | Amazon S3, `us-west-2` |
| OS / host | Amazon Linux 2023, `c7i.48xlarge` (192 vCPU) |
| zarr | 3.x |

## Reproduction

**Standalone script, ~130 statements, depending only on icechunk, zarr, numpy and an S3 bucket:
`mre_v3_fork_pool.py`, staged with the upstream issue rather than committed here.** It reproduced on
the first attempt, and on both 2.1.1 and 2.2.0.

| arm | outcome |
|---|---|
| cap 1 | two to three of four coordinator processes park **permanently** in `commit`/`merge` with the background `rebase` parked; native frames identical across dumps 86-127 s apart; ~3 CPU ticks in 20 minutes |
| cap 1, `--no-forks` | also deadlocks — the fork pool is not the necessary ingredient |
| cap 2, identical load | clean in ~100 s, with 66 catch-ups that each did real work |

A runtime census of a parked process: all 191 tokio worker threads idle in `park_internal`. Nothing
in flight, nothing on the network.

**Why the four earlier reductions failed, which is the useful part.** None of them had a write
running concurrently with a rebase that had real work to do; in one, the rebase kept
short-circuiting with "No rebase is needed" because the branch tip had not moved. A control whose
catch-ups have nothing to cross proves nothing, and that is what made this look irreducible for a
day. All four ran against real S3 at `max_concurrent_requests = 1` and all four completed cleanly:

| reduction | result |
|---|---|
| 4 processes, each writing its own array then `rebase` + `commit` on one branch | no deadlock |
| the same, plus a background thread rebasing the live session while the main thread writes it | no deadlock |
| one process, one commit carrying 4,096 chunk references | no deadlock, committed in 0.7 s |
| one process committing while a background thread holds a read-only session and lists nodes | no deadlock, committed in under 0.02 s |

**A local-filesystem store can never show this**, because `max_concurrent_requests` governs HTTP
request concurrency and is therefore inert without an object store.

**Tooling note.** `py-spy --native` failed with `UNW_EBADREG` on the reproducing host; `gdb` gave
the native frames.

### Our own workload, for the record

The fault was first pinned with the full write path: four coordinator processes, each spawning eight
fork workers that receive a *pickled* session and write large shard objects through their own
repository handle, while the coordinator concurrently rebases its session and commits the merged
result. All 36 processes opened the repository at `max_concurrent_requests = 1`, and three of four
coordinators parked within about three minutes and never recovered. The harness is
`prod_scale_repro.py` in the `wedge_repro` scoping directory on the unmerged branch
`dev/publication-density-harness`.

### Also reproduces on the current release, 2.2.0

The identical configuration was re-run against icechunk 2.2.0 in a fresh virtualenv: zero cells
published, three of four coordinators stuck at teardown, native stacks 124 seconds apart showing
identical parked frames in the same processes. So the fault is not fixed upstream, and the release
notes for 2.1.2 and 2.2.0 do not mention it.

## Bisection

Same load, same store geometry, same process shape; only the value changed.

| `max_concurrent_requests` | outcome |
|---|---|
| 1 | **deadlock** in `commit` and in `rebase`/`diff` |
| 2 | all writers complete, clean |
| 4 | all writers complete, clean |

The cap-2 control published every cell in 146 seconds. The cap-1 arm parked three of four writers
within about two minutes and was still parked when killed 29 minutes later.

## Native stacks

Both taken twice, 105 seconds apart on the same pids, with identical frames, and corroborated by an
independent third dump. Abridged:

**Park 1, `commit`:**

```
syscall (libc.so.6)
tokio::runtime::park::Inner::park::hc45134453c707800 (_icechunk_python.abi3.so)
commit (icechunk/session.py:442)
```

**Park 2, `diff` on the rebase path:**

```
syscall (libc.so.6)
tokio::runtime::park::Inner::park::hc45134453c707800 (_icechunk_python.abi3.so)
diff (icechunk/repository.py:1456)
```

icechunk's own tracing for the stalled commit ends on repeats of:

```
icechunk::session::do_commit_rebasing with rebase_attempts: 1000,
    max_concurrent_nodes: 1, allow_empty: false
```

In the same process the main thread was separately blocked in an ordinary `zarr.open_group` read
against the same repository — a second victim of the same exhausted permit, showing the process had
no runnable thread left.

## What we think is happening

A hypothesis, not a diagnosis: a single in-flight operation appears to need more than one concurrent
request internally, so with exactly one permit available it holds that permit while awaiting a
second that cannot be granted. That would explain why 2 is sufficient, why a lone sequential writer
is unaffected, and why the deadlock lands inside the rebase-and-commit path, which is the part that
fans out across snapshot and manifest objects. The reproducer sharpens it: what is needed is a write
in flight AT THE SAME TIME as a rebase with real commits to cross, which is exactly the pair that
would contend for one permit.

## Why it mattered so much

The operations that deadlock are the same ones a recovery path uses. Our recovery, on detecting a
catch-up that would not return, re-homed onto a fresh session and called `diff` — which parked in
exactly the same way. So the fault defeated its own mitigation, and a stalled writer produced no
error, no timeout and no diagnostic. It was located only by attaching a native profiler, which the
normal runtime does not permit (Fargate denies `CAP_SYS_PTRACE`); the workload had to be rebuilt on
a plain instance to get a stack.

## What we would ask upstream for

In rough order of preference:

1. Make the operation not deadlock at a permit budget of 1, or acquire its permits as a set.
2. Reject `max_concurrent_requests = 1` at construction with an error naming the minimum, or clamp
   it to the minimum the implementation actually needs and log that it did.
3. Document a minimum. The docstring gives only the default of 256, so 1 reads as a legitimate if
   conservative choice. Ours was computed by dividing a fleet request budget by a worker count, and
   it reached 1 without anyone choosing it.
4. Bound the wait so it surfaces as an error rather than an indefinite park.

## Our mitigation

**We have stopped setting the value at all and taken icechunk's default (256).** Uncapped at
production fan-out, including 32 fork workers per fill across the fleet, zero throttling was measured
on every arm — so nothing was bought by capping it in the first place. The recovery machinery built
on the assumption that these operations return is removed with it. See
`assembly-wedges-during-fork-phase-2026-09-04.md`.
