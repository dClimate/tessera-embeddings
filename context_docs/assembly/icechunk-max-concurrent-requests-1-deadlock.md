# icechunk: `max_concurrent_requests = 1` deadlocks commit and rebase

**Status 2026-09-09: root cause of the 2026-09-04 assembly wedges, confirmed on real S3 with
native stacks on our pinned 2.1.1 AND the current release 2.2.0.** Python 3.12, S3 `us-west-2`.
Our record of the fault and our mitigation; the upstream issue drafted from it is not yet filed.

## Summary

A repository opened with `RepositoryConfig.max_concurrent_requests = 1` deadlocks: `Session.commit`
or `Repository.diff`/`Session.rebase` parks inside icechunk's tokio runtime and never returns. The
process stays alive with no runnable thread, writes nothing further, and does not time out.

**The necessary ingredient is an ordinary object-store write in flight at the same time as a
`rebase` that has real commits to cross, through one request permit.** A lone sequential writer at
cap 1 is fine. A fork pool is not required (`--no-forks` deadlocks too); forks only change which
call is caught holding the permit. Cap 2 clears it entirely under identical load.

It stranded five compute clusters for four days and cost roughly 177 units of work, because the
operations that never return are the same ones our recovery path used.

## Bisection

Same load, same store geometry, same process shape; only the value changed.

| `max_concurrent_requests` | outcome |
|---|---|
| 1 | **deadlock** in `commit` and in `rebase`/`diff` |
| 2 | all writers complete, clean (146 s) |
| 4 | all writers complete, clean |

The cap-1 arm parked three of four writers within about two minutes and was still parked when
killed 29 minutes later.

## Reproduction

A standalone ~130-statement script, `mre_v3_fork_pool.py`, depending only on icechunk, zarr, numpy
and a bucket. It is staged with the upstream issue rather than committed here. It reproduced on the
first attempt, on both 2.1.1 and 2.2.0.

| arm | outcome |
|---|---|
| cap 1 | 2-3 of 4 coordinators park permanently in `commit`/`merge` with the background `rebase` parked; native frames identical across dumps 86-127 s apart; ~3 CPU ticks in 20 min |
| cap 1, `--no-forks` | also deadlocks |
| cap 2, identical load | clean in ~100 s, with 66 catch-ups that each did real work |

A runtime census of a parked process: all 191 tokio worker threads idle in `park_internal`. Nothing
in flight, nothing on the network.

**Why four earlier reductions failed, which is the useful part.** None had a write running
concurrently with a rebase that had real work; in one the rebase kept short-circuiting with "No
rebase is needed" because the tip had not moved. A control whose catch-ups have nothing to cross
proves nothing, and that is what made the fault look irreducible for a day. All four ran against
real S3 at cap 1 and all four completed cleanly: four processes each writing its own array then
rebasing and committing; the same plus a background thread rebasing the live session; one commit
carrying 4,096 chunk references; one commit alongside a background read-only session listing nodes.

**A local-filesystem store can never show this**: the setting governs HTTP request concurrency and
is inert without an object store.

The fault was first pinned with our full write path — four coordinators, each spawning eight fork
workers on pickled sessions, all 36 processes at cap 1 — via `prod_scale_repro.py` in the
`wedge_repro` harness on the unmerged branch `dev/publication-density-harness`.

**Tooling.** `py-spy --native` failed with `UNW_EBADREG` on that host; `gdb` gave the frames.

## Native stacks

Both taken twice, 105 s apart on the same pids, identical frames, corroborated by an independent
third dump. Abridged:

```
syscall (libc.so.6)                                                  # park 1: commit
tokio::runtime::park::Inner::park::hc45134453c707800 (_icechunk_python.abi3.so)
commit (icechunk/session.py:442)

syscall (libc.so.6)                                                  # park 2: diff, on rebase
tokio::runtime::park::Inner::park::hc45134453c707800 (_icechunk_python.abi3.so)
diff (icechunk/repository.py:1456)
```

icechunk's own tracing for the stalled commit ends on repeats of `icechunk::session::
do_commit_rebasing with rebase_attempts: 1000, max_concurrent_nodes: 1, allow_empty: false`. In the
same process the main thread was separately blocked in an ordinary `zarr.open_group` read against
the same repository — a second victim of the same exhausted permit.

## What we think is happening

A hypothesis, not a diagnosis: a single in-flight operation appears to need more than one
concurrent request internally, so with exactly one permit it holds that permit while awaiting a
second that cannot be granted. That fits cap 2 being sufficient, a lone writer being unaffected,
the deadlock landing in the rebase-and-commit path, and the reproducer's requirement that a write
and a real rebase overlap.

## What we would ask upstream for

In rough order of preference: make the operation not deadlock at a permit budget of 1, or acquire
its permits as a set; reject or clamp `max_concurrent_requests = 1` at construction; document a
minimum (the docstring gives only the default of 256, so 1 reads as merely conservative — ours came
from dividing a fleet request budget by a worker count and reached 1 without anyone choosing it);
bound the wait so it surfaces as an error rather than an indefinite park.

## Our mitigation

**We have stopped setting the value and taken icechunk's default (256).** Uncapped at production
fan-out, including 32 fork workers per fill across the fleet, zero throttling was measured on every
arm — so capping bought nothing. The recovery machinery built on the assumption that these
operations return is removed with it; see
[`assembly-wedges-during-fork-phase-2026-09-04.md`](assembly-wedges-during-fork-phase-2026-09-04.md).
