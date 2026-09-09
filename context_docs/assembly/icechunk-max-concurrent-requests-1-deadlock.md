# icechunk: `max_concurrent_requests = 1` deadlocks commit and rebase under concurrent writers

**Status 2026-09-09: root cause of the 2026-09-04 assembly wedges, confirmed on real S3 with native
stacks. Draft for an upstream icechunk issue.** icechunk version **2.1.1**, Python 3.12.

## Summary

A `Repository` opened with `RepositoryConfig.max_concurrent_requests = 1` deadlocks: a
`Session.commit` or a `Repository.diff` / `Session.rebase` parks forever inside icechunk's tokio
runtime and never returns. It happens when several processes commit to the same branch concurrently
(so each session must rebase against a moved tip). It does **not** happen at `max_concurrent_requests
>= 2` under the identical load, and does not happen for a lone sequential writer at cap 1. The
single request permit is held by one in-flight operation while the same operation (or its rebase)
awaits a second concurrent request that can never be granted.

## Evidence (native py-spy, real S3, `us-west-2`)

Four processes, each opening the same store at `max_concurrent_requests = 1`, each writing a cell and
committing to `main` while a background catch-up rebases the session. Three of four parked within
~2 minutes and stayed parked across dumps 105 s apart (frames, thread ids and sizes identical).

Park one — commit:
```
tokio::runtime::park::Inner::park  (_icechunk_python.abi3.so)
commit                             (icechunk/session.py:442)
```
Park two — rebase/diff:
```
tokio::runtime::park::Inner::park  (_icechunk_python.abi3.so)
diff                               (icechunk/repository.py:1456)
```
icechunk's own tracing for the stuck commit ends on repeats of
`icechunk::session::do_commit_rebasing with rebase_attempts: 1000, max_concurrent_nodes: 1,
allow_empty: false`.

## Bisection (only the cap changes; same load, same store geometry, same process shape)

| `max_concurrent_requests` | outcome |
|---|---|
| 1 | deadlock in commit and in diff/rebase |
| 2 | all cells publish, clean |
| 4 | all cells publish, clean |

A lone single-process sequential commit at cap 1 does **not** hang, and a commit under a background
readonly reader at cap 1 does **not** hang — the deadlock needs concurrent WRITERS advancing the
branch so a rebase has real work while holding the single permit.

## Reproducer

Faithful reproducer: `scripts/scoping/wedge_repro/prod_scale_repro.py` (this repo) against an S3
store, `--arm reproduce --max-concurrent-requests 1 --process-shape production --coordinators 4
--cells-per-coordinator 1 --live-shards 200 --n-workers 8`. The minimal shape is: >= 2 processes,
one object-store repo opened at `max_concurrent_requests = 1`, each process in a loop writing then
`rebase` then `commit` to one branch. (A single-process local-filesystem reduction was attempted and
does not reproduce it — local FS is flagged unsafe for concurrent commits and a same-chunk race
raises `ConflictError` rather than deadlocking.)

## Our mitigation (independent of the upstream fix)

Never open the store at cap 1: floor the per-worker request cap at 2 in `_s3_budget_split`
(`inference/assembly.py`). Our budget split, `max(1, TARGET_AGGREGATE_S3_CONCURRENCY // n_workers)`,
produced 1 on the live campaign (`per_worker_s3_cap: 1` in the assembly summary), and raising the
fork count to 32 keeps it pinned at 1 while adding contention.
