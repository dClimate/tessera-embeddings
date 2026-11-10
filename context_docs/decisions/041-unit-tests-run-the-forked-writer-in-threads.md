# 041 — Unit tests run the forked writer in threads

**Status:** Accepted 2026-11-05.

## Context

`storage/shard_writer.py::run_forked` writes a zone-year's or a tile's shards through a
spawn-context `ProcessPoolExecutor`, one fresh pool per call, even for a single payload. That is
right in production: a spawned worker cannot inherit a wedged lock, a parent's thread state or a
half-configured icechunk client, and the stall watchdog can kill it.

In the unit suite it was the largest single cost. Every spawned child starts an interpreter and
re-imports `tessera_embeddings`, about 0.8 s, and every test that assembles a store or writes a
year pays it. Measured on 2026-11-04: one assemble took 1.23 s through the spawned pool and 0.12 s
in-process, and the whole default run fell from 618 to 231 CPU-seconds when the pool was replaced.

## Decision

**Unit tests run `run_forked` on an in-process stand-in by default.** An autouse fixture in
`tests/unit/conftest.py` replaces `shard_writer.ProcessPoolExecutor` with a thread pool whose
`submit` pickles the worker function, its payload and its result across the call. So each worker
still gets its own unpickled fork session, and an unpicklable payload still fails, as under spawn.
The stand-in does not run the pool's `initializer`, which configures logging: a thread inherits the
parent's logging, which pytest's `caplog` owns.

**Two tests keep the real spawn pool**, marked `real_fork_pool`: one payload through the real pool
(`test_run_forked_ships_index_and_interval_to_workers`) and a multi-worker assemble compared with a
serial one (`test_assemble_multiband_parallel_matches_serial`). Tests of the watchdog, the
termination path and the initializer install their own fake executor and need neither.

## Consequences

- A defect that only exists across a process boundary (a worker reading parent-process state
  that a spawned child would not have) is caught by the two marked tests or not at all. Mark a new
  test `real_fork_pool` when its subject is that boundary.
- Worker bodies must stay free of process-global side effects, which they are today: they write
  only to their fork and to the shared progress counters.
