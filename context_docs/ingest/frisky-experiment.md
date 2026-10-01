# Frisky as the ingest's scheduler: design, findings, test plan

**Status: experimental, opt-in, off by default.** Frisky 0.7.2, wired in
`src/tessera_embeddings/providers/frisky.py`. Nothing changes for a run that does not ask for it.
This is the record: why Frisky is wired this way and what was found. How to use it is
[`docs/frisky.md`](../../docs/frisky.md).

[Frisky](https://getfrisky.dev/) is Matthew Rocklin's reimplementation of Dask's scheduler and
workers in Rust: about 100x cheaper scheduling (around 3 µs a task) and far more telemetry,
exposed to people and agents through the `frisky observe` CLI. Two things make it worth an
experiment here. The ingest's scheduler is a known bound: graph size and scheduler RAM shape most
of the levers in `docs/ingest-performance.md`. And the profiling we have is hand-built: a scheduler
heartbeat plugin and a capped Dask performance report.

**Packaging.** On this branch Frisky is a core dependency, because the branch exists to test it as
the engine. Whether it stays core when the work reaches `main` depends on the experiment's results
and on two constraints. Frisky is not open source: it ships as a free binary under its author's
own licence, which permits commercial use and redistribution but not modification, and this
repository is public and Apache-2.0. And Frisky installs only on Python below 3.15, so a core
dependency caps where the package installs.

## How it is wired

The providers build the same Dask cluster as always. `hijack()` then loads Frisky onto it, using
Frisky's own `frisky.hijack` plus one plugin of ours.

```text
 flow runner                      Dask scheduler process            each Dask worker process
 ───────────                      ──────────────────────            ────────────────────────
 ecs_cluster(frisky=True)         Dask scheduler                    Dask worker
   cluster.scale(max_workers) ──► ├─ Frisky scheduler  ◄─────────── ├─ Frisky worker (runs the tasks)
   hijack(client)                 └─ dashboard :8787 → Frisky's      ├─ _MatchDaskWorker (ours)
                                                                     └─ on one worker, a Dask thread
 Prefect DaskTaskRunner ─────────────────────────────────────────────►  runs the Prefect ingest task,
                                                                         which does connect(get_client())
```

`connect()` gives the ingest one client. It sends `compute`, `persist`, `submit`, `map`,
`gather` and `scatter` to Frisky (the set is `FRISKY_METHODS`) and everything else to Dask.
Creating it also points `dask.compute` and every bare `.compute()` in that process at Frisky,
from any thread.

| Stays on Dask | Moves to Frisky |
|---|---|
| Provisioning, teardown, ECS tags, the cancellation sweep | Every task the ingest computes |
| The Prefect task runner: the ingest task is still a Dask task on one worker | Scheduling, data transfer, spilling |
| `register_plugin` and `run`, which Frisky's client lacks: credential broadcast, read-failure capture | The dashboard on port 8787, and its API |
| `SchedulerResourceLogger` | |

The Dask control plane still reaches Frisky's tasks because both share each worker *process*.
`test_dask_plugins_and_run_reach_the_processes_frisky_tasks_run_in` pins that.

**Code identity.** Nothing inside the ingest or inference import closures changed, so
`ingest_code_identity()` (`ingcode-0869839717820a9c`) and `inference_code_identity()`
(`infcode-c9905e5aa8f93d3e`) are the same with and without Frisky. A mosaic started on one
engine appends on the other, which the parity tests justify.

## What differs from Dask, and what handles it

Four things. The first two are in none of Frisky's docs and showed only when the real ingest
ran; the last two follow from reading Dask's adaptive scaling and Frisky's client.

**1. Thread state is destroyed after every task.** This was the blocker. Frisky enters Python
with `PyGILState_Ensure` and `PyGILState_Release` around each piece of work, and the release
destroys the thread's Python state, so every `threading.local` is rebuilt per task. Measured on
one OS thread, four tasks in a row each stored a value and read back what the one before had
stored:

| Engine | Values read back |
|---|---|
| Dask | `None, 0, 1, 2` |
| Frisky | `None, None, None, None` |

That is a performance cost for rasterio's environment, odc's sessions and our per-thread STAC
clients. For pyproj 3.7 it is a crash. pyproj frees the thread's PROJ context through an object
held in a `threading.local`, but keeps the raw pointer in OS-thread storage (`PyThread_tss`). The
next task on that thread uses freed memory and segfaults the worker. Dask restarts the worker,
the replacement dies on its first image read too, and the ingest waits forever.

Frisky's own telemetry found it. Its event log showed each replacement worker removed 20 to 50 ms
after it joined, and the tasks placed on it but never completed were odc's image reads. With
`PYTHONFAULTHANDLER=1`, the dump named the frame: `pyproj/crs/crs.py` in `_crs`.

The smallest reproduction is a pyproj `CRS` passed as a task argument: 19 worker restarts before
it was stopped. The fix is `_pin_thread_state`, one unmatched `PyGILState_Ensure` per thread. It is
installed by wrapping `pickle.loads` on each worker, because unpickling is the first Python a
Frisky thread runs for any task. With the pin, the reproduction shows 0 restarts, and the real
Denver ingest completes with 10 of 13 dates passing the coverage gate, as on Dask, in 64 to 68 s.

To see the failure again, remove the `pickle.loads` wrap from `_MatchDaskWorker.setup` and run
`test_real_imagery_produces_an_equivalent_store_on_frisky`: it hangs while the workers crash-loop.
`test_frisky_threads_keep_their_python_state_between_tasks` fails the same way in seconds.

The pin relies on Frisky looking `pickle.loads` up after our plugin runs.
`test_frisky_threads_keep_their_python_state_between_tasks` fails if an upgrade changes that.
The bug belongs upstream: a long-lived worker thread should keep one thread state.

**2. Exception chains lose their cause.** A task's exception keeps its `__cause__` only if tblib
has registered its classes. In a worker that happens once, when Dask is imported, and covers the
classes that exist at that moment. Dask also registers the failing chain on every failure; Frisky
never does. So rasterio's GDAL errors, imported later, come back with `__notes__` but without
`__cause__`, which is the GDAL reason every read-failure verdict is decided from
(`context_docs/ingest/source-read-failures.md`).
`_MatchDaskWorker` registers the classes once, after importing the ingest's import closure,
because tblib covers only classes that exist when it runs. Combined with `loader_failures.keep_causes_picklable`, which the ingest already
installs, the whole chain arrives.

Two caveats:

- Exception classes created later (botocore's per-service errors) still arrive without their
  cause.
- With tblib but without our reducer, Frisky's client fails to unpickle GDAL's exception. It
  raises a `RuntimeError` holding the raw pickle bytes, GDAL's message included, which a text
  classifier could misread. Production cannot reach that state, because the ingest refuses to
  read until the reducer is verified on every worker.

**3. Adaptive scaling cannot see Frisky's load.** Dask sizes an adaptive fleet from its own task
table, which under a hijack holds one task: the driver. It would shrink the fleet to
`min_workers` and retire workers holding Frisky's data. `ecs_cluster(frisky=True)` calls
`cluster.scale(max_workers)` instead, and `min_workers` is ignored.

**4. The client lacks part of Dask's API.** Frisky's client has no `run`, `register_plugin`,
`run_on_scheduler`, `scheduler_info` or `cancel`. Hence the routing client, rather than handing
the ingest Frisky's client directly.

**Measured and ruled out: thread stack size.** Frisky's task threads get Rust's default 2 MiB
stack; Dask's get 16 MiB on macOS (and typically 8 MiB on Linux). Raising Frisky's with
`RUST_MIN_STACK` did not stop the crash above, and once thread state is pinned the real ingest
succeeds at 2 MiB, so nothing changes it. If a future crash really is a stack overflow,
`RUST_MIN_STACK` set in the workers' environment (`worker_env_overrides`) is the lever.

## Verified so far

Done locally (macOS, Python 3.13), and green again in CI on Linux (Python 3.12 and 3.13) on PR
#205:

| Suite | Covers | Result |
|---|---|---|
| `tests/integration/test_frisky.py` | Every compute path the ingest uses; Dask plugins and `run` reaching Frisky's processes; a driver running as a Dask task (the Prefect shape), including a second thread; overlapped icechunk writes identical to Dask's and committing nothing when a window fails; thread state and the CRS reproduction; spans readable by `frisky observe`; span capture never failing a run | 7 passed, about 5 s |
| `tests/integration/test_read_failure_cause_over_dask.py` | Frisky's before-and-after for the cause chain; the read-failure classification on both engines | 16 passed |
| `tests/parity/test_ingest_s2_roi_frisky_parity.py` | The S2 domain ingest on Dask versus Frisky: offline synthetic dates byte for byte (with `pipeline_dates` and a mid-run gate failure), and Denver July 2024 real imagery within 1e-6 | 2 passed, about 2 min |

The dev-account test plan, and the record of its runs, is
[`frisky-dev-test-plan.md`](frisky-dev-test-plan.md). How to use Frisky, and the do's and
don'ts that follow from the findings above, are reference documentation:
[`docs/frisky.md`](../../docs/frisky.md).

## Open items

- Report the thread-state bug and the dropped exception cause upstream, at
  [mrocklin/frisky-issues](https://github.com/mrocklin/frisky-issues). Drafts with standalone
  reproductions are written but not filed. Either fix upstream lets
  the matching step of `_MatchDaskWorker` go.
- Decide whether Frisky stays a core dependency when this reaches `main`, on the experiment's
  results and the constraints under Packaging above.
