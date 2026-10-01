# Frisky as the ingest's scheduler: design, findings, test plan

**Status: experimental, opt-in, off by default.** Frisky 0.7.2, wired in
`src/tessera_embeddings/providers/frisky.py`. Nothing changes for a run that does not ask for it.

[Frisky](https://getfrisky.dev/) is Matthew Rocklin's reimplementation of Dask's scheduler and
workers in Rust: about 100x cheaper scheduling (around 3 µs a task) and far more telemetry,
exposed to people and agents through the `frisky observe` CLI. Two things make it worth an
experiment here. The ingest's scheduler is a known bound: graph size and scheduler RAM shape most
of the levers in `docs/ingest-performance.md`. And the profiling we have is hand-built: a scheduler
heartbeat plugin and a capped Dask performance report.

**Licence.** Frisky is not open source. It ships as a free binary under its author's own licence,
which permits commercial use and redistribution but not modification. That is why it is the
optional `frisky` extra and is kept out of `all`. It is also why it never becomes a base
dependency of this public repository.

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

**2. Exception chains lose their cause.** Frisky returns a task's exception with plain pickle,
which keeps `__notes__` but drops `__cause__`. The cause is the GDAL reason every read-failure
verdict is decided from (`context_docs/ingest/source-read-failures.md`). Dask avoids this by
installing tblib's chain-preserving reducers on every failure. `_MatchDaskWorker` installs them
once, after importing the ingest's import closure, because tblib covers only classes that exist
when it runs. Combined with `loader_failures.keep_causes_picklable`, which the ingest already
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

## Do's and don'ts

**Do:**

- **Turn it on per run.** Pass `use_frisky=True` to `ingest_s2_roi_reflectance` or
  `ingest_s1_roi_sar`, or set `frisky: true` in the plain runner's config. The campaign and fill
  flows deliberately do not pass it through.
- **Install the `frisky` extra in both images:** the flow runner's, which calls
  `frisky.hijack`, and the one the Fargate scheduler and workers run (`DASK_ECR_IMAGE_URI`),
  where the hijack's plugins import Frisky.
- **Size the fleet with `max_workers`.** The fleet is fixed at that size.
- **Capture telemetry with `perf_report_uri`.** On a Frisky run it receives Frisky's spans as
  JSON, readable after the cluster is gone with `frisky observe overview spans.json`.
- **Watch a live run through the existing SSM port-forward to 8787**, which now serves Frisky's
  dashboard. Start with `frisky observe overview http://localhost:8787`.
- **To upgrade Frisky,** move the bound in `pyproject.toml`, then run `test_frisky.py`,
  `test_read_failure_cause_over_dask.py` and `test_ingest_s2_roi_frisky_parity.py`. The Frisky
  before-and-after test fails when an upgrade makes the tblib step unnecessary.

**Don't:**

- **Call Frisky's API outside `providers/frisky.py`.**
- **Use a raw `frisky.hijack` for the ingest.** Without `_MatchDaskWorker`, the first real image
  read segfaults the worker.
- **Reference the `frisky` module from a function pickled by value** (one defined in `__main__`,
  a notebook or a test's lambda). Frisky replaces its module's class, and cloudpickle cannot
  pickle the module object.
- **Pass `pure=` to Frisky's `submit`.** It is forwarded to your function as an argument. Frisky
  keys every call uniquely anyway.
- **Keep a Frisky client open longer than its work in a shared process,** such as a test session.
  It reroutes every bare `.compute()` in that process.
- **Count on Dask's memory management for Frisky's data.** Dask's pause and spill cannot see it,
  but the nanny still kills a worker process at 95% of its memory limit. Frisky's own thresholds
  are `FRISKY_SPILL_FRACTION` and `FRISKY_SPILL_TARGET_FRACTION`, settable through
  `worker_env_overrides`.
- **Read the heartbeat's task fields as the ingest's load.** Under Frisky,
  `SchedulerResourceLogger`'s `tasks`, `processing` and `wmanaged` count only Dask's single driver
  task. Its `cpu`, `rss` and
  `lag` still measure the scheduler process, which now also hosts Frisky's scheduler.
- **Mistake the periodic stdout summary for an error.** Frisky prints one by default;
  `FRISKY_SUMMARY=off` silences it.

## Test plan

### a) Does it work?

Done locally (macOS, Python 3.13), and green again in CI on Linux (Python 3.12 and 3.13) on PR
#205:

| Suite | Covers | Result |
|---|---|---|
| `tests/integration/test_frisky.py` | Every compute path the ingest uses; Dask plugins and `run` reaching Frisky's processes; a driver running as a Dask task (the Prefect shape), including a second thread; overlapped icechunk writes identical to Dask's and committing nothing when a window fails; thread state and the CRS reproduction; spans readable by `frisky observe`; span capture never failing a run | 8 passed, about 8 s |
| `tests/integration/test_read_failure_cause_over_dask.py` | Frisky's before-and-after for the cause chain; the read-failure classification on both engines | 16 passed |
| `tests/parity/test_ingest_s2_roi_frisky_parity.py` | The S2 domain ingest on Dask versus Frisky: offline synthetic dates byte for byte (with `pipeline_dates` and a mid-run gate failure), and Denver July 2024 real imagery within 1e-6 | 2 passed, about 2 min |

Next, on the dev account:

1. **Smoke.** One small S2 probe rung with `use_frisky=True` and about 10 workers. Check:
   - The log line "Frisky loaded onto the cluster" appears.
   - Every worker appears in `frisky observe workers`. This also proves the security group lets
     workers reach the Frisky scheduler's random port.
   - CloudWatch shows no "Restarting worker".
   - The store passes the checks a Dask run does.
2. **Equivalence at scale.** The same zone tile and window ingested by both engines into
   sibling stores, compared with `assert_zarr_equivalent`. Do this once for S2 and once for one S1
   orbit; S1 has no Frisky parity test yet.
3. **A full fleet.** Confirm the worker count reaches `max_workers`, and that workers which join
   late run tasks. Most of a Fargate fleet joins after the hijack.

### b) Is it faster?

- **Same rungs.** Use the probe rungs already used for Dask
  (`context_docs/ingest/campaign-ingest-measurements.md`), at two or three ROI sizes.
- **Pair the arms.** Run Dask with `min_workers == max_workers`, so the comparison measures the
  engine rather than autoscaling. Repeat each pair two or three times, alternating which engine
  goes first.
- **Record each rung's configuration** with its result, so any number here can be rerun: commit,
  image tag, Frisky version, ROI and date window, `max_workers`, worker CPU and memory, and every
  ingest flag passed (`batch_dates`, `pipeline_dates`, `overlap_window_writes`).
- **Measure:**
  - wall clock per leg;
  - the per-batch `Batch timings` lines (`build`, `gate`, `write`, `stall`);
  - the scheduler heartbeat's `cpu`, `rss` and `lag`;
  - worker-hours.
- **The hypothesis to test** is that Frisky helps where the scheduler bounds the run (wide fleets,
  large `batch_dates`, the width at which `MAX_PIPELINE_DATES_WORKERS` stops pipelining) and not
  where image reads do.
- **The only evidence so far** is the network-bound Denver toy, where the engines are
  indistinguishable. Both ran on a laptop with 2 workers of 2 threads and 2 GB each, default
  ingest flags, Denver July 2024 (13 dates, 10 written). Dask's figures are from
  `test_s2_roi_parity`'s domain run; Frisky's from the same call on a hijacked cluster of the same
  shape:

| Engine | `gate` per batch (s) | `write` per batch (s) |
|---|---|---|
| Dask | 3.7, 6.4, 1.9 | 13.6, 13.7, 11.2 |
| Frisky | 5.0, 3.5, 6.1 | 15.3, 14.8, 11.4 |

### c) Is the telemetry better?

- **Capture on every rung.** Set `perf_report_uri` on each Frisky rung. Read the spans offline
  with `frisky observe overview`, `prefixes`, `stragglers` and `transfers`.
- **Compare like with like.** Set the Dask performance report from the equivalent Dask rung
  alongside them.
- **The test is concrete:** for the slowest date of a rung, can each engine's telemetry say why
  (critical path, GIL time, transfer time, a straggling worker), and at what cost (capture size,
  capture time, scheduler overhead)?
- **First evidence is this branch's own debugging.** Dask's logs said only "Restarting worker".
  Frisky's event log named the dying workers' lifetimes and the read tasks that killed them.
- **Limits to know:**
  - The capture keeps the most recent `SPANS_CAPTURE_LIMIT` (500,000) spans.
  - Workers record up to `FRISKY_TRACING_CAPACITY` spans (default 1,000,000).
  - The scheduler's event log keeps `FRISKY_EVENT_LOG_CAPACITY` events.

## Open items

- Report the thread-state bug and the dropped exception cause upstream, at
  [mrocklin/frisky-issues](https://github.com/mrocklin/frisky-issues). Either fix upstream lets
  the matching step of `_MatchDaskWorker` go.
- `uv sync --all-extras`, which CI and the contributor instructions use, now installs Frisky. That
  is what runs these tests in CI, at the price of a closed-source binary in every contributor's
  environment (`--no-extra frisky` opts out).
