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

Five things. The first two are in none of Frisky's docs and showed only when the real ingest
ran; the next two follow from reading Dask's adaptive scaling and Frisky's client; the fifth came
from a whole run's spans on dev.

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

The hijack does not wait for that fleet. A Dask `Client` given the cluster object first waits for
every requested worker's ECS task to reach RUNNING, which after the scale-up held the hijack back
73 to 77 s at 60 workers on dev, so `ecs_cluster` connects its short-lived clients by address.
Every worker therefore joins after the hijack. Dask hands a joining worker its plugins in
registration order, `_MatchDaskWorker` before Frisky's, and Frisky spreads the work it queued
while no worker existed onto the workers as they arrive
(`test_workers_that_join_after_the_hijack_run_its_queued_work`). One ordering is weaker than
Dask's. Dask holds a joining worker's tasks until all its plugins are set up, but Frisky's worker
starts inside its own plugin, so the plugins the ingest registers after the hijack (the
credential broadcast and the read-failure capture) finish setting up just after it. Theirs take
microseconds, and none of 2,000 tasks probed locally on 20 late joiners ran before them; a plugin
whose setup sleeps 0.5 s lost the race on every task. A read that did lose it would fail closed,
not drop data: `is_unreadable_source` excludes credential failures, so they propagate.

**4. The client lacks part of Dask's API.** Frisky's client has no `run`, `register_plugin`,
`run_on_scheduler`, `scheduler_info` or `cancel`. Hence the routing client, rather than handing
the ingest Frisky's client directly.

**5. The client pickles the whole graph before the first task runs.** Dask's client sends a graph
once, and its scheduler pickles each task as it dispatches it, while the fleet works. Frisky's
client lowers the graph to a task dict and pickles every task before it submits any, so the fleet
waits for all of it. On the drained Iowa S2 run at 2048 the whole fleet stood idle 32 s and 25 s
before the two write batches, against 8 s and 7 s on Dask, and about 2 s before each of the 11
gate computes, which ran one after another where Dask overlaps them. The same tasks took 9% less
time on Frisky, so this idle time is its whole per-date gap: 80 s of the 208 s run, against 20 s
on Dask.

About half of a write batch's pickling was avoidable. icechunk's merge reduction wraps its two
functions in a `functools.wraps` closure, which plain pickle refuses, so each task holding one
(about 22,000 of an Iowa batch's 47,000) went through cloudpickle by value. `connect()` swaps in a
partial of module-level functions, which pickles by reference and computes the same thing
(`_picklable_merge_reduction`). Locally, on an Iowa-sized batch of 47,104 tasks with 11,264
writes, the time from the write call to the first task falls from 9.9 s to 5.5 s:

| Driver, write call to first task (local, s) | Dask | Frisky | Frisky with the partial |
|---|---|---|---|
| Graph build | 1.2 | 1.2 | 1.2 |
| Merge reduction and optimisation | 1.3 | 1.3 | 0.4 |
| Lower to a dict, then pickle every task | – | 7.2 | 3.7 |
| Scheduler takes the graph, first task starts | 3.9 | 0.2 | 0.2 |
| **Total** | **6.4** | **9.9** | **5.5** |

Locally the Dask scheduler shares the driver's process; on Fargate it runs on its own task. Most
of what remains on Frisky, lowering the graph and pickling the other tasks, is how its client
works. On dev, side by side with Dask on Iowa, the fix cut the idle stretch before each write
batch from 32 s and 25 s to 10 s and 8 s (Dask's are 9 s and 8 s), and S2 took 17.9 s a date on
both engines.

At zone scale that remainder dominates. On 35N at 60 workers a date's write graph holds 108,000 to
168,000 tasks, and the fleet sat idle 15 to 29 s before each one while the driver converted it.
`pipeline_dates` does not hide it: it prepares the next date in the background, but a write's
conversion happens inside the write. So the S2 ingest passes its client to the store write, which
splits a date's windows into `WRITE_SUBMISSION_GROUPS` (4) contiguous groups and submits them one
after another with `client.compute`. That call returns as soon as a group's graph is converted and
submitted, so the fleet runs the first group while the driver converts the next. The groups'
results merge into the session together, so a date is still one commit, and a failing group
commits nothing. Locally, one synthetic date (11 variables, row-band windows, instant reads, 4
workers of 2 threads), two runs each:

| One date, local | Tasks | Write call to first task | Write, wall |
|---|---|---|---|
| Frisky, one graph | 47,000 | 4.3 to 4.5 s | 28 to 31 s |
| Frisky, four groups | 47,000 | 1.6 s | 26 s |
| Frisky, one graph | 106,000 | 11.1 to 11.4 s | 67 to 69 s |
| Frisky, four groups | 106,000 | 2.9 s | 60 to 68 s |
| Dask, one graph | 47,000 | 5.3 to 5.7 s | 44 to 45 s |
| Dask, four groups | 47,000 | 2.6 s | 45 s |
| Dask, one graph | 106,000 | 11.9 to 14.0 s | 114 to 120 s |
| Dask, four groups | 106,000 | 4.5 to 5.1 s | 114 to 129 s |

At 106,000 tasks on Frisky the four groups were submitted 2.8, 4.6 to 4.8, 6.6 to 7.0 and 8.5 to
9.0 s after the write call, so the first group's tasks ran while the other three were still being
converted. The conversion itself also fell, from about 9.4 s to 7.5 s, because `client.compute`
lowers the graph more cheaply than the blocking compute's route does (`client.lower_graph` totals
about 4.4 s, against about 7 s of `client.convert_legacy`). On Dask the first task also starts
sooner, but the tasks then take longer, and the write's wall clock is unchanged within the runs'
spread; locally Dask's scheduler shares the driver's process, which Fargate's does not.

On 35N on dev (version C, with `pipeline_dates`) the grouping pays only on Frisky. Frisky's writes
ran 3% to 16% shorter from the second date on, and its time between dates fell from 165 s to 152 s.
Dask's writes ran 31% to 35% longer on every date, and its time between dates rose from 211 s to
269 s: its scheduler, already at 100% of its core, now takes four graphs a date instead of one.

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
| `tests/integration/test_frisky.py` | Every compute path the ingest uses; Dask plugins and `run` reaching Frisky's processes; a driver running as a Dask task (the Prefect shape), including a second thread; overlapped icechunk writes identical to Dask's and committing nothing when a window fails, as one graph and submitted in groups; thread state and the CRS reproduction; work queued before any worker exists, run on workers that join after the hijack; spans readable by `frisky observe`; their merge functions pickling by reference; the span drain keeping every span exactly once; span capture never failing a run; a drained run's live snapshot reading no spans | 12 passed, about 27 s |
| `tests/integration/test_read_failure_cause_over_dask.py` | Frisky's before-and-after for the cause chain; the read-failure classification on both engines | 16 passed |
| `tests/parity/test_ingest_s2_roi_frisky_parity.py` | The S2 domain ingest on Dask versus Frisky: offline synthetic dates byte for byte (with `pipeline_dates` and a mid-run gate failure), and Denver July 2024 real imagery within 1e-6 | 2 passed, about 2 min |

## On the dev account

The runs, their figures and how to rerun them are in
[`frisky-dev-test-plan.md`](frisky-dev-test-plan.md). Up to zone scale at production width:

- **Stable and exact.** No worker died in any run, and Frisky's stores are bit-identical to Dask's
  at both chunk sizes. One Frisky scheduler died, at the end of a zone run, from its span buffers
  (below).
- **Faster at zone scale, because Dask's scheduler saturates.** On 35N for a week Frisky took 193 s
  a date against Dask's 275 s, and 152 s against 269 s with `pipeline_dates` and grouped writes
  (above). Dask's scheduler sat at 95% to 101% of its core and ran the coverage gate with the
  fleet under a third busy; Frisky's kept the fleet 97% busy through every write.
- **As fast as Dask at Iowa scale, once its graphs pickle cheaply.** Frisky's tasks are as fast or
  faster (store writes 0.97 of Dask's median, band reads 0.89), but it first took 25 to 40% longer
  per S2 date, idling while the driver pickled each graph (item 5 above). With the fix it matches
  Dask on S2 (17.9 s a date each) and edges it on S1 (13.1 s against 13.8 s). Dask's scheduler is
  not the bound at that scale, so Frisky's scheduling advantage has nothing to recover yet. At
  4096 a second cost shows: each batch ends on a long tail, because one worker is handed up to 1.8
  times the mean work and keeps it while the others go idle.
- **Frisky's scheduler does more work per heartbeat the longer a run goes.** Its own summary line
  in the scheduler's log (`sched: busy=… heartbeat=…`) shows the time spent on worker heartbeats
  rising steadily with the work submitted: on Iowa from about 20 ms a window at the start to
  850 ms an hour in, and the loop's busy share from 2% to 17–18%, where it levelled off. Both Iowa
  runs follow the same curve, with span buffers of 1,000,000 and of 200,000; and locally 4,000
  small graphs take it from 1 to 10 ms whatever `FRISKY_EVENT_LOG_CAPACITY` or
  `FRISKY_TRACING_CAPACITY` is, so it is neither the spans nor the event log. On the first
  year-long Iowa run the slowdown began as the busy share levelled, about an hour in: gates rose
  from 10 to 93 s and writes from 52 to 228 s, with gaps of 11 to 61 s between small tasks while
  the workers idled, and an overview query took 87 s. It is inside Frisky, so it goes upstream; a
  zone-year runs for hours, so it decides whether Frisky can take one.
- **The end-of-run capture keeps only a run's tail.** At Iowa scale Frisky emits about 15,000 spans
  a second, so the 500,000 captured at the end cover about half a minute. Each process keeps its
  own span buffer (1,000,000 by default), which a worker fills in about an hour, so the run's spans
  are still there to drain as it goes: `frisky_drain_spans` copies the task, transfer and spill
  spans every minute. They are a fifth of the spans and hold all of the task time; spans under
  1 ms would be another five times smaller but lose a quarter of the transfer time. On Iowa the
  drain kept every span Frisky did, and Frisky's tracing itself misses about one in 2,000. It asks
  for 15 seconds at a time: on 35N whole-minute requests timed out at the dashboard proxy (HTTP
  504), and in slices none of version C's requests failed, 4.8 million spans over the 35N week.
- **Frisky's default span buffer is too big for a long or zone-scale run.** Each process keeps
  `FRISKY_TRACING_CAPACITY` spans, 1,000,000 by default. On the
  pipelined 35N rerun the scheduler's memory doubled every two minutes to 5.7 GiB of its 8 GiB,
  held there once the buffer was full, reached 6.8 GiB under the end-of-run capture's three large
  span queries, and the process died without logging a close, after all seven dates had committed;
  `cluster.close()` then timed out on it and failed the flow. `ecs_cluster(frisky=True)` now
  sets 200,000, and the end-of-run capture asks the scheduler for spans once, computing both
  overviews from that file in the flow's process. At 200,000 on version C the scheduler still
  stepped up at each live snapshot while the buffers filled, and only then: 0.6 to 2.7 GiB at the
  first (5 minutes in), 3.5 GiB at the second, then flat, on 35N; 0.7 to 2.5 and 3.0 GiB on Iowa.
  The drain's queries, every minute, moved it not at all. The difference is the filter: the
  snapshot's overview asks for the latest 50,000 spans, which the scheduler finds by assembling
  every process's buffer, while the drain asks for one span name over a 15-second slice. So a
  drained run's live snapshot now reads `frisky observe cluster`, the counts alone, and the
  drain's parts hold the spans. The end-of-run `spans.json`, the one unfiltered query left, then
  took the same scheduler from 3.4 to 6.6 GiB and timed out at the dashboard proxy (HTTP 504);
  with the drain it is now the drain's own tail, so a drained run asks the scheduler only for
  filtered slices.
- **The end-of-run capture took 45 s with the drain on, against Dask's 7 s, while the whole fleet
  was billed.** The bundle's S3 timestamps on the `-fix` run split it: the final drain about 9 s,
  the 500,000-span `spans.json` 18 s before its 167 MB upload began (five pages from the
  dashboard), and that upload and the five CLI calls the last 18 s, one after another. Locally, on
  a hijacked cluster holding 600,000 spans, three changes take the same exit from 8.4 s to 2.1 s.
  The drain writes each part with one `json.dumps` at gzip level 6, where streaming `json.dump` ran
  the pure-Python encoder into level 9: a 173,000-span part takes 0.53 s instead of 2.35 s and is
  9% larger. With the drain on, `spans.json` is the drain's last 100,000 spans. And the final
  captures run concurrently. On version C's 35N run the capture took 16 s after the last date,
  most of it the failed `spans.json` query above.

The 2048-px chunk changes pixels slightly, through GDAL's approximate warp transformer rather than
through either engine; the plan's B1 result has the measurement.

How to use Frisky, and the do's and don'ts that follow from these findings, are reference
documentation: [`docs/frisky.md`](../../docs/frisky.md).

## Open items

- Report the thread-state bug and the dropped exception cause upstream, at
  [mrocklin/frisky-issues](https://github.com/mrocklin/frisky-issues). Drafts with standalone
  reproductions are written but not filed. Either fix upstream lets
  the matching step of `_MatchDaskWorker` go.
- Report the up-front graph pickling and the 4096 tail upstream, with these numbers, and ask
  icechunk to make `computing_meta` pickle by reference so `_picklable_merge_reduction` can go.
- Confirm the shorter end-of-run capture on dev: a drained Iowa run's bundle should land within
  about 10 s of its last batch, not 45.
- Decide the grouped write submission: on 35N it cut Frisky's time a date by 8% and raised
  Dask's by 27%. Grouping only on Frisky, or not at all, changes the ingest code identity either
  way.
- Report the heartbeat growth upstream, with the local reproduction.
- Find why the gate's computes run one after another on Frisky.
- Decide whether Frisky stays a core dependency when this reaches `main`, on the experiment's
  results and the constraints under Packaging above.
