# Running the ingest on Frisky

[Frisky](https://getfrisky.dev/) is a Rust reimplementation of Dask's scheduler and workers, with
much cheaper scheduling and far more telemetry. The S2 and S1 ingest can run their compute on it
instead of Dask. It is experimental and off by default; a run that does not ask for it is
unchanged.

Frisky is a core dependency on this branch. It is closed-source (a free binary licence) and installs
only on Python below 3.15. Why it is wired the way it is, and the evidence behind every rule
below: [`context_docs/ingest/frisky-experiment.md`](../context_docs/ingest/frisky-experiment.md).

## Turning it on

| Entry point | How |
|---|---|
| `ingest_s2_roi_reflectance` flow | `use_frisky=True` |
| `ingest_s1_roi_sar` flow | `use_frisky=True` |
| Plain runner | `frisky: true` in the YAML config |

The campaign flows (`ingest_zone_year`, the fills) do not pass it through. To ingest a zone on
Frisky, dispatch the S2 or S1 ingest flow directly against the zone's ROI.

What changes when it is on:

- The same Dask cluster is built, and Frisky is loaded onto it: a Frisky scheduler inside the
  Dask scheduler process, a Frisky worker inside every Dask worker process.
- **The fleet is fixed at `max_workers`.** Dask's adaptive scaling cannot see Frisky's tasks, so
  `min_workers` is ignored.
- `perf_report_uri` receives Frisky's spans as JSON instead of a Dask performance report.
- The dashboard on port 8787 is Frisky's. Dask's own pages (`/workers`, `/health`) keep their
  paths.

## What a deployment needs

- **Both images built from this branch:** the flow runner's, which loads Frisky onto the cluster,
  and the Dask image (`DASK_ECR_IMAGE_URI`), where the scheduler and workers import it. An image
  without Frisky fails the run at cluster start.
- **A security group that admits Frisky's scheduler port.** It is a random port beside Dask's
  8786, dialled on the scheduler's private IP. A group that admits only 8786 and 8787 leaves every
  Frisky worker unregistered.

## Watching a run

Use the SSM port-forward the flow logs for the Dask dashboard; it now opens Frisky's. Then, from
your machine:

```bash
frisky observe overview http://localhost:8787     # start here: state, costliest spans, stragglers
frisky observe workers  --url http://localhost:8787
frisky observe events   http://localhost:8787 --kind worker_removed   # workers that died, and when
frisky observe --help                             # the rest: prefixes, blocked, transfers, ...
```

After the run, read the spans `perf_report_uri` captured:

```bash
frisky observe overview spans.json
```

That capture keeps the most recent 500,000 spans (`SPANS_CAPTURE_LIMIT`), so it holds only a run's
tail: at Iowa scale (60 workers, 2048-px chunks) Frisky emits about 15,000 spans a second, and the
capture covers the last half minute.

**To keep the whole run,** also set `frisky_drain_spans=True`. Every minute the run then copies its
task, transfer and spill spans (`SPAN_DRAIN_NAMES`, a fifth of all spans and every second of task
time) to `<perf_report_uri>/spans/part-NNNNNN.json.gz`. That is about 9 GB a day gzipped per 60
workers. `spans.json` then keeps only the last 100,000 spans (`SPANS_CAPTURE_LIMIT_DRAINED`), about
7 s at Iowa scale, because the parts already hold every task span. To read a run's parts as one
file:

```bash
python -c "import glob, gzip, json, sys; json.dump([s for p in sorted(glob.glob(sys.argv[1] + '/part-*.json.gz')) for s in json.load(gzip.open(p, 'rt'))], sys.stdout)" spans > run-spans.json
frisky observe overview run-spans.json
```

Frisky's tracing misses about one span in 2,000, so a task count taken from spans runs that much
short; the drain itself loses none. The parts are raw material, not a record: once a run is
analysed, delete them with
`aws s3 rm --recursive <perf_report_uri>/spans/`. The rest of the bundle is small and stays.

Frisky also prints a periodic cluster summary to stdout; `FRISKY_SUMMARY=off` silences it.

## Do's and don'ts

**Do:**

- Size the fleet with `max_workers`.
- Set `perf_report_uri` on any run you want to analyse after its cluster is gone, and
  `frisky_drain_spans` as well if the analysis needs more than the run's last half minute.
- Delete `<perf_report_uri>/spans/` when the analysis is done. A global run's parts reach
  terabytes.
- Pass Frisky's own settings through `worker_env_overrides`. Examples are
  `FRISKY_SPILL_FRACTION` and `FRISKY_SPILL_TARGET_FRACTION` for its spill thresholds, and
  `FRISKY_TRACING_CAPACITY` for spans kept per worker.
- After upgrading Frisky, run `tests/integration/test_frisky.py`,
  `tests/integration/test_read_failure_cause_over_dask.py` and
  `tests/parity/test_ingest_s2_roi_frisky_parity.py`. They pin the library behaviour the ingest
  relies on.

**Don't:**

- **Call Frisky's API outside `providers/frisky.py`.** That module works around four Frisky
  behaviours (see the context doc), and code that bypasses it loses the workarounds.
- **Put a closure, lambda or `functools.wraps` wrapper in a big graph.** Frisky's client pickles
  every task before it submits any, and the fleet waits meanwhile. Plain pickle refuses those, so
  each task holding one goes through cloudpickle by value. `connect` already fixes icechunk's
  merge closures, which were half of a write batch's pickling, and the ROI mask's block reader
  is module-level for the same reason: as a closure it was most of each coverage gate's.
- **Use a raw `frisky.hijack` for the ingest.** Without our worker plugin, the first real image
  read segfaults the worker and the run hangs.
- **Reference the `frisky` module from a function pickled by value,** such as one defined in
  `__main__`, a notebook or a test's lambda. cloudpickle cannot pickle the module object.
- **Pass `pure=` to Frisky's `submit`.** It is forwarded to your function as an argument. Frisky
  keys every call uniquely anyway.
- **Keep a Frisky client open longer than its work in a shared process,** such as a test session.
  It reroutes every bare `.compute()` in that process to Frisky.
- **Give a worker plugin a slow `setup`.** On a worker that joins after the hijack, which on
  Fargate is every worker, Frisky's tasks can start before a plugin registered after the hijack
  has finished setting up.
- **Count on Dask's memory management for Frisky's data.** Dask's pause and spill cannot see it,
  but Dask's nanny still kills a worker process at 95% of its memory limit.
- **Read the scheduler heartbeat's task counts as the ingest's load.** Under Frisky, the
  `tasks`, `processing` and `wmanaged` fields count only Dask's one driver task. Its `cpu`, `rss`
  and `lag` still measure the scheduler process, which now also hosts Frisky's scheduler.
