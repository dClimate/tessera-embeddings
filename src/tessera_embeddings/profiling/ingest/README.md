# Ingest performance harness

> **Scope: the INGEST half of the pipeline** — the Dask / Fargate stage that
> downloads and mosaics S1/S2 into per-zone stores. For the inference half (the
> Ray / GPU fill), see [`../inference/`](../inference/README.md). Start at
> [`../README.md`](../README.md) for which harness to reach for.

At UTM scale the ingest **scheduler** — the single event-loop process that
builds every task graph and routes every task — is the named saturation risk,
not the workers. Its stress is otherwise invisible until it stalls (the
built-in `Event loop was unresponsive for Ns` warning) or the task is
OOM-killed. This harness makes the run-up traceable and machine-consumable.

Deployment-agnostic. Credentials come from the ambient AWS chain, so
`export AWS_PROFILE=<your-profile>` once and the examples below run as written;
`--profile` / `--region` override per call. `--log-group` defaults to
`/ecs/tessera/dask`, which is where this repo's Dask provider ships. All tools
need only CloudWatch Logs read access.

## In-process instrumentation (already on the scheduler)

`SchedulerResourceLogger` (in `tessera_embeddings.providers.aws.dask`) is
registered on every ingest scheduler and emits a 30 s heartbeat to the
`dask-scheduler` stream:

```
scheduler health: cpu=100% rss=1.50GiB mem=18% lag=5.0s fds=42 threads=9 workers=250 tasks=90000 processing=480 no-worker=220
```

- `cpu` / `rss` / `mem` — scheduler-process load and the OOM predictor.
- `lag` — event-loop lag (how late the probe fired): the direct stall signal.
- `processing` / `no-worker` — tasks in flight vs stuck waiting; a rising
  `no-worker` backlog is the scheduler falling behind on assignment.

When `cpu` or `lag` crosses a threshold it additionally logs a **`scheduler
stack sample`** line — a collapsed tally of the busiest code locations across
the process's threads, sampled off the event loop, to attribute a stall (graph
build vs comms vs work-stealing vs GC) without a process-attach profiler.

Optionally, an ingest can capture a full Dask `performance_report` (task
stream / worker profile / bandwidth panels): set `IngestSettings.perf_report_uri`
to a base URI and `ingest-zone-year` writes a per-child HTML there
(`s2.html`, `s1-<orbit>.html`). Off by default; use it on a probe rung, not the
campaign (large graphs make the report heavy).

## Tools

- **`watch_scheduler.py`** — the centerpiece.
  - `--live`: tail the scheduler stream, parse each health line into a rolling
    window, derive rates (task-count change, backlog slope, worker churn),
    evaluate saturation thresholds, and emit one JSON snapshot per line to
    **stdout** (with ALERT keys), plus a human line to stderr.
  - `--report --since ... --until ...`: post-hoc profile — full time series,
    per-metric peaks, saturation-onset timestamps, cpu∧backlog co-occurrence,
    worker join/exit events — as JSON (consumed by `report.py`); `--markdown`
    adds a paste-ready section.
- **`ingest_log_queries.py`** — a Logs-Insights query pack answering "scheduler
  or external?": catalog HTTP retries by service (CMR vs earth-search/STAC) and
  status (429 throttle vs 503 SlowDown), S1 (ASF) download-retry storms, S3
  SlowDown, and worker exit/restart events with reasons. `--list` shows the
  queries; runs one, several, or all over a run window and returns JSON.
- **`report.py`** — assembles the per-run **dossier**: merges the
  `watch_scheduler --report` JSON + the `ingest_log_queries` JSON (+ an optional
  `performance_report` link) into a markdown skeleton with an **Interpretation**
  section for the operator to fill in (bottleneck verdict, limiting factor, next
  rung). Pure over the JSON — the raw JSON stays the system of record.

## Per-run workflow

```
export AWS_PROFILE=<your-profile>   # or add --profile <p> to each call below

# 1. while the run is live — machine-readable snapshots + alerts
te-watch-scheduler --live | tee live.jsonl

# 2. after the run — scheduler profile + external-service aggregates
te-watch-scheduler --report --since <t0> --until <t1> > sched.json
te-ingest-log-queries --since <t0> --until <t1> > logs.json

# 3. assemble the dossier, then write the interpretation
te-ingest-report --scheduler sched.json --logs logs.json \
    --run-id <id> --zone <z> --year <y> --max-workers <n> --out dossier.md
```

## Live scheduler dashboard

The scheduler task runs with `enableExecuteCommand`, so its Dask dashboard
(`:8787`) is reachable over an SSM port-forward — `ecs_cluster` logs the exact
`aws ssm start-session ...` command at startup (see `log_dashboard_ssm_command`
in `providers/aws/dask.py`). The dashboard is a convenience; the CloudWatch
heartbeat + these tools are the durable record (they survive teardown).
