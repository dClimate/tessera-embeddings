# Profiling harnesses

The Tessera pipeline has two compute-heavy stages, and they saturate on
completely different resources. Each has its own profiling harness here — pick
by **which stage you are watching**, not by which account you are on (both
harnesses are deployment-agnostic; you pass the profile / region / log group).

| Stage | What runs it | Saturates on | Harness |
| --- | --- | --- | --- |
| **Ingest** — download + mosaic S1/S2 into per-zone stores | Dask on AWS Fargate (one scheduler + many workers) | the **scheduler** event loop (graph build + task routing), and external catalog/S3 throttling | [`ingest/`](ingest/README.md) |
| **Inference** — Ray GPU fill of embeddings from mosaics | Ray on EC2 GPU workers (g6e) | **GPU / host RAM** on the workers | [`inference/`](inference/README.md) |

## Which one do I want?

- **"Is the ingest scheduler falling behind at N workers?"** → `ingest/` —
  `watch_scheduler.py` (live scheduler heartbeat → JSON + alerts),
  `ingest_log_queries.py` (429/503/retry/worker-exit aggregates across every
  worker stream), `report.py` (assemble the per-run dossier).
- **"Are the GPUs busy? are workers OOMing?"** → `inference/` —
  `observe_cluster.py` (live GPU/RAM pollers + post-hoc CloudWatch rollups),
  plus the `compare_*` output-equivalence gates.

## Why they are separate

Inference profiling is mature and GPU/RAM-centric: the workers are the
interesting thing, and the tooling discovers them by Ray tags and samples them
over SSM. Ingest profiling is scheduler-centric: the interesting thing is a
single event-loop process whose stress shows up as 30-second heartbeat lines
and gradual backlog growth across thousands of worker log streams — a bad fit
for human watching, which is why `ingest/` leans on machine-readable JSON
snapshots, threshold alerts, and Logs-Insights aggregates meant to be consumed
and summarized by an agent. Keeping the two harnesses in sibling directories
(rather than one grab-bag) keeps each one's discovery model, log conventions,
and output formats coherent.
