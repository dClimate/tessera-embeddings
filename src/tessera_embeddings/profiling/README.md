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
  `te-watch-scheduler` (live scheduler heartbeat → JSON + alerts),
  `te-ingest-log-queries` (429/503/retry/worker-exit aggregates across every
  worker stream), `te-ingest-report` (assemble the per-run dossier).
- **"Are the GPUs busy? are workers OOMing?"** → `inference/` —
  `te-observe-cluster` (live GPU/RAM pollers + post-hoc CloudWatch rollups),
  plus the `te-compare-*` output-equivalence gates.

## Invocation

Every tool is installed as a console script, so any project depending on this
library gets them on `PATH` — no checkout or path juggling:

| Command | Module under `tessera_embeddings.profiling` |
| --- | --- |
| `te-watch-scheduler` | `ingest.watch_scheduler` |
| `te-ingest-log-queries` | `ingest.ingest_log_queries` |
| `te-ingest-report` | `ingest.report` |
| `te-observe-cluster` | `inference.observe_cluster` |
| `te-compare-outputs` | `inference.compare_outputs` |
| `te-compare-stores` | `inference.compare_coarsened_stores` |

Each is equally runnable from a checkout as
`python -m tessera_embeddings.profiling.<stage>.<tool>`, which is what to use
when iterating on the tools themselves.

The three AWS-facing tools read CloudWatch, ECS and EC2, so they need the `aws`
extra (`pip install tessera_embeddings[aws]`) for boto3; the comparison gates
need only the base install. Nothing in the library imports this subpackage, so
an ordinary `import tessera_embeddings` never pulls a cloud SDK in on its
account — the tools load only when a command runs.

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
