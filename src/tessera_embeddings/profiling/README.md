# Profiling harnesses

The Tessera pipeline has two compute-heavy stages, and they saturate on
completely different resources. Each has its own profiling harness here — pick
by **which stage you are watching**, not by which account you are on (both
harnesses are deployment-agnostic; you supply the credentials / region / log
group).

| Stage | What runs it | Saturates on | Harness |
| --- | --- | --- | --- |
| **Ingest** — download + mosaic S1/S2 into per-zone stores | Dask on AWS Fargate (one scheduler + many workers) | the **scheduler** event loop (graph build + task routing), and external catalog/S3 throttling | [`ingest/`](ingest/README.md) |
| **Inference** — Ray GPU fill of embeddings from mosaics | Ray on EC2 GPU workers (g6e) | **GPU / host RAM** on the workers | [`inference/`](inference/README.md) |

## Which one do I want?

| Question | Reach for |
| --- | --- |
| Is the ingest scheduler falling behind at N workers? | `te-watch-scheduler` (live heartbeat → JSON + alerts), `te-ingest-log-queries` (429/503/retry/worker-exit aggregates), `te-ingest-report` (per-run dossier) |
| Where does a date's time go, and what did a mode change buy? | `te-ingest-log-queries` — `date_stage_timings`, `batch_timings` (its `batch_dates > 1` counterpart: one shared write per batch, divide by `n_dates`), `pipeline_stalls` (emitted in both modes, so the A/B is one query) |
| Are the GPUs busy? Are workers OOMing? | `te-observe-cluster` (live GPU/RAM pollers + CloudWatch rollups), plus the `te-compare-*` equivalence gates |

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

**Install the `aws` extra** (`pip install "tessera_embeddings[aws]"`) for any of
these against real infrastructure: the three AWS-facing tools need boto3 for
CloudWatch/ECS/EC2, and the comparison gates need `s3fs` the moment their inputs
are `s3://` URIs (they run on the base install only for local paths).

Nothing in the library imports this subpackage, so an ordinary
`import tessera_embeddings` never pulls a cloud SDK in on its account — the tools
load only when a command runs. An architecture rule enforces that, so the
isolation can't quietly lapse.

**Credentials.** No tool hardcodes an account: they resolve credentials through
the ambient AWS chain, so `export AWS_PROFILE=<your-profile>` (or an instance
role) is enough, and `--profile` / `--region` override per call. Only
`--log-group` carries a deployment-shaped default — `/ecs/tessera/dask` for
ingest, the yield Ray group for inference — so pass it on other deployments.

## Why two harnesses, and why AWS-coupled

They are separate because the interesting object differs. Inference profiling
watches **workers** — GPU and host RAM — discovered by Ray tags and sampled over
SSM. Ingest profiling watches a **single event-loop process**, whose stress
shows up as 30-second heartbeat lines and slow backlog growth across thousands
of worker streams. That is a poor fit for human watching, which is why `ingest/`
emits machine-readable JSON snapshots and threshold alerts meant to be consumed
by an agent. Sibling directories keep each one's discovery model, log
conventions and output formats coherent.

They are AWS-coupled because there is no vendor-neutral way to read CloudWatch
Logs, ECS task metadata, EC2 tags, or reach a worker over SSM.

**The structure is a template, not a dead end**, and PRs generalizing it are
welcome. What carries over to any cloud is the shape: an in-process health
heartbeat, a log-derived time series parsed into snapshots with threshold
alerts, a query pack separating "our scheduler is saturated" from "an external
service is throttling us", and a per-run dossier a human or agent interprets.
The parsers, threshold rules and dossier assembly are already
provider-agnostic — only the log-read and query layers are CloudWatch-shaped.
See `docs/providers/adding-your-own.md` for how the rest of the codebase handles
the same split.
