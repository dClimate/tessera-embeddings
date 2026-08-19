"""Ingest-stage profiling: the Dask/Fargate scheduler and its workers.

At UTM scale the scheduler — the single event-loop process that builds every
task graph and routes every task — is the saturation risk, not the workers.

- :mod:`.watch_scheduler` — live JSON snapshots + threshold alerts from the
  scheduler health heartbeat, and the post-hoc run profile.
- :mod:`.ingest_log_queries` — Logs-Insights aggregates across every worker
  stream (catalog throttling, retry storms, S3 SlowDown, worker lifecycle):
  the scheduler-vs-external discriminator.
- :mod:`.report` — assembles the two into a per-run dossier.

See ``README.md`` in this directory for the per-run workflow.
"""

from __future__ import annotations

#: CloudWatch log group the ingest Dask cluster ships to — the single source for
#: both tools. It mirrors ``providers.aws.dask.DEFAULT_CLOUDWATCH_LOG_GROUP``, and
#: a test pins the two together: importing the provider from here instead would
#: drag dask/distributed/dask-cloudprovider into a CLI that only needs boto3.
DEFAULT_INGEST_LOG_GROUP = "/ecs/tessera/dask"
