"""Ingest-stage profiling: the Dask/Fargate scheduler and its workers.

At UTM scale the scheduler — the single event-loop process that builds every task graph
and routes every task — is the saturation risk, not the workers.

- :mod:`.watch_scheduler` — live JSON snapshots + threshold alerts from the scheduler
  health heartbeat, and the post-hoc run profile.
- :mod:`.ingest_log_queries` — Logs-Insights aggregates across every worker stream
  (catalog throttling, retry storms, S3 SlowDown, worker lifecycle): the
  scheduler-vs-external discriminator.
- :mod:`.report` — assembles the two into a per-run dossier.

See ``README.md`` in this directory for the per-run workflow.
"""

from __future__ import annotations

import argparse

#: CloudWatch log group the ingest Dask cluster ships to — the single source for both
#: tools. Mirrors ``providers.aws.dask.DEFAULT_CLOUDWATCH_LOG_GROUP`` (a test pins the two
#: together); importing the provider instead would drag dask/distributed/dask-cloudprovider
#: into a CLI that only needs boto3.
DEFAULT_INGEST_LOG_GROUP = "/ecs/tessera/dask"


def ingest_log_parser(description: str | None) -> argparse.ArgumentParser:
    """A parser carrying the options both tools share: AWS profile, region and log group."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--profile", default=None, help="AWS profile; default uses the ambient credential chain / AWS_PROFILE"
    )
    parser.add_argument("--region", default="us-west-2")
    parser.add_argument("--log-group", default=DEFAULT_INGEST_LOG_GROUP)
    return parser
