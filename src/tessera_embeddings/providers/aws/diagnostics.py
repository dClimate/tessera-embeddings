"""CloudWatch-backed log fetcher for the inference diagnostics shim.

The pure-domain ``inference.diagnostics`` module accepts a callable
``fetch_events(instance_id) -> list[dict]`` that returns CloudWatch-style
event dicts (each with a ``"message"`` key). This module provides the
AWS implementation: a factory that returns such a callable, bound to a
specific log group, region, and stream suffix.

Pair with :func:`tessera_embeddings.inference.diagnostics.log_worker_failure_diagnostic`::

    from tessera_embeddings.providers.aws.diagnostics import make_cloudwatch_fetcher
    from tessera_embeddings.inference.diagnostics import log_worker_failure_diagnostic

    fetch_events = make_cloudwatch_fetcher(log_group="/ec2/tessera/ray")
    log_worker_failure_diagnostic(
        instance_id, chunk_label, error_msg, log, fetch_events=fetch_events
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import boto3

logger = logging.getLogger(__name__)


def make_cloudwatch_fetcher(
    *,
    log_group: str,
    region: str = "us-west-2",
    stream_suffix: str = "actors",
    limit: int = 500,
    filter_pattern: str | None = None,
) -> Callable[[str], list[dict]]:
    """Return a CloudWatch-backed ``fetch_events`` callable.

    Args:
        log_group: CloudWatch log group name. Should match the
            ``cloudwatch_log_group`` passed to :func:`ray_cluster`.
        region: AWS region.
        stream_suffix: Suffix on the per-instance log stream name. Ray's
            CloudWatch agent template (see
            ``providers/aws/cloudwatch-agent.json.tpl``) writes streams
            named ``{instance_id}/{stream_suffix}`` for each log type
            (``"actors"``, ``"raylet"``, ``"workers"``, ``"other"``).
        limit: Maximum events fetched per call.
        filter_pattern: Optional CloudWatch filter pattern applied at the
            API level.

    Returns:
        A callable ``(instance_id) -> list[dict]`` suitable for the
        ``fetch_events`` parameter on
        :func:`tessera_embeddings.inference.diagnostics.build_worker_failure_diagnostic`.
    """
    client = boto3.client("logs", region_name=region)

    def _fetch(instance_id: str) -> list[dict]:
        stream_name = f"{instance_id}/{stream_suffix}"
        kwargs: dict = {
            "logGroupName": log_group,
            "logStreamNames": [stream_name],
            "limit": limit,
            "interleaved": False,
        }
        if filter_pattern:
            kwargs["filterPattern"] = filter_pattern

        events: list[dict] = []
        try:
            paginator = client.get_paginator("filter_log_events")
            for page in paginator.paginate(**kwargs):
                events.extend(page.get("events", []))
        except client.exceptions.ResourceNotFoundException:
            logger.debug("CloudWatch stream %s not found in %s", stream_name, log_group)
        except Exception:
            logger.debug("CloudWatch query failed for %s", stream_name, exc_info=True)
        return events

    return _fetch
