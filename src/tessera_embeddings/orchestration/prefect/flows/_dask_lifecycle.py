"""Shared Dask/ECS teardown hook for cluster-owning ingest flows.

The Dask analogue of :mod:`._ray_lifecycle`, for the leak found on the first
real campaign cell: ``ecs_cluster`` tears down in a ``finally``, which a hard
cancel skips (Prefect kills the flow process first), leaving the scheduler and
workers running in ECS. (NOT the provider's ``skip_cleanup`` flag — that only
disables dask-cloudprovider's startup sweep for debris from PRIOR runs, and
turning it off breaks cluster construction under AWS SSO.)

Simpler than the Ray hook because it is tag-based end to end: the flows tag
every cluster resource with their flow-run id at provisioning
(``resource_tags``), and the hook re-derives that tag from nothing but the
``flow_run`` argument — no module state to survive Prefect's fresh-import hook
process. Register :func:`dask_cleanup_on_cancellation` as BOTH
``on_cancellation`` and ``on_crashed``: a crashed run leaks exactly like a
cancelled one.
"""

from __future__ import annotations

import logging

from tessera_embeddings.providers.aws.dask import stop_ecs_tasks_by_tag

#: Tag key stamped on every ECS resource a flow's Dask cluster creates.
DASK_FLOW_RUN_TAG = "tessera-flow-run-id"


def dask_resource_tags(flow_run_id: object) -> dict[str, str] | None:
    """The ``resource_tags`` for this flow run's cluster (None outside a run)."""
    return {DASK_FLOW_RUN_TAG: str(flow_run_id)} if flow_run_id else None


def dask_cleanup_on_cancellation(flow: object, flow_run: object, state: object) -> None:  # noqa: ARG001
    """Emergency ECS teardown when an ingest flow is cancelled OR crashes.

    Must remain idempotent — see the module docstring.
    """
    log = logging.getLogger(__name__)
    run_id = getattr(flow_run, "id", None)
    if not run_id:
        log.warning("No flow-run id on the cancellation hook — cannot sweep ECS tasks.")
        return
    log.warning("Flow cancelled/crashed — sweeping ECS tasks for run %s", run_id)
    stop_ecs_tasks_by_tag(DASK_FLOW_RUN_TAG, str(run_id), log=log)
