"""Shared Dask/ECS teardown hook for cluster-owning ingest flows.

The Dask analogue of :mod:`._ray_lifecycle`, for the leak found on the first real campaign cell:
``ecs_cluster`` tears down in a ``finally``, which a hard cancel skips (Prefect kills the flow
process first), leaving the scheduler and workers running in ECS. (NOT the provider's
``skip_cleanup`` flag — that only disables dask-cloudprovider's startup sweep for debris from
PRIOR runs, and turning it off breaks cluster construction under AWS SSO.)

Simpler than the Ray hook because it is tag-based end to end: the flows tag every cluster resource
with their flow-run id at provisioning (``resource_tags``), and the hook re-derives that tag from
nothing but the ``flow_run`` argument — no module state to survive Prefect's fresh-import hook
process. Register :func:`dask_cleanup_on_cancellation` as BOTH ``on_cancellation`` and
``on_crashed``: a crashed run leaks exactly like a cancelled one.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prefect_dask import DaskTaskRunner

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
    _stop_ecs_tasks_by_tag(DASK_FLOW_RUN_TAG, str(run_id), log=log)


def _stop_ecs_tasks_by_tag(tag_key: str, tag_value: str, *, log: logging.Logger) -> None:
    """Lazy-import wrapper over the AWS provider's ECS sweep.

    Deferred, NOT imported at module scope: both ingest flows import this module unconditionally,
    and ``providers.aws.dask`` pulls in ``dask-cloudprovider`` and ``psutil`` from the ``aws``
    extra, so a module-scope import would make a supported ``tessera_embeddings[prefect]`` install
    unable to so much as IMPORT the ingest flows — including to run them with ``use_local=True`` or
    a non-AWS provider, which need no AWS packages at all. Deferring costs an import only on the
    AWS teardown path, where a cluster is being swept anyway. Same pattern as
    ``run_global_campaign``'s ``_resolve_code_identity``.

    An ``ImportError`` here is NOT a failure: a ``use_local=True`` run on a ``[prefect]``-only
    install still reaches this hook on cancellation, where the import that never mattered — there
    is no ECS cluster to sweep — would otherwise raise and fail the terminal hook. The absence of
    the AWS provider is itself the proof there is nothing to tear down, so it is logged and
    swallowed. Every OTHER exception propagates: on a real AWS run a failed sweep means leaked ECS
    tasks that keep billing, which is precisely what this hook exists to prevent.
    """
    try:
        from tessera_embeddings.providers.aws.dask import stop_ecs_tasks_by_tag
    except ImportError as exc:
        log.info("AWS provider not installed (%s) — no ECS cluster to sweep for this run.", exc)
        return

    stop_ecs_tasks_by_tag(tag_key, tag_value, log=log)


def get_task_runner_for_cluster(scheduler_address: str) -> DaskTaskRunner:
    """Create a ``DaskTaskRunner`` connected to ``scheduler_address``.

    Apply it via ``inner_flow.with_options(task_runner=...)`` so Prefect binds the runner
    late, after the cluster context manager has produced a real scheduler address.

    ``prefect_dask`` is imported here rather than at module scope to keep this module
    import-light, matching :mod:`._ray_lifecycle`, which defers its provider import for the
    same reason. Prefect re-imports a hook module in a fresh process, and a teardown hook is
    the wrong place to add import-time weight.
    """
    from prefect_dask import DaskTaskRunner

    return DaskTaskRunner(
        address=scheduler_address,
        client_kwargs={"timeout": "300s"},
    )
