"""Internal helper to construct a Prefect ``DaskTaskRunner`` for a cluster."""

from __future__ import annotations

from prefect_dask import DaskTaskRunner


def get_task_runner_for_cluster(scheduler_address: str) -> DaskTaskRunner:
    """Create a ``DaskTaskRunner`` connected to ``scheduler_address``.

    Apply it via ``inner_flow.with_options(task_runner=...)`` so Prefect binds the runner
    late, after the cluster context manager has produced a real scheduler address.
    """
    return DaskTaskRunner(
        address=scheduler_address,
        client_kwargs={"timeout": "300s"},
    )
