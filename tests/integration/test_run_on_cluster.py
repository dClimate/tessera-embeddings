"""``run_on_cluster`` keeps a Dask-run task and its logs on the calling flow run.

Integration rather than unit: it boots a Prefect API server (``prefect_test_harness``), ~15 s.
"""

from __future__ import annotations

import pytest
from dask.distributed import LocalCluster
from prefect import flow, get_run_logger, task
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import LogFilter, LogFilterFlowRunId
from prefect.logging.handlers import APILogWorker
from prefect.testing.utilities import prefect_test_harness

from tessera_embeddings.orchestration.prefect.flows import _dask_lifecycle as mod

pytestmark = pytest.mark.integration


@task
def _double(x: int) -> int:
    get_run_logger().info("progress from the task")
    return 2 * x


@flow
def _one_flow() -> int:
    with LocalCluster(n_workers=1, processes=False, dashboard_address=None) as cluster:
        return mod.run_on_cluster(cluster.scheduler_address, _double, {"x": 21})


def test_run_on_cluster_logs_onto_the_calling_flow_run():
    """The task's result comes back, and the task and its logs belong to the ONE flow run."""
    with prefect_test_harness():
        state = _one_flow(return_state=True)
        assert state.result() == 42
        run_id = state.state_details.flow_run_id
        APILogWorker.drain_all(timeout=5)  # logs ship in batches
        with get_client(sync_client=True) as client:
            assert [r.id for r in client.read_flow_runs()] == [run_id]
            assert [t.flow_run_id for t in client.read_task_runs()] == [run_id]
            logs = client.read_logs(log_filter=LogFilter(flow_run_id=LogFilterFlowRunId(any_=[run_id])))
        assert "progress from the task" in [log.message for log in logs]
