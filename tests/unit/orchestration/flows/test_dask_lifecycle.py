"""The ingest flows' Dask helpers: teardown hook wiring and tag derivation, and run_on_cluster.

The 03S incident's secondary finding: a hard-cancelled ingest leaked its whole
Dask cluster because ``ecs_cluster`` tears down in a ``finally`` the kill skips,
and the ingest flows registered no hook. These tests pin the wiring — both
terminal states on both flows, and the sweep driven purely from the flow_run
argument, because Prefect runs hooks in a fresh import where module state is
gone.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from dask.distributed import LocalCluster
from prefect import flow, get_run_logger, task
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import LogFilter, LogFilterFlowRunId
from prefect.logging.handlers import APILogWorker
from prefect.testing.utilities import prefect_test_harness

from tessera_embeddings.orchestration.prefect.flows import _dask_lifecycle as mod
from tessera_embeddings.orchestration.prefect.flows.ingest_s1_roi_sar import ingest_s1_roi_sar
from tessera_embeddings.orchestration.prefect.flows.ingest_s2_roi_reflectance import ingest_s2_roi_reflectance


def test_both_ingest_flows_register_the_hook_for_both_terminal_states():
    """Cancelled AND crashed: a crash leaks exactly like a cancellation."""
    for f in (ingest_s2_roi_reflectance, ingest_s1_roi_sar):
        assert mod.dask_cleanup_on_cancellation in f.on_cancellation_hooks
        assert mod.dask_cleanup_on_cancellation in f.on_crashed_hooks


def test_hook_sweeps_by_the_flow_runs_tag(monkeypatch):
    """The sweep is derived from nothing but flow_run.id — fresh-import safe."""
    calls: list = []
    monkeypatch.setattr(mod, "_stop_ecs_tasks_by_tag", lambda k, v, *, log: calls.append((k, v)))
    mod.dask_cleanup_on_cancellation(None, SimpleNamespace(id="run-123"), None)
    assert calls == [(mod.DASK_FLOW_RUN_TAG, "run-123")]


def test_hook_without_a_run_id_is_a_noop(monkeypatch, caplog):
    calls: list = []
    monkeypatch.setattr(mod, "_stop_ecs_tasks_by_tag", lambda *a, **k: calls.append(a))
    with caplog.at_level(logging.WARNING):
        mod.dask_cleanup_on_cancellation(None, SimpleNamespace(), None)
    assert calls == []


def test_resource_tags_shape():
    assert mod.dask_resource_tags("abc") == {"tessera-flow-run-id": "abc"}
    assert mod.dask_resource_tags(None) is None  # outside a Prefect run: untagged, not crashed


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
