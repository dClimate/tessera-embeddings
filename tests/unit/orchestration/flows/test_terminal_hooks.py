"""Terminal hooks: what every flow that starts billable work runs on cancel and crash.

The Ray teardown hook runs in a freshly imported copy of its module (Prefect kills
the flow child process first), so it must be able to tear the cluster down from
nothing but the hook's ``flow_run`` argument. These tests pin that contract once,
on the shared hook, and then pin that each flow registers its hook for BOTH
terminal states. The module-scope import rule that keeps these flows importable
without the ``aws`` extra is ruff's TID253 (``ruff.toml``).
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tessera_embeddings.orchestration.prefect.flows._ray_lifecycle as lifecycle_mod
import tessera_embeddings.orchestration.prefect.flows.fill_zone_year as fill_mod
import tessera_embeddings.orchestration.prefect.flows.fill_zones_sequential as seq_mod
import tessera_embeddings.orchestration.prefect.flows.ingest_zone_year as izy_mod
import tessera_embeddings.orchestration.prefect.flows.run_global_campaign as campaign_mod
import tessera_embeddings.orchestration.prefect.flows.tessera_embeddings as flows_mod
from tessera_embeddings.orchestration.prefect.flows import _dask_lifecycle as dask_mod
from tessera_embeddings.orchestration.prefect.flows.ingest_s1_roi_sar import ingest_s1_roi_sar
from tessera_embeddings.orchestration.prefect.flows.ingest_s2_roi_reflectance import ingest_s2_roi_reflectance
from tessera_embeddings.providers.aws.ray import RAY_DOWN_TIMEOUT_S

_RUN_ID = "1cb5e1da-7454-4bd6-89f8-e20cd020dbaa"


# ---------------------------------------------------------------------------
# The shared teardown hook (_ray_lifecycle). Every Ray-owning flow registers
# this one function, so its guarantees are pinned once, here.
# ---------------------------------------------------------------------------


def test_lifecycle_hook_bounds_ray_down_and_terminates_by_tag(tmp_path: Path) -> None:
    """The shared campaign teardown hook bounds `ray down`; a hang/launch failure
    still cleans up the tempfile and terminates the cluster by tag (no leaked fleet).
    """
    yaml_file = tmp_path / "resolved.yaml"
    yaml_file.write_text("cluster_name: x\n")
    with (
        patch.object(lifecycle_mod, "_active_resolved_yaml", str(yaml_file)),
        patch.object(lifecycle_mod, "_active_cluster_name", "tessera-inference-fill01"),
        patch("tessera_embeddings.providers.aws.ray.terminate_ray_instances_by_tag") as terminate,
        patch("tessera_embeddings.providers.aws.ray.cleanup_ray_tempfiles") as cleanup,
        patch.object(
            lifecycle_mod.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="ray down", timeout=RAY_DOWN_TIMEOUT_S),
        ) as ray_down,
    ):
        lifecycle_mod.ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    assert ray_down.call_args.kwargs["timeout"] == RAY_DOWN_TIMEOUT_S
    cleanup.assert_called_once_with(str(yaml_file))
    terminate.assert_called_once_with(cluster_name="tessera-inference-fill01", log=terminate.call_args.kwargs["log"])


def test_lifecycle_hook_swallows_ray_down_launch_error(tmp_path: Path) -> None:
    """An OSError before `ray down` produces a return code (e.g. `ray` not on PATH)
    must not escape the hook — cleanup and tag-based termination still run.
    """
    yaml_file = tmp_path / "resolved.yaml"
    yaml_file.write_text("cluster_name: x\n")
    with (
        patch.object(lifecycle_mod, "_active_resolved_yaml", str(yaml_file)),
        patch.object(lifecycle_mod, "_active_cluster_name", "tessera-inference-fill01"),
        patch("tessera_embeddings.providers.aws.ray.terminate_ray_instances_by_tag") as terminate,
        patch("tessera_embeddings.providers.aws.ray.cleanup_ray_tempfiles") as cleanup,
        patch.object(lifecycle_mod.subprocess, "run", side_effect=OSError("ray: command not found")),
    ):
        lifecycle_mod.ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    cleanup.assert_called_once_with(str(yaml_file))
    terminate.assert_called_once()


def test_lifecycle_hook_prefers_the_recorded_cluster_name(tmp_path: Path) -> None:
    """With no YAML to `ray down`, the name recorded by activate() wins over the
    derived one — same-process cancellation tears down the cluster actually built.
    """
    with (
        patch.object(lifecycle_mod, "_active_resolved_yaml", None),
        patch.object(lifecycle_mod, "_active_cluster_name", "tessera-inference-stored99"),
        patch("tessera_embeddings.providers.aws.ray.terminate_ray_instances_by_tag") as terminate,
        patch.object(lifecycle_mod.subprocess, "run") as ray_down,
    ):
        lifecycle_mod.ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    assert terminate.call_args.kwargs["cluster_name"] == "tessera-inference-stored99"
    ray_down.assert_not_called()


def test_lifecycle_hook_derives_cluster_name_in_fresh_process(tmp_path: Path) -> None:
    """In a fresh import (module globals unset) the shared hook re-derives the
    cluster name from flow_run.id and terminates by tag — the deterministic-name
    fix so a cancel before activate() records the name cannot leak the fleet.
    """
    with (
        patch.object(lifecycle_mod, "_active_resolved_yaml", None),
        patch.object(lifecycle_mod, "_active_cluster_name", None),
        patch("tessera_embeddings.providers.aws.ray.terminate_ray_instances_by_tag") as terminate,
    ):
        lifecycle_mod.ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    terminate.assert_called_once()
    assert terminate.call_args.kwargs["cluster_name"] == "tessera-inference-1cb5e1da"


def test_dask_hook_sweeps_by_the_flow_runs_tag(monkeypatch) -> None:
    """A hard-cancelled ingest leaks its Dask cluster, because ``ecs_cluster`` tears down
    in a ``finally`` the kill skips. The sweep is derived from nothing but flow_run.id,
    so it works from the fresh import Prefect runs hooks in.
    """
    calls: list = []
    monkeypatch.setattr(dask_mod, "_stop_ecs_tasks_by_tag", lambda k, v, *, log: calls.append((k, v)))
    dask_mod.dask_cleanup_on_cancellation(None, SimpleNamespace(id="run-123"), None)
    assert calls == [(dask_mod.DASK_FLOW_RUN_TAG, "run-123")]


# ---------------------------------------------------------------------------
# Hook REGISTRATION: every flow that starts billable work cleans up on BOTH a
# cancellation AND a crash.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flow", "hook"),
    [
        # A `ray up` head persists until torn down, so a crashed GPU run leaks its fleet.
        (fill_mod.fill_zone_year_flow, lifecycle_mod.ray_cleanup_on_cancellation),
        (seq_mod.fill_zones_sequential_flow, lifecycle_mod.ray_cleanup_on_cancellation),
        (flows_mod.tessera_embeddings, lifecycle_mod.ray_cleanup_on_cancellation),
        # `ecs_cluster` tears down in a `finally` a hard cancel skips.
        (ingest_s2_roi_reflectance, dask_mod.dask_cleanup_on_cancellation),
        (ingest_s1_roi_sar, dask_mod.dask_cleanup_on_cancellation),
        # Each dispatched child is an independent run that keeps writing and billing.
        (campaign_mod.run_global_campaign, campaign_mod._cancel_children_on_cancellation),
        (seq_mod.fill_zones_sequential_flow, seq_mod._cancel_child_ingests_on_cancellation),
        (izy_mod.ingest_zone_year, izy_mod._cancel_children_on_cancellation),
    ],
    ids=lambda v: getattr(v, "name", None) or getattr(v, "__name__", None),
)
def test_the_hook_is_registered_for_cancel_and_crash(flow, hook) -> None:
    """Both terminal states: a crashed run leaks exactly like a cancelled one. The GPU flows
    share ONE Ray hook rather than per-flow copies, so a teardown fix lands in one place.
    """
    assert hook in flow.on_cancellation_hooks
    assert hook in flow.on_crashed_hooks


def test_every_ray_owning_flow_clears_the_hook_state_on_the_failure_path() -> None:
    """The shared hook reads process-wide module state, and Prefect reuses worker
    processes — so state left set by a failed run outlives it.

    A later run's cancellation hook then PREFERS that stale cluster name over its own
    flow-run fallback, tears down a cluster that is already gone, and leaks the live
    fleet it was called to reclaim. Clearing only on success is therefore not clearing:
    the exception path is exactly the one the hook exists for. Asserted structurally
    across all three flows, because the leak only shows up on a reused worker after a
    failure — no unit run reproduces it.
    """
    import ast

    for mod in (fill_mod, seq_mod, flows_mod):
        src = Path(inspect.getfile(mod)).read_text()
        tree = ast.parse(src)
        deactivates = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Try)
            for handler in node.finalbody
            if isinstance(handler, ast.Expr)
            and isinstance(handler.value, ast.Call)
            and getattr(handler.value.func, "id", None) == "deactivate"
        ]
        assert deactivates, f"{Path(inspect.getfile(mod)).name}: deactivate() must run from a finally block"
