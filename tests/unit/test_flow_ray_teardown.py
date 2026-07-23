"""Ray-cluster teardown paths of the tessera_embeddings and fill-zone-year flows.

The cancellation/crash hook runs in a freshly imported copy of the flow
module (Prefect kills the flow child process first), so it must be able
to tear the cluster down from nothing but the hook's ``flow_run``
argument. These tests pin that contract for both flows' hooks.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tessera_embeddings.orchestration.prefect.flows.fill_zone_year as fill_mod
import tessera_embeddings.orchestration.prefect.flows.tessera_embeddings as flows_mod
from tessera_embeddings.orchestration.prefect.flows.tessera_embeddings import (
    _cluster_name_for_flow_run,
    _ray_cleanup_on_cancellation,
)

_RUN_ID = "1cb5e1da-7454-4bd6-89f8-e20cd020dbaa"


def test_cluster_name_is_deterministic_and_template_based() -> None:
    name = _cluster_name_for_flow_run(_RUN_ID)
    assert name == "tessera-inference-1cb5e1da"
    assert name == _cluster_name_for_flow_run(_RUN_ID)


def test_hook_terminates_by_derived_tag_in_fresh_process() -> None:
    # Fresh-import conditions: module globals unset, only flow_run.id known.
    with (
        patch.object(flows_mod, "_active_resolved_yaml", None),
        patch.object(flows_mod, "_active_cluster_name", None),
        patch.object(flows_mod, "terminate_ray_instances_by_tag") as terminate,
        patch.object(flows_mod.subprocess, "run") as ray_down,
    ):
        _ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    terminate.assert_called_once()
    assert terminate.call_args.kwargs["cluster_name"] == "tessera-inference-1cb5e1da"
    ray_down.assert_not_called()


def test_hook_prefers_same_process_cluster_name() -> None:
    with (
        patch.object(flows_mod, "_active_resolved_yaml", None),
        patch.object(flows_mod, "_active_cluster_name", "tessera-inference-stored99"),
        patch.object(flows_mod, "terminate_ray_instances_by_tag") as terminate,
    ):
        _ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    assert terminate.call_args.kwargs["cluster_name"] == "tessera-inference-stored99"


def test_hook_bounds_ray_down_and_falls_through_on_hang(tmp_path: Path) -> None:
    """A hung `ray down` (same-process YAML fast path) must not block the
    authoritative tag-based termination: the call is time-bounded, and a timeout
    still cleans up the tempfile and terminates the cluster by tag.
    """
    yaml_file = tmp_path / "resolved.yaml"
    yaml_file.write_text("cluster_name: x\n")
    with (
        patch.object(flows_mod, "_active_resolved_yaml", str(yaml_file)),
        patch.object(flows_mod, "_active_cluster_name", "tessera-inference-stored99"),
        patch.object(flows_mod, "terminate_ray_instances_by_tag") as terminate,
        patch.object(flows_mod, "cleanup_ray_tempfiles") as cleanup,
        patch.object(
            flows_mod.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="ray down", timeout=flows_mod.RAY_DOWN_TIMEOUT_S),
        ) as ray_down,
    ):
        _ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    # `ray down` was invoked with a finite timeout, and the hang did not prevent
    # tempfile cleanup (of the resolved YAML) or the authoritative tag-based termination.
    assert ray_down.call_args.kwargs["timeout"] == flows_mod.RAY_DOWN_TIMEOUT_S
    cleanup.assert_called_once_with(str(yaml_file))
    terminate.assert_called_once()
    assert terminate.call_args.kwargs["cluster_name"] == "tessera-inference-stored99"


def test_hook_falls_through_on_ray_down_launch_error(tmp_path: Path) -> None:
    """An OSError before `ray down` even runs (e.g. `ray` not on PATH) must be caught
    so execution still reaches tempfile cleanup and tag-based termination.
    """
    yaml_file = tmp_path / "resolved.yaml"
    yaml_file.write_text("cluster_name: x\n")
    with (
        patch.object(flows_mod, "_active_resolved_yaml", str(yaml_file)),
        patch.object(flows_mod, "_active_cluster_name", "tessera-inference-stored99"),
        patch.object(flows_mod, "terminate_ray_instances_by_tag") as terminate,
        patch.object(flows_mod, "cleanup_ray_tempfiles") as cleanup,
        patch.object(flows_mod.subprocess, "run", side_effect=OSError("ray: command not found")),
    ):
        _ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    cleanup.assert_called_once_with(str(yaml_file))
    terminate.assert_called_once()


# ---------------------------------------------------------------------------
# fill-zone-year teardown hook (separate implementation, same guarantees)
# ---------------------------------------------------------------------------


def test_fill_hook_bounds_ray_down_and_terminates_by_tag(tmp_path: Path) -> None:
    """The fill flow's teardown hook bounds `ray down`; a hang/launch failure still
    cleans up the tempfile and terminates the cluster by tag (no leaked GPU fleet).
    """
    yaml_file = tmp_path / "resolved.yaml"
    yaml_file.write_text("cluster_name: x\n")
    with (
        patch.object(fill_mod, "_active_resolved_yaml", str(yaml_file)),
        patch.object(fill_mod, "_active_cluster_name", "tessera-inference-fill01"),
        patch.object(fill_mod, "terminate_ray_instances_by_tag") as terminate,
        patch.object(fill_mod, "cleanup_ray_tempfiles") as cleanup,
        patch.object(
            fill_mod.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="ray down", timeout=fill_mod.RAY_DOWN_TIMEOUT_S),
        ) as ray_down,
    ):
        fill_mod._ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    assert ray_down.call_args.kwargs["timeout"] == fill_mod.RAY_DOWN_TIMEOUT_S
    cleanup.assert_called_once_with(str(yaml_file))
    terminate.assert_called_once_with(cluster_name="tessera-inference-fill01", log=terminate.call_args.kwargs["log"])


def test_fill_hook_derives_cluster_name_in_fresh_process(tmp_path: Path) -> None:
    """In a fresh import (module globals unset) the fill hook re-derives the cluster
    name from flow_run.id and terminates by tag — the deterministic-name fix so a
    cancel before the YAML is read cannot leak the fleet.
    """
    with (
        patch.object(fill_mod, "_active_resolved_yaml", None),
        patch.object(fill_mod, "_active_cluster_name", None),
        patch.object(fill_mod, "terminate_ray_instances_by_tag") as terminate,
    ):
        fill_mod._ray_cleanup_on_cancellation(None, SimpleNamespace(id=_RUN_ID), None)
    terminate.assert_called_once()
    assert terminate.call_args.kwargs["cluster_name"] == "tessera-inference-1cb5e1da"


# ---------------------------------------------------------------------------
# Hook REGISTRATION: every GPU (Ray-owning) flow must clean up on BOTH a
# cancellation AND a crash — a crashed run leaks the head EC2 node forever.
# ---------------------------------------------------------------------------


def test_fill_zone_year_registers_teardown_on_cancel_and_crash() -> None:
    """The per-cell GPU fill must tear down on cancel AND crash (a crashed run
    would otherwise leak the whole Ray GPU fleet — the Dask ingest flows self-heal
    via their scheduler idle-timeout, but a `ray up` head persists until torn down).
    """
    flow = fill_mod.fill_zone_year_flow
    assert fill_mod._ray_cleanup_on_cancellation in flow.on_cancellation_hooks
    assert fill_mod._ray_cleanup_on_cancellation in flow.on_crashed_hooks


def test_tessera_embeddings_registers_teardown_on_cancel_and_crash() -> None:
    """The single-ROI GPU sibling has the same both-hooks contract."""
    flow = flows_mod.tessera_embeddings
    assert flows_mod._ray_cleanup_on_cancellation in flow.on_cancellation_hooks
    assert flows_mod._ray_cleanup_on_cancellation in flow.on_crashed_hooks
