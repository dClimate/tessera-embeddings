"""Ray-cluster teardown paths of the tessera_embeddings flow.

The cancellation/crash hook runs in a freshly imported copy of the flow
module (Prefect kills the flow child process first), so it must be able
to tear the cluster down from nothing but the hook's ``flow_run``
argument. These tests pin that contract.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    # tempfile cleanup or the authoritative tag-based termination.
    assert ray_down.call_args.kwargs["timeout"] == flows_mod.RAY_DOWN_TIMEOUT_S
    cleanup.assert_called_once()
    terminate.assert_called_once()
    assert terminate.call_args.kwargs["cluster_name"] == "tessera-inference-stored99"
