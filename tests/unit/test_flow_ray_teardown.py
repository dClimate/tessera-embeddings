"""Ray-cluster teardown: the one hook every GPU flow registers.

The cancellation/crash hook runs in a freshly imported copy of its module
(Prefect kills the flow child process first), so it must be able to tear the
cluster down from nothing but the hook's ``flow_run`` argument. These tests pin
that contract once, on the shared hook, and then pin that each Ray-owning flow
registers THAT hook for both terminal states — a crashed run leaks the head EC2
node exactly like a cancelled one.
"""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import tessera_embeddings.orchestration.prefect.flows._ray_lifecycle as lifecycle_mod
import tessera_embeddings.orchestration.prefect.flows.fill_zone_year as fill_mod
import tessera_embeddings.orchestration.prefect.flows.fill_zones_sequential as seq_mod
import tessera_embeddings.orchestration.prefect.flows.tessera_embeddings as flows_mod
from tessera_embeddings.providers.aws.ray import RAY_DOWN_TIMEOUT_S, cluster_name_for_flow_run

_RUN_ID = "1cb5e1da-7454-4bd6-89f8-e20cd020dbaa"


def test_cluster_name_is_deterministic_and_template_based() -> None:
    """Every flow pins its cluster name from this, and the hook re-derives it."""
    name = cluster_name_for_flow_run(_RUN_ID)
    assert name == "tessera-inference-1cb5e1da"
    assert name == cluster_name_for_flow_run(_RUN_ID)


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


# ---------------------------------------------------------------------------
# Hook REGISTRATION: every GPU (Ray-owning) flow must clean up on BOTH a
# cancellation AND a crash — a crashed run leaks the head EC2 node forever.
# ---------------------------------------------------------------------------


def test_fill_zone_year_registers_teardown_on_cancel_and_crash() -> None:
    """The per-cell GPU fill must tear down on cancel AND crash (a crashed run
    would otherwise leak the whole Ray GPU fleet — the Dask ingest flows self-heal
    via their scheduler idle-timeout, but a `ray up` head persists until torn down).
    It uses the SHARED :mod:`._ray_lifecycle` hook (not a per-flow copy).
    """
    flow = fill_mod.fill_zone_year_flow
    assert lifecycle_mod.ray_cleanup_on_cancellation in flow.on_cancellation_hooks
    assert lifecycle_mod.ray_cleanup_on_cancellation in flow.on_crashed_hooks


def test_fill_zones_sequential_registers_teardown_on_cancel_and_crash() -> None:
    """The shared-cluster GPU fill has the same both-hooks contract via the shared
    hook — a crashed year-long run must not leak its long-lived Ray cluster.
    """
    flow = seq_mod.fill_zones_sequential_flow
    assert lifecycle_mod.ray_cleanup_on_cancellation in flow.on_cancellation_hooks
    assert lifecycle_mod.ray_cleanup_on_cancellation in flow.on_crashed_hooks


def test_tessera_embeddings_registers_teardown_on_cancel_and_crash() -> None:
    """The single-ROI GPU sibling registers the SAME shared hook, not a copy of it.

    It carried its own duplicate for a while; teardown failures leak billed GPU
    instances, so one implementation means one place to fix when that is found.
    """
    flow = flows_mod.tessera_embeddings
    assert lifecycle_mod.ray_cleanup_on_cancellation in flow.on_cancellation_hooks
    assert lifecycle_mod.ray_cleanup_on_cancellation in flow.on_crashed_hooks


def test_both_gpu_flows_terminate_retired_actors_instances() -> None:
    """Idle actors must take their EC2 instance with them, on either GPU path.

    End-of-run teardown covers the fleet, but actors go idle at the TAIL, while the
    last chunks finish — and after ``ray.kill()`` the autoscaler's idle timeout is
    unreliable, because it relies on the node self-reporting empty
    (providers/aws/gotchas.md). Without the retire hook those nodes bill until the
    run ends. Asserted as source, since wiring it requires a live Ray context.
    """
    for module in (fill_mod, flows_mod):
        source = inspect.getsource(module)
        assert "on_actor_retire" in source, module.__name__
        assert "make_instance_terminator" in source, module.__name__


def test_ray_owning_flows_do_not_import_the_aws_provider_at_module_scope() -> None:
    """A `tessera_embeddings[prefect]` install must be able to IMPORT these flows.

    ``providers.aws.ray`` pulls in ``boto3`` from the ``aws`` extra and ``ray`` from
    ``inference``. Imported at module scope, that makes registering or inspecting these
    flows impossible without both — including to run them locally or against a non-AWS
    provider, which need neither. The Dask twin (``_dask_lifecycle``) already defers for
    this reason; a static check is what keeps the Ray side from drifting back, since a
    module-scope import only fails on an install this test suite does not run under.
    """
    import ast

    for mod in (lifecycle_mod, fill_mod, seq_mod, flows_mod):
        tree = ast.parse(Path(inspect.getfile(mod)).read_bytes())
        offenders = [
            node.module
            for node in tree.body  # module scope only — a deferred import is nested
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tessera_embeddings.providers.aws")
        ]
        assert not offenders, f"{Path(inspect.getfile(mod)).name} imports {offenders} at module scope"
