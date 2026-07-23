"""allow_s2_only wiring at the flow layer.

Two concerns the flag introduces beyond the per-pixel dataset gate:

* the single-ROI flow must not let a *resume* silently mix S1-gated and
  S2-only staged chunks (``_resolve_run_id``), and
* the flag must be reachable through the documented end-to-end path
  (``tessera_full_pipeline`` forwards it to the embeddings deployment).
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from prefect.states import StateType

import tessera_embeddings.orchestration.prefect.flows.tessera_embeddings as emb_mod
import tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline as fp_mod
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.orchestration.prefect.flows.tessera_embeddings import (
    S2_ONLY_RUN_PREFIX,
    _resolve_run_id,
)

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


# ── _resolve_run_id: the resume mode-mixing guard ──


def test_fresh_run_id_is_bare_uuid_by_default() -> None:
    """Flag off → the historical bare 12-hex run_id; the single-ROI staging
    layout is unchanged when allow_s2_only is False.
    """
    run_id = _resolve_run_id(None, allow_s2_only=False, assembly_only=False)
    assert not run_id.startswith(S2_ONLY_RUN_PREFIX)
    assert len(run_id) == 12


def test_fresh_run_id_encodes_s2_only_mode() -> None:
    run_id = _resolve_run_id(None, allow_s2_only=True, assembly_only=False)
    assert run_id.startswith(S2_ONLY_RUN_PREFIX)


@pytest.mark.parametrize(
    ("previous", "flag"),
    [("abc123def456", False), (f"{S2_ONLY_RUN_PREFIX}abc123", True)],
)
def test_resume_with_matching_mode_returns_run_id(previous: str, flag: bool) -> None:
    assert _resolve_run_id(previous, allow_s2_only=flag, assembly_only=False) == previous


@pytest.mark.parametrize(
    ("previous", "flag"),
    [("abc123def456", True), (f"{S2_ONLY_RUN_PREFIX}abc123", False)],
)
def test_resume_with_flipped_mode_is_refused(previous: str, flag: bool) -> None:
    """The footgun: continuing a run under the other per-pixel S1 mode would
    publish a mix of S1-gated and S2-only tiles. Reject it loudly.
    """
    with pytest.raises(ValueError, match="mix of S1-gated and S2-only"):
        _resolve_run_id(previous, allow_s2_only=flag, assembly_only=False)


@pytest.mark.parametrize("flag", [False, True])
def test_assembly_only_resume_is_exempt_from_the_guard(flag: bool) -> None:
    """Assembly-only re-publishes whatever is staged and never runs the per-pixel
    gate, so a mode mismatch can't change its output — the guard must not block it.
    """
    previous = f"{S2_ONLY_RUN_PREFIX}abc123"
    assert _resolve_run_id(previous, allow_s2_only=flag, assembly_only=True) == previous


def test_flow_refuses_resume_mode_flip_before_any_io(monkeypatch) -> None:
    """End-to-end at the flow boundary: a resumed run with a flipped flag raises
    at the run_id step, before Ray or any store is touched.
    """
    monkeypatch.setattr(emb_mod, "get_run_logger", lambda: logging.getLogger("test-emb"))
    with pytest.raises(ValueError, match="mix of S1-gated and S2-only"):
        emb_mod.tessera_embeddings.fn(
            roi_name="demo",
            time_window_end="June 2025",
            paths=_PATHS,
            ami_ssm_name="ami",
            allow_s2_only=True,
            dev_params=emb_mod.EmbeddingsDevParams(previous_run_id="plain-run-000"),
        )


# ── tessera_full_pipeline: the flag reaches the embeddings deployment ──


def _run_master_pipeline(monkeypatch, **kwargs) -> dict:
    """Drive the master flow with every deployment dispatch and cluster-sizing
    touchpoint mocked; return {deployment_ref: parameters} for each stage.
    """
    calls: dict = {}

    async def fake_arun(deployment, parameters):
        calls[deployment] = parameters
        return SimpleNamespace(
            id=f"run-{deployment}", state=SimpleNamespace(type=StateType.COMPLETED, name="COMPLETED")
        )

    monkeypatch.setattr(fp_mod, "arun_deployment", fake_arun)
    monkeypatch.setattr(fp_mod, "_count_roi_chunks", lambda *a, **k: 10)
    monkeypatch.setattr(fp_mod, "compute_pipeline_cluster_sizing", lambda *a, **k: (1, 2, 4))
    monkeypatch.setattr(fp_mod, "get_run_logger", lambda: logging.getLogger("test-fp"))

    asyncio.run(fp_mod.tessera_full_pipeline.fn(paths=_PATHS, time_window_end="June 2025", roi_name="demo", **kwargs))
    return calls


def test_full_pipeline_forwards_allow_s2_only(monkeypatch) -> None:
    calls = _run_master_pipeline(monkeypatch, allow_s2_only=True)
    emb_params = calls[fp_mod.PipelineDeployments().tessera_embeddings]
    assert emb_params["allow_s2_only"] is True


def test_full_pipeline_allow_s2_only_defaults_off(monkeypatch) -> None:
    calls = _run_master_pipeline(monkeypatch)
    emb_params = calls[fp_mod.PipelineDeployments().tessera_embeddings]
    assert emb_params["allow_s2_only"] is False
