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
from tessera_embeddings.config.inference import S1_ORBIT_NONE
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.orchestration.prefect.flows.tessera_embeddings import (
    S2_ONLY_RUN_PREFIX,
    _resolve_run_id,
    staged_s2_only_mode,
)

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


class _StopError(Exception):
    """Ends the flow once the assertion's subject has been reached."""


# ── _resolve_run_id: the resume mode-mixing guard ──


def test_fresh_run_id_is_bare_uuid_by_default() -> None:
    """Flag off → the historical bare 12-hex run_id; the single-ROI staging
    layout is unchanged when allow_s2_only is False.
    """
    run_id = _resolve_run_id(None, model_version="v1.1", allow_s2_only=False, assembly_only=False)
    assert not run_id.startswith(S2_ONLY_RUN_PREFIX)
    assert len(run_id) == 12


def test_fresh_run_id_encodes_s2_only_mode() -> None:
    run_id = _resolve_run_id(None, model_version="v1.1", allow_s2_only=True, assembly_only=False)
    assert run_id.startswith(S2_ONLY_RUN_PREFIX)


@pytest.mark.parametrize(
    ("previous", "flag"),
    [("abc123def456", False), (f"{S2_ONLY_RUN_PREFIX}abc123", True)],
)
def test_resume_with_matching_mode_returns_run_id(previous: str, flag: bool) -> None:
    assert _resolve_run_id(previous, model_version="v1.1", allow_s2_only=flag, assembly_only=False) == previous


@pytest.mark.parametrize(
    ("previous", "flag"),
    [("abc123def456", True), (f"{S2_ONLY_RUN_PREFIX}abc123", False)],
)
def test_resume_with_flipped_mode_is_refused(previous: str, flag: bool) -> None:
    """The footgun: continuing a run under the other per-pixel S1 mode would
    publish a mix of S1-gated and S2-only tiles. Reject it loudly.
    """
    with pytest.raises(ValueError, match="mix of S1-gated and S2-only"):
        _resolve_run_id(previous, model_version="v1.1", allow_s2_only=flag, assembly_only=False)


@pytest.mark.parametrize("flag", [False, True])
def test_assembly_only_resume_is_exempt_from_the_guard(flag: bool) -> None:
    """Assembly-only re-publishes whatever is staged and never runs the per-pixel
    gate, so a mode mismatch can't change its output — the guard must not block it.
    """
    previous = f"{S2_ONLY_RUN_PREFIX}abc123"
    assert _resolve_run_id(previous, model_version="v1.1", allow_s2_only=flag, assembly_only=True) == previous


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


# ── the EFFECTIVE mode, not the requested one ──


def test_the_run_id_is_minted_from_the_effective_mode_not_the_requested_one(monkeypatch) -> None:
    """``InferenceConfig`` FORCES allow_s2_only when the orbit resolves to none — every
    pixel there has zero S1 observations, so the default gate would skip all of them.

    Minting the run_id from the REQUESTED flag missed that, so S2-only chunks landed
    under an unprefixed staging prefix: an explicit S2-only resume of them is then
    refused, and the same bare id can be reused under the S1-gated mode the prefix exists
    to separate.
    """
    monkeypatch.setattr(emb_mod, "get_run_logger", lambda: logging.getLogger("test-emb"))
    monkeypatch.setattr(emb_mod, "resolve_s1_orbit", lambda *a, **k: S1_ORBIT_NONE)
    # The mint is the call under test. The early guard no longer reaches _resolve_run_id
    # on a fresh run at all: it can only settle the request-is-True direction, because
    # the effective flag is forced ON and never off, so a request of False may still
    # become True. Asserting the flag the MINT saw pins the behaviour this test names;
    # asserting how many times the helper was called pinned the old implementation.
    calls: list[bool] = []

    def record(previous, *, allow_s2_only, assembly_only, **_):
        calls.append(allow_s2_only)
        raise _StopError

    monkeypatch.setattr(emb_mod, "_resolve_run_id", record)
    with pytest.raises(_StopError):
        emb_mod.tessera_embeddings.fn(
            roi_name="demo",
            time_window_end="June 2025",
            paths=_PATHS,
            ami_ssm_name="ami",
            require_s1=False,
            allow_s2_only=False,  # requested off; the resolved orbit forces it on
        )
    assert calls == [True], "the mint must see the EFFECTIVE mode, not the request"


def test_a_forced_s2_only_resume_is_not_refused_by_the_early_guard(monkeypatch) -> None:
    """The false refusal the narrowed guard exists to stop.

    A radar-free ROI stages under the FORCED flag, so its run_id carries the S2-only
    prefix. Resuming it with ``require_s1=False`` and the flag left at its default had
    the early guard compare the prefix against the REQUEST and refuse — before any orbit
    was resolved, and for a resume that the late check (on the effective flag) accepts.
    """
    monkeypatch.setattr(emb_mod, "get_run_logger", lambda: logging.getLogger("test-emb"))
    monkeypatch.setattr(emb_mod, "resolve_s1_orbit", lambda *a, **k: S1_ORBIT_NONE)
    seen: list[bool] = []

    def record(previous, *, allow_s2_only, assembly_only, **_):
        seen.append(allow_s2_only)
        raise _StopError

    monkeypatch.setattr(emb_mod, "_resolve_run_id", record)
    with pytest.raises(_StopError):
        emb_mod.tessera_embeddings.fn(
            roi_name="demo",
            time_window_end="June 2025",
            paths=_PATHS,
            ami_ssm_name="ami",
            require_s1=False,
            allow_s2_only=False,
            dev_params=emb_mod.EmbeddingsDevParams(previous_run_id=f"{emb_mod.S2_ONLY_RUN_PREFIX}abc123def456"),
        )
    assert seen == [True], "the guard must defer to the effective flag, which is forced on"


def test_an_assembly_only_resume_publishes_under_the_staged_mode() -> None:
    """Assembly-only never runs the per-pixel gate, so the requested flag cannot change
    WHICH pixels it publishes — but it still reaches the EmbeddingManifest.

    Exempting it outright let a staged S2-only run be published as flag-off, after which
    a later flag-off append mixes incompatible slices into one store. The run_id prefix
    is the record of what the staged pixels ARE, so the mode is read off it.
    """
    assert staged_s2_only_mode(f"{S2_ONLY_RUN_PREFIX}abc123") is True
    assert staged_s2_only_mode("plain-run-000") is False
    assert staged_s2_only_mode(None) is False


# ── the master pipeline can actually select the second model ──


def test_full_pipeline_forwards_model_version(monkeypatch) -> None:
    """The child flow HAS a default, so an unforwarded selector fails silently: the run
    produces v1.1 embeddings while the operator believes they chose v2.
    """
    calls = _run_master_pipeline(monkeypatch, model_version="v2-large")
    emb_params = calls[fp_mod.PipelineDeployments().tessera_embeddings]
    assert emb_params["model_version"] == "v2-large"


def test_full_pipeline_model_version_defaults_to_v11(monkeypatch) -> None:
    calls = _run_master_pipeline(monkeypatch)
    emb_params = calls[fp_mod.PipelineDeployments().tessera_embeddings]
    assert emb_params["model_version"] == "v1.1"
