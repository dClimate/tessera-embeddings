"""The library runner must claim a staging run before honouring resume state.

Regression test for the review finding that `run_inference` — the documented,
orchestrator-neutral entry point — adopted staged chunks on the strength of their
labels alone, with the identity check living only in the Prefect wrapper. A direct
caller reusing a ``run_id`` under a different model, ROI or time window could
therefore blend two encoders into one store.

The claim is the first thing `run_inference` does after validating ``num_actors``,
so these tests reach it without a Ray cluster: a conflicting claim raises before
any actor is requested. That ordering is the property under test, not an accident
of the harness — if the claim ever moved below actor creation, these would start
needing Ray and fail.
"""

from __future__ import annotations

import logging

import pytest

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.assembly import ZarrWriter, staging_fingerprint
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.runner import run_inference

MOSAIC = "s3://inputs/mosaics/roi"


def _cfg(**over) -> InferenceConfig:
    kwargs = dict(
        time_window=parse_time_window("October 2025"),
        model_version="v2-large",
        representation_dim=128,
        checkpoint_path="s3://bucket/models/student_large.pt",
        s1_orbit="both",
        num_obs_checkpoints=(8, 16),
    )
    kwargs.update(over)
    return InferenceConfig(**kwargs)  # type: ignore[arg-type]


def _chunks() -> list[ChunkSpec]:
    return [ChunkSpec(row=0, col=0, y_start=0, y_stop=4, x_start=0, x_stop=4)]


def _run(cfg: InferenceConfig, staging: str, run_id: str, mosaic: str = MOSAIC, **kw):
    return run_inference(
        num_actors=1,
        config=cfg,
        chunks=_chunks(),
        mosaic_base=mosaic,
        staging_base=staging,
        run_id=run_id,
        t0=0.0,
        log=logging.getLogger(__name__),
        **kw,
    )


def test_resuming_a_v2_run_as_v11_is_refused(tmp_path):
    """The defect this closes: finishing a v2 staging run with the v1.1 encoder."""
    staging = str(tmp_path / "staging")
    v2 = _cfg()
    ZarrWriter(staging).claim_run("shared", staging_fingerprint(v2, MOSAIC))

    v11 = _cfg(
        model_version="v1.1", representation_dim=192, checkpoint_path="s3://b/models/tessera_v1_1_aws_encoder.pt"
    )
    with pytest.raises(ValueError, match="different configuration"):
        _run(v11, staging, "shared")


def test_resuming_under_a_different_roi_is_refused(tmp_path):
    """Chunk labels are grid positions, so another ROI's chunk_0_0 collides by name."""
    staging = str(tmp_path / "staging")
    cfg = _cfg()
    ZarrWriter(staging).claim_run("shared", staging_fingerprint(cfg, "s3://inputs/mosaics/alps"))
    with pytest.raises(ValueError, match="different configuration"):
        _run(cfg, staging, "shared", mosaic="s3://inputs/mosaics/iowa")


def test_resuming_under_a_different_year_is_refused(tmp_path):
    """Same ROI, different window: labels are IDENTICAL, so only this catches it."""
    staging = str(tmp_path / "staging")
    ZarrWriter(staging).claim_run("shared", staging_fingerprint(_cfg(), MOSAIC))
    other_year = _cfg(time_window=parse_time_window("October 2024"))
    with pytest.raises(ValueError, match="different configuration"):
        _run(other_year, staging, "shared")


def test_unclaimed_legacy_staging_is_refused_by_default(tmp_path):
    staging = str(tmp_path / "staging")
    (tmp_path / "staging" / "legacy").mkdir(parents=True)
    ZarrWriter(staging).write_skip_marker(_chunks()[0], run_id="legacy")
    with pytest.raises(ValueError, match="no identity manifest"):
        _run(_cfg(), staging, "legacy")


def test_claim_precedes_actor_creation(tmp_path):
    """Pins the ordering these tests depend on.

    If the claim moved below actor creation, a conflicting resume would spin up
    GPU actors before refusing — and this test would fail on a missing Ray
    context rather than on the ValueError.
    """
    staging = str(tmp_path / "staging")
    ZarrWriter(staging).claim_run("shared", staging_fingerprint(_cfg(), MOSAIC))
    with pytest.raises(ValueError, match="different configuration"):
        _run(_cfg(s1_orbit="ascending"), staging, "shared")


def test_num_actors_validation_still_precedes_the_claim(tmp_path):
    """Cheap argument checks stay first; no staging side effect from a bad call."""
    staging = str(tmp_path / "staging")
    with pytest.raises(ValueError, match="num_actors"):
        run_inference(
            num_actors=0,
            config=_cfg(),
            chunks=_chunks(),
            mosaic_base=MOSAIC,
            staging_base=staging,
            run_id="never",
            t0=0.0,
            log=logging.getLogger(__name__),
        )
    assert not (tmp_path / "staging" / "never").exists()
