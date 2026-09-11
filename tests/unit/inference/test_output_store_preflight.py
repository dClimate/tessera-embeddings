"""The single-ROI preflight gate: reject an impossible append before provisioning.

``assemble`` validates the embedding manifest against an existing store anyway, but
it runs after inference. On the single-ROI path that means a model change against an
existing store is discovered once the GPU fleet has already done the work. The gate
moves the same check to flow entry, where it costs one metadata read.
"""

from __future__ import annotations

import pytest
import zarr

from tessera_embeddings.config.inference import DEFAULT_MODEL_VERSION, EMBEDDING_DIM, encoder_url
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.orchestration_helpers import (
    assert_output_store_accepts,
    build_embedding_manifest,
    build_inference_config,
    embedding_store_path,
)
from tessera_embeddings.storage.manifest import ConfigMismatchError
from tessera_embeddings.storage.zarr_store import open_or_create_repo

_ROI = "story_county"


def _config(tmp_path, checkpoint: str = "tessera_v1_aws.pt"):
    return build_inference_config(
        s1_orbit="both",
        time_window=parse_time_window("June 2025"),
        checkpoint_path=f"{tmp_path}/models/{checkpoint}",
        inputs_bucket=str(tmp_path / "inputs"),
        output_bucket=str(tmp_path / "outputs"),
    )


def _existing_store(tmp_path, config) -> str:
    """An output store carrying the manifest a run with *config* would write."""
    path = embedding_store_path(str(tmp_path / "outputs"), _ROI)
    repo, _ = open_or_create_repo(path)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    root.create_array("embeddings", shape=(1, 8, 8, EMBEDDING_DIM), dtype="int8", fill_value=0)
    root.attrs["_manifest"] = build_embedding_manifest(config=config, mosaic_base=None).to_dict()
    session.commit("seed")
    return path


def test_absent_store_is_not_a_failure(tmp_path):
    """The first run for an ROI has nothing to append to."""
    assert_output_store_accepts(
        output_bucket=str(tmp_path / "outputs"),
        roi_name=_ROI,
        output_name_suffix="",
        config=_config(tmp_path),
        mosaic_base=None,
    )


def test_matching_store_passes(tmp_path):
    config = _config(tmp_path)
    _existing_store(tmp_path, config)

    assert_output_store_accepts(
        output_bucket=str(tmp_path / "outputs"),
        roi_name=_ROI,
        output_name_suffix="",
        config=config,
        mosaic_base=None,
    )


def test_changed_model_is_rejected_before_any_compute(tmp_path):
    """The case worth paying a metadata read to catch.

    The store was written by one checkpoint and the run embeds with another;
    ``assemble`` would refuse the append, so refuse it now instead of after the
    inference bill.
    """
    _existing_store(tmp_path, _config(tmp_path, checkpoint="tessera_v1_aws.pt"))

    with pytest.raises(ConfigMismatchError, match="model_checkpoint"):
        assert_output_store_accepts(
            output_bucket=str(tmp_path / "outputs"),
            roi_name=_ROI,
            output_name_suffix="",
            config=_config(tmp_path, checkpoint="tessera_v1_mpc.pt"),
            mosaic_base=None,
        )


def test_a_suffixed_run_targets_its_own_store(tmp_path):
    """``output_name_suffix`` names a different store, so it is not gated on this one."""
    _existing_store(tmp_path, _config(tmp_path, checkpoint="tessera_v1_aws.pt"))

    assert_output_store_accepts(
        output_bucket=str(tmp_path / "outputs"),
        roi_name=_ROI,
        output_name_suffix="-experiment",
        config=_config(tmp_path, checkpoint="tessera_v1_mpc.pt"),
        mosaic_base=None,
    )


def test_the_output_path_has_one_definition():
    """The gate and the assembly task both call this, so they cannot target
    different stores.
    """
    assert embedding_store_path("s3://b/out/", _ROI, "-x") == f"s3://b/out/embeddings/{_ROI}-x.zarr"
    assert embedding_store_path("s3://b/out", _ROI) == f"s3://b/out/embeddings/{_ROI}.zarr"


# ── the encoder family, checked here rather than after the GPU bill ──


def _store_published_by(tmp_path, config, encoder_url_value: str | None) -> str:
    """An output store whose root advertises *encoder_url_value* (None = legacy, no attr)."""
    path = _existing_store(tmp_path, config)
    repo, _ = open_or_create_repo(path)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    if encoder_url_value is None:
        # `_existing_store` never writes the attr, so a legacy store is the store as built.
        # Popping a key that was never there leaves the session empty, and icechunk refuses an
        # empty commit — hence no commit on this branch rather than a defensive try/except.
        if "geoemb:model" not in root.attrs:
            return path
        del root.attrs["geoemb:model"]
    else:
        root.attrs["geoemb:model"] = encoder_url_value
    session.commit("set encoder identity")
    return path


def test_a_cross_family_append_is_refused_before_any_compute(tmp_path) -> None:
    """The whole point of this gate. The mismatch is DETERMINISTIC — store plus config, nothing
    inference discovers — so leaving it to `assemble` meant the fleet ran to completion and the
    result could then never be published. The manifest cannot stand in: it compares a checkpoint
    stem, and is skipped outright for a legacy store.
    """
    config = _config(tmp_path)
    _store_published_by(tmp_path, config, encoder_url(DEFAULT_MODEL_VERSION))

    v2_config = build_inference_config(
        s1_orbit="both",
        time_window=parse_time_window("June 2025"),
        checkpoint_path=f"{tmp_path}/models/student_large.pt",
        inputs_bucket=str(tmp_path / "inputs"),
        output_bucket=str(tmp_path / "outputs"),
        model_version="v2-large",
    )
    with pytest.raises(ValueError, match="Refusing to append"):
        assert_output_store_accepts(
            output_bucket=str(tmp_path / "outputs"),
            roi_name=_ROI,
            output_name_suffix="",
            config=v2_config,
            mosaic_base=None,
        )


def test_a_legacy_store_with_no_encoder_attr_is_read_as_v11(tmp_path) -> None:
    """Absence dates the store rather than leaving the question open — and it is the case most
    likely to also lack a `_manifest`, so both doors would otherwise stand open together.
    """
    config = _config(tmp_path)
    _store_published_by(tmp_path, config, None)

    v2_config = build_inference_config(
        s1_orbit="both",
        time_window=parse_time_window("June 2025"),
        checkpoint_path=f"{tmp_path}/models/student_large.pt",
        inputs_bucket=str(tmp_path / "inputs"),
        output_bucket=str(tmp_path / "outputs"),
        model_version="v2-large",
    )
    with pytest.raises(ValueError, match="Refusing to append"):
        assert_output_store_accepts(
            output_bucket=str(tmp_path / "outputs"),
            roi_name=_ROI,
            output_name_suffix="",
            config=v2_config,
            mosaic_base=None,
        )


def test_the_matching_family_still_passes_the_gate(tmp_path) -> None:
    """The path that must keep working — a same-family append is the ordinary case, and a gate
    that blocked it would be worse than none.
    """
    config = _config(tmp_path)
    _store_published_by(tmp_path, config, encoder_url(DEFAULT_MODEL_VERSION))
    assert_output_store_accepts(
        output_bucket=str(tmp_path / "outputs"),
        roi_name=_ROI,
        output_name_suffix="",
        config=config,
        mosaic_base=None,
    )
