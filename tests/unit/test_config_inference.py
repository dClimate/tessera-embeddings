"""Unit tests for config/inference.py validation and helpers."""

from __future__ import annotations

import pytest

from tessera_embeddings.config.inference import (
    InferenceConfig,
    _normalize_obs_checkpoints,
    checkpoint_filename,
)
from tessera_embeddings.config.time_windows import parse_time_window

# ---------------------------------------------------------------------------
# checkpoint_filename
# ---------------------------------------------------------------------------


def test_checkpoint_filename_aws() -> None:
    assert checkpoint_filename("aws") == "tessera_v1_1_aws_encoder.pt"


def test_checkpoint_filename_mpc() -> None:
    assert checkpoint_filename("mpc") == "tessera_v1_1_mpc_encoder.pt"


def test_checkpoint_filename_default_is_aws() -> None:
    assert checkpoint_filename() == checkpoint_filename("aws")


def test_checkpoint_filename_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown norm_source"):
        checkpoint_filename("bogus")


def test_checkpoint_filename_v2_large() -> None:
    assert checkpoint_filename(model_version="v2-large") == "student_large.pt"


def test_checkpoint_filename_v2_ignores_norm_source() -> None:
    """v2 ships one checkpoint per student size — no norm_source split."""
    assert checkpoint_filename("mpc", model_version="v2-large") == checkpoint_filename(model_version="v2-large")


# ---------------------------------------------------------------------------
# _normalize_obs_checkpoints
# ---------------------------------------------------------------------------


def test_normalize_obs_checkpoints_sorts() -> None:
    assert _normalize_obs_checkpoints((8, 32, 16)) == (8, 16, 32)


def test_normalize_obs_checkpoints_deduplicates() -> None:
    assert _normalize_obs_checkpoints((8, 8, 16)) == (8, 16)


def test_normalize_obs_checkpoints_filters_nonpositive() -> None:
    assert _normalize_obs_checkpoints((0, -1, 8, 16)) == (8, 16)


def test_normalize_obs_checkpoints_coerces_floats() -> None:
    assert _normalize_obs_checkpoints((8.0, 16.0)) == (8, 16)  # type: ignore[arg-type]


def test_normalize_obs_checkpoints_empty_after_filter_raises() -> None:
    with pytest.raises(ValueError, match="at least one positive integer"):
        _normalize_obs_checkpoints((0, -5))


def test_normalize_obs_checkpoints_empty_input_raises() -> None:
    with pytest.raises(ValueError):
        _normalize_obs_checkpoints(())


# ---------------------------------------------------------------------------
# InferenceConfig.__post_init__ validation
# ---------------------------------------------------------------------------


def _minimal_config(**kwargs) -> InferenceConfig:
    defaults = dict(
        time_window=parse_time_window("August 2024"),
        checkpoint_path="/tmp/model.pt",
        inputs_bucket="s3://inputs",
        output_bucket="s3://outputs",
        num_gpus=0,
    )
    defaults.update(kwargs)
    return InferenceConfig(**defaults)


def test_inference_config_default_norm_source_is_aws() -> None:
    cfg = _minimal_config()
    assert cfg.norm_source == "aws"


def test_inference_config_invalid_norm_source_raises() -> None:
    with pytest.raises(ValueError, match="Invalid norm_source"):
        _minimal_config(norm_source="xyz")


def test_inference_config_invalid_s1_orbit_raises() -> None:
    with pytest.raises(ValueError, match="Invalid s1_orbit"):
        _minimal_config(s1_orbit="diagonal")


def test_inference_config_valid_s1_orbits() -> None:
    for orbit in ("ascending", "descending", "both"):
        cfg = _minimal_config(s1_orbit=orbit)
        assert cfg.s1_orbit == orbit


def test_inference_config_num_obs_checkpoints_coerced() -> None:
    cfg = _minimal_config(num_obs_checkpoints=(16, 8, 8, 0))
    assert cfg.num_obs_checkpoints == (8, 16)


def test_inference_config_empty_num_obs_checkpoints_raises() -> None:
    with pytest.raises(ValueError):
        _minimal_config(num_obs_checkpoints=())


def test_inference_config_default_model_version_is_v11() -> None:
    cfg = _minimal_config()
    assert cfg.model_version == "v1.1"


def test_inference_config_invalid_model_version_raises() -> None:
    with pytest.raises(ValueError, match="Invalid model_version"):
        _minimal_config(model_version="v3")
