"""Unit tests for config/inference.py validation and helpers."""

from __future__ import annotations

from dataclasses import fields

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


def test_inference_config_compute_std_forced_false() -> None:
    cfg = _minimal_config(compute_std=True)  # type: ignore[call-arg]
    assert cfg.compute_std is False


def test_the_newest_field_is_the_last_field() -> None:
    """InferenceConfig is public API (docs/public-api.md) and a positional
    dataclass. A new field inserted mid-list would silently rebind later
    positional args in downstream construction (e.g. a positional num_gpus would
    land in compute_std). New fields must be appended, and this pins the last one.

    Moving this assertion is the expected cost of adding a field, and it is worth paying: the
    tripwire fires on an insertion and on an append alike, so whoever moves it has to look at
    where their field landed. It named allow_s2_only until 2026-08-13, then optical_min_obs, and now
    model_version — which arrived on the v2-large branch INSERTED second, before this
    rule existed on that branch, and was moved to the end during the merge rather than
    left to rebind every positional argument after ``time_window``.
    """
    assert fields(InferenceConfig)[-1].name == "model_version"


def test_the_minimum_depth_rule_defaults_to_no_rule() -> None:
    """None, not zero, and not the module constant. A config that inherited a refusal line from
    whatever the module happened to hold would publish under a rule nobody chose, and zero would
    read as a configured rule that refuses nothing.
    """
    assert _minimal_config().optical_min_obs is None
    assert _minimal_config(optical_min_obs=25).optical_min_obs == 25


def test_a_rule_that_refuses_nothing_is_refused_by_the_config() -> None:
    with pytest.raises(ValueError, match="refuses nothing"):
        _minimal_config(optical_min_obs=0)


def test_allow_s2_only_defaults_off() -> None:
    assert _minimal_config().allow_s2_only is False
    assert _minimal_config(allow_s2_only=True).allow_s2_only is True


def test_an_explicitly_empty_norm_source_is_refused_not_defaulted() -> None:
    """`or "aws"` accepted every falsy value and silently selected AWS statistics.

    Pairing a checkpoint with the wrong band statistics produces embeddings that are wrong and
    perfectly well-formed — no shape error, no NaN, nothing downstream to object. Only the UNSET
    case may default; anything supplied goes through validation.
    """
    with pytest.raises(ValueError, match="Invalid norm_source"):
        _minimal_config(norm_source="")  # type: ignore[arg-type]
    assert _minimal_config(norm_source=None).norm_source == "aws", "unset still defaults"
    assert _minimal_config(norm_source="mpc").norm_source == "mpc"
