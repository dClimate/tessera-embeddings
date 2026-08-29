"""Unit tests for config/inference.py validation and helpers."""

from __future__ import annotations

from dataclasses import fields

import pytest

from tessera_embeddings.config.inference import (
    MIN_GPU_BATCH_SIZE,
    TUNED_GPU_GIB,
    InferenceConfig,
    _normalize_obs_checkpoints,
    batch_size_for_gpu,
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
    where their field landed. It named allow_s2_only until 2026-08-13, when optical_min_obs was
    appended after it, and actor_request_headroom appended after that.
    """
    assert fields(InferenceConfig)[-1].name == "actor_request_headroom"


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


class TestActorRequestPolicy:
    """The combined meaning of the two actor-request knobs, decided in one place.

    `actor_request_batch_size` carried a sentinel (0 = all at once) before
    `actor_request_headroom` arrived with another (None = off). Nothing reconciled them,
    so contradictory pairs were accepted and then half-honoured somewhere downstream.
    These pin what each combination means, at the only layer that can enforce it.
    """

    def test_a_headroom_at_or_below_zero_is_refused(self) -> None:
        """It would clamp the opening request to nothing and fail on an empty pool.

        Zero is the dangerous one: it reads as "disable the bound" and does the
        opposite, so it has to be refused rather than interpreted.
        """
        for bad in (0, -1):
            with pytest.raises(ValueError, match="bounds the fleet at or below nothing"):
                _minimal_config(num_gpus=1.0, actor_request_headroom=bad)
        # The neighbouring legal value, so the refusal is a line and not a blanket.
        assert _minimal_config(num_gpus=1.0, actor_request_headroom=1).actor_request_headroom == 1

    def test_a_headroom_with_batching_disabled_is_refused(self) -> None:
        """Two opposite instructions, so neither is silently chosen for the caller.

        A batch size of 0 asks for the whole fleet at once; a headroom asks never to
        exceed what has been placed. Honouring one and dropping the other is how a run
        came to open at the reduced width and stay there.
        """
        with pytest.raises(ValueError, match="two opposite things"):
            _minimal_config(num_gpus=1.0, actor_request_headroom=25, actor_request_batch_size=0)
        # Each alone is still legal — it is only the pair that has no meaning.
        assert _minimal_config(actor_request_batch_size=0).actor_request_batch_size == 0
        assert _minimal_config(num_gpus=1.0, actor_request_headroom=25, actor_request_batch_size=50)

    def test_a_headroom_without_a_gpu_reservation_is_refused(self) -> None:
        """The bound counts actor slots derived from GPUs, and a CPU run places none.

        Refused loudly rather than supported: the bound exists to protect an EC2 launch
        quota that a CPU-only run does not draw on, and the alternative to a clear
        refusal here is a fleet that silently never grows.
        """
        with pytest.raises(ValueError, match="needs a GPU reservation"):
            _minimal_config(num_gpus=0, actor_request_headroom=25)
        # A CPU run that asks for no bound is untouched, which is what keeps this a
        # refusal of one combination rather than of CPU runs.
        assert _minimal_config(num_gpus=0).actor_request_headroom is None
        assert _minimal_config(num_gpus=1.0, actor_request_headroom=25).actor_request_headroom == 25

    def test_the_opening_request_is_decided_in_one_place(self) -> None:
        """Both the runner and the scheduler read this, so they cannot disagree again."""
        # Batching disabled: the whole fleet up front.
        assert _minimal_config(actor_request_batch_size=0).initial_actor_request(250) == 250
        # Batching on, no bound: one batch.
        assert _minimal_config(actor_request_batch_size=50).initial_actor_request(250) == 50
        # Bounded: the cold start is the whole allowance, since nothing is placed yet.
        bounded = _minimal_config(num_gpus=1.0, actor_request_batch_size=50, actor_request_headroom=25)
        assert bounded.initial_actor_request(250) == 25
        # A target smaller than either still wins, so a tiny run does not over-ask.
        assert bounded.initial_actor_request(10) == 10


# ---------------------------------------------------------------------------
# batch_size_for_gpu — the card decides how much of a batch fits
# ---------------------------------------------------------------------------


class TestBatchSizeForGpu:
    """The tuned batch is an L40S number; a smaller card has to be given a smaller one."""

    @pytest.mark.parametrize(
        ("total_gib", "expected"),
        [
            # The card it was tuned on, and anything larger, is left alone.
            (44.7, 7168),  # L40S in g6e.xlarge
            (TUNED_GPU_GIB, 7168),  # exactly the reference — still unscaled
            (79.0, 7168),  # a bigger card is not given a bigger batch
            # The A10G is the case this exists for: 22.06 GiB, and 7168 does not fit.
            (22.06, 3593),  # g5.2xlarge — int(7168 * 22.06 / 44.0)
            (24.0, 3909),
        ],
    )
    def test_scales_only_below_the_tuned_card(self, total_gib: float, expected: int) -> None:
        assert batch_size_for_gpu(7168, total_gib) == expected

    def test_unknown_memory_leaves_the_tuned_value_alone(self) -> None:
        """A CPU device reports nothing; scaling on a guess is worse than the tuned default."""
        assert batch_size_for_gpu(7168, None) == 7168

    def test_a_tiny_card_stops_at_the_floor(self) -> None:
        """Below the floor the per-forward overhead dominates and the card is starved."""
        assert batch_size_for_gpu(7168, 0.5) == MIN_GPU_BATCH_SIZE

    def test_the_a10g_batch_leaves_real_headroom(self) -> None:
        """The observed failure was a 2.5 GiB allocation with 1.2 GiB free at 20.9 GiB in use.

        Activations dominate, so the scaled batch has to bring that 23.4 GiB requirement
        under the card's 22.06 — this pins the margin rather than just the arithmetic.
        """
        fitted = batch_size_for_gpu(7168, 22.06)
        projected_gib = 23.4 * fitted / 7168
        assert projected_gib < 22.06 * 0.75

    def test_a_configured_batch_below_the_floor_is_not_raised(self) -> None:
        """The floor bounds SCALING, not the caller: a small batch was asked for on purpose."""
        assert batch_size_for_gpu(64, 22.06) == 64
