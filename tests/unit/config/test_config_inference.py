"""Unit tests for config/inference.py validation and helpers."""

from __future__ import annotations

import math
from dataclasses import fields

import pytest

from tessera_embeddings.config.inference import (
    DEFAULT_NUM_OBS_CHECKPOINTS,
    TUNED_BATCH_SIZE,
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

#: The A10G in ``g5.2xlarge``, the smallest rung `fleet_mix.GPU_RUNGS` offers.
A10G_TOTAL_GIB = 22.06

#: Deepest sub-batch an A10G was measured to complete, in tokens: 7,168 pixels at 325
#: tokens each, the last rung of the forward sweep in
#: ``context_docs/inference/inference-on-gpus.md``. The rung above it refuses.
A10G_MEASURED_TOKEN_CEILING = TUNED_BATCH_SIZE * 325

#: Most tokens one pixel can carry: the sampler clips each of the two streams to the
#: deepest checkpoint, so this is a property of the bucketing, not of any geography.
DEEPEST_TOKENS_PER_PIXEL = 2 * max(DEFAULT_NUM_OBS_CHECKPOINTS)


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
        assert batch_size_for_gpu(TUNED_BATCH_SIZE, total_gib) == expected

    def test_unknown_memory_leaves_the_tuned_value_alone(self) -> None:
        """A CPU device reports nothing; scaling on a guess is worse than the tuned default."""
        assert batch_size_for_gpu(TUNED_BATCH_SIZE, None) == TUNED_BATCH_SIZE

    def test_a_tiny_card_gets_what_fits_rather_than_a_round_number(self) -> None:
        """No floor: a batch rounded UP to look reasonable is the failure being prevented."""
        assert batch_size_for_gpu(7168, 0.5) == int(TUNED_BATCH_SIZE * 0.5 / TUNED_GPU_GIB)

    def test_a_request_above_the_calibration_is_capped_not_scaled(self) -> None:
        """The fit comes off the calibrated batch, so an over-ask cannot ride through it."""
        assert batch_size_for_gpu(10_000, A10G_TOTAL_GIB) == batch_size_for_gpu(TUNED_BATCH_SIZE, A10G_TOTAL_GIB)

    @pytest.mark.parametrize("gpu_fraction", [0.6, 0.4])
    def test_the_share_is_what_ray_packs_not_what_was_reserved(self, gpu_fraction: float) -> None:
        """Ray fits ``floor(1 / num_gpus)`` actors on a card, so 0.6 owns all of it and 0.4 half.

        Reading the reservation as the share would cut a batch that fits, for no memory
        gained: at 0.6 no second actor can be placed beside this one.
        """
        packed = math.floor(1 / gpu_fraction)
        assert batch_size_for_gpu(TUNED_BATCH_SIZE, 44.7, gpu_fraction=gpu_fraction) == batch_size_for_gpu(
            TUNED_BATCH_SIZE, 44.7 / packed
        )

    @pytest.mark.parametrize(
        ("num_obs_checkpoints", "gpu_fraction"),
        [
            (DEFAULT_NUM_OBS_CHECKPOINTS, 1.0),  # production: one actor, the tuned ladder
            ((512,), 1.0),  # a ladder deeper than the one the batch was tuned against
            ((4096,), 1.0),  # a ladder deep enough that the safe batch is a few hundred
            (DEFAULT_NUM_OBS_CHECKPOINTS, 0.5),  # a fractional reservation packs two per card
            (DEFAULT_NUM_OBS_CHECKPOINTS, 0.1),  # ten actors to a card
            ((512,), 0.5),  # both at once, which must compound rather than pick one
        ],
    )
    def test_the_fitted_batch_holds_the_deepest_bucket_the_sampler_can_build(
        self, num_obs_checkpoints: tuple[int, ...], gpu_fraction: float
    ) -> None:
        """The batch must fit the DEEPEST sub-batch, not the one that happened to fail.

        A sub-batch's working set is linear in ``batch x (t_s2 + t_s1)``, and the sampler
        clips both sequences to ``max(num_obs_checkpoints)`` — so the worst case is fixed
        before a tile is read rather than by which geography a fill draws. That is what
        makes a batch fitted once safe for every bucket, and it holds only if the fit sees
        everything that moves it: the card, the ladder's depth, and how many actors share
        the card. Each row here is a way to make the deepest sub-batch bigger; the demand
        on the card must not move.
        """
        fitted = batch_size_for_gpu(
            TUNED_BATCH_SIZE,
            A10G_TOTAL_GIB,
            num_obs_checkpoints=num_obs_checkpoints,
            gpu_fraction=gpu_fraction,
        )
        # What the CARD is asked for, not what one actor is: a fractional reservation puts
        # that many concurrent forwards on the same memory.
        per_card = fitted * 2 * max(num_obs_checkpoints) * math.floor(1 / gpu_fraction)
        assert per_card <= A10G_MEASURED_TOKEN_CEILING

    def test_the_unfitted_batch_does_not_hold_it(self) -> None:
        """The bound above is only evidence if the batch it replaced fails the same test."""
        assert TUNED_BATCH_SIZE * DEEPEST_TOKENS_PER_PIXEL > A10G_MEASURED_TOKEN_CEILING

    def test_a_small_configured_batch_is_never_raised(self) -> None:
        """The request is a ceiling: a small batch (a tiny test model) was asked for on purpose."""
        assert batch_size_for_gpu(64, A10G_TOTAL_GIB) == 64
