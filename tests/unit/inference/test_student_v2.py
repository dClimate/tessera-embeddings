"""Tests for the v2 Large student port: construction, shapes, final LayerNorm.

These run everywhere — they build randomly-initialised models at the real v2
Large dimensions but never touch the 175 MB checkpoint. The bit-level agreement
with upstream on the real weights is pinned in ``test_student_v2_golden.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tessera_embeddings.config.inference import InferenceConfig, band_stats
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
from tessera_embeddings.inference.inference import run_inference
from tessera_embeddings.inference.models.builder import (
    _build_v2_inference_model,
    _fit_v2_to_card,
    _verify_v2_args,
)
from tessera_embeddings.inference.models.ssl_model import GRAPH_CHUNK_PIXELS, run_in_chunks
from tessera_embeddings.inference.models.student_v2 import (
    E4M3_MAX,
    AttentionPooling,
    QKNormEncoderLayer,
    StudentTransformerEncoder,
    _add_layer_norm,
    _encoder_layer_forward,
    _scale_rows,
    build_fp8_layers,
    fp8_weight,
)

# The v2 Large checkpoint's stored ``args``.
V2_LARGE_ARGS = {
    "repr_dim": 128,
    "latent_dim": 160,
    "num_layers": 4,
    "nhead": 4,
    "dim_feedforward": 2560,
    "max_seq_len": 256,
    "matryoshka_dims": "16,32,64,128",
    "student_type": "transformer",
    "enable_qk_norm": False,
}


def v2_config(**overrides) -> InferenceConfig:
    """A v2-large config with tiny buckets so tests stay fast."""
    kwargs = dict(
        time_window=parse_time_window("June 2025"),
        model_version="v2-large",
        num_obs_checkpoints=(8, 16),
        dropout=0.0,
        checkpoint_path="dummy",
    )
    kwargs.update(overrides)
    return InferenceConfig(**kwargs)  # type: ignore[arg-type]


# ── config: version selection ──


def test_v2_config_adopts_v2_architecture() -> None:
    """model_version="v2-large" replaces the v1.1 architecture defaults."""
    cfg = v2_config()
    assert (cfg.latent_dim, cfg.representation_dim) == (160, 128)
    assert (cfg.nhead, cfg.num_encoder_layers, cfg.dim_feedforward) == (4, 4, 2560)


def test_v2_config_stored_args_match_the_spec() -> None:
    """Our v2 arch spec agrees with the checkpoint's stored ``args``."""
    cfg = v2_config()
    _verify_v2_args(cfg, V2_LARGE_ARGS)  # raises on mismatch


def test_v2_config_rejects_conflicting_arch_field() -> None:
    with pytest.raises(ValueError, match="conflicts with model_version"):
        v2_config(latent_dim=64)


def test_v2_config_rejects_norm_source() -> None:
    with pytest.raises(ValueError, match="does not apply to model_version"):
        v2_config(norm_source="aws")


def test_v11_config_still_resolves_norm_source() -> None:
    cfg = InferenceConfig(time_window=parse_time_window("June 2025"))
    assert (cfg.model_version, cfg.norm_source, cfg.latent_dim) == ("v1.1", "aws", 192)


def test_v2_band_stats_differ_from_v11() -> None:
    """v2 uses its own hard-coded stats, not either v1.1 set."""
    v2 = band_stats("v2-large")
    assert v2["s2_mean"][0] == pytest.approx(1633.0042)
    assert v2["s1_asc_mean"] != v2["s1_desc_mean"]  # per-orbit stats, still split
    for source in ("aws", "mpc"):
        assert band_stats("v1.1", source)["s2_mean"] != v2["s2_mean"]


def test_v2_band_stats_ignore_norm_source() -> None:
    assert band_stats("v2-large", "mpc") is band_stats("v2-large", None)


def test_verify_v2_args_rejects_mismatched_checkpoint() -> None:
    cfg = v2_config()
    with pytest.raises(ValueError, match="does not match config"):
        _verify_v2_args(cfg, {**V2_LARGE_ARGS, "latent_dim": 96})


def test_verify_v2_args_requires_nhead() -> None:
    """Require nhead: the one arch value not recoverable from parameter shapes.

    ``nn.MultiheadAttention`` stores ``in_proj_weight`` as (3*d_model, d_model)
    and ``out_proj`` as (d_model, d_model) — neither depends on the head count.
    A checkpoint omitting ``nhead`` would otherwise load under ``strict=True``
    with whatever head count the config happened to carry, and silently produce
    embeddings computed over the wrong head partition.
    """
    cfg = v2_config()
    args = {k: v for k, v in V2_LARGE_ARGS.items() if k != "nhead"}
    with pytest.raises(ValueError, match="nhead"):
        _verify_v2_args(cfg, args)


def test_verify_v2_args_still_tolerates_shape_recoverable_absences() -> None:
    """Keys a strict load would already catch stay optional — no new brittleness."""
    cfg = v2_config()
    for key in ("latent_dim", "repr_dim", "num_layers", "dim_feedforward"):
        _verify_v2_args(cfg, {k: v for k, v in V2_LARGE_ARGS.items() if k != key})


def test_verify_v2_args_rejects_buckets_over_max_seq_len() -> None:
    cfg = v2_config(num_obs_checkpoints=(8, 512))
    with pytest.raises(ValueError, match="max_seq_len"):
        _verify_v2_args(cfg, V2_LARGE_ARGS)


# ── blocks ──


def test_attention_pooling_single_timestep_passes_through() -> None:
    """T == 1 returns the lone timestep unchanged (no softmax over one element)."""
    pool = AttentionPooling(8)
    x = torch.randn(3, 1, 8)
    torch.testing.assert_close(pool(x), x.squeeze(1))


def test_attention_pooling_empty_sequence_is_zeros() -> None:
    pool = AttentionPooling(8)
    out = pool(torch.randn(3, 0, 8))
    assert out.shape == (3, 8)
    assert torch.all(out == 0)


def test_attention_pooling_is_a_convex_combination() -> None:
    """Softmax weights sum to 1, so the pool lies inside the sequence's range."""
    pool = AttentionPooling(6)
    x = torch.randn(2, 7, 6)
    out = pool(x)
    assert torch.all(out <= x.amax(dim=1) + 1e-5)
    assert torch.all(out >= x.amin(dim=1) - 1e-5)


@pytest.mark.parametrize("t", [1, 13], ids=["radar-free-step", "sequence"])
def test_hand_written_encoder_layer_computes_the_module(t: int) -> None:
    """The CUDA path's layer, given the plain add + LayerNorm, is the backbone's own layers' forward."""
    torch.manual_seed(0)
    enc = StudentTransformerEncoder(
        band_num=10, latent_dim=16, nhead=4, num_encoder_layers=2, dim_feedforward=96
    ).eval()
    x = torch.randn(3, t, 64)
    with torch.no_grad():
        for layer in enc.transformer_encoder.layers:
            torch.testing.assert_close(
                _encoder_layer_forward(layer, x, _add_layer_norm), layer(x), atol=1e-5, rtol=1e-5
            )
            x = layer(x)


def test_qk_norm_layer_preserves_shape() -> None:
    """The QK-norm variant (unused by the Large checkpoint) still runs."""
    layer = QKNormEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0).eval()
    x = torch.randn(2, 5, 32)
    assert layer(x).shape == x.shape


def test_qk_norm_layer_rejects_indivisible_head_dim() -> None:
    with pytest.raises(ValueError, match="nhead"):
        QKNormEncoderLayer(d_model=30, nhead=4, dim_feedforward=64)


# ── assembled model ──


@pytest.fixture(scope="module")
def v2_model():
    """Randomly-initialised v2 Large model at production dimensions, eval mode."""
    model = _build_v2_inference_model(v2_config(), torch.device("cpu"))
    model.eval()
    return model


def test_v2_model_parameter_count(v2_model) -> None:
    """The Large student is 43.83M parameters."""
    assert sum(p.numel() for p in v2_model.parameters()) == 43_831_170


def test_v2_model_state_dict_keys_match_checkpoint_layout(v2_model) -> None:
    """Backbone/reducer key names match the published payload exactly."""
    keys = set(v2_model.state_dict())
    assert "s2_backbone.attn_pool.query.weight" in keys
    assert "s1_backbone.attn_pool.query.weight" in keys
    assert {"dim_reducer.0.weight", "dim_reducer.1.weight", "dim_reducer.4.weight"} <= keys
    # No GRU/LayerNorm pooling params (v1.1's TemporalAwarePooling) anywhere.
    assert not [k for k in keys if "temporal_context" in k or "attn_pool.layer_norm" in k]
    assert not [k for k in keys if k.startswith(("projector.", "segmented_matryoshka_projector."))]


def test_v2_output_is_layer_normalised_per_row(v2_model) -> None:
    """The final non-affine LayerNorm puts every row at mean 0 / std 1."""
    with torch.no_grad():
        out = v2_model(torch.randn(8, 16, 11), torch.randn(8, 8, 3))
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(8), atol=1e-5, rtol=0)
    # LayerNorm normalises by the biased (population) std.
    torch.testing.assert_close(out.std(dim=-1, unbiased=False), torch.ones(8), atol=1e-4, rtol=0)


@pytest.mark.parametrize("t_s1", [5, 0], ids=["with-radar", "radar-free"])
def test_v2_runs_the_unmodified_inference_loop(v2_model, sample_chunk_data, t_s1: int) -> None:
    """The bucketing/sampling/quantize path needs no v2 changes: (H, W, 128) int8 out.

    ``save_dim = min(EMBEDDING_DIM, representation_dim)`` is 128 for v2, so the
    v1.1 slice is the identity and the store layout is unchanged. The radar-free chunk
    puts every pixel in v2's one-step S1 bucket, a length no other bucket has.
    """
    config = v2_config(batch_size=8)
    chunk = sample_chunk_data(height=4, width=4, t_s2=10, t_s1a=t_s1, t_s1d=t_s1)
    dataset = MosaicChunkInferenceDataset(
        chunk,
        num_obs_checkpoints=config.num_obs_checkpoints,
        # Both halves from the config. This call previously passed v2 STATISTICS while leaving
        # the version at its v1.1 default — so it normalised as v2 and resampled as v1.1, which
        # is exactly the mismatch that made these two separate arguments a defect. The test was
        # demonstrating the bug it should have caught.
        model_version=config.model_version,
        norm_source=config.norm_source,
        allow_s2_only=True,
    )

    result = run_inference(v2_model, dataset, config, torch.device("cpu"))

    assert result.embeddings.shape == (4, 4, 128)
    assert result.embeddings.dtype.name == "int8"
    assert result.scales.shape == (4, 4)
    assert not any(np.isnan(result.scales.ravel()))


def test_chunked_forward_matches_the_whole_batch(v2_model) -> None:
    """Chunking, short last chunk included, changes no output; the chunk size comes from the card.

    On CUDA each chunk replays a captured graph; here the same loop runs the model eagerly on the
    static buffers, which is all a CPU can check.
    """
    chunk = GRAPH_CHUNK_PIXELS["NVIDIA L40S"]
    assert v2_model.chunk_pixels is None  # a CPU model runs the whole batch

    n = 2 * chunk + 5
    s2_x, s1_x = torch.randn(n, 16, 11), torch.randn(n, 8, 3)
    s2_static, s1_static = s2_x[:chunk].clone(), s1_x[:chunk].clone()
    with torch.no_grad():
        whole = v2_model(s2_x, s1_x)
        z_static = torch.empty(chunk, whole.shape[1])

        def run_chunk() -> None:
            z_static.copy_(v2_model(s2_static, s1_static))

        chunked = run_in_chunks(s2_x, s1_x, (s2_static, s1_static, z_static), run_chunk)
    torch.testing.assert_close(chunked, whole, atol=1e-5, rtol=1e-5)


# ── FP8 (ADR 043) ──
# F.scaled_mm runs row-wise E4M3 on the CPU, so the layer's arithmetic is checked here; capture, replay and the
# card's handling of non-finite rows are checked on an L40S (context_docs/inference/fp8-completion-plan.md §3.1).


@pytest.fixture(scope="module")
def fp8_encoder() -> tuple[StudentTransformerEncoder, list]:
    """A small random encoder at v2's width, and its FP8 layers."""
    torch.manual_seed(0)
    enc = StudentTransformerEncoder(band_num=10, latent_dim=160, nhead=4, num_encoder_layers=3, dim_feedforward=256)
    return enc.eval(), build_fp8_layers(enc)


def test_fp8_layers_quantise_two_linears_and_leave_the_first_qkv(fp8_encoder) -> None:
    """QKV of every layer after the first and every first FFN projection; each scale maps its channel's max to 448."""
    enc, layers = fp8_encoder
    assert [layer.qkv is None for layer in layers] == [True, False, False]
    w = enc.transformer_encoder.layers[1].linear1.weight
    q = layers[1].ffn_up
    assert q.weight.dtype == torch.float8_e4m3fn and q.weight.shape == w.t().shape
    torch.testing.assert_close(q.scale[0], w.abs().amax(dim=1) / E4M3_MAX)
    # E4M3 keeps 3 mantissa bits (half a step is 1/16 relative); its subnormals step by 2**-9 of the scale.
    assert ((q.weight.float() * q.scale - w.t()).abs() <= w.t().abs() / 16 + q.scale * 2**-10).all()


def test_fp8_weights_survive_the_bf16_cast_and_stay_out_of_the_state_dict(fp8_encoder) -> None:
    enc, layers = fp8_encoder
    copy = StudentTransformerEncoder(band_num=10, latent_dim=160, nhead=4, num_encoder_layers=3, dim_feedforward=256)
    keys = set(copy.state_dict())
    copy.fp8_layers = layers
    copy.bfloat16()
    assert copy.fp8_layers[1].qkv.weight.dtype == torch.float8_e4m3fn
    assert copy.fp8_layers[1].qkv.scale.dtype == torch.float32
    assert set(copy.state_dict()) == keys


@pytest.mark.parametrize("t", [1, 9])
def test_fp8_layer_tracks_bf16_and_quantises_each_row_alone(fp8_encoder, t: int) -> None:
    """Close to today's layer (a transposed weight or a scale on the wrong axis is far off), and per row.

    A pixel's codes must not depend on its batch, so every scale is per token row. (The CPU's matrix
    multiplies themselves may round by batch size; the card's are checked for that on an L40S.)
    """
    enc, layers = fp8_encoder
    layer, fp8 = enc.transformer_encoder.layers[1], layers[1]
    x = torch.randn(5, t, 640)
    with torch.no_grad():
        ref = _encoder_layer_forward(layer, x, _add_layer_norm)
        out = _encoder_layer_forward(layer, x, _add_layer_norm, fp8)
    assert torch.nn.functional.cosine_similarity(out.flatten(1), ref.flatten(1)).min() > 0.999
    rows = x.reshape(-1, 640)
    codes, scales = _scale_rows(rows)
    alone_codes, alone_scales = _scale_rows(rows[3:4])
    assert torch.equal(alone_codes.view(torch.uint8), codes[3:4].view(torch.uint8))
    assert torch.equal(alone_scales, scales[3:4])


def test_fp8_layer_keeps_a_zero_pixel_finite_and_a_nan_pixel_to_itself(fp8_encoder) -> None:
    """An all-zero token row scales to the floor, not to a NaN; a NaN stays in its own pixel."""
    enc, layers = fp8_encoder
    layer, fp8 = enc.transformer_encoder.layers[1], layers[1]
    x = torch.randn(4, 6, 640)
    x[0] = 0.0
    x[2, 3, 7] = float("nan")
    with torch.no_grad():
        out = _encoder_layer_forward(layer, x, _add_layer_norm, fp8)
        clean = _encoder_layer_forward(layer, x[[0, 1, 3]], _add_layer_norm, fp8)
    assert torch.isfinite(out[[0, 1, 3]]).all()
    assert not torch.isfinite(out[2]).all()
    torch.testing.assert_close(out[[0, 1, 3]], clean, atol=1e-6, rtol=1e-6)


def test_fp8_weight_clamps_rather_than_overflowing() -> None:
    """A channel's own max lands on 448 exactly; nothing becomes a NaN code."""
    w = torch.tensor([[1.0, -3.0, 0.5], [0.0, 0.0, 0.0]])
    q = fp8_weight(w)
    assert torch.isfinite(q.weight.float()).all()
    assert q.weight.float()[1, 0] == -E4M3_MAX


@pytest.mark.parametrize(
    ("card", "precision", "chunk", "fp8"),
    [
        ("NVIDIA L40S", "fp8", 64, True),
        ("NVIDIA L40S", "bf16", 64, False),
        ("NVIDIA A10G", "fp8", None, False),
        ("NVIDIA L4", "fp8", 32, False),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "fp8", None, False),
    ],
)
def test_only_an_fp8_card_in_an_fp8_store_builds_fp8_layers(card: str, precision: str, chunk, fp8: bool) -> None:
    model = _build_v2_inference_model(v2_config(precision=precision), torch.device("cpu"))
    assert _fit_v2_to_card(model, model_config := v2_config(precision=precision), card) is fp8
    assert model.chunk_pixels == chunk
    for encoder in (model.s2_backbone, model.s1_backbone):
        assert (encoder.fp8_layers is not None) is fp8
    assert model_config.precision == precision


def test_precision_defaults_per_model_and_v11_refuses_fp8() -> None:
    assert v2_config().precision == "fp8"
    assert InferenceConfig(time_window=parse_time_window("June 2025"), checkpoint_path="x").precision == "bf16"
    with pytest.raises(ValueError, match="precision"):
        InferenceConfig(time_window=parse_time_window("June 2025"), checkpoint_path="x", precision="fp8")
