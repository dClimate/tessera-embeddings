"""Tests for the v2 Large student port: construction, shapes, final LayerNorm.

These run everywhere — they build randomly-initialised models at the real v2
Large dimensions but never touch the 175 MB checkpoint. The bit-level agreement
with upstream on the real weights is pinned in ``test_student_v2_golden.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tessera_embeddings.config.inference import MODEL_ARCHS, InferenceConfig, band_stats
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.models.builder import _build_v2_inference_model, _verify_v2_args
from tessera_embeddings.inference.models.student_v2 import (
    AttentionPooling,
    QKNormEncoderLayer,
    StudentTransformerEncoder,
    build_v2_dim_reducer,
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


def test_verify_v2_args_rejects_buckets_over_max_seq_len() -> None:
    cfg = v2_config(num_obs_checkpoints=(8, 512))
    with pytest.raises(ValueError, match="max_seq_len"):
        _verify_v2_args(cfg, V2_LARGE_ARGS)


# ── blocks ──


def test_attention_pooling_shape() -> None:
    pool = AttentionPooling(64)
    assert pool(torch.randn(4, 10, 64)).shape == (4, 64)


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


def test_v2_backbone_has_no_recurrence() -> None:
    """v2 pools with plain attention — no GRU for the builder to fuse."""
    enc = StudentTransformerEncoder(band_num=10, latent_dim=16, nhead=4, num_encoder_layers=1)
    assert isinstance(enc.attn_pool, AttentionPooling)
    assert not any(isinstance(m, torch.nn.GRU | torch.nn.GRUCell) for m in enc.modules())


@pytest.mark.parametrize(("band_num", "channels"), [(10, 11), (2, 3)])
def test_v2_backbone_output_shape(band_num: int, channels: int) -> None:
    enc = StudentTransformerEncoder(band_num=band_num, latent_dim=16, nhead=4, num_encoder_layers=2)
    assert enc(torch.randn(4, 12, channels)).shape == (4, 64)


def test_qk_norm_layer_preserves_shape() -> None:
    """The QK-norm variant (unused by the Large checkpoint) still runs."""
    layer = QKNormEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.0).eval()
    x = torch.randn(2, 5, 32)
    assert layer(x).shape == x.shape


def test_qk_norm_layer_rejects_indivisible_head_dim() -> None:
    with pytest.raises(ValueError, match="nhead"):
        QKNormEncoderLayer(d_model=30, nhead=4, dim_feedforward=64)


def test_v2_dim_reducer_shapes_and_keys() -> None:
    """The trailing non-affine LayerNorm adds no parameters."""
    reducer = build_v2_dim_reducer(latent_dim=160, repr_dim=128)
    assert reducer(torch.randn(3, 160 * 4 * 2)).shape == (3, 128)
    assert set(reducer.state_dict()) == {
        "0.weight",
        "0.bias",
        "1.weight",
        "1.bias",
        "4.weight",
        "4.bias",
    }


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


def test_v2_model_output_shape_is_128(v2_model) -> None:
    with torch.no_grad():
        out = v2_model(torch.randn(5, 16, 11), torch.randn(5, 24, 3))
    assert out.shape == (5, 128)


def test_v2_output_is_layer_normalised_per_row(v2_model) -> None:
    """The final non-affine LayerNorm puts every row at mean 0 / std 1."""
    with torch.no_grad():
        out = v2_model(torch.randn(8, 16, 11), torch.randn(8, 8, 3))
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(8), atol=1e-5, rtol=0)
    # LayerNorm normalises by the biased (population) std.
    torch.testing.assert_close(out.std(dim=-1, unbiased=False), torch.ones(8), atol=1e-4, rtol=0)


def test_quantization_is_version_agnostic(v2_model) -> None:
    """Per-pixel abs-max int8 + fp32 scale needs no v2 changes.

    v2 rows are LayerNorm-ed (|x| ~ 1) rather than v1.1's unnormalised
    activations, but the scheme derives every scale from the row itself, so the
    round-trip error stays bounded by ``scale / 2`` per channel.
    """
    from tessera_embeddings.inference.quantization import quantize_rows, quantize_rows_torch

    with torch.no_grad():
        rows = v2_model(torch.randn(16, 16, 11), torch.randn(16, 16, 3))

    quantized, scales = quantize_rows(rows.numpy())
    assert quantized.shape == (16, 128)
    assert quantized.dtype.name == "int8"
    assert scales.shape == (16,)
    reconstructed = quantized.astype("float32") * scales[:, None]
    assert abs(reconstructed - rows.numpy()).max() <= scales.max() / 2 + 1e-6

    # The on-device path must agree bit-for-bit with the CPU path (v1.1 contract).
    q_torch, s_torch = quantize_rows_torch(rows)
    torch.testing.assert_close(q_torch, torch.from_numpy(quantized), atol=0, rtol=0)
    torch.testing.assert_close(s_torch, torch.from_numpy(scales), atol=0, rtol=0)


def test_v2_runs_the_unmodified_inference_loop(v2_model, sample_chunk_data) -> None:
    """The bucketing/sampling/quantize path needs no v2 changes: (H, W, 128) int8 out.

    ``save_dim = min(EMBEDDING_DIM, representation_dim)`` is 128 for v2, so the
    v1.1 slice is the identity and the store layout is unchanged.
    """
    from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
    from tessera_embeddings.inference.inference import run_inference

    config = v2_config(batch_size=8)
    chunk = sample_chunk_data(height=4, width=4, t_s2=10, t_s1a=5, t_s1d=5)
    dataset = MosaicChunkInferenceDataset(
        chunk,
        num_obs_checkpoints=config.num_obs_checkpoints,
        stats=band_stats(config.model_version, config.norm_source),
    )

    result = run_inference(v2_model, dataset, config, torch.device("cpu"))

    assert result.embeddings.shape == (4, 4, 128)
    assert result.embeddings.dtype.name == "int8"
    assert result.scales.shape == (4, 4)
    assert not any(np.isnan(result.scales.ravel()))


def test_v2_arch_spec_is_registered() -> None:
    assert MODEL_ARCHS["v2-large"].enable_qk_norm is False
    assert MODEL_ARCHS["v2-large"].representation_dim == 128
