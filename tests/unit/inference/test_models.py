"""Tests for v1.1 model construction and forward pass shapes."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from tessera_embeddings.inference.models.builder import _build_inference_model
from tessera_embeddings.inference.models.modules import (
    CustomGRU,
    CustomGRUCell,
    TemporalAwarePooling,
    TemporalPositionalEncoder,
    V11TransformerEncoder,
)
from tessera_embeddings.inference.models.ssl_model import (
    MultimodalBTInferenceModel,
    build_dim_reducer,
)
from tessera_embeddings.inference.models.student_v2 import StudentTransformerEncoder
from tests._paths import FIXTURES


class TestTransformerEncoder:
    """Tests for V11TransformerEncoder forward pass shapes."""

    def test_s2_output_shape(self):
        """S2 backbone pools to (B, latent_dim * 4)."""
        enc = V11TransformerEncoder(band_num=10, latent_dim=32, nhead=4, num_encoder_layers=2)
        x = torch.randn(4, 15, 11)
        assert enc(x).shape == (4, 128)

    def test_s1_output_shape(self):
        """S1 backbone pools to (B, latent_dim * 4)."""
        enc = V11TransformerEncoder(band_num=2, latent_dim=32, nhead=4, num_encoder_layers=2)
        x = torch.randn(4, 15, 3)
        assert enc(x).shape == (4, 128)


def _upstream_encoder(file: str) -> type[torch.nn.Module]:
    """An upstream model file's own ``TemporalPositionalEncoder``, imported by path."""
    spec = importlib.util.spec_from_file_location(f"upstream_{Path(file).stem}", FIXTURES / "upstream" / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TemporalPositionalEncoder


class TestTemporalPositionalEncoder:
    """Tests for positional encoding output shape."""

    def test_output_shape(self):
        """DOY values of shape (B, T) map to (B, T, d_model)."""
        pe = TemporalPositionalEncoder(d_model=64)
        doy = torch.randint(1, 366, (4, 20))
        assert pe(doy).shape == (4, 20, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="exp() rounds the same either way on a CPU-only host")
    @pytest.mark.parametrize(
        ("upstream_file", "d_model", "on_cpu"),
        [("v1_1_reference/modules.py", 768, True), ("v2_student_reference.py", 640, False)],
    )
    def test_matches_its_upstream_on_gpu(self, upstream_file, d_model, on_cpu):
        """Each model's encoding is bit-identical to its own upstream's on the GPU."""
        doy = torch.arange(1, 366, device="cuda", dtype=torch.float32).repeat(3, 1)
        ours = TemporalPositionalEncoder(d_model, frequencies_on_cpu=on_cpu)(doy, out_dtype=torch.float32)
        theirs = _upstream_encoder(upstream_file)(d_model)(doy)
        torch.testing.assert_close(ours, theirs, atol=0.0, rtol=0.0)


def _reference_pool(pool: TemporalAwarePooling, x: torch.Tensor) -> torch.Tensor:
    """The pooling head as the model was trained: CustomGRU over the sequence, LayerNorm, attention."""
    context, _ = pool.temporal_context(x)
    weights = torch.softmax(pool.query(pool.layer_norm(context)), dim=1)
    return (weights * x).sum(dim=1)


def _trained_like_pool(dim: int) -> TemporalAwarePooling:
    """A pooling head with biases spread out, so its reset gate is far from 1, as the real checkpoint's is."""
    torch.manual_seed(0)
    pool = TemporalAwarePooling(dim).eval()
    with torch.no_grad():
        for b in (pool.temporal_context.gru_cell.b_r, pool.temporal_context.gru_cell.b_z, pool.query.bias):
            b.normal_(0.0, 1.0)
    return pool


class TestTemporalAwarePooling:
    """The fused pooling head must reproduce the trained, step-at-a-time head."""

    @pytest.mark.parametrize(("batch", "steps"), [(4, 1), (4, 10), (7, 40), (1, 33)])
    def test_matches_the_trained_formula(self, batch, steps):
        """Matches CustomGRU + LayerNorm + attention across the 32-step projection block and at odd batches."""
        pool = _trained_like_pool(64)
        x = torch.randn(batch, steps, 64)
        with torch.no_grad():
            torch.testing.assert_close(pool(x), _reference_pool(pool, x), atol=1e-5, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="the compiled step runs on CUDA only")
    @pytest.mark.parametrize("batch", [1, 7, 7167])
    def test_compiled_step_matches_the_trained_formula_on_gpu(self, batch):
        """The compiled BF16 path stays at rounding distance from the FP32 reference, odd batches included."""
        pool = _trained_like_pool(128).cuda()
        x = torch.randn(batch, 40, 128, device="cuda")
        with torch.no_grad():
            reference = _reference_pool(pool, x)
            fused = pool.to(torch.bfloat16)(x.to(torch.bfloat16)).float()
        cos = torch.nn.functional.cosine_similarity(fused, reference, dim=-1)
        assert cos.min() > 0.9999


class TestCustomGRU:
    """Tests for CustomGRUCell and CustomGRU."""

    def test_cell_output_shape(self):
        """A single cell step produces a (B, hidden_size) hidden state."""
        cell = CustomGRUCell(input_size=64, hidden_size=32)
        h_new = cell(torch.randn(4, 64), torch.randn(4, 32))
        assert h_new.shape == (4, 32)

    def test_gru_output_shape(self):
        """CustomGRU matches nn.GRU(batch_first=True) output contract."""
        gru = CustomGRU(input_size=64, hidden_size=32)
        outputs, h_t = gru(torch.randn(4, 10, 64))
        assert outputs.shape == (4, 10, 32)
        assert h_t.shape == (1, 4, 32)

    def test_tessera_gru_formulation(self):
        """CustomGRUCell matches tessera's variant: reset before matmul, z selects new."""
        cell = CustomGRUCell(8, 16)
        torch.manual_seed(42)
        x_t = torch.randn(4, 8)
        h_prev = torch.randn(4, 16)
        with torch.no_grad():
            r = torch.sigmoid(cell.W_ir(x_t) + cell.W_hr(h_prev) + cell.b_r)
            z = torch.sigmoid(cell.W_iz(x_t) + cell.W_hz(h_prev) + cell.b_z)
            n = torch.tanh(cell.W_ih(x_t) + cell.W_hh(r * h_prev) + cell.b_h)
            expected = (1 - z) * h_prev + z * n
        torch.testing.assert_close(cell(x_t, h_prev), expected, atol=1e-6, rtol=1e-6)


class TestMultimodalBTInferenceModel:
    """v1.1 inference model: two backbones + MLP dim_reducer, output (B, repr_dim)."""

    def test_forward_shape_concat(self):
        """Concat fusion forward pass returns (B, representation_dim)."""
        latent_dim, repr_dim = 32, 16
        s2_enc = V11TransformerEncoder(band_num=10, latent_dim=latent_dim, nhead=4, num_encoder_layers=2)
        s1_enc = V11TransformerEncoder(band_num=2, latent_dim=latent_dim, nhead=4, num_encoder_layers=2)
        reducer = build_dim_reducer(latent_dim=latent_dim, active_backbones=2, repr_dim=repr_dim)
        model = MultimodalBTInferenceModel(s2_enc, s1_enc, reducer, fusion_method="concat")
        model.eval()

        s2_x = torch.randn(4, 15, 11)
        s1_x = torch.randn(4, 15, 3)
        with torch.no_grad():
            out = model(s2_x, s1_x)
        assert out.shape == (4, repr_dim)

    def test_invalid_fusion_raises(self):
        """An unknown fusion method raises ValueError on forward."""
        s2_enc = V11TransformerEncoder(band_num=10, latent_dim=32, nhead=4, num_encoder_layers=2)
        s1_enc = V11TransformerEncoder(band_num=2, latent_dim=32, nhead=4, num_encoder_layers=2)
        reducer = build_dim_reducer(latent_dim=32, active_backbones=2, repr_dim=16)
        model = MultimodalBTInferenceModel(s2_enc, s1_enc, reducer, fusion_method="invalid")
        with pytest.raises(ValueError, match="Unknown fusion method"):
            model(torch.randn(1, 5, 11), torch.randn(1, 5, 3))


class TestBuildInferenceModel:
    """Smoke test that the builder wires together a forward-able v1.1 model."""

    def test_build_from_config(self, inference_config):
        """The builder produces a model whose forward runs on a tiny batch."""
        model = _build_inference_model(inference_config, torch.device("cpu"))
        assert hasattr(model, "s2_backbone")
        assert hasattr(model, "s1_backbone")
        assert hasattr(model, "dim_reducer")
        # Forward pass must not raise on a tiny synthetic batch sized to the
        # smallest bucket in the test config.
        s2_target = inference_config.num_obs_checkpoints[0]
        s1_target = inference_config.num_obs_checkpoints[0]
        s2_x = torch.randn(2, s2_target, 11)
        s1_x = torch.randn(2, s1_target, 3)
        with torch.no_grad():
            out = model(s2_x, s1_x)
        assert out.shape == (2, inference_config.representation_dim)


class TestLoadCheckpoint:
    """Tests for load_v11_checkpoint: state-key fallback, prefix stripping, head dropping."""

    def _save(self, tmp_path, payload):
        """Write *payload* to a .pt file and return its path."""
        path = tmp_path / "ckpt.pt"
        torch.save(payload, path)
        return str(path)

    def test_strips_orig_mod_prefix(self, tmp_path):
        """`_orig_mod.` (torch.compile) prefixes are removed from param keys."""
        from tessera_embeddings.inference.models.builder import load_v11_checkpoint

        path = self._save(
            tmp_path,
            {
                "model_state": {
                    "_orig_mod.s2_backbone.weight": torch.ones(2),
                    "s1_backbone.bias": torch.zeros(3),
                }
            },
        )
        cleaned = load_v11_checkpoint(path, torch.device("cpu"))
        assert "s2_backbone.weight" in cleaned
        assert "_orig_mod.s2_backbone.weight" not in cleaned
        assert "s1_backbone.bias" in cleaned

    def test_drops_training_only_heads(self, tmp_path):
        """projector.* and segmented_matryoshka_projector.* params are dropped."""
        from tessera_embeddings.inference.models.builder import load_v11_checkpoint

        path = self._save(
            tmp_path,
            {
                "model_state": {
                    "s2_backbone.weight": torch.ones(2),
                    "projector.0.weight": torch.ones(4),
                    "segmented_matryoshka_projector.fc.bias": torch.ones(8),
                }
            },
        )
        cleaned = load_v11_checkpoint(path, torch.device("cpu"))
        assert set(cleaned) == {"s2_backbone.weight"}

    def test_falls_back_to_model_state_dict_key(self, tmp_path):
        """When 'model_state' is absent, 'model_state_dict' is used."""
        from tessera_embeddings.inference.models.builder import load_v11_checkpoint

        path = self._save(
            tmp_path,
            {
                "model_state_dict": {"s2_backbone.weight": torch.ones(2)},
            },
        )
        cleaned = load_v11_checkpoint(path, torch.device("cpu"))
        assert "s2_backbone.weight" in cleaned

    def test_prefers_model_state_over_model_state_dict(self, tmp_path):
        """When both keys exist, 'model_state' wins."""
        from tessera_embeddings.inference.models.builder import load_v11_checkpoint

        path = self._save(
            tmp_path,
            {
                "model_state": {"from_model_state": torch.ones(1)},
                "model_state_dict": {"from_model_state_dict": torch.ones(1)},
            },
        )
        cleaned = load_v11_checkpoint(path, torch.device("cpu"))
        assert "from_model_state" in cleaned
        assert "from_model_state_dict" not in cleaned

    def test_raises_when_no_state_key(self, tmp_path):
        """A checkpoint with neither state key raises KeyError listing available keys."""
        from tessera_embeddings.inference.models.builder import load_v11_checkpoint

        path = self._save(tmp_path, {"optimizer": {}, "epoch": 5})
        with pytest.raises(KeyError, match="model_state_dict"):
            load_v11_checkpoint(path, torch.device("cpu"))

    def test_preserves_tensor_values(self, tmp_path):
        """Param tensors survive cleaning unmodified."""
        from tessera_embeddings.inference.models.builder import load_v11_checkpoint

        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        path = self._save(tmp_path, {"model_state": {"_orig_mod.layer.weight": weight}})
        cleaned = load_v11_checkpoint(path, torch.device("cpu"))
        torch.testing.assert_close(cleaned["layer.weight"], weight)


#: One FP32 unit in the last place for values in [-1, 1]. Multi-threaded CPU sin/cos rounds a few elements
#: (~0.005%) one ulp differently depending on how the tensor is split across threads, so the per-pixel
#: reference and the 367-row table can differ by this much on CPU. Single-threaded, after the cast to
#: BF16, and on the GPU (elementwise, no split) they are identical.
_ONE_FP32_ULP = 2.0**-24


class TestPositionalEncoderBitIdentity:
    """The day-of-year table reproduces the textbook per-pixel FP32 encoding."""

    @staticmethod
    def _reference_forward(d_model: int, doy: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        """The textbook FP32 encoding, written independently of the module."""
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        position = doy.float().unsqueeze(-1)
        pe = torch.zeros(doy.shape[0], doy.shape[1], d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe.to(out_dtype or doy.dtype)

    def test_float32_matches_within_one_ulp(self):
        enc = TemporalPositionalEncoder(d_model=32)
        doy = torch.randint(1, 366, (4, 20)).float()
        torch.testing.assert_close(enc(doy), self._reference_forward(32, doy), atol=_ONE_FP32_ULP, rtol=0.0)

    def test_bf16_module_still_computes_the_fp32_encoding(self):
        """``.bfloat16()`` must not reach the frequencies or the DOY arithmetic.

        The production path calls ``model.bfloat16()``. If ``div_term`` were a
        registered buffer it would be swept into BF16 and sin/cos arguments would
        shift by up to ~0.70 rad at DOY 365 — a silent divergence from the FP32
        graph the checkpoints were trained and golden-tested against. The only
        permitted loss is the final cast of the result.
        """
        enc = TemporalPositionalEncoder(d_model=64).bfloat16()
        doy = torch.randint(1, 366, (2, 9)).float()
        expected = self._reference_forward(64, doy, out_dtype=torch.bfloat16)
        actual = enc(doy, out_dtype=torch.bfloat16)
        assert actual.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    @pytest.mark.parametrize(("out_dtype", "atol"), [(torch.bfloat16, 0.0), (torch.float32, _ONE_FP32_ULP)])
    def test_every_day_of_year_matches_the_per_pixel_encoding(self, out_dtype, atol):
        """Every day the pipeline can produce, 0 through 366: exact in BF16, the production dtype."""
        enc = TemporalPositionalEncoder(d_model=768)
        doy = torch.arange(367, dtype=torch.float32).flip(0).repeat(3, 1)
        expected = self._reference_forward(768, doy, out_dtype=out_dtype)
        torch.testing.assert_close(enc(doy, out_dtype=out_dtype), expected, atol=atol, rtol=0.0)

    def test_doy_above_256_keeps_one_day_resolution(self):
        """DOY 257 and 258 must not collapse onto the same encoding.

        BF16 carries 8 mantissa bits, so it steps by 2 above 256 — days 257
        through 365 (mid-September onward) would otherwise round onto their even
        neighbour before the encoder ever sees them.
        """
        enc = TemporalPositionalEncoder(d_model=32)
        doy = torch.tensor([[257.0, 258.0]])
        out = enc(doy, out_dtype=torch.bfloat16)
        assert not torch.equal(out[0, 0], out[0, 1])
        torch.testing.assert_close(out, self._reference_forward(32, doy, out_dtype=torch.bfloat16), atol=0.0, rtol=0.0)


class TestEncoderComputeDtype:
    """Backbones cast bands to the weight dtype and leave DOY alone.

    The output dtype alone is NOT sufficient evidence: an encoder that cast the
    whole input to BF16 before splitting off DOY would also return BF16, having
    silently collapsed day 257 onto 256. So each test intercepts the tensor the
    temporal encoder actually receives and checks it is still exact FP32.
    """

    @staticmethod
    def _captured_doy(enc: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run *enc* on *x*, returning the DOY tensor its temporal encoder saw."""
        seen: list[torch.Tensor] = []
        handle = enc.temporal_encoder.register_forward_pre_hook(
            lambda _module, args: seen.append(args[0].detach().clone()) and None
        )
        try:
            with torch.no_grad():
                out = enc(x)
        finally:
            handle.remove()
        assert len(seen) == 1
        assert out.dtype == torch.bfloat16, "reduced-precision compute dtype must survive to the output"
        return seen[0]

    def _assert_doy_survives(self, enc: torch.nn.Module, band_num: int) -> None:
        # 257 and 258 straddle the BF16 integer-exactness boundary: BF16 steps by
        # 2 above 256, so a whole-tensor cast would deliver 256 and 258.
        x = torch.randn(1, 2, band_num + 1)
        x[0, :, -1] = torch.tensor([257.0, 258.0])
        doy = self._captured_doy(enc.bfloat16().eval(), x)
        assert doy.dtype == torch.float32
        torch.testing.assert_close(doy, torch.tensor([[257.0, 258.0]]), atol=0.0, rtol=0.0)

    def test_v11_backbone_hands_the_encoder_exact_fp32_doy(self):
        self._assert_doy_survives(V11TransformerEncoder(band_num=10, latent_dim=8, nhead=2, num_encoder_layers=1), 10)

    def test_v2_backbone_hands_the_encoder_exact_fp32_doy(self):
        self._assert_doy_survives(StudentTransformerEncoder(band_num=2, latent_dim=8, nhead=2, num_encoder_layers=1), 2)
