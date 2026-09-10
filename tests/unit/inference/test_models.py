"""Tests for v1.1 model construction and forward pass shapes."""

from __future__ import annotations

import pytest
import torch

from tessera_embeddings.inference.models.builder import _build_inference_model
from tessera_embeddings.inference.models.modules import (
    CustomGRU,
    CustomGRUCell,
    TemporalAwarePooling,
    TemporalPositionalEncoder,
    TransformerEncoder,
)
from tessera_embeddings.inference.models.ssl_model import (
    MultimodalBTInferenceModel,
    build_dim_reducer,
)


class TestTransformerEncoder:
    """Tests for TransformerEncoder forward pass shapes."""

    def test_s2_output_shape(self):
        """S2 backbone pools to (B, latent_dim * 4)."""
        enc = TransformerEncoder(band_num=10, latent_dim=32, nhead=4, num_encoder_layers=2)
        x = torch.randn(4, 15, 11)
        assert enc(x).shape == (4, 128)

    def test_s1_output_shape(self):
        """S1 backbone pools to (B, latent_dim * 4)."""
        enc = TransformerEncoder(band_num=2, latent_dim=32, nhead=4, num_encoder_layers=2)
        x = torch.randn(4, 15, 3)
        assert enc(x).shape == (4, 128)


class TestTemporalPositionalEncoder:
    """Tests for positional encoding output shape."""

    def test_output_shape(self):
        """DOY values of shape (B, T) map to (B, T, d_model)."""
        pe = TemporalPositionalEncoder(d_model=64)
        doy = torch.randint(1, 366, (4, 20))
        assert pe(doy).shape == (4, 20, 64)


class TestTemporalAwarePooling:
    """Tests for temporal-aware pooling."""

    def test_output_shape(self):
        """Pooling collapses the temporal axis: (B, T, dim) -> (B, dim)."""
        pool = TemporalAwarePooling(64)
        x = torch.randn(4, 10, 64)
        assert pool(x).shape == (4, 64)


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

    def test_custom_gru_to_nn_gru_fusion_approx(self):
        """Fusing CustomGRU into nn.GRU with negated z gate produces close output.

        Reset gate placement differs between tessera (``W_hh(r*h)``) and nn.GRU
        (``r*(W_hh(h)+b)``); when r ~ 1 (forced here via large ``b_r``), the
        approximation is tight.
        """
        input_size, hidden_size, seq_len, batch = 64, 32, 15, 4
        custom = CustomGRU(input_size, hidden_size)
        cell = custom.gru_cell
        with torch.no_grad():
            cell.b_r.fill_(5.0)

        fused = torch.nn.GRU(input_size, hidden_size, batch_first=True)
        zeros_h = torch.zeros(hidden_size)
        with torch.no_grad():
            fused.weight_ih_l0.copy_(torch.cat([cell.W_ir.weight, -cell.W_iz.weight, cell.W_ih.weight]))
            fused.weight_hh_l0.copy_(torch.cat([cell.W_hr.weight, -cell.W_hz.weight, cell.W_hh.weight]))
            fused.bias_ih_l0.copy_(torch.cat([cell.b_r, -cell.b_z, cell.b_h]))
            fused.bias_hh_l0.copy_(torch.cat([zeros_h, zeros_h, zeros_h]))

        torch.manual_seed(0)
        x = torch.randn(batch, seq_len, input_size)
        out_custom, _ = custom(x)
        out_fused, _ = fused(x)
        torch.testing.assert_close(out_custom, out_fused, atol=1e-2, rtol=1e-2)


class TestMultimodalBTInferenceModel:
    """v1.1 inference model: two backbones + MLP dim_reducer, output (B, repr_dim)."""

    def test_forward_shape_concat(self):
        """Concat fusion forward pass returns (B, representation_dim)."""
        latent_dim, repr_dim = 32, 16
        s2_enc = TransformerEncoder(band_num=10, latent_dim=latent_dim, nhead=4, num_encoder_layers=2)
        s1_enc = TransformerEncoder(band_num=2, latent_dim=latent_dim, nhead=4, num_encoder_layers=2)
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
        s2_enc = TransformerEncoder(band_num=10, latent_dim=32, nhead=4, num_encoder_layers=2)
        s1_enc = TransformerEncoder(band_num=2, latent_dim=32, nhead=4, num_encoder_layers=2)
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
    """Tests for load_checkpoint: state-key fallback, prefix stripping, head dropping."""

    def _save(self, tmp_path, payload):
        """Write *payload* to a .pt file and return its path."""
        path = tmp_path / "ckpt.pt"
        torch.save(payload, path)
        return str(path)

    def test_strips_orig_mod_prefix(self, tmp_path):
        """`_orig_mod.` (torch.compile) prefixes are removed from param keys."""
        from tessera_embeddings.inference.models.builder import load_checkpoint

        path = self._save(
            tmp_path,
            {
                "model_state": {
                    "_orig_mod.s2_backbone.weight": torch.ones(2),
                    "s1_backbone.bias": torch.zeros(3),
                }
            },
        )
        cleaned = load_checkpoint(path, torch.device("cpu"))
        assert "s2_backbone.weight" in cleaned
        assert "_orig_mod.s2_backbone.weight" not in cleaned
        assert "s1_backbone.bias" in cleaned

    def test_drops_training_only_heads(self, tmp_path):
        """projector.* and segmented_matryoshka_projector.* params are dropped."""
        from tessera_embeddings.inference.models.builder import load_checkpoint

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
        cleaned = load_checkpoint(path, torch.device("cpu"))
        assert set(cleaned) == {"s2_backbone.weight"}

    def test_falls_back_to_model_state_dict_key(self, tmp_path):
        """When 'model_state' is absent, 'model_state_dict' is used."""
        from tessera_embeddings.inference.models.builder import load_checkpoint

        path = self._save(
            tmp_path,
            {
                "model_state_dict": {"s2_backbone.weight": torch.ones(2)},
            },
        )
        cleaned = load_checkpoint(path, torch.device("cpu"))
        assert "s2_backbone.weight" in cleaned

    def test_prefers_model_state_over_model_state_dict(self, tmp_path):
        """When both keys exist, 'model_state' wins."""
        from tessera_embeddings.inference.models.builder import load_checkpoint

        path = self._save(
            tmp_path,
            {
                "model_state": {"from_model_state": torch.ones(1)},
                "model_state_dict": {"from_model_state_dict": torch.ones(1)},
            },
        )
        cleaned = load_checkpoint(path, torch.device("cpu"))
        assert "from_model_state" in cleaned
        assert "from_model_state_dict" not in cleaned

    def test_raises_when_no_state_key(self, tmp_path):
        """A checkpoint with neither state key raises KeyError listing available keys."""
        from tessera_embeddings.inference.models.builder import load_checkpoint

        path = self._save(tmp_path, {"optimizer": {}, "epoch": 5})
        with pytest.raises(KeyError, match="model_state_dict"):
            load_checkpoint(path, torch.device("cpu"))

    def test_preserves_tensor_values(self, tmp_path):
        """Param tensors survive cleaning unmodified."""
        from tessera_embeddings.inference.models.builder import load_checkpoint

        weight = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        path = self._save(tmp_path, {"model_state": {"_orig_mod.layer.weight": weight}})
        cleaned = load_checkpoint(path, torch.device("cpu"))
        torch.testing.assert_close(cleaned["layer.weight"], weight)


class TestPositionalEncoderBitIdentity:
    """empty + strided sin/cos fill is bit-identical to the historical zeros fill."""

    @staticmethod
    def _reference_forward(enc: TemporalPositionalEncoder, doy: torch.Tensor) -> torch.Tensor:
        position = doy.unsqueeze(-1).float()
        pe = torch.zeros(doy.shape[0], doy.shape[1], enc.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * enc.div_term.float())
        pe[:, :, 1::2] = torch.cos(position * enc.div_term.float())
        return pe.to(doy.dtype)

    def test_bit_identical_float32(self):
        enc = TemporalPositionalEncoder(d_model=32)
        doy = torch.randint(1, 366, (4, 20)).float()
        torch.testing.assert_close(enc(doy), self._reference_forward(enc, doy), atol=0.0, rtol=0.0)

    def test_bit_identical_bfloat16_cast_path(self):
        enc = TemporalPositionalEncoder(d_model=64).bfloat16()
        doy = torch.randint(1, 366, (2, 9)).bfloat16()
        torch.testing.assert_close(enc(doy), self._reference_forward(enc, doy), atol=0.0, rtol=0.0)
