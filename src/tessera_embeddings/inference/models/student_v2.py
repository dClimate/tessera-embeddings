"""Tessera v2 student encoder blocks (128-d Matryoshka pixel encoder).

Architecture (same two-backbone topology as v1.1, different pooling head and a
final normalisation)::

    S2 stream ──► StudentTransformerEncoder (latent_dim=160) ─┐
                                                              ├─ concat ─► dim_reducer ──► (B, 128)
    S1 merged ──► StudentTransformerEncoder (latent_dim=160) ─┘

Differences from the v1.1 blocks in ``modules.py`` / ``ssl_model.py``:

* **Pooling head.** v1.1 pools with ``TemporalAwarePooling`` (CustomGRU +
  LayerNorm + attention); v2 pools with a plain single-head softmax attention
  (:class:`AttentionPooling` — one ``Linear(D, 1)``, no recurrence). There is no
  GRU anywhere in v2, so ``builder._fuse_custom_gru`` has nothing to fuse.
* **Final LayerNorm.** The v2 ``dim_reducer`` ends in a non-affine
  ``LayerNorm(repr_dim)``, so every output vector has per-pixel mean 0 / std 1
  across its 128 dimensions. It carries no parameters, so the state dict and
  parameter count are unchanged by it.
* **Native 128-d output.** v2 emits ``repr_dim=128`` directly and the dims are
  Matryoshka-ordered (the first K are independently usable for K in
  {16, 32, 64, 128}); v1.1 emits 192 and the pipeline saves the first 128.

The parameter names here match the upstream v2 ``PixelStudent`` exactly
(``s2_backbone.*``, ``s1_backbone.*``, ``dim_reducer.*``), which is also what
:class:`~tessera_embeddings.inference.models.ssl_model.MultimodalBTInferenceModel`
uses — so the runtime graph is assembled from that wrapper (it adds the
dual-CUDA-stream backbone execution) and the checkpoint loads ``strict=True``.
``tests/unit/test_student_v2_golden.py`` pins our output against the vendored
upstream reference on the real checkpoint.

Ported from ``geotessera/TESSERA-V-2.0-2B-L`` (Hugging Face), ``model.py``
— identical to ``ucam-eo/tessera``'s ``tessera_infer_v2/student/model.py``.
Changes: type hints, ruff formatting, and the positional encoder is the shared
:class:`~tessera_embeddings.inference.models.modules.TemporalPositionalEncoder`
(bit-identical in fp32; it caches ``div_term`` and casts its output back to the
input dtype so a BF16 model does not silently upcast the whole transformer).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — upstream spelling

from .modules import TemporalPositionalEncoder


class AttentionPooling(nn.Module):
    """Plain single-head softmax attention pool over the temporal axis.

    A single ``Linear(D, 1) -> softmax -> weighted sum``. No GRU, no LayerNorm —
    this is the v2 student's pooling head.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool ``(B, T, D)`` to ``(B, D)``."""
        b, t, d = x.shape
        if t == 0:
            return torch.zeros(b, d, device=x.device, dtype=x.dtype)
        if t == 1:
            return x.squeeze(1)
        w = torch.softmax(self.query(x), dim=1)
        return (w * x).sum(dim=1)


class QKNormEncoderLayer(nn.Module):
    """Encoder layer identical to ``nn.TransformerEncoderLayer`` plus QK-norm.

    Post-LN, ReLU FFN, batch-first — the only difference from the standard layer
    is per-head RMSNorm applied to Q and K before scaled-dot-product attention.
    Parameter count is effectively the same (q/k/v as three separate Linears
    matches the standard layer's combined ``in_proj``, plus two tiny RMSNorm
    gains).

    The v2 Large checkpoint was trained with ``enable_qk_norm=False``, so this
    layer is not part of its graph; it is ported because it is part of the
    upstream v2 student architecture.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} % nhead {nhead} != 0")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._attn_p = float(dropout)

    def _sa(self, x: torch.Tensor) -> torch.Tensor:
        """Self-attention with per-head RMSNorm on Q and K."""
        b, t, d = x.shape
        h, hd = self.nhead, self.head_dim
        q = self.q_proj(x).view(b, t, h, hd)
        k = self.k_proj(x).view(b, t, h, hd)
        v = self.v_proj(x).view(b, t, h, hd)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # (B, H, T, hd)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self._attn_p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(b, t, d)
        return self.out_proj(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Post-LN self-attention block followed by the post-LN FFN block."""
        x = self.norm1(x + self.dropout1(self._sa(x)))
        return self.norm2(x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x))))))


class StudentTransformerEncoder(nn.Module):
    """Per-pixel band embedding + DOY positional encoding + transformer + attention pool.

    Upstream calls this ``TransformerEncoder``; renamed here only to keep it
    distinguishable from the v1.1 :class:`modules.TransformerEncoder` (which
    differs in its pooling head).

    ``enable_qk_norm=False`` (default) uses the standard
    ``nn.TransformerEncoderLayer`` stack (ReLU, no QK-norm); ``True`` uses
    :class:`QKNormEncoderLayer`.
    """

    def __init__(
        self,
        band_num: int,
        latent_dim: int,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        enable_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        _ = max_seq_len  # Unused; kept to match the upstream signature.
        self.embedding = nn.Sequential(
            nn.Linear(band_num, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

        self.enable_qk_norm = bool(enable_qk_norm)
        self.transformer_encoder: nn.Module
        if self.enable_qk_norm:
            self.transformer_encoder = nn.ModuleList(
                [QKNormEncoderLayer(latent_dim * 4, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)]
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim * 4,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attn_pool = AttentionPooling(latent_dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, T, band_num + 1)`` — last channel is raw integer DOY.

        Returns:
            Pooled representation of shape ``(B, latent_dim * 4)``.
        """
        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        x = self.embedding(bands) + self.temporal_encoder(doy)
        if self.enable_qk_norm:
            for layer in cast("nn.ModuleList", self.transformer_encoder):
                x = layer(x)
        else:
            x = self.transformer_encoder(x)
        return self.attn_pool(x)


def build_v2_dim_reducer(latent_dim: int, repr_dim: int) -> nn.Sequential:
    """Build the v2 student's fusion MLP (concat fusion, two backbones).

    Shapes match upstream ``PixelStudent`` exactly::

        fused_in = latent_dim * 4 * 2
        Linear(fused_in, fused_in*2) → LayerNorm → ReLU → Dropout(0.2)
          → Linear(fused_in*2, repr_dim) → LayerNorm(repr_dim, affine=False)

    The trailing non-affine LayerNorm is the v2 addition: it puts every output
    vector at per-pixel mean 0 / std 1 and carries no parameters, so the
    checkpoint's ``dim_reducer.{0,1,4}`` keys are unchanged.
    """
    fused_in = latent_dim * 4 * 2
    return nn.Sequential(
        nn.Linear(fused_in, fused_in * 2),
        nn.LayerNorm(fused_in * 2),
        nn.ReLU(inplace=False),
        nn.Dropout(0.2),
        nn.Linear(fused_in * 2, repr_dim),
        nn.LayerNorm(repr_dim, elementwise_affine=False),
    )
