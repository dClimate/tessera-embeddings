"""Multimodal BarlowTwins SSL model and inference variant.

Ported from tessera_infer/src/models/ssl_model.py with type hints and ruff compliance.
Logic is unchanged from the original implementation.
"""

import logging
import time

import torch
import torch.nn as nn

from .modules import ProjectionHead, TransformerEncoder

logger = logging.getLogger(__name__)


class MultimodalBTModel(nn.Module):
    """Full multimodal BarlowTwins model (training + inference).

    Contains S2 and S1 backbones, fusion, dimension reducer, and projection head.
    Used to load the full checkpoint; the projection head is then stripped for inference.
    """

    def __init__(
        self,
        s2_backbone: TransformerEncoder,
        s1_backbone: TransformerEncoder,
        projector: ProjectionHead,
        fusion_method: str = "concat",
        return_repr: bool = False,
        latent_dim: int = 128,
    ) -> None:
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.projector = projector
        self.fusion_method = fusion_method
        self.return_repr = return_repr

        if fusion_method == "concat":
            in_dim = 8 * latent_dim
        elif fusion_method == "sum":
            in_dim = 4 * latent_dim
        else:
            msg = f"Unknown fusion method: {fusion_method}"
            raise ValueError(msg)

        self.dim_reducer = nn.Sequential(nn.Linear(in_dim, latent_dim))

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both backbones, fusion, and projection.

        Args:
            s2_x: S2 input of shape (B, seq_len, 11) — 10 bands + DOY.
            s1_x: S1 input of shape (B, seq_len, 3) — 2 bands + DOY.

        Returns:
            If return_repr is True: (projected_features, fused_representation).
            Otherwise: projected_features only.
        """
        s2_repr = self.s2_backbone(s2_x)
        s1_repr = self.s1_backbone(s1_x)

        if self.fusion_method == "concat":
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        else:
            fused = s2_repr + s1_repr

        fused = self.dim_reducer(fused)
        feats = self.projector(fused)

        if self.return_repr:
            return feats, fused
        return feats


class MultimodalBTInferenceModel(nn.Module):
    """Inference-only model: backbones + fusion + dim_reducer, no projection head.

    Output is the 128-dim fused representation (the embedding).
    """

    def __init__(
        self,
        s2_backbone: TransformerEncoder,
        s1_backbone: TransformerEncoder,
        fusion_method: str,
        dim_reducer: nn.Module,
    ) -> None:
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.fusion_method = fusion_method
        self.dim_reducer = dim_reducer

        # ADDED: Pre-create CUDA streams for parallel backbone execution.
        # Streams are lightweight handles — creating them once avoids per-call overhead.
        # On CPU these are unused (forward falls back to sequential execution).
        self._s2_stream: torch.cuda.Stream | None = None
        self._s1_stream: torch.cuda.Stream | None = None

    def _ensure_streams(self, device: torch.device) -> None:
        """Lazily initialize CUDA streams on first forward pass.

        Deferred to forward() because the model may be built before .to(device),
        and stream creation requires knowing the CUDA device.
        """
        if device.type == "cuda" and self._s2_stream is None:
            self._s2_stream = torch.cuda.Stream(device=device)
            self._s1_stream = torch.cuda.Stream(device=device)

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing 128-dim embeddings.

        CHANGED from original: S2 and S1 backbones run on separate CUDA streams
        so their kernels can overlap on the GPU. The two backbones are completely
        independent (no shared parameters or state) until the fusion step, so
        this is safe and produces bit-identical results. On CPU, falls back to
        sequential execution.

        Args:
            s2_x: S2 input of shape (B, seq_len, 11) — 10 bands + DOY.
            s1_x: S1 input of shape (B, seq_len, 3) — 2 bands + DOY.

        Returns:
            Embedding tensor of shape (B, 128).
        """
        self._ensure_streams(s2_x.device)
        profile = getattr(self, "_profile", False)

        if profile and s2_x.is_cuda:
            torch.cuda.synchronize()
            tf0 = time.monotonic()

        # CHANGED from original: run backbones on parallel CUDA streams.
        # During the profiled batch, run sequentially so per-backbone timing is accurate.
        if profile and s2_x.is_cuda:
            # Sequential execution for accurate per-backbone timing
            torch.cuda.synchronize()
            ts2_start = time.monotonic()
            s2_repr = self.s2_backbone(s2_x)
            torch.cuda.synchronize()
            ts2_end = time.monotonic()
            s1_repr = self.s1_backbone(s1_x)
            torch.cuda.synchronize()
            ts1_end = time.monotonic()
        elif self._s2_stream is not None and self._s1_stream is not None:
            with torch.cuda.stream(self._s2_stream):
                s2_repr = self.s2_backbone(s2_x)
            with torch.cuda.stream(self._s1_stream):
                s1_repr = self.s1_backbone(s1_x)
            torch.cuda.current_stream().wait_stream(self._s2_stream)
            torch.cuda.current_stream().wait_stream(self._s1_stream)
        else:
            s2_repr = self.s2_backbone(s2_x)
            s1_repr = self.s1_backbone(s1_x)

        if profile and s2_x.is_cuda:
            tf1 = time.monotonic()

        if self.fusion_method == "sum":
            fused = s2_repr + s1_repr
        elif self.fusion_method == "concat":
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        else:
            msg = f"Unknown fusion method: {self.fusion_method}"
            raise ValueError(msg)

        result = self.dim_reducer(fused)

        if profile and s2_x.is_cuda:
            torch.cuda.synchronize()
            tf2 = time.monotonic()
            s2_ms = (ts2_end - ts2_start) * 1000
            s1_ms = (ts1_end - ts2_end) * 1000
            sequential_ms = s2_ms + s1_ms
            logger.debug(
                "PROFILE MultimodalBTInferenceModel: "
                "s2_backbone=%.1fms  s1_backbone=%.1fms  sequential_total=%.1fms  "
                "fusion+reducer=%.1fms  "
                "s2_repr dtype=%s  s1_repr dtype=%s  output dtype=%s  TOTAL=%.1fms",
                s2_ms,
                s1_ms,
                sequential_ms,
                (tf2 - tf1) * 1000,
                s2_repr.dtype,
                s1_repr.dtype,
                result.dtype,
                (tf2 - tf0) * 1000,
            )

        return result
