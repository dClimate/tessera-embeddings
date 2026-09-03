"""Multimodal Tessera v1.1 inference model (merged S1 backbone).

Architecture::

    S2 stream ──► TransformerEncoder (latent_dim=192) ─┐
                                                       ├─ concat ─► dim_reducer ──► (B, 192)
    S1 merged ──► TransformerEncoder (latent_dim=192) ─┘

The S1 "merged" stream is ascending + descending concatenated time-wise, each
pre-normalised with its own per-modality mean/std (done at the dataset level).

``dim_reducer`` matches v1.1 training exactly::

    Linear(in, in*2) → LayerNorm(in*2) → ReLU → Dropout(0.2) → Linear(in*2, 192)

where ``in = latent_dim * 4 * 2 = 1536`` for concat fusion.

The projector and segmented-matryoshka-projector from pretraining are stripped
at checkpoint load time — they are not part of the inference graph.

Ported from ``ucam-eo/tessera`` tag ``v1.1``
(``tessera_infer_QAT/src/models/ssl_model_v1_1.py``).
"""

from __future__ import annotations

import logging
import time

import torch
import torch.nn as nn

from .modules import TransformerEncoder

logger = logging.getLogger(__name__)


class MultimodalBTInferenceModel(nn.Module):
    """Two-backbone Tessera v1.1 inference model (concat fusion).

    Output is the 192-dim representation. Callers that want the canonical 128-d
    downstream save should slice ``out[:, :128]`` — this is done in the inference
    loop, not the model, so the forward stays aligned with the tessera v1.1
    reference implementation.
    """

    def __init__(
        self,
        s2_backbone: TransformerEncoder,
        s1_backbone: TransformerEncoder,
        dim_reducer: nn.Module,
        fusion_method: str = "concat",
    ) -> None:
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.dim_reducer = dim_reducer
        self.fusion_method = fusion_method

        # Pre-create CUDA streams for parallel backbone execution.
        self._s2_stream: torch.cuda.Stream | None = None
        self._s1_stream: torch.cuda.Stream | None = None

    def _ensure_streams(self, device: torch.device) -> None:
        if device.type == "cuda" and self._s2_stream is None:
            self._s2_stream = torch.cuda.Stream(device=device)
            self._s1_stream = torch.cuda.Stream(device=device)

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing 192-dim representations.

        Args:
            s2_x: S2 input, shape ``(B, T_s2, 11)`` — 10 bands + DOY.
            s1_x: Merged S1 input, shape ``(B, T_s1, 3)`` — 2 bands + DOY.

        Returns:
            Representation tensor of shape ``(B, 192)``.
        """
        self._ensure_streams(s2_x.device)
        profile = getattr(self, "_profile", False)

        if profile and s2_x.is_cuda:
            torch.cuda.synchronize()
            tf0 = time.monotonic()

        # On the profiled batch run backbones sequentially for accurate per-backbone timing.
        if profile and s2_x.is_cuda:
            torch.cuda.synchronize()
            ts2_start = time.monotonic()
            s2_repr = self.s2_backbone(s2_x)
            torch.cuda.synchronize()
            ts2_end = time.monotonic()
            s1_repr = self.s1_backbone(s1_x)
            torch.cuda.synchronize()
            ts1_end = time.monotonic()
        elif self._s2_stream is not None and self._s1_stream is not None:
            # The backbone streams must first wait on the caller's stream: inputs may still be
            # in flight there (async H2D from pinned memory + dtype cast). With pageable copies
            # that H2D was synchronous, which hid the missing edge.
            current = torch.cuda.current_stream()
            self._s2_stream.wait_stream(current)
            self._s1_stream.wait_stream(current)
            # wait_stream orders EXECUTION, but the caching allocator tracks a tensor's liveness
            # only on its ALLOCATION stream. It cannot see that the side streams still read
            # s2_x/s1_x, so a later `current`-stream allocation (the next pipeline iteration's
            # H2D) could reuse that storage mid-read — silent, intermittent corruption.
            # record_stream marks a tensor in use on the streams that actually consume it.
            s2_x.record_stream(self._s2_stream)
            s1_x.record_stream(self._s1_stream)
            with torch.cuda.stream(self._s2_stream):
                s2_repr = self.s2_backbone(s2_x)
            with torch.cuda.stream(self._s1_stream):
                s1_repr = self.s1_backbone(s1_x)
            current.wait_stream(self._s2_stream)
            current.wait_stream(self._s1_stream)
            # Symmetric to the inputs above: s2_repr/s1_repr were allocated on the backbone
            # streams but are consumed on `current` by the fusion below.
            s2_repr.record_stream(current)
            s1_repr.record_stream(current)
        else:
            s2_repr = self.s2_backbone(s2_x)
            s1_repr = self.s1_backbone(s1_x)

        if profile and s2_x.is_cuda:
            tf1 = time.monotonic()

        if self.fusion_method == "concat":
            fused = torch.cat([s2_repr, s1_repr], dim=-1)
        elif self.fusion_method == "sum":
            fused = s2_repr + s1_repr
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


def build_dim_reducer(latent_dim: int, active_backbones: int, repr_dim: int) -> nn.Sequential:
    """Build the v1.1 MLP dim_reducer.

    The shapes match the v1.1 training-time reducer exactly::

        in_features = latent_dim * 4 * active_backbones
        Linear(in, in*2) → LayerNorm(in*2) → ReLU → Dropout(0.2) → Linear(in*2, repr_dim)
    """
    in_features = latent_dim * 4 * active_backbones
    return nn.Sequential(
        nn.Linear(in_features, in_features * 2),
        nn.LayerNorm(in_features * 2),
        nn.ReLU(inplace=False),
        nn.Dropout(0.2),
        nn.Linear(in_features * 2, repr_dim),
    )
