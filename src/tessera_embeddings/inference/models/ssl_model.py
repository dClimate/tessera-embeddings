"""Multimodal Tessera v1.1 inference model (merged S1 backbone).

Architecture::

    S2 stream ──► V11TransformerEncoder (latent_dim=192) ─┐
                                                       ├─ concat ─► dim_reducer ──► (B, 192)
    S1 merged ──► V11TransformerEncoder (latent_dim=192) ─┘

The S1 "merged" stream is ascending + descending concatenated time-wise, each
pre-normalised with its own per-modality mean/std (done at the dataset level).

``dim_reducer`` matches v1.1 training exactly::

    Linear(in, in*2) → LayerNorm(in*2) → ReLU → Dropout(0.2) → Linear(in*2, 192)

where ``in = latent_dim * 4 * 2 = 1536`` for concat fusion.

The projector and segmented-matryoshka-projector from pretraining are stripped
at checkpoint load time — they are not part of the inference graph.

Ported from ``ucam-eo/tessera`` tag ``v1.1``
(``tessera_infer_QAT/src/models/ssl_model_v1_1.py``).

``MultimodalBTInferenceModel`` also hosts the v2 Large student: its topology,
parameter names, and fusion math are the same, only the backbone pooling head
and the reducer's trailing LayerNorm differ (see ``student_v2.py``).
``build_dim_reducer`` below stays v1.1-specific.

On a CUDA device in eval mode with gradients off, a v2 model runs each sub-batch in chunks of
``GRAPH_CHUNK_PIXELS`` pixels, each chunk replaying a CUDA graph captured once per bucket shape::

    sub-batch (B pixels) ─► chunk 1 ─► copy into static inputs ─► replay graph ─► copy rows out ─┐
                            chunk 2 ─► …                                                        ├─► (B, 128)
                            last chunk (m ≤ 64 rows) ─► same, keeping only m rows ──────────────┘

Small chunks keep the intermediates in the GPU's cache; the graph removes the per-operation
launch cost that would otherwise make small chunks slower. v1.1, the CPU, and any forward with
gradients on or in training mode run the whole batch at once. Measurements:
``context_docs/inference/inference-on-gpus.md``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Pixels per replayed chunk for v2: 64 was the fastest size measured on an L40S, and 128 was slower.
GRAPH_CHUNK_PIXELS = 64

_Graph = tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor]
"""A captured forward: (graph, static S2 input, static S1 input, static output)."""


def run_in_chunks(
    s2_x: torch.Tensor,
    s1_x: torch.Tensor,
    s2_static: torch.Tensor,
    s1_static: torch.Tensor,
    z_static: torch.Tensor,
    run_chunk: Callable[[], None],
) -> torch.Tensor:
    """Run a batch through fixed-size static buffers, one chunk at a time.

    Each chunk's rows are copied into ``s2_static``/``s1_static``, ``run_chunk`` fills ``z_static``
    from them (on CUDA it is a captured graph's ``replay``), and the chunk's rows are copied out.
    The chunk size is the static buffers' first dimension.
    """
    n, c = s2_x.shape[0], s2_static.shape[0]
    out = z_static.new_empty((n, z_static.shape[1]))
    for i in range(0, n, c):
        # A short last chunk leaves stale rows past m; pixels are embedded independently, so they never reach kept rows.
        m = min(c, n - i)
        s2_static[:m].copy_(s2_x[i : i + m])
        s1_static[:m].copy_(s1_x[i : i + m])
        run_chunk()
        out[i : i + m] = z_static[:m]
    return out


class MultimodalBTInferenceModel(nn.Module):
    """Two-backbone Tessera inference model (concat fusion).

    Output is the model's native representation: 192-dim under v1.1, 128-dim
    under v2 Large. Callers that want the canonical 128-d downstream save slice
    ``out[:, :128]`` — done in the inference loop, not the model, so the forward
    stays aligned with the upstream reference implementations.

    The parameter names (``s2_backbone``, ``s1_backbone``, ``dim_reducer``) and
    the fusion math are identical in both versions, so this wrapper hosts either
    version's backbones and reducer; the version-specific blocks live in
    ``modules.py`` (v1.1) and ``student_v2.py`` (v2). ``builder.py`` picks.

    ``chunk_pixels`` turns on chunks of that many pixels replaying a captured CUDA graph (see
    the module docstring); ``None`` runs every batch whole. A graph is captured the first time a
    ``(T_s2, T_s1)`` bucket shape is seen and kept for the life of the model, with no eviction:
    there is at most one per bucket shape, which the default ladder of 32 sequence lengths per
    stream bounds at about a thousand. All of a model's graphs share one memory pool, which is
    safe because they never replay at the same time and each chunk's output is copied out before
    the next replay; it keeps graph memory near one capture's peak however many bucket shapes a
    worker sees.
    """

    def __init__(
        self,
        s2_backbone: nn.Module,
        s1_backbone: nn.Module,
        dim_reducer: nn.Module,
        fusion_method: str = "concat",
        chunk_pixels: int | None = None,
    ) -> None:
        super().__init__()
        self.s2_backbone = s2_backbone
        self.s1_backbone = s1_backbone
        self.dim_reducer = dim_reducer
        self.fusion_method = fusion_method
        self.chunk_pixels = chunk_pixels
        # Per (T_s2, T_s1); a plain dict, so graphs never enter state_dict.
        self._graphs: dict[tuple[int, int], _Graph] = {}
        self._graph_pool: torch.cuda._POOL_HANDLE | None = None

        # Pre-create CUDA streams for parallel backbone execution.
        self._s2_stream: torch.cuda.Stream | None = None
        self._s1_stream: torch.cuda.Stream | None = None

    def _ensure_streams(self, device: torch.device) -> None:
        if device.type == "cuda" and self._s2_stream is None:
            self._s2_stream = torch.cuda.Stream(device=device)
            self._s1_stream = torch.cuda.Stream(device=device)

    def replays_graphs(self, s2_x: torch.Tensor) -> bool:
        """Whether ``forward`` on this input runs chunks replaying captured CUDA graphs."""
        return self.chunk_pixels is not None and s2_x.is_cuda and not torch.is_grad_enabled() and not self.training

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing the model's native representation.

        Args:
            s2_x: S2 input, shape ``(B, T_s2, 11)`` — 10 bands + DOY.
            s1_x: Merged S1 input, shape ``(B, T_s1, 3)`` — 2 bands + DOY.

        Returns:
            Representation tensor of shape ``(B, 192)`` under v1.1, ``(B, 128)`` under v2 Large.
        """
        if not self.replays_graphs(s2_x):
            return self._forward_whole_batch(s2_x, s1_x)
        key = (s2_x.shape[1], s1_x.shape[1])
        if key not in self._graphs:
            self._graphs[key] = self._capture(s2_x, s1_x)
        graph, s2_static, s1_static, z_static = self._graphs[key]
        return run_in_chunks(s2_x, s1_x, s2_static, s1_static, z_static, graph.replay)

    def _capture(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> _Graph:
        """Capture one chunk's forward for this bucket shape, on static zero inputs."""
        assert self.chunk_pixels is not None
        s2_static = s2_x.new_zeros((self.chunk_pixels, *s2_x.shape[1:]))
        s1_static = s1_x.new_zeros((self.chunk_pixels, *s1_x.shape[1:]))
        # Warm-up runs any lazy first-call work (compilation, per-device caches) outside the capture.
        for _ in range(3):
            self._forward_whole_batch(s2_static, s1_static)
        if self._graph_pool is None:
            self._graph_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool):
            z_static = self._forward_whole_batch(s2_static, s1_static)
        return graph, s2_static, s1_static, z_static

    def _forward_whole_batch(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        """The forward over the whole batch at once; also the body a captured graph records."""
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
