"""Symmetric INT8 quantization for embedding outputs.

Quantizes float32 embeddings to int8 for ~4x storage reduction, matching the
tessera beta QAT pipeline's scheme. The scale is per-PIXEL rather than
per-tensor: each spatial location's scale comes from the max absolute value
across its own 128 embedding channels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def quantize_rows_torch(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """On-device equivalent of :func:`quantize_rows` for a ``(N, D)`` GPU tensor.

    Quantizes in the forward's compute stream so the seq-length-independent
    quantize cost folds into GPU work instead of running as a separate CPU stage
    (which gates the pipeline on low-observation chunks, where the forward is
    tiny). Returns device tensors — ``(N, D)`` int8 and ``(N,)`` float32 — that
    the caller copies to host together, shrinking the D2H transfer ~4x against
    copying float32 embeddings and quantizing on the CPU.

    The arithmetic mirrors :func:`quantize_rows` exactly (cast to float32, per-row
    max-abs scale with the same 1e-8 floor, round-half-to-even, clip to
    [-127, 127]), so the int8 output is bit-identical to the CPU path on the same
    float32 input. ``rows`` is cast to float32 first because the forward emits
    bf16/fp16; the CPU path likewise quantizes the ``.float()`` output.

    Unlike the CPU path, this does NOT validate finiteness on-device: an
    ``isfinite().all()`` here forces a host sync plus an extra full pass over the
    activations on every sub-batch, stalling the GPU pipeline. Any non-finite row
    necessarily produces a non-finite scale (NaN/Inf propagates through
    ``amax``/``clamp``), so callers validate the scales on the host after the D2H
    copy instead — see :func:`raise_on_nonfinite_scales`.
    """
    import torch

    rows = rows.float()
    abs_max = rows.abs().amax(dim=-1)  # (N,)
    scales = torch.clamp(abs_max / 127.0, min=1e-8)  # (N,) float32, avoid div-by-zero
    # torch.round is round-half-to-even, matching numpy's np.round in quantize_rows.
    quantized = torch.clamp(torch.round(rows / scales[:, None]), -127, 127).to(torch.int8)
    return quantized, scales


def raise_on_nonfinite_scales(scales_host: np.ndarray) -> None:
    """Reject a batch whose per-row scales carry NaN/Inf (host-side, post-D2H).

    The GPU quantizer skips on-device finiteness validation (see
    :func:`quantize_rows_torch`); a non-finite embedding row always surfaces as
    a non-finite scale, so this cheap ``(N,)`` check on the already-copied host
    array is equivalent to the full on-device check without the pipeline stall.
    """
    if not np.isfinite(scales_host).all():
        raise ValueError("embeddings contain non-finite values (NaN or Inf); cannot quantize")


def quantize_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32 embedding rows to int8 using symmetric per-row quantization.

    Row-wise variant of :func:`quantize_embeddings` for ``(N, D)`` inputs, so a
    bucket's rows can be quantized as they come off the GPU instead of buffering
    the whole chunk in float32. Each row's scale comes from its own channels, so it
    is numerically identical to quantizing the full ``(H, W, D)`` array.

    Args:
        rows: Array of shape ``(N, D)``, dtype float32.

    Returns:
        Tuple of (quantized, scales):
        - quantized: ``(N, D)`` int8 array.
        - scales: ``(N,)`` float32 array — multiply ``quantized * scales[:, None]``
          to recover approximate float32 values.
    """
    if not np.isfinite(rows).all():
        raise ValueError("embeddings contain non-finite values (NaN or Inf); cannot quantize")

    # Per-row max absolute value across embedding channels
    abs_max = np.abs(rows).max(axis=-1)  # (N,)
    scales = np.maximum(abs_max / 127.0, 1e-8).astype(np.float32)  # avoid div-by-zero

    quantized = np.clip(np.round(rows / scales[:, np.newaxis]), -127, 127).astype(np.int8)
    return quantized, scales


def quantize_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32 embeddings to int8 using symmetric per-pixel quantization.

    Args:
        embeddings: Array of shape ``(H, W, D)``, dtype float32.

    Returns:
        Tuple of (quantized, scales):
        - quantized: ``(H, W, D)`` int8 array.
        - scales: ``(H, W)`` float32 array — multiply ``quantized * scales[..., None]``
          to recover approximate float32 values.
    """
    h, w, d = embeddings.shape
    quantized, scales = quantize_rows(embeddings.reshape(h * w, d))
    return quantized.reshape(h, w, d), scales.reshape(h, w)


def dequantize_embeddings(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize int8 embeddings back to float32.

    Args:
        quantized: ``(H, W, D)`` int8 array.
        scales: ``(H, W)`` float32 array (from :func:`quantize_embeddings`).

    Returns:
        ``(H, W, D)`` float32 array.
    """
    return quantized.astype(np.float32) * scales[..., np.newaxis]
