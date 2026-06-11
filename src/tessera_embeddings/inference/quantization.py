"""Symmetric INT8 quantization for embedding outputs.

Quantizes float32 embeddings to int8 with a per-pixel scale factor for ~4x
storage reduction. Matches the tessera beta QAT pipeline's quantization scheme.

Per-pixel (not per-tensor) scale: each spatial location gets its own scale
factor computed from the max absolute value across its 128 embedding channels.
"""

from __future__ import annotations

import numpy as np


def quantize_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize float32 embedding rows to int8 using symmetric per-row quantization.

    Row-wise variant of :func:`quantize_embeddings` for ``(N, D)`` inputs, so a
    bucket's rows can be quantized as they come off the GPU instead of buffering
    the whole chunk in float32. Each row's scale comes from its own channels, so
    this is numerically identical to quantizing the full ``(H, W, D)`` array.

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
