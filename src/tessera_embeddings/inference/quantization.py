"""Symmetric INT8 quantization for embedding outputs.

Quantizes float32 embeddings to int8 with a per-pixel scale factor for ~4x
storage reduction. Matches the tessera beta QAT pipeline's quantization scheme.

Per-pixel (not per-tensor) scale: each spatial location gets its own scale
factor computed from the max absolute value across its 128 embedding channels.
"""

from __future__ import annotations

import numpy as np


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
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values (NaN or Inf); cannot quantize")

    # Per-pixel max absolute value across embedding channels
    abs_max = np.abs(embeddings).max(axis=-1)  # (H, W)
    scales = np.maximum(abs_max / 127.0, 1e-8).astype(np.float32)  # avoid div-by-zero

    quantized = np.clip(np.round(embeddings / scales[..., np.newaxis]), -127, 127).astype(np.int8)
    return quantized, scales


def dequantize_embeddings(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Dequantize int8 embeddings back to float32.

    Args:
        quantized: ``(H, W, D)`` int8 array.
        scales: ``(H, W)`` float32 array (from :func:`quantize_embeddings`).

    Returns:
        ``(H, W, D)`` float32 array.
    """
    return quantized.astype(np.float32) * scales[..., np.newaxis]
