"""Synthetic (and optional real) embedding data for the scale tests.

Two properties matter for the tests to be meaningful:

* **Determinism.** ``embedding_block`` is a pure function of ``(seed, block
  index)`` so read-back verification recomputes the expected bytes without
  storing the inputs.
* **Spatial coherence.** ``land_mask`` produces contiguous "land" blobs, not iid
  noise — only land chunks are written, ocean chunks stay unwritten (fill), and
  that coherence is what makes the ref-count and region-write metrics realistic.

Random int8 is deliberately worst-case for PCodec; pass ``--real-sample`` (see
:class:`RealSample`) on the T1 winner/runner-up to sanity-check compressibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _rng(seed: int, *tags: int) -> np.random.Generator:
    """Deterministic Generator keyed on a base seed plus integer tags."""
    return np.random.default_rng([seed, *tags])


def _box_smooth(field: np.ndarray, passes: int) -> np.ndarray:
    """Smooth a 2-D field with repeated 3x3 box blurs (no scipy dependency)."""
    out = field.astype("float64")
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        acc = np.zeros_like(out)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                acc += padded[dy : dy + out.shape[0], dx : dx + out.shape[1]]
        out = acc / 9.0
    return out


def land_mask(n_y_chunks: int, n_x_chunks: int, *, fraction: float = 0.7, seed: int = 0) -> np.ndarray:
    """Return a boolean ``(n_y_chunks, n_x_chunks)`` mask of coherent "land".

    A coarse random field is smoothed into blobs and thresholded at the quantile
    that yields ~``fraction`` True cells. Deterministic in ``seed``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    field = _rng(seed, 7411).random((n_y_chunks, n_x_chunks))
    smoothed = _box_smooth(field, passes=max(1, min(n_y_chunks, n_x_chunks) // 4))
    threshold = np.quantile(smoothed, 1.0 - fraction)
    mask = smoothed >= threshold
    if not mask.any():  # degenerate tiny grids: guarantee at least one land cell
        mask[0, 0] = True
    return mask


def embedding_block(shape: tuple[int, ...], *, seed: int, block_index: tuple[int, ...]) -> np.ndarray:
    """Return a deterministic int8 embedding block for one chunk position.

    Pure function of ``(seed, block_index)`` so a reader can recompute and verify
    without the writer persisting anything.
    """
    rng = _rng(seed, *block_index)
    return rng.integers(-127, 128, size=shape, dtype=np.int8)


def scales_block(shape: tuple[int, ...], *, seed: int, block_index: tuple[int, ...]) -> np.ndarray:
    """Return a deterministic positive float32 per-pixel scale block."""
    rng = _rng(seed, 999, *block_index)
    return (rng.random(size=shape, dtype="float32") * 0.05 + 1e-3).astype("float32")


class RealSample:
    """Tiles a real-embedding ``.npy`` sample to fill requested blocks.

    The sample is flattened to ``(N, band)`` rows; each block draws a
    deterministic permutation of rows keyed on its block index, so blocks differ
    but the values stay real. Float samples are per-row absmax-quantized to int8
    (the store dtype); int8 samples are used as-is.
    """

    def __init__(self, path: Path, band: int) -> None:
        """Load and reshape the sample from ``path`` to ``(N, band)`` int8 rows."""
        arr = np.load(path)
        rows = arr.reshape(-1, arr.shape[-1])
        if rows.shape[-1] != band:
            raise ValueError(f"real sample has {rows.shape[-1]} channels, expected {band}")
        if np.issubdtype(rows.dtype, np.floating):
            from tessera_embeddings.inference.quantization import quantize_rows

            quantized, _ = quantize_rows(rows.astype("float32"))
            self._rows = quantized
        else:
            self._rows = rows.astype("int8")

    def block(self, shape: tuple[int, ...], *, block_index: tuple[int, ...]) -> np.ndarray:
        """Return an int8 block of ``shape`` filled from permuted real rows."""
        n_vectors = int(np.prod(shape[:-1]))
        rng = _rng(0, *block_index)
        idx = rng.integers(0, self._rows.shape[0], size=n_vectors)
        return self._rows[idx].reshape(shape)
