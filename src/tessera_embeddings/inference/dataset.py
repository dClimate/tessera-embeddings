"""Mosaicked Zarr chunk dataset for inference.

Replaces tessera's SingleTileInferenceDataset, operating on numpy arrays
from data_loading.py rather than .npy files on disk.

Same valid-pixel filtering logic as tessera: pixels must have non-zero S2 bands,
≥min_valid_timesteps valid S2 frames, and ≥min_valid_timesteps total S1 frames.

Source arrays are kept in memory in their original (T, H, W, bands) layout
(native store dtype: uint16, 2 bytes). ``get_batch()`` does vectorized fancy
indexing per call — ``source[:, rows, cols, :]`` — to pull an entire batch of
~3584 pixels in one numpy operation. The per-batch copy is ~5 MB, negligible
compared to the ~900 ms GPU forward pass. Arrays keep their native store dtype
throughout; sampling upcasts to float32 for standardization; the model receives
float16 via ``.half()``.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from tessera_embeddings.config.inference import S1_BAND_MEAN, S1_BAND_STD, S2_BAND_MEAN, S2_BAND_STD
from tessera_embeddings.inference.data_loading import ChunkData

logger = logging.getLogger(__name__)


class MosaicChunkInferenceDataset:
    """Dataset yielding per-pixel time series for one spatial chunk.

    Each item returns the full time series for one valid pixel (no random sampling).
    Sampling is deferred to the inference loop so it can be repeated and averaged.

    CHANGED: Source arrays are kept in (T, H, W, bands) layout and indexed lazily
    per batch via get_batch(). No pre-extraction step — eliminates the peak memory
    spike where source + extracted arrays coexisted.

    CHANGED: Added ``s1_orbit`` parameter to auto-halve the S1 validity
    threshold when only one orbit direction is used. Each orbit contributes
    roughly half the total S1 observations, so the threshold is halved to
    avoid excluding pixels that have sufficient data from the active orbit.

    Args:
        chunk_data: Loaded chunk data from ``data_loading.load_chunk()``.
        min_valid_timesteps: Minimum valid timesteps required for both S2 and S1.
        s1_orbit: Which S1 orbit direction(s) are active. The S1 threshold
            is halved since only one orbit contributes data.
    """

    def __init__(
        self,
        chunk_data: ChunkData,
        min_valid_timesteps: int = 10,
        s1_orbit: Literal["ascending", "descending"] = "ascending",
    ) -> None:
        self.min_valid_timesteps = min_valid_timesteps
        self.s1_orbit = s1_orbit
        self.H = chunk_data.height
        self.W = chunk_data.width

        self.s2_band_mean = np.array(S2_BAND_MEAN, dtype=np.float32)
        self.s2_band_std = np.array(S2_BAND_STD, dtype=np.float32)
        self.s1_band_mean = np.array(S1_BAND_MEAN, dtype=np.float32)
        self.s1_band_std = np.array(S1_BAND_STD, dtype=np.float32)

        # Find valid pixels and store source array references for lazy access.
        # CHANGED: Replaces _pre_extract which copied valid pixel data into
        # contiguous (N, T, bands) arrays. Now just identifies valid pixels
        # and keeps the source arrays for per-batch fancy indexing.
        self._find_valid_pixels(chunk_data)

    def _find_valid_pixels(self, chunk_data: ChunkData) -> None:
        """Identify valid pixels and store source arrays for lazy batch access.

        Valid pixels must have:
        - Non-zero S2 bands in at least one timestep
        - ≥min_valid_timesteps valid S2 frames (from SCL mask)
        - ≥threshold total valid S1 frames (threshold halved for single-orbit)

        Source arrays are stored as instance attributes in their original
        (T, H, W, bands) layout. get_batch() and __getitem__() do per-call
        fancy indexing from these arrays.
        """
        s2_bands = chunk_data.s2_bands  # (T_s2, H, W, 10)
        s2_masks = chunk_data.s2_masks  # (T_s2, H, W)
        s1_asc = chunk_data.s1_asc_bands  # (T_s1a, H, W, 2)
        s1_desc = chunk_data.s1_desc_bands  # (T_s1d, H, W, 2)

        # --- Valid pixel detection ---
        # Use pre-pruning valid count when available (from selective timestep loading).
        # This prevents the min_valid_timesteps check from being applied to a
        # pruned time axis where fewer timesteps make the threshold unreachable.
        if chunk_data.s2_obs_count is not None:
            s2_valid_count = chunk_data.s2_obs_count  # (H, W) — from full SCL
        else:
            s2_valid_count = s2_masks.sum(axis=0)  # (H, W) — fallback for eager loader

        # CHANGED: Per-timestep nonzero check to avoid materializing a
        # (T, H, W, 10) boolean temporary. For a 2000x2000 chunk with 109
        # timesteps (~1 year of data), the one-shot np.any(s2_bands != 0, axis=(0,3)) would
        # create a ~6.8 GB boolean array. The per-timestep loop creates only
        # (H, W, 10) = ~62 MB per iteration.
        s2_nonzero = np.zeros((self.H, self.W), dtype=bool)
        for t in range(s2_bands.shape[0]):
            s2_nonzero |= np.any(s2_bands[t] != 0, axis=-1)

        s1_asc_valid = np.any(s1_asc != 0, axis=-1).sum(axis=0)  # (H, W)
        s1_desc_valid = np.any(s1_desc != 0, axis=-1).sum(axis=0)  # (H, W)
        s1_total_valid = s1_asc_valid + s1_desc_valid

        # CHANGED: When using a single orbit, halve the S1 threshold since each
        # orbit provides roughly half the total S1 observations.
        s1_threshold = (self.min_valid_timesteps + 1) // 2
        s1_threshold = max(s1_threshold, 1)

        valid_mask = s2_nonzero & (s2_valid_count >= self.min_valid_timesteps) & (s1_total_valid >= s1_threshold)

        # Valid pixel coordinates for per-batch fancy indexing
        self._rows, self._cols = np.where(valid_mask)
        n_valid = len(self._rows)

        # Global flat indices for writing results back to (H, W) grid
        self._global_idxs = (self._rows * self.W + self._cols).astype(np.int64)  # (N,)

        # CHANGED: Store references to source arrays in their original
        # (T, H, W, bands) layout instead of extracting into (N, T, bands).
        # Eliminates the extraction step that doubled peak memory.
        # get_batch() does per-call fancy indexing: source[:, rows, cols, :]
        # creates a ~5 MB batch copy — negligible vs the 900ms forward pass.
        self._s2_bands = s2_bands  # (T_s2, H, W, 10)
        self._s2_masks = s2_masks  # (T_s2, H, W)
        self._s1_asc_bands = s1_asc  # (T_s1a, H, W, 2)
        self._s1_desc_bands = s1_desc  # (T_s1d, H, W, 2)

        # DOYs are shared across all pixels (not per-pixel), stored once
        self._s2_doys = chunk_data.s2_doys  # (T_s2,)
        self._s1_asc_doys = chunk_data.s1_asc_doys  # (T_s1a,)
        self._s1_desc_doys = chunk_data.s1_desc_doys  # (T_s1d,)

        logger.info(
            "MosaicChunkInferenceDataset: %d valid pixels out of %d total (%.1f%%)",
            n_valid,
            self.H * self.W,
            100.0 * n_valid / max(self.H * self.W, 1),
        )

    def __len__(self) -> int:
        """Number of valid pixels in this chunk."""
        return len(self._global_idxs)

    def get_batch(self, start: int, end: int) -> dict[str, np.ndarray]:
        """Return a batch of valid pixels via vectorized fancy indexing.

        Replaces the DataLoader + ``__getitem__`` pattern. That pattern called
        ``__getitem__`` once per pixel (B=3584 Python function calls), each
        producing a tiny single-pixel dict, then ran ``default_collate`` —
        another Python loop that stacked every key across the list of dicts,
        copying all the data a second time. The combined overhead (thousands
        of function calls + collation copy) could burn 50-100 ms per batch,
        which is pure waste next to a ~900 ms forward pass.

        This method does the same work in a single call: one vectorized
        numpy fancy-index per key pulls all B pixels at once from the source
        (T, H, W, bands) arrays, producing the batched output directly with
        no per-pixel Python loop and no collation step.

        Args:
            start: Start index into valid pixel array (inclusive).
            end: End index into valid pixel array (exclusive).

        Returns:
            Dict of batched arrays:
                global_idxs: (B,) int64
                s2_bands: (B, T_s2, 10) uint16
                s2_masks: (B, T_s2) int32
                s2_doys: (B, T_s2) — broadcast from shared DOY array
                s1_asc_bands: (B, T_s1a, 2) uint16
                s1_asc_doys: (B, T_s1a) — broadcast from shared DOY array
                s1_desc_bands: (B, T_s1d, 2) uint16
                s1_desc_doys: (B, T_s1d) — broadcast from shared DOY array
        """
        b = end - start
        rows = self._rows[start:end]
        cols = self._cols[start:end]

        # Fancy index source[:, rows, cols, :] → (T, B, bands), then
        # transpose to (B, T, bands) for consistency with downstream consumers.
        return {
            "global_idxs": self._global_idxs[start:end],
            "s2_bands": self._s2_bands[:, rows, cols, :].transpose(1, 0, 2),
            "s2_masks": self._s2_masks[:, rows, cols].T,
            "s2_doys": np.broadcast_to(self._s2_doys[None, :], (b, len(self._s2_doys))),
            "s1_asc_bands": self._s1_asc_bands[:, rows, cols, :].transpose(1, 0, 2),
            "s1_asc_doys": np.broadcast_to(self._s1_asc_doys[None, :], (b, len(self._s1_asc_doys))),
            "s1_desc_bands": self._s1_desc_bands[:, rows, cols, :].transpose(1, 0, 2),
            "s1_desc_doys": np.broadcast_to(self._s1_desc_doys[None, :], (b, len(self._s1_desc_doys))),
        }
