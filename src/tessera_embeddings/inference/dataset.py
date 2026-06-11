"""Per-chunk pixel dataset for Tessera v1.1 bucketed inference.

Groups valid pixels by ``(s2_bin, s1_bin)`` — the bucket sizes chosen for the
pixel's S2 and merged-S1 observation counts — so the inference loop can feed
the transformer rectangular batches. Each bucket's batch is pre-resampled via
``sampling.resample_s2_bucket`` / ``resample_s1_bucket``, which apply per-modality
normalisation and the deterministic bucketed resampling (see ``sampling.py``).

Source arrays stay in their native ``(T, H, W, bands)`` layout; per-batch fancy
indexing pulls ``(B, T, bands)`` slices as needed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Literal

import numpy as np

from tessera_embeddings.config.inference import (
    DEFAULT_NUM_OBS_CHECKPOINTS,
    S1_ASC_BAND_MEAN,
    S1_ASC_BAND_STD,
    S1_DESC_BAND_MEAN,
    S1_DESC_BAND_STD,
    S2_BAND_MEAN,
    S2_BAND_STD,
    _normalize_obs_checkpoints,
)
from tessera_embeddings.inference.data_loading import ChunkData
from tessera_embeddings.inference.sampling import compute_bin_keys, resample_s1_bucket, resample_s2_bucket

logger = logging.getLogger(__name__)


class MosaicChunkInferenceDataset:
    """Bucketed pixel dataset for one spatial chunk.

    Valid pixels are grouped by ``(s2_bin, s1_bin)``. ``iter_buckets()`` yields
    one bucket at a time; ``get_bucket_batch()`` returns a pre-resampled,
    standardised numpy slice ready for GPU transfer.

    Args:
        chunk_data: Loaded chunk data from ``data_loading.load_chunk()``.
        num_obs_checkpoints: Sorted bucket sizes. A pixel with ``k`` valid
            observations is mapped to the smallest checkpoint ``>= k``.
        s1_orbit: Which S1 orbit direction(s) are active. Only affects logging.
    """

    def __init__(
        self,
        chunk_data: ChunkData,
        num_obs_checkpoints: tuple[int, ...] = DEFAULT_NUM_OBS_CHECKPOINTS,
        s1_orbit: Literal["ascending", "descending", "both"] = "ascending",
    ) -> None:
        self.num_obs_checkpoints = _normalize_obs_checkpoints(num_obs_checkpoints)
        self.s1_orbit = s1_orbit
        self.H = chunk_data.height
        self.W = chunk_data.width

        self.s2_band_mean = np.array(S2_BAND_MEAN, dtype=np.float32)
        self.s2_band_std = np.array(S2_BAND_STD, dtype=np.float32)
        self.s1a_band_mean = np.array(S1_ASC_BAND_MEAN, dtype=np.float32)
        self.s1a_band_std = np.array(S1_ASC_BAND_STD, dtype=np.float32)
        self.s1d_band_mean = np.array(S1_DESC_BAND_MEAN, dtype=np.float32)
        self.s1d_band_std = np.array(S1_DESC_BAND_STD, dtype=np.float32)

        self._bucket_pixels: dict[tuple[int, int], np.ndarray] = {}
        self._rows = np.empty((0,), dtype=np.int64)
        self._cols = np.empty((0,), dtype=np.int64)
        self._global_idxs = np.empty((0,), dtype=np.int64)

        self._find_and_bucket_pixels(chunk_data)

    def _find_and_bucket_pixels(self, chunk_data: ChunkData) -> None:
        """Identify valid pixels and assign each to a ``(s2_bin, s1_bin)`` bucket."""
        s2_bands = chunk_data.s2_bands
        s2_masks = chunk_data.s2_masks
        s1_asc = chunk_data.s1_asc_bands
        s1_desc = chunk_data.s1_desc_bands

        # Use the pre-pruning s2_obs_count when available so pixels aren't
        # under-bucketed just because the loader dropped empty S2 timesteps.
        if chunk_data.s2_obs_count is not None:
            s2_valid_count = chunk_data.s2_obs_count.astype(np.int32)
        else:
            s2_valid_count = s2_masks.sum(axis=0).astype(np.int32)

        # Any-nonzero S2 check avoids ingesting pixels with zero reflectance
        # everywhere (sensor gap, out-of-swath).
        s2_nonzero = np.zeros((self.H, self.W), dtype=bool)
        for t in range(s2_bands.shape[0]):
            s2_nonzero |= np.any(s2_bands[t] != 0, axis=-1)

        if chunk_data.s1_asc_obs_count is not None:
            s1_asc_valid = chunk_data.s1_asc_obs_count.astype(np.int32)
        else:
            s1_asc_valid = np.any(s1_asc != 0, axis=-1).sum(axis=0).astype(np.int32)
        if chunk_data.s1_desc_obs_count is not None:
            s1_desc_valid = chunk_data.s1_desc_obs_count.astype(np.int32)
        else:
            s1_desc_valid = np.any(s1_desc != 0, axis=-1).sum(axis=0).astype(np.int32)
        s1_total_valid = s1_asc_valid + s1_desc_valid

        valid_mask = s2_nonzero & (s2_valid_count > 0) & (s1_total_valid > 0)
        rows, cols = np.where(valid_mask)

        self._rows = rows
        self._cols = cols
        self._global_idxs = (rows * self.W + cols).astype(np.int64)

        self._s2_bands = s2_bands
        self._s2_masks = s2_masks
        self._s1_asc_bands = s1_asc
        self._s1_desc_bands = s1_desc
        self._s2_doys = chunk_data.s2_doys
        self._s1_asc_doys = chunk_data.s1_asc_doys
        self._s1_desc_doys = chunk_data.s1_desc_doys

        n_valid = len(rows)
        if n_valid == 0:
            logger.info(
                "MosaicChunkInferenceDataset: 0 valid pixels out of %d total",
                self.H * self.W,
            )
            return

        pixel_s2_counts = s2_valid_count[rows, cols]
        pixel_s1_counts = s1_total_valid[rows, cols]
        keys = compute_bin_keys(pixel_s2_counts, pixel_s1_counts, self.num_obs_checkpoints)

        # Group pixel indices by (s2, s1) bucket. Pack structured (int32, int32) into
        # one int64 for a stable argsort.
        flat_keys = keys.view(np.int64)
        sort_order = np.argsort(flat_keys, kind="stable")
        sorted_keys = flat_keys[sort_order]
        change_points = np.concatenate(([0], np.where(np.diff(sorted_keys) != 0)[0] + 1, [len(sorted_keys)]))

        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i + 1]
            pixel_indices = sort_order[start:end].astype(np.int64, copy=False)
            k = keys[sort_order[start]]
            self._bucket_pixels[(int(k["s2"]), int(k["s1"]))] = pixel_indices

        logger.info(
            "MosaicChunkInferenceDataset: %d valid pixels out of %d total (%.1f%%) in %d buckets",
            n_valid,
            self.H * self.W,
            100.0 * n_valid / max(self.H * self.W, 1),
            len(self._bucket_pixels),
        )

    def __len__(self) -> int:
        """Number of valid pixels in this chunk."""
        return int(self._global_idxs.shape[0])

    def iter_buckets(self, *, largest_first: bool = True) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
        """Yield each ``(bucket_key, pixel_indices_into_valid_arrays)`` pair.

        ``largest_first=True`` sorts by ``s2_target * s1_target`` descending so
        the GPU's first batch sees the largest sequence lengths — this warms up
        CUDA kernels at peak memory pressure and stabilises timing telemetry.
        """
        items = list(self._bucket_pixels.items())
        if largest_first:
            items.sort(key=lambda kv: kv[0][0] * kv[0][1], reverse=True)
        yield from items

    def bucket_sizes(self) -> dict[tuple[int, int], int]:
        """Return ``{bucket_key: num_pixels_in_bucket}``."""
        return {k: int(v.size) for k, v in self._bucket_pixels.items()}

    def get_bucket_batch(
        self,
        bucket_key: tuple[int, int],
        pixel_start: int,
        pixel_end: int,
    ) -> dict[str, np.ndarray]:
        """Return a pre-resampled, standardised batch for pixels in one bucket.

        Args:
            bucket_key: ``(s2_target, s1_target)`` bucket identifier.
            pixel_start: Start index into this bucket's pixel-index array.
            pixel_end: End index (exclusive).

        Returns:
            Dict with:
                global_idxs: ``(B,)`` int64 — flat (H*W) indices.
                s2: ``(B, s2_target, 11)`` float32 — standardised S2 + DOY.
                s1: ``(B, s1_target, 3)`` float32 — per-modality-normalised
                    merged S1 stream + DOY.
        """
        s2_target, s1_target = bucket_key
        pixel_indices = self._bucket_pixels[bucket_key][pixel_start:pixel_end]
        rows = self._rows[pixel_indices]
        cols = self._cols[pixel_indices]
        b = len(rows)

        # S2: (T_s2, H, W, 10) → (T_s2, B, 10) → (B, T_s2, 10)
        s2_bands = self._s2_bands[:, rows, cols, :].transpose(1, 0, 2)
        s2_masks = self._s2_masks[:, rows, cols].T
        s2_doys = np.broadcast_to(self._s2_doys[None, :], (b, len(self._s2_doys))).astype(np.int32, copy=False)

        s1_asc_bands = self._s1_asc_bands[:, rows, cols, :].transpose(1, 0, 2)
        s1_asc_doys = np.broadcast_to(self._s1_asc_doys[None, :], (b, len(self._s1_asc_doys))).astype(
            np.int32, copy=False
        )
        s1_desc_bands = self._s1_desc_bands[:, rows, cols, :].transpose(1, 0, 2)
        s1_desc_doys = np.broadcast_to(self._s1_desc_doys[None, :], (b, len(self._s1_desc_doys))).astype(
            np.int32, copy=False
        )

        s2 = resample_s2_bucket(
            s2_bands,
            s2_masks,
            s2_doys,
            target=s2_target,
            s2_mean=self.s2_band_mean,
            s2_std=self.s2_band_std,
        )

        s1 = resample_s1_bucket(
            s1_asc_bands,
            s1_asc_doys,
            s1_desc_bands,
            s1_desc_doys,
            target=s1_target,
            s1a_mean=self.s1a_band_mean,
            s1a_std=self.s1a_band_std,
            s1d_mean=self.s1d_band_mean,
            s1d_std=self.s1d_band_std,
        )

        return {
            "global_idxs": self._global_idxs[pixel_indices],
            "s2": s2,
            "s1": s1,
        }
