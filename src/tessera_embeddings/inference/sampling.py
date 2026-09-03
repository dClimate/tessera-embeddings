"""Deterministic bucketed resampling for Tessera v1.1 inference.

A deterministic "use every valid observation" scheme: a pixel with ``k`` valid observations is
mapped to the next bucket size ``t`` in ``num_obs_checkpoints`` and resampled to exactly ``t``
rows. Downsizes pick the median element of each even chunk; upsizes keep every original index and
fill with evenly-spaced duplicates.

Pixels sharing a ``(s2_bin, s1_bin)`` bucket are batched together so the transformer sees
rectangular sequences. Per-modality S1 normalisation is applied BEFORE concatenating ascending
and descending orbits into the merged S1 stream.
"""

from __future__ import annotations

import functools

import numpy as np


def build_resample_indices(valid_len: int, target_size: int) -> np.ndarray:
    """Deterministic index vector resampling ``valid_len`` rows to ``target_size``.

    - ``target_size == valid_len``: identity.
    - ``target_size < valid_len``: median index of each evenly-split chunk.
    - ``target_size > valid_len``: keep every original index, fill extras at
      evenly-spaced positions (duplicates allowed).

    Matches the tessera v1.1 implementation (``ssl_dataset_v1_1.py``).
    """
    if valid_len <= 0:
        return np.zeros((0,), dtype=np.int64)
    if target_size == valid_len:
        return np.arange(valid_len, dtype=np.int64)
    if target_size < valid_len:
        chunks = np.array_split(np.arange(valid_len), target_size)
        return np.array([c[len(c) // 2] for c in chunks if len(c) > 0], dtype=np.int64)
    extra = target_size - valid_len
    anchors = np.linspace(0, valid_len - 1, num=extra + 2, dtype=np.float64)[1:-1]
    extras = np.clip(np.rint(anchors).astype(np.int64), 0, valid_len - 1)
    return np.concatenate([np.arange(valid_len, dtype=np.int64), extras], axis=0)


# build_resample_indices depends only on (valid_len, target), both small ints (valid_len <= T of
# the chunk, target one of ~32 checkpoints), so the index vectors are memoised. Callers must treat
# the returned arrays as READ-ONLY: they are shared across every pixel with the same pair.
_resample_indices_cached = functools.lru_cache(maxsize=None)(build_resample_indices)


def _local_resample_matrix(counts: np.ndarray, target: int) -> np.ndarray:
    """Stack per-pixel local resample indices into a ``(B, target)`` matrix.

    ``counts[i]`` is pixel *i*'s valid-observation count; row *i* of the result is
    ``build_resample_indices(counts[i], target)``. Rows with ``counts == 0`` are left as zeros —
    callers must exclude them from any gather.

    The Python-level work is one cache lookup per *unique* count (a handful per bucket) rather
    than per pixel, which is what makes the bucket resamplers vectorised instead of
    2 x batch_size Python iterations per sub-batch.
    """
    local = np.zeros((len(counts), target), dtype=np.int64)
    for count in np.unique(counts):
        if count == 0:
            continue
        local[counts == count] = _resample_indices_cached(int(count), target)
    return local


def _row_starts(counts: np.ndarray) -> np.ndarray:
    """Offsets of each row's first entry in a row-major ``np.nonzero`` output."""
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return starts[:-1]


def bucket_for_count(n: int, ckps: tuple[int, ...]) -> int:
    """Return the smallest checkpoint >= *n*, clamped to the largest checkpoint.

    ``ckps`` must be sorted ascending. ``n <= 0`` also returns ``ckps[0]``.
    """
    for c in ckps:
        if n <= c:
            return c
    return ckps[-1]


def compute_bin_keys(
    s2_valid: np.ndarray,
    s1_valid: np.ndarray,
    ckps: tuple[int, ...],
) -> np.ndarray:
    """Vectorised per-pixel bucket assignment.

    Args:
        s2_valid: ``(N,)`` S2 valid-observation counts.
        s1_valid: ``(N,)`` S1 (asc + desc) valid-observation counts.
        ckps: Sorted checkpoint sizes.

    Returns:
        Structured array of shape ``(N,)`` with fields ``s2`` and ``s1`` (int32).
    """
    ckps_arr = np.asarray(ckps, dtype=np.int32)
    s2_idx = np.searchsorted(ckps_arr, np.clip(s2_valid, 1, None), side="left")
    s1_idx = np.searchsorted(ckps_arr, np.clip(s1_valid, 1, None), side="left")
    s2_idx = np.clip(s2_idx, 0, len(ckps_arr) - 1)
    s1_idx = np.clip(s1_idx, 0, len(ckps_arr) - 1)
    keys = np.empty(len(s2_valid), dtype=np.dtype([("s2", np.int32), ("s1", np.int32)]))
    keys["s2"] = ckps_arr[s2_idx]
    keys["s1"] = ckps_arr[s1_idx]
    return keys


def resample_s2_bucket(
    s2_bands: np.ndarray,
    s2_masks: np.ndarray,
    s2_doys: np.ndarray,
    target: int,
    s2_mean: np.ndarray,
    s2_std: np.ndarray,
) -> np.ndarray:
    """Resample a batch of pixels' S2 observations to a uniform length.

    Args:
        s2_bands: ``(B, T, 10)`` uint16 bands.
        s2_masks: ``(B, T)`` bool binary validity mask from SCL.
        s2_doys: ``(B, T)`` int32 day-of-year, broadcast across pixels.
        target: Target sequence length (bucket size).
        s2_mean: ``(10,)`` per-band mean.
        s2_std: ``(10,)`` per-band std.

    Returns:
        ``(B, target, 11)`` float32, standardised bands + DOY as last column.
    """
    b, t, c = s2_bands.shape
    out = np.zeros((b, target, c + 1), dtype=np.float32)
    if t == 0 or target == 0:
        return out

    # Vectorised across pixels: np.nonzero on the (B, T) mask yields each row's valid time indices
    # contiguously in row-major order, _row_starts locates each row's slice, and the memoised
    # local-index matrix maps every pixel straight to its absolute gather indices. Bit-identical
    # to a per-pixel loop (same indices, same per-element arithmetic) — golden reference test in
    # test_sampling_v11.py.
    counts = s2_masks.sum(axis=1).astype(np.int64)
    rows = np.nonzero(counts > 0)[0]
    if rows.size == 0:
        return out
    _, nz_cols = np.nonzero(s2_masks)
    starts = _row_starts(counts)
    local = _local_resample_matrix(counts, target)

    real = nz_cols[starts[rows, None] + local[rows]]  # (Bv, target) absolute time idx
    gathered = s2_bands[rows[:, None], real].astype(np.float32, copy=False)
    out[rows, :, :c] = (gathered - s2_mean) / (s2_std + 1e-9)
    out[rows, :, c] = s2_doys[rows[:, None], real]
    return out


def resample_s1_bucket(
    s1_asc_bands: np.ndarray,
    s1_asc_doys: np.ndarray,
    s1_desc_bands: np.ndarray,
    s1_desc_doys: np.ndarray,
    target: int,
    s1a_mean: np.ndarray,
    s1a_std: np.ndarray,
    s1d_mean: np.ndarray,
    s1d_std: np.ndarray,
) -> np.ndarray:
    """Resample a batch of pixels' merged S1 observations to a uniform length.

    Ascending and descending are each normalised with their OWN mean/std before concatenation,
    matching v1.1 training preprocessing. A timestep is valid iff any band is non-zero, so
    zero-padded SAR timesteps are skipped.

    Args:
        s1_asc_bands: ``(B, T_asc, 2)`` uint16.
        s1_asc_doys: ``(B, T_asc)`` int32, broadcast across pixels.
        s1_desc_bands: ``(B, T_desc, 2)`` uint16.
        s1_desc_doys: ``(B, T_desc)`` int32, broadcast across pixels.
        target: Target sequence length (bucket size).
        s1a_mean: ``(2,)`` ascending-orbit mean.
        s1a_std: ``(2,)`` ascending-orbit std.
        s1d_mean: ``(2,)`` descending-orbit mean.
        s1d_std: ``(2,)`` descending-orbit std.

    Returns:
        ``(B, target, 3)`` float32, standardised bands + DOY as last column.
    """
    b = s1_asc_bands.shape[0] if s1_asc_bands.shape[0] > 0 else s1_desc_bands.shape[0]
    out = np.zeros((b, target, 3), dtype=np.float32)
    if target == 0 or b == 0:
        return out

    # Vectorised over pixels. Each pixel's merged stream is its valid ascending rows followed by
    # its valid descending rows; a merged local index below ``ca`` (the pixel's asc count)
    # resolves into the ascending arrays and the rest into descending, so the two sides are
    # gathered separately and combined with np.where. Per-element arithmetic matches a per-pixel
    # loop (normalise-then-gather == gather-then-normalise elementwise) — golden reference test
    # in test_sampling_v11.py.
    valid_a = np.any(s1_asc_bands != 0, axis=-1) if s1_asc_bands.shape[1] > 0 else np.zeros((b, 0), dtype=bool)
    valid_d = np.any(s1_desc_bands != 0, axis=-1) if s1_desc_bands.shape[1] > 0 else np.zeros((b, 0), dtype=bool)
    ca = valid_a.sum(axis=1).astype(np.int64)
    cd = valid_d.sum(axis=1).astype(np.int64)
    total = ca + cd
    rows = np.nonzero(total > 0)[0]
    if rows.size == 0:
        return out

    local = _local_resample_matrix(total, target)[rows]  # (Bv, target) merged idx
    ca_v = ca[rows, None]
    from_asc = local < ca_v

    # Masked-out lanes get a guarded index of 0 BEFORE the gather so every
    # index is in-bounds; their gathered values are discarded by np.where.
    def _gather_side(
        valid: np.ndarray,
        bands: np.ndarray,
        doys: np.ndarray,
        side_local: np.ndarray,
        lanes: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        _, nz_cols = np.nonzero(valid)
        if nz_cols.size == 0:
            return (
                np.zeros((rows.size, target, 2), dtype=np.float32),
                np.zeros((rows.size, target), dtype=np.float32),
            )
        counts = valid.sum(axis=1).astype(np.int64)
        idx = np.where(lanes, _row_starts(counts)[rows, None] + side_local, 0)
        t_idx = nz_cols[idx]  # (Bv, target) absolute time idx; garbage where ~lanes
        vals = bands[rows[:, None], t_idx].astype(np.float32, copy=False)
        norm = (vals - mean) / (std + 1e-9)
        return norm, doys[rows[:, None], t_idx].astype(np.float32, copy=False)

    a_norm, a_doys = _gather_side(valid_a, s1_asc_bands, s1_asc_doys, local, from_asc, s1a_mean, s1a_std)
    d_norm, d_doys = _gather_side(valid_d, s1_desc_bands, s1_desc_doys, local - ca_v, ~from_asc, s1d_mean, s1d_std)

    out[rows, :, :2] = np.where(from_asc[..., None], a_norm, d_norm)
    out[rows, :, 2] = np.where(from_asc, a_doys, d_doys)
    return out
