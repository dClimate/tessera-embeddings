"""Deterministic bucketed resampling for Tessera v1.1 inference.

v1.1 replaces v1.0's random weighted sampling + repeat averaging with a
deterministic "use every valid observation" scheme: a pixel with ``k`` valid
observations is mapped to the next bucket size ``t`` in ``num_obs_checkpoints``
and resampled to exactly ``t`` rows. Downsizes pick the median element of each
even chunk; upsizes keep every original index and fill with evenly-spaced
duplicates.

Pixels sharing a ``(s2_bin, s1_bin)`` bucket are batched together so the
transformer sees rectangular sequences. Per-modality S1 normalisation is
applied BEFORE concatenating ascending and descending orbits into the merged
S1 stream.
"""

from __future__ import annotations

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
        s2_masks: ``(B, T)`` int32 binary validity mask from SCL.
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

    for i in range(b):
        mask = s2_masks[i]
        valid = np.nonzero(mask)[0]
        if len(valid) == 0:
            continue
        local = build_resample_indices(len(valid), target)
        real = valid[local]
        sub_b = s2_bands[i, real].astype(np.float32, copy=False)
        sub_b = (sub_b - s2_mean) / (s2_std + 1e-9)
        out[i, :, :c] = sub_b
        out[i, :, c] = s2_doys[i, real]
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

    Ascending and descending are each normalised with their OWN mean/std before
    concatenation — this matches v1.1 training preprocessing. A timestep is
    valid iff any band is non-zero (zero-padded SAR timesteps are skipped).

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

    for i in range(b):
        parts_b: list[np.ndarray] = []
        parts_d: list[np.ndarray] = []

        if s1_asc_bands.shape[1] > 0:
            stream = s1_asc_bands[i]
            valid = np.nonzero(np.any(stream != 0, axis=-1))[0]
            if len(valid) > 0:
                norm = (stream[valid].astype(np.float32, copy=False) - s1a_mean) / (s1a_std + 1e-9)
                parts_b.append(norm)
                parts_d.append(s1_asc_doys[i, valid].astype(np.float32, copy=False))

        if s1_desc_bands.shape[1] > 0:
            stream = s1_desc_bands[i]
            valid = np.nonzero(np.any(stream != 0, axis=-1))[0]
            if len(valid) > 0:
                norm = (stream[valid].astype(np.float32, copy=False) - s1d_mean) / (s1d_std + 1e-9)
                parts_b.append(norm)
                parts_d.append(s1_desc_doys[i, valid].astype(np.float32, copy=False))

        if not parts_b:
            continue

        all_b = np.concatenate(parts_b, axis=0)
        all_d = np.concatenate(parts_d, axis=0)
        local = build_resample_indices(len(all_b), target)
        out[i, :, :2] = all_b[local]
        out[i, :, 2] = all_d[local]
    return out
