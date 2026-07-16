"""Unit tests for v1.1 deterministic bucketed sampling (sampling.py)."""

from __future__ import annotations

import numpy as np

from tessera_embeddings.inference.sampling import (
    bucket_for_count,
    build_resample_indices,
    compute_bin_keys,
    resample_s1_bucket,
    resample_s2_bucket,
)

CKPS = tuple(range(8, 33, 8))  # (8, 16, 24, 32) — small for tests


# ── build_resample_indices ──


def test_build_resample_indices_identity() -> None:
    idx = build_resample_indices(5, 5)
    np.testing.assert_array_equal(idx, [0, 1, 2, 3, 4])


def test_build_resample_indices_zero_len() -> None:
    idx = build_resample_indices(0, 4)
    assert idx.shape == (0,)


def test_build_resample_indices_downsample() -> None:
    idx = build_resample_indices(10, 5)
    assert idx.shape == (5,)
    assert all(0 <= i < 10 for i in idx)
    # All indices must be distinct when downsampling
    assert len(set(idx.tolist())) == 5


def test_build_resample_indices_upsample() -> None:
    idx = build_resample_indices(3, 8)
    assert idx.shape == (8,)
    # Original indices must be present
    assert 0 in idx and 1 in idx and 2 in idx


# ── bucket_for_count ──


def test_bucket_for_count_exact_match() -> None:
    assert bucket_for_count(8, CKPS) == 8


def test_bucket_for_count_between() -> None:
    assert bucket_for_count(9, CKPS) == 16


def test_bucket_for_count_above_max_clamps() -> None:
    assert bucket_for_count(100, CKPS) == 32


def test_bucket_for_count_zero_returns_first() -> None:
    assert bucket_for_count(0, CKPS) == 8


def test_bucket_for_count_negative_returns_first() -> None:
    assert bucket_for_count(-1, CKPS) == 8


# ── compute_bin_keys ──


def test_compute_bin_keys_shape_and_dtype() -> None:
    s2 = np.array([3, 8, 25], dtype=np.int32)
    s1 = np.array([0, 16, 32], dtype=np.int32)
    keys = compute_bin_keys(s2, s1, CKPS)
    assert keys.shape == (3,)
    assert keys.dtype.names == ("s2", "s1")


def test_compute_bin_keys_values() -> None:
    s2 = np.array([3, 8, 25], dtype=np.int32)
    s1 = np.array([0, 16, 32], dtype=np.int32)
    keys = compute_bin_keys(s2, s1, CKPS)
    # n=3 → bucket 8, n=8 → bucket 8, n=25 → bucket 32
    np.testing.assert_array_equal(keys["s2"], [8, 8, 32])
    # n=0 → bucket 8, n=16 → bucket 16, n=32 → bucket 32
    np.testing.assert_array_equal(keys["s1"], [8, 16, 32])


# ── resample_s2_bucket ──


def test_resample_s2_bucket_output_shape() -> None:
    rng = np.random.default_rng(0)
    b, t, c = 4, 20, 10
    bands = rng.integers(0, 3000, (b, t, c), dtype=np.uint16)
    masks = np.ones((b, t), dtype=np.int32)
    doys = np.tile(np.arange(t, dtype=np.int32), (b, 1))
    mean = np.zeros(c, dtype=np.float32)
    std = np.ones(c, dtype=np.float32)

    out = resample_s2_bucket(bands, masks, doys, target=16, s2_mean=mean, s2_std=std)
    assert out.shape == (b, 16, c + 1)
    assert out.dtype == np.float32


def test_resample_s2_bucket_no_valid_pixels_returns_zeros() -> None:
    b, t, c = 2, 10, 10
    bands = np.zeros((b, t, c), dtype=np.uint16)
    masks = np.zeros((b, t), dtype=np.int32)
    doys = np.zeros((b, t), dtype=np.int32)
    mean = np.zeros(c, dtype=np.float32)
    std = np.ones(c, dtype=np.float32)

    out = resample_s2_bucket(bands, masks, doys, target=8, s2_mean=mean, s2_std=std)
    np.testing.assert_array_equal(out, np.zeros_like(out))


# ── resample_s1_bucket ──


def test_resample_s1_bucket_both_orbits_output_shape() -> None:
    rng = np.random.default_rng(1)
    b, t = 4, 12
    asc_bands = rng.standard_normal((b, t, 2)).astype(np.float32)
    asc_doys = np.tile(np.arange(t, dtype=np.int32), (b, 1))
    desc_bands = rng.standard_normal((b, t, 2)).astype(np.float32)
    desc_doys = np.tile(np.arange(t, dtype=np.int32), (b, 1))
    mean2 = np.zeros(2, dtype=np.float32)
    std2 = np.ones(2, dtype=np.float32)

    out = resample_s1_bucket(
        asc_bands,
        asc_doys,
        desc_bands,
        desc_doys,
        target=8,
        s1a_mean=mean2,
        s1a_std=std2,
        s1d_mean=mean2,
        s1d_std=std2,
    )
    # 2 asc bands + 1 DOY = 3 features
    assert out.shape == (b, 8, 3)
    assert out.dtype == np.float32


def test_resample_s1_bucket_single_orbit_empty_desc() -> None:
    rng = np.random.default_rng(2)
    b, t = 3, 8
    asc_bands = rng.standard_normal((b, t, 2)).astype(np.float32)
    asc_doys = np.tile(np.arange(t, dtype=np.int32), (b, 1))
    empty_bands = np.zeros((b, 0, 2), dtype=np.float32)
    empty_doys = np.zeros((b, 0), dtype=np.int32)
    mean2 = np.zeros(2, dtype=np.float32)
    std2 = np.ones(2, dtype=np.float32)

    out = resample_s1_bucket(
        asc_bands,
        asc_doys,
        empty_bands,
        empty_doys,
        target=8,
        s1a_mean=mean2,
        s1a_std=std2,
        s1d_mean=mean2,
        s1d_std=std2,
    )
    assert out.shape == (b, 8, 3)


# ── golden reference: vectorised resamplers == the original per-pixel loops ──
#
# The vectorised implementations must be BIT-IDENTICAL to the per-pixel loops
# they replaced (ADR 012: these are kept in the bit-exact class). The loops are
# frozen here as reference implementations.


def _reference_resample_s2_bucket(s2_bands, s2_masks, s2_doys, target, s2_mean, s2_std):
    b, t, c = s2_bands.shape
    out = np.zeros((b, target, c + 1), dtype=np.float32)
    if t == 0 or target == 0:
        return out
    for i in range(b):
        valid = np.nonzero(s2_masks[i])[0]
        if len(valid) == 0:
            continue
        real = valid[build_resample_indices(len(valid), target)]
        sub_b = s2_bands[i, real].astype(np.float32, copy=False)
        out[i, :, :c] = (sub_b - s2_mean) / (s2_std + 1e-9)
        out[i, :, c] = s2_doys[i, real]
    return out


def _reference_resample_s1_bucket(
    s1_asc_bands, s1_asc_doys, s1_desc_bands, s1_desc_doys, target, s1a_mean, s1a_std, s1d_mean, s1d_std
):
    b = s1_asc_bands.shape[0] if s1_asc_bands.shape[0] > 0 else s1_desc_bands.shape[0]
    out = np.zeros((b, target, 3), dtype=np.float32)
    if target == 0 or b == 0:
        return out
    for i in range(b):
        parts_b, parts_d = [], []
        for bands, doys, mean, std in (
            (s1_asc_bands, s1_asc_doys, s1a_mean, s1a_std),
            (s1_desc_bands, s1_desc_doys, s1d_mean, s1d_std),
        ):
            if bands.shape[1] > 0:
                stream = bands[i]
                valid = np.nonzero(np.any(stream != 0, axis=-1))[0]
                if len(valid) > 0:
                    parts_b.append((stream[valid].astype(np.float32, copy=False) - mean) / (std + 1e-9))
                    parts_d.append(doys[i, valid].astype(np.float32, copy=False))
        if not parts_b:
            continue
        all_b = np.concatenate(parts_b, axis=0)
        all_d = np.concatenate(parts_d, axis=0)
        local = build_resample_indices(len(all_b), target)
        out[i, :, :2] = all_b[local]
        out[i, :, 2] = all_d[local]
    return out


def _random_s2_case(rng, b, t, invalid_frac=0.3, dead_pixel_frac=0.2):
    bands = rng.integers(0, 6000, size=(b, t, 10)).astype(np.uint16)
    masks = rng.random((b, t)) > invalid_frac
    dead = rng.random(b) < dead_pixel_frac
    masks[dead] = False  # some pixels have zero valid observations
    doys = np.broadcast_to(rng.integers(1, 366, size=t).astype(np.int32)[None, :], (b, t))
    return bands, masks, doys


class TestVectorisedS2MatchesLoop:
    """Vectorised S2 resampler is bit-identical to the frozen per-pixel loop."""

    def test_bit_identical_across_targets(self) -> None:
        rng = np.random.default_rng(42)
        mean = np.array(np.arange(10) * 100.0, dtype=np.float32)
        std = np.array(np.arange(10) + 50.0, dtype=np.float32)
        for t, target in [(20, 8), (20, 20), (5, 16), (33, 24), (1, 8)]:
            bands, masks, doys = _random_s2_case(rng, b=64, t=t)
            got = resample_s2_bucket(bands, masks, doys, target, mean, std)
            want = _reference_resample_s2_bucket(bands, masks, doys, target, mean, std)
            np.testing.assert_array_equal(got, want, err_msg=f"t={t} target={target}")

    def test_all_pixels_dead(self) -> None:
        bands = np.zeros((4, 6, 10), dtype=np.uint16)
        masks = np.zeros((4, 6), dtype=bool)
        doys = np.zeros((4, 6), dtype=np.int32)
        mean = np.zeros(10, dtype=np.float32)
        std = np.ones(10, dtype=np.float32)
        got = resample_s2_bucket(bands, masks, doys, 8, mean, std)
        np.testing.assert_array_equal(got, np.zeros((4, 8, 11), dtype=np.float32))


class TestVectorisedS1MatchesLoop:
    """Vectorised merged-S1 resampler is bit-identical to the frozen loop."""

    def _stats(self):
        return (
            np.array([5000.0, 3000.0], dtype=np.float32),
            np.array([1700.0, 1600.0], dtype=np.float32),
            np.array([5100.0, 2900.0], dtype=np.float32),
            np.array([1650.0, 1700.0], dtype=np.float32),
        )

    def _random_orbit(self, rng, b, t, zero_frac=0.4):
        bands = rng.integers(1, 8000, size=(b, t, 2)).astype(np.uint16)
        zero = rng.random((b, t)) < zero_frac
        bands[zero] = 0  # zeroed timesteps = invalid for that pixel
        doys = np.broadcast_to(rng.integers(1, 366, size=t).astype(np.int32)[None, :], (b, t))
        return bands, doys

    def test_bit_identical_both_orbits(self) -> None:
        rng = np.random.default_rng(7)
        s1a_mean, s1a_std, s1d_mean, s1d_std = self._stats()
        for ta, td, target in [(6, 5, 8), (10, 10, 16), (3, 0, 8), (0, 4, 8), (12, 9, 8)]:
            asc, asc_doys = self._random_orbit(rng, 48, ta) if ta else (
                np.empty((48, 0, 2), dtype=np.uint16),
                np.empty((48, 0), dtype=np.int32),
            )
            desc, desc_doys = self._random_orbit(rng, 48, td) if td else (
                np.empty((48, 0, 2), dtype=np.uint16),
                np.empty((48, 0), dtype=np.int32),
            )
            got = resample_s1_bucket(asc, asc_doys, desc, desc_doys, target, s1a_mean, s1a_std, s1d_mean, s1d_std)
            want = _reference_resample_s1_bucket(
                asc, asc_doys, desc, desc_doys, target, s1a_mean, s1a_std, s1d_mean, s1d_std
            )
            np.testing.assert_array_equal(got, want, err_msg=f"ta={ta} td={td} target={target}")

    def test_pixels_with_all_zero_sar_stay_zero(self) -> None:
        s1a_mean, s1a_std, s1d_mean, s1d_std = self._stats()
        asc = np.zeros((3, 4, 2), dtype=np.uint16)
        desc = np.zeros((3, 4, 2), dtype=np.uint16)
        doys = np.zeros((3, 4), dtype=np.int32)
        got = resample_s1_bucket(asc, doys, desc, doys, 8, s1a_mean, s1a_std, s1d_mean, s1d_std)
        np.testing.assert_array_equal(got, np.zeros((3, 8, 3), dtype=np.float32))
