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
