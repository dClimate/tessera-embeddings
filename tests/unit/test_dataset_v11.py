"""Unit tests for v1.1 bucketed MosaicChunkInferenceDataset."""

from __future__ import annotations

import numpy as np

from tessera_embeddings.inference.data_loading import ChunkData
from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset

CKPS = (8, 16, 24, 32)

# ── helpers ──


def _make_chunk_data(
    H: int = 4,
    W: int = 4,
    n_s2: int = 12,
    n_s1a: int = 6,
    n_s1d: int = 0,
    *,
    rng: np.random.Generator | None = None,
) -> ChunkData:
    """Build a minimal synthetic ChunkData for testing."""
    if rng is None:
        rng = np.random.default_rng(42)

    s2_bands = rng.integers(100, 3000, (n_s2, H, W, 10), dtype=np.uint16)
    s2_masks = rng.integers(0, 2, (n_s2, H, W)).astype(bool)
    s2_doys = np.arange(n_s2, dtype=np.int32) * 16

    if n_s1a > 0:
        s1_asc_bands = rng.standard_normal((n_s1a, H, W, 2)).astype(np.float32)
        s1_asc_doys = np.arange(n_s1a, dtype=np.int32) * 30
    else:
        s1_asc_bands = np.empty((0, H, W, 2), dtype=np.float32)
        s1_asc_doys = np.empty((0,), dtype=np.int32)

    if n_s1d > 0:
        s1_desc_bands = rng.standard_normal((n_s1d, H, W, 2)).astype(np.float32)
        s1_desc_doys = np.arange(n_s1d, dtype=np.int32) * 30
    else:
        s1_desc_bands = np.empty((0, H, W, 2), dtype=np.float32)
        s1_desc_doys = np.empty((0,), dtype=np.int32)

    s2_obs_count = s2_masks.sum(axis=0).astype(np.uint16)

    return ChunkData(
        s2_bands=s2_bands,
        s2_masks=s2_masks,
        s2_doys=s2_doys,
        s1_asc_bands=s1_asc_bands,
        s1_asc_doys=s1_asc_doys,
        s1_desc_bands=s1_desc_bands,
        s1_desc_doys=s1_desc_doys,
        height=H,
        width=W,
        s2_obs_count=s2_obs_count,
    )


# ── tests ──


def test_bucket_assignment_valid_pixels() -> None:
    chunk_data = _make_chunk_data(H=4, W=4, n_s2=12, n_s1a=6)
    ds = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)
    total = sum(len(v) for v in ds._bucket_pixels.values())
    assert len(ds) == total
    assert len(ds) > 0


def test_iter_buckets_largest_first() -> None:
    chunk_data = _make_chunk_data(H=8, W=8, n_s2=20, n_s1a=8)
    ds = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)

    bucket_list = list(ds.iter_buckets(largest_first=True))
    if len(bucket_list) > 1:
        sizes = [len(idxs) for _, idxs in bucket_list]
        assert sizes == sorted(sizes, reverse=True)


def test_iter_buckets_keys_are_valid_checkpoints() -> None:
    chunk_data = _make_chunk_data(H=4, W=4, n_s2=12, n_s1a=6)
    ds = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)
    for (s2_bin, s1_bin), _ in ds.iter_buckets():
        assert s2_bin in CKPS
        assert s1_bin in CKPS


def test_get_bucket_batch_shapes() -> None:
    chunk_data = _make_chunk_data(H=4, W=4, n_s2=12, n_s1a=6)
    ds = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)

    for (s2_bin, s1_bin), idxs in ds.iter_buckets():
        batch = ds.get_bucket_batch((s2_bin, s1_bin), 0, min(4, len(idxs)))
        b = batch["s2"].shape[0]
        assert batch["s2"].shape == (b, s2_bin, 11)  # 10 bands + DOY
        assert batch["s1"].shape == (b, s1_bin, 3)  # 2 bands + DOY
        assert batch["global_idxs"].shape == (b,)
        break  # one bucket is enough for shape checks


def test_no_valid_pixels_empty_dataset() -> None:
    """A chunk where all S2 masks are zero has no valid pixels."""
    H, W = 4, 4
    chunk_data = ChunkData(
        s2_bands=np.zeros((10, H, W, 10), dtype=np.uint16),
        s2_masks=np.zeros((10, H, W), dtype=bool),
        s2_doys=np.arange(10, dtype=np.int32),
        s1_asc_bands=np.zeros((6, H, W, 2), dtype=np.float32),
        s1_asc_doys=np.arange(6, dtype=np.int32),
        s1_desc_bands=np.empty((0, H, W, 2), dtype=np.float32),
        s1_desc_doys=np.empty((0,), dtype=np.int32),
        height=H,
        width=W,
    )
    ds = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)
    assert len(ds) == 0
    assert list(ds.iter_buckets()) == []


def test_bucket_sizes_consistency() -> None:
    chunk_data = _make_chunk_data(H=6, W=6, n_s2=16, n_s1a=8)
    ds = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)

    sizes = ds.bucket_sizes()
    for key, count in sizes.items():
        assert len(ds._bucket_pixels[key]) == count


# ── allow_s2_only: the optional per-pixel S1 requirement ──


def test_s2_only_pixels_dropped_by_default() -> None:
    """Pins the historical gate: S2-valid pixels with ZERO S1 observations are
    skipped entirely when allow_s2_only is off (the default).
    """
    chunk_data = _make_chunk_data(H=4, W=4, n_s2=12, n_s1a=0, n_s1d=0)
    ds = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)
    assert len(ds) == 0
    assert list(ds.iter_buckets()) == []


def test_allow_s2_only_embeds_s1_empty_pixels_with_upstream_convention() -> None:
    """With allow_s2_only, S2-valid/S1-empty pixels are kept and receive the
    upstream v1.1 missing-S1 input: the SMALLEST S1 bucket and an all-zeros
    (normalized-space) S1 slice — exactly ucam-eo/tessera's
    ``_sample_s1_merged`` zero return.
    """
    chunk_data = _make_chunk_data(H=4, W=4, n_s2=12, n_s1a=0, n_s1d=0)
    ds = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS, allow_s2_only=True)
    assert len(ds) > 0  # every S2-valid pixel is now embedded
    for (s2_bin, s1_bin), idxs in ds.iter_buckets():
        assert s1_bin == CKPS[0]  # zero S1 count clips into the smallest bucket
        batch = ds.get_bucket_batch((s2_bin, s1_bin), 0, len(idxs))
        assert batch["s1"].shape[1:] == (CKPS[0], 3)
        np.testing.assert_array_equal(batch["s1"], 0.0)  # neutral all-zeros slice
        assert np.all(np.isfinite(batch["s2"]))


def test_allow_s2_only_mixed_chunk_adds_exactly_the_s1_gap_pixels() -> None:
    """A chunk with SAR over only part of its area: the flag adds exactly the
    S2-valid pixels inside the SAR gap, and keeps every previously-valid pixel
    (S1-informed pixels are unaffected).
    """
    chunk_data = _make_chunk_data(H=4, W=4, n_s2=12, n_s1a=6)
    # Kill SAR over the right half — an in-zone S1 coverage gap.
    chunk_data.s1_asc_bands[:, :, 2:, :] = 0.0

    ds_default = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS)
    ds_flag = MosaicChunkInferenceDataset(chunk_data, num_obs_checkpoints=CKPS, allow_s2_only=True)

    s2_ok = chunk_data.s2_obs_count > 0  # s2_bands are all non-zero in the fixture
    s1_ok = np.any(np.any(chunk_data.s1_asc_bands != 0, axis=-1), axis=0)
    assert len(ds_default) == int((s2_ok & s1_ok).sum())
    assert len(ds_flag) == int(s2_ok.sum())

    # The default set is a strict subset: nothing previously embedded changes.
    default_idxs = set(ds_default._global_idxs.tolist())
    flag_idxs = set(ds_flag._global_idxs.tolist())
    assert default_idxs < flag_idxs
    gap_rows, gap_cols = np.where(s2_ok & ~s1_ok)
    assert set((gap_rows * 4 + gap_cols).tolist()) == flag_idxs - default_idxs
