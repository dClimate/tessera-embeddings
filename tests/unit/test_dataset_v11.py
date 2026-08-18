"""Unit tests for v1.1 bucketed MosaicChunkInferenceDataset."""

from __future__ import annotations

import numpy as np
import pytest

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


class TestTheMinimumOpticalDepthGate:
    """A pixel below the line is not embedded at all, and the three refusal reasons stay apart.

    This is the one gate in the pipeline whose effect a consumer cannot undo: a filtered pixel can
    be unfiltered, whereas a refused pixel has no embedding and recovering it means re-running its
    whole shard. So these pin the boundary exactly, and pin that the off state is bit-identical to
    the historical behaviour rather than merely similar to it.
    """

    @staticmethod
    def _chunk(obs_counts: np.ndarray) -> ChunkData:
        """Real reflectance everywhere, with per-pixel optical depth supplied directly.

        ``s2_obs_count`` is the pre-pruning count the loader passes when it has one, and it is what
        the gate reads — so a depth of 400 needs no 400 fabricated timesteps.
        """
        h, w = obs_counts.shape
        return ChunkData(
            s2_bands=np.ones((1, h, w, 10), dtype=np.uint16),
            s2_masks=np.ones((1, h, w), dtype=bool),
            s1_asc_bands=np.ones((1, h, w, 2), dtype=np.float32),
            s1_desc_bands=np.ones((1, h, w, 2), dtype=np.float32),
            s2_doys=np.array([100], dtype=np.int32),
            s1_asc_doys=np.array([100], dtype=np.int32),
            s1_desc_doys=np.array([100], dtype=np.int32),
            height=h,
            width=w,
            s2_obs_count=obs_counts.astype(np.uint16),
        )

    def test_the_boundary_keeps_the_line_and_refuses_below_it(self):
        """29 out at a line of 30, 30 in. An off-by-one here silently changes what a petabyte
        contains, and the plan states the arithmetic once, as at-least.
        """
        ds = MosaicChunkInferenceDataset(self._chunk(np.array([[29, 30]])), allow_s2_only=True, optical_min_obs=30)
        assert len(ds) == 1
        assert ds.refused_thin == 1

    def test_none_embeds_everything_with_any_optical_input(self):
        chunk = self._chunk(np.array([[1, 5, 29, 100]]))
        ds = MosaicChunkInferenceDataset(chunk, allow_s2_only=True, optical_min_obs=None)
        assert len(ds) == 4
        assert ds.refused_thin == 0

    def test_the_off_state_is_identical_to_no_gate_at_all(self):
        """What protects every non-campaign caller: None must select exactly the pixels the
        historical code selected, not almost those.
        """
        chunk = self._chunk(np.array([[1, 14, 15, 400]]))
        without = MosaicChunkInferenceDataset(chunk, allow_s2_only=True)
        explicit = MosaicChunkInferenceDataset(chunk, allow_s2_only=True, optical_min_obs=None)
        assert np.array_equal(without._global_idxs, explicit._global_idxs)

    def test_zero_is_refused_rather_than_treated_as_off(self):
        """Zero refuses nothing while presenting as a configured rule, so a campaign whose value
        resolved to zero would publish under no rule while believing it had one.
        """
        with pytest.raises(ValueError, match="refuses nothing"):
            MosaicChunkInferenceDataset(self._chunk(np.array([[10]])), optical_min_obs=0)

    def test_the_two_optical_reasons_partition_rather_than_overlap(self):
        """No optical input at all is a fact about the imagery; too little of it is this
        campaign's quality rule. Counting the first in both would overrun the shard's eligible
        total, and the per-shard record's invariant is that the parts sum to it.
        """
        chunk = self._chunk(np.array([[0, 0, 5, 40]]))
        chunk.s2_bands[:, 0, 0:2, :] = 0  # no reflectance where the count is zero
        ds = MosaicChunkInferenceDataset(chunk, allow_s2_only=True, optical_min_obs=30)
        assert (ds.refused_no_optical, ds.refused_thin, len(ds)) == (2, 1, 1)

    def test_a_radar_refusal_counts_only_pixels_that_passed_the_optical_test(self):
        """Otherwise a thin pixel with no radar lands in two counters at once."""
        chunk = self._chunk(np.array([[5, 40]]))
        chunk.s1_asc_bands[:] = 0.0
        chunk.s1_desc_bands[:] = 0.0
        ds = MosaicChunkInferenceDataset(chunk, allow_s2_only=False, optical_min_obs=30)
        assert (ds.refused_thin, ds.refused_no_radar, len(ds)) == (1, 1, 0)
