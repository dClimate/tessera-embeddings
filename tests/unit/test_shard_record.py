"""The per-shard record's invariants — the ones that make its numbers safe to divide.

Every figure a reader derives from this row is a ratio of two of its counts, so the tests are about
the denominators and the partition rather than about the plumbing: a row whose parts do not sum to
its eligible total produces shares that are wrong in a way nothing downstream checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera_embeddings.inference import shard_record

SIDE = shard_record.SHARD_PX


def _masks(side: int = 64):
    """A square footprint where everything is eligible and everything is embedded."""
    eligible = np.ones((side, side), dtype=bool)
    embedded = np.ones((side, side), dtype=bool)
    empty = np.zeros((side, side), dtype=bool)
    obs = np.full((side, side), 40, dtype=np.uint16)
    return eligible, embedded, empty.copy(), empty.copy(), empty.copy(), obs


def _build(**overrides):
    eligible, embedded, thin, no_optical, no_radar, obs = _masks()
    kwargs = dict(
        zone="33N",
        year=2024,
        tile_row=3,
        tile_col=4,
        eligible=eligible,
        embedded=embedded,
        s2_obs=obs,
        refused_thin=thin,
        refused_no_optical=no_optical,
        refused_no_radar=no_radar,
        optical_min_obs=25,
    )
    kwargs.update(overrides)
    return shard_record.build(**kwargs)


class TestTheDenominator:
    """``n_eligible_px`` is the only honest divisor, and the parts have to reach it."""

    def test_ocean_is_not_counted_as_refused(self):
        """A live shard includes the mask's sea buffer, so a coastal shard can be majority water.
        Counting that water anywhere would make every share a measure of coastline.
        """
        eligible, embedded, thin, no_optical, no_radar, obs = _masks()
        eligible[:, 32:] = False  # half the shard is sea
        embedded[:, 32:] = False
        row = _build(eligible=eligible, embedded=embedded)
        assert row.n_eligible_px == 64 * 32
        assert row.n_embedded_px == 64 * 32

    def test_the_parts_must_sum_to_the_eligible_total(self):
        """The check exists because the refusal masks come from the inference dataset and the
        eligibility mask from the land mask — two modules, and nothing else compares them.
        """
        eligible, embedded, thin, no_optical, no_radar, obs = _masks()
        embedded[:8, :] = False  # eight rows embedded nowhere and refused for no stated reason
        with pytest.raises(ValueError, match="eligibility mask and the refusal masks disagree"):
            _build(embedded=embedded)

    def test_a_correctly_partitioned_shard_builds(self):
        eligible, embedded, thin, no_optical, no_radar, obs = _masks()
        embedded[:8, :] = False
        thin[:5, :] = True
        no_optical[5:7, :] = True
        no_radar[7:8, :] = True
        row = _build(embedded=embedded, refused_thin=thin, refused_no_optical=no_optical, refused_no_radar=no_radar)
        assert (
            row.n_embedded_px + row.n_refused_thin_px + row.n_refused_no_optical_px + row.n_refused_no_radar_px
            == row.n_eligible_px
        )


class TestDepthStatistics:
    """Which population the depth figures describe."""

    def test_depth_describes_the_eligible_pixels_not_the_embedded_ones(self):
        """Over embedded pixels the statistics would describe only the population already known to
        clear the line, and the record exists to say what the refused ones looked like.
        """
        eligible, embedded, thin, no_optical, no_radar, obs = _masks()
        embedded[:32, :] = False
        thin[:32, :] = True
        obs[:32, :] = 5  # the refused half is thin
        obs[32:, :] = 45
        row = _build(embedded=embedded, refused_thin=thin, s2_obs=obs)
        assert row.s2_obs_mean == pytest.approx(25.0)  # both halves, not just the kept one
        assert row.s2_obs_p10 == pytest.approx(5.0)


class TestTheDepthHistogram:
    """What makes a later top-up plannable from the row instead of from the store."""

    def test_the_bins_sum_to_the_thin_refusals(self):
        """The invariant that makes the histogram checkable: it counts the thin refusals and only
        those, so a zero-observation pixel appearing in it would double-count a refusal.
        """
        eligible, embedded, thin, no_optical, no_radar, obs = _masks()
        embedded[:] = False
        thin[:] = True
        obs[:] = np.tile(np.array([1, 7, 12, 18, 23, 40, 3, 9], dtype=np.uint16), (64, 8))
        row = _build(embedded=embedded, refused_thin=thin, s2_obs=obs)
        assert sum(row.refused_depth_hist) == row.n_refused_thin_px

    def test_zero_observation_pixels_are_not_in_the_histogram(self):
        depths = np.array([0, 0, 3, 12], dtype=np.uint16)
        assert sum(shard_record.depth_histogram(depths)) == 2

    def test_the_bins_separate_nearly_rescued_from_hopeless(self):
        """The whole reason the histogram exists: a shard whose refusals sit just under the line is
        rescued by a small backfill, one whose refusals sit near zero is not, and a mean of the two
        is the same number.
        """
        nearly = shard_record.depth_histogram(np.array([23, 24, 22], dtype=np.uint16))
        hopeless = shard_record.depth_histogram(np.array([1, 2, 3], dtype=np.uint16))
        assert nearly != hopeless
        assert nearly[-2] == 3 and hopeless[0] == 3


class TestTheChunkBitmask:
    """Chunk-level detail in eight bytes, and what counts as skipped."""

    def test_one_fully_refused_chunk_sets_exactly_one_bit_in_the_right_place(self):
        side = shard_record.CHUNKS_PER_EDGE * 8
        eligible = np.ones((side, side), dtype=bool)
        refused = np.zeros((side, side), dtype=bool)
        refused[8:16, 16:24] = True  # chunk (1, 2) => bit 1*8 + 2
        mask, n_eligible, n_skipped = shard_record.chunk_skip_mask(refused, eligible)
        assert mask == 1 << 10
        assert (n_eligible, n_skipped) == (shard_record.CHUNKS_PER_SHARD, 1)

    def test_a_partly_refused_chunk_sets_no_bit(self):
        """The bit means the whole chunk is gone, because that is the granularity at which a hole
        is visible in the product and at which a top-up would rewrite.
        """
        side = shard_record.CHUNKS_PER_EDGE * 8
        eligible = np.ones((side, side), dtype=bool)
        refused = np.zeros((side, side), dtype=bool)
        refused[8:16, 16:20] = True  # half of chunk (1, 2)
        mask, _, n_skipped = shard_record.chunk_skip_mask(refused, eligible)
        assert (mask, n_skipped) == (0, 0)

    def test_a_chunk_with_no_land_is_not_skipped(self):
        """Otherwise the count of skipped chunks becomes a measure of how much ocean the shard
        touches, which is the same error as using 2048 squared as a denominator.
        """
        side = shard_record.CHUNKS_PER_EDGE * 8
        eligible = np.ones((side, side), dtype=bool)
        eligible[:8, :8] = False  # chunk (0,0) is all sea
        refused = np.zeros((side, side), dtype=bool)
        mask, n_eligible, n_skipped = shard_record.chunk_skip_mask(refused, eligible)
        assert (mask, n_skipped) == (0, 0)
        assert n_eligible == shard_record.CHUNKS_PER_SHARD - 1

    def test_the_top_bit_survives_the_row_encoding(self):
        """Parquet's integers are signed, so the highest chunk's bit would round-trip negative if
        it were stored as a number. It is stored as fixed-width hex.
        """
        side = shard_record.CHUNKS_PER_EDGE * 8
        eligible = np.ones((side, side), dtype=bool)
        refused = np.zeros((side, side), dtype=bool)
        refused[-8:, -8:] = True  # the last chunk => bit 63
        mask, _, _ = shard_record.chunk_skip_mask(refused, eligible)
        assert mask == 1 << 63
        row = _build(
            eligible=eligible,
            embedded=~refused & eligible,
            refused_thin=refused,
            s2_obs=np.full((side, side), 40, dtype=np.uint16),
        ).as_row()
        assert row["chunks_skipped_mask"] == "8000000000000000"


class TestTheRuleTravelsWithTheRow:
    """A row is only comparable to another row produced under the same line."""

    def test_the_row_records_which_line_produced_it(self):
        """Two rows written under different lines are not comparable, and the only thing that can
        say so is the row itself.
        """
        assert _build(optical_min_obs=25).optical_min_obs == 25
        assert _build(optical_min_obs=None).optical_min_obs is None

    def test_no_percentages_are_stored(self):
        """Counts only: a stored share is a second copy of a truth that drifts from its numerator."""
        row = _build().as_row()
        assert not [k for k in row if "pct" in k or "percent" in k or "share" in k]
