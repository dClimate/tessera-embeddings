"""``auto_batch_dates`` — the size threshold that decides date batching.

Pinned as a THRESHOLD rather than a curve on purpose: the measured benefit is not
monotonic in ROI size (it wins at the smallest sizes, regresses in the middle, and
is neutral at the largest), so the constant marks the upper edge of the range where
batching was measured to win. These tests therefore assert the shape of the rule and
that the boundary is inclusive — not any interpolated relationship.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.config.ingest import (
    AUTO_BATCH_DATES,
    AUTO_BATCH_DATES_MAX_COVERED_CHUNKS,
    auto_batch_dates,
)

THRESHOLD = AUTO_BATCH_DATES_MAX_COVERED_CHUNKS


def test_batches_at_and_below_the_threshold() -> None:
    """The boundary is inclusive, so a run exactly at it still batches."""
    assert auto_batch_dates(THRESHOLD) == AUTO_BATCH_DATES
    assert auto_batch_dates(THRESHOLD - 1) == AUTO_BATCH_DATES


def test_does_not_batch_above_the_threshold() -> None:
    """One chunk over is enough to fall back to one commit per date."""
    assert auto_batch_dates(THRESHOLD + 1) == 1


@pytest.mark.parametrize("covered", [0, 1, 2, 42])
def test_tiny_rois_batch(covered: int) -> None:
    """A near-empty ROI is where the commit dominates a date, so it batches."""
    assert auto_batch_dates(covered) == AUTO_BATCH_DATES


@pytest.mark.parametrize("covered", [930, 2631, 100_000])
def test_large_rois_do_not_batch(covered: int) -> None:
    """Mid and large ROIs get 1: this is where batching was measured to cost."""
    assert auto_batch_dates(covered) == 1


def test_never_returns_less_than_one() -> None:
    """The result feeds ``batch_dates``, whose contract is >= 1 at every size."""
    assert all(auto_batch_dates(c) >= 1 for c in (0, 1, THRESHOLD, THRESHOLD + 1, 10**6))


def test_batch_size_is_the_measured_one() -> None:
    """Guards against drifting to an unmeasured k without a new measurement."""
    assert AUTO_BATCH_DATES == 4
