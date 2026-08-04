"""A zone with no usable radar must ingest, not fail forever.

Some land has no dual-pol VV+VH radar in principle: over ice Sentinel-1 runs Extra Wide
swath with HH/HV polarisation, which the OPERA query correctly discards. Zone 23N's live land
is Greenland, and its 2021 catalogue holds ~45,000 ascending and ~138,000 descending granules
of which the ingest can use NONE. Requiring a SAR store there failed the cell permanently.

The asymmetry these tests pin is the point. The INGEST may resolve to no radar, because it has
just queried both orbits and its log says whether the source offered anything usable. A
CONSUMER reading a finished mosaic cannot tell a radar-free ROI from a lost orbit, so for
readers the absence stays an error.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.data_loading import (
    S1_ORBIT_NONE,
    _active_orbits,
    resolve_s1_orbit,
)


def test_none_activates_no_orbit() -> None:
    """The coverage gate builds its store list from this, so 'none' must yield nothing."""
    assert _active_orbits(S1_ORBIT_NONE) == ()


def test_none_is_not_an_accepted_request() -> None:
    """Nobody asks for 'none' — it is only ever a resolved outcome."""
    with pytest.raises(ValueError, match="Invalid s1_orbit"):
        _active_orbits("neither")


def test_a_reader_still_fails_when_no_radar_store_exists(tmp_path) -> None:
    """Default behaviour is unchanged: silently embedding without radar is the old bug."""
    base = f"file://{tmp_path}/mosaics/23N/2021"
    with pytest.raises(InsufficientCoverageError, match="no SAR stores found"):
        resolve_s1_orbit(base, "both")


def test_the_ingest_may_resolve_to_no_radar(tmp_path, caplog) -> None:
    """Opt-in, and it must say so loudly enough to be found in a fleet-wide log."""
    base = f"file://{tmp_path}/mosaics/23N/2021"
    with caplog.at_level("WARNING"):
        assert resolve_s1_orbit(base, "both", allow_none=True) == S1_ORBIT_NONE
    assert "NO SAR store exists" in caplog.text
    assert base in caplog.text, "the warning must name the mosaic it applies to"


@pytest.mark.parametrize("orbit", ["ascending", "descending"])
def test_a_named_orbit_is_never_downgraded(tmp_path, orbit: str) -> None:
    """An operator who named one orbit asked for something specific; allow_none must not apply."""
    base = f"file://{tmp_path}/mosaics/23N/2021"
    assert resolve_s1_orbit(base, orbit, allow_none=True) == orbit


class TestAbandoningAnUnusableOrbit:
    """Leading empty batches abandon an orbit; a mid-window gap must not.

    Zone 23N paginated 182,542 granules across a year, one 30-day batch at a time, to write
    nothing — every granule is HH/HV Extra Wide over ice and dual-pol VV+VH is required. The
    saving measured on that zone is 190s of catalogue querying per zone-year down to 59s.

    The distinction the threshold turns on is LEADING versus consecutive-anywhere. A zone imaged
    only in summer has a run of empty batches in January, so a rule that fired on any run of
    zeros would abandon it before reaching the data. Counting only while the running total is
    still zero makes that unreachable, and these tests pin exactly that.
    """

    def test_no_item_count_threshold_exists(self) -> None:
        """The rule is deliberately absent: no threshold on item counts alone is safe.

        Pinned as a test so a future attempt has to confront the seasonal case below rather
        than rediscover it in production, where the loss is silent.
        """
        import tessera_embeddings.ingest.s1_roi as mod

        assert not hasattr(mod, "LEADING_EMPTY_BATCHES_BEFORE_SKIP")

    @staticmethod
    def _consume(items_per_batch: list[int], threshold: int) -> int:
        """Batches actually consumed under the leading-empty rule."""
        total_seen = 0
        leading_empty = 0
        for i, seen in enumerate(items_per_batch, start=1):
            total_seen += seen
            if seen == 0 and total_seen == 0:
                leading_empty += 1
                if leading_empty >= threshold:
                    return i
        return len(items_per_batch)

    def test_a_threshold_would_have_saved_the_ice_case(self) -> None:
        """What the optimisation is worth: 3 batches instead of 12 on a permanently empty orbit."""
        assert self._consume([0] * 12, 3) == 3

    def test_why_that_threshold_is_unsafe(self) -> None:
        """A summer-only zone: the same rule abandons it in March and loses every summer date.

        This is the test that stopped the optimisation shipping. Leading emptiness does not
        distinguish "this terrain is imaged in the wrong mode" from "this window has no
        acquisitions yet", so only the polarisation skip count can gate it.
        """
        summer_only = [0, 0, 0, 0, 0, 900, 900, 900, 0, 0, 0, 0]
        assert self._consume(summer_only, 3) == 3, "fires in March"
        assert sum(summer_only[3:]) == 2700, "and 2,700 usable items are never queried"

    def test_a_mid_window_gap_would_have_been_safe(self) -> None:
        """The one case the leading-only formulation did get right, kept for the eventual fix."""
        assert self._consume([500] + [0] * 11, 3) == 12
