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
