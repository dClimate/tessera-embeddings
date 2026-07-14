"""Zone-grid registry: 120 zones, shard-aligned, pinned to the EPSG registry."""

from __future__ import annotations

import numpy as np
import pytest

from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage import zone_grid as zg


def test_120_zones_named_by_epsg():
    assert len(zg.ZONES) == 120
    for utm in range(1, 61):
        assert f"326{utm:02d}" in zg.ZONES
        assert f"327{utm:02d}" in zg.ZONES
    n = zg.ZONES["32601"]
    assert n.hemisphere == "N" and n.utm_zone == 1
    assert zg.ZONES["32760"].hemisphere == "S" and zg.ZONES["32760"].utm_zone == 60
    assert n.group_name == "32601" and n.crs == "EPSG:32601"


@pytest.mark.parametrize("epsg", list(zg.ZONES))
def test_extents_pinned_to_pyproj(epsg):
    # The static table must match a fresh pyproj derivation, so an EPSG-database
    # change fails here instead of silently moving the grid.
    spec = zg.ZONES[epsg]
    easting, northing = zg.derive_extent(int(epsg))
    assert (easting, northing) == (spec.easting, spec.northing)


@pytest.mark.parametrize("epsg", ["32601", "32630", "32660", "32701", "32731", "32760"])
def test_shard_aligned(epsg):
    spec = zg.ZONES[epsg]
    assert spec.width % SHARD_PX == 0
    assert spec.height % SHARD_PX == 0


def test_coords_monotonic_and_sized():
    spec = zg.ZONES["32601"]
    east = zg.easting_coords(spec)
    north = zg.northing_coords(spec)
    assert east.shape == (spec.width,)
    assert north.shape == (spec.height,)
    assert np.all(np.diff(east) > 0)  # ascending
    assert np.all(np.diff(north) < 0)  # descending (row 0 = top)
    assert spec.easting[0] < east[0] < east[-1] < spec.easting[1]
    assert spec.northing[0] < north[-1] < north[0] < spec.northing[1]


def test_calendar_year_times():
    t = zg.calendar_year_times()
    assert t.dtype == np.dtype("datetime64[ns]")
    assert len(t) == 9
    assert str(t[0]).startswith("2017-01-01") and str(t[-1]).startswith("2025-01-01")


def test_northern_vs_southern_extents_differ_only_in_northing():
    n = zg.ZONES["32610"]
    s = zg.ZONES["32710"]
    assert n.easting == s.easting
    assert n.northing != s.northing  # N starts at 0; S is shifted for the false-northing offset
