"""Zone grids: both layouts, shard-aligned, pinned to the EPSG registry (ADR 037)."""

from __future__ import annotations

import numpy as np
import pytest

from tessera_embeddings.config.fault_injection import _CANONICAL_ZONE
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage import zone_grid as zg
from tessera_embeddings.storage.time_axis import calendar_year_times

_ALL = [spec for layout in ("hemispheres", "utm_zones") for spec in zg.zones(layout).values()]


def test_zone_tables_by_layout():
    hemi, cols = zg.zones("hemispheres"), zg.zones("utm_zones")
    assert sorted(hemi) == sorted(f"{u:02d}{h}" for u in range(1, 61) for h in "NS")
    assert sorted(cols) == [f"utm{u:02d}" for u in range(1, 61)]
    assert hemi["60S"].epsg == "32760" and hemi["01N"].crs == "EPSG:32601"
    # A utm_zones column is in its NORTHERN CRS, whichever hemisphere it reaches.
    assert cols["utm60"].epsg == "32660" and cols["utm60"].hemisphere is None
    assert zg.zone("utm33") is cols["utm33"] and zg.zone("33S") is hemi["33S"]


@pytest.mark.parametrize("spec", _ALL, ids=lambda s: s.group_name)
def test_extents_pinned_to_pyproj(spec):
    # The static tables must match a fresh pyproj derivation, so an EPSG-database
    # change fails here instead of silently moving the grid.
    also = (int(spec.epsg) + 100,) if spec.hemisphere is None else ()
    assert zg.derive_extent(int(spec.epsg), *also) == (spec.easting, spec.northing)
    assert spec.width % SHARD_PX == 0 and spec.height % SHARD_PX == 0


def test_column_shares_pixels_not_shards_with_the_southern_zone():
    # Why a utm_zones store needs its own mask and mosaics: the column and the S zone agree on
    # every pixel (10,000 km is whole pixels) but not on shard boundaries.
    col, south, north = zg.zone("utm10"), zg.zone("10S"), zg.zone("10N")
    offset = col.northing[1] - (south.northing[1] - 10_000_000.0)
    assert offset % zg.PIXEL_M == 0 and offset % zg.PITCH_M != 0
    assert col.northing[1] == north.northing[1] and col.easting == north.easting


def test_zone_layout_of():
    assert zg.zone_layout_of(["01N", "60S", "not-a-zone"]) == "hemispheres"
    assert zg.zone_layout_of(["utm01"]) == "utm_zones"
    assert zg.zone_layout_of([]) is None
    with pytest.raises(ValueError, match="mix"):
        zg.zone_layout_of(["01N", "utm01"])


def test_coords_monotonic_and_sized():
    spec = zg.zone("01N")
    east = zg.easting_coords(spec)
    north = zg.northing_coords(spec)
    assert east.shape == (spec.width,)
    assert north.shape == (spec.height,)
    assert np.all(np.diff(east) > 0)  # ascending
    assert np.all(np.diff(north) < 0)  # descending (row 0 = top)
    assert spec.easting[0] < east[0] < east[-1] < spec.easting[1]
    assert spec.northing[0] < north[-1] < north[0] < spec.northing[1]


def test_calendar_year_times():
    t = calendar_year_times()
    assert t.dtype == np.dtype("datetime64[ns]")
    assert len(t) == 9
    assert str(t[0]).startswith("2017-01-01") and str(t[-1]).startswith("2025-01-01")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("33N", "33N"),
        ("15S", "15S"),
        ("1N", "01N"),  # zero-padded
        ("01N", "01N"),
        ("60S", "60S"),
        ("33n", "33N"),  # case-insensitive hemisphere
        (" 7s ", "07S"),  # surrounding whitespace + zero-pad
        ("utm33", "utm33"),
        ("UTM7", "utm07"),
    ],
)
def test_canonicalize_zone(raw, expected):
    name = zg.canonicalize_zone(raw)
    assert name == expected
    zg.zone(name)  # every parse maps to a real zone
    assert _CANONICAL_ZONE.fullmatch(name)  # the fault drill's copy of this format agrees


@pytest.mark.parametrize(
    "bad", ["", "33", "N33", "0N", "61N", "33X", "33NN", "abc", "-5N", "utm", "utm0", "utm61", "utm33N"]
)
def test_canonicalize_zone_rejects_malformed(bad):
    with pytest.raises(ValueError, match="UTM zone"):
        zg.canonicalize_zone(bad)
