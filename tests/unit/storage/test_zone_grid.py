"""Zone grids: both layouts, shard-aligned, pinned to the EPSG registry (ADR 037), filled by halves (ADR 038)."""

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


def test_every_extent_is_pinned_to_pyproj_and_shard_aligned():
    # The static tables must match a fresh pyproj derivation, so an EPSG-database
    # change fails here instead of silently moving the grid. A column spans both CRSs.
    for spec in _ALL:
        also = (int(spec.epsg) + 100,) if spec.hemisphere is None else ()
        assert zg.derive_extent(int(spec.epsg), *also) == (spec.easting, spec.northing), spec.group_name
        assert spec.width % SHARD_PX == 0 and spec.height % SHARD_PX == 0, spec.group_name


def test_column_shares_pixels_not_shards_with_the_southern_zone():
    # Why a utm_zones store needs its own mask and mosaics: the column and the S zone agree on
    # every pixel (10,000 km is whole pixels) but not on shard boundaries.
    col, south, north = zg.zone("utm10"), zg.zone("10S"), zg.zone("10N")
    offset = col.northing[1] - (south.northing[1] - 10_000_000.0)
    assert offset % zg.PIXEL_M == 0 and offset % zg.PITCH_M != 0
    assert col.northing[1] == north.northing[1] and col.easting == north.easting


def test_a_column_is_filled_as_two_halves_split_on_a_shard_row():
    # ADR 038: the equator (northing 0 in 326xx) falls on a shard boundary, so the halves own
    # disjoint shard rows that together are the column, each on the column's own grid.
    col, north, south = zg.zone("utm10"), zg.zone("utm10n"), zg.zone("utm10s")
    assert (north.name, north.group_name, south.name, south.group_name) == ("utm10n", "utm10", "utm10s", "utm10")
    assert (north.northing, north.easting, north.epsg) == (col.northing, col.easting, col.epsg)
    assert col.northing[1] % zg.PITCH_M == 0
    assert north.tile_rows == (0, south.tile_rows[0]) and south.tile_rows[1] == col.tile_rows[1]
    assert (
        zg.northing_coords(col)[north.tile_rows[1] * SHARD_PX]
        < 0
        < zg.northing_coords(col)[north.tile_rows[1] * SHARD_PX - 1]
    )
    assert sorted(zg.fill_zones("utm_zones")) == sorted(f"utm{u:02d}{h}" for u in range(1, 61) for h in "ns")
    assert zg.fill_zones("hemispheres") is zg.zones("hemispheres")
    assert [zg.group_and_half(n) for n in ("utm10s", "10S", "")] == [("utm10", "s"), ("10S", None), ("", None)]
    with pytest.raises(ValueError, match="filled as its halves"):
        zg.group_and_half("utm10")
    equator = north.tile_rows[1]
    assert [zg.fill_zones_of("utm10", r) for r in (None, equator - 1, equator)] == [
        ("utm10n", "utm10s"),
        ("utm10n",),
        ("utm10s",),
    ]
    assert zg.fill_zones_of("UTM10S") == ("utm10s",) and zg.fill_zones_of("10s", 0) == ("10S",)


def test_zone_layout_of():
    assert zg.zone_layout_of(["01N", "60S", "not-a-zone"]) == "hemispheres"
    assert zg.zone_layout_of(["utm01"]) == zg.zone_layout_of(["utm01s"]) == "utm_zones"
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
        ("1N", "01N"),  # zero-padded
        ("60S", "60S"),
        ("33n", "33N"),  # case-insensitive hemisphere
        (" 7s ", "07S"),  # surrounding whitespace + zero-pad
        ("utm33", "utm33"),
        ("UTM7", "utm07"),
        ("UTM33S", "utm33s"),  # a column half
    ],
)
def test_canonicalize_zone(raw, expected):
    name = zg.canonicalize_zone(raw)
    assert name == expected
    zg.zone(name)  # every parse maps to a real zone
    # The fault drill's copy of this format agrees, for the fill zones it can fire on (not a whole column).
    assert bool(_CANONICAL_ZONE.fullmatch(name)) == (zg.fill_zones_of(name) == (name,))


@pytest.mark.parametrize("bad", ["N33", "0N", "61N", "33X", "utm61", "utm33x", "utm33ns"])
def test_canonicalize_zone_rejects_malformed(bad):
    with pytest.raises(ValueError, match="UTM zone"):
        zg.canonicalize_zone(bad)


def test_a_tile_box_steps_by_one_shard_pitch_per_row() -> None:
    """The label's row/col index the SHARD grid — the same reading assembly uses to place a
    tile — so a box must step by one tile pitch per row, not by a pixel or by a chunk of some
    other size. A pitch error yields boxes that look plausible and describe the wrong ground.
    """
    spec = zg.zone("16S")
    top = zg.tile_range_bbox_wgs84(spec, 0, 1, 0, 1)
    next_row = zg.tile_range_bbox_wgs84(spec, 1, 2, 0, 1)
    deg_per_m = 1.0 / 111_320.0

    # One tile south, and by one tile PITCH: ~20.48 km of latitude per row.
    assert next_row[3] < top[3] and next_row[1] < top[1], "row 1 is south of row 0"
    assert abs((top[3] - top[1]) - SHARD_PX * zg.PIXEL_M * deg_per_m) < 0.01

    # Adjacent boxes OVERLAP rather than abut, and that is the containment guarantee showing
    # through: each is the WGS84 envelope of its own projected quad, so along the northing line
    # the two tiles share, one box takes that curve's maximum latitude and the other its
    # minimum. What must never happen is a GAP — land in neither box.
    assert next_row[3] >= top[1], "no gap between vertically adjacent tiles"
    assert (next_row[3] - top[1]) < 10.0 * deg_per_m, "and the overlap stays inside one pixel"
