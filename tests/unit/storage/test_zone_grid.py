"""Zone-grid registry: 120 zones, shard-aligned, pinned to the EPSG registry."""

from __future__ import annotations

import numpy as np
import pytest

from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage import zone_grid as zg
from tessera_embeddings.storage.time_axis import calendar_year_times


def test_120_zones_named_by_common_name():
    assert len(zg.ZONES) == 120
    for utm in range(1, 61):
        assert f"{utm:02d}N" in zg.ZONES
        assert f"{utm:02d}S" in zg.ZONES
    n = zg.ZONES["01N"]
    assert n.hemisphere == "N" and n.utm_zone == 1 and n.epsg == "32601"
    assert zg.ZONES["60S"].hemisphere == "S" and zg.ZONES["60S"].utm_zone == 60 and zg.ZONES["60S"].epsg == "32760"
    assert n.group_name == "01N" and n.crs == "EPSG:32601"


@pytest.mark.parametrize("name", list(zg.ZONES))
def test_extents_pinned_to_pyproj(name):
    # The static table must match a fresh pyproj derivation, so an EPSG-database
    # change fails here instead of silently moving the grid.
    spec = zg.ZONES[name]
    easting, northing = zg.derive_extent(int(spec.epsg))
    assert (easting, northing) == (spec.easting, spec.northing)


@pytest.mark.parametrize("name", ["01N", "30N", "60N", "01S", "31S", "60S"])
def test_shard_aligned(name):
    spec = zg.ZONES[name]
    assert spec.width % SHARD_PX == 0
    assert spec.height % SHARD_PX == 0


def test_coords_monotonic_and_sized():
    spec = zg.ZONES["01N"]
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


def test_northern_vs_southern_extents_differ_only_in_northing():
    n = zg.ZONES["10N"]
    s = zg.ZONES["10S"]
    assert n.easting == s.easting
    assert n.northing != s.northing  # N starts at 0; S is shifted for the false-northing offset


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
    ],
)
def test_canonicalize_zone(raw, expected):
    name = zg.canonicalize_zone(raw)
    assert name == expected
    assert name in zg.ZONES  # every parse maps to a real seeded zone


@pytest.mark.parametrize("bad", ["", "33", "N33", "0N", "61N", "33X", "33NN", "abc", "-5N"])
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
