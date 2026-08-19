"""Land-mask v1.1 → per-zone coverage bitmaps (ADR-010)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import rasterio
import zarr
from rasterio.transform import from_bounds

from tessera_embeddings.config.store_layout import INNER_PX, SHARD_PX
from tessera_embeddings.ingest import land_mask
from tessera_embeddings.ingest.live_windows import live_chunk_grid_from_keys
from tessera_embeddings.ingest.roi import read_roi_metadata
from tessera_embeddings.storage import zone_grid
from tessera_embeddings.storage.zarr_store import open_or_create_repo, open_store_as_zarr_group
from tests.unit.coverage_repo import make_coverage


# --------------------------------------------------------------------------- #
# Pure geometry
# --------------------------------------------------------------------------- #
def test_parse_cell_name_handles_negatives() -> None:
    assert land_mask.parse_cell_name("grid_-0.05_10.05.tiff") == (-0.05, 10.05)
    assert land_mask.parse_cell_name("grid_-0.05_-16.05.tiff") == (-0.05, -16.05)
    assert land_mask.parse_cell_name("grid_179.95_-59.45.tiff") == (179.95, -59.45)


@pytest.mark.parametrize("bad", ["red.tiff", "grid_1.tiff", "grid_1_2_3.tiff", "grid_1_2.png"])
def test_parse_cell_name_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        land_mask.parse_cell_name(bad)


@pytest.mark.parametrize(
    "bad",
    [
        "grid_nan_10.05.tiff",  # non-finite
        "grid_200.05_10.05.tiff",  # lon out of range
        "grid_-0.05_95.05.tiff",  # lat out of range
        "grid_2.4_48.85.tiff",  # lon off the 0.1° cell-centre lattice (2.4*20=48, even)
        "grid_-0.05_10.0.tiff",  # lat on a cell EDGE, not a centre
    ],
)
def test_parse_cell_name_rejects_invalid_coords(bad: str) -> None:
    with pytest.raises(ValueError):
        land_mask.parse_cell_name(bad)


def test_zone_for_cell_matches_delivery_samples() -> None:
    # Verified against the partner sample TIFFs' embedded CRS.
    assert land_mask.zone_for_cell(-0.05, 10.05) == "30N"
    assert land_mask.zone_for_cell(-0.05, -16.05) == "30S"
    # Band extremes and the equator sign split.
    assert land_mask.zone_for_cell(-179.95, 0.05) == "01N"
    assert land_mask.zone_for_cell(179.95, 0.05) == "60N"
    assert land_mask.zone_for_cell(0.05, 0.05) == "31N"
    assert land_mask.zone_for_cell(0.05, -0.05) == "31S"


def test_project_cells_to_pixel_boxes_is_on_grid() -> None:
    spec = zone_grid.zone("31N")
    # A cell near the central meridian, mid-latitude.
    r0, r1, c0, c1 = land_mask.project_cells_to_pixel_boxes(np.array([2.35]), np.array([48.85]), spec)
    assert 0 <= r0[0] < r1[0] <= spec.height
    assert 0 <= c0[0] < c1[0] <= spec.width
    # Roughly one 0.1° cell: ~1113 px in latitude, ~730 px in longitude at
    # 48.85°N — a sanity band, not the whole zone and not empty.
    assert 1000 <= (r1[0] - r0[0]) <= 1200
    assert 600 <= (c1[0] - c0[0]) <= 900


# --------------------------------------------------------------------------- #
# Bitmap build
# --------------------------------------------------------------------------- #
def test_build_zone_coverage_consistency_and_or() -> None:
    spec = zone_grid.zone("31N")
    lons = np.array([2.35, 2.45])
    lats = np.array([48.85, 48.85])
    cov = land_mask.build_zone_coverage("31N", lons, lats)

    assert cov.tile_live.shape == (spec.height // SHARD_PX, spec.width // SHARD_PX)
    assert cov.chunk_live.shape == (spec.height // INNER_PX, spec.width // INNER_PX)
    assert cov.n_cells == 2
    assert cov.n_live_tiles >= 1
    # The invariant the runner + validation rely on.
    ratio = SHARD_PX // INNER_PX
    assert np.array_equal(cov.tile_live, land_mask._block_any(cov.chunk_live, ratio))


def test_build_zone_coverage_empty_zone_is_all_false() -> None:
    cov = land_mask.build_zone_coverage("01N", np.empty(0), np.empty(0))
    assert cov.n_cells == 0
    assert not cov.tile_live.any()
    assert not cov.chunk_live.any()


def _write_registry(tmp_path, names: list[str]) -> str:
    reg = tmp_path / "registry.txt"
    reg.write_text("".join(f"{n} 0000\n" for n in names))
    return str(reg)


def _geo_zones() -> list[str]:
    return sorted({land_mask.zone_for_cell(lon, lat) for lon, lat, *_ in land_mask._GEO_CHECKS})


def _land_cell_names() -> list[str]:
    return [f"grid_{lon}_{lat}.tiff" for lon, lat, live, _ in land_mask._GEO_CHECKS if live]


def test_build_all_and_validate(tmp_path) -> None:
    """Build the geo-check zones from a synthetic registry; validate end-to-end.

    Restricting to the geo-check zones means validation's known land points are
    live (their cells are in the registry) and the ocean point is dead (its zone
    has no cells) — exercising the geographic self-check for real.
    """
    names = _land_cell_names()
    registry = _write_registry(tmp_path, names)
    zones = _geo_zones()
    dest = str(tmp_path / "coverage.icechunk")

    result = land_mask.build_all(dest, registry_uri=registry, delivery_uri="s3://x/y/", zones=zones)
    assert result.n_zones == len(zones)
    assert result.zones_with_cells == len(names)
    assert result.n_cells == len(names)
    assert result.n_live_tiles >= len(names)

    for z in zones:
        spec = zone_grid.zone(z)
        node = open_store_as_zarr_group(dest, group=z)
        assert node.attrs["zone"] == z
        assert node.attrs["crs"] == spec.crs
        assert list(node.attrs["grid_shape"]) == [spec.height, spec.width]
        assert node.attrs["tile_px"] == SHARD_PX
        assert node.attrs["source"] == "s3://x/y/"
        assert node.attrs["registry_sha256"] == result.registry_sha256
        assert np.asarray(node["tile_live_2048"]).shape == (spec.height // SHARD_PX, spec.width // SHARD_PX)

    # Structural + geographic self-checks (land live, ocean dead).
    land_mask.validate_coverage(dest, zones=zones)


def test_build_all_rebuild_is_a_new_commit(tmp_path) -> None:
    registry = _write_registry(tmp_path, _land_cell_names())
    zones = _geo_zones()
    dest = str(tmp_path / "coverage.icechunk")

    first = land_mask.build_all(dest, registry_uri=registry, zones=zones)
    second = land_mask.build_all(dest, registry_uri=registry, zones=zones)
    assert first.snapshot_id != second.snapshot_id
    repo, _ = open_or_create_repo(dest)
    ancestry = [c.id for c in repo.ancestry(branch="main")]
    assert first.snapshot_id in ancestry and second.snapshot_id in ancestry


def test_validate_rejects_inconsistent_bitmaps(tmp_path) -> None:
    z = "01N"
    spec = zone_grid.zone(z)
    dest = str(tmp_path / "cov.icechunk")
    repo, _ = open_or_create_repo(dest)
    session = repo.writable_session("main")
    node = zarr.open_group(session.store, mode="a").require_group(z)
    tile_live = np.zeros((spec.height // SHARD_PX, spec.width // SHARD_PX), dtype=bool)
    chunk_live = np.zeros((spec.height // INNER_PX, spec.width // INNER_PX), dtype=bool)
    chunk_live[0, 0] = True  # a live inner chunk with no corresponding live tile
    node.create_array(
        "tile_live_2048", data=tile_live, chunks=tile_live.shape, dimension_names=("tile_row", "tile_col")
    )
    node.create_array(
        "chunk_live_256", data=chunk_live, chunks=chunk_live.shape, dimension_names=("chunk_row", "chunk_col")
    )
    node.attrs.update({"zone": z, "crs": spec.crs, "grid_shape": [spec.height, spec.width], "n_cells": 0})
    session.commit("inconsistent")

    with pytest.raises(ValueError, match="block-any"):
        land_mask.validate_coverage(dest, zones=[z])


# --------------------------------------------------------------------------- #
# Delivery spot-check (guards the all-1s assumption)
# --------------------------------------------------------------------------- #
def _write_tile(
    path: str, lon: float, lat: float, value: int, crs: str | None = None, east_shift_m: float = 0.0
) -> None:
    spec = zone_grid.zone(land_mask.zone_for_cell(lon, lat))
    west, south, east, north = land_mask._expected_cell_bounds(lon, lat, spec)
    h = w = 64
    arr = np.full((h, w), value, dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="uint8",
        crs=crs or spec.crs,
        transform=from_bounds(west, south, east + east_shift_m, north, w, h),
    ) as ds:
        ds.write(arr, 1)


def test_spot_check_delivery_passes_on_valid_tiles(tmp_path) -> None:
    names = ["grid_2.35_48.85.tiff", "grid_2.45_48.85.tiff"]
    for n in names:
        lon, lat = land_mask.parse_cell_name(n)
        _write_tile(str(tmp_path / n), lon, lat, value=1)
    result = land_mask.spot_check_delivery(names, delivery_uri=str(tmp_path), n=10)
    assert result.checked == 2
    assert result.all_ones == result.crs_ok == result.bounds_ok == 2


def test_spot_check_delivery_rejects_non_all_ones(tmp_path) -> None:
    n = "grid_2.35_48.85.tiff"
    lon, lat = land_mask.parse_cell_name(n)
    _write_tile(str(tmp_path / n), lon, lat, value=0)  # v1-era all-zero tile
    with pytest.raises(ValueError, match="all-1s"):
        land_mask.spot_check_delivery([n], delivery_uri=str(tmp_path), n=10)


def test_spot_check_delivery_rejects_wrong_crs(tmp_path) -> None:
    n = "grid_2.35_48.85.tiff"
    lon, lat = land_mask.parse_cell_name(n)
    _write_tile(str(tmp_path / n), lon, lat, value=1, crs="EPSG:32601")  # not the cell's zone
    with pytest.raises(ValueError, match="CRS"):
        land_mask.spot_check_delivery([n], delivery_uri=str(tmp_path), n=10)


def test_spot_check_delivery_rejects_shifted_right_edge(tmp_path) -> None:
    # Correct west/top but a wrong width (right edge off by 100 m) must fail —
    # the check compares all four bounds, not only left/top.
    n = "grid_2.35_48.85.tiff"
    lon, lat = land_mask.parse_cell_name(n)
    _write_tile(str(tmp_path / n), lon, lat, value=1, east_shift_m=100.0)
    with pytest.raises(ValueError, match="bounds"):
        land_mask.spot_check_delivery([n], delivery_uri=str(tmp_path), n=10)


# --------------------------------------------------------------------------- #
# Zone ROI synthesis (export_zone_roi)
# --------------------------------------------------------------------------- #
def test_export_zone_roi_ocean_returns_none(tmp_path) -> None:
    cov = make_coverage(tmp_path, "01N", [])
    assert land_mask.export_zone_roi("01N", land_mask_path=cov, dest_path=str(tmp_path / "roi.zarr")) is None


def test_export_zone_roi_removes_a_superseded_mask_when_a_zone_turns_all_ocean(tmp_path) -> None:
    """A new delivery that removes a zone's land must not leave the old ROI readable.

    The ROI path is derived from the zone name, so ingest callers resolve it directly and
    never see this function's `None`. Left in place, a superseded delivery's land would go on
    being ingested while the current coverage bitmap declares there is none — the stale file
    winning purely by existing.
    """
    dest = tmp_path / "zone_01N.zarr"
    # A real ROI from a delivery that DID have land, written to the path 01N would use.
    land_mask.export_zone_roi("31N", land_mask_path=make_coverage(tmp_path, "31N", [(0, 0)]), dest_path=str(dest))
    assert dest.exists(), "fixture failed: nothing to supersede"

    # The next delivery finds no land in the zone at that path.
    assert (
        land_mask.export_zone_roi("01N", land_mask_path=make_coverage(tmp_path, "01N", []), dest_path=str(dest)) is None
    )
    assert not dest.exists(), "the superseded ROI is still discoverable by a direct ingest caller"


def test_export_zone_roi_roundtrip_matches_zone_grid(tmp_path) -> None:
    """The synthesized ROI reconstructs to the EXACT zone grid via the real
    consumer (read_roi_metadata) — the acceptance test the fill relies on.
    """
    zone = "31N"
    spec = zone_grid.zone(zone)
    cov = make_coverage(tmp_path, zone, [(10, 5), (11, 5)])
    dest = str(tmp_path / "zone_31N.zarr")
    assert land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest) == dest

    meta = read_roi_metadata(dest)  # the real ingest consumer
    assert meta.native_crs == spec.crs
    assert (meta.height, meta.width) == (spec.height, spec.width)
    # The stored transform reconstructs the zone's pixel-center coordinates.
    a, _, c, _, e, f = zarr.open(dest, mode="r").attrs["transform"]
    east = c + (np.arange(spec.width) + 0.5) * a
    north = f + (np.arange(spec.height) + 0.5) * e
    np.testing.assert_allclose([east[0], east[-1]], zone_grid.easting_coords(spec)[[0, -1]])
    np.testing.assert_allclose([north[0], north[-1]], zone_grid.northing_coords(spec)[[0, -1]])


def test_export_zone_roi_mask_upsamples_live_tiles(tmp_path) -> None:
    zone = "31N"
    cov = make_coverage(tmp_path, zone, [(10, 5)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    z = zarr.open(dest, mode="r")
    live = z[10 * SHARD_PX : 11 * SHARD_PX, 5 * SHARD_PX : 6 * SHARD_PX]
    dead = z[10 * SHARD_PX : 11 * SHARD_PX, 6 * SHARD_PX : 7 * SHARD_PX]
    assert bool(np.asarray(live).all())  # live tile upsampled to all-True pixels
    assert not bool(np.asarray(dead).any())  # neighbouring dead tile stays fill


def test_export_zone_roi_bbox_contains_live_tiles(tmp_path) -> None:
    zone = "31N"
    spec = zone_grid.zone(zone)
    cov = make_coverage(tmp_path, zone, [(10, 5), (11, 5)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    minx, miny, maxx, maxy = zarr.open(dest, mode="r").attrs["bbox_wgs84"]
    # The centre of live tile (10, 5) projected to WGS84 must lie inside the bbox.
    e = spec.easting[0] + (5 * SHARD_PX + SHARD_PX / 2) * land_mask.PIXEL_M
    n = spec.northing[1] - (10 * SHARD_PX + SHARD_PX / 2) * land_mask.PIXEL_M
    lon, lat = zone_grid.to_wgs84(int(spec.epsg)).transform(e, n)
    assert minx <= lon <= maxx
    assert miny <= lat <= maxy


def test_export_zone_roi_idempotent(tmp_path) -> None:
    zone = "31N"
    cov = make_coverage(tmp_path, zone, [(10, 5)])
    dest = str(tmp_path / "roi.zarr")
    first = land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    second = land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)  # matching grid -> skip
    assert first == second == dest
    z = zarr.open(dest, mode="r")
    assert bool(np.asarray(z[10 * SHARD_PX : 11 * SHARD_PX, 5 * SHARD_PX : 6 * SHARD_PX]).all())


def test_export_zone_roi_rebuilds_after_partial_write(tmp_path) -> None:
    """The coverage-sha discriminator is stamped LAST, so an ROI missing it (a
    crash before the final attr write) is rebuilt rather than trusted: idempotency
    must never skip a partially-written array.
    """
    zone = "31N"
    cov = make_coverage(tmp_path, zone, [(10, 5)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    # Simulate a crash that landed shape/CRS/transform but not the final sha.
    z = zarr.open(dest, mode="a")
    del z.attrs["coverage_sha256"]
    assert "coverage_sha256" not in zarr.open(dest, mode="r").attrs
    # The retry must NOT short-circuit — it rebuilds and re-stamps the sha.
    assert land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest) == dest
    assert zarr.open(dest, mode="r").attrs.get("coverage_sha256") is not None


def test_export_zone_roi_bbox_handles_antimeridian(tmp_path) -> None:
    """A live-tile strip spanning zone 01N's ±180 edge yields a NARROW bbox, not
    the near-global box naive min/max would give (which would span ~360°).
    """
    zone = "01N"
    spec = zone_grid.zone(zone)
    row = spec.height // SHARD_PX // 2  # a mid-latitude tile row
    # A west→east strip that crosses the -180 boundary (west cols wrap to +179,
    # east cols stay near -176).
    cov = make_coverage(tmp_path, zone, [(row, c) for c in range(8)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    minx, miny, maxx, maxy = zarr.open(dest, mode="r").attrs["bbox_wgs84"]
    # Crossing → GeoJSON west>east; else a normal box. Either way the effective
    # longitude span is a single 6° zone's width, never ~360°.
    span = (180.0 - minx) + (maxx + 180.0) if minx > maxx else maxx - minx
    assert span < 30.0


# --------------------------------------------------------------------------- #
# Pre-generation gate (live_chunk_count / validate_zone_roi)
# --------------------------------------------------------------------------- #
def test_live_chunk_count_equals_written_chunk_objects(tmp_path) -> None:
    """The invariant the placement check rests on: the export writes exactly one
    chunk object per live ingest chunk, so the coverage-derived count and the
    stored-object count are the same number.
    """
    zone = "31N"
    tiles_per_chunk = land_mask.INGEST_CHUNK_SIZE // SHARD_PX
    # Two tiles inside one chunk block, plus one in a different block: the count
    # must coarsen (3 tiles -> 2 chunks), not simply track tiles.
    cov = make_coverage(tmp_path, zone, [(0, 0), (0, 1), (4 * tiles_per_chunk, 3 * tiles_per_chunk)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)

    expected = land_mask.live_chunk_count(zone, land_mask_path=cov)
    assert expected == 2
    grid = live_chunk_grid_from_keys(dest, zarr.open(dest, mode="r"))
    assert grid is not None  # layout the ingest's cropping fast path requires
    assert int(grid.sum()) == expected


def test_validate_zone_roi_accepts_freshly_exported(tmp_path) -> None:
    zone = "31N"
    cov = make_coverage(tmp_path, zone, [(10, 5), (11, 5)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    assert land_mask.validate_zone_roi(zone, land_mask_path=cov, roi_path=dest) == []


def test_validate_zone_roi_catches_misplaced_land_at_an_equal_count(tmp_path) -> None:
    """Count-only validation is blind to the error this gate exists to catch.

    Swap one live chunk for a dead one and the totals still match, so a mask with
    land in the wrong part of the zone passes a sum check. Positions are compared
    against the coverage bitmap coarsened to ingest chunks.
    """
    zone = "31N"
    tiles_per_chunk = land_mask.INGEST_CHUNK_SIZE // SHARD_PX
    cov = make_coverage(tmp_path, zone, [(0, 0), (4 * tiles_per_chunk, 3 * tiles_per_chunk)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    assert land_mask.validate_zone_roi(zone, land_mask_path=cov, roi_path=dest) == []

    # Move one stored chunk object to a block the bitmap says is ocean: same count,
    # wrong ground position.
    root = pathlib.Path(dest) / "c"
    src_chunk = next(f for f in root.rglob("*") if f.is_file() and f.parent.name != "0")
    moved = root / "9" / "9"
    moved.parent.mkdir(parents=True, exist_ok=True)
    src_chunk.rename(moved)

    problems = land_mask.validate_zone_roi(zone, land_mask_path=cov, roi_path=dest)
    assert any("misplaced" in p for p in problems), problems


def test_validate_zone_roi_reports_absent_mask(tmp_path) -> None:
    cov = make_coverage(tmp_path, "31N", [(10, 5)])
    problems = land_mask.validate_zone_roi("31N", land_mask_path=cov, roi_path=str(tmp_path / "missing.zarr"))
    assert len(problems) == 1
    assert "cannot open" in problems[0]


def test_validate_zone_roi_rejects_wrong_transform(tmp_path) -> None:
    """A mask on the right grid shape but the wrong origin passes every structural
    check and writes data to the wrong ground position — so the transform is
    compared, not just the shape.
    """
    zone = "31N"
    cov = make_coverage(tmp_path, zone, [(10, 5)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    z = zarr.open(dest, mode="a")
    shifted = list(z.attrs["transform"])
    shifted[2] += land_mask.PIXEL_M  # origin off by one pixel east
    z.attrs["transform"] = shifted
    problems = land_mask.validate_zone_roi(zone, land_mask_path=cov, roi_path=dest)
    assert any("transform" in p for p in problems)


def test_validate_zone_roi_rejects_superseded_coverage(tmp_path) -> None:
    """A mask built from a previous land-mask delivery is not valid for the
    current one, even though its grid is identical — the sha is the discriminator.
    """
    zone = "31N"
    cov = make_coverage(tmp_path, zone, [(10, 5)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    z = zarr.open(dest, mode="a")
    z.attrs["coverage_sha256"] = "sha-from-an-older-delivery"
    problems = land_mask.validate_zone_roi(zone, land_mask_path=cov, roi_path=dest)
    assert any("coverage_sha256" in p for p in problems)


def test_validate_zone_roi_detects_missing_chunk_object(tmp_path) -> None:
    """The placement check has teeth: losing one chunk leaves a mask whose grid,
    CRS, transform and sha all still validate, but which no longer marks all the
    land the coverage bitmap does.
    """
    zone = "31N"
    tiles_per_chunk = land_mask.INGEST_CHUNK_SIZE // SHARD_PX
    cov = make_coverage(tmp_path, zone, [(0, 0), (4 * tiles_per_chunk, 3 * tiles_per_chunk)])
    dest = str(tmp_path / "roi.zarr")
    land_mask.export_zone_roi(zone, land_mask_path=cov, dest_path=dest)
    assert land_mask.validate_zone_roi(zone, land_mask_path=cov, roi_path=dest) == []

    chunks = sorted(pathlib.Path(dest).glob("c/*/*"))
    assert len(chunks) == 2
    chunks[0].unlink()
    problems = land_mask.validate_zone_roi(zone, land_mask_path=cov, roi_path=dest)
    assert any("stored chunk" in p for p in problems)
