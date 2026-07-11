"""Unit tests for the ROI grid-snap foundation (ingest/feature_grid.py).

The load-bearing property is **exact pixel-subset alignment**: a per-feature ROI
rasterized by ``rasterize_feature_roi`` must have coordinates that are an exact
subset of the master geobox's, because the downstream region write places data
positionally (``resolve_region`` → ``merge_feature_into_master``). We assert that
directly, plus the disjoint-overlap guard, the snap-window math, the full masked
ROI, and the ``classify_mask`` / ``classify_store`` triage.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr
import zarr
from affine import Affine
from odc.geo.geobox import GeoBox
from shapely.geometry import box

from tessera_embeddings.ingest.feature_grid import (
    CORRUPTED,
    MISSING,
    PRESENT,
    FeatureRecord,
    assert_features_disjoint,
    classify_mask,
    classify_store,
    feature_window,
    load_features,
    rasterize_feature_roi,
    rasterize_full_masked_roi,
)
from tessera_embeddings.ingest.roi import read_roi_metadata
from tessera_embeddings.storage.manifest import RoiManifest, extract_manifest
from tessera_embeddings.storage.zarr_store import write_dataset

_CRS = "EPSG:5070"
# A small master grid: 10 m, descending northing, origin on a whole-metre boundary.
_MASTER = GeoBox(shape=(200, 300), affine=Affine(10, 0, 500000, 0, -10, 4000000), crs=_CRS)


def _feature(fid: str, geom) -> FeatureRecord:
    """A FeatureRecord whose master-CRS and wgs84 geoms are the same (tests that
    don't reproject work directly in the master CRS).
    """
    return FeatureRecord(feature_id=fid, geometry=geom, geometry_wgs84=geom)


def _master_cells(rows: slice, cols: slice):
    """A box covering master cells [rows.start:rows.stop, cols.start:cols.stop]."""
    minx = 500000 + 10 * cols.start
    maxx = 500000 + 10 * cols.stop
    maxy = 4000000 - 10 * rows.start
    miny = 4000000 - 10 * rows.stop
    return box(minx, miny, maxx, maxy)


# --------------------------------------------------------------------------- #
# Snap-window math
# --------------------------------------------------------------------------- #


def test_window_snaps_outward_and_clamps():
    """A box over master cells [rows 30:50, cols 40:90] resolves to a sub-geobox of
    exactly that shape, aligned at that offset on the master grid.
    """
    sub = feature_window(_MASTER, _master_cells(slice(30, 50), slice(40, 90)))
    assert (sub.height, sub.width) == (20, 50)
    assert sub.affine == _MASTER[30:50, 40:90].affine


def test_window_clamps_to_master_extent():
    huge = box(0, 0, 10_000_000, 10_000_000)
    sub = feature_window(_MASTER, huge)
    assert (sub.height, sub.width) == (_MASTER.height, _MASTER.width)
    assert sub.affine == _MASTER.affine


def test_window_outside_master_raises():
    far = box(0, 0, 100, 100)  # west/south of the master origin entirely
    with pytest.raises(ValueError, match="outside the master grid"):
        feature_window(_MASTER, far)


# --------------------------------------------------------------------------- #
# Disjoint guard
# --------------------------------------------------------------------------- #


def test_disjoint_guard_passes_for_separated_boxes():
    assert_features_disjoint([_feature("a", box(0, 0, 10, 10)), _feature("b", box(20, 20, 30, 30))])


def test_disjoint_guard_allows_touching_borders():
    """Shared edges have zero overlap area — allowed."""
    assert_features_disjoint([_feature("a", box(0, 0, 10, 10)), _feature("b", box(10, 0, 20, 10))])


def test_disjoint_guard_rejects_overlap():
    feats = [_feature("a", box(0, 0, 10, 10)), _feature("b", box(5, 5, 15, 15))]
    with pytest.raises(ValueError, match="overlap"):
        assert_features_disjoint(feats)


# --------------------------------------------------------------------------- #
# Rasterization → exact pixel-subset of master (the load-bearing property)
# --------------------------------------------------------------------------- #


def test_feature_roi_is_exact_subset_of_master(tmp_path):
    feat = _feature("t1", _master_cells(slice(30, 50), slice(40, 90)))
    out = str(tmp_path / "feat.zarr")
    rasterize_feature_roi(out, master_geobox=_MASTER, feature=feat)

    fm = read_roi_metadata(out)
    assert (fm.width, fm.height) == (50, 20)
    assert fm.native_crs == _CRS

    mc, fc = _MASTER.coordinates, fm.geobox.coordinates
    mx, my = np.round(mc["x"].values, 3), np.round(mc["y"].values, 3)
    fx, fy = np.round(fc["x"].values, 3), np.round(fc["y"].values, 3)
    assert np.all(np.isin(fx, mx)), "feature easting not a subset of master easting"
    assert np.all(np.isin(fy, my)), "feature northing not a subset of master northing"
    assert np.isclose(fx[0], mx[40]) and np.isclose(fy[0], my[30])


def test_feature_roi_writes_compatible_attrs(tmp_path):
    """The mask carries the same attrs read_roi_metadata + the ingest path expect."""
    feat = _feature("t1", box(500400, 3999000, 500900, 3999500))
    out = str(tmp_path / "feat.zarr")
    rasterize_feature_roi(out, master_geobox=_MASTER, feature=feat)

    z = zarr.open(out, mode="r")
    assert z.dtype == np.dtype("bool")
    assert z.attrs["crs"] == _CRS
    assert len(z.attrs["transform"]) == 6
    assert len(z.attrs["bbox_wgs84"]) == 4
    assert RoiManifest.from_dict(extract_manifest(z.attrs)) == RoiManifest(resolution=10.0, chunk_size=4000, crs=_CRS)


# --------------------------------------------------------------------------- #
# Full masked ROI — features burned onto the master's full grid
# --------------------------------------------------------------------------- #


def test_full_masked_roi_marks_only_covered(tmp_path):
    feats = [
        _feature("a", _master_cells(slice(10, 30), slice(20, 50))),
        _feature("b", _master_cells(slice(100, 140), slice(200, 260))),
    ]
    out = str(tmp_path / "masked.zarr")
    rasterize_full_masked_roi(out, master_geobox=_MASTER, features=feats)

    fm = read_roi_metadata(out)
    assert (fm.width, fm.height) == (_MASTER.width, _MASTER.height)
    assert fm.geobox.affine == _MASTER.affine

    mask = zarr.open(out, mode="r")[:]
    assert mask.dtype == np.dtype("bool")
    assert mask[10:30, 20:50].all()
    assert mask[100:140, 200:260].all()
    assert not mask[60, 120]
    assert int(mask.sum()) == 20 * 30 + 40 * 60  # exactly the two disjoint footprints


def test_full_masked_roi_equals_union_of_feature_windows(tmp_path):
    feats = [
        _feature("a", _master_cells(slice(10, 30), slice(20, 50))),
        _feature("b", _master_cells(slice(100, 140), slice(200, 260))),
    ]
    full = str(tmp_path / "masked.zarr")
    rasterize_full_masked_roi(full, master_geobox=_MASTER, features=feats)
    full_mask = zarr.open(full, mode="r")[:]

    union = np.zeros((_MASTER.height, _MASTER.width), dtype=bool)
    for feat in feats:
        win = str(tmp_path / f"win_{feat.feature_id}.zarr")
        rasterize_feature_roi(win, master_geobox=_MASTER, feature=feat)
        yroi, xroi = _MASTER.overlap_roi(feature_window(_MASTER, feat.geometry))
        union[yroi, xroi] |= zarr.open(win, mode="r")[:]

    assert np.array_equal(full_mask, union)


def test_full_masked_roi_empty_features_raises(tmp_path):
    with pytest.raises(ValueError, match="features is empty"):
        rasterize_full_masked_roi(str(tmp_path / "x.zarr"), master_geobox=_MASTER, features=[])


# --------------------------------------------------------------------------- #
# classify_mask — the full-masked-ROI skip check
# --------------------------------------------------------------------------- #


def _write_mask(tmp_path, name="masked.zarr"):
    feats = [_feature("a", _master_cells(slice(10, 30), slice(20, 50)))]
    out = str(tmp_path / name)
    rasterize_full_masked_roi(out, master_geobox=_MASTER, features=feats)
    return out


def test_classify_mask_present_for_valid_mask(tmp_path):
    assert classify_mask(_write_mask(tmp_path), _MASTER).status == PRESENT


def test_classify_mask_missing_when_absent(tmp_path):
    assert classify_mask(str(tmp_path / "nope.zarr"), _MASTER).status == MISSING


def test_classify_mask_corrupted_on_shape_mismatch(tmp_path):
    out = _write_mask(tmp_path)
    bigger = GeoBox(shape=(201, 300), affine=_MASTER.affine, crs=_CRS)
    diag = classify_mask(out, bigger)
    assert diag.status == CORRUPTED
    assert "stale mask" in diag.detail


def test_classify_mask_corrupted_when_manifest_missing(tmp_path):
    out = _write_mask(tmp_path)
    z = zarr.open(out, mode="a")
    del z.attrs["_manifest"]
    diag = classify_mask(out, _MASTER)
    assert diag.status == CORRUPTED
    assert "_manifest" in diag.detail


def test_classify_mask_corrupted_on_wrong_dtype(tmp_path):
    out = str(tmp_path / "wrong.zarr")
    z = zarr.open(out, mode="w", shape=(_MASTER.height, _MASTER.width), chunks=(64, 64), dtype="uint8")
    z.attrs["_manifest"] = {}
    diag = classify_mask(out, _MASTER)
    assert diag.status == CORRUPTED
    assert "bool" in diag.detail


# --------------------------------------------------------------------------- #
# classify_store — the per-feature mosaic triage
# --------------------------------------------------------------------------- #


def _write_mosaic(path: str, data_vars: dict[str, str]):
    """Write a tiny (time, northing, easting) mosaic with the named uint16 vars."""
    box_geobox = _MASTER[10:30, 20:50]
    coords = box_geobox.coordinates
    ds = xr.Dataset(
        {name: (("time", "northing", "easting"), np.zeros((1, 20, 30), dt)) for name, dt in data_vars.items()},
        coords={
            "time": np.array(["2024-01-15"], dtype="datetime64[ns]"),
            "northing": coords["y"].values,
            "easting": coords["x"].values,
        },
    )
    write_dataset(path, ds, tile_id="x", baselines={}, chunks={"time": 1, "northing": 40, "easting": 40}, crs=_CRS)


def test_classify_store_present_with_full_schema(tmp_path):
    path = str(tmp_path / "mosaic.zarr")
    _write_mosaic(path, {"VV": "uint16", "VH": "uint16"})
    diag = classify_store(path, {"VV", "VH"})
    assert diag.status == PRESENT


def test_classify_store_missing_when_absent(tmp_path):
    assert classify_store(str(tmp_path / "nope.zarr"), {"VV"}).status == MISSING


def test_classify_store_corrupted_when_var_missing(tmp_path):
    path = str(tmp_path / "mosaic.zarr")
    _write_mosaic(path, {"VV": "uint16"})  # no VH
    diag = classify_store(path, {"VV", "VH"})
    assert diag.status == CORRUPTED
    assert "VH" in diag.detail


def test_classify_store_corrupted_when_unopenable(tmp_path):
    """Bytes on disk that aren't a valid store are CORRUPTED, not MISSING."""
    path = tmp_path / "junk.zarr"
    path.mkdir()
    (path / "not_a_store.txt").write_text("garbage")
    diag = classify_store(str(path), {"VV"})
    assert diag.status == CORRUPTED


# --------------------------------------------------------------------------- #
# load_features against a GeoJSON FeatureCollection
# --------------------------------------------------------------------------- #


def _write_geojson(tmp_path) -> str:
    """A 2-feature disjoint FeatureCollection in WGS84, keyed by tile_id."""
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"tile_id": "1"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-100.0, 40.0], [-99.7, 40.0], [-99.7, 40.3], [-100.0, 40.3], [-100.0, 40.0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"tile_id": "2"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-99.6, 40.0], [-99.3, 40.0], [-99.3, 40.3], [-99.6, 40.3], [-99.6, 40.0]]],
                },
            },
        ],
    }
    path = tmp_path / "regions.geojson"
    path.write_text(json.dumps(fc))
    return str(path)


def test_load_features_reprojects_and_rasterizes_to_subset(tmp_path):
    """End to end: features load in the master CRS, are disjoint, and each
    rasterizes to an exact subset of a master built over their union.
    """
    from tessera_embeddings.ingest.roi import rasterize_roi_zarr

    geojson = _write_geojson(tmp_path)
    master_path = str(tmp_path / "master.zarr")
    rasterize_roi_zarr(master_path, resolution=100.0, force_crs=_CRS, input_path=geojson)
    master = read_roi_metadata(master_path)

    feats = load_features(geojson, target_crs=master.native_crs, id_property="tile_id")
    assert [f.feature_id for f in feats] == ["1", "2"]
    assert_features_disjoint(feats)

    mc = master.geobox.coordinates
    mx, my = np.round(mc["x"].values, 3), np.round(mc["y"].values, 3)
    for feat in feats:
        out = str(tmp_path / f"feat_{feat.feature_id}.zarr")
        rasterize_feature_roi(out, master_geobox=master.geobox, feature=feat)
        fc_coords = read_roi_metadata(out).geobox.coordinates
        assert np.all(np.isin(np.round(fc_coords["x"].values, 3), mx))
        assert np.all(np.isin(np.round(fc_coords["y"].values, 3), my))


def test_load_features_missing_id_property_raises(tmp_path):
    with pytest.raises(ValueError, match="missing id property"):
        load_features(_write_geojson(tmp_path), target_crs=_CRS, id_property="does_not_exist")


def _poly(minx, miny, maxx, maxy) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
    }


def test_load_features_duplicate_id_raises(tmp_path):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"tile_id": "dup"}, "geometry": _poly(-100.0, 40.0, -99.9, 40.1)},
            {"type": "Feature", "properties": {"tile_id": "dup"}, "geometry": _poly(-99.5, 40.0, -99.4, 40.1)},
        ],
    }
    path = tmp_path / "dup.geojson"
    path.write_text(json.dumps(fc))
    with pytest.raises(ValueError, match="Duplicate feature id"):
        load_features(str(path), target_crs=_CRS, id_property="tile_id")
