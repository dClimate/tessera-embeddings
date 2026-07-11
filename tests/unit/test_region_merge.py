"""Unit tests for the region-merge domain layer (storage/region_merge.py).

The load-bearing properties:

* ``gather_time_union`` returns the sorted, de-duplicated union of feature dates.
* ``merge_feature_into_master`` places each feature's pixels in exactly its
  spatial window AND only on its own dates — the heterogeneous-date case a single
  time-range write would corrupt (a feature must not touch a master date it lacks).
* the fill-masked overlay never clobbers a neighbour's real pixels with fill.
* the CPU-stall watchdog kills a wedged worker pool but never a progressing copy.
* ``delete_store`` removes a store and tolerates a missing one.

These exercise the real ``write_dataset`` / ``open_store`` path on small local
Icechunk stores, so the merge is verified for real, not mocked.
"""

from __future__ import annotations

import itertools
import os
import signal
import threading
import time

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from tessera_embeddings.ingest.roi import ROIMetadata, check_output_exists
from tessera_embeddings.storage import region_merge
from tessera_embeddings.storage.empty_store import VarSpec, create_empty_store, create_empty_store_from_coords
from tessera_embeddings.storage.region_merge import (
    _chunk_blocks,
    delete_store,
    gather_time_union,
    merge_feature_into_master,
    merge_stores,
    read_master_axes,
    read_store_times,
)
from tessera_embeddings.storage.zarr_store import open_store, write_dataset

_CRS = "EPSG:5070"
_MASTER = GeoBox(shape=(80, 100), affine=Affine(10, 0, 500000, 0, -10, 4000000), crs=_CRS)
_CHUNKS = {"time": 1, "northing": 40, "easting": 40}


def _write_mosaic(path: str, geobox: GeoBox, dates: list[str], value: int) -> None:
    """Write a tiny single-var uint16 mosaic over ``geobox`` for ``dates``."""
    coords = geobox.coordinates
    times = np.array(dates, dtype="datetime64[ns]")
    shape = (len(dates), geobox.shape[0], geobox.shape[1])
    ds = xr.Dataset(
        {"VV": (("time", "northing", "easting"), np.full(shape, value, "uint16"))},
        coords={"time": times, "northing": coords["y"].values, "easting": coords["x"].values},
    )
    write_dataset(path, ds, tile_id="x", baselines={}, chunks=_CHUNKS, crs=_CRS)


def _write_ds(path: str, geobox: GeoBox, dates: list[str], data_vars: dict, chunks: dict | None = None) -> None:
    """Write a mosaic with arbitrary named ``(time, northing, easting)`` arrays.

    Used by the multi-variable / float-fill / mismatch tests that need control
    over var names, dtypes, and per-cell values that ``_write_mosaic`` (single
    uint16 var, uniform value) can't express.
    """
    coords = geobox.coordinates
    times = np.array(dates, dtype="datetime64[ns]")
    ds = xr.Dataset(
        {name: (("time", "northing", "easting"), arr) for name, arr in data_vars.items()},
        coords={"time": times, "northing": coords["y"].values, "easting": coords["x"].values},
    )
    write_dataset(path, ds, tile_id="x", baselines={}, chunks=chunks or _CHUNKS, crs=_CRS)


def _seed_master(path: str, times: np.ndarray, chunks: dict | None = None) -> None:
    """Seed an all-zero master mosaic over the full master grid for ``times``."""
    coords = _MASTER.coordinates
    shape = (len(times), 80, 100)
    ds = xr.Dataset(
        {"VV": (("time", "northing", "easting"), np.zeros(shape, "uint16"))},
        coords={"time": times, "northing": coords["y"].values, "easting": coords["x"].values},
    )
    write_dataset(path, ds, tile_id="m", baselines={}, chunks=chunks or _CHUNKS, crs=_CRS)


def _merge(master: str, feature: str, **kwargs) -> int:
    """Merge a feature into the master, reading the master axes the way a flow does.

    ``merge_feature_into_master`` requires the caller to read the master's date
    axis and variable set once and pass them in (a flow reads them once per
    master, not per feature), so the tests mirror that here.
    """
    master_times, master_vars = read_master_axes(master)
    return merge_feature_into_master(master, feature, master_times=master_times, master_vars=master_vars, **kwargs)


def _roi() -> ROIMetadata:
    """An ROIMetadata over the shared ``_MASTER`` grid, for ``merge_stores`` seeding.

    Constructed directly from the test geobox (no rasterized ROI store needed) —
    ``create_empty_store`` only reads ``geobox.coordinates`` / ``height`` / ``width``
    / ``native_crs``.
    """
    return ROIMetadata(bbox_wgs84=(-100.0, 40.0, -99.0, 41.0), native_crs=_CRS, geobox=_MASTER, width=100, height=80)


# --------------------------------------------------------------------------- #
# Date union
# --------------------------------------------------------------------------- #


def test_gather_time_union_dedups_and_sorts(tmp_path):
    a, b = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_mosaic(a, _MASTER[10:30, 20:50], ["2024-03-15", "2024-01-15"], 11)
    _write_mosaic(b, _MASTER[40:60, 60:90], ["2024-02-15", "2024-03-15"], 22)
    union = gather_time_union([a, b])
    assert [str(d)[:10] for d in union] == ["2024-01-15", "2024-02-15", "2024-03-15"]


def test_gather_time_union_tolerates_missing_store(tmp_path):
    a = str(tmp_path / "a.zarr")
    _write_mosaic(a, _MASTER[10:30, 20:50], ["2024-01-15"], 11)
    union = gather_time_union([a, str(tmp_path / "nonexistent.zarr")])
    assert [str(d)[:10] for d in union] == ["2024-01-15"]


def test_read_store_times_missing_is_empty(tmp_path):
    assert read_store_times(str(tmp_path / "nope.zarr")).size == 0


# --------------------------------------------------------------------------- #
# Region merge — the heterogeneous-date correctness case
# --------------------------------------------------------------------------- #


def test_merge_places_features_by_window_and_date(tmp_path):
    """Two disjoint features with overlapping-but-different date sets merge so that
    each feature's pixels land in its window on its dates only — and a feature
    never writes to a master date it does not have.
    """
    fa_box, fb_box = _MASTER[10:30, 20:50], _MASTER[40:60, 60:90]
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15", "2024-03-15"], 11)
    _write_mosaic(pb, fb_box, ["2024-02-15", "2024-03-15"], 22)

    master = str(tmp_path / "master.zarr")
    union = gather_time_union([pa, pb])
    _seed_master(master, union)

    assert _merge(master, pa) == 2
    assert _merge(master, pb) == 2

    out = open_store(master)["VV"]
    jan = np.datetime64("2024-01-15", "ns")
    feb = np.datetime64("2024-02-15", "ns")
    mar = np.datetime64("2024-03-15", "ns")

    # A in its window on its dates.
    assert (out.sel(time=jan).values[10:30, 20:50] == 11).all()
    assert (out.sel(time=mar).values[10:30, 20:50] == 11).all()
    # B in its window on its dates.
    assert (out.sel(time=feb).values[40:60, 60:90] == 22).all()
    assert (out.sel(time=mar).values[40:60, 60:90] == 22).all()
    # A has NO Feb date -> its window on Feb must remain the seed (0).
    assert (out.sel(time=feb).values[10:30, 20:50] == 0).all()
    # Nothing bled outside the two windows: exactly 4 feature-dates x 600 cells.
    assert int((out.values != 0).sum()) == 4 * (20 * 30)


def test_merge_interleaved_dates_place_correctly(tmp_path):
    """A feature whose dates are interleaved by a sibling on every other master date
    must still place each date's pixels at its exact (non-contiguous) master time
    index. The copy pairs feature array position to master index per date, so a
    gap-riddled date axis is no different from a contiguous one.
    """
    # A on Jan/Mar/May, B on Feb/Apr — so A's master indices are 0,2,4 (gaps at
    # B's dates) and B's are 1,3.
    fa_box, fb_box = _MASTER[10:30, 20:50], _MASTER[40:60, 60:90]
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15", "2024-03-15", "2024-05-15"], 11)
    _write_mosaic(pb, fb_box, ["2024-02-15", "2024-04-15"], 22)
    union = gather_time_union([pa, pb])

    master = str(tmp_path / "master.zarr")
    _seed_master(master, union)
    assert _merge(master, pa) == 3
    assert _merge(master, pb) == 2

    out = open_store(master)["VV"]
    # A on each of its three dates, in its window; B's window on those dates stays seed.
    for d in ("2024-01-15", "2024-03-15", "2024-05-15"):
        t = np.datetime64(d, "ns")
        assert (out.sel(time=t).values[10:30, 20:50] == 11).all()
        assert (out.sel(time=t).values[40:60, 60:90] == 0).all()
    # B on each of its two dates, in its window; A's window on those dates stays seed.
    for d in ("2024-02-15", "2024-04-15"):
        t = np.datetime64(d, "ns")
        assert (out.sel(time=t).values[40:60, 60:90] == 22).all()
        assert (out.sel(time=t).values[10:30, 20:50] == 0).all()


def test_merge_features_sharing_a_boundary_chunk_both_survive(tmp_path):
    """Two pixel-adjacent features that land in the SAME master chunk on the same
    date must both survive. Chunks are 40px; features at [20:40] and [40:60]
    northing both fall in (and straddle) the chunk row starting at 40, so the merge
    relies on read-modify-write across the sequential per-feature commits — the
    second feature must not clobber the first's pixels in the shared chunk.
    """
    # Northing 20:60 spans two 40px chunk rows (0:40, 40:80); both features touch
    # the second chunk row, so they share Zarr chunks on every shared date.
    fa_box, fb_box = _MASTER[20:42, 0:40], _MASTER[38:60, 0:40]
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15"], 11)
    _write_mosaic(pb, fb_box, ["2024-01-15"], 22)
    union = gather_time_union([pa, pb])

    master = str(tmp_path / "master.zarr")
    _seed_master(master, union)
    assert _merge(master, pa) == 1
    assert _merge(master, pb) == 1

    out = open_store(master)["VV"].sel(time=np.datetime64("2024-01-15", "ns")).values
    # A owns 20:42, B owns 38:60; they overlap on rows 38:42 where B (written
    # second) wins. The key property: A's non-overlapping rows survived the RMW.
    assert (out[20:38, 0:40] == 11).all(), "first feature's pixels clobbered in shared chunk"
    assert (out[42:60, 0:40] == 22).all()
    assert (out[38:42, 0:40] == 22).all(), "overlap rows: last writer wins"


def test_merge_does_not_clobber_neighbor_with_fill_in_overlapping_window(tmp_path):
    """A feature must not overwrite a prior feature's valid pixels with its own FILL.

    Two geometry-disjoint features routinely have OVERLAPPING bounding windows. In
    that window overlap a pixel is owned by at most one feature; the other feature
    carries FILL (0, the nodata sentinel) there because the pixel is outside its
    polygon. A blind full-window copy would write that fill over the owner's real
    data — silently corrupting the master.

    Here A's window [10:30, 20:50] is all valid (11). B's window [20:40, 30:60]
    overlaps A on [20:30, 30:50], but B has NO data there (fill 0) and real data
    (22) elsewhere. Merging A then B must leave A's pixels in the overlap intact: a
    feature only contributes where it is not fill.
    """
    fa_box, fb_box = _MASTER[10:30, 20:50], _MASTER[20:40, 30:60]
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15"], 11)

    # B: 22 everywhere except its overlap with A's window, which is fill (0).
    b_vals = np.full((1, fb_box.shape[0], fb_box.shape[1]), 22, "uint16")
    # B-local indices of the [20:30, 30:50] master overlap: rows 0:10, cols 0:20.
    b_vals[0, 0:10, 0:20] = 0
    _write_ds(pb, fb_box, ["2024-01-15"], {"VV": b_vals})

    master = str(tmp_path / "master.zarr")
    _seed_master(master, gather_time_union([pa, pb]))
    assert _merge(master, pa) == 1
    assert _merge(master, pb) == 1

    out = open_store(master)["VV"].sel(time=np.datetime64("2024-01-15", "ns")).values
    # The overlap: A owns it (11), B is fill there — A must survive.
    assert (out[20:30, 30:50] == 11).all(), "feature A's valid pixels clobbered by B's fill"
    # A's non-overlap pixels untouched.
    assert (out[10:20, 20:50] == 11).all()
    assert (out[20:30, 20:30] == 11).all()
    # B's genuine (non-fill) pixels landed.
    assert (out[30:40, 30:60] == 22).all()
    assert (out[20:30, 50:60] == 22).all()


def test_merge_all_real_block_short_circuits_master_read(tmp_path, monkeypatch):
    """An all-real interior block writes ``src`` directly and skips the master read.

    For a block fully inside the polygon ``is_fill`` is all-False, so the masked
    overlay ``where(is_fill, master, src)`` degenerates to ``src``. The merge then
    writes ``src`` without first reading the master block. ``np.where`` is invoked
    only on the partial-edge read-modify-write path, so its call count is a faithful
    proxy for "took the masked path"; here it must never be called.

    ``max_workers=1`` runs the copy inline in this process so the spy on
    ``region_merge.np.where`` sees the calls.
    """
    fa_box = _MASTER[0:40, 0:40]
    pa = str(tmp_path / "a.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15", "2024-03-15"], 11)
    master = str(tmp_path / "master.zarr")
    _seed_master(master, gather_time_union([pa]))

    where_calls = {"n": 0}
    real_where = region_merge.np.where

    def _counting_where(*a, **k):
        where_calls["n"] += 1
        return real_where(*a, **k)

    monkeypatch.setattr(region_merge.np, "where", _counting_where)

    assert _merge(master, pa, max_workers=1) == 2
    assert where_calls["n"] == 0, "all-real interior blocks took the masked read-modify-write path"

    out = open_store(master)["VV"]
    for d in ("2024-01-15", "2024-03-15"):
        t = np.datetime64(d, "ns")
        assert (out.sel(time=t).values[0:40, 0:40] == 11).all()
    # Nothing bled outside the feature window: exactly 2 dates x 40x40 cells.
    assert int((out.values != 0).sum()) == 2 * (40 * 40)


def test_merge_partial_edge_block_uses_masked_read(tmp_path, monkeypatch):
    """A block carrying out-of-polygon fill must take the masked read-modify-write.

    The complement of the short-circuit: when a block mixes real and fill pixels the
    merge MUST read the master and overlay only the real pixels (``np.where``), or it
    would clobber a neighbour. Here feature B's window overlaps A's with explicit fill
    in the overlap, so at least one block is partial — ``np.where`` must fire.
    """
    fa_box, fb_box = _MASTER[10:30, 20:50], _MASTER[20:40, 30:60]
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15"], 11)

    b_vals = np.full((1, fb_box.shape[0], fb_box.shape[1]), 22, "uint16")
    b_vals[0, 0:10, 0:20] = 0  # B's fill where it overlaps A's window
    _write_ds(pb, fb_box, ["2024-01-15"], {"VV": b_vals})

    master = str(tmp_path / "master.zarr")
    _seed_master(master, gather_time_union([pa, pb]))
    assert _merge(master, pa, max_workers=1) == 1

    where_calls = {"n": 0}
    real_where = region_merge.np.where

    def _counting_where(*a, **k):
        where_calls["n"] += 1
        return real_where(*a, **k)

    monkeypatch.setattr(region_merge.np, "where", _counting_where)

    assert _merge(master, pb, max_workers=1) == 1
    assert where_calls["n"] > 0, "partial-fill block did not take the masked read-modify-write path"

    out = open_store(master)["VV"].sel(time=np.datetime64("2024-01-15", "ns")).values
    assert (out[20:30, 30:50] == 11).all(), "feature A's pixels clobbered by B's fill"


def test_merge_disjoint_windows_sharing_a_chunk_both_survive(tmp_path):
    """Two features with NON-overlapping windows that still fall in the SAME master
    chunk must both survive. The windows are disjoint (no shared pixel), but the
    master chunk grid is coarse enough that one Zarr chunk spans both — so each
    feature writes a disjoint SUB-region of the shared chunk.

    Chunks are 40px. A owns northing [0:20], B owns [20:40]; both sub-regions live in
    the single chunk row [0:40]. Correctness rests on zarr's chunk-level
    read-modify-write across the sequential per-feature commits.
    """
    fa_box, fb_box = _MASTER[0:20, 0:40], _MASTER[20:40, 0:40]
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15"], 11)
    _write_mosaic(pb, fb_box, ["2024-01-15"], 22)

    master = str(tmp_path / "master.zarr")
    _seed_master(master, gather_time_union([pa, pb]))
    assert _merge(master, pa) == 1
    assert _merge(master, pb) == 1

    out = open_store(master)["VV"].sel(time=np.datetime64("2024-01-15", "ns")).values
    assert (out[0:20, 0:40] == 11).all(), "first feature's rows lost in the shared chunk"
    assert (out[20:40, 0:40] == 22).all(), "second feature's rows missing"


def test_merge_empty_feature_is_noop(tmp_path):
    """A missing feature mosaic contributes nothing and does not raise."""
    master = str(tmp_path / "master.zarr")
    _seed_master(master, np.array(["2024-01-15"], dtype="datetime64[ns]"))
    assert _merge(master, str(tmp_path / "missing.zarr")) == 0


def test_merge_disjoint_vars_raises(tmp_path):
    """Feature store with no variables in common with the master raises immediately
    rather than silently no-oping and reporting success.
    """
    master = str(tmp_path / "master.zarr")
    union = np.array(["2024-01-15"], dtype="datetime64[ns]")
    _seed_master(master, union)

    feature = str(tmp_path / "feature.zarr")
    shape = (1, 20, 30)
    _write_ds(feature, _MASTER[10:30, 20:50], ["2024-01-15"], {"scl": np.zeros(shape, "uint8")})

    with pytest.raises(ValueError, match="shares no variables"):
        _merge(master, feature)


# --------------------------------------------------------------------------- #
# Extensions beyond the ported suite
# --------------------------------------------------------------------------- #


def test_merge_multi_var_writes_intersection_only(tmp_path):
    """A master with {VV, VH} and a feature with {VV, extra} merges VV, ignores
    ``extra`` (feature-only), and leaves VH untouched — the intersection contract of
    ``_shared_feature_vars`` beyond the no-overlap raise.
    """
    coords = _MASTER.coordinates
    shape = (1, 80, 100)
    ds_m = xr.Dataset(
        {
            "VV": (("time", "northing", "easting"), np.zeros(shape, "uint16")),
            "VH": (("time", "northing", "easting"), np.zeros(shape, "uint16")),
        },
        coords={
            "time": np.array(["2024-01-15"], dtype="datetime64[ns]"),
            "northing": coords["y"].values,
            "easting": coords["x"].values,
        },
    )
    master = str(tmp_path / "master.zarr")
    write_dataset(master, ds_m, tile_id="m", baselines={}, chunks=_CHUNKS, crs=_CRS)

    fbox = _MASTER[10:30, 20:50]
    feature = str(tmp_path / "feature.zarr")
    _write_ds(
        feature,
        fbox,
        ["2024-01-15"],
        {
            "VV": np.full((1, 20, 30), 11, "uint16"),
            "extra": np.full((1, 20, 30), 99, "uint16"),
        },
    )

    assert _merge(master, feature) == 1
    out = open_store(master)
    vv = out["VV"].sel(time=np.datetime64("2024-01-15", "ns")).values
    assert (vv[10:30, 20:50] == 11).all()
    assert (vv[0:10, :] == 0).all(), "VV bled outside the feature window"
    # VH was never in the feature's shared vars — must stay all seed (0).
    assert (out["VH"].values == 0).all()
    # extra is feature-only; it must not have been added to the master.
    assert "extra" not in out.data_vars


def test_merge_float_nan_fill_does_not_clobber_neighbor(tmp_path):
    """The fill-overlay path on a FLOAT store: NaN is the fill sentinel, so a feature's
    NaN cells must not clobber a neighbour's real float values (mirrors the uint16
    fill-overlay test with ``np.nan``).
    """
    coords = _MASTER.coordinates
    ds_m = xr.Dataset(
        {"VV": (("time", "northing", "easting"), np.zeros((1, 80, 100), "float32"))},
        coords={
            "time": np.array(["2024-01-15"], dtype="datetime64[ns]"),
            "northing": coords["y"].values,
            "easting": coords["x"].values,
        },
    )
    master = str(tmp_path / "master.zarr")
    write_dataset(master, ds_m, tile_id="m", baselines={}, chunks=_CHUNKS, crs=_CRS)

    fa_box, fb_box = _MASTER[10:30, 20:50], _MASTER[20:40, 30:60]
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_ds(pa, fa_box, ["2024-01-15"], {"VV": np.full((1, 20, 30), 1.5, "float32")})

    b_vals = np.full((1, fb_box.shape[0], fb_box.shape[1]), 2.5, "float32")
    b_vals[0, 0:10, 0:20] = np.nan  # B's fill where it overlaps A's window
    _write_ds(pb, fb_box, ["2024-01-15"], {"VV": b_vals})

    assert _merge(master, pa) == 1
    assert _merge(master, pb) == 1

    out = open_store(master)["VV"].sel(time=np.datetime64("2024-01-15", "ns")).values
    assert (out[20:30, 30:50] == 1.5).all(), "feature A's float pixels clobbered by B's NaN fill"
    assert (out[30:40, 30:60] == 2.5).all(), "feature B's real float pixels missing"
    assert (out[20:30, 50:60] == 2.5).all()


def test_merge_feature_date_absent_from_master_raises(tmp_path):
    """A feature carrying a date the master was not seeded with is a seeding bug,
    surfaced loudly by ``_feature_master_indices`` rather than a silent misplacement.
    """
    master = str(tmp_path / "master.zarr")
    _seed_master(master, np.array(["2024-01-15"], dtype="datetime64[ns]"))

    feature = str(tmp_path / "feature.zarr")
    _write_mosaic(feature, _MASTER[10:30, 20:50], ["2024-02-15"], 11)

    with pytest.raises(ValueError, match="absent from the master axis"):
        _merge(master, feature)


def test_merge_grid_not_pixel_subset_raises(tmp_path):
    """A feature at a different resolution is not an exact pixel-subset of the master;
    the resolved spatial slice length won't equal the feature's pixel count, so the
    master-snap guard raises rather than misplacing pixels.
    """
    # 20 m feature over roughly the same extent as a 10 m master sub-box: the master
    # coords the feature spans outnumber the feature's own pixels, so the lengths
    # disagree and the guard fires.
    coarse = GeoBox(shape=(10, 15), affine=Affine(20, 0, 500000, 0, -20, 4000000), crs=_CRS)
    feature = str(tmp_path / "feature.zarr")
    _write_ds(feature, coarse, ["2024-01-15"], {"VV": np.full((1, 10, 15), 11, "uint16")})

    master = str(tmp_path / "master.zarr")
    _seed_master(master, np.array(["2024-01-15"], dtype="datetime64[ns]"))

    with pytest.raises(ValueError, match=r"master-snapped|pixel-subset"):
        _merge(master, feature)


# --------------------------------------------------------------------------- #
# Time-chunk disjointness guard (OSS hardening addition)
# --------------------------------------------------------------------------- #


def test_time_chunk_guard_raises_when_dates_share_chunk(tmp_path):
    """A master chunked ``time=2`` into which a feature maps two dates that fall in
    the same time chunk must raise: the two per-date copy units would be written by
    different forks with no conflict resolution.
    """
    fa_box = _MASTER[10:30, 20:50]
    pa = str(tmp_path / "a.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15", "2024-01-16"], 11)
    master = str(tmp_path / "master.zarr")
    union = gather_time_union([pa])  # two adjacent dates -> master indices 0,1
    _seed_master(master, union, chunks={"time": 2, "northing": 40, "easting": 40})

    with pytest.raises(ValueError, match="time chunk"):
        _merge(master, pa)


def test_time_chunk_guard_allows_single_date_in_multichunk_master(tmp_path):
    """The guard must not overfire: a single-date feature into a ``time=2``-chunked
    master trivially keeps each date in its own chunk, so the merge succeeds.
    """
    master = str(tmp_path / "master.zarr")
    _seed_master(
        master,
        np.array(["2024-01-15", "2024-01-16"], dtype="datetime64[ns]"),
        chunks={"time": 2, "northing": 40, "easting": 40},
    )
    pa = str(tmp_path / "a.zarr")
    _write_mosaic(pa, _MASTER[10:30, 20:50], ["2024-01-15"], 11)

    assert _merge(master, pa) == 1
    out = open_store(master)["VV"].sel(time=np.datetime64("2024-01-15", "ns")).values
    assert (out[10:30, 20:50] == 11).all()


def test_time_chunk_guard_allows_dates_in_distinct_chunks(tmp_path):
    """A ``time=2``-chunked master with a feature whose two dates land in DISTINCT
    time chunks (master indices 0 and 2) must succeed — only actual chunk sharing is
    rejected.
    """
    master = str(tmp_path / "master.zarr")
    _seed_master(
        master,
        np.array(["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"], dtype="datetime64[ns]"),
        chunks={"time": 2, "northing": 40, "easting": 40},
    )
    pa = str(tmp_path / "a.zarr")
    # Dates at master indices 0 and 2 -> chunks 0 and 1, distinct.
    _write_mosaic(pa, _MASTER[10:30, 20:50], ["2024-01-15", "2024-01-17"], 11)

    assert _merge(master, pa) == 2
    out = open_store(master)["VV"]
    for d in ("2024-01-15", "2024-01-17"):
        t = np.datetime64(d, "ns")
        assert (out.sel(time=t).values[10:30, 20:50] == 11).all()


def test_merge_unreadable_feature_raises(tmp_path, monkeypatch):
    """A feature store that exists but can't be read (corruption, expired credentials,
    a transient S3 error) must fail the merge LOUDLY, not be silently skipped as a
    no-op. Only a genuinely-absent store returns 0; swallowing an unreadable one would
    let the master commit as successful with this feature's data missing, after which
    the caller may delete the temp store.
    """
    master = str(tmp_path / "master.zarr")
    _seed_master(master, np.array(["2024-01-15"], dtype="datetime64[ns]"))
    feature = str(tmp_path / "feature.zarr")
    _write_mosaic(feature, _MASTER[10:30, 20:50], ["2024-01-15"], 11)
    # Read the master axes before patching (the merge reads the feature via open_store).
    master_times, master_vars = read_master_axes(master)

    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated corrupt/unreadable store")

    monkeypatch.setattr(region_merge, "open_store", boom)
    with pytest.raises(RuntimeError, match="corrupt/unreadable"):
        merge_feature_into_master(master, feature, master_times=master_times, master_vars=master_vars)


def test_merge_duplicate_feature_dates_raises(tmp_path):
    """A feature with duplicate dates maps two copy units to the SAME master slot. With a
    time=1 master that bypasses the time-chunk guard, so a dedicated distinct-slot check
    must reject it rather than let two forks race last-writer on one chunk.
    """
    fbox = _MASTER[10:30, 20:50]
    fc = fbox.coordinates
    dup = np.array(["2024-01-15", "2024-01-15"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {"VV": (("time", "northing", "easting"), np.full((2, 20, 30), 11, "uint16"))},
        coords={"time": dup, "northing": fc["y"].values, "easting": fc["x"].values},
    )
    feature = str(tmp_path / "feature.zarr")
    write_dataset(feature, ds, tile_id="x", baselines={}, chunks=_CHUNKS, crs=_CRS)

    master = str(tmp_path / "master.zarr")
    _seed_master(master, np.array(["2024-01-15"], dtype="datetime64[ns]"))  # union dedups to one date

    with pytest.raises(ValueError, match="duplicate dates"):
        _merge(master, feature)


def test_merge_non_uniform_master_chunk_grid_raises(tmp_path):
    """A master whose shared variables were seeded with DIFFERENT chunk grids must be
    rejected: the copy tiles every variable on one grid, so a coarser-chunked variable
    could have a chunk split across units sharded to different forks — a silent
    same-chunk race. Guard, don't corrupt.
    """
    coords = _MASTER.coordinates
    union = np.array(["2024-01-15"], dtype="datetime64[ns]")
    master = str(tmp_path / "master.zarr")
    create_empty_store_from_coords(
        master,
        coords={"time": union, "northing": coords["y"].values, "easting": coords["x"].values},
        var_specs={
            "VV": VarSpec(dims=("time", "northing", "easting"), dtype=np.dtype("uint16"), chunks=(1, 40, 40)),
            "VH": VarSpec(dims=("time", "northing", "easting"), dtype=np.dtype("uint16"), chunks=(1, 80, 100)),
        },
        commit_msg="seed non-uniform master",
        attrs={"crs": _CRS},
    )
    feature = str(tmp_path / "feature.zarr")
    _write_ds(
        feature,
        _MASTER[10:30, 20:50],
        ["2024-01-15"],
        {"VV": np.full((1, 20, 30), 11, "uint16"), "VH": np.full((1, 20, 30), 22, "uint16")},
    )
    with pytest.raises(ValueError, match="non-uniform chunk grids"):
        _merge(master, feature)


def test_merge_unsorted_master_axis_raises(tmp_path):
    """`np.searchsorted` (feature date → master index) assumes a sorted master axis.
    `read_master_axes` returns store order and the empty-store helpers persist dates
    as-is, so an unsorted master must be rejected rather than silently map a feature
    to the wrong time row.
    """
    coords = _MASTER.coordinates
    unsorted = np.array(["2024-01-15", "2024-01-10"], dtype="datetime64[ns]")  # not ascending
    master = str(tmp_path / "master.zarr")
    create_empty_store_from_coords(
        master,
        coords={"time": unsorted, "northing": coords["y"].values, "easting": coords["x"].values},
        var_specs={"VV": VarSpec(dims=("time", "northing", "easting"), dtype=np.dtype("uint16"), chunks=(1, 40, 40))},
        commit_msg="unsorted master",
        attrs={"crs": _CRS},
    )
    feature = str(tmp_path / "feature.zarr")
    _write_mosaic(feature, _MASTER[10:30, 20:50], ["2024-01-15"], 11)
    with pytest.raises(ValueError, match="not sorted"):
        _merge(master, feature)


def test_merge_dtype_mismatch_raises(tmp_path):
    """A feature and master sharing a variable name but not its dtype would let the raw
    positional assignment silently cast (uint16 → uint8 truncation); reject as schema drift.
    """
    coords = _MASTER.coordinates
    union = np.array(["2024-01-15"], dtype="datetime64[ns]")
    master = str(tmp_path / "master.zarr")
    create_empty_store_from_coords(
        master,
        coords={"time": union, "northing": coords["y"].values, "easting": coords["x"].values},
        var_specs={"VV": VarSpec(dims=("time", "northing", "easting"), dtype=np.dtype("uint8"), chunks=(1, 40, 40))},
        commit_msg="uint8 master",
        attrs={"crs": _CRS},
    )
    feature = str(tmp_path / "feature.zarr")
    _write_mosaic(feature, _MASTER[10:30, 20:50], ["2024-01-15"], 11)  # uint16 VV
    with pytest.raises(ValueError, match="dtype differs"):
        _merge(master, feature)


def test_merge_reversed_feature_axis_raises(tmp_path):
    """A feature with the SAME spatial bounds but a reversed axis (ascending northing over
    a descending master) passes the extent-length guard yet would transpose pixels; the
    elementwise coordinate check must catch it.
    """
    fbox = _MASTER[10:30, 20:50]
    c = fbox.coordinates
    ds = xr.Dataset(
        {"VV": (("time", "northing", "easting"), np.full((1, 20, 30), 11, "uint16"))},
        coords={
            "time": np.array(["2024-01-15"], dtype="datetime64[ns]"),
            "northing": c["y"].values[::-1],  # reversed vs the master's descending axis
            "easting": c["x"].values,
        },
    )
    feature = str(tmp_path / "feature.zarr")
    write_dataset(feature, ds, tile_id="x", baselines={}, chunks=_CHUNKS, crs=_CRS)
    master = str(tmp_path / "master.zarr")
    _seed_master(master, np.array(["2024-01-15"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="coordinates do not match"):
        _merge(master, feature)


# --------------------------------------------------------------------------- #
# merge_stores — the full seed → merge → cleanup driver
# --------------------------------------------------------------------------- #


def test_merge_stores_end_to_end(tmp_path):
    """merge_stores seeds a master over the feature date UNION and places each
    feature's pixels in its own window on its own dates — heterogeneous dates and all.
    """
    pa, pb = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    _write_mosaic(pa, _MASTER[10:30, 20:50], ["2024-01-15", "2024-03-15"], 11)
    _write_mosaic(pb, _MASTER[40:60, 60:90], ["2024-02-15", "2024-03-15"], 22)
    master = str(tmp_path / "master.zarr")

    summary = merge_stores(master, [pa, pb], roi=_roi(), var_dtypes={"VV": np.dtype("uint16")}, tile_id="m")

    assert summary["n_dates"] == 3  # union of jan/feb/mar (mar shared)
    assert summary["merged"] == {pa: 2, pb: 2}
    assert summary["skipped"] is False
    assert summary["deleted"] == []
    out = open_store(master)["VV"]
    assert (out.sel(time=np.datetime64("2024-01-15", "ns")).values[10:30, 20:50] == 11).all()
    assert (out.sel(time=np.datetime64("2024-02-15", "ns")).values[40:60, 60:90] == 22).all()
    # a has no Feb date → its window stays fill (0) on Feb.
    assert (out.sel(time=np.datetime64("2024-02-15", "ns")).values[10:30, 20:50] == 0).all()


def test_merge_stores_delete_temp_removes_features(tmp_path):
    """delete_temp=True drops the per-feature stores after a successful merge."""
    pa = str(tmp_path / "a.zarr")
    _write_mosaic(pa, _MASTER[10:30, 20:50], ["2024-01-15"], 11)
    master = str(tmp_path / "master.zarr")

    summary = merge_stores(
        master, [pa], roi=_roi(), var_dtypes={"VV": np.dtype("uint16")}, tile_id="m", delete_temp=True
    )
    assert summary["deleted"] == [pa]
    assert read_store_times(pa).size == 0  # feature store is gone


def test_merge_stores_refuses_then_overwrites_existing_master(tmp_path):
    """An existing master is refused unless overwrite_master=True."""
    pa = str(tmp_path / "a.zarr")
    _write_mosaic(pa, _MASTER[10:30, 20:50], ["2024-01-15"], 11)
    master = str(tmp_path / "master.zarr")
    dt = {"VV": np.dtype("uint16")}

    merge_stores(master, [pa], roi=_roi(), var_dtypes=dt, tile_id="m")
    with pytest.raises(FileExistsError, match="already exists"):
        merge_stores(master, [pa], roi=_roi(), var_dtypes=dt, tile_id="m")
    summary = merge_stores(master, [pa], roi=_roi(), var_dtypes=dt, tile_id="m", overwrite_master=True)
    assert summary["merged"] == {pa: 1}


def test_merge_stores_resume_requires_existing_master(tmp_path):
    """resume=True with no master is a loud error, not a silent seed."""
    with pytest.raises(FileNotFoundError, match="resume=True"):
        merge_stores(
            str(tmp_path / "master.zarr"),
            [str(tmp_path / "a.zarr")],
            roi=_roi(),
            var_dtypes={"VV": np.dtype("uint16")},
            tile_id="m",
            resume=True,
        )


def test_merge_stores_empty_feature_paths_raises(tmp_path):
    with pytest.raises(ValueError, match="feature_paths is empty"):
        merge_stores(str(tmp_path / "m.zarr"), [], roi=_roi(), var_dtypes={"VV": np.dtype("uint16")}, tile_id="m")


def test_merge_stores_no_dates_skips(tmp_path):
    """When no feature carries any date, there is nothing to seed or merge."""
    master = str(tmp_path / "master.zarr")
    summary = merge_stores(
        master, [str(tmp_path / "missing.zarr")], roi=_roi(), var_dtypes={"VV": np.dtype("uint16")}, tile_id="m"
    )
    assert summary["skipped"] is True
    assert summary["n_dates"] == 0
    assert not check_output_exists(master)  # nothing seeded


def test_empty_store_is_a_valid_merge_target(tmp_path):
    """The empty-store seeder and the merge are designed as a pair: a master seeded by
    ``create_empty_store`` accepts a ``merge_feature_into_master`` write, the feature's
    pixels landing in its window on its date. Exercises the seed → merge handoff
    end-to-end (moved here from the empty-store suite when the merge split into its own
    PR, so it stays with the merge it exercises).
    """
    master = str(tmp_path / "master.zarr")
    times = np.array(["2024-01-15"], dtype="datetime64[ns]")
    create_empty_store(master, roi=_roi(), times=times, var_dtypes={"VV": np.dtype("uint16")}, tile_id="m", crs=_CRS)

    feature = str(tmp_path / "feature.zarr")
    _write_mosaic(feature, _MASTER[10:30, 20:50], ["2024-01-15"], 11)

    master_times, master_vars = read_master_axes(master)
    assert merge_feature_into_master(master, feature, master_times=master_times, master_vars=master_vars) == 1
    out = open_store(master)["VV"].sel(time=np.datetime64("2024-01-15", "ns")).values
    assert (out[10:30, 20:50] == 11).all()
    assert (out[0:10, :] == 0).all(), "feature bled outside its window"


# --------------------------------------------------------------------------- #
# _chunk_blocks (pure function)
# --------------------------------------------------------------------------- #


def test_chunk_blocks_aligned_span():
    assert _chunk_blocks(0, 80, 40) == [slice(0, 40), slice(40, 80)]


def test_chunk_blocks_unaligned_edges():
    # start mid-chunk, stop mid-chunk -> partial first and last blocks.
    assert _chunk_blocks(10, 70, 40) == [slice(10, 40), slice(40, 70)]


def test_chunk_blocks_within_single_chunk():
    assert _chunk_blocks(5, 30, 40) == [slice(5, 30)]


@pytest.mark.parametrize(
    ("start", "stop", "chunk"),
    [(0, 80, 40), (10, 70, 40), (5, 30, 40), (0, 100, 40), (37, 123, 40), (0, 1, 7)],
)
def test_chunk_blocks_tiles_span_and_respects_grid(start, stop, chunk):
    blocks = _chunk_blocks(start, stop, chunk)
    # Blocks tile [start, stop) exactly, in order, without gaps or overlap.
    assert blocks[0].start == start
    assert blocks[-1].stop == stop
    for prev, nxt in itertools.pairwise(blocks):
        assert prev.stop == nxt.start
    # Each block lies within a single chunk of the grid.
    for b in blocks:
        assert b.start // chunk == (b.stop - 1) // chunk


# --------------------------------------------------------------------------- #
# Hang protection — a wedged worker must fail loudly, not hang the run
# --------------------------------------------------------------------------- #


def _hang_forever(fork, feature_path, units, threads, max_concurrent_requests=None):
    """Stand-in for ``_copy_units_in_process`` that never returns (simulates a worker
    parked on a dead socket). Module-level so it survives the fork into the child.
    """
    while True:
        time.sleep(3600)


def test_merge_times_out_and_kills_wedged_workers(tmp_path, monkeypatch):
    """A worker that hangs must make ``merge_feature_into_master`` raise ``TimeoutError``
    within a bounded wall-time — not hang forever — and leave no orphan workers.

    Forces the multi-shard process-pool path with ``max_workers=2`` and replaces the
    copy worker with one that sleeps forever. This uses the production ``spawn`` start
    method (not ``fork``): under spawn the submitted callable pickles by qualified name
    (``test_region_merge._hang_forever``) and the worker re-imports it, so the
    monkeypatched worker still reaches the child — without forking a parent that has
    already initialised icechunk's Rust/tokio runtime, which is not fork-safe (a
    fork-based variant crashes the pool on macOS in both this repo and yield-embeddings).
    ``stall_poll_sec`` is small so the ``feature_timeout_sec`` ceiling is checked
    promptly; ``feature_retries=0`` so the single attempt fails fast. A guard thread
    bounds the whole test so a regression can never wedge the suite itself.
    """
    fa_box = _MASTER[10:30, 20:50]
    pa = str(tmp_path / "a.zarr")
    _write_mosaic(pa, fa_box, ["2024-01-15", "2024-03-15"], 11)
    master = str(tmp_path / "master.zarr")
    union = gather_time_union([pa])
    _seed_master(master, union)
    master_times, master_vars = read_master_axes(master)

    monkeypatch.setattr(region_merge, "_copy_units_in_process", _hang_forever)

    seen_pids: list[int] = []
    real_pool_cls = region_merge.ProcessPoolExecutor

    class _TrackingPool(real_pool_cls):
        def submit(self, *a, **k):
            fut = super().submit(*a, **k)
            seen_pids[:] = list(self._processes.keys())
            return fut

    monkeypatch.setattr(region_merge, "ProcessPoolExecutor", _TrackingPool)

    result: dict = {}

    def _run():
        try:
            merge_feature_into_master(
                master,
                pa,
                master_times=master_times,
                master_vars=master_vars,
                max_workers=2,
                feature_timeout_sec=2.0,
                stall_poll_sec=0.25,
                feature_retries=0,
            )
            result["ok"] = True
        except BaseException as e:
            result["exc"] = e

    runner = threading.Thread(target=_run, daemon=True)
    runner.start()
    # Generous bound: 2s timeout + spawn/teardown slack. If the merge hung, this fails
    # the test instead of hanging the whole suite.
    runner.join(timeout=60.0)

    assert not runner.is_alive(), "merge_feature_into_master hung instead of timing out"
    assert "ok" not in result, "merge unexpectedly succeeded with a wedged worker"
    assert isinstance(result.get("exc"), TimeoutError), f"expected TimeoutError, got {result.get('exc')!r}"
    assert str(pa) in str(result["exc"])

    # No orphan workers: every captured child PID must be gone after the kill.
    deadline = time.monotonic() + 5.0
    survivors = []
    for pid in seen_pids:
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            survivors.append(pid)
    # Reap any straggler we may have missed so we don't leak into the next test.
    for pid in survivors:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    assert not survivors, f"orphan worker process(es) survived the timeout teardown: {survivors}"


# --------------------------------------------------------------------------- #
# Stall watchdog (CPU-progress based)
# --------------------------------------------------------------------------- #


class _NeverFuture:
    """A future stand-in that never completes, so the watchdog runs to a decision.

    ``_wait_with_stall_detection`` only calls ``concurrent.futures.wait`` on the list;
    we patch that to a no-op so the loop is driven purely by the monkeypatched CPU
    clock and never blocks on a real future.
    """


def _drive_watchdog(monkeypatch, cpu_samples, *, grace_sec, poll_sec):
    """Run the watchdog against a scripted CPU series; return its (completed, pending).

    ``cpu_samples`` is consumed one reading per loop iteration (the last value repeats
    if the loop outlives the script). ``concurrent.futures.wait`` is stubbed to return
    "still pending" instantly and ``time.monotonic`` is advanced by ``poll_sec`` each
    iteration, so the test is deterministic and runs in microseconds regardless of the
    real grace window.
    """
    samples = list(cpu_samples)
    state = {"i": 0, "t": 0.0}

    def _fake_cpu():
        i = min(state["i"], len(samples) - 1)
        return samples[i]

    def _fake_wait(_futures, timeout=None):
        # Advance the scripted clock + sample cursor one step, report all pending.
        state["i"] += 1
        state["t"] += poll_sec
        return set(), {object()}

    def _fake_monotonic():
        return state["t"]

    monkeypatch.setattr(region_merge, "_busy_cpu_seconds", _fake_cpu)
    monkeypatch.setattr(region_merge.concurrent.futures, "wait", _fake_wait)
    monkeypatch.setattr(region_merge.time, "monotonic", _fake_monotonic)

    return region_merge._wait_with_stall_detection(
        [_NeverFuture()],
        grace_sec=grace_sec,
        poll_sec=poll_sec,
        hard_timeout_sec=None,
        log=region_merge.logger,
        feature_path="s3://bucket/feature.zarr",
    )


def test_watchdog_does_not_kill_while_cpu_climbs(monkeypatch):
    """A copy that keeps burning CPU is never flagged, even far past the grace window."""
    grace_sec, poll_sec = 60.0, 10.0
    climbing = [float(i * 50) for i in range(100)]  # 100 polls = 1000s ~ 16x grace
    st = {"i": 0, "t": 0.0}
    monkeypatch.setattr(region_merge, "_busy_cpu_seconds", lambda: climbing[min(st["i"], len(climbing) - 1)])
    monkeypatch.setattr(region_merge.time, "monotonic", lambda: st["t"])

    def _wait(_futures, timeout=None):
        st["i"] += 1
        st["t"] += poll_sec
        if st["i"] >= len(climbing):
            return {object()}, set()  # series exhausted with no kill → completes cleanly
        return set(), {object()}

    monkeypatch.setattr(region_merge.concurrent.futures, "wait", _wait)

    completed, pending = region_merge._wait_with_stall_detection(
        [_NeverFuture()],
        grace_sec=grace_sec,
        poll_sec=poll_sec,
        hard_timeout_sec=None,
        log=region_merge.logger,
        feature_path="s3://bucket/feature.zarr",
    )
    assert not pending, "watchdog killed a CPU-progressing copy (false stall)"


def test_watchdog_flags_stall_when_cpu_flat(monkeypatch):
    """Flat CPU for the full grace window returns the unfinished futures as pending."""
    grace_sec, poll_sec = 60.0, 10.0
    flat = [1000.0] * 50
    completed, pending = _drive_watchdog(monkeypatch, flat, grace_sec=grace_sec, poll_sec=poll_sec)
    assert pending, "watchdog failed to flag a flat-CPU stall"


def test_watchdog_hard_ceiling_overrides_cpu_progress(monkeypatch):
    """``hard_timeout_sec`` flags pending even while CPU is climbing (escape hatch)."""
    poll_sec = 10.0
    climbing = [float(i * 100) for i in range(50)]
    samples = list(climbing)
    st = {"i": 0, "t": 0.0}
    monkeypatch.setattr(region_merge, "_busy_cpu_seconds", lambda: samples[min(st["i"], len(samples) - 1)])
    monkeypatch.setattr(region_merge.time, "monotonic", lambda: st["t"])

    def _wait(_futures, timeout=None):
        st["i"] += 1
        st["t"] += poll_sec
        return set(), {object()}

    monkeypatch.setattr(region_merge.concurrent.futures, "wait", _wait)

    completed, pending = region_merge._wait_with_stall_detection(
        [_NeverFuture()],
        grace_sec=10_000.0,  # so high the CPU-stall path can never trigger
        poll_sec=poll_sec,
        hard_timeout_sec=30.0,  # 3 polls
        log=region_merge.logger,
        feature_path="s3://bucket/feature.zarr",
    )
    assert pending, "hard ceiling failed to flag a long-running (but live) copy"


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #


def test_delete_store_removes_and_tolerates_missing(tmp_path):
    p = str(tmp_path / "a.zarr")
    _write_mosaic(p, _MASTER[10:30, 20:50], ["2024-01-15"], 11)
    assert check_output_exists(p)
    assert delete_store(p) is True
    assert not check_output_exists(p)
    # second delete is a tolerated no-op
    assert delete_store(p) is False
