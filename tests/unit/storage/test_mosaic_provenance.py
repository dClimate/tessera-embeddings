"""The ingest half of provenance at the store: the granule lists and the switch.

* Each date's granule list lands at its own slot in the same commit as its pixels, under both
  write paths, and a failed batch leaves no list behind.
* A day the ingest skipped whole is listed off the time axis, once the mosaic exists, and a retry
  replaces its list.
* The ``provenance`` switch is a manifest term whose absence means off: a store keeps the setting
  it was seeded with, and a store from before the term resumes with the switch off.
"""

from __future__ import annotations

import json

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

import tessera_embeddings.storage.zarr_store as zs
from tessera_embeddings.config.ingest import (
    GRANULES_ARRAY,
    PROVENANCE_VERSION,
    SKIPPED_DAYS_ARRAY,
    SKIPPED_GRANULES_ARRAY,
    SOURCE_INDEX_VAR,
)
from tessera_embeddings.errors import ConfigMismatchError
from tessera_embeddings.ingest.provenance import SkippedDays
from tessera_embeddings.ingest.solar_days import normalize_to_solar_day
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import open_store, open_store_as_zarr_group, write_days_windows
from tests.stac_items import fake_stac_item

CHUNKS = {"time": 1, "northing": 4, "easting": 4}
WINDOWS = [(0, 4, 0, 8), (4, 8, 0, 4)]
ON = IngestManifest(roi_manifest_hash="abc", provenance_version=PROVENANCE_VERSION)
OFF = IngestManifest(roi_manifest_hash="abc")


class _Roi:
    geobox = GeoBox((8, 8), Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0), "EPSG:32601")
    height = 8
    width = 8


def _day(date: str, index: int | None) -> xr.Dataset:
    shape, chunks = (1, 8, 8), (1, 4, 4)
    # A ramp, not a constant: every window's blocks must land where they belong.
    ramp = da.from_array(np.arange(64, dtype=np.uint16).reshape(shape) + 7, chunks=chunks)
    data = {"band": (("time", "northing", "easting"), ramp)}
    if index is not None:
        data[SOURCE_INDEX_VAR] = (("time", "northing", "easting"), ramp % 3 + index)
    return xr.Dataset(data, coords={"time": np.array([np.datetime64(date, "ns")])})


def _expected(index: int | None) -> np.ndarray:
    """What the store holds after the windows are written: the ramp (or its index) inside them, fill outside."""
    ramp = np.arange(64, dtype=np.uint16).reshape(8, 8) + 7
    values = ramp if index is None else ramp % 3 + index
    out = np.zeros((8, 8), dtype=np.uint16)
    for y0, y1, x0, x1 in WINDOWS:
        out[y0:y1, x0:x1] = values[y0:y1, x0:x1]
    return out


def _granules(date: str) -> str:
    return json.dumps([{"granule_id": f"G-{date}"}])


def _write(store: str, dates: list[str], manifest: IngestManifest, *, parallel: bool = False) -> None:
    on = manifest.provenance_version is not None
    write_days_windows(
        store,
        [(_day(d, 1 if on else None), list(WINDOWS)) for d in dates],
        roi=_Roi(),
        manifest=manifest,
        baselines={},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks=CHUNKS,
        parallel_windows=parallel,
        granules=[_granules(d) for d in dates] if on else None,
    )


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "overlapped"])
def test_each_slot_holds_its_own_granule_list(tmp_path, parallel) -> None:
    store = str(tmp_path / "s.zarr")
    _write(store, ["2024-06-01"], ON, parallel=parallel)
    _write(store, ["2024-06-02", "2024-06-03"], ON, parallel=parallel)

    root = open_store_as_zarr_group(store)
    lists = root[GRANULES_ARRAY][:].tolist()
    assert lists == [_granules(d) for d in ("2024-06-01", "2024-06-02", "2024-06-03")]
    assert len(lists) == root["time"].shape[0]
    np.testing.assert_array_equal(open_store(store)[SOURCE_INDEX_VAR].values, np.broadcast_to(_expected(1), (3, 8, 8)))
    np.testing.assert_array_equal(open_store(store)["band"].values, np.broadcast_to(_expected(None), (3, 8, 8)))


def test_a_failed_batch_leaves_no_granule_list(tmp_path, monkeypatch) -> None:
    """The list rides in the batch's one commit, so a batch that fails leaves no slot and no list."""
    store = str(tmp_path / "s.zarr")
    _write(store, ["2024-06-01"], ON)

    def boom(*_args, **_kwargs):
        raise RuntimeError("window write failed")

    monkeypatch.setattr(zs, "to_icechunk", boom)
    with pytest.raises(RuntimeError, match="window write failed"):
        _write(store, ["2024-06-02"], ON)
    root = open_store_as_zarr_group(store)
    assert root[GRANULES_ARRAY].shape == root["time"].shape == (1,)


def test_a_store_without_the_record_has_no_granule_array(tmp_path) -> None:
    store = str(tmp_path / "s.zarr")
    _write(store, ["2024-06-01"], OFF)
    root = open_store_as_zarr_group(store)
    assert GRANULES_ARRAY not in root and SOURCE_INDEX_VAR not in root


@pytest.mark.parametrize(("seeded", "resumed"), [(ON, OFF), (OFF, ON)], ids=["on-then-off", "off-then-on"])
def test_a_store_keeps_the_setting_it_was_seeded_with(tmp_path, seeded, resumed) -> None:
    """Off cannot leave an on store's list short, and on cannot claim a record an off store lacks."""
    store = str(tmp_path / "s.zarr")
    _write(store, ["2024-06-01"], seeded)
    with pytest.raises(ConfigMismatchError, match="provenance_version"):
        _write(store, ["2024-06-02"], resumed)


def test_a_store_from_before_the_switch_resumes_with_it_off(tmp_path) -> None:
    """Such a store was written without the record, which is what off means."""
    store = str(tmp_path / "s.zarr")
    _write(store, ["2024-06-01"], OFF)
    assert "provenance_version" not in open_store(store).attrs["_manifest"]
    _write(store, ["2024-06-02"], OFF)


def test_the_granule_lists_must_match_the_dates(tmp_path) -> None:
    with pytest.raises(ValueError, match="one granule list per date"):
        write_days_windows(
            str(tmp_path / "s.zarr"),
            [(_day("2024-06-01", 1), list(WINDOWS))],
            roi=_Roi(),
            manifest=ON,
            baselines={},
            tile_id="roi.zarr",
            crs="EPSG:32601",
            chunks=CHUNKS,
            granules=[],
        )


def test_a_skipped_day_waits_for_the_mosaic_and_a_retry_replaces_its_list(tmp_path) -> None:
    """Skipped before the first write, a day is committed once that write creates the mosaic, off
    the time axis; a day reaching no live window is not listed; listing a day again replaces it.
    """
    store = str(tmp_path / "s.zarr")
    source = {"provider": "p", "collection": "c", "catalog_url": "https://cat", "collection_id": "c"}
    recorder = SkippedDays(store, source, exists=False, s3_region=None)

    def copies(day: str, granule: str) -> list:
        return normalize_to_solar_day([fake_stac_item(day, item_id=granule)], mid_longitude=0.0)

    recorder.record("2024-01-01", copies("2024-01-01", "first-attempt"), "coverage")
    recorder.record("2024-01-02", copies("2024-01-02", "over-the-sea"), "no-live-window")
    _write(store, ["2024-01-03"], ON)
    recorder.written()
    recorder.record("2024-01-01", copies("2024-01-01", "retry"), "unreadable")

    root = open_store_as_zarr_group(store)
    assert root["time"].shape == (1,), "the time axis holds only the written day"
    assert root[SKIPPED_DAYS_ARRAY][:].tolist() == ["2024-01-01"]
    (listed,) = json.loads(str(root[SKIPPED_GRANULES_ARRAY][0]))
    assert (listed["granule_id"], listed["status"]) == ("retry", "unreadable")
