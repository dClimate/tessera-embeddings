"""write_day_windows: the cropped counterpart of write_dataset.

Pins the bookkeeping contract the relocation trace flagged: same attr set as
write_dataset on create, append-path merge semantics (baselines union, doy
concat, last_appended bump), manifest validated before anything is written, one
snapshot per date, and windows landing while everything outside stays fill.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from tessera_embeddings.errors import ConfigMismatchError
from tessera_embeddings.storage.manifest import IngestManifest, extract_manifest
from tessera_embeddings.storage.zarr_store import (
    _open_repo,
    get_existing_dates,
    open_store_as_zarr_group,
    write_day_windows,
)

CHUNKS = {"time": 1, "northing": 4, "easting": 4}
GEOBOX = GeoBox((8, 8), Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0), "EPSG:32601")
MANIFEST = IngestManifest(roi_manifest_hash="abc")


class _Roi:
    geobox = GEOBOX
    height = 8
    width = 8


def _day_ds(date: str, band_val: int) -> xr.Dataset:
    """A lazy full-extent one-date dataset, dask-backed like odc.stac.load's."""
    shape = (1, 8, 8)
    return xr.Dataset(
        {
            "band": (("time", "northing", "easting"), da.full(shape, band_val, dtype=np.uint16, chunks=(1, 4, 4))),
            "scl": (("time", "northing", "easting"), da.full(shape, 4, dtype=np.uint8, chunks=(1, 4, 4))),
        },
        coords={"time": np.array([np.datetime64(date, "ns")])},
    )


def _write(store: str, date: str, band_val: int, windows=((0, 4, 0, 8),)) -> None:
    write_day_windows(
        store,
        _day_ds(date, band_val),
        list(windows),
        roi=_Roi(),
        manifest=MANIFEST,
        baselines={date: 5},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks=CHUNKS,
    )


def test_first_date_seeds_then_windows_land_and_rest_stays_fill(tmp_path):
    store = str(tmp_path / "reflectance.zarr")
    _write(store, "2024-06-01", 7, windows=[(0, 4, 0, 8), (4, 8, 4, 8)])

    g = open_store_as_zarr_group(store)
    band = np.asarray(g["band"])
    assert (band[0, 0:4, 0:8] == 7).all() and (band[0, 4:8, 4:8] == 7).all()
    assert (band[0, 4:8, 0:4] == 0).all()  # outside every window: seed fill, i.e. today's zeroed value
    # The attr set write_dataset creates, present at seed time.
    attrs = dict(g.attrs)
    assert attrs["tile_id"] == "roi.zarr" and attrs["crs"] == "EPSG:32601"
    assert attrs["baselines_applied"] == {"2024-06-01": 5}
    assert attrs["doy"] == [153]
    assert extract_manifest(attrs) is not None


def test_later_dates_are_one_snapshot_each_with_merged_attrs(tmp_path):
    store = str(tmp_path / "reflectance.zarr")
    _write(store, "2024-06-01", 7)
    before = len(list(_open_repo(store).ancestry(branch="main")))
    _write(store, "2024-06-11", 9, windows=[(0, 4, 0, 8), (4, 8, 0, 8)])

    assert len(list(_open_repo(store).ancestry(branch="main"))) == before + 1  # windows + attrs: ONE commit
    assert get_existing_dates(store) == {"2024-06-01", "2024-06-11"}
    g = open_store_as_zarr_group(store)
    attrs = dict(g.attrs)
    assert attrs["baselines_applied"] == {"2024-06-01": 5, "2024-06-11": 5}
    assert attrs["doy"] == [153, 163]  # concat in append order, like write_dataset
    assert attrs["last_appended"] != attrs["created_at"]
    band = np.asarray(g["band"])
    assert (band[0, 0:4, :] == 7).all() and (band[1] == 9).all()


def test_manifest_mismatch_fails_before_any_write(tmp_path):
    store = str(tmp_path / "reflectance.zarr")
    _write(store, "2024-06-01", 7)
    bad = IngestManifest(roi_manifest_hash="DIFFERENT")
    with pytest.raises(ConfigMismatchError):
        write_day_windows(
            store,
            _day_ds("2024-06-11", 9),
            [(0, 4, 0, 8)],
            roi=_Roi(),
            manifest=bad,
            baselines={"2024-06-11": 5},
            tile_id="roi.zarr",
            crs="EPSG:32601",
            chunks=CHUNKS,
        )
    assert get_existing_dates(store) == {"2024-06-01"}  # nothing committed


def test_multi_date_dataset_refused(tmp_path):
    ds = xr.concat([_day_ds("2024-06-01", 1), _day_ds("2024-06-11", 2)], dim="time")
    with pytest.raises(ValueError, match="one date per call"):
        write_day_windows(
            str(tmp_path / "s.zarr"),
            ds,
            [(0, 4, 0, 8)],
            roi=_Roi(),
            manifest=None,
            baselines={},
            tile_id="t",
            crs="EPSG:32601",
            chunks=CHUNKS,
        )
