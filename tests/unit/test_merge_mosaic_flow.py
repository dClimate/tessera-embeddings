"""Unit test for the ``merge-mosaic`` Prefect flow.

The flow is a thin wrapper over ``region_merge.merge_stores`` (tested directly in
``test_region_merge.py``); this verifies the flow's own responsibilities: the
``var_dtypes`` string → ``np.dtype`` conversion across the Prefect parameter
boundary, and delegation to the driver.

Invoked via ``merge_mosaic.fn(...)`` (bypassing the Prefect engine) with
``get_run_logger`` and ``read_roi_metadata`` monkeypatched — the same idiom the
parity flow tests use — so no Prefect runtime or ROI store is needed.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

pytest.importorskip("prefect")

from tessera_embeddings.ingest.roi import ROIMetadata
from tessera_embeddings.orchestration.prefect.flows import merge_mosaic as flow_mod
from tessera_embeddings.storage.zarr_store import open_store, write_dataset

_CRS = "EPSG:5070"
_MASTER = GeoBox(shape=(80, 100), affine=Affine(10, 0, 500000, 0, -10, 4000000), crs=_CRS)


def _roi() -> ROIMetadata:
    return ROIMetadata(bbox_wgs84=(-100.0, 40.0, -99.0, 41.0), native_crs=_CRS, geobox=_MASTER, width=100, height=80)


def _write_feature(path: str, box: GeoBox, dates: list[str], value: int) -> None:
    coords = box.coordinates
    ds = xr.Dataset(
        {"VV": (("time", "northing", "easting"), np.full((len(dates), box.shape[0], box.shape[1]), value, "uint16"))},
        coords={
            "time": np.array(dates, dtype="datetime64[ns]"),
            "northing": coords["y"].values,
            "easting": coords["x"].values,
        },
    )
    write_dataset(path, ds, tile_id="x", baselines={}, chunks={"time": 1, "northing": 40, "easting": 40}, crs=_CRS)


def test_merge_mosaic_flow_end_to_end(tmp_path, monkeypatch):
    """The flow converts string dtypes, reads the ROI, and delegates to merge_stores —
    proven end to end by a real (tiny) merge whose seed uint16 must round-trip.
    """
    monkeypatch.setattr(flow_mod, "get_run_logger", lambda: logging.getLogger("test"))
    monkeypatch.setattr(flow_mod, "read_roi_metadata", lambda _path: _roi())

    pa = str(tmp_path / "a.zarr")
    _write_feature(pa, _MASTER[10:30, 20:50], ["2024-01-15"], 11)
    master = str(tmp_path / "master.zarr")

    summary = flow_mod.merge_mosaic.fn(
        master_path=master,
        feature_paths=[pa],
        roi_zarr_path="unused://roi",
        var_dtypes={"VV": "uint16"},  # string, as it crosses the Prefect param boundary
        tile_id="m",
    )

    assert summary["merged"] == {pa: 1}
    assert summary["n_dates"] == 1
    out = open_store(master)["VV"]
    assert (out.sel(time=np.datetime64("2024-01-15", "ns")).values[10:30, 20:50] == 11).all()
