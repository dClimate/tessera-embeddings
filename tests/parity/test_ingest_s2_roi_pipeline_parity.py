"""Parity: ``pipeline_dates`` must not change a single byte of the store.

The unit tests pin the loop's counters and log lines; this pins the artefact. It
drives the real domain function against a real ``LocalCluster`` and real Icechunk
stores, twice over the same toy dates, and compares the two stores for exact
equality. Only the STAC band load is synthetic — it is what makes the run
deterministic and offline, and what lets a date be forced to FAIL the coverage
gate in the middle of the run.

That mid-run gate failure is the case worth the fixture: it is where the pipeline
is draining a skip while a write is in flight, so a date written twice, dropped,
or committed out of order would show up here as a store mismatch rather than as
a subtly wrong mosaic six hours into a campaign.

No network and no cassette: unlike its sibling module this one never reaches
Earth Search, so it is a ``parity`` test but not an ``integration`` one.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from dask.distributed import Client

from tessera_embeddings.config.satellites import S2_SCL_INVALID_CLASSES
from tessera_embeddings.ingest import s2_roi
from tessera_embeddings.storage.zarr_store import get_existing_dates, open_store
from tests.parity.helpers import assert_zarr_equivalent

SIZE = 64
BANDS = ("blue", "green", "nir")
VALID_CLASS = next(c for c in range(12) if c not in S2_SCL_INVALID_CLASSES)
INVALID_CLASS = sorted(S2_SCL_INVALID_CLASSES)[0]

# The gate-failing date sits in the MIDDLE, between dates that must still be written.
DATES = ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
GATE_FAILS_ON = "2024-01-03"

# Denver-ish 10 m UTM 13N grid; only its internal consistency matters here.
TRANSFORM = [10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0]
CRS = "EPSG:32613"


def _stage_roi(tmp_path: Path) -> Path:
    """Write an all-land ROI mask with the attrs ``read_roi_metadata`` requires."""
    path = tmp_path / "roi.zarr"
    array = zarr.open(str(path), mode="w", shape=(SIZE, SIZE), chunks=(SIZE, SIZE), dtype="bool")
    array[:] = True
    array.attrs["crs"] = CRS
    array.attrs["transform"] = TRANSFORM
    array.attrs["bbox_wgs84"] = [-105.0, 39.0, -104.99, 39.01]
    return path


def _toy_day(date: str) -> xr.Dataset:
    """One date's mosaic, deterministic in the date so both runs load identical pixels.

    ``GATE_FAILS_ON`` gets all-invalid SCL, which is what drops it at the gate; every
    other date is fully valid and must be written.
    """
    # Seeded from the date itself, not from hash(), which is salted per process:
    # the two runs must see the same pixels, and so must a re-run of this test.
    rng = np.random.default_rng(int(date.replace("-", "")))
    valid = date != GATE_FAILS_ON
    data = {
        band: (
            ("time", "northing", "easting"),
            da.from_array(rng.integers(100, 5000, size=(1, SIZE, SIZE), dtype="uint16"), chunks=(1, SIZE, SIZE)),
        )
        for band in BANDS
    }
    scl = np.full((1, SIZE, SIZE), VALID_CLASS if valid else INVALID_CLASS, dtype="uint8")
    data["scl"] = (("time", "northing", "easting"), da.from_array(scl, chunks=(1, SIZE, SIZE)))
    return xr.Dataset(
        data,
        coords={
            "time": [np.datetime64(f"{date}T10:00:00", "ns")],
            "northing": TRANSFORM[5] + TRANSFORM[4] * np.arange(SIZE),
            "easting": TRANSFORM[2] + TRANSFORM[0] * np.arange(SIZE),
        },
    )


def _item(date: str):
    """The minimum of a STAC item the date loop touches: its datetime and cloud cover."""
    return SimpleNamespace(datetime=datetime.fromisoformat(f"{date}T10:00:00"), properties={"eo:cloud_cover": 0.0})


def _ingest(
    *,
    roi_zarr: Path,
    store_path: Path,
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_dates: bool,
):
    """Run the real domain ingest over the toy dates, with only the band load stubbed."""
    monkeypatch.setattr(s2_roi, "query_stac_items", lambda **_kwargs: ([_item(d) for d in DATES], {}))
    monkeypatch.setattr(
        s2_roi,
        "load_stac_items",
        lambda day_items, **_kwargs: _toy_day(day_items[0].datetime.strftime("%Y-%m-%d")),
    )
    log = logging.getLogger(f"parity-s2-pipeline-{pipeline_dates}")
    log.setLevel(logging.INFO)
    return s2_roi.ingest_s2_roi_reflectance(
        roi_zarr_path=str(roi_zarr),
        start_date=DATES[0],
        end_date=DATES[-1],
        store_path=str(store_path),
        client=client,
        min_valid_coverage=50.0,
        log=log,
        stream_stac_monthly=False,
        pipeline_dates=pipeline_dates,
    )


@pytest.mark.parity
def test_pipelining_dates_produces_an_identical_store(
    tmp_path: Path,
    parity_cluster: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serial and pipelined runs of the same dates must agree byte for byte."""
    roi_zarr = _stage_roi(tmp_path)
    serial_store, pipelined_store = tmp_path / "serial", tmp_path / "pipelined"

    serial = _ingest(
        roi_zarr=roi_zarr,
        store_path=serial_store,
        client=parity_cluster,
        monkeypatch=monkeypatch,
        pipeline_dates=False,
    )
    pipelined = _ingest(
        roi_zarr=roi_zarr,
        store_path=pipelined_store,
        client=parity_cluster,
        monkeypatch=monkeypatch,
        pipeline_dates=True,
    )

    # The counters first: an identical store reached by writing a different set of
    # dates would mean the toy data, not the pipeline, was doing the agreeing.
    assert serial.status == pipelined.status == "success"
    assert serial.dates_processed == pipelined.dates_processed == len(DATES) - 1
    assert serial.dates_filtered_coverage == pipelined.dates_filtered_coverage == 1

    # An equality check over two EMPTY stores passes; this is what stops that from
    # being mistaken for parity.
    written = open_store(str(pipelined_store / "reflectance.zarr"))
    try:
        assert sorted(written.data_vars) == sorted([*BANDS, "scl"])
        assert written.sizes["time"] == len(DATES) - 1
    finally:
        written.close()

    assert_zarr_equivalent(pipelined_store / "reflectance.zarr", serial_store / "reflectance.zarr")


@pytest.mark.parity
def test_the_gate_failing_date_is_absent_from_both_stores(
    tmp_path: Path,
    parity_cluster: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mid-run skip must be a real skip, not a date the comparison never noticed.

    Without this, a bug that wrote every date under both modes would still pass the
    equality check above — the two stores would simply be identically wrong.
    """
    roi_zarr = _stage_roi(tmp_path)
    store = tmp_path / "pipelined"
    _ingest(
        roi_zarr=roi_zarr,
        store_path=store,
        client=parity_cluster,
        monkeypatch=monkeypatch,
        pipeline_dates=True,
    )

    dates = get_existing_dates(str(store / "reflectance.zarr"))
    assert dates == {d for d in DATES if d != GATE_FAILS_ON}
