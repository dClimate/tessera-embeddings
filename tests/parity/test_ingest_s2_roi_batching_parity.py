"""Parity: ``batch_dates`` must not change a single byte of the store.

The unit tests pin the batch write's storage semantics (one snapshot per batch,
atomicity, ordering guards); this pins the artefact end to end. It drives the real
domain function against a real ``LocalCluster`` and real Icechunk stores, once per
mode over the same toy dates, and compares the stores for exact equality. Only the
STAC band load is synthetic — it is what makes the run deterministic and offline,
and what lets a date be forced to FAIL the coverage gate in the middle of a batch.

That mid-batch gate failure is the case worth the fixture: a skipped date must not
occupy a batch slot, shift its neighbours into the wrong commit, or change any
written byte — and the trailing partial batch must still flush. Both runs use the
cropped path, because the batched write is the windowed write lifted across dates
and has no full-extent form.
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
from tessera_embeddings.storage.zarr_store import _open_repo, get_existing_dates, open_store
from tests.parity.helpers import assert_zarr_equivalent

SIZE = 64
BANDS = ("blue", "green", "nir")
VALID_CLASS = next(c for c in range(12) if c not in S2_SCL_INVALID_CLASSES)
INVALID_CLASS = sorted(S2_SCL_INVALID_CLASSES)[0]

# The gate-failing date sits MID-BATCH at batch_dates=3: passing dates are
# 01, 02, 04, 05 -> batches [01, 02, 04] and the flushed partial [05].
DATES = ("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05")
GATE_FAILS_ON = "2024-01-03"
BATCH = 3

TRANSFORM = [10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0]
CRS = "EPSG:32613"


def _stage_roi(tmp_path: Path) -> Path:
    path = tmp_path / "roi.zarr"
    array = zarr.open(str(path), mode="w", shape=(SIZE, SIZE), chunks=(SIZE, SIZE), dtype="bool")
    array[:] = True
    array.attrs["crs"] = CRS
    array.attrs["transform"] = TRANSFORM
    array.attrs["bbox_wgs84"] = [-105.0, 39.0, -104.99, 39.01]
    return path


def _toy_day(date: str) -> xr.Dataset:
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
    return SimpleNamespace(datetime=datetime.fromisoformat(f"{date}T10:00:00"), properties={"eo:cloud_cover": 0.0})


def _ingest(
    *,
    roi_zarr: Path,
    store_path: Path,
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    batch_dates: int,
    pipeline_dates: bool = False,
):
    monkeypatch.setattr(s2_roi, "query_stac_items", lambda **_kwargs: ([_item(d) for d in DATES], {}))
    monkeypatch.setattr(
        s2_roi,
        "load_stac_items",
        lambda day_items, **_kwargs: _toy_day(day_items[0].datetime.strftime("%Y-%m-%d")),
    )
    log = logging.getLogger(f"parity-s2-batch-{batch_dates}")
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
        crop_to_live_windows=True,
        batch_dates=batch_dates,
        pipeline_dates=pipeline_dates,
    )


def _snapshots(store: Path) -> int:
    return len(list(_open_repo(str(store / "reflectance.zarr")).ancestry(branch="main")))


@pytest.mark.parity
def test_batched_dates_produce_an_identical_store(
    tmp_path: Path,
    parity_cluster: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-date and batched runs of the same dates must agree byte for byte."""
    roi_zarr = _stage_roi(tmp_path)
    serial_store, batched_store = tmp_path / "serial", tmp_path / "batched"

    serial = _ingest(
        roi_zarr=roi_zarr, store_path=serial_store, client=parity_cluster, monkeypatch=monkeypatch, batch_dates=1
    )
    batched = _ingest(
        roi_zarr=roi_zarr, store_path=batched_store, client=parity_cluster, monkeypatch=monkeypatch, batch_dates=BATCH
    )

    # Counters first: an identical store reached by writing a different set of
    # dates would mean the toy data, not the batching, was doing the agreeing.
    assert serial.status == batched.status == "success"
    assert serial.dates_processed == batched.dates_processed == len(DATES) - 1
    assert serial.dates_filtered_coverage == batched.dates_filtered_coverage == 1

    written = open_store(str(batched_store / "reflectance.zarr"))
    assert written.sizes["time"] == len(DATES) - 1
    written.close()
    assert get_existing_dates(str(batched_store / "reflectance.zarr")) == set(DATES) - {GATE_FAILS_ON}

    assert_zarr_equivalent(
        str(serial_store / "reflectance.zarr"),
        str(batched_store / "reflectance.zarr"),
    )

    # The commit-granularity contract, both directions: per-date mode commits each
    # date alone; batched mode commits [01, 02, 04] together and flushes [05].
    assert _snapshots(serial_store) - _snapshots(batched_store) == (len(DATES) - 1) - 2


def test_batch_dates_validation_rejects_bad_combinations(tmp_path: Path) -> None:
    """The refusals fire before any cluster or store work."""
    kwargs = dict(
        roi_zarr_path=str(tmp_path / "roi.zarr"),
        start_date="2024-01-01",
        end_date="2024-01-02",
        store_path=str(tmp_path / "s"),
        client=None,  # never reached: validation refuses first
    )
    with pytest.raises(ValueError, match="batch_dates must be >= 1"):
        s2_roi.ingest_s2_roi_reflectance(**kwargs, batch_dates=0)
    with pytest.raises(ValueError, match="requires crop_to_live_windows"):
        s2_roi.ingest_s2_roi_reflectance(**kwargs, batch_dates=2)


@pytest.mark.parity
def test_batching_composed_with_pipelining_is_identical(
    tmp_path: Path,
    parity_cluster: Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batching + pipelining must equal plain per-date writing, byte for byte.

    The composition is where a preparation running on the background thread overlaps a
    BATCH's write, with the look-ahead sized to the batch. That is the arrangement most
    likely to reorder or drop a date — and the gate-failing date mid-batch means a skip
    is in flight while a batch is being written, which is exactly the race worth pinning.
    """
    roi_zarr = _stage_roi(tmp_path)
    plain, composed = tmp_path / "plain", tmp_path / "composed"

    a = _ingest(roi_zarr=roi_zarr, store_path=plain, client=parity_cluster, monkeypatch=monkeypatch, batch_dates=1)
    b = _ingest(
        roi_zarr=roi_zarr,
        store_path=composed,
        client=parity_cluster,
        monkeypatch=monkeypatch,
        batch_dates=BATCH,
        pipeline_dates=True,
    )

    assert a.status == b.status == "success"
    assert a.dates_processed == b.dates_processed == len(DATES) - 1
    assert a.dates_filtered_coverage == b.dates_filtered_coverage == 1
    assert get_existing_dates(str(composed / "reflectance.zarr")) == set(DATES) - {GATE_FAILS_ON}
    assert_zarr_equivalent(str(plain / "reflectance.zarr"), str(composed / "reflectance.zarr"))
