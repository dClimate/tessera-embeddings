"""Parity: the S2 date loop's write modes must not change a single byte of the store.

Batching dates into one commit (``batch_dates``) and preparing the next date while one writes
(``pipeline_dates``) are both performance options, so each, and the two composed, must
produce exactly the store plain per-date writing does. The unit tests pin each mode's
mechanics; this pins the artefact. It drives the real domain function against a real
``LocalCluster`` and real Icechunk stores over the same toy dates and compares the stores for
exact equality. Only the STAC band load is synthetic — it is what makes the run deterministic
and offline, and what lets a date be forced to FAIL the coverage gate mid-run.

That mid-run gate failure is the case worth the fixture: a skipped date must not occupy a
batch slot, shift its neighbours into the wrong commit, or be written while the pipeline
drains it — and the trailing partial batch must still flush. No network and no cassette, so
this is a ``parity`` test but not an ``integration`` one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from dask.distributed import Client

from tessera_embeddings.config.ingest import SKIPPED_DAYS_ARRAY, SKIPPED_GRANULES_ARRAY
from tessera_embeddings.config.satellites import S2_SCL_INVALID_CLASSES
from tessera_embeddings.ingest import s2_roi
from tessera_embeddings.storage.zarr_store import get_existing_dates, open_repo, open_store
from tests.parity.helpers import assert_zarr_equivalent
from tests.stac_items import fake_stac_item

SIZE = 64
BANDS = ("blue", "green", "nir")
VALID_CLASS = next(c for c in range(12) if c not in S2_SCL_INVALID_CLASSES)
INVALID_CLASS = sorted(S2_SCL_INVALID_CLASSES)[0]

# The gate-failing date sits in the MIDDLE: at batch_dates=3 the passing dates batch as
# [01, 02, 04] and a flushed partial [05], so a skip is drained mid-batch.
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
    """One date's mosaic, deterministic in the date so every run loads identical pixels.

    ``GATE_FAILS_ON`` gets all-invalid SCL, which is what drops it at the gate; every
    other date is fully valid and must be written.
    """
    # Seeded from the date itself, not from hash(), which is salted per process:
    # every run must see the same pixels, and so must a re-run of this test.
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


def _ingest(
    *,
    roi_zarr: Path,
    store_path: Path,
    client: Client,
    monkeypatch: pytest.MonkeyPatch,
    **mode,
):
    """Run the real domain ingest over the toy dates, with only the band load stubbed.

    ``mode`` is the write mode under test (``batch_dates``, ``pipeline_dates``); without one the
    run is plain per-date writing, the baseline every mode is compared against.
    """
    mode = {"batch_dates": 1, "pipeline_dates": False, **mode}
    monkeypatch.setattr(s2_roi, "query_stac_items", lambda **_kwargs: ([fake_stac_item(d) for d in DATES], {}))
    monkeypatch.setattr(
        s2_roi,
        "load_stac_items",
        lambda day_items, **_kwargs: _toy_day(day_items[0].datetime.strftime("%Y-%m-%d")),
    )
    log = logging.getLogger(f"parity-s2-batch-{mode['batch_dates']}-pipeline-{mode['pipeline_dates']}")
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
        **mode,
    )


def _snapshots(store: Path) -> int:
    return len(list(open_repo(str(store / "reflectance.zarr")).ancestry(branch="main")))


@pytest.mark.parity
@pytest.mark.parametrize(
    ("mode", "commits"),
    [
        pytest.param({"batch_dates": 3}, 2, id="batched"),
        pytest.param({"pipeline_dates": True}, 4, id="pipelined"),
        # Where a preparation on the background thread overlaps a BATCH's write: the
        # arrangement most likely to reorder or drop a date.
        pytest.param({"batch_dates": 3, "pipeline_dates": True}, 2, id="batched-and-pipelined"),
    ],
)
def test_a_write_mode_produces_the_per_date_store(
    tmp_path: Path,
    parity_cluster: Client,
    monkeypatch: pytest.MonkeyPatch,
    mode: dict,
    commits: int,
) -> None:
    """A write mode and plain per-date writing must agree byte for byte, in ``commits`` commits."""
    roi_zarr = _stage_roi(tmp_path)
    plain_store, mode_store = tmp_path / "plain", tmp_path / "mode"
    run = {"roi_zarr": roi_zarr, "client": parity_cluster, "monkeypatch": monkeypatch}
    plain = _ingest(store_path=plain_store, **run)
    moded = _ingest(store_path=mode_store, **run, **mode)

    # Counters first: an identical store reached by writing a different set of dates would
    # mean the toy data, not the mode, was doing the agreeing.
    assert plain.status == moded.status == "success"
    assert plain.dates_processed == moded.dates_processed == len(DATES) - 1
    assert plain.dates_filtered_coverage == moded.dates_filtered_coverage == 1

    # An equality check over two EMPTY stores, or two that both wrote the skipped date,
    # passes; these are what stop either from being mistaken for parity.
    written = open_store(str(mode_store / "reflectance.zarr"))
    try:
        # The gated date is recorded off the time axis, with its granules (ingest/provenance.py).
        assert sorted(written.data_vars) == sorted([*BANDS, "scl", SKIPPED_DAYS_ARRAY, SKIPPED_GRANULES_ARRAY])
        assert written.sizes["time"] == len(DATES) - 1
    finally:
        written.close()
    assert get_existing_dates(str(mode_store / "reflectance.zarr")) == set(DATES) - {GATE_FAILS_ON}

    assert_zarr_equivalent(str(plain_store / "reflectance.zarr"), str(mode_store / "reflectance.zarr"))

    # The commit-granularity contract: per-date writing commits each date alone; a batch of
    # three commits [01, 02, 04] together and flushes [05].
    assert _snapshots(plain_store) - _snapshots(mode_store) == (len(DATES) - 1) - commits
