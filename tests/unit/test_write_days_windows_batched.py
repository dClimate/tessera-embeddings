"""write_days_windows: several dates' windows computed together, ONE commit for the batch.

Pins what makes the batch form safe to trust:

* byte-identity — a batch of k dates produces exactly the store k single-date calls
  produce, arrays and merged attrs alike, under both the sequential and the
  overlapped compute;
* atomicity at batch granularity — one snapshot per batch, and a failure mid-batch
  commits none of its dates (the no-date-before-its-pixels invariant, batch-sized);
* the guards — dates out of order or duplicated refuse before anything is written,
  and an empty batch is a no-op rather than an empty commit.

The batch is one commit BY CONSTRUCTION (per-date sessions would conflict on the
time-axis resize), so there is deliberately no test asserting per-date snapshots
from a batch — that shape cannot exist.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import (
    _open_repo,
    get_existing_dates,
    open_store,
    write_day_windows,
    write_days_windows,
)

CHUNKS = {"time": 1, "northing": 4, "easting": 4}
GEOBOX = GeoBox((8, 8), Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0), "EPSG:32601")
MANIFEST = IngestManifest(roi_manifest_hash="abc")
WINDOWS = [(0, 4, 0, 8), (4, 8, 0, 4)]


class _Roi:
    geobox = GEOBOX
    height = 8
    width = 8


def _day_ds(date: str, band_val: int) -> xr.Dataset:
    shape = (1, 8, 8)
    return xr.Dataset(
        {
            "band": (("time", "northing", "easting"), da.full(shape, band_val, dtype=np.uint16, chunks=(1, 4, 4))),
            "scl": (("time", "northing", "easting"), da.full(shape, 4, dtype=np.uint8, chunks=(1, 4, 4))),
        },
        coords={"time": np.array([np.datetime64(date, "ns")])},
    )


def _batch(store: str, dates: list[tuple[str, int]], *, parallel: bool = False) -> None:
    write_days_windows(
        store,
        [(_day_ds(d, v), list(WINDOWS)) for d, v in dates],
        roi=_Roi(),
        manifest=MANIFEST,
        baselines={d: 5 for d, _ in dates},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks=CHUNKS,
        parallel_windows=parallel,
    )


def _single(store: str, date: str, band_val: int) -> None:
    write_day_windows(
        store,
        _day_ds(date, band_val),
        list(WINDOWS),
        roi=_Roi(),
        manifest=MANIFEST,
        baselines={date: 5},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks=CHUNKS,
    )


def _snapshots(store: str) -> int:
    return len(list(_open_repo(store).ancestry(branch="main")))


DATES = [("2024-06-01", 7), ("2024-06-02", 9), ("2024-06-03", 11)]


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "overlapped"])
def test_batch_matches_singles_byte_for_byte(tmp_path, parallel):
    """The batch's arrays AND merged attrs equal k single-date writes'."""
    ref = str(tmp_path / "ref.zarr")
    got = str(tmp_path / "got.zarr")
    for d, v in DATES:
        _single(ref, d, v)
    _batch(got, DATES, parallel=parallel)

    a, b = open_store(ref), open_store(got)
    assert list(a.time.values) == list(b.time.values)
    for var in ("band", "scl"):
        np.testing.assert_array_equal(a[var].values, b[var].values)
    for key in ("baselines_applied", "doy", "tile_id", "crs"):
        assert a.attrs[key] == b.attrs[key], key
    assert get_existing_dates(got) == {d for d, _ in DATES}


def test_batch_is_one_snapshot(tmp_path):
    store = str(tmp_path / "s.zarr")
    _batch(store, DATES)
    after_first = _snapshots(store)
    _batch(store, [("2024-06-04", 13), ("2024-06-05", 15)])
    # exactly ONE commit per batch, whatever its size
    assert _snapshots(store) == after_first + 1


def test_out_of_order_dates_refused_before_any_write(tmp_path):
    store = str(tmp_path / "s.zarr")
    with pytest.raises(ValueError, match="strictly increasing"):
        _batch(store, [("2024-06-02", 9), ("2024-06-01", 7)])
    with pytest.raises(Exception):  # noqa: B017 — store must simply not exist
        _snapshots(store)


def test_duplicate_date_in_batch_refused(tmp_path):
    store = str(tmp_path / "s.zarr")
    with pytest.raises(ValueError, match="strictly increasing"):
        _batch(store, [("2024-06-01", 7), ("2024-06-01", 9)])


def test_empty_batch_is_a_noop(tmp_path):
    store = str(tmp_path / "s.zarr")
    write_days_windows(
        store,
        [],
        roi=_Roi(),
        manifest=MANIFEST,
        baselines={},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks=CHUNKS,
    )
    with pytest.raises(Exception):  # noqa: B017 — no store, no commit, nothing
        _snapshots(store)


def test_failed_batch_commits_none_of_its_dates(tmp_path, monkeypatch):
    """Atomicity at batch granularity: a crash mid-batch leaves every date retryable."""
    store = str(tmp_path / "s.zarr")
    _batch(store, [("2024-05-01", 3)])  # seed + one committed date
    before = _snapshots(store)

    import tessera_embeddings.storage.zarr_store as zs

    real = zs.to_icechunk
    calls = {"n": 0}

    def _explode_on_third(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # first date lands, second date's first window dies
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(zs, "to_icechunk", _explode_on_third)
    with pytest.raises(RuntimeError, match="boom"):
        _batch(store, DATES)

    assert _snapshots(store) == before, "a failed batch must commit nothing"
    assert get_existing_dates(store) == {"2024-05-01"}
