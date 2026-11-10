"""write_day_windows and write_days_windows: the cropped counterparts of write_dataset.

Pins the bookkeeping contract the relocation trace flagged: same attr set as
write_dataset on create, append-path merge semantics (baselines union, doy
concat, last_appended bump), manifest validated before anything is written, one
snapshot per date, and windows landing while everything outside stays fill.

The batch form (several dates' windows computed together, ONE commit for the batch) is
pinned for what makes it safe to trust: byte-identity with k single-date calls under both
the sequential and the overlapped compute, atomicity at batch granularity, and its guards.
The batch is one commit BY CONSTRUCTION (per-date sessions would conflict on the time-axis
resize), so there is deliberately no test asserting per-date snapshots from a batch.
"""

from __future__ import annotations

import sys

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from odc.geo.geobox import GeoBox

from tessera_embeddings.config.ingest import INGEST_INNER_CHUNKS
from tessera_embeddings.errors import ConfigMismatchError
from tessera_embeddings.storage import zarr_store
from tessera_embeddings.storage.empty_store import create_empty_store
from tessera_embeddings.storage.manifest import MIXED_CODE_IDENTITIES_ATTR, IngestManifest, extract_manifest
from tessera_embeddings.storage.zarr_store import (
    get_existing_dates,
    open_repo,
    open_store,
    open_store_as_zarr_group,
    write_day_windows,
    write_days_windows,
)

CHUNKS = {"time": 1, "northing": 4, "easting": 4}
GEOBOX = GeoBox((8, 8), Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0), "EPSG:32601")
MANIFEST = IngestManifest(roi_manifest_hash="abc")
WINDOWS = [(0, 4, 0, 8), (4, 8, 0, 4)]  # two disjoint windows, one partial row


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


def _write(
    store: str, date: str, band_val: int, windows=((0, 4, 0, 8),), manifest=MANIFEST, *, parallel: bool = False
) -> None:
    write_day_windows(
        store,
        _day_ds(date, band_val),
        list(windows),
        roi=_Roi(),
        manifest=manifest,
        baselines={date: 5},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks=CHUNKS,
        parallel_windows=parallel,
    )


def _snapshots(store: str) -> int:
    return len(list(open_repo(store).ancestry(branch="main")))


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
    before = len(list(open_repo(store).ancestry(branch="main")))
    _write(store, "2024-06-11", 9, windows=[(0, 4, 0, 8), (4, 8, 0, 8)])

    assert len(list(open_repo(store).ancestry(branch="main"))) == before + 1  # windows + attrs: ONE commit
    assert get_existing_dates(store) == {"2024-06-01", "2024-06-11"}
    g = open_store_as_zarr_group(store)
    attrs = dict(g.attrs)
    assert attrs["baselines_applied"] == {"2024-06-01": 5, "2024-06-11": 5}
    assert attrs["doy"] == [153, 163]  # concat in append order, like write_dataset
    assert attrs["last_appended"] != attrs["created_at"]
    band = np.asarray(g["band"])
    assert (band[0, 0:4, :] == 7).all() and (band[1] == 9).all()


def test_an_override_append_records_both_code_identities_on_the_store(tmp_path):
    """The safety argument for permitting the append: the store says, durably, that it happened."""
    store = str(tmp_path / "reflectance.zarr")
    old = IngestManifest(roi_manifest_hash="abc", ingest_code_identity="ing-A")
    new = IngestManifest(roi_manifest_hash="abc", ingest_code_identity="ing-B")
    forced = IngestManifest(roi_manifest_hash="abc", ingest_code_identity="ing-B", allow_ingest_code_mismatch=True)

    _write(store, "2024-06-01", 7, manifest=old)
    with pytest.raises(ConfigMismatchError):
        _write(store, "2024-06-11", 9, manifest=new)  # off by default, refused before any write
    assert get_existing_dates(store) == {"2024-06-01"}, "a refused append must commit nothing"
    assert MIXED_CODE_IDENTITIES_ATTR not in dict(open_store_as_zarr_group(store).attrs)

    _write(store, "2024-06-11", 9, manifest=forced)

    assert get_existing_dates(store) == {"2024-06-01", "2024-06-11"}
    assert dict(open_store_as_zarr_group(store).attrs)[MIXED_CODE_IDENTITIES_ATTR] == ["ing-A", "ing-B"]


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


def test_crash_between_seed_and_first_windows_is_retryable(tmp_path):
    """The seed commits an EMPTY axis, so no date exists before its pixels do.

    Review catch (da-code-reviewer, HIGH): seeding times=[first_date] committed
    the date before its windows — a crash in between left an all-fill timestep
    that get_existing_dates reported as ingested, so the retry hit the
    duplicate-date guard and the STAC dedupe filtered the date forever. Now a
    crash after the seed leaves a zero-date store, and the retry appends the
    date atomically with its windows.
    """
    store = str(tmp_path / "reflectance.zarr")
    # Simulate the crash: store seeded, no date committed.
    create_empty_store(
        store,
        roi=_Roi(),
        times=np.array([], dtype="datetime64[ns]"),
        var_dtypes={"band": np.dtype("uint16"), "scl": np.dtype("uint8")},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks=CHUNKS,
        manifest=MANIFEST,
    )
    assert get_existing_dates(store) == set()  # the dedupe sees nothing ingested

    _write(store, "2024-06-01", 7)  # the retry
    assert get_existing_dates(store) == {"2024-06-01"}
    g = open_store_as_zarr_group(store)
    assert (np.asarray(g["band"])[0, 0:4, :] == 7).all()
    assert dict(g.attrs)["doy"] == [153]


# --- the overlapped window write -------------------------------------------------
#
# One dask compute for all of a date's windows instead of one blocking compute per
# window. Equivalence with the sequential path is the shipping gate (the batch
# byte-identity test below compares it against sequential single-date writes): the
# parallel path lifts icechunk's own fork/merge machinery, and these tests pin that the
# lift changes nothing about what lands in the store or when a commit happens.


def test_parallel_failure_commits_nothing_and_retry_succeeds(tmp_path):
    """A poisoned window fails the single compute; the date never lands; a clean
    retry writes it — the abandoned-session contract, unchanged by overlap.
    """
    store = str(tmp_path / "s")
    _write(store, "2024-01-01", 7, [(0, 4, 0, 8)], parallel=True)
    before = _snapshots(store)

    def _poison(block):
        raise RuntimeError("poisoned window")

    bad = _day_ds("2024-01-02", 3)
    bad["band"] = (("time", "northing", "easting"), bad["band"].data.map_blocks(_poison, dtype=np.uint16))
    with pytest.raises(RuntimeError, match="poisoned window"):
        write_day_windows(
            store,
            bad,
            [(0, 4, 0, 8), (4, 8, 0, 4)],
            roi=_Roi(),
            manifest=MANIFEST,
            baselines={"2024-01-02": 5},
            tile_id="roi.zarr",
            crs="EPSG:32601",
            chunks=CHUNKS,
            parallel_windows=True,
        )
    assert _snapshots(store) == before, "a failed parallel write must commit nothing"
    assert get_existing_dates(store) == {"2024-01-01"}

    _write(store, "2024-01-02", 9, [(0, 4, 0, 8)], parallel=True)  # clean retry
    assert get_existing_dates(store) == {"2024-01-01", "2024-01-02"}
    g = open_store_as_zarr_group(store)
    assert (g["band"][1, :4, :] == 9).all()


def test_parallel_falls_back_when_private_api_missing(tmp_path, monkeypatch, caplog):
    """Private-API drift degrades to the sequential path, not to a failure.

    Simulated by poisoning ``sys.modules`` so only FRESH imports of the icechunk
    xarray module fail: the overlapped helper imports at call time and must decline,
    while the sequential path's ``to_icechunk`` was bound at module load and keeps
    working — the same topology as a real drift, where the private symbol moves but
    the public API stays.
    """
    monkeypatch.setitem(sys.modules, "icechunk.xarray", None)
    store = str(tmp_path / "s")
    with caplog.at_level("WARNING"):
        _write(store, "2024-01-01", 7, WINDOWS, parallel=True)
    assert "writing sequentially" in caplog.text
    g = open_store_as_zarr_group(store)
    assert (g["band"][0, :4, :] == 7).all()
    assert (g["band"][0, 4:, :4] == 7).all()


def test_a_sharded_store_takes_whole_shard_windows_and_reads_back(tmp_path, monkeypatch):
    """With 2-px inner chunks the 4-px blocks become shards; windows land intact, date after date."""
    monkeypatch.setitem(INGEST_INNER_CHUNKS, "northing", 2)
    monkeypatch.setitem(INGEST_INNER_CHUNKS, "easting", 2)
    store = str(tmp_path / "reflectance.zarr")
    _write(store, "2024-06-01", 7, windows=[(0, 4, 0, 8), (4, 8, 4, 8)])
    _write(store, "2024-06-11", 9, windows=[(4, 8, 0, 4)])

    g = open_store_as_zarr_group(store)
    assert g["band"].chunks == (1, 2, 2) and g["band"].shards == (1, 4, 4)
    band = np.asarray(g["band"])
    assert (band[0, 0:4, :] == 7).all() and (band[0, 4:8, 4:8] == 7).all() and (band[0, 4:8, 0:4] == 0).all()
    assert (band[1, 4:8, 0:4] == 9).all() and (band[1, 0:4, :] == 0).all()


# --- the batch form ---------------------------------------------------------------

DATES = [("2024-06-01", 7), ("2024-06-02", 9), ("2024-06-03", 11)]


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


@pytest.mark.parametrize("parallel", [False, True], ids=["sequential", "overlapped"])
def test_batch_matches_singles_byte_for_byte(tmp_path, parallel):
    """The batch's arrays AND merged attrs equal k sequential single-date writes'."""
    ref = str(tmp_path / "ref.zarr")
    got = str(tmp_path / "got.zarr")
    for d, v in DATES:
        _write(ref, d, v, WINDOWS)
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


@pytest.mark.parametrize(
    "dates",
    [[("2024-06-02", 9), ("2024-06-01", 7)], [("2024-06-01", 7), ("2024-06-01", 9)]],
    ids=["out-of-order", "duplicate"],
)
def test_unordered_dates_refused_before_any_write(tmp_path, dates):
    store = str(tmp_path / "s.zarr")
    with pytest.raises(ValueError, match="strictly increasing"):
        _batch(store, dates)
    with pytest.raises(Exception):  # noqa: B017 — store must simply not exist
        _snapshots(store)


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

    real = zarr_store.to_icechunk
    calls = {"n": 0}

    def _explode_on_third(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # first date lands, second date's first window dies
            raise RuntimeError("boom")
        return real(*args, **kwargs)

    monkeypatch.setattr(zarr_store, "to_icechunk", _explode_on_third)
    with pytest.raises(RuntimeError, match="boom"):
        _batch(store, DATES)

    assert _snapshots(store) == before, "a failed batch must commit nothing"
    assert get_existing_dates(store) == {"2024-05-01"}
