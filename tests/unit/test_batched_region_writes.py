"""The per-date write unit: one session, N window writes, one commit.

Pins the properties the live-window ingest path stands on — one snapshot per
date regardless of window count, atomicity on failure, the time-axis append
being visible to the session's own writes, and the appended date round-tripping
through the production reader (`get_existing_dates`), because that reader is
what dedupes the next run's STAC query.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera_embeddings.storage.empty_store import VarSpec, create_empty_store_from_coords
from tessera_embeddings.storage.zarr_store import (
    _open_repo,
    batched_region_writes,
    get_existing_dates,
    open_store_as_zarr_group,
    read_time_values,
)

D1, D2 = np.datetime64("2024-06-01", "ns"), np.datetime64("2024-06-11", "ns")


@pytest.fixture()
def store(tmp_path) -> str:
    """A seeded mosaic-shaped store: one date, 8x8 px, 4-px chunks, two dtypes."""
    path = str(tmp_path / "mosaic.zarr")
    create_empty_store_from_coords(
        path,
        coords={"time": np.array([D1]), "northing": np.arange(8.0), "easting": np.arange(8.0)},
        var_specs={
            "band": VarSpec(dims=("time", "northing", "easting"), dtype=np.dtype("uint16"), chunks=(1, 4, 4)),
            "scl": VarSpec(dims=("time", "northing", "easting"), dtype=np.dtype("uint8"), chunks=(1, 4, 4)),
        },
        commit_msg="seed",
        attrs={"crs": "EPSG:32601", "baselines_applied": {"2024-06-01": 5}},
    )
    return path


def _snapshots(path: str) -> int:
    return len(list(_open_repo(path).ancestry(branch="main")))


def test_append_plus_windows_is_one_snapshot(store):
    before = _snapshots(store)
    with batched_region_writes(store, message="date 2024-06-11") as batch:
        t = batch.append_time_slot(D2)
        assert t == 1
        batch.write_window(t, slice(0, 4), slice(0, 8), {"band": np.full((4, 8), 7, np.uint16)})
        batch.write_window(t, slice(4, 8), slice(4, 8), {"band": np.full((4, 4), 9, np.uint16)})
    assert _snapshots(store) == before + 1  # append + both windows: ONE commit

    g = open_store_as_zarr_group(store)
    assert list(read_time_values(g).astype("datetime64[D]").astype(str)) == ["2024-06-01", "2024-06-11"]
    band = np.asarray(g["band"])
    assert (band[1, 0:4, 0:8] == 7).all() and (band[1, 4:8, 4:8] == 9).all()
    assert (band[1, 4:8, 0:4] == 0).all()  # untouched window stays fill
    assert (band[0] == 0).all()  # seeded date untouched
    assert (np.asarray(g["scl"])[1] == 0).all()  # unwritten var's new slot stays fill


def test_failed_batch_commits_nothing(store):
    before = _snapshots(store)
    with pytest.raises(RuntimeError, match="mid-batch"), batched_region_writes(store, message="doomed") as batch:
        batch.append_time_slot(D2)
        raise RuntimeError("mid-batch")
    assert _snapshots(store) == before  # no snapshot
    assert get_existing_dates(store) == {"2024-06-01"}  # axis unchanged for readers


def test_duplicate_date_refused(store):
    with (
        pytest.raises(ValueError, match="already on the time axis"),
        batched_region_writes(store, message="dup") as batch,
    ):
        batch.append_time_slot(D1)


def test_appended_date_reaches_the_dedupe_reader(store):
    """`get_existing_dates` feeds the next run's STAC filter — the round trip
    through TIME_ENCODING must hold or re-runs re-download every written date.
    """
    with batched_region_writes(store, message="d2") as batch:
        batch.append_time_slot(D2)
    assert get_existing_dates(store) == {"2024-06-01", "2024-06-11"}


def test_attr_edits_land_in_the_same_commit(store):
    before = _snapshots(store)
    with batched_region_writes(store, message="attrs") as batch:
        t = batch.append_time_slot(D2)
        batch.write_window(t, slice(0, 4), slice(0, 4), {"band": np.full((4, 4), 3, np.uint16)})
        merged = dict(batch.group.attrs.get("baselines_applied", {}))
        merged["2024-06-11"] = 5
        batch.group.attrs["baselines_applied"] = merged
        batch.group.attrs["last_appended"] = "2026-07-24T00:00:00Z"
    assert _snapshots(store) == before + 1
    attrs = dict(open_store_as_zarr_group(store).attrs)
    assert attrs["baselines_applied"] == {"2024-06-01": 5, "2024-06-11": 5}
    assert attrs["last_appended"] == "2026-07-24T00:00:00Z"
    assert attrs["crs"] == "EPSG:32601"  # pre-existing attrs preserved
