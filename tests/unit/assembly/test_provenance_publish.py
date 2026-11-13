"""Publishing a run's provenance record, and reading it back: tables, planes and the checks between them.

Built on the real pieces end to end, on local stores: mosaics written by the ingest's own writer
with the switch on, shards staged by the real staging writer from pieces read off those mosaics,
publication reading both, and the reader helpers reading what it published. A 2 x 2 grid of 4-pixel
shards, three sensors, two dates each; the optical lists also name a copy the ingest dropped, and a
day it skipped whole.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

import dask.array as da
import icechunk
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
import xarray as xr
import zarr
from affine import Affine
from odc.geo.geobox import GeoBox

from tessera_embeddings.config.inference import ZoneLayout
from tessera_embeddings.config.ingest import PROVENANCE_VERSION, SOURCE_INDEX_VAR
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.provenance import PROVENANCE_ATTR, SNAPSHOTS_ATTR, SourcePlanes, read_source_pieces
from tessera_embeddings.inference.provenance_publish import ProvenanceCheckError, ProvenanceRun, publish_provenance
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.provenance import (
    granules_uri,
    pixel_sources_group,
    pixel_sources_uri,
    shard_dates_uri,
    update_pixels_uri,
)
from tessera_embeddings.storage.provenance_reader import PixelLocation, ProvenanceRecord, locate
from tessera_embeddings.storage.published_store import store_runs_stamp
from tessera_embeddings.storage.registry_master import STORE_RUNS_KEY
from tessera_embeddings.storage.zarr_store import (
    open_repo,
    open_store_as_zarr_group,
    record_skipped_days,
    write_days_windows,
)
from tests.unit.inference_stack import STACK

SIZE, SHARD = 8, 4
DATES = ["2024-06-01", "2024-06-13"]
#: A day the optical ingest skipped whole, so off the time axis.
SKIPPED = "2024-06-07"
STORES = {"reflectance": "s2", "sar_ascending": "s1_asc", "sar_descending": "s1_desc"}
ZONE, YEAR, RUN, UPDATE = "33N", 2024, "33N-2024-abcd1234", "33N-2024-holes123"

#: The pixels a holes update embedded in each of its embedding shards: the top two rows.
EMBEDDED_BY_UPDATE = np.zeros((SHARD, SHARD), dtype=bool)
EMBEDDED_BY_UPDATE[:2] = True


class _Roi:
    geobox = GeoBox((SIZE, SIZE), Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0), "EPSG:32633")
    height = SIZE
    width = SIZE


def _granule(sensor: str, date: str, k: int) -> dict:
    return {
        "granule_id": f"{sensor}-{date}-{k}",
        "image_key": f"key-{k}",
        "platform": "S2A" if sensor == "s2" else "S1A",
        "acquired_at": f"{date}T10:0{k}:00Z",
        "cloud_cover": 3.0 if sensor == "s2" else None,
        "assets": {"band": f"https://example.com/{sensor}/{date}/{k}.tif"},
        "stepped_down": False,
        "geometry": {"type": "Polygon", "coordinates": [[[14, 45], [15, 45], [15, 46], [14, 46], [14, 45]]]},
        "bbox": [14, 45, 15, 46],
    }


def _granule_list(sensor: str, date: str) -> list[dict]:
    """Two fused granules, and for optical a losing duplicate listed after them."""
    fused = [_granule(sensor, date, 0), _granule(sensor, date, 1)]
    return fused + ([_granule(sensor, date, 9) | {"status": "duplicate"}] if sensor == "s2" else [])


def _mosaics(base: Path, *, on: bool = True, seed: int = 0) -> dict[str, str]:
    """Three mosaic stores with two dates each; each slot's index uses granules 1 and 2."""
    rng = np.random.default_rng(seed)
    pins = {}
    for store, sensor in STORES.items():
        path = str(base / f"{store}.zarr")
        days = []
        for date in DATES:
            data = {"band": da.full((1, SIZE, SIZE), 7, dtype=np.uint16, chunks=(1, SHARD, SHARD))}
            if on:
                index = rng.integers(0, 3, size=(1, SIZE, SIZE)).astype(np.uint16)
                data[SOURCE_INDEX_VAR] = da.from_array(index, chunks=(1, SHARD, SHARD))
            ds = xr.Dataset(
                {k: (("time", "northing", "easting"), v) for k, v in data.items()},
                coords={"time": [np.datetime64(date, "ns")]},
            )
            days.append(ds)
        write_days_windows(
            path,
            [(d, [(0, SIZE, 0, SIZE)]) for d in days],
            roi=_Roi(),
            manifest=IngestManifest(roi_manifest_hash="abc", provenance_version=PROVENANCE_VERSION if on else None),
            baselines={},
            tile_id="roi.zarr",
            crs="EPSG:32633",
            chunks={"time": 1, "northing": SHARD, "easting": SHARD},
            granules=[json.dumps(_granule_list(sensor, d)) for d in DATES] if on else None,
        )
        if on and sensor == "s2":
            # A skipped day, and one an earlier attempt skipped and a later one wrote, whose slot wins.
            skipped = {
                day: json.dumps([_granule(sensor, day, 0) | {"status": "low_coverage"}]) for day in (SKIPPED, DATES[0])
            }
            record_skipped_days(path, skipped)
        pins[path] = open_repo(path).lookup_branch("main")
    return pins


def _chunk(row: int, col: int) -> ChunkSpec:
    return ChunkSpec(
        row=row, col=col, y_start=row * SHARD, y_stop=(row + 1) * SHARD, x_start=col * SHARD, x_stop=(col + 1) * SHARD
    )


#: Every observation valid for the model.
ALL_VALID = np.ones((len(DATES), SHARD, SHARD), dtype=bool)

#: Half of each shard rejected on the first date, all of it on the second: a fully cloudy date.
SOME_REJECTED = ALL_VALID.copy()
SOME_REJECTED[0, :, : SHARD // 2] = False
SOME_REJECTED[1] = False


def _planes(pins: dict[str, str], chunk: ChunkSpec, valid: np.ndarray = ALL_VALID) -> SourcePlanes:
    """The actor's record of ``chunk``, read off the pinned mosaics; ``valid`` is the model's test."""
    planes = SourcePlanes(SHARD, SHARD, pins)
    for path, pin in pins.items():
        root = open_store_as_zarr_group(path, snapshot_id=pin)
        times = np.arange(len(DATES))
        ys, xs = slice(chunk.y_start, chunk.y_stop), slice(chunk.x_start, chunk.x_stop)
        planes.add(
            STORES[Path(path).stem],
            read_source_pieces(root, times, (ys, xs), valid, (slice(0, SHARD), slice(0, SHARD))),
        )
    return planes


def _stage(
    staging: Path,
    pins: dict[str, str],
    *,
    pixels: np.ndarray | None = None,
    run: str = RUN,
    valid: np.ndarray = ALL_VALID,
) -> tuple[list[str], list[str]]:
    """Three shards embed, one is refused; all stage planes. Returns (embedded, refused).

    ``pixels`` makes the run a holes update that embedded those pixels of each embedding shard.
    """
    writer = ZarrWriter(str(staging))
    embedded = []
    for row, col in ((0, 0), (0, 1), (1, 0)):
        chunk = _chunk(row, col)
        writer.write_chunk(
            chunk,
            np.zeros((SHARD, SHARD, 128), dtype=np.int8),
            run,
            scales=np.ones((SHARD, SHARD), dtype=np.float32),
            coverage={"embedded_px": SHARD * SHARD if pixels is None else int(pixels.sum())},
            stack=STACK,
            planes=_planes(pins, chunk, valid),
            pixels=pixels,
        )
        embedded.append(chunk.label)
    refused = _chunk(1, 1)
    writer.write_coverage_only(
        refused,
        run,
        {"s2_obs_count": np.zeros((SHARD, SHARD), dtype=np.uint16)},
        None,
        planes=_planes(pins, refused, valid),
        pixels=None if pixels is None else np.zeros_like(pixels),
    )
    return embedded, [refused.label]


def _publish(tmp_path: Path, pins: dict[str, str], embedded: list[str], refused: list[str], **run) -> dict | None:
    """Publish one run's record; ``run`` overrides the fill's ``run_id`` or ``mode``."""
    run = {"run_id": RUN, "mode": "fill"} | run
    return publish_provenance(
        ProvenanceRun(
            staging_base=str(tmp_path / "staging"),
            registry_root=str(tmp_path / "registry"),
            zone=ZONE,
            year=YEAR,
            embedded=embedded,
            refused=refused,
            mosaic_snapshots=pins,
            grid_shape=(SIZE, SIZE),
            shard_px=SHARD,
            **run,
        )
    )


@pytest.fixture
def published(tmp_path: Path):
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins)
    return tmp_path, pins, _publish(tmp_path, pins, embedded, refused)


def _granules_uri(tmp_path: Path, pins: dict[str, str], sensor: str) -> str:
    pin = next(pin for path, pin in pins.items() if STORES[Path(path).stem] == sensor)
    return granules_uri(str(tmp_path / "registry"), ZONE, YEAR, sensor, pin)


def _shard_dates(tmp_path: Path, run: str = RUN):
    return pq.read_table(shard_dates_uri(str(tmp_path / "registry"), ZONE, YEAR, run))


def test_granules_list_each_snapshots_granules_in_fusion_order(published) -> None:
    tmp_path, pins, summary = published
    tables = {sensor: pq.read_table(_granules_uri(tmp_path, pins, sensor)) for sensor in STORES.values()}
    assert summary["granules"] == sum(t.num_rows for t in tables.values()) == 3 * len(DATES) * 2 + len(DATES) + 1
    rows = tables["s2"].to_pylist()
    dropped = [(r["granule_id"], r["fusion_order"], r["status"]) for r in rows if r["status"] != "fused"]
    assert dropped == [(f"s2-{d}-9", None, "duplicate") for d in DATES] + [(f"s2-{SKIPPED}-0", None, "low_coverage")]
    first = next(r for r in rows if r["fusion_order"] == 1)
    assert (str(first["solar_day"]), first["granule_id"]) == ("2024-06-01", "s2-2024-06-01-1")
    assert first["snapshot_id"] == pins[str(tmp_path / "mosaics" / "reflectance.zarr")]
    assert shapely.from_wkb(first["geometry"]).bounds == (14.0, 45.0, 15.0, 46.0)
    assert b"geo" in tables["s2"].schema.metadata
    assert tables["s2"].schema.metadata[b"first_published_by"] == RUN.encode()


def test_a_snapshots_granules_are_written_once_and_checked_by_every_later_run(published) -> None:
    """A second run reading the same mosaics keeps the first run's file, and refuses one that differs."""
    tmp_path, pins, _ = published
    uri = _granules_uri(tmp_path, pins, "s2")
    embedded, refused = _stage(tmp_path / "staging", pins, run=UPDATE)
    _publish(tmp_path, pins, embedded, refused, run_id=UPDATE)
    table = pq.read_table(uri)
    assert table.schema.metadata[b"first_published_by"] == RUN.encode()
    assert set(_shard_dates(tmp_path, UPDATE).column("snapshot_id").to_pylist()) == set(pins.values())
    pq.write_table(table.slice(1), uri)
    with pytest.raises(ProvenanceCheckError, match="already published"):
        _publish(tmp_path, pins, embedded, refused, run_id=UPDATE)


@pytest.mark.parametrize("valid", [ALL_VALID, SOME_REJECTED], ids=["all-valid", "some-rejected"])
def test_shard_dates_and_planes_name_the_same_granules(tmp_path: Path, valid: np.ndarray) -> None:
    """The plane is the mosaic's index, signed, every row's sources are its values, and the group names each snapshot.

    Positive where the model used the observation, negative where it rejected it; a fully rejected
    date keeps its rows.
    """
    pins = _mosaics(tmp_path / "mosaics")
    _publish(tmp_path, pins, *_stage(tmp_path / "staging", pins, valid=valid))
    rows = _shard_dates(tmp_path).to_pylist()
    assert {r["tile"] for r in rows} == {"chunk_0_0", "chunk_0_1", "chunk_1_0", "chunk_1_1"}
    repo = open_repo(pixel_sources_uri(str(tmp_path / "registry")))
    group = zarr.open_group(repo.readonly_session("main").store, path=pixel_sources_group(ZONE, YEAR, RUN), mode="r")
    for sensor in STORES.values():
        dates = group[sensor].shape[0]
        assert (group[sensor].chunks, group[sensor].shards) == ((dates, SHARD, SHARD), (dates, SHARD, SHARD)), (
            "a chunk holds every date"
        )
    for row in rows:
        r, c = (int(x) for x in row["tile"].removeprefix("chunk_").split("_"))
        t = DATES.index(str(row["solar_day"]))
        plane = group[row["sensor"]][t, r * SHARD : (r + 1) * SHARD, c * SHARD : (c + 1) * SHARD]
        store = next(p for p in pins if STORES[Path(p).stem] == row["sensor"])
        assert row["snapshot_id"] == pins[store] == group.attrs[f"snapshot_{row['sensor']}"]
        index = open_store_as_zarr_group(store, snapshot_id=pins[store])[SOURCE_INDEX_VAR][
            t, r * SHARD : (r + 1) * SHARD, c * SHARD : (c + 1) * SHARD
        ]
        np.testing.assert_array_equal(plane, np.where(valid[t], index, -index.astype(int)), err_msg=str(row))
        assert row["sources"] == sorted(set(np.abs(plane[plane != 0]).astype(int) - 1))
        assert row["valid_px"] == int(np.count_nonzero(plane > 0))
        assert row["embedded_px"] == (0 if row["tile"] == "chunk_1_1" else SHARD * SHARD)
    assert {str(r["solar_day"]) for r in rows} == set(DATES), "a fully rejected date keeps its rows"


def test_publishing_again_rewrites_the_same_record(published) -> None:
    tmp_path, pins, _ = published
    before = _shard_dates(tmp_path)
    _publish(tmp_path, pins, ["chunk_0_0", "chunk_0_1", "chunk_1_0"], ["chunk_1_1"])
    assert _shard_dates(tmp_path).equals(before)


def test_mosaics_without_the_record_publish_nothing(tmp_path: Path) -> None:
    pins = _mosaics(tmp_path / "mosaics", on=False)
    assert _publish(tmp_path, pins, [], []) is None
    assert not (tmp_path / "registry").exists()


def test_a_plane_that_disagrees_with_its_row_fails_the_publication(tmp_path: Path) -> None:
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins)
    group = zarr.open_group(str(tmp_path / "staging" / RUN / "chunk_0_0.zarr"), mode="a")
    record = dict(group.attrs[PROVENANCE_ATTR])
    record["s2"] = dict(record["s2"], valid_px=[v + 1 for v in record["s2"]["valid_px"]])
    group.attrs[PROVENANCE_ATTR] = record
    with pytest.raises(ProvenanceCheckError, match="its row says"):
        _publish(tmp_path, pins, embedded, refused)


def test_a_shard_no_pass_of_one_sensor_covered_publishes_without_it(tmp_path: Path) -> None:
    """A shard no descending pass covered stages no descending record, and has no descending rows."""
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins)
    group = zarr.open_group(str(tmp_path / "staging" / RUN / "chunk_0_0.zarr"), mode="a")
    group.attrs[PROVENANCE_ATTR] = {k: v for k, v in group.attrs[PROVENANCE_ATTR].items() if k != "s1_desc"}
    _publish(tmp_path, pins, embedded, refused)
    sensors = {r["sensor"] for r in _shard_dates(tmp_path).to_pylist() if r["tile"] == "chunk_0_0"}
    assert sensors == {"s2", "s1_asc"}


def test_a_shard_read_from_other_mosaic_snapshots_fails_the_publication(tmp_path: Path) -> None:
    """Its planes number another version's granules, so describing it from the run's would mislead."""
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins)
    group = zarr.open_group(str(tmp_path / "staging" / RUN / "chunk_0_1.zarr"), mode="a")
    group.attrs[SNAPSHOTS_ATTR] = dict.fromkeys(pins, "another-version")
    with pytest.raises(ProvenanceCheckError, match=r"\['chunk_0_1'\] were read from mosaic snapshots other"):
        _publish(tmp_path, pins, embedded, refused)
    assert not (tmp_path / "registry").exists(), "refused before anything is written"


def test_an_embedded_shard_with_no_record_fails_the_publication(tmp_path: Path) -> None:
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins)
    group = zarr.open_group(str(tmp_path / "staging" / RUN / "chunk_0_0.zarr"), mode="a")
    del group.attrs[PROVENANCE_ATTR]
    with pytest.raises(ProvenanceCheckError, match="staged no provenance"):
        _publish(tmp_path, pins, embedded, refused)


def test_a_holes_update_publishes_the_pixels_it_embedded_and_planes_inside_them(tmp_path: Path) -> None:
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins, pixels=EMBEDDED_BY_UPDATE)
    _publish(tmp_path, pins, embedded, refused)
    masks = zarr.open_group(update_pixels_uri(str(tmp_path / "registry"), ZONE, YEAR, RUN), mode="r")
    assert sorted(masks.array_keys()) == ["chunk_0_0", "chunk_0_1", "chunk_1_0", "chunk_1_1"]
    np.testing.assert_array_equal(masks["chunk_0_0"][:], EMBEDDED_BY_UPDATE)
    assert not masks["chunk_1_1"][:].any()
    rows = _shard_dates(tmp_path).to_pylist()
    assert rows and all(r["valid_px"] <= int(EMBEDDED_BY_UPDATE.sum()) for r in rows)
    assert {r["tile"] for r in rows} == set(embedded)


def test_a_holes_mask_that_is_not_what_the_update_embedded_fails_the_publication(tmp_path: Path) -> None:
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins, pixels=EMBEDDED_BY_UPDATE)
    group = zarr.open_group(str(tmp_path / "staging" / RUN / "chunk_0_0.zarr"), mode="a")
    group.attrs["coverage"] = {"embedded_px": int(EMBEDDED_BY_UPDATE.sum()) + 1}
    with pytest.raises(ProvenanceCheckError, match="in its mask"):
        _publish(tmp_path, pins, embedded, refused)


@pytest.fixture
def updated(published):
    """The fill, then a holes update embedding the top two rows of each embedding shard, and the master.

    The master's ``source_runs`` names both runs for the three shards the update wrote, and is null
    for the fourth, as for a shard with no complete record.
    """
    tmp_path, pins, _ = published
    embedded, refused = _stage(tmp_path / "staging", pins, pixels=EMBEDDED_BY_UPDATE, run=UPDATE)
    _publish(tmp_path, pins, embedded, refused, run_id=UPDATE, mode="holes")
    _master(tmp_path / "registry", embedded + refused, [[RUN, UPDATE]] * 3 + [None])
    return str(tmp_path / "registry"), pins


def _store_attrs(run_ids: list[str]) -> dict:
    """A zone group's run records for YEAR: the first run as the fill, the rest as updates."""
    fill, *updates = run_ids
    return {"runs": {str(YEAR): {"run_id": fill}}, "updates": {str(YEAR): [{"run_id": u} for u in updates]}}


def _record_attrs(registry: Path, attrs: dict) -> None:
    """Make the store beside ``registry`` hold ``attrs`` on its zone group, as each run's commit leaves them."""
    repo = icechunk.Repository.open_or_create(icechunk.local_filesystem_storage(f"{registry}.icechunk"))
    session = repo.writable_session("main")
    zarr.open_group(session.store, path=ZONE, mode="a").attrs.update(attrs)
    session.commit("record runs")


def _master(registry: Path, tiles: list[str], source_runs: list) -> None:
    """A registry master naming each shard's ``source_runs``, merged after every run it names landed."""
    master = registry / "master" / f"zone={ZONE}" / f"year={YEAR}"
    master.mkdir(parents=True)
    run_ids = list(dict.fromkeys(run for runs in source_runs if runs for run in runs))
    _record_attrs(registry, _store_attrs(run_ids))
    stamp = {STORE_RUNS_KEY: store_runs_stamp(_store_attrs(run_ids), YEAR)}
    table = pa.table({"tile": tiles, "source_runs": pa.array(source_runs, pa.list_(pa.string()))}, metadata=stamp)
    pq.write_table(table, master / "part-0.parquet")


@pytest.mark.parametrize(
    ("lat", "lon", "layout", "where"),
    [
        (45.5071, 14.3761, "hemispheres", PixelLocation("33N", "chunk_209_14", 1908, 70)),
        (45.5149, 14.3794, "hemispheres", PixelLocation("33N", "chunk_209_14", 1822, 96)),
        (45.5071, 14.3761, "utm_zones", PixelLocation("utm33", "chunk_209_14", 1908, 70)),
        (-33.9, 18.4, "hemispheres", PixelLocation("34S", "chunk_184_4", 51, 1382)),
        (-33.9, 18.4, "utm_zones", PixelLocation("utm34", "chunk_639_4", 627, 1382)),
    ],
)
def test_a_point_is_located_on_the_stores_grid(
    lat: float, lon: float, layout: ZoneLayout, where: PixelLocation
) -> None:
    """The dev test's pixels where it found them; south of the equator the two layouts' grids differ."""
    assert locate(lat, lon, layout) == where


@pytest.mark.parametrize(("row", "col", "run"), [(0, 0, UPDATE), (3, 3, RUN)], ids=["update-marked", "fill"])
def test_a_pixel_is_read_from_the_run_that_claims_it(updated, row: int, col: int, run: str) -> None:
    """The update claims the pixels its mask marks, the fill the rest; each date names the mosaic's granule."""
    registry, pins = updated
    got = ProvenanceRecord(registry).pixel_granules_at(PixelLocation(ZONE, "chunk_0_1", row, col), YEAR)
    expected = []
    for path, pin in pins.items():
        sensor = STORES[Path(path).stem]
        index = open_store_as_zarr_group(path, snapshot_id=pin)[SOURCE_INDEX_VAR][:, row, SHARD + col]
        expected += [(sensor, DATES[t], True, f"{sensor}-{DATES[t]}-{v - 1}") for t, v in enumerate(index) if v]
    assert set(got["run_id"]) == {run}
    rows = [(r["sensor"], str(r["solar_day"]), r["valid"], r["granule_id"]) for r in got.iter_rows(named=True)]
    assert sorted(rows) == sorted(expected)


def test_a_rejected_observation_names_its_granule_and_says_so(tmp_path: Path) -> None:
    """A pixel the model rejected on both dates still lists each date's granule, as not valid."""
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins, valid=SOME_REJECTED)
    _publish(tmp_path, pins, embedded, refused)
    _master(tmp_path / "registry", embedded + refused, [[RUN]] * 4)
    got = ProvenanceRecord(str(tmp_path / "registry")).pixel_granules_at(PixelLocation(ZONE, "chunk_0_0", 0, 0), YEAR)
    index = {
        STORES[Path(path).stem]: open_store_as_zarr_group(path, snapshot_id=pin)[SOURCE_INDEX_VAR][:, 0, 0]
        for path, pin in pins.items()
    }
    expected = [(s, DATES[t], f"{s}-{DATES[t]}-{v - 1}") for s, idx in index.items() for t, v in enumerate(idx) if v]
    assert sorted((r["sensor"], str(r["solar_day"]), r["granule_id"]) for r in got.iter_rows(named=True)) == sorted(
        expected
    )
    assert expected and not got["valid"].any()


def test_a_shard_with_no_complete_record_has_no_pixel_answer(updated) -> None:
    registry, _ = updated
    with pytest.raises(LookupError, match="no complete provenance record"):
        ProvenanceRecord(registry).pixel_granules_at(PixelLocation(ZONE, "chunk_1_1", 0, 0), YEAR)


@pytest.mark.parametrize("row", [-1, SHARD])
def test_a_location_outside_its_shard_is_refused(updated, row: int) -> None:
    registry, _ = updated
    with pytest.raises(ValueError, match="must lie in"):
        ProvenanceRecord(registry).pixel_granules_at(PixelLocation(ZONE, "chunk_0_1", row, 0), YEAR)


def test_a_refused_shard_that_staged_no_record_has_no_pixel_answer(tmp_path: Path) -> None:
    """Its run lists it as unrecorded, so a master naming the run still gives no answer for it."""
    pins = _mosaics(tmp_path / "mosaics")
    embedded, refused = _stage(tmp_path / "staging", pins)
    shutil.rmtree(next((tmp_path / "staging" / RUN).glob(f"{refused[0]}*")))
    _publish(tmp_path, pins, embedded, refused)
    _master(tmp_path / "registry", embedded + refused, [[RUN]] * 4)
    with pytest.raises(LookupError, match="staged no provenance record"):
        ProvenanceRecord(str(tmp_path / "registry")).pixel_granules_at(PixelLocation(ZONE, refused[0], 0, 0), YEAR)


@pytest.mark.parametrize("wrote", [True, False], ids=["update-wrote-shards", "update-wrote-nothing"])
def test_a_master_merged_before_the_latest_shard_writing_run_has_no_pixel_answer(published, wrote: bool) -> None:
    """A run that wrote shards after the merge may own the pixel, so the lineage is refused until the
    next merge; one that recorded itself without writing (a null commit) changes no tile, and is not.
    """
    tmp_path, _pins, _ = published
    registry = tmp_path / "registry"
    _master(registry, ["chunk_0_0", "chunk_0_1", "chunk_1_0", "chunk_1_1"], [[RUN]] * 4)
    where, record = PixelLocation(ZONE, "chunk_0_0", 0, 0), ProvenanceRecord(str(registry))
    assert not record.pixel_granules_at(where, YEAR).is_empty()
    update = {"run_id": UPDATE, "commit": {"snapshot": "s"} if wrote else None}
    _record_attrs(registry, _store_attrs([RUN]) | {"updates": {str(YEAR): [update]}})
    if wrote:
        with pytest.raises(LookupError, match="merged before the store's latest run"):
            record.pixel_granules_at(where, YEAR)
    else:
        assert not record.pixel_granules_at(where, YEAR).is_empty()


@pytest.mark.parametrize("written", [True, False], ids=["before-source-runs", "never-filled"])
def test_a_master_without_source_runs_has_no_pixel_answer(tmp_path: Path, written: bool) -> None:
    master = tmp_path / "master" / f"zone={ZONE}" / f"year={YEAR}"
    master.mkdir(parents=True)
    if written:
        pq.write_table(pa.table({"tile": ["chunk_0_0"]}), master / "part-0.parquet")
    with pytest.raises(LookupError, match="no complete provenance record"):
        ProvenanceRecord(str(tmp_path)).pixel_granules_at(PixelLocation(ZONE, "chunk_0_0", 0, 0), YEAR)


def test_a_day_and_a_shard_read_through_the_snapshot_each_run_names(updated) -> None:
    registry, _ = updated
    day = datetime.date(2024, 6, 1)
    for run in (None, RUN, UPDATE):
        got = ProvenanceRecord(registry).day_granules(ZONE, YEAR, "s2", day, run_id=run)
        assert list(zip(got["granule_id"], got["status"], strict=True)) == [
            ("s2-2024-06-01-0", "fused"),
            ("s2-2024-06-01-1", "fused"),
            ("s2-2024-06-01-9", "duplicate"),
        ]
    shard = ProvenanceRecord(registry).shard_granules(ZONE, YEAR, "chunk_0_0", RUN)
    rows = _shard_dates(Path(registry).parent).to_pylist()
    assert shard.height == sum(len(r["sources"]) for r in rows if r["tile"] == "chunk_0_0")
    assert all(g.startswith(f"{s}-{d}-") for s, d, g in shard.select("sensor", "solar_day", "granule_id").iter_rows())
