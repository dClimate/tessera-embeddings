"""Publishing a run's provenance record, and reading it back: tables, planes and the checks between them.

Built on the real pieces end to end, on local stores: mosaics written by the ingest's own writer
with the switch on, shards staged by the real staging writer from pieces read off those mosaics,
publication reading both, and the reader helpers reading what it published. A 2 x 2 grid of 4-pixel
shards, three sensors, two dates each.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import dask.array as da
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
import xarray as xr
import zarr
from affine import Affine
from odc.geo.geobox import GeoBox

from tessera_embeddings.config.ingest import PROVENANCE_VERSION, SOURCE_INDEX_VAR
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.provenance import PROVENANCE_ATTR, SourcePlanes, read_source_pieces
from tessera_embeddings.inference.provenance_publish import ProvenanceCheckError, publish_provenance
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.provenance import (
    granules_uri,
    pixel_sources_group,
    pixel_sources_uri,
    shard_dates_uri,
    update_pixels_uri,
)
from tessera_embeddings.storage.provenance_reader import (
    PixelLocation,
    day_granules,
    locate,
    pixel_granules_at,
    shard_granules,
)
from tessera_embeddings.storage.zarr_store import open_repo, open_store_as_zarr_group, write_days_windows
from tests.unit.inference_stack import STACK

SIZE, SHARD = 8, 4
DATES = ["2024-06-01", "2024-06-13"]
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
            granules=[json.dumps([_granule(sensor, d, 0), _granule(sensor, d, 1)]) for d in DATES] if on else None,
        )
        pins[path] = open_repo(path).lookup_branch("main")
    return pins


def _chunk(row: int, col: int) -> ChunkSpec:
    return ChunkSpec(
        row=row, col=col, y_start=row * SHARD, y_stop=(row + 1) * SHARD, x_start=col * SHARD, x_stop=(col + 1) * SHARD
    )


def _planes(pins: dict[str, str], chunk: ChunkSpec) -> SourcePlanes:
    """The actor's record of ``chunk``: every pixel valid, read off the pinned mosaics."""
    planes = SourcePlanes(SHARD, SHARD)
    for path, pin in pins.items():
        root = open_store_as_zarr_group(path, snapshot_id=pin)
        times = np.arange(len(DATES))
        valid = np.ones((len(DATES), SHARD, SHARD), dtype=bool)
        ys, xs = slice(chunk.y_start, chunk.y_stop), slice(chunk.x_start, chunk.x_stop)
        planes.add(
            STORES[Path(path).stem],
            read_source_pieces(root, times, ys, xs, valid, rows=slice(0, SHARD), cols=slice(0, SHARD)),
        )
    return planes


def _stage(
    staging: Path, pins: dict[str, str], *, pixels: np.ndarray | None = None, run: str = RUN
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
            planes=_planes(pins, chunk),
            pixels=pixels,
        )
        embedded.append(chunk.label)
    refused = _chunk(1, 1)
    writer.write_coverage_only(
        refused,
        run,
        {"s2_obs_count": np.zeros((SHARD, SHARD), dtype=np.uint16)},
        None,
        planes=_planes(pins, refused),
        pixels=None if pixels is None else np.zeros_like(pixels),
    )
    return embedded, [refused.label]


def _publish(
    tmp_path: Path, pins: dict[str, str], embedded: list[str], refused: list[str], run: str = RUN
) -> dict | None:
    return publish_provenance(
        staging_base=str(tmp_path / "staging"),
        registry_root=str(tmp_path / "registry"),
        zone=ZONE,
        year=YEAR,
        run_id=run,
        embedded=embedded,
        refused=refused,
        mosaic_snapshots=pins,
        grid_shape=(SIZE, SIZE),
        shard_px=SHARD,
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
    assert summary["granules"] == sum(t.num_rows for t in tables.values()) == 3 * len(DATES) * 2
    rows = tables["s2"].to_pylist()
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
    _publish(tmp_path, pins, embedded, refused, run=UPDATE)
    table = pq.read_table(uri)
    assert table.schema.metadata[b"first_published_by"] == RUN.encode()
    assert set(_shard_dates(tmp_path, UPDATE).column("snapshot_id").to_pylist()) == set(pins.values())
    pq.write_table(table.slice(1), uri)
    with pytest.raises(ProvenanceCheckError, match="already published"):
        _publish(tmp_path, pins, embedded, refused, run=UPDATE)


def test_shard_dates_and_planes_name_the_same_granules(published) -> None:
    """Every row's sources are what its plane ranks, and the plane is the mosaic's index."""
    tmp_path, pins, _ = published
    rows = _shard_dates(tmp_path).to_pylist()
    assert {r["tile"] for r in rows} == {"chunk_0_0", "chunk_0_1", "chunk_1_0", "chunk_1_1"}
    repo = open_repo(pixel_sources_uri(str(tmp_path / "registry")))
    group = zarr.open_group(repo.readonly_session("main").store, path=pixel_sources_group(ZONE, YEAR, RUN), mode="r")
    for row in rows:
        r, c = (int(x) for x in row["tile"].removeprefix("chunk_").split("_"))
        t = DATES.index(str(row["solar_day"]))
        plane = group[row["sensor"]][t, r * SHARD : (r + 1) * SHARD, c * SHARD : (c + 1) * SHARD]
        store = next(p for p in pins if STORES[Path(p).stem] == row["sensor"])
        assert row["snapshot_id"] == pins[store]
        index = open_store_as_zarr_group(store, snapshot_id=pins[store])[SOURCE_INDEX_VAR][
            t, r * SHARD : (r + 1) * SHARD, c * SHARD : (c + 1) * SHARD
        ]
        named = np.where(plane > 0, np.asarray(row["sources"] + [0])[plane.astype(int) - 1] + 1, 0)
        np.testing.assert_array_equal(named, index, err_msg=str(row))
        assert row["valid_px"] == int(np.count_nonzero(plane))
        assert row["embedded_px"] == (0 if row["tile"] == "chunk_1_1" else SHARD * SHARD)


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
    _publish(tmp_path, pins, embedded, refused, run=UPDATE)
    master = tmp_path / "registry" / "master" / f"zone={ZONE}" / f"year={YEAR}"
    master.mkdir(parents=True)
    runs = pa.array([[RUN, UPDATE], [RUN, UPDATE], [RUN, UPDATE], None], pa.list_(pa.string()))
    pq.write_table(pa.table({"tile": embedded + refused, "source_runs": runs}), master / "part-0.parquet")
    return str(tmp_path / "registry"), pins


@pytest.mark.parametrize(
    ("lat", "lon", "where"),
    [
        (45.5071, 14.3761, PixelLocation("33N", "chunk_209_14", 1908, 70)),
        (45.5149, 14.3794, PixelLocation("33N", "chunk_209_14", 1822, 96)),
    ],
)
def test_a_point_is_located_on_the_stores_grid(lat: float, lon: float, where: PixelLocation) -> None:
    """The two pixels the dev test traced, at the shard, row and column it found them."""
    assert locate(lat, lon) == where


@pytest.mark.parametrize(("row", "col", "run"), [(0, 0, UPDATE), (3, 3, RUN)], ids=["update-marked", "fill"])
def test_a_pixel_is_read_from_the_run_that_claims_it(updated, row: int, col: int, run: str) -> None:
    """The update claims the pixels its mask marks, the fill the rest; each date names the mosaic's granule."""
    registry, pins = updated
    got = pixel_granules_at(registry, PixelLocation(ZONE, "chunk_0_1", row, col), YEAR)
    expected = []
    for path, pin in pins.items():
        sensor = STORES[Path(path).stem]
        index = open_store_as_zarr_group(path, snapshot_id=pin)[SOURCE_INDEX_VAR][:, row, SHARD + col]
        expected += [(sensor, DATES[t], f"{sensor}-{DATES[t]}-{v - 1}") for t, v in enumerate(index) if v]
    assert set(got["run_id"]) == {run}
    rows = [(r["sensor"], str(r["solar_day"]), r["granule_id"]) for r in got.iter_rows(named=True)]
    assert sorted(rows) == sorted(expected)


def test_a_shard_with_no_complete_record_has_no_pixel_answer(updated) -> None:
    registry, _ = updated
    with pytest.raises(LookupError, match="no complete provenance record"):
        pixel_granules_at(registry, PixelLocation(ZONE, "chunk_1_1", 0, 0), YEAR)


def test_a_day_and_a_shard_read_through_the_snapshot_each_run_names(updated) -> None:
    registry, _ = updated
    day = datetime.date(2024, 6, 1)
    for run in (None, RUN, UPDATE):
        assert day_granules(registry, ZONE, YEAR, "s2", day, run_id=run)["granule_id"].to_list() == [
            "s2-2024-06-01-0",
            "s2-2024-06-01-1",
        ]
    shard = shard_granules(registry, ZONE, YEAR, "chunk_0_0", RUN)
    rows = _shard_dates(Path(registry).parent).to_pylist()
    assert shard.height == sum(len(r["sources"]) for r in rows if r["tile"] == "chunk_0_0")
    assert all(g.startswith(f"{s}-{d}-") for s, d, g in shard.select("sensor", "solar_day", "granule_id").iter_rows())
