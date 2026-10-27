"""Publishing a run's provenance record: tables, pixel planes and the checks between them.

Built on the real pieces end to end, on local stores: mosaics written by the ingest's own writer
with the switch on, shards staged by the real staging writer from pieces read off those mosaics, and
publication reading both. A 2 x 2 grid of 4-pixel shards, three sensors, two dates each.
"""

from __future__ import annotations

import json
from pathlib import Path

import dask.array as da
import numpy as np
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
    GRANULES,
    SHARD_DATES,
    pixel_sources_group,
    pixel_sources_uri,
    table_uri,
    update_pixels_uri,
)
from tessera_embeddings.storage.zarr_store import open_repo, open_store_as_zarr_group, write_days_windows
from tests.unit.inference_stack import STACK

SIZE, SHARD = 8, 4
DATES = ["2024-06-01", "2024-06-13"]
STORES = {"reflectance": "s2", "sar_ascending": "s1_asc", "sar_descending": "s1_desc"}
ZONE, YEAR, RUN = "33N", 2024, "33N-2024-abcd1234"


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


def _stage(staging: Path, pins: dict[str, str], *, pixels: np.ndarray | None = None) -> tuple[list[str], list[str]]:
    """Three shards embed, one is refused; all stage planes. Returns (embedded, refused)."""
    writer = ZarrWriter(str(staging))
    embedded = []
    for row, col in ((0, 0), (0, 1), (1, 0)):
        chunk = _chunk(row, col)
        writer.write_chunk(
            chunk,
            np.zeros((SHARD, SHARD, 128), dtype=np.int8),
            RUN,
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
        RUN,
        {"s2_obs_count": np.zeros((SHARD, SHARD), dtype=np.uint16)},
        None,
        planes=_planes(pins, refused),
        pixels=pixels,
    )
    return embedded, [refused.label]


def _publish(tmp_path: Path, pins: dict[str, str], embedded: list[str], refused: list[str]) -> dict | None:
    return publish_provenance(
        staging_base=str(tmp_path / "staging"),
        registry_root=str(tmp_path / "registry"),
        zone=ZONE,
        year=YEAR,
        run_id=RUN,
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


def _table(tmp_path: Path, table: str):
    return pq.read_table(table_uri(str(tmp_path / "registry"), table, ZONE, YEAR, RUN))


def test_granules_list_every_slots_granules_in_fusion_order(published) -> None:
    tmp_path, _, summary = published
    table = _table(tmp_path, GRANULES)
    assert summary["granules"] == table.num_rows == 3 * len(DATES) * 2
    rows = table.to_pylist()
    first = next(r for r in rows if r["sensor"] == "s2" and r["fusion_order"] == 1)
    assert (str(first["solar_day"]), first["granule_id"]) == ("2024-06-01", "s2-2024-06-01-1")
    assert shapely.from_wkb(first["geometry"]).bounds == (14.0, 45.0, 15.0, 46.0)
    assert b"geo" in table.schema.metadata and table.schema.metadata[b"run_id"] == RUN.encode()


def test_shard_dates_and_planes_name_the_same_granules(published) -> None:
    """Every row's sources are what its plane ranks, and the plane is the mosaic's index."""
    tmp_path, pins, _ = published
    rows = _table(tmp_path, SHARD_DATES).to_pylist()
    assert {r["tile"] for r in rows} == {"chunk_0_0", "chunk_0_1", "chunk_1_0", "chunk_1_1"}
    repo = open_repo(pixel_sources_uri(str(tmp_path / "registry")))
    group = zarr.open_group(repo.readonly_session("main").store, path=pixel_sources_group(ZONE, YEAR, RUN), mode="r")
    for row in rows:
        r, c = (int(x) for x in row["tile"].removeprefix("chunk_").split("_"))
        t = DATES.index(str(row["solar_day"]))
        plane = group[row["sensor"]][t, r * SHARD : (r + 1) * SHARD, c * SHARD : (c + 1) * SHARD]
        store = next(p for p in pins if STORES[Path(p).stem] == row["sensor"])
        index = open_store_as_zarr_group(store, snapshot_id=pins[store])[SOURCE_INDEX_VAR][
            t, r * SHARD : (r + 1) * SHARD, c * SHARD : (c + 1) * SHARD
        ]
        named = np.where(plane > 0, np.asarray(row["sources"] + [0])[plane.astype(int) - 1] + 1, 0)
        np.testing.assert_array_equal(named, index, err_msg=str(row))
        assert row["valid_px"] == int(np.count_nonzero(plane))
        assert row["embedded_px"] == (0 if row["tile"] == "chunk_1_1" else SHARD * SHARD)


def test_publishing_again_rewrites_the_same_record(published) -> None:
    tmp_path, pins, _ = published
    before = {t: _table(tmp_path, t) for t in (GRANULES, SHARD_DATES)}
    _publish(tmp_path, pins, ["chunk_0_0", "chunk_0_1", "chunk_1_0"], ["chunk_1_1"])
    for table, old in before.items():
        assert _table(tmp_path, table).equals(old), table


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


def test_a_holes_update_publishes_its_masks_and_planes_inside_them(tmp_path: Path) -> None:
    pins = _mosaics(tmp_path / "mosaics")
    holes = np.zeros((SHARD, SHARD), dtype=bool)
    holes[:2] = True
    embedded, refused = _stage(tmp_path / "staging", pins, pixels=holes)
    _publish(tmp_path, pins, embedded, refused)
    masks = zarr.open_group(update_pixels_uri(str(tmp_path / "registry"), ZONE, YEAR, RUN), mode="r")
    assert sorted(masks.array_keys()) == ["chunk_0_0", "chunk_0_1", "chunk_1_0", "chunk_1_1"]
    np.testing.assert_array_equal(masks["chunk_0_0"][:], holes)
    rows = _table(tmp_path, SHARD_DATES).to_pylist()
    assert all(r["valid_px"] <= int(holes.sum()) for r in rows)
