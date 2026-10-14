"""An update of a single-area store: some tiles, or only their holes, rewritten in a window it holds."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.assembly import SpatialCoords, ZarrWriter
from tessera_embeddings.inference.chunk_spec import enumerate_chunks
from tessera_embeddings.orchestration.runners.plain import _plan_update, _write_update
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.update import base_tag, read_updates

log = logging.getLogger("test_plain_update")

_BAND, _TILE, _INNER = 8, 64, 32
# A whole row of tiles over a short one, as a real area's edge leaves it.
_NY, _NX = _TILE + _TILE // 2, 2 * _TILE
_EMB = ArrayLayout(DIMS_4D, (1, _INNER, _INNER, _BAND), "int8", 0, "zstd", shards=(1, _TILE, _TILE, _BAND))
_SCL = ArrayLayout(DIMS_3D, (1, _INNER, _INNER), "float32", float("nan"), "pcodec", shards=(1, _TILE, _TILE))
SMALL = StoreLayout(name="small", arrays={"embeddings": _EMB, "scales": _SCL})
WINDOW = parse_time_window("June 2025")
CHUNKS = enumerate_chunks(_NY, _NX, chunk_size=_TILE)
#: The grid a store assembled without mosaics is given: pixel indices.
GRID = SpatialCoords(northing=np.arange(_NY), easting=np.arange(_NX))


def _stage(writer: ZarrWriter, run_id: str, value: int, *, holes: bool) -> None:
    """Stage every tile at ``value``; ``holes`` refuses the left half of chunk_0_0 and all of chunk_1_1."""
    for chunk in CHUNKS:
        emb = np.full((chunk.height, chunk.width, _BAND), value, dtype=np.int8)
        scales = np.ones((chunk.height, chunk.width), dtype=np.float32)
        refused = np.zeros_like(scales, dtype=bool)
        if holes and chunk.label == "chunk_0_0":
            refused[:, : chunk.width // 2] = True
        if holes and chunk.label == "chunk_1_1":
            refused[:] = True
        emb[refused], scales[refused] = 0, np.nan
        writer.write_chunk(chunk, emb, run_id, scales=scales)


@pytest.fixture()
def area(tmp_path):
    """A single-area store holding one window, filled with 1s around its holes."""
    roi = str(tmp_path / "roi.zarr")
    zarr.open(roi, mode="w", shape=(_NY, _NX), chunks=(_NY, _NX), dtype="bool")[:] = True
    writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=_BAND)
    _stage(writer, "fill", 1, holes=True)
    output = str(tmp_path / "out.zarr")
    writer.assemble(CHUNKS, _NY, _NX, "fill", output, roi_zarr_path=roi, time_window=WINDOW, n_workers=1, layout=SMALL)
    return {"writer": writer, "output": output, "holes": str(tmp_path / "holes")}


def _plan(area, window: str = WINDOW.window_end_label, spatial: SpatialCoords = GRID, **update):
    return _plan_update(
        {"name": "u1", **update},
        output_path=area["output"],
        roi_name="roi",
        window=window,
        chunks=CHUNKS,
        spatial=spatial,
        extent=(_NY, _NX),
        holes_prefix=area["holes"],
        log=log,
    )


def _update(area, **update):
    """Plan, stage every planned tile at 2s, and write; return the store's root afterwards."""
    planned = _plan(area, **update)
    _stage(area["writer"], "upd", 2, holes=False)
    results = [{"chunk": c.label, "status": "success"} for c in planned.chunks]
    _write_update(area["writer"], planned, output_path=area["output"], run_id="upd", results=results, log=log)
    return zarr.open_group(open_global_repo(area["output"]).readonly_session(branch="main").store, mode="r")


def test_a_holes_update_fills_only_the_holes_and_records_itself(area):
    root = _update(area)
    emb = np.asarray(root["embeddings"][0, :, :, 0])
    assert (emb[:_TILE, : _TILE // 2] == 2).all() and (emb[_TILE:, _TILE:] == 2).all()  # the holes
    assert (emb[:_TILE, _TILE // 2 : _TILE] == 1).all() and (emb[:, :_TILE][_TILE:] == 1).all()  # never touched
    assert not np.isnan(np.asarray(root["scales"][0])).any()
    [record] = read_updates(root, 2025)
    assert (record["update"], record["mode"], record["written"]) == ("u1", "holes", ["chunk_0_0", "chunk_1_1"])
    assert record["kept"] == ["chunk_0_1", "chunk_1_0"]
    assert base_tag("u1") in open_global_repo(area["output"]).list_tags()
    assert _plan(area) is None, "an update already recorded is not run again"


def test_a_shards_update_rewrites_only_its_tiles_whole(area):
    emb = np.asarray(_update(area, mode="shards", tiles=["chunk_1_1"])["embeddings"][0, :, :, 0])
    assert (emb[_TILE:, _TILE:] == 2).all()
    assert (emb[:_TILE, _TILE // 2 :] == 1).all() and (emb[:_TILE, : _TILE // 2] == 0).all()


@pytest.mark.parametrize(
    ("plan", "match"),
    [
        ({"tile": ["chunk_0_0"]}, "takes name, mode and tiles"),
        ({"mode": "all"}, "mode must be"),
        ({"tiles": ["chunk_9_9"]}, "outside the area"),
        ({"window": parse_time_window("June 2024").window_end_label}, "run without `update` first"),
        # Tiles are written by position, so mosaics a pixel off the store's grid would land a pixel off.
        (
            {"spatial": SpatialCoords(northing=GRID.northing, easting=GRID.easting + 10.0)},
            "do not lie on existing store",
        ),
    ],
)
def test_an_update_that_cannot_be_planned_is_refused(area, plan, match):
    with pytest.raises(ValueError, match=match):
        _plan(area, **plan)
