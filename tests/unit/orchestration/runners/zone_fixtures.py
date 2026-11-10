"""A miniature campaign zone for the zone-fill and zone-update runner tests.

A 2x2 grid of 64-px tiles on zone 01N, seeded for 2024 and 2025 under a small layout, with helpers
for the land mask, the input mosaics and an inference stand-in that stages every tile.
"""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, DIMS_4D_BAND_FIRST, ArrayLayout, StoreLayout
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.orchestration.runners import zone_fill
from tessera_embeddings.storage import global_store
from tessera_embeddings.storage.empty_store import VarSpec, create_empty_store_from_coords
from tessera_embeddings.storage.zarr_store import open_or_create_repo
from tessera_embeddings.storage.zone_grid import ZoneSpec, easting_coords, northing_coords
from tests.unit.inference_stack import STACK

log = logging.getLogger("test_zone_fill")

BAND = 8
TILE = 64  # miniature shard pitch (== inference tile size, D3)
INNER = 32
NY = NX = 2 * TILE  # 2x2 tile grid
ZONE = "01N"
YEARS = (2024, 2025)

_EMB = ArrayLayout(DIMS_4D, (1, INNER, INNER, BAND), "int8", 0, "zstd", shards=(1, TILE, TILE, BAND))
_SCL = ArrayLayout(DIMS_3D, (1, INNER, INNER), "float32", float("nan"), "pcodec", shards=(1, TILE, TILE))
SMALL = StoreLayout(name="small", arrays={"embeddings": _EMB, "scales": _SCL}, bands=BAND)
# v2's shape in miniature: band-first, two band shards of two-band chunks, in level group 0 (ADR 031).
_EMB_V2 = ArrayLayout(DIMS_4D_BAND_FIRST, (1, 2, INNER, INNER), "int8", 0, "zstd", shards=(1, BAND // 2, TILE, TILE))
SMALL_V2 = StoreLayout(name="small-v2", arrays={"embeddings": _EMB_V2, "scales": _SCL}, bands=BAND, level="0")
LAYOUTS = pytest.mark.parametrize("layout", [SMALL, SMALL_V2], ids=["v1", "v2"])

# 10 m pixels -> metre extents for a NY x NX pixel zone.
SPEC = ZoneSpec("32601", "N", 1, (0.0, NX * 10.0), (0.0, NY * 10.0))


def calendar_window(year: int) -> TimeWindow:
    """The exact Jan-Dec calendar-year window the global store requires (strict gate)."""
    return TimeWindow(
        window_start=(year, 1),
        window_end=(year, 12),
        months=tuple((year, m) for m in range(1, 13)),
        window_end_label=f"{year}-12-01",
    )


WINDOW = calendar_window(2025)


def zone_config(year: int = 2025) -> InferenceConfig:
    return InferenceConfig(time_window=calendar_window(year), chunk_size=TILE, num_gpus=0)


def runner_kwargs(tmp_path, store, **overrides) -> dict:
    """This module's standard runner arguments, with ``overrides`` applied.

    Every test drives the same runner against the same miniature zone, so the
    eight arguments that never vary are defaulted here and each test passes only
    what it is actually about. Paths are derived from ``tmp_path`` so a test that
    does not build a mask or a mosaic still names one — several gates fire before
    either is opened, and pointing at an absent path is the honest way to say so.
    """
    kwargs = {
        "store_path": store,
        "zone": ZONE,
        "year": 2025,
        "land_mask_path": str(tmp_path / "mask.zarr"),
        "mosaic_base": str(tmp_path / "mosaics"),
        "staging_base": str(tmp_path / "staging"),
        "config": zone_config(),
        "num_actors": 1,
        "log": log,
    }
    return {**kwargs, **overrides}


def run_fill(tmp_path, store, **overrides):
    """``fill_zone_year`` with the standard wiring."""
    return zone_fill.fill_zone_year(**runner_kwargs(tmp_path, store, **overrides))


def run_infer(tmp_path, store, **overrides):
    """``infer_zone_year`` with the standard wiring."""
    return zone_fill.infer_zone_year(**runner_kwargs(tmp_path, store, **overrides))


def seed_global(tmp_path, layout: StoreLayout = SMALL) -> str:
    store = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [SPEC], years=YEARS, layout=layout)
    return store


def make_mosaic_store(path: str, north: np.ndarray, east: np.ndarray, crs: str = SPEC.crs) -> None:
    """One mosaic child store (reflectance or SAR) on the given grid."""
    create_empty_store_from_coords(
        path,
        coords={
            "time": np.array(["2025-01-01"], dtype="datetime64[ns]"),
            "northing": north,
            "easting": east,
        },
        var_specs={
            "red": VarSpec(
                dims=("time", "northing", "easting"), dtype=np.dtype("uint16"), chunks=(1, len(north), len(east))
            )
        },
        commit_msg="seed test mosaic store",
        attrs={"crs": crs},
    )


def make_mosaic(tmp_path, ny: int = NY, nx: int = NX, *, sar: bool = True) -> str:
    """Reflectance + both SAR stores on the zone grid (CRS + coords validated by the runner).

    The runner validates every ACTIVE store's grid, so a realistic mosaic carries
    the SAR stores too; ``sar=False`` omits them to exercise a missing-SAR mosaic.
    """
    base = str(tmp_path / "mosaics")
    north = northing_coords(SPEC) if ny == NY else np.arange(ny, dtype="float64")
    east = easting_coords(SPEC) if nx == NX else np.arange(nx, dtype="float64")
    make_mosaic_store(f"{base}/reflectance.zarr", north, east)
    if sar:
        for orbit in ("ascending", "descending"):
            make_mosaic_store(f"{base}/sar_{orbit}.zarr", north, east)
    return base


def make_mask(
    tmp_path,
    live_tiles: list[tuple[int, int]],
    *,
    attrs: dict | None = None,
    tile_shape: tuple[int, int] | None = None,
) -> str:
    """A coverage-bitmap Icechunk repo (ADR-010) with the given tiles live.

    Mirrors :func:`ingest.land_mask.build_all` for one zone: a ``tile_live_2048``
    bool array on the tile grid plus the zone/crs/grid_shape attrs the runner
    guards. ``attrs`` overrides those attrs, so tests can construct wrong-zone / wrong-crs /
    wrong-shape masks that must be rejected.
    """
    nr, nc = tile_shape if tile_shape is not None else (NY // TILE, NX // TILE)
    path = str(tmp_path / "coverage.icechunk")
    repo, _ = open_or_create_repo(path)
    session = repo.writable_session("main")
    node = zarr.open_group(session.store, mode="a").require_group(ZONE)
    tile_live = np.zeros((nr, nc), dtype=bool)
    for row, col in live_tiles:
        tile_live[row, col] = True
    node.create_array("tile_live_2048", data=tile_live, chunks=(nr, nc), dimension_names=("tile_row", "tile_col"))
    node.attrs.update({"zone": ZONE, "crs": SPEC.crs, "grid_shape": [NY, NX]} | (attrs or {}))
    session.commit("seed test coverage")
    return path


def staging_inference_stub(staged: dict[str, np.ndarray], *, band_first: bool = False):
    """A run_inference stand-in that stages every tile and reports success.

    Records what it staged into ``staged`` (label -> embeddings, as ``(y, x, band)``) so tests
    can compare the assembled store against it. *band_first* stages as a v2 actor does.
    """

    def stub(*args, **kwargs):
        # Bound against the real signature, so the runner's positional call is checked too.
        call = inspect.signature(run_inference).bind(*args, **kwargs).arguments
        chunks, staging_base, run_id = call["chunks"], call["staging_base"], call["run_id"]
        rng = np.random.default_rng(5)
        writer = ZarrWriter(staging_base, embedding_dim=BAND, band_first=band_first)
        results = []
        for chunk in chunks:
            emb = rng.integers(-100, 100, size=(chunk.height, chunk.width, BAND)).astype(np.int8)
            scales = rng.random((chunk.height, chunk.width)).astype(np.float32)
            writer.write_chunk(chunk, emb, run_id, scales=scales, stack=STACK)
            staged[chunk.label] = emb
            results.append({"chunk": chunk.label, "status": "success", "valid_pixels": 1, "elapsed_sec": 0.0})
        return results

    return stub
