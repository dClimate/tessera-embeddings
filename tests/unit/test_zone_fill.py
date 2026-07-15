"""Zone-fill runner: mask → (stubbed) inference → shard assembly → tag."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.orchestration.runners import zone_fill
from tessera_embeddings.storage import global_store
from tessera_embeddings.storage.empty_store import VarSpec, create_empty_store_from_coords
from tessera_embeddings.storage.zarr_store import open_or_create_repo
from tessera_embeddings.storage.zone_grid import ZoneSpec, easting_coords, northing_coords

log = logging.getLogger("test_zone_fill")

_BAND = 8
_TILE = 64  # miniature shard pitch (== inference tile size, D3)
_INNER = 32
_NY = _NX = 2 * _TILE  # 2x2 tile grid
_ZONE = "32601"
_YEARS = (2024, 2025)

_EMB = ArrayLayout(DIMS_4D, (1, _INNER, _INNER, _BAND), "int8", 0, "zstd", shards=(1, _TILE, _TILE, _BAND))
_SCL = ArrayLayout(DIMS_3D, (1, _INNER, _INNER), "float32", float("nan"), "pcodec", shards=(1, _TILE, _TILE))
SMALL = StoreLayout(name="small", arrays={"embeddings": _EMB, "scales": _SCL})

# 10 m pixels -> metre extents for a _NY x _NX pixel zone.
_SPEC = ZoneSpec(_ZONE, "N", 1, (0.0, _NX * 10.0), (0.0, _NY * 10.0))

_WINDOW = TimeWindow(
    window_start=(2024, 7),
    window_end=(2025, 6),
    months=tuple([(2024, m) for m in range(7, 13)] + [(2025, m) for m in range(1, 7)]),
    window_end_label="2025-06-01",
)


def _config() -> InferenceConfig:
    return InferenceConfig(time_window=_WINDOW, chunk_size=_TILE, num_gpus=0)


def _seed_global(tmp_path) -> str:
    store = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_SPEC], years=_YEARS, layout=SMALL)
    return store


def _make_mosaic(tmp_path, ny: int = _NY, nx: int = _NX) -> str:
    """A minimal reflectance store on the zone's real grid (CRS + coords validated by the runner)."""
    base = str(tmp_path / "mosaics")
    north = northing_coords(_SPEC) if ny == _NY else np.arange(ny, dtype="float64")
    east = easting_coords(_SPEC) if nx == _NX else np.arange(nx, dtype="float64")
    create_empty_store_from_coords(
        f"{base}/reflectance.zarr",
        coords={
            "time": np.array(["2025-01-01"], dtype="datetime64[ns]"),
            "northing": north,
            "easting": east,
        },
        var_specs={"red": VarSpec(dims=("time", "northing", "easting"), dtype=np.dtype("uint16"), chunks=(1, ny, nx))},
        commit_msg="seed test mosaic",
        attrs={"crs": _SPEC.crs},
    )
    return base


def _make_mask(
    tmp_path,
    live_tiles: list[tuple[int, int]],
    *,
    zone_attr: str = _ZONE,
    crs: str | None = None,
    grid_shape: list[int] | None = None,
    tile_shape: tuple[int, int] | None = None,
) -> str:
    """A coverage-bitmap Icechunk repo (ADR-010) with the given tiles live.

    Mirrors :func:`ingest.land_mask.build_all` for one zone: a ``tile_live_2048``
    bool array on the tile grid plus the zone/crs/grid_shape attrs the runner
    guards. The overrides let tests construct wrong-zone / wrong-crs /
    wrong-shape masks that must be rejected.
    """
    crs = _SPEC.crs if crs is None else crs
    grid_shape = [_NY, _NX] if grid_shape is None else grid_shape
    nr, nc = tile_shape if tile_shape is not None else (_NY // _TILE, _NX // _TILE)
    path = str(tmp_path / "coverage.icechunk")
    repo, _ = open_or_create_repo(path)
    session = repo.writable_session("main")
    node = zarr.open_group(session.store, mode="a").require_group(_ZONE)
    tile_live = np.zeros((nr, nc), dtype=bool)
    for row, col in live_tiles:
        tile_live[row, col] = True
    node.create_array("tile_live_2048", data=tile_live, chunks=(nr, nc), dimension_names=("tile_row", "tile_col"))
    node.attrs.update({"zone": zone_attr, "crs": crs, "grid_shape": grid_shape})
    session.commit("seed test coverage")
    return path


def _staging_inference_stub(staged: dict[str, np.ndarray]):
    """A run_inference stand-in that stages every tile and reports success.

    Records what it staged into ``staged`` (label -> embeddings) so tests can
    compare the assembled store against it.
    """

    def stub(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        rng = np.random.default_rng(5)
        writer = ZarrWriter(staging_base, embedding_dim=_BAND)
        results = []
        for chunk in chunks:
            emb = rng.integers(-100, 100, size=(chunk.height, chunk.width, _BAND)).astype(np.int8)
            scales = rng.random((chunk.height, chunk.width)).astype(np.float32)
            writer.write_chunk(chunk, emb, run_id, scales=scales)
            staged[chunk.label] = emb
            results.append({"chunk": chunk.label, "status": "success", "valid_pixels": 1, "elapsed_sec": 0.0})
        return results

    return stub


def test_fill_zone_year_end_to_end(tmp_path, monkeypatch):
    """Live tiles land as shards at the year index; the cell is tagged and cleaned up."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0), (0, 1), (1, 1)])
    staged: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", _staging_inference_stub(staged))

    summary = zone_fill.fill_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2025,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="runZ",
    )

    assert summary["empty"] is False
    assert summary["live_tiles"] == 3
    assert summary["total_tiles"] == 4
    assert summary["succeeded"] == 3
    assert summary["tag"] == "zone-32601-2025"

    repo = global_store.open_global_repo(store)
    assert repo.lookup_tag("zone-32601-2025") == summary["snapshot_id"]
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    assert node.attrs["years_complete"] == [2025]
    assert node.attrs["runs"]["2025"]["run_id"] == "runZ"
    # 2025 is index 1 on the (2024, 2025) axis; staged tiles match, ocean tile is fill.
    result = np.asarray(node["embeddings"][1])
    assert np.all(result[_TILE:, :_TILE] == 0), "unmasked tile must stay at fill"
    for label, emb in staged.items():
        row, col = (int(p) for p in label.split("_")[1:])
        got = result[row * _TILE : (row + 1) * _TILE, col * _TILE : (col + 1) * _TILE]
        np.testing.assert_array_equal(got, emb, err_msg=label)
    # Staging cleaned up after success.
    assert not (tmp_path / "staging" / "runZ").exists()


def test_all_ocean_cell_marked_complete_without_inference(tmp_path, monkeypatch):
    """An all-ocean cell lands (years_complete + tag) with no inference at all."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [])  # nothing lives

    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_inference must not be called for an all-ocean cell")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)

    summary = zone_fill.fill_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2024,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="runE",
    )

    assert summary["empty"] is True
    assert summary["live_tiles"] == 0
    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    assert node.attrs["years_complete"] == [2024]
    assert node.attrs["runs"]["2024"] == {**node.attrs["runs"]["2024"], "run_id": "runE", "empty": True}
    assert repo.lookup_tag("zone-32601-2024") == summary["snapshot_id"]


def test_unseeded_zone_raises(tmp_path):
    store = _seed_global(tmp_path)
    with pytest.raises(ValueError, match="not seeded"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone="32660",
            year=2025,
            land_mask_path=str(tmp_path / "mask.zarr"),
            mosaic_base=str(tmp_path / "mosaics"),
            staging_base=str(tmp_path / "staging"),
            config=_config(),
            num_actors=1,
            log=log,
        )


def test_chunk_size_shard_mismatch_raises(tmp_path):
    store = _seed_global(tmp_path)
    config = InferenceConfig(time_window=_WINDOW, chunk_size=_TILE * 2, num_gpus=0)
    with pytest.raises(ValueError, match="1 inference tile == 1 shard"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2025,
            land_mask_path=str(tmp_path / "mask.zarr"),
            mosaic_base=str(tmp_path / "mosaics"),
            staging_base=str(tmp_path / "staging"),
            config=config,
            num_actors=1,
            log=log,
        )


def test_mosaic_grid_mismatch_raises(tmp_path):
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path, ny=_NY + _TILE)  # taller than the zone grid
    with pytest.raises(ValueError, match="does not match"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2025,
            land_mask_path=_make_mask(tmp_path, [(0, 0)]),
            mosaic_base=mosaic_base,
            staging_base=str(tmp_path / "staging"),
            config=_config(),
            num_actors=1,
            log=log,
        )


def test_inference_failure_aborts_before_assembly(tmp_path, monkeypatch):
    """A failed tile raises and neither assembles nor tags."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0)])

    def failing_inference(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        return [{"chunk": chunks[0].label, "status": "failed", "error": "boom"}]

    monkeypatch.setattr(zone_fill, "run_inference", failing_inference)

    with pytest.raises(RuntimeError, match="1 tiles failed"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2025,
            land_mask_path=mask,
            mosaic_base=mosaic_base,
            staging_base=str(tmp_path / "staging"),
            config=_config(),
            num_actors=1,
            log=log,
        )

    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    assert node.attrs["years_complete"] == []
    assert "zone-32601-2025" not in repo.list_tags()


def test_completed_and_tagged_cell_short_circuits(tmp_path, monkeypatch):
    """A retry of a landed, tagged cell returns its snapshot without re-running anything."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0)])
    staged: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", _staging_inference_stub(staged))

    first = zone_fill.fill_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2025,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="run1",
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a completed+tagged cell must not re-run inference")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    retry = zone_fill.fill_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2025,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="run2",
    )
    assert retry["already_complete"] is True
    assert retry["snapshot_id"] == first["snapshot_id"]
    assert retry["tag"] == first["tag"]


def test_all_tiles_skipped_marks_complete_empty(tmp_path, monkeypatch):
    """Live tiles that all skip (zero valid pixels) land as an empty cell, tagged."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0), (1, 1)])

    def all_skip_inference(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        writer = ZarrWriter(staging_base, embedding_dim=_BAND)
        results = []
        for chunk in chunks:
            writer.write_skip_marker(chunk, run_id)
            results.append({"chunk": chunk.label, "status": "skipped", "valid_pixels": 0, "elapsed_sec": 0.0})
        return results

    monkeypatch.setattr(zone_fill, "run_inference", all_skip_inference)

    summary = zone_fill.fill_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2025,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="runS",
    )

    assert summary["empty"] is True
    assert summary["skipped"] == 2
    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    assert node.attrs["years_complete"] == [2025]
    assert node.attrs["runs"]["2025"]["empty"] is True
    assert repo.lookup_tag("zone-32601-2025") == summary["snapshot_id"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"zone_attr": "32602"}, "read positionally"),
        ({"crs": "EPSG:99999"}, "read positionally"),
        ({"grid_shape": [_NY // 2, _NX // 2]}, "read positionally"),
    ],
)
def test_wrong_zone_coverage_mask_raises(tmp_path, kwargs, match):
    """A coverage group whose zone/CRS/grid_shape attrs disagree with the seeded
    group must be loud — all same-hemisphere zones share one shape, so a
    wrong-zone mask would otherwise be read positionally and misclassify tiles.
    """
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0)], **kwargs)

    with pytest.raises(ValueError, match=match):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2025,
            land_mask_path=mask,
            mosaic_base=mosaic_base,
            staging_base=str(tmp_path / "staging"),
            config=_config(),
            num_actors=1,
            log=log,
        )


def test_coverage_bitmap_shape_mismatch_raises(tmp_path):
    """A tile_live array inconsistent with the (attr-declared) grid is rejected."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    # grid_shape attr says the full zone, but the bitmap is the wrong tile shape.
    mask = _make_mask(tmp_path, [], tile_shape=(1, 1))

    with pytest.raises(ValueError, match="inconsistent with the seeded grid"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2025,
            land_mask_path=mask,
            mosaic_base=mosaic_base,
            staging_base=str(tmp_path / "staging"),
            config=_config(),
            num_actors=1,
            log=log,
        )


def test_all_ocean_cell_skips_missing_mosaic(tmp_path, monkeypatch):
    """An all-ocean cell is marked empty WITHOUT reading the mosaic — the coverage
    check precedes read_spatial_coords, so a zone whose mosaic was never ingested
    still converges instead of failing on a missing reflectance store.
    """
    store = _seed_global(tmp_path)
    mask = _make_mask(tmp_path, [])  # no live tiles

    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_inference must not run for an all-ocean cell")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    summary = zone_fill.fill_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2024,
        land_mask_path=mask,
        mosaic_base=str(tmp_path / "does_not_exist"),  # missing mosaic must not be read
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="runNM",
    )
    assert summary["empty"] is True
    assert summary["live_tiles"] == 0


def test_zone_has_live_tiles_true(tmp_path):
    """The preflight reports live coverage so the flow provisions a cluster."""
    mask = _make_mask(tmp_path, [(0, 0)])
    assert zone_fill.zone_has_live_tiles(mask, _ZONE) is True


def test_zone_has_live_tiles_false(tmp_path):
    """An all-ocean coverage bitmap reports no live tiles (skip the cluster)."""
    mask = _make_mask(tmp_path, [])
    assert zone_fill.zone_has_live_tiles(mask, _ZONE) is False


def test_off_axis_year_fails_before_inference(tmp_path, monkeypatch):
    """An off-axis year dies before any inference is dispatched."""
    store = _seed_global(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("inference must not run for an off-axis year")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    window = TimeWindow(
        window_start=(2015, 1),
        window_end=(2015, 12),
        months=tuple((2015, m) for m in range(1, 13)),
        window_end_label="2015-12-01",
    )
    with pytest.raises(ValueError, match="not on 32601's pre-allocated time axis"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2015,
            land_mask_path=str(tmp_path / "mask.zarr"),
            mosaic_base=str(tmp_path / "mosaics"),
            staging_base=str(tmp_path / "staging"),
            config=InferenceConfig(time_window=window, chunk_size=_TILE, num_gpus=0),
            num_actors=1,
            log=log,
        )


def test_window_year_mismatch_raises(tmp_path):
    """A time_window that never touches the target year is an operator error."""
    store = _seed_global(tmp_path)
    # _WINDOW covers 2024-07..2025-06; year=2024 and 2025 are both fine — but a
    # cloned invocation editing year to a slot outside the window must die.
    window = TimeWindow(
        window_start=(2025, 1),
        window_end=(2025, 12),
        months=tuple((2025, m) for m in range(1, 13)),
        window_end_label="2025-12-01",
    )
    with pytest.raises(ValueError, match="must overlap the calendar-year slot"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2024,
            land_mask_path=str(tmp_path / "mask.zarr"),
            mosaic_base=str(tmp_path / "mosaics"),
            staging_base=str(tmp_path / "staging"),
            config=InferenceConfig(time_window=window, chunk_size=_TILE, num_gpus=0),
            num_actors=1,
            log=log,
        )


def test_landed_but_untagged_cell_is_retagged_without_rerun(tmp_path, monkeypatch):
    """A crash between the fill commit and the tag: retry tags the tip, no re-inference.

    The crash state is constructed by running the fill WITHOUT the tag step
    (assemble_global directly) — icechunk forbids recreating a deleted tag, so
    deleting a landed tag would not reproduce the crash window.
    """
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0)])
    # Stage one tile and land the fill commit (years_complete advances) with no tag.
    rng = np.random.default_rng(5)
    writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=_BAND)
    chunk_00 = zone_fill.enumerate_chunks(_NY, _NX, _TILE)[0]
    writer.write_chunk(
        chunk_00,
        rng.integers(-100, 100, size=(_TILE, _TILE, _BAND)).astype(np.int8),
        "run1",
        scales=rng.random((_TILE, _TILE)).astype(np.float32),
    )
    writer.assemble_global(store, _ZONE, year=2025, run_id="run1", n_workers=1)
    repo = global_store.open_global_repo(store)
    assert "zone-32601-2025" not in repo.list_tags()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a landed cell must not re-run inference just to re-tag")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    retry = zone_fill.fill_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2025,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="run2",
    )
    assert retry["already_complete"] is True
    assert repo.lookup_tag("zone-32601-2025") == retry["snapshot_id"]
