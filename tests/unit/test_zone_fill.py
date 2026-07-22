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
_ZONE = "01N"
_YEARS = (2024, 2025)

_EMB = ArrayLayout(DIMS_4D, (1, _INNER, _INNER, _BAND), "int8", 0, "zstd", shards=(1, _TILE, _TILE, _BAND))
_SCL = ArrayLayout(DIMS_3D, (1, _INNER, _INNER), "float32", float("nan"), "pcodec", shards=(1, _TILE, _TILE))
SMALL = StoreLayout(name="small", arrays={"embeddings": _EMB, "scales": _SCL})

# 10 m pixels -> metre extents for a _NY x _NX pixel zone.
_SPEC = ZoneSpec("32601", "N", 1, (0.0, _NX * 10.0), (0.0, _NY * 10.0))


def _window(year: int) -> TimeWindow:
    """The exact Jan-Dec calendar-year window the global store requires (strict gate)."""
    return TimeWindow(
        window_start=(year, 1),
        window_end=(year, 12),
        months=tuple((year, m) for m in range(1, 13)),
        window_end_label=f"{year}-12-01",
    )


_WINDOW = _window(2025)


def _config(year: int = 2025) -> InferenceConfig:
    return InferenceConfig(time_window=_window(year), chunk_size=_TILE, num_gpus=0)


def _seed_global(tmp_path) -> str:
    store = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [_SPEC], years=_YEARS, layout=SMALL)
    return store


def _make_mosaic_store(path: str, north: np.ndarray, east: np.ndarray, ny: int, nx: int, crs: str) -> None:
    """One mosaic child store (reflectance or SAR) on the given grid."""
    create_empty_store_from_coords(
        path,
        coords={
            "time": np.array(["2025-01-01"], dtype="datetime64[ns]"),
            "northing": north,
            "easting": east,
        },
        var_specs={"red": VarSpec(dims=("time", "northing", "easting"), dtype=np.dtype("uint16"), chunks=(1, ny, nx))},
        commit_msg="seed test mosaic store",
        attrs={"crs": crs},
    )


def _make_mosaic(tmp_path, ny: int = _NY, nx: int = _NX, *, sar: bool = True) -> str:
    """Reflectance + both SAR stores on the zone grid (CRS + coords validated by the runner).

    The runner validates every ACTIVE store's grid, so a realistic mosaic carries
    the SAR stores too; ``sar=False`` omits them to exercise a missing-SAR mosaic.
    """
    base = str(tmp_path / "mosaics")
    north = northing_coords(_SPEC) if ny == _NY else np.arange(ny, dtype="float64")
    east = easting_coords(_SPEC) if nx == _NX else np.arange(nx, dtype="float64")
    _make_mosaic_store(f"{base}/reflectance.zarr", north, east, ny, nx, _SPEC.crs)
    if sar:
        for orbit in ("ascending", "descending"):
            _make_mosaic_store(f"{base}/sar_{orbit}.zarr", north, east, ny, nx, _SPEC.crs)
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
    assert summary["tag"] == "zone-01N-2025"

    repo = global_store.open_global_repo(store)
    assert repo.lookup_tag("zone-01N-2025") == summary["snapshot_id"]
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    assert node.attrs["years_complete"] == [2025]
    assert node.attrs["runs"]["2025"]["run_id"] == "runZ"
    # Label accuracy under the calendar-year GUARANTEE: the slot's `time` point is
    # Jan 1 (the window start) and the seeded time_bnds state the slot's true
    # interval [Jan 1, Dec 31] — exactly the window the strict gate required.
    assert node["time"].attrs["bounds"] == "time_bnds"
    bnds = np.asarray(node["time_bnds"][1]).astype("datetime64[ns]")  # 2025 slot = index 1
    assert list(bnds.astype("datetime64[D]").astype(str)) == list(_WINDOW.to_date_range())
    assert np.asarray(node["time"][1]).astype("datetime64[ns]").astype("datetime64[D]").astype(str) == "2025-01-01"
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
        config=_config(2024),
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
    assert repo.lookup_tag("zone-01N-2024") == summary["snapshot_id"]


def test_unseeded_zone_raises(tmp_path):
    store = _seed_global(tmp_path)
    with pytest.raises(ValueError, match="not seeded"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone="60N",
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


def test_wrong_grid_sar_store_raises(tmp_path, monkeypatch):
    """A SAR store on a different grid than reflectance is rejected — SAR is read
    by positional slice, so a stale/wrong-zone SAR store (right shape, wrong
    coordinates) would silently misgeoreference the fill.
    """
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path, sar=False)  # reflectance on the zone grid, no SAR yet
    # sar_ascending on a SHIFTED easting grid (right shape, wrong coordinates);
    # sar_descending correct, so the failure is unambiguously the ascending store.
    shifted_east = easting_coords(_SPEC) + 10_000.0
    _make_mosaic_store(f"{mosaic_base}/sar_ascending.zarr", northing_coords(_SPEC), shifted_east, _NY, _NX, _SPEC.crs)
    _make_mosaic_store(
        f"{mosaic_base}/sar_descending.zarr", northing_coords(_SPEC), easting_coords(_SPEC), _NY, _NX, _SPEC.crs
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("inference must not run when a SAR grid is invalid")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    with pytest.raises(ValueError, match="SAR ascending"):
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
    assert "zone-01N-2025" not in repo.list_tags()


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
    assert repo.lookup_tag("zone-01N-2025") == summary["snapshot_id"]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"zone_attr": "02N"}, "read positionally"),
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
        config=_config(2024),
        num_actors=1,
        log=log,
        run_id="runNM",
    )
    assert summary["empty"] is True
    assert summary["live_tiles"] == 0


def test_zone_year_complete_reflects_years_complete(tmp_path, monkeypatch):
    """The preflight helper reports False before a fill (and for an unseeded
    zone) and True once the (zone, year) has landed.
    """
    store = _seed_global(tmp_path)
    assert zone_fill.zone_year_complete(store, _ZONE, 2025) is False
    assert zone_fill.zone_year_complete(store, "60N", 2025) is False  # not seeded in this store

    monkeypatch.setattr(zone_fill, "run_inference", _staging_inference_stub({}))
    zone_fill.fill_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2025,
        land_mask_path=_make_mask(tmp_path, [(0, 0)]),
        mosaic_base=_make_mosaic(tmp_path),
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="rc",
    )
    assert zone_fill.zone_year_complete(store, _ZONE, 2025) is True
    assert zone_fill.zone_year_complete(store, _ZONE, 2024) is False


def test_zone_year_on_axis(tmp_path):
    """The preflight reports True for a seeded year, False off-axis / unseeded —
    so the flow can decline Ray for a year the runner would reject anyway.
    """
    store = _seed_global(tmp_path)
    assert zone_fill.zone_year_on_axis(store, _ZONE, 2025) is True
    assert zone_fill.zone_year_on_axis(store, _ZONE, 2024) is True
    assert zone_fill.zone_year_on_axis(store, _ZONE, 2026) is False  # off the seeded axis
    assert zone_fill.zone_year_on_axis(store, "60N", 2025) is False  # not seeded in this store


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
    with pytest.raises(ValueError, match="not on 01N's pre-allocated time axis"):
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
    """A window for a DIFFERENT year than the target slot is an operator error
    (e.g. a cloned invocation whose year was edited but not its time_window).
    """
    store = _seed_global(tmp_path)
    with pytest.raises(ValueError, match="guarantees calendar-year slots"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2024,
            land_mask_path=str(tmp_path / "mask.zarr"),
            mosaic_base=str(tmp_path / "mosaics"),
            staging_base=str(tmp_path / "staging"),
            config=_config(2025),  # exact Jan-Dec 2025 window, but year=2024
            num_actors=1,
            log=log,
        )


def test_rolling_window_rejected(tmp_path):
    """The calendar-year gate: a rolling 12-month window that merely OVERLAPS the
    target year is rejected — the store guarantees calendar-year slots (a rolling
    window's label would be inaccurate, CF containment), and the error points
    non-calendar consumers at the single-ROI `12mo_window_end` path.
    """
    store = _seed_global(tmp_path)
    rolling = TimeWindow(
        window_start=(2024, 7),
        window_end=(2025, 6),
        months=tuple([(2024, m) for m in range(7, 13)] + [(2025, m) for m in range(1, 7)]),
        window_end_label="2025-06-01",
    )
    with pytest.raises(ValueError, match="12mo_window_end"):
        zone_fill.fill_zone_year(
            store_path=store,
            zone=_ZONE,
            year=2025,  # rolling window overlaps 2025 — still rejected
            land_mask_path=str(tmp_path / "mask.zarr"),
            mosaic_base=str(tmp_path / "mosaics"),
            staging_base=str(tmp_path / "staging"),
            config=InferenceConfig(time_window=rolling, chunk_size=_TILE, num_gpus=0),
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
    assert "zone-01N-2025" not in repo.list_tags()

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
    assert repo.lookup_tag("zone-01N-2025") == retry["snapshot_id"]


# ===========================================================================
# Phase split: infer_zone_year / assemble_zone_year (the sequential runner's API)
# ===========================================================================


def test_phase_split_matches_composed_fill(tmp_path, monkeypatch):
    """Infer → handoff → assemble lands the same store state and summary shape
    as the composed fill_zone_year (which is their back-to-back composition).
    """
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0), (1, 1)])
    staged: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", _staging_inference_stub(staged))

    handoff = zone_fill.infer_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2025,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="runP",
    )
    assert handoff.done is None
    assert len(handoff.live) == 2 and len(handoff.results) == 2

    summary = zone_fill.assemble_zone_year(
        handoff,
        store_path=store,
        staging_base=str(tmp_path / "staging"),
        log=log,
    )
    assert summary["empty"] is False and summary["succeeded"] == 2
    assert summary["tag"] == zone_fill.zone_year_tag(_ZONE, 2025)
    # The year is recorded complete — the composed path's end state.
    assert zone_fill.zone_year_complete(store, _ZONE, 2025)


def test_infer_phase_threads_retirement_gate(tmp_path, monkeypatch):
    """The sequential runner's retire_idle_actors=False must reach run_inference —
    it is what keeps a zone tail from draining the shared cluster.
    """
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0)])
    seen: dict = {}
    staged: dict[str, np.ndarray] = {}
    base_stub = _staging_inference_stub(staged)

    def recording_stub(*args, **kwargs):
        seen.update(kwargs)
        return base_stub(*args, **kwargs)

    monkeypatch.setattr(zone_fill, "run_inference", recording_stub)
    zone_fill.infer_zone_year(
        store_path=store,
        zone=_ZONE,
        year=2025,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        staging_base=str(tmp_path / "staging"),
        config=_config(),
        num_actors=1,
        log=log,
        run_id="runR",
        retire_idle_actors=False,
    )
    assert seen["retire_idle_actors"] is False


def test_terminal_handoff_passes_through_assembly(tmp_path):
    done = {"zone": _ZONE, "year": 2025, "already_complete": True}
    handoff = zone_fill.ZoneFillHandoff(
        zone=_ZONE, year=2025, run_id="r", t0=0.0, summary={}, live=[], results=[], done=done
    )
    assert zone_fill.assemble_zone_year(handoff, store_path="unused", staging_base="unused", log=log) is done


def test_zone_live_tile_count(tmp_path):
    mask = _make_mask(tmp_path, [(0, 0), (0, 1), (1, 0)])
    assert zone_fill.zone_live_tile_count(mask, _ZONE) == 3
    assert zone_fill.zone_has_live_tiles(mask, _ZONE) is True
    empty_mask = _make_mask(tmp_path / "m2", [])
    assert zone_fill.zone_live_tile_count(empty_mask, _ZONE) == 0
    assert zone_fill.zone_has_live_tiles(empty_mask, _ZONE) is False
