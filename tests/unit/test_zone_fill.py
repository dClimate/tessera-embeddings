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
from tessera_embeddings.storage import global_store, shard_writer
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


def _fill(tmp_path, store, **overrides):
    """Call ``fill_zone_year`` with this module's standard wiring.

    Every test drives the same runner against the same miniature zone, so the
    eight arguments that never vary are defaulted here and each test passes only
    what it is actually about. Paths are derived from ``tmp_path`` so a test that
    does not build a mask or a mosaic still names one — several gates fire before
    either is opened, and pointing at an absent path is the honest way to say so.
    """
    kwargs = {
        "store_path": store,
        "zone": _ZONE,
        "year": 2025,
        "land_mask_path": str(tmp_path / "mask.zarr"),
        "mosaic_base": str(tmp_path / "mosaics"),
        "staging_base": str(tmp_path / "staging"),
        "config": _config(),
        "num_actors": 1,
        "log": log,
    }
    return zone_fill.fill_zone_year(**{**kwargs, **overrides})


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


def _resuming_inference_stub(staged: dict[str, np.ndarray]):
    """A run_inference stand-in that resumes from staging exactly as the real one does.

    Mirrors the production resume prologue: scan the staging prefix, drop everything it
    already resolved, infer only the remainder, and report the resolved tiles as
    ``resumed`` successes. That last part is the hazard a resume-safety test needs — the
    scan folds skip markers into "already done", so a leg that finishes a run reports a
    SUCCESS for a tile it staged nothing for and whose skip marker an earlier leg wrote.
    """

    def stub(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        writer = ZarrWriter(staging_base, embedding_dim=_BAND)
        already = writer.scan_existing_staged_chunks(run_id, chunks, log=log)
        results = [
            {"chunk": label, "status": "success", "valid_pixels": 0, "elapsed_sec": 0.0, "resumed": True}
            for label in sorted(already)
        ]
        rng = np.random.default_rng(7)
        for chunk in (c for c in chunks if c.label not in already):
            emb = rng.integers(-100, 100, size=(chunk.height, chunk.width, _BAND)).astype(np.int8)
            writer.write_chunk(chunk, emb, run_id, scales=rng.random((chunk.height, chunk.width)).astype(np.float32))
            staged[chunk.label] = emb
            results.append({"chunk": chunk.label, "status": "success", "valid_pixels": 1, "elapsed_sec": 0.0})
        return results

    return stub


def test_a_resumed_fill_records_skips_an_earlier_leg_wrote(tmp_path, monkeypatch):
    """The published skip record must come from the staging prefix, not the last leg.

    Skip markers persist across legs of one run id, and the resume scan folds them into
    "already staged" — so the leg that finishes a fill reports a resumed SUCCESS for a
    tile it staged nothing for. A record built from that leg's own results would say the
    year lost nothing while publishing fill over the skipped tiles, which is precisely
    the silent loss the field exists to close.
    """
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0), (1, 1)])

    # Leg 1 of run "runRS": tile (1,1) had no valid pixels and left a skip marker;
    # nothing else landed before the leg ended.
    leg1 = ZarrWriter(str(tmp_path / "staging"), embedding_dim=_BAND)
    chunk_11 = next(c for c in zone_fill.enumerate_chunks(_NY, _NX, _TILE) if (c.row, c.col) == (1, 1))
    leg1.write_skip_marker(chunk_11, "runRS")

    # Leg 2 stages only (0,0) and reports (1,1) as a resumed success.
    staged: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", _resuming_inference_stub(staged))
    summary = _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="runRS")

    assert summary["empty"] is False
    assert list(staged) == ["chunk_0_0"], "leg 2 must have staged nothing for the skipped tile"
    assert summary["resumed"] == 1

    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    skips = dict(node.attrs["runs"])["2025"]["optical_skips"]
    assert skips["labels"] == [chunk_11.label]
    assert skips["tiles_skipped"] == 1
    assert skips["tiles_live"] == 2
    # And the record describes what the year actually holds: that tile reads as fill.
    assert np.all(np.asarray(node["embeddings"][1])[_TILE:, _TILE:] == 0)


def test_fill_zone_year_end_to_end(tmp_path, monkeypatch):
    """Live tiles land as shards at the year index; the cell is tagged and cleaned up."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0), (0, 1), (1, 1)])
    staged: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", _staging_inference_stub(staged))

    summary = _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="runZ")

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
    # interval, half-open [Jan 1 2025, Jan 1 2026) — the window the strict gate required.
    assert node["time"].attrs["bounds"] == "time_bnds"
    bnds = np.asarray(node["time_bnds"][1]).astype("datetime64[ns]")  # 2025 slot = index 1
    assert list(bnds.astype("datetime64[D]").astype(str)) == ["2025-01-01", "2026-01-01"]
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

    summary = _fill(
        tmp_path,
        store,
        year=2024,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        config=_config(2024),
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
        _fill(tmp_path, store, zone="60N")


@pytest.mark.parametrize("bad", [0, -1, -16])
def test_an_unusable_assembly_worker_override_is_refused_before_inference(tmp_path, bad):
    """A bad worker count must not cost a whole GPU run.

    The value is consumed only after inference, and `n or default` reads 0 as "unset"
    while a negative rides through to `ProcessPoolExecutor(max_workers=<0)` — so both
    used to surface as an assembly-time failure on a cell whose expensive half had
    already run. This calls the runner with no mask and no mosaic on disk: reaching
    the ValueError at all proves the check fires before anything is opened.
    """
    store = _seed_global(tmp_path)
    with pytest.raises(ValueError, match="n_assembly_workers must be >= 1"):
        _fill(tmp_path, store, n_assembly_workers=bad)


def test_the_runner_refuses_a_rule_the_store_does_not_advertise(tmp_path):
    """The store's minimum-depth rule is write-once root identity, so the STORE is the
    authority on what rule its zones were filled under — not the caller's config.

    The Prefect adapter reads the root and substitutes the value before calling. This runner
    is the orchestrator-agnostic entry point and was trusting that: called directly, it would
    run inference under any threshold, publish, and mark the year complete. Zones filled under
    different lines would then sit under one root advertising a single rule, and no later check
    could tell which pixels came from which — a refused pixel is indistinguishable from one
    that had no optical input.

    Asserted rather than substituted: silently replacing a caller's value would let a direct
    caller believe it had configured a line it had not.
    """
    store = _seed_global(tmp_path)  # seeded with no rule, so the store advertises None
    config = InferenceConfig(time_window=_WINDOW, chunk_size=_TILE, num_gpus=0, optical_min_obs=25)
    with pytest.raises(ValueError, match="write-once root identity"):
        _fill(tmp_path, store, config=config)


def test_the_runner_accepts_a_config_matching_the_store(tmp_path):
    """The adapter's substitution must satisfy the check, not trip it — otherwise every
    Prefect fill would fail on the guard meant to protect direct callers.
    """
    store = _seed_global(tmp_path)
    config = InferenceConfig(time_window=_WINDOW, chunk_size=_TILE, num_gpus=0, optical_min_obs=None)
    # This wiring points at an absent mask on purpose, so the fill still fails — just LATER,
    # on the mask, which is what proves the rule check passed rather than short-circuited.
    with pytest.raises(Exception) as exc:
        _fill(tmp_path, store, config=config)
    assert "write-once root identity" not in str(exc.value), "the matching config must clear the rule check"


def test_chunk_size_shard_mismatch_raises(tmp_path):
    store = _seed_global(tmp_path)
    config = InferenceConfig(time_window=_WINDOW, chunk_size=_TILE * 2, num_gpus=0)
    with pytest.raises(ValueError, match="1 inference tile == 1 shard"):
        _fill(tmp_path, store, config=config)


def test_mosaic_grid_mismatch_raises(tmp_path):
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path, ny=_NY + _TILE)  # taller than the zone grid
    with pytest.raises(ValueError, match="does not match"):
        _fill(tmp_path, store, land_mask_path=_make_mask(tmp_path, [(0, 0)]), mosaic_base=mosaic_base)


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
        _fill(tmp_path, store, land_mask_path=_make_mask(tmp_path, [(0, 0)]), mosaic_base=mosaic_base)


def test_inference_failure_aborts_before_assembly(tmp_path, monkeypatch):
    """A failed tile raises and neither assembles nor tags."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0)])

    def failing_inference(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        return [{"chunk": chunks[0].label, "status": "failed", "error": "boom"}]

    monkeypatch.setattr(zone_fill, "run_inference", failing_inference)

    with pytest.raises(RuntimeError, match="1 tiles failed"):
        _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base)

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

    first = _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run1")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a completed+tagged cell must not re-run inference")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    retry = _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run2")
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

    summary = _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="runS")

    assert summary["empty"] is True
    assert summary["skipped"] == 2
    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    assert node.attrs["years_complete"] == [2025]
    assert node.attrs["runs"]["2025"]["empty"] is True
    # The wholly-empty case is stated by the flag alone: a per-tile skip list would
    # restate the zone's land mask on a group attribute and say nothing more.
    assert "optical_skips" not in dict(node.attrs["runs"])["2025"]
    assert repo.lookup_tag("zone-01N-2025") == summary["snapshot_id"]


def test_an_all_skipped_retry_clears_the_previous_attempt_s_shards(tmp_path, monkeypatch):
    """A year marked empty must not leave an earlier attempt's embeddings readable.

    A year lands in two commits — shards, then the completion attrs — so an attempt
    that crashes between them leaves data on a year nothing has marked, and the
    campaign re-dispatches it. If the retry's mosaic makes every tile skip where the
    first attempt found valid pixels, marking the year empty without writing would
    publish those old embeddings under a completion mark and a zone-year tag that both
    say the cell holds nothing.
    """
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0), (1, 1)])

    # Attempt 1 produces real embeddings and its SHARD commit lands, then the attrs
    # commit fails — the exact gap `write_year_shards` documents. The year is left
    # holding data that nothing has marked and nothing has tagged.
    staged: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", _staging_inference_stub(staged))

    def _attrs_commit_fails(*_a, **_k):
        raise RuntimeError("attrs commit exhausted its retries")

    monkeypatch.setattr(shard_writer, "commit_year_attrs", _attrs_commit_fails)
    with pytest.raises(RuntimeError, match="attrs commit"):
        _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run1")
    monkeypatch.undo()

    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    assert np.any(np.asarray(node["embeddings"][1]) != 0), "attempt 1's shards must have landed"
    assert node.attrs["years_complete"] == [], "and the year must be unmarked"

    # Attempt 2: every tile skips.
    def all_skip_inference(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        writer = ZarrWriter(staging_base, embedding_dim=_BAND)
        results = []
        for chunk in chunks:
            writer.write_skip_marker(chunk, run_id)
            results.append({"chunk": chunk.label, "status": "skipped", "valid_pixels": 0, "elapsed_sec": 0.0})
        return results

    monkeypatch.setattr(zone_fill, "run_inference", all_skip_inference)
    summary = _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run2")

    assert summary["empty"] is True
    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[_ZONE]
    assert node.attrs["years_complete"] == [2025]
    assert node.attrs["runs"]["2025"]["empty"] is True
    # The published year now reads exactly as an unwritten one: fill, not attempt 1.
    assert np.all(np.asarray(node["embeddings"][1]) == 0)
    assert np.all(np.isnan(np.asarray(node["scales"][1])))


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
        _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base)


def test_coverage_bitmap_shape_mismatch_raises(tmp_path):
    """A tile_live array inconsistent with the (attr-declared) grid is rejected."""
    store = _seed_global(tmp_path)
    mosaic_base = _make_mosaic(tmp_path)
    # grid_shape attr says the full zone, but the bitmap is the wrong tile shape.
    mask = _make_mask(tmp_path, [], tile_shape=(1, 1))

    with pytest.raises(ValueError, match="inconsistent with the seeded grid"):
        _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base)


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
    summary = _fill(
        tmp_path,
        store,
        year=2024,
        land_mask_path=mask,
        mosaic_base=str(tmp_path / "does_not_exist"),
        # missing mosaic must not be read
        staging_base=str(tmp_path / "staging"),
        config=_config(2024),
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
    _fill(
        tmp_path,
        store,
        land_mask_path=_make_mask(tmp_path, [(0, 0)]),
        mosaic_base=_make_mosaic(tmp_path),
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
        _fill(tmp_path, store, year=2015, config=InferenceConfig(time_window=window, chunk_size=_TILE, num_gpus=0))


def _rolling_window() -> TimeWindow:
    """A 12-month window that merely OVERLAPS the target year."""
    return TimeWindow(
        window_start=(2024, 7),
        window_end=(2025, 6),
        months=tuple([(2024, m) for m in range(7, 13)] + [(2025, m) for m in range(1, 7)]),
        window_end_label="2025-06-01",
    )


def _partial_window() -> TimeWindow:
    """Every month inside the target year, but fewer than twelve of them."""
    return TimeWindow(
        window_start=(2025, 1),
        window_end=(2025, 6),
        months=tuple((2025, m) for m in range(1, 7)),
        window_end_label="2025-06-01",
    )


@pytest.mark.parametrize(
    "year,window,match",
    [
        # A window for a DIFFERENT year than the target slot — an operator error, e.g. a
        # cloned invocation whose year was edited but not its time_window.
        pytest.param(2024, _window(2025), "guarantees calendar-year slots", id="wrong-year"),
        # A rolling window overlapping the target year. Rejected even though it is 12
        # months: the slot's seeded time_bnds advertise Jan-Dec, so a rolling window's
        # label would be inaccurate (CF containment). The message points non-calendar
        # consumers at the single-ROI `12mo_window_end` path.
        pytest.param(2025, _rolling_window(), "12mo_window_end", id="rolling"),
        # A same-year PARTIAL window. The gate compares the whole month SET, not just
        # "every month falls in `year`" — a short window would under-fill the slot.
        pytest.param(2025, _partial_window(), "guarantees calendar-year slots", id="partial"),
    ],
)
def test_non_calendar_year_windows_are_rejected(tmp_path, year, window, match):
    """The slot must hold exactly the Jan-Dec year its seeded time_bnds advertise.

    All three rejections happen before any mask or mosaic is opened, which is why
    this can run against paths that do not exist.
    """
    store = _seed_global(tmp_path)
    with pytest.raises(ValueError, match=match):
        _fill(tmp_path, store, year=year, config=InferenceConfig(time_window=window, chunk_size=_TILE, num_gpus=0))


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
    retry = _fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run2")
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


def test_fill_accepts_a_cropped_written_mosaic(tmp_path, monkeypatch):
    """The exact-grid validation passes against a write_day_windows mosaic.

    The cropped ingest seeds schema-only from the zone ROI's geobox and writes
    only live windows; the fill's grid assert reads full-length coords, CRS and
    endpoints. This pins the load-bearing equivalence: a sparse mosaic's DECLARED
    grid — geobox-derived coords included — is byte-compatible with the zone grid
    the fill validates against, so cropping never trips the assert.
    """
    import dask.array as dask_array
    import xarray as xr
    from affine import Affine
    from odc.geo.geobox import GeoBox

    from tessera_embeddings.storage.zarr_store import write_day_windows

    store = _seed_global(tmp_path)
    mask = _make_mask(tmp_path, [(0, 0)])

    # The production zone geobox, exactly as export_zone_roi encodes it:
    # north-up, top-left origin at (easting_min, northing_max).
    geobox = GeoBox((_NY, _NX), Affine(10.0, 0.0, _SPEC.easting[0], 0.0, -10.0, _SPEC.northing[1]), _SPEC.crs)

    class _Roi:
        pass

    roi = _Roi()
    roi.geobox, roi.height, roi.width = geobox, _NY, _NX

    day_ds = xr.Dataset(
        {
            "red": (
                ("time", "northing", "easting"),
                dask_array.full((1, _NY, _NX), 3, dtype=np.uint16, chunks=(1, _TILE, _TILE)),
            )
        },
        coords={"time": np.array(["2025-06-01"], dtype="datetime64[ns]")},
    )
    base = str(tmp_path / "mosaics")
    for name in ("reflectance", "sar_ascending", "sar_descending"):
        write_day_windows(
            f"{base}/{name}.zarr",
            day_ds,
            [(0, _TILE, 0, _TILE)],  # one live window; the rest of the grid is never written
            roi=roi,
            manifest=None,
            baselines={},
            tile_id="roi.zarr",
            crs=_SPEC.crs,
            chunks={"time": 1, "northing": _TILE, "easting": _TILE},
        )

    staged: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", _staging_inference_stub(staged))
    summary = _fill(tmp_path, store, land_mask_path=mask, mosaic_base=base, run_id="runCropped")
    assert summary["empty"] is False and summary["succeeded"] == 1


def test_calendar_gate_is_callable_before_provisioning():
    """The gate decides from config alone, so both flows call it pre-cluster.

    An offset or same-year-partial window is rejected by planning regardless —
    but only after a billable GPU fleet is up (and, on the chained path, after
    look-ahead ingests have started). One definition, two call sites.
    """
    zone_fill.assert_calendar_year_window(_window(2025), 2025)  # exact Jan-Dec: passes

    offset = TimeWindow(
        window_start=(2024, 2),
        window_end=(2025, 1),
        months=tuple([(2024, m) for m in range(2, 13)] + [(2025, 1)]),
        window_end_label="2025-01-01",
    )
    with pytest.raises(ValueError, match="exact January-December 2025 window"):
        zone_fill.assert_calendar_year_window(offset, 2025)

    partial = TimeWindow(
        window_start=(2025, 1),
        window_end=(2025, 6),
        months=tuple((2025, m) for m in range(1, 7)),
        window_end_label="2025-06-01",
    )
    with pytest.raises(ValueError, match="exact January-December 2025 window"):
        zone_fill.assert_calendar_year_window(partial, 2025)  # same year, only 6 months


def test_non_affine_mosaic_axis_raises(tmp_path, monkeypatch):
    """Endpoints and length are not enough to pin an axis, and inference writes positionally.

    A mosaic whose easting axis has the right count and the right first and last values
    but a permuted interior passes every earlier check — shape, CRS, endpoints — while
    describing pixels in a different order than the seeded grid. Inference reads it by
    positional slice, so those pixels would be published at the wrong coordinates with
    nothing anywhere to signal it. The campaign's own ingest cannot produce this (odc
    builds every load against the zone geobox), but ``ingest=False`` accepts a mosaic the
    operator staged, which is the path this guards.
    """
    store = _seed_global(tmp_path)
    mosaic_base = str(tmp_path / "scrambled-mosaics")
    scrambled = easting_coords(_SPEC).copy()
    # Swap two interior samples: same endpoints, same length, same CRS, wrong order.
    # Note this ALSO defeats a per-step spacing test that allows half a pixel of slack —
    # the two bad steps are +/- one pixel, and a wander-and-return pattern would defeat it
    # entirely. Comparing whole vectors against the seeded axis has no such gap.
    scrambled[1], scrambled[2] = scrambled[2], scrambled[1]
    _make_mosaic_store(f"{mosaic_base}/reflectance.zarr", northing_coords(_SPEC), scrambled, _NY, _NX, _SPEC.crs)
    for orbit in ("ascending", "descending"):
        _make_mosaic_store(
            f"{mosaic_base}/sar_{orbit}.zarr", northing_coords(_SPEC), easting_coords(_SPEC), _NY, _NX, _SPEC.crs
        )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("inference must not run on a non-affine mosaic grid")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    with pytest.raises(ValueError, match=r"does not match zone .* seeded axis"):
        _fill(tmp_path, store, land_mask_path=_make_mask(tmp_path, [(0, 0)]), mosaic_base=mosaic_base)
