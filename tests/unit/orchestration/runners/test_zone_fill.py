"""Zone-fill runner: mask → (stubbed) inference → shard assembly → tag."""

from __future__ import annotations

import logging

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.orchestration.runners import zone_fill
from tessera_embeddings.storage import global_store, shard_writer
from tessera_embeddings.storage.update import UpdateTarget, commit_update_record, update_record
from tessera_embeddings.storage.zone_grid import easting_coords, northing_coords
from tests.unit.inference_stack import STACK
from tests.unit.orchestration.runners.zone_fixtures import (
    BAND,
    NX,
    NY,
    SMALL,
    SPEC,
    TILE,
    WINDOW,
    YEARS,
    ZONE,
    calendar_window,
    log,
    make_mask,
    make_mosaic,
    make_mosaic_store,
    run_fill,
    run_infer,
    seed_global,
    staging_inference_stub,
    zone_config,
)


@pytest.fixture
def staged(monkeypatch) -> dict[str, np.ndarray]:
    """Replaces inference with a stand-in that stages every tile; label -> what it staged."""
    out: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", staging_inference_stub(out))
    return out


def _resuming_inference_stub(staged: dict[str, np.ndarray]):
    """A run_inference stand-in that resumes from staging exactly as the real one does.

    Mirrors the production resume prologue: scan the staging prefix, drop everything it
    already resolved, infer only the remainder, and report the resolved tiles as
    ``resumed`` successes. That last part is the hazard a resume-safety test needs — the
    scan folds skip markers into "already done", so a leg that finishes a run reports a
    SUCCESS for a tile it staged nothing for and whose skip marker an earlier leg wrote.
    """

    def stub(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        writer = ZarrWriter(staging_base, embedding_dim=BAND)
        already = writer.scan_existing_staged_chunks(run_id, chunks, log=log)
        results = [
            {"chunk": label, "status": "success", "valid_pixels": 0, "elapsed_sec": 0.0, "resumed": True}
            for label in sorted(already)
        ]
        rng = np.random.default_rng(7)
        for chunk in (c for c in chunks if c.label not in already):
            emb = rng.integers(-100, 100, size=(chunk.height, chunk.width, BAND)).astype(np.int8)
            writer.write_chunk(
                chunk,
                emb,
                run_id,
                scales=rng.random((chunk.height, chunk.width)).astype(np.float32),
                stack=STACK,
            )
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
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0), (1, 1)])

    # Leg 1 of run "runRS": tile (1,1) had no valid pixels and left a skip marker;
    # nothing else landed before the leg ended.
    leg1 = ZarrWriter(str(tmp_path / "staging"), embedding_dim=BAND)
    chunk_11 = next(c for c in zone_fill.enumerate_chunks(NY, NX, TILE) if (c.row, c.col) == (1, 1))
    leg1.write_skip_marker(chunk_11, "runRS")

    # Leg 2 stages only (0,0) and reports (1,1) as a resumed success.
    staged: dict[str, np.ndarray] = {}
    monkeypatch.setattr(zone_fill, "run_inference", _resuming_inference_stub(staged))
    summary = run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="runRS")

    assert summary["empty"] is False
    assert list(staged) == ["chunk_0_0"], "leg 2 must have staged nothing for the skipped tile"
    assert summary["resumed"] == 1

    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[ZONE]
    skips = dict(node.attrs["runs"])["2025"]["optical_skips"]
    assert skips["labels"] == [chunk_11.label]
    assert skips["tiles_skipped"] == 1
    assert skips["tiles_live"] == 2
    # And the record describes what the year actually holds: that tile reads as fill.
    assert np.all(np.asarray(node["embeddings"][1])[TILE:, TILE:] == 0)


def test_fill_zone_year_end_to_end(tmp_path, monkeypatch, staged):
    """Live tiles land as shards at the year index; the cell is tagged and cleaned up."""
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0), (0, 1), (1, 1)])

    summary = run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="runZ")

    assert summary["empty"] is False
    assert summary["live_tiles"] == 3
    assert summary["total_tiles"] == 4
    assert summary["succeeded"] == 3
    assert summary["tag"] == "zone-01N-2025"

    repo = global_store.open_global_repo(store)
    assert repo.lookup_tag("zone-01N-2025") == summary["snapshot_id"]
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[ZONE]
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
    assert np.all(result[TILE:, :TILE] == 0), "unmasked tile must stay at fill"
    for label, emb in staged.items():
        row, col = (int(p) for p in label.split("_")[1:])
        got = result[row * TILE : (row + 1) * TILE, col * TILE : (col + 1) * TILE]
        np.testing.assert_array_equal(got, emb, err_msg=label)
    # Staging cleaned up after success.
    assert not (tmp_path / "staging" / "runZ").exists()


def test_unseeded_zone_raises(tmp_path):
    store = seed_global(tmp_path)
    with pytest.raises(ValueError, match="not seeded"):
        run_fill(tmp_path, store, zone="60N")


@pytest.mark.parametrize("bad", [0, -1])
def test_an_unusable_assembly_worker_override_is_refused_before_inference(tmp_path, bad):
    """A bad worker count must not cost a whole GPU run.

    The value is consumed only after inference, and `n or default` reads 0 as "unset"
    while a negative rides through to `ProcessPoolExecutor(max_workers=<0)` — so both
    used to surface as an assembly-time failure on a cell whose expensive half had
    already run. This calls the runner with no mask and no mosaic on disk: reaching
    the ValueError at all proves the check fires before anything is opened.
    """
    store = seed_global(tmp_path)
    with pytest.raises(ValueError, match="n_assembly_workers must be >= 1"):
        run_fill(tmp_path, store, n_assembly_workers=bad)


@pytest.mark.parametrize(
    ("root_rule", "is_update", "config_rule", "refused"),
    [
        (None, False, 25, "advertises"),
        (None, False, None, None),
        (15, False, 1, "advertises"),
        (15, True, 1, None),
        (15, True, None, "advertises"),
        (15, True, 20, "only loosen"),
        (None, True, 1, "only loosen"),
    ],
)
def test_the_runner_applies_only_the_stores_rule_or_an_updates_looser_one(
    tmp_path, root_rule, is_update, config_rule, refused
):
    """The STORE is the authority on what rule a cell is filled under, not the caller's config: a
    fill applies the root's rule exactly, and an update may apply a looser one but never a stricter.

    The Prefect adapter reads the store and substitutes the value before calling. This runner
    is the orchestrator-agnostic entry point: trusting the config, a direct call would fill
    under any threshold, and no later check could tell which pixels came from which — a refused
    pixel is indistinguishable from one that had no optical input. Asserted rather than
    substituted, so a direct caller never believes it configured a line it had not.
    """
    store = _seed_global_with_rule(tmp_path, root_rule)
    update = UpdateTarget("u", "holes", "keep", "global", None, "h") if is_update else None
    config = InferenceConfig(time_window=WINDOW, chunk_size=TILE, num_gpus=0, optical_min_obs=config_rule)
    # A config that clears the check still fails LATER (the cell is unfilled, the mask absent),
    # which proves the check passed rather than short-circuited.
    with pytest.raises(Exception) as exc:
        run_fill(tmp_path, store, config=config, update=update)
    assert all((word in str(exc.value)) is (word == refused) for word in ("advertises", "only loosen"))


def _record_updates(store: str, *lines: tuple[str, int]) -> None:
    """Mark the test cell filled and record earlier updates of it, each at its own line."""
    session = global_store.open_global_repo(store).writable_session("main")
    zarr.open_group(session.store, mode="a")[ZONE].attrs["years_complete"] = [2025]
    session.commit("filled")
    for name, line in lines:
        target = UpdateTarget(name, "shards", "keep", "global", None, "h")
        record = update_record(target, run_id=name, written=["chunk_0_0"], kept=[], optical_min_obs=line)
        commit_update_record(global_store.open_global_repo(store), ZONE, 2025, record=record)


def _update_at(tmp_path, store: str, name: str, line: int | None):
    config = InferenceConfig(time_window=WINDOW, chunk_size=TILE, num_gpus=0, optical_min_obs=line or 15)
    update = UpdateTarget(name, "shards", "keep", "global", None, "h")
    return run_fill(tmp_path, store, config=config, update=update)


@pytest.mark.parametrize(("line", "refused"), [(None, True), (5, True), (1, False)])
def test_an_update_may_not_tighten_a_cell_an_earlier_update_loosened(tmp_path, line, refused):
    """After an update at 1, a later one at the root's 15 or at 5 would drop (``shards``) or count as
    thin (``holes``) the pixels the first one embedded, so the runner refuses it before inference.
    """
    store = _seed_global_with_rule(tmp_path, 15)
    _record_updates(store, ("earlier", 1))
    with pytest.raises(Exception) as exc:
        _update_at(tmp_path, store, "later", line)
    assert ("earlier update" in str(exc.value)) is refused


def test_rerunning_a_recorded_update_is_not_a_tightening(tmp_path):
    """Update a at 5, then b at 1: rerunning a (to catch up its registry master) does no inference and
    writes no pixel, so it settles as already recorded rather than being refused.
    """
    store = _seed_global_with_rule(tmp_path, 15)
    _record_updates(store, ("a", 5), ("b", 1))
    assert _update_at(tmp_path, store, "a", 5)["already_recorded"]


def test_chunk_size_shard_mismatch_raises(tmp_path):
    store = seed_global(tmp_path)
    config = InferenceConfig(time_window=WINDOW, chunk_size=TILE * 2, num_gpus=0)
    with pytest.raises(ValueError, match="1 inference tile == 1 shard"):
        run_fill(tmp_path, store, config=config)


def test_mosaic_grid_mismatch_raises(tmp_path):
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path, ny=NY + TILE)  # taller than the zone grid
    with pytest.raises(ValueError, match="does not match"):
        run_fill(tmp_path, store, land_mask_path=make_mask(tmp_path, [(0, 0)]), mosaic_base=mosaic_base)


def test_wrong_grid_sar_store_raises(tmp_path, monkeypatch):
    """A SAR store on a different grid than reflectance is rejected — SAR is read
    by positional slice, so a stale/wrong-zone SAR store (right shape, wrong
    coordinates) would silently misgeoreference the fill.
    """
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path, sar=False)  # reflectance on the zone grid, no SAR yet
    # sar_ascending on a SHIFTED easting grid (right shape, wrong coordinates);
    # sar_descending correct, so the failure is unambiguously the ascending store.
    shifted_east = easting_coords(SPEC) + 10_000.0
    make_mosaic_store(f"{mosaic_base}/sar_ascending.zarr", northing_coords(SPEC), shifted_east)
    make_mosaic_store(f"{mosaic_base}/sar_descending.zarr", northing_coords(SPEC), easting_coords(SPEC))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("inference must not run when a SAR grid is invalid")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    with pytest.raises(ValueError, match="SAR ascending"):
        run_fill(tmp_path, store, land_mask_path=make_mask(tmp_path, [(0, 0)]), mosaic_base=mosaic_base)


def test_inference_failure_aborts_before_assembly(tmp_path, monkeypatch):
    """A failed tile raises and neither assembles nor tags."""
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0)])

    def failing_inference(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        return [{"chunk": chunks[0].label, "status": "failed", "error": "boom"}]

    monkeypatch.setattr(zone_fill, "run_inference", failing_inference)

    with pytest.raises(RuntimeError, match="1 tiles failed"):
        run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base)

    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[ZONE]
    assert node.attrs["years_complete"] == []
    assert "zone-01N-2025" not in repo.list_tags()


def test_completed_and_tagged_cell_short_circuits(tmp_path, monkeypatch, staged):
    """A retry of a landed, tagged cell returns its snapshot without re-running anything."""
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0)])

    first = run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run1")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a completed+tagged cell must not re-run inference")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    retry = run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run2")
    assert retry["already_complete"] is True
    assert retry["snapshot_id"] == first["snapshot_id"]
    assert retry["tag"] == first["tag"]


def test_an_all_skipped_retry_clears_the_previous_attempt_s_shards(tmp_path, monkeypatch, staged):
    """A year marked empty must not leave an earlier attempt's embeddings readable.

    A year lands in two commits — shards, then the completion attrs — so an attempt
    that crashes between them leaves data on a year nothing has marked, and the
    campaign re-dispatches it. If the retry's mosaic makes every tile skip where the
    first attempt found valid pixels, marking the year empty without writing would
    publish those old embeddings under a completion mark and a zone-year tag that both
    say the cell holds nothing.
    """
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0), (1, 1)])

    # Attempt 1 produces real embeddings and its SHARD commit lands, then the attrs
    # commit fails — the exact gap `write_year_shards` documents. The year is left
    # holding data that nothing has marked and nothing has tagged.

    def _attrs_commit_fails(*_a, **_k):
        raise RuntimeError("attrs commit exhausted its retries")

    monkeypatch.setattr(shard_writer, "commit_year_attrs", _attrs_commit_fails)
    with pytest.raises(RuntimeError, match="attrs commit"):
        run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run1")
    monkeypatch.undo()

    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[ZONE]
    assert np.any(np.asarray(node["embeddings"][1]) != 0), "attempt 1's shards must have landed"
    assert node.attrs["years_complete"] == [], "and the year must be unmarked"

    # Attempt 2: every tile skips.
    def all_skip_inference(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        writer = ZarrWriter(staging_base, embedding_dim=BAND)
        results = []
        for chunk in chunks:
            writer.write_skip_marker(chunk, run_id)
            results.append({"chunk": chunk.label, "status": "skipped", "valid_pixels": 0, "elapsed_sec": 0.0})
        return results

    monkeypatch.setattr(zone_fill, "run_inference", all_skip_inference)
    summary = run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run2")

    assert summary["empty"] is True
    assert summary["skipped"] == 2
    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[ZONE]
    assert node.attrs["years_complete"] == [2025]
    assert node.attrs["runs"]["2025"]["empty"] is True
    # The wholly-empty case is stated by the flag alone: a per-tile skip list would
    # restate the zone's land mask on a group attribute and say nothing more.
    assert "optical_skips" not in dict(node.attrs["runs"])["2025"]
    assert repo.lookup_tag("zone-01N-2025") == summary["snapshot_id"]
    # The published year now reads exactly as an unwritten one: fill, not attempt 1.
    assert np.all(np.asarray(node["embeddings"][1]) == 0)
    assert np.all(np.isnan(np.asarray(node["scales"][1])))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"attrs": {"zone": "02N"}}, "read positionally"),
        ({"attrs": {"crs": "EPSG:99999"}}, "read positionally"),
        ({"attrs": {"grid_shape": [NY // 2, NX // 2]}}, "read positionally"),
    ],
)
def test_wrong_zone_coverage_mask_raises(tmp_path, kwargs, match):
    """A coverage group whose zone/CRS/grid_shape attrs disagree with the seeded
    group must be loud — all same-hemisphere zones share one shape, so a
    wrong-zone mask would otherwise be read positionally and misclassify tiles.
    """
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0)], **kwargs)

    with pytest.raises(ValueError, match=match):
        run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base)


def test_coverage_bitmap_shape_mismatch_raises(tmp_path):
    """A tile_live array inconsistent with the (attr-declared) grid is rejected."""
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path)
    # grid_shape attr says the full zone, but the bitmap is the wrong tile shape.
    mask = make_mask(tmp_path, [], tile_shape=(1, 1))

    with pytest.raises(ValueError, match="inconsistent with the seeded grid"):
        run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base)


def test_all_ocean_cell_skips_missing_mosaic(tmp_path, monkeypatch):
    """An all-ocean cell is marked empty WITHOUT reading the mosaic — the coverage
    check precedes read_spatial_coords, so a zone whose mosaic was never ingested
    still converges instead of failing on a missing reflectance store.
    """
    store = seed_global(tmp_path)
    mask = make_mask(tmp_path, [])  # no live tiles

    def fail_if_called(*args, **kwargs):
        raise AssertionError("run_inference must not run for an all-ocean cell")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    summary = run_fill(
        tmp_path,
        store,
        year=2024,
        land_mask_path=mask,
        mosaic_base=str(tmp_path / "does_not_exist"),
        # missing mosaic must not be read
        staging_base=str(tmp_path / "staging"),
        config=zone_config(2024),
        run_id="runNM",
    )
    assert summary["empty"] is True
    assert summary["live_tiles"] == 0
    repo = global_store.open_global_repo(store)
    node = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[ZONE]
    assert node.attrs["years_complete"] == [2024]
    assert node.attrs["runs"]["2024"] == {**node.attrs["runs"]["2024"], "run_id": "runNM", "empty": True}
    assert repo.lookup_tag("zone-01N-2024") == summary["snapshot_id"]


def test_zone_year_complete_reflects_years_complete(tmp_path):
    """The preflight helper reports False before a fill and for an unseeded zone; the True
    side is asserted after a landed fill by test_phase_split_matches_composed_fill.
    """
    store = seed_global(tmp_path)
    assert zone_fill.zone_year_complete(store, ZONE, 2025) is False
    assert zone_fill.zone_year_complete(store, "60N", 2025) is False  # not seeded in this store


def test_zone_year_on_axis(tmp_path):
    """The preflight reports True for a seeded year, False off-axis / unseeded —
    so the flow can decline Ray for a year the runner would reject anyway.
    """
    store = seed_global(tmp_path)
    assert zone_fill.zone_year_on_axis(store, ZONE, 2025) is True
    assert zone_fill.zone_year_on_axis(store, ZONE, 2024) is True
    assert zone_fill.zone_year_on_axis(store, ZONE, 2026) is False  # off the seeded axis
    assert zone_fill.zone_year_on_axis(store, "60N", 2025) is False  # not seeded in this store


def test_off_axis_year_fails_before_inference(tmp_path, monkeypatch):
    """An off-axis year dies before any inference is dispatched."""
    store = seed_global(tmp_path)

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
        run_fill(tmp_path, store, year=2015, config=InferenceConfig(time_window=window, chunk_size=TILE, num_gpus=0))


def _rolling_window() -> TimeWindow:
    """A 12-month window that merely OVERLAPS the target year."""
    return TimeWindow(
        window_start=(2024, 7),
        window_end=(2025, 6),
        months=tuple([(2024, m) for m in range(7, 13)] + [(2025, m) for m in range(1, 7)]),
        window_end_label="2025-06-01",
    )


@pytest.mark.parametrize(
    "year,window,match",
    [
        # A window for a DIFFERENT year than the target slot — an operator error, e.g. a
        # cloned invocation whose year was edited but not its time_window.
        pytest.param(2024, calendar_window(2025), "guarantees calendar-year slots", id="wrong-year"),
        # A rolling window overlapping the target year. Rejected even though it is 12
        # months: the slot's seeded time_bnds advertise Jan-Dec, so a rolling window's
        # label would be inaccurate (CF containment). The message points non-calendar
        # consumers at the single-ROI `12mo_window_end` path.
        pytest.param(2025, _rolling_window(), "12mo_window_end", id="rolling"),
    ],
)
def test_non_calendar_year_windows_are_rejected(tmp_path, year, window, match):
    """The slot must hold exactly the Jan-Dec year its seeded time_bnds advertise.

    Both rejections happen before any mask or mosaic is opened, which is why
    this can run against paths that do not exist.
    """
    store = seed_global(tmp_path)
    with pytest.raises(ValueError, match=match):
        run_fill(tmp_path, store, year=year, config=InferenceConfig(time_window=window, chunk_size=TILE, num_gpus=0))


def test_landed_but_untagged_cell_is_retagged_without_rerun(tmp_path, monkeypatch):
    """A crash between the fill commit and the tag: retry tags the tip, no re-inference.

    The crash state is constructed by running the fill WITHOUT the tag step
    (assemble_global directly) — icechunk forbids recreating a deleted tag, so
    deleting a landed tag would not reproduce the crash window.
    """
    store = seed_global(tmp_path)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0)])
    # Stage one tile and land the fill commit (years_complete advances) with no tag.
    rng = np.random.default_rng(5)
    writer = ZarrWriter(str(tmp_path / "staging"), embedding_dim=BAND)
    chunk_00 = zone_fill.enumerate_chunks(NY, NX, TILE)[0]
    writer.write_chunk(
        chunk_00,
        rng.integers(-100, 100, size=(TILE, TILE, BAND)).astype(np.int8),
        "run1",
        scales=rng.random((TILE, TILE)).astype(np.float32),
        stack=STACK,
    )
    writer.assemble_global(store, ZONE, year=2025, run_id="run1", n_workers=1)
    repo = global_store.open_global_repo(store)
    assert "zone-01N-2025" not in repo.list_tags()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("a landed cell must not re-run inference just to re-tag")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    retry = run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=mosaic_base, run_id="run2")
    assert retry["already_complete"] is True
    assert repo.lookup_tag("zone-01N-2025") == retry["snapshot_id"]


# ===========================================================================
# Phase split: infer_zone_year / assemble_zone_year (the sequential runner's API)
# ===========================================================================


def test_phase_split_matches_composed_fill(tmp_path, monkeypatch, staged):
    """Infer → handoff → assemble lands the same store state and summary shape
    as the composed fill_zone_year (which is their back-to-back composition).
    """
    store = seed_global(tmp_path)
    make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0), (1, 1)])

    handoff = run_infer(tmp_path, store, land_mask_path=mask, run_id="runP")
    assert handoff.done is None
    assert len(handoff.live) == 2 and len(handoff.results) == 2

    summary = zone_fill.assemble_zone_year(
        handoff,
        store_path=store,
        staging_base=str(tmp_path / "staging"),
        log=log,
    )
    assert summary["empty"] is False and summary["succeeded"] == 2
    assert summary["tag"] == zone_fill.zone_year_tag(ZONE, 2025)
    # The year is recorded complete — the composed path's end state.
    assert zone_fill.zone_year_complete(store, ZONE, 2025)


def test_staging_cleanup_is_handed_to_the_caller_rather_than_run_inline(tmp_path, monkeypatch, staged):
    """THE REGRESSION. The staging delete used to run on the trailing-assembly thread.

    Measured in production on 2026-08-31 at roughly TWO HOURS per cell — 34N-2018 published at
    02:27Z and the next step on that thread did not begin until 04:27Z — during which the
    cluster's next assembly could not start even with its tiles fully staged. Seven of nine
    clusters were idle 91-117 minutes for this reason.

    The oracle is that `cleanup_staging` is NOT called while `assemble_zone_year` runs, and IS
    reachable afterwards through what the caller was handed. Asserting only that a deferral
    callable fired would pass even if the work never happened.
    """
    store = seed_global(tmp_path)
    make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0), (1, 1)])
    deleted: list[str] = []
    monkeypatch.setattr(
        ZarrWriter, "cleanup_staging", lambda self, run_id, log=None: deleted.append(run_id), raising=True
    )
    handoff = run_infer(tmp_path, store, land_mask_path=mask, run_id="runDefer")
    handed: list = []
    zone_fill.assemble_zone_year(
        handoff,
        store_path=store,
        staging_base=str(tmp_path / "staging"),
        log=log,
        defer_cleanup=handed.append,
    )
    assert deleted == [], f"the staging delete ran inline despite a deferral: {deleted}"
    assert len(handed) == 1, "nothing was handed to the caller, so the delete would never happen"
    handed[0]()
    assert deleted == ["runDefer"], "what was handed over does not delete this run's staging"


def _seed_global_with_rule(tmp_path, rule: int | None) -> str:
    """A store whose root declares a depth rule (or none), so the fill's gate accepts that one."""
    store = str(tmp_path / "global.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, [SPEC], years=YEARS, layout=SMALL, optical_min_obs=rule)
    return store


def _published_part(registry_root: str) -> dict:
    """The one part under ``registry_root``, read back as a column -> values dict."""
    ds = pytest.importorskip("pyarrow.dataset")
    table = ds.dataset(f"{registry_root}/parts", partitioning="hive").to_table()
    return table.to_pydict()


def _master_tiles(registry_root: str) -> list[str]:
    """The tiles in the cell's ``master/`` file — present only if the fill brought it up to date."""
    pq = pytest.importorskip("pyarrow.parquet")
    return pq.read_table(f"{registry_root}/master/zone={ZONE}/year=2025/part-0.parquet").column("tile").to_pylist()


def test_the_registry_records_the_depth_rule_the_cell_was_filled_under(tmp_path, monkeypatch, staged):
    """The runner's config must reach the published row, through three forwarding hops.

    Worth a test rather than a reading, because the failure is INVISIBLE: every hop defaults to
    ``None``, and ``None`` in the column means "this fill applied no depth rule" — a legitimate,
    plausible value. A dropped argument therefore publishes a registry that looks fine and silently
    unreadable, since `obs_max` and `median_obs_where_any` are distances from a line the rows would
    no longer name.
    """
    store = _seed_global_with_rule(tmp_path, 15)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0), (1, 1)])
    registry_root = str(tmp_path / "registry")

    summary = run_fill(
        tmp_path,
        store,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        registry_root=registry_root,
        config=InferenceConfig(time_window=calendar_window(2025), chunk_size=TILE, num_gpus=0, optical_min_obs=15),
        run_id="ruleRun",
    )
    assert summary["empty"] is False

    cols = _published_part(registry_root)
    assert cols["tile"], "a part was published"
    assert set(cols["optical_min_obs"]) == {15}, "every row states the store's rule"
    assert set(cols["embedded"]) == {True}
    assert _master_tiles(registry_root) == sorted(cols["tile"]), "and the cell's master is current"


def _fill_one_tile(tmp_path, registry_root: str):
    """A one-tile cell filled under a depth rule, with its registry beside it."""
    return run_fill(
        tmp_path,
        _seed_global_with_rule(tmp_path, 15),
        land_mask_path=make_mask(tmp_path, [(0, 0)]),
        mosaic_base=make_mosaic(tmp_path),
        registry_root=registry_root,
        config=InferenceConfig(time_window=calendar_window(2025), chunk_size=TILE, num_gpus=0, optical_min_obs=15),
    )


def test_a_failed_master_update_does_not_fail_a_landed_cell(tmp_path, monkeypatch, caplog):
    """The cell has committed by then; the master is an index beside it, and repairable."""
    monkeypatch.setattr(zone_fill, "run_inference", staging_inference_stub({}))

    def unreachable(*_args: object, **_kwargs: object) -> list[str]:
        raise OSError("S3 500")

    monkeypatch.setattr(zone_fill, "merge_cell", unreachable)
    with caplog.at_level(logging.ERROR):
        summary = _fill_one_tile(tmp_path, str(tmp_path / "registry"))
    assert summary["tag"], "the cell landed and was tagged"
    assert "registry master NOT merged" in caplog.text


def _all_skip_inference(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
    """Every tile refused, with nothing staged: the store holds nothing for them before or after."""
    writer = ZarrWriter(staging_base, embedding_dim=BAND)
    for chunk in chunks:
        writer.write_skip_marker(chunk, run_id, {"label": chunk.label, "refused": {"no_optical": 4}})
    return [{"chunk": c.label, "status": "skipped", "valid_pixels": 0, "elapsed_sec": 0.0} for c in chunks]


@pytest.mark.parametrize(
    ("inference", "part_survives"),
    [(staging_inference_stub({}), True), (_all_skip_inference, False)],
    ids=["its-part-published", "an-all-refused-run-whose-part-never-published"],
)
def test_a_retry_merges_the_master_a_fill_died_before_reaching(tmp_path, monkeypatch, caplog, inference, part_survives):
    """The retry finds the cell complete and skips the fill, so planning merges the master or nothing does.

    An all-refused run over a tile the store held nothing for still shows in its commit's diff, so
    its retry refuses while its part is missing.
    """
    monkeypatch.setattr(zone_fill, "run_inference", inference)
    store, registry_root = _seed_global_with_rule(tmp_path, 15), str(tmp_path / "registry")
    cell = {
        "land_mask_path": make_mask(tmp_path, [(0, 0)]),
        "mosaic_base": make_mosaic(tmp_path),
        "registry_root": registry_root,
        "config": InferenceConfig(time_window=calendar_window(2025), chunk_size=TILE, num_gpus=0, optical_min_obs=15),
    }
    merge = zone_fill._merge_registry_master

    def died(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(zone_fill, "_merge_registry_master", died)
    with pytest.raises(KeyboardInterrupt):
        run_fill(tmp_path, store, **cell)
    if not part_survives:
        for part in (tmp_path / "registry" / "parts").rglob("*.parquet"):
            part.unlink()
    monkeypatch.setattr(zone_fill, "_merge_registry_master", merge)
    with caplog.at_level(logging.ERROR):
        assert run_fill(tmp_path, store, **cell)["already_complete"] is True
    if part_survives:
        assert _master_tiles(registry_root) == ["chunk_0_0"]
    else:
        assert "last wrote 1 tile(s) no part describes" in caplog.text


def test_the_master_learns_of_a_refill_that_fails_at_the_tag_step(tmp_path, monkeypatch):
    """A deliberate refill fails at tagging by design — zone-year tags are write-once — after its data
    and part have landed. Updating the master after the tag would never reach it.
    """
    monkeypatch.setattr(zone_fill, "run_inference", staging_inference_stub({}))

    def spent(*_args: object, **_kwargs: object) -> str:
        raise ValueError("tag zone-01N-2025 already exists")

    monkeypatch.setattr(zone_fill, "tag_zone_year", spent)
    registry_root = str(tmp_path / "registry")
    with pytest.raises(ValueError, match="already exists"):
        _fill_one_tile(tmp_path, registry_root)
    assert _master_tiles(registry_root) == ["chunk_0_0"]


def test_an_all_refused_cell_still_records_the_depth_rule(tmp_path, monkeypatch):
    """The all-skipped branch is a SEPARATE forwarding site, and it is the cell with the most to say.

    It also cannot recover the value any other way: no staged tile is opened, so the rule has to
    arrive as an argument or not at all.
    """
    store = _seed_global_with_rule(tmp_path, 15)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0), (1, 1)])
    registry_root = str(tmp_path / "registry")

    def all_skip_inference(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        writer = ZarrWriter(staging_base, embedding_dim=BAND)
        results = []
        for chunk in chunks:
            writer.write_skip_marker(chunk, run_id, {"label": chunk.label, "refused": {"thin": 4}})
            results.append({"chunk": chunk.label, "status": "skipped", "valid_pixels": 0, "elapsed_sec": 0.0})
        return results

    monkeypatch.setattr(zone_fill, "run_inference", all_skip_inference)
    summary = run_fill(
        tmp_path,
        store,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        registry_root=registry_root,
        config=InferenceConfig(time_window=calendar_window(2025), chunk_size=TILE, num_gpus=0, optical_min_obs=15),
        run_id="ruleRunEmpty",
    )
    assert summary["empty"] is True

    cols = _published_part(registry_root)
    assert set(cols["embedded"]) == {False}, "every tile refused"
    assert set(cols["optical_min_obs"]) == {15}
    assert set(cols["refused_thin_px"]) == {4}, "and the reason survived alongside the rule"
    assert _master_tiles(registry_root) == sorted(cols["tile"]), "this branch updates the master too"


def test_a_partly_refused_tile_publishes_what_the_gate_removed(tmp_path, monkeypatch):
    """The actor's success-path coverage record must reach the published row.

    Worth an end-to-end test because the failure is silent and reads as good news: without the
    forwarding, a tile the depth gate partly refused publishes as embedded with every measurement
    null, which a consumer reads as fully covered ground. The number was always computed — the actor
    accumulates refusal reasons on the success path too — so nothing looks wrong anywhere.
    """
    store = _seed_global_with_rule(tmp_path, 15)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0), (1, 1)])
    registry_root = str(tmp_path / "registry")

    def partly_refusing_stub(num_actors, config, chunks, mosaic_base, staging_base, run_id, t0, log, **kwargs):
        rng = np.random.default_rng(5)
        writer = ZarrWriter(staging_base, embedding_dim=BAND)
        results = []
        for i, chunk in enumerate(chunks):
            emb = rng.integers(-100, 100, size=(chunk.height, chunk.width, BAND)).astype(np.int8)
            writer.write_chunk(
                chunk,
                emb,
                run_id,
                scales=rng.random((chunk.height, chunk.width)).astype(np.float32),
                stack=STACK,
            )
            results.append(
                {
                    "chunk": chunk.label,
                    "status": "success",
                    "valid_pixels": 1,
                    "elapsed_sec": 0.0,
                    "s1_free_pixels": 0,
                    "s1_thin_pixels": 0,
                    "s2_thin_pixels": 0,
                    "s2_thin_below_obs": 15,
                    "coverage": {
                        "refused": {"thin": 7 * (i + 1), "no_optical": 0, "no_radar": 0},
                        "eligible_px": 100,
                        "chunk_px": 100,
                        "s2_obs": {"px_with_any": 95, "max": 14, "median_where_any": 11.0},
                        "px_with_any_radar": 100,
                        "radar_rule_enforced": False,
                    },
                }
            )
        return results

    monkeypatch.setattr(zone_fill, "run_inference", partly_refusing_stub)
    summary = run_fill(
        tmp_path,
        store,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        registry_root=registry_root,
        config=InferenceConfig(time_window=calendar_window(2025), chunk_size=TILE, num_gpus=0, optical_min_obs=15),
        run_id="partialRun",
    )
    assert summary["empty"] is False

    cols = _published_part(registry_root)
    rows = {t: i for i, t in enumerate(cols["tile"])}
    assert set(cols["embedded"]) == {True}, "both tiles hold embeddings"
    # ...and both report the pixels the depth gate took out of them, per tile, not pooled.
    refused = {t: cols["refused_px"][i] for t, i in rows.items()}
    assert sorted(refused.values()) == [7, 14], "each tile's own count, not a shared one"
    assert all(cols["obs_max"][i] == 14 for i in rows.values())
    assert all(cols["median_obs_where_any"][i] == 11.0 for i in rows.values())
    assert all(cols["optical_min_obs"][i] == 15 for i in rows.values()), "and the line they missed"


def test_a_resumed_success_publishes_null_rather_than_zero(tmp_path, monkeypatch, staged):
    """A resumed tile is a synthetic success carrying no coverage record. It must publish as
    unmeasured, not as a tile the gate refused nothing from — the second is a claim nobody made.
    """
    store = _seed_global_with_rule(tmp_path, 15)
    mosaic_base = make_mosaic(tmp_path)
    mask = make_mask(tmp_path, [(0, 0)])
    registry_root = str(tmp_path / "registry")

    run_fill(
        tmp_path,
        store,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        registry_root=registry_root,
        config=InferenceConfig(time_window=calendar_window(2025), chunk_size=TILE, num_gpus=0, optical_min_obs=15),
        run_id="resumedRun",
    )
    cols = _published_part(registry_root)
    assert cols["embedded"] == [True]
    assert cols["refused_px"] == [None], "no record reached it, so nothing is claimed"
    assert cols["optical_min_obs"] == [15], "the cell's policy is known regardless"


def test_terminal_handoff_passes_through_assembly(tmp_path):
    done = {"zone": ZONE, "year": 2025, "already_complete": True}
    handoff = zone_fill.ZoneFillHandoff(
        zone=ZONE, year=2025, run_id="r", t0=0.0, summary={}, live=[], results=[], done=done
    )
    assert zone_fill.assemble_zone_year(handoff, store_path="unused", staging_base="unused", log=log) is done


def test_zone_live_tile_count(tmp_path):
    mask = make_mask(tmp_path, [(0, 0), (0, 1), (1, 0)])
    assert zone_fill.zone_live_tile_count(mask, ZONE) == 3
    assert zone_fill.zone_has_live_tiles(mask, ZONE) is True
    empty_mask = make_mask(tmp_path / "m2", [])
    assert zone_fill.zone_live_tile_count(empty_mask, ZONE) == 0
    assert zone_fill.zone_has_live_tiles(empty_mask, ZONE) is False


def test_fill_accepts_a_cropped_written_mosaic(tmp_path, monkeypatch, staged):
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

    store = seed_global(tmp_path)
    mask = make_mask(tmp_path, [(0, 0)])

    # The production zone geobox, exactly as export_zone_roi encodes it:
    # north-up, top-left origin at (easting_min, northing_max).
    geobox = GeoBox((NY, NX), Affine(10.0, 0.0, SPEC.easting[0], 0.0, -10.0, SPEC.northing[1]), SPEC.crs)

    class _Roi:
        pass

    roi = _Roi()
    roi.geobox, roi.height, roi.width = geobox, NY, NX

    day_ds = xr.Dataset(
        {
            "red": (
                ("time", "northing", "easting"),
                dask_array.full((1, NY, NX), 3, dtype=np.uint16, chunks=(1, TILE, TILE)),
            )
        },
        coords={"time": np.array(["2025-06-01"], dtype="datetime64[ns]")},
    )
    base = str(tmp_path / "mosaics")
    for name in ("reflectance", "sar_ascending", "sar_descending"):
        write_day_windows(
            f"{base}/{name}.zarr",
            day_ds,
            [(0, TILE, 0, TILE)],  # one live window; the rest of the grid is never written
            roi=roi,
            manifest=None,
            baselines={},
            tile_id="roi.zarr",
            crs=SPEC.crs,
            chunks={"time": 1, "northing": TILE, "easting": TILE},
        )

    summary = run_fill(tmp_path, store, land_mask_path=mask, mosaic_base=base, run_id="runCropped")
    assert summary["empty"] is False and summary["succeeded"] == 1


def test_calendar_gate_is_callable_before_provisioning():
    """The gate decides from config alone, so both flows call it pre-cluster.

    An offset or same-year-partial window is rejected by planning regardless —
    but only after a billable GPU fleet is up (and, on the chained path, after
    look-ahead ingests have started). One definition, two call sites.
    """
    zone_fill.assert_calendar_year_window(calendar_window(2025), 2025)  # exact Jan-Dec: passes

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
    store = seed_global(tmp_path)
    mosaic_base = str(tmp_path / "scrambled-mosaics")
    scrambled = easting_coords(SPEC).copy()
    # Swap two interior samples: same endpoints, same length, same CRS, wrong order.
    # Note this ALSO defeats a per-step spacing test that allows half a pixel of slack —
    # the two bad steps are +/- one pixel, and a wander-and-return pattern would defeat it
    # entirely. Comparing whole vectors against the seeded axis has no such gap.
    scrambled[1], scrambled[2] = scrambled[2], scrambled[1]
    make_mosaic_store(f"{mosaic_base}/reflectance.zarr", northing_coords(SPEC), scrambled)
    for orbit in ("ascending", "descending"):
        make_mosaic_store(f"{mosaic_base}/sar_{orbit}.zarr", northing_coords(SPEC), easting_coords(SPEC))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("inference must not run on a non-affine mosaic grid")

    monkeypatch.setattr(zone_fill, "run_inference", fail_if_called)
    with pytest.raises(ValueError, match=r"does not match zone .* seeded axis"):
        run_fill(tmp_path, store, land_mask_path=make_mask(tmp_path, [(0, 0)]), mosaic_base=mosaic_base)


def _recording_publish(calls: list[dict], store: str, *, fail: bool = False):
    """A stand-in for ``publish_provenance`` recording what it was given and whether the cell was tagged."""

    def publish(**kwargs):
        calls.append({**kwargs, "tagged": "zone-01N-2025" in global_store.open_global_repo(store).list_tags()})
        if fail:
            raise zone_fill.ProvenanceCheckError("a plane disagrees with its row")
        return {}

    return publish


def test_provenance_is_published_before_the_tag_from_the_runs_shards(tmp_path, monkeypatch):
    store = seed_global(tmp_path)
    monkeypatch.setattr(zone_fill, "run_inference", staging_inference_stub({}))
    calls: list[dict] = []
    monkeypatch.setattr(zone_fill, "publish_provenance", _recording_publish(calls, store))
    summary = run_fill(
        tmp_path,
        store,
        land_mask_path=make_mask(tmp_path, [(0, 0), (1, 1)]),
        mosaic_base=make_mosaic(tmp_path),
        registry_root=str(tmp_path / "registry"),
    )
    assert summary["tag"]
    [call] = calls
    assert call["tagged"] is False, "published before the tag"
    assert call["embedded"] == ["chunk_0_0", "chunk_1_1"] and call["refused"] == []
    assert sorted(path.rsplit("/", 1)[-1] for path in call["mosaic_snapshots"]) == [
        "reflectance.zarr",
        "sar_ascending.zarr",
        "sar_descending.zarr",
    ]


def test_a_failed_publication_leaves_the_cell_untagged_and_its_retry_republishes(tmp_path, monkeypatch):
    """The cell has landed, so the retry takes the landed branch; that branch must publish before it tags."""
    store = seed_global(tmp_path)
    mask, mosaic_base = make_mask(tmp_path, [(0, 0)]), make_mosaic(tmp_path)
    monkeypatch.setattr(zone_fill, "run_inference", staging_inference_stub({}))
    calls: list[dict] = []
    monkeypatch.setattr(zone_fill, "publish_provenance", _recording_publish(calls, store, fail=True))
    with pytest.raises(zone_fill.ProvenanceCheckError):
        run_fill(
            tmp_path,
            store,
            land_mask_path=mask,
            mosaic_base=mosaic_base,
            registry_root=str(tmp_path / "registry"),
            run_id="r1",
        )
    assert len(calls) == 1, "a failed check is not retried"
    assert "zone-01N-2025" not in global_store.open_global_repo(store).list_tags()

    monkeypatch.setattr(zone_fill, "publish_provenance", _recording_publish(calls, store))
    retry = run_fill(
        tmp_path,
        store,
        land_mask_path=mask,
        mosaic_base=mosaic_base,
        registry_root=str(tmp_path / "registry"),
        run_id="r1",
    )
    assert retry["already_complete"] is True
    assert calls[-1]["tagged"] is False and calls[-1]["embedded"] == ["chunk_0_0"]
    assert "zone-01N-2025" in global_store.open_global_repo(store).list_tags()
