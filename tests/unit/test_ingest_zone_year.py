"""ingest_zone_year flow: ocean-skip, marker-skip, dispatch, verify-gate.

The flow body is orchestration (arun_deployment); tests call it via ``.fn`` with
every external touchpoint mocked, so no Prefect engine, Dask cluster, or S3 runs.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import numpy as np
import pytest
from prefect.states import StateType

import tessera_embeddings.orchestration.prefect.flows.ingest_zone_year as mod
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.errors import InsufficientCoverageError

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


def _completed_run(rid: str = "r") -> SimpleNamespace:
    return SimpleNamespace(id=rid, state=SimpleNamespace(type=StateType.COMPLETED, name="Completed"))


@pytest.fixture()
def wired(monkeypatch):
    """Mock every external touchpoint; return a record of what the flow did."""
    rec: dict = {"arun": [], "markers_written": [], "roi_exported": [], "coverage_checked": []}

    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-ingest-zone-year"))
    # Credentials: the flow lazily imports this symbol; patch it on the source module.
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", object(), raising=False
    )

    async def fake_arun(dep, parameters=None):
        rec["arun"].append((dep, parameters))
        return _completed_run()

    monkeypatch.setattr(mod, "arun_deployment", fake_arun)
    monkeypatch.setattr(mod, "export_zone_roi", lambda z, **kw: rec["roi_exported"].append((z, kw)))
    monkeypatch.setattr(mod, "check_time_window_coverage", lambda *a, **k: rec["coverage_checked"].append((a, k)))
    monkeypatch.setattr(mod, "_write_ingest_marker", lambda store, fp, **kw: rec["markers_written"].append(store))
    monkeypatch.setattr(mod, "delete_prefix", lambda uri, **kw: rec.setdefault("deletes", []).append(uri))
    # Fingerprint inputs: coverage sha + orbit resolution ("ascending" resolves to
    # itself without probing, so _resolved_stores is deterministic in tests).
    monkeypatch.setattr(mod, "_coverage_sha", lambda *a, **k: "cov-sha-1")
    monkeypatch.setattr(mod, "resolve_s1_orbit", lambda mosaic_base, orbit, **k: orbit)
    return rec


def _run(**kwargs):
    return asyncio.run(mod.ingest_zone_year.fn(zone="33N", year=2025, paths=_PATHS, **kwargs))


def test_ocean_zone_skips_everything(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: False)
    result = _run()
    assert result["status"] == "skipped_ocean"
    assert wired["arun"] == [] and wired["roi_exported"] == [] and wired["markers_written"] == []


def test_matching_markers_skip_ingest(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    fp = {
        "window": ["2025-01-01", "2025-12-31"],
        "min_valid_coverage": 0.1,
        "s1_orbit": "ascending",
        "allow_partial_window": False,
        "coverage_sha256": "cov-sha-1",
    }
    # Every candidate store exists and carries the exact fingerprint.
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (True, fp))
    result = _run(s1_orbit="ascending")
    assert result["status"] == "already_ingested"
    assert result["fingerprint"] == fp
    assert wired["arun"] == [] and wired["roi_exported"] == [] and wired.get("deletes", []) == []


def test_dispatches_and_marks_on_success(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))  # nothing ingested yet
    result = _run(s1_orbit="ascending", ingest_settings=mod.IngestSettings(min_valid_coverage=0.2))

    assert result["status"] == "ingested"
    # ROI synthesized once, onto the campaign's ROI path for this zone.
    assert wired["roi_exported"] and wired["roi_exported"][0][0] == "33N"
    # One S1 (ascending) + one S2 dispatch, both onto the per-year mosaic base.
    assert len(wired["arun"]) == 2
    params = [p for _, p in wired["arun"]]
    assert all(p["store_path"] == "s3://in/mosaics/33N/2025" for p in params)
    assert all(p["roi_zarr_path"] == "s3://in/rois/zarrs/zone_33N.zarr" for p in params)
    assert all(p["start_date"] == "2025-01-01" and p["end_date"] == "2025-12-31" for p in params)
    s2_params = next(p for p in params if "min_valid_coverage" in p)
    assert s2_params["min_valid_coverage"] == 0.2
    s1_params = next(p for p in params if "orbit" in p)
    assert s1_params["orbit"] == "ascending"
    # Coverage verified, then a marker on each required store (reflectance + sar_asc).
    assert wired["coverage_checked"]
    assert wired["markers_written"] == [
        "s3://in/mosaics/33N/2025/reflectance.zarr",
        "s3://in/mosaics/33N/2025/sar_ascending.zarr",
    ]


def test_perf_report_uri_scoped_by_cell_then_child(wired, monkeypatch):
    """The base URI is scoped by (zone, year) FIRST, then per child.

    run-global-campaign hands the same IngestSettings to every cell, so a
    base-only path would have concurrent cells racing on one s2.html and later
    cells silently overwriting earlier ones.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
    _run(s1_orbit="ascending", ingest_settings=mod.IngestSettings(perf_report_uri="s3://in/perf/"))

    params = [p for _, p in wired["arun"]]
    s2 = next(p for p in params if "min_valid_coverage" in p)
    s1 = next(p for p in params if "orbit" in p)
    # zone 33N / year 2025 (see _run's defaults) separates the cells.
    assert s2["perf_report_uri"] == "s3://in/perf/33N-2025/s2.html"
    assert s1["perf_report_uri"] == "s3://in/perf/33N-2025/s1-ascending.html"


def test_perf_report_uri_none_by_default(wired, monkeypatch):
    """With no perf base set, children receive perf_report_uri=None (off)."""
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
    _run(s1_orbit="ascending")

    params = [p for _, p in wired["arun"]]
    assert all(p["perf_report_uri"] is None for p in params)


def test_s3_region_threaded_through_metadata_opens(wired, monkeypatch):
    """The flow's s3_region reaches its Icechunk metadata opens — the mask liveness
    probe, ROI synthesis, and the coverage gate — so a non-default-region deployment
    reads the same stores the fill will (the campaign now forwards this region).
    """
    live_kw: dict = {}
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: live_kw.update(k) or True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
    _run(s1_orbit="ascending", s3_region="eu-west-1")
    assert live_kw.get("s3_region") == "eu-west-1"  # mask liveness probe
    assert wired["roi_exported"][0][1].get("s3_region") == "eu-west-1"  # ROI synthesis
    assert wired["coverage_checked"][0][1].get("s3_region") == "eu-west-1"  # coverage gate


def test_stale_marker_triggers_reingest(wired, monkeypatch):
    """A present marker with a different fingerprint (e.g. rebuilt coverage) does
    NOT short-circuit — the mosaic is re-ingested under the new inputs.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    stale = {
        "window": ["2025-01-01", "2025-12-31"],
        "min_valid_coverage": 0.1,
        "s1_orbit": "ascending",
        "coverage_sha256": "OLD-sha",  # != current "cov-sha-1"
    }
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (True, stale))
    result = _run(s1_orbit="ascending")
    assert result["status"] == "ingested"  # re-ingested despite a present (stale) marker
    assert wired["arun"]
    # Stale stores cleared before the rebuild so re-ingest doesn't append onto old
    # data — per store, since a sibling carrying the current fingerprint is fine.
    assert sorted(wired.get("deletes", [])) == [
        "s3://in/mosaics/33N/2025/reflectance.zarr",
        "s3://in/mosaics/33N/2025/sar_ascending.zarr",
        "s3://in/mosaics/33N/2025/sar_descending.zarr",
    ]


def test_markerless_partial_mosaic_is_cleared(wired, monkeypatch):
    """A prior attempt that wrote reflectance then crashed before any SAR store or
    marker landed must be cleared, not appended onto: the resolved-orbit probe
    can't see it (no SAR -> resolve raises), so existence over the maximal
    candidate set is what triggers the clean rebuild.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    # Reflectance exists (markerless half-write); the SAR stores never landed.
    monkeypatch.setattr(
        mod, "_probe_marker", lambda store, **kw: (True, None) if store.endswith("reflectance.zarr") else (False, None)
    )
    # No SAR store yet -> the FIRST orbit resolution (the probe) raises; the
    # SECOND (post-ingest verification) succeeds once the ingesters have written.
    calls = {"n": 0}

    def flaky_resolve(mosaic_base, orbit, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise InsufficientCoverageError("no SAR store yet")
        return orbit

    monkeypatch.setattr(mod, "resolve_s1_orbit", flaky_resolve)
    result = _run(s1_orbit="ascending")
    assert result["status"] == "ingested"
    # Only the markerless partial is cleared; the SAR stores never landed, so
    # there is nothing there to delete.
    assert wired.get("deletes") == ["s3://in/mosaics/33N/2025/reflectance.zarr"]


def test_partial_window_marker_does_not_satisfy_strict_run(wired, monkeypatch):
    """A mosaic accepted under allow_partial_window=True must NOT short-circuit a
    later strict run — the policy is in the fingerprint, so the strict run clears
    and re-ingests (re-running the strict coverage gate) instead of reusing a
    partial mosaic whose fill would fail strict preflight forever.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    partial = {
        "window": ["2025-01-01", "2025-12-31"],
        "min_valid_coverage": 0.1,
        "s1_orbit": "ascending",
        "allow_partial_window": True,  # accepted under the relaxed policy
        "coverage_sha256": "cov-sha-1",
    }
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (True, partial))
    # Default run is strict (allow_partial_window=False): fingerprint differs → re-ingest.
    result = _run(s1_orbit="ascending")
    assert result["status"] == "ingested"
    assert sorted(wired.get("deletes", [])) == [  # policy-mismatch stores cleared
        "s3://in/mosaics/33N/2025/reflectance.zarr",
        "s3://in/mosaics/33N/2025/sar_ascending.zarr",
        "s3://in/mosaics/33N/2025/sar_descending.zarr",
    ]


def test_coverage_failure_leaves_no_marker(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))

    def _raise(*a, **k):
        raise InsufficientCoverageError("reflectance store starts at 2025-03, but window requires 2025-01")

    monkeypatch.setattr(mod, "check_time_window_coverage", _raise)
    with pytest.raises(InsufficientCoverageError):
        _run(s1_orbit="ascending")
    assert wired["markers_written"] == []  # not marked complete on a coverage gap


class TestChunkScaledWorkers:
    """Cropped cells size their Dask fleet from live chunks, not zone extent.

    The 03S incident: a 4-tile zone hit the 50-worker ceiling because the fleet
    was sized for a full-extent mosaic. With writes cropped to live windows the
    chunk count IS the work measure, so max_workers scales with it — floor keeps
    a tiny cell from starving, and the settings ceiling stays the quota cap.
    """

    def _dispatch(self, wired, monkeypatch, *, tile_live, **settings_kwargs):

        monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
        monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))

        def fake_coverage(*a, **k):
            return {"tile_live_2048": np.asarray(tile_live, dtype=bool)}

        # Two patch targets, deliberately: the flow reads the coverage sha through
        # its own namespace, while the chunk count is land_mask's function reading
        # through land_mask's. Patching only one leaves the other reaching for S3.
        monkeypatch.setattr(mod, "open_store_as_zarr_group", fake_coverage)
        monkeypatch.setattr("tessera_embeddings.ingest.land_mask.open_store_as_zarr_group", fake_coverage)
        wired["arun"].clear()  # the fixture accumulates; a test may dispatch more than once
        _run(ingest_settings=mod.IngestSettings(crop_to_live_windows=True, **settings_kwargs), s1_orbit="ascending")
        # Dispatch order is S1 orbits then S2, so the LAST entry is always S2's width.
        return [p["max_workers"] for _, p in wired["arun"]]

    def test_sparse_zone_gets_the_floor_not_the_ceiling(self, wired, monkeypatch):
        # 03S in miniature: 4 live tiles in one 4096-chunk -> 1 chunk -> floor(10).
        s1, s2 = self._dispatch(wired, monkeypatch, tile_live=[[True, True], [True, True]])
        assert s2 == 10
        # S1 takes its fraction of that, which on a tiny zone is a couple of workers. A
        # fixed S1 width would have exceeded S2's own fleet here.
        assert s1 == 2

    def test_dense_zone_is_capped_by_settings(self, wired, monkeypatch):

        tiles = np.ones((40, 40), dtype=bool)  # 400 live chunks -> 200 > cap
        s1, s2 = self._dispatch(wired, monkeypatch, tile_live=tiles, max_workers=50)
        assert s2 == 50
        assert s1 == 11  # round(50 * 0.22)

    def test_mid_zone_scales_half_worker_per_chunk(self, wired, monkeypatch):

        tiles = np.zeros((20, 20), dtype=bool)
        tiles[::2, ::2] = True  # every 2x2 tile block live -> all 100 chunks live
        s1, s2 = self._dispatch(wired, monkeypatch, tile_live=tiles, max_workers=200)
        assert s2 == 50
        assert s1 == 11

    def test_s1_is_never_wider_than_s2_nor_below_min_workers(self, wired, monkeypatch):
        """The two clamps that keep a narrow S2 fleet from being out-sized by its S1 pair.

        A cell's duration is set by S2, so an S1 orbit wider than S2 buys nothing and holds
        quota that limits how many cells run at once.
        """
        tiles = [[True, True], [True, True]]  # smallest zone: S2 lands on the floor
        s1, s2 = self._dispatch(wired, monkeypatch, tile_live=tiles, s1_worker_fraction=1.0)
        assert s1 == s2, "fraction 1.0 must give parity, never more"
        s1, s2 = self._dispatch(wired, monkeypatch, tile_live=tiles, min_workers=6, s1_worker_fraction=0.01)
        assert s1 == 6, "a tiny fraction must still respect min_workers"

    def test_crop_off_keeps_the_settings_value(self, wired, monkeypatch):
        monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
        monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
        _run(ingest_settings=mod.IngestSettings(max_workers=37), s1_orbit="ascending")
        # S2 takes the settings value verbatim; S1 still takes its fraction of it, because
        # the ratio is about the WORK split between sensors, not about cropping.
        assert [p["max_workers"] for _, p in wired["arun"]] == [8, 37]  # round(37 * 0.22) = 8


def test_a_completed_sibling_store_survives_the_other_sensors_failure(wired, monkeypatch):
    """The clear must not destroy work that is already correct.

    S1 and S2 ingest concurrently into one prefix. When one fails it leaves no
    marker, so the next attempt sees a not-clean mosaic — and clearing the whole
    prefix threw away the sibling's COMPLETED, correctly-marked output. Because
    such failures are usually deterministic, every retry re-paid that ingest.
    A store already carrying the current fingerprint is complete regardless of what
    happened to its siblings, so it is kept.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    fp = {
        "window": ["2025-01-01", "2025-12-31"],
        "min_valid_coverage": 0.1,
        "s1_orbit": "ascending",
        "allow_partial_window": False,
        "coverage_sha256": "cov-sha-1",
    }

    def probe(store, **kw):
        if store.endswith("sar_ascending.zarr"):
            return (True, fp)  # SAR finished and marked
        if store.endswith("reflectance.zarr"):
            return (True, None)  # S2 crashed part-way, unmarked
        return (False, None)  # descending never ran

    monkeypatch.setattr(mod, "_probe_marker", probe)
    result = _run(s1_orbit="ascending")

    assert result["status"] == "ingested"
    assert wired.get("deletes") == ["s3://in/mosaics/33N/2025/reflectance.zarr"], (
        "the completed SAR store must survive the S2 failure"
    )
