"""ingest_zone_year flow: ocean-skip, marker-skip, dispatch, verify-gate.

The flow body is orchestration (arun_deployment); tests call it via ``.fn`` with
every external touchpoint mocked, so no Prefect engine, Dask cluster, or S3 runs.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

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
    # Stale mosaic cleared before the rebuild so re-ingest doesn't append onto old data.
    assert wired.get("deletes") == ["s3://in/mosaics/33N/2025"]


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
    # The markerless partial was cleared before the rebuild.
    assert wired.get("deletes") == ["s3://in/mosaics/33N/2025"]


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
    assert wired.get("deletes") == ["s3://in/mosaics/33N/2025"]  # policy-mismatch mosaic cleared


def test_coverage_failure_leaves_no_marker(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))

    def _raise(*a, **k):
        raise InsufficientCoverageError("reflectance store starts at 2025-03, but window requires 2025-01")

    monkeypatch.setattr(mod, "check_time_window_coverage", _raise)
    with pytest.raises(InsufficientCoverageError):
        _run(s1_orbit="ascending")
    assert wired["markers_written"] == []  # not marked complete on a coverage gap
