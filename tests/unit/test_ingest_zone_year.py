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
from tessera_embeddings.errors import ConfigMismatchError, InsufficientCoverageError
from tessera_embeddings.ingest import land_mask

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


def _completed_run(rid: str = "r") -> SimpleNamespace:
    return SimpleNamespace(id=rid, state=SimpleNamespace(type=StateType.COMPLETED, name="Completed"))


#: Live chunks the fixture reports. Small enough that _scaled_max_workers lands on its
#: floor, so a test not about sizing gets a stable, uninteresting fleet.
_LIVE_CHUNKS = 4


@pytest.fixture()
def wired(monkeypatch):
    """Mock every external touchpoint; return a record of what the flow did."""
    rec: dict = {"arun": [], "markers_written": [], "manifests_checked": [], "roi_exported": [], "coverage_checked": []}

    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-ingest-zone-year"))
    # Credentials: the flow lazily imports this symbol; patch it on the source module.
    monkeypatch.setattr(
        "tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", object(), raising=False
    )

    async def fake_arun(dep, parameters=None, tags=None):
        rec["arun"].append((dep, parameters))
        rec.setdefault("tags", []).append(tags)
        return _completed_run()

    monkeypatch.setattr(mod, "arun_deployment", fake_arun)
    monkeypatch.setattr(mod, "export_zone_roi", lambda z, **kw: rec["roi_exported"].append((z, kw)))
    monkeypatch.setattr(mod, "check_time_window_coverage", lambda *a, **k: rec["coverage_checked"].append((a, k)))
    monkeypatch.setattr(mod, "_write_ingest_marker", lambda store, fp, **kw: rec["markers_written"].append(store))
    # Reads each store's existing manifest — an external touchpoint like the marker write,
    # and the one gate a zero-append resume has. A test about it overrides this itself.
    monkeypatch.setattr(
        mod, "_assert_store_manifest_matches", lambda store, roi, **kw: rec["manifests_checked"].append(store)
    )
    # Fingerprint inputs: coverage sha + orbit resolution ("ascending" resolves to
    # itself without probing, so _resolved_stores is deterministic in tests).
    monkeypatch.setattr(mod, "_coverage_sha", lambda *a, **k: "cov-sha-1")
    monkeypatch.setattr(mod, "resolve_s1_orbit", lambda mosaic_base, orbit, **k: orbit)
    # Fleet sizing reads the zone's live chunk count on EVERY run now that cropping is
    # unconditional, so it is an external touchpoint like the rest. A test that cares about
    # the resulting worker count overrides this in its own body.
    monkeypatch.setattr(mod, "live_chunk_count", lambda zone, **kw: _LIVE_CHUNKS)
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
    assert wired["arun"] == [] and wired["roi_exported"] == []


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


def test_a_complete_store_under_a_different_fingerprint_raises(wired, monkeypatch):
    """A campaign holds its inputs fixed, so this means a parameter changed — not a crash.

    Neither automatic answer is defensible. Resuming would append dates admitted under one
    configuration onto dates admitted under another and stamp one fingerprint over the
    mixture; clearing would destroy a complete, correct mosaic because someone mistyped a
    window. It raises and a human decides.

    This replaces a test asserting the stores were cleared. That behaviour is what made
    every interruption re-pay a whole cell.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    stale = {
        "window": ["2025-01-01", "2025-12-31"],
        "min_valid_coverage": 0.1,
        "s1_orbit": "ascending",
        "coverage_sha256": "OLD-sha",  # != current "cov-sha-1"
    }
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (True, stale))
    with pytest.raises(ConfigMismatchError, match="different"):
        _run(s1_orbit="ascending")
    assert wired["arun"] == [], "nothing may be dispatched once the mismatch is known"


def test_markerless_partial_mosaic_is_cleared(wired, monkeypatch):
    """A prior attempt that wrote reflectance then crashed before any SAR store or
    marker landed must be cleared, not appended onto: existence over the maximal
    candidate set is what triggers the clean rebuild, and the flow must reach that
    clearing without first asking the orbit probe about the wreckage.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    # Reflectance exists (markerless half-write); the SAR stores never landed.
    monkeypatch.setattr(
        mod, "_probe_marker", lambda store, **kw: (True, None) if store.endswith("reflectance.zarr") else (False, None)
    )

    # Resolution keyed on STATE, not on call ordinality: there is no SAR store to
    # resolve until the ingesters have been dispatched. So a resolution attempted before
    # the dispatch raises, and the post-ingest one succeeds — and the test stays honest
    # about which side of the dispatch the flow probes from.
    def resolve_once_ingested(mosaic_base, orbit, **k):
        if not wired["arun"]:
            raise InsufficientCoverageError("no SAR store yet")
        return orbit

    monkeypatch.setattr(mod, "resolve_s1_orbit", resolve_once_ingested)
    result = _run(s1_orbit="ascending")
    assert result["status"] == "ingested"
    # The markerless partial is RESUMED, not cleared — the flow has no delete path left,
    # so its absence is structural rather than something this asserts around.
    assert not hasattr(mod, "delete_prefix"), "the resume path must not reintroduce a clear"


def test_partial_window_marker_does_not_satisfy_strict_run(wired, monkeypatch):
    """A mosaic accepted under allow_partial_window=True must NOT short-circuit a strict run.

    The policy is in the fingerprint, so a strict run sees a complete store under different
    inputs and RAISES rather than silently reusing a partial mosaic (whose fill would fail
    strict preflight forever) or discarding it. A campaign does not flip this policy
    mid-flight, so reaching here means someone changed it deliberately and should say so.
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
    # Default run is strict (allow_partial_window=False), so the fingerprint differs.
    with pytest.raises(ConfigMismatchError, match="different"):
        _run(s1_orbit="ascending")


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
        # These tests are ABOUT the chunk count, so undo the fixture's stub of it and let
        # the real function derive it from `tile_live` above.
        monkeypatch.setattr(mod, "live_chunk_count", land_mask.live_chunk_count)
        wired["arun"].clear()  # the fixture accumulates; a test may dispatch more than once
        _run(ingest_settings=mod.IngestSettings(**settings_kwargs), s1_orbit="ascending")
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

    def test_the_settings_value_is_a_cap_not_the_fleet_size(self, wired, monkeypatch):
        """Sizing always scales from live chunks; max_workers only bounds the result.

        This replaces a test for the crop-off path, which used to take the settings value
        verbatim. There is no crop-off path now — an extent-sized fleet for a 4-tile zone
        was wrong by orders of magnitude (the 03S incident), and the only reason it was
        ever reachable was a flag that no longer exists.
        """
        monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
        monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
        # Far more live chunks than the cap allows, so the cap is what binds.
        monkeypatch.setattr(mod, "live_chunk_count", lambda zone, **kw: 10_000)
        _run(ingest_settings=mod.IngestSettings(max_workers=37), s1_orbit="ascending")
        # S2 is capped at the settings value; S1 still takes its fraction of what S2 got,
        # because the ratio is about the WORK split between sensors.
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
    # Stronger than before: the completed SAR store survives because NOTHING is deleted,
    # not because the clear was scoped narrowly enough to spare it.
    assert not hasattr(mod, "delete_prefix"), "the resume path must not reintroduce a clear"


def test_child_ingests_receive_the_configured_region(wired, monkeypatch):
    """The children CREATE the mosaic repos, so the region has to reach them.

    This flow's own probes use `s3_region`, so a non-default-region campaign gets
    through preflight and then fails inside the multi-hour S1/S2 jobs, signing
    against the storage layer's default instead.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
    _run(s3_region="eu-west-1")
    dispatched = [p for _, p in wired["arun"]]
    assert dispatched, "no child ingests dispatched"
    assert all(p["s3_region"] == "eu-west-1" for p in dispatched), dispatched


class TestChildRunsAreCancellable:
    """Cancelling this flow has to stop the S1/S2 jobs it started.

    Each is an independent deployment run, so killing this one leaves them writing
    into the mosaic prefix with their Dask fleets billing — and a retry clears and
    rebuilds that same prefix, which is the one race the recovery cannot survive.
    The campaign already sweeps ITS children by tag; that reaches this flow but
    stops there, because these grandchildren carried no tag of their own.
    """

    def test_the_terminal_hooks_are_registered(self):
        """Both, not just cancellation: a crashed parent orphans children identically."""
        assert mod._cancel_children_on_cancellation in mod.ingest_zone_year.on_cancellation_hooks
        assert mod._cancel_children_on_cancellation in mod.ingest_zone_year.on_crashed_hooks

    def test_every_child_carries_the_sweep_tag(self, wired, monkeypatch):
        """Derived from the flow-run id alone, because the hook runs in a fresh import
        after the process is killed and nothing in memory survives to tell it what
        was started.
        """
        monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
        monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
        monkeypatch.setattr(mod.flow_run_ctx, "id", "RUN123", raising=False)
        _run(s1_orbit="both")
        assert wired["tags"], "no child ingests dispatched"
        assert all(t == ["ingest-zone-year:RUN123"] for t in wired["tags"]), wired["tags"]

    def test_no_run_id_means_no_tag_rather_than_a_bogus_one(self, wired, monkeypatch):
        """A direct .fn() call dispatches nothing worth sweeping, so an id-less run must
        not stamp a tag that would later match every other id-less run.
        """
        monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
        monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
        monkeypatch.setattr(mod.flow_run_ctx, "id", None, raising=False)
        _run()
        assert all(t is None for t in wired["tags"]), wired["tags"]


def test_a_rootless_store_is_resumed_rather_than_wedging_every_retry(wired, monkeypatch):
    """A repo created but crashed before its root group is committed must be REBUILT.

    `resolve_s1_orbit` re-raises GroupNotFoundError deliberately — at fill time a
    rootless SAR store must never read as an absent orbit and quietly halve the radar.
    Resolving before the clear put that raise in front of the only code that repairs
    the wreckage, so the prefix stayed wedged for every retry.
    """
    import zarr

    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    # The ascending SAR store exists but is rootless; the others were never written.
    monkeypatch.setattr(
        mod,
        "_probe_marker",
        lambda store, **kw: (True, None) if store.endswith("sar_ascending.zarr") else (False, None),
    )

    def resolve_on_a_rootless_store(mosaic_base, orbit, **k):
        # Keyed on STATE, not on call ordinality: the store stays rootless until a child
        # re-seeds it, so resolution must fail until the ingesters have been dispatched.
        if not wired["arun"]:
            raise zarr.errors.GroupNotFoundError(f"{mosaic_base}/sar_ascending.zarr")
        return orbit

    monkeypatch.setattr(mod, "resolve_s1_orbit", resolve_on_a_rootless_store)
    result = _run(s1_orbit="ascending")
    # The point of the test: the run gets PAST the probe. Orbit resolution re-raises
    # GroupNotFoundError by design, and resolving before the ingesters were dispatched put
    # that raise in front of the only code that repairs the store — every retry died there
    # and the prefix stayed wedged. It is now skipped whenever a store is incomplete.
    assert result["status"] == "ingested"
    assert wired["arun"], "the rootless store must be resumed, which means children run"


def test_local_mode_does_not_require_the_aws_provider() -> None:
    """The documented dev path must be reachable without the optional `aws` extra.

    `use_local=True` runs against local stores that need no IAM callback, but the flow
    imported `providers.aws.credentials` unconditionally before reaching the branch —
    so an install without botocore could not use the mode it was told to use. Asserted
    as source, because the failure is an ImportError on a machine that HAS the extra
    installed, and a test cannot uninstall it.
    """
    import inspect

    import tessera_embeddings.orchestration.prefect.flows.ingest_zone_year as mod

    source = inspect.getsource(mod.ingest_zone_year.fn)
    line = next(i for i, ln in enumerate(source.splitlines()) if "providers.aws" in ln)
    guard = next(i for i, ln in enumerate(source.splitlines()) if "if not use_local:" in ln)
    assert guard < line, "the AWS provider import must sit inside the non-local branch"


# ===========================================================================
# Automatic leg retry
# ===========================================================================


class TestLegRetryClassification:
    """Which leg failures a re-dispatch is allowed to retry.

    The polarity is the point: retry by DEFAULT, and exclude only failures that are
    deterministic in the input. A re-dispatch resumes — committed dates are skipped, not
    rewritten — so a wasted retry costs one leg's remaining work, while a missed retry
    leaves a mosaic incomplete until a human notices. That is not hypothetical: on
    2026-08-04 an expired source credential and a warp error each killed a leg hours in,
    and 35N, 12N and 53N each needed manual re-dispatch.
    """

    @pytest.mark.parametrize(
        "detail",
        [
            "ingest_s1_roi_sar (ascending): PermissionError: The provided token has expired.",
            "ingest_s2_roi_reflectance: Exception: WarpOperationError('Chunk and warp failed')",
            "ingest_s2_roi_reflectance did not complete successfully (state=Crashed)",
            "ClientError: An error occurred (ThrottlingException) when calling ListTasks",
            "KilledWorker: worker died",
            "TimeoutError",
        ],
    )
    def test_transient_failures_are_retried(self, detail: str) -> None:
        assert mod._is_retryable_leg_failure(detail) is True

    @pytest.mark.parametrize(
        "detail",
        [
            "InsufficientCoverageError: s1_orbit='both' but no SAR stores found under s3://...",
            "ObjectNotFound: None",
            "ValidationError: 3 validation errors for IngestSettings",
            "rejected by the calendar-year gate",
            "zone has no live tiles in the campaign coverage mask",
        ],
    )
    def test_deterministic_failures_are_not_retried(self, detail: str) -> None:
        assert mod._is_retryable_leg_failure(detail) is False

    def test_matching_is_case_insensitive(self) -> None:
        """The marker list is matched against exception text whose casing we do not own."""
        assert mod._is_retryable_leg_failure("insufficientcoverageerror: none") is False

    def test_an_unrecognised_failure_is_retried(self) -> None:
        """The default must be to retry: a failure nobody has classified yet is far more
        likely to be a transient one than a deterministic one, and the cost of being wrong
        in this direction is one resumable leg.
        """
        assert mod._is_retryable_leg_failure("SomeNewError: nobody has seen this before") is True


class TestLegFailureDetail:
    """The detail string that classification reads."""

    def test_a_completed_leg_yields_no_detail(self) -> None:
        run = SimpleNamespace(state=SimpleNamespace(type=StateType.COMPLETED, name="Completed"))
        assert mod._leg_failure_detail(run, "s2") is None

    def test_an_exception_result_is_reported_verbatim(self) -> None:
        detail = mod._leg_failure_detail(RuntimeError("dispatch blew up"), "s2")
        assert detail is not None
        assert "dispatch blew up" in detail
        assert detail.startswith("s2:")

    def test_the_state_message_is_included_not_just_the_name(self) -> None:
        """Classification needs the exception text; the state NAME ("Failed") carries none.

        This is what lets an expired-credential failure be told apart from a coverage gate.
        """
        run = SimpleNamespace(
            state=SimpleNamespace(
                type=StateType.FAILED,
                name="Failed",
                message="PermissionError: The provided token has expired.",
            )
        )
        detail = mod._leg_failure_detail(run, "s1 (ascending)")
        assert detail is not None
        assert "token has expired" in detail
        assert mod._is_retryable_leg_failure(detail) is True

    def test_a_missing_state_message_still_yields_a_detail(self) -> None:
        run = SimpleNamespace(state=SimpleNamespace(type=StateType.FAILED, name="Failed", message=None))
        detail = mod._leg_failure_detail(run, "s2")
        assert detail is not None
        assert "did not complete successfully" in detail


def test_max_leg_attempts_defaults_to_retrying() -> None:
    """Off by default would leave the behaviour that cost three zones a day."""
    from tessera_embeddings.config.ingest import IngestSettings

    assert IngestSettings().max_leg_attempts >= 2


def test_a_terminal_leg_failure_is_not_forgotten_when_a_sibling_retries(wired, monkeypatch):
    """A leg that can never succeed must fail the flow even if its siblings recover.

    ``errors`` is rebuilt each attempt and only retryable legs are re-dispatched, so
    retrying around a terminal failure used to erase it. For ``s1_orbit="both"`` that
    is silent data loss: the surviving orbit resolves, passes the coverage gate, and
    gets stamped with a "both" marker, after which every later run reads the marker
    and skips the cell — half its radar gone for good.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
    dispatched: list[str] = []

    async def fake_arun(dep, parameters=None, tags=None):
        orbit = parameters.get("orbit")
        dispatched.append(str(orbit))
        if orbit == "ascending":
            # Not retryable: the deployment itself is missing.
            return SimpleNamespace(
                id="r",
                state=SimpleNamespace(
                    type=StateType.FAILED, name="Failed", message="ObjectNotFound: no such deployment"
                ),
            )
        if orbit == "descending":
            return SimpleNamespace(
                id="r",
                state=SimpleNamespace(
                    type=StateType.FAILED, name="Failed", message="PermissionError: The provided token has expired."
                ),
            )
        return _completed_run()

    monkeypatch.setattr(mod, "arun_deployment", fake_arun)

    with pytest.raises(RuntimeError, match="ObjectNotFound"):
        _run(s1_orbit="both", ingest_settings=mod.IngestSettings(max_leg_attempts=3))

    # Aborted on the first attempt rather than retrying the recoverable legs around a
    # failure that a re-dispatch cannot fix.
    assert sorted(dispatched) == ["None", "ascending", "descending"]  # one dispatch each, no retry
    assert wired["markers_written"] == []  # and nothing was marked complete


def test_the_manifest_is_checked_before_any_marker_is_stamped(wired, monkeypatch):
    """A resume that appends NOTHING has had no manifest check at all.

    The ingest's append path validates on every batch, which is what makes resuming an
    interrupted store safe. But an attempt that wrote every date and crashed before its
    marker leaves a store the next run adopts as unmarked; the child legs then skip every
    date as already present and write nothing, so the append check never fires — and the
    marker gets stamped over a mosaic built under a different mask, threshold or ingest
    code. Every later run then reads that marker and skips the cell.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (True, None))  # unmarked = resume
    order: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "_assert_store_manifest_matches", lambda s, roi, **kw: order.append(("check", s)))
    monkeypatch.setattr(mod, "_write_ingest_marker", lambda s, fp, **kw: order.append(("mark", s)))

    _run(s1_orbit="ascending")

    # Every store checked, and every check before every mark: a disagreement on the last
    # store must not leave the earlier ones already marked.
    assert [s for kind, s in order if kind == "check"] == [s for kind, s in order if kind == "mark"]
    assert [kind for kind, _ in order] == ["check", "check", "mark", "mark"]


def test_a_manifest_disagreement_stops_the_marker(wired, monkeypatch):
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (True, None))

    def refuse(store, roi, **kw):
        raise ConfigMismatchError(f"{store}: coverage_sha256 changed")

    monkeypatch.setattr(mod, "_assert_store_manifest_matches", refuse)
    with pytest.raises(ConfigMismatchError, match="coverage_sha256"):
        _run(s1_orbit="ascending")
    assert wired["markers_written"] == []


def test_only_the_optical_store_is_held_to_the_admission_threshold(wired, monkeypatch):
    """The SAR legs apply no coverage threshold, so their manifests carry none —
    expecting one would fail every radar store on a path that only exists to protect them.
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)
    monkeypatch.setattr(mod, "_probe_marker", lambda store, **kw: (False, None))
    seen: dict[str, float | None] = {}
    monkeypatch.setattr(
        mod,
        "_assert_store_manifest_matches",
        lambda s, roi, *, min_valid_coverage, **kw: seen.update({s.rsplit("/", 1)[-1]: min_valid_coverage}),
    )

    _run(s1_orbit="ascending", ingest_settings=mod.IngestSettings(min_valid_coverage=0.25))
    assert seen == {"reflectance.zarr": 0.25, "sar_ascending.zarr": None}
