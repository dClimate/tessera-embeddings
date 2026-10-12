"""run_global_campaign: preflight, chained-fill dispatch, retry rounds, staging identity.

The work-list logic itself is covered in test_campaign (campaign_work_list); here the flow
body is driven via ``.fn`` with mocked dispatch.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prefect.states import StateType

import tessera_embeddings.orchestration.prefect.flows.run_global_campaign as mod
from tessera_embeddings.config.paths import BucketPaths

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


def _campaign(**kwargs) -> dict:
    """Drive the campaign flow body with mocked dispatch."""
    return asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", **kwargs))


def _completed_run(rid: str = "r") -> SimpleNamespace:
    return SimpleNamespace(id=rid, state=SimpleNamespace(type=StateType.COMPLETED, name="Completed"))


def _defer_milestone(*_a, **_k):
    raise ValueError("not all 120 zones complete yet")  # tag_year_complete stub → milestone deferred


def _shrinking(cells: list[tuple[str, int]], wired: dict):
    """A `campaign_work_list` stub that drops cells as their fills land.

    The real one is re-derived from the store every round, and the campaign now trusts
    it over a child's failure list — a child can report success having never attempted
    a cell. A constant stub models a store that records nothing, so the campaign would
    re-dispatch the same work until it ran out of attempts.
    """

    def _work(_status=None, _tags=None, *, expected_zones=None, years=None):
        wanted = set(years) if years else None
        # `expected_zones` is HONOURED, as the real one honours it. A stub that ignored it
        # would hand a zone-scoped caller every zone, which is the shape of the bug a
        # zone-scoped caller exists to avoid.
        scope = set(expected_zones) if expected_zones is not None else None
        return [
            c
            for c in cells
            if c not in wired["landed"] and (wanted is None or c[1] in wanted) and (scope is None or c[0] in scope)
        ]

    return _work


def _dispatched_cells(params: dict) -> list[tuple[str, int]]:
    """The (zone, year) cells one chained-fill dispatch covers; anything else covers none."""
    return [(z, y) for z, y in params.get("cells", [])]


@pytest.fixture()
def wired(monkeypatch):
    """Mock the store + dispatch; return records of what the flow did."""
    rec: dict = {"arun": [], "deletes": []}
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-campaign"))
    monkeypatch.setattr(mod, "open_global_repo", lambda *a, **k: SimpleNamespace(list_tags=lambda: set()))
    # A seeded, unfilled status: one zone group, nothing complete (so no retag skip).
    status = SimpleNamespace(zones={"33N": ()}, has=lambda z, y: False)
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: True)  # requested years are on-axis
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: True)  # default: a live (non-ocean) cell
    # The work list must SHRINK as fills land, because that is what the real one does —
    # it is re-derived from the store every round, and the campaign now trusts it rather
    # than a child's failure list (a child can report success having skipped cells). A
    # constant stub models a store that never records anything, so every round would
    # find the same work and the campaign would exhaust its attempts.
    _landed: set[tuple[str, int]] = set()
    rec["landed"] = _landed

    def _land(parameters: dict | None) -> None:
        """Record a completed dispatch's cells as present in the store.

        Exposed on the fixture because tests that install their OWN `arun_deployment`
        replace the default wholesale, and a dispatch that lands nothing makes the
        campaign re-read an unchanged store and re-dispatch forever.
        """
        _landed.update(_dispatched_cells(parameters or {}))

    rec["land"] = _land
    monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [c for c in [("33N", 2025)] if c not in _landed])
    monkeypatch.setattr(mod, "tag_year_complete", _defer_milestone)
    # Pinned AMI id (resolved once, threaded into every fill's provisioning) — mocked
    # so tests make no SSM call.
    monkeypatch.setattr(mod, "_resolve_ami_id", lambda *a, **k: "ami-test-id")
    # Seeded-model gate: real in production (a metadata read of the store root),
    # stubbed here so tests exercise dispatch rather than store contents.
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: None)
    # Store reads for the staging fingerprint: each mosaic store's branch-tip snapshot
    # (the authoritative term) plus its provenance attrs.
    monkeypatch.setattr(
        mod,
        "open_store_group_and_tip",
        lambda *a, **k: (
            SimpleNamespace(attrs={"registry_sha256": "cov-sha-test", "last_appended": "2026-01-01T00:00:00Z"}),
            "SNAPSHOT0",
        ),
    )

    async def fake_arun(dep, parameters=None, tags=None):
        rec["arun"].append((dep, parameters))
        rec.setdefault("tags", []).append(tags)
        # A completed fill puts its cell(s) in the store, which is what the next
        # round's work list is derived from.
        _land(parameters)
        return _completed_run()

    monkeypatch.setattr(mod, "arun_deployment", fake_arun)

    monkeypatch.setattr(mod, "delete_prefix", lambda uri, **k: rec["deletes"].append(uri))
    # The fleet-wide caps are real Prefect API writes; record them instead.
    monkeypatch.setattr(
        mod, "_upsert_limit", lambda name, limit, **k: rec.setdefault("limits", []).append((name, limit))
    )
    return rec


class TestCampaignDefaults:
    """The scheduling defaults are load-bearing and expensive to get wrong.

    Pinned as literals rather than derived, so a change has to be made on purpose
    and shows up as a failing test rather than a surprise on a live campaign.
    """

    def test_ten_clusters_share_sixty_ingest_slots(self):
        """Six UTM zones ingesting per cluster: 10 x 6 = the 60-zone fleet-wide cap.

        The campaign's planned width (``context_docs/campaign/campaign-plan.md`` §3), and 10 x 250
        actors is what reaches the 2,500-actor quota.

        The two numbers are a pair, not independent knobs — the per-cluster ingest window is the cap
        divided by the cluster count, so changing either changes how many zones a cluster keeps in
        flight. The plan named 61; 60 is used instead precisely because of the assertion below.
        """
        sig = inspect.signature(mod.run_global_campaign.fn)
        clusters = sig.parameters["max_parallel_clusters"].default
        ingests = sig.parameters["max_parallel_ingest"].default
        assert (clusters, ingests) == (10, 60)
        assert ingests % clusters == 0, "an uneven split rounds up and overshoots the cap"

    def test_the_actor_count_reaches_the_quota_without_exceeding_the_assembly_ceiling(self):
        """250 per cluster x 10 = the 2,500-actor quota, and each stays under ~275.

        Both bounds matter and they pull opposite ways: fewer actors leaves quota unused, more
        pushes a cluster's single assembly thread past the width it was measured to sustain.
        """
        sig = inspect.signature(mod.run_global_campaign.fn)
        actors = sig.parameters["num_actors"].default
        clusters = sig.parameters["max_parallel_clusters"].default
        assert actors == 250
        assert actors * clusters == 2500, "the fleet should reach the actor quota exactly"
        assert actors < 275, "one cluster's assembly thread is the ceiling"

    def test_every_gate_is_published_to_the_server(self, wired):
        """Both gates live in CHILD flow runs, so a gate only binds if it reaches the
        Prefect limit they name. Writing them from this flow is what stops the server's
        numbers drifting from the ones chosen here — and, for the pause gate, what makes
        pausing reachable without an operator creating anything first.

        There were three. The commit gate is gone — nothing bounds committers any more, and
        `test_no_committer_gate_is_published` pins that it stays gone.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
        assert dict(wired["limits"]) == {
            "tessera-global-ingests": 60,
            "tessera-global-inference": 1,
        }
        assert wired["arun"][0][1]["ingest_limit_name"] == "tessera-global-ingests"
        assert wired["arun"][0][1]["inference_pause_gate"] == "tessera-global-inference"

    def test_the_pause_gate_is_published_as_running_not_as_a_cap(self, wired):
        """One, not the cluster count. The gate is READ as a flag rather than acquired, so a
        larger number would imply a capacity it does not enforce — and zero, the value that
        pauses, must never be what a campaign start writes.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=8))
        assert dict(wired["limits"])["tessera-global-inference"] == 1

    def test_a_start_clears_a_pause_somebody_left_behind(self, wired):
        """The gate is upserted at every start, so a campaign cannot inherit a zero from a
        previous run's pause and sit there looking healthy while doing nothing.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
        writes = [v for k, v in wired["limits"] if k == "tessera-global-inference"]
        assert writes == [1]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"num_actors": 0}, "num_actors must be >= 1"),
            # A card name is checked HERE and not only inside the child. The child's own
            # refusal arrives after it has primed its look-ahead mosaics and entered
            # `ray_cluster`, so a typo would spend real ingest work per cluster before
            # anything said so.
            ({"gpu_fallback_instance_types": ["g5.xlarge"]}, "unsupported type"),
        ],
    )
    def test_a_doomed_invocation_touches_no_shared_limit(self, wired, kwargs, match):
        """A call that cannot possibly run must not have changed the fleet's shared controls.

        The pause gate is the one that matters: ZERO is how an operator pauses a campaign already
        in flight, and a start RESETS the gate to running. So an invocation that reset the gate and
        then died on a typo would have resumed somebody else's paused fleet on its way out — a side
        effect on shared state from a call that accomplished nothing. The validations therefore run
        before any upsert, and this asserts the absence rather than the ordering.
        """
        with pytest.raises(ValueError, match=match):
            asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", **kwargs))
        assert wired.get("limits", []) == [], "no cap, and above all no pause-gate reset"

    def test_a_rejected_invocation_never_touches_the_shared_gates(self, wired, monkeypatch):
        """Resuming somebody else's paused fleet on the way out is the failure to avoid.

        Zero on the pause gate is how an operator pauses a campaign ALREADY IN FLIGHT, and the
        gate is fleet-wide. A run that wrote it and then rejected would have resumed that fleet
        as a side effect of accomplishing nothing. The parameter checks were already ahead of the
        writes; the store read, the model gate, the year-axis probe and the seeded-zone guards
        sit between them and can reject too, which is what this pins.
        """
        # A work cell whose zone is not seeded — the last guard before dispatch.
        monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [("99N", 2025)])

        with pytest.raises(ValueError, match="not seeded"):
            asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))

        assert wired.get("limits", []) == [], "a rejected run must leave every shared gate alone"

    def test_a_no_work_invocation_never_touches_the_shared_gates(self, wired, monkeypatch):
        """The same hazard without an error: nothing to dispatch, so nothing to resume for."""
        monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [])

        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))

        assert wired.get("limits", []) == [], "a run with no work must not reset a live pause"
        assert wired["arun"] == [], "and must dispatch nothing"

    def test_no_ingest_cap_is_published_when_ingest_is_off(self, wired):
        """Prebuilt mosaics mean no ingest to gate. The inference pause is still published,
        because inference is exactly what this run still does and pausing it must stay
        available.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False))
        assert dict(wired["limits"]) == {"tessera-global-inference": 1}

    @pytest.mark.parametrize("clusters", [1, 4, 8, 16, 40])
    def test_no_committer_gate_is_published_at_any_cluster_count(self, wired, clusters):
        """Commits are ungated, and the cluster count no longer implies a committer cap.

        The removed gate derived ``min(clusters, 8)`` from the run-1 contention curve. It was
        removed because what it bounded is a SLOWDOWN, not a failure: all 120 zone groups do
        share one repo and one branch tip, and run 1 measured that contention as 2.2 s commits
        at 16 simultaneous committers and 15 s at 120, with ZERO unresolvable conflicts at every
        N. The curve itself is kept in
        ``context_docs/storage/writing-to-the-global-store.md`` together with the scale
        (N>=16) at which it would matter again — this is parametrised past that scale on
        purpose, so a reader sees the removal is unconditional rather than tuned.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=clusters))
        assert "tessera-global-commits" not in dict(wired["limits"])

    def test_the_flow_takes_no_commit_limit_parameter(self):
        """Structural: there is no knob left to re-enable a fleet-wide committer bound, so a
        dispatch cannot reintroduce one without a code change.
        """
        import inspect

        flow = getattr(mod.run_global_campaign, "fn", mod.run_global_campaign)
        assert "commit_limit_name" not in inspect.signature(flow).parameters


def test_driver_reads_are_credentialed(wired, monkeypatch):
    """The driver's own store reads (open_global_repo + the on-axis probe) carry the
    IAM callback — a callback-only store must authenticate on the driver too, before
    any child flow is dispatched, not just inside the children.
    """
    captured: dict = {}
    monkeypatch.setattr(
        mod, "open_global_repo", lambda *a, **k: captured.update(open_kw=k) or SimpleNamespace(list_tags=lambda: set())
    )
    onaxis: dict = {}
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: onaxis.update(k) or True)

    _campaign()
    assert captured["open_kw"].get("get_credentials") is not None
    assert onaxis.get("get_credentials") is not None


def _dispatched_zones(params: dict) -> list[str]:
    """The zones of a chained dispatch, from its `(zone, year)` pairs.

    The child takes pairs so a cluster can span years. The tests that use this carry one
    year, so every pair in one dispatch shares it; asserting on zones keeps them about the
    PARTITION rather than about the parameter shape.
    """
    years = {y for _, y in params["cells"]}
    assert len(years) == 1, f"a single-year work list must dispatch one year per cluster, got {years}"
    return [z for z, _ in params["cells"]]


_STAGING = {
    "inputs_bucket": "s3://in",
    "min_valid_coverage": 0.001,
    "s1_orbit": "both",
    "allow_partial_window": False,
    "allow_s2_only": False,
    "optical_min_obs": None,
    "code_identity": "infcode-x",
}

#: An ingest marker is policy, so a rebuild under identical settings reproduces it exactly.
_MARKER = {"window": ["2025-01-01", "2025-12-31"], "coverage_sha256": "cov"}


def _run_id(monkeypatch, *, attrs: dict | None = None, tip: str = "SNAPSHOT0", **overrides) -> str:
    """The staging run_id of 33N-2025 over mosaic stores carrying ``attrs`` at snapshot ``tip``."""
    attrs = {"last_appended": "2026-01-01T00:00:00Z"} if attrs is None else attrs
    monkeypatch.setattr(mod, "open_store_group_and_tip", lambda *a, **k: (SimpleNamespace(attrs=attrs), tip))
    return mod._staging_run_id("33N", 2025, **(_STAGING | overrides))


class TestStagingRunId:
    """The prefix the chained fill stages each cell under (:func:`_staging_run_id`).

    Deterministic per input set, so a retry resumes the same prefix; any change to what decides
    the embeddings starts a fresh one, so tiles staged under old inputs are never resumed.
    """

    def test_it_is_cell_scoped_and_stable(self, monkeypatch):
        rid = _run_id(monkeypatch)
        assert rid.startswith("33N-2025-") and _run_id(monkeypatch) == rid

    @pytest.mark.parametrize(
        "override",
        [
            {"min_valid_coverage": 0.5},
            # allow_s2_only changes WHICH pixels get embeddings, so a retry across a flipped
            # flag must never resume mixed tiles.
            {"allow_s2_only": True},
            # Staging lives under `outputs/staging/{zone}/{year}`, not namespaced by store, so
            # a store with a different depth rule must not resume tiles staged under the old.
            {"optical_min_obs": 25},
            {"optical_min_obs": 30},
            {"code_identity": "infcode-y"},
        ],
        ids=lambda o: next(iter(o)),
    )
    def test_every_input_that_decides_the_embeddings_moves_it(self, monkeypatch, override):
        assert _run_id(monkeypatch, **override) != _run_id(monkeypatch)

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            # Rebuilt under identical policy: the marker is byte-identical while the pixels can
            # differ (a reprocessed or late-published granule); ingest_completed_at moves.
            (
                {"attrs": {"ingest_marker": _MARKER, "ingest_completed_at": "2026-01-01T00:00:00Z"}},
                {"attrs": {"ingest_marker": _MARKER, "ingest_completed_at": "2026-02-02T00:00:00Z"}},
            ),
            # A prebuilt (ingest=False) mosaic replaced upstream: its own last_appended moves.
            (
                {"attrs": {"last_appended": "2026-01-01T00:00:00Z"}},
                {"attrs": {"last_appended": "2026-06-01T00:00:00Z"}},
            ),
            # The snapshot tip is the authority: a region overwrite or hand repair changes
            # pixels while every provenance attr stays exactly as it was.
            (
                {"attrs": {"ingest_marker": _MARKER, "ingest_completed_at": "2026-01-01"}, "tip": "SNAPSHOT_BEFORE"},
                {"attrs": {"ingest_marker": _MARKER, "ingest_completed_at": "2026-01-01"}, "tip": "SNAPSHOT_AFTER"},
            ),
            # A store with no provenance attrs at all is identified by its tip alone.
            ({"attrs": {}, "tip": "SNAPSHOT_A"}, {"attrs": {}, "tip": "SNAPSHOT_B"}),
        ],
        ids=["rebuilt", "prebuilt-replaced", "snapshot-moved", "attrless"],
    )
    def test_a_changed_mosaic_moves_it(self, monkeypatch, before, after):
        first = _run_id(monkeypatch, **before)
        assert _run_id(monkeypatch, **before) == first, "the same mosaic must resume"
        assert _run_id(monkeypatch, **after) != first

    def test_only_active_orbits_are_fingerprinted(self, monkeypatch):
        """A single requested orbit opens reflectance + that SAR store only — never the opposite
        orbit, even if a stale one is present, which the fill never reads and which could
        wrongly raise or perturb the run_id.
        """
        opened: list[str] = []

        def _capture(path, **_k):
            opened.append(path)
            return SimpleNamespace(attrs={}), "SNAPSHOT0"

        monkeypatch.setattr(mod, "open_store_group_and_tip", _capture)
        mod._staging_run_id("33N", 2025, **(_STAGING | {"s1_orbit": "ascending"}))
        assert [p.rsplit("/", 1)[-1] for p in opened] == ["reflectance.zarr", "sar_ascending.zarr"]


def _chained_staging_identity(wired, **kwargs) -> str:
    """The staging identity the campaign hands its chained fill."""
    wired["arun"].clear()
    wired["landed"].clear()
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", **kwargs))
    params = next(p for d, p in wired["arun"] if "fill-zones-sequential" in d)
    return params["staging_code_identity"]


class TestTheChainedFillGetsTheNarrowedStagingIdentity:
    """The narrowing and its escape hatches have to reach the child that stages.

    Left to itself the chained fill recomputes the AMI-plus-tarball artifact identity — so a
    re-bake would abandon every staged tile and `force_staging_reuse`/`force_staging_restage`
    would be documented on a flow that never forwarded them.
    """

    def test_the_chained_child_is_handed_the_narrowed_identity(self, wired, monkeypatch):
        monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-FIXED")
        assert _chained_staging_identity(wired) == "infcode-FIXED"

    def test_a_rebaked_ami_does_not_change_it(self, wired, monkeypatch):
        """The inversion the narrowing was for."""
        base = _chained_staging_identity(wired)
        monkeypatch.setattr(mod, "_resolve_code_identity", lambda *a, **k: "ami=ami-REBAKED")
        assert _chained_staging_identity(wired) == base

    def test_an_inference_change_does(self, wired, monkeypatch):
        base = _chained_staging_identity(wired)
        monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-DIFFERENT")
        assert _chained_staging_identity(wired) != base

    def test_both_escape_hatches_reach_it(self, wired):
        base = _chained_staging_identity(wired)
        assert _chained_staging_identity(wired, force_staging_reuse=True) != base
        assert _chained_staging_identity(wired, force_staging_restage="tok") != base

    def test_force_staging_reuse_survives_an_inference_change(self, wired, monkeypatch):
        """The escape hatch for a change the author knows is output-neutral."""
        base = _chained_staging_identity(wired, force_staging_reuse=True)
        monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-DIFFERENT")
        assert _chained_staging_identity(wired, force_staging_reuse=True) == base
        # ...and it really is an override, not a no-op: without it the same change moves it.
        assert _chained_staging_identity(wired) != base

    def test_force_staging_restage_forces_a_fresh_prefix(self, wired):
        """The escape hatch for a change the source hash cannot see — a torch upgrade."""
        base = _chained_staging_identity(wired)
        bumped = _chained_staging_identity(wired, force_staging_restage="torch-2.9")
        assert bumped != base
        # Same token → same prefix, so a retry after the upgrade still resumes.
        assert _chained_staging_identity(wired, force_staging_restage="torch-2.9") == bumped
        assert _chained_staging_identity(wired, force_staging_restage="torch-3.0") not in {base, bumped}


class TestAPinnedStagingIdentityReachesAnEarlierCampaignsPrefix:
    """The lever for restarting a campaign onto the tiles it already staged.

    Both other hatches DERIVE the identity from the code in front of them, so neither can
    reproduce the value that changed code replaced — which is why a restart across an
    orchestration fix abandoned every staged tile. Stating the identity outright is the only
    form that reaches backwards.
    """

    def test_it_survives_an_inference_change(self, wired, monkeypatch):
        pinned = _chained_staging_identity(wired, staging_code_identity="infcode-EARLIER")
        assert pinned == "infcode-EARLIER"
        monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-DIFFERENT")
        assert _chained_staging_identity(wired, staging_code_identity="infcode-EARLIER") == pinned

    def test_it_survives_a_changed_tarball(self, wired, monkeypatch):
        """The term that bites hardest on a tarball deployment: re-uploading moves it."""
        monkeypatch.setattr(mod, "_resolve_tarball_identity", lambda *a, **k: "tarball=NEW")
        assert _chained_staging_identity(wired, staging_code_identity="infcode-EARLIER|tarball=OLD") == (
            "infcode-EARLIER|tarball=OLD"
        )

    def test_an_empty_pin_derives_exactly_as_before(self, wired, monkeypatch):
        """The control and the default: off must cost nothing, including the derivation."""
        monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-FIXED")
        assert _chained_staging_identity(wired, staging_code_identity="") == "infcode-FIXED"

    def test_the_identity_is_announced_so_a_later_campaign_can_state_it(self, wired, monkeypatch, caplog):
        """The value has to be READABLE or the lever cannot be used.

        Each chained fill records it as a parameter, but the driver's own log is the one place
        it is stated once for the whole campaign, and by the time anyone wants it the code that
        produced it has moved.
        """
        monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-FIXED")
        with caplog.at_level("INFO"):
            _chained_staging_identity(wired)
        announced = [r.getMessage() for r in caplog.records if "Staging code identity" in r.getMessage()]
        assert announced, "the driver never said what identity it resolved"
        assert "infcode-FIXED" in announced[0]
        assert len(announced) == 1, f"announced once per campaign, not per cell: {len(announced)}"

    def test_the_derived_identity_is_never_frozen_by_the_announcement(self, wired, monkeypatch, caplog):
        """Announcing once must not mean deriving once.

        The tarball ETag is the one term that can move WHILE a campaign runs, and the safe
        response to a replaced tarball is a FRESH prefix. Caching the identity to make the log
        line fire once would instead send later dispatches back to the prefix the OLD code
        staged — two code versions in one write-once zone-year, which is the failure the
        fingerprint exists to prevent. A pinned identity is a constant and never consults the
        tarball, so this is the derived path's hazard alone.
        """
        tarballs = iter(["tarball=FIRST", "tarball=SECOND"])
        monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-FIXED")
        monkeypatch.setattr(mod, "_resolve_tarball_identity", lambda *a, **k: next(tarballs, "tarball=SECOND"))
        with caplog.at_level("INFO"):
            first = _chained_staging_identity(wired)
            second = _chained_staging_identity(wired)
        assert first == "infcode-FIXED|tarball=FIRST"
        assert second == "infcode-FIXED|tarball=SECOND", "a replaced tarball must start a fresh prefix"
        announced = [r.getMessage() for r in caplog.records if "Staging code identity" in r.getMessage()]
        assert len(announced) == 2, "a mid-campaign swap must announce itself, not hide behind the first line"

    @pytest.mark.parametrize(
        ("kwargs", "needle"),
        [
            ({"force_staging_reuse": True}, "force_staging_reuse"),
            ({"force_staging_restage": "torch-2.9"}, "force_staging_restage"),
        ],
    )
    def test_it_is_refused_alongside_a_derived_hatch(self, wired, kwargs, needle) -> None:
        """Three levers each CLAIMING the identity is a contradiction, not a combination.

        Refused rather than ordered by precedence: silently preferring one lands the run on a
        prefix nobody chose, and the two directions of that mistake cost a campaign's
        re-inference and a mixed-code zone-year respectively.
        """
        with pytest.raises(ValueError, match="Pass exactly one") as excinfo:
            _chained_staging_identity(wired, staging_code_identity="infcode-EARLIER", **kwargs)
        # The message has to NAME the lever that conflicts: an operator restarting a campaign
        # passed three staging arguments and needs to know which one to drop.
        assert needle in str(excinfo.value)

    def test_the_two_derived_hatches_are_refused_together_too(self, wired) -> None:
        """The pair that predates this change, and the reason the rule is stated over all three.

        Reuse collapsed the identity to a constant and simply ignored the restage token, so an
        operator asking for FRESH staging during a dependency upgrade silently got the reused
        prefix — the exact mixing the token exists to prevent. An enumeration that has to be
        complete to be safe cannot stop at the pairs the newest lever happens to be in.
        """
        with pytest.raises(ValueError, match="Pass exactly one") as excinfo:
            _chained_staging_identity(wired, force_staging_reuse=True, force_staging_restage="torch-2.9")
        assert "force_staging_reuse and force_staging_restage" in str(excinfo.value)


def test_the_ami_is_still_pinned_even_though_it_no_longer_fingerprints(wired, monkeypatch):
    """Narrowing the fingerprint must not stop the AMI being resolved once and pinned.

    One campaign must never straddle two images: the fingerprint no longer notices, so the
    pinning is now the ONLY thing preventing it. Also pins the region — the AMI SSM param
    lives where Ray provisions (None → us-west-2), not in a non-default storage region.
    """
    calls: list = []
    monkeypatch.setattr(mod, "_resolve_ami_id", lambda *a, **k: calls.append(a) or "ami-test")
    _campaign(s3_region="eu-west-1")
    assert calls, "the AMI must still be resolved"
    assert all(c[-1] != "eu-west-1" for c in calls), calls
    assert _dep(wired, "fill-zones-sequential")[1]["ami_id"] == "ami-test", "and pinned into the fill"


class TestValidationDeploymentForwarding:
    """An explicit validator reaches the chained fill; an absent one is left ALONE.

    The absent case is the load-bearing one and it is not symmetric with the rest of this
    flow's parameters. Every other setting is forwarded unconditionally, which is right when
    the campaign is the authority. Here it is not: the validator is a consumer's flow, named
    on the fill deployment itself, so passing None would override a correctly-registered
    ref with nothing and run a whole campaign unvalidated while every deployment still looked
    configured.
    """

    def test_the_chained_fill_receives_an_explicit_ref(self, wired):
        asyncio.run(
            mod.run_global_campaign.fn(
                paths=_PATHS, ami_ssm_name="ami", validation_deployment="validate-zone-year/validate-zone-year-x"
            )
        )
        params = next(p for d, p in wired["arun"] if "fill-zones-sequential" in d)
        assert params["validation_deployment"] == "validate-zone-year/validate-zone-year-x"

    @pytest.mark.parametrize("branch", [None, "some-branch"])
    def test_an_unset_validator_is_omitted_rather_than_forwarded_as_none(self, wired, branch):
        """So the CHILD deployment's own registered value stands — including its branch form.

        Both branch cases, because the validator is the one child ref this flow does NOT
        derive from the slug: there is no base name here to suffix, and the branch-scoped
        registration on the child already carries it.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", branch=branch))
        params = next(p for d, p in wired["arun"] if "fill-zones-sequential" in d)
        assert "validation_deployment" not in params


def test_the_fill_carries_the_campaigns_s3_region(wired):
    """A non-default-region deployment's fill — and the ingests it dispatches — must read and
    write the right buckets.
    """
    _campaign(s3_region="eu-west-1")
    assert _dep(wired, "fill-zones-sequential")[1]["s3_region"] == "eu-west-1"


def test_default_run_rejects_unseeded_work_zone(wired, monkeypatch):
    """A partially-seeded store (default zones=None) fails BEFORE any ingest when the
    work list includes an unseeded zone — not after an expensive per-cell ingest.
    """
    monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [("02N", 2025)])  # 02N not in status.zones
    with pytest.raises(ValueError, match="not seeded"):
        _campaign()
    assert wired["arun"] == []  # nothing dispatched


def test_rejects_zero_parallel_ingest(wired):
    with pytest.raises(ValueError, match="max_parallel_ingest"):
        _campaign(max_parallel_ingest=0)


def test_off_axis_year_rejected(wired, monkeypatch):
    """An off-axis year fails up front, before any ingest/fill is dispatched."""
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: False)
    with pytest.raises(ValueError, match="not on the store's pre-allocated axis"):
        _campaign(years=(2026,))
    assert wired["arun"] == []  # nothing dispatched


def test_unseeded_explicit_zone_rejected(wired):
    """An explicit zones= entry that isn't seeded fails before dispatch."""
    with pytest.raises(ValueError, match="not seeded"):
        _campaign(zones=["02N"])
    assert wired["arun"] == []


def test_duplicate_years_dispatch_once(wired):
    """The dispatch loop dedupes years — years=(2025, 2025) runs the cell once."""
    _campaign(years=(2025, 2025))
    assert [p["cells"] for _, p in wired["arun"]] == [[["33N", 2025]]]


def test_chained_fill_dispatches_one_run_per_year(wired):
    """ONE fill-zones-sequential run per year under the year barrier: no driver-side ingest
    (the child's look-ahead owns it), no driver-side mosaic cleanup, cells passed as
    (zone, year) pairs.
    """
    # overlap_years is the DEFAULT now, and this test is about the per-year dispatch shape, so the
    # barrier is asked for explicitly rather than relied on.
    result = asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", overlap_years=False))
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zones-sequential/fill-zones-sequential"]
    params = wired["arun"][0][1]
    assert _dispatched_zones(params) == ["33N"] and params["cells"] == [["33N", 2025]]
    # One cell dispatched, so the cluster list is one long and it carries the whole ingest bound as
    # its window: cap 60 means 59 cells BEYOND the current one.
    assert params["ingest"] is True and 1 + params["look_ahead"] == 60
    assert params["ingest_deployment"] == "ingest-zone-year/ingest-zone-year"
    # 32 assembly workers, asked for EXPLICITLY. The library default is 16, safe on the 64 GiB
    # inference family that every other caller runs on; only this deployment is the 244 GiB
    # `assembly_large` family that can hold a 32-process pool. See AssemblyConfig.
    assert params["n_assembly_workers"] == mod.ASSEMBLY_WORKERS_ON_THE_LARGE_RUNNER == 32
    # Mosaic lifecycle belongs to the child in this mode.
    assert wired["deletes"] == []
    assert result["dispatched"] == 1


def test_chained_fill_forwards_ingest_false(wired):
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False))
    assert wired["arun"][0][1]["ingest"] is False


def test_chained_fills_shard_by_live_tiles(wired, monkeypatch):
    """max_parallel_clusters > 1 = that many chained clusters:
    zones are LPT-partitioned by WORK (tiles weighted by the latitude band's observation
    count) and each cluster's child divides the global ingest look-ahead bound.
    """
    status = SimpleNamespace(zones={"33N": (), "34N": (), "35N": ()}, has=lambda z, y: False)
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    monkeypatch.setattr(mod, "campaign_work_list", _shrinking([("33N", 2025), ("34N", 2025), ("35N", 2025)], wired))
    counts = {"33N": 500, "34N": 300, "35N": 250}
    monkeypatch.setattr(mod, "zone_work_weight", lambda mask, zone, **k: float(counts[zone]))

    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS,
            ami_ssm_name="ami",
            max_parallel_clusters=2,
            max_parallel_ingest=6,
        )
    )
    assert [d for d, _ in wired["arun"]] == ["fill-zones-sequential/fill-zones-sequential"] * 2
    clusters = [_dispatched_zones(p) for _, p in wired["arun"]]
    # LPT: 500 alone; 300+250 together — balanced totals (500 vs 550).
    assert sorted(map(sorted, clusters)) == [["33N"], ["34N", "35N"]]
    # The global ingest bound is divided across clusters: 6 over 2 clusters = 3 each,
    # and a cluster's width is 1 + look_ahead.
    assert all(1 + p["look_ahead"] == 3 for _, p in wired["arun"])


class TestIngestBoundAcrossClusters:
    """A cluster's ingest concurrency is ``1 + look_ahead``, because ``look_ahead``
    counts the cells kept in flight BEYOND the current one — that is the contract
    ``fill_zones_sequential`` states and sizes its driver by (``max_parallel=1 +
    look_ahead``). So the fleet-wide figure is ``(1 + look_ahead) * clusters``.

    These assert the WIDTH rather than the raw parameter: pinning the parameter is
    what let the two ends of this contract disagree, with the campaign handing over a
    cluster's whole share and the fill then adding one to it.
    """

    @staticmethod
    def _look_ahead(wired, monkeypatch, *, zones: int, clusters: int, ingest_cap: int) -> int:
        names = [f"{i + 1:02d}N" for i in range(zones)]
        status = SimpleNamespace(zones=dict.fromkeys(names, ()), has=lambda z, y: False)
        monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
        monkeypatch.setattr(mod, "campaign_work_list", _shrinking([(z, 2025) for z in names], wired))
        monkeypatch.setattr(mod, "zone_work_weight", lambda mask, zone, **k: 100.0)
        wired["arun"].clear()
        wired["landed"].clear()
        asyncio.run(
            mod.run_global_campaign.fn(
                paths=_PATHS,
                ami_ssm_name="ami",
                max_parallel_clusters=clusters,
                max_parallel_ingest=ingest_cap,
            )
        )
        # The cluster's ingest WIDTH, which is what the cap is expressed in.
        return 1 + wired["arun"][0][1]["look_ahead"]

    def test_an_exact_division_delivers_the_requested_width(self, wired, monkeypatch):
        assert self._look_ahead(wired, monkeypatch, zones=4, clusters=4, ingest_cap=8) == 2

    def test_the_default_split_is_five_zones_per_cluster(self, wired, monkeypatch):
        """8 clusters over a 40-zone cap: the shape the campaign actually runs."""
        assert self._look_ahead(wired, monkeypatch, zones=8, clusters=8, ingest_cap=40) == 5

    def test_a_remainder_rounds_up_rather_than_under_delivering(self, wired, monkeypatch):
        """Flooring would hand a cluster less ingest width than was asked for, with
        nothing in the logs saying so. The global gate is the hard ceiling, so
        rounding up overshoots the intent and never the actual cap.
        """
        assert self._look_ahead(wired, monkeypatch, zones=4, clusters=4, ingest_cap=6) == 2

    def test_a_cap_below_the_cluster_count_still_gives_every_cluster_one(self, wired, monkeypatch):
        """A cluster with a window of zero could never start a zone, so the real
        floor is one per cluster however low the cap is set. The gate still holds
        the fleet to the cap; the clusters just queue on it.
        """
        assert self._look_ahead(wired, monkeypatch, zones=4, clusters=4, ingest_cap=1) == 1


def test_every_cluster_opens_on_one_of_the_densest_zones(wired, monkeypatch):
    """The property the GPU-start rule depends on.

    A cluster requests its fleet as soon as its FIRST zone has ingested, which is
    only safe if that zone is big enough to keep the fleet busy while the rest of
    the window ingests behind it. The LPT assignment gives that for free — every
    total starts at zero, so the N densest zones go one to each of the N clusters —
    but it falls out of the assignment order rather than being an explicit step, so
    it is pinned here.
    """
    counts = {"01N": 900, "02N": 800, "03N": 700, "04N": 50, "05N": 40, "06N": 30}
    status = SimpleNamespace(zones=dict.fromkeys(counts, ()), has=lambda z, y: False)
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    monkeypatch.setattr(mod, "campaign_work_list", _shrinking([(z, 2025) for z in counts], wired))
    monkeypatch.setattr(mod, "zone_work_weight", lambda mask, zone, **k: float(counts[zone]))

    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=3))
    dispatched = [_dispatched_zones(p) for _, p in wired["arun"]]
    assert {zs[0] for zs in dispatched} == {"01N", "02N", "03N"}, dispatched
    # ...and each cluster tapers from dense to sparse rather than the reverse.
    for zs in dispatched:
        tiles = [counts[z] for z in zs]
        assert tiles == sorted(tiles, reverse=True), zs


def test_sequential_single_shard_reads_no_tile_counts(wired, monkeypatch):
    """One work zone → one cluster → the partitioner must not read the mask."""

    def boom(*a, **k):
        raise AssertionError("zone weights must not be read for a single cluster")

    monkeypatch.setattr(mod, "zone_work_weight", boom)
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    assert len(wired["arun"]) == 1


def test_partition_zero_weights_zones_with_nothing_left_to_do(monkeypatch):
    """Retag-only cells cost the child no GPU time, so they must not skew the
    LPT balance — and their mask reads must be skipped entirely.
    """
    counts = {"01N": 100, "02N": 90, "03N": 80}

    def count(mask, zone, **k):
        if zone not in counts:
            raise AssertionError(f"zone weight read for a zone with no pending years: {zone}")
        return float(counts[zone])

    monkeypatch.setattr(mod, "zone_work_weight", count)
    clusters = mod._partition_by_live_tiles(
        ["01N", "02N", "03N", "04N"],
        2,
        land_mask_path="mask",
        pending_years={"01N": 1, "02N": 1, "03N": 1, "04N": 0},
    )
    # LPT over {100, 90, 80}: [01N] vs [02N, 03N]; 04N rides along at zero cost.
    assert sorted(map(sorted, clusters)) == [["01N", "04N"], ["02N", "03N"]]


def test_partition_weights_a_zone_by_the_years_it_carries(monkeypatch):
    """A cluster receives every pending year of the zones it owns, so a zone missing
    five years is five times the work of one missing one.

    Weighed once, the two looked identical — irrelevant while every zone owed the same
    single year, and wrong as soon as overlap_years let a batch span several or a repair
    run left uneven gaps. One cluster then drains the extra years while the rest idle,
    and its finish time is the campaign's.
    """
    monkeypatch.setattr(mod, "zone_work_weight", lambda mask, zone, **k: 10.0)
    clusters = mod._partition_by_live_tiles(
        ["01N", "02N", "03N"],
        2,
        land_mask_path="mask",
        pending_years={"01N": 4, "02N": 2, "03N": 2},
    )
    # 01N alone (40) balances the other two together (20 + 20). Unweighted by years,
    # all three score 10 and the split would be 2-vs-1 the other way round.
    assert sorted(map(sorted, clusters)) == [["01N"], ["02N", "03N"]]


# ── branch-scoped deployment routing ──


def _dep(rec: dict, needle: str) -> tuple[str, dict]:
    """The (dep_ref, params) of the dispatched deployment whose ref contains needle."""
    return next((d, p) for d, p in rec["arun"] if needle in d)


def test_dpl_routes_only_when_branch_is_a_real_slug() -> None:
    """The routing primitive: a blank branch (None/empty/whitespace) is a no-op;
    a real slug is appended to the deployment name.
    """
    assert mod._dpl("flow/name", None) == "flow/name"
    assert mod._dpl("flow/name", "") == "flow/name"
    assert mod._dpl("flow/name", "   ") == "flow/name"
    assert mod._dpl("flow/name", "x") == "flow/name-x"


def _ingest_deployments(branch: str | None) -> dict:
    """The S1/S2 grandchild refs one ingest dispatch carries, as the chained fill builds it."""
    return mod._ingest_dispatch_params(
        "33N",
        2025,
        paths=_PATHS,
        mask_name="global",
        s1_orbit="both",
        ingest_settings=mod.IngestSettings(),
        allow_partial_window=False,
        branch=branch,
    )["deployments"]


@pytest.mark.parametrize(("branch", "suffix"), [(None, ""), ("global-tessera", "-global-tessera")])
def test_branch_routes_every_derived_ref(wired, branch, suffix):
    """No branch keeps every child ref — direct AND grandchild — at its unsuffixed production
    default. A slug routes the fill, the ingest, AND the S1/S2 grandchildren that
    ingest_zone_year dispatches (the ones that used to 404).
    """
    _campaign(branch=branch)
    dep, params = _dep(wired, "fill-zones-sequential")
    assert dep == f"fill-zones-sequential/fill-zones-sequential{suffix}"
    assert params["ingest_deployment"] == f"ingest-zone-year/ingest-zone-year{suffix}"
    assert _ingest_deployments(params["branch"]) == {
        "ingest_s1_roi_sar": f"ingest_s1_roi_sar/ingest-s1-roi-sar{suffix}",
        "ingest_s2_roi_reflectance": f"ingest_s2_roi_reflectance/ingest-s2-roi-reflectance{suffix}",
    }


def test_explicit_direct_ref_is_verbatim_but_grandchildren_still_route(wired):
    """Precedence: an explicit fill/ingest ref is used verbatim (never re-suffixed),
    yet the S1/S2 grandchildren still follow `branch` — they have no explicit knob.
    """
    _campaign(branch="b", chained_fill_deployment="custom/fill", ingest_deployment="custom/ingest")
    dep, params = _dep(wired, "custom/fill")
    assert dep == "custom/fill"  # not custom/fill-b
    assert params["ingest_deployment"] == "custom/ingest"
    assert _ingest_deployments(params["branch"])["ingest_s1_roi_sar"].endswith("-b")


def test_branch_routes_fill_when_ingest_disabled(wired):
    """ingest=False: the fill ref is still branch-routed."""
    _campaign(ingest=False, branch="b")
    assert [d for d, _ in wired["arun"]] == ["fill-zones-sequential/fill-zones-sequential-b"]


@pytest.mark.parametrize("flag", ["allow_model_mismatch", "allow_s2_only"])
def test_operator_overrides_reach_the_fill(wired, flag: str):
    """A campaign-level flag has to reach the gate that fires, which is in the child.

    The campaign checks the seeded model once before dispatching, but the child re-checks it;
    without forwarding, it falls back to its default and rejects the same store after the
    cell's ingest has been paid for. ``allow_s2_only`` decides which pixels get embeddings.
    """
    _campaign(**{flag: True})
    assert _dep(wired, "fill-zones-sequential")[1][flag] is True


@pytest.mark.parametrize("case", ["retag-only", "all-ocean"])
def test_a_cluster_that_starts_no_ray_neither_resolves_nor_pins_the_ami(wired, monkeypatch, case):
    """A cluster whose cells never provision Ray must not force an SSM read.

    ``_ami_id`` is deliberately lazy so a pure tag-repair or all-ocean campaign makes no AWS
    call; pinning the AMI into every dispatch would undo that and turn a missing parameter —
    or a role without SSM access — into a failure on exactly the cheap recovery path that has
    no use for the answer.
    """

    def _boom(*a, **k):
        raise AssertionError("a cluster that starts no Ray must not resolve the AMI")

    monkeypatch.setattr(mod, "_resolve_ami_id", _boom)
    if case == "retag-only":
        status = SimpleNamespace(zones={"33N": (2025,)}, has=lambda z, y: True)
        monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    else:
        monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: False)
    _campaign()
    fill = _dep(wired, "fill-zones-sequential")[1]
    assert fill["ami_id"] is None
    # The SSM pointer still travels, so the fill's own fallback stays available.
    assert fill["ami_ssm_name"] == "ami"


def test_every_dispatched_child_carries_the_campaign_tag(wired):
    """The tag is the only thing the terminal hooks can work from.

    Prefect runs them in a fresh process after killing the flow, so nothing in
    memory records what was dispatched — an untagged child is an unsweepable one.
    """
    with patch.object(mod.flow_run_ctx, "id", "run-abc"):
        _campaign()
    assert wired["tags"], "no children dispatched"
    assert all(t == ["campaign:run-abc"] for t in wired["tags"]), wired["tags"]


def test_children_go_untagged_outside_a_flow_run(wired):
    """A direct .fn() call has no run id, and dispatches nothing worth sweeping."""
    _campaign()
    assert all(t is None for t in wired["tags"])


def test_campaign_registers_child_cancellation_on_cancel_and_crash():
    """Cancelling the campaign must stop the ingests and fills it started.

    Each dispatch is an INDEPENDENT run: without this the parent dies and its
    children keep writing, with their Dask and Ray fleets still billing.
    """
    flow = mod.run_global_campaign
    assert mod._cancel_children_on_cancellation in flow.on_cancellation_hooks
    assert mod._cancel_children_on_cancellation in flow.on_crashed_hooks


def test_model_gate_runs_before_any_dispatch(wired, monkeypatch):
    """A store seeded for another encoder must cost a metadata read, not a mosaic.

    Each fill re-checks this, but the campaign dispatches ingest first and runs cells
    concurrently — so by the time the first fill fails there is a multi-terabyte
    mosaic per in-flight zone, and failed cells retain theirs to keep a resume cheap.
    """

    def _reject(*a, **k):
        raise ValueError("store was seeded for encoder https://x/OLD")

    monkeypatch.setattr(mod, "_assert_seeded_model_matches", _reject)
    with pytest.raises(ValueError, match="seeded for encoder"):
        _campaign()
    assert wired["arun"] == [], "nothing may be dispatched once the gate fails"


def test_model_gate_override_lets_the_campaign_proceed(wired, monkeypatch):
    """The escape hatch is threaded through rather than reimplemented."""
    seen: dict = {}
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: seen.update(k))
    _campaign(allow_model_mismatch=True)
    assert seen["allow_model_mismatch"] is True


class TestFailedZonesAreRetried:
    """One zone must not cost the campaign the rest of the year, nor later years.

    Interruptions are expected here — the orphan sweeper cancels child runs by
    design — and an interrupted mosaic is resumed rather than rebuilt, so a
    re-dispatch is cheap. Each round re-reads the STORE for what is still missing,
    which is also what makes "progress" measurable rather than guessed.
    """

    @staticmethod
    def _wire(monkeypatch, wired, zones, *, complete_after):
        """Zones that become complete once ``complete_after`` rounds have run."""
        rounds = {"n": 0}
        state: set[str] = set()

        def status_now(*_a, **_k):
            return SimpleNamespace(zones=dict.fromkeys(zones, ()), has=lambda z, y: z in state)

        monkeypatch.setattr(mod, "campaign_status", status_now)
        monkeypatch.setattr(
            mod, "campaign_work_list", lambda st, *a, **k: [(z, 2025) for z in zones if not st.has(z, 2025)]
        )

        async def arun(dep, parameters=None, tags=None):
            wired["arun"].append((dep, parameters))
            if "fill-zones-sequential" in dep:
                rounds["n"] += 1
                state.update(complete_after.get(rounds["n"], ()))
                if any(z not in state for z, _y in parameters["cells"]):
                    raise RuntimeError("cluster failed")
            wired["land"](parameters)
            return _completed_run()

        monkeypatch.setattr(mod, "arun_deployment", arun)
        return rounds

    def test_a_failed_zone_is_re_dispatched(self, wired, monkeypatch):
        """Round one leaves 02N unfilled; round two lands it and the campaign passes."""
        rounds = self._wire(monkeypatch, wired, ["01N", "02N"], complete_after={1: {"01N"}, 2: {"01N", "02N"}})
        result = asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1))
        assert rounds["n"] == 2, "the failed cell was never re-dispatched"
        assert result["dispatched"] >= 1

    def test_a_retry_targets_only_what_is_still_missing(self, wired, monkeypatch):
        """A cluster can fail having landed most of its zones; those must not be
        re-dispatched, or a retry would repeat hours of finished work.
        """
        self._wire(
            monkeypatch,
            wired,
            ["01N", "02N", "03N"],
            complete_after={1: {"01N", "02N"}, 2: {"01N", "02N", "03N"}},
        )
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1))
        dispatched = [_dispatched_zones(p) for d, p in wired["arun"] if "fill-zones-sequential" in d]
        assert dispatched[0] == ["01N", "02N", "03N"]
        assert dispatched[1] == ["03N"], f"second round re-dispatched landed zones: {dispatched[1]}"

    def test_a_round_that_makes_no_progress_stops_the_retries(self, wired, monkeypatch):
        """What a DETERMINISTIC failure looks like from the campaign — a coverage
        gate, a fingerprint mismatch. It wants a human, not another cluster, so the
        retries stop rather than burning a fleet per attempt.
        """
        rounds = self._wire(monkeypatch, wired, ["01N", "02N"], complete_after={1: {"01N"}})  # 02N never lands
        summary = asyncio.run(
            mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1, max_dispatch_rounds=5)
        )
        assert summary["unfilled"] == {2025: ["02N"]}
        # One attempt, one retry that achieves nothing, then stop — the minimum
        # possible without guessing at the child's failure type.
        assert rounds["n"] == 2, f"kept retrying a deterministic failure: {rounds['n']} rounds"

    def test_the_campaign_reports_every_unfilled_cell_and_still_succeeds(self, wired, monkeypatch, caplog):
        """A few cells failing every attempt is an expected outcome at this scale, not a failed
        campaign — so the run SUCCEEDS and the list is the product.

        Raising would say the opposite: it would mark a thousand landed cells a failure because
        three did not land, fire the driver-stopped alert (whose meaning is "nothing is being
        filled any more"), and leave an operator reading a stack trace to find a list.
        """
        self._wire(monkeypatch, wired, ["01N", "02N"], complete_after={})
        with caplog.at_level(logging.WARNING):
            summary = asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1))
        # The list is DATA in the summary, not prose to be re-parsed out of a message.
        assert summary["unfilled"] == {2025: ["01N", "02N"]}
        # And it is loud: a banner, the whole list, and what a re-run would do.
        logged = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert "UNFILLED CELL(S)" in logged
        assert "01N" in logged and "02N" in logged
        assert "resumes from here" in logged

    def test_a_campaign_that_lands_everything_still_carries_the_key(self, wired, monkeypatch):
        """Empty rather than absent: a caller that tests for the key must not have to know
        which of two shapes it gets.
        """
        self._wire(monkeypatch, wired, ["01N"], complete_after={1: {"01N"}})
        summary = asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1))
        assert summary["unfilled"] == {}

    def test_retries_are_disabled_at_one_attempt(self, wired, monkeypatch):
        rounds = self._wire(monkeypatch, wired, ["01N"], complete_after={})
        summary = asyncio.run(
            mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1, max_dispatch_rounds=1)
        )
        assert rounds["n"] == 1
        assert summary["unfilled"] == {2025: ["01N"]}

    def test_zero_attempts_is_rejected(self, wired):
        with pytest.raises(ValueError, match="max_dispatch_rounds"):
            asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_dispatch_rounds=0))


# --- overlap_years: dropping the year barrier -------------------------------------------


def _multi_year(monkeypatch, zones, years, wired):
    """Wire a work list of every (zone, year) pair, with nothing yet complete."""
    status = SimpleNamespace(zones=dict.fromkeys(zones, ()), has=lambda z, y: False, years=tuple(years))
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    monkeypatch.setattr(mod, "campaign_work_list", _shrinking([(z, y) for y in years for z in zones], wired))
    monkeypatch.setattr(mod, "zone_work_weight", lambda mask, zone, **k: 100.0)
    return status


def test_overlapping_years_is_now_the_default(wired, monkeypatch):
    """One batch covering every requested year, without anyone asking for it.

    This assertion is inverted from what it was, deliberately. It used to pin year-serial as the
    default, on the reasoning that year-serial was the path that had actually been run and a silent
    flip would change the campaign's shape. That premise expired: `overlap_years` was certified on
    six cells each carrying both radar orbits, including a same-zone year rollover inside one
    cluster, and `context_docs/campaign/campaign-plan.md` §3 makes it the campaign's shape.

    The guard's PURPOSE is unchanged and is why this still exists rather than being deleted — the
    default must not move without a test saying so. Overlapping years is what makes the 60-zone
    ingest cap reachable; with the barrier in place the ceiling is 45 regardless of the cap.
    """
    _multi_year(monkeypatch, ["01N", "02N"], [2025, 2024], wired)
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1))
    # One dispatch carrying BOTH years, so year N+1's ingest overlaps year N's inference.
    years_per_dispatch = [{y for _z, y in p["cells"]} for _, p in wired["arun"]]
    assert years_per_dispatch == [{2025, 2024}], years_per_dispatch


def test_the_year_barrier_is_still_reachable_when_asked_for(wired, monkeypatch):
    """The complement, so the test above cannot be satisfied by the option disappearing.

    A repair pass over one year, or a diagnosis that wants years kept apart, still needs this.
    """
    _multi_year(monkeypatch, ["01N", "02N"], [2025, 2024], wired)
    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS,
            ami_ssm_name="ami",
            max_parallel_clusters=1,
            overlap_years=False,
        )
    )
    years_per_dispatch = [{y for _z, y in p["cells"]} for _, p in wired["arun"]]
    assert years_per_dispatch == [{2025}, {2024}], years_per_dispatch


def test_overlap_years_dispatches_every_year_in_one_batch(wired, monkeypatch):
    """The barrier gone: one round covering all requested years, so year N+1's ingest
    can overlap year N's inference instead of waiting for it.
    """
    _multi_year(monkeypatch, ["01N", "02N"], [2025, 2024], wired)
    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS,
            ami_ssm_name="ami",
            max_parallel_clusters=1,
            overlap_years=True,
        )
    )
    assert len(wired["arun"]) == 1, [d for d, _ in wired["arun"]]
    cells = {(z, y) for z, y in wired["arun"][0][1]["cells"]}
    assert cells == {("01N", 2025), ("01N", 2024), ("02N", 2025), ("02N", 2024)}


def test_overlap_years_keeps_every_year_of_a_zone_in_one_cluster(wired, monkeypatch):
    """The safety property the whole design rests on.

    Two clusters writing two years of the SAME zone would contend on that group's
    attributes. The partition is over ZONES rather than cells precisely so a zone's every
    year lands in one cluster, where the runner's single trailing-assembly thread
    serializes them. If the partition were ever over cells, this fails.
    """
    _multi_year(monkeypatch, ["01N", "02N", "03N", "04N"], [2025, 2024, 2023], wired)
    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS,
            ami_ssm_name="ami",
            max_parallel_clusters=4,
            overlap_years=True,
        )
    )
    owner: dict[str, int] = {}
    for i, (_d, p) in enumerate(wired["arun"]):
        for z, _y in p["cells"]:
            assert owner.setdefault(z, i) == i, f"zone {z} was split across clusters"
    # Every zone placed, and each cluster carries all three years of the zones it owns.
    assert set(owner) == {"01N", "02N", "03N", "04N"}
    for _d, p in wired["arun"]:
        by_zone: dict[str, set[int]] = {}
        for z, y in p["cells"]:
            by_zone.setdefault(z, set()).add(y)
        assert all(yrs == {2025, 2024, 2023} for yrs in by_zone.values()), by_zone


def test_a_resumed_cluster_is_credited_only_with_the_years_it_owns(wired, monkeypatch):
    """Attribution follows the PARTITION, not the batch.

    A fresh campaign hands every cluster every year, which is what made crediting
    `batch_years` look right. A RESUMED one partitions over what is still missing, so a
    cluster can hold zones each short a DIFFERENT single year — here 01N wants only 2024
    and 02N only 2025, one zone per cluster. Crediting the batch gave each run both years:
    `dispatched` came out 4 for 2 runs, and each year claimed a run that never touched it.
    """
    status = SimpleNamespace(zones=dict.fromkeys(["01N", "02N"], ()), has=lambda z, y: False, years=(2025, 2024))
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    monkeypatch.setattr(mod, "campaign_work_list", _shrinking([("01N", 2024), ("02N", 2025)], wired))
    monkeypatch.setattr(mod, "zone_work_weight", lambda mask, zone, **k: 100.0)
    summary = asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS,
            ami_ssm_name="ami",
            max_parallel_clusters=2,
            overlap_years=True,
        )
    )
    # Two clusters, one cell each, so exactly two runs — not two runs counted twice.
    assert summary["dispatched"] == 2, summary["runs_by_year"]
    assert {y: len(runs) for y, runs in summary["runs_by_year"].items()} == {2024: 1, 2025: 1}


def test_overlap_years_still_reports_unfilled_cells_per_year(wired, monkeypatch):
    """The failure report stays YEAR-keyed, so an operator reads it the same way.

    A batch spanning years must not collapse into one undifferentiated list — the whole
    point of the report is telling you which zone-years to look at.
    """
    _multi_year(monkeypatch, ["01N"], [2025, 2024], wired)

    async def arun(dep, parameters=None, tags=None):
        wired["arun"].append((dep, parameters))
        if "fill-zones-sequential" in dep:
            raise RuntimeError("cluster failed")
        return _completed_run()

    monkeypatch.setattr(mod, "arun_deployment", arun)
    summary = asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS,
            ami_ssm_name="ami",
            max_parallel_clusters=1,
            overlap_years=True,
        )
    )
    # Both years accounted for SEPARATELY, each with its zone: one run spanning years must not
    # collapse into one entry, because the second wave is dispatched per cell.
    assert summary["unfilled"] == {2024: ["01N"], 2025: ["01N"]}, summary["unfilled"]


def test_production_staging_identity_is_unchanged_by_the_tarball_term(monkeypatch):
    """A baked-AMI deploy passes no code_bucket, so the term is empty and prod is untouched.

    The point of adding it at all is that it costs production nothing: `code_bucket=None`
    means no tarball overlays the image, there is nothing to fingerprint, and the identity
    stays exactly the source hash the 2026-07-30 narrowing settled on.
    """
    called: list = []
    monkeypatch.setattr(mod, "source_tarball_identity", lambda *a, **k: called.append(a) or "x", raising=False)
    assert mod._resolve_tarball_identity(None, "", None) == ""
    assert not called, "no bucket means no S3 call at all, not an empty answer from one"


def test_the_dev_overlay_folds_the_tarball_etag_into_the_staging_identity(monkeypatch):
    """On the dev path the workers run a tarball the flow runner never imports.

    Replacing it changes what executes while the local source hash stays put, so a retry
    would resume tiles staged by the old code and add new ones from the new — mixing two
    code versions in one write-once zone-year. The ETag is what makes the fingerprint move.
    """
    import tessera_embeddings.providers.aws.ray as ray_mod

    monkeypatch.setattr(ray_mod, "source_tarball_identity", lambda b, s, r: f"tarball=etag-{b}{s}")
    assert mod._resolve_tarball_identity("bkt", "-branch", None) == "tarball=etag-bkt-branch"


# --- immediate_refill: replacing a settled fill without waiting for the round ------------


def _zones_of(params: dict) -> tuple[str, ...]:
    """The zones one dispatch covers, sorted — multi-year safe, unlike `_dispatched_zones`."""
    return tuple(sorted({z for z, _y in _dispatched_cells(params)}))


def _run_in(state: str) -> SimpleNamespace:
    """A settled child run in the named terminal state."""
    return SimpleNamespace(id=f"r-{state}", state=SimpleNamespace(type=getattr(StateType, state), name=state.title()))


class _Round:
    """A round in which one cluster settles while another is HELD open.

    The immediate path only fires while a sibling is still running, so every test here —
    the refusals included — needs one; otherwise a refusal would pass for the wrong
    reason. `hold` names the zone whose dispatch parks until the driver has decided,
    which is what makes "before its sibling settled" an observable fact rather than a
    timing hope.
    """

    def __init__(self, wired, *, ends: dict[str, str], hold: str):
        self.wired = wired
        self.ends = ends  # a dispatch's first zone -> terminal state name, or "raise"
        self.hold = hold
        self.release = asyncio.Event()
        #: ("dispatch" | "settle", zones) in the order it happened. The ORDER is the subject.
        self.timeline: list[tuple[str, tuple[str, ...]]] = []
        self.held: set[str] = set()
        #: Zones a dispatch found already held. RECORDED rather than raised: the driver
        #: catches a dispatch's exception and records it as a cluster failure, so an
        #: assertion thrown in here would be swallowed and the test would pass.
        self.overlaps: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def dispatches(self) -> list[tuple[str, ...]]:
        return [zs for kind, zs in self.timeline if kind == "dispatch"]

    def dispatched_before(self, sibling: tuple[str, ...]) -> list[tuple[str, ...]]:
        """Every dispatch that happened before ``sibling`` settled."""
        cut = self.timeline.index(("settle", sibling))
        return [zs for kind, zs in self.timeline[:cut] if kind == "dispatch"]

    async def arun(self, dep, parameters=None, tags=None):
        zones = _zones_of(parameters or {})
        self.wired["arun"].append((dep, parameters))
        if "fill-zones-sequential" not in dep:
            self.wired["land"](parameters)
            return _completed_run()
        # DISJOINTNESS, checked on entry and therefore across the whole run: two
        # dispatches must never hold one zone at the same time, whatever order they
        # settle in and whether or not a replacement was issued.
        if self.held & set(zones):
            self.overlaps.append((zones, tuple(sorted(self.held & set(zones)))))
        self.held |= set(zones)
        self.timeline.append(("dispatch", zones))
        try:
            if zones and zones[0] == self.hold:
                await self.release.wait()
                # RE-ARM. The hold has to work in every round, not just the first: a
                # bound that spans rounds cannot be observed if only round one has a
                # sibling to hold.
                self.release.clear()
            ends = self.ends.get(zones[0], "COMPLETED") if zones else "COMPLETED"
            if ends == "raise":
                raise RuntimeError("dispatch failed")
            if ends == "COMPLETED":
                self.wired["land"](parameters)
            return _run_in(ends)
        finally:
            self.timeline.append(("settle", zones))
            self.held -= set(zones)

    async def drive(self, **kwargs):
        """Run the campaign, releasing the held cluster once the driver has decided.

        The release waits a budget of event-loop TURNS rather than a delay: everything the
        decision touches is stubbed, so it needs a handful of turns and no wall clock. A
        driver that needed more would fail these tests rather than hang. Pumped in a loop
        rather than once, so a held cluster works in every round.
        """
        task = asyncio.ensure_future(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", **kwargs))
        for _ in range(50):  # rounds, generously; the assert below catches a real hang
            if task.done():
                break
            for _ in range(200):
                await asyncio.sleep(0)
                if task.done():
                    break
            self.release.set()
        assert task.done() or not self.timeline, "the campaign did not settle within the turn budget"
        return await task


@contextmanager
def caplog_at_info():
    """Collect the driver's log messages at INFO, as a list of strings.

    A plain handler rather than the `caplog` fixture, because these tests drive the flow
    inside `asyncio.run` from a helper and want the records regardless of propagation.
    """
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("test-campaign")
    handler = _Collect()
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _round(monkeypatch, wired, round_, zones=("01N", "02N"), years=(2025,)):
    """Wire a pending (zone, year) grid and drive dispatch through ``round_``."""
    cells = [(z, y) for y in years for z in zones]
    status = SimpleNamespace(zones=dict.fromkeys(zones, ()), has=lambda z, y: False)
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    monkeypatch.setattr(mod, "campaign_work_list", _shrinking(cells, wired))
    monkeypatch.setattr(mod, "zone_work_weight", lambda mask, zone, **k: 100.0)
    monkeypatch.setattr(mod, "arun_deployment", round_.arun)
    # The settling wait is real seconds in production. Zeroed here by default; the tests
    # that are ABOUT the wait put it back.
    monkeypatch.setattr(mod, "_SETTLE_DELAY_S", 0)
    return round_


@pytest.mark.asyncio
async def test_still_running_ignores_a_task_that_finished_but_was_not_harvested():
    """Not `bool(the map)`. The drain loop harvests tasks when it wakes, so a task can be
    finished and still present — and a truthiness test would then report a round as open
    when everything in it had stopped, which is the difference between a replacement that
    saves wall clock and one that only spends a fleet.

    Tested directly because the window it guards is a scheduling race that cannot be forced
    from the flow; asserting the helper's contract is what makes the distinction real rather
    than assumed.
    """
    done: asyncio.Future = asyncio.get_running_loop().create_future()
    done.set_result(None)
    pending: asyncio.Future = asyncio.get_running_loop().create_future()
    try:
        assert mod._still_running({pending: 0}) is True
        assert mod._still_running({done: 0}) is False, "a finished task read as still running"
        assert mod._still_running({done: 0, pending: 1}) is True
        assert mod._still_running({}) is False
    finally:
        pending.cancel()


class TestImmediateRefill:
    """A settled fill's roster must not wait on the slowest cluster in its round.

    The dispatch round is a barrier: it collects every cluster's outcome before re-reading
    the store. A cluster owns a roster of zones, so one dying early costs that whole roster
    the wait for its slowest sibling. These pin when the driver may skip that wait and —
    more importantly — when it may not.
    """

    @pytest.mark.parametrize("flag,replaced_before_sibling", [(False, False), (True, True)])
    def test_the_flag_decides_whether_a_replacement_waits_for_the_round(
        self, wired, monkeypatch, flag, replaced_before_sibling
    ):
        """The whole point of the change, and the whole safety argument for merging it.

        Parametrised as a PAIR deliberately. A "the default is unchanged" assertion on its
        own passes whether or not the change is present, which makes it worse than no test;
        pinned against the flag-on case it cannot.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))
        asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=flag))
        before = r.dispatched_before(("02N",))
        assert (before == [("01N",), ("02N",), ("01N",)]) is replaced_before_sibling, before
        # Either way the cell is re-dispatched — the flag decides WHEN, not whether. The extra
        # attempt is the flag's whole cost, and it is exactly one: the round's own budget is
        # untouched, so a cell that keeps failing gains one fleet, not a loop of them.
        assert r.dispatches().count(("01N",)) == 2 + int(replaced_before_sibling)

    def test_the_replacement_covers_only_the_dead_clusters_still_missing_cells(self, wired, monkeypatch):
        """Its predecessor's zones, and only the cells the STORE still lacks.

        Never the child's failure list: a child can report success on cells it never
        attempted, so the store is the only thing that may decide — exactly as the round
        decides.
        """
        r = _round(
            monkeypatch,
            wired,
            _Round(wired, ends={"01N": "FAILED"}, hold="02N"),
            zones=("01N", "02N"),
            years=(2025, 2024),
        )
        wired["landed"].add(("01N", 2024))  # one of the dead cluster's cells did land
        asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, overlap_years=True))
        # Located by happening BEFORE the sibling settled, so this cannot pass instead on a
        # NEXT-ROUND re-dispatch that happens to have the same scope. Asserted on the shape
        # rather than the sequence: the partition decides which cluster is dealt first.
        before = r.dispatched_before(("02N",))
        assert len(before) == 3 and before.count(("01N",)) == 2, before
        refill = [p for d, p in wired["arun"] if "fill-zones-sequential" in d][2]
        assert sorted(map(tuple, refill["cells"])) == [("01N", 2025)]

    def test_no_two_dispatches_ever_hold_one_zone_at_once(self, wired, monkeypatch):
        """The two-writer property, asserted across the whole run rather than at a moment.

        Nothing locks a zone: the zone partition IS the guarantee, and an immediate
        replacement is the one thing that could dispatch a zone while something else still
        held it. ``_Round.arun`` checks disjointness on every entry, so this fails on the
        first overlap wherever in the run it occurs. The second assertion is the same
        property stated over the whole run: a zone is only ever dispatched as part of one
        shard, so no replacement ever widened past the roster it inherited.
        """
        r = _round(
            monkeypatch,
            wired,
            _Round(wired, ends={"01N": "FAILED", "03N": "FAILED"}, hold="02N"),
            zones=("01N", "02N", "03N", "04N"),
            years=(2025, 2024),
        )
        asyncio.run(r.drive(max_parallel_clusters=4, immediate_refill=True, overlap_years=True))
        assert r.overlaps == [], f"a zone was dispatched while another dispatch still held it: {r.overlaps}"
        owner: dict[str, tuple[str, ...]] = {}
        for zs in r.dispatches():
            for z in zs:
                assert owner.setdefault(z, zs) == zs, f"zone {z} appeared in two different shards"

    @pytest.mark.parametrize("ends", ["CRASHED", "CANCELLED"])
    def test_a_fill_that_cannot_prove_it_stopped_waits_for_the_round(self, wired, monkeypatch, ends):
        """Neither state carries the proof the immediate path needs.

        A crashed run's ``finally`` never ran, and the verdict can be reached from missed
        heartbeats while the run is still committing; a cancellation is a REQUEST, and its
        descendants are cancelled one level further out. Both wait for the round, which is
        what they do today.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": ends}, hold="02N"))
        asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True))
        assert r.dispatched_before(("02N",)) == [("01N",), ("02N",)]

    def test_a_dispatch_that_raised_is_not_replaced_immediately(self, wired, monkeypatch, caplog):
        """A dispatch that failed to RETURN may still have dispatched.

        So it establishes nothing at all about what is running, and its cells wait. The
        REASON is asserted as well as the outcome: a state check alone would also decline
        this (an exception carries no state), so without the reason nothing here would
        distinguish the rule from that accident.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "raise"}, hold="02N"))
        with caplog.at_level(logging.INFO):
            asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True))
        assert r.dispatched_before(("02N",)) == [("01N",), ("02N",)]
        assert "the dispatch itself raised" in "\n".join(rec.getMessage() for rec in caplog.records)

    def test_a_replacement_waits_for_the_settling_delay_before_dispatching(self, wired, monkeypatch):
        """The one thing standing between a settled fill and its replacement, now that the
        live-run census is gone.

        A census could only report what was true a moment ago, and could not make a
        lingering child stop. The wait gives an asynchronous cancellation the one thing it
        actually needs.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))
        monkeypatch.setattr(mod, "_SETTLE_DELAY_S", mod.CANCELLATION_CONFIRM_S)
        slept: list[float] = []
        real_sleep = asyncio.sleep

        async def spy(delay, *a, **k):
            slept.append(delay)
            return await real_sleep(0)  # the wait is asserted, never actually served

        monkeypatch.setattr(mod.asyncio, "sleep", spy)
        asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=1))
        assert mod.CANCELLATION_CONFIRM_S in slept, f"the replacement did not wait: {slept}"
        assert r.dispatches().count(("01N",)) == 2, "the replacement never went out"

    def test_settling_delays_run_concurrently_not_in_series(self, wired, monkeypatch):
        """Each settled slot waits out its OWN delay, at the same time as the others.

        Awaiting a planner inline would stop the drain loop harvesting anything else, so
        several clusters failing together would serialise their waits end to end — and a
        sibling finishing mid-wait could not close the round until that wait expired.
        """
        r = _round(
            monkeypatch,
            wired,
            _Round(wired, ends={"01N": "FAILED", "02N": "FAILED"}, hold="03N"),
            zones=("01N", "02N", "03N"),
        )
        monkeypatch.setattr(mod, "_SETTLE_DELAY_S", mod.CANCELLATION_CONFIRM_S)
        marks: list[str] = []
        real_sleep = asyncio.sleep

        async def spy(delay, *a, **k):
            if delay != mod.CANCELLATION_CONFIRM_S:
                return await real_sleep(0)
            marks.append("start")
            for _ in range(5):
                await real_sleep(0)
            marks.append("end")
            return None

        monkeypatch.setattr(mod.asyncio, "sleep", spy)
        asyncio.run(r.drive(max_parallel_clusters=3, immediate_refill=True, max_dispatch_rounds=1))
        assert marks.count("start") == 2, f"both settled rosters should have waited: {marks}"
        assert marks[:2] == ["start", "start"], f"the settling delays ran in series: {marks}"

    def test_a_planner_is_dropped_once_no_dispatch_is_left_to_outrun(self, wired, monkeypatch):
        """The barrier this feature removes, reintroduced at the end of it.

        The last sibling settles while a failed roster is still waiting out its delay, so the
        replacement can no longer buy any wall clock — with the round over, the loop
        re-dispatches the same cells on the same terms. Serving the rest of that wait would
        hold the store re-read for the whole settling budget on behalf of a dispatch that
        cannot happen.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))
        monkeypatch.setattr(mod, "_SETTLE_DELAY_S", mod.CANCELLATION_CONFIRM_S)
        served: list[float] = []
        real_sleep = asyncio.sleep

        async def spy(delay, *a, **k):
            if delay != mod.CANCELLATION_CONFIRM_S:
                return await real_sleep(0)
            # The last sibling settles DURING the wait, which is the case under test.
            r.release.set()
            for _ in range(10):
                await real_sleep(0)
            served.append(delay)  # reached only if the planner was never dropped
            return None

        monkeypatch.setattr(mod.asyncio, "sleep", spy)
        asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=1))
        assert served == [], "the settling wait was served out after the round had nothing left to outrun"
        assert r.dispatches().count(("01N",)) == 1, "a replacement went out with no sibling to outrun"
        assert not r.overlaps, f"two dispatches held one zone: {r.overlaps}"

    def test_the_settling_delay_is_derived_from_the_cancellation_budget(self):
        """Derived, not chosen. A fill's teardown spends this budget waiting for its own
        children before it goes terminal; the delay is the same budget again for the level
        below, whose cancellation nothing waited for. One constant, so the two cannot drift
        into disagreeing about how long an asynchronous cancellation takes.
        """
        assert mod._SETTLE_DELAY_S == mod.CANCELLATION_CONFIRM_S

    def test_a_roster_with_nothing_left_to_do_does_not_pay_the_wait(self, wired, monkeypatch):
        """Most clusters settle having landed everything, and those must not hold the drain
        loop for a wait that protects a dispatch which is not going to happen. So the store
        is read once BEFORE the wait, purely to decide whether the wait is worth paying.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={}, hold="02N"))  # 01N completes cleanly
        monkeypatch.setattr(mod, "_SETTLE_DELAY_S", mod.CANCELLATION_CONFIRM_S)
        slept: list[float] = []
        real_sleep = asyncio.sleep

        async def spy(delay, *a, **k):
            slept.append(delay)
            return await real_sleep(0)

        monkeypatch.setattr(mod.asyncio, "sleep", spy)
        asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=1))
        assert mod.CANCELLATION_CONFIRM_S not in slept, "waited for a roster that had nothing missing"

    def test_the_store_is_re_read_between_the_wait_and_the_dispatch(self, wired, monkeypatch):
        """The dispatch is based on the SETTLED picture, not on what was true before the
        descendants were given time to stop — a cell that landed meanwhile must not cost a
        fleet.

        Asserted as "a read happens between the wait and the next dispatch", not merely as
        "a read happens after the wait": the round always re-reads the store when it closes,
        so the weaker form is satisfied by a read that has nothing to do with this decision.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))
        monkeypatch.setattr(mod, "_SETTLE_DELAY_S", mod.CANCELLATION_CONFIRM_S)
        status = SimpleNamespace(zones=dict.fromkeys(("01N", "02N"), ()), has=lambda z, y: False)
        real_sleep = asyncio.sleep

        def read(*_a, **_k):
            r.timeline.append(("read", ()))
            return status

        async def spy(delay, *a, **k):
            if delay == mod.CANCELLATION_CONFIRM_S:
                r.timeline.append(("wait", ()))
            return await real_sleep(0)

        monkeypatch.setattr(mod, "campaign_status", read)
        monkeypatch.setattr(mod.asyncio, "sleep", spy)
        asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=1))
        kinds = [kind for kind, _ in r.timeline]
        assert "wait" in kinds, "the replacement never waited"
        after = kinds[kinds.index("wait") :]
        upto_dispatch = after[: after.index("dispatch")]
        assert "read" in upto_dispatch, f"the store was not re-read between the wait and the dispatch: {kinds}"

    def test_one_generation_per_slot_per_round(self, wired, monkeypatch):
        """A replacement is not itself replaced — that is what bounds the extra fleets.

        Without it a persistent shortage (the account unable to supply a single GPU) turns
        one round into a dispatch loop.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))
        asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=1))
        assert r.dispatches().count(("01N",)) == 2, r.dispatches()

    def test_the_last_cluster_to_settle_is_left_to_the_round(self, wired, monkeypatch):
        """With nothing else running there is nothing to skip waiting for.

        The round re-dispatches it anyway, so a replacement here would buy an extra fleet
        for no wall clock at all — which is what keeps a cell's total attempts bounded.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="_none"), zones=("01N",))
        asyncio.run(r.drive(max_parallel_clusters=1, immediate_refill=True, max_dispatch_rounds=1))
        assert r.dispatches() == [("01N",)], r.dispatches()

    def test_a_round_that_is_already_over_is_not_even_planned_for(self, wired, monkeypatch):
        """The gate before planning is a COST filter, not the safety one.

        The commit gate below it is what actually prevents a pointless replacement, and it
        would decline this case on its own. This one exists so the driver does not pay for
        a store re-read and a full live-run scan per settled cluster once the round is
        over — which, at the fleet's cluster count, is a burst of orchestrator queries for
        an answer that is already known.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="_none"), zones=("01N",))
        status = SimpleNamespace(zones={"01N": ()}, has=lambda z, y: False)
        reads = {"n": 0}

        def counting_read(*_a, **_k):
            reads["n"] += 1
            return status

        monkeypatch.setattr(mod, "campaign_status", counting_read)
        asyncio.run(r.drive(max_parallel_clusters=1, immediate_refill=True, max_dispatch_rounds=1))
        # One read at flow start and one when the round closes. A third would mean the
        # planner ran for a round that was already over.
        assert reads["n"] == 2, f"the planner ran for a round that had already finished ({reads['n']} reads)"


class TestTheDispatchDecisionIsAtomic:
    """Nothing that can await or raise may sit between deciding to dispatch and dispatching.

    A starred ``gather`` gave both halves of that for free: it consumed its whole generator
    before scheduling anything, and its tasks never existed outside the awaited expression.
    A hand-rolled drain loop has to earn them.
    """

    def test_a_parameter_build_that_raises_dispatches_nothing(self, wired, monkeypatch):
        """Partial dispatch is the worst outcome available here.

        The parameter build probes the land mask and reads SSM, so it can fail for a later
        cluster after earlier ones are already scheduled. Those fills would then keep
        running while the campaign fails — and an ordinary FAILED state does not fire the
        child-cancellation hook, so nothing would sweep them.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={}, hold="_none"), zones=("01N", "02N"))

        def probe(_mask, zone, **_k):
            if zone == "02N":
                raise RuntimeError("land-mask read failed")
            return True

        monkeypatch.setattr(mod, "zone_has_live_tiles", probe)
        with pytest.raises(RuntimeError, match="land-mask read failed"):
            asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True))
        assert r.dispatches() == [], f"a fill was dispatched before the build failed: {r.dispatches()}"

    def test_a_round_that_ends_while_the_refill_is_planned_does_not_get_one(self, wired, monkeypatch):
        """The gate checked before planning is stale by the time the dispatch happens.

        Planning reads the store and queries the orchestrator, and a sibling can finish
        across those awaits while its task sits in ``live`` unharvested. Dispatching then
        buys an extra fleet and an extra attempt for no wall clock at all, because the round
        is already over and would re-dispatch anyway.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))

        monkeypatch.setattr(mod, "_SETTLE_DELAY_S", mod.CANCELLATION_CONFIRM_S)
        real_sleep = asyncio.sleep

        async def wait_that_lets_the_sibling_finish(delay, *a, **k):
            if delay == mod.CANCELLATION_CONFIRM_S:
                r.release.set()  # 02N's dispatch is parked on this
                for _ in range(20):
                    await real_sleep(0)
                return None
            return await real_sleep(0)

        monkeypatch.setattr(mod.asyncio, "sleep", wait_that_lets_the_sibling_finish)
        with caplog_at_info() as records:
            asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=1))
        assert r.dispatched_before(("02N",)) == [("01N",), ("02N",)]
        assert "the round finished while the refill was being planned" in "\n".join(records)

    def test_an_unwinding_drain_waits_for_its_dispatches_to_finish_unwinding(self, wired, monkeypatch):
        """`cancel()` is a REQUEST, and returning on it is not the same as waiting.

        A dispatch caught mid-unwind may still be registering a child run server-side. If
        the flow reaches its terminal hook first, that child appears AFTER the hook's
        one-shot tag sweep has run and nothing collects it. `gather` waited for its children
        before the outer await raised; the drain loop that replaced it has to do the same.
        """

        class _DrainInterruptedError(Exception):
            pass

        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))
        real_wait = asyncio.wait
        calls = {"n": 0}

        async def wait_then_fail(tasks, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:  # after both dispatches exist and one has settled
                raise _DrainInterruptedError("the drain loop was interrupted")
            return await real_wait(tasks, **kwargs)

        monkeypatch.setattr(mod.asyncio, "wait", wait_then_fail)

        async def _go():
            task = asyncio.ensure_future(
                mod.run_global_campaign.fn(
                    paths=_PATHS,
                    ami_ssm_name="ami",
                    max_parallel_clusters=2,
                    immediate_refill=True,
                )
            )
            for _ in range(200):
                await asyncio.sleep(0)
                if task.done():
                    break
            with pytest.raises(_DrainInterruptedError):
                await task
            # Captured while the loop is still running, so this cannot be satisfied by the
            # tidy-up `asyncio.run` does to pending tasks on its way out.
            return list(r.timeline)

        timeline = asyncio.run(_go())
        assert ("settle", ("02N",)) in timeline, (
            f"the held dispatch had not finished unwinding when the flow returned: {timeline}"
        )


class TestUncertaintyIsNotPermission:
    """Every read the refill decision makes can fail, and a failure must DECLINE.

    Declining costs a round's wait. Proceeding on an unestablished fact risks a second
    writer; escaping fails the campaign while sibling fills are still running, which an
    ordinary FAILED state does not sweep.
    """

    def test_a_store_read_failure_declines_instead_of_failing_the_campaign(self, wired, monkeypatch):
        """`campaign_status` and `list_tags` reach the store over a network."""
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))
        status = SimpleNamespace(zones=dict.fromkeys(("01N", "02N"), ()), has=lambda z, y: False)
        calls = {"n": 0}

        def flaky_status(*_a, **_k):
            calls["n"] += 1
            if calls["n"] == 2:  # the refill's own re-read; the round's own reads still work
                raise RuntimeError("store unreachable")
            return status

        monkeypatch.setattr(mod, "campaign_status", flaky_status)
        with caplog_at_info() as records:
            summary = asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=1))
        assert summary["unfilled"] == {2025: ["01N"]}, "the campaign should finish, not fail"
        assert r.dispatched_before(("02N",)) == [("01N",), ("02N",)]
        assert "Could not establish what is safe to refill" in "\n".join(records)

    def test_a_failing_replacement_parameter_build_declines_instead_of_failing_the_campaign(self, wired, monkeypatch):
        """The replacement's OWN build probes the land mask and reads SSM, exactly as the
        round's does — and it runs mid-round, with siblings still writing. Left outside the
        guard it would end the campaign over a transient read.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED"}, hold="02N"))
        dispatched: list[int] = []

        def probe(_mask, _zone, **_k):
            # Fails only once a fill has settled, so the round's own builds succeed and only
            # the replacement's fails.
            if r.timeline and any(kind == "settle" for kind, _z in r.timeline):
                raise RuntimeError("land-mask read failed")
            dispatched.append(1)
            return True

        monkeypatch.setattr(mod, "zone_has_live_tiles", probe)
        with caplog_at_info() as records:
            summary = asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=1))
        assert summary["unfilled"] == {2025: ["01N"]}, "the campaign should finish, not fail"
        assert r.dispatched_before(("02N",)) == [("01N",), ("02N",)]
        assert "Could not establish what is safe to refill" in "\n".join(records)


class TestTheExtraAttemptIsBoundedAcrossRounds:
    """The bound advertised is per CELL for the whole campaign: at most one attempt beyond
    the round budget. Rounds are re-entered, so a marker scoped to a round would hand that
    extra attempt out again on every one.
    """

    def test_a_cell_gets_one_extra_attempt_for_the_campaign_not_one_per_round(self, wired, monkeypatch):
        """Both zones keep failing, so every round has a sibling to hold and the immediate
        path is genuinely reachable in each. `max_dispatch_rounds=2` therefore permits two
        originals plus ONE refill, not one refill per round.
        """
        r = _round(monkeypatch, wired, _Round(wired, ends={"01N": "FAILED", "02N": "FAILED"}, hold="02N"))
        summary = asyncio.run(r.drive(max_parallel_clusters=2, immediate_refill=True, max_dispatch_rounds=2))
        assert summary["unfilled"] == {2025: ["01N", "02N"]}
        assert r.dispatches().count(("01N",)) == 3, r.dispatches()
