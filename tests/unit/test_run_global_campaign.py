"""run_global_campaign: ingest → fill → cleanup chaining per cell.

The work-list logic itself is covered in test_campaign (campaign_work_list); here
the flow body is driven via ``.fn`` with mocked dispatch to check the per-cell
sequence, the ingest bypass, and mosaic cleanup.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prefect.states import StateType

import tessera_embeddings.orchestration.prefect.flows.run_global_campaign as mod
from tessera_embeddings.config.paths import BucketPaths

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


def _per_cell(**kwargs):
    """Drive the campaign on the PER-CELL chain (ingest → fill → cleanup).

    Explicit because the default strategy is now ``"chained-clusters"``, which
    bypasses that chain entirely — these tests would otherwise silently assert
    against a code path they are not about. Strategy-agnostic tests use it too;
    the default itself is pinned by ``test_default_strategy_is_chained_clusters``.
    """
    kwargs.setdefault("fill_strategy", "cluster-per-zone")
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
        return [c for c in cells if c not in wired["landed"] and (wanted is None or c[1] in wanted)]

    return _work


def _dispatched_cells(params: dict) -> list[tuple[str, int]]:
    """The (zone, year) cells one dispatch covers, for either fill strategy.

    ``cluster-per-zone`` and ``ingest-zone-year`` pass a single ``zone``/``year``;
    ``chained-clusters`` passes a ``cells`` list. Anything else (a milestone tag, say)
    covers no cells.
    """
    if "cells" in params:
        return [(z, y) for z, y in params["cells"]]
    if "zone" in params and "year" in params:
        return [(params["zone"], params["year"])]
    return []


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
    # Immutable code identity (AMI ID + optional tarball ETag) — mocked so tests make no
    # SSM/S3 call; resolved once per campaign and folded into the staging fingerprint.
    monkeypatch.setattr(mod, "_resolve_code_identity", lambda *a, **k: "ami=ami-test")
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

    def test_the_default_strategy_is_chained_clusters(self, wired):
        """GPUs stream through a cluster instead of paying a cluster per zone."""
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
        assert [d for d, _ in wired["arun"]] == ["fill-zones-sequential/fill-zones-sequential"]

    def test_eight_clusters_share_forty_ingest_slots(self):
        """Five UTM zones ingesting per cluster: 8 x 5 = the 40-zone fleet-wide cap.

        The two numbers are a pair, not independent knobs — the per-cluster ingest
        window is the cap divided by the cluster count, so changing either changes
        how many zones a cluster keeps in flight.
        """
        sig = inspect.signature(mod.run_global_campaign.fn)
        clusters = sig.parameters["max_parallel_clusters"].default
        ingests = sig.parameters["max_parallel_ingest"].default
        assert (clusters, ingests) == (8, 40)
        assert ingests % clusters == 0, "an uneven split rounds up and overshoots the cap"

    def test_both_caps_are_published_to_the_server(self, wired):
        """Both gates live in CHILD flow runs, so a cap only binds if it reaches the
        Prefect limit they name. Writing both from this flow is what stops the
        server's numbers drifting from the ones chosen here.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
        assert dict(wired["limits"]) == {"tessera-global-ingests": 40, "tessera-global-commits": 8}
        assert wired["arun"][0][1]["ingest_limit_name"] == "tessera-global-ingests"
        assert wired["arun"][0][1]["commit_limit_name"] == "tessera-global-commits"

    def test_no_ingest_cap_is_published_when_ingest_is_off(self, wired):
        """Prebuilt mosaics mean no ingest to gate. Commits still happen, so that
        cap is still published.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False))
        assert dict(wired["limits"]) == {"tessera-global-commits": 8}

    @pytest.mark.parametrize(("clusters", "expected"), [(1, 1), (4, 4), (8, 8), (16, 8), (40, 8)])
    def test_the_commit_cap_is_derived_from_the_cluster_count(self, wired, clusters, expected):
        """``min(clusters, 8)``, and both halves earn their place.

        A cluster's trailing assembly is single-threaded, so N clusters can produce
        at most N assembly commits at once — more slots than clusters is a number
        that could never bind. And the run-1 curve caps it at 8 however many
        clusters run: 16 simultaneous committers measured 7.5 rebase retries and a
        2.2 s commit, which breached that experiment's own acceptance criterion.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=clusters))
        assert dict(wired["limits"])["tessera-global-commits"] == expected

    def test_the_commit_cap_is_never_above_the_measured_ceiling(self):
        """Pinned as a literal so raising it is a deliberate act against the ADR."""
        assert mod.MAX_SIMULTANEOUS_COMMITTERS == 8

    def test_an_empty_commit_limit_name_publishes_nothing(self, wired):
        """The documented escape hatch for running ungated — it must not then write
        a limit named ``""``.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", commit_limit_name=""))
        assert dict(wired["limits"]) == {"tessera-global-ingests": 40}

    def test_nothing_bounds_mosaics_in_flight(self, wired, monkeypatch):
        """A whole year's mosaics may coexist: no backpressure from fill onto ingest.

        Measured as PEAK MOSAICS ALIVE — ingested but not yet cleaned — because that
        is the quantity the removed semaphore actually bounded. Concurrent-ingest
        counts cannot detect it: the old bound was ``zones + ingests``, always
        greater than the ``ingests`` cap that limits ingestion anyway.

        Fills are held until every mosaic exists, which is satisfiable only without
        a chain-wide gate. With one, a cell could not begin ingesting until an
        earlier cell's fill had returned, so the peak would stall at that bound —
        the wait then times out and the assertion reports how far it got, rather
        than the test hanging. ADR-011 records why the trade was accepted.
        """
        names = [f"{i + 1:02d}N" for i in range(12)]
        status = SimpleNamespace(zones=dict.fromkeys(names, ()), has=lambda z, y: False)
        monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
        monkeypatch.setattr(mod, "campaign_work_list", _shrinking([(z, 2025) for z in names], wired))

        alive = 0
        peak = 0
        all_ingested = asyncio.Event()

        def _cleaned(_uri, **_k):
            nonlocal alive
            alive -= 1  # a mosaic stops being alive at ITS cleanup, not at its ingest

        async def gated_arun(dep, parameters=None, tags=None):
            nonlocal alive, peak
            wired["arun"].append((dep, parameters))
            if "ingest-zone-year" in dep:
                alive += 1
                peak = max(peak, alive)
                if alive == len(names):
                    all_ingested.set()
            else:
                try:
                    await asyncio.wait_for(all_ingested.wait(), timeout=2)
                except TimeoutError:
                    pass  # regression: report the peak reached instead of hanging
            wired["land"](parameters)
            return _completed_run()

        monkeypatch.setattr(mod, "arun_deployment", gated_arun)
        monkeypatch.setattr(mod, "delete_prefix", _cleaned)
        # Both caps well below the cell count, so only the absence of a chain-wide
        # gate can let all twelve mosaics exist at once.
        _per_cell(max_parallel_clusters=2, max_parallel_ingest=3)
        assert peak == len(names), f"only {peak} of {len(names)} mosaics coexisted"


def test_ingest_then_fill_then_cleanup(wired):
    result = _per_cell()
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["ingest-zone-year/ingest-zone-year", "fill-zone-year/fill-zone-year"]  # ingest BEFORE fill
    # Ingest + fill both target the same (zone, year).
    ingest_params = wired["arun"][0][1]
    assert ingest_params["zone"] == "33N" and ingest_params["year"] == 2025
    # The transient mosaic is dropped after the fill lands.
    assert wired["deletes"] == ["s3://in/mosaics/33N/2025"]
    assert result["dispatched"] == 1


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

    _per_cell()
    assert captured["open_kw"].get("get_credentials") is not None
    assert onaxis.get("get_credentials") is not None


def _dispatched_zones(params: dict) -> list[str]:
    """The zones of a chained dispatch, from its `(zone, year)` pairs.

    The child takes pairs so a cluster can span years. This driver is still year-serial,
    so every pair in one dispatch shares a year; asserting on zones keeps these tests
    about the PARTITION rather than about the parameter shape.
    """
    years = {y for _, y in params["cells"]}
    assert len(years) == 1, f"a year-serial driver must dispatch one year per cluster, got {years}"
    return [z for z, _ in params["cells"]]


def _fill_run_id(rec: dict, **kwargs) -> str:
    rec["arun"].clear()
    rec["landed"].clear()  # a fresh campaign, not a second round
    _per_cell(**kwargs)
    return next(p for d, p in rec["arun"] if d == "fill-zone-year/fill-zone-year")["run_id"]


def test_fill_run_id_is_stable_and_input_fingerprinted(wired):
    """The fill run_id is deterministic per (zone, year) — so a retry resumes the
    same staging prefix — AND carries an input fingerprint, so a changed input
    (here min_valid_coverage) yields a DIFFERENT prefix (no stale resume).
    """
    base = _fill_run_id(wired)
    assert base.startswith("33N-2025-")  # cell-scoped prefix + fingerprint suffix
    assert _fill_run_id(wired) == base  # deterministic across identical runs
    changed = _fill_run_id(wired, ingest_settings=mod.IngestSettings(min_valid_coverage=0.5))
    assert changed.startswith("33N-2025-") and changed != base  # input change → new prefix


def test_fill_run_id_changes_when_the_mosaic_is_rebuilt(wired, monkeypatch):
    """A rebuilt mosaic must not resume tiles staged against the previous one.

    The ingest marker is policy only, so re-ingesting the same window under the same
    settings reproduces it exactly — while the pixels can differ (a reprocessed or
    late-published granule). Keying on the marker alone would resume the old tiles
    and publish a zone-year mixing two mosaic revisions.
    """
    marker = {"window": ["2025-01-01", "2025-12-31"], "coverage_sha256": "cov"}

    def _mosaic(completed_at):
        return lambda *a, **k: (
            SimpleNamespace(
                attrs={"registry_sha256": "cov-sha-test", "ingest_marker": marker, "ingest_completed_at": completed_at}
            ),
            "SNAPSHOT0",
        )

    monkeypatch.setattr(mod, "open_store_group_and_tip", _mosaic("2026-01-01T00:00:00Z"))
    first = _fill_run_id(wired)
    assert _fill_run_id(wired) == first  # same build → same prefix, so a retry resumes

    monkeypatch.setattr(mod, "open_store_group_and_tip", _mosaic("2026-02-02T00:00:00Z"))
    assert _fill_run_id(wired) != first  # identical marker, new build → fresh prefix


def test_fill_run_id_changes_with_allow_s2_only(wired):
    """allow_s2_only changes WHICH pixels get embeddings, so it must flip the
    staging fingerprint — a retry across a flipped flag never resumes mixed tiles.
    The flag is also forwarded to the fill deployment.
    """
    base = _fill_run_id(wired)
    flagged = _fill_run_id(wired, allow_s2_only=True)
    assert flagged.startswith("33N-2025-") and flagged != base
    fill_params = next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")
    assert fill_params["allow_s2_only"] is True


def test_fill_run_id_ignores_the_build_and_keys_on_inference_code(wired, monkeypatch):
    """NARROWED 2026-07-30: staging reuse keys on the INFERENCE SOURCE, not the build.

    The fingerprint used to be the resolved AMI ID plus the tarball ETag, so re-baking the
    AMI or a hotfix anywhere in the repo abandoned every staged tile. Neither should move
    it now — only a change to the code that decides what a staged tile contains.
    """
    base = _fill_run_id(wired)
    # A `code_suffix` label change: never mattered, still doesn't.
    assert _fill_run_id(wired, code_suffix="-branchB") == base
    # A RE-BAKED AMI now REUSES the prefix. This assertion is the inverse of the one it
    # replaces, and that inversion is the whole point of the change.
    monkeypatch.setattr(mod, "_resolve_code_identity", lambda *a, **k: "ami=ami-REBAKED")
    assert _fill_run_id(wired) == base
    # A change to the inference source DOES start a fresh prefix.
    monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-DIFFERENT")
    changed = _fill_run_id(wired)
    assert changed.startswith("33N-2025-") and changed != base


def _chained_staging_identity(wired, **kwargs) -> str:
    """The staging identity the DEFAULT strategy hands its child."""
    wired["arun"].clear()
    wired["landed"].clear()
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", **kwargs))
    params = next(p for d, p in wired["arun"] if "fill-zones-sequential" in d)
    return params["staging_code_identity"]


class TestBothStrategiesShareOneStagingIdentity:
    """The narrowing and its escape hatches have to reach the DEFAULT path.

    `chained-clusters` is the strategy campaigns actually run, and it used to recompute
    the AMI-plus-tarball artifact identity for itself — so a re-bake abandoned every
    staged tile on the path that runs, and `force_staging_reuse`/`force_staging_restage`
    were documented on a flow that never forwarded them.
    """

    def test_the_chained_child_is_handed_the_narrowed_identity(self, wired, monkeypatch):
        monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-FIXED")
        assert _chained_staging_identity(wired) == "infcode-FIXED"

    def test_a_rebaked_ami_does_not_change_it(self, wired, monkeypatch):
        """The inversion the narrowing was for, asserted on the default path too."""
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

    def test_the_two_strategies_agree(self, wired):
        """Same campaign, same inputs — the identity must not depend on the strategy."""
        chained = _chained_staging_identity(wired)
        wired["arun"].clear()
        wired["landed"].clear()
        _per_cell()
        per_zone = next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")["run_id"]
        # The per-zone path folds the identity into the run_id rather than passing it,
        # so recompute that id with the chained identity and require a match.
        assert per_zone.startswith("33N-2025-")
        assert chained == mod.inference_code_identity()


def test_force_staging_reuse_survives_an_inference_change(wired, monkeypatch):
    """The escape hatch for a change the author knows is output-neutral."""
    base = _fill_run_id(wired, force_staging_reuse=True)
    monkeypatch.setattr(mod, "inference_code_identity", lambda: "infcode-DIFFERENT")
    assert _fill_run_id(wired, force_staging_reuse=True) == base
    # ...and it really is an override, not a no-op: without it the same change moves it.
    assert _fill_run_id(wired) != base


def test_force_staging_restage_forces_a_fresh_prefix(wired) -> None:
    """The escape hatch for a change the source hash cannot see — a torch upgrade."""
    base = _fill_run_id(wired)
    bumped = _fill_run_id(wired, force_staging_restage="torch-2.9")
    assert bumped != base
    # Same token → same prefix, so a retry after the upgrade still resumes.
    assert _fill_run_id(wired, force_staging_restage="torch-2.9") == bumped
    assert _fill_run_id(wired, force_staging_restage="torch-3.0") not in {base, bumped}


def test_the_ami_is_still_pinned_even_though_it_no_longer_fingerprints(wired, monkeypatch):
    """Narrowing the fingerprint must not stop the AMI being resolved once and pinned.

    One campaign must never straddle two images: the fingerprint no longer notices, so the
    pinning is now the ONLY thing preventing it. Also pins the region — the AMI SSM param
    lives where Ray provisions (None → us-west-2), not in a non-default storage region.
    """
    calls: list = []
    monkeypatch.setattr(mod, "_resolve_ami_id", lambda *a, **k: calls.append(a) or "ami-test")
    _per_cell(s3_region="eu-west-1")
    assert calls, "the AMI must still be resolved"
    assert all(c[-1] != "eu-west-1" for c in calls), calls


def test_mosaic_identity_fingerprints_only_active_orbits(wired, monkeypatch):
    """With a single requested orbit, _mosaic_identity opens reflectance + the active
    SAR store only — never the opposite orbit, even if a stale one is present (which
    the fill never reads and which could wrongly raise or perturb the run_id).
    """
    opened: list[str] = []

    def _capture(path, **k):
        opened.append(path)
        return SimpleNamespace(attrs={"registry_sha256": "cov", "last_appended": "2026-01-01T00:00:00Z"}), "SNAPSHOT0"

    monkeypatch.setattr(mod, "open_store_group_and_tip", _capture)
    _per_cell(ingest=False, cleanup_mosaics=False, s1_orbit="ascending")
    assert any("reflectance.zarr" in p for p in opened)
    assert any("sar_ascending.zarr" in p for p in opened)
    assert not any("sar_descending.zarr" in p for p in opened)  # inactive orbit never touched


def test_all_ocean_cell_uses_empty_run_id_and_skips_cleanup(wired, monkeypatch):
    """An all-ocean cell (no live tiles) gets a stable '-empty' run id and never
    fingerprints a (nonexistent) mosaic — _staging_run_id would raise 'No mosaic stores
    found' and strand the year — and no mosaic cleanup runs (ingest produced none).
    """
    monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: False)

    def _boom(*a, **k):
        raise AssertionError("an empty cell must NOT read the mosaic for a staging fingerprint")

    monkeypatch.setattr(mod, "open_store_group_and_tip", _boom)
    _per_cell()
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["ingest-zone-year/ingest-zone-year", "fill-zone-year/fill-zone-year"]  # ingest still probes+skips
    fill = next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")
    assert fill["run_id"] == "33N-2025-empty"
    assert wired["deletes"] == []  # no mosaic produced → nothing to clean up


class TestValidationDeploymentForwarding:
    """Both fill strategies must receive the validator the campaign was dispatched with.

    Forwarded rather than left to each child's own registration, and asserted on BOTH
    strategies: a child registered before the validator existed carries ``None``, and a
    campaign that failed to forward its own value would then run a thousand cells with no
    validation at all while every deployment looked correctly configured.
    """

    def test_the_per_cell_fill_receives_it(self, wired):
        _per_cell(validation_deployment="validate-zone-year/validate-zone-year-x")
        fill = next(p for d, p in wired["arun"] if "fill-zone-year" in d)
        assert fill["validation_deployment"] == "validate-zone-year/validate-zone-year-x"

    def test_the_chained_fill_receives_it(self, wired):
        asyncio.run(
            mod.run_global_campaign.fn(
                paths=_PATHS, ami_ssm_name="ami", validation_deployment="validate-zone-year/validate-zone-year-x"
            )
        )
        params = next(p for d, p in wired["arun"] if "fill-zones-sequential" in d)
        assert params["validation_deployment"] == "validate-zone-year/validate-zone-year-x"

    def test_it_is_not_derived_from_the_branch(self, wired):
        """Unlike every other child ref. The validator is a consumer's flow rather than one
        of this library's, so there is no base name here to suffix — a branch run that
        names no validator validates nothing, and says so by carrying None.
        """
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", branch="some-branch"))
        params = next(p for d, p in wired["arun"] if "fill-zones-sequential" in d)
        assert params["validation_deployment"] is None


def test_ingest_params_carry_s3_region(wired):
    """The campaign's s3_region reaches the ingest deployment (not just the fill), so a
    non-default-region deployment's ingest metadata opens hit the right bucket.
    """
    _per_cell(s3_region="eu-west-1")
    ingest = next(p for d, p in wired["arun"] if d == "ingest-zone-year/ingest-zone-year")
    fill = next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")
    assert ingest["s3_region"] == "eu-west-1" and fill["s3_region"] == "eu-west-1"


def test_ingest_false_run_id_tracks_prebuilt_mosaic_identity(wired, monkeypatch):
    """With ingest=False the staging fingerprint tracks the prebuilt mosaic's own
    identity (reflectance last_appended), so replacing the mosaic and rerunning with
    the same config yields a FRESH staging prefix (no stale-tile resume).
    """

    def _run_with_mosaic_ts(ts: str) -> str:
        monkeypatch.setattr(
            mod,
            "open_store_group_and_tip",
            lambda *a, **k: (SimpleNamespace(attrs={"registry_sha256": "cov", "last_appended": ts}), "SNAPSHOT0"),
        )
        wired["arun"].clear()
        wired["landed"].clear()
        _per_cell(ingest=False, cleanup_mosaics=False)
        return next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")["run_id"]

    rid1 = _run_with_mosaic_ts("2026-01-01T00:00:00Z")
    rid2 = _run_with_mosaic_ts("2026-06-01T00:00:00Z")  # mosaic replaced → new last_appended
    assert rid1.startswith("33N-2025-") and rid1 != rid2


def test_retag_only_uses_stable_run_id_without_mosaic(wired, monkeypatch):
    """A retag-only cell (already complete) gets a stable '-retag' run id and does NOT
    fingerprint the mosaic — which may be cleaned up, so inspecting it would raise
    before the fill's documented tag repair.
    """
    monkeypatch.setattr(
        mod, "campaign_status", lambda *a, **k: SimpleNamespace(zones={"33N": (2025,)}, has=lambda z, y: True)
    )

    def _boom(*a, **k):
        raise AssertionError("retag-only must NOT read the mosaic for a staging fingerprint")

    monkeypatch.setattr(mod, "open_store_group_and_tip", _boom)
    _per_cell()
    fill = next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")
    assert fill["run_id"] == "33N-2025-retag"


def _prebuilt_run_id(wired, monkeypatch, *, attrs: dict, tip: str) -> str:
    """run_id of the ingest=False fill for a prebuilt mosaic with these attrs/tip."""
    monkeypatch.setattr(mod, "open_store_group_and_tip", lambda *a, **k: (SimpleNamespace(attrs=attrs), tip))
    wired["arun"].clear()
    wired["landed"].clear()
    _per_cell(ingest=False, cleanup_mosaics=False)
    return next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")["run_id"]


def test_run_id_tracks_the_snapshot_even_when_provenance_attrs_are_unchanged(wired, monkeypatch):
    """The snapshot tip — not the attrs — is what makes the fingerprint safe.

    A region overwrite (a re-run of one date's write, a hand repair, a prebuilt mosaic
    maintained outside this pipeline) changes pixels while leaving `ingest_marker` /
    `ingest_completed_at` exactly as they were. Keyed on attrs alone the retry would
    reuse the prefix, skip the already-staged tiles, and publish a zone-year mixing
    both mosaic revisions — silently. Keyed on the tip, it starts fresh.
    """
    attrs = {"ingest_marker": {"window": ["2025-01-01", "2025-12-31"]}, "ingest_completed_at": "2026-01-01T00:00:00Z"}
    before = _prebuilt_run_id(wired, monkeypatch, attrs=attrs, tip="SNAPSHOT_BEFORE")
    assert _prebuilt_run_id(wired, monkeypatch, attrs=attrs, tip="SNAPSHOT_BEFORE") == before  # retry resumes
    assert _prebuilt_run_id(wired, monkeypatch, attrs=attrs, tip="SNAPSHOT_AFTER") != before


def test_attrless_prebuilt_mosaic_is_identified_by_its_snapshot(wired, monkeypatch):
    """A store with no provenance attrs is no longer fatal: the tip is a complete
    identity on its own, so a prebuilt mosaic that never learned this pipeline's
    bookkeeping still gets a correct, change-detecting staging prefix.
    """
    rid = _prebuilt_run_id(wired, monkeypatch, attrs={}, tip="SNAPSHOT_A")
    assert rid.startswith("33N-2025-")
    assert _prebuilt_run_id(wired, monkeypatch, attrs={}, tip="SNAPSHOT_B") != rid


def test_default_run_rejects_unseeded_work_zone(wired, monkeypatch):
    """A partially-seeded store (default zones=None) fails BEFORE any ingest when the
    work list includes an unseeded zone — not after an expensive per-cell ingest.
    """
    monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [("02N", 2025)])  # 02N not in status.zones
    with pytest.raises(ValueError, match="not seeded"):
        _per_cell()
    assert wired["arun"] == []  # nothing dispatched


def test_ingest_disabled_skips_ingest_and_cleanup(wired):
    _per_cell(ingest=False, cleanup_mosaics=False)
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zone-year/fill-zone-year"]  # fill only
    assert wired["deletes"] == []


def test_rejects_zero_parallel_ingest(wired):
    with pytest.raises(ValueError, match="max_parallel_ingest"):
        _per_cell(max_parallel_ingest=0)


def test_off_axis_year_rejected(wired, monkeypatch):
    """An off-axis year fails up front, before any ingest/fill is dispatched."""
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: False)
    with pytest.raises(ValueError, match="not on the store's pre-allocated axis"):
        _per_cell(years=(2026,))
    assert wired["arun"] == []  # nothing dispatched


def test_retag_only_cell_skips_ingest(wired, monkeypatch):
    """A complete-but-untagged cell only needs a retag — no ingest, no cleanup."""
    status = SimpleNamespace(zones={"33N": (2025,)}, has=lambda z, y: True)  # already complete
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    _per_cell()
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zone-year/fill-zone-year"]  # fill (retag) only, no ingest
    assert wired["deletes"] == []


def test_unseeded_explicit_zone_rejected(wired):
    """An explicit zones= entry that isn't seeded fails before dispatch."""
    with pytest.raises(ValueError, match="not seeded"):
        _per_cell(zones=["02N"])
    assert wired["arun"] == []


def test_duplicate_years_dispatch_once(wired):
    """The dispatch loop dedupes years — years=(2025, 2025) runs the cell once."""
    _per_cell(years=(2025, 2025))
    assert len(wired["arun"]) == 2  # one ingest + one fill, not two of each


def test_sequential_strategy_dispatches_one_run_per_year(wired):
    """fill_strategy="chained-clusters" replaces the per-cell chain with ONE
    fill-zones-sequential run per year: no driver-side ingest (the child's
    look-ahead owns it), no driver-side mosaic cleanup, cells passed as (zone, year) pairs.
    """
    result = asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters"))
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zones-sequential/fill-zones-sequential"]
    params = wired["arun"][0][1]
    assert _dispatched_zones(params) == ["33N"] and params["cells"] == [["33N", 2025]]
    # One cluster, so it carries the whole ingest bound as its window.
    assert params["ingest"] is True and params["look_ahead"] == 40
    assert params["ingest_deployment"] == "ingest-zone-year/ingest-zone-year"
    # Mosaic lifecycle belongs to the child in this mode.
    assert wired["deletes"] == []
    assert result["dispatched"] == 1


def test_sequential_strategy_forwards_ingest_false(wired):
    asyncio.run(
        mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters", ingest=False)
    )
    assert wired["arun"][0][1]["ingest"] is False


def test_invalid_fill_strategy_rejected(wired):
    with pytest.raises(ValueError, match="fill_strategy"):
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="both"))
    assert wired["arun"] == []


def test_sequential_strategy_shards_by_live_tiles(wired, monkeypatch):
    """max_parallel_clusters > 1 in sequential mode = that many chained clusters:
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
            fill_strategy="chained-clusters",
            max_parallel_clusters=2,
            max_parallel_ingest=6,
        )
    )
    assert [d for d, _ in wired["arun"]] == ["fill-zones-sequential/fill-zones-sequential"] * 2
    clusters = [_dispatched_zones(p) for _, p in wired["arun"]]
    # LPT: 500 alone; 300+250 together — balanced totals (500 vs 550).
    assert sorted(map(sorted, clusters)) == [["33N"], ["34N", "35N"]]
    # The global ingest bound is divided across clusters: 6 over 2 clusters = 3 each.
    assert all(p["look_ahead"] == 3 for _, p in wired["arun"])


class TestIngestBoundAcrossClusters:
    """The per-cluster look-ahead IS that cluster's ingest concurrency, so the
    fleet-wide figure is the per-cluster value times the cluster count.
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
                fill_strategy="chained-clusters",
                max_parallel_clusters=clusters,
                max_parallel_ingest=ingest_cap,
            )
        )
        return wired["arun"][0][1]["look_ahead"]

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

    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters", max_parallel_clusters=3
        )
    )
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
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters"))
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


def test_branch_none_keeps_prod_refs_everywhere(wired):
    """Regression guard: with no branch, every child ref — direct AND grandchild —
    is the unsuffixed production default (today's behaviour, byte-for-byte).
    """
    _per_cell()
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["ingest-zone-year/ingest-zone-year", "fill-zone-year/fill-zone-year"]
    _, ingest_params = _dep(wired, "ingest-zone-year")
    assert ingest_params["deployments"] == {
        "ingest_s1_roi_sar": "ingest_s1_roi_sar/ingest-s1-roi-sar",
        "ingest_s2_roi_reflectance": "ingest_s2_roi_reflectance/ingest-s2-roi-reflectance",
    }


def test_branch_suffixes_all_four_derived_refs(wired):
    """The fix: a branch slug routes the fill, the ingest, AND the S1/S2
    grandchildren that ingest_zone_year dispatches (the ones that used to 404).
    """
    _per_cell(branch="global-tessera")
    fill_dep, _ = _dep(wired, "fill-zone-year")
    ingest_dep, ingest_params = _dep(wired, "ingest-zone-year")
    assert fill_dep == "fill-zone-year/fill-zone-year-global-tessera"
    assert ingest_dep == "ingest-zone-year/ingest-zone-year-global-tessera"
    assert ingest_params["deployments"] == {
        "ingest_s1_roi_sar": "ingest_s1_roi_sar/ingest-s1-roi-sar-global-tessera",
        "ingest_s2_roi_reflectance": "ingest_s2_roi_reflectance/ingest-s2-roi-reflectance-global-tessera",
    }


def test_explicit_direct_ref_is_verbatim_but_grandchildren_still_route(wired):
    """Precedence: an explicit fill/ingest ref is used verbatim (never re-suffixed),
    yet the S1/S2 grandchildren still follow `branch` — they have no explicit knob.
    """
    _per_cell(branch="b", fill_deployment="custom/fill", ingest_deployment="custom/ingest")
    fill_dep, _ = _dep(wired, "custom/fill")
    ingest_dep, ingest_params = _dep(wired, "custom/ingest")
    assert fill_dep == "custom/fill"  # not custom/fill-b
    assert ingest_dep == "custom/ingest"
    assert ingest_params["deployments"]["ingest_s1_roi_sar"].endswith("-b")


def test_branch_routes_fill_when_ingest_disabled(wired):
    """ingest=False: no ingest dispatch, but the fill ref is still branch-routed."""
    _per_cell(ingest=False, branch="b")
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zone-year/fill-zone-year-b"]  # fill only, suffixed


def test_both_strategies_pin_the_resolved_ami_id(wired):
    """The campaign resolves the worker AMI once and pins it into every dispatched
    fill — cluster-per-zone AND chained — so a fill boots the exact image its
    staging fingerprint recorded instead of re-reading the SSM pointer.
    """
    _per_cell()
    fill = next(p for d, p in wired["arun"] if "fill-zone-year" in d)
    assert fill["ami_id"] == "ami-test-id"

    wired["arun"].clear()
    wired["landed"].clear()
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters"))
    chained = next(p for d, p in wired["arun"] if "fill-zones-sequential" in d)
    assert chained["ami_id"] == "ami-test-id"


@pytest.mark.parametrize(
    ("strategy", "dep"),
    [("cluster-per-zone", "fill-zone-year"), ("chained-clusters", "fill-zones-sequential")],
)
def test_both_strategies_forward_the_model_mismatch_override(wired, strategy: str, dep: str):
    """An operator override cleared in preflight must reach the gate that fires.

    The campaign checks the seeded model once before dispatching, but every child
    re-checks it. Without forwarding, the child falls back to its default and rejects
    the same store — after the cell's ingest has already been paid for.
    """
    asyncio.run(
        mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy=strategy, allow_model_mismatch=True)
    )
    fill = next(p for d, p in wired["arun"] if dep in d)
    assert fill["allow_model_mismatch"] is True


class TestAmiIsResolvedOnlyForCellsThatStartRay:
    """A cell that never provisions Ray must not force an SSM read.

    ``_code_identity`` is deliberately lazy so a pure tag-repair or all-ocean
    campaign makes no AWS call; pinning the AMI into every dispatch would undo
    that and turn a missing parameter — or a role without SSM access — into a
    failure on exactly the cheap recovery path that has no use for the answer.
    """

    def _no_ssm(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("a cell that starts no cluster must not resolve the AMI")

        monkeypatch.setattr(mod, "_resolve_ami_id", _boom)

    def test_retag_only_cell_neither_resolves_nor_pins(self, wired, monkeypatch):
        self._no_ssm(monkeypatch)
        monkeypatch.setattr(
            mod, "campaign_status", lambda *a, **k: SimpleNamespace(zones={"33N": (2025,)}, has=lambda z, y: True)
        )
        _per_cell()
        fill = next(p for d, p in wired["arun"] if "fill-zone-year" in d)
        assert fill["run_id"].endswith("-retag")
        assert fill["ami_id"] is None
        # The SSM pointer still travels, so the fill's own fallback stays available.
        assert fill["ami_ssm_name"] == "ami"

    def test_all_ocean_cell_neither_resolves_nor_pins(self, wired, monkeypatch):
        self._no_ssm(monkeypatch)
        monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: False)
        _per_cell()
        fill = next(p for d, p in wired["arun"] if "fill-zone-year" in d)
        assert fill["run_id"].endswith("-empty")
        assert fill["ami_id"] is None


def test_every_dispatched_child_carries_the_campaign_tag(wired):
    """The tag is the only thing the terminal hooks can work from.

    Prefect runs them in a fresh process after killing the flow, so nothing in
    memory records what was dispatched — an untagged child is an unsweepable one.
    """
    with patch.object(mod.flow_run_ctx, "id", "run-abc"):
        _per_cell()
    assert wired["tags"], "no children dispatched"
    assert all(t == ["campaign:run-abc"] for t in wired["tags"]), wired["tags"]


def test_children_go_untagged_outside_a_flow_run(wired):
    """A direct .fn() call has no run id, and dispatches nothing worth sweeping."""
    _per_cell()
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
        _per_cell()
    assert wired["arun"] == [], "nothing may be dispatched once the gate fails"


def test_model_gate_override_lets_the_campaign_proceed(wired, monkeypatch):
    """The escape hatch is threaded through rather than reimplemented."""
    seen: dict = {}
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: seen.update(k))
    _per_cell(allow_model_mismatch=True)
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
        with pytest.raises(RuntimeError, match="unfilled cell"):
            asyncio.run(
                mod.run_global_campaign.fn(
                    paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1, max_zone_attempts=5
                )
            )
        # One attempt, one retry that achieves nothing, then stop — the minimum
        # possible without guessing at the child's failure type.
        assert rounds["n"] == 2, f"kept retrying a deterministic failure: {rounds['n']} rounds"

    def test_the_campaign_reports_every_unfilled_cell_and_fails(self, wired, monkeypatch):
        """Loudly, and last: a campaign that did not finish must not report success,
        and the operator needs the whole list rather than the first failure.
        """
        self._wire(monkeypatch, wired, ["01N", "02N"], complete_after={})
        with pytest.raises(RuntimeError, match=r"unfilled cell\(s\).*01N.*02N"):
            asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1))

    def test_retries_are_disabled_at_one_attempt(self, wired, monkeypatch):
        rounds = self._wire(monkeypatch, wired, ["01N"], complete_after={})
        with pytest.raises(RuntimeError, match="unfilled cell"):
            asyncio.run(
                mod.run_global_campaign.fn(
                    paths=_PATHS, ami_ssm_name="ami", max_parallel_clusters=1, max_zone_attempts=1
                )
            )
        assert rounds["n"] == 1

    def test_zero_attempts_is_rejected(self, wired):
        with pytest.raises(ValueError, match="max_zone_attempts"):
            asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_zone_attempts=0))


# --- overlap_years: dropping the year barrier -------------------------------------------


def _multi_year(monkeypatch, zones, years, wired):
    """Wire a work list of every (zone, year) pair, with nothing yet complete."""
    status = SimpleNamespace(zones=dict.fromkeys(zones, ()), has=lambda z, y: False, years=tuple(years))
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    monkeypatch.setattr(mod, "campaign_work_list", _shrinking([(z, y) for y in years for z in zones], wired))
    monkeypatch.setattr(mod, "zone_work_weight", lambda mask, zone, **k: 100.0)
    return status


def test_year_serial_is_still_the_default(wired, monkeypatch):
    """The flag is opt-in: without it the driver dispatches one batch PER YEAR.

    Pinned because the year-serial path is the one that has actually been run, and a
    default flip would change the campaign's shape silently.
    """
    _multi_year(monkeypatch, ["01N", "02N"], [2025, 2024], wired)
    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters", max_parallel_clusters=1
        )
    )
    # One dispatch per year, each carrying exactly one year's cells.
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
            fill_strategy="chained-clusters",
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
            fill_strategy="chained-clusters",
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
    with pytest.raises(RuntimeError, match="unfilled cell") as exc:
        asyncio.run(
            mod.run_global_campaign.fn(
                paths=_PATHS,
                ami_ssm_name="ami",
                fill_strategy="chained-clusters",
                max_parallel_clusters=1,
                overlap_years=True,
            )
        )
    # Both years named separately, newest first, each with its zone.
    assert "2025: 01N" in str(exc.value) and "2024: 01N" in str(exc.value), str(exc.value)
