"""run_global_campaign: ingest → fill → cleanup chaining per cell.

The work-list logic itself is covered in test_campaign (campaign_work_list); here
the flow body is driven via ``.fn`` with mocked dispatch to check the per-cell
sequence, the ingest bypass, and mosaic cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prefect.states import StateType

import tessera_embeddings.orchestration.prefect.flows.run_global_campaign as mod
from tessera_embeddings.config.paths import BucketPaths

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


def _completed_run(rid: str = "r") -> SimpleNamespace:
    return SimpleNamespace(id=rid, state=SimpleNamespace(type=StateType.COMPLETED, name="Completed"))


def _defer_milestone(*_a, **_k):
    raise ValueError("not all 120 zones complete yet")  # tag_year_complete stub → milestone deferred


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
    monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [("33N", 2025)])
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
        return _completed_run()

    monkeypatch.setattr(mod, "arun_deployment", fake_arun)
    monkeypatch.setattr(mod, "delete_prefix", lambda uri, **k: rec["deletes"].append(uri))
    return rec


def test_ingest_then_fill_then_cleanup(wired):
    result = asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
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

    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    assert captured["open_kw"].get("get_credentials") is not None
    assert onaxis.get("get_credentials") is not None


def _fill_run_id(rec: dict, **kwargs) -> str:
    rec["arun"].clear()
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", **kwargs))
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


def test_fill_run_id_tracks_resolved_code_artifact_not_suffix(wired, monkeypatch):
    """The staging fingerprint keys on the RESOLVED code artifact (AMI ID + tarball
    ETag), not the mutable `code_suffix`. So overwriting the tarball / re-baking the AMI
    (a new resolved identity) yields a fresh prefix, while a `code_suffix` label change
    that resolves to the same artifact does NOT — closing the mixed-version-year hole.
    """
    base = _fill_run_id(wired)
    # A different code_suffix that resolves to the SAME artifact keeps the prefix stable
    # (the mock ignores its args) — the fingerprint no longer keys on the raw suffix.
    assert _fill_run_id(wired, code_suffix="-branchB") == base
    # A changed resolved artifact (re-baked AMI / overwritten tarball) → fresh prefix.
    monkeypatch.setattr(mod, "_resolve_code_identity", lambda *a, **k: "ami=ami-REBAKED")
    changed = _fill_run_id(wired)
    assert changed.startswith("33N-2025-") and changed != base


def test_code_identity_resolves_in_ray_region_not_storage_region(wired, monkeypatch):
    """The code-artifact lookup must use the RAY provisioning region (None → us-west-2,
    the fill's ray_cluster default), NOT the storage s3_region — the AMI SSM param and
    tarball live where Ray provisions, which may differ from a non-default-region store.
    """
    calls: list = []
    monkeypatch.setattr(mod, "_resolve_code_identity", lambda *a: calls.append(a) or "ami=ami-test")
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", s3_region="eu-west-1"))
    # Called once (memoized) with region (4th positional arg) = None, NOT "eu-west-1".
    assert calls and calls[0][3] is None


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
    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS, ami_ssm_name="ami", ingest=False, cleanup_mosaics=False, s1_orbit="ascending"
        )
    )
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
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["ingest-zone-year/ingest-zone-year", "fill-zone-year/fill-zone-year"]  # ingest still probes+skips
    fill = next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")
    assert fill["run_id"] == "33N-2025-empty"
    assert wired["deletes"] == []  # no mosaic produced → nothing to clean up


def test_ingest_params_carry_s3_region(wired):
    """The campaign's s3_region reaches the ingest deployment (not just the fill), so a
    non-default-region deployment's ingest metadata opens hit the right bucket.
    """
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", s3_region="eu-west-1"))
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
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False, cleanup_mosaics=False))
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
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    fill = next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")
    assert fill["run_id"] == "33N-2025-retag"


def _prebuilt_run_id(wired, monkeypatch, *, attrs: dict, tip: str) -> str:
    """run_id of the ingest=False fill for a prebuilt mosaic with these attrs/tip."""
    monkeypatch.setattr(mod, "open_store_group_and_tip", lambda *a, **k: (SimpleNamespace(attrs=attrs), tip))
    wired["arun"].clear()
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False, cleanup_mosaics=False))
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
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    assert wired["arun"] == []  # nothing dispatched


def test_ingest_disabled_skips_ingest_and_cleanup(wired):
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False, cleanup_mosaics=False))
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zone-year/fill-zone-year"]  # fill only
    assert wired["deletes"] == []


def test_rejects_zero_parallel_ingest(wired):
    with pytest.raises(ValueError, match="max_parallel_ingest"):
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_ingest=0))


def test_off_axis_year_rejected(wired, monkeypatch):
    """An off-axis year fails up front, before any ingest/fill is dispatched."""
    monkeypatch.setattr(mod, "zone_year_on_axis", lambda *a, **k: False)
    with pytest.raises(ValueError, match="not on the store's pre-allocated axis"):
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", years=(2026,)))
    assert wired["arun"] == []  # nothing dispatched


def test_retag_only_cell_skips_ingest(wired, monkeypatch):
    """A complete-but-untagged cell only needs a retag — no ingest, no cleanup."""
    status = SimpleNamespace(zones={"33N": (2025,)}, has=lambda z, y: True)  # already complete
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zone-year/fill-zone-year"]  # fill (retag) only, no ingest
    assert wired["deletes"] == []


def test_unseeded_explicit_zone_rejected(wired):
    """An explicit zones= entry that isn't seeded fails before dispatch."""
    with pytest.raises(ValueError, match="not seeded"):
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", zones=["02N"]))
    assert wired["arun"] == []


def test_duplicate_years_dispatch_once(wired):
    """The dispatch loop dedupes years — years=(2025, 2025) runs the cell once."""
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", years=(2025, 2025)))
    assert len(wired["arun"]) == 2  # one ingest + one fill, not two of each


def test_sequential_strategy_dispatches_one_run_per_year(wired):
    """fill_strategy="chained-clusters" replaces the per-cell chain with ONE
    fill-zones-sequential run per year: no driver-side ingest (the child's
    look-ahead owns it), no driver-side mosaic cleanup, zones passed as a list.
    """
    result = asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters"))
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zones-sequential/fill-zones-sequential"]
    params = wired["arun"][0][1]
    assert params["zones"] == ["33N"] and params["year"] == 2025
    # The child's ingest look-ahead inherits the driver's ingest bound + params.
    assert params["ingest"] is True and params["look_ahead"] == 2
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
    """max_parallel_zones > 1 in sequential mode = that many chained clusters:
    zones are LPT-partitioned by live-tile count and each shard's child divides
    the global ingest look-ahead bound.
    """
    status = SimpleNamespace(zones={"33N": (), "34N": (), "35N": ()}, has=lambda z, y: False)
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: status)
    monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [("33N", 2025), ("34N", 2025), ("35N", 2025)])
    counts = {"33N": 500, "34N": 300, "35N": 250}
    monkeypatch.setattr(mod, "zone_live_tile_count", lambda mask, zone, **k: counts[zone])

    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters", max_parallel_zones=2
        )
    )
    assert [d for d, _ in wired["arun"]] == ["fill-zones-sequential/fill-zones-sequential"] * 2
    shards = [p["zones"] for _, p in wired["arun"]]
    # LPT: 500 alone; 300+250 together — balanced totals (500 vs 550).
    assert sorted(map(sorted, shards)) == [["33N"], ["34N", "35N"]]
    # The global ingest bound (max_parallel_ingest=2) is divided across shards.
    assert all(p["look_ahead"] == 1 for _, p in wired["arun"])


def test_sequential_single_shard_reads_no_tile_counts(wired, monkeypatch):
    """One work zone → one shard → the partitioner must not read the mask."""

    def boom(*a, **k):
        raise AssertionError("tile counts must not be read for a single shard")

    monkeypatch.setattr(mod, "zone_live_tile_count", boom)
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="chained-clusters"))
    assert len(wired["arun"]) == 1


def test_partition_zero_weights_known_complete_cells(monkeypatch):
    """Retag-only cells cost the child no GPU time, so they must not skew the
    LPT balance — and their mask reads must be skipped entirely.
    """
    counts = {"01N": 100, "02N": 90, "03N": 80}

    def count(mask, zone, **k):
        if zone not in counts:
            raise AssertionError(f"tile count read for known-complete zone {zone}")
        return counts[zone]

    monkeypatch.setattr(mod, "zone_live_tile_count", count)
    shards = mod._partition_by_live_tiles(
        ["01N", "02N", "03N", "04N"], 2, land_mask_path="mask", known_complete={"04N"}
    )
    # LPT over {100, 90, 80}: [01N] vs [02N, 03N]; 04N rides along at zero cost.
    assert sorted(map(sorted, shards)) == [["01N", "04N"], ["02N", "03N"]]


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
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
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
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", branch="global-tessera"))
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
    asyncio.run(
        mod.run_global_campaign.fn(
            paths=_PATHS,
            ami_ssm_name="ami",
            branch="b",
            fill_deployment="custom/fill",
            ingest_deployment="custom/ingest",
        )
    )
    fill_dep, _ = _dep(wired, "custom/fill")
    ingest_dep, ingest_params = _dep(wired, "custom/ingest")
    assert fill_dep == "custom/fill"  # not custom/fill-b
    assert ingest_dep == "custom/ingest"
    assert ingest_params["deployments"]["ingest_s1_roi_sar"].endswith("-b")


def test_branch_routes_fill_when_ingest_disabled(wired):
    """ingest=False: no ingest dispatch, but the fill ref is still branch-routed."""
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False, branch="b"))
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zone-year/fill-zone-year-b"]  # fill only, suffixed


def test_both_strategies_pin_the_resolved_ami_id(wired):
    """The campaign resolves the worker AMI once and pins it into every dispatched
    fill — cluster-per-zone AND chained — so a fill boots the exact image its
    staging fingerprint recorded instead of re-reading the SSM pointer.
    """
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    fill = next(p for d, p in wired["arun"] if "fill-zone-year" in d)
    assert fill["ami_id"] == "ami-test-id"

    wired["arun"].clear()
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
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
        fill = next(p for d, p in wired["arun"] if "fill-zone-year" in d)
        assert fill["run_id"].endswith("-retag")
        assert fill["ami_id"] is None
        # The SSM pointer still travels, so the fill's own fallback stays available.
        assert fill["ami_ssm_name"] == "ami"

    def test_all_ocean_cell_neither_resolves_nor_pins(self, wired, monkeypatch):
        self._no_ssm(monkeypatch)
        monkeypatch.setattr(mod, "zone_has_live_tiles", lambda *a, **k: False)
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
        fill = next(p for d, p in wired["arun"] if "fill-zone-year" in d)
        assert fill["run_id"].endswith("-empty")
        assert fill["ami_id"] is None


def test_every_dispatched_child_carries_the_campaign_tag(wired):
    """The tag is the only thing the terminal hooks can work from.

    Prefect runs them in a fresh process after killing the flow, so nothing in
    memory records what was dispatched — an untagged child is an unsweepable one.
    """
    with patch.object(mod.flow_run_ctx, "id", "run-abc"):
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    assert wired["tags"], "no children dispatched"
    assert all(t == ["campaign:run-abc"] for t in wired["tags"]), wired["tags"]


def test_children_go_untagged_outside_a_flow_run(wired):
    """A direct .fn() call has no run id, and dispatches nothing worth sweeping."""
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
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
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    assert wired["arun"] == [], "nothing may be dispatched once the gate fails"


def test_model_gate_override_lets_the_campaign_proceed(wired, monkeypatch):
    """The escape hatch is threaded through rather than reimplemented."""
    seen: dict = {}
    monkeypatch.setattr(mod, "_assert_seeded_model_matches", lambda *a, **k: seen.update(k))
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", allow_model_mismatch=True))
    assert seen["allow_model_mismatch"] is True
