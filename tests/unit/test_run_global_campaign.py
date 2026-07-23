"""run_global_campaign: ingest → fill → cleanup chaining per cell.

The work-list logic itself is covered in test_campaign (campaign_work_list); here
the flow body is driven via ``.fn`` with mocked dispatch to check the per-cell
sequence, the ingest bypass, and mosaic cleanup.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

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
    # Store reads for the staging fingerprint: the coverage delivery sha (global per
    # delivery) and, for ingest=False, each mosaic store's `last_appended` identity.
    monkeypatch.setattr(
        mod,
        "open_store_as_zarr_group",
        lambda *a, **k: SimpleNamespace(
            attrs={"registry_sha256": "cov-sha-test", "last_appended": "2026-01-01T00:00:00Z"}
        ),
    )

    async def fake_arun(dep, parameters=None):
        rec["arun"].append((dep, parameters))
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
    changed = _fill_run_id(wired, min_valid_coverage=0.5)
    assert changed.startswith("33N-2025-") and changed != base  # input change → new prefix


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
        return SimpleNamespace(attrs={"registry_sha256": "cov", "last_appended": "2026-01-01T00:00:00Z"})

    monkeypatch.setattr(mod, "open_store_as_zarr_group", _capture)
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

    monkeypatch.setattr(mod, "open_store_as_zarr_group", _boom)
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
            "open_store_as_zarr_group",
            lambda *a, **k: SimpleNamespace(attrs={"registry_sha256": "cov", "last_appended": ts}),
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

    monkeypatch.setattr(mod, "open_store_as_zarr_group", _boom)
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami"))
    fill = next(p for d, p in wired["arun"] if d == "fill-zone-year/fill-zone-year")
    assert fill["run_id"] == "33N-2025-retag"


def test_ingest_false_fails_closed_without_mosaic_identity(wired, monkeypatch):
    """ingest=False fails closed when a prebuilt mosaic store carries no identity
    attr — better than fingerprinting a partial view and risking a stale resume.
    """
    monkeypatch.setattr(mod, "open_store_as_zarr_group", lambda *a, **k: SimpleNamespace(attrs={}))
    with pytest.raises(RuntimeError, match="last_appended"):
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False, cleanup_mosaics=False))


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
