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
    monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [("33N", 2025)])
    monkeypatch.setattr(mod, "tag_year_complete", _defer_milestone)
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


def test_sequential_strategy_dispatches_one_run_per_year(wired):
    """fill_strategy="sequential" replaces the per-cell chain with ONE
    fill-zones-sequential run per year: no driver-side ingest (the child's
    look-ahead owns it), no driver-side mosaic cleanup, zones passed as a list.
    """
    result = asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="sequential"))
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
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="sequential", ingest=False))
    assert wired["arun"][0][1]["ingest"] is False


def test_invalid_fill_strategy_rejected(wired):
    with pytest.raises(ValueError, match="fill_strategy"):
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", fill_strategy="both"))
    assert wired["arun"] == []
