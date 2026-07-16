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
    monkeypatch.setattr(mod, "campaign_status", lambda *a, **k: object())
    monkeypatch.setattr(mod, "campaign_work_list", lambda *a, **k: [("33N", 2025)])
    monkeypatch.setattr(mod, "tag_year_complete", _defer_milestone)

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


def test_ingest_disabled_skips_ingest_and_cleanup(wired):
    asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", ingest=False, cleanup_mosaics=False))
    deps = [d for d, _ in wired["arun"]]
    assert deps == ["fill-zone-year/fill-zone-year"]  # fill only
    assert wired["deletes"] == []


def test_rejects_zero_parallel_ingest(wired):
    with pytest.raises(ValueError, match="max_parallel_ingest"):
        asyncio.run(mod.run_global_campaign.fn(paths=_PATHS, ami_ssm_name="ami", max_parallel_ingest=0))
