"""Shared fixtures for the Prefect flow tests."""

from __future__ import annotations

import pytest

from tessera_embeddings.orchestration.prefect.flows import fill_zone_year, fill_zones_sequential, run_global_campaign


@pytest.fixture(autouse=True)
def _store_holds_v11(monkeypatch: pytest.MonkeyPatch) -> None:
    """The store each fill flow reads its model from holds v1.1, unless a test patches it otherwise.

    The flows resolve the model from the store before anything else, so without this every test
    of theirs would need a real seeded store just to get past that read.
    """
    for module in (fill_zone_year, fill_zones_sequential, run_global_campaign):
        monkeypatch.setattr(module, "store_model_version", lambda *_args, **_kwargs: "v1.1")
