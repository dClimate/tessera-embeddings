"""A run's Sentinel-2 sources arrive as Prefect JSON and are checked before any fleet starts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tessera_embeddings.config.ingest import IngestSettings


def test_sources_from_json_parameters_are_read_per_year() -> None:
    settings = IngestSettings.model_validate({"s2_sources": {"2017": ["planetary-computer", "sentinel-2-l2a"]}})
    assert settings.s2_sources == {2017: ("planetary-computer", "sentinel-2-l2a")}


@pytest.mark.parametrize("source", [["nowhere", "sentinel-2-l2a"], ["earth-search", "sentinel-2-c9"]])
def test_an_unknown_source_is_refused_at_run_start(source: list[str]) -> None:
    with pytest.raises(ValidationError, match="s2_sources\\[2017\\]"):
        IngestSettings(s2_sources={2017: source})
