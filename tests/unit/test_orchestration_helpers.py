"""Unit tests for inference/orchestration_helpers.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tessera_embeddings.inference.orchestration_helpers import read_upstream_manifests


@pytest.fixture()
def _mock_stores():
    """Mock open_store and extract_manifest for manifest reading tests."""
    fake_manifest = {"version": "1.0", "dates": ["2024-01-01"]}
    mock_ds = MagicMock()
    mock_ds.attrs = {"_manifest": fake_manifest}

    with (
        patch(
            "tessera_embeddings.inference.orchestration_helpers.open_store",
            return_value=mock_ds,
        ),
        patch(
            "tessera_embeddings.inference.orchestration_helpers.extract_manifest",
            return_value=fake_manifest,
        ) as mock_extract,
    ):
        yield {"extract": mock_extract, "ds": mock_ds}


class TestReadUpstreamManifests:
    """Verify read_upstream_manifests handles all orbit values."""

    def test_both_includes_both_sar_stores(self, _mock_stores):
        result = read_upstream_manifests("/tmp/mosaic", "both")
        assert "reflectance" in result
        assert "sar_ascending" in result
        assert "sar_descending" in result
        assert len(result) == 3

    def test_ascending_includes_only_ascending(self, _mock_stores):
        result = read_upstream_manifests("/tmp/mosaic", "ascending")
        assert "sar_ascending" in result
        assert "sar_descending" not in result
        assert len(result) == 2

    def test_descending_includes_only_descending(self, _mock_stores):
        result = read_upstream_manifests("/tmp/mosaic", "descending")
        assert "sar_descending" in result
        assert "sar_ascending" not in result
        assert len(result) == 2
