"""Unit tests for inference/orchestration_helpers.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tessera_embeddings.inference.orchestration_helpers import read_upstream_manifests


@pytest.fixture()
def _mock_stores():
    """Mock the raw-zarr opener and extract_manifest for manifest reading tests."""
    fake_manifest = {"version": "1.0", "dates": ["2024-01-01"]}
    mock_ds = MagicMock()
    mock_ds.attrs = {"_manifest": fake_manifest}

    with (
        patch(
            "tessera_embeddings.inference.orchestration_helpers.open_store_as_zarr_group",
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

    @pytest.mark.parametrize(
        ("orbit", "stores"),
        [
            ("both", {"reflectance", "sar_ascending", "sar_descending"}),
            ("ascending", {"reflectance", "sar_ascending"}),
            ("descending", {"reflectance", "sar_descending"}),
        ],
    )
    def test_reads_reflectance_and_each_selected_orbit(self, _mock_stores, orbit, stores):
        assert set(read_upstream_manifests("/tmp/mosaic", orbit)) == stores
