"""Unit tests for orchestration/runners/plain.py — orbit decomposition."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from tessera_embeddings.orchestration.runners.plain import _run_ingest


@pytest.fixture()
def _mock_ingest_deps():
    """Mock Dask client, S2 ingest, and S1 ingest for _run_ingest tests."""
    mock_client = MagicMock()
    mock_client.scheduler_info.return_value = {"address": "tcp://localhost:1234"}

    with (
        patch("tessera_embeddings.orchestration.runners.plain._dask_client") as mock_dask_ctx,
        patch("tessera_embeddings.orchestration.runners.plain.ingest_s2_roi_reflectance") as mock_s2,
        patch("tessera_embeddings.orchestration.runners.plain.ingest_s1_roi_sar") as mock_s1,
    ):
        mock_dask_ctx.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_dask_ctx.return_value.__exit__ = MagicMock(return_value=False)
        mock_s2.return_value = MagicMock()
        mock_s1.return_value = MagicMock()
        yield {"s2": mock_s2, "s1": mock_s1}


def _common_kwargs() -> dict:
    paths = MagicMock()
    paths.store_for.return_value = "/tmp/mosaics/reflectance.zarr"
    return dict(
        roi_path="/tmp/roi.zarr",
        paths=paths,
        roi_name="test-roi",
        start_date="2024-01-01",
        end_date="2024-07-01",
        n_workers=2,
        log=logging.getLogger("test"),
        storage_options=None,
        s1_use_s3_direct=False,
    )


class TestRunIngestOrbitDecomposition:
    """Verify _run_ingest decomposes s1_orbit='both' into two ingest calls."""

    def test_both_calls_ingest_s1_twice(self, _mock_ingest_deps):
        _run_ingest(**_common_kwargs(), s1_orbit="both")

        mock_s1 = _mock_ingest_deps["s1"]
        assert mock_s1.call_count == 2
        orbits_called = [call.kwargs["orbit"] for call in mock_s1.call_args_list]
        assert orbits_called == ["ascending", "descending"]

    def test_ascending_calls_ingest_s1_once(self, _mock_ingest_deps):
        _run_ingest(**_common_kwargs(), s1_orbit="ascending")

        mock_s1 = _mock_ingest_deps["s1"]
        assert mock_s1.call_count == 1
        assert mock_s1.call_args.kwargs["orbit"] == "ascending"

    def test_descending_calls_ingest_s1_once(self, _mock_ingest_deps):
        _run_ingest(**_common_kwargs(), s1_orbit="descending")

        mock_s1 = _mock_ingest_deps["s1"]
        assert mock_s1.call_count == 1
        assert mock_s1.call_args.kwargs["orbit"] == "descending"

    def test_s2_always_called_once(self, _mock_ingest_deps):
        _run_ingest(**_common_kwargs(), s1_orbit="both")
        assert _mock_ingest_deps["s2"].call_count == 1
