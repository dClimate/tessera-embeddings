"""Unit tests for orchestration/runners/plain.py — orbit decomposition."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from tessera_embeddings.config.paths import BucketPaths
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


# ── the pre-flight, which the Prefect flow had and this runner did not ──


def test_the_runner_preflights_the_output_store_before_inferring(monkeypatch, tmp_path) -> None:
    """The encoder mismatch is decided by the store and this config alone — nothing inference
    discovers — so finding it in `assemble` means every chunk was inferred and none of it can be
    published. Ordering is the whole assertion: the gate must fire with no inference behind it.
    """
    import tessera_embeddings.orchestration.runners.plain as plain_mod

    order: list[str] = []

    class _StopError(Exception):
        """Ends the run once the gate under test has fired."""

    def _preflight(**kwargs):
        order.append("preflight")
        raise _StopError

    def _inference(**kwargs):
        order.append("inference")
        return []

    monkeypatch.setattr(plain_mod, "assert_output_store_accepts", _preflight)
    monkeypatch.setattr(plain_mod, "run_inference", _inference)
    # Orbit resolution and chunk enumeration precede the gate. Both are metadata reads, so the
    # gate still runs before anything expensive; they are stubbed only because this test has
    # no mosaic stores on disk.
    monkeypatch.setattr(plain_mod, "resolve_s1_orbit", lambda *a, **k: "both")
    monkeypatch.setattr(plain_mod, "enumerate_mosaic_chunks", lambda *a, **k: ([], 8, 8))
    monkeypatch.setattr(plain_mod, "filter_chunks_by_roi_mask", lambda chunks, roi: chunks)

    with pytest.raises(_StopError):
        plain_mod._run_inference_and_assemble(
            roi_path=str(tmp_path / "roi.zarr"),
            roi_name="demo",
            paths=BucketPaths(inputs=str(tmp_path / "in"), outputs=str(tmp_path / "out")),
            time_window_end="June 2025",
            s1_orbit="both",
            checkpoint_dir=None,
            checkpoint_url=None,
            model_version="v2-large",
            num_gpus=0,
            log=logging.getLogger("test-plain"),
        )

    assert order == ["preflight"], "the gate must run before any inference, not after it"


def test_the_run_id_prefix_helper_is_public() -> None:
    """`run_inference` is public and refuses an unprefixed id under v2, so the rule for building
    a valid id cannot itself be internal — that made the advertised direct v2 path depend on an
    undocumented convention.
    """
    import tessera_embeddings as pkg

    assert "run_id_prefix" in pkg.__all__
    assert pkg.run_id_prefix("v2-large") == "v2-"
    assert pkg.run_id_prefix("v1.1") == ""
