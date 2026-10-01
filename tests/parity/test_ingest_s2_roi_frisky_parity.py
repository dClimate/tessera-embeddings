"""Parity: running the S2 ingest on Frisky must not change the store.

Frisky (``providers/frisky.py``) is an engine switch, the second kind of parity in this tier's
README: the same domain function on the same inputs, under Dask and under Frisky loaded onto an
identical cluster, compared store against store. Each Frisky run also asserts Frisky scheduled
the work, so a silent fall-back to Dask cannot pass as parity.

Two inputs. The toy dates from ``test_ingest_s2_roi_pipeline_parity`` are offline and
deterministic, and run with ``pipeline_dates`` on, so a second thread computes and a date fails
the coverage gate mid-run. The Denver cassette reads real COGs over the network, which is where an
engine is likeliest to differ: GDAL and rasterio running in Frisky's worker threads rather than
Dask's.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from dask.distributed import Client

from tessera_embeddings.ingest.s2_roi import IngestResult, ingest_s2_roi_reflectance
from tessera_embeddings.providers.frisky import connect
from tessera_embeddings.providers.local.dask import local_cluster
from tests.parity.helpers import assert_zarr_equivalent, stage_quickstart_roi
from tests.parity.test_ingest_s2_roi_parity import CASSETTE_NAME, DENVER_DATES
from tests.parity.test_ingest_s2_roi_pipeline_parity import DATES, _ingest, _stage_roi

pytest.importorskip("frisky")


@pytest.fixture
def default_cassette_name() -> str:
    """Replay the S2 parity test's cassette: one ingest's STAC requests, rewound for the second."""
    return CASSETTE_NAME


def _frisky_tasks_seen(dask_scheduler: Any) -> int:
    return dask_scheduler.plugins["frisky-scheduler"].scheduler.total_tasks_seen


@contextmanager
def _frisky_client() -> Iterator[Client]:
    """The ingest's client on a hijacked cluster shaped like ``parity_cluster``; asserts Frisky ran."""
    with (
        local_cluster(n_workers=2, threads_per_worker=2, memory_limit="2GB", frisky=True) as cluster,
        Client(cluster) as dask_client,
        connect(dask_client) as client,
    ):
        yield client
        assert dask_client.run_on_scheduler(_frisky_tasks_seen) > 0, "the ingest never reached Frisky"


def _on_both(
    run: Callable[[Client, Path], IngestResult],
    dask: Client,
    tmp_path: Path,
    between: Callable[[], None] = lambda: None,
) -> tuple[IngestResult, ...]:
    """``run`` under Dask, then under Frisky, into sibling stores; returns both results."""
    dask_result = run(dask, tmp_path / "dask")
    between()
    with _frisky_client() as client:
        frisky_result = run(client, tmp_path / "frisky")
    return dask_result, frisky_result


@pytest.mark.parity
def test_toy_dates_produce_an_identical_store_on_frisky(
    tmp_path: Path, parity_cluster: Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Byte for byte, with the date counters agreeing first so empty stores cannot pass."""
    roi_zarr = _stage_roi(tmp_path)
    results = _on_both(
        lambda client, store: _ingest(
            roi_zarr=roi_zarr, store_path=store, client=client, monkeypatch=monkeypatch, pipeline_dates=True
        ),
        parity_cluster,
        tmp_path,
    )
    assert {(r.status, r.dates_processed, r.dates_filtered_coverage) for r in results} == {
        ("success", len(DATES) - 1, 1)
    }
    assert_zarr_equivalent(tmp_path / "frisky" / "reflectance.zarr", tmp_path / "dask" / "reflectance.zarr")


@pytest.mark.parity
@pytest.mark.integration
@pytest.mark.vcr
def test_real_imagery_produces_an_equivalent_store_on_frisky(
    tmp_path: Path, fixture_quickstart_roi: Path, parity_cluster: Client, vcr: Any
) -> None:
    """Denver, July 2024, with the S2 parity test's cassette and its float32-ULP tolerance.

    The cassette is rewound between the runs rather than allowed to repeat, which would break
    pagination (see ``vcr_config``): each run replays the recordings once, in order.
    """
    roi_zarr = stage_quickstart_roi(tmp_path, fixture_quickstart_roi)
    results = _on_both(
        lambda client, store: ingest_s2_roi_reflectance(
            roi_zarr_path=str(roi_zarr),
            start_date=DENVER_DATES[0],
            end_date=DENVER_DATES[1],
            store_path=str(store),
            client=client,
            log=logging.getLogger("parity-s2-frisky"),
        ),
        parity_cluster,
        tmp_path,
        between=vcr.rewind,
    )
    assert [r.status for r in results] == ["success", "success"], results
    assert results[0].dates_processed == results[1].dates_processed > 0
    assert_zarr_equivalent(
        tmp_path / "frisky" / "reflectance.zarr", tmp_path / "dask" / "reflectance.zarr", rtol=1e-6, atol=1e-6
    )
