"""Frisky on a real hijacked cluster: the library behaviour the ingest relies on.

Frisky is pre-1.0 and closed-source, so the bar is higher than for Dask: these pin the library's
own behaviour as well as our wiring, and an upgrade that changes any of it fails here before an
ingest does. Each test also proves its work ran on Frisky, because a compute that quietly fell
back to Dask would pass every equality check. The cluster comes from our own providers
(``local_cluster(frisky=True)`` and ``connect``) with spawned workers, so nothing is inherited from
this process.

Function-scoped on purpose: a live Frisky client reroutes every bare ``.compute()`` in this
process, so a cluster outliving its test would capture other modules' tests on the same xdist
worker.

Covered elsewhere: the read-failure cause chain beside its Dask counterpart, in
``test_read_failure_cause_over_dask.py``, and store equivalence for whole ingests in
``tests/parity/test_ingest_s2_roi_frisky_parity.py``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from typing import Any

import dask
import dask.array as da
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from dask.distributed import Client, WorkerPlugin, get_client
from odc.geo.geobox import GeoBox
from pyproj import CRS

from tessera_embeddings.providers.frisky import connect, maybe_capture_spans
from tessera_embeddings.providers.local.dask import local_cluster
from tessera_embeddings.storage.manifest import IngestManifest
from tessera_embeddings.storage.zarr_store import (
    get_existing_dates,
    open_repo,
    open_store_as_zarr_group,
    write_day_windows,
)

pytestmark = pytest.mark.integration

LOG = logging.getLogger(__name__)


@pytest.fixture
def hijacked() -> Iterator[tuple[Client, Client]]:
    """``(dask_client, client)``: the cluster's Dask client, and the ingest's client after the hijack."""
    with (
        local_cluster(n_workers=2, threads_per_worker=2, dashboard_address=":0", frisky=True) as cluster,
        Client(cluster) as dask_client,
        connect(dask_client) as client,
    ):
        yield dask_client, client


def _scheduler_counts(dask_scheduler: Any) -> tuple[int, int]:
    """Tasks Frisky's scheduler has seen and task transitions Dask's has made (runs on the scheduler)."""
    return dask_scheduler.plugins["frisky-scheduler"].scheduler.total_tasks_seen, dask_scheduler.transition_counter


@contextlib.contextmanager
def _ran_on(dask_client: Client, *, frisky: bool) -> Iterator[None]:
    """Assert the block's tasks reached Frisky, or no cluster scheduler at all; never Dask's."""
    frisky_before, dask_before = dask_client.run_on_scheduler(_scheduler_counts)
    yield
    frisky_after, dask_after = dask_client.run_on_scheduler(_scheduler_counts)
    assert (frisky_after > frisky_before) is frisky, f"Frisky saw {frisky_after - frisky_before} tasks"
    assert dask_after == dask_before, "Dask's scheduler ran tasks that belonged to Frisky"


def test_the_ingests_compute_paths_run_on_frisky_and_agree_with_numpy(hijacked) -> None:
    """Every way the ingest computes: bare ``.compute()``, ``client.compute`` (the S2 coverage
    gate), and ``client.persist`` whose result a later graph reuses (the gate's mask).
    """
    dask_client, client = hijacked
    values = np.random.default_rng(0).random((600, 600))
    x = da.from_array(values, chunks=100)
    with _ran_on(dask_client, frisky=True):
        assert float((x + 1).mean().compute()) == pytest.approx((values + 1).mean())
        assert float(client.compute(x.sum()).result()) == pytest.approx(values.sum())
        mask, total = client.persist([x > 0.5, x.sum()])
        assert int((mask & (x < 0.9)).sum().compute()) == int(((values > 0.5) & (values < 0.9)).sum())
        assert float(total.compute()) == pytest.approx(values.sum())


_THREAD_LOCAL = threading.local()


def _previous_on_this_thread(value: int) -> tuple[int, int | None]:
    """Store ``value`` in a ``threading.local``; return the thread and what its last task left there."""
    previous = getattr(_THREAD_LOCAL, "value", None)
    _THREAD_LOCAL.value = value
    return threading.get_native_id(), previous


def _epsg(crs: CRS) -> int | None:
    return crs.to_epsg()


def test_frisky_threads_keep_their_python_state_between_tasks(hijacked) -> None:
    """Frisky destroys a thread's Python state after each task unless the hijack pins it, and every
    ``threading.local`` cache in the read stack goes with it. pyproj's goes worst: it keeps the freed
    PROJ context's pointer, so a CRS passed as an argument segfaults the next task on its thread.
    """
    _, client = hijacked
    last: dict[int, int] = {}
    for value in range(12):  # more tasks than the cluster has threads, so some thread repeats
        thread, previous = client.submit(_previous_on_this_thread, value).result()
        assert previous == last.get(thread), f"thread {thread} lost its threading.local between tasks"
        last[thread] = value
    assert len(last) < 12

    crs = CRS.from_epsg(32613)
    assert {client.submit(_epsg, crs).result(timeout=30) for _ in range(20)} == {32613}


class _MarkProcess(WorkerPlugin):
    """Stands in for the credential broadcast: a Dask plugin that sets process state."""

    name = "tessera-test-mark"

    def setup(self, worker: object) -> None:
        os.environ["TESSERA_TEST_MARK"] = "set-by-a-dask-plugin"


def _mark_and_pid(_: int) -> tuple[str | None, int]:
    return os.environ.get("TESSERA_TEST_MARK"), os.getpid()


def test_dask_plugins_and_run_reach_the_processes_frisky_tasks_run_in(hijacked) -> None:
    """The credential broadcast and the read-failure capture are Dask plugins, read back with
    ``client.run``; they only work if Frisky's tasks run in the Dask workers' processes.
    """
    _, client = hijacked
    client.register_plugin(_MarkProcess())
    seen = client.gather(client.map(_mark_and_pid, range(8)))
    assert {mark for mark, _ in seen} == {"set-by-a-dask-plugin"}
    assert {pid for _, pid in seen} <= set(client.run(os.getpid).values())


def _driver() -> tuple[int, int, int]:
    """What the Prefect task shell does on a Dask worker: connect, then compute from two threads."""
    with connect(get_client()) as client:
        ones = da.ones((400, 400), chunks=100)
        background: list[int] = []
        thread = threading.Thread(target=lambda: background.append(int(ones.sum().compute())))
        thread.start()
        thread.join()
        return int(ones.sum().compute()), int(client.compute(ones.sum()).result()), background[0]


def test_a_driver_running_as_a_dask_task_computes_on_frisky(hijacked) -> None:
    """The Prefect path, where the ingest itself is a Dask task and ``pipeline_dates`` computes
    from a second thread. Dask schedules the driver, so only Frisky's count is asserted.
    """
    dask_client, _ = hijacked
    frisky_before, _ = dask_client.run_on_scheduler(_scheduler_counts)
    assert dask_client.submit(_driver, pure=False).result(timeout=120) == (160_000,) * 3
    frisky_after, _ = dask_client.run_on_scheduler(_scheduler_counts)
    assert frisky_after > frisky_before


class _Roi:
    geobox = GeoBox((8, 8), Affine(10.0, 0.0, 0.0, 0.0, -10.0, 80.0), "EPSG:32601")
    height = width = 8


def _poison(block: np.ndarray) -> np.ndarray:
    raise RuntimeError("poisoned window")


def _write(store: str, date: str, value: int, blocks: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
    """One date through the ingest's overlapped window write: icechunk's fork and merge reduction.

    Two variables, as every real mosaic has: the overlapped write fails on a one-variable dataset
    under any scheduler.
    """
    band = da.full((1, 8, 8), value, dtype=np.uint16, chunks=(1, 4, 4))
    if blocks is not None:
        band = band.map_blocks(blocks, dtype=np.uint16)
    dims = ("time", "northing", "easting")
    day = xr.Dataset(
        {"band": (dims, band), "scl": (dims, da.full((1, 8, 8), 4, dtype=np.uint8, chunks=(1, 4, 4)))},
        coords={"time": np.array([np.datetime64(date, "ns")])},
    )
    write_day_windows(
        store,
        day,
        [(0, 4, 0, 8), (4, 8, 4, 8)],
        roi=_Roi(),
        manifest=IngestManifest(roi_manifest_hash="abc"),
        baselines={date: 5},
        tile_id="roi.zarr",
        crs="EPSG:32601",
        chunks={"time": 1, "northing": 4, "easting": 4},
        parallel_windows=True,
    )


def test_overlapped_window_writes_match_dask_and_commit_nothing_on_failure(hijacked, tmp_path) -> None:
    """The store write is where an engine could differ silently: forked icechunk sessions are
    pickled to workers and merged back. A failing window must still commit nothing.
    """
    dask_client, _ = hijacked
    dates = (("2024-06-01", 7), ("2024-06-11", 9))
    reference, store = str(tmp_path / "reference"), str(tmp_path / "frisky")
    with _ran_on(dask_client, frisky=False), dask.config.set(scheduler="sync"):
        for date, value in dates:
            _write(reference, date, value)
    with _ran_on(dask_client, frisky=True):
        for date, value in dates:
            _write(store, date, value)
        with pytest.raises(RuntimeError, match="poisoned window"):
            _write(store, "2024-06-21", 3, _poison)

    np.testing.assert_array_equal(
        np.asarray(open_store_as_zarr_group(store)["band"]), np.asarray(open_store_as_zarr_group(reference)["band"])
    )
    assert get_existing_dates(store) == {"2024-06-01", "2024-06-11"}
    snapshots = [len(list(open_repo(s).ancestry(branch="main"))) for s in (store, reference)]
    assert snapshots[0] == snapshots[1], "the poisoned date committed something"


def test_captured_spans_are_what_frisky_observe_reads(hijacked, tmp_path) -> None:
    """The telemetry a Fargate run leaves behind: worker spans, via the Dask dashboard's proxy,
    in the file format the ``frisky observe`` CLI analyses offline.
    """
    dask_client, _ = hijacked
    spans_file = tmp_path / "spans.json"
    with maybe_capture_spans(dask_client.dashboard_link, str(spans_file), LOG):
        da.ones(10_000, chunks=100).sum().compute()

    names = {span["name"] for span in json.loads(spans_file.read_text())}
    assert any(name.startswith("worker.exec") for name in names), sorted(names)
    overview = subprocess.run(
        [sys.executable, "-m", "frisky.cli", "observe", "overview", str(spans_file)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert overview.returncode == 0, overview.stderr


def test_span_capture_never_fails_or_masks_the_run(tmp_path, caplog) -> None:
    """Diagnostics must not change a run's outcome: an unreachable dashboard only warns."""
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(ValueError, match="the run's own failure"),
        maybe_capture_spans("http://127.0.0.1:9", str(tmp_path / "spans.json"), LOG),
    ):
        raise ValueError("the run's own failure")
    assert "failed to capture Frisky spans" in caplog.text
