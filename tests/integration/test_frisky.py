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
import gzip
import json
import logging
import os
import pickle
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import dask
import dask.array as da
import frisky
import numpy as np
import pytest
import xarray as xr
from affine import Affine
from dask.distributed import Client, WorkerPlugin, get_client
from odc.geo.geobox import GeoBox
from pyproj import CRS

from tessera_embeddings.providers import frisky as frisky_engine
from tessera_embeddings.providers.frisky import connect, maybe_capture_telemetry
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


def _nth_epsg(_: int, crs: CRS) -> int | None:
    return crs.to_epsg()


def _hijack_state() -> tuple[bool, bool]:
    """Whether this worker pins thread state and has the ingest imported (runs on a worker)."""
    return getattr(pickle.loads, "_tessera_pins", False), "tessera_embeddings.ingest.s1_roi" in sys.modules


def test_workers_that_join_after_the_hijack_run_its_queued_work() -> None:
    """A Fargate fleet boots after the hijack, and a restarted worker rejoins the same way: work
    queued while no worker exists must run on the late joiners, each pinned and with the ingest
    imported before its Frisky worker takes a task.
    """
    with (
        local_cluster(n_workers=0, threads_per_worker=2, dashboard_address=":0", frisky=True) as cluster,
        Client(cluster) as dask_client,
        connect(dask_client) as client,
        _ran_on(dask_client, frisky=True),
    ):
        futures = client.map(_nth_epsg, range(20), crs=CRS.from_epsg(32613))
        cluster.scale(2)
        assert {future.result(timeout=120) for future in futures} == {32613}
        assert set(dask_client.run(_hijack_state).values()) == {(True, True)}


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


def test_the_telemetry_bundle_is_written_live_and_at_the_end(hijacked, tmp_path, caplog) -> None:
    """What a Fargate run leaves behind, through the Dask dashboard's proxy: live snapshots while
    it runs, then a bundle the ``frisky observe`` CLI reads after the cluster is gone.
    """
    dask_client, _ = hijacked
    bundle = tmp_path / "bundle"
    with (
        caplog.at_level(logging.INFO),
        maybe_capture_telemetry(dask_client.dashboard_link, str(bundle), LOG, interval_s=1),
    ):
        for _ in range(4):  # long enough for live snapshots to land
            da.ones(10_000, chunks=100).sum().compute()
            time.sleep(1)

    assert (bundle / "live" / "overview.json").exists()
    assert "frisky state: workers=2 " in caplog.text
    for name in ("spans.json", "overview.txt", "overview.json", "events.json", "logs.json"):
        assert (bundle / name).exists(), name
    lifecycle = json.loads((bundle / "events.json").read_text())["lifecycle"]["events"]
    assert sorted(e["kind"] for e in lifecycle) == ["worker_added", "worker_added"], "a worker left mid-run"
    names = {span["name"] for span in json.loads((bundle / "spans.json").read_text())}
    assert any(name.startswith("worker.exec") for name in names), sorted(names)
    overview = subprocess.run(
        [sys.executable, "-m", "frisky.cli", "observe", "overview", str(bundle / "spans.json")],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert overview.returncode == 0, overview.stderr


def _held(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


def _span_id(span: dict[str, Any]) -> tuple[str, int]:
    """A span id is numbered per process, so a span is its process and its id."""
    return span["worker"], span["span_id"]


def test_the_span_drain_keeps_every_task_once(hijacked, tmp_path, monkeypatch) -> None:
    """``drain_spans`` keeps the whole run in parts written while it runs: every task exactly once,
    including one running across several drains, and nothing outside the drained names.
    """
    dask_client, client = hijacked
    monkeypatch.setattr(frisky_engine, "SPAN_DRAIN_INTERVAL_S", 0.3)
    monkeypatch.setattr(frisky_engine, "_SPAN_DRAIN_LAG_NS", 100_000_000)
    with maybe_capture_telemetry(dask_client.dashboard_link, str(tmp_path), LOG, interval_s=3600, drain_spans=True):
        assert client.submit(_held, 1.5).result(timeout=30) == 1.5
        for batch in range(3):
            client.gather(client.map(_held, [(batch * 100 + i) * 1e-9 for i in range(100)]))
            time.sleep(0.5)
    buffered = {
        _span_id(span)
        for name in frisky_engine.SPAN_DRAIN_NAMES
        for span in frisky.query_spans(name=name, limit=sys.maxsize, dashboard_url=dask_client.dashboard_link)
    }

    parts = sorted((tmp_path / "spans").glob("part-*.json.gz"))
    spans = [span for part in parts for span in json.loads(gzip.decompress(part.read_bytes()))]
    assert len(parts) > 1
    assert len({_span_id(span) for span in spans}) == len(spans), "a span was written twice"
    assert {_span_id(span) for span in spans} == buffered, "the parts are not everything Frisky kept"
    assert any(span["name"] == "worker.exec.call" and span["duration_ns"] >= 1.5e9 for span in spans)


def test_span_capture_never_fails_or_masks_the_run(tmp_path, caplog) -> None:
    """Diagnostics must not change a run's outcome: an unreachable dashboard only warns."""
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(ValueError, match="the run's own failure"),
        maybe_capture_telemetry("http://127.0.0.1:9", str(tmp_path / "spans.json"), LOG, drain_spans=True),
    ):
        raise ValueError("the run's own failure")
    assert "failed to capture Frisky spans" in caplog.text
    assert "Frisky span drain failed" in caplog.text
