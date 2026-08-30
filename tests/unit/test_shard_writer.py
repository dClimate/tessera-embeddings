"""Shard-aligned land-masked writer + commit discipline (W3)."""

from __future__ import annotations

import inspect
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import Future

import icechunk
import numpy as np
import pytest
import zarr

from tessera_embeddings.config.fault_injection import DIE_BETWEEN_COMMITS, DRILL_EXIT_STATUS, FaultInjection
from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.inference import assembly
from tessera_embeddings.storage import (
    global_store,
    icechunk_logging,
    session_catch_up,
    shard_writer,
    zarr_store,
)
from tessera_embeddings.storage.icechunk_logging import COMMIT_LOG_FILTER, commit_tracing
from tessera_embeddings.storage.shard_writer import (
    PhaseTimer,
    _await_forks,
    _write_shards_worker,
    commit_with_rebase,
    run_forked,
    write_year_shards,
)
from tessera_embeddings.storage.zone_grid import ZoneSpec

_BAND = 8  # small band for a light test
_SHARD = 512
_CHUNK = 256

# Small layout: 512^2 shards of 256^2 inner chunks - whole-shard blocks are a few MB.
_EMB = ArrayLayout(DIMS_4D, (1, _CHUNK, _CHUNK, _BAND), "int8", 0, "zstd", shards=(1, _SHARD, _SHARD, _BAND))
_SCL = ArrayLayout(DIMS_3D, (1, _CHUNK, _CHUNK), "float32", float("nan"), "pcodec", shards=(1, _SHARD, _SHARD))
SMALL = StoreLayout(name="small", arrays={"embeddings": _EMB, "scales": _SCL})

# Zone spanning 2 shards tall x 1 wide (1024 x 512 px).
_ZONE = ZoneSpec("32601", "N", 1, (0.0, 5_120.0), (0.0, 10_240.0))
_ZONE_B = ZoneSpec("32701", "S", 1, (0.0, 5_120.0), (1_105_920.0, 1_116_160.0))


class _OneInnerChunkSource:
    """Writes shard (0,0): one 256^2 inner chunk of real data, the rest fill."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def live_shards(self):
        return [(0, 0)]

    def load(self, shard):
        rng = np.random.default_rng(self.seed)
        emb = np.zeros((1, _SHARD, _SHARD, _BAND), dtype="int8")
        scl = np.full((1, _SHARD, _SHARD), np.nan, dtype="float32")
        emb[0, 0:_CHUNK, 0:_CHUNK, :] = rng.integers(-127, 128, size=(_CHUNK, _CHUNK, _BAND), dtype="int8")
        scl[0, 0:_CHUNK, 0:_CHUNK] = rng.random((_CHUNK, _CHUNK), dtype="float32")
        return {"embeddings": emb, "scales": scl}


#: How long `_SlowLoadSource` stalls. Sized to DOMINATE the real write it is compared
#: against, not merely to be measurable: the assertion is `write_s < read_s`, and the write
#: is unbounded real work whose duration depends on the runner. At 0.05 s this flaked on CI
#: with read_s=0.054 against write_s=0.068 — a contended runner made the genuine write take
#: longer than the injected stall, failing a test whose subject is attribution, not speed.
_STALL_S = 0.3


class _SlowLoadSource(_OneInnerChunkSource):
    """A source whose load stalls, standing in for a slow staged-tile GET."""

    def load(self, shard):
        time.sleep(_STALL_S)
        return super().load(shard)


def _seed(tmp_path, zones=(_ZONE,)):
    # band is derived from the SMALL layout's embeddings band chunk (_BAND).
    store = str(tmp_path / "g.icechunk")
    repo = global_store.create_global_repo(store)
    global_store.seed_zone_groups(repo, zones, years=(2023, 2024, 2025), layout=SMALL)
    return store, repo


def test_writes_one_shard_land_masked(tmp_path):
    store, repo = _seed(tmp_path)
    write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(seed=1), n_workers=1, shard_px=_SHARD)

    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    expected = np.random.default_rng(1).integers(-127, 128, size=(_CHUNK, _CHUNK, _BAND), dtype="int8")
    assert np.array_equal(g["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :], expected)
    # ocean within the written shard is elided -> reads as fill
    assert (g["embeddings"][2, 300:512, 300:512, :] == 0).all()
    assert np.isnan(g["scales"][2, 300:512, 300:512]).all()
    # the OTHER shard (rows 512:1024) was never written
    assert (g["embeddings"][2, 512:1024, 0:_CHUNK, :] == 0).all()
    # exactly one 512^2 shard touched (== 4 inner-chunk positions); the second
    # shard's 4 inner chunks stay uninitialized. (Ocean bytes within the written
    # shard are elided by the codec - verified at scale in d3v2 E4.)
    assert g["embeddings"].nchunks_initialized == 4


def test_years_complete_updated_in_commit(tmp_path):
    store, repo = _seed(tmp_path)
    write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    assert g.attrs["years_complete"] == [2025]


def test_other_years_stay_empty(tmp_path):
    store, repo = _seed(tmp_path)
    write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)
    g = zarr_store.open_store_as_zarr_group(store, group="01N")
    assert (g["embeddings"][0, 0:_CHUNK, 0:_CHUNK, :] == 0).all()  # 2023 untouched


def test_commit_with_rebase_resolves_concurrent_disjoint_commits(tmp_path):
    store, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
    # Two sessions from the same tip write disjoint groups; both must commit.
    s1 = repo.writable_session("main")
    s2 = repo.writable_session("main")
    zarr.open_group(s1.store, mode="a")["01N"]["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :] = 1
    zarr.open_group(s2.store, mode="a")["01S"]["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :] = 2
    id1 = commit_with_rebase(s1, "write zone A")
    id2 = commit_with_rebase(s2, "write zone B")  # tip moved -> auto-rebase
    assert id1 and id2 and id1 != id2


class TestPhaseTimer:
    """The wall/CPU phase accumulator behind the ASSEMBLY_SUMMARY record."""

    def test_spans_with_the_same_name_accumulate(self):
        # Per-tile spans re-enter the same phase thousands of times; a timer
        # that RESET on re-entry would report only the last tile and the whole
        # aggregation would silently understate every phase.
        timer = PhaseTimer()
        with timer.phase("read"):
            time.sleep(0.02)
        with timer.phase("read"):
            time.sleep(0.02)
        assert timer.stats()["read_s"] >= 0.035

    def test_a_blocked_span_reports_wall_time_without_cpu_time(self):
        # The whole point of the two clocks: a span blocked on I/O (here, a
        # sleep) must show wall time with near-zero CPU, or blocked-on-S3 time
        # would be indistinguishable from compression time.
        timer = PhaseTimer()
        with timer.phase("write"):
            time.sleep(0.05)
        stats = timer.stats()
        assert stats["write_s"] >= 0.045
        assert stats["write_cpu_s"] < stats["write_s"] / 2

    def test_phases_do_not_bleed_into_each_other(self):
        timer = PhaseTimer()
        with timer.phase("read"):
            time.sleep(0.03)
        with timer.phase("write"):
            pass
        stats = timer.stats()
        assert stats["read_s"] >= 0.025
        assert stats["write_s"] < 0.01

    def test_overall_wall_covers_time_outside_any_phase(self):
        # wall_s runs from construction, so per-phase walls plus the un-phased
        # residue account for the worker's whole life — the invariant that lets
        # a summary reader attribute every second of a slow worker.
        timer = PhaseTimer()
        with timer.phase("read"):
            time.sleep(0.01)
        time.sleep(0.02)  # un-phased residue
        stats = timer.stats()
        assert stats["wall_s"] >= stats["read_s"] + 0.015


class TestForkTelemetry:
    """run_forked and write_year_shards report what the fill did, not just that it did."""

    def test_run_forked_returns_worker_stats_in_payload_order(self, tmp_path):
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        result = run_forked(session, lambda p: (p["fork"], {"tag": p["tag"]}), [{"tag": "only"}])
        assert result["workers"] == [{"tag": "only"}]
        assert result["wall_s"] >= result["merge_s"] >= 0

    def test_write_year_shards_fills_the_telemetry_out_param(self, tmp_path):
        store, repo = _seed(tmp_path)
        telemetry: dict = {}
        write_year_shards(
            repo,
            "01N",
            year_index=2,
            source=_OneInnerChunkSource(),
            n_workers=1,
            shard_px=_SHARD,
            telemetry=telemetry,
        )
        (worker,) = telemetry["workers"]
        assert worker["tiles"] == 1
        assert worker["writes"] == 2  # embeddings + scales
        # Uncompressed bytes handed to zarr: one whole shard per variable.
        assert worker["bytes"] == (_SHARD * _SHARD * _BAND) + (_SHARD * _SHARD * 4)
        assert worker["read_s"] >= 0 and worker["write_s"] >= 0
        assert telemetry["commit_s"] >= 0 and telemetry["attrs_commit_s"] >= 0
        assert telemetry["fill_wall_s"] >= telemetry["merge_s"]

    def test_no_telemetry_param_changes_nothing(self, tmp_path):
        # The out-param is optional by design: every existing caller that does
        # not ask for telemetry must keep getting a plain snapshot id back.
        store, repo = _seed(tmp_path)
        sid = write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)
        assert isinstance(sid, str) and sid

    def test_a_stalled_load_is_attributed_to_the_read_phase(self, tmp_path):
        # A slow source must surface in read_s, not write_s: attributing it to
        # the write would send an operator chasing S3 PUT concurrency when the
        # staged GETs are the bottleneck — the exact confusion the phase split
        # exists to prevent.
        store, repo = _seed(tmp_path)
        telemetry: dict = {}
        write_year_shards(
            repo,
            "01N",
            year_index=2,
            source=_SlowLoadSource(),
            n_workers=1,
            shard_px=_SHARD,
            telemetry=telemetry,
        )
        (worker,) = telemetry["workers"]
        assert worker["read_s"] >= _STALL_S * 0.9
        assert worker["write_s"] < worker["read_s"]
        # The stall is blocked time, so it must NOT appear as read CPU either.
        assert worker["read_cpu_s"] < worker["read_s"]


class TestForkProgressReporting:
    """A forked write reports what is outstanding, on a timer, without reordering forks.

    Exercises :func:`_await_forks` against plain futures rather than a real process
    pool: the behaviour under test is the coordinator's waiting policy, and driving it
    with resolvable futures makes the timing deterministic instead of dependent on how
    fast a spawned interpreter starts.
    """

    @staticmethod
    def _resolved(value):
        future: Future = Future()
        future.set_result(value)
        return future

    @staticmethod
    def _progress_lines(caplog):
        return [r.getMessage() for r in caplog.records if "Assembly progress" in r.getMessage()]

    def test_results_follow_submission_order_not_completion_order(self):
        # The slow fork is submitted FIRST but finishes LAST; merge order must still be
        # the caller's band order, so a completion-ordered collect would fail here.
        slow: Future = Future()
        quick = self._resolved("second")
        threading.Timer(0.05, lambda: slow.set_result("first")).start()
        assert _await_forks([slow, quick], 10.0) == ["first", "second"]

    def test_progress_is_reported_on_a_timer_not_only_on_completions(self, caplog):
        # ONE outstanding fork, so there is no completion to report until the very end.
        # Per-completion reporting would leave this silent — which is the failure mode
        # the timer exists to remove.
        outstanding: Future = Future()
        threading.Timer(0.15, lambda: outstanding.set_result("done")).start()
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.storage.shard_writer"):
            assert _await_forks([outstanding], 0.01) == ["done"]
        lines = self._progress_lines(caplog)
        assert len(lines) >= 2, f"a single long band reported {len(lines)} progress line(s)"
        assert "1/1 partitions outstanding" in lines[0]

    def test_nothing_outstanding_reports_nothing(self, caplog):
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.storage.shard_writer"):
            assert _await_forks([self._resolved(1), self._resolved(2)], 0.01) == [1, 2]
        assert self._progress_lines(caplog) == []

    def test_a_failed_fork_still_raises(self):
        # Waiting must not swallow a band failure into a partial merge.
        failed: Future = Future()
        failed.set_exception(RuntimeError("band 0 failed"))
        with pytest.raises(RuntimeError, match="band 0 failed"):
            _await_forks([failed, self._resolved("ok")], 10.0)

    def test_a_failed_fork_terminates_the_workers_still_running(self, tmp_path, monkeypatch):
        """`cancel_futures` reaches only work that has not STARTED.

        Without a second step a multi-hour shard writer keeps running — and keeps writing
        fork objects — after the coordinator has raised, and Python's executor atexit hook
        then blocks interpreter shutdown on it. A retry in the same process would overlap the
        previous attempt.

        Safe to kill because nothing a worker has written is in the store: a fork joins the
        repository only when the coordinator merges and commits, and neither happens here.
        Icechunk documents dropping a ForkSession without merging as the way to orphan chunks
        deliberately, so the cost is unreferenced objects that GC reclaims.
        """

        class _Proc:
            def __init__(self, alive):
                self._alive, self.terminated = alive, False

            def is_alive(self):
                return self._alive

            def terminate(self):
                self.terminated = True

        running, finished = _Proc(True), _Proc(False)

        class _FakeExecutor:
            def __init__(self, *a, **k):
                self._processes = {1: running, 2: finished}
                self.shutdown_args = None

            def submit(self, fn, payload):
                f: Future = Future()
                f.set_exception(RuntimeError("worker died"))
                return f

            def shutdown(self, **kwargs):
                self.shutdown_args = kwargs
                # CPython drops the reference here, to release file descriptors. The stub must
                # do it too: without this the test passes against code that reads `_processes`
                # AFTER shutdown, which is exactly the bug it is meant to catch.
                self._processes = None

        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", _FakeExecutor)
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")

        with pytest.raises(RuntimeError, match="worker died"):
            run_forked(session, lambda p: p, [{"tag": "a"}, {"tag": "b"}])

        assert running.terminated, "a worker still running must be terminated, not left to finish"
        assert not finished.terminated, "an already-finished worker needs no signal"

    def test_progress_line_names_the_callers_unit(self, caplog):
        # A payload is a band for one caller and a round-robin tile partition for
        # another; a fixed noun misdescribes the work for one of them. The caller
        # names it, and the default is the neutral "partitions".
        def _one_line(**kwargs):
            outstanding: Future = Future()
            threading.Timer(0.05, lambda: outstanding.set_result("done")).start()
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="tessera_embeddings.storage.shard_writer"):
                _await_forks([outstanding], 0.01, **kwargs)
            return self._progress_lines(caplog)[0]

        assert "band writes outstanding" in _one_line(unit="band writes")
        default_line = _one_line()
        assert "partitions outstanding" in default_line
        assert "band writes" not in default_line

    def test_progress_lines_go_to_the_callers_logger(self, caplog):
        # The module logger reaches only the process's own log stream; the whole
        # point of the log parameter is that a flow can pass its run logger and
        # make the wait visible to the orchestrator. Lines emitted on the module
        # logger instead would silently undo that routing.
        outstanding: Future = Future()
        threading.Timer(0.05, lambda: outstanding.set_result("done")).start()
        with caplog.at_level(logging.INFO, logger="test.callers.logger"):
            _await_forks([outstanding], 0.01, log=logging.getLogger("test.callers.logger"))
        progress = [r for r in caplog.records if "Assembly progress" in r.getMessage()]
        assert progress, "no progress line was captured at all"
        assert {r.name for r in progress} == {"test.callers.logger"}


class TestWorkerProgressReporting:
    """A worker reports its own within-partition progress, on a timer.

    The coordinator only sees whole payloads complete, so on a real fill its
    outstanding-count line cannot move until the end — the worker's own lines are
    the only in-flight progress signal. Driven by calling the worker body directly,
    in-process, against a real (throwaway) fork: the behaviour under test is the
    worker's reporting policy, not the process pool.
    """

    @staticmethod
    def _payload(repo, *, interval, shards=((0, 0), (1, 0)), worker_index=3):
        session = repo.writable_session("main")
        return {
            "fork": session.fork(),
            "group": "01N",
            "year_index": 2,
            "shard_px": _SHARD,
            "source": _SlowLoadSource(),  # each load outlasts a tiny interval
            "shards": list(shards),
            "worker_index": worker_index,
            "progress_interval_s": interval,
        }

    @staticmethod
    def _worker_lines(caplog):
        return [r.getMessage() for r in caplog.records if "Assembly worker" in r.getMessage()]

    def test_worker_reports_done_and_total_between_shards(self, tmp_path, caplog):
        store, repo = _seed(tmp_path)
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.storage.shard_writer"):
            _write_shards_worker(self._payload(repo, interval=0.01))
        lines = self._worker_lines(caplog)
        assert lines, "a multi-shard partition produced no worker progress line"
        # After the first slow shard the timer has expired, so the first line
        # must say one of two shards is written — the done/total pair a reader
        # aggregates across workers.
        assert "1/2 shards written" in lines[0]
        # The worker names itself by its payload index: separate processes share
        # no counter, so the index is what makes the lines aggregatable.
        assert "worker 3" in lines[0]

    def test_worker_stays_quiet_within_the_interval(self, tmp_path, caplog):
        # The report is a TIMER, not an every-N-shards cadence: a short write
        # must stay silent rather than narrate every shard.
        store, repo = _seed(tmp_path)
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.storage.shard_writer"):
            _write_shards_worker(self._payload(repo, interval=300.0))
        assert self._worker_lines(caplog) == []

    def test_run_forked_ships_index_and_interval_to_workers(self, tmp_path):
        # The worker's clock comes from the coordinator via the payload; if the
        # injection is dropped the worker silently falls back to the default
        # interval and a caller-tuned period never takes effect.
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        result = run_forked(
            session,
            lambda p: (p["fork"], {"idx": p["worker_index"], "interval": p["progress_interval_s"]}),
            [{}],
            progress_interval_s=1.25,
        )
        assert result["workers"] == [{"idx": 0, "interval": 1.25}]

    def test_write_year_shards_threads_log_and_unit_through(self, tmp_path, monkeypatch):
        # write_year_shards' payloads are tile partitions, and its coordinator
        # lines must go to the CALLER's logger (the flow's run logger under
        # Prefect) — both die silently if the pass-through is dropped.
        store, repo = _seed(tmp_path)
        captured: dict = {}
        real = shard_writer.run_forked

        def spy(session, worker_fn, payloads, **kwargs):
            captured.update(kwargs)
            return real(session, worker_fn, payloads, **kwargs)

        monkeypatch.setattr(shard_writer, "run_forked", spy)
        marker = logging.getLogger("test.fill.logger")
        write_year_shards(
            repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD, log=marker
        )
        assert captured["log"] is marker
        assert captured["unit"] == "tile partitions"


class TestTheDrillDeathBetweenTheTwoCommits:
    """The one state no operator can manufacture: shards landed, year unmarked.

    A zone-year lands in two commits, and the interesting failure is a death
    between them. The gap is bounded by two lines of one function, so there is
    nothing outside the process to aim a kill at — which is why the drill is
    injected rather than performed, and why what follows checks the state left
    behind rather than only that something exited.
    """

    @staticmethod
    def _armed(zone: str = "01N", year: int = 2025):
        return FaultInjection(fault=DIE_BETWEEN_COMMITS, zone=zone, year=year).arm(
            ssm_prefix="/global-tessera-dev/ray/", supports=(DIE_BETWEEN_COMMITS,), log=logging.getLogger("drill")
        )

    @staticmethod
    def _intercept_the_exit(monkeypatch) -> list:
        """Replace the hard exit with a raise, recording the order of what ran.

        The production path flushes handlers and then leaves the process, and
        neither half can happen inside a test runner — the flush would close
        pytest's own handlers and the exit would take the session with it. Both
        are intercepted, and the ORDER is recorded, because an exit before the
        flush loses the announcement that says the damage was deliberate.
        """
        events: list = []

        def _flushed() -> None:
            events.append("flushed")

        def _exited(status: int) -> None:
            events.append(("exited", status))
            raise SystemExit(status)

        monkeypatch.setattr(logging, "shutdown", _flushed)
        monkeypatch.setattr(os, "_exit", _exited)
        return events

    def test_the_shards_land_and_the_year_stays_unmarked(self, tmp_path, monkeypatch):
        store, repo = _seed(tmp_path)
        events = self._intercept_the_exit(monkeypatch)

        with pytest.raises(SystemExit) as exc:
            write_year_shards(
                repo,
                "01N",
                year_index=2,
                source=_OneInnerChunkSource(seed=1),
                n_workers=1,
                shard_px=_SHARD,
                fault=self._armed(),
            )

        assert exc.value.code == DRILL_EXIT_STATUS, "the status must name the drill, not look like an OOM kill"
        assert events == ["flushed", ("exited", DRILL_EXIT_STATUS)], "announce, flush, THEN leave"
        g = zarr_store.open_store_as_zarr_group(store, group="01N")
        expected = np.random.default_rng(1).integers(-127, 128, size=(_CHUNK, _CHUNK, _BAND), dtype="int8")
        assert np.array_equal(g["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :], expected), "the shard commit must have landed"
        assert g.attrs["years_complete"] == [], "and nothing may mark the year complete"
        assert "runs" not in dict(g.attrs), "nor record a run against it"

    def test_a_fault_aimed_at_another_cell_leaves_this_one_alone(self, tmp_path, monkeypatch):
        # The fault is a hard death on a code path every cell shares, so the cell it
        # names is the only thing standing between one drilled cell and a run losing
        # a cell it was never aimed at.
        store, repo = _seed(tmp_path)
        self._intercept_the_exit(monkeypatch)

        write_year_shards(
            repo,
            "01N",
            year_index=2,
            source=_OneInnerChunkSource(),
            n_workers=1,
            shard_px=_SHARD,
            fault=self._armed(zone="01N", year=2024),
        )

        g = zarr_store.open_store_as_zarr_group(store, group="01N")
        assert g.attrs["years_complete"] == [2025], "a fill the fault does not name must complete normally"

    def test_an_unarmed_fill_cannot_reach_the_exit_at_all(self, tmp_path, monkeypatch):
        # The property that matters more than the fault working: with no request
        # passed, the injected code is unreachable. The exit is booby-trapped, so a
        # path that consulted a default or a module-level fault would be caught here.
        store, repo = _seed(tmp_path)
        events = self._intercept_the_exit(monkeypatch)

        write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)

        assert events == [], "nothing may flush or exit when no fault was requested"
        g = zarr_store.open_store_as_zarr_group(store, group="01N")
        assert g.attrs["years_complete"] == [2025]


def test_every_commit_is_timed_including_the_terminal_path(tmp_path, caplog):
    """The removal's reopen criterion is commit LATENCY, so the detector has to see every commit.

    `commit_s` in ASSEMBLY_SUMMARY was the obvious candidate and misses the dominant source: a
    terminal cell marks itself through `mark_zone_year_empty` and returns without ever reaching
    `assemble_global`, and terminal cells were 72 of the first 78 completions. `commit_with_rebase`
    is the one site both paths pass through.
    """
    store, repo = _seed(tmp_path)
    session = repo.writable_session("main")
    zarr.open_group(session.store, mode="a")["01N"]["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :] = 1
    with caplog.at_level("INFO", logger="tessera_embeddings.storage.shard_writer"):
        commit_with_rebase(session, "mark 01N year 2020 complete")
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("COMMIT ")]
    assert lines, "every commit must emit a COMMIT <secs> line; it is the reopen detector"
    assert "mark 01N year 2020 complete" in lines[0]


class TestAssemblyProgressReportsShards:
    """The only progress a flow run can see. It counted PAYLOADS, which read `0/16` for the
    entire write since all workers return at the end; shard counts come through shared memory.
    """

    def test_the_worker_publishes_into_the_slots_it_was_initialised_with(self) -> None:
        slots = multiprocessing.get_context("spawn").Array("l", 4, lock=False)
        shard_writer._init_fork_worker(slots)
        try:
            shard_writer.report_shard_progress(0, 7, 100)
            shard_writer.report_shard_progress(1, 9, 50)
            assert list(slots) == [7, 100, 9, 50]
        finally:
            shard_writer._PROGRESS_SLOTS = None

    def test_reporting_off_the_pool_path_is_a_no_op(self) -> None:
        """The single-payload path runs in the coordinator, which has no slots."""
        shard_writer._PROGRESS_SLOTS = None
        shard_writer.report_shard_progress(0, 1, 2)  # must not raise

    def test_the_coordinator_logs_shards_and_a_percentage(self, caplog) -> None:
        slots = multiprocessing.get_context("spawn").Array("l", 4, lock=False)
        slots[0], slots[1], slots[2], slots[3] = 30, 100, 20, 100
        futures: list[Future] = []
        for _ in range(2):
            f: Future = Future()
            f.set_running_or_notify_cancel()
            threading.Timer(0.05, lambda fut=f: fut.set_result("done")).start()
            futures.append(f)
        with caplog.at_level(logging.INFO, logger=shard_writer.__name__):
            shard_writer._await_forks(futures, 0.01, unit="tile partitions", slots=slots)
        line = next(r.getMessage() for r in caplog.records if "Assembly progress" in r.getMessage())
        # Summed across BOTH workers, which is the point — one worker's share is not progress.
        assert "50/200 shards written (25%)" in line
        assert "tile partitions outstanding" in line

    def test_before_any_worker_reports_it_falls_back_to_payloads(self, caplog) -> None:
        """A zeroed total must not divide, and a caller whose worker never reports still gets a
        line — it just cannot claim a shard count it does not have.
        """
        slots = multiprocessing.get_context("spawn").Array("l", 2, lock=False)
        future: Future = Future()
        future.set_running_or_notify_cancel()
        threading.Timer(0.05, lambda: future.set_result("done")).start()
        with caplog.at_level(logging.INFO, logger=shard_writer.__name__):
            shard_writer._await_forks([future], 0.01, unit="bands", slots=slots)
        line = next(r.getMessage() for r in caplog.records if "Assembly progress" in r.getMessage())
        assert "shards" not in line
        assert "1/1 bands outstanding" in line


class TestShardCountsArePublishedAtBothEnds:
    """The timed checkpoint sits at the TOP of the loop, so alone it publishes neither the
    denominator before the first shard nor the final count after the last. The coordinator sums
    totals across workers, so a worker yet to report shortens the denominator, and a finished
    one would sit at its last checkpoint understating progress.
    """

    def test_the_worker_reports_the_total_first_and_the_count_last(self, tmp_path, monkeypatch):
        calls: list[tuple[int, int, int]] = []
        monkeypatch.setattr(
            shard_writer,
            "report_shard_progress",
            lambda w, done, total: calls.append((w, done, total)),
        )
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        payload = {
            "fork": session.fork(),
            "group": "01N",
            "year_index": 2,
            "shard_px": _SHARD,
            "source": _SlowLoadSource(),
            "shards": [(0, 0), (1, 0)],
            "worker_index": 3,
            # Long enough that no timed checkpoint fires, so only the two ends can report.
            "progress_interval_s": 3600.0,
        }
        _write_shards_worker(payload)
        assert calls[0] == (3, 0, 2), "the denominator must be published before the first shard"
        assert calls[-1] == (3, 2, 2), "the final count must be published after the last"


class TestForkedWorkersGetLoggingConfigured:
    """A spawned process inherits no logging config, so an unconfigured worker's records are
    discarded. Set on the POOL so a worker added later cannot omit it, which is what this pins.
    """

    def test_the_pool_configures_logging_in_every_child(self, tmp_path, monkeypatch) -> None:
        seen: dict[str, object] = {}

        class _Boom:
            def __init__(self, *args, **kwargs):
                seen.update(kwargs)
                raise RuntimeError("pool not built")

        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", _Boom)
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        # Two payloads, so the pool path is taken rather than the in-process one.
        with pytest.raises(RuntimeError, match="pool not built"):
            run_forked(session, lambda p: p, [{"tag": "a"}, {"tag": "b"}])
        assert seen.get("initializer") is shard_writer._init_fork_worker
        assert seen.get("initargs")  # the shared progress slots travel with it

    def test_the_initializer_configures_logging(self, monkeypatch) -> None:
        """The initializer is where it happens now, so pin that it still does."""
        called: list[bool] = []
        monkeypatch.setattr(shard_writer, "configure_logging", lambda: called.append(True))
        slots = multiprocessing.get_context("spawn").Array("l", 2, lock=False)
        try:
            shard_writer._init_fork_worker(slots)
            assert called == [True]
            assert shard_writer._PROGRESS_SLOTS is slots
        finally:
            shard_writer._PROGRESS_SLOTS = None


class TestCommitTracing:
    """icechunk's own tracing, raised only while a commit runs.

    The commit is the one step whose insides we cannot see from our own logs: our last line is
    printed before the call and the next only after it returns, so a commit that never returns
    tells us nothing about where it stopped.
    """

    @pytest.fixture
    def no_base(self, monkeypatch):
        """No standing filter — the production condition. The suite itself sets one."""
        monkeypatch.setattr(icechunk_logging, "_base_log_filter", None)
        monkeypatch.setattr(icechunk_logging, "_tracing_depth", 0)

    def test_the_filter_is_raised_and_restored(self, monkeypatch, no_base):
        calls = []
        monkeypatch.setattr(icechunk, "set_logs_filter", calls.append)
        with commit_tracing():
            assert calls == [COMMIT_LOG_FILTER]
        assert calls == [COMMIT_LOG_FILTER, None]

    def test_overlapping_commits_restore_only_once_the_last_one_leaves(self, monkeypatch, no_base):
        """Two commits really can overlap: ``plan()`` commits an all-ocean cell on the feeder
        thread while the trailing thread commits an assembly. They share one global filter, so
        the FIRST one out must not switch tracing off under the one still running — which is
        precisely the commit whose trace we would need if it stalled.
        """
        calls = []
        monkeypatch.setattr(icechunk, "set_logs_filter", calls.append)
        with commit_tracing():
            with commit_tracing():
                assert calls == [COMMIT_LOG_FILTER]
            assert calls == [COMMIT_LOG_FILTER], "inner exit must not restore"
        assert calls == [COMMIT_LOG_FILTER, None]

    def test_a_configured_filter_is_left_completely_alone(self, monkeypatch):
        """Someone who configured a filter chose it deliberately, and it is not ours to edit.

        Appending would be worse than replacing: ``EnvFilter`` resolves a target to its most
        specific directive, so adding ``icechunk::session=debug`` beside an ``icechunk=trace``
        would DOWNGRADE the module they asked to watch.
        """
        monkeypatch.setattr(icechunk_logging, "_base_log_filter", "icechunk=trace")
        monkeypatch.setattr(icechunk_logging, "_tracing_depth", 0)
        calls = []
        monkeypatch.setattr(icechunk, "set_logs_filter", calls.append)
        with commit_tracing():
            pass
        assert calls == []

    def test_a_rejected_directive_is_not_recorded_as_the_base(self, monkeypatch):
        """Record only what actually got installed.

        Recording first and swallowing the error left every later `commit_tracing` believing
        an operator filter was in force, so it skipped the commit diagnostics — on the
        strength of a filter icechunk had refused — while the caller heard nothing.
        """
        monkeypatch.setattr(icechunk_logging, "_base_log_filter", "icechunk=warn")
        monkeypatch.setattr(icechunk_logging, "_tracing_depth", 0)

        def _reject(_directive):
            raise ValueError("malformed directive")

        monkeypatch.setattr(icechunk, "set_logs_filter", _reject)
        with pytest.raises(ValueError, match="malformed"):
            icechunk_logging.set_base_logs_filter("not a directive")
        assert icechunk_logging._base_log_filter == "icechunk=warn", "the prior base was lost"

    def test_the_base_filter_is_applied_and_is_what_a_scope_returns_to(self, monkeypatch):
        calls = []
        monkeypatch.setattr(icechunk, "set_logs_filter", calls.append)
        monkeypatch.setattr(icechunk_logging, "_tracing_depth", 0)
        monkeypatch.setattr(icechunk_logging, "_base_log_filter", None)
        icechunk_logging.set_base_logs_filter("icechunk::storage::object_store=error")
        assert calls == ["icechunk::storage::object_store=error"]
        assert icechunk_logging._base_log_filter == "icechunk::storage::object_store=error"

    def test_a_failure_to_set_the_filter_does_not_break_the_commit(self, monkeypatch, no_base):
        """Diagnostics are never worth losing a commit over."""

        def boom(_):
            raise RuntimeError("no filter here")

        monkeypatch.setattr(icechunk, "set_logs_filter", boom)
        with commit_tracing():
            pass
        assert icechunk_logging._tracing_depth == 0, "a failed raise must not leak a reference"

    def test_an_icechunk_without_the_hook_is_tolerated(self, monkeypatch, no_base):
        monkeypatch.delattr(icechunk, "set_logs_filter", raising=False)
        with commit_tracing():
            pass

    def test_the_filter_is_active_while_the_commit_runs(self, tmp_path, monkeypatch, no_base):
        """The load-bearing one: raised around the call, not merely somewhere nearby.

        A commit that stalls only produces useful lines if the filter is already up when it
        enters icechunk. Recording the filter state from inside the commit is the only way to
        pin that; asserting on call order alone would still pass if the wrapper were moved.
        """
        _, repo = _seed(tmp_path, zones=(_ZONE,))
        session = repo.writable_session("main")
        zarr.open_group(session.store, mode="a")["01N"]["embeddings"][2, 0:_CHUNK, 0:_CHUNK, :] = 1
        state = {"filter": None}
        monkeypatch.setattr(icechunk, "set_logs_filter", lambda f: state.__setitem__("filter", f))
        real_commit = type(session).commit
        seen = {}

        def spy(self, *args, **kwargs):
            seen["during"] = state["filter"]
            return real_commit(self, *args, **kwargs)

        monkeypatch.setattr(type(session), "commit", spy)
        commit_with_rebase(session, "traced")
        assert seen["during"] == COMMIT_LOG_FILTER


def test_every_assembly_commit_goes_through_the_traced_helper():
    """No bare ``session.commit`` anywhere an assembly can reach.

    The single-ROI path commits directly three times — schema, overwrite, time-axis
    extension — and every one can stall exactly as the fill's commits can. A commit outside
    a tracing scope produces the same silence that made the 2026-08-29 incident take a day
    to localise, so the property worth pinning is not "these three are wrapped" but "none is
    left bare", which also catches the fourth someone adds later.
    """
    import ast

    from tessera_embeddings.inference import assembly as assembly_mod

    bare = []
    for module in (assembly_mod, shard_writer):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "commit"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"session", "fork"}
            ):
                bare.append(f"{module.__name__}:{node.lineno}")
    assert not bare, f"commit(s) outside a tracing scope: {bare}; use traced_commit"


class _SlowSession:
    """Minimal stand-in: a commit that takes as long as it is told to."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def commit(self, message, **kwargs):
        time.sleep(self.seconds)
        return "snapshot-id"


class TestTheStalledCommitAlarm:
    """A stalled commit is silent by construction; this is what breaks the silence."""

    def test_the_filter_keeps_a_baseline_for_every_other_target(self):
        """`set_logs_filter` REPLACES the filter, it does not add to it.

        A directive naming only icechunk would switch every other target down for the length
        of the commit — including warnings and errors, and on a stall that is forever. The
        bare level in front keeps everyone else where they were.
        """
        directives = COMMIT_LOG_FILTER.split(",")
        assert any("=" not in d for d in directives), (
            f"{COMMIT_LOG_FILTER!r} names only targets, so every other target loses its level"
        )
        assert directives[0] == "warn"

    def test_a_commit_still_happens_when_no_thread_can_be_started(self, monkeypatch, caplog):
        """The alarm is a diagnostic; it must never be the reason a commit does not happen.

        `Thread.start` raises `RuntimeError` when the OS will not give out another thread, and
        that fires before the commit — so an unguarded start would abort the commit for want
        of an alarm about it.
        """

        class _NoThreads:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                raise RuntimeError("can't start new thread")

        monkeypatch.setattr(icechunk_logging.threading, "Thread", _NoThreads)
        assert icechunk_logging.traced_commit(_SlowSession(0.0), "unwatched") == "snapshot-id"

    def test_a_normal_commit_says_nothing(self, monkeypatch, caplog):
        monkeypatch.setattr(icechunk_logging, "COMMIT_ALARM_S", 5.0)
        with caplog.at_level(logging.CRITICAL, logger="tessera_embeddings.storage.icechunk_logging"):
            assert icechunk_logging.traced_commit(_SlowSession(0.0), "quick") == "snapshot-id"
        assert not [r for r in caplog.records if "STALLED" in r.getMessage()]

    def test_a_stalled_commit_is_announced_repeatedly(self, monkeypatch, caplog):
        """Once is not enough: one line followed by silence reads the same as recovery."""
        monkeypatch.setattr(icechunk_logging, "COMMIT_ALARM_S", 0.05)
        with caplog.at_level(logging.CRITICAL, logger="tessera_embeddings.storage.icechunk_logging"):
            icechunk_logging.traced_commit(_SlowSession(0.4), "13N-2018 shard commit")
        said = [r.getMessage() for r in caplog.records if "ASSEMBLY COMMIT STALLED" in r.getMessage()]
        assert len(said) >= 2, f"a stalled commit must keep announcing itself: {said}"
        assert "13N-2018 shard commit" in said[0], "the alarm must name which commit stopped"

    def test_the_first_alarm_dumps_every_thread_stack(self, monkeypatch, caplog):
        """The artefact that cannot be got from outside the process.

        py-spy needs CAP_SYS_PTRACE, which Fargate does not grant, and /proc/<pid>/syscall
        and /proc/<pid>/stack are gated the same way — all three refused against a live
        stalled task. faulthandler runs inside the process and needs no permission at all.
        """
        monkeypatch.setattr(icechunk_logging, "COMMIT_ALARM_S", 0.05)
        dumps = []
        monkeypatch.setattr(icechunk_logging.faulthandler, "dump_traceback", lambda: dumps.append(1))
        with caplog.at_level(logging.CRITICAL, logger="tessera_embeddings.storage.icechunk_logging"):
            icechunk_logging.traced_commit(_SlowSession(0.2), "stuck")
        assert dumps, "the first alarm must dump the stacks; there is no other way to get them"

    def test_a_failing_emission_does_not_end_the_alarm(self, monkeypatch, caplog):
        """One bad log line must not hand the silence back."""
        monkeypatch.setattr(icechunk_logging, "COMMIT_ALARM_S", 0.05)
        real = icechunk_logging._log.critical
        state = {"failed": False}

        def _boom_once(msg, *args, **kwargs):
            if not state["failed"]:
                state["failed"] = True
                raise RuntimeError("log transport down")
            return real(msg, *args, **kwargs)

        monkeypatch.setattr(icechunk_logging._log, "critical", _boom_once)
        with caplog.at_level(logging.CRITICAL, logger="tessera_embeddings.storage.icechunk_logging"):
            icechunk_logging.traced_commit(_SlowSession(0.4), "stuck")
        assert state["failed"], "the first emission was meant to fail"
        assert [r for r in caplog.records if "STALLED" in r.getMessage()], "the alarm died on one failure"


class _FakeDiff:
    """A duck-typed :class:`icechunk.Diff` for the pure-function cases.

    Used ONLY for :func:`_diff_touches`, which is a predicate over path strings and has no
    icechunk behaviour to get wrong. Everything about `catch_up_to_branch` is exercised
    against a real repository below, because a stub of the thing under test cannot falsify it.
    """

    def __init__(self, *, chunks=(), groups=(), arrays=(), moved=()):
        self.updated_chunks = {p: [[0, 0, 0]] for p in chunks}
        self.updated_groups = set(groups)
        self.new_groups = set()
        self.deleted_groups = set()
        self.new_arrays = set(arrays)
        self.updated_arrays = set()
        self.deleted_arrays = set()
        # EIGHT members, matching icechunk.Diff exactly. The stub used to have seven, which is
        # precisely how `moved_nodes` went unchecked in the code as well — a stub that is a
        # subset of the real type makes an incomplete guard look complete.
        self.moved_nodes = list(moved)


class TestDiffTouches:
    """Which changes count as "somebody else wrote our cell"."""

    def test_a_chunk_write_inside_the_group_is_seen(self):
        assert session_catch_up._diff_touches(_FakeDiff(chunks=["/01N/embeddings"]), "01N")

    def test_a_group_attribute_change_is_seen(self):
        # `mark <zone> year N complete` writes group attrs and no chunks at all, so a
        # chunks-only test would wave through the second of every cell's two commits.
        assert session_catch_up._diff_touches(_FakeDiff(groups=["/01N"]), "01N")

    def test_a_new_array_inside_the_group_is_seen(self):
        assert session_catch_up._diff_touches(_FakeDiff(arrays=["/01N/extra"]), "01N")

    def test_another_zone_is_not_our_business(self):
        assert not session_catch_up._diff_touches(_FakeDiff(chunks=["/01S/embeddings"]), "01N")

    def test_an_empty_diff_touches_nothing(self):
        assert not session_catch_up._diff_touches(_FakeDiff(), "01N")

    def test_a_rename_of_our_own_array_is_seen(self):
        # A rename populates `moved_nodes` and NOTHING else — verified against a real repo —
        # so a guard reading the other seven fields waves it through.
        assert session_catch_up._diff_touches(_FakeDiff(moved=[("/01N/embeddings", "/01N/emb2")]), "01N")

    def test_a_rename_into_our_group_is_seen(self):
        assert session_catch_up._diff_touches(_FakeDiff(moved=[("/01S/embeddings", "/01N/stolen")]), "01N")

    def test_the_stub_carries_every_member_of_the_real_diff(self):
        """A stub missing a field cannot fail on that field, which is how one got missed.

        Pins the stub to `icechunk.Diff` so adding a member upstream breaks this rather than
        quietly shrinking what the guard is tested against.
        """
        real = {a for a in dir(icechunk.Diff) if not a.startswith("_") and a != "is_empty"}
        assert real <= set(vars(_FakeDiff(chunks=["/x/y"]))), f"stub is missing {real - set(vars(_FakeDiff()))}"

    def test_a_group_whose_name_merely_starts_with_ours_is_not_ours(self):
        # "/01N2/..." starts with "01N" as a STRING but is a different zone. Matching on a
        # bare prefix would block every catch-up behind an unrelated neighbour, which fails
        # in the safe direction but disables the fix — worth pinning either way.
        assert not session_catch_up._diff_touches(_FakeDiff(chunks=["/01N2/embeddings"]), "01N")


class TestCatchUpToBranch:
    """Keeping an idle coordinator session current — and refusing to when that would hide a conflict.

    Against a real repository, because the whole question is what icechunk does with a session
    whose base has moved while its forks were outstanding.
    """

    @staticmethod
    def _commit_to(repo, group, value):
        session = repo.writable_session("main")
        node = zarr.open_group(session.store, mode="a")
        node[group]["embeddings"][0, 0:_CHUNK, 0:_CHUNK, :] = value
        return session.commit(f"fill {group}")

    def test_nothing_to_do_when_already_at_the_tip(self, tmp_path):
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "current"

    def test_advances_past_a_commit_to_another_zone(self, tmp_path):
        _, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
        session = repo.writable_session("main")
        base = session.snapshot_id
        tip = self._commit_to(repo, "01S", 7)
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "advanced"
        assert session.snapshot_id == tip != base

    def test_refuses_to_skip_a_commit_that_touched_our_own_zone(self, tmp_path):
        _, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
        session = repo.writable_session("main")
        base = session.snapshot_id
        self._commit_to(repo, "01N", 7)
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "blocked"
        assert session.snapshot_id == base, "a blocked catch-up must leave the base alone"

    def test_a_blocked_catch_up_leaves_the_collision_for_the_commit_to_raise(self, tmp_path):
        """THE SAFETY PROPERTY, and the reason the guard exists.

        Moving the base past a commit is also a decision to stop checking it for conflicts.
        A catch-up without the guard sails past a competing writer and then overwrites it in
        silence; with the guard the commit still sees it and refuses, which is what icechunk
        does today and what must not regress.
        """
        _, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
        session = repo.writable_session("main")

        # ORDER MATTERS, and matching production is the whole point. During the fork phase
        # the coordinator's session is EMPTY — the forks have not been merged — so the
        # catch-up runs with nothing to compare, which is exactly why an unguarded one sails
        # past a competing writer without noticing. Writing to the session first would make
        # `rebase` raise for a different reason and the test would pass without testing this.
        self._commit_to(repo, "01N", 7)  # a competing writer lands on OUR zone
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "blocked"

        # ...and only now does the write arrive, as the merge would deliver it.
        node = zarr.open_group(session.store, mode="a")
        node["01N"]["embeddings"][0, 0:_CHUNK, 0:_CHUNK, :] = 3
        with pytest.raises(icechunk.RebaseFailedError):
            session.commit("fill 01N", rebase_with=icechunk.ConflictDetector())

    def test_repeated_catch_ups_keep_the_gap_at_zero(self, tmp_path):
        # The point of running it on a timer rather than once: after each unrelated commit
        # the session is back at the tip, so no single catch-up ever walks far.
        _, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
        session = repo.writable_session("main")
        for value in (1, 2, 3):
            tip = self._commit_to(repo, "01S", value)
            assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "advanced"
            assert session.snapshot_id == tip


class TestCatchUpRaceAndBestEffort:
    """The two things that decide whether this is safe to run unattended for hours."""

    def test_advancing_past_an_unvetted_conflicting_commit_raises(self, tmp_path, monkeypatch):
        """`rebase` takes no target snapshot, so it lands on whatever the tip is when it runs.

        A commit that arrives between the diff check and the rebase is skipped without being
        vetted. It cannot be undone — an empty session's rebase never raises whatever it walks
        over — so the only correct outcome is to notice and fail loudly. Simulated by landing
        the competing commit inside the rebase call itself, which is exactly the window.
        """
        _, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
        session = repo.writable_session("main")
        TestCatchUpToBranch._commit_to(repo, "01S", 1)  # vetted as clean

        real_rebase = type(session).rebase

        def rebase_with_a_latecomer(self, solver):
            TestCatchUpToBranch._commit_to(repo, "01N", 9)  # slips into the window
            return real_rebase(self, solver)

        monkeypatch.setattr(type(session), "rebase", rebase_with_a_latecomer)
        with pytest.raises(session_catch_up.CaughtUpPastAConflictError, match="01N"):
            session_catch_up.catch_up_to_branch(repo, session, "01N")

    def test_best_effort_turns_a_failure_into_a_tally_entry(self, tmp_path, monkeypatch):
        # A rename anywhere in the store makes `rebase` raise even on an empty session, and a
        # reset branch makes `diff` raise. Neither may cost a three-hour fill.
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        monkeypatch.setattr(
            session_catch_up, "catch_up_to_branch", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("s3 hiccup"))
        )
        assert session_catch_up.catch_up_best_effort(repo, session, "01N") == "failed"

    def test_a_failure_after_the_base_moved_is_not_best_effort(self, tmp_path, monkeypatch):
        """The rule is whether the base moved, NOT which exception was raised.

        `rebase` advances incrementally and can leave the session advanced when a later step
        throws, and the post-rebase verification is itself a network call. A handler keyed on
        exception type therefore swallows exactly the case that matters: commits crossed
        without being vetted, then forks merged onto that base, putting a same-group collision
        beyond the commit's conflict detection.
        """
        _, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
        session = repo.writable_session("main")
        TestCatchUpToBranch._commit_to(repo, "01S", 1)
        before = session.snapshot_id

        def advance_then_fail(*args, **kwargs):
            session.rebase(icechunk.ConflictDetector())  # the base really moves
            raise RuntimeError("storage failed during the post-rebase check")

        monkeypatch.setattr(session_catch_up, "catch_up_to_branch", advance_then_fail)
        with pytest.raises(session_catch_up.CaughtUpPastAConflictError, match="never checked"):
            session_catch_up.catch_up_best_effort(repo, session, "01N")
        assert session.snapshot_id != before, "the test did not actually move the base"

    def test_best_effort_still_lets_the_unsafe_one_through(self, tmp_path, monkeypatch):
        """The one exception that must NOT be swallowed, or the guard is decorative."""
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")

        def unsafe(*a, **k):
            raise session_catch_up.CaughtUpPastAConflictError("advanced past a conflict")

        monkeypatch.setattr(session_catch_up, "catch_up_to_branch", unsafe)
        with pytest.raises(session_catch_up.CaughtUpPastAConflictError):
            session_catch_up.catch_up_best_effort(repo, session, "01N")


class TestTheCatchUpIsActuallyWiredIn:
    """Without this, every other test here describes a function nothing calls.

    Deleting the `catch_up=` argument from the one production call site passed the entire
    3,600-test suite. A mechanism can be perfectly tested and completely disconnected.
    """

    def test_write_year_shards_asks_for_a_catch_up_on_its_own_repo_session_and_group(self, tmp_path, monkeypatch):
        # The timer is the ONLY caller now — there is no final synchronous catch-up — so the
        # interval has to be short enough to fire inside this small fill.
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.005)
        _, repo = _seed(tmp_path)
        calls: list[tuple] = []

        def recording(repo_arg, session_arg, group_arg, **kwargs):
            calls.append((repo_arg, session_arg, group_arg))
            return "current"

        monkeypatch.setattr(session_catch_up, "catch_up_to_branch", recording)
        write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), n_workers=1, shard_px=_SHARD)
        assert calls, "write_year_shards never asked for a catch-up"
        seen_repo, seen_session, seen_group = calls[0]
        assert seen_repo is repo
        assert seen_group == "01N"
        assert isinstance(seen_session, icechunk.Session)

    def test_the_guard_is_asked_about_the_whole_gap_not_just_the_newest_commit(self, tmp_path):
        """Production is explicitly multi-commit — depths of 4, 8 and 16 were measured.

        Every other catch-up test creates exactly one intervening snapshot, so a guard that
        inspected only the most recent commit would look correct. Here the offending commit is
        the OLDER of two.
        """
        _, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
        session = repo.writable_session("main")
        TestCatchUpToBranch._commit_to(repo, "01N", 7)  # ours, and now not the newest
        TestCatchUpToBranch._commit_to(repo, "01S", 8)  # someone else's, newest
        assert session_catch_up.catch_up_to_branch(repo, session, "01N") == "blocked"

    def test_every_enumerated_diff_field_exists_on_a_real_diff(self, tmp_path):
        """The enumeration IS the guard's coverage.

        Four of the six node fields have no stub test of their own, so dropping them from the
        tuple changes nothing any test sees. Asserting against a real `icechunk.Diff` means an
        upstream rename breaks this rather than silently narrowing what is checked.
        """
        _, repo = _seed(tmp_path)
        base = repo.lookup_branch("main")
        tip = TestCatchUpToBranch._commit_to(repo, "01N", 5)
        real = repo.diff(from_snapshot_id=base, to_snapshot_id=tip)
        members = {a for a in dir(real) if not a.startswith("_")} - {"is_empty"}
        # EQUALITY, both directions. `hasattr` on each entry would catch an upstream rename but
        # not someone shortening the tuple, which is the likelier accident and the one that
        # narrows the guard in silence.
        handled_separately = {"updated_chunks", "moved_nodes"}
        assert set(session_catch_up._DIFF_NODE_FIELDS) | handled_separately == members, (
            f"the guard covers {set(session_catch_up._DIFF_NODE_FIELDS) | handled_separately} "
            f"but icechunk.Diff has {members}"
        )

    def test_the_rebase_detects_conflicts_rather_than_solving_them(self, tmp_path, monkeypatch):
        """A solver would resolve a collision silently — the exact failure the guard prevents.

        The guard is the first line; this argument is the second. Swapping `ConflictDetector`
        for `BasicConflictSolver` passed every other test.
        """
        _, repo = _seed(tmp_path, zones=(_ZONE, _ZONE_B))
        session = repo.writable_session("main")
        TestCatchUpToBranch._commit_to(repo, "01S", 4)
        seen: list[object] = []
        real_rebase = type(session).rebase
        monkeypatch.setattr(
            type(session),
            "rebase",
            lambda self, solver: (seen.append(solver), real_rebase(self, solver))[1],
        )
        session_catch_up.catch_up_to_branch(repo, session, "01N")
        assert isinstance(seen[0], icechunk.ConflictDetector), f"rebased with {type(seen[0])}"


class TestRunForkedCatchUp:
    """`run_forked` runs the catch-up on a timer for the whole fork phase — and only then.

    There is deliberately no final synchronous catch-up: it would sit outside the timer, so
    nothing would bound it, and a hang there would be silent where a hung COMMIT at least
    raises the stall alarm. Every test here therefore uses a short interval and a worker slow
    enough for the timer to fire, which is also the production shape.
    """

    @staticmethod
    def _slow_worker(payload):
        time.sleep(0.2)
        return payload["fork"], {}

    def test_the_tally_reaches_the_telemetry(self, tmp_path, monkeypatch):
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.05)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        result = run_forked(session, self._slow_worker, [{"tag": "only"}], catch_up=lambda: "current")
        assert result["catch_ups"].get("current", 0) >= 1, "no catch-up was counted"

    def test_the_tally_separates_the_outcomes(self, tmp_path, monkeypatch):
        # The tally's stated job is to say "how often the guard refused". Counting every
        # outcome under one key satisfies a single-outcome test while destroying that.
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.05)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        outcomes = iter(["blocked"] + ["current"] * 200)
        result = run_forked(session, self._slow_worker, [{"tag": "only"}], catch_up=lambda: next(outcomes))
        assert result["catch_ups"]["blocked"] == 1, f"outcomes were merged: {result['catch_ups']}"
        assert result["catch_ups"]["current"] >= 1, f"only one outcome recorded: {result['catch_ups']}"

    def test_the_tally_reaches_the_summary_record_not_just_the_dict(self):
        """`assemble_global` selects fields by hand for `_assembly_summary_line`.

        A field can therefore sit in the telemetry dict and never appear in the record an
        operator reads — which is where this one was, while its own documentation called it
        the only evidence the mechanism ran.
        """
        source = inspect.getsource(assembly.ZarrWriter.assemble_global)
        assert "catch_ups=telemetry.get(" in source, "assemble_global does not forward catch_ups into ASSEMBLY_SUMMARY"

    def test_no_tally_when_no_catch_up_was_asked_for(self, tmp_path):
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        assert "catch_ups" not in run_forked(session, lambda p: (p["fork"], {}), [{"tag": "only"}])

    def test_every_catch_up_happens_before_the_merge(self, tmp_path, monkeypatch):
        """Ordering is the safety argument: a catch-up must only ever see an EMPTY session.

        Records the order DIRECTLY. An earlier version asserted `has_uncommitted_changes is
        False` at catch-up time, which is constant across both orderings when the worker hands
        back an untouched fork — so it passed with the catch-up moved after the merge, the
        exact inversion it claims to forbid.
        """
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.05)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        order: list[str] = []
        real_merge = type(session).merge

        def recording_merge(self, *forks):
            order.append("merge")
            return real_merge(self, *forks)

        monkeypatch.setattr(type(session), "merge", recording_merge)
        run_forked(
            session,
            self._slow_worker,
            [{"tag": "only"}],
            catch_up=lambda: (order.append("catch_up"), "current")[1],
        )
        assert order.count("catch_up") >= 2, f"no periodic catch-up fired: {order}"
        assert order[-1] == "merge", f"a catch-up ran at or after the merge: {order}"
        assert "merge" not in order[:-1], f"the merge is not last: {order}"

    def test_the_timer_fires_repeatedly_during_the_fork_phase(self, tmp_path, monkeypatch):
        """Driven through the SINGLE-payload path: the one a waiting-loop hook never reached."""
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.05)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        ticks: list[int] = []
        run_forked(
            session,
            self._slow_worker,
            [{"tag": "only"}],
            catch_up=lambda: (ticks.append(1), "current")[1],
        )
        assert len(ticks) >= 3, f"the timer fired {len(ticks)} time(s); the periodic catch-up is dead"

    def test_a_catch_up_that_will_not_stop_aborts_rather_than_merging(self, tmp_path, monkeypatch):
        """`join(timeout=...)` returns whether or not the thread noticed the stop.

        A HUNG catch-up is the failure this module exists to bound, so the stop path must not
        assume it stopped. Carrying on would hand the session to `merge` while another thread
        is still inside it.
        """
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.05)
        monkeypatch.setattr(session_catch_up, "CATCH_UP_STOP_TIMEOUT_S", 0.2)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        wedged = threading.Event()

        def never_returns() -> str:
            wedged.set()
            time.sleep(30)  # the daemon dies with the test; nothing waits on it
            return "current"

        def worker_until_wedged(payload):
            wedged.wait(timeout=5)
            return payload["fork"], {}

        with pytest.raises(session_catch_up.CatchUpDidNotStopError, match="still running"):
            run_forked(session, worker_until_wedged, [{"tag": "only"}], catch_up=never_returns)

    def test_a_failed_catch_up_stops_the_wait_instead_of_finishing_the_fill(self):
        """A fill already known to be uncommittable must not spend three more hours writing."""
        outstanding: Future = Future()
        abort = threading.Event()
        threading.Timer(0.05, abort.set).start()
        threading.Timer(5.0, lambda: outstanding.set_result("never reached")).start()
        with pytest.raises(session_catch_up.CatchUpAbortedTheWaitError, match="no longer commit"):
            _await_forks([outstanding], 0.01, abort=abort)

    def test_the_wait_is_unaffected_when_nothing_aborted(self):
        outstanding: Future = Future()
        threading.Timer(0.05, lambda: outstanding.set_result("done")).start()
        assert _await_forks([outstanding], 0.01, abort=threading.Event()) == ["done"]

    def test_a_timer_failure_reaches_the_caller(self, tmp_path, monkeypatch):
        """A daemon thread's exception is discarded, and one of these must fail the fill."""
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.05)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        # RAISES ONCE, from the TIMER. If it raised on every call a later tick would raise on
        # the caller's thread anyway, and the test would pass with the re-raise deleted.
        calls: list[int] = []

        def boom_once() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise session_catch_up.CaughtUpPastAConflictError("crossed an unvetted commit")
            return "current"

        with pytest.raises(session_catch_up.CaughtUpPastAConflictError, match="unvetted"):
            run_forked(session, self._slow_worker, [{"tag": "only"}], catch_up=boom_once)

    def test_a_failing_catch_up_is_not_swallowed(self, tmp_path, monkeypatch):
        # `run_forked` itself does not catch — best-effort is applied at the CALL SITE by
        # `catch_up_best_effort`. Keeping this layer honest means a caller that wants the raw
        # behaviour still gets it.
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.05)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")

        def boom() -> str:
            raise RuntimeError("catch-up exploded")

        with pytest.raises(RuntimeError, match="catch-up exploded"):
            run_forked(session, self._slow_worker, [{"tag": "only"}], catch_up=boom)
