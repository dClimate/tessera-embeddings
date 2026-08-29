"""Shard-aligned land-masked writer + commit discipline (W3)."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.fault_injection import DIE_BETWEEN_COMMITS, DRILL_EXIT_STATUS, FaultInjection
from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.storage import global_store, shard_writer, zarr_store
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


class TestTheForkPhaseBudgetIsDerivedFromTheWork:
    """A fill's fork-write budget comes from how many shards it has, not from a clock.

    The phase runs for hours and its length tracks the shard count, so one fixed cutoff
    would be too tight for a dense zone-year and too loose for a sparse one at the same
    time. Rates and derivation: ``context_docs/design/assembly-deadlines-2026_08.md``.
    """

    def test_it_scales_with_the_shard_count(self):
        budget = shard_writer.fork_phase_budget_s
        assert budget(18_000, 16) == pytest.approx(2 * budget(9_000, 16))

    def test_it_scales_inversely_with_the_worker_count(self):
        """The same work on a quarter of the workers takes four times as long.

        The measured rates are aggregates over sixteen workers. A budget derived from the
        shard count alone would hold a narrow fill to a sixteen-worker pace and kill it
        while it was working perfectly well — the one thing this budget must never do.
        """
        budget = shard_writer.fork_phase_budget_s
        assert budget(9_000, 4) == pytest.approx(4 * budget(9_000, 16))

    def test_a_sparse_year_still_gets_the_floor(self):
        # A zone-year with almost nothing to write must not be held to a budget of
        # seconds just because its work is small.
        assert shard_writer.fork_phase_budget_s(1, 1) == shard_writer.FORK_PHASE_FLOOR_S

    def test_it_clears_the_slowest_fill_on_record(self):
        # The slowest fork phase yet measured wrote 9,030 shards in 213 minutes on sixteen
        # workers. The budget must clear that by the safety factor at least, or a cell
        # slower than anything in a narrow sample gets killed while it is working fine.
        slowest_on_record_s = 213 * 60
        budget = shard_writer.fork_phase_budget_s(9_030, 16)
        assert budget >= shard_writer.FORK_PHASE_SAFETY_FACTOR * slowest_on_record_s


class TestTheForkPhaseIsAbandonedOnItsBudget:
    """Outstanding forks past the budget stop being waited on.

    Cheap to abandon, uniquely among the phases: the caller's handler terminates the
    worker processes, and nothing a worker wrote is in the store until its fork is
    merged — so the cost of the kill is unreferenced objects.
    """

    def test_outstanding_forks_past_the_budget_raise(self):
        outstanding: Future = Future()  # never resolves
        with pytest.raises(shard_writer.AssemblyDeadlineError, match="fork-write phase"):
            _await_forks([outstanding], 0.01, budget_s=0.0)

    def test_the_deadline_lands_when_it_falls_not_one_progress_interval_later(self):
        # Production reports progress every 5 minutes, so a wait that always runs the full
        # interval would overshoot the budget by up to that much — and if the last fork
        # landed during the overshoot, nothing would be abandoned at all. Each wait is
        # capped by what is left of the budget instead.
        outstanding: Future = Future()  # never resolves
        started = time.monotonic()
        with pytest.raises(shard_writer.AssemblyDeadlineError, match="fork-write phase"):
            _await_forks([outstanding], 10.0, budget_s=0.1)
        assert time.monotonic() - started < 5.0, "the wait ran its whole progress interval"

    def test_a_terminated_fork_phase_does_not_report_a_leaked_thread(self):
        """The incident line is rendered from this, so it must not describe the other path.

        A fork-write overrun kills its worker processes and a post-fork overrun leaks a
        thread. An announcement that asserted either lifecycle for both would hand operators
        the opposite diagnosis half the time.
        """
        outstanding: Future = Future()  # never resolves
        with pytest.raises(shard_writer.AssemblyDeadlineError) as caught:
            _await_forks([outstanding], 0.01, budget_s=0.0)
        assert "terminated" in caught.value.abandoned
        assert "leak" not in caught.value.abandoned

    def test_a_fill_inside_its_budget_is_untouched(self):
        outstanding: Future = Future()
        threading.Timer(0.05, lambda: outstanding.set_result("done")).start()
        assert _await_forks([outstanding], 0.01, budget_s=30.0) == ["done"]

    def test_no_budget_never_abandons(self):
        # The default every other caller keeps: a fork phase with no budget waits.
        outstanding: Future = Future()
        threading.Timer(0.05, lambda: outstanding.set_result("done")).start()
        assert _await_forks([outstanding], 0.01) == ["done"]

    def test_a_single_payload_fill_is_bounded_too(self, tmp_path):
        """The in-process path is where a SPARSE cell lands, and it must not be exempt.

        `partition_round_robin` yields one payload for a one-shard fill whatever
        `n_workers` says, and one payload runs on the caller's thread. Unbounded, such a
        cell would block its fill's finalizer exactly as the stall this exists for does.
        """
        release = threading.Event()
        entered = threading.Event()

        def _wedged_worker(payload):
            entered.set()
            release.wait(30)
            return payload["fork"], {}

        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        try:
            with pytest.raises(shard_writer.AssemblyDeadlineError, match="fork-write phase"):
                run_forked(session, _wedged_worker, [{"tag": "only"}], fork_budget_s=0.2)
            assert entered.is_set(), "the worker never ran, so the test proved nothing"
        finally:
            release.set()

    def test_a_stuck_session_fork_is_bounded(self):
        """The clock starts before `session.fork()`, not after it.

        That is an icechunk call of the same family as the ones that stalled, and a caller
        whose fork never returns is wedged exactly as one whose commit never returns.
        """
        release = threading.Event()
        entered = threading.Event()

        class _Session:
            def fork(self):
                entered.set()
                release.wait(30)
                return object()

        try:
            with pytest.raises(shard_writer.AssemblyDeadlineError, match="session fork") as caught:
                run_forked(_Session(), lambda p: p, [{"tag": "a"}], fork_budget_s=0.2)
            assert entered.is_set(), "the fork never ran, so the test proved nothing"
            # One clock serves both phases, and this text is quoted into the greppable
            # incident line. A fork that never returned must not be reported as having
            # stalled AFTER the forks — that is the wrong side of the boundary an operator
            # is trying to find.
            assert "after the forks" not in str(caught.value)
        finally:
            release.set()

    def test_a_single_payload_fill_with_no_budget_is_untouched(self, tmp_path):
        # The default every other caller keeps: in-process, on the caller's thread.
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        seen: list[threading.Thread] = []

        def _worker(payload):
            seen.append(threading.current_thread())
            return payload["fork"], {}

        run_forked(session, _worker, [{"tag": "only"}])
        assert seen == [threading.current_thread()]

    def test_the_deadline_terminates_the_workers_still_running(self, tmp_path, monkeypatch):
        """Abandoning the wait must not orphan the pool.

        A worker left running keeps writing fork objects nobody will merge, and Python's
        executor atexit hook then blocks interpreter shutdown on it — the same reasoning
        as the failed-fork path, reached by a different door.
        """

        class _Proc:
            def __init__(self):
                self.terminated = False

            def is_alive(self):
                return True

            def terminate(self):
                self.terminated = True

        running = _Proc()

        class _FakeExecutor:
            def __init__(self, **kwargs):
                self._processes = {"a": running}

            def submit(self, fn, payload):
                return Future()  # never resolves

            def shutdown(self, **kwargs):
                self._processes = None

        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", _FakeExecutor)
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        with pytest.raises(shard_writer.AssemblyDeadlineError, match="fork-write phase"):
            # Small but NON-zero: a zero budget is now refused before `session.fork()` runs,
            # which would never reach the wait this test is about.
            run_forked(session, lambda p: p, [{"tag": "a"}, {"tag": "b"}], progress_interval_s=0.01, fork_budget_s=0.3)
        assert running.terminated, "an abandoned fork phase must terminate its workers"


class TestThePhaseAfterTheForksIsBounded:
    """The tight bound, and the one that can catch a stall.

    Everything after the forks return does a fixed amount of work — one pool teardown,
    one merge, two commits — however many shards were written, and it has never been
    measured taking more than a couple of seconds. So a ceiling here is worth hundreds
    of times the observed maximum and still catches a phase that has stopped.
    """

    @staticmethod
    def _started(budget_s):
        deadline = shard_writer._Deadline(budget_s)
        deadline.start()
        return deadline

    def test_no_deadline_runs_inline_on_the_callers_thread(self):
        # The default every caller that names no budget keeps: no thread, no hand-off.
        seen: list[threading.Thread] = []
        shard_writer._run_under_deadline(None, lambda: seen.append(threading.current_thread()), "merge")
        assert seen == [threading.current_thread()]

    def test_abandonable_work_is_not_started_once_the_shared_budget_is_gone(self):
        # The phases share one clock, so an earlier one can spend all of it. Launching work
        # that would be abandoned a microsecond later leaves a thread running that nobody
        # will ever observe, bought for no chance of finishing in time.
        ran: list[bool] = []
        spent = shard_writer._Deadline(0.0)
        spent.start()
        with pytest.raises(shard_writer.AssemblyDeadlineError, match="was not started"):
            shard_writer._run_under_deadline(spent, lambda: ran.append(True), "merge")
        assert ran == [], "the work was launched despite having no budget left"

    def test_a_commit_with_no_budget_left_still_runs(self):
        """A step that will be WAITED on either way must not be refused for lack of budget.

        Refusing is only worth it when the alternative is an unobserved thread. A commit is
        never abandoned, so refusing to start one would throw away a cell whose commit was
        about to take 200 ms — because an earlier step happened to spend the budget.
        """
        ran: list[bool] = []
        spent = shard_writer._Deadline(0.0)
        spent.start()
        shard_writer._run_under_deadline(spent, lambda: ran.append(True), "shard commit", abandonable=False)
        assert ran == [True]

    def test_an_abandoned_step_reports_a_leaked_thread(self):
        # The counterpart of the fork path's terminated workers: here nothing is killed.
        release = threading.Event()
        try:
            with pytest.raises(shard_writer.AssemblyDeadlineError) as caught:
                shard_writer._run_under_deadline(self._started(0.1), lambda: release.wait(30), "merge")
            assert "leaked" in caught.value.abandoned
        finally:
            release.set()

    def test_a_failure_inside_the_budget_reaches_the_caller(self):
        # A real commit failure must not be swallowed by the hand-off thread.
        def _refused():
            raise RuntimeError("rebase refused")

        with pytest.raises(RuntimeError, match="rebase refused"):
            shard_writer._run_under_deadline(self._started(30.0), _refused, "merge")

    def test_the_caller_is_freed_while_the_work_is_still_running(self):
        """The property that matters, since the work itself cannot be cancelled.

        These calls block below the interpreter, in icechunk's Rust extension, so the
        thread doing them is abandoned rather than stopped. What the budget buys is the
        CALLER's thread — in the campaign runner that is the single-slot finalizer every
        later cell queues behind.
        """
        release = threading.Event()
        entered = threading.Event()
        finished = threading.Event()

        def _wedged():
            entered.set()
            release.wait(30)
            finished.set()

        try:
            with pytest.raises(shard_writer.AssemblyDeadlineError, match="pool teardown"):
                shard_writer._run_under_deadline(self._started(0.2), _wedged, "pool teardown")
            assert entered.is_set(), "the work never started"
            assert not finished.is_set(), "the caller waited for work it was supposed to abandon"
        finally:
            release.set()
        assert finished.wait(10), "the abandoned work is leaked, not killed — it still runs"

    def test_a_stuck_commit_is_announced_repeatedly_and_never_abandoned(self, tmp_path, monkeypatch, caplog):
        """A commit is the one step that mutates the published store, so it is WAITED on.

        A thread cannot be killed. Abandoning a commit would leave a writer that can still
        land, minutes later, after the caller has already failed the cell — and nobody would
        be watching when it did. So the budget only decides when to start shouting: the line
        repeats every budget, because one line followed by hours of silence cannot be told
        apart from the stall having cleared.
        """
        monkeypatch.setattr(shard_writer, "POST_FORK_BUDGET_S", 0.2)
        release = threading.Event()
        entered = threading.Event()
        real_commit = shard_writer.commit_with_rebase

        def _wedged_commit(session, message, **kwargs):
            entered.set()
            release.wait(60)
            return real_commit(session, message, **kwargs)

        monkeypatch.setattr(shard_writer, "commit_with_rebase", _wedged_commit)
        store, repo = _seed(tmp_path)
        done = threading.Event()

        def _said():
            return [r.getMessage() for r in caplog.records if "ASSEMBLY COMMIT OVERDUE" in r.getMessage()]

        def _fill():
            write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), shard_px=_SHARD)
            done.set()

        with caplog.at_level(logging.CRITICAL, logger="tessera_embeddings.storage.shard_writer"):
            threading.Thread(target=_fill, daemon=True).start()
            try:
                assert entered.wait(30), "the commit never ran, so the test proved nothing"
                until = time.monotonic() + 15
                while len(_said()) < 2 and time.monotonic() < until:
                    time.sleep(0.05)
                assert len(_said()) >= 2, f"a stuck commit must keep announcing itself: {_said()}"
                assert not done.is_set(), "the fill returned; a commit must never be abandoned"
            finally:
                release.set()
            assert done.wait(30), "the fill did not finish once its commit returned"
        assert "shard commit" in _said()[0], "the announcement must name which commit stopped"

    def test_a_stuck_teardown_still_kills_the_workers(self, tmp_path, monkeypatch):
        """Abandoning the teardown must not abandon the pool with it.

        A leaked executor is worse than a leaked thread: `concurrent.futures` joins
        executor manager threads at interpreter exit, so a run could still hang at the
        end. A stuck `shutdown` is stuck in a JOIN, so ending the processes is the thing
        most likely to release it — and it cannot be done by calling `shutdown` again,
        which would block on the lock the abandoned thread is holding.
        """
        monkeypatch.setattr(shard_writer, "POST_FORK_BUDGET_S", 0.2)
        release = threading.Event()

        class _Proc:
            def __init__(self):
                self.terminated = False

            def is_alive(self):
                return True

            def terminate(self):
                self.terminated = True

        running = _Proc()

        class _FakeExecutor:
            def __init__(self, **kwargs):
                self._processes = {"a": running}

            def submit(self, fn, payload):
                future: Future = Future()
                future.set_result((payload["fork"], {}))
                return future

            def shutdown(self, **kwargs):
                release.wait(30)  # never returns inside the budget

        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", _FakeExecutor)
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        deadline = shard_writer._Deadline(shard_writer.POST_FORK_BUDGET_S)
        try:
            with pytest.raises(shard_writer.AssemblyDeadlineError, match="pool teardown"):
                run_forked(session, lambda p: p, [{"tag": "a"}, {"tag": "b"}], post_fork=deadline)
            assert running.terminated, "an abandoned teardown must still kill the pool"
        finally:
            release.set()

    def test_a_healthy_fill_is_not_touched_by_the_bound(self, tmp_path):
        # The bound is live on every fill now, so pin that an ordinary one still lands.
        store, repo = _seed(tmp_path)
        write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(), shard_px=_SHARD)
        assert zarr_store.open_store_as_zarr_group(store, group="01N").attrs["years_complete"] == [2025]
