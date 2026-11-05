"""Shard-aligned land-masked writer + commit discipline (W3)."""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from typing import ClassVar

import icechunk
import numpy as np
import pytest
import zarr

from tessera_embeddings.config.fault_injection import DIE_BETWEEN_COMMITS, DRILL_EXIT_STATUS, FaultInjection
from tessera_embeddings.config.store_layout import DIMS_3D, DIMS_4D, ArrayLayout, StoreLayout
from tessera_embeddings.storage import (
    global_store,
    icechunk_logging,
    published_store,
    session_catch_up,
    shard_writer,
    zarr_store,
)
from tessera_embeddings.storage.icechunk_logging import COMMIT_LOG_FILTER
from tessera_embeddings.storage.shard_writer import (
    PhaseTimer,
    _await_forks,
    _write_shards_worker,
    commit_with_rebase,
    run_forked,
    write_year_shards,
)
from tessera_embeddings.storage.zone_grid import ZoneSpec
from tests.unit.deadline import run_under_deadline

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


# `run_forked` worker bodies live at MODULE level: every payload count goes through a spawned
# pool, and `ex.submit` cannot pickle a lambda or a function defined inside a test.
def _echo_injection_worker(payload):
    return payload["fork"], {"idx": payload["worker_index"], "interval": payload["progress_interval_s"]}


def _no_stats_worker(payload):
    return payload["fork"], {}


def _slow_no_stats_worker(payload):
    # Long enough for a catch-up on a 0.05 s timer to fire and WEDGE before this returns; a
    # parent-side Event cannot be used, the worker being a spawned process.
    time.sleep(0.4)
    return payload["fork"], {}


class _OneInnerChunkSource:
    """Writes shard (0,0): one 256^2 inner chunk of real data, the rest fill.

    ``owned`` makes it a targeted update's source: only those pixels of the shard are its own.
    """

    def __init__(self, seed: int = 0, owned: np.ndarray | None = None):
        self.seed = seed
        self.owned = owned

    def live_shards(self):
        return [(0, 0)]

    def pixels(self, shard):
        return self.owned

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
    assert (g["embeddings"][0, 0:_CHUNK, 0:_CHUNK, :] == 0).all()  # other years (2023) untouched


def test_a_source_that_owns_some_pixels_merges_into_the_shard(tmp_path):
    """A targeted update's merge: owned pixels take the new values in every array, the rest stay."""
    store, repo = _seed(tmp_path)
    write_year_shards(repo, "01N", year_index=2, source=_OneInnerChunkSource(seed=1), n_workers=1, shard_px=_SHARD)
    before = zarr_store.open_store_as_zarr_group(store, group="01N")
    old_emb, old_scl = (
        np.asarray(before["embeddings"][2, :_SHARD, :_SHARD]),
        np.asarray(before["scales"][2, :_SHARD, :_SHARD]),
    )
    owned = np.zeros((_SHARD, _SHARD), dtype=bool)
    owned[:16, :16] = True
    recorded: list[tuple[str, int]] = []

    write_year_shards(
        repo,
        "01N",
        year_index=2,
        source=_OneInnerChunkSource(seed=2, owned=owned),
        n_workers=1,
        shard_px=_SHARD,
        commit_attrs=lambda repo_, group, year, commit: (
            recorded.append((group, year, commit["snapshot"])) or repo_.lookup_branch("main")
        ),
    )

    after = zarr_store.open_store_as_zarr_group(store, group="01N")
    new = _OneInnerChunkSource(seed=2).load((0, 0))
    emb, scl = np.asarray(after["embeddings"][2, :_SHARD, :_SHARD]), np.asarray(after["scales"][2, :_SHARD, :_SHARD])
    np.testing.assert_array_equal(emb[owned], new["embeddings"][0][owned])
    np.testing.assert_array_equal(scl[owned], new["scales"][0][owned])
    np.testing.assert_array_equal(emb[~owned], old_emb[~owned])
    np.testing.assert_array_equal(scl[~owned], old_scl[~owned])
    # The hook REPLACED the year attrs commit: the update recorded itself, and the fill's mark
    # is exactly as the first write left it.
    assert recorded == [("01N", 2025, repo.lookup_branch("main"))]
    assert after.attrs["years_complete"] == [2025]


def test_the_year_records_its_shard_commit_and_a_refill_keeps_the_run_before_it(tmp_path):
    """The registry orders runs by the shard commit and checks its parts against every run that wrote
    the year, so the record must name the commit by the store's own clock and survive a refill.
    """
    store, repo = _seed(tmp_path)
    commits = []
    for run in ("fill", "refill"):
        telemetry: dict = {}
        source = _OneInnerChunkSource()
        write_year_shards(repo, "01N", 2, source, n_workers=1, shard_px=_SHARD, run_id=run, telemetry=telemetry)
        commits.append(telemetry["commit"])
    assert commits[1]["committed_at"] == repo.lookup_snapshot(commits[1]["snapshot"]).written_at.isoformat()
    record = zarr_store.open_store_as_zarr_group(store, group="01N").attrs["runs"]["2025"]
    assert (record["run_id"], record["commit"]) == ("refill", commits[1])
    assert [(p["run_id"], p["commit"]) for p in record["previous"]] == [("fill", commits[0])]


def test_a_write_that_dies_before_its_record_is_still_in_the_commit_history(tmp_path, monkeypatch):
    """The runs record lands in a second commit, which a dying process misses. The shard commit's own
    tag is what lets the registry see such a run, so it must land with the pixels.
    """
    store, repo = _seed(tmp_path)

    def dies(*_args, **_kwargs):
        raise RuntimeError("process died between the two commits")

    monkeypatch.setattr(shard_writer, "commit_year_attrs", dies)
    with pytest.raises(RuntimeError, match="between the two commits"):
        write_year_shards(repo, "01N", 2, _OneInnerChunkSource(), n_workers=1, shard_px=_SHARD, run_id="lost")
    assert "runs" not in zarr_store.open_store_as_zarr_group(store, group="01N").attrs
    assert [c.run_id for c in published_store.shard_writes(repo)["01N", 2025]] == ["lost"]


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

    # Campaign logs interleave cells, so an unlabelled line cannot be attributed to one.
    def test_the_progress_line_names_what_is_being_assembled(self, caplog):
        outstanding: Future = Future()
        threading.Timer(0.15, lambda: outstanding.set_result("done")).start()
        with caplog.at_level(logging.INFO, logger="tessera_embeddings.storage.shard_writer"):
            assert _await_forks([outstanding], 0.01, label="15N") == ["done"]
        assert self._progress_lines(caplog)[0].startswith("Assembly progress [15N]: ")

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

    @pytest.mark.real_fork_pool
    def test_run_forked_ships_index_and_interval_to_workers(self, tmp_path):
        # The worker's clock comes from the coordinator via the payload; if the
        # injection is dropped the worker silently falls back to the default
        # interval and a caller-tuned period never takes effect.
        store, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        result = run_forked(session, _echo_injection_worker, [{}], progress_interval_s=1.25)
        assert result["workers"] == [{"idx": 0, "interval": 1.25}]


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


def test_the_filter_is_active_while_the_commit_runs(tmp_path, monkeypatch):
    """The load-bearing one: raised around the call, not merely somewhere nearby.

    A commit that stalls only produces useful lines if the filter is already up when it
    enters icechunk. Recording the filter state from inside the commit is the only way to
    pin that; asserting on call order alone would still pass if the wrapper were moved.
    """
    monkeypatch.setattr(icechunk_logging, "_base_log_filter", None)
    monkeypatch.setattr(icechunk_logging, "_tracing_depth", 0)
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


def test_write_year_shards_asks_for_a_catch_up_on_its_own_repo_session_and_group(tmp_path, monkeypatch):
    """Without this, every catch-up test describes a function nothing calls: deleting the
    `catch_up=` argument from this one production call site passed the whole suite.
    """
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

    def test_no_tally_when_no_catch_up_was_asked_for(self, tmp_path):
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        assert "catch_ups" not in run_forked(session, _no_stats_worker, [{"tag": "only"}])

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

    def test_a_wedged_catch_up_fails_the_fill_rather_than_merging(self, tmp_path, monkeypatch):
        """A catch-up parked inside icechunk cannot be stopped, and the session it holds cannot
        be declared free — so the fill fails and its cell is re-dispatched.

        A recovery that re-homed the finished forks onto a fresh session lived here until
        2026-09-09. It was removed with the request cap that caused the wedge: its own `diff`
        call parked in the identical icechunk deadlock, so it never recovered a cell.
        """
        monkeypatch.setattr(shard_writer, "CATCH_UP_INTERVAL_S", 0.05)
        monkeypatch.setattr(session_catch_up, "CATCH_UP_STOP_TIMEOUT_S", 0.2)

        def never_returns() -> str:
            time.sleep(30)  # daemon; dies with the test
            return "current"

        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        with pytest.raises(session_catch_up.CatchUpDidNotStopError, match="still running"):
            run_forked(session, _slow_no_stats_worker, [{"tag": "only"}], catch_up=never_returns)


class TestForkStallWatchdog:
    """A fork phase that stops making progress fails in bounded time instead of hanging.

    Every test drives the REAL ``run_forked`` through a fake executor, so a never-resolving future
    and a shutdown that will not return are both deterministic; the timeout is shrunk, not the
    mechanism.
    """

    class _Proc:
        def __init__(self, alive: bool = True) -> None:
            self.alive, self.terminated = alive, False

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.terminated, self.alive = True, False

    @classmethod
    def _executor(cls, futures_for, *, shutdown_blocks_until_terminated: bool = False):
        """A stand-in pool; ``futures_for(payload)`` says what ``submit`` hands back.
        ``shutdown_blocks_until_terminated`` models a worker that returned its result but will
        not exit, so a joining shutdown would never return.
        """
        procs = [cls._Proc(), cls._Proc()]
        released = threading.Event()

        class _Fake:
            instances: ClassVar[list] = []

            def __init__(self, *a, **k):
                self._processes = dict(enumerate(procs))
                self.shutdown_calls: list[dict] = []
                self.handed_out: list[Future] = []
                self.terminated_after_done: list[bool] = []
                _Fake.instances.append(self)

            def submit(self, fn, payload):
                future = futures_for(payload)
                self.handed_out.append(future)
                return future

            def shutdown(self, wait=True, cancel_futures=False):
                self.shutdown_calls.append({"wait": wait, "cancel_futures": cancel_futures})
                if wait and shutdown_blocks_until_terminated:
                    released.wait(timeout=10)
                self._processes = None  # CPython drops it here; the stub must too

        def _terminate(p, real=cls._Proc.terminate):
            real(p)
            released.set()
            for fake in _Fake.instances:
                fake.terminated_after_done.append(all(f.done() for f in fake.handed_out))
            # A real pool resolves pending futures as BrokenProcessPool once a worker dies.
            for fake in _Fake.instances:
                for future in fake.handed_out:
                    if not future.done():
                        future.set_exception(BrokenProcessPool("worker terminated"))

        for p in procs:
            p.terminate = _terminate.__get__(p)  # type: ignore[method-assign]
        return _Fake, procs

    @staticmethod
    def _stall_lines(caplog):
        return [r for r in caplog.records if "FORK PHASE STALLED" in r.getMessage()]

    @pytest.mark.parametrize("n_payloads", [1, 2])
    def test_a_worker_that_never_finishes_fails_the_fill_in_bounded_time(
        self, n_payloads, tmp_path, monkeypatch, caplog
    ):
        """The 2026-09-04 shape: a partition outstanding forever, which used to mean days.

        Also pins ``run_forked``'s reporting CONTRACT — the watchdog reads only the shared
        counters, so a worker still running that never reports (this fake) is killed as stalled,
        which is what ``_fill_band_worker`` was until 2026-09-09. And ONE payload is watched like
        any other: ``compute_n_workers`` returns 1 for a small cell, so the old in-process
        shortcut left every shard of such a cell unwatched.
        """
        fake, procs = self._executor(lambda p: Future())
        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", fake)
        dumps: list[int] = []
        dumped = threading.Event()

        def _dump() -> None:
            dumps.append(1)
            dumped.set()

        monkeypatch.setattr(shard_writer.faulthandler, "dump_traceback", _dump)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")

        with (
            caplog.at_level(logging.WARNING, logger="tessera_embeddings.storage.shard_writer"),
            pytest.raises(shard_writer.ForkPhaseStalledError, match="no shard progress"),
        ):
            run_under_deadline(
                15,
                lambda: run_forked(
                    session,
                    lambda p: p,
                    [{"tag": str(i)} for i in range(n_payloads)],
                    progress_interval_s=0.02,
                    fork_stall_timeout_s=0.3,
                ),
            )
        assert all(p.terminated for p in procs), "the stuck workers must be killed, not left to run"
        # WAITED FOR, not assumed: the diagnostics deliberately run AFTER the signal and the
        # teardown, so the fill can raise before the watchdog thread has finished dumping. The
        # critical line is emitted before the dump, so this wait covers both.
        assert dumped.wait(10.0), "the stall must dump every thread's stack: the only way to get one here"
        assert dumps == [1], "the stall dumped more than once; the watchdog is meant to be one-shot"
        assert len(self._stall_lines(caplog)) == 1

    def test_the_workers_are_killed_when_the_stall_is_declared_not_at_the_next_wake(
        self, tmp_path, monkeypatch, caplog
    ):
        """The watchdog tears the pool down ITSELF rather than only setting a flag, which
        ``_await_forks`` would not see for up to a full progress interval — five minutes in
        production. The reporting interval here is longer than the stall timeout, so a pool
        killed only at the next wake fails this.
        """
        fake, procs = self._executor(lambda p: Future())
        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", fake)
        monkeypatch.setattr(shard_writer.faulthandler, "dump_traceback", lambda: None)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        outcome: dict = {}

        def _go():
            try:
                run_forked(
                    session,
                    lambda p: p,
                    [{"tag": "a"}, {"tag": "b"}],
                    progress_interval_s=30.0,
                    fork_stall_timeout_s=0.3,
                )
            except BaseException as exc:  # inspected on the test thread below
                outcome["exc"] = exc

        with caplog.at_level(logging.CRITICAL, logger="tessera_embeddings.storage.shard_writer"):
            thread = threading.Thread(target=_go, daemon=True)
            thread.start()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not all(p.terminated for p in procs):
                time.sleep(0.02)
            assert all(p.terminated for p in procs), (
                "the pool was still alive long past the stall timeout: the watchdog only set a flag"
            )
            thread.join(5.0)
        assert not thread.is_alive(), "the fill never returned"
        assert isinstance(outcome.get("exc"), shard_writer.ForkPhaseStalledError)

    def test_a_healthy_write_is_never_interrupted(self, tmp_path, monkeypatch, caplog):
        """Progress, not elapsed time: a wall-clock timeout would kill every dense zone, whose
        fork phase runs three hours. The worker here outlasts several timeouts while its counter
        keeps moving, and is terminated only once its result is in hand.
        """
        done: Future = Future()
        fake, procs = self._executor(lambda p: done)
        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", fake)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        slots_seen: list = []
        real_watchdog = shard_writer._fork_stall_watchdog

        def capturing(slots, *a, **k):
            slots_seen.append(slots)
            return real_watchdog(slots, *a, **k)

        monkeypatch.setattr(shard_writer, "_fork_stall_watchdog", capturing)

        def keep_writing():
            deadline = time.monotonic() + 1.0  # > 3x the timeout below
            while time.monotonic() < deadline:
                if slots_seen:
                    slots_seen[0][0] += 1  # worker 0 wrote one more shard
                time.sleep(0.02)
            done.set_result(("fork", {}))

        threading.Thread(target=keep_writing, daemon=True).start()
        with caplog.at_level(logging.CRITICAL, logger="tessera_embeddings.storage.shard_writer"):
            fake.instances.clear()
            monkeypatch.setattr(type(session), "merge", lambda self, *forks: None)
            telemetry = run_under_deadline(
                15,
                lambda: run_forked(
                    session,
                    lambda p: p,
                    [{"tag": "a"}, {"tag": "b"}],
                    progress_interval_s=0.02,
                    fork_stall_timeout_s=0.3,
                ),
            )
        assert telemetry["wall_s"] >= 0.9, "the write did not actually outlast the timeout; the test proves nothing"
        assert not self._stall_lines(caplog), "a write that kept making progress was declared stalled"
        assert all(fake.instances[0].terminated_after_done), "a worker was terminated before its result was in hand"

    def test_finished_workers_are_terminated_never_joined(self, tmp_path, monkeypatch, caplog):
        """Every result is in hand but a worker will not exit. A joining shutdown parks forever
        there and would make the watchdog inert; the forks are already ours, so it is killed.
        """
        resolved: Future = Future()
        resolved.set_result(("fork", {}))
        fake, procs = self._executor(lambda p: resolved, shutdown_blocks_until_terminated=True)
        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", fake)
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")
        monkeypatch.setattr(type(session), "merge", lambda self, *forks: None)

        with caplog.at_level(logging.CRITICAL, logger="tessera_embeddings.storage.shard_writer"):
            telemetry = run_under_deadline(
                10,
                lambda: run_forked(
                    session,
                    lambda p: p,
                    [{"tag": "a"}, {"tag": "b"}],
                    progress_interval_s=0.02,
                    fork_stall_timeout_s=0.3,
                ),
            )
        assert all(p.terminated for p in procs)
        assert not any(c["wait"] for c in fake.instances[0].shutdown_calls), "the pool must never be joined"
        assert not self._stall_lines(caplog), "nothing stalled; terminating a finished worker is not a recovery"
        assert len(telemetry["workers"]) == 2, "the finished results were thrown away"

    def test_the_watchdog_stops_with_the_fork_phase(self):
        """Left running it would fire on the merge and commit, which write no shards — a false
        stall on every healthy fill.
        """
        ctx = multiprocessing.get_context("spawn")
        slots = ctx.Array("l", 4, lock=False)

        class _Ex:
            _processes: ClassVar[dict] = {}

            def shutdown(self, **k):
                pass

        with shard_writer._fork_stall_watchdog(slots, 2, threading.Event(), _Ex(), timeout_s=0.1):  # type: ignore[arg-type]
            pass
        time.sleep(0.05)
        assert not [t for t in threading.enumerate() if t.name == "fork-stall-watchdog" and t.is_alive()], (
            "the watchdog thread outlived the fork phase"
        )

    @pytest.mark.parametrize("blocking", ["logger", "dump"])
    def test_a_diagnostic_that_blocks_does_not_stop_the_teardown(self, blocking, tmp_path, monkeypatch):
        """BLOCKING, not raising: `suppress(Exception)` covers the second and nothing covers the
        first. A full container log pipe or an unreachable remote handler parks the write to
        stderr, so a watchdog that diagnosed before it signalled could be wedged by the very
        thing it exists to end. Both diagnostics are exercised because both are downstream.
        """
        gate = threading.Event()  # never set while the subject runs: the diagnostic never returns

        class _BlockingCritical(logging.Logger):
            def critical(self, *args, **kwargs):  # type: ignore[override]
                gate.wait()

        fake, procs = self._executor(lambda p: Future())
        monkeypatch.setattr(shard_writer, "ProcessPoolExecutor", fake)
        if blocking == "dump":
            monkeypatch.setattr(shard_writer.faulthandler, "dump_traceback", gate.wait)
            log: logging.Logger | None = None
        else:
            monkeypatch.setattr(shard_writer.faulthandler, "dump_traceback", lambda: None)
            log = _BlockingCritical("blocking-critical")
        _, repo = _seed(tmp_path)
        session = repo.writable_session("main")

        try:
            with pytest.raises(shard_writer.ForkPhaseStalledError, match="no shard progress"):
                run_under_deadline(
                    15,
                    lambda: run_forked(
                        session,
                        lambda p: p,
                        [{"tag": "a"}, {"tag": "b"}],
                        progress_interval_s=0.02,
                        fork_stall_timeout_s=0.3,
                        log=log,
                    ),
                    what="the fill whose watchdog blocked in a diagnostic",
                )
            assert all(p.terminated for p in procs), "the pool was left up by a blocked diagnostic"
        finally:
            gate.set()  # let the parked watchdog thread finish rather than leaking it
