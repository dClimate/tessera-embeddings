"""icechunk's process-global log filter and the stalled-commit alarm (``storage.icechunk_logging``)."""

from __future__ import annotations

import logging
import time

import icechunk
import pytest

from tessera_embeddings.storage import icechunk_logging
from tessera_embeddings.storage.icechunk_logging import COMMIT_LOG_FILTER, commit_tracing


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

    def test_a_failure_to_set_the_filter_does_not_break_the_commit(self, monkeypatch, no_base):
        """Diagnostics are never worth losing a commit over."""

        def boom(_):
            raise RuntimeError("no filter here")

        monkeypatch.setattr(icechunk, "set_logs_filter", boom)
        with commit_tracing():
            pass
        assert icechunk_logging._tracing_depth == 0, "a failed raise must not leak a reference"


class _SlowSession:
    """Minimal stand-in: a commit that takes as long as it is told to."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def commit(self, message, **kwargs):
        time.sleep(self.seconds)
        return "snapshot-id"


class TestTheStalledCommitAlarm:
    """A stalled commit is silent by construction; this is what breaks the silence."""

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
