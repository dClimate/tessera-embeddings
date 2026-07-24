"""Unit tests for the AWS Dask provider's cluster-start retry policy.

These cover the retry *decision* — which cluster-start failures are transient
and worth retrying vs. which must fail fast — without provisioning real
Fargate. The predicate is the load-bearing piece (a misclassification either
burns retries on a permanent misconfiguration or gives up on a recoverable
ENI/control-plane race), so it is tested directly, and the tenacity policy it
feeds is exercised against a stub callable to confirm the wiring.
"""

from __future__ import annotations

import pytest
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_fixed

pytest.importorskip("dask_cloudprovider", reason="dask-cloudprovider not installed (AWS extras)")

from tessera_embeddings.providers.aws.dask import (
    _RETRYABLE_CLUSTER_START_ERRORS,
    SchedulerResourceLogger,
    _is_retryable_cluster_start_error,
)


class TestRetryablePredicate:
    """``_is_retryable_cluster_start_error`` classification."""

    @pytest.mark.parametrize("needle", _RETRYABLE_CLUSTER_START_ERRORS)
    def test_known_transient_errors_retry(self, needle: str) -> None:
        """Each documented transient substring is matched, wrapped as _start() wraps it."""
        exc = RuntimeError(f"Cluster failed to start: {needle} (xyz)")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_scheduler_failed_to_start_retries(self) -> None:
        """The ENI-exhaustion failure observed in the fan-out → merge handoff."""
        exc = RuntimeError("Cluster failed to start: Scheduler failed to start")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_eni_placement_failure_retries(self) -> None:
        """RunTask placement rejection: dask raises RuntimeError(response) with
        the whole run_task response, whose failure reason is ``RESOURCE:ENI``.
        """
        exc = RuntimeError("{'tasks': [], 'failures': [{'reason': 'RESOURCE:ENI'}]}")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_describe_tasks_race_retries(self) -> None:
        """The original describe_tasks read-after-write race still matches."""
        exc = RuntimeError("Cluster failed to start: not enough values to unpack (expected 1, got 0)")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_unrelated_runtime_error_does_not_retry(self) -> None:
        """A RuntimeError that isn't a known transient must fail fast."""
        exc = RuntimeError("Cluster failed to start: image pull failed (manifest unknown)")
        assert _is_retryable_cluster_start_error(exc) is False

    def test_non_runtime_error_does_not_retry(self) -> None:
        """Even a matching substring on a non-RuntimeError is not retried — the
        isinstance guard keeps the policy scoped to dask-cloudprovider's wrapper.
        """
        exc = ValueError("Scheduler failed to start")
        assert _is_retryable_cluster_start_error(exc) is False


class TestRetryPolicyWiring:
    """The tenacity policy built from the predicate retries/aborts as intended.

    Uses ``wait_fixed(0)`` to keep the test instant — the wait *schedule* is a
    config value asserted by reading it, not something to sit through here.
    """

    def _run(self, fn, *, max_attempts: int = 4):
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable_cluster_start_error),
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(0),
            reraise=True,
        )
        return retrying(fn)

    def test_retries_until_success(self) -> None:
        """A transient failure on early attempts is ridden out; the call succeeds."""
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Cluster failed to start: Scheduler failed to start")
            return "cluster"

        assert self._run(flaky) == "cluster"
        assert calls["n"] == 3  # failed twice, succeeded on the third

    def test_gives_up_after_max_attempts_and_reraises_original(self) -> None:
        """A persistently-transient failure exhausts attempts and re-raises the
        ORIGINAL RuntimeError (reraise=True), not a tenacity RetryError.
        """
        calls = {"n": 0}

        def always_transient():
            calls["n"] += 1
            raise RuntimeError("Cluster failed to start: Scheduler failed to start")

        with pytest.raises(RuntimeError, match="Scheduler failed to start"):
            self._run(always_transient, max_attempts=4)
        assert calls["n"] == 4  # 1 initial + 3 retries

    def test_non_transient_fails_immediately(self) -> None:
        """A non-matching error is not retried — it surfaces on the first attempt."""
        calls = {"n": 0}

        def misconfigured():
            calls["n"] += 1
            raise RuntimeError("Cluster failed to start: image pull failed")

        with pytest.raises(RuntimeError, match="image pull failed"):
            self._run(misconfigured)
        assert calls["n"] == 1  # no retries burned on a permanent failure


class _FakeWorker:
    """Minimal WorkerState stand-in exposing only what the logger reads."""

    def __init__(self, processing: int) -> None:
        self.processing = list(range(processing))


class _FakeLoop:
    def __init__(self, t: float) -> None:
        self._t = t

    def time(self) -> float:
        return self._t


class _FakeScheduler:
    """Scheduler stand-in with the attributes ``_log_usage`` touches."""

    def __init__(self, *, workers_processing, tasks: int, unrunnable: int, now: float = 1000.0) -> None:
        self.workers = {f"tcp://w{i}": _FakeWorker(p) for i, p in enumerate(workers_processing)}
        self.tasks = dict.fromkeys(range(tasks))
        self.unrunnable = set(range(unrunnable))
        self.loop = _FakeLoop(now)


class TestSchedulerResourceLogger:
    """The scheduler health heartbeat plugin.

    ``_log_usage`` is exercised directly against a fake scheduler so the line
    contents are asserted deterministically, without standing up a cluster or
    sleeping for a PeriodicCallback tick.
    """

    def _emit(self, plugin: SchedulerResourceLogger, sched: _FakeScheduler, caplog) -> str:
        """Run one probe with the plugin bound to ``sched`` and return the line."""
        import logging

        plugin._scheduler = sched
        with caplog.at_level(logging.INFO, logger="distributed.scheduler"):
            plugin._log_usage()
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("scheduler health:")]
        assert len(lines) == 1, f"expected one health line, got {lines}"
        return lines[0]

    def _make_started(self) -> SchedulerResourceLogger:
        """A logger past ``start()`` but with its PeriodicCallback torn down so
        no timer fires during the test.
        """
        # Sampling off: these tests assert the health line only, and a triggered
        # sampler would spawn a thread and add nondeterministic WARNING lines.
        plugin = SchedulerResourceLogger(interval_s=30.0, stack_sampling=False)
        import psutil

        plugin._proc = psutil.Process()
        plugin._proc.cpu_percent(None)
        plugin._mem_limit_bytes = 8 * 1024**3
        return plugin

    def test_line_reports_all_signals(self, caplog) -> None:
        """One probe emits CPU, memory, lag, fds/threads and the task/worker
        backlog — every field the operator diagnoses from.
        """
        plugin = self._make_started()
        sched = _FakeScheduler(workers_processing=[3, 5], tasks=42, unrunnable=7)
        line = self._emit(plugin, sched, caplog)

        assert "cpu=" in line
        assert "rss=" in line and "GiB" in line
        assert "mem=" in line
        assert "lag=" in line
        assert "fds=" in line
        assert "threads=" in line
        assert "workers=2" in line
        assert "tasks=42" in line
        assert "processing=8" in line  # 3 + 5 summed across workers
        assert "no-worker=7" in line

    def test_memory_percent_uses_limit(self, caplog) -> None:
        """mem% is RSS as a fraction of the configured container limit, so a
        tiny limit drives the percentage up for the same process.
        """
        plugin = self._make_started()
        plugin._mem_limit_bytes = 64 * 1024**2  # 64 MiB: any real RSS exceeds this
        sched = _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0)
        line = self._emit(plugin, sched, caplog)

        mem_token = next(t for t in line.split() if t.startswith("mem="))
        pct = float(mem_token.removeprefix("mem=").removesuffix("%"))
        assert pct > 100.0  # RSS far exceeds a 64 MiB ceiling

    def test_lag_is_zero_on_first_probe_then_tracks_overrun(self, caplog) -> None:
        """The first probe has no prior schedule to compare against (lag 0);
        the next probe measures how far past the expected fire time it ran.
        """
        plugin = self._make_started()
        # First probe at t=1000 — primes _expected_next_s = 1000 + 30.
        first = self._emit(plugin, _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0, now=1000.0), caplog)
        assert "lag=0.0s" in first

        caplog.clear()
        # Next probe fires 5s late: clock at 1035 vs expected 1030.
        second = self._emit(plugin, _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0, now=1035.0), caplog)
        assert "lag=5.0s" in second

    def test_probe_before_start_is_a_noop(self, caplog) -> None:
        """Defensive: a probe with no primed process (pre-``start``) logs nothing
        rather than raising.
        """
        import logging

        plugin = SchedulerResourceLogger()
        with caplog.at_level(logging.INFO, logger="distributed.scheduler"):
            plugin._log_usage()
        assert not [r for r in caplog.records if r.getMessage().startswith("scheduler health:")]

    def test_before_close_stops_callback(self) -> None:
        """``before_close`` must stop the PeriodicCallback so a torn-down
        scheduler isn't left with a live timer.
        """
        import asyncio

        plugin = SchedulerResourceLogger()

        class _FakeCallback:
            def __init__(self) -> None:
                self.stopped = False

            def stop(self) -> None:
                self.stopped = True

        cb = _FakeCallback()
        plugin._callback = cb
        asyncio.run(plugin.before_close())

        assert cb.stopped is True
        assert plugin._callback is None

    def test_stack_sampler_logs_and_clears(self, caplog) -> None:
        """The worker logs one collapsed stack tally with the cpu/lag context
        and clears the single-flight flag when done.
        """
        import logging

        plugin = SchedulerResourceLogger(stack_samples=2, stack_sample_gap_s=0.0)
        plugin._sampler_active.set()
        with caplog.at_level(logging.WARNING, logger="distributed.scheduler"):
            plugin._sample_stacks_worker(95.0, 4.0)
        lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("scheduler stack sample")]
        assert len(lines) == 1
        assert "cpu=95% lag=4.0s, 2 samples" in lines[0]
        assert plugin._sampler_active.is_set() is False

    def test_stack_sampler_single_flight(self) -> None:
        """A second trigger while a sampler is active is a no-op."""
        plugin = SchedulerResourceLogger()
        started: list = []
        plugin._sample_stacks_worker = lambda *a: started.append(a)  # type: ignore[method-assign]
        plugin._sampler_active.set()  # pretend one is already running
        plugin._maybe_sample_stacks(99.0, 5.0)
        assert started == []

    def test_stack_sampler_triggers_when_idle(self) -> None:
        """When idle, a trigger spawns the worker off the event loop."""
        import threading

        plugin = SchedulerResourceLogger()
        ran = threading.Event()

        def _fake(cpu: float, lag: float) -> None:
            ran.set()
            plugin._sampler_active.clear()

        plugin._sample_stacks_worker = _fake  # type: ignore[method-assign]
        plugin._maybe_sample_stacks(99.0, 5.0)
        assert ran.wait(timeout=2.0)

    def test_log_usage_calls_sampler_on_high_lag(self, caplog) -> None:
        """The health probe hands a duress reading (lag>=3s) to the sampler."""
        plugin = self._make_started()
        plugin._stack_sampling = True  # helper disables it; re-enable for this test
        calls: list = []
        plugin._maybe_sample_stacks = lambda cpu, lag: calls.append((cpu, lag))  # type: ignore[method-assign]
        self._emit(plugin, _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0, now=1000.0), caplog)
        caplog.clear()  # _emit asserts exactly one health line per call
        self._emit(plugin, _FakeScheduler(workers_processing=[0], tasks=0, unrunnable=0, now=1035.0), caplog)
        assert calls, "expected the sampler to be triggered by lag>=3s"
        assert calls[-1][1] == 5.0


@pytest.mark.integration
class TestSchedulerResourceLoggerOnCluster:
    """End-to-end: registering the plugin on a real (local) scheduler makes it
    log on its event loop. Guards the registration/start wiring that the unit
    tests stub out.
    """

    def test_registered_plugin_emits_on_real_scheduler(self, caplog) -> None:
        import logging
        import time

        from distributed import Client, LocalCluster

        with (
            caplog.at_level(logging.INFO, logger="distributed.scheduler"),
            LocalCluster(n_workers=1, dashboard_address=None, processes=False) as cluster,
            Client(cluster) as client,
        ):
            client.register_plugin(
                SchedulerResourceLogger(interval_s=0.3),
                name=SchedulerResourceLogger.name,
            )
            client.gather(client.map(lambda x: x * x, range(50)))
            time.sleep(1.0)  # allow a couple of PeriodicCallback ticks

        health = [r.getMessage() for r in caplog.records if r.getMessage().startswith("scheduler health:")]
        assert health, "expected at least one health line from the running scheduler"
        assert "workers=1" in health[-1]


def test_maybe_performance_report_noop_without_uri() -> None:
    """With no URI the helper is a pure pass-through — no client, no Dask calls."""
    import logging

    from tessera_embeddings.providers.aws.dask import maybe_performance_report

    ran = []
    with maybe_performance_report("tcp://unused:8786", None, logging.getLogger("t")):
        ran.append(True)
    assert ran == [True]


def test_ingest_settings_perf_report_uri_defaults_none() -> None:
    """The perf-report knob is off by default, so normal runs never capture one."""
    from tessera_embeddings.config.ingest import IngestSettings

    assert IngestSettings().perf_report_uri is None
