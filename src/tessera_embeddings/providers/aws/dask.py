"""AWS-backed Dask cluster provisioning (Fargate or hybrid EC2 scheduler).

The :func:`ecs_cluster` context manager constructs a Dask
:class:`FargateCluster` (or :class:`ECSCluster` with an EC2 scheduler)
from environment variables that are expected to be injected by the
orchestrator's job template (the Prefect work pool template, in our
reference deployment). No Prefect imports — the orchestration layer
binds this to Prefect via a thin task shell in
``orchestration/prefect/``.

# Environment-variable contract

The provider reads the following env vars at construction time. The
expected source is the orchestrator's job template (``env`` block on
the work-pool definition); a deployment is responsible for setting
them. They are not hardcoded in source.

| Variable | Purpose |
|---|---|
| ``ECS_CLUSTER_ARN`` | Target ECS cluster ARN |
| ``DASK_ECR_IMAGE_URI`` (or ``ECR_IMAGE_URI``) | Container image for scheduler/workers |
| ``VPC_ID`` | VPC for cluster networking |
| ``PRIVATE_SUBNETS`` | Comma-separated private subnet IDs |
| ``SECURITY_GROUP_ID`` | Security group for tasks |
| ``ECS_EXECUTION_ROLE_ARN`` | ECS execution role |
| ``DASK_TASK_ROLE_ARN`` | Task role for Dask workers |
| ``CLOUDWATCH_LOG_GROUP`` | CloudWatch log group (default ``/ecs/tessera/dask``) |
| ``EC2_SCHEDULER_CAPACITY_PROVIDER`` | Required only when ``ec2_scheduler=True`` |
| ``EC2_SCHEDULER_SUBNET`` | Optional subnet override when ``ec2_scheduler=True`` |

This contract is documented for the open-source release in
``providers/aws/gotchas.md``.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import random
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from distributed import Scheduler

import dask
import fsspec
import psutil
from dask_cloudprovider.aws import ECSCluster, FargateCluster
from distributed import Client, performance_report
from distributed.diagnostics.plugin import SchedulerPlugin
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from tornado.ioloop import PeriodicCallback

# Default Fargate task sizes for ingest workloads. Callers can override
# per-call via ``ecs_cluster``'s ``worker_cpu`` / ``worker_mem`` arguments.
#
# Memory is sized for the ONE worker that runs the ingest task, not for the average
# one. That worker holds the STAC query's retained items — the month being processed
# plus the month prefetched behind it (``ingest.stac.stream_stac_months``) — on top of
# its share of the per-date graph. Every worker gets the same size, so this is paid
# across the whole fleet to accommodate one of them.
#
# RAISING THIS IS A WEAK LEVER against the UNMANAGED baseline: a worker settles at
# roughly 72% of whatever limit it is given — caches and allocator arenas expand into
# whatever space exists — while Dask pauses at 80%, so the steady-state margin is ~8%
# of the limit at ANY size. What protects the worker is keeping its PEAK close to that
# steady ceiling, i.e. not retaining more per date than necessary.
#
# The size is set by ONE rule: leave the driver worker's peak a comfortable multiple
# below the PAUSE threshold. Not below the container limit, and not below the spill
# threshold — spilling the retained items was measured to cost nothing, because the
# spilled bytes are precisely the prefetched month nobody reads until the boundary.
#
# Spilling is NOT the protection: Dask has nothing it is allowed to evict when the
# memory is unmanaged. And a paused worker does not recover — work waiting on data it
# holds can never complete, so the run DEADLOCKS with the rest of the fleet idle. That
# is the observed failure this size exists to prevent, and it is why the margin is
# stated as a multiple rather than a few hundred MiB: undersizing costs a whole run,
# not a retry.
#
# A larger limit is a WEAK lever against that, per the 72% rule above — most of the
# extra space becomes cache rather than margin. Pruning the retained items is what
# actually moved the demand, and it moved it far enough that a smaller limit became
# affordable: memory costs nothing in quota terms but it is not free in dollars, and
# every worker in a cell pays it to accommodate one of them.
#
# See context_docs/design/ingest_optimization_campaign_2026_07.md for the measured
# peaks and the margin this size was chosen against — deliberately not inlined here,
# because a calibration goes stale while the rule above does not.
#
# The vCPU stays at 4 deliberately: the Fargate quota is counted in vCPU, so doubling
# the CPU would halve the workers a cell can run. Valid pairings for 4 vCPU are
# 8192-30720 MiB in 1024 steps.
DEFAULT_INGEST_WORKER_CPU = 4096
DEFAULT_INGEST_WORKER_MEM = 16384

# Schedulers don't need much memory but benefit from a few cores so
# graph construction and dashboard responsiveness stay smooth.
DEFAULT_INGEST_SCHEDULER_CPU = 4096
DEFAULT_INGEST_SCHEDULER_MEM = 8192

DEFAULT_CLOUDWATCH_LOG_GROUP = "/ecs/tessera/dask"

#: Task-stream buffer for DIAGNOSTIC runs only.
#:
#: Dask's task stream is a bounded deque
#: (``distributed.scheduler.dashboard.tasks.task-stream-length``, default 100,000). A
#: performance report built from a capped stream holds only the run's TAIL, and dividing it
#: by the run's date count understates packed task work — which put this campaign's packing
#: ceiling at 1.60x when the true figure is ~2.8x. It survived review because the report's
#: TOTAL rectangle count looks unremarkable once inter-worker transfers pad it.
#:
#: At ~25k tasks per date, 100,000 covers about 4 dates; 3,000,000 covers a ~120-date run.
#: Applied only when a performance report is requested: it is scheduler memory, and campaign
#: runs neither need it nor should pay for it.
DIAGNOSTIC_TASK_STREAM_LENGTH = 3_000_000

# How often the scheduler logs its own resource usage. The default scheduler
# logs are event-driven (worker register/connect) and say nothing about the
# scheduler process's own load, so a slow event loop only shows up as the
# after-the-fact "unresponsive for Ns" warning. A steady heartbeat makes the
# run-up to that visible.
DEFAULT_SCHEDULER_PROFILE_INTERVAL_S = 30.0

# Under-duress stack sampling. When the scheduler process crosses either
# threshold, a short-lived background thread snapshots ``sys._current_frames()``
# a few times and logs a collapsed tally of the busiest code locations — the
# py-spy substitute for Fargate, where process-attach (SYS_PTRACE) isn't
# granted. It attributes a stall to graph construction vs comms vs work-stealing
# vs GC. Runs OFF the event loop, so it never adds to the lag it measures.
DEFAULT_STACK_TRIGGER_CPU_PCT = 90.0
DEFAULT_STACK_TRIGGER_LAG_S = 3.0
DEFAULT_STACK_SAMPLES = 5
DEFAULT_STACK_SAMPLE_GAP_S = 0.2


class SchedulerResourceLogger(SchedulerPlugin):
    """Log the scheduler *process's own* health on a fixed interval.

    The cluster dashboard surfaces aggregate worker load, but stress on the
    scheduler itself — the single event-loop process that builds every graph
    and routes every task — is otherwise invisible until it stalls and emits
    the built-in ``Event loop was unresponsive for Ns`` warning, or the ECS
    task is OOM-killed with no warning at all. This plugin runs a
    :class:`~tornado.ioloop.PeriodicCallback` on the scheduler's event loop and
    logs the signals that precede those failures to the ``dask-scheduler``
    stream, so the run-up is traceable rather than only the aftermath.

    Each line reports:

    - ``cpu`` — scheduler process CPU %, averaged over the interval (can exceed
      100% across threads). Sustained ~100% on the GIL-bound loop is the
      precursor to event-loop stalls.
    - ``rss`` / ``mem`` — process resident memory, absolute and as a percent of
      the container memory limit. The OOM predictor; the whole reason the
      scheduler task is sized at all.
    - ``lag`` — event-loop lag: how late this callback fired versus its
      schedule. The direct, leading measure of the "unresponsive" warning.
    - ``fds`` / ``threads`` — open file descriptors and thread count, to catch
      connection/fd leaks and thread blow-ups.
    - ``workers`` / ``tasks`` — cluster size and total tracked tasks, with a
      breakdown of tasks in flight (``processing``) and stuck waiting
      (``no-worker``) — a rising backlog signals the scheduler falling behind.
    - ``wmem`` / ``wmanaged`` / ``wspill`` / ``wmax`` — FLEET memory, summed
      across workers from the per-worker state the scheduler already tracks, plus
      the hottest single worker (a sum alone cannot distinguish an even fleet from
      one worker holding the graph). Deliberately not scheduler health: worker
      memory is a second failure mode, independent of every signal above, and a
      run can die of it with the scheduler entirely nominal. ``wspill`` is the
      leading indicator — spill means the graph no longer fits the fleet, and it
      precedes worker kills rather than following them. A fleet that fits is the
      precondition for the scheduler metrics to mean anything at all.

    When ``cpu`` or ``lag`` crosses the stack-sampling thresholds, a one-shot
    background thread additionally logs a ``scheduler stack sample`` line: a
    collapsed tally of the busiest code locations across the process's threads,
    to attribute the stall (graph build vs comms vs stealing vs GC). It is
    single-flight and runs off the event loop, so it never worsens the lag it
    measures; pass ``stack_sampling=False`` to disable it entirely.

    Instantiated on the client but pickled to and run inside the scheduler
    process, so ``psutil.Process()`` measures the scheduler, not the client.
    """

    name = "scheduler-resource-logger"

    def __init__(
        self,
        interval_s: float = DEFAULT_SCHEDULER_PROFILE_INTERVAL_S,
        *,
        stack_sampling: bool = True,
        stack_trigger_cpu_pct: float = DEFAULT_STACK_TRIGGER_CPU_PCT,
        stack_trigger_lag_s: float = DEFAULT_STACK_TRIGGER_LAG_S,
        stack_samples: int = DEFAULT_STACK_SAMPLES,
        stack_sample_gap_s: float = DEFAULT_STACK_SAMPLE_GAP_S,
    ) -> None:
        self.interval_s = interval_s
        self._proc: psutil.Process | None = None
        self._callback: PeriodicCallback | None = None
        self._scheduler: Any = None
        self._mem_limit_bytes: int | None = None
        self._expected_next_s: float | None = None
        # Under-duress stack sampling (see class docstring).
        self._stack_sampling = stack_sampling
        self._stack_trigger_cpu_pct = stack_trigger_cpu_pct
        self._stack_trigger_lag_s = stack_trigger_lag_s
        self._stack_samples = stack_samples
        self._stack_sample_gap_s = stack_sample_gap_s
        # Created in start(), NOT here: this object is pickled to the remote
        # scheduler by Client.register_plugin, and a threading.Event owns a
        # _thread.lock that neither pickle nor cloudpickle can serialize. Building
        # it in __init__ makes registration raise — which ecs_cluster swallows as
        # best-effort, silently leaving the run with no heartbeat at all.
        self._sampler_active: threading.Event | None = None
        # Likewise recorded in start(): the id of the thread the scheduler's event
        # loop runs on, which is the thread `lag` measures and the one the stack
        # sampler reports separately (see _sample_stacks_worker).
        self._loop_tid: int | None = None

    async def start(self, scheduler: Scheduler) -> None:
        """Bind to the scheduler and start the periodic probe (called in-process).

        Async to match :class:`SchedulerPlugin`'s declared signature; the body
        is non-blocking (no ``await``) — it just wires up the PeriodicCallback.
        """
        self._scheduler = scheduler
        # Safe here: start() runs in the scheduler process, after unpickling.
        self._sampler_active = threading.Event()
        # This coroutine is awaited ON the scheduler's event loop, so the current
        # thread IS the loop thread. Recording it beats inferring it later.
        self._loop_tid = threading.get_ident()
        self._proc = psutil.Process()
        # Prime cpu_percent so the first real reading is an interval delta
        # rather than the meaningless 0.0 the first call always returns.
        self._proc.cpu_percent(None)
        # Container memory ceiling (cgroup limit on Fargate). psutil reads the
        # cgroup-aware total inside a container; fall back to None if unknown.
        try:
            self._mem_limit_bytes = psutil.virtual_memory().total
        except psutil.Error:
            self._mem_limit_bytes = None
        self._callback = PeriodicCallback(self._log_usage, self.interval_s * 1000)
        self._callback.start()

    async def before_close(self) -> None:
        """Stop the periodic probe before the scheduler shuts down.

        Async to match the supertype; the body is non-blocking.
        """
        if self._callback is not None:
            self._callback.stop()
            self._callback = None

    @staticmethod
    def _worker_memory(sched: Scheduler) -> tuple[float, float, float, float]:
        """Fleet memory as ``(process, managed, spilled, hottest_process)`` GiB.

        Read from :class:`distributed.scheduler.MemoryState` on each worker, which
        the scheduler maintains anyway — no worker-side agent, no extra comms.

        Isolated in its own guard rather than sharing ``_log_usage``'s: a change
        in ``WorkerState`` internals must cost the four fleet numbers, not the
        whole heartbeat. Returns NaNs (rendered ``nan``, which the profiler parses
        as unknown) when the scheduler ref is absent or the fleet is empty.
        """
        try:
            mems = [w.memory for w in sched.workers.values()]
            if not mems:
                return (0.0, 0.0, 0.0, 0.0)
            gib = 1024**3
            return (
                sum(m.process for m in mems) / gib,
                sum(m.managed for m in mems) / gib,
                sum(m.spilled for m in mems) / gib,
                max(m.process for m in mems) / gib,
            )
        except (AttributeError, TypeError, ValueError):
            nan = float("nan")
            return (nan, nan, nan, nan)

    def _log_usage(self) -> None:
        proc = self._proc
        sched = self._scheduler
        if proc is None:
            return
        log = logging.getLogger("distributed.scheduler")
        try:
            # Event-loop lag: how far past the scheduled fire time we actually
            # ran. Large values mean the loop was blocked (GIL-bound work, big
            # graph) — the same condition behind the "unresponsive" warnings.
            now = sched.loop.time() if sched is not None else None
            lag_s = 0.0
            if now is not None and self._expected_next_s is not None:
                lag_s = max(0.0, now - self._expected_next_s)
            if now is not None:
                self._expected_next_s = now + self.interval_s

            cpu_pct = proc.cpu_percent(None)
            rss = proc.memory_info().rss
            rss_gib = rss / 1024**3
            mem_pct = (100.0 * rss / self._mem_limit_bytes) if self._mem_limit_bytes else float("nan")

            n_workers = len(sched.workers) if sched is not None else -1
            n_tasks = len(sched.tasks) if sched is not None else -1
            processing = sum(len(w.processing) for w in sched.workers.values()) if sched is not None else -1
            no_worker = len(getattr(sched, "unrunnable", ())) if sched is not None else -1

            w_mem, w_managed, w_spill, w_max = self._worker_memory(sched) if sched is not None else (float("nan"),) * 4

            log.info(
                "scheduler health: cpu=%.0f%% rss=%.2fGiB mem=%.0f%% lag=%.1fs "
                "fds=%d threads=%d workers=%d tasks=%d processing=%d no-worker=%d "
                "wmem=%.2fGiB wmanaged=%.2fGiB wspill=%.2fGiB wmax=%.2fGiB",
                cpu_pct,
                rss_gib,
                mem_pct,
                lag_s,
                proc.num_fds(),
                proc.num_threads(),
                n_workers,
                n_tasks,
                processing,
                no_worker,
                w_mem,
                w_managed,
                w_spill,
                w_max,
            )
            if self._stack_sampling and (cpu_pct >= self._stack_trigger_cpu_pct or lag_s >= self._stack_trigger_lag_s):
                self._maybe_sample_stacks(cpu_pct, lag_s)
        except (psutil.Error, AttributeError) as e:
            log.warning("scheduler health probe failed: %s", e)

    def _maybe_sample_stacks(self, cpu_pct: float, lag_s: float) -> None:
        """Kick off a one-shot background stack sample unless one is running.

        Single-flight via ``_sampler_active`` so a sustained stall spawns at most
        one sampler at a time, and threaded so sampling runs off the event loop.
        """
        if self._sampler_active is None:
            # Defensive: a probe driven without start() (tests) still samples.
            self._sampler_active = threading.Event()
        if self._sampler_active.is_set():
            return
        self._sampler_active.set()
        threading.Thread(
            target=self._sample_stacks_worker,
            args=(cpu_pct, lag_s),
            name="sched-stack-sampler",
            daemon=True,
        ).start()

    def _sample_stacks_worker(self, cpu_pct: float, lag_s: float) -> None:
        """Snapshot thread stacks a few times and log a collapsed leaf-frame tally.

        Reports the **event-loop thread separately** from the rest, because a
        single global tally is misleading: idle threads park on one unchanging
        wait frame and so accumulate a high count, while the genuinely busy loop
        spreads its samples across many frames and can fall off the end of the
        list — the exact opposite of what we need to attribute a stall. The loop
        thread is the one that matters here (it is what ``lag`` measures), so its
        frames are tallied and reported on their own.

        This sampler's own thread is excluded — it is guaranteed to be running
        (it is doing the sampling) and would otherwise take a slot in every
        report.
        """
        log = logging.getLogger("distributed.scheduler")
        try:
            me = threading.current_thread().ident
            # Recorded by start() from the loop thread itself. Falling back to
            # MainThread covers a probe driven without start() (tests) and matches
            # where dask-cloudprovider's `dask-scheduler` entrypoint runs the loop;
            # attributing the stall to the wrong thread would only mislabel which
            # of the two tallies below is which, never crash.
            loop_tid = self._loop_tid if self._loop_tid is not None else threading.main_thread().ident
            names = {t.ident: t.name for t in threading.enumerate()}
            loop_counts: collections.Counter[str] = collections.Counter()
            other_counts: collections.Counter[str] = collections.Counter()
            for i in range(self._stack_samples):
                for tid, frame in sys._current_frames().items():
                    if tid == me:
                        continue
                    code = frame.f_code
                    fname = code.co_filename.rsplit("/", 1)[-1]
                    if tid == loop_tid:
                        loop_counts[f"{fname}:{code.co_name}:{frame.f_lineno}"] += 1
                    else:
                        other_counts[f"{names.get(tid, tid)}:{fname}:{code.co_name}:{frame.f_lineno}"] += 1
                if i < self._stack_samples - 1:
                    time.sleep(self._stack_sample_gap_s)
            loop_top = " | ".join(f"{loc}={n}" for loc, n in loop_counts.most_common(6)) or "(no samples)"
            other_top = " | ".join(f"{loc}={n}" for loc, n in other_counts.most_common(4)) or "(none)"
            log.warning(
                "scheduler stack sample (cpu=%.0f%% lag=%.1fs, %d samples) loop[%s]: %s -- other threads: %s",
                cpu_pct,
                lag_s,
                self._stack_samples,
                names.get(loop_tid, "?"),
                loop_top,
                other_top,
            )
        except Exception as e:  # diagnostics thread must never crash the scheduler
            log.warning("scheduler stack sampling failed: %s", e)
        finally:
            if self._sampler_active is not None:
                self._sampler_active.clear()


@contextlib.contextmanager
def maybe_performance_report(
    scheduler_address: str,
    uri: str | None,
    # Both call sites are Prefect flows, whose get_run_logger() returns a
    # LoggerAdapter, not a Logger. Only .warning/.info are used, which both have.
    log: logging.Logger | logging.LoggerAdapter[Any],
) -> Iterator[None]:
    """Optionally capture a Dask ``performance_report`` for the wrapped compute.

    No-op when ``uri`` is falsy (the default), so normal runs pay nothing. When
    set, it opens a short-lived client on ``scheduler_address`` — needed because
    ``performance_report`` talks to the scheduler through the *current* client,
    and the Prefect task runner owns a separate one — captures the
    task-stream/profile/bandwidth HTML to a temp file, then uploads it to ``uri``
    (any fsspec target). Intended for probe rungs, not the campaign (large graphs
    make report assembly heavy).

    **The diagnostics are fully isolated from the wrapped body**, because this is
    an optional probe-only artifact that must never affect the ingest it observes.
    Three failure paths are each contained:

    1. *Setup* (client connect, report start) — logged, then the body runs with no
       diagnostics at all, rather than an unreachable scheduler dashboard
       preventing the ingest from running.
    2. *Rendering* (``performance_report.__exit__``, which builds the HTML and is
       the expensive part) — logged. Left unguarded it would fail an ingest that
       had already completed, and on a FAILING ingest it would propagate in place
       of the body's exception, hiding the real error behind a diagnostics one.
    3. *Upload* — logged. fsspec backends raise anything (botocore ClientError for
       a denied PUT or missing bucket, credential errors), none of which subclass
       OSError.

    The report is shipped whether or not the body succeeded: a failed at-scale
    run is exactly when its task stream is most worth reading.
    """
    if not uri:
        yield
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        local = Path(tmpdir) / "dask-performance-report.html"
        diagnostics = contextlib.ExitStack()
        try:
            diagnostics.enter_context(Client(scheduler_address))
            diagnostics.enter_context(performance_report(filename=str(local)))
        except Exception as e:
            log.warning("could not start Dask performance report (%s); continuing without it", e)
            with contextlib.suppress(Exception):
                diagnostics.close()
            yield
            return

        try:
            yield
        finally:
            # Nothing below may raise: that is what keeps a diagnostics artifact
            # from failing a completed ingest or masking a failing one.
            rendered = False
            try:
                diagnostics.close()  # renders the HTML
                rendered = True
            except Exception as e:
                log.warning("Dask performance report generation failed: %s", e)
            if rendered:
                try:
                    with local.open("rb") as fh, fsspec.open(uri, "wb") as out:
                        out.write(fh.read())
                    log.info("wrote Dask performance report to %s", uri)
                except Exception as e:
                    log.warning("failed to upload Dask performance report to %s: %s", uri, e)


# Substrings of transient cluster-start failures worth retrying. All are ECS
# races/limits that clear on their own, not configuration errors:
#   * "not enough values to unpack" — dask-cloudprovider's Task._update_task
#     (ecs.py) calls run_task then immediately describe_tasks on the returned
#     ARN; when the ECS control plane hasn't caught up, describe_tasks returns
#     no task and the ``[self.task] = ...`` unpack raises, wrapped by _start()
#     as RuntimeError("Cluster failed to start: not enough values to unpack").
#   * "RESOURCE:ENI" — ECS RunTask rejects placement before creating a task
#     when the subnet's ENI/IP capacity is exhausted. dask-cloudprovider's
#     Task.start (ecs.py) raises ``RuntimeError(response)`` with the whole
#     run_task response when ``tasks`` is empty, and AWS reports that placement
#     failure reason as ``RESOURCE:ENI`` in that response.
#   * "Scheduler failed to start" — the scheduler Fargate task didn't reach
#     RUNNING in time, typically ENI/IP exhaustion when many clusters launch in
#     a burst (e.g. a fan-out ingest) and their interfaces haven't drained yet.
#     NOTE: dask-cloudprovider raises this same generic string for *any* task
#     that fails to reach RUNNING, including permanent misconfigurations (bad
#     image/command). It is not distinguishable from the transient ENI case by
#     message alone, so we treat it as retryable; the cost of a wrong guess is
#     bounded (a permanent failure burns ~3 backed-off retries before surfacing
#     the original error via reraise=True).
# distributed's spec._start() calls _close() before re-raising, so a failed
# scheduler task is torn down before the exception surfaces — a fresh
# constructor on retry starts clean, with no orphaned task/ENI to leak.
_RETRYABLE_CLUSTER_START_ERRORS = (
    "not enough values to unpack",
    "RESOURCE:ENI",
    "Scheduler failed to start",
)


def _is_retryable_cluster_start_error(exc: BaseException) -> bool:
    """True if ``exc`` is a transient cluster-start failure worth retrying.

    Matches the RuntimeErrors dask-cloudprovider / distributed raise for the
    ECS races in ``_RETRYABLE_CLUSTER_START_ERRORS``. Anything else (bad subnet,
    IAM denial, image pull failure) is a real misconfiguration and must fail
    fast rather than burn retries.
    """
    return isinstance(exc, RuntimeError) and any(s in str(exc) for s in _RETRYABLE_CLUSTER_START_ERRORS)


@dataclass(frozen=True)
class FargateConfig:
    """Resolved Fargate cluster configuration.

    All values are sourced from environment variables (see module
    docstring). Construct via :func:`get_fargate_config` rather than
    instantiating directly.
    """

    cluster_arn: str
    image: str
    vpc: str
    subnets: list[str]
    security_groups: list[str]
    execution_role_arn: str
    task_role_arn: str
    scheduler_cpu: int
    scheduler_mem: int
    worker_cpu: int
    worker_mem: int
    cloudwatch_logs_group: str
    environment: dict[str, str] = field(default_factory=dict)
    #: STABLE Dask task-definition ARNs. When both are set the provider reuses them and
    #: registers NOTHING; when either is unset it falls back to registering a fresh pair
    #: per cluster (the historical behaviour), so this is safe to ship before the
    #: infrastructure that supplies them exists.
    scheduler_task_definition_arn: str = ""
    worker_task_definition_arn: str = ""

    def to_cluster_kwargs(self) -> dict[str, Any]:
        """Translate to kwargs accepted by :class:`FargateCluster`."""
        kwargs: dict[str, Any] = {
            "cluster_arn": self.cluster_arn,
            "image": self.image,
            "vpc": self.vpc,
            "subnets": self.subnets,
            "security_groups": self.security_groups,
            "execution_role_arn": self.execution_role_arn,
            "task_role_arn": self.task_role_arn,
            # NOTE (corrected 2026-08-03): an earlier comment here said definitions had
            # to be created dynamically "so the scheduler address is injected into worker
            # commands". That is not how the library works. The scheduler address, worker
            # name, --nthreads and --memory-limit all travel as per-run CONTAINER
            # OVERRIDES (`Task._overrides["command"]`, sent on every run_task), so the
            # command baked into a registered definition is never used. Pinning stable
            # definitions therefore loses nothing — and it takes RegisterTaskDefinition
            # calls to zero, which is what tripped the account-wide rate limit at 37
            # concurrent cells (ThrottlingException: Rate exceeded).
            "scheduler_cpu": self.scheduler_cpu,
            "scheduler_mem": self.scheduler_mem,
            "worker_cpu": self.worker_cpu,
            "worker_mem": self.worker_mem,
            "cloudwatch_logs_group": self.cloudwatch_logs_group,
            "cloudwatch_logs_stream_prefix": "dask",
            # Use private IPs for scheduler/worker communication within the VPC
            "fargate_use_private_ip": True,
            # Enable ECS Exec for SSM port forwarding to the dashboard
            "scheduler_task_kwargs": {"enableExecuteCommand": True},
            "environment": {
                "DASK_DISTRIBUTED__ADMIN__TICK__LIMIT": "10s",
                "DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT": "60s",
                "DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP": "120s",
                # Use unsigned requests for public S3 buckets (e.g. sentinel-s2-l1c)
                "AWS_NO_SIGN_REQUEST": "YES",
            },
            # Keep True. dask-cloudprovider's startup sweep for stale resources
            # from prior runs iterates all IAM roles, which fails on AWS SSO
            # reserved roles — flipping to False breaks cluster construction
            # every time, not just after a crash. It also only targets debris
            # from runs that died without closing; it does NOT speed reclamation
            # of ENIs from clusters that closed normally (the source of
            # post-burst start failures — see _RETRYABLE_CLUSTER_START_ERRORS).
            # CDK manages our cleanup, so we skip the sweep here.
            "skip_cleanup": True,
        }
        if self.environment:
            kwargs["environment"].update(self.environment)
        # BOTH or NEITHER. The library takes one ARN per role and registers whichever it
        # was not given, so pinning only one would still leave a registration per cluster
        # — the exact thing this removes. Unset means the historical dynamic path.
        if self.scheduler_task_definition_arn and self.worker_task_definition_arn:
            kwargs["scheduler_task_definition_arn"] = self.scheduler_task_definition_arn
            kwargs["worker_task_definition_arn"] = self.worker_task_definition_arn
        return kwargs


def get_fargate_config(
    *,
    scheduler_cpu: int = DEFAULT_INGEST_SCHEDULER_CPU,
    scheduler_mem: int = DEFAULT_INGEST_SCHEDULER_MEM,
    worker_cpu: int = DEFAULT_INGEST_WORKER_CPU,
    worker_mem: int = DEFAULT_INGEST_WORKER_MEM,
    cloudwatch_log_group: str | None = None,
) -> FargateConfig:
    """Build :class:`FargateConfig` from environment variables.

    Pins Dask to a single AZ to avoid cross-AZ data transfer costs, but
    randomises which AZ to spread utilisation over time.

    Args:
        scheduler_cpu: Scheduler CPU units (Fargate reservation).
        scheduler_mem: Scheduler memory in MiB.
        worker_cpu: Worker CPU units.
        worker_mem: Worker memory in MiB.
        cloudwatch_log_group: Override for the log group. ``None`` reads
            ``CLOUDWATCH_LOG_GROUP`` from the env, falling back to
            :data:`DEFAULT_CLOUDWATCH_LOG_GROUP`.
    """
    subnets_str = os.environ.get("PRIVATE_SUBNETS", "")
    subnets = [s.strip() for s in subnets_str.split(",") if s.strip()]
    if subnets:
        subnets = [random.choice(subnets)]

    security_group = os.environ.get("SECURITY_GROUP_ID", "")
    security_groups = [security_group] if security_group else []

    image = os.environ.get("DASK_ECR_IMAGE_URI") or os.environ.get("ECR_IMAGE_URI", "")

    log_group = cloudwatch_log_group or os.environ.get("CLOUDWATCH_LOG_GROUP", DEFAULT_CLOUDWATCH_LOG_GROUP)

    return FargateConfig(
        cluster_arn=os.environ.get("ECS_CLUSTER_ARN", ""),
        image=image,
        vpc=os.environ.get("VPC_ID", ""),
        subnets=subnets,
        security_groups=security_groups,
        execution_role_arn=os.environ.get("ECS_EXECUTION_ROLE_ARN", ""),
        task_role_arn=os.environ.get("DASK_TASK_ROLE_ARN", ""),
        scheduler_task_definition_arn=os.environ.get("DASK_SCHEDULER_TASK_DEFINITION_ARN", ""),
        worker_task_definition_arn=os.environ.get("DASK_WORKER_TASK_DEFINITION_ARN", ""),
        scheduler_cpu=scheduler_cpu,
        scheduler_mem=scheduler_mem,
        worker_cpu=worker_cpu,
        worker_mem=worker_mem,
        cloudwatch_logs_group=log_group,
    )


def log_dashboard_ssm_command(
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    cluster: ECSCluster,
) -> None:
    """Log a copy-pasteable SSM port-forward command for the Dask dashboard.

    The SSM target must point at the *scheduler* Fargate task (where the
    dashboard runs). ``--region`` is filled in from the task ARN — the cluster's
    actual region, so a caller whose default region differs doesn't get an
    "instance not found" from SSM. ``--profile`` stays a placeholder: it is a
    property of the caller's credentials, not of the cluster.

    Uses ``AWS-StartPortForwardingSession``, which targets a port on the
    scheduler container itself. The ``...ToRemoteHost`` variant addresses hosts
    reachable *from* the container, and current SSM agents reject loopback
    destinations for it ("Forwarding to IP address localhost is forbidden").

    Best-effort: if the scheduler task metadata isn't shaped as expected
    (e.g. an EC2 scheduler), logs a warning and returns without raising.
    """
    try:
        scheduler = cluster.scheduler
        cluster_name, task_id = scheduler.task_arn.rsplit("/", 2)[1:]
        # arn:aws:ecs:<region>:<account>:task/<cluster>/<task-id>
        region = scheduler.task_arn.split(":")[3]
        # The scheduler task has multiple containers (dask-scheduler plus
        # injected sidecars like the SSM/ECS-Exec guard); ordering isn't
        # stable, so look up by name.
        scheduler_container = next(c for c in scheduler.task["containers"] if c["name"] == "dask-scheduler")
        runtime_id = scheduler_container["runtimeId"]
        ssm_target = f"ecs:{cluster_name}_{task_id}_{runtime_id}"
    except (AttributeError, KeyError, IndexError, StopIteration, ValueError) as exc:
        log.warning("Could not build SSM dashboard command: %s", exc)
        return

    log.info(
        "To view the Dask dashboard, run (supply your own --profile):\n\n"
        "aws ssm start-session \\\n"
        f"  --target {ssm_target} \\\n"
        "  --document-name AWS-StartPortForwardingSession \\\n"
        f"  --region {region} \\\n"
        '  --parameters \'{"portNumber":["8787"],"localPortNumber":["8787"]}\' \\\n'
        "  --profile <your-aws-profile>\n\n"
        "Then open http://localhost:8787/status"
    )


@contextlib.contextmanager
def ecs_cluster(
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    *,
    min_workers: int = 1,
    max_workers: int = 50,
    worker_cpu: int | None = None,
    worker_mem: int | None = None,
    scheduler_cpu: int | None = None,
    scheduler_mem: int | None = None,
    worker_nthreads: int | None = None,
    worker_nprocs: int | None = None,
    extra_worker_env: dict[str, str] | None = None,
    extra_scheduler_env: dict[str, str] | None = None,
    ec2_scheduler: bool = False,
    diagnostic_task_stream: bool = False,
    image: str | None = None,
    resource_tags: dict[str, str] | None = None,
) -> Iterator[ECSCluster]:
    """Provision a Dask cluster backed by AWS Fargate (or hybrid EC2 scheduler).

    Reads the environment-variable contract documented in this module's
    docstring. The cluster adapts between ``min_workers`` and
    ``max_workers``.

    Args:
        log: Logger.
        min_workers: Minimum Fargate tasks for adaptive scaling.
        max_workers: Maximum Fargate tasks for adaptive scaling.
        worker_cpu: Override worker CPU units.
        worker_mem: Override worker memory in MiB.
        scheduler_cpu: Override scheduler CPU units.
        scheduler_mem: Override scheduler memory in MiB.
        worker_nthreads: Threads per worker process.
        worker_nprocs: Worker processes per Fargate task. Set
            ``worker_nprocs > 1`` with ``worker_nthreads=1`` for
            GIL-bound workloads (e.g. TensorFlow) to use all vCPUs via
            multiprocessing.
        extra_worker_env: Additional env vars to set on every worker.
            Merged after the defaults.
        extra_scheduler_env: Env vars set on the flow-runner process
            *before* :func:`get_fargate_config` is called. Use to
            override worker resource defaults that
            :class:`FargateConfig` reads from env vars.
        ec2_scheduler: Run the Dask scheduler on EC2 instead of
            Fargate. Provides better single-threaded CPU performance
            for large-graph planning. Workers still run on Fargate.
            Requires ``EC2_SCHEDULER_CAPACITY_PROVIDER`` env var.
        diagnostic_task_stream: Raise the scheduler's task-stream buffer to
            :data:`DIAGNOSTIC_TASK_STREAM_LENGTH` so a performance report covers the whole
            run rather than only its last few dates. Pass this whenever a report is being
            captured; leave it off for campaign runs, where the buffer is scheduler memory
            spent on data nobody reads.
        image: Override Docker image URI for scheduler and workers.
        resource_tags: Extra tags applied to every AWS resource the cluster
            creates (scheduler + worker ECS tasks included). The flows tag with
            their flow-run id so an emergency teardown can find this run's
            tasks from nothing but the flow_run (see
            :func:`stop_ecs_tasks_by_tag`).

    Yields:
        The :class:`ECSCluster`/``FargateCluster``.
    """
    # Tightened timeouts compared to dask defaults so worker death is detected
    # quickly enough to drive autoscaling reliably. Heartbeat at 10s gives the
    # scheduler near-immediate awareness of worker exits.
    dask.config.set(
        {
            "distributed.scheduler.worker-ttl": "180s",
            "distributed.worker.heartbeat-interval": "10s",
            "distributed.comm.timeouts.tcp": "180s",
            "distributed.comm.timeouts.connect": "180s",
            "distributed.client.heartbeat": "30s",
        }
    )

    if extra_scheduler_env:
        for key, val in extra_scheduler_env.items():
            os.environ[key] = val

    config = get_fargate_config()
    cluster_kwargs = config.to_cluster_kwargs()

    if extra_worker_env:
        cluster_kwargs["environment"].update(extra_worker_env)

    if diagnostic_task_stream:
        cluster_kwargs["environment"]["DASK_DISTRIBUTED__SCHEDULER__DASHBOARD__TASKS__TASK_STREAM_LENGTH"] = str(
            DIAGNOSTIC_TASK_STREAM_LENGTH
        )

    if ec2_scheduler:
        capacity_provider = os.environ.get("EC2_SCHEDULER_CAPACITY_PROVIDER", "")
        if not capacity_provider:
            raise ValueError("EC2_SCHEDULER_CAPACITY_PROVIDER env var must be set when ec2_scheduler=True")
        cluster_kwargs.setdefault("scheduler_task_kwargs", {})
        cluster_kwargs["scheduler_task_kwargs"]["capacityProviderStrategy"] = [
            {"capacityProvider": capacity_provider, "weight": 1, "base": 1}
        ]
        # Pin all tasks (scheduler + workers) to the ASG's subnet so everything
        # is co-located in the same AZ — avoids cross-AZ data transfer costs.
        scheduler_subnet = os.environ.get("EC2_SCHEDULER_SUBNET", "")
        if scheduler_subnet:
            cluster_kwargs["subnets"] = [scheduler_subnet]
        log.info("EC2 scheduler enabled via capacity provider: %s", capacity_provider)

    if worker_cpu is not None:
        cluster_kwargs["worker_cpu"] = worker_cpu
    if worker_mem is not None:
        cluster_kwargs["worker_mem"] = worker_mem
    if scheduler_cpu is not None:
        cluster_kwargs["scheduler_cpu"] = scheduler_cpu
    if scheduler_mem is not None:
        cluster_kwargs["scheduler_mem"] = scheduler_mem
    if image is not None:
        cluster_kwargs["image"] = image
    if resource_tags:
        cluster_kwargs["tags"] = dict(resource_tags)

    worker_extra_args = ["--death-timeout", "300"]
    if worker_nprocs is not None:
        worker_extra_args.extend(["--nworkers", str(worker_nprocs)])
    if worker_nthreads is not None:
        worker_extra_args.extend(["--nthreads", str(worker_nthreads)])
    cluster_kwargs["worker_extra_args"] = worker_extra_args
    log.info("Worker args: %s", worker_extra_args)

    safe_kwargs = {k: v for k, v in cluster_kwargs.items() if k != "environment"}

    # Retry the cluster constructor on transient ECS start failures (the
    # describe_tasks read-after-write race and scheduler-start/ENI exhaustion —
    # see ``_RETRYABLE_CLUSTER_START_ERRORS``). dask-cloudprovider itself only
    # retries ThrottlingException, so we wrap the whole constructor. Randomized
    # exponential backoff (up to 120s) gives a drained ENI/IP pool time to
    # recover after a burst of cluster launches; the jitter desynchronizes the
    # many flows that exhausted ENIs together so they don't wake and re-collide
    # on RunTask in lockstep, which a deterministic schedule would preserve.
    retrying = Retrying(
        retry=retry_if_exception(_is_retryable_cluster_start_error),
        stop=stop_after_attempt(4),
        wait=wait_random_exponential(multiplier=10, max=120),
        reraise=True,
        before_sleep=lambda rs: log.warning(
            "FargateCluster start failed (%s); retry %d",
            rs.outcome.exception() if rs.outcome is not None else "unknown",
            rs.attempt_number,
        ),
    )

    # HLG culling and graph construction run on the client before any tasks are
    # submitted; the scheduler sees no activity during that window and shuts
    # itself down at its 300s default idle timeout. Raise it so the scheduler
    # stays alive through the client-side prep phase.
    cluster_kwargs["scheduler_timeout"] = "600s"

    if ec2_scheduler:
        log.info("Creating ECSCluster (EC2 scheduler) with config: %s", safe_kwargs)
        cluster = retrying(lambda: ECSCluster(fargate_scheduler=False, fargate_workers=True, **cluster_kwargs))
    else:
        log.info("Creating FargateCluster with config: %s", safe_kwargs)
        cluster = retrying(lambda: FargateCluster(**cluster_kwargs))

    log.info("Cluster created: %s", cluster)
    log.info("Dashboard: %s", cluster.dashboard_link)
    log_dashboard_ssm_command(log, cluster)

    cluster.adapt(minimum=min_workers, maximum=max_workers)
    log.info("Adaptive scaling configured: min=%d, max=%d", min_workers, max_workers)

    # Register the scheduler health heartbeat. A short-lived Client is the only
    # way to push a SchedulerPlugin onto a remote scheduler; the plugin persists
    # on the scheduler after this Client closes, independent of the task-runner
    # Client the flow opens next. Best-effort: a registration failure must not
    # take down the run, since this is diagnostics only.
    try:
        with Client(cluster, timeout="60s") as client:
            client.register_plugin(SchedulerResourceLogger(), name=SchedulerResourceLogger.name)
        log.info("Scheduler health logging enabled (every %.0fs)", DEFAULT_SCHEDULER_PROFILE_INTERVAL_S)
    except Exception as e:
        log.warning("Could not enable scheduler health logging: %s", e)

    try:
        yield cluster
    finally:
        log.info("Closing cluster...")
        cluster.close()


def _region_from_arn(arn: str) -> str | None:
    """The region encoded in an AWS ARN, or ``None`` if it does not look like one.

    ``arn:aws:ecs:<region>:<account>:cluster/<name>`` — field 3. Returning ``None``
    rather than guessing lets boto3 fall back to its own resolution, which is the
    right behaviour for a malformed or non-standard ARN.
    """
    parts = arn.split(":")
    return parts[3] if len(parts) > 4 and parts[3] else None


def ecs_inventory_client(region: str | None = None):  # noqa: ANN201 — botocore client, untyped
    """A boto3 ECS client configured to survive a cluster-wide task enumeration.

    **Use this for any code that paginates ``list_tasks``/``describe_tasks``**, in either
    repo. Walking a wide cluster is ~2 calls per 100 tasks, and both land in ECS's
    *cluster service resource read* bucket, which refills at **one request per second**
    with a burst of ten — the tightest bucket ECS has. At 36 concurrent cells (~1,200
    tasks) one walk is ~24 calls, and boto3's default of four legacy retries gives up
    partway with ``ThrottlingException: Rate exceeded``.

    ``adaptive`` mode is the fix rather than a bigger ``max_attempts``: it adds
    client-side rate limiting that learns the throttle and paces requests into it,
    instead of retrying into a bucket that is still empty. Slower is the intended
    outcome — an enumeration that takes a minute is worth far more than one that fails.

    This mattered because the orphan-fleet sweep is what enumerates: it failed on every
    scheduled run during a 36-cell ingest (2026-08-03), which is exactly when a leaked
    fleet is most likely and most expensive. A safety mechanism that fails under load is
    worse than a slow one. The durable fix is also filed as a quota raise — see
    ``docs/aws-quota-requests.md`` in yield-embeddings — but the client must not depend
    on that being granted.
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "ecs",
        region_name=region,
        config=Config(retries={"max_attempts": 10, "mode": "adaptive"}),
    )


def stop_ecs_tasks_by_tag(
    tag_key: str,
    tag_value: str,
    *,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
    cluster_arn: str | None = None,
) -> int:
    """Stop every running ECS task in the ingest cluster carrying ``tag_key=tag_value``.

    The emergency teardown for a Dask ingest cluster whose owning flow was
    cancelled or crashed: ``ecs_cluster`` tears down in a ``finally``, which a
    hard cancel skips (the process is killed first), leaving scheduler + workers
    running in ECS. The flows tag every cluster resource with their flow-run id
    (``resource_tags``), so this can find the leak from nothing but that id — the
    same tag-based pattern as ``ray.terminate_ray_instances_by_tag``. Idempotent:
    already-stopped tasks simply no longer match.

    ``cluster_arn`` defaults to the module's env contract (``ECS_CLUSTER_ARN``).
    Returns the number of tasks stopped; 0 with a warning when the env var is
    absent (nothing to sweep against) rather than raising — the hook must never
    mask the flow's own terminal state.
    """
    arn = cluster_arn or os.environ.get("ECS_CLUSTER_ARN", "")
    if not arn:
        log.warning("ECS_CLUSTER_ARN unset — cannot sweep tasks for %s=%s", tag_key, tag_value)
        return 0
    # Built in the CLUSTER's region, not the ambient one. A cluster ARN carries its
    # region, and passing one to a default-region client does not redirect the call —
    # so a cross-region sweep would fail to list anything and leave the fleet running
    # and billing, at exactly the moment this hook exists to prevent that.
    ecs = ecs_inventory_client(_region_from_arn(arn))
    task_arns: list[str] = []
    paginator = ecs.get_paginator("list_tasks")
    for page in paginator.paginate(cluster=arn):
        task_arns.extend(page.get("taskArns", []))
    stopped = 0
    for i in range(0, len(task_arns), 100):  # describe_tasks caps at 100 per call
        desc = ecs.describe_tasks(cluster=arn, tasks=task_arns[i : i + 100], include=["TAGS"])
        for task in desc.get("tasks", []):
            if any(t.get("key") == tag_key and t.get("value") == tag_value for t in task.get("tags", [])):
                ecs.stop_task(cluster=arn, task=task["taskArn"], reason=f"tessera teardown: {tag_key}={tag_value}")
                stopped += 1
    log.warning("Stopped %d ECS task(s) tagged %s=%s", stopped, tag_key, tag_value)
    return stopped
