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

import contextlib
import logging
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import dask
from dask_cloudprovider.aws import ECSCluster, FargateCluster
from tenacity import (
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

# Default Fargate task sizes for ingest workloads. Callers can override
# per-call via ``ecs_cluster``'s ``worker_cpu`` / ``worker_mem`` arguments.
DEFAULT_INGEST_WORKER_CPU = 4096
DEFAULT_INGEST_WORKER_MEM = 16384

# Schedulers benefit from a few cores for graph construction and dashboard
# responsiveness. 16 GiB gives headroom for large HLG graphs without OOM.
DEFAULT_INGEST_SCHEDULER_CPU = 4096
DEFAULT_INGEST_SCHEDULER_MEM = 16384

DEFAULT_CLOUDWATCH_LOG_GROUP = "/ecs/tessera/dask"

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
            # Let dask-cloudprovider create task definitions dynamically so
            # the scheduler address is injected into worker commands.
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
    dashboard runs). The caller supplies their own AWS profile via
    ``--profile`` — it is deliberately not baked into the logged command.

    Best-effort: if the scheduler task metadata isn't shaped as expected
    (e.g. an EC2 scheduler), logs a warning and returns without raising.
    """
    try:
        scheduler = cluster.scheduler
        cluster_name, task_id = scheduler.task_arn.rsplit("/", 2)[1:]
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
        "  --document-name AWS-StartPortForwardingSessionToRemoteHost \\\n"
        '  --parameters \'{"host":["localhost"],"portNumber":["8787"],"localPortNumber":["8787"]}\' \\\n'
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
    image: str | None = None,
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
        image: Override Docker image URI for scheduler and workers.

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

    config = get_fargate_config(
        scheduler_cpu=scheduler_cpu if scheduler_cpu is not None else DEFAULT_INGEST_SCHEDULER_CPU,
        scheduler_mem=scheduler_mem if scheduler_mem is not None else DEFAULT_INGEST_SCHEDULER_MEM,
        worker_cpu=worker_cpu if worker_cpu is not None else DEFAULT_INGEST_WORKER_CPU,
        worker_mem=worker_mem if worker_mem is not None else DEFAULT_INGEST_WORKER_MEM,
    )
    cluster_kwargs = config.to_cluster_kwargs()

    if extra_worker_env:
        cluster_kwargs["environment"].update(extra_worker_env)

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

    try:
        yield cluster
    finally:
        log.info("Closing cluster...")
        cluster.close()
