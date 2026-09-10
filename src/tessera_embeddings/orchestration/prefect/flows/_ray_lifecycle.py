"""Shared Ray-cluster teardown hook for every cluster-owning flow.

``fill_zone_year``, ``fill_zones_sequential`` and the single-ROI ``tessera_embeddings``
flow all provision a Ray cluster the same way and need the same emergency teardown when the
flow is cancelled OR crashes. ONE implementation deliberately: teardown failures leak
billed GPU instances, and a second copy is a second thing to fix when that is discovered.
A flow calls :func:`activate` right after ``ray_cluster`` yields, :func:`deactivate` on
normal exit, and registers :func:`ray_cleanup_on_cancellation` as BOTH its
``on_cancellation`` and ``on_crashed`` hook — a crashed run (OOM, host loss, unhandled
error) is exactly as leak-prone as a cancelled one, since the ``ray up`` head node persists
on EC2 until torn down.

Prefect can run the hook in a FRESH import of this module (the flow's child process is
killed first), where the module globals are unset — so the flows pin a deterministic
``cluster_name`` from their flow-run id
(:func:`~tessera_embeddings.providers.aws.ray.cluster_name_for_flow_run`) and the hook
re-derives the same name as its fallback, terminating the fleet by tag from nothing but the
``flow_run`` argument.

**This hook MUST stay idempotent, and it is the expensive one.** Cancelling a parent run
and its child together delivers the transition twice and runs the hook twice (diagnosed on
the Dask side 2026-07-25; Prefect's invocation model is not at fault), and two concurrent
``ray down`` invocations against one cluster is materially worse than two ECS sweeps — so
prefer cancelling ONE run. Both paths tolerate it: ``ray down`` is bounded and its failure
falls through to tag-based termination, and ``terminate_ray_instances_by_tag`` filters live
instances by tag, so a second pass finds fewer or none. Keep it that way.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml

_active_resolved_yaml: str | None = None
_active_cluster_name: str | None = None


def activate(resolved_yaml: str | None) -> None:
    """Record the live cluster (and its name, parsed from the resolved YAML)."""
    global _active_resolved_yaml, _active_cluster_name
    _active_resolved_yaml = resolved_yaml
    if resolved_yaml and Path(resolved_yaml).exists():
        with Path(resolved_yaml).open() as f:
            _active_cluster_name = yaml.safe_load(f).get("cluster_name")


def deactivate() -> None:
    """Clear the recorded cluster after a normal teardown."""
    global _active_resolved_yaml, _active_cluster_name
    _active_resolved_yaml = None
    _active_cluster_name = None


def ray_cleanup_on_cancellation(flow: object, flow_run: object, state: object) -> None:  # noqa: ARG001
    """Emergency Ray teardown when the flow is cancelled OR crashes.

    Registered as both ``on_cancellation`` and ``on_crashed`` — see the module
    docstring for why a crash is as leak-prone as a cancellation, and why this
    must stay idempotent.
    """
    log = logging.getLogger(__name__)
    # Deferred, NOT module scope. Three Prefect flows import this module unconditionally to
    # register the hook, and ``providers.aws.ray`` pulls in ``boto3`` from the ``aws`` extra
    # and ``ray`` from ``inference`` — at module scope a supported
    # ``tessera_embeddings[prefect]`` install could not so much as IMPORT those flows, local
    # or non-AWS runs included. Same pattern as ``_dask_lifecycle``.
    #
    # An ImportError here is NOT a failure: no AWS provider means no AWS cluster to tear
    # down. Every other exception propagates, since a failed teardown leaks billed GPUs.
    try:
        from tessera_embeddings.providers.aws.ray import (
            RAY_DOWN_TIMEOUT_S,
            cleanup_ray_tempfiles,
            cluster_name_for_flow_run,
            terminate_ray_instances_by_tag,
        )
    except ImportError as exc:
        log.info("AWS provider not installed (%s) — no Ray cluster to tear down for this run.", exc)
        return

    log.warning("Flow cancelled/crashed — tearing down Ray cluster")
    # Fresh-import fallback: the flows pin cluster_name_for_flow_run(flow_run_ctx.id) at
    # provisioning, so the same name is re-derivable when the module globals are unset.
    fallback_cluster = _active_cluster_name or cluster_name_for_flow_run(getattr(flow_run, "id", None))
    if _active_resolved_yaml and Path(_active_resolved_yaml).exists():
        # Bound the call and swallow launch failures: a hung `ray down` (unreachable head,
        # wedged CLI) or an OSError before it produces a return code (`ray` not on PATH) must
        # not block the tag-based fallback and leak billed EC2 workers. Both count as failure
        # (rc=-1) so the fallback fires; tempfile cleanup runs in `finally` regardless.
        rc = -1
        try:
            rc = subprocess.run(
                ["ray", "down", _active_resolved_yaml, "-y"], check=False, timeout=RAY_DOWN_TIMEOUT_S
            ).returncode
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("`ray down` did not complete (%s) — terminating instances by tag", exc)
        finally:
            cleanup_ray_tempfiles(_active_resolved_yaml)
        # A non-zero/timed-out/failed `ray down` leaves EC2 instances running; terminate by
        # cluster tag rather than leak them.
        if rc != 0 and fallback_cluster:
            log.warning("`ray down` exited %d — terminating instances for cluster %r by tag", rc, fallback_cluster)
            terminate_ray_instances_by_tag(cluster_name=fallback_cluster, log=log)
    elif fallback_cluster:
        terminate_ray_instances_by_tag(cluster_name=fallback_cluster, log=log)
    else:
        log.warning("Cancellation fired before the cluster was provisioned — check the AWS console manually.")
