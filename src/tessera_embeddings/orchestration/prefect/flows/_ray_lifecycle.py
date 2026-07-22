"""Shared Ray-cluster cancellation hook for cluster-owning campaign flows.

``fill_zone_year`` and ``fill_zones_sequential`` provision a Ray cluster the
same way and need the same emergency teardown when cancelled from the Prefect
UI; this module holds the one hook (and the module state it reads) so the
pattern isn't copied per flow. A flow calls :func:`activate` right after
``ray_cluster`` yields and :func:`deactivate` on normal exit, and registers
:func:`ray_cleanup_on_cancellation` as its ``on_cancellation`` hook.

Prefect can run the hook in a FRESH import of this module (the flow's child
process is killed first), where the module globals are unset — so both flows
pin a deterministic ``cluster_name`` from their flow-run id
(:func:`~tessera_embeddings.providers.aws.ray.cluster_name_for_flow_run`) and
the hook re-derives the same name as its fallback, terminating the fleet by
tag even from nothing but the ``flow_run`` argument.
(:mod:`.tessera_embeddings` keeps its own variant of the same pattern.)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml

from tessera_embeddings.providers.aws.ray import (
    RAY_DOWN_TIMEOUT_S,
    cleanup_ray_tempfiles,
    cluster_name_for_flow_run,
    terminate_ray_instances_by_tag,
)

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
    """Emergency Ray teardown when the flow is cancelled via the Prefect UI."""
    log = logging.getLogger(__name__)
    log.warning("Flow cancelled — tearing down Ray cluster")
    # Fresh-import fallback: the flows pin cluster_name_for_flow_run(flow_run_ctx.id)
    # at provisioning, so the hook can re-derive the same name when the module
    # globals are unset (Prefect killed the flow process before running the hook).
    fallback_cluster = _active_cluster_name or cluster_name_for_flow_run(getattr(flow_run, "id", None))
    if _active_resolved_yaml and Path(_active_resolved_yaml).exists():
        # Bound the call and swallow launch failures: a hung `ray down` (unreachable
        # head, wedged CLI) OR an OSError before it even produces a return code (e.g.
        # `ray` not on PATH) must not block the tag-based termination fallback and leak
        # billed EC2 workers. Both are treated as failure (rc=-1) so the fallback fires,
        # and tempfile cleanup runs in `finally` regardless of outcome.
        rc = -1
        try:
            rc = subprocess.run(
                ["ray", "down", _active_resolved_yaml, "-y"], check=False, timeout=RAY_DOWN_TIMEOUT_S
            ).returncode
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("`ray down` did not complete (%s) — terminating instances by tag", exc)
        finally:
            cleanup_ray_tempfiles(_active_resolved_yaml)
        # A non-zero/timed-out/failed `ray down` leaves EC2 instances running; fall back
        # to terminating them by cluster tag rather than silently leaking them.
        if rc != 0 and fallback_cluster:
            log.warning("`ray down` exited %d — terminating instances for cluster %r by tag", rc, fallback_cluster)
            terminate_ray_instances_by_tag(cluster_name=fallback_cluster, log=log)
    elif fallback_cluster:
        terminate_ray_instances_by_tag(cluster_name=fallback_cluster, log=log)
    else:
        log.warning("Cancellation fired before the cluster was provisioned — check the AWS console manually.")
