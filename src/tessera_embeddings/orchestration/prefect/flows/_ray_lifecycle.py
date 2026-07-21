"""Shared Ray-cluster cancellation hook for cluster-owning campaign flows.

``fill_zone_year`` and ``fill_zones_sequential`` provision a Ray cluster the
same way and need the same emergency teardown when cancelled from the Prefect
UI; this module holds the one hook (and the module state it reads) so the
pattern isn't copied per flow. A flow calls :func:`activate` right after
``ray_cluster`` yields and :func:`deactivate` on normal exit, and registers
:func:`ray_cleanup_on_cancellation` as its ``on_cancellation`` hook.

(:mod:`.tessera_embeddings` keeps its own variant: its hook must also work
from a FRESH process import, so it re-derives the cluster name from the
flow-run id — a contract these campaign flows don't have.)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml

from tessera_embeddings.providers.aws.ray import cleanup_ray_tempfiles, terminate_ray_instances_by_tag

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
    if _active_resolved_yaml and Path(_active_resolved_yaml).exists():
        rc = subprocess.run(["ray", "down", _active_resolved_yaml, "-y"], check=False).returncode
        cleanup_ray_tempfiles(_active_resolved_yaml)
        # A non-zero `ray down` leaves EC2 instances running; fall back to
        # terminating them by cluster tag rather than silently leaking them.
        if rc != 0 and _active_cluster_name:
            log.warning("`ray down` exited %d — terminating instances for cluster %r by tag", rc, _active_cluster_name)
            terminate_ray_instances_by_tag(cluster_name=_active_cluster_name, log=log)
    elif _active_cluster_name:
        terminate_ray_instances_by_tag(cluster_name=_active_cluster_name, log=log)
    else:
        log.warning("Cancellation fired before the cluster was provisioned — check the AWS console manually.")
