"""Fill one (zone, year) of the global embeddings store via Ray (ADR-008 W5).

Provisions a Ray GPU cluster, runs the zone-fill runner
(:func:`tessera_embeddings.orchestration.runners.zone_fill.fill_zone_year` —
coverage mask → inference → shard assembly → tag), and tears the cluster down.
Mirrors :mod:`tessera_embeddings`'s single-flow Ray pattern (``ray_cluster``
context manager + module-state cancellation hook).

**Concurrency model (ADR-008 D6).** Inference is embarrassingly parallel across
zones — nothing is shared and nothing commits — so many of these flow runs can
do GPU work at once. Only the *commit* contends: the runner gates
``assemble_global``'s commit on ``gate`` and leaves ``run_inference`` ungated.
Pass ``commit_limit_name`` to make ``gate`` a **Prefect global concurrency
limit**, so no more than that limit's slots' worth of fill runs commit
simultaneously across the whole fleet (avoiding S3 rebase/commit storms) while
inference stays unbounded. Same-zone serialization (whose attr commits genuinely
conflict) is the campaign driver's job, not this flow's.
"""

from __future__ import annotations

import logging
import subprocess
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any

import yaml
from prefect import flow, get_run_logger
from prefect.concurrency.sync import concurrency

from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.data_loading import resolve_s1_orbit
from tessera_embeddings.inference.orchestration_helpers import build_inference_config
from tessera_embeddings.orchestration.runners.zone_fill import (
    fill_zone_year,
    zone_has_live_tiles,
    zone_year_complete,
    zone_year_on_axis,
)
from tessera_embeddings.providers.aws.ray import cleanup_ray_tempfiles, terminate_ray_instances_by_tag

# Module-level state for the cancellation hook (set on entry, cleared on exit).
_active_resolved_yaml: str | None = None
_active_cluster_name: str | None = None


class _PrefectCommitGate(AbstractContextManager):
    """A ``CommitGate`` backed by a Prefect global concurrency limit.

    Each ``with gate:`` acquires one slot of the named limit for the duration of
    a commit and releases it after, so the campaign's committer count is bounded
    fleet-wide (across separate flow runs / machines) without limiting inference.
    A fresh :func:`concurrency` context is opened per entry so the gate is
    reusable across the (few) commits a single fill performs.
    """

    def __init__(self, name: str, occupy: int = 1) -> None:
        self._name = name
        self._occupy = occupy
        self._cm: AbstractContextManager[Any] | None = None

    def __enter__(self) -> None:
        self._cm = concurrency(self._name, occupy=self._occupy)
        self._cm.__enter__()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        assert self._cm is not None
        self._cm.__exit__(exc_type, exc, tb)
        self._cm = None


def _ray_cleanup_on_cancellation(flow: object, flow_run: object, state: object) -> None:  # noqa: ARG001
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


@flow(name="fill-zone-year", on_cancellation=[_ray_cleanup_on_cancellation])
def fill_zone_year_flow(
    *,
    zone: str,
    year: int,
    paths: BucketPaths,
    ami_ssm_name: str,
    time_window_end: str | None = None,
    store_name: str = "tessera",
    mask_name: str = "global",
    ssm_prefix: str = "/tessera/ray/",
    cloudwatch_log_group: str = "/ec2/tessera/ray",
    code_bucket: str | None = None,
    code_suffix: str = "",
    num_actors: int = 20,
    s1_orbit: str = "both",
    commit_limit_name: str | None = None,
    cleanup_staging: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Fill one ``(zone, year)`` cell of the global store on a Ray cluster.

    Args:
        zone: Zone group name — UTM common name, e.g. ``"30N"``/``"19S"``.
        year: Campaign calendar year to fill (must be on the seeded axis).
        paths: Deployment storage contract (global store, land mask, mosaics).
        ami_ssm_name: SSM parameter name for the Ray GPU AMI ID.
        time_window_end: End month of the inference window as ``"Month Year"``;
            defaults to ``"December {year}"`` (the calendar-year window). The
            runner requires it to overlap ``year``.
        store_name: Global-store repo basename (``paths.global_store``).
        mask_name: Coverage-store repo basename (``paths.land_mask_store``).
        ssm_prefix: SSM prefix for the Ray cluster resource IDs.
        cloudwatch_log_group: CloudWatch log group the Ray workers write to.
        code_bucket: S3 bucket workers pull the source tarball from (``None`` =
            AMI-baked source). See :mod:`tessera_embeddings`.
        code_suffix: Source-tarball filename suffix (lets branches coexist).
        num_actors: GPU actor count for inference.
        s1_orbit: ``"ascending"``, ``"descending"``, or ``"both"``.
        commit_limit_name: Prefect global concurrency limit that bounds the
            fleet's simultaneous committers (D6). ``None`` = ungated (a single
            isolated run has no commit contention).
        cleanup_staging: Delete staged tiles after a successful fill.
        run_id: Reuse a prior run's id to resume it (staged tiles are skipped).

    Returns:
        The zone-fill summary dict (zone, year, run_id, snapshot_id, tag,
        tile/inference counts, ``empty`` flag, elapsed seconds).
    """
    log = get_run_logger()

    # Lazily import the AWS providers so the flow file imports on machines
    # without ray/boto installed (arch tests, local inspection).
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials
    from tessera_embeddings.providers.aws.ray import ray_cluster

    store_path = paths.global_store(store_name)
    land_mask_path = paths.land_mask_store(mask_name)
    mosaic_base = f"{paths.inputs.rstrip('/')}/mosaics/{zone}"

    # Preflight in ascending cost order — cheapest global-store metadata reads
    # first, then the mask, then (only if we will truly infer) the mosaic probe
    # and Ray. GPU work is needed ONLY for a cell that is on-axis, not already
    # complete, AND has live coverage. Each cheaper short-circuit avoids the more
    # expensive reads below it; fill_zone_year re-validates as the authority.
    #  - already-complete cell: the campaign dispatches landed-but-untagged cells
    #    for crash recovery, where fill_zone_year merely re-creates the tag. This
    #    needs ONLY the global store — so it is checked before the land mask,
    #    whose unavailability must never block a retag-only recovery.
    #  - off-axis / unseeded year: fill_zone_year rejects it outright, so resolve
    #    it here (a global-store read) BEFORE standing up a cluster we'd tear
    #    straight back down.
    #  - all-ocean cell (no live tiles): may have no mosaic to probe; fill empty.
    already_complete = zone_year_complete(store_path, zone, year, get_credentials=iam_icechunk_credentials)
    on_axis = already_complete or zone_year_on_axis(store_path, zone, year, get_credentials=iam_icechunk_credentials)
    # Only touch the mask once completion + axis are cleared: an unavailable mask
    # must not block retag-only recovery, and an off-axis year needs no mask.
    has_live = (
        zone_has_live_tiles(land_mask_path, zone, get_credentials=iam_icechunk_credentials)
        if (on_axis and not already_complete)
        else False
    )
    needs_cluster = on_axis and not already_complete and has_live

    # resolve_s1_orbit probes the mosaics (with the same credential callback the
    # rest of the fill uses), so only do it when we'll actually infer.
    resolved_s1 = (
        resolve_s1_orbit(mosaic_base, s1_orbit, get_credentials=iam_icechunk_credentials) if needs_cluster else s1_orbit
    )
    # Default to the strict Jan-Dec calendar-year window for `year` (our global
    # convention: `December {year}` yields a 12-month window spanning Jan-Dec).
    # `time_window_end` overrides for rolling windows — the runner's window check
    # is deliberately permissive so non-campaign consumers can use them.
    config = build_inference_config(
        s1_orbit=resolved_s1,
        time_window=parse_time_window(time_window_end or f"December {year}"),
        checkpoint_path=f"{paths.inputs.rstrip('/')}/models/{checkpoint_filename()}",
        inputs_bucket=paths.inputs,
        output_bucket=paths.outputs,
        chunk_size=SHARD_PX,  # 1 inference tile == 1 shard (D3)
    )
    gate = _PrefectCommitGate(commit_limit_name) if commit_limit_name else None

    fill_kwargs: dict[str, Any] = {
        "store_path": store_path,
        "zone": zone,
        "year": year,
        "land_mask_path": land_mask_path,
        "mosaic_base": mosaic_base,
        "staging_base": f"{paths.outputs.rstrip('/')}/staging",
        "config": config,
        "num_actors": num_actors,
        "log": log,
        "run_id": run_id,
        "gate": gate,
        "cleanup_staging": cleanup_staging,
        "get_credentials": iam_icechunk_credentials,
    }

    if not needs_cluster:
        if already_complete:
            reason = "already complete (retag only)"
        elif not on_axis:
            reason = "year off the pre-allocated axis (runner will reject)"
        else:
            reason = "no live tiles (all-ocean)"
        log.info("Zone %s year %d %s — no Ray cluster", zone, year, reason)
        return fill_zone_year(**fill_kwargs)

    global _active_resolved_yaml, _active_cluster_name
    with ray_cluster(
        log,
        ami_ssm_name=ami_ssm_name,
        ssm_prefix=ssm_prefix,
        cloudwatch_log_group=cloudwatch_log_group,
        code_bucket=code_bucket,
        code_suffix=code_suffix,
    ) as resolved_yaml:
        _active_resolved_yaml = resolved_yaml
        if resolved_yaml and Path(resolved_yaml).exists():
            with Path(resolved_yaml).open() as f:
                _active_cluster_name = yaml.safe_load(f).get("cluster_name")

        summary = fill_zone_year(**fill_kwargs)
    _active_resolved_yaml = None
    _active_cluster_name = None

    log.info("Zone %s year %d filled: %s", zone, year, summary.get("tag"))
    return summary
