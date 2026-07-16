"""Drive the global embeddings campaign: fill every pending (zone, year).

Reads the live fill progress from the seeded store
(:func:`~tessera_embeddings.storage.campaign.campaign_status`) and dispatches a
``fill-zone-year`` deployment run per pending cell via ``arun_deployment``,
mirroring :mod:`tessera_full_pipeline`'s driver pattern.

**Scheduling (ADR-008 D6 + the runner contract).** Inference is parallel across
zones; only commits contend, and only *same-zone* fills conflict (shared
``years_complete``/``runs`` attrs → ``RebaseFailedError``). So the driver runs
**year by year** (an outer serial loop) and, within a year, dispatches its zones
**concurrently** up to ``max_parallel_zones`` — all distinct zones, so no
same-zone overlap is ever possible. The fleet-wide committer bound is a separate
knob: ``commit_limit_name`` (a Prefect global concurrency limit) is passed to
every fill so commits stay under the storm threshold while inference runs free.
``pending()`` is year-major for exactly this drain pattern.
"""

from __future__ import annotations

import asyncio
from typing import Any

from prefect import flow, get_run_logger
from prefect.deployments import arun_deployment

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.inference.assembly import TARGET_AGGREGATE_S3_CONCURRENCY
from tessera_embeddings.orchestration.prefect.flows.ingest_zone_year import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
from tessera_embeddings.orchestration.runners.zone_fill import zone_year_on_axis
from tessera_embeddings.storage.campaign import campaign_status, campaign_work_list, tag_year_complete, zone_year_tag
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.zone_grid import CAMPAIGN_YEARS, canonicalize_zone


@flow(name="run-global-campaign")
async def run_global_campaign(
    *,
    paths: BucketPaths,
    ami_ssm_name: str,
    fill_deployment: str = "fill-zone-year/fill-zone-year",
    store_name: str = "tessera",
    years: tuple[int, ...] | None = None,
    zones: list[str] | None = None,
    max_parallel_zones: int = 8,
    commit_limit_name: str = "tessera-global-commits",
    num_actors: int = 20,
    s1_orbit: str = "both",
    ssm_prefix: str = "/tessera/ray/",
    cloudwatch_log_group: str = "/ec2/tessera/ray",
    code_bucket: str | None = None,
    code_suffix: str = "",
    # Campaign-triggered per-zone ingestion
    ingest: bool = True,
    ingest_deployment: str = "ingest-zone-year/ingest-zone-year",
    mask_name: str = "global",
    max_parallel_ingest: int = 2,
    cleanup_mosaics: bool = True,
    ingest_min_workers: int = 1,
    ingest_max_workers: int = 50,
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
    batch_days: int = 30,
    allow_partial_window: bool = False,
    sweep_orphan_mosaics: bool = False,
) -> dict[str, Any]:
    """Fill every pending (zone, year), year-serial with bounded zone parallelism.

    Args:
        paths: Deployment storage contract.
        ami_ssm_name: SSM parameter name for the Ray GPU AMI (forwarded to fills).
        fill_deployment: ``flow-name/deployment-name`` of the fill deployment.
        store_name: Global-store repo basename.
        years: Campaign years to drive (default: all campaign years).
        zones: Restrict the fill chain (inference + assembly) to these UTM zones in
            the ergonomic ``"<1-60><N|S>"`` form, e.g. ``["33N", "15S"]``; ``None``
            (default) drives all 120. Either way only cells still needing work are
            dispatched, so a default re-run of a partially-complete year skips the
            finished zones and fills only the unfinished ones (see
            :func:`campaign_work_list`).
        max_parallel_zones: Max concurrent fill runs *within a year* (bounds
            simultaneous Ray clusters — a cost knob, distinct from the commit
            gate).
        commit_limit_name: Prefect global concurrency limit bounding fleet-wide
            simultaneous committers (D6); forwarded to every fill.
        num_actors: GPU actor count, forwarded to each fill.
        s1_orbit: S1 orbit selection, forwarded to both ingest and fill.
        ssm_prefix: SSM prefix for Ray resources, forwarded to each fill.
        cloudwatch_log_group: CloudWatch log group, forwarded to each fill.
        code_bucket: Source-tarball bucket, forwarded to each fill.
        code_suffix: Source-tarball suffix, forwarded to each fill.
        ingest: Trigger per-zone ingestion (``ingest_zone_year``) before each
            fill (default). Set False when the mosaics already exist upstream.
        ingest_deployment: ``flow-name/deployment-name`` of the ingest deployment.
        mask_name: Coverage-store basename, forwarded to ingest.
        max_parallel_ingest: Max concurrent ingests (each provisions its own Dask
            cluster) — a separate, smaller knob than the fill cap.
        cleanup_mosaics: Delete ``mosaics/{zone}/{year}`` (all versions) after the
            fill lands (default; the mosaic is a transient input). Keep for dev.
        ingest_min_workers: Lower Dask worker bound for ingest.
        ingest_max_workers: Upper Dask worker bound for ingest.
        min_valid_coverage: S2 per-solar-day keep threshold forwarded to ingest.
        batch_days: S1 CMR batch window forwarded to ingest.
        allow_partial_window: Relax the coverage gate (ingest + fill) to
            "non-empty" for legitimately partial edge zones.
        sweep_orphan_mosaics: Before the run, delete mosaics for cells that are
            already complete+tagged in scope — recovering orphans left by a
            per-cell cleanup that failed after tagging (that cell is no longer in
            `work`, so it is never retried otherwise). Off by default; the zero-cost
            backstop for transient mosaics is an S3 lifecycle rule on the mosaics
            prefix, which this complements for immediate reclamation.

    Returns:
        Summary: pending count at start, dispatched run ids per year, totals.
    """
    log = get_run_logger()
    if max_parallel_zones < 1:
        raise ValueError(f"max_parallel_zones must be >= 1, got {max_parallel_zones} (Semaphore(0) blocks forever)")
    if max_parallel_ingest < 1:
        raise ValueError(f"max_parallel_ingest must be >= 1, got {max_parallel_ingest} (Semaphore(0) blocks forever)")
    campaign_years = tuple(years) if years is not None else CAMPAIGN_YEARS

    # Lazy AWS import so the flow file imports on non-AWS machines (arch tests).
    # The driver reads the global store directly (status, tags, on-axis probe)
    # BEFORE any child flow is dispatched, so a deployment whose store authenticates
    # only through the callback needs it here too — not just in the children.
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

    store_path = paths.global_store(store_name)
    repo = open_global_repo(store_path, get_credentials=iam_icechunk_credentials)
    status = campaign_status(repo, years=campaign_years)
    existing_tags = set(repo.list_tags())

    # Reject off-axis years up front, before dispatching any ingest/fill: the time
    # axis is fixed at seeding (ADR-008 D1), so e.g. `years=(2026,)` against a
    # 2017-2025 store would otherwise ingest + provision Ray for every zone only
    # for the fill to reject each one. Validate against a seeded zone's axis.
    seeded_zones = sorted(status.zones)
    if seeded_zones:
        off_axis = [
            y
            for y in campaign_years
            if not zone_year_on_axis(store_path, seeded_zones[0], y, get_credentials=iam_icechunk_credentials)
        ]
        if off_axis:
            raise ValueError(
                f"Year(s) {off_axis} are not on the store's pre-allocated axis "
                f"(fixed at seeding, ADR-008 D1) — reseed the store or drop them from `years`."
            )

    # Normalize the requested zones to canonical common names ("33N", "07S") so a
    # bad id fails loudly here and matches the store's group names exactly.
    expected_zones = [canonicalize_zone(z) for z in zones] if zones is not None else None
    # An explicit zones subset must be seeded: a typo that canonicalizes to a valid
    # but unseeded zone would otherwise be treated as pending and ingest+fill it
    # only for the fill to reject an unseeded group. (The default None spans all 120;
    # a not-fully-seeded store surfaces at the fill, as before.)
    if expected_zones is not None:
        unseeded = [z for z in expected_zones if z not in status.zones]
        if unseeded:
            raise ValueError(f"Requested zone(s) {unseeded} are not seeded in {store_path} — run the seed flow first.")

    # Work list = cells still needing a fill, restricted to `zones` (default: all
    # 120). campaign_work_list is tag-aware: it skips landed-and-tagged cells (so a
    # default re-run of a partially-complete year only touches the unfinished zones)
    # and INCLUDES complete-but-untagged cells (crash between fill and tag → the
    # runner's idempotent retag path runs). See its docstring for the two use cases.
    work = campaign_work_list(status, existing_tags, expected_zones=expected_zones, years=campaign_years)

    # Every work cell must be seeded before we dispatch. The explicit-zones guard
    # above covers a requested subset, but the default `zones=None` expands to all
    # 120 in campaign_work_list — so a partially-seeded store would otherwise ingest
    # each unseeded cell (expensive) only for the fill to reject the missing group.
    # Validate the actual work zones here to catch that before any ingestion.
    unseeded_work = sorted({z for z, _ in work if z not in status.zones})
    if unseeded_work:
        raise ValueError(
            f"Zone(s) {unseeded_work} in the work list are not seeded in {store_path} — run the seed flow "
            "first (a partially-seeded store would ingest each cell only for the fill to reject it)."
        )

    # Orphan-mosaic recovery: a per-cell cleanup that failed after tagging leaves
    # the mosaic behind, and that cell is no longer in `work`, so it is never
    # retried. Sweep complete-and-tagged cells in scope (best-effort) before the
    # run. Opt-in — the recommended zero-cost backstop is an S3 lifecycle rule.
    if sweep_orphan_mosaics and ingest and cleanup_mosaics:
        work_set = set(work)
        sweep_zones = expected_zones if expected_zones is not None else sorted(status.zones)
        swept = 0
        for y in set(campaign_years):
            for z in sweep_zones:
                if (z, y) not in work_set and status.has(z, y) and zone_year_tag(z, y) in existing_tags:
                    delete_prefix(f"{paths.inputs.rstrip('/')}/mosaics/{z}/{y}", log=log)
                    swept += 1
        log.info("Orphan-mosaic sweep: reclaimed %d completed-cell mosaic prefix(es)", swept)

    log.info(
        "Campaign: %d (zone, year) cell(s) need work across %d year(s); <=%d concurrent fills/year, commit limit %r",
        len(work),
        len({y for _, y in work}),
        max_parallel_zones,
        commit_limit_name,
    )

    def _fill_params(zone: str, year: int) -> dict[str, Any]:
        return {
            "zone": zone,
            "year": year,
            "paths": paths.model_dump(),
            # Deterministic, cell-unique run_id: a retry of a failed/cancelled fill
            # then RESUMES the same staging prefix (and that prefix is findable for
            # cleanup) instead of the runner minting a fresh uuid every attempt and
            # orphaning terabytes of un-resumable staged shards. Cell-unique (zone AND
            # year) so it can never collide with another cell's staged labels.
            "run_id": f"{zone}-{year}",
            "ami_ssm_name": ami_ssm_name,
            "store_name": store_name,
            "mask_name": mask_name,
            "num_actors": num_actors,
            "s1_orbit": s1_orbit,
            "commit_limit_name": commit_limit_name,
            "allow_partial_window": allow_partial_window,
            # Divide the fleet S3-PUT budget across concurrent fills so K shard-write
            # phases don't burst K times the target PUTs (the ~800-req SlowDown). D6
            # gates committers; this bounds the ungated upload phase.
            "s3_concurrency": max(1, TARGET_AGGREGATE_S3_CONCURRENCY // max_parallel_zones),
            "ssm_prefix": ssm_prefix,
            "cloudwatch_log_group": cloudwatch_log_group,
            "code_bucket": code_bucket,
            "code_suffix": code_suffix,
        }

    def _ingest_params(zone: str, year: int) -> dict[str, Any]:
        return {
            "zone": zone,
            "year": year,
            "paths": paths.model_dump(),
            "mask_name": mask_name,
            "s1_orbit": s1_orbit,
            "min_workers": ingest_min_workers,
            "max_workers": ingest_max_workers,
            "min_valid_coverage": min_valid_coverage,
            "batch_days": batch_days,
            "allow_partial_window": allow_partial_window,
        }

    fill_sem = asyncio.Semaphore(max_parallel_zones)
    ingest_sem = asyncio.Semaphore(max_parallel_ingest)

    async def _process(zone: str, year: int) -> str:
        # Ingest (if enabled) → fill → drop the transient mosaic. Ingest and fill
        # have separate concurrency caps; the mosaic is deleted only after the
        # fill is tagged complete.
        #
        # A complete-but-untagged cell (in years_complete, missing its tag) is in
        # the work list only so the fill re-creates the tag — no inference, no
        # mosaic. Skip ingest and cleanup for it entirely.
        retag_only = status.has(zone, year)
        did_ingest = ingest and not retag_only
        if did_ingest:
            async with ingest_sem:  # each ingest provisions its own Dask cluster
                irun = await arun_deployment(ingest_deployment, parameters=_ingest_params(zone, year))
                _check_completed(irun, f"ingest {zone}-{year}")
        async with fill_sem:  # bound concurrent fills (hence Ray clusters) within a year
            frun = await arun_deployment(fill_deployment, parameters=_fill_params(zone, year))
            _check_completed(frun, f"fill {zone}-{year}")
        # Only delete a mosaic THIS campaign produced: with ingest=False the mosaic
        # is an upstream input we must not remove.
        if did_ingest and cleanup_mosaics:
            delete_prefix(f"{paths.inputs.rstrip('/')}/mosaics/{zone}/{year}", log=log)
        return str(frun.id)

    # Descending: the campaign fills the current year first, then backwards
    # (ADR-008) — deliver the most-recent year soonest. Override via `years`.
    # Dedupe (campaign_work_list already dedupes years, so a duplicate here would
    # re-dispatch the same static work entries against a non-refreshed status).
    runs_by_year: dict[int, list[str]] = {}
    for year in sorted(set(campaign_years), reverse=True):
        year_zones = [z for z, y in work if y == year]
        if year_zones:
            log.info("Year %d: dispatching %d zone fill(s)", year, len(year_zones))
            # All zones in a year are distinct groups → safe to fill concurrently.
            # The outer loop is serial, so the SAME zone never fills two years at once.
            # return_exceptions=True so a single zone's failure doesn't abandon its
            # siblings mid-flight (default gather() would leave them running orphaned);
            # we let the whole year settle, then fail loudly with every failure.
            results = await asyncio.gather(*(_process(z, year) for z in year_zones), return_exceptions=True)
            failures = [(z, r) for z, r in zip(year_zones, results, strict=True) if isinstance(r, BaseException)]
            if failures:
                raise RuntimeError(
                    f"year {year}: {len(failures)}/{len(year_zones)} zone fill(s) failed "
                    f"(e.g. {failures[0][0]}: {failures[0][1]})"
                )
            runs_by_year[year] = [str(r) for r in results]
            log.info("Year %d: %d fill(s) landed", year, len(runs_by_year[year]))

        # Backfill the year-complete milestone/retention tag even when NO fills
        # ran this pass — a prior run may have completed every zone for the year
        # but crashed before tagging. The milestone is defined over ALL 120 zones
        # (expected_zones defaults to the full set, NOT this run's `zones` subset,
        # so a subset/repair run never stamps the global tag prematurely — and
        # tags are write-once). ValueError = not yet all-120 complete → defer.
        try:
            year_tag = tag_year_complete(repo, year)
            log.info("Year %d tagged complete (all 120 zones): %s", year, year_tag)
        except ValueError as exc:
            log.debug("Year %d not yet complete across all 120 zones (%s) — milestone tag deferred", year, exc)

    dispatched = sum(len(v) for v in runs_by_year.values())
    log.info("Campaign dispatch complete: %d fill run(s) across %d year(s)", dispatched, len(runs_by_year))
    return {"work_at_start": len(work), "dispatched": dispatched, "runs_by_year": runs_by_year}
