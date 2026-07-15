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
from tessera_embeddings.orchestration.prefect.flows.tessera_full_pipeline import _check_completed
from tessera_embeddings.storage.campaign import campaign_status
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.zone_grid import CAMPAIGN_YEARS


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
) -> dict[str, Any]:
    """Fill every pending (zone, year), year-serial with bounded zone parallelism.

    Args:
        paths: Deployment storage contract.
        ami_ssm_name: SSM parameter name for the Ray GPU AMI (forwarded to fills).
        fill_deployment: ``flow-name/deployment-name`` of the fill deployment.
        store_name: Global-store repo basename.
        years: Campaign years to drive (default: all campaign years).
        zones: Restrict to these EPSG strings (default: all 120).
        max_parallel_zones: Max concurrent fill runs *within a year* (bounds
            simultaneous Ray clusters — a cost knob, distinct from the commit
            gate).
        commit_limit_name: Prefect global concurrency limit bounding fleet-wide
            simultaneous committers (D6); forwarded to every fill.
        num_actors: GPU actor count, forwarded to each fill.
        s1_orbit: S1 orbit selection, forwarded to each fill.
        ssm_prefix: SSM prefix for Ray resources, forwarded to each fill.
        cloudwatch_log_group: CloudWatch log group, forwarded to each fill.
        code_bucket: Source-tarball bucket, forwarded to each fill.
        code_suffix: Source-tarball suffix, forwarded to each fill.

    Returns:
        Summary: pending count at start, dispatched run ids per year, totals.
    """
    log = get_run_logger()
    if max_parallel_zones < 1:
        raise ValueError(f"max_parallel_zones must be >= 1, got {max_parallel_zones} (Semaphore(0) blocks forever)")
    campaign_years = tuple(years) if years is not None else CAMPAIGN_YEARS

    repo = open_global_repo(paths.global_store(store_name))
    status = campaign_status(repo, years=campaign_years)
    pending = status.pending(expected_zones=zones, years=campaign_years)
    if not pending:
        log.info("Campaign already complete for the requested zones/years — nothing to dispatch")
        return {"pending_at_start": 0, "dispatched": 0, "runs_by_year": {}}

    log.info(
        "Campaign: %d pending (zone, year) cells across %d year(s); <=%d concurrent fills/year, commit limit %r",
        len(pending),
        len({y for _, y in pending}),
        max_parallel_zones,
        commit_limit_name,
    )

    def _params(zone: str, year: int) -> dict[str, Any]:
        return {
            "zone": zone,
            "year": year,
            "paths": paths.model_dump(),
            "ami_ssm_name": ami_ssm_name,
            "store_name": store_name,
            "num_actors": num_actors,
            "s1_orbit": s1_orbit,
            "commit_limit_name": commit_limit_name,
            "ssm_prefix": ssm_prefix,
            "cloudwatch_log_group": cloudwatch_log_group,
            "code_bucket": code_bucket,
            "code_suffix": code_suffix,
        }

    sem = asyncio.Semaphore(max_parallel_zones)

    async def _fill(zone: str, year: int) -> str:
        async with sem:  # bound concurrent fills (hence Ray clusters) within a year
            run = await arun_deployment(fill_deployment, parameters=_params(zone, year))
            _check_completed(run, f"fill {zone}-{year}")
            return str(run.id)

    runs_by_year: dict[int, list[str]] = {}
    for year in campaign_years:
        year_zones = [z for z, y in pending if y == year]
        if not year_zones:
            continue
        log.info("Year %d: dispatching %d zone fill(s)", year, len(year_zones))
        # All zones in a year are distinct groups → safe to fill concurrently.
        # The outer loop is serial, so the SAME zone never fills two years at once.
        # return_exceptions=True so a single zone's failure doesn't abandon its
        # siblings mid-flight (default gather() would leave them running orphaned);
        # we let the whole year settle, then fail loudly with every failure.
        results = await asyncio.gather(*(_fill(z, year) for z in year_zones), return_exceptions=True)
        failures = [(z, r) for z, r in zip(year_zones, results, strict=True) if isinstance(r, BaseException)]
        if failures:
            raise RuntimeError(
                f"year {year}: {len(failures)}/{len(year_zones)} zone fill(s) failed "
                f"(e.g. {failures[0][0]}: {failures[0][1]})"
            )
        runs_by_year[year] = [str(r) for r in results]
        log.info("Year %d complete: %d fill(s) landed", year, len(runs_by_year[year]))

    dispatched = sum(len(v) for v in runs_by_year.values())
    log.info("Campaign dispatch complete: %d fill run(s) across %d year(s)", dispatched, len(runs_by_year))
    return {"pending_at_start": len(pending), "dispatched": dispatched, "runs_by_year": runs_by_year}
