"""Pre-generate and validate every zone's ingest ROI mask, ahead of the campaign.

The campaign's per-cell ingest (:mod:`.ingest_zone_year`) exports the zone mask it
needs on the fly, so this flow adds no capability the campaign lacks. What it adds
is *sequencing*: the export moves off the critical path, and — the reason it
exists — a bad mask is found in a cheap batch run instead of hours into a fill.
The mask pins the ingest's grid and the fill validates the resulting mosaic
against the same :class:`~tessera_embeddings.storage.zone_grid.ZoneSpec`, so a
wrong transform surfaces late and expensively.

Every zone is independent, so the run is a flat fan-out over zones with no
barrier: one task per zone, ``max_parallel_zones`` in flight. The work is S3 PUT
latency (one object per live ingest chunk), which is why threads are the right
substrate and why in-region concurrency dominates the runtime — a laptop managed
about four chunk-writes a second on 35N.

**Safe to re-run, and safe to run before the campaign.** ``export_zone_roi`` is
idempotent on the coverage delivery's ``registry_sha256``, so a mask this flow
wrote is byte-identical to the one the campaign would have written and the
campaign skips it. A new land-mask delivery changes the sha and both paths
rebuild. ``validate_only=True`` re-checks without writing, which is how the
"every mask exists and validates" gate is re-asserted cheaply.

The flow FAILS if any zone fails validation, so its terminal state is the gate —
a green run is the evidence, not the log.
"""

from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger, task
from prefect.task_runners import ThreadPoolTaskRunner

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.ingest.land_mask import export_zone_roi, live_chunk_count, validate_zone_roi
from tessera_embeddings.storage.zone_grid import ZONES, canonicalize_zone

#: Zone-level fan-out width. The work is S3 request latency rather than CPU, so
#: the useful ceiling is well above the runner's core count; 12 keeps peak memory
#: modest (each in-flight zone upsamples one 4096-px bool block at a time) while
#: making the run bounded by the largest single zone rather than by their sum.
DEFAULT_MAX_PARALLEL_ZONES = 12


@task(name="export-zone-roi", task_run_name="roi-{zone}", retries=2, retry_delay_seconds=15)
def export_one_zone_roi(
    zone: str,
    *,
    land_mask_path: str,
    roi_path: str,
    validate_only: bool,
    s3_region: str | None,
) -> dict[str, Any]:
    """Export (unless ``validate_only``) and validate one zone's ROI mask.

    Retried: the body is a few thousand independent S3 writes, and a transient
    failure part-way through should not sink a 112-zone run. A retry is safe
    because the export is a whole-artifact rewrite, not an append — and because
    the coverage sha is stamped only after the last pixel, a mask abandoned
    mid-write is never mistaken for a current one.

    Returns:
        A row with the zone's ``status`` (``"exported"``, ``"validated"``,
        ``"all_ocean"``, or ``"invalid"``), its live chunk count, and any
        validation ``problems``. An all-ocean zone has no mask by design and is
        neither written nor validated.
    """
    log = get_run_logger()
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

    creds = iam_icechunk_credentials
    n_chunks = live_chunk_count(zone, land_mask_path=land_mask_path, get_credentials=creds, s3_region=s3_region)
    if n_chunks == 0:
        log.info("Zone %s is all-ocean — no ROI mask expected", zone)
        return {"zone": zone, "status": "all_ocean", "live_chunks": 0, "problems": []}

    if not validate_only:
        export_zone_roi(
            zone,
            land_mask_path=land_mask_path,
            dest_path=roi_path,
            get_credentials=creds,
            s3_region=s3_region,
        )

    problems = validate_zone_roi(
        zone, land_mask_path=land_mask_path, roi_path=roi_path, get_credentials=creds, s3_region=s3_region
    )
    if problems:
        log.error("Zone %s ROI mask is INVALID at %s: %s", zone, roi_path, "; ".join(problems))
        return {"zone": zone, "status": "invalid", "live_chunks": n_chunks, "problems": problems}

    log.info("Zone %s ROI mask OK: %d live chunk(s) at %s", zone, n_chunks, roi_path)
    return {
        "zone": zone,
        "status": "validated" if validate_only else "exported",
        "live_chunks": n_chunks,
        "problems": [],
    }


@flow(name="export-zone-rois-impl")
def _export_zone_rois_impl(
    *,
    zones: list[str],
    paths: BucketPaths,
    land_mask_path: str,
    validate_only: bool,
    s3_region: str | None,
) -> list[dict[str, Any]]:
    """Inner flow: submit one task per zone to the configured thread pool.

    Submitted in one pass with no barrier between zones — a slow dense zone never
    holds up the rest, and the run's wall clock is the largest single zone rather
    than the sum.
    """
    futures = [
        export_one_zone_roi.submit(
            zone,
            land_mask_path=land_mask_path,
            roi_path=paths.zone_roi_store(zone),
            validate_only=validate_only,
            s3_region=s3_region,
        )
        for zone in zones
    ]
    return [f.result() for f in futures]


@flow(name="export-zone-rois")
def export_zone_rois(
    *,
    paths: BucketPaths,
    zones: list[str] | None = None,
    mask_name: str = "global",
    max_parallel_zones: int = DEFAULT_MAX_PARALLEL_ZONES,
    validate_only: bool = False,
    s3_region: str | None = None,
) -> dict[str, Any]:
    """Export and validate the ROI masks for ``zones`` (default: all 120).

    The two-flow split is the repo's standard task-runner idiom: ``task_runner=``
    binds at flow-definition time, so the fan-out width can only be a runtime
    parameter if the outer flow passes it to an inner one via ``with_options``.

    Args:
        paths: Deployment storage contract. Masks are written to
            ``paths.zone_roi_store(zone)`` — the same path the ingest reads.
        zones: UTM zone common names (e.g. ``["33N", "35N"]``). Defaults to every
            zone in the grid; the all-ocean ones cost one bitmap read and are
            reported as such rather than being excluded up front, so the land
            zone count is an output of the run and not a hardcoded constant.
        mask_name: Land-mask coverage repo basename.
        max_parallel_zones: Zones in flight at once.
        validate_only: Check the existing masks and write nothing. Use this to
            re-assert the gate, and to distinguish "not built" from "built wrong".
        s3_region: Region for the coverage-repo reads, if not the default.

    Returns:
        Summary counts plus the per-zone rows, and ``invalid_zones`` naming any
        that failed.

    Raises:
        ValueError: If any zone's mask fails validation — the flow's terminal
            state is what makes a green run usable as the campaign's entry gate.
    """
    log = get_run_logger()
    todo = [canonicalize_zone(z) for z in (zones if zones is not None else list(ZONES))]
    log.info(
        "%s ROI masks for %d zone(s), %d in flight",
        "Validating" if validate_only else "Exporting",
        len(todo),
        max_parallel_zones,
    )

    rows = _export_zone_rois_impl.with_options(  # type: ignore[call-overload]
        task_runner=ThreadPoolTaskRunner(max_workers=max_parallel_zones)
    )(
        zones=todo,
        paths=paths,
        land_mask_path=paths.land_mask_store(mask_name),
        validate_only=validate_only,
        s3_region=s3_region,
    )

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    invalid = [row["zone"] for row in rows if row["status"] == "invalid"]
    summary = {
        "zones": len(rows),
        "by_status": by_status,
        "live_chunks": sum(row["live_chunks"] for row in rows),
        "invalid_zones": invalid,
        "rows": rows,
    }
    log.info("ROI masks: %s over %d live chunk(s)", by_status, summary["live_chunks"])
    if invalid:
        raise ValueError(f"{len(invalid)} zone ROI mask(s) failed validation: {', '.join(invalid)}")
    return summary
