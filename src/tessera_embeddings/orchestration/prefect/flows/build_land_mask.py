"""Build the campaign land-mask coverage store (ADR-010).

Turns the partner's per-0.1°-cell TIFF delivery into per-zone tile-liveness
bitmaps in one Icechunk repo (:meth:`BucketPaths.land_mask_store`), which the
zone-fill runner reads to pick live tiles. This sits alongside the ingest and
embeddings flows as the campaign's mask-preparation step.

No Dask cluster required — the build is pure geometry over ``registry.txt``
(project each land cell's footprint into its UTM zone, OR into bitmaps), so it
runs entirely on the flow runner in well under a minute, like
:mod:`generate_roi`. Optional pre-build verification reads a sample of delivery
TIFFs to guard the v1.1 all-1s assumption; post-build validation checks bitmap
consistency and a few known land/ocean points.

Credentials: the build/verify path uses the ambient AWS credential chain (env /
instance profile) for the registry read, the bucket listing, and the coverage
repo — this is a one-off admin job, not the credential-scoped campaign hot path.
Deploy it where that chain resolves to the delivery + inputs buckets.
"""

from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger

from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.ingest.land_mask import DEFAULT_DELIVERY_URI
from tessera_embeddings.orchestration.prefect.tasks.land_mask import (
    build_land_mask_coverage,
    validate_land_mask_coverage,
    verify_land_mask_delivery,
)


@flow(name="build-land-mask-coverage")
def build_land_mask(
    *,
    paths: BucketPaths,
    name: str = "global",
    registry_uri: str | None = None,
    delivery_uri: str = DEFAULT_DELIVERY_URI,
    zones: list[str] | None = None,
    verify: bool = True,
    spot_check: int = 500,
    reconcile: bool = True,
    run_validation: bool = True,
) -> dict[str, Any]:
    """Build per-zone coverage bitmaps and (optionally) verify + validate them.

    Args:
        paths: Deployment storage contract; the coverage repo is
            ``paths.land_mask_store(name)`` (under ``inputs``).
        name: Coverage repo basename (default ``"global"``).
        registry_uri: Registry file URI; defaults to ``{delivery_uri}/registry.txt``.
        delivery_uri: Partner delivery prefix.
        zones: Restrict to these EPSG strings (default: all 120).
        verify: When True, reconcile the registry against the bucket and
            spot-check delivery tiles *before* building (guards the all-1s
            assumption the design rests on).
        spot_check: Number of delivery TIFFs to sample when verifying (0 = skip).
        reconcile: When verifying, also LIST the bucket and assert registry ⊆
            bucket. Off for a fast verify that only samples tiles.
        run_validation: When True, run structural + geographic self-checks on
            the built store.

    Returns:
        The build summary dict (snapshot id, counts, registry sha256), plus a
        ``verify`` sub-dict when verification ran.
    """
    log = get_run_logger()
    dest = paths.land_mask_store(name)
    resolved_registry = registry_uri or f"{delivery_uri.rstrip('/')}/registry.txt"

    # The three steps are strictly sequential (verify → build → validate), so
    # call the tasks directly — each still runs as a tracked Prefect task; there
    # is no concurrency to warrant submit()/futures.
    summary: dict[str, Any] = {}
    if verify:
        log.info("Verifying delivery %s before build", delivery_uri)
        summary["verify"] = verify_land_mask_delivery(
            registry_uri=resolved_registry,
            delivery_uri=delivery_uri,
            spot_check=spot_check,
            reconcile=reconcile,
        )

    log.info("Building coverage store %s", dest)
    result = build_land_mask_coverage(
        dest=dest,
        registry_uri=resolved_registry,
        delivery_uri=delivery_uri,
        zones=zones,
    )
    summary.update(result)

    if run_validation:
        log.info("Validating coverage store %s", dest)
        validate_land_mask_coverage(dest=dest, zones=zones)

    log.info("Land-mask coverage ready at %s (%s)", dest, result["snapshot_id"])
    return summary
