"""Prefect task shells for the land-mask coverage domain functions.

Thin wrappers (ADR-002 shape-C): pull the run logger from Prefect context,
delegate to the :mod:`tessera_embeddings.ingest.land_mask` domain functions, and
convert the dataclass result to a dict at the boundary. No Dask client is pulled
— the coverage build is pure geometry over the registry and runs entirely on the
flow runner (no cluster), like :mod:`generate_roi`.

This file is one of the few places in the package that imports from
:mod:`prefect`. Domain modules under ``ingest/`` never do.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from prefect import get_run_logger, task

from tessera_embeddings.ingest.land_mask import (
    DEFAULT_DELIVERY_URI,
    build_all,
    read_registry,
    reconcile_with_bucket,
    spot_check_delivery,
    validate_coverage,
)


@task(name="build-land-mask-coverage")
def build_land_mask_coverage(
    *,
    dest: str,
    registry_uri: str | None = None,
    delivery_uri: str = DEFAULT_DELIVERY_URI,
    zones: list[str] | None = None,
) -> dict[str, Any]:
    """Prefect task: build per-zone coverage bitmaps into ``dest`` (one commit)."""
    result = build_all(
        dest,
        registry_uri=registry_uri,
        delivery_uri=delivery_uri,
        zones=zones,
        log=get_run_logger(),
    )
    return asdict(result)


@task(name="verify-land-mask-delivery")
def verify_land_mask_delivery(
    *,
    registry_uri: str,
    delivery_uri: str = DEFAULT_DELIVERY_URI,
    spot_check: int = 500,
    reconcile: bool = True,
) -> dict[str, Any]:
    """Prefect task: reconcile the registry against the bucket + spot-check tiles.

    Guards the load-bearing v1.1 all-1s assumption *before* a build is trusted.
    """
    log = get_run_logger()
    names, sha = read_registry(registry_uri)
    out: dict[str, Any] = {"registry_sha256": sha, "n_registry": len(names)}
    if reconcile:
        _, n_bucket, n_extras = reconcile_with_bucket(names, delivery_uri=delivery_uri, log=log)
        out.update(n_bucket=n_bucket, n_extras=n_extras)
    if spot_check > 0:
        result = spot_check_delivery(names, delivery_uri=delivery_uri, n=spot_check, log=log)
        out["spot_check"] = asdict(result)
    return out


@task(name="validate-land-mask-coverage")
def validate_land_mask_coverage(*, dest: str, zones: list[str] | None = None) -> None:
    """Prefect task: structural + geographic self-checks on a built coverage repo."""
    validate_coverage(dest, zones=zones, log=get_run_logger())
