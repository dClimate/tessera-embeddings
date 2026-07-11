"""Merge many master-snapped feature stores into one master mosaic (a Prefect flow).

The orchestration-visible entry point for the multiple-regional-inserts use case:
given the per-feature stores an upstream fan-out produced (each an exact
pixel-subset of a shared master grid), region-write them all into a single master
mosaic, one commit per feature, and optionally delete the temps. See
``context_docs/design/region-merge.md`` for the design and
``docs/region-merge.md`` for the how-to.

This is a **thin** flow over the domain driver
:func:`tessera_embeddings.storage.region_merge.merge_stores` — it pulls the
Prefect run logger, reads the master ROI grid, and delegates. All the merge logic
(date-union seeding, sequential region writes, manifest splitting, cleanup) lives
in the domain layer, so the plain/`examples` path and this flow share one
implementation. Like ``generate_roi``, it needs **no Dask/Ray cluster** — the
merge is process-parallel raw-Zarr chunk movement on the flow runner — so there is
no task-runner / provider-cluster dance.
"""

from __future__ import annotations

import numpy as np
from prefect import flow, get_run_logger

from tessera_embeddings.ingest.roi import read_roi_metadata
from tessera_embeddings.storage.region_merge import merge_stores


@flow(name="merge-mosaic", log_prints=True)
def merge_mosaic(
    *,
    master_path: str,
    feature_paths: list[str],
    roi_zarr_path: str,
    var_dtypes: dict[str, str],
    tile_id: str,
    crs: str | None = None,
    delete_temp: bool = False,
    overwrite_master: bool = False,
    resume: bool = False,
    max_workers: int | None = None,
    threads_per_process: int | None = None,
    max_concurrent_requests: int | None = None,
    feature_retries: int = 1,
) -> dict:
    """Seed a master over the feature date union and region-write each feature into it.

    Args:
        master_path: Destination master store URI (created here unless ``resume``).
        feature_paths: Per-feature store URIs to merge, in application order. Each
            must be **master-snapped** — an exact pixel-subset of the master grid
            (same CRS, resolution, axis order; coords a contiguous subset). Produce
            them on a window of the master grid (e.g. ``master_geobox.enclosing``);
            the merge validates this and raises if it is not met.
        roi_zarr_path: Master ROI mask URI (the grid authority) — read once and
            passed to the seed.
        var_dtypes: ``{var_name: dtype_string}`` the master is seeded with, e.g.
            ``{"0_VV": "uint16", "0_VH": "uint16"}``. Strings (not ``np.dtype``) so
            they cross the Prefect parameter boundary cleanly; converted here.
        tile_id: Store-metadata identifier for the seeded master.
        crs: CRS authority code; defaults to the ROI's native CRS.
        delete_temp: Delete each feature store after a successful merge (default
            False — the caller owns temp lifecycle).
        overwrite_master: If the master exists (and not ``resume``): False (default)
            fails; True rebuilds it.
        resume: Merge into an existing master without re-seeding (pick up a partial
            run); pass only the not-yet-merged features.
        max_workers: Worker processes per feature copy (default: one per core).
        threads_per_process: Thread-pool width inside each worker (default: small).
        max_concurrent_requests: Caps per-repo/per-fork S3 concurrency AND each
            worker's feature-store read; lower it (e.g. 64) to avoid 503 SlowDown
            when many processes hit one prefix. ``None`` leaves icechunk's 256.
        feature_retries: Retries per feature on a worker stall (default 1; the
            region write is idempotent, so a fresh-session retry is safe).

    Returns:
        The :func:`merge_stores` summary dict (``master_path``, ``n_dates``,
        ``merged``, ``deleted``, ``skipped``, ``elapsed_sec``).
    """
    log = get_run_logger()
    roi = read_roi_metadata(roi_zarr_path)
    dtypes = {name: np.dtype(dt) for name, dt in var_dtypes.items()}
    log.info(
        "merge-mosaic: %d feature(s) → %s (%dx%dpx, crs=%s)",
        len(feature_paths),
        master_path,
        roi.width,
        roi.height,
        roi.native_crs,
    )
    return merge_stores(
        master_path,
        feature_paths,
        roi=roi,
        var_dtypes=dtypes,
        tile_id=tile_id,
        crs=crs,
        delete_temp=delete_temp,
        overwrite_master=overwrite_master,
        resume=resume,
        max_workers=max_workers,
        threads_per_process=threads_per_process,
        max_concurrent_requests=max_concurrent_requests,
        feature_retries=feature_retries,
        log=log,
    )
