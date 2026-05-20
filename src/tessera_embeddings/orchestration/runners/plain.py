"""Orchestrator-free pipeline runner.

Default: runs ROI generation → S2 + S1 ingestion → CPU inference →
assembly end-to-end on a small example ROI using local Ray + local
Dask. Slow (~30+ min on a developer laptop) but proves the layer
separation is real — every step calls the same domain function the
Prefect flows do.

``--skip-inference`` short-circuits after ingestion for fast contributor
iteration on the ingest path.
"""

from __future__ import annotations

import argparse
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

import yaml
from dask.distributed import Client

from tessera_embeddings.config.dask import AssemblyConfig
from tessera_embeddings.config.inference import DEFAULT_CHUNK_SIZE, checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.inference.assembly import ZarrWriter
from tessera_embeddings.inference.chunk_spec import filter_chunks_by_roi_mask
from tessera_embeddings.inference.data_loading import _active_orbits, resolve_s1_orbit
from tessera_embeddings.inference.orchestration_helpers import (
    build_inference_config,
    checkpoint_to_version,
    compute_assembly_worker_counts,
    enumerate_mosaic_chunks,
    read_upstream_manifests,
)
from tessera_embeddings.inference.runner import run_inference
from tessera_embeddings.ingest.auth import get_s3_credentials, set_s3_credentials
from tessera_embeddings.ingest.roi import rasterize_roi_zarr
from tessera_embeddings.ingest.roi_processing import DEFAULT_MIN_VALID_COVERAGE
from tessera_embeddings.ingest.s1_roi import ingest_s1_roi_sar
from tessera_embeddings.ingest.s2_roi import ingest_s2_roi_reflectance
from tessera_embeddings.providers.local.dask import local_cluster
from tessera_embeddings.providers.local.ray import ray_cluster
from tessera_embeddings.storage.manifest import EmbeddingManifest


@contextmanager
def _dask_client(n_workers: int) -> Iterator[Client]:
    """Create a local Dask cluster + connected client."""
    with local_cluster(n_workers=n_workers) as cluster, Client(cluster) as client:
        yield client


def _run_ingest(
    *,
    roi_path: str,
    paths: BucketPaths,
    roi_name: str,
    start_date: str,
    end_date: str,
    s1_orbit: Literal["ascending", "descending", "both"],
    n_workers: int,
    log: logging.Logger,
    storage_options: dict | None,
    s1_use_s3_direct: bool,
) -> None:
    """Run S2 + S1 ingestion sequentially against a local Dask cluster.

    The two stages share one cluster but run sequentially because the
    plain runner is single-process; the cost of separate clusters
    outweighs any concurrency win on a laptop.
    """
    mosaic_base = paths.store_for(roi_name, "reflectance").rsplit("/", 1)[0]

    with _dask_client(n_workers) as client:
        log.info("Local Dask cluster ready: scheduler=%s", client.scheduler_info()["address"])

        log.info("Ingesting S2 reflectance into %s", mosaic_base)
        s2_result = ingest_s2_roi_reflectance(
            roi_zarr_path=roi_path,
            start_date=start_date,
            end_date=end_date,
            store_path=mosaic_base,
            client=client,
            log=log,
            storage_options=storage_options,
        )
        log.info("S2 ingestion: %s", s2_result)

        for orbit in _active_orbits(s1_orbit):
            log.info("Ingesting S1 SAR (orbit=%s, use_s3_direct=%s)", orbit, s1_use_s3_direct)
            # When S3 direct is on, the domain function expects callables that fetch
            # ASF STS credentials and broadcast them to the cluster. The Prefect
            # flow wires these the same way; without them GDAL hits /vsis3/ with no
            # AWS env vars and ASF rejects the request.
            s1_result = ingest_s1_roi_sar(
                roi_zarr_path=roi_path,
                start_date=start_date,
                end_date=end_date,
                store_path=mosaic_base,
                client=client,
                orbit=orbit,  # type: ignore[arg-type]
                use_s3_direct=s1_use_s3_direct,
                edl_credentials_fn=get_s3_credentials if s1_use_s3_direct else None,
                apply_credentials_fn=set_s3_credentials if s1_use_s3_direct else None,
                log=log,
                storage_options=storage_options,
            )
            log.info("S1 ingestion (%s): %s", orbit, s1_result)


def _run_inference_and_assemble(
    *,
    roi_path: str,
    roi_name: str,
    paths: BucketPaths,
    time_window_end: str,
    s1_orbit: Literal["ascending", "descending", "both"],
    checkpoint_dir: str | None,
    log: logging.Logger,
) -> dict[str, Any]:
    """Run CPU inference under local Ray, then assemble under local Dask."""
    inputs_bucket = paths.inputs
    output_bucket = paths.outputs

    time_window = parse_time_window(time_window_end)
    model_dir = checkpoint_dir or f"{inputs_bucket.rstrip('/')}/models"
    checkpoint_path = f"{model_dir.rstrip('/')}/{checkpoint_filename()}"

    mosaic_base = paths.store_for(roi_name, "reflectance").rsplit("/", 1)[0]
    staging_base = f"{output_bucket.rstrip('/')}/staging"

    effective_orbit = resolve_s1_orbit(mosaic_base, s1_orbit)
    if effective_orbit != s1_orbit:
        log.info("s1_orbit resolved: %s → %s", s1_orbit, effective_orbit)

    config = build_inference_config(
        s1_orbit=effective_orbit,
        time_window=time_window,
        checkpoint_path=checkpoint_path,
        inputs_bucket=inputs_bucket,
        output_bucket=output_bucket,
        num_gpus=0,
    )

    chunks, total_y, total_x = enumerate_mosaic_chunks(mosaic_base, config.chunk_size or DEFAULT_CHUNK_SIZE, log)
    live_chunks = filter_chunks_by_roi_mask(chunks, roi_path)
    log.info("ROI filter: %d/%d chunks intersect the ROI", len(live_chunks), len(chunks))

    run_id = uuid.uuid4().hex[:12]
    t0 = time.monotonic()

    log.warning(
        "Running CPU inference on %d chunks. Expect this to take a while; "
        "use --skip-inference for ingest-only sanity checks.",
        len(live_chunks),
    )

    with ray_cluster(num_gpus=0):
        results = run_inference(
            num_actors=1,
            config=config,
            chunks=live_chunks,
            mosaic_base=mosaic_base,
            staging_base=staging_base,
            run_id=run_id,
            t0=t0,
            log=log,
        )

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        msg = f"{len(failed)} chunks failed during inference"
        raise RuntimeError(msg)

    # Assembly under local Dask
    n_live = len(live_chunks)
    _min_workers, max_workers = compute_assembly_worker_counts(n_live, AssemblyConfig())
    output_path = f"{output_bucket.rstrip('/')}/embeddings/{roi_name}.zarr"
    writer = ZarrWriter(staging_base)
    model_version = checkpoint_to_version(config.checkpoint_path)
    upstream_manifests = read_upstream_manifests(mosaic_base, config.s1_orbit)
    manifest = EmbeddingManifest.from_upstream_stores(
        model_checkpoint=model_version,
        num_obs_checkpoints=config.num_obs_checkpoints,
        upstream_manifests=upstream_manifests,
    )

    with _dask_client(min(2, max_workers)):
        writer.assemble(
            chunks,
            total_y,
            total_x,
            run_id,
            output_path,
            roi_zarr_path=roi_path,
            mosaic_base=mosaic_base,
            log=log,
            time_window=time_window,
            tile_id=roi_name,
            model_version=model_version,
            manifest=manifest,
            n_workers=max_workers,
        )

    return {
        "run_id": run_id,
        "roi_name": roi_name,
        "output_path": output_path,
        "n_live_chunks": n_live,
        "elapsed_sec": time.monotonic() - t0,
    }


def run_plain(config_path: Path, *, skip_inference: bool = False) -> dict[str, Any] | None:
    """Run the full pipeline from a YAML config.

    Args:
        config_path: Path to the YAML config. Schema::

            paths:
              inputs: file:///tmp/tessera/inputs
              outputs: file:///tmp/tessera/outputs
              preprocessed: file:///tmp/tessera/preprocessed
            roi:
              name: my-roi
              geojson: examples/quickstart/roi.geojson    # optional
              resolution: 10.0
              chunk_size: 2000
              force_crs: null                              # or "EPSG:32615"
            time_window_end: "June 2025"
            time_range:
              start: "2024-07-01"
              end: "2025-07-01"
            s1_orbit: ascending    # or "descending" or "both"
            n_workers: 2
            checkpoint_dir: null    # override model directory; null → {inputs}/models/
            storage_options: null

        skip_inference: When True, stop after ingestion. Useful for
            iterating on the ingest path without paying CPU inference
            cost.

    Returns:
        Final assembly summary dict when inference + assembly run,
        else ``None``.
    """
    cfg = yaml.safe_load(config_path.read_text())
    log = logging.getLogger("tessera_embeddings.plain")
    if not log.handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    paths = BucketPaths(**cfg["paths"])
    roi_cfg = cfg["roi"]
    roi_name: str = roi_cfg["name"]
    roi_path = paths.store_for(roi_name, "roi")

    if not skip_inference:
        log.warning(
            "Plain runner will execute the full pipeline end-to-end with CPU inference. "
            "Expect ~30+ minutes on a developer laptop. Pass --skip-inference for an "
            "ingest-only sanity check."
        )

    # Stage 1: rasterize the ROI (no cluster needed)
    if roi_cfg.get("geojson"):
        log.info("Rasterising ROI from %s → %s", roi_cfg["geojson"], roi_path)
        rasterize_roi_zarr(
            output_path=roi_path,
            resolution=roi_cfg.get("resolution", 10.0),
            chunk_size=roi_cfg.get("chunk_size", DEFAULT_CHUNK_SIZE),
            force_crs=roi_cfg.get("force_crs"),
            input_path=roi_cfg["geojson"],
        )
    else:
        log.info("Using pre-rasterised ROI at %s", roi_path)

    # Stage 2: S2 + S1 ingestion
    time_range = cfg["time_range"]
    _run_ingest(
        roi_path=roi_path,
        paths=paths,
        roi_name=roi_name,
        start_date=time_range["start"],
        end_date=time_range["end"],
        s1_orbit=cfg.get("s1_orbit", "ascending"),
        n_workers=cfg.get("n_workers", 2),
        log=log,
        storage_options=cfg.get("storage_options"),
        # The default in the domain layer is True (S3 direct from us-west-2).
        # On a laptop outside us-west-2, default to CloudFront-signed HTTPS so
        # the quickstart works without ASF S3 STS credentials.
        s1_use_s3_direct=cfg.get("s1_use_s3_direct", False),
    )

    # ``--skip-inference`` ends here, with ingest output verified.
    if skip_inference:
        log.info("Skipping inference + assembly (--skip-inference)")
        return None

    # Stages 3 + 4: CPU inference + assembly
    summary = _run_inference_and_assemble(
        roi_path=roi_path,
        roi_name=roi_name,
        paths=paths,
        time_window_end=cfg["time_window_end"],
        s1_orbit=cfg.get("s1_orbit", "ascending"),
        checkpoint_dir=cfg.get("checkpoint_dir"),
        log=log,
    )
    log.info("Pipeline complete: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Tessera embeddings plain runner")
    parser.add_argument("config_path", type=Path, help="Path to YAML config")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Stop after ingestion; do not run CPU inference + assembly",
    )
    parser.add_argument(
        "--min-valid-coverage",
        type=float,
        default=DEFAULT_MIN_VALID_COVERAGE,
        help="Minimum SCL valid-pixel coverage to keep a date (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    run_plain(args.config_path, skip_inference=args.skip_inference)


if __name__ == "__main__":
    main()
