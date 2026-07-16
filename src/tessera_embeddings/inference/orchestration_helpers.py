"""Pure helpers shared by Prefect task shells, the plain runner, and tests.

These functions used to live in the reference repo's ``flow_utils/inference.py``
alongside the ``@task`` shells. They have no Prefect coupling and don't
belong in any individual flow file, so they live here in the inference
domain layer.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from typing import Any

from tessera_embeddings.config.inference import InferenceConfig, TimeWindow
from tessera_embeddings.inference.chunk_spec import ChunkSpec, enumerate_chunks_from_dataset
from tessera_embeddings.inference.data_loading import _active_orbits
from tessera_embeddings.storage.manifest import extract_manifest
from tessera_embeddings.storage.zarr_store import open_store, open_store_as_zarr_group


def read_upstream_manifests(
    mosaic_base: str,
    s1_orbit: str,
    *,
    get_credentials: Callable[[], Any] | None = None,
    s3_region: str | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Read ``_manifest`` attrs from all active ingest stores under ``mosaic_base``.

    Opens via the raw-zarr metadata reader with the caller's credential callback /
    region (the same ones the assemble write uses), so a callback-only or
    non-default-region deployment doesn't fall back to the default Icechunk chain
    and fail this manifest read on the flow runner before ``writer.assemble()``.
    """
    store_names = ["reflectance"]
    for orbit in _active_orbits(s1_orbit):
        store_names.append(f"sar_{orbit}")

    manifests: dict[str, dict[str, Any] | None] = {}
    for name in store_names:
        path = f"{mosaic_base}/{name}.zarr"
        # Raw-zarr open reads only the root attrs off metadata — no Dask graph over
        # a full-zone mosaic's chunks just to read one attr dict.
        root = open_store_as_zarr_group(path, get_credentials=get_credentials, region=s3_region)
        manifests[name] = extract_manifest(dict(root.attrs))
    return manifests


def checkpoint_to_version(checkpoint_path: str) -> str:
    """Derive a model version string from a checkpoint path.

    Extracts the filename stem, e.g.
    ``"s3://bucket/models/best_model_fsdp_20250608_220648_QAT.pt"`` →
    ``"best_model_fsdp_20250608_220648_QAT"``.
    """
    return checkpoint_path.rsplit("/", 1)[-1].removesuffix(".pt")


def enumerate_mosaic_chunks(
    mosaic_base: str,
    chunk_size: int,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger],
) -> tuple[list[ChunkSpec], int, int]:
    """Open the reflectance store, enumerate the spatial chunk grid, then close it.

    The store is explicitly closed and garbage-collected after
    enumeration to release the Icechunk session before the Ray cluster
    starts (Ray's pickling chokes on a live session).
    """
    reflectance_path = f"{mosaic_base}/reflectance.zarr"
    log.info("Reading store metadata from %s", reflectance_path)
    # chunks=None: only the dim sizes are read here (never pixels), so skip
    # building a Dask task graph over every chunk of the full-zone mosaic.
    ds = open_store(reflectance_path, chunks=None)
    try:
        chunks = enumerate_chunks_from_dataset(ds, chunk_size)
        total_y, total_x = ds.sizes["northing"], ds.sizes["easting"]
    finally:
        ds.close()
        del ds  # ensure the Icechunk session is released before Ray starts
        gc.collect()
    log.info("Enumerated %d chunks for (%d x %d) mosaic", len(chunks), total_y, total_x)
    return chunks, total_y, total_x


def build_inference_config(
    *,
    s1_orbit: str,
    time_window: TimeWindow,
    checkpoint_path: str,
    inputs_bucket: str,
    output_bucket: str,
    **overrides: Any,  # noqa: ANN401 — pass-through to InferenceConfig dataclass kwargs
) -> InferenceConfig:
    """Build an :class:`InferenceConfig` from flow-level parameters.

    The reference repo's helper used a ``dev: bool`` toggle that
    selected hardcoded buckets and checkpoints; this version takes
    them as explicit caller arguments so the same code works for any
    deployment.
    """
    return InferenceConfig(
        time_window=time_window,
        s1_orbit=s1_orbit,  # type: ignore[arg-type]
        checkpoint_path=checkpoint_path,
        inputs_bucket=inputs_bucket,
        output_bucket=output_bucket,
        **overrides,
    )
