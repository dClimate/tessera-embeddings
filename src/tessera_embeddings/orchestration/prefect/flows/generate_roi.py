"""Generate a Zarr ROI mask from a GeoJSON polygon or S2 tile footprints.

Two input modes (mutually exclusive):

* **GeoJSON mode** (``roi_name``): reads
  ``{roi_bucket}/geojsons/{roi_name}.geojson`` and writes to
  ``{roi_bucket}/zarrs/{roi_name}.zarr``.
* **Tile mode** (``tile_names``): fetches Sentinel-2 MGRS tile
  footprints from a GeoJSON index on S3 and writes to
  ``{roi_bucket}/zarrs/{tile1}_{tile2}.zarr``.

No Dask cluster required — rasterisation is chunked and sequential,
running entirely on the flow runner.
"""

from __future__ import annotations

import zarr
from prefect import flow, get_run_logger

from tessera_embeddings.errors import ConfigMismatchError  # noqa: F401  (re-exported for callers)
from tessera_embeddings.ingest.roi import (
    check_output_exists,
    load_s2_tile_geometry,
    rasterize_roi_zarr,
)
from tessera_embeddings.storage.manifest import RoiManifest, extract_manifest


def _crs_suffix(crs: str | None) -> str:
    """Return a path-friendly suffix for the given CRS string.

    Empty string when ``crs`` is ``None`` so the caller's filename
    convention round-trips through the auto-CRS path unchanged.
    """
    return f"_{crs.replace(':', '').lower()}" if crs else ""


@flow(name="generate-roi")
def generate_roi(
    *,
    roi_bucket: str,
    roi_name: str | None = None,
    tile_names: str | None = None,
    output_name: str | None = None,
    resolution: float = 10.0,
    chunk_size: int = 2000,
    force_crs: str | None = None,
) -> str:
    """Generate a chunked Zarr ROI mask and write it to the configured bucket.

    Exactly one of ``roi_name`` or ``tile_names`` must be provided.

    Args:
        roi_bucket: Base URI for ROI storage (e.g.
            ``"s3://my-bucket/rois"`` or a local path). Caller-supplied
            so the flow works for any deployment — the reference repo's
            ``dev: bool`` toggle is gone.
        roi_name: Name of the ROI. Reads the GeoJSON from
            ``{roi_bucket}/geojsons/{roi_name}.geojson`` and writes to
            ``{roi_bucket}/zarrs/{roi_name}.zarr``.
        tile_names: Comma-separated Sentinel-2 MGRS tile IDs (e.g.
            ``"14TPK"`` or ``"14TPK,14TQK"``). Output name is derived
            from the tile IDs unless ``output_name`` is set.
        output_name: Override for the derived output filename in tile
            mode. Ignored in GeoJSON mode (use ``roi_name`` directly).
        resolution: Output pixel size in metres.
        chunk_size: Spatial chunk size in pixels (default 2000, matches
            the ingestion pipeline's TESSERA_CHUNKS).
        force_crs: Override CRS as an EPSG string (e.g.
            ``"EPSG:32633"``). Default: auto-select UTM zone from the
            geometry centroid.

    Returns:
        Output Zarr URI.
    """
    log = get_run_logger()

    if roi_name and tile_names:
        raise ValueError("Provide exactly one of roi_name or tile_names, not both")
    if not roi_name and not tile_names:
        raise ValueError("Provide exactly one of roi_name or tile_names")

    suffix = _crs_suffix(force_crs)
    if tile_names:
        derived_name = "_".join(n.strip() for n in tile_names.split(",") if n.strip())
        output_path = f"{roi_bucket}/zarrs/{output_name or derived_name}{suffix}.zarr"
        log.info("Tile mode: fetching footprints for %s", tile_names)
        geometries = load_s2_tile_geometry(tile_names, roi_bucket=roi_bucket)
        input_path = None
    else:
        output_path = f"{roi_bucket}/zarrs/{roi_name}{suffix}.zarr"
        input_path = f"{roi_bucket}/geojsons/{roi_name}.geojson"
        log.info("GeoJSON mode: reading %s", input_path)
        geometries = None

    # Skip if a matching ROI already exists at the output path.
    if check_output_exists(output_path):
        z = zarr.open(output_path, mode="r")
        existing_manifest = extract_manifest(z.attrs)
        if existing_manifest is None:
            log.warning(
                "No _manifest in %s — legacy store, regenerating to add manifest safety checks",
                output_path,
            )
        else:
            current = RoiManifest(resolution=resolution, chunk_size=chunk_size, crs=force_crs)
            current.validate_against(existing_manifest, output_path)
            log.warning("ROI Zarr already exists with matching config, skipping generation: %s", output_path)
            return output_path

    log.info("Writing ROI Zarr to %s (resolution=%sm, chunk_size=%d)", output_path, resolution, chunk_size)
    result = rasterize_roi_zarr(
        output_path=output_path,
        resolution=resolution,
        chunk_size=chunk_size,
        force_crs=force_crs,
        input_path=input_path,
        geometries=geometries,
    )
    log.info("Done: %s", result)
    return result
