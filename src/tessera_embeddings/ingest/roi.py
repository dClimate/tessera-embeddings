"""ROI (Region of Interest) utilities for reading and generating Zarr ROI masks.

Reading
-------
``read_roi_metadata`` and ``read_roi_mask`` load an existing Zarr ROI store
(local or S3) and extract spatial metadata needed for STAC queries and
odc.stac.load: WGS84 bounding box, native CRS, and grid dimensions.

Generation
----------
``rasterize_roi_zarr`` converts GeoJSON polygons (or pre-loaded Shapely
geometries) into a chunked boolean Zarr mask on a UTM grid.  Rasterization
is performed per spatial chunk so the full output is never held in memory.

Helper functions handle CRS detection, UTM zone selection, reprojection,
and grid computation.  ``load_s2_tile_geometry`` fetches Sentinel-2 MGRS
tile footprints from a GeoJSON index on S3 as a shortcut for tile-aligned
ROIs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import dask.array as da
import fsspec
import numpy as np
import zarr
from affine import Affine
from odc.geo.geobox import GeoBox
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform
from shapely.ops import unary_union

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE
from tessera_embeddings.storage.manifest import RoiManifest

logger = logging.getLogger(__name__)

_S2_TILES_FILENAME = "sentinel2_tiles.geojson"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ROIMetadata:
    """Spatial metadata extracted from a Zarr ROI mask.

    Attributes:
        bbox_wgs84: Bounding box in EPSG:4326 (minx, miny, maxx, maxy).
        native_crs: CRS string (e.g., "EPSG:32615").
        geobox: odc GeoBox matching the ROI's exact grid (CRS, transform, shape).
        width: Raster width in pixels.
        height: Raster height in pixels.
    """

    bbox_wgs84: tuple[float, float, float, float]
    native_crs: str
    geobox: GeoBox
    width: int
    height: int


@dataclass
class GridSpec:
    """Computed output grid specification."""

    width: int
    height: int
    transform: Affine
    target_crs: CRS
    resolution: float


# ---------------------------------------------------------------------------
# Reading utilities
# ---------------------------------------------------------------------------


def read_roi_metadata(roi_path: str) -> ROIMetadata:
    """Read spatial metadata from a Zarr ROI store.

    Args:
        roi_path: Path to the Zarr ROI store (local or s3://).

    Returns:
        ROIMetadata with WGS84 bbox, native CRS string, and grid dimensions.
    """
    z = zarr.open(roi_path, mode="r")
    assert isinstance(z, zarr.Array), f"Expected Zarr Array, got {type(z).__name__}"
    native_crs = str(z.attrs["crs"])
    transform = Affine(*cast(list, z.attrs["transform"]))
    height, width = z.shape

    geobox = GeoBox(shape=(height, width), affine=transform, crs=native_crs)

    bbox_wgs84 = cast(tuple[float, float, float, float], tuple(cast(list, z.attrs["bbox_wgs84"])))

    logger.info(f"ROI metadata (zarr): crs={native_crs}, {width}x{height}px, geobox={geobox}, bbox_wgs84={bbox_wgs84}")

    return ROIMetadata(
        bbox_wgs84=bbox_wgs84,
        native_crs=native_crs,
        geobox=geobox,
        width=width,
        height=height,
    )


StorageOptions = dict | Callable[[], "dict | None"] | None
"""fsspec options, or a callable resolving them — see :func:`resolve_storage_options`."""


def resolve_storage_options(storage_options: StorageOptions) -> dict | None:
    """Resolve ``storage_options``, calling it if it is a provider.

    **Accepting a callable is what keeps these credentials fresh, and it belongs here
    rather than at each call site.** A dict is a snapshot: resolved once, it is still
    the value being used however much later the read happens, and an IAM credential
    outlives neither a long leg nor its own TTL. A provider is re-invoked at each read,
    which is where the credential is actually consumed.

    Resolving inside the reader rather than at the call sites means a new call site
    cannot silently reintroduce the frozen behaviour by forgetting to re-resolve.
    """
    return storage_options() if callable(storage_options) else storage_options


def read_roi_mask(
    roi_path: str,
    chunks: dict[str, int],
    storage_options: StorageOptions = None,
) -> da.Array:
    """Read ROI mask from a Zarr store as a chunked dask array.

    Args:
        roi_path: Path to the Zarr ROI store (local or s3:// or any fsspec URI).
        chunks: Dict with ``"northing"`` and ``"easting"`` chunk sizes.
        storage_options: fsspec storage options, or a callable returning them.
            Prefer the callable for S3: it is re-invoked per read, so the credential
            cannot be older than this call. When ``None``, fsspec infers credentials
            from the environment — which on the radar path holds the SOURCE's
            short-lived token rather than our own role, so ``None`` is only safe
            locally.

    Returns:
        Chunked dask boolean array aligned to the target grid (True = inside ROI).
    """
    return da.from_zarr(
        roi_path,
        chunks=(chunks["northing"], chunks["easting"]),
        storage_options=resolve_storage_options(storage_options),
    )


# ---------------------------------------------------------------------------
# Geometry loading
# ---------------------------------------------------------------------------


def load_geometries_from_geojson(input_path: str) -> tuple[list, CRS]:
    """Load geometries and detect CRS from a GeoJSON file (local or S3).

    Handles FeatureCollection, Feature, and raw geometry types.
    If the GeoJSON contains a ``crs`` property (e.g. from gdal_polygonize),
    that CRS is returned; otherwise WGS84 is assumed.

    Args:
        input_path: Path to the GeoJSON file. Any fsspec-compatible URI
            (``s3://``, ``gs://``, ``file://``, absolute local path).

    Returns:
        Tuple of (list of Shapely geometries, detected CRS).
    """
    with fsspec.open(input_path, "r") as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        geometries = [shape(feat["geometry"]) for feat in geojson["features"]]
    elif geojson.get("type") == "Feature":
        geometries = [shape(geojson["geometry"])]
    else:
        geometries = [shape(geojson)]

    input_crs = CRS.from_epsg(4326)
    crs_info = geojson.get("crs", {}).get("properties", {}).get("name")
    if crs_info:
        try:
            input_crs = CRS.from_user_input(crs_info)
            logger.info("Detected input CRS: %s", input_crs)
        except Exception:
            logger.warning("Could not parse GeoJSON CRS, assuming WGS84")

    if not geometries:
        raise ValueError("No geometries found in GeoJSON")

    logger.info("Loaded %d geometry(ies) from %s", len(geometries), input_path)
    return geometries, input_crs


def load_s2_tile_geometry(tile_names: str, roi_bucket: str) -> list:
    """Load Sentinel-2 tile polygon(s) from the S3 tile index.

    Accepts a single MGRS tile identifier or a comma-separated list of
    identifiers.  The GeoJSON index is fetched once regardless of how many
    tiles are requested.

    Args:
        tile_names: One or more MGRS tile identifiers, comma-separated
            (e.g. ``"14TPK"`` or ``"14TPK,14TQK,15TPK"``).
        roi_bucket: Base URI for ROI storage (e.g. ``s3://my-bucket/rois``
            or ``/local/path/rois``). Any fsspec-compatible URI is accepted.

    Returns:
        List of Shapely geometries for the requested tiles.
    """
    names = [n.strip() for n in tile_names.split(",") if n.strip()]
    if not names:
        raise ValueError("No tile names provided")

    tiles_uri = f"{roi_bucket.rstrip('/')}/{_S2_TILES_FILENAME}"

    logger.info("Fetching tile(s) %s from %s", names, tiles_uri)
    with fsspec.open(tiles_uri, "r") as f:
        geojson = json.load(f)

    name_set = set(names)
    geometries: list = []
    found: set[str] = set()
    for feat in geojson["features"]:
        feat_name = feat["properties"]["Name"]
        if feat_name in name_set:
            geometries.append(shape(feat["geometry"]))
            found.add(feat_name)

    missing = name_set - found
    if missing:
        raise ValueError(f"Tile(s) {sorted(missing)} not found in {tiles_uri}")

    logger.info("Found %d geometry(ies) for tile(s) %s", len(geometries), sorted(found))
    return geometries


# ---------------------------------------------------------------------------
# CRS and grid helpers
# ---------------------------------------------------------------------------


def determine_utm_zone(lon: float, lat: float) -> int:
    """Determine the best UTM EPSG code from a lon/lat coordinate."""
    zone_number = int((lon + 180) / 6) + 1
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32
    if 72 <= lat < 84:
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37
    return 32600 + zone_number if lat >= 0 else 32700 + zone_number


def determine_target_crs(
    geometries: list,
    input_crs: CRS,
    force_crs: str | None = None,
) -> CRS:
    """Determine the target CRS for reprojection.

    Priority:
        1. ``force_crs`` if provided.
        2. ``input_crs`` if it is not WGS84 (i.e. already projected).
        3. Auto-select UTM zone from geometry centroid.

    Args:
        geometries: List of Shapely geometries (in input_crs).
        input_crs: CRS of the input geometries.
        force_crs: Optional EPSG string to override auto-detection.

    Returns:
        Target CRS.
    """
    if force_crs:
        target_crs = CRS.from_user_input(force_crs)
        logger.info("Using specified CRS: %s", target_crs)
    elif input_crs != CRS.from_epsg(4326):
        target_crs = input_crs
        logger.info("Using input CRS as target: %s", target_crs)
    else:
        centroid = unary_union(geometries).centroid
        epsg = determine_utm_zone(centroid.x, centroid.y)
        target_crs = CRS.from_epsg(epsg)
        logger.info("Auto-selected UTM CRS: EPSG:%d (centroid: %.4f, %.4f)", epsg, centroid.x, centroid.y)

    if target_crs.is_geographic:
        raise ValueError(
            f"Target CRS {target_crs} is geographic (units are degrees, not meters). "
            "Provide a projected CRS via force_crs (e.g. 'EPSG:32633')."
        )
    return target_crs


def reproject_geometries(geometries: list, input_crs: CRS, target_crs: CRS) -> list:
    """Reproject geometries from input_crs to target_crs.

    If the CRSes match, returns the geometries unchanged.

    Args:
        geometries: List of Shapely geometries.
        input_crs: Source CRS.
        target_crs: Destination CRS.

    Returns:
        List of reprojected Shapely geometries.
    """
    if input_crs == target_crs:
        logger.info("Input CRS matches target - skipping reprojection")
        return geometries

    transformer = Transformer.from_crs(input_crs, target_crs, always_xy=True).transform
    return [shp_transform(transformer, geom) for geom in geometries]


def compute_grid(geometries: list, resolution: float, target_crs: CRS) -> GridSpec:
    """Compute output grid dimensions and transform from geometries.

    Args:
        geometries: List of Shapely geometries (already in target_crs).
        resolution: Pixel size in CRS units (meters for UTM).
        target_crs: CRS for the output grid.

    Returns:
        GridSpec with width, height, transform, target_crs, and resolution.

    Raises:
        ValueError: If computed grid dimensions are non-positive.
    """
    union_geom = unary_union(geometries)
    minx, miny, maxx, maxy = union_geom.bounds
    res = float(resolution)
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid grid dimensions: {width}x{height}. Check bounds and resolution.")

    transform = from_origin(minx, maxy, res, res)
    logger.info("Output grid: %dx%d pixels at %sm, CRS=%s", width, height, res, target_crs)
    logger.info("Bounds (projected): %.2f, %.2f -> %.2f, %.2f", minx, miny, maxx, maxy)

    return GridSpec(width=width, height=height, transform=transform, target_crs=target_crs, resolution=res)


# ---------------------------------------------------------------------------
# ROI Zarr generation
# ---------------------------------------------------------------------------


def check_output_exists(output_path: str) -> bool:
    """Check whether a Zarr store already exists at the given path.

    Args:
        output_path: Any fsspec-compatible URI.

    Returns:
        True if the path exists.
    """
    fs = fsspec.filesystem(fsspec.utils.get_protocol(output_path))
    return fs.exists(output_path)


def rasterize_roi_zarr(
    output_path: str,
    resolution: float,
    chunk_size: int = INGEST_CHUNK_SIZE,
    force_crs: str | None = None,
    input_path: str | None = None,
    geometries: list | None = None,
) -> str:
    """Rasterize GeoJSON polygons into a chunked boolean Zarr ROI mask.

    Rasterizes per spatial chunk so the full output is never held in memory.
    Each chunk calls ``rasterio.features.rasterize`` with a chunk-local
    transform - only geometries that intersect the chunk contribute pixels.

    Zarr attrs stored:
        crs: CRS string (e.g. ``"EPSG:32615"``)
        transform: 6 Affine coefficients ``[a, b, c, d, e, f]``
        resolution: pixel size in meters
        bbox_wgs84: WGS84 bounding box ``[minx, miny, maxx, maxy]``,
            computed from the original geometry before reprojection

    Args:
        output_path: Path to output Zarr store (local directory or ``s3://`` URI).
        resolution: Output pixel size in meters.
        chunk_size: Spatial chunk size in pixels (default
            ``INGEST_CHUNK_SIZE``, matching the ingestion pipeline's
            ``load_chunks`` so the ROI mask read during ingest is a clean
            merge rather than a cross-chunk shuffle).
        force_crs: Optional EPSG string (e.g. ``"EPSG:32633"``) to override
            automatic UTM zone selection.
        input_path: Path to input GeoJSON file (local or ``s3://``).
        geometries: Pre-loaded list of Shapely geometries (alternative to input_path).

    Returns:
        Path to the output Zarr store.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    # Default to WGS84; overridden if the GeoJSON carries its own CRS
    input_crs = CRS.from_epsg(4326)

    if geometries is None:
        if input_path is None:
            raise ValueError("Either input_path or geometries must be provided")
        geometries, input_crs = load_geometries_from_geojson(input_path)

    if not geometries:
        raise ValueError("No geometries found")

    # Compute the WGS84 bounding box from the original geometry *before*
    # reprojection.  This ensures the STAC query extent is projection-
    # invariant — reprojecting to different target CRSes and back-projecting
    # the axis-aligned grid bounds can inflate the bbox significantly.
    wgs84_crs = CRS.from_epsg(4326)
    wgs84_geoms = reproject_geometries(geometries, input_crs, wgs84_crs) if input_crs != wgs84_crs else geometries
    bbox_wgs84 = unary_union(wgs84_geoms).bounds  # (minx, miny, maxx, maxy)

    # Project geometries to UTM (or user-specified CRS) and compute the
    # output pixel grid (width, height, affine transform).
    target_crs = determine_target_crs(geometries, input_crs, force_crs)
    reprojected = reproject_geometries(geometries, input_crs, target_crs)
    grid = compute_grid(reprojected, resolution, target_crs)

    # Pre-build (geometry, burn_value) pairs once — reused in every chunk's
    # rasterize call. Only geometries intersecting a chunk's extent actually
    # contribute pixels; rasterio handles the spatial filtering internally.
    geom_pairs = [(mapping(g), 1) for g in reprojected]

    # Create the output Zarr array with spatial chunking matching the
    # ingestion pipeline so downstream da.from_zarr reads are zero-copy.
    z = zarr.open(
        output_path,
        mode="w",
        shape=(grid.height, grid.width),
        chunks=(chunk_size, chunk_size),
        dtype="bool",
    )
    assert isinstance(z, zarr.Array)
    # Store spatial metadata so read_roi_metadata() can reconstruct the
    # GeoBox and WGS84 bbox without the original GeoJSON.
    z.attrs["crs"] = str(grid.target_crs)
    z.attrs["transform"] = list(grid.transform)[:6]
    z.attrs["resolution"] = grid.resolution
    z.attrs["bbox_wgs84"] = list(bbox_wgs84)

    # Rasterize one spatial chunk at a time to bound memory usage.
    # Each iteration shifts the base affine to the chunk's top-left corner.
    valid_pixels = 0
    total_chunks = 0
    for y0 in range(0, grid.height, chunk_size):
        for x0 in range(0, grid.width, chunk_size):
            ch = min(chunk_size, grid.height - y0)
            cw = min(chunk_size, grid.width - x0)
            chunk_transform = grid.transform * Affine.translation(x0, y0)

            chunk = rasterize(
                geom_pairs,
                out_shape=(ch, cw),
                transform=chunk_transform,
                fill=0,
                dtype="uint8",
            )
            z[y0 : y0 + ch, x0 : x0 + cw] = chunk > 0
            valid_pixels += int(chunk.sum())
            total_chunks += 1

    # Write manifest only after rasterization completes successfully,
    # so interrupted runs don't leave a partial store that looks valid.
    manifest = RoiManifest(resolution=grid.resolution, chunk_size=chunk_size, crs=str(grid.target_crs))
    z.attrs["_manifest"] = manifest.to_dict()
    logger.info("Wrote _manifest to %s", output_path)

    total_pixels = grid.height * grid.width
    logger.info(
        "ROI coverage: %d/%d pixels (%.1f%%)",
        valid_pixels,
        total_pixels,
        100 * valid_pixels / total_pixels,
    )
    logger.info(
        "Wrote ROI Zarr: %s (%d chunks of %dx%d)",
        output_path,
        total_chunks,
        chunk_size,
        chunk_size,
    )
    return output_path
