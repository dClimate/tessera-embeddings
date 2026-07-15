"""GeoZarr convention attribute builders for embedding stores.

Implements the proj:, spatial:, and geoemb: Zarr conventions as purely
additive metadata on the root group. No store layout changes.

The geoembeddings (``geoemb:``) convention supersedes the earlier ``tessera:``
convention: it records encoder-model provenance, source datasets, and a
structured quantization/dequantization description (see the ``geoemb:`` fields
below). The mapping follows the convention repo's own ``tessera_example.json``.

References:
    - proj:    https://github.com/zarr-conventions/geo-proj
    - spatial: https://github.com/zarr-conventions/spatial
    - geoemb:  https://github.com/geo-embeddings/embeddings-zarr-convention
"""

from __future__ import annotations

import logging

import numpy as np
from pyproj import CRS

logger = logging.getLogger(__name__)

# Convention registration metadata (UUID + schema URLs)
_PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-experimental/geo-proj/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-experimental/geo-proj/blob/v1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj:",
    "description": "Coordinate reference system information for geospatial data",
}

_SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial:",
    "description": "Spatial coordinate information",
}

_GEOEMB_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/geo-embeddings/embeddings-zarr-convention/refs/tags/v1/schema.json",
    "spec_url": "https://github.com/geo-embeddings/embeddings-zarr-convention/blob/v1/README.md",
    "uuid": "61c12cc5-0e28-4056-999a-480cf3fb7e4c",
    "name": "geoemb:",
    "description": "Geoembeddings convention for geospatial embedding arrays with model provenance",
}

# --- geoemb: field defaults (this pipeline's fixed provenance) --------------
#: Encoder checkpoint version (v1.1 pipeline). Versions both the model URL and
#: ``geoemb:build_version``; overridable per call via ``model_version``.
ENCODER_VERSION = "1.1"
#: Model reference URL, versioned by the encoder checkpoint.
_MODEL_URL_TEMPLATE = "https://geotessera.org/model/{version}"
#: Precise source datasets we pull from: Sentinel-2 L2A COGs (Earth Search AWS
#: Open Data) and OPERA RTC-S1 (ASF datapool).
DEFAULT_SOURCE_DATA: tuple[str, ...] = (
    "s3://sentinel-cogs",
    "https://datapool.asf.alaska.edu/RTC/OPERA-S1",
)
#: Ground sample distance in metres (10 m embeddings grid).
GSD_METERS = 10.0
#: Storage dtype of the quantized embeddings.
QUANTIZED_DTYPE = "int8"


def tile_id_to_epsg(tile_id: str) -> str | None:
    """Derive EPSG code from a Sentinel-2 MGRS tile ID.

    The first two characters encode the UTM zone number (01-60). The third
    character is the latitude band letter: bands N-X are northern hemisphere
    (EPSG:326xx), bands C-M are southern hemisphere (EPSG:327xx).

    Args:
        tile_id: MGRS tile ID, e.g. ``"37PBM"``, ``"33UWP"``, ``"56HKH"``.

    Returns:
        EPSG authority code string (e.g. ``"EPSG:32637"``), or ``None`` if
        *tile_id* doesn't look like a valid MGRS tile ID.
    """
    if not tile_id or len(tile_id) < 3:
        return None
    try:
        zone = int(tile_id[:2])
    except ValueError:
        return None
    band = tile_id[2].upper()
    if not band.isalpha() or band in ("A", "B", "Y", "Z", "I", "O"):
        return None
    if not 1 <= zone <= 60:
        return None
    hemisphere_base = 32600 if band >= "N" else 32700
    return f"EPSG:{hemisphere_base + zone}"


def _crs_fields_from_epsg(epsg_code: str) -> dict[str, str | dict]:
    """Derive all proj: fields from an EPSG code string (e.g. ``"EPSG:32637"``).

    Uses pyproj to produce ``proj:code``, ``proj:wkt2``, and ``proj:projjson``.

    Returns:
        Dict of ``proj:*`` keys to their values. Empty if derivation fails.
    """
    result: dict[str, str | dict] = {}
    try:
        crs = CRS.from_user_input(epsg_code)

        result["proj:code"] = epsg_code

        wkt2 = crs.to_wkt("WKT2_2019")
        if wkt2:
            result["proj:wkt2"] = wkt2

        projjson = crs.to_json_dict()
        if projjson:
            result["proj:projjson"] = projjson
    except Exception:
        logger.debug("Failed to derive CRS fields from %s", epsg_code, exc_info=True)
    return result


def _compute_affine_transform(y_coords: np.ndarray, x_coords: np.ndarray) -> list[float]:
    """Compute a 6-element affine transform from coordinate arrays.

    For axis-aligned grids the transform is::

        [scale_x, 0, origin_x, 0, scale_y, origin_y]

    The pixel size (scale) is derived from the **median** spacing between
    consecutive coordinate values rather than just ``coords[1] - coords[0]``.
    This is more robust for non-UTM projections where floating-point
    representation of geographic coordinates can introduce small rounding
    differences along the axis.

    A warning is logged if the coordinate spacing is not uniform (max
    deviation > 1 % of the median), which would indicate a non-regular grid
    where an affine transform is only an approximation.
    """
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)

    res_x = float(np.median(dx))
    res_y = float(np.median(dy))

    # Warn if spacing is significantly non-uniform
    if dx.size > 1:
        dx_dev = float(np.max(np.abs(dx - res_x)))
        if dx_dev > abs(res_x) * 0.01:
            logger.warning(
                "X coordinate spacing is non-uniform (max deviation %.4f vs median %.4f). "
                "Affine transform may be approximate.",
                dx_dev,
                res_x,
            )
    if dy.size > 1:
        dy_dev = float(np.max(np.abs(dy - res_y)))
        if dy_dev > abs(res_y) * 0.01:
            logger.warning(
                "Y coordinate spacing is non-uniform (max deviation %.4f vs median %.4f). "
                "Affine transform may be approximate.",
                dy_dev,
                res_y,
            )

    origin_x = float(x_coords[0])
    origin_y = float(y_coords[0])
    return [res_x, 0.0, origin_x, 0.0, res_y, origin_y]


def _compute_bbox(y_coords: np.ndarray, x_coords: np.ndarray) -> list[float]:
    """Compute a bounding box [xmin, ymin, xmax, ymax] for pixel-registered data.

    For pixel-registered grids the bbox extends by half a pixel beyond the
    outermost coordinate centres on all sides, matching the ``spatial:``
    convention definition of pixel registration.
    """
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    res_x = float(np.median(dx))
    res_y = float(np.median(dy))

    # Pixel registration: bbox is the outer edge of the outermost pixels.
    # The coordinates mark pixel centres, so extend by half a pixel.
    half_x = abs(res_x) / 2
    half_y = abs(res_y) / 2

    x_min = float(min(x_coords[0], x_coords[-1])) - half_x
    x_max = float(max(x_coords[0], x_coords[-1])) + half_x
    y_min = float(min(y_coords[0], y_coords[-1])) - half_y
    y_max = float(max(y_coords[0], y_coords[-1])) + half_y

    return [x_min, y_min, x_max, y_max]


def build_convention_attrs(
    *,
    tile_id: str | None = None,
    epsg_code: str | None = None,
    total_y: int,
    total_x: int,
    embedding_dim: int,
    y_coords: np.ndarray | None = None,
    x_coords: np.ndarray | None = None,
    model_version: str | None = None,
    data_type: str = QUANTIZED_DTYPE,
    gsd: float = GSD_METERS,
    spatial_layout: str | None = None,
    source_data: tuple[str, ...] = DEFAULT_SOURCE_DATA,
) -> dict:
    """Build GeoZarr convention attributes for the root group.

    Returns a flat dict of attributes to set on the Zarr root group.
    Includes ``zarr_conventions`` registration, ``proj:*``, ``spatial:*``,
    and ``geoemb:*`` metadata (the geoembeddings convention).

    *epsg_code* takes precedence over *tile_id* for determining the CRS.
    This allows callers who reproject data to record the actual output CRS
    instead of the original UTM zone derived from the MGRS tile ID.

    If neither *epsg_code* nor *tile_id* resolve to a valid EPSG code,
    ``proj:`` conventions are omitted (no CRS info available).

    The ``geoemb:`` fields record encoder-model provenance and quantization:
    *model_version* (the encoder checkpoint version, default
    :data:`ENCODER_VERSION`) versions both ``geoemb:model`` and
    ``geoemb:build_version``; *data_type* is the quantized storage dtype;
    *gsd* the ground sample distance in metres (derived from the coordinate
    spacing when coords are given, else this nominal value); *spatial_layout*
    is ``"utm_zones"``/``"global"`` and is OMITTED when ``None`` (a single-ROI
    store has no utmNN/global groups); *source_data* the source-dataset URLs.
    """
    conventions: list[dict] = []
    attrs: dict = {}

    # --- proj: convention ---
    # An explicit *epsg_code* takes precedence over tile_id derivation so that
    # callers who reproject data can record the actual output CRS.
    proj_fields: dict[str, str | dict] = {}
    effective_epsg = epsg_code
    if not effective_epsg and tile_id:
        effective_epsg = tile_id_to_epsg(tile_id)
    if effective_epsg:
        proj_fields = _crs_fields_from_epsg(effective_epsg)

    if proj_fields:
        conventions.append(_PROJ_CONVENTION)
        attrs.update(proj_fields)

    # --- spatial: convention ---
    if y_coords is not None and x_coords is not None and len(y_coords) > 1 and len(x_coords) > 1:
        conventions.append(_SPATIAL_CONVENTION)
        attrs["spatial:dimensions"] = ["northing", "easting"]
        attrs["spatial:transform_type"] = "affine"
        attrs["spatial:transform"] = _compute_affine_transform(y_coords, x_coords)
        attrs["spatial:shape"] = [total_y, total_x]
        attrs["spatial:bbox"] = _compute_bbox(y_coords, x_coords)
        attrs["spatial:registration"] = "pixel"

    # --- geoemb: convention ---
    version = model_version or ENCODER_VERSION
    conventions.append(_GEOEMB_CONVENTION)
    attrs["geoemb:type"] = "pixel"  # per-pixel embeddings (not chip)
    attrs["geoemb:dimensions"] = embedding_dim
    attrs["geoemb:model"] = _MODEL_URL_TEMPLATE.format(version=version)
    attrs["geoemb:source_data"] = list(source_data)
    attrs["geoemb:data_type"] = data_type
    # Prefer the actual pixel size from the coordinate spacing (an ROI may be
    # coarsened, e.g. 20 m) over the nominal default.
    if x_coords is not None and len(x_coords) > 1:
        attrs["geoemb:gsd"] = abs(float(np.median(np.diff(x_coords))))
    else:
        attrs["geoemb:gsd"] = gsd
    # spatial_layout is OPTIONAL and only meaningful for a store organised into
    # utmNN / global groups. A single-ROI store writes arrays at its own root,
    # so omit it unless a caller (e.g. the 120-group campaign) sets it.
    if spatial_layout is not None:
        attrs["geoemb:spatial_layout"] = spatial_layout
    attrs["geoemb:build_version"] = version
    attrs["geoemb:quantization"] = {
        "method": "per_pixel_scale",  # absmax-per-pixel: value = quantized * scale
        "original_dtype": "float32",
        "quantized_dtype": data_type,
        # Per-pixel float32 dequantization factors live in the `scales` array;
        # absent/ocean pixels carry NaN there (assembly fills float vars NaN).
        "scale": {"type": "array", "array_name": "scales", "nodata": "NaN"},
    }

    if conventions:
        attrs["zarr_conventions"] = conventions

    return attrs
