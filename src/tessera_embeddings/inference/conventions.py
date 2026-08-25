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
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

import numpy as np
from pyproj import CRS

from tessera_embeddings.config.inference import DEFAULT_MODEL_VERSION, ModelVersion, encoder_url

logger = logging.getLogger(__name__)


def _software_version() -> str:
    """Version of this package that built the store (``geoemb:build_version``)."""
    try:
        return _dist_version("tessera_embeddings")
    except PackageNotFoundError:
        return "unknown"


def _is_metre_crs(epsg_code: str | None) -> bool:
    """Whether an EPSG code's horizontal axes are metres.

    ``geoemb:gsd`` is defined in metres, so the coordinate spacing may only be
    used as the GSD for a metre-based (projected) CRS. A geographic CRS
    (e.g. EPSG:4326, degrees) or a US-survey-foot CRS must not have its spacing
    mislabelled as metres.
    """
    if not epsg_code:
        return False
    try:
        crs = CRS.from_user_input(epsg_code)
    except Exception:
        return False
    units = [getattr(a, "unit_name", "").lower() for a in crs.axis_info]
    return bool(units) and all(u in ("metre", "meter") for u in units)


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

# The convention repo has not cut a `v1` tag, so `refs/tags/v1` 404s. We pin the
# schema/spec URLs to an immutable commit SHA that DOES resolve, so the emitted
# registration is dereferenceable. Switch to `refs/tags/v1` once upstream tags it.
_GEOEMB_REF = "0655212938f36351245dbd3e5e8868f811d43663"
_GEOEMB_CONVENTION = {
    "schema_url": f"https://raw.githubusercontent.com/geo-embeddings/embeddings-zarr-convention/{_GEOEMB_REF}/schema.json",
    "spec_url": f"https://github.com/geo-embeddings/embeddings-zarr-convention/blob/{_GEOEMB_REF}/README.md",
    "uuid": "61c12cc5-0e28-4056-999a-480cf3fb7e4c",
    "name": "geoemb:",
    "description": "Geoembeddings convention for geospatial embedding arrays with model provenance",
}

# --- geoemb: field defaults (this pipeline's fixed provenance) --------------
#: v1.1's public encoder version. The live ``geoemb:model`` lookup is
#: :func:`~tessera_embeddings.config.inference.encoder_url`, keyed by model
#: family; this remains the v1.1 identifier. (``geoemb:build_version`` is the
#: software/package version, not this.)
ENCODER_VERSION = "1.1"
#: Precise source datasets we pull from: Sentinel-2 L2A COGs (Earth Search AWS
#: Open Data) and OPERA RTC-S1 (ASF datapool).
DEFAULT_SOURCE_DATA: tuple[str, ...] = (
    "s3://sentinel-cogs",
    "https://datapool.asf.alaska.edu/RTC/OPERA-S1",
)
#: Storage dtype of the quantized embeddings.
QUANTIZED_DTYPE = "int8"


def expected_model_url(model_url: str | None = None, model_version: ModelVersion | None = None) -> str:
    """The ``geoemb:model`` URL this build stamps for the current encoder version.

    A seeded store records this once at its root; the fill re-derives it to verify
    the running code embeds with the SAME encoder the store was seeded for — a
    mismatch means a model upgrade slipped in between seeding and filling. Passing
    *model_url* mirrors the seed-time override so a store seeded with a custom URL
    round-trips.

    Routed through :func:`~tessera_embeddings.config.inference.encoder_url` rather than a
    module constant, and taking *model_version*, on the instruction that function's own
    docstring left for this merge: with one hard-coded encoder version the gate compares a
    v2 fill against v1.1's URL — rejecting a correct run, or waving through a real mismatch
    when the seed happened to carry the same stale constant. Omitting *model_version* keeps
    the historical single-encoder behaviour.
    """
    return model_url or encoder_url(model_version or DEFAULT_MODEL_VERSION)


def assert_encoder_matches(published: str | None, *, model_version: ModelVersion | None, where: str) -> None:
    """Refuse a write whose encoder disagrees with the store's published one.

    **One implementation because there are several readers.** The assembly append gate and the
    pre-flight that runs before any compute is provisioned both have to answer the same question,
    and two copies of "is this the same encoder" is how they end up disagreeing about one store —
    worse than either answer alone, because a run is then refused or admitted depending on which
    door it arrived through.

    ``published is None`` is read as v1.1, NOT waved through: ``geoemb:model`` postdates v1.1 and
    predates v2, so a store lacking it necessarily holds v1.1 embeddings. Treating absence as "no
    claim" is what let a v2 run append to a legacy store and restamp the whole thing.
    """
    expected = expected_model_url(model_version=model_version)
    actual = published or expected_model_url(model_version=DEFAULT_MODEL_VERSION)
    if actual != expected:
        raise ValueError(
            f"Refusing to append to {where}: it was published by encoder {actual!r} and this run "
            f"uses {expected!r}. Appending would mix two model families in one store and restamp "
            "the whole thing with this run's encoder. Assemble into a store of the matching "
            "family, or a fresh path. (A store carrying no geoemb:model at all is read as v1.1 — "
            "the attr predates v2, so its absence dates the store rather than leaving it open.)"
        )


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


def _geoemb_fields(
    *,
    embedding_dim: int,
    model_version: str | None,
    model_url: str | None,
    data_type: str,
    gsd: float | None,
    spatial_layout: str | None,
    source_data: tuple[str, ...],
    encoder_version: ModelVersion | None = None,
) -> dict:
    """The ``geoemb:*`` (+ ``checkpoint_id``) attrs, shared by the single-ROI root
    builder and the multi-group root builder.

    *gsd* is already resolved to a trustworthy metre value or ``None`` — the caller
    decides how (derived from a metre-CRS grid, or explicit). See
    :func:`build_convention_attrs` for the meaning of each ``geoemb:`` field.
    """
    attrs: dict = {
        "geoemb:type": "pixel",  # per-pixel embeddings (not chip)
        "geoemb:dimensions": embedding_dim,
        # The PUBLIC encoder reference, resolved from the model FAMILY and never from
        # *model_version* — which in production is an internal checkpoint filename stem
        # (`checkpoint_to_version(...)`). That stem is kept separately as `checkpoint_id`
        # below, so a v1.0 / future / custom checkpoint cannot advertise a synthetic public
        # URL. Keyed by encoder_version so v2 Large stamps its own URL rather than v1.1's.
        "geoemb:model": model_url or encoder_url(encoder_version or DEFAULT_MODEL_VERSION),
        "geoemb:source_data": list(source_data),
        "geoemb:data_type": data_type,
        # build_version is the SOFTWARE build (this package) per the convention.
        "geoemb:build_version": _software_version(),
        "geoemb:quantization": {
            "method": "per_pixel_scale",  # absmax-per-pixel: value = quantized * scale
            "original_dtype": "float32",
            "quantized_dtype": data_type,
            # Per-pixel float32 dequantization factors live in the `scales` array;
            # absent/ocean pixels carry NaN there (assembly fills float vars NaN).
            "scale": {"type": "array", "array_name": "scales", "nodata": "NaN"},
        },
    }
    if model_version:
        attrs["checkpoint_id"] = model_version
    if gsd is not None:
        attrs["geoemb:gsd"] = gsd
    if spatial_layout is not None:
        attrs["geoemb:spatial_layout"] = spatial_layout
    return attrs


def build_geoemb_root_attrs(
    *,
    embedding_dim: int,
    spatial_layout: str,
    gsd: float | None = None,
    model_version: str | None = None,
    encoder_version: ModelVersion | None = None,
    model_url: str | None = None,
    data_type: str = QUANTIZED_DTYPE,
    source_data: tuple[str, ...] = DEFAULT_SOURCE_DATA,
) -> dict:
    """``geoemb:`` attrs for the ROOT of a multi-group store (``utm_zones`` / ``global``).

    In the geoembeddings ``utm_zones`` layout the encoder/quantization provenance is
    stated ONCE at the root (it is identical across zones), while ``proj:``/``spatial:``
    live on each per-zone group (their CRS and grid differ) — build those with
    :func:`build_convention_attrs` and ``include_geoemb=False``. *gsd* is explicit
    here because the root has no single coordinate grid to derive it from.
    """
    attrs = _geoemb_fields(
        embedding_dim=embedding_dim,
        model_version=model_version,
        encoder_version=encoder_version,
        model_url=model_url,
        data_type=data_type,
        gsd=gsd,
        spatial_layout=spatial_layout,
        source_data=source_data,
    )
    attrs["zarr_conventions"] = [_GEOEMB_CONVENTION]
    return attrs


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
    encoder_version: ModelVersion | None = None,
    model_url: str | None = None,
    data_type: str = QUANTIZED_DTYPE,
    gsd: float | None = None,
    spatial_layout: str | None = None,
    source_data: tuple[str, ...] = DEFAULT_SOURCE_DATA,
    include_geoemb: bool = True,
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
    ``geoemb:model`` is the PUBLIC encoder reference URL — *model_url* when a
    caller supplies the exact public URI for the encoder it used, else looked up
    from *encoder_version* (the model FAMILY, e.g. ``"v2-large"``). It is NEVER
    built from *model_version*, which in production is an internal checkpoint
    filename stem; that is recorded as a plain ``checkpoint_id`` provenance attr.
    Passing the family matters: omitting it stamps the default model's URL, so a
    v2 store would advertise itself as v1.1. ``geoemb:build_version`` is the
    software/package version. *data_type* is the quantized storage dtype.
    *gsd* (metres) is emitted only when trustworthy — derived from a metre-based
    CRS's coordinate spacing, or an explicit *gsd* the caller vouches for; for a
    non-metre CRS (e.g. degrees/feet) or absent coords it is OMITTED (optional
    field, no false metre value). *spatial_layout* is ``"utm_zones"``/``"global"``
    and OMITTED when ``None`` (a single-ROI store has no utmNN/global groups);
    *source_data* the source-dataset URLs.

    Set *include_geoemb* to False to emit only ``proj:``/``spatial:`` (no ``geoemb:``):
    the multi-group campaign store carries geoemb: once at the root
    (:func:`build_geoemb_root_attrs`) and uses this builder only for each zone's
    CRS/grid.
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
    # Omitted when *include_geoemb* is False: the multi-group campaign store states
    # geoemb: ONCE at the root (build_geoemb_root_attrs) and uses this function only
    # for per-zone proj:/spatial:, whose CRS/grid differ by zone.
    if include_geoemb:
        conventions.append(_GEOEMB_CONVENTION)
        # geoemb:gsd is OPTIONAL and defined in metres, so emit it ONLY with a
        # trustworthy metric value: derived from the coordinate spacing of a
        # metre-based CRS, or an explicit gsd the caller vouches for. A non-metre CRS
        # (EPSG:4326 degrees, foot-based) or missing coords → OMIT it (no false value).
        if x_coords is not None and len(x_coords) > 1 and _is_metre_crs(effective_epsg):
            gsd_val: float | None = abs(float(np.median(np.diff(x_coords))))
        else:
            gsd_val = gsd
        attrs.update(
            _geoemb_fields(
                embedding_dim=embedding_dim,
                model_version=model_version,
                encoder_version=encoder_version,
                model_url=model_url,
                data_type=data_type,
                gsd=gsd_val,
                spatial_layout=spatial_layout,
                source_data=source_data,
            )
        )

    if conventions:
        attrs["zarr_conventions"] = conventions

    return attrs
