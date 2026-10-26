"""GeoZarr convention attribute builders for embedding stores.

Implements the proj:, spatial:, and geoemb: Zarr conventions as purely additive
metadata on the root group. No store layout changes.

The geoembeddings (``geoemb:``) convention supersedes the earlier ``tessera:`` one
(still stripped from pre-existing stores by ``assembly``): it records encoder-model
provenance, source datasets, and a structured quantization/dequantization description,
following the convention repo's own ``tessera_example.json``.

References:
    - proj:    https://github.com/zarr-conventions/geo-proj
    - spatial: https://github.com/zarr-conventions/spatial
    - geoemb:  https://github.com/geo-embeddings/embeddings-zarr-convention
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from typing import cast

import numpy as np
from pyproj import CRS

from tessera_embeddings.config.inference import ModelVersion, encoder_url, model_family, model_version_for_encoder

logger = logging.getLogger(__name__)


def _software_version() -> str:
    """Version of this package that built the store (``geoemb:build_version``)."""
    try:
        return _dist_version("tessera_embeddings")
    except PackageNotFoundError:
        return "unknown"


def _is_metre_crs(epsg_code: str | None) -> bool:
    """Whether an EPSG code's horizontal axes are metres.

    ``geoemb:gsd`` is defined in metres, so coordinate spacing may only be used as the GSD
    for a metre-based (projected) CRS: a geographic CRS (EPSG:4326, degrees) or a
    US-survey-foot one must not have its spacing mislabelled as metres.
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
#
# **Each entry is the registration object its own schema REQUIRES, field for field.** The `v0.1`
# schemas pin every field with `const` and set `additionalProperties: false`, so a registration that
# differs anywhere does not validate against the schema it advertises. Pinning `refs/tags/v1` — a
# tag neither convention has cut — additionally 404'd all four URLs, and a registration nobody can
# follow is worse than none because it reads as a version claim.
#
# **The three conventions genuinely disagree about `name`, and that is not a typo here.** The proj
# and spatial schemas require the BARE name (`"proj"`, `"spatial"`); the geoembeddings schema
# requires the key prefix WITH its colon (`"geoemb:"`). Each was read from the schema it pins.
# Anything matching these by name must therefore not assume a uniform shape — match on `uuid`,
# which is the field all three describe as permanently identifying the convention.
#
# `proj:` also moved organisation twice: `zarr-experimental/geo-proj` and `zarr-conventions/geo-proj`
# both still redirect, but the repository its own schema names is `zarr-conventions/proj`.
# Re-pin to a later tag only in a change that re-reads these schemas, since the URLs are `const`.
_PROJ_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/proj/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/proj/blob/v0.1/README.md",
    "uuid": "f17cb550-5864-4468-aeb7-f3180cfb622f",
    "name": "proj",
    "description": "Coordinate reference system information for geospatial data",
}

_SPATIAL_CONVENTION = {
    "schema_url": "https://raw.githubusercontent.com/zarr-conventions/spatial/refs/tags/v0.1/schema.json",
    "spec_url": "https://github.com/zarr-conventions/spatial/blob/v0.1/README.md",
    "uuid": "689b58e2-cf7b-45e0-9fff-9cfc0883d6b4",
    "name": "spatial",
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
#: Precise source datasets we pull from: Sentinel-2 L2A COGs (Earth Search AWS
#: Open Data) and OPERA RTC-S1 (ASF datapool).
DEFAULT_SOURCE_DATA: tuple[str, ...] = (
    "s3://sentinel-cogs",
    "https://datapool.asf.alaska.edu/RTC/OPERA-S1",
)
#: Storage dtype of the quantized embeddings.
QUANTIZED_DTYPE = "int8"


def expected_model_url(model_url: str | None = None, *, model_version: ModelVersion | None) -> str:
    """The ``geoemb:model`` URL a store of *model_version* carries: *model_url* if given, else its record's.

    A seeded store records this once at its root; the fill re-derives it to check the running
    code embeds with the SAME encoder the store was seeded for. Passing *model_url* mirrors the
    seed-time override so a store seeded with a custom URL round-trips. With neither, the call is
    refused: a default here once compared a v2 fill against v1.1's URL.
    """
    if model_version is not None:
        model_family(model_version)  # a checkpoint name passed under the old meaning fails here, loudly
    if model_url:
        return model_url
    if model_version is None:
        raise ValueError("expected_model_url needs a model_version or an explicit model_url")
    return encoder_url(model_version)


def assert_encoder_matches(published: str | None, *, model_version: ModelVersion, where: str) -> None:
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
    expected = encoder_url(model_version)
    actual = published or encoder_url(model_version_for_encoder(None))
    if actual != expected:
        raise ValueError(
            f"Refusing to append to {where}: it was published by encoder {actual!r} and this run "
            f"uses {expected!r}. Appending would mix two model families in one store and restamp "
            "the whole thing with this run's encoder. Assemble into a store of the matching "
            "family, or a fresh path. (A store carrying no geoemb:model at all is read as v1.1 — "
            "the attr predates v2, so its absence dates the store rather than leaving it open.)"
        )


#: Root and staged-tile attr naming the library stack that computed the embeddings:
#: ``{"torch": torch.__version__, "cuda": torch.version.cuda}``, CUDA ``None`` on a CPU build.
STACK_ATTR = "inference_stack"


def _describe_stack(stack: Mapping[str, object]) -> str:
    cuda = stack.get("cuda")
    return f"PyTorch {stack.get('torch')} " + (f"with CUDA {cuda}" if cuda else "on CPU")


def staged_stack(
    attrs: Mapping[str, object], where: str, *, expected: Mapping[str, str | None] | None = None
) -> dict[str, str | None]:
    """The library stack a staged tile was computed on, refusing a tile that differs from *expected*.

    The inference actor records it as it stages the tile, never the driver: the flow runner runs the
    lockfile's CPU build whatever the workers run. A tile without it predates the record, so the
    stack that computed it is unknown and it is refused rather than guessed. *expected* is the
    stack of the run's other tiles, since one run never mixes two.
    """
    stack = attrs.get(STACK_ATTR)
    if not isinstance(stack, Mapping):
        raise ValueError(
            f"Staged tile {where} does not record the PyTorch that computed it: it was staged before "
            "tiles recorded their stack. Re-infer it with a fresh force_staging_restage token, or by "
            "deleting the staging prefix."
        )
    found = {"torch": stack.get("torch"), "cuda": stack.get("cuda")}
    if expected is not None and found != dict(expected):
        raise ValueError(
            f"Staged tile {where} was computed on {_describe_stack(found)}, and this run's other tiles "
            f"on {_describe_stack(expected)}. One run never mixes two stacks; re-infer it on one."
        )
    return found


def assert_stack_matches(stored: object, staged: Mapping[str, str | None], *, allow_mismatch: bool, where: str) -> None:
    """Refuse a write whose tiles were computed on a different library stack from the store's.

    ADR 025 rule 2: a store is finished on the stack it began on, because another PyTorch or CUDA
    shifts the embeddings slightly and nothing downstream would notice the mix. ``stored is None``
    is a store written before the record, which the write stamps rather than refuses. One
    implementation for the single-area and global write paths, as :func:`assert_encoder_matches`
    is for the model.
    """
    if stored is None or allow_mismatch or dict(cast("Mapping[str, object]", stored)) == dict(staged):
        return
    raise ValueError(
        f"Refusing to write into {where}: it was computed on "
        f"{_describe_stack(cast('Mapping[str, object]', stored))}, and this run's tiles on "
        f"{_describe_stack(staged)}. Another stack shifts the embeddings slightly, so one store never "
        "mixes two (ADR 025). Run on the store's stack: the global flows pick its worker image through "
        "ami_ssm_name_by_model, and a local run needs that PyTorch installed (docs/environment-setup.md). "
        "allow_stack_mismatch=True accepts the mix, for a deliberate experiment only."
    )


#: Staged-tile attr naming the model that computed the embeddings:
#: ``{"model_version": <ModelVersion>, "checkpoint_id": <checkpoint file stem>}``.
MODEL_ATTR = "inference_model"


def checkpoint_stem(checkpoint: str) -> str:
    """A checkpoint path or file name reduced to its stem, the form ``checkpoint_id`` compares in."""
    return checkpoint.rsplit("/", 1)[-1].removesuffix(".pt")


def staged_model(
    attrs: Mapping[str, object], where: str, *, expected: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The model and checkpoint a staged tile was computed with, refusing one that differs from *expected*.

    A tile without the record predates it, so the model that computed it is unknown and it is
    refused rather than guessed, as :func:`staged_stack` refuses a tile without its stack.
    *expected* is the probe tile's model: a resumed run can hold tiles from two checkpoints.
    """
    model = attrs.get(MODEL_ATTR)
    if not isinstance(model, Mapping):
        raise ValueError(
            f"Staged tile {where} does not record the model that computed it: it was staged before "
            "tiles recorded their model. Re-infer it with a fresh force_staging_restage token, or by "
            "deleting the staging prefix."
        )
    found = {"model_version": str(model.get("model_version")), "checkpoint_id": str(model.get("checkpoint_id"))}
    if expected is not None and found != dict(expected):
        raise ValueError(
            f"Staged tile {where} was computed with {found}, and this run's other tiles with "
            f"{dict(expected)}. One run never mixes two models; re-infer it with one."
        )
    return found


def assert_model_matches(
    root_attrs: Mapping[str, object], model: Mapping[str, str], *, allow_mismatch: bool, where: str
) -> None:
    """Refuse a write whose model or checkpoint disagrees with the store root's.

    **One implementation for every reader:** the fills' pre-flight, which answers before any GPU
    is provisioned, and the global write path, which every write into a store passes through (the
    Prefect flows, a direct runner call, the plain runner's updates). The encoder is compared as
    :func:`assert_encoder_matches` compares it, a root without ``geoemb:model`` reading as v1.1.
    The checkpoint is compared only when the root records a ``checkpoint_id``, as stems, because a
    global root records the file name and a single-area root its stem. ``allow_mismatch`` is the
    deliberate override, for a store seeded with a custom ``model_url``.
    """
    if allow_mismatch:
        return
    published = root_attrs.get("geoemb:model")
    assert_encoder_matches(
        published if isinstance(published, str) else None,
        model_version=cast("ModelVersion", model["model_version"]),
        where=where,
    )
    stored = root_attrs.get("checkpoint_id")
    if stored is not None and checkpoint_stem(str(stored)) != checkpoint_stem(model["checkpoint_id"]):
        raise ValueError(
            f"Refusing to write into {where}: it holds checkpoint {stored!r} and this run's tiles come "
            f"from {model['checkpoint_id']!r}. The encoder matches but the checkpoint (norm source) "
            "differs, so the embeddings would be mixed. Reseed, or pass allow_model_mismatch=True to "
            "override deliberately."
        )


def tile_id_to_epsg(tile_id: str) -> str | None:
    """Derive EPSG code from a Sentinel-2 MGRS tile ID.

    The first two characters encode the UTM zone number (01-60). The third is the latitude
    band letter: bands N-X are northern hemisphere (EPSG:326xx), bands C-M southern
    (EPSG:327xx).

    Args:
        tile_id: MGRS tile ID, e.g. ``"37PBM"``, ``"33UWP"``, ``"56HKH"``.

    Returns:
        An EPSG authority code (e.g. ``"EPSG:32637"``), or ``None`` if *tile_id* does not
        look like a valid MGRS tile ID.
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
    """Derive ``proj:code``, ``proj:wkt2`` and ``proj:projjson`` from an EPSG code via pyproj.

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
    """Compute a 6-element affine transform from PIXEL-CENTRE coordinate arrays.

    For axis-aligned grids the transform is::

        [scale_x, 0, origin_x, 0, scale_y, origin_y]

    **The origin is the OUTER CORNER of the first pixel, not its centre.** The ``spatial:``
    convention defines the transform over array indices where ``(0, 0)`` is the top-left
    corner of the top-left pixel and the first pixel's centre is at ``(0.5, 0.5)``, so ``c``
    is the western-most coordinate of the X axis and ``f`` the northern-most of the Y axis —
    the same rule as a GDAL geotransform or a rasterio ``Affine``. Our coordinate arrays are
    pixel CENTRES on both paths that reach here (``zone_grid`` builds them as
    ``edge + (i + 0.5) * pitch``, and ``odc`` writes centres into every mosaic), so half a
    pixel is stepped back off each axis to reach the corner the convention asks for.

    Emitting the centre instead is not a harmless convention difference: it displaces every
    pixel half a cell diagonally for any consumer that reads the transform, and it
    contradicts :func:`_compute_bbox`, which is already edge-based. The two must agree —
    ``c`` is ``bbox[0]`` and ``f`` is ``bbox[3]`` for a north-up grid — and a test pins that.

    The scale is the **median** spacing between consecutive coordinates, not
    ``coords[1] - coords[0]``: in non-UTM projections the float representation of
    geographic coordinates introduces small rounding differences along the axis.

    Spacing more than 1 % off the median logs a warning — a non-regular grid, where an
    affine transform is only an approximation.
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

    # Half a pixel back from the first CENTRE, along each axis's own sign: for a north-up
    # grid res_y is negative, so this moves the Y origin UP to the northern edge and the X
    # origin LEFT to the western edge. Signed rather than `abs()`, so a south-up or
    # west-negative grid lands on its own leading edge instead of a pixel inside the grid.
    origin_x = float(x_coords[0]) - res_x / 2
    origin_y = float(y_coords[0]) - res_y / 2
    return [res_x, 0.0, origin_x, 0.0, res_y, origin_y]


def _compute_bbox(y_coords: np.ndarray, x_coords: np.ndarray) -> list[float]:
    """Compute a bounding box [xmin, ymin, xmax, ymax] for pixel-registered data.

    Per the ``spatial:`` convention's pixel registration, the bbox extends half a pixel
    beyond the outermost coordinate centres on all sides.
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
    checkpoint_id: str | None,
    model_url: str | None,
    data_type: str,
    gsd: float | None,
    spatial_layout: str | None,
    source_data: tuple[str, ...],
    model_version: ModelVersion | None,
) -> dict:
    """The ``geoemb:*`` (+ ``checkpoint_id``) attrs, shared by the single-ROI and multi-group builders.

    *gsd* arrives already resolved to a trustworthy metre value or ``None``. See
    :func:`build_convention_attrs` for the meaning of each ``geoemb:`` field.
    """
    attrs: dict = {
        "geoemb:type": "pixel",  # per-pixel embeddings (not chip)
        "geoemb:dimensions": embedding_dim,
        # The PUBLIC encoder reference, resolved from the model and never from the checkpoint's
        # filename stem, which is kept separately as `checkpoint_id` below, so a custom checkpoint
        # cannot advertise a synthetic public URL.
        "geoemb:model": expected_model_url(model_url, model_version=model_version),
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
    if checkpoint_id:
        attrs["checkpoint_id"] = checkpoint_id
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
    checkpoint_id: str | None = None,
    model_version: ModelVersion | None = None,
    model_url: str | None = None,
    data_type: str = QUANTIZED_DTYPE,
    source_data: tuple[str, ...] = DEFAULT_SOURCE_DATA,
) -> dict:
    """``geoemb:`` attrs for the ROOT of a multi-group store (``utm_zones`` / ``global``).

    In the ``utm_zones`` layout the encoder/quantization provenance is stated ONCE at the
    root, being identical across zones, while ``proj:``/``spatial:`` live on each per-zone
    group whose CRS and grid differ — build those with :func:`build_convention_attrs` and
    ``include_geoemb=False``. *gsd* is explicit here because the root has no single
    coordinate grid to derive it from.
    """
    attrs = _geoemb_fields(
        embedding_dim=embedding_dim,
        checkpoint_id=checkpoint_id,
        model_version=model_version,
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
    checkpoint_id: str | None = None,
    model_version: ModelVersion | None = None,
    model_url: str | None = None,
    data_type: str = QUANTIZED_DTYPE,
    gsd: float | None = None,
    spatial_layout: str | None = None,
    source_data: tuple[str, ...] = DEFAULT_SOURCE_DATA,
    include_geoemb: bool = True,
) -> dict:
    """Build GeoZarr convention attributes for the root group.

    Returns a flat dict for the Zarr root group: the ``zarr_conventions`` registration plus
    ``proj:*``, ``spatial:*`` and ``geoemb:*`` metadata. *epsg_code* takes precedence over
    *tile_id*, so a caller that reprojects records the actual output CRS rather than the
    MGRS tile's original UTM zone; if neither resolves to a valid EPSG code the ``proj:``
    convention is omitted.

    The ``geoemb:`` fields record encoder-model provenance and quantization:

    * ``geoemb:model`` — the PUBLIC encoder reference URL: *model_url* when the caller
      supplies the exact public URI, else *model_version*'s (e.g. ``"v2-large"``); one of the
      two is required. Never built from the checkpoint's filename stem, which is recorded as
      a plain ``checkpoint_id`` provenance attr instead.
    * ``geoemb:build_version`` — the software/package version.
    * ``geoemb:data_type`` — the quantized storage dtype (*data_type*).
    * ``geoemb:gsd`` — metres, and emitted only when trustworthy: from a metre-based CRS's
      coordinate spacing, or an explicit *gsd* the caller vouches for. OMITTED for a
      non-metre CRS (degrees, feet) or absent coords rather than given a false value.
    * ``geoemb:spatial_layout`` — ``"utm_zones"``/``"global"``; OMITTED when ``None``, since
      a single-ROI store has no utmNN/global groups.
    * ``geoemb:source_data`` — the source-dataset URLs.

    *include_geoemb* False emits only ``proj:``/``spatial:``: the multi-group campaign store
    carries geoemb: once at the root (:func:`build_geoemb_root_attrs`) and uses this builder
    only for each zone's CRS/grid.
    """
    conventions: list[dict] = []
    attrs: dict = {}

    # --- proj: convention ---
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
    if include_geoemb:
        conventions.append(_GEOEMB_CONVENTION)
        # geoemb:gsd is OPTIONAL and in metres, so derive it only from a metre-based CRS's
        # spacing; otherwise fall back to the caller's explicit value (possibly None).
        if x_coords is not None and len(x_coords) > 1 and _is_metre_crs(effective_epsg):
            gsd_val: float | None = abs(float(np.median(np.diff(x_coords))))
        else:
            gsd_val = gsd
        attrs.update(
            _geoemb_fields(
                embedding_dim=embedding_dim,
                checkpoint_id=checkpoint_id,
                model_version=model_version,
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
