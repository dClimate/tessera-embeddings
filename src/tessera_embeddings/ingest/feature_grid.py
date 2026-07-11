"""Rasterize FeatureCollection features onto a shared master ROI grid.

The geometry foundation for the ROI fan-out → region-merge pipeline: ingest one
mosaic per FeatureCollection feature, then region-write each per-feature mosaic
into a single master mosaic (see :mod:`tessera_embeddings.storage.region_merge`).
Region writes place data **positionally**, so a write is only correct when each
feature grid's coordinates are an **exact pixel-subset** of the master grid's —
same CRS, resolution, *and* pixel-aligned origin.

:func:`~tessera_embeddings.ingest.roi.rasterize_roi_zarr` can't give us that: it
has no geobox parameter and anchors every ROI to *its own* bounds
(``from_origin(minx, maxy, ...)``), so two independently-rasterized features land
at fractional-pixel offsets from each other. Alignment must instead come from the
**master geobox** (an input — the grid authority). :func:`rasterize_feature_roi`
burns each feature onto ``master_geobox.enclosing(geom)`` — a grid-aligned window
of the master that odc guarantees shares the parent origin and resolution exactly
— so exact-subset coordinates are true *by construction*, not by post-hoc
validation. The output store is byte-for-byte the format ``rasterize_roi_zarr``
produces (same ``crs`` / ``transform`` / ``bbox_wgs84`` / ``_manifest`` attrs,
bool dtype, chunking) so the ingest path reads it unchanged.

:func:`classify_mask` (a bool ROI mask) and :func:`classify_store` (a mosaic
store) are the fan-out's skip/triage checks — both return a :class:`StoreDiagnosis`
of ``PRESENT`` / ``MISSING`` / ``CORRUPTED``.

Domain-layer rules apply: stdlib logging only (no orchestrator imports), storage
via the fsspec-backed helpers (no boto3).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import fsspec
import zarr
from affine import Affine
from odc.geo.geom import Geometry
from rasterio.crs import CRS
from rasterio.features import rasterize
from shapely.geometry import mapping, shape

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE
from tessera_embeddings.ingest.roi import check_output_exists, reproject_geometries
from tessera_embeddings.storage.manifest import RoiManifest
from tessera_embeddings.storage.zarr_store import open_store

if TYPE_CHECKING:
    from odc.geo.geobox import GeoBox
    from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)

_WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True)
class FeatureRecord:
    """One FeatureCollection feature, ready to rasterize onto the master grid.

    Attributes:
        feature_id: Stringified value of the feature's id property (e.g.
            ``tile_id``). Used to derive the per-feature ROI / mosaic name.
        geometry: Feature geometry reprojected into the master CRS — the geometry
            burned into the per-feature mask, and the one the disjoint guard checks.
        geometry_wgs84: Feature geometry in EPSG:4326, for the ``bbox_wgs84`` attr
            (kept projection-invariant the same way ``rasterize_roi_zarr`` does:
            computed from the geometry before the master-CRS reprojection).
    """

    feature_id: str
    geometry: BaseGeometry
    geometry_wgs84: BaseGeometry


def load_features(
    geojson_path: str,
    *,
    target_crs: str,
    id_property: str,
) -> list[FeatureRecord]:
    """Load FeatureCollection features as per-feature records in the master CRS.

    Unlike :func:`~tessera_embeddings.ingest.roi.load_geometries_from_geojson`
    (which unions away per-feature identity), this keeps each feature distinct and
    carries its id property through, because the fan-out needs one ROI / mosaic
    *per feature*. CRS detection mirrors that loader: an embedded GeoJSON ``crs``
    name wins, else WGS84 is assumed.

    Args:
        geojson_path: Any fsspec URI to a GeoJSON FeatureCollection.
        target_crs: CRS to reproject geometries into — pass the master ROI's
            ``native_crs`` so every feature shares the master grid's projection.
        id_property: Feature property holding the per-feature identifier (e.g.
            ``"tile_id"``). Must be present and unique across features.

    Returns:
        One :class:`FeatureRecord` per feature, geometry in ``target_crs``.

    Raises:
        ValueError: empty collection, missing id property, or duplicate ids.
    """
    features, crs_name = _read_feature_collection(geojson_path)
    input_crs = _detect_input_crs(crs_name)
    target = CRS.from_user_input(target_crs)

    geoms_native = [shape(feat["geometry"]) for feat in features]
    geoms_target = reproject_geometries(geoms_native, input_crs, target)
    geoms_wgs84 = reproject_geometries(geoms_native, input_crs, _WGS84) if input_crs != _WGS84 else geoms_native

    records = _build_records(features, geoms_target, geoms_wgs84, id_property=id_property, source=geojson_path)
    logger.info("Loaded %d feature(s) from %s, reprojected to %s", len(records), geojson_path, target)
    return records


def _read_feature_collection(geojson_path: str) -> tuple[list[dict], str | None]:
    """Open a GeoJSON FeatureCollection; return its non-empty features and CRS name.

    The second element is the embedded ``crs`` name (``None`` if absent), passed
    on to :func:`_detect_input_crs` so CRS resolution stays decoupled from IO.

    Raises:
        ValueError: the document is not a FeatureCollection, or carries no features.
    """
    with fsspec.open(geojson_path, "r") as f:
        geojson = json.load(f)

    if geojson.get("type") != "FeatureCollection":
        raise ValueError(
            f"Expected a FeatureCollection at {geojson_path}, got {geojson.get('type')!r}. "
            "The fan-out needs one ROI per feature."
        )
    features = geojson.get("features") or []
    if not features:
        raise ValueError(f"No features found in {geojson_path}")
    crs_name = geojson.get("crs", {}).get("properties", {}).get("name")
    return features, crs_name


def _detect_input_crs(crs_name: str | None) -> CRS:
    """Resolve the input CRS from an embedded GeoJSON ``crs`` name.

    A parseable embedded name wins; otherwise (absent or unparseable) WGS84 is
    assumed, with a warning for the unparseable case.
    """
    if not crs_name:
        return _WGS84
    try:
        return CRS.from_user_input(crs_name)
    except Exception:  # any unparseable CRS name falls back to WGS84
        logger.warning("Could not parse GeoJSON CRS %r, assuming WGS84", crs_name)
        return _WGS84


def _build_records(
    features: list[dict],
    geoms_target: list[BaseGeometry],
    geoms_wgs84: list[BaseGeometry],
    *,
    id_property: str,
    source: str,
) -> list[FeatureRecord]:
    """Pair each feature's id with its reprojected geometries into a record.

    Raises:
        ValueError: a feature is missing ``id_property``, or two features share an id.
    """
    records: list[FeatureRecord] = []
    seen: set[str] = set()
    for feat, geom_t, geom_w in zip(features, geoms_target, geoms_wgs84, strict=True):
        props = feat.get("properties") or {}
        if id_property not in props:
            raise ValueError(f"Feature is missing id property {id_property!r} (have {sorted(props)}) in {source}")
        fid = str(props[id_property])
        if fid in seen:
            raise ValueError(f"Duplicate feature id {fid!r} (property {id_property!r}) in {source}")
        seen.add(fid)
        records.append(FeatureRecord(feature_id=fid, geometry=geom_t, geometry_wgs84=geom_w))
    return records


def assert_features_disjoint(features: list[FeatureRecord]) -> None:
    """Guard: every pair of feature geometries must have zero-area overlap.

    Region writes overwrite committed cells, so two overlapping features would
    silently last-writer-win on the shared cells of the master mosaic. The fan-out
    contract is that features are disjoint; this fails loud if they are not, naming
    the offending pair, before any (expensive) ingest is dispatched.

    Geometries must already be in a common (projected) CRS — pass the master-CRS
    geometries from :func:`load_features`. Touching borders (shared edges, zero
    overlap area) are allowed; only positive-area intersection is rejected.

    This is a geometry-level check, but region writes work on *pixels*. The two are
    reconciled by :func:`rasterize_feature_roi`'s ``all_touched=False``: every
    feature burns onto the shared master grid under the center-rule, so a pixel
    whose center lies on a shared edge is claimed by at most one feature. That keeps
    geometry-disjoint features pixel-disjoint without an explicit pixel check here.
    """
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a, b = features[i], features[j]
            if not a.geometry.intersects(b.geometry):
                continue
            overlap = a.geometry.intersection(b.geometry).area
            if overlap > 0:
                raise ValueError(
                    f"Features {a.feature_id!r} and {b.feature_id!r} overlap by {overlap:.4g} "
                    "projected-unit² — the fan-out requires disjoint features (region writes "
                    "overwrite, so overlap would silently last-writer-win on shared cells)."
                )


def feature_window(master_geobox: GeoBox, geometry: BaseGeometry) -> GeoBox:
    """Return the master-grid window enclosing *geometry*, clamped to the master.

    ``master_geobox.enclosing`` snaps the geometry's bounds outward to whole master
    pixels and shares the master's origin/resolution exactly; ``overlap_roi`` then
    clamps that window to the master extent. The returned sub-geobox is therefore a
    grid-aligned subset of the master by construction — which is what makes the
    later region write align positionally.

    Args:
        master_geobox: The grid authority (from ``read_roi_metadata(...).geobox``).
        geometry: Feature geometry in the master CRS.

    Raises:
        ValueError: the geometry falls entirely outside the master grid.
    """
    window = master_geobox.enclosing(Geometry(geometry, crs=master_geobox.crs))
    yroi, xroi = master_geobox.overlap_roi(window)
    # A window that doesn't overlap the master yields a slice whose stop is at or
    # below its start (overlap_roi clamps to [0, n], so the empty intersection
    # collapses). Catch it here: slicing the geobox with such a range would build a
    # negative-size GeoBox and raise odc.geo's opaque "negative sizes" error before
    # the friendly guard below could run.
    if yroi.stop <= yroi.start or xroi.stop <= xroi.start:
        raise ValueError(
            f"Feature bounds {geometry.bounds} fall outside the master grid "
            f"({master_geobox.width}x{master_geobox.height}px). Every feature must lie within the master ROI."
        )
    return master_geobox[yroi, xroi]


def rasterize_feature_roi(
    output_path: str,
    *,
    master_geobox: GeoBox,
    feature: FeatureRecord,
    chunk_size: int = INGEST_CHUNK_SIZE,
) -> str:
    """Rasterize one feature onto a grid-aligned window of the master geobox.

    Writes a chunked boolean Zarr ROI mask in the exact format
    :func:`~tessera_embeddings.ingest.roi.rasterize_roi_zarr` produces (``crs`` /
    ``transform`` / ``bbox_wgs84`` / ``_manifest`` attrs, bool dtype, ``chunk_size``
    spatial chunking) so the ingest path reads it with no special-casing. The grid
    is the master window (see :func:`feature_window`) — guaranteeing the mask's
    coordinates are an exact pixel-subset of the master's.

    Args:
        output_path: Destination Zarr store URI for the per-feature ROI mask.
        master_geobox: The grid authority (shared origin / resolution / CRS).
        feature: The feature to burn (geometry already in the master CRS).
        chunk_size: Spatial chunk size in pixels (default ``INGEST_CHUNK_SIZE``,
            matching the ingest pipeline so reads are a clean merge).

    Returns:
        ``output_path``.
    """
    sub = feature_window(master_geobox, feature.geometry)
    valid_pixels = _write_bool_mask(
        output_path,
        geobox=sub,
        geom_pairs=[(mapping(feature.geometry), 1)],
        bbox_wgs84=feature.geometry_wgs84.bounds,
        chunk_size=chunk_size,
    )

    total = sub.height * sub.width
    logger.info(
        "Wrote feature ROI %s: %dx%dpx, coverage %d/%d (%.1f%%)",
        output_path,
        sub.width,
        sub.height,
        valid_pixels,
        total,
        100 * valid_pixels / total if total else 0.0,
    )
    return output_path


def rasterize_full_masked_roi(
    output_path: str,
    *,
    master_geobox: GeoBox,
    features: list[FeatureRecord],
    chunk_size: int = INGEST_CHUNK_SIZE,
) -> str:
    """Burn every feature onto the FULL master grid as one boolean coverage mask.

    Where :func:`rasterize_feature_roi` rasterizes a single feature onto a snapped
    *window* of the master (for the per-feature ingest), this produces a mask with
    the master's **exact shape and grid**: ``True`` for any pixel covered by a
    feature, ``False`` everywhere else. This is the single ROI a downstream
    consumer (e.g. inference chunk-filtering) reads, so the mask marks only the
    regions features actually cover, not the master's full rectangular extent.

    The output is byte-for-byte the format ``rasterize_roi_zarr`` produces (``crs``
    / ``transform`` / ``resolution`` / ``bbox_wgs84`` / ``_manifest`` attrs, bool
    dtype, ``chunk_size`` spatial chunking). ``bbox_wgs84`` is the union of the
    features' WGS84 bounds. All features burn onto the *same* master pixel grid
    under the center-rule (``all_touched=False``), matching how
    :func:`rasterize_feature_roi` burns each window — so this mask is exactly the
    union of the per-feature masks.

    Args:
        output_path: Destination Zarr store URI for the full master-grid mask.
        master_geobox: The grid authority (shared origin / resolution / CRS) — the
            mask adopts its full shape, transform, and CRS.
        features: Every feature to burn (geometries already in the master CRS).
        chunk_size: Spatial chunk size in pixels (default ``INGEST_CHUNK_SIZE``).

    Returns:
        ``output_path``.

    Raises:
        ValueError: ``features`` is empty.
    """
    if not features:
        raise ValueError("features is empty; nothing to rasterize into the full masked ROI")

    valid_pixels = _write_bool_mask(
        output_path,
        geobox=master_geobox,
        geom_pairs=[(mapping(f.geometry), 1) for f in features],
        bbox_wgs84=_union_bounds(f.geometry_wgs84 for f in features),
        chunk_size=chunk_size,
    )

    total = master_geobox.height * master_geobox.width
    logger.info(
        "Wrote full masked ROI %s: %dx%dpx, coverage %d/%d (%.1f%%) over %d feature(s)",
        output_path,
        master_geobox.width,
        master_geobox.height,
        valid_pixels,
        total,
        100 * valid_pixels / total if total else 0.0,
        len(features),
    )
    return output_path


def _write_bool_mask(
    output_path: str,
    *,
    geobox: GeoBox,
    geom_pairs: list[tuple[dict, int]],
    bbox_wgs84: tuple[float, float, float, float],
    chunk_size: int,
) -> int:
    """Burn ``geom_pairs`` onto ``geobox`` as a chunked boolean Zarr ROI mask.

    The shared rasterization body behind :func:`rasterize_feature_roi` (one
    feature, a snapped master window) and :func:`rasterize_full_masked_roi` (all
    features, the full master grid). Writes the exact format
    :func:`~tessera_embeddings.ingest.roi.rasterize_roi_zarr` produces (``crs`` /
    ``transform`` / ``resolution`` / ``bbox_wgs84`` / ``_manifest`` attrs, bool
    dtype, ``chunk_size`` spatial chunking). Rasterizes one spatial chunk at a time
    so the full mask is never held in memory.

    Args:
        output_path: Destination Zarr store URI.
        geobox: The output grid — fixes shape, transform, resolution, and CRS.
        geom_pairs: ``(geojson_mapping, burn_value)`` pairs (geometries in the
            geobox CRS), passed straight to ``rasterio.features.rasterize``.
        bbox_wgs84: WGS84 ``(minx, miny, maxx, maxy)`` for the ``bbox_wgs84`` attr.
        chunk_size: Spatial chunk size in pixels.

    Returns:
        Count of pixels burned ``True``.
    """
    height, width = geobox.height, geobox.width
    transform = geobox.affine
    crs_str = str(geobox.crs)
    resolution = abs(transform.a)

    z = zarr.open(
        output_path,
        mode="w",
        shape=(height, width),
        chunks=(chunk_size, chunk_size),
        dtype="bool",
    )
    assert isinstance(z, zarr.Array)
    z.attrs["crs"] = crs_str
    z.attrs["transform"] = list(transform)[:6]
    z.attrs["resolution"] = resolution
    z.attrs["bbox_wgs84"] = list(bbox_wgs84)

    valid_pixels = 0
    for y0 in range(0, height, chunk_size):
        for x0 in range(0, width, chunk_size):
            ch = min(chunk_size, height - y0)
            cw = min(chunk_size, width - x0)
            chunk_transform = transform * Affine.translation(x0, y0)
            # all_touched=False (the rasterio default, pinned here deliberately) is
            # what keeps the per-feature masks a clean partition of the master grid.
            # Every feature burns onto the SAME master pixel grid (feature_window and
            # the full-grid burn share the master origin/resolution), and the
            # center-rule burns a pixel only when its CENTER lies inside the polygon.
            # Two features that share an edge (allowed by assert_features_disjoint —
            # zero-area overlap) therefore cannot claim the same pixel: a center falls
            # inside at most one of two interior-disjoint polygons. So the full-grid
            # mask equals the union of the per-feature window masks exactly. Flipping
            # this to all_touched=True would burn every boundary-crossing pixel for
            # BOTH neighbours, so shared edges would yield shared pixels and the region
            # merge would silently last-writer-win on them. Do not change without a
            # pixel-overlap guard.
            chunk = rasterize(
                geom_pairs,
                out_shape=(ch, cw),
                transform=chunk_transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            )
            z[y0 : y0 + ch, x0 : x0 + cw] = chunk > 0
            valid_pixels += int(chunk.sum())

    manifest = RoiManifest(resolution=resolution, chunk_size=chunk_size, crs=crs_str)
    z.attrs["_manifest"] = manifest.to_dict()
    return valid_pixels


def _union_bounds(geometries: Iterable[BaseGeometry]) -> tuple[float, float, float, float]:
    """Axis-aligned bounds enclosing all *geometries* (minx, miny, maxx, maxy)."""
    bounds = [g.bounds for g in geometries]
    return (
        min(b[0] for b in bounds),
        min(b[1] for b in bounds),
        max(b[2] for b in bounds),
        max(b[3] for b in bounds),
    )


# --------------------------------------------------------------------------- #
# Store triage — the fan-out's skip/re-run/manual-curation state machine
# --------------------------------------------------------------------------- #

# Store states the fan-out triages on, in escalating-attention order.
PRESENT = "present"  # opens with the expected schema — a functioning store
MISSING = "missing"  # nothing at the path — never written (or a genuinely-empty run)
CORRUPTED = "corrupted"  # won't open OR malformed (missing expected var(s)) — manual curation
STORE_STATES: tuple[str, ...] = (PRESENT, MISSING, CORRUPTED)


@dataclass(frozen=True)
class StoreDiagnosis:
    """A store's triage state plus a human-readable reason.

    ``status`` is one of :data:`STORE_STATES`; ``detail`` explains what was found
    (e.g. ``"opens with all 2 variable(s)"``, ``"missing variable(s) ['scl']"``) so
    the fan-out can log complete, actionable triage without re-opening the store.
    """

    status: str
    detail: str


def classify_mask(mask_path: str, master_geobox: GeoBox) -> StoreDiagnosis:
    """Diagnose a boolean ROI mask store so the fan-out can skip re-rasterizing it.

    The mask is a single boolean Zarr array (not an Icechunk mosaic), so "valid" is
    checked against the format :func:`_write_bool_mask` writes:

    * ``MISSING`` — nothing at the path. Rasterize it.
    * ``PRESENT`` — opens as a ``bool`` array of the master's exact shape **and**
      carries the ``_manifest`` attr. :func:`_write_bool_mask` writes ``_manifest``
      **last** (after every spatial chunk), so its presence proves the write ran to
      completion — the mask's completeness signal. Skip the rasterize.
    * ``CORRUPTED`` — bytes exist but the store won't open, isn't a ``bool`` array,
      has no ``_manifest`` (a write interrupted mid-stream), or its shape disagrees
      with the current master grid (a stale mask from a different master). A mask is
      fully reproducible from (features + master grid), so the caller can just
      re-rasterize — the status is surfaced for logging, not to gate work.

    Args:
        mask_path: Full ROI-mask store URI.
        master_geobox: The grid authority the mask must match (shape check).

    Returns:
        A :class:`StoreDiagnosis`.
    """
    if not check_output_exists(mask_path):
        return StoreDiagnosis(MISSING, "no mask at path")

    try:
        z = zarr.open(mask_path, mode="r")
    except Exception as e:  # bytes on disk that won't open is "corrupted", not missing
        return StoreDiagnosis(CORRUPTED, f"exists but cannot open: {e}")

    if not isinstance(z, zarr.Array) or z.dtype != "bool":
        kind = type(z).__name__ if not isinstance(z, zarr.Array) else f"dtype {z.dtype}"
        return StoreDiagnosis(CORRUPTED, f"opens but is not a bool array ({kind})")
    expected = (master_geobox.height, master_geobox.width)
    if tuple(z.shape) != expected:
        return StoreDiagnosis(CORRUPTED, f"shape {tuple(z.shape)} != master grid {expected} (stale mask)")
    if "_manifest" not in z.attrs:
        return StoreDiagnosis(CORRUPTED, "opens but write did not finish (no _manifest attr)")
    return StoreDiagnosis(PRESENT, f"bool mask matching master grid {expected}")


def classify_store(store_path: str, expected_vars: Iterable[str]) -> StoreDiagnosis:
    """Diagnose a per-feature mosaic store against an expected variable set.

    The fan-out classifies every (feature, kind) store before dispatching so it can
    skip finished work, re-run what was never started, and surface anything
    malformed for manual curation:

    * ``MISSING`` — nothing exists at the path. Either never ingested, or a
      completed run that found genuinely no data and wrote no store. The caller
      (re-)ingests; that run is cheap and non-destructive.
    * ``CORRUPTED`` — the store is not wholly formed: either it won't open (a
      partially-written repo, a leftover non-store directory), or it opens but is
      missing an expected data variable (schema drift). Existing data can't be
      trusted, so the caller must not silently overwrite — manual curation.
    * ``PRESENT`` — opens and carries every variable in ``expected_vars``: a
      functioning store.

    Args:
        store_path: Full store URI.
        expected_vars: The variable names a wholly-formed store must contain (e.g.
            a mosaic kind's band set). A store missing any is ``CORRUPTED``.

    Returns:
        A :class:`StoreDiagnosis`.
    """
    expected = set(expected_vars)
    if not check_output_exists(store_path):
        return StoreDiagnosis(MISSING, "no store at path")

    try:
        ds = open_store(store_path)
    except Exception as e:  # bytes on disk that won't open is "corrupted", not missing
        return StoreDiagnosis(CORRUPTED, f"exists but cannot open: {e}")
    try:
        missing_vars = sorted(expected - set(ds.data_vars))
    finally:
        ds.close()

    if missing_vars:
        return StoreDiagnosis(CORRUPTED, f"opens but malformed — missing variable(s) {missing_vars}")
    return StoreDiagnosis(PRESENT, f"opens with all {len(expected)} variable(s)")


__all__ = [
    "CORRUPTED",
    "MISSING",
    "PRESENT",
    "STORE_STATES",
    "FeatureRecord",
    "StoreDiagnosis",
    "assert_features_disjoint",
    "classify_mask",
    "classify_store",
    "feature_window",
    "load_features",
    "rasterize_feature_roi",
    "rasterize_full_masked_roi",
]
