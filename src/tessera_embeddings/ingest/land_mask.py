"""Land-mask v1.1 → per-zone coverage bitmaps (ADR-010).

The partner delivers the global land mask as ~1.59M per-0.1°-cell GeoTIFFs
(``grid_{lon}_{lat}.tiff``) plus a ``registry.txt`` and ``SHA256SUM`` under
``s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/``. **v1.1 semantics
(from the partner): every registry-listed tile is all-1s** — v1 followed the
coastline per-pixel but cropped land, so v1.1 dropped per-pixel masking, and
coverage additionally extends ~1 cell (~11 km) into the sea. The consequence is
decisive: *tile presence / the registry listing IS the mask*. No TIFF pixel
needs to be read to build our runtime artifact — only the registry file and
geometry. (Empirically confirmed on the delivery sample: registry membership
tracks all-1s exactly; v1-era all-zero tiles are absent from the registry.)

So the runtime artifact is **per-zone coverage bitmaps**, not a pixel mask:

* ``tile_live_2048`` — bool ``(n_tile_rows, n_tile_cols)``: OUR 2048-px tile is
  live iff ≥1 registry cell's projected footprint intersects it. This is the
  array the zone-fill runner reads (one ~1 KB GET replaces ~15k windowed reads).
* ``chunk_live_256`` — bool ``(n_chunk_rows, n_chunk_cols)``: the same coverage
  at inner-chunk granularity, kept for future coverage-edge shard leaning. By
  construction ``tile_live_2048 == 8x8-block-any(chunk_live_256)``.

All 120 UTM zone groups live in one Icechunk repo (``BucketPaths.land_mask_store()``),
mirroring the global store's layout so the consumer read is exactly
``open_store_as_zarr_group(path, group=zone)`` — the same helper, timeout/retry
hardening, and credential path the rest of the campaign uses. The all-ocean
zones (8 of 120) get empty bitmaps, so the runner always finds a coverage group
and its ``mark_zone_year_empty`` path fires naturally.

Building is pure geometry over the registry (sub-minute, no data-plane reads):
stream ``registry.txt`` → assign each cell to its NOMINAL 6° UTM band by
filename math → per zone, vectorized pyproj projection of each cell's footprint
→ snap to our zone-grid pixel indices → OR the covered tile/chunk rectangles
into the bitmaps → one atomic commit carrying the registry sha256.

Domain-layer rules apply: stdlib logging only (no orchestrator imports), storage
via fsspec / the icechunk helpers (no boto3).
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import cast

import fsspec
import numpy as np
import rasterio
import zarr
from pyproj import Transformer

from tessera_embeddings.config.store_layout import INNER_PX, SHARD_PX
from tessera_embeddings.storage import zone_grid
from tessera_embeddings.storage.zarr_store import open_or_create_repo, open_store_as_zarr_group
from tessera_embeddings.storage.zone_grid import PIXEL_M, ZONE_SCHEME, ZONES, ZoneSpec
from tessera_embeddings.utils import utcnow_iso

logger = logging.getLogger(__name__)

#: Default delivery prefix (the partner's canonical copy).
DEFAULT_DELIVERY_URI = "s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/"
#: Registry filename inside the delivery prefix.
REGISTRY_NAME = "registry.txt"

#: 0.1° cell edge and half-edge (cell CENTERS are at ±(0.05 + 0.1·k)°).
CELL_DEG = 0.1
HALF_DEG = CELL_DEG / 2.0

#: Number of WGS84 sample points per cell (3x3: corners, edge midpoints,
#: centre) whose projected bbox bounds the cell's footprint in the zone CRS.
#: A 0.1° cell projects to a near-rectangle, but the sea buffer lives at zone
#: edges where meridian curvature bows the top/bottom edges, so the midpoints
#: are cheap insurance against under-covering an edge tile.
_SAMPLE_OFFSETS = np.array([-HALF_DEG, 0.0, HALF_DEG])


@dataclass(frozen=True)
class ZoneCoverage:
    """One zone's coverage bitmaps, ready to write onto a group node."""

    zone: str
    tile_live: np.ndarray  # bool (n_tile_rows, n_tile_cols)
    chunk_live: np.ndarray  # bool (n_chunk_rows, n_chunk_cols)
    n_cells: int

    @property
    def n_live_tiles(self) -> int:
        """Count of live 2048-px tiles."""
        return int(self.tile_live.sum())


@dataclass(frozen=True)
class BuildResult:
    """Summary of a full :func:`build_all` run (dict-friendly for the task shell)."""

    dest: str
    snapshot_id: str
    n_zones: int
    zones_with_cells: int
    n_cells: int
    n_live_tiles: int
    registry_sha256: str


# --------------------------------------------------------------------------- #
# Cell → zone geometry (pure, unit-testable)
# --------------------------------------------------------------------------- #
def parse_cell_name(name: str) -> tuple[float, float]:
    """``grid_{lon}_{lat}.tiff`` → ``(lon_center, lat_center)`` in degrees.

    The lon/lat are the cell CENTERS (e.g. ``grid_-0.05_10.05.tiff`` →
    ``(-0.05, 10.05)``). Raises ``ValueError`` on anything not of that form —
    a malformed registry line must fail loudly, never be silently skipped.
    """
    if not (name.startswith("grid_") and name.endswith(".tiff")):
        raise ValueError(f"Registry entry {name!r} is not of the form 'grid_<lon>_<lat>.tiff'")
    body = name[len("grid_") : -len(".tiff")]
    parts = body.split("_")
    if len(parts) != 2:
        raise ValueError(f"Registry entry {name!r} does not parse to exactly (lon, lat)")
    lon, lat = float(parts[0]), float(parts[1])
    _validate_cell_center(name, lon, lat)
    return lon, lat


def _validate_cell_center(name: str, lon: float, lat: float) -> None:
    """Reject non-finite, out-of-range, or off-lattice registry cell centres.

    Centres sit on the 0.1° lattice at ±(0.05 + 0.1·k)°, so ``value * 20`` is an
    odd integer. A typo (``NaN``, ``|lon| > 180``, or a centre off the lattice)
    would otherwise be assigned to the wrong zone or produce shifted/clamped
    coverage — the registry is load-bearing, so fail loudly here.
    """
    for axis, value, limit in (("lon", lon, 180.0), ("lat", lat, 90.0)):
        if not math.isfinite(value):
            raise ValueError(f"Registry entry {name!r} has non-finite {axis} {value}")
        if abs(value) > limit:
            raise ValueError(f"Registry entry {name!r} has out-of-range {axis} {value}")
        twentieths = value * 20.0
        nearest = round(twentieths)
        if abs(twentieths - nearest) > 1e-6 or nearest % 2 == 0:
            raise ValueError(f"Registry entry {name!r} {axis}={value} is not on the 0.1° cell-centre lattice")


@cache
def _transformer(epsg: int) -> Transformer:
    """WGS84 → zone-CRS transformer, cached per zone (there are only 120)."""
    return Transformer.from_crs(4326, epsg, always_xy=True)


def zone_for_cell(lon: float, lat: float) -> str:
    """Cell centre → EPSG code string of its zone (e.g. ``"32630"``).

    Uses the NOMINAL 6° UTM band (``utm_6deg_nominal``, ADR-008) — filename
    math only, never the MGRS exceptions in :func:`ingest.roi.determine_utm_zone`.
    Cells never straddle a 6° band edge (a 6° band is exactly 60 0.1° cells and
    centres are offset 0.05° from every boundary), so the ``floor`` is exact.
    Hemisphere is the sign of the centre latitude (never exactly 0).
    """
    band = math.floor((lon + 180.0) / 6.0) + 1
    band = min(max(band, 1), 60)
    prefix = "326" if lat >= 0 else "327"
    return f"{prefix}{band:02d}"


def project_cells_to_pixel_boxes(
    lons: np.ndarray, lats: np.ndarray, spec: ZoneSpec
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized: cell centres → half-open pixel boxes ``(r0, r1, c0, c1)``.

    Projects each cell's 3x3 sample grid into ``spec``'s CRS, takes the
    projected bbox, and snaps it (floor low / ceil high) to integer pixel
    indices on the zone grid, clamped to ``[0, height] x [0, width]``. Rows are
    measured from the top (north): the northing axis descends, so the *max*
    projected northing maps to the *min* row. Clamping absorbs the ~1-cell sea
    buffer that projects beyond a zone's fixed extent.
    """
    transformer = _transformer(int(spec.epsg))
    dlon, dlat = np.meshgrid(_SAMPLE_OFFSETS, _SAMPLE_OFFSETS)
    sample_lon = lons[:, None] + dlon.ravel()[None, :]  # (n, 9)
    sample_lat = lats[:, None] + dlat.ravel()[None, :]
    east, north = transformer.transform(sample_lon.ravel(), sample_lat.ravel())
    east = np.asarray(east, dtype="float64").reshape(len(lons), _SAMPLE_OFFSETS.size**2)
    north = np.asarray(north, dtype="float64").reshape(len(lons), _SAMPLE_OFFSETS.size**2)

    e0 = spec.easting[0]
    n_top = spec.northing[1]
    c0 = np.clip(np.floor((east.min(axis=1) - e0) / PIXEL_M).astype("int64"), 0, spec.width)
    c1 = np.clip(np.ceil((east.max(axis=1) - e0) / PIXEL_M).astype("int64"), 0, spec.width)
    r0 = np.clip(np.floor((n_top - north.max(axis=1)) / PIXEL_M).astype("int64"), 0, spec.height)
    r1 = np.clip(np.ceil((n_top - north.min(axis=1)) / PIXEL_M).astype("int64"), 0, spec.height)
    return r0, r1, c0, c1


def build_zone_coverage(zone: str, lons: np.ndarray, lats: np.ndarray) -> ZoneCoverage:
    """Build one zone's coverage bitmaps by OR-ing every cell's footprint.

    ``lons``/``lats`` are the centres of the registry cells assigned to ``zone``
    (empty for an all-ocean zone). The zone-grid dimensions are exact multiples
    of :data:`SHARD_PX` and :data:`INNER_PX`, so tile/chunk ranges never spill a
    partial edge. The tile bitmap is derived from the same clamped pixel box as
    the chunk bitmap via the identity ``ceil(ceil(x/256)/8) == ceil(x/2048)``,
    which is what makes ``tile_live == 8x8-block-any(chunk_live)`` hold globally.
    """
    spec = zone_grid.zone(zone)
    tile_live = np.zeros((spec.height // SHARD_PX, spec.width // SHARD_PX), dtype=bool)
    chunk_live = np.zeros((spec.height // INNER_PX, spec.width // INNER_PX), dtype=bool)
    n_cells = len(lons)
    if n_cells:
        r0, r1, c0, c1 = project_cells_to_pixel_boxes(lons, lats, spec)
        for i in range(n_cells):
            if r1[i] <= r0[i] or c1[i] <= c0[i]:
                continue  # footprint clamped entirely outside the zone extent
            tile_live[
                r0[i] // SHARD_PX : _ceil_div(r1[i], SHARD_PX),
                c0[i] // SHARD_PX : _ceil_div(c1[i], SHARD_PX),
            ] = True
            chunk_live[
                r0[i] // INNER_PX : _ceil_div(r1[i], INNER_PX),
                c0[i] // INNER_PX : _ceil_div(c1[i], INNER_PX),
            ] = True
    return ZoneCoverage(zone=zone, tile_live=tile_live, chunk_live=chunk_live, n_cells=n_cells)


def _ceil_div(a: int, b: int) -> int:
    """``ceil(a / b)`` for non-negative ints (the exclusive end of a pixel range).

    ``a`` arrives as a NumPy scalar from the projected pixel arrays; integer
    floor-division keeps it exact, and NumPy ints index zarr slices fine.
    """
    return -(-a // b)


# --------------------------------------------------------------------------- #
# Registry ingest
# --------------------------------------------------------------------------- #
def read_registry(registry_uri: str) -> tuple[list[str], str]:
    """Read the whole registry file → (tile names, sha256 of the raw bytes).

    Reads in one shot (the file is ~120 MB) so the sha256 covers the exact
    bytes we built from — recorded in every zone's attrs as build provenance.
    """
    with fsspec.open(registry_uri, "rb") as fh:
        data = fh.read()
    sha = hashlib.sha256(data).hexdigest()
    names = [line.split()[0] for line in data.decode("utf-8").splitlines() if line.strip()]
    return names, sha


def group_cells_by_zone(names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Parse registry names and group cell centres by zone EPSG string."""
    lon_by_zone: dict[str, list[float]] = defaultdict(list)
    lat_by_zone: dict[str, list[float]] = defaultdict(list)
    for name in names:
        lon, lat = parse_cell_name(name)
        z = zone_for_cell(lon, lat)
        lon_by_zone[z].append(lon)
        lat_by_zone[z].append(lat)
    return {
        z: (np.asarray(lon_by_zone[z], dtype="float64"), np.asarray(lat_by_zone[z], dtype="float64"))
        for z in lon_by_zone
    }


# --------------------------------------------------------------------------- #
# Build (all zones, one commit)
# --------------------------------------------------------------------------- #
def _write_coverage_arrays(node: zarr.Group, cov: ZoneCoverage) -> None:
    """Create the two bool bitmaps as single-chunk arrays on a group node.

    ``overwrite=True`` so a re-delivery rebuilds cleanly into the same repo (a
    new commit replacing the arrays; the old coverage stays in history).
    """
    node.create_array(
        "tile_live_2048",
        data=cov.tile_live,
        chunks=cov.tile_live.shape,
        dimension_names=("tile_row", "tile_col"),
        overwrite=True,
    )
    node.create_array(
        "chunk_live_256",
        data=cov.chunk_live,
        chunks=cov.chunk_live.shape,
        dimension_names=("chunk_row", "chunk_col"),
        overwrite=True,
    )


def build_all(
    dest: str,
    *,
    registry_uri: str | None = None,
    delivery_uri: str = DEFAULT_DELIVERY_URI,
    zones: list[str] | None = None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> BuildResult:
    """Build per-zone coverage bitmaps for all requested zones in one commit.

    Args:
        dest: URI of the coverage Icechunk repo (``BucketPaths.land_mask_store()``).
        registry_uri: Registry file URI; defaults to ``{delivery_uri}/registry.txt``.
        delivery_uri: Partner delivery prefix (recorded in attrs as ``source``).
        zones: Restrict to these EPSG strings (default: all 120). A restricted
            build still commits; unlisted zones simply aren't touched.
        log: Optional logger.

    Returns:
        A :class:`BuildResult` summary (snapshot id, counts, registry sha256).
    """
    log = log or logger
    registry_uri = registry_uri or _join(delivery_uri, REGISTRY_NAME)
    log.info("Reading registry %s", registry_uri)
    names, sha = read_registry(registry_uri)
    cells_by_zone = group_cells_by_zone(names)
    log.info("Registry: %d cells across %d zones", len(names), len(cells_by_zone))

    target_zones = zones if zones is not None else list(ZONES)
    created_at = utcnow_iso()

    repo, _ = open_or_create_repo(dest)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")

    n_cells_total = 0
    n_live_total = 0
    zones_with_cells = 0
    for zone in target_zones:
        spec = zone_grid.zone(zone)
        lons, lats = cells_by_zone.get(zone, (np.empty(0), np.empty(0)))
        cov = build_zone_coverage(zone, lons, lats)
        node = root.require_group(zone)
        _write_coverage_arrays(node, cov)
        node.attrs.update(
            {
                "zone": zone,
                "crs": spec.crs,
                "zone_scheme": ZONE_SCHEME,
                "grid_shape": [spec.height, spec.width],
                "tile_px": SHARD_PX,
                "inner_px": INNER_PX,
                "n_cells": cov.n_cells,
                "n_live_tiles": cov.n_live_tiles,
                "registry_sha256": sha,
                "source": delivery_uri,
                "created_at": created_at,
            }
        )
        n_cells_total += cov.n_cells
        n_live_total += cov.n_live_tiles
        zones_with_cells += int(cov.n_cells > 0)
        log.debug("Zone %s: %d cells → %d live tiles", zone, cov.n_cells, cov.n_live_tiles)

    snapshot = session.commit(
        f"build land-mask coverage: {len(target_zones)} zones, {n_cells_total} cells, registry {sha[:12]}"
    )
    log.info(
        "Committed coverage %s: %d zones (%d with cells), %d cells, %d live tiles",
        snapshot,
        len(target_zones),
        zones_with_cells,
        n_cells_total,
        n_live_total,
    )
    return BuildResult(
        dest=dest,
        snapshot_id=snapshot,
        n_zones=len(target_zones),
        zones_with_cells=zones_with_cells,
        n_cells=n_cells_total,
        n_live_tiles=n_live_total,
        registry_sha256=sha,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _block_any(arr: np.ndarray, factor: int) -> np.ndarray:
    """Down-sample a bool grid by OR-ing each ``factor x factor`` block."""
    nr, nc = arr.shape
    return np.asarray(arr.reshape(nr // factor, factor, nc // factor, factor).any(axis=(1, 3)))


def _group(root: zarr.Group, zone: str) -> zarr.Group:
    """Typed access to a zone subgroup (attrs are JSON, arrays are Array)."""
    return cast("zarr.Group", root[zone])


def _bitmap(node: zarr.Group, name: str) -> np.ndarray:
    """Read a named bitmap array off a group node as a numpy array."""
    return np.asarray(cast("zarr.Array", node[name]))


def validate_coverage(
    dest: str,
    *,
    zones: list[str] | None = None,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> None:
    """Structural + geographic self-checks on a built coverage repo.

    Raises ``ValueError`` on the first inconsistency:

    * bitmap shapes match ``grid_shape`` / ``tile_px`` / ``inner_px``;
    * ``tile_live_2048 == 8x8-block-any(chunk_live_256)`` (the ratio is
      ``SHARD_PX // INNER_PX``);
    * a zone with cells has ≥1 live tile (coverage can't vanish);
    * known-land points are live and a known deep-ocean point is dead.
    """
    log = log or logger
    ratio = SHARD_PX // INNER_PX
    target_zones = zones if zones is not None else list(ZONES)
    root = open_store_as_zarr_group(dest)  # one repo open; groups are indexed off the root
    for zone in target_zones:
        node = _group(root, zone)
        tile_live = _bitmap(node, "tile_live_2048")
        chunk_live = _bitmap(node, "chunk_live_256")
        grid_shape = cast("list[int]", node.attrs["grid_shape"])
        ny, nx = grid_shape[0], grid_shape[1]
        if tile_live.shape != (ny // SHARD_PX, nx // SHARD_PX):
            raise ValueError(f"Zone {zone}: tile_live shape {tile_live.shape} inconsistent with grid_shape {ny}x{nx}")
        if chunk_live.shape != (ny // INNER_PX, nx // INNER_PX):
            raise ValueError(f"Zone {zone}: chunk_live shape {chunk_live.shape} inconsistent with grid_shape {ny}x{nx}")
        if not np.array_equal(tile_live, _block_any(chunk_live, ratio)):
            raise ValueError(f"Zone {zone}: tile_live != block-any(chunk_live, {ratio}) — bitmaps are inconsistent")
        n_cells = cast("int", node.attrs["n_cells"])
        if n_cells > 0 and not tile_live.any():
            raise ValueError(f"Zone {zone}: {n_cells} cells but no live tiles")

    _check_geo_points(root, set(target_zones), log)


#: (lon, lat, expected_live, label) sanity points, all on exact 0.1° cell
#: centres. Land points are in the registry (so must be live); the ocean point
#: is a deep-Pacific gyre far from any buffered coast (so must be dead) — a
#: coarse guard against an inverted or mis-snapped build, not a coastline test.
_GEO_CHECKS: tuple[tuple[float, float, bool, str], ...] = (
    (2.35, 48.85, True, "Paris"),
    (-47.85, -15.75, True, "Brasília"),
    (-160.05, -30.05, False, "South Pacific gyre"),
)


def _check_geo_points(
    root: zarr.Group, validated_zones: set[str], log: logging.Logger | logging.LoggerAdapter[logging.Logger]
) -> None:
    """Assert each in-scope :data:`_GEO_CHECKS` point's tile matches expectation.

    Points whose zone was not validated (a restricted ``--zones`` run) are
    skipped, so subset validation never touches an unbuilt group.
    """
    for lon, lat, expected, label in _GEO_CHECKS:
        zone = zone_for_cell(lon, lat)
        if zone not in validated_zones:
            continue
        spec = zone_grid.zone(zone)
        r0, _, c0, _ = project_cells_to_pixel_boxes(np.array([lon]), np.array([lat]), spec)
        tile_live = _bitmap(_group(root, zone), "tile_live_2048")
        tr, tc = int(r0[0]) // SHARD_PX, int(c0[0]) // SHARD_PX
        if not (0 <= tr < tile_live.shape[0] and 0 <= tc < tile_live.shape[1]):
            raise ValueError(f"Geo check {label}: tile ({tr},{tc}) outside {zone} grid {tile_live.shape}")
        got = bool(tile_live[tr, tc])
        if got != expected:
            raise ValueError(f"Geo check {label} ({lon},{lat}) in {zone}: live={got}, expected {expected}")
        log.debug("Geo check %s: live=%s (ok)", label, got)


# --------------------------------------------------------------------------- #
# Delivery spot-check (guards the load-bearing all-1s assumption)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SpotCheckResult:
    """Outcome of :func:`spot_check_delivery`."""

    checked: int
    all_ones: int
    crs_ok: int
    bounds_ok: int


def spot_check_delivery(
    names: list[str],
    *,
    delivery_uri: str = DEFAULT_DELIVERY_URI,
    n: int = 500,
    bounds_tol_px: float = 2.0,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> SpotCheckResult:
    """Sample ``n`` registry TIFFs from the bucket and verify the v1.1 contract.

    This is the ONLY place a delivery pixel is read, and it exists solely to
    guard the assumption the whole design rests on. For each sampled tile it
    asserts: pixels are all-1s; the file CRS equals the filename-derived zone;
    and the raster bounds match our projected cell bbox within ``bounds_tol_px``
    pixels. Any violation raises ``ValueError`` — the all-1s assumption is not
    something to discover a wrong answer from later.

    Sampling is a deterministic stride over the registry (reproducible across
    runs), not random.
    """
    log = log or logger
    if not names:
        raise ValueError("Registry is empty — nothing to spot-check")
    stride = max(1, len(names) // n)
    sample = names[::stride][:n]
    tol_m = bounds_tol_px * PIXEL_M
    all_ones = crs_ok = bounds_ok = 0
    for name in sample:
        lon, lat = parse_cell_name(name)
        zone = zone_for_cell(lon, lat)
        spec = zone_grid.zone(zone)
        with fsspec.open(_join(delivery_uri, name), "rb") as fh:
            data = fh.read()
        with rasterio.MemoryFile(data) as mf, mf.open() as ds:
            arr = ds.read(1)
            lo, hi = int(arr.min()), int(arr.max())
            if not (lo == 1 and hi == 1):
                raise ValueError(f"{name}: not all-1s (min={lo}, max={hi}) — v1.1 all-1s assumption violated")
            all_ones += 1
            if ds.crs is None or ds.crs.to_epsg() != int(spec.epsg):
                raise ValueError(f"{name}: CRS {ds.crs} != zone {zone} ({spec.crs})")
            crs_ok += 1
            west, south, east, north = _expected_cell_bounds(lon, lat, spec)
            b = ds.bounds
            # Check ALL four edges: a wrong width/height (shifted right/bottom)
            # must not pass the load-bearing georeferencing check.
            if max(abs(b.left - west), abs(b.bottom - south), abs(b.right - east), abs(b.top - north)) > tol_m:
                raise ValueError(
                    f"{name}: bounds {tuple(round(x, 1) for x in b)} off expected "
                    f"({west:.1f},{south:.1f},{east:.1f},{north:.1f}) by >{tol_m} m"
                )
            bounds_ok += 1
    log.info("Delivery spot-check: %d tiles all-1s, CRS + bounds OK", len(sample))
    return SpotCheckResult(checked=len(sample), all_ones=all_ones, crs_ok=crs_ok, bounds_ok=bounds_ok)


def _expected_cell_bounds(lon: float, lat: float, spec: ZoneSpec) -> tuple[float, float, float, float]:
    """Projected ``(west, south, east, north)`` bounds of a cell in its zone CRS.

    Rasterio ``BoundingBox`` order (left, bottom, right, top).
    """
    corner_lon = np.array([lon - HALF_DEG, lon + HALF_DEG, lon + HALF_DEG, lon - HALF_DEG])
    corner_lat = np.array([lat - HALF_DEG, lat - HALF_DEG, lat + HALF_DEG, lat + HALF_DEG])
    ex, ny = _transformer(int(spec.epsg)).transform(corner_lon, corner_lat)
    return float(np.min(ex)), float(np.min(ny)), float(np.max(ex)), float(np.max(ny))


def reconcile_with_bucket(
    names: list[str],
    *,
    delivery_uri: str = DEFAULT_DELIVERY_URI,
    log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
) -> tuple[int, int, int]:
    """List the delivery prefix and reconcile it against the registry.

    Returns ``(n_registry, n_bucket_tiffs, n_extras)``. The registry is the
    authority (presence-based), so ``registry ⊆ bucket`` must hold; ``n_extras``
    are bucket TIFFs absent from the registry (expected: v1-era leftovers,
    including all-zero tiles). Raises if any registry entry is missing from the
    bucket. Lists ~1.7M objects — a manual ``verify`` step, not a hot path.
    """
    log = log or logger
    fs, _ = fsspec.core.url_to_fs(delivery_uri)
    listed = fs.find(delivery_uri)
    bucket = {p.rsplit("/", 1)[-1] for p in listed if p.endswith(".tiff")}
    registry = set(names)
    missing = registry - bucket
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(f"{len(missing)} registry tiles absent from bucket (e.g. {sample}) — delivery incomplete")
    extras = len(bucket - registry)
    log.info("Bucket reconciliation: registry %d ⊆ bucket %d tiffs; %d extras", len(registry), len(bucket), extras)
    return len(registry), len(bucket), extras


def _join(base: str, name: str) -> str:
    """Join a prefix and a name with exactly one separator."""
    return f"{base.rstrip('/')}/{name}"
