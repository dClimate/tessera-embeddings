"""The UTM-zone grids for the global embeddings store (ADR-008, ADR-034).

The world is one Icechunk repo of zone groups, each in its own CRS, in one of two
layouts (:data:`~tessera_embeddings.config.inference.ZoneLayout`). A new store takes its
model's layout; an existing store keeps the one its groups already use.

* ``"utm_zones"`` (v2 onward) — the geoembeddings ``utm_zones`` layout, as on Source
  Coop: 60 groups ``utm01`` … ``utm60``, each a full pole-to-pole column in its
  NORTHERN CRS (EPSG:326xx), the southern hemisphere carried as negative northings.
* ``"hemispheres"`` (v1.1) — 120 groups named by UTM **common name** (``"01N"`` …
  ``"60S"``), each in its hemisphere's CRS (EPSG:326xx north, EPSG:327xx south).

A zone name says which layout it belongs to, so every zone resolves through one
:func:`zone` lookup; only "every zone" questions take a layout. The southern half of a
``utm`` zone is NOT the matching ``S`` zone moved: 10 000 km of false northing is a whole
number of pixels but not of shards, so the two grids share pixels and disagree on shard
boundaries.

**The campaign fills a column in two halves (ADR 036).** ``utm33n`` and ``utm33s`` are the
work units for group ``utm33``: each has the column's whole grid but owns only the shard rows
on its side of the equator, which falls on a shard boundary. :func:`zones` lists the store's
groups; :func:`fill_zones` lists the units the campaign fills, which in the ``hemispheres``
layout are the groups themselves.

**Boundary policy — pure nominal 6° longitude bands.** Zone extents are the
*nominal* 6°-wide UTM bands; coverage is disjoint and every pixel-center belongs
to exactly one zone. The MGRS width exceptions around Norway and Svalbard
(zone 32V widened; 31X-37X irregular) are **NOT** honored — they exist for
navigation charts, not data grids, and would make zone extents irregular for no
storage benefit. Downstream consumers must not assume MGRS zone widths here;
this is recorded machine-readably in the dataset attr
``zone_scheme = "utm_6deg_nominal"``.

**Extents come from an authoritative source, not hand-typed.** Each zone's
projected bounding box is derived from that CRS's official *area of use* in the
EPSG registry (via ``pyproj``), then snapped outward to the 2048-px shard pitch
(:data:`SHARD_PX` x :data:`PIXEL_M` = 20 480 m). Because every 6° band has the
same geometry in its own CRS, all zones of one kind (north, south, full column)
share one extent template — but :func:`derive_extent` and the pinning test
(``tests/unit/storage/test_zone_grid.py``) re-derive per zone from pyproj so an
EPSG-database change can't silently move the grid.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Iterable
from functools import cache
from typing import Literal

import numpy as np
from pyproj import CRS, Transformer

from tessera_embeddings.config.inference import ZoneLayout
from tessera_embeddings.config.store_layout import SHARD_PX

#: Ground sample distance in metres (10 m embeddings grid).
PIXEL_M: float = 10.0
#: Shard pitch in metres — extents snap to this so the grid is shard-aligned.
PITCH_M: float = SHARD_PX * PIXEL_M  # 20 480 m


#: Machine-readable marker of the boundary policy, for dataset attrs.
ZONE_SCHEME: str = "utm_6deg_nominal"

Extent = tuple[float, float]  # (low, high) in projected metres
Half = Literal["n", "s"]


@dataclasses.dataclass(frozen=True)
class ZoneSpec:
    """One UTM zone: CRS, hemisphere, and shard-aligned projected extents."""

    epsg: str  # e.g. "32601"
    hemisphere: str | None  # "N" | "S", or None for a utm_zones column spanning both
    utm_zone: int  # 1..60
    easting: Extent
    northing: Extent
    half: Half | None = None  # a column's fill half; None for a whole group

    @property
    def group_name(self) -> str:
        """Zarr group name for this zone: ``"utm01"`` for a column, else its UTM common name (``"01N"``)."""
        return f"utm{self.utm_zone:02d}" if self.hemisphere is None else f"{self.utm_zone:02d}{self.hemisphere}"

    @property
    def name(self) -> str:
        """The zone's own name: its group's, or a column half's (``"utm01s"``)."""
        return self.group_name + (self.half or "")

    @property
    def tile_rows(self) -> tuple[int, int]:
        """The ``[start, stop)`` shard rows this zone owns: all of them, or a half's side of the equator."""
        n_rows = self.height // SHARD_PX
        if self.half is None:
            return 0, n_rows
        equator = round(self.northing[1] / PITCH_M)
        return (0, equator) if self.half == "n" else (equator, n_rows)

    @property
    def crs(self) -> str:
        """CRS authority string, e.g. ``"EPSG:32601"``."""
        return f"EPSG:{self.epsg}"

    @property
    def width(self) -> int:
        """Easting extent in pixels (a whole multiple of :data:`SHARD_PX`)."""
        return round((self.easting[1] - self.easting[0]) / PIXEL_M)

    @property
    def height(self) -> int:
        """Northing extent in pixels (a whole multiple of :data:`SHARD_PX`)."""
        return round((self.northing[1] - self.northing[0]) / PIXEL_M)


def derive_extent(epsg: int, *also: int) -> tuple[Extent, Extent]:
    """Derive a zone's shard-aligned ``(easting, northing)`` extent from pyproj.

    Projects the EPSG area-of-use (WGS84 lon/lat box, densely sampled to capture
    meridian curvature) of ``epsg`` and of each CRS in ``also`` into ``epsg``'s CRS,
    then snaps the bounding box outward to the shard pitch — a ``utm_zones`` column
    passes its southern CRS in ``also``. This is the authoritative derivation; the
    static zone tables are pinned against it by the unit test.
    """
    crs = CRS.from_epsg(epsg)
    to_utm = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    xs, ys = [], []
    for code in (epsg, *also):
        aou = CRS.from_epsg(code).area_of_use
        if aou is None:
            raise ValueError(f"EPSG:{code} has no area_of_use")
        for i in range(9):
            for j in range(9):
                x, y = to_utm.transform(
                    aou.west + (aou.east - aou.west) * i / 8, aou.south + (aou.north - aou.south) * j / 8
                )
                if math.isfinite(x) and math.isfinite(y):
                    xs.append(x)
                    ys.append(y)
    easting = (math.floor(min(xs) / PITCH_M) * PITCH_M, math.ceil(max(xs) / PITCH_M) * PITCH_M)
    northing = (math.floor(min(ys) / PITCH_M) * PITCH_M, math.ceil(max(ys) / PITCH_M) * PITCH_M)
    return easting, northing


# Static, pinned extent templates (derive_extent gives these for every zone;
# uniform per hemisphere because all 6° bands share geometry in their own CRS).
# The pinning test re-derives per zone and asserts equality — an EPSG-db drift
# fails CI rather than silently moving the grid.
_NORTH_EASTING: Extent = (163_840.0, 839_680.0)
_NORTH_NORTHING: Extent = (0.0, 9_338_880.0)
_SOUTH_EASTING: Extent = (163_840.0, 839_680.0)
_SOUTH_NORTHING: Extent = (1_105_920.0, 10_014_720.0)
# A utm_zones column: both areas of use in the northern CRS, the south as negative northings.
_COLUMN_NORTHING: Extent = (-8_888_320.0, 9_338_880.0)


def _build_zones() -> dict[ZoneLayout, dict[str, ZoneSpec]]:
    """Build both layouts' group tables, each keyed by group name."""
    hemispheres: dict[str, ZoneSpec] = {}
    columns: dict[str, ZoneSpec] = {}
    for utm in range(1, 61):
        for spec in (
            ZoneSpec(f"326{utm:02d}", "N", utm, _NORTH_EASTING, _NORTH_NORTHING),
            ZoneSpec(f"327{utm:02d}", "S", utm, _SOUTH_EASTING, _SOUTH_NORTHING),
        ):
            hemispheres[spec.group_name] = spec
        column = ZoneSpec(f"326{utm:02d}", None, utm, _NORTH_EASTING, _COLUMN_NORTHING)
        columns[column.group_name] = column
    return {"hemispheres": hemispheres, "utm_zones": columns}


_ZONES_BY_LAYOUT: dict[ZoneLayout, dict[str, ZoneSpec]] = _build_zones()
#: What the campaign fills: each column as its two halves, each hemisphere group whole.
_FILL_ZONES_BY_LAYOUT: dict[ZoneLayout, dict[str, ZoneSpec]] = {
    "hemispheres": _ZONES_BY_LAYOUT["hemispheres"],
    "utm_zones": {
        half.name: half
        for column in _ZONES_BY_LAYOUT["utm_zones"].values()
        for half in (dataclasses.replace(column, half="n"), dataclasses.replace(column, half="s"))
    },
}


def zones(layout: ZoneLayout) -> dict[str, ZoneSpec]:
    """Every group of *layout*, keyed by name: ``"01N"``…``"60S"`` or ``"utm01"``…``"utm60"``."""
    return _ZONES_BY_LAYOUT[layout]


def fill_zones(layout: ZoneLayout) -> dict[str, ZoneSpec]:
    """Every unit the campaign fills in *layout*, keyed by name: the groups, or each column's two halves."""
    return _FILL_ZONES_BY_LAYOUT[layout]


def zone(name: str) -> ZoneSpec:
    """Look up a group or a column half of either layout by name (``"01N"``, ``"utm33"``, ``"utm33s"``)."""
    for tables in (_ZONES_BY_LAYOUT, _FILL_ZONES_BY_LAYOUT):
        for table in tables.values():
            if name in table:
                return table[name]
    raise KeyError(name)


def group_and_half(name: str) -> tuple[str, Half | None]:
    """The store group a fill zone writes, and the column half it owns there.

    ``"utm33s"`` → ``("utm33", "s")``; any other name — a group, or a single-area store's ``""``
    — is its own group, whole. The one mapping from a fill zone to what it addresses in the store,
    where a column's two halves share one group (ADR 036).
    """
    spec = _FILL_ZONES_BY_LAYOUT["utm_zones"].get(name)
    return (spec.group_name, spec.half) if spec else (name, None)


def zone_layout_of(names: Iterable[str]) -> ZoneLayout | None:
    """The layout the zone *names* belong to, or ``None`` when none is a zone.

    How a store's existing groups decide its layout. Names that are no zone's are ignored;
    zones of BOTH layouts are refused, since one store holding both would cover the world twice.
    """
    found = {
        layout
        for name in names
        for tables in (_ZONES_BY_LAYOUT, _FILL_ZONES_BY_LAYOUT)
        for layout, table in tables.items()
        if name in table
    }
    if len(found) > 1:
        raise ValueError("zone names mix the 'hemispheres' and 'utm_zones' layouts; one store holds one layout.")
    return found.pop() if found else None


def canonicalize_zone(zone: str) -> str:
    """Normalize a UTM-zone id to its canonical zone (group) name.

    Accepts ``"33N"``, ``"15s"``, ``" 7S "``, ``"07S"`` → ``"33N"`` / ``"15S"`` /
    ``"07S"``; ``"utm33"``, ``"UTM7"`` → ``"utm33"`` / ``"utm07"``; and a column half,
    ``"UTM33S"`` → ``"utm33s"``. Zone number is
    1-60, hemisphere ``N``/``S`` and prefix case-insensitive, optional leading zero and
    surrounding whitespace. Raises ``ValueError`` on a malformed or out-of-range id.

    This is the single parser for the zone identifier used across group names,
    mosaic paths, tags, and flow params; the EPSG code stays available via
    ``zone(name).epsg`` / ``.crs``.
    """
    text = zone.strip().upper()
    m = re.fullmatch(r"(?:(UTM)(\d{1,2})([NS]?)|(\d{1,2})([NS]))", text)
    if not m:
        raise ValueError(
            f"UTM zone {zone!r} is malformed — expected 'utm<1-60>[n|s]' or '<1-60><N|S>', e.g. 'utm33' or '33N'."
        )
    number = int(m.group(2) or m.group(4))
    if not 1 <= number <= 60:
        raise ValueError(f"UTM zone number {number} out of range (1-60) in {zone!r}.")
    return f"utm{number:02d}{m.group(3).lower()}" if m.group(1) else f"{number:02d}{m.group(5)}"


def easting_coords(spec: ZoneSpec) -> np.ndarray:
    """1-D easting pixel-center coordinates (ascending), float64."""
    return spec.easting[0] + (np.arange(spec.width, dtype="float64") + 0.5) * PIXEL_M


def northing_coords(spec: ZoneSpec) -> np.ndarray:
    """1-D northing pixel-center coordinates (descending — row 0 is the top)."""
    return spec.northing[1] - (np.arange(spec.height, dtype="float64") + 0.5) * PIXEL_M


@cache
def to_wgs84(epsg: int) -> Transformer:
    """Zone-CRS -> WGS84 transformer, cached per zone."""
    return Transformer.from_crs(epsg, 4326, always_xy=True)


def tile_range_bbox_wgs84(spec: ZoneSpec, r0: int, r1: int, c0: int, c1: int) -> tuple[float, float, float, float]:
    """WGS84 ``(west, south, east, north)`` covering tile rows ``[r0, r1)`` x cols ``[c0, c1)``.

    The projected rectangle's perimeter is densified before projecting to WGS84 so meridian
    curvature cannot clip the extremes (which may fall mid-edge, not at a corner) — the WGS84
    envelope of the perimeter therefore CONTAINS the whole projected rectangle, which is what
    lets callers treat a catalogue miss against the box as a miss against everything inside it.

    Lives here, with the northing convention and ``PITCH_M`` it depends on, because two callers
    need it — the ingest's catalogue preflight and the published registry's per-tile box — and a
    second copy is how one of them silently stops describing the grid the other uses.
    """
    tile_m = PITCH_M
    e_lo = spec.easting[0] + c0 * tile_m
    e_hi = spec.easting[0] + c1 * tile_m
    n_hi = spec.northing[1] - r0 * tile_m  # top (max northing)
    n_lo = spec.northing[1] - r1 * tile_m  # bottom (min northing)

    # 64 samples per edge, chosen by measurement. Callers rely on the docstring's containment
    # guarantee — `preflight_optical_source` treats a catalogue miss against this box as a miss
    # against everything inside it — and a sampled perimeter only contains the true envelope if no
    # extremum hides between two samples. Measured worst-case under-coverage across zone edges and
    # latitudes:
    #
    #     16 -> 325 m     32 -> 63 m     64 -> 9.8 m     128 -> 0.61 m
    #
    # 64 is the first that lands inside one pixel (10 m), i.e. below the resolution anything
    # downstream can act on. The cost is ~30 us per box against ~8 us, and the campaign
    # computes at most a few hundred boxes per cell: about eight seconds across all 1,008
    # cells, which is not a number worth trading a correctness guarantee for.
    n = 64  # samples per edge
    es = np.linspace(e_lo, e_hi, n)
    ns = np.linspace(n_lo, n_hi, n)
    perim_e = np.concatenate([es, np.full(n, e_hi), es[::-1], np.full(n, e_lo)])
    perim_n = np.concatenate([np.full(n, n_hi), ns[::-1], np.full(n, n_lo), ns])
    lon, lat = to_wgs84(int(spec.epsg)).transform(perim_e, perim_n)
    lon = np.asarray(lon, dtype="float64")
    lat = np.asarray(lat, dtype="float64")
    lat_min, lat_max = float(lat.min()), float(lat.max())

    # Antimeridian: zones 01* / 60* snap slightly past +/-180, so inverse projection
    # wraps some perimeter points to the far side. Plain min/max would then span
    # nearly the whole globe. A span > 180 degrees signals the wrap; represent the crossing
    # per the GeoJSON/STAC convention (west > east) — west is the eastern (>0)
    # cluster's min, east the western (<0) cluster's max — keeping the box narrow.
    if float(lon.max() - lon.min()) > 180.0:
        west = float(lon[lon > 0.0].min())
        east = float(lon[lon < 0.0].max())
        return west, lat_min, east, lat_max
    return float(lon.min()), lat_min, float(lon.max()), lat_max


def tile_row_latitudes(spec: ZoneSpec, n_rows: int) -> np.ndarray:
    """Approximate latitude of each 2048-px tile row's centre, row 0 at the top.

    For bucketing tile rows into wide latitude bands: satellite observation counts
    vary about twofold with latitude, so anything sizing WORK rather than AREA needs
    to know roughly where a zone's live tiles sit, not just how many there are.

    Deliberately arithmetic rather than a pyproj transform. A degree of latitude is
    ~111.32 km everywhere to within 0.3%, three orders of magnitude finer than the
    20-degree bands this feeds, and it spares the caller a projection round-trip per
    tile row. Southern zones carry UTM's 10,000,000 m false northing, so their
    northings are shifted first; both hemispheres' extents confirm the convention
    (north tops out at 83.9 degrees, south bottoms at -79.9). A ``utm_zones`` column
    is in its northern CRS, whose southern northings are already negative.
    """
    northing = spec.northing[1] - (np.arange(n_rows, dtype="float64") + 0.5) * PITCH_M
    if spec.hemisphere == "S":
        northing = northing - 10_000_000.0
    return northing / 111_320.0
