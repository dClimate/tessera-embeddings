"""The 120 UTM-zone grid for the global embeddings store (ADR-008).

The world is one Icechunk repo of 120 Zarr groups — 60 UTM zones x 2
hemispheres — each in its own CRS (EPSG:326xx north, EPSG:327xx south) and
named by its UTM **common name** (``"01N"`` … ``"60S"``: zero-padded zone
number + hemisphere). The EPSG code is retained as the CRS only
(:attr:`ZoneSpec.epsg` / :attr:`ZoneSpec.crs`). This is a deliberate deviation
from the geoembeddings ``utm_zones`` spec, whose ``utm{NN}`` group name cannot
express the hemisphere.

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
same geometry in its own CRS, all northern zones share one extent template and
all southern zones another — but :func:`derive_extent` and the pinning test
(``tests/unit/test_zone_grid.py``) re-derive per zone from pyproj so an
EPSG-database change can't silently move the grid.
"""

from __future__ import annotations

import dataclasses
import math
import re

import numpy as np
from pyproj import CRS, Transformer

from tessera_embeddings.config.store_layout import SHARD_PX

#: Ground sample distance in metres (10 m embeddings grid).
PIXEL_M: float = 10.0
#: Shard pitch in metres — extents snap to this so the grid is shard-aligned.
PITCH_M: float = SHARD_PX * PIXEL_M  # 20 480 m

#: The campaign's annual timesteps (2025 filled first, then backwards).
CAMPAIGN_YEARS: tuple[int, ...] = tuple(range(2017, 2026))

#: Machine-readable marker of the boundary policy, for dataset attrs.
ZONE_SCHEME: str = "utm_6deg_nominal"

Extent = tuple[float, float]  # (low, high) in projected metres


@dataclasses.dataclass(frozen=True)
class ZoneSpec:
    """One UTM zone: CRS, hemisphere, and shard-aligned projected extents."""

    epsg: str  # e.g. "32601"
    hemisphere: str  # "N" | "S"
    utm_zone: int  # 1..60
    easting: Extent
    northing: Extent

    @property
    def group_name(self) -> str:
        """Zarr group name for this zone: its UTM common name, e.g. ``"01N"``/``"60S"``."""
        return f"{self.utm_zone:02d}{self.hemisphere}"

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


def derive_extent(epsg: int) -> tuple[Extent, Extent]:
    """Derive a zone's shard-aligned ``(easting, northing)`` extent from pyproj.

    Projects the CRS's EPSG area-of-use (WGS84 lon/lat box, densely sampled to
    capture meridian curvature) into the zone CRS, then snaps the bounding box
    outward to the shard pitch. This is the authoritative derivation; the static
    :data:`ZONES` table is pinned against it by the unit test.
    """
    crs = CRS.from_epsg(epsg)
    aou = crs.area_of_use
    if aou is None:
        raise ValueError(f"EPSG:{epsg} has no area_of_use")
    to_utm = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    lons = [aou.west + (aou.east - aou.west) * i / 8 for i in range(9)]
    lats = [aou.south + (aou.north - aou.south) * j / 8 for j in range(9)]
    xs, ys = [], []
    for lon in lons:
        for lat in lats:
            x, y = to_utm.transform(lon, lat)
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


def _build_zones() -> dict[str, ZoneSpec]:
    """Build the 120-zone registry, keyed by UTM common name (``"01N"``…``"60S"``)."""
    zones: dict[str, ZoneSpec] = {}
    for utm in range(1, 61):
        north = ZoneSpec(f"326{utm:02d}", "N", utm, _NORTH_EASTING, _NORTH_NORTHING)
        south = ZoneSpec(f"327{utm:02d}", "S", utm, _SOUTH_EASTING, _SOUTH_NORTHING)
        zones[north.group_name] = north
        zones[south.group_name] = south
    return zones


#: The 120 zones, keyed by UTM common name (``"01N"``…``"60S"``).
ZONES: dict[str, ZoneSpec] = _build_zones()


def zone(name: str) -> ZoneSpec:
    """Look up a zone by its UTM common name (e.g. ``"01N"``, ``"60S"``)."""
    return ZONES[name]


def canonicalize_zone(zone: str) -> str:
    """Normalize a UTM-zone id to the canonical zero-padded common name.

    Accepts ``"33N"``, ``"15s"``, ``" 7S "``, ``"07S"`` → ``"33N"`` / ``"15S"`` /
    ``"07S"``. Zone number is 1-60, hemisphere ``N``/``S`` (case-insensitive),
    optional leading zero and surrounding whitespace. Raises ``ValueError`` on a
    malformed or out-of-range id.

    This is the single parser for the zone identifier used across group names,
    mosaic paths, tags, and flow params; the EPSG code stays available via
    ``ZONES[name].epsg`` / ``.crs``.
    """
    text = zone.strip().upper()
    m = re.fullmatch(r"(\d{1,2})([NS])", text)
    if not m:
        raise ValueError(f"UTM zone {zone!r} is malformed — expected '<1-60><N|S>', e.g. '33N' or '15S'.")
    number = int(m.group(1))
    if not 1 <= number <= 60:
        raise ValueError(f"UTM zone number {number} out of range (1-60) in {zone!r}.")
    return f"{number:02d}{m.group(2)}"


def easting_coords(spec: ZoneSpec) -> np.ndarray:
    """1-D easting pixel-center coordinates (ascending), float64."""
    return spec.easting[0] + (np.arange(spec.width, dtype="float64") + 0.5) * PIXEL_M


def northing_coords(spec: ZoneSpec) -> np.ndarray:
    """1-D northing pixel-center coordinates (descending — row 0 is the top)."""
    return spec.northing[1] - (np.arange(spec.height, dtype="float64") + 0.5) * PIXEL_M


def tile_row_latitudes(spec: ZoneSpec, n_rows: int) -> np.ndarray:
    """Approximate latitude of each 2048-px tile row's centre, row 0 at the top.

    For bucketing tile rows into wide latitude bands: satellite observation counts
    vary about twofold with latitude, so anything sizing WORK rather than AREA needs
    to know roughly where a zone's live tiles sit, not just how many there are.

    Deliberately arithmetic rather than a pyproj transform. A degree of latitude is
    ~111.32 km everywhere to within 0.3%, three orders of magnitude finer than the
    20-degree bands this feeds, and it keeps the caller free of a projection
    round-trip per tile row. Southern zones carry UTM's 10,000,000 m false northing,
    so their northings are shifted first; both hemispheres' extents confirm the
    convention (north tops out at 83.9 degrees, south bottoms at -79.9).
    """
    northing = spec.northing[1] - (np.arange(n_rows, dtype="float64") + 0.5) * PITCH_M
    if spec.hemisphere == "S":
        northing = northing - 10_000_000.0
    return northing / 111_320.0


def year_timestamp(year: int) -> np.datetime64:
    """The Q2 calendar-year convention, encoded once: ``year`` → ``YYYY-01-01`` ns."""
    return np.datetime64(f"{year}-01-01", "ns")


def year_of(value: np.datetime64 | np.ndarray) -> int:
    """Inverse of :func:`year_timestamp`: a (scalar) timestamp's calendar year."""
    return int(np.asarray(value).astype("datetime64[Y]").astype(int)) + 1970


def calendar_year_times(years: tuple[int, ...] = CAMPAIGN_YEARS) -> np.ndarray:
    """Return one ``datetime64[ns]`` per year at ``YYYY-01-01`` (Q2 convention)."""
    return np.array([year_timestamp(y) for y in years])
