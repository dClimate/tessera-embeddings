"""The 120 UTM-zone grid for the global embeddings store (ADR-008).

The world is one Icechunk repo of 120 Zarr groups — 60 UTM zones x 2
hemispheres — each in its own CRS (EPSG:326xx north, EPSG:327xx south) and
named by its EPSG code string (``"32601"`` … ``"32760"``).

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
        """Zarr group name for this zone (its EPSG code string)."""
        return self.epsg

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
    """Build the 120-zone registry from the pinned per-hemisphere templates."""
    zones: dict[str, ZoneSpec] = {}
    for utm in range(1, 61):
        n_epsg = f"326{utm:02d}"
        s_epsg = f"327{utm:02d}"
        zones[n_epsg] = ZoneSpec(n_epsg, "N", utm, _NORTH_EASTING, _NORTH_NORTHING)
        zones[s_epsg] = ZoneSpec(s_epsg, "S", utm, _SOUTH_EASTING, _SOUTH_NORTHING)
    return zones


#: The 120 zones, keyed by EPSG code string.
ZONES: dict[str, ZoneSpec] = _build_zones()


def zone(epsg: str) -> ZoneSpec:
    """Look up a zone by EPSG code string (e.g. ``"32601"``)."""
    return ZONES[epsg]


def easting_coords(spec: ZoneSpec) -> np.ndarray:
    """1-D easting pixel-center coordinates (ascending), float64."""
    return spec.easting[0] + (np.arange(spec.width, dtype="float64") + 0.5) * PIXEL_M


def northing_coords(spec: ZoneSpec) -> np.ndarray:
    """1-D northing pixel-center coordinates (descending — row 0 is the top)."""
    return spec.northing[1] - (np.arange(spec.height, dtype="float64") + 0.5) * PIXEL_M


def calendar_year_times(years: tuple[int, ...] = CAMPAIGN_YEARS) -> np.ndarray:
    """Return one ``datetime64[ns]`` per year at ``YYYY-01-01`` (Q2 convention)."""
    return np.array([np.datetime64(f"{y}-01-01", "ns") for y in years])
