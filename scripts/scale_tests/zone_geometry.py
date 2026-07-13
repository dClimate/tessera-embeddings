"""Mock UTM-zone geometry: shapes, coordinate arrays, and the annual time axis.

A scaled-down stand-in for one real UTM zone. ``tiny`` runs on a laptop; ``bench``
is the ~1/170th-area zone used for real numbers; ``full_height`` exercises the
metadata-only seeding path (T4) at realistic node/coord sizes without writing
any chunk data.
"""

from __future__ import annotations

import dataclasses

import numpy as np

#: The annual timesteps of the campaign (2025 filled first, then backwards).
YEARS: tuple[int, ...] = tuple(range(2017, 2026))

#: 10 m pixel spacing, in metres (matches the real 10 m embeddings grid).
PIXEL_M: float = 10.0
#: Arbitrary but plausible UTM origins so coord values look real.
_NORTHING_TOP_M: float = 9_000_000.0  # northing descends from here
_EASTING_LEFT_M: float = 500_000.0  # easting ascends from here


@dataclasses.dataclass(frozen=True)
class MockZone:
    """A scaled mock zone: pixel extent plus a CRS authority code."""

    height: int
    width: int
    epsg: str = "EPSG:32601"

    @property
    def shape2d(self) -> tuple[int, int]:
        """(height, width) in pixels."""
        return (self.height, self.width)


def zone_for(scale: str, *, full_height: bool = False) -> MockZone:
    """Return the mock zone for a run scale.

    Args:
        scale: ``"tiny"`` (laptop) or ``"bench"`` (real numbers).
        full_height: When True, return the tall, narrow metadata-only geometry
            used by T4 (never fully written), regardless of ``scale``.
    """
    if full_height:
        return MockZone(height=1_000_000, width=4_096)
    if scale == "tiny":
        return MockZone(height=2_048, width=2_048)
    return MockZone(height=20_000, width=20_000)


def times(years: tuple[int, ...] = YEARS) -> np.ndarray:
    """Return one ``datetime64[ns]`` timestamp per year (Jan 1), ascending."""
    return np.array([np.datetime64(f"{y}-01-01", "ns") for y in years])


def coords(zone: MockZone, years: tuple[int, ...] = YEARS) -> dict[str, np.ndarray]:
    """Return the ``time``/``northing``/``easting`` coordinate arrays for a zone.

    ``northing`` descends and ``easting`` ascends (real UTM convention), both
    float64 at 10 m spacing — matching what ``resolve_region`` expects.
    """
    northing = _NORTHING_TOP_M - np.arange(zone.height, dtype="float64") * PIXEL_M
    easting = _EASTING_LEFT_M + np.arange(zone.width, dtype="float64") * PIXEL_M
    return {"time": times(years), "northing": northing, "easting": easting}
