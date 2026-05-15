"""Reference grid definitions for coarsening embedding stores.

Reads a reference grid definition (CRS, origin, step) from an existing zarr
store.  The actual regridding logic lives in :mod:`src.inference.coarsen`.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import zarr

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ReferenceGrid:
    """Spatial grid definition extracted from a reference zarr store.

    Defines an infinite regular grid by its CRS, origin (first coordinate
    value), and step size along each axis.
    """

    crs: str
    """EPSG authority code, e.g. ``"EPSG:5070"``."""

    origin_x: float
    """First x (easting) coordinate value (pixel-center)."""

    origin_y: float
    """First y (northing) coordinate value (pixel-center)."""

    step_x: float
    """X spacing between pixel centers (positive = eastward)."""

    step_y: float
    """Y spacing between pixel centers (negative = southward)."""


def read_reference_grid(zarr_path: str) -> ReferenceGrid:
    """Read a reference grid definition from a zarr store.

    Extracts CRS from the ``proj:code`` root attribute and derives origin/step
    from the ``x`` and ``y`` coordinate arrays.

    Args:
        zarr_path: Path to a zarr store (local or ``s3://``).

    Returns:
        A :class:`ReferenceGrid` describing the store's spatial grid.

    Raises:
        ValueError: If ``proj:code`` is missing or coordinate arrays are
            too short to derive a step.
    """
    root = zarr.open_group(zarr_path, mode="r")

    attrs = dict(root.attrs)
    crs = attrs.get("proj:code")
    if not crs:
        msg = f"Reference zarr at {zarr_path} has no 'proj:code' attribute"
        raise ValueError(msg)

    x = root["x"][:]
    y = root["y"][:]
    if len(x) < 2 or len(y) < 2:
        msg = f"Reference zarr coordinate arrays too short: x={len(x)}, y={len(y)}"
        raise ValueError(msg)

    step_x = float(np.median(np.diff(x)))
    step_y = float(np.median(np.diff(y)))

    return ReferenceGrid(
        crs=str(crs),
        origin_x=float(x[0]),
        origin_y=float(y[0]),
        step_x=step_x,
        step_y=step_y,
    )
