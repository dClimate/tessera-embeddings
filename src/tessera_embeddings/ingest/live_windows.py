"""Derive grid-aligned live windows from an ROI mask.

Ingest cost scales with the extent it computes, not the land it keeps, so the
mosaic loads are restricted to windows that actually intersect the ROI
(``context_docs/design/ingest-live-tile-cropping.md``). This module is the pure
geometry half: read the boolean ROI mask (a bare zarr array on the fixed grid —
the artifact both ``rasterize_roi_zarr`` and ``export_zone_roi`` write), coarsen
it to the ingest chunk grid, and emit one window per live chunk-row spanning that
row's live columns.

This module is general-purpose: it serves a SINGLE run (any sparse ROI — scattered
fields, a coastline, any footprint much smaller than its bounding box) and a
GLOBAL campaign zone identically, because both produce the same mask artifact.
The strategy choice was measured on the GLOBAL campaign's coverage specifically
(all 112 land zones; table and per-zone JSON in the design note): row bands
compute within 1% (median) of the exact live-chunk floor, while a single bounding
box captures less than half the win. A single ROI sees the same behaviour scaled
to its own extent-vs-content gap.

Windows are snapped to the chunk grid by construction, which makes them
chunk-disjoint — the property that lets one session write every window of a date
and commit once, with no shared-chunk reconciliation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import zarr

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE


@dataclass(frozen=True)
class LiveWindow:
    """One chunk-aligned pixel window: ``[y0, y1) x [x0, x1)`` on the mask grid.

    ``y0``/``x0`` are chunk-grid multiples; ``y1``/``x1`` are clamped to the mask
    extent (the last chunk of an axis may be partial).
    """

    y0: int
    y1: int
    x0: int
    x1: int


def _open_mask(mask_path: str) -> zarr.Array:
    """Open + validate the ROI mask array (the shape both mask writers produce)."""
    z = zarr.open(mask_path, mode="r")
    if not isinstance(z, zarr.Array) or z.ndim != 2 or z.dtype != np.bool_:
        raise ValueError(f"ROI mask at {mask_path} is not a 2-D boolean zarr array")
    return z


def live_chunk_grid(mask_path: str, *, chunk_px: int = INGEST_CHUNK_SIZE) -> np.ndarray:
    """Coarsen the ROI mask onto the ingest chunk grid: True where any pixel is live.

    Reads the mask row-band by row-band (one chunk-row of pixels at a time) rather
    than whole — a zone mask decompresses to tens of GB — or per-chunk — thousands
    of object reads. Plain zarr, no dask: this is a metadata-scale scan that must
    not cost a task graph.
    """
    z = _open_mask(mask_path)
    height, width = z.shape
    rows, cols = math.ceil(height / chunk_px), math.ceil(width / chunk_px)
    live = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        band = np.asarray(z[r * chunk_px : min((r + 1) * chunk_px, height), :])
        padded = np.zeros((band.shape[0], cols * chunk_px), dtype=bool)
        padded[:, :width] = band
        live[r] = padded.reshape(band.shape[0], cols, chunk_px).any(axis=(0, 2))
    return live


def row_band_windows(
    live: np.ndarray, *, height: int, width: int, chunk_px: int = INGEST_CHUNK_SIZE
) -> list[LiveWindow]:
    """One window per live chunk-row, spanning that row's first..last live column.

    Windows are chunk-aligned (hence mutually chunk-disjoint) and clamped to the
    ``height x width`` pixel extent. An all-ocean grid yields no windows.
    """
    if live.shape != (math.ceil(height / chunk_px), math.ceil(width / chunk_px)):
        raise ValueError(f"live grid {live.shape} does not match extent {height}x{width} at {chunk_px}px chunks")
    windows: list[LiveWindow] = []
    for r in np.flatnonzero(live.any(axis=1)):
        cols = np.flatnonzero(live[r])
        windows.append(
            LiveWindow(
                y0=int(r) * chunk_px,
                y1=min((int(r) + 1) * chunk_px, height),
                x0=int(cols[0]) * chunk_px,
                x1=min((int(cols[-1]) + 1) * chunk_px, width),
            )
        )
    return windows


def live_windows_for_mask(mask_path: str, *, chunk_px: int = INGEST_CHUNK_SIZE) -> list[LiveWindow]:
    """The one-call form: mask store → row-band live windows."""
    height, width = _open_mask(mask_path).shape
    return row_band_windows(
        live_chunk_grid(mask_path, chunk_px=chunk_px), height=height, width=width, chunk_px=chunk_px
    )
