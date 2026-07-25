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

The grid itself is normally read from the mask's chunk KEYS rather than its pixels:
an all-ocean chunk is never written, so the set of stored chunks IS the set of live
cells, and one listing replaces a read per chunk position. The pixel scan remains as
the fallback for any store whose layout is not positively recognised.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import fsspec
import numpy as np
import zarr

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE

logger = logging.getLogger(__name__)

#: Zarr v3's default chunk key encoding for a 2-D array: ``c/<row>/<col>`` under the
#: array root. Anchored at the end so a store nested beneath a longer prefix still
#: matches, while a key of different rank (``c/1/2/3``) deliberately does not.
_CHUNK_KEY_RE = re.compile(r"(?:^|/)c/(\d+)/(\d+)$")


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


def live_chunk_grid_from_keys(
    mask_path: str, mask: zarr.Array, *, chunk_px: int = INGEST_CHUNK_SIZE
) -> np.ndarray | None:
    """The live-chunk grid read from the mask's stored chunk KEYS, not its pixels.

    An all-ocean chunk is never written — the campaign writer skips it explicitly
    (``export_zone_roi``), and the single-ROI writer gets the same result because an
    all-False boolean chunk equals the fill value and zarr does not write all-fill
    chunks. So a chunk object exists exactly where the ROI has pixels, and one
    listing yields what :func:`live_chunk_grid` otherwise derives with a read per
    chunk position (a UTM zone has ~3,700 of them, ~2 minutes in-region).

    Errs only toward MORE work. A chunk that was written but holds no live pixel
    would be reported live, widening one window; the worst case is the full-extent
    behaviour we already had. It cannot under-report, because a chunk containing a
    live pixel must exist as an object — which is the property that makes this safe
    to prefer over reading.

    Returns ``None`` — meaning "fall back to the block scan" — for any store whose
    layout this does not positively recognise: sharded, non-default chunk key
    encoding, chunking that does not match ``chunk_px``, an index outside the
    grid, or a listing that fails. Never guesses, because a grid derived from a
    misread layout would crop to the wrong windows and silently drop land.
    """
    if mask.chunks != (chunk_px, chunk_px):
        # Keys would not map 1:1 onto the grid being built.
        logger.debug("mask chunks %s != %d; using the block scan", mask.chunks, chunk_px)
        return None
    if getattr(mask, "shards", None) is not None:
        # A shard object holds many chunks, so its key is not a chunk index.
        logger.debug("mask is sharded; using the block scan")
        return None

    height, width = mask.shape
    rows, cols = math.ceil(height / chunk_px), math.ceil(width / chunk_px)
    try:
        fs, root = fsspec.core.url_to_fs(mask_path)
        keys = fs.find(root)
    except (OSError, ValueError) as exc:
        logger.debug("cannot list %s (%s); using the block scan", mask_path, exc)
        return None

    live = np.zeros((rows, cols), dtype=bool)
    matched = 0
    for key in keys:
        m = _CHUNK_KEY_RE.search(key)
        if m is None:
            continue  # zarr.json and anything else that is not a chunk
        r, c = int(m.group(1)), int(m.group(2))
        if r >= rows or c >= cols:
            logger.debug("chunk key %s outside the %dx%d grid; using the block scan", key, rows, cols)
            return None
        live[r, c] = True
        matched += 1

    # No chunks at all is a legitimate answer (an ROI with no live pixels), not a
    # failed listing: the listing succeeded and found only metadata.
    logger.info("Derived %d live chunk(s) of %d from the mask's keys", matched, rows * cols)
    return live


def live_chunk_grid(mask_path: str, *, chunk_px: int = INGEST_CHUNK_SIZE) -> np.ndarray:
    """Coarsen the ROI mask onto the ingest chunk grid: True where any pixel is live.

    Reads one chunk-sized block at a time (~16 MB at the 4096 default), reducing
    each straight into its ``live`` cell — never a whole row band (a full-width
    band of a wide zone is hundreds of MB, and an arbitrary single-ROI width is
    unbounded) and never the whole mask (tens of GB decompressed). The block
    reads hit the same underlying zarr chunk objects a wider read would — zarr
    fetches per chunk either way — so this bounds memory without extra I/O.
    Plain zarr, no dask: a metadata-scale scan must not cost a task graph.
    """
    z = _open_mask(mask_path)
    height, width = z.shape
    rows, cols = math.ceil(height / chunk_px), math.ceil(width / chunk_px)
    live = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        y = slice(r * chunk_px, min((r + 1) * chunk_px, height))
        for c in range(cols):
            block = z[y, c * chunk_px : min((c + 1) * chunk_px, width)]
            live[r, c] = bool(np.asarray(block).any())
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


def live_windows_for_mask(
    mask_path: str, *, chunk_px: int = INGEST_CHUNK_SIZE, prefer_keys: bool = True
) -> list[LiveWindow]:
    """The one-call form: mask store → row-band live windows.

    Derives the live-chunk grid from the mask's chunk keys when the store's layout
    is recognised (:func:`live_chunk_grid_from_keys` — one listing) and falls back
    to reading every chunk position (:func:`live_chunk_grid`) when it is not. Both
    routes return the same grid; the difference is one listing against thousands of
    sequential reads.

    ``prefer_keys=False`` forces the read path, which is how the two are held to
    the same answer in tests.
    """
    mask = _open_mask(mask_path)
    height, width = mask.shape
    live = live_chunk_grid_from_keys(mask_path, mask, chunk_px=chunk_px) if prefer_keys else None
    if live is None:
        live = live_chunk_grid(mask_path, chunk_px=chunk_px)
    return row_band_windows(live, height=height, width=width, chunk_px=chunk_px)
