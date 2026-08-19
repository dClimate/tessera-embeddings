"""Derive grid-aligned live windows from an ROI mask.

Ingest cost scales with the extent it computes, not the land it keeps, so the
mosaic loads are restricted to windows that actually intersect the ROI
(``context_docs/design/ingest-live-tile-cropping.md``). This module is the pure
geometry half, in two stages: coarsen the boolean ROI mask (a bare zarr array on
the fixed grid — the artifact both ``rasterize_roi_zarr`` and ``export_zone_roi``
write) to the ingest chunk grid and emit one window per live chunk-row
(:func:`row_band_windows`), then group vertically adjacent bands into fewer, taller
windows (:func:`merge_bands`).

The second stage exists because the two costs a window strategy trades are nothing
alike. Chunk area is computed in PARALLEL across the fleet, so more of it widens
the graph — which is what the fleet wants. A window BOUNDARY is a serial, blocking
region write the whole fleet waits through. Minimising area alone therefore buys
the cheap thing and pays for the expensive one; grouping does the reverse.

This module is general-purpose: it serves a SINGLE run (any sparse ROI — scattered
fields, a coastline, any footprint much smaller than its bounding box) and a GLOBAL
campaign zone identically, because both produce the same mask artifact.

Windows are snapped to the chunk grid by construction, which makes them
chunk-disjoint — the property that lets one session write every window of a date
and commit once, with no shared-chunk reconciliation. Grouping preserves it, since
it only unions adjacent chunk-row ranges.

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
from typing import cast

import fsspec
import numpy as np
import zarr

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE
from tessera_embeddings.ingest.roi import StorageOptions, resolve_storage_options

logger = logging.getLogger(__name__)

#: Zarr v3's default chunk key encoding for a 2-D array: ``c/<row>/<col>`` under the
#: array root. Anchored at the end so a store nested beneath a longer prefix still
#: matches, while a key of different rank (``c/1/2/3``) deliberately does not.
_CHUNK_KEY_RE = re.compile(r"(?:^|/)c/(\d+)/(\d+)$")

#: Cap on the graph a single region write may build, in TASKS — the unit that
#: actually saturates, because the scheduler dispatches tasks and its event loop is
#: single-threaded. Denominated in tasks rather than chunk area because the tasks a
#: chunk costs depends on band count and block geometry, so a chunk-denominated cap
#: silently changes meaning whenever either moves. See :func:`merge_bands`.
MAX_TASKS_PER_WINDOW = 24_000

#: Graph tasks one window chunk is assumed to cost, for turning the task cap into a
#: chunk area. Deliberately conservative — the cap should bind slightly early rather
#: than late — and overridable by callers that have measured their own geometry.
DEFAULT_TASKS_PER_CHUNK = 200

#: What one window write costs when each window is its own SEQUENTIAL blocking region
#: write, expressed as the chunk area that costs the same. Extra computed area is
#: parallel and cheap; a serial window boundary is not. Calibration in
#: ``context_docs/design/ingest-live-tile-cropping.md``. Still the right value for the
#: sequential write path — S1 is on it (``overlap_window_writes`` defaults False there).
WINDOW_COST_IN_CHUNKS = 200

#: The same exchange rate once a date's windows share ONE dask graph
#: (``overlap_window_writes``), which is what makes a window boundary cheap rather than
#: a serial stall. A window then costs a client-side subgraph, one leaf in the merge
#: reduction and one changeset — order 15 chunks against the 200 a serial write cost.
#:
#: Lower is not free: paying less per window makes the DP merge less, so it stops
#: dragging ocean into ragged-edge merges (the dead area this recovers) but issues more
#: windows. Measured over all 112 zone masks, 200 -> 20 cuts covered area 6.0% for 14.4%
#: more windows, and total submitted tasks FALL 6% because area dominates. The knee is
#: 10-20: 200->50 buys 74 chunks per added window, 50->20 buys 28, 20->10 buys 13.5.
#: Numbers and the per-zone spread in
#: ``yield-embeddings/context_docs/measurements/`` (2026-07-27 sweep).
WINDOW_COST_IN_CHUNKS_OVERLAPPED = 20


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


def _open_mask(mask_path: str, storage_options: StorageOptions = None) -> zarr.Array:
    """Open + validate the ROI mask array (the shape both mask writers produce).

    ``storage_options`` mirrors :func:`ingest.roi.read_roi_mask` in both senses: a
    deployment whose mask needs non-default fsspec/S3 settings must pass them here too,
    or window derivation fails where the ingest's own mask read succeeds — and a callable
    is resolved here, at the read, so a credential cannot be older than this call.
    """
    z = zarr.open(mask_path, mode="r", storage_options=resolve_storage_options(storage_options))
    if not isinstance(z, zarr.Array) or z.ndim != 2 or z.dtype != np.bool_:
        raise ValueError(f"ROI mask at {mask_path} is not a 2-D boolean zarr array")
    return z


def live_chunk_grid_from_keys(
    mask_path: str,
    mask: zarr.Array,
    *,
    chunk_px: int = INGEST_CHUNK_SIZE,
    storage_options: StorageOptions = None,
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
    # POSITIVELY recognise the key layout _CHUNK_KEY_RE assumes: zarr v3 with the
    # default "c/<row>/<col>" encoding. Without this the regex simply matches
    # nothing on a v2 mask (keys like "0.0") or a custom separator, and "no chunk
    # keys found" is indistinguishable from "no live pixels" — so cropped ingest
    # would derive zero windows and write an EMPTY mosaic while reporting success.
    # A caller-supplied single-ROI mask is the realistic v2 source.
    encoding = getattr(mask.metadata, "chunk_key_encoding", None)
    if type(encoding).__name__ != "DefaultChunkKeyEncoding" or getattr(encoding, "separator", None) != "/":
        logger.debug("mask chunk-key encoding %r is not the v3 default; using the block scan", encoding)
        return None

    height, width = mask.shape
    rows, cols = math.ceil(height / chunk_px), math.ceil(width / chunk_px)
    try:
        fs, root = fsspec.core.url_to_fs(mask_path, **(resolve_storage_options(storage_options) or {}))
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

    # No chunks at all is a legitimate answer (an ROI with no live pixels) ONLY if
    # the listing genuinely found nothing but metadata. If it returned other keys
    # that this parser did not recognise, the layout is not what we think it is —
    # fall back rather than report an empty ROI and silently skip every pixel.
    if matched == 0 and any(not key.endswith(("zarr.json", ".zarray", ".zattrs", ".zgroup")) for key in keys):
        logger.debug("listing found unrecognised keys and no chunk keys; using the block scan")
        return None
    logger.info("Derived %d live chunk(s) of %d from the mask's keys", matched, rows * cols)
    return live


def live_chunk_grid(
    mask_path: str, *, chunk_px: int = INGEST_CHUNK_SIZE, storage_options: StorageOptions = None
) -> np.ndarray:
    """Coarsen the ROI mask onto the ingest chunk grid: True where any pixel is live.

    Reads one chunk-sized block at a time (~16 MB at the 4096 default), reducing
    each straight into its ``live`` cell — never a whole row band (a full-width
    band of a wide zone is hundreds of MB, and an arbitrary single-ROI width is
    unbounded) and never the whole mask (tens of GB decompressed). The block
    reads hit the same underlying zarr chunk objects a wider read would — zarr
    fetches per chunk either way — so this bounds memory without extra I/O.
    Plain zarr, no dask: a metadata-scale scan must not cost a task graph.
    """
    z = _open_mask(mask_path, storage_options)
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


def _chunk_area(w: LiveWindow, chunk_px: int) -> int:
    """A window's area in whole chunks (the unit both merge bounds are counted in)."""
    return math.ceil((w.y1 - w.y0) / chunk_px) * math.ceil((w.x1 - w.x0) / chunk_px)


def _union(windows: list[LiveWindow]) -> LiveWindow:
    """The bounding window of a group — still chunk-aligned, since its inputs are."""
    return LiveWindow(
        y0=min(w.y0 for w in windows),
        y1=max(w.y1 for w in windows),
        x0=min(w.x0 for w in windows),
        x1=max(w.x1 for w in windows),
    )


def merge_bands(
    windows: list[LiveWindow],
    *,
    chunk_px: int = INGEST_CHUNK_SIZE,
    max_tasks_per_window: int = MAX_TASKS_PER_WINDOW,
    tasks_per_chunk: int = DEFAULT_TASKS_PER_CHUNK,
    window_cost_in_chunks: int = WINDOW_COST_IN_CHUNKS,
) -> list[LiveWindow]:
    """Group vertically-adjacent row bands to minimise total ingest cost.

    One window per chunk-row computes the least *area*, but area is not what a
    windowed ingest is billed for. Each window is a separate BLOCKING region write,
    so a date costs about ``n_windows x F + chunk_area x V`` — and the two are
    wildly different in scale, because area is computed in parallel across the
    fleet while a window boundary is a serial stall the whole fleet waits through.
    ``window_cost_in_chunks`` is that exchange rate: the chunk area worth as much
    as one saved write. It is large, so grouping is usually worth it, and a
    strategy that minimises area alone leaves most of the win unclaimed.

    This therefore minimises ``n_windows x window_cost_in_chunks + total_area``
    EXACTLY, by dynamic programming over groupings of CONSECUTIVE bands — O(n^2)
    in a zone's live chunk-rows, a few hundred at most. Optimising the true
    objective matters more than it sounds: a heuristic bound on wasted area cannot
    express "extra area is nearly free", so it under-merges precisely on the sparse
    ROIs where the absolute waste is trivial.

    ``max_tasks_per_window`` bounds one window's GRAPH, converted to a chunk area
    through ``tasks_per_chunk``. Tasks are the right unit: the scheduler dispatches
    tasks on a single-threaded event loop, and past its throughput extra area stops
    being cheap — which is how an unbounded objective over-merges into a saturated
    scheduler and gets slower. A single row band over the cap is still emitted:
    being one chunk-row, it cannot be split.

    Grouping only unions adjacent chunk-row ranges, so windows stay chunk-aligned
    and mutually chunk-disjoint — the property that lets one session write a whole
    date and commit once.

    Calibration, the campaign-wide effect, and the cap sweep are in
    ``context_docs/design/ingest-live-tile-cropping.md``.
    """
    n = len(windows)
    if n == 0:
        return []
    max_chunks = max(1, max_tasks_per_window // max(1, tasks_per_chunk))

    # best[i] = min cost of covering the first i bands; cut[i] = where its last
    # group starts. Costs are in CHUNK-EQUIVALENTS, so no seconds are needed —
    # only the ratio between a window and a chunk, which is what we measured.
    best = [0] + [math.inf] * n
    cut = [0] * (n + 1)
    for i in range(1, n + 1):
        # Walk j DOWNWARD so the group grows: windows[j:i] is one band at j=i-1 and
        # the whole prefix at j=0. Once it exceeds the cap every smaller j is worse,
        # so we can stop — but a lone band over the cap is still emitted, since a
        # single chunk-row cannot be split.
        for j in range(i - 1, -1, -1):
            group = _union(windows[j:i])
            area = _chunk_area(group, chunk_px)
            if area > max_chunks and i - j > 1:
                break
            candidate = best[j] + window_cost_in_chunks + area
            if candidate < best[i]:
                best[i], cut[i] = candidate, j

    out: list[LiveWindow] = []
    i = n
    while i > 0:
        j = cut[i]
        out.append(_union(windows[j:i]))
        i = j
    out.reverse()
    return out


def coarsen_live_grid(live: np.ndarray, factor: int) -> np.ndarray:
    """Coarsen a live-chunk grid by ``factor``: a coarse cell is live if any fine one is.

    Lets windows be derived on a COARSER grid than the mask is chunked at, which is
    what keeps them aligned to the ingest's load blocks when those are a multiple of
    the store's chunks. Deriving on the fine grid and snapping outward afterwards
    would be wrong: two windows on adjacent fine rows can snap into the same coarse
    block and stop being chunk-disjoint, which the single-session write requires.
    """
    if factor == 1:
        return live
    rows, cols = live.shape
    r_pad, c_pad = math.ceil(rows / factor) * factor, math.ceil(cols / factor) * factor
    padded = np.zeros((r_pad, c_pad), dtype=bool)
    padded[:rows, :cols] = live
    # cast: np.any is typed as possibly-scalar, but a tuple axis over a 4-D array
    # always yields a 2-D array.
    return cast("np.ndarray", padded.reshape(r_pad // factor, factor, c_pad // factor, factor).any(axis=(1, 3)))


#: Window cells added around every reprojected footprint, on all four sides.
#:
#: The footprint comes from reprojecting each item's lon/lat bounding box and taking
#: the result's bounds, which for a curved reprojection can fall marginally inside
#: the true image. Under-covering would silently drop data, so the footprint is
#: padded until that is impossible: one window cell is tens of kilometres, orders of
#: magnitude more than the curvature error of a scene-sized box. The cost is a little
#: extra computed area, which is the cheap direction.
FOOTPRINT_PAD_CELLS = 1


def grid_from_windows(
    windows: list[LiveWindow], *, height: int, width: int, chunk_px: int = INGEST_CHUNK_SIZE
) -> np.ndarray:
    """The boolean cell grid covered by ``windows`` — the inverse of :func:`row_band_windows`.

    Lets a window list be intersected with another grid and re-derived, without
    re-listing the mask it originally came from.
    """
    grid = np.zeros((math.ceil(height / chunk_px), math.ceil(width / chunk_px)), dtype=bool)
    for w in windows:
        grid[w.y0 // chunk_px : math.ceil(w.y1 / chunk_px), w.x0 // chunk_px : math.ceil(w.x1 / chunk_px)] = True
    return grid


def footprint_grid(
    bboxes: list[tuple[float, float, float, float]],
    geobox: object,
    *,
    chunk_px: int = INGEST_CHUNK_SIZE,
) -> np.ndarray | None:
    """Cells a set of lon/lat bounding boxes could put data in, or ``None`` if unsure.

    ``bboxes`` are ``(west, south, east, north)`` in EPSG:4326 — the form STAC
    publishes — and ``geobox`` is the grid the ingest writes on. Only geometry
    matters here, so callers pass bounding boxes rather than catalog items and this
    module stays free of catalog types.

    **Returning ``None`` means "assume everything"**, and every uncertain path takes
    it: an unusable bounding box, a geobox that cannot be read, a projection that
    fails. The caller then behaves exactly as it did before this function existed.
    That asymmetry is the whole safety argument — a footprint that is too LARGE only
    costs computed area that was going to be discarded anyway, while one that is too
    SMALL silently drops imagery from a mosaic and nothing downstream would notice.
    Every rounding here goes outward for the same reason, and
    :data:`FOOTPRINT_PAD_CELLS` widens the result again on top.

    An empty result (no cell set) is meaningful and distinct from ``None``: those
    items fall entirely outside this grid.
    """
    try:
        from odc.geo.geom import box

        height, width = int(geobox.height), int(geobox.width)  # type: ignore[attr-defined]
        inverse = ~geobox.transform  # type: ignore[attr-defined]
        crs = geobox.crs  # type: ignore[attr-defined]
    except Exception:  # any geobox we cannot read means "assume everything"
        logger.debug("footprint: unusable geobox; computing the full extent", exc_info=True)
        return None

    rows, cols = math.ceil(height / chunk_px), math.ceil(width / chunk_px)
    grid = np.zeros((rows, cols), dtype=bool)

    for bbox in bboxes:
        try:
            west, south, east, north = (float(v) for v in bbox)
        except (TypeError, ValueError):
            logger.debug("footprint: unusable bbox %r; computing the full extent", bbox)
            return None
        # A west > east box is the STAC convention for crossing the antimeridian.
        # Splitting keeps each half an ordinary west-to-east box, which is what
        # reprojection expects — the same treatment the catalog query already applies.
        halves = (
            [(west, south, 180.0, north), (-180.0, south, east, north)] if west > east else [(west, south, east, north)]
        )
        for w, s, e, n in halves:
            try:
                bounds = box(w, s, e, n, "EPSG:4326").to_crs(crs).boundingbox
                c0, r0 = inverse * (bounds.left, bounds.top)
                c1, r1 = inverse * (bounds.right, bounds.bottom)
            except Exception:  # a projection failure means "assume everything"
                logger.debug("footprint: could not project %r; computing the full extent", bbox, exc_info=True)
                return None
            # A projection can return NaN/inf without raising, and math.floor/ceil on
            # those throws OUTSIDE the guard above — turning this function's documented
            # conservative fallback into an aborted date, or an aborted ingest. Checked
            # here so a non-finite corner takes the same "assume everything" path as a
            # projection that failed outright.
            if not all(math.isfinite(v) for v in (r0, r1, c0, c1)):
                logger.debug("footprint: non-finite projection of %r; computing the full extent", bbox)
                return None
            # Outward to whole cells, then padded, then clamped to the grid.
            cr0 = max(math.floor(min(r0, r1) / chunk_px) - FOOTPRINT_PAD_CELLS, 0)
            cr1 = min(math.ceil(max(r0, r1) / chunk_px) + FOOTPRINT_PAD_CELLS, rows)
            cc0 = max(math.floor(min(c0, c1) / chunk_px) - FOOTPRINT_PAD_CELLS, 0)
            cc1 = min(math.ceil(max(c0, c1) / chunk_px) + FOOTPRINT_PAD_CELLS, cols)
            if cr0 < cr1 and cc0 < cc1:
                grid[cr0:cr1, cc0:cc1] = True
    return grid


def windows_for_date(
    windows: list[LiveWindow],
    bboxes: list[tuple[float, float, float, float]],
    geobox: object,
    *,
    chunk_px: int = INGEST_CHUNK_SIZE,
    merge: bool = True,
    window_cost_in_chunks: int = WINDOW_COST_IN_CHUNKS,
) -> list[LiveWindow]:
    """A run's live windows, narrowed to what ONE date's imagery can actually fill.

    A run's windows describe where the ROI has land, and they are identical on every
    date. A single date is not: an optical satellite images a fraction of a wide ROI
    per pass, so most of those windows hold nothing for a given date. Tasks built over
    them still run, find no data, and write nothing — the mosaic is the same either
    way, because an all-fill chunk is never stored.

    So this removes work whose result is already discarded, which is why it cannot
    change what a mosaic contains. What it changes is cost, in both terms that matter:
    the graph shrinks with the area, and windows the date misses entirely disappear,
    taking their serial region writes with them.

    Returns the input unchanged when the footprint cannot be determined
    (:func:`footprint_grid`), so the conservative behaviour is the fallback rather
    than something a caller opts into. An EMPTY result means this date reaches no
    live cell at all and there is nothing to write.

    ``window_cost_in_chunks`` must be the value the RUN's windows were built with
    (:func:`live_windows_for_mask`): the re-merge below is the same exchange rate
    decision made again on a smaller grid, and leaving it at the sequential default
    would buy dead area to save window boundaries that the overlapped write path has
    already made cheap — undoing the calibration for every narrowed date.
    """
    if not windows:
        return windows
    footprint = footprint_grid(bboxes, geobox, chunk_px=chunk_px)
    if footprint is None:
        return windows
    height, width = int(geobox.height), int(geobox.width)  # type: ignore[attr-defined]
    covered = grid_from_windows(windows, height=height, width=width, chunk_px=chunk_px) & footprint
    if not covered.any():
        return []
    bands = row_band_windows(covered, height=height, width=width, chunk_px=chunk_px)
    return merge_bands(bands, chunk_px=chunk_px, window_cost_in_chunks=window_cost_in_chunks) if merge else bands


def live_windows_for_mask(
    mask_path: str,
    *,
    chunk_px: int = INGEST_CHUNK_SIZE,
    window_px: int | None = None,
    prefer_keys: bool = True,
    merge: bool = True,
    window_cost_in_chunks: int = WINDOW_COST_IN_CHUNKS,
    storage_options: StorageOptions = None,
) -> list[LiveWindow]:
    """The one-call form: mask store → live windows to write.

    Derives the live-chunk grid from the mask's chunk keys when the store's layout
    is recognised (:func:`live_chunk_grid_from_keys` — one listing) and falls back
    to reading every chunk position (:func:`live_chunk_grid`) when it is not. Both
    routes return the same grid; the difference is one listing against thousands of
    sequential reads. Row bands are then merged into taller bands
    (:func:`merge_bands`), because the per-window write cost is what dominates.

    ``window_px`` snaps the windows to a COARSER grid than the mask's own chunking —
    set it to the ingest's load-block size so every window lands on whole load
    blocks. It must be a multiple of ``chunk_px``; ``None`` means the mask's grid.
    The mask itself stays at ``chunk_px``, so the fast key-listing path is unaffected.

    ``window_cost_in_chunks`` is what one window is assumed to cost, in chunk area, and
    the caller owns it because it depends on how that caller WRITES: pass
    :data:`WINDOW_COST_IN_CHUNKS_OVERLAPPED` when the windows of a date share one dask
    graph, and leave the default when each window is its own blocking write. Getting
    this backwards is a real regression rather than a mis-tuning — a sequential writer
    on the overlapped rate pays extra serial boundaries for area it does not care about.

    ``prefer_keys=False`` forces the read path and ``merge=False`` the unmerged row
    bands, which is how each is held against the other in tests.
    ``storage_options`` mirrors :func:`ingest.roi.read_roi_mask`, so a deployment
    whose mask needs non-default fsspec/S3 settings derives windows where its
    ingest already reads the mask.
    """
    window_px = window_px or chunk_px
    if window_px % chunk_px:
        raise ValueError(f"window_px {window_px} must be a multiple of chunk_px {chunk_px}")
    mask = _open_mask(mask_path, storage_options)
    height, width = mask.shape
    live = (
        live_chunk_grid_from_keys(mask_path, mask, chunk_px=chunk_px, storage_options=storage_options)
        if prefer_keys
        else None
    )
    if live is None:
        live = live_chunk_grid(mask_path, chunk_px=chunk_px, storage_options=storage_options)
    live = coarsen_live_grid(live, window_px // chunk_px)
    windows = row_band_windows(live, height=height, width=width, chunk_px=window_px)
    if not merge:
        return windows
    return merge_bands(windows, chunk_px=window_px, window_cost_in_chunks=window_cost_in_chunks)
