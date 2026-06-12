"""Load and prepare data from Icechunk/Zarr stores for inference.

Reads spatial chunks from three stores (reflectance, sar_ascending, sar_descending),
stacks bands in the correct order, derives masks from SCL, and extracts DOY values.

Under Tessera v1.1 the model uses every valid observation per pixel (bucketed
resampling at the dataset layer), so all S2 timesteps with any valid pixel in
the chunk are loaded. Timesteps with zero coverage across the chunk are pruned
to avoid paying to load data the sampler can't use anyway.

SAR data is ~10x smaller than S2 (2 bands, fewer timesteps); both orbits are
loaded eagerly.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.inference import S2_BAND_ORDER, SCL_VALID_CLASSES
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.storage.zarr_store import compute_doy, open_store_as_zarr_group

logger = logging.getLogger(__name__)

# A function that maps a store path to an open zarr group. The default is
# ``open_store_as_zarr_group`` (a fresh repo open per call). The strip loop
# passes a memoizing opener (see ``make_store_opener``) so every strip of a
# chunk shares one repo handle — and thus one persistent 512 MB chunk cache —
# rather than reopening cold and discarding the cache between strips.
StoreOpener = Callable[[str], zarr.Group]


def make_store_opener() -> StoreOpener:
    """Return a store opener that opens each distinct path once and reuses it.

    Holding the opened zarr groups alive keeps their icechunk chunk cache warm
    across calls. Intended to live for the duration of one chunk's strip loop:
    overlapping northing strips re-touch the same ``(time=1, 4000, 4000)`` store
    chunks, so a shared cache turns the per-strip re-reads into cache hits
    instead of repeated S3 fetches.
    """
    cache: dict[str, zarr.Group] = {}

    def _open(store_path: str) -> zarr.Group:
        group = cache.get(store_path)
        if group is None:
            group = open_store_as_zarr_group(store_path)
            cache[store_path] = group
        return group

    return _open


@dataclass
class ChunkData:
    """Loaded and prepared data for one spatial chunk.

    All arrays have spatial dimensions matching the chunk (H, W).
    Time dimensions may differ across sensors.

    Attributes:
        s2_bands: S2 reflectance bands, shape (T_s2, H, W, 10), uint16.
        s2_masks: Binary valid-pixel mask from SCL, shape (T_s2, H, W), bool.
        s2_doys: Day-of-year for each S2 timestep, shape (T_s2,), int32.
        s1_asc_bands: S1 ascending VV+VH, shape (T_s1a, H, W, 2), native store dtype.
        s1_asc_doys: DOY for ascending, shape (T_s1a,), int32.
        s1_desc_bands: S1 descending VV+VH, shape (T_s1d, H, W, 2), native store dtype.
        s1_desc_doys: DOY for descending, shape (T_s1d,), int32.
        height: Spatial height of the chunk.
        width: Spatial width of the chunk.
        s2_obs_count: Per-pixel count of valid S2 timesteps, shape (H, W), uint16.
        s1_asc_obs_count: Per-pixel count of valid S1-asc timesteps, (H, W), uint16.
        s1_desc_obs_count: Per-pixel count of valid S1-desc timesteps, (H, W), uint16.
    """

    s2_bands: np.ndarray
    s2_masks: np.ndarray
    s2_doys: np.ndarray
    s1_asc_bands: np.ndarray
    s1_asc_doys: np.ndarray
    s1_desc_bands: np.ndarray
    s1_desc_doys: np.ndarray
    height: int
    width: int
    s2_obs_count: np.ndarray | None = None  # (H, W), uint16
    s1_asc_obs_count: np.ndarray | None = None  # (H, W), uint16
    s1_desc_obs_count: np.ndarray | None = None  # (H, W), uint16


def _load_sar_bands_from_zarr(
    root: zarr.Group,
    time_indices: np.ndarray,
    y_slice: slice,
    x_slice: slice,
) -> tuple[np.ndarray, np.ndarray]:
    """Read SAR VV+VH directly from zarr, dropping timesteps with no coverage.

    Returns:
        Tuple of (bands, kept_indices) with shapes (T_kept, H, W, 2) and (T_kept,).
    """
    vv = root["0_VV"].oindex[time_indices, y_slice, x_slice]  # (T, H, W)
    valid_t = (vv != 0).any(axis=(1, 2))
    kept = np.where(valid_t)[0]
    n_kept = len(kept)
    n_total = len(time_indices)

    h = y_slice.stop - y_slice.start
    w = x_slice.stop - x_slice.start
    result = np.empty((n_kept, h, w, 2), dtype=vv.dtype)
    result[:, :, :, 0] = vv[kept]
    del vv
    result[:, :, :, 1] = root["0_VH"].oindex[time_indices[kept], y_slice, x_slice]

    if n_kept < n_total:
        logger.info("SAR: dropped %d/%d empty timesteps from chunk", n_total - n_kept, n_total)
    return result, kept


def _load_s2_bands(
    root: zarr.Group,
    time_indices: np.ndarray,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """Stack S2 bands in canonical order, reading directly from zarr.

    Returns:
        Array of shape (T, H, W, 10), uint16.
    """
    ref = root[S2_BAND_ORDER[0]]
    t = len(time_indices)
    h = y_slice.stop - y_slice.start
    w = x_slice.stop - x_slice.start
    result = np.empty((t, h, w, len(S2_BAND_ORDER)), dtype=ref.dtype)
    for i, band in enumerate(S2_BAND_ORDER):
        result[:, :, :, i] = root[band].oindex[time_indices, y_slice, x_slice]
    return result


def _load_scl_mask(
    root: zarr.Group,
    time_indices: np.ndarray,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """Read SCL directly from zarr and derive the binary validity mask.

    Returns:
        Binary mask of shape (T, H, W), bool. True = valid, False = invalid.
    """
    scl = root["scl"].oindex[time_indices, y_slice, x_slice]
    return np.isin(scl, list(SCL_VALID_CLASSES))


# ---------------------------------------------------------------------------
# Time window filtering
# ---------------------------------------------------------------------------


def _filter_times_from_zarr(root: zarr.Group, window: TimeWindow) -> tuple[np.ndarray, np.ndarray]:
    """Filter the time coordinate of a raw zarr group to a TimeWindow.

    Returns:
        Tuple of (window_indices, doys): absolute store-level integer indices of
        the matching timesteps, and int32 DOY values for those timesteps.
    """
    times = root["time"][:].astype("datetime64[ns]")
    years = times.astype("datetime64[Y]").astype(int) + 1970
    months_arr = times.astype("datetime64[M]").astype(int) % 12 + 1
    month_set = set(window.months)
    mask = np.array([(int(y), int(m)) in month_set for y, m in zip(years, months_arr, strict=True)])
    indices = np.where(mask)[0]
    if len(indices) == 0:
        msg = f"No observations found within time window {window.months[0]}-{window.months[-1]}"
        raise RuntimeError(msg)
    return indices, compute_doy(times[indices])


def count_s2_window_timesteps(
    mosaic_base: str, time_window: TimeWindow, store_opener: StoreOpener | None = None
) -> int:
    """Return how many S2 timesteps fall inside ``time_window`` for this mosaic.

    Reads only the reflectance store's 1-D ``time`` coordinate (no spatial
    data), so it is a cheap probe of the upper bound on the per-chunk valid
    timestep count ``T`` before any band load. Used by the inference actor to
    size northing strips ahead of the first full read. The true post-pruning
    ``T_kept`` is <= this value, so sizing on it is conservative (strips end up
    no wider than the budget allows).

    ``store_opener`` defaults to a fresh repo open; pass a shared opener (see
    ``make_store_opener``) to reuse the reflectance handle the strip loads use,
    so the probe primes the chunk cache instead of opening a throwaway repo.
    """
    if store_opener is None:
        store_opener = open_store_as_zarr_group
    root = store_opener(f"{mosaic_base}/reflectance.zarr")
    window_indices, _ = _filter_times_from_zarr(root, time_window)
    return len(window_indices)


def _active_orbits(s1_orbit: str) -> tuple[str, ...]:
    """Return the orbit directions active for a given ``s1_orbit`` setting."""
    if s1_orbit == "both":
        return ("ascending", "descending")
    if s1_orbit in {"ascending", "descending"}:
        return (s1_orbit,)
    raise ValueError(f"Invalid s1_orbit: {s1_orbit!r}. Must be 'ascending', 'descending', or 'both'.")


def resolve_s1_orbit(mosaic_base: str, s1_orbit: str) -> str:
    """Downgrade ``s1_orbit="both"`` to a single orbit when only one store exists.

    ``"both"`` is a request, not a guarantee — if upstream ingestion only wrote
    one SAR store, the inference pipeline transparently falls back to
    single-orbit rather than failing. Probing once at flow entry keeps every
    downstream callsite aligned on the same effective orbit value.

    ``"ascending"`` / ``"descending"`` are returned unchanged without probing;
    a missing store at that point is an error surfaced downstream.
    """
    if s1_orbit != "both":
        _active_orbits(s1_orbit)  # validates
        return s1_orbit

    present = []
    for orbit in ("ascending", "descending"):
        path = f"{mosaic_base}/sar_{orbit}.zarr"
        try:
            open_store_as_zarr_group(path)
            present.append(orbit)
        except (FileNotFoundError, icechunk.IcechunkError):
            logger.info("SAR %s store not present at %s — will be excluded", orbit, path)

    if not present:
        msg = f"s1_orbit='both' but no SAR stores found under {mosaic_base}"
        raise InsufficientCoverageError(msg)
    if len(present) == 1:
        logger.warning("s1_orbit='both' requested but only %s store is present — falling back", present[0])
        return present[0]
    return "both"


def check_time_window_coverage(
    mosaic_base: str,
    window: TimeWindow,
    s1_orbit: str = "ascending",
    skip_coverage_check: bool = False,
) -> None:
    """Verify that source stores span the requested time window.

    Raises:
        InsufficientCoverageError: If any required store does not span the window.
    """
    earliest = window.months[0]
    latest = window.months[-1]

    stores = [("reflectance", f"{mosaic_base}/reflectance.zarr")]
    for orbit in _active_orbits(s1_orbit):
        stores.append((f"sar_{orbit}", f"{mosaic_base}/sar_{orbit}.zarr"))

    for label, path in stores:
        root = open_store_as_zarr_group(path)
        times = root["time"][:].astype("datetime64[ns]")
        if len(times) == 0:
            msg = f"{label} store at {path} has no time entries"
            raise InsufficientCoverageError(msg)

        if skip_coverage_check:
            continue

        store_min = times.min()
        store_max = times.max()
        store_min_ym = (
            int(store_min.astype("datetime64[Y]").astype(int)) + 1970,
            int(store_min.astype("datetime64[M]").astype(int)) % 12 + 1,
        )
        store_max_ym = (
            int(store_max.astype("datetime64[Y]").astype(int)) + 1970,
            int(store_max.astype("datetime64[M]").astype(int)) % 12 + 1,
        )

        if store_min_ym > earliest:
            msg = (
                f"{label} store starts at {store_min_ym[0]}-{store_min_ym[1]:02d}, "
                f"but window requires data from {earliest[0]}-{earliest[1]:02d}"
            )
            raise InsufficientCoverageError(msg)
        if store_max_ym < latest:
            msg = (
                f"{label} store ends at {store_max_ym[0]}-{store_max_ym[1]:02d}, "
                f"but window requires data through {latest[0]}-{latest[1]:02d}"
            )
            raise InsufficientCoverageError(msg)

    logger.info(
        "Time window coverage verified: %d-%02d through %d-%02d",
        earliest[0],
        earliest[1],
        latest[0],
        latest[1],
    )


# ---------------------------------------------------------------------------
# Main loading
# ---------------------------------------------------------------------------


def _empty_sar_arrays(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return empty SAR band and DOY arrays for a skipped orbit direction."""
    return (
        np.empty((0, height, width, 2), dtype=np.uint16),
        np.empty((0,), dtype=np.int32),
    )


def _resolve_y_slice(chunk: ChunkSpec, y_sub: slice | None) -> slice:
    """Resolve the absolute store-level northing slice for a (sub-)chunk read.

    ``y_sub`` is an offset *relative to the chunk* — it narrows the read within
    the chunk's existing read-tile to a horizontal strip. When ``None`` the
    full chunk extent is read, reproducing the unstriped behaviour exactly.
    """
    if y_sub is None:
        return slice(chunk.y_start, chunk.y_stop)
    return slice(chunk.y_start + y_sub.start, chunk.y_start + y_sub.stop)


def _load_s2(
    mosaic_base: str,
    chunk: ChunkSpec,
    time_window: TimeWindow,
    y_sub: slice | None = None,
    store_opener: StoreOpener | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all S2 timesteps in the window that have any valid pixel in the (sub-)chunk.

    ``y_sub`` is an optional chunk-relative northing slice. When given, only
    that horizontal strip of the chunk is read (full easting width); pruning and
    obs counts are computed on the strip's own pixels. ``None`` reads the full
    chunk.

    Returns:
        Tuple of (s2_bands, s2_masks, s2_doys, s2_obs_count):
          - s2_bands: (T_kept, H, W, 10) uint16
          - s2_masks: (T_kept, H, W) bool — binary SCL validity
          - s2_doys: (T_kept,) int32
          - s2_obs_count: (H, W) uint16 — per-pixel valid-timestep count from full mask
    """
    if store_opener is None:
        store_opener = open_store_as_zarr_group
    store_path = f"{mosaic_base}/reflectance.zarr"
    root = store_opener(store_path)
    window_indices, s2_doys_full = _filter_times_from_zarr(root, time_window)
    y_slice = _resolve_y_slice(chunk, y_sub)
    x_slice = slice(chunk.x_start, chunk.x_stop)

    abs_indices = window_indices
    s2_masks_full = _load_scl_mask(root, abs_indices, y_slice, x_slice)
    t_full = s2_masks_full.shape[0]
    # Compute obs_count from the full mask before pruning so pixels aren't under-counted.
    s2_obs_count = s2_masks_full.sum(axis=0).astype(np.uint16)
    logger.info("Loaded SCL for %d S2 timesteps", t_full)

    # Prune timesteps with no valid pixels anywhere in the chunk — the v1.1
    # per-pixel resampler only draws from valid indices, so fully-empty
    # timesteps would never be read.
    nonempty_t = s2_masks_full.any(axis=(1, 2))
    kept = np.where(nonempty_t)[0]
    n_kept = len(kept)
    if n_kept < t_full:
        logger.info("Pruned %d/%d empty S2 timesteps", t_full - n_kept, t_full)
    abs_kept = abs_indices[kept]
    s2_masks = s2_masks_full[kept]

    # Free T_full-sized mask before allocating T_kept-sized bands (saves memory on large ROIs).
    del s2_masks_full

    s2_doys = s2_doys_full[kept]

    s2_bands = _load_s2_bands(root, time_indices=abs_kept, y_slice=y_slice, x_slice=x_slice)
    logger.info("Loaded S2 bands shape %s", s2_bands.shape)

    # Keep the mask bool (1 byte/elem) rather than widening to int32 — it stays
    # resident in ChunkData for the whole chunk, and the only consumers
    # (np.nonzero in resampling, .sum for obs_count) handle bool directly.
    return s2_bands, s2_masks, s2_doys, s2_obs_count


def _load_sar_orbit(
    mosaic_base: str,
    chunk: ChunkSpec,
    orbit: str,
    time_window: TimeWindow,
    y_sub: slice | None = None,
    store_opener: StoreOpener | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load SAR bands and DOYs for a single orbit direction.

    ``y_sub`` is an optional chunk-relative northing slice (full easting width);
    ``None`` reads the full chunk.

    Returns:
        Tuple of (bands, doys) with shapes (T, H, W, 2) uint16 and (T,) int32.
    """
    if store_opener is None:
        store_opener = open_store_as_zarr_group
    root = store_opener(f"{mosaic_base}/sar_{orbit}.zarr")
    window_indices, doys_full = _filter_times_from_zarr(root, time_window)
    y_slice = _resolve_y_slice(chunk, y_sub)
    x_slice = slice(chunk.x_start, chunk.x_stop)
    bands, kept = _load_sar_bands_from_zarr(root, window_indices, y_slice, x_slice)
    return bands, doys_full[kept]


def load_chunk(
    chunk: ChunkSpec,
    mosaic_base: str,
    time_window: TimeWindow,
    s1_orbit: Literal["ascending", "descending", "both"] = "ascending",
    y_sub: slice | None = None,
    store_opener: StoreOpener | None = None,
) -> ChunkData:
    """Load all data for one spatial chunk (or a northing strip of it) for v1.1 inference.

    Args:
        chunk: Spatial chunk specification.
        mosaic_base: Base path for the mosaic stores.
        time_window: 12-month time window for temporal filtering.
        s1_orbit: Which S1 orbit direction(s) to load — ``"ascending"``,
            ``"descending"``, or ``"both"``.
        store_opener: Maps a store path to an open zarr group. Defaults to a
            fresh repo open per call; the strip loop passes one shared opener
            (see ``make_store_opener``) for all strips of a chunk so the 512 MB
            chunk cache persists across strips instead of starting cold on each.
        y_sub: Optional chunk-relative northing slice bounding the resident
            input working set. When given, only that horizontal strip of the
            chunk is read (full easting width) and the returned ``ChunkData``
            describes a self-contained strip — its ``height`` reflects the
            strip, ``width`` stays full, and pruning / obs counts are computed
            on the strip's own pixels (so a strip may keep a different T_kept
            than its neighbour). ``None`` reads the whole chunk, reproducing
            the unstriped behaviour byte-for-byte.

    Returns:
        ChunkData with all S2 timesteps that have any valid pixel in the
        (sub-)region and all SAR timesteps within the time window.
    """
    active = set(_active_orbits(s1_orbit))
    height = chunk.height if y_sub is None else (y_sub.stop - y_sub.start)
    width = chunk.width
    logger.info(
        "Loading chunk %s from %s (s1_orbit=%s, time_window=%s, y_sub=%s)",
        chunk.label,
        mosaic_base,
        s1_orbit,
        f"{time_window.months[0]}-{time_window.months[-1]}",
        y_sub,
    )

    s2_bands, s2_masks, s2_doys, s2_obs_count = _load_s2(
        mosaic_base, chunk, time_window, y_sub=y_sub, store_opener=store_opener
    )

    s1_asc_bands, s1_asc_doys = (
        _load_sar_orbit(
            mosaic_base, chunk, "ascending", time_window=time_window, y_sub=y_sub, store_opener=store_opener
        )
        if "ascending" in active
        else _empty_sar_arrays(height, width)
    )
    s1_desc_bands, s1_desc_doys = (
        _load_sar_orbit(
            mosaic_base, chunk, "descending", time_window=time_window, y_sub=y_sub, store_opener=store_opener
        )
        if "descending" in active
        else _empty_sar_arrays(height, width)
    )

    s1_asc_obs_count = np.any(s1_asc_bands != 0, axis=-1).sum(axis=0).astype(np.uint16)
    s1_desc_obs_count = np.any(s1_desc_bands != 0, axis=-1).sum(axis=0).astype(np.uint16)

    logger.info(
        "Loaded %s: S2 %s, SAR asc %s (%d dates), SAR desc %s (%d dates)",
        chunk.label,
        s2_bands.shape,
        s1_asc_bands.shape,
        len(s1_asc_doys),
        s1_desc_bands.shape,
        len(s1_desc_doys),
    )

    return ChunkData(
        s2_bands=s2_bands,
        s2_masks=s2_masks,
        s2_doys=s2_doys,
        s1_asc_bands=s1_asc_bands,
        s1_asc_doys=s1_asc_doys,
        s1_desc_bands=s1_desc_bands,
        s1_desc_doys=s1_desc_doys,
        height=height,
        width=width,
        s2_obs_count=s2_obs_count,
        s1_asc_obs_count=s1_asc_obs_count,
        s1_desc_obs_count=s1_desc_obs_count,
    )
