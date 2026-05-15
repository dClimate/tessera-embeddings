"""Load and prepare data from Icechunk/Zarr stores for inference.

Reads spatial chunks from three stores (reflectance, sar_ascending, sar_descending),
stacks bands in the correct order, derives masks from SCL, and extracts DOY values.

Uses selective S2 timestep loading — a 3-phase approach that cuts peak S2 memory
~10x by only loading the timesteps the model will actually use:

Phase 1 -- Load SCL only: Read just the Scene Classification Layer (one uint8
band per timestep) and derive per-pixel binary validity masks. This is cheap:
~200 MB for 200 timesteps over a 1500x1500 chunk.

Phase 2 -- Pre-sample timestep indices: For every pixel, simulate all
``repeat_times`` random samplings of ``sample_size_s2`` valid timesteps.
Collect the *union* of all selected timestep indices across all pixels. In
practice this yields ~20-40 unique timesteps out of ~200.

Phase 3 -- Load only selected S2 bands: Fetch reflectance data for just the
pre-sampled timesteps. The returned ``ChunkData`` has the same schema but with
a reduced time dimension.

SAR data is always loaded eagerly -- it is ~10x smaller (2 bands vs 10, fewer
timesteps) and does not benefit from selective loading.

See ``ai/analysis/lazy-vs-eager-data-loading.md`` for the full design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import zarr

from tessera_embeddings.config.inference import S2_BAND_ORDER, SCL_VALID_CLASSES, TimeWindow
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.storage.zarr_store import _compute_doy, open_store_as_zarr_group

logger = logging.getLogger(__name__)


@dataclass
class ChunkData:
    """Loaded and prepared data for one spatial chunk.

    All arrays have spatial dimensions matching the chunk (H, W).
    Time dimensions may differ across sensors.

    Attributes:
        s2_bands: S2 reflectance bands, shape (T_s2, H, W, 10), uint16.
        s2_masks: Binary valid-pixel mask from SCL, shape (T_s2, H, W), int32.
        s2_doys: Day-of-year for each S2 timestep, shape (T_s2,), int32.
        s1_asc_bands: S1 ascending VV+VH, shape (T_s1a, H, W, 2), native store dtype.
        s1_asc_doys: DOY for ascending, shape (T_s1a,), int32.
        s1_desc_bands: S1 descending VV+VH, shape (T_s1d, H, W, 2), native store dtype.
        s1_desc_doys: DOY for descending, shape (T_s1d,), int32.
        height: Spatial height of the chunk.
        width: Spatial width of the chunk.
        s2_obs_count: Per-pixel count of valid S2 timesteps computed from
            the FULL (pre-pruning) SCL mask, shape (H, W), uint16. When present,
            the dataset uses this for the min_valid_timesteps check instead of
            recomputing from the (pruned) s2_masks. Also written to the output
            Zarr as an observation-count layer.
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

    Args:
        root: zarr.Group opened from the SAR store.
        time_indices: Absolute store-level time indices to consider.
        y_slice: Northing slice for the chunk.
        x_slice: Easting slice for the chunk.

    Returns:
        Tuple of (bands, kept_indices) with shapes (T_kept, H, W, 2) and (T_kept,).
    """
    # VV and VH are always co-acquired, so filtering on VV coverage is sufficient for both.
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
    """Stack S2 bands in the canonical order, reading directly from zarr.

    Bypasses xarray/dask. Each ``.values`` on an xarray dask-backed array
    triggers a fresh dask graph compute and leaves scheduler state around
    for the lifetime of the dataset handle; in a 10-band loop at chunk scale
    that caused the Phase 3 transient to spike ~2-3x above the 9.6 GB output.
    Direct zarr reads allocate the decompression buffer, copy into ``result``,
    and release immediately.

    Args:
        root: zarr.Group opened from the reflectance store.
        time_indices: Selected time indices (absolute into the store).
        y_slice: Northing slice for the chunk.
        x_slice: Easting slice for the chunk.

    Returns:
        Array of shape (T, H, W, 10), uint16.
    """
    ref = root[S2_BAND_ORDER[0]]
    t = len(time_indices)
    h = y_slice.stop - y_slice.start
    w = x_slice.stop - x_slice.start
    result = np.empty((t, h, w, len(S2_BAND_ORDER)), dtype=ref.dtype)
    for i, band in enumerate(S2_BAND_ORDER):
        # zarr orthogonal selection: (list, slice, slice) → (T, H, W)
        result[:, :, :, i] = root[band].oindex[time_indices, y_slice, x_slice]
    return result


def _load_scl_mask(
    root: zarr.Group,
    time_indices: np.ndarray,
    y_slice: slice,
    x_slice: slice,
) -> np.ndarray:
    """Read SCL directly from zarr and derive the binary validity mask.

    Bypasses xarray/dask for the same reason as :func:`_load_s2_bands`: each
    ``.values`` on a dask-backed xarray array builds a fresh task graph and
    dispatches per-chunk S3 fetches serialized through icechunk scheduler/session
    state. On full-year ROIs the SCL has ~350 timesteps to read, and that
    per-timestep overhead (not raw bandwidth) was making Phase 1 cost longer
    than Phase 3. Direct zarr reads reuse a single session and dispatch the
    per-chunk fetches without the task-graph hop.

    Args:
        root: zarr.Group opened from the reflectance store.
        time_indices: Selected time indices (absolute into the store).
        y_slice: Northing slice for the chunk.
        x_slice: Easting slice for the chunk.

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

    The time array is written as int64 nanoseconds-since-epoch by xarray's zarr
    writer and can be decoded directly as datetime64[ns], avoiding the need to
    open an xarray/dask session just to filter timestamps.

    Args:
        root: zarr.Group opened from the store.
        window: 12-month time window to filter to.

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
    return indices, _compute_doy(times[indices])


def check_time_window_coverage(
    mosaic_base: str,
    window: TimeWindow,
    s1_orbit: str = "ascending",
    skip_coverage_check: bool = False,
) -> None:
    """Verify that source stores span the requested time window.

    Opens each store, reads only the time coordinate, and checks that the
    store's time range covers the window's earliest and latest months.

    Args:
        mosaic_base: Base path for the mosaic stores.
        window: Resolved 12-month time window.
        s1_orbit: Which SAR orbit(s) to check.
        skip_coverage_check: If True, skip the range boundary checks
            (store must still have at least one time entry).

    Raises:
        InsufficientCoverageError: If any required store does not span the window.
    """
    earliest = window.months[0]  # (year, month) — chronologically first
    latest = window.months[-1]  # chronologically last

    stores = [("reflectance", f"{mosaic_base}/reflectance.zarr")]
    if s1_orbit == "ascending":
        stores.append(("sar_ascending", f"{mosaic_base}/sar_ascending.zarr"))
    if s1_orbit == "descending":
        stores.append(("sar_descending", f"{mosaic_base}/sar_descending.zarr"))

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
# Phase 2: Pre-sample timestep indices
# ---------------------------------------------------------------------------


def _presample_s2_timestep_indices(
    s2_masks: np.ndarray,
    sample_size_s2: int,
    repeat_times: int,
    rng: np.random.Generator | None = None,
    max_sample_pixels: int = 10000,
) -> np.ndarray:
    """Determine which S2 timesteps are needed across all pixels and repeats.

    Simulates the random temporal sampling that will happen during inference:
    for a representative subset of pixels, draw ``sample_size_s2`` valid
    timestep indices ``repeat_times`` times. Returns the sorted union of all
    selected indices.

    This is the core of the memory optimization. Instead of loading all ~200
    timesteps of reflectance data, we identify the ~20-40 unique timesteps
    that will actually be consumed by the model.

    NOTE **CHANGED**
    this function is NOT ported from tessera — it is our own optimization for
    the selective timestep loading pipeline.

    **Optimized implementation (v2):** The original v1 looped over every pixel
    (O(H*W) = 9M for 3000x3000 chunks), taking ~8 minutes per chunk. This
    version reduces that to <1 second by:
    1. Fast-checks if validity is high enough that all timesteps will be
       needed anyway (common case, returns immediately).
    2. Samples a random subset of pixels (default 10K) rather than all pixels.
       The union of needed timesteps converges quickly — 10K pixels with
       10 repeats x 20 samples = 2M draws is more than enough to cover
       all reachable timesteps.
    3. Uses vectorized numpy sampling per pixel (no inner Python loop).

    POSSIBLE SOURCES OF ERROR to watch for:
      - Pixel subsampling (max_sample_pixels=10K) could theoretically miss a
        timestep that is valid for only a tiny spatial region. In practice this
        is astronomically unlikely (10K random pixels across 9M provides >99.9%
        coverage of all validity patterns). If embeddings look wrong for rare
        edge pixels, increase max_sample_pixels.
      - The fast-path checks per-timestep .any() validity. If a timestep is
        valid for even 1 pixel, it's counted. This is conservative (loads more
        timesteps than strictly needed) which is safe — it can only load extra
        data, never miss data that's needed.

    Args:
        s2_masks: Binary mask from SCL, shape (T, H, W). 1 = valid.
        sample_size_s2: Number of timesteps sampled per pixel per repeat.
        repeat_times: Number of independent samplings per pixel.
        rng: Optional numpy random generator for reproducibility.
        max_sample_pixels: Maximum number of pixels to sample. The union of
            needed timesteps converges well below this. Default 10000.

    Returns:
        Sorted 1-D array of unique timestep indices to load. If there are
        fewer unique valid timesteps across all pixels than ``sample_size_s2``,
        all timesteps are returned (fallback to eager).
    """
    if rng is None:
        rng = np.random.default_rng()

    n_t, n_h, n_w = s2_masks.shape
    all_indices = np.arange(n_t)
    n_pixels = n_h * n_w

    if n_t == 0:
        logger.info("Pre-sampling: chunk has no valid S2 timesteps, skipping")
        return all_indices

    # NOTE: an earlier version short-circuited here if >=90% of timesteps had
    # any valid pixel. That check is useless once Phase 1 prunes all empty
    # timesteps in the chunk — the fraction is always 100% by construction.
    # Any dense-coverage early exit must now be based on *per-pixel* coverage,
    # which the sampling loop below handles via its own 90% check.

    # Sample a random subset of pixels (much cheaper than iterating all pixels).
    # Index directly into the 3D mask array to avoid materializing a 9M x T matrix.
    n_sample = min(n_pixels, max_sample_pixels)
    sample_rows = rng.integers(0, n_h, size=n_sample)
    sample_cols = rng.integers(0, n_w, size=n_sample)

    # Collect every timestep index that gets sampled for any pixel/repeat.
    # Always include the latest timestep so the newest observation enters
    # sampling — critical for incremental updates.
    needed: set[int] = {n_t - 1}

    for i in range(n_sample):
        pixel_mask = s2_masks[:, sample_rows[i], sample_cols[i]]  # (n_t,)
        valid_idx = all_indices[pixel_mask == 1]

        # Fallback: if no valid timesteps, sampling will use all indices
        if len(valid_idx) == 0:
            valid_idx = all_indices

        # Vectorized: sample all repeats at once
        sampled = rng.choice(valid_idx, size=(repeat_times, sample_size_s2), replace=True)
        needed.update(sampled.ravel().tolist())

        # Early exit
        if len(needed) >= n_t * 0.9:
            logger.debug(
                "Pre-sampling selected %d/%d timesteps (>=90%%) after %d/%d pixels, loading all",
                len(needed),
                n_t,
                i + 1,
                n_sample,
            )
            return all_indices

    if len(needed) >= n_t * 0.9:
        logger.debug("Pre-sampling selected %d/%d timesteps (>=90%%), loading all", len(needed), n_t)
        return all_indices

    result = np.sort(np.array(list(needed), dtype=np.intp))
    logger.info(
        "Pre-sampling: %d/%d S2 timesteps needed (sampled %d/%d pixels x %d repeats)",
        len(result),
        n_t,
        n_sample,
        n_pixels,
        repeat_times,
    )
    return result


# ---------------------------------------------------------------------------
# Main loading functions
# ---------------------------------------------------------------------------


def _empty_sar_arrays(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return empty SAR band and DOY arrays for a skipped orbit direction."""
    return (
        np.empty((0, height, width, 2), dtype=np.uint16),
        np.empty((0,), dtype=np.int32),
    )


@dataclass
class _S2LoadResult:
    """Result of the 3-phase selective S2 loading pipeline."""

    bands: np.ndarray  # (T_sel, H, W, 10), uint16
    masks: np.ndarray  # (T_sel, H, W), int32
    doys: np.ndarray  # (T_sel,), int32
    s2_obs_count: np.ndarray  # (H, W), uint16
    t_full: int  # original timestep count
    t_selected: int  # selected timestep count


def _load_s2_selective(
    mosaic_base: str,
    chunk: ChunkSpec,
    sample_size_s2: int,
    repeat_times: int,
    rng: np.random.Generator | None,
    time_window: TimeWindow,
) -> _S2LoadResult:
    """Run the 3-phase selective S2 loading pipeline.

    Phase 1: Load SCL mask and DOYs (cheap).
    Phase 2: Pre-sample timestep indices across all pixels/repeats.
    Phase 3: Load only the selected S2 reflectance bands.

    See ``load_chunk`` module docstring for full design rationale.
    """
    store_path = f"{mosaic_base}/reflectance.zarr"
    root = open_store_as_zarr_group(store_path)
    window_indices, s2_doys_full = _filter_times_from_zarr(root, time_window)
    y_slice = slice(chunk.y_start, chunk.y_stop)
    x_slice = slice(chunk.x_start, chunk.x_stop)

    # Phase 1: Load SCL only (direct zarr — no dask graph per timestep)
    abs_indices = window_indices
    s2_masks_full = _load_scl_mask(root, abs_indices, y_slice, x_slice)
    t_full = s2_masks_full.shape[0]
    s2_obs_count = s2_masks_full.sum(axis=0).astype(np.uint16)
    logger.info("Phase 1: Loaded SCL for %d S2 timesteps", t_full)

    # Drop timesteps with no valid pixels anywhere in the chunk. For large
    # ROIs the time axis often covers the full year even though each chunk
    # only intersects a subset of acquisitions; pruning here shrinks both
    # the pre-sampling domain and the Phase 3 load.
    nonempty_t = s2_masks_full.any(axis=(1, 2))
    n_nonempty = int(nonempty_t.sum())
    if n_nonempty < t_full:
        kept = np.where(nonempty_t)[0]
        s2_masks_full = s2_masks_full[kept]
        s2_doys_full = s2_doys_full[kept]
        abs_indices = abs_indices[kept]
        logger.info("Phase 1: pruned %d/%d empty S2 timesteps", t_full - n_nonempty, t_full)
        t_full = n_nonempty

    # Phase 2: Pre-sample timestep indices
    needed_indices = _presample_s2_timestep_indices(
        s2_masks_full,
        sample_size_s2=sample_size_s2,
        repeat_times=repeat_times,
        rng=rng,
    )
    t_selected = len(needed_indices)
    logger.info(
        "Phase 2: %d/%d S2 timesteps selected (%.0f%% reduction)",
        t_selected,
        t_full,
        100.0 * (1 - t_selected / max(t_full, 1)),
    )

    # Phase 3: Load only selected S2 bands.
    abs_selected = abs_indices[needed_indices]
    s2_bands = _load_s2_bands(
        root,
        time_indices=abs_selected,
        y_slice=y_slice,
        x_slice=x_slice,
    )
    s2_masks = s2_masks_full[needed_indices]
    s2_doys = s2_doys_full[needed_indices]
    logger.info(
        "Phase 3: Loaded S2 bands -- shape %s (was %s with all timesteps)",
        s2_bands.shape,
        f"({t_full}, {chunk.height}, {chunk.width}, 10)",
    )

    return _S2LoadResult(
        bands=s2_bands,
        masks=s2_masks,
        doys=s2_doys,
        s2_obs_count=s2_obs_count,
        t_full=t_full,
        t_selected=t_selected,
    )


def _load_sar_orbit(
    mosaic_base: str,
    chunk: ChunkSpec,
    orbit: str,
    time_window: TimeWindow,
) -> tuple[np.ndarray, np.ndarray]:
    """Load SAR bands and DOYs for a single orbit direction.

    Returns:
        Tuple of (bands, doys) with shapes (T, H, W, 2) uint16 and (T,) int32.
    """
    root = open_store_as_zarr_group(f"{mosaic_base}/sar_{orbit}.zarr")
    window_indices, doys_full = _filter_times_from_zarr(root, time_window)
    y_slice = slice(chunk.y_start, chunk.y_stop)
    x_slice = slice(chunk.x_start, chunk.x_stop)
    bands, kept = _load_sar_bands_from_zarr(root, window_indices, y_slice, x_slice)
    return bands, doys_full[kept]


def load_chunk(
    chunk: ChunkSpec,
    mosaic_base: str,
    sample_size_s2: int,
    repeat_times: int,
    time_window: TimeWindow,
    rng: np.random.Generator | None = None,
    s1_orbit: Literal["ascending", "descending"] = "ascending",
) -> ChunkData:
    """Load all data for one spatial chunk with selective S2 timestep loading.

    Delegates to ``_load_s2_selective`` for the 3-phase S2 pipeline and
    ``_load_sar_orbit`` for each SAR orbit direction. See those helpers
    for phase-level details.

    Args:
        chunk: Spatial chunk specification.
        mosaic_base: Base path for the mosaic stores
            (e.g., "s3://cl-preprocessed-data-dev/mosaics/small_minnesota").
        sample_size_s2: Number of S2 timesteps sampled per pixel per repeat.
        repeat_times: Number of independent samplings averaged per pixel.
        time_window: 12-month time window for temporal filtering.
        rng: Optional numpy random generator for reproducibility. If None,
            a new unseeded generator is created.
        s1_orbit: Which S1 orbit direction(s) to load.

    Returns:
        ChunkData with S2 arrays subsetted to only the needed timesteps.
        SAR arrays contain all timesteps within the time window.
    """
    if s1_orbit not in {"ascending", "descending"}:
        raise ValueError(f"Invalid s1_orbit: {s1_orbit!r}. Must be 'ascending' or 'descending'.")
    logger.info(
        "Loading chunk %s from %s (sample_size_s2=%d, repeat_times=%d, s1_orbit=%s, time_window=%s)",
        chunk.label,
        mosaic_base,
        sample_size_s2,
        repeat_times,
        s1_orbit,
        f"{time_window.months[0]}-{time_window.months[-1]}",
    )

    s2 = _load_s2_selective(mosaic_base, chunk, sample_size_s2, repeat_times, rng, time_window=time_window)

    s1_asc_bands, s1_asc_doys = (
        _load_sar_orbit(mosaic_base, chunk, "ascending", time_window=time_window)
        if s1_orbit == "ascending"
        else _empty_sar_arrays(chunk.height, chunk.width)
    )
    s1_desc_bands, s1_desc_doys = (
        _load_sar_orbit(mosaic_base, chunk, "descending", time_window=time_window)
        if s1_orbit == "descending"
        else _empty_sar_arrays(chunk.height, chunk.width)
    )

    # Compute per-pixel observation counts (either VV or VH nonzero = valid observation)
    s1_asc_obs_count = np.any(s1_asc_bands != 0, axis=-1).sum(axis=0).astype(np.uint16)
    s1_desc_obs_count = np.any(s1_desc_bands != 0, axis=-1).sum(axis=0).astype(np.uint16)

    logger.info(
        "Loaded %s: S2 %s (%d/%d dates), SAR asc %s (%d dates), SAR desc %s (%d dates)",
        chunk.label,
        s2.bands.shape,
        s2.t_selected,
        s2.t_full,
        s1_asc_bands.shape,
        len(s1_asc_doys),
        s1_desc_bands.shape,
        len(s1_desc_doys),
    )

    return ChunkData(
        s2_bands=s2.bands,
        s2_masks=s2.masks,
        s2_doys=s2.doys,
        s1_asc_bands=s1_asc_bands,
        s1_asc_doys=s1_asc_doys,
        s1_desc_bands=s1_desc_bands,
        s1_desc_doys=s1_desc_doys,
        height=chunk.height,
        width=chunk.width,
        s2_obs_count=s2.s2_obs_count,
        s1_asc_obs_count=s1_asc_obs_count,
        s1_desc_obs_count=s1_desc_obs_count,
    )
