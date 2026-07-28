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

import datetime
import logging
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.inference import S2_BAND_ORDER, SCL_VALID_CLASSES
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.storage.zarr_store import (
    ASSESSED_WINDOW_ATTR,
    compute_doy,
    is_missing_repo,
    open_store_as_zarr_group,
)

logger = logging.getLogger(__name__)


# Worker cap for the concurrent S2 band read (see ``_load_s2_bands``). The 10
# bands are independent zarr arrays, so their reads fan out across a thread
# pool. Unlike the latency-bound ROI probe (``chunk_spec._ROI_PROBE_WORKERS``,
# oversubscribed x4), this read is decompression-bound: with time chunked at 1,
# each band's ``oindex`` already issues its per-timestep S3 GETs concurrently,
# so the serial cost we remove is the 10 decompression waves, not GET latency.
# Decompression runs in Rust with the GIL released, so it scales to cores — but
# only to cores, so we cap at the allocated CPU count (never above the band
# count) to keep decompression cores busy without oversubscribing. Uses
# ``sched_getaffinity`` where available (Linux inference actors) so a
# cgroup/affinity-pinned worker sees its real allocation, not the host's.
def _band_read_workers(reserve_cpus: int = 0) -> int:
    """Decompression pool size for the concurrent S2 band read.

    ``reserve_cpus`` leaves that many cores free for concurrent CPU work.
    Callers loading in the background while the GPU runs (the intra-chunk
    strip prefetch) reserve cores for the batch-prep workers that feed the
    GPU — on the 4-vCPU g6e.xlarge, four decompression threads competing with
    batch prep produced ~500 ms get_batch spikes that starved the GPU.
    Foreground loads (the serial chunk prologue, GPU idle) reserve nothing.
    """
    try:
        allocated = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:  # macOS/Windows: no affinity API
        allocated = os.cpu_count() or 4
    # Reserve cores from the ALLOCATION first, then cap at the band count, so a
    # host with more CPUs than bands still runs the full reader set (reserving
    # after the cap would needlessly drop readers, e.g. 12 CPUs, 2 reserved ->
    # min(10,12)-2=8 instead of min(10,10)=10). On the 4-vCPU worker both give 2.
    return max(1, min(len(S2_BAND_ORDER), allocated - reserve_cpus))


# A function that maps a store path to an open zarr group. The default is
# ``open_store_as_zarr_group`` (a fresh repo open per call). The strip loop
# passes a memoizing opener (see ``make_store_opener``) so every strip of a
# chunk reuses one repo handle rather than re-paying the repo-open and
# manifest-load cost on each strip.
StoreOpener = Callable[[str], zarr.Group]


def make_store_opener(region: str | None = None) -> StoreOpener:
    """Return a store opener that opens each distinct path once and reuses it.

    Intended to live for the duration of one chunk's strip loop: each strip of
    a split chunk would otherwise reopen the same three stores, re-paying the
    icechunk repo-open and manifest load every time. (It does not amortise chunk
    *data* re-reads: a dense strip's working set far exceeds icechunk's chunk
    cache, so cross-strip reuse does not hit — see
    ``zarr_store._default_repo_config``.)

    ``region`` is the S3 region for the mosaic repos (credentials are injected
    separately, via the actor's :func:`credentials_provider` context); a
    non-default-region fill must thread it so the actor's reads open the store in
    the same region the preflight/assembly paths use.
    """
    cache: dict[str, zarr.Group] = {}

    def _open(store_path: str) -> zarr.Group:
        group = cache.get(store_path)
        if group is None:
            group = open_store_as_zarr_group(store_path, region=region)
            cache[store_path] = group
        return group

    return _open


@dataclass
class S2MaskBundle:
    """Full-chunk S2 SCL validity, loaded once and shared across northing strips.

    The strip loop reads SCL for the whole chunk a single time, then slices this
    bundle per strip instead of re-decompressing SCL on every strip. SCL chunks
    on disk are ``(time=1, 4000, 4000)``, so any sub-region read decompresses the
    whole chunk anyway; loading once and slicing turns the per-strip SCL re-reads
    into pure in-memory views. The mask is also what sizes the strip height (see
    ``actors._strip_height_for_density``): its ``T_kept`` is the true post-pruning
    timestep count for *this* chunk, so sparse chunks get tall strips (often the
    whole chunk) and only genuinely dense chunks split.

    Timesteps are pruned at the *chunk* level — any timestep with no valid pixel
    anywhere in the chunk is dropped, since the per-pixel resampler can never draw
    it. A strip may therefore carry a timestep that is empty within its own rows;
    that wastes a little band memory for that strip but keeps pruning a single
    whole-chunk decision rather than a per-strip one.

    Attributes:
        mask: Binary SCL validity, shape (T_kept, chunk_height, W), bool.
        doys: Day-of-year per kept timestep, shape (T_kept,), int32.
        abs_indices: Absolute store-level time indices of kept timesteps, (T_kept,).
        obs_count: Per-pixel valid-timestep count from the full (pre-prune) mask,
            shape (chunk_height, W), uint16.
    """

    mask: np.ndarray
    doys: np.ndarray
    abs_indices: np.ndarray
    obs_count: np.ndarray


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
    # Full-chunk-width SAR obs counts, populated ONLY for x-cropped loads
    # (x_sub): the saved obs-count layers must keep full-extent fidelity, but
    # the cropped grid above can't carry it. SAR is read full-width regardless
    # (it is ~10x smaller than S2); these hold the pre-crop counts.
    s1_asc_obs_count_full: np.ndarray | None = None  # (H, full W), uint16
    s1_desc_obs_count_full: np.ndarray | None = None  # (H, full W), uint16


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
    reserve_cpus: int = 0,
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

    # The 10 bands are independent zarr arrays and each thread writes a distinct
    # ``result[..., i]`` column, so the reads have no shared mutable state and
    # run concurrently. Reading one band per iteration is serial across 10
    # decompression waves; fanning them out collapses those to one wave bounded
    # by the allocated cores (see ``_band_read_workers``). Concurrent reads on
    # one readonly icechunk session store (an immutable snapshot view) are safe
    # — only concurrent *commits* are not; verified against a real store.
    def _read_band(i: int, band: str) -> None:
        result[:, :, :, i] = root[band].oindex[time_indices, y_slice, x_slice]

    with ThreadPoolExecutor(max_workers=_band_read_workers(reserve_cpus)) as pool:
        # Drain the map so any read exception propagates instead of being
        # swallowed with a partially-filled ``result``.
        list(pool.map(lambda ib: _read_band(*ib), enumerate(S2_BAND_ORDER)))
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


def _active_orbits(s1_orbit: str) -> tuple[str, ...]:
    """Return the orbit directions active for a given ``s1_orbit`` setting."""
    if s1_orbit == "both":
        return ("ascending", "descending")
    if s1_orbit in {"ascending", "descending"}:
        return (s1_orbit,)
    raise ValueError(f"Invalid s1_orbit: {s1_orbit!r}. Must be 'ascending', 'descending', or 'both'.")


def resolve_s1_orbit(
    mosaic_base: str,
    s1_orbit: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> str:
    """Downgrade ``s1_orbit="both"`` to a single orbit when only one store exists.

    ``"both"`` is a request, not a guarantee — if upstream ingestion only wrote
    one SAR store, the inference pipeline transparently falls back to
    single-orbit rather than failing. Probing once at flow entry keeps every
    downstream callsite aligned on the same effective orbit value.

    ``"ascending"`` / ``"descending"`` are returned unchanged without probing;
    a missing store at that point is an error surfaced downstream.

    The SAR stores are opened with the SAME credential callback / region that
    the runner uses for the rest of the fill (``get_credentials`` / ``s3_region``);
    without them the probe would fall back to the default Icechunk credential
    chain and fail at orbit resolution in any deployment that needs the callback.
    """
    if s1_orbit != "both":
        _active_orbits(s1_orbit)  # validates
        return s1_orbit

    present = []
    for orbit in ("ascending", "descending"):
        path = f"{mosaic_base}/sar_{orbit}.zarr"
        try:
            open_store_as_zarr_group(path, get_credentials=get_credentials, region=s3_region)
            present.append(orbit)
        except zarr.errors.GroupNotFoundError:
            # PRESENT but rootless (created, then crashed before the schema commit).
            # GroupNotFoundError subclasses FileNotFoundError, so without this clause a
            # damaged orbit reads as an absent one — and with s1_orbit="both" and the
            # sibling healthy, the campaign would quietly resolve to a single orbit and
            # permanently publish half the radar it was asked for. Fail closed.
            raise
        except FileNotFoundError:
            logger.info("SAR %s store not present at %s — will be excluded", orbit, path)
        except icechunk.IcechunkError as exc:
            # ONLY a genuinely-absent repo means "exclude this orbit". A timeout,
            # auth failure, or corruption must not silently downgrade `both` to a
            # single orbit and permanently drop the other's data — re-raise so the
            # caller fails loudly instead.
            if not is_missing_repo(exc):
                raise
            logger.info("SAR %s store not present at %s — will be excluded", orbit, path)

    if not present:
        msg = f"s1_orbit='both' but no SAR stores found under {mosaic_base}"
        raise InsufficientCoverageError(msg)
    if len(present) == 1:
        logger.warning("s1_orbit='both' requested but only %s store is present — falling back", present[0])
        return present[0]
    return "both"


def _months_within_assessed(months: list[tuple[int, int]], assessed: object) -> set[tuple[int, int]]:
    """Of ``months``, those lying ENTIRELY inside an ``assessed_window`` attribute.

    Returns an empty set for any unusable attribute — absent, malformed, unparseable — so a
    damaged record makes the gate STRICTER rather than more permissive. That asymmetry is the
    safety argument: over-excusing a month publishes a mosaic with a hole in it, while
    under-excusing one costs a re-ingest.
    """
    if not isinstance(assessed, (list, tuple)) or len(assessed) != 2:
        return set()
    try:
        start = datetime.date.fromisoformat(str(assessed[0]))
        end = datetime.date.fromisoformat(str(assessed[1]))
    except ValueError:
        logger.warning("Unparseable %s attribute %r — treating as absent", ASSESSED_WINDOW_ATTR, assessed)
        return set()
    inside = set()
    for year, month in months:
        first = datetime.date(year, month, 1)
        last = datetime.date(year + month // 12, month % 12 + 1, 1) - datetime.timedelta(days=1)
        if start <= first and last <= end:
            inside.add((year, month))
    return inside


def check_time_window_coverage(
    mosaic_base: str,
    window: TimeWindow,
    s1_orbit: str = "both",
    skip_coverage_check: bool = False,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> None:
    """Verify that source stores span the requested time window.

    Opens each store with the caller's credential callback / region (same as the
    rest of the fill); ``skip_coverage_check=True`` still hard-fails an EMPTY
    store (no in-window data at all) but skips the month-span check, the escape
    hatch for a legitimately partial window (e.g. an arctic-only edge zone).

    Raises:
        InsufficientCoverageError: If any required store does not span the window.
    """
    earliest = window.months[0]
    latest = window.months[-1]

    stores = [("reflectance", f"{mosaic_base}/reflectance.zarr")]
    for orbit in _active_orbits(s1_orbit):
        stores.append((f"sar_{orbit}", f"{mosaic_base}/sar_{orbit}.zarr"))

    required_months = set(window.months)
    for label, path in stores:
        root = open_store_as_zarr_group(path, get_credentials=get_credentials, region=s3_region)
        times = root["time"][:].astype("datetime64[ns]")
        if len(times) == 0:
            msg = f"{label} store at {path} has no time entries"
            raise InsufficientCoverageError(msg)

        years = times.astype("datetime64[Y]").astype(int) + 1970
        months = times.astype("datetime64[M]").astype(int) % 12 + 1
        present_months = set(zip(years.tolist(), months.tolist(), strict=True))

        if skip_coverage_check:
            # Partial-window mode still requires at least one timestamp INSIDE the
            # window — a store with only out-of-window (e.g. prior-year) dates is
            # non-empty but useless, and must not pass the preflight / be marked
            # ingested only for the fill to later find zero in-window observations.
            if not (present_months & required_months):
                msg = (
                    f"{label} store at {path} has no timestamps within the window "
                    f"{earliest[0]}-{earliest[1]:02d}..{latest[0]}-{latest[1]:02d}"
                )
                raise InsufficientCoverageError(msg)
            continue

        # Require EVERY month of the window to be present, not just that the
        # min/max span it: a mosaic with only January + December (or out-of-window
        # dates bracketing the year) would otherwise pass despite missing every
        # intervening month, and the write-once tag would make that partial year
        # permanent. Month granularity matches the campaign's calendar-year window.
        missing = sorted(required_months - present_months)

        # A month can be absent because the ingest EXAMINED it and found nothing reachable,
        # which is a finding, or because the ingest never covered it, which is a gap. Only
        # the second is an error, and `assessed_window` is what tells them apart: the ingest
        # records the range it processed in full, so a month wholly inside that range was
        # looked at. A satellite pass covers a swath rather than a whole UTM zone, and some
        # zones have an orbit that reaches their land on no date of the year at all.
        #
        # Requires the month to be COVERED ENTIRELY. A partially-assessed month could hide
        # unexamined days, so it stays an error — strict here costs nothing on the campaign's
        # calendar-year windows, where months are always wholly inside.
        if missing:
            assessed = root.attrs.get(ASSESSED_WINDOW_ATTR)
            examined = _months_within_assessed(missing, assessed)
            if examined:
                logger.info(
                    "%s store at %s: %d month(s) absent but inside the assessed window %s "
                    "— examined and holding no reachable imagery, not a gap (e.g. %s)",
                    label,
                    path,
                    len(examined),
                    assessed,
                    ", ".join(f"{y}-{m:02d}" for y, m in sorted(examined)[:6]),
                )
                missing = [m for m in missing if m not in examined]

        if missing:
            preview = ", ".join(f"{y}-{m:02d}" for y, m in missing[:6])
            assessed = root.attrs.get(ASSESSED_WINDOW_ATTR)
            why = (
                f"assessed window {assessed} does not cover them"
                if assessed
                else "the store records no assessed window, so absence cannot be explained"
            )
            msg = (
                f"{label} store at {path} is missing {len(missing)} of the window's "
                f"{len(required_months)} month(s) (e.g. {preview}) — "
                f"window {earliest[0]}-{earliest[1]:02d}..{latest[0]}-{latest[1]:02d}; {why}"
            )
            raise InsufficientCoverageError(msg)

    logger.info(
        "Time window coverage verified (every month present): %d-%02d through %d-%02d",
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


def _resolve_x_slice(chunk: ChunkSpec, x_sub: slice | None) -> slice:
    """Resolve the absolute store-level easting slice for a (sub-)chunk read.

    ``x_sub`` is chunk-relative, mirroring ``_resolve_y_slice``. ``None`` reads
    the full chunk width.
    """
    if x_sub is None:
        return slice(chunk.x_start, chunk.x_stop)
    return slice(chunk.x_start + x_sub.start, chunk.x_start + x_sub.stop)


def load_s2_mask_bundle(
    mosaic_base: str,
    chunk: ChunkSpec,
    time_window: TimeWindow,
    store_opener: StoreOpener | None = None,
) -> S2MaskBundle:
    """Load and prune the full-chunk S2 SCL mask once for reuse across strips.

    Reads SCL for the entire chunk extent (all easting, all northing), computes
    the per-pixel obs count from the un-pruned mask, then drops timesteps with no
    valid pixel anywhere in the chunk. The result is shared by every strip's
    band load (sliced, not re-read) and is what sizes the strip height.

    SCL is 1 byte/pixel — far cheaper than the 20 bytes/pixel of reflectance — so
    loading the whole chunk's mask up front is a small fixed cost that removes the
    per-strip SCL re-decompression entirely. See :class:`S2MaskBundle`.
    """
    if store_opener is None:
        store_opener = open_store_as_zarr_group
    root = store_opener(f"{mosaic_base}/reflectance.zarr")
    window_indices, doys_full = _filter_times_from_zarr(root, time_window)
    y_slice = slice(chunk.y_start, chunk.y_stop)
    x_slice = slice(chunk.x_start, chunk.x_stop)

    mask_full = _load_scl_mask(root, window_indices, y_slice, x_slice)
    t_full = mask_full.shape[0]
    # obs_count from the full (pre-prune) mask so pixels aren't under-counted.
    obs_count = mask_full.sum(axis=0).astype(np.uint16)
    logger.info("Loaded full-chunk SCL for %d S2 timesteps", t_full)

    # Prune timesteps with no valid pixel anywhere in the chunk — the v1.1
    # per-pixel resampler only draws from valid indices, so fully-empty
    # timesteps would never be read by any strip.
    nonempty_t = mask_full.any(axis=(1, 2))
    kept = np.where(nonempty_t)[0]
    if len(kept) < t_full:
        logger.info("Pruned %d/%d empty S2 timesteps (chunk-level)", t_full - len(kept), t_full)

    return S2MaskBundle(
        mask=mask_full[kept],
        doys=doys_full[kept],
        abs_indices=window_indices[kept],
        obs_count=obs_count,
    )


def _load_s2(
    mosaic_base: str,
    chunk: ChunkSpec,
    time_window: TimeWindow,
    y_sub: slice | None = None,
    store_opener: StoreOpener | None = None,
    mask_bundle: S2MaskBundle | None = None,
    reserve_cpus: int = 0,
    x_sub: slice | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all S2 timesteps in the window that have any valid pixel in the (sub-)chunk.

    ``y_sub`` is an optional chunk-relative northing slice. When given, only
    that horizontal strip of the chunk is read (full easting width). ``None``
    reads the full chunk.

    ``mask_bundle`` is an optional precomputed full-chunk SCL bundle (see
    :func:`load_s2_mask_bundle`). When supplied, the SCL is sliced from it rather
    than re-read — chunk-level timestep pruning has already happened, so the band
    load reads exactly the bundle's kept timesteps. When ``None`` (the unstriped
    default path), SCL is loaded and pruned inline as before.

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
    x_slice = _resolve_x_slice(chunk, x_sub)

    if mask_bundle is not None:
        # Slice the shared full-chunk mask to this strip's rows (and, when
        # x-cropped, to the valid-bbox columns); the chunk-level prune already
        # chose the kept timesteps, so band reads match the bundle.
        rows = y_sub if y_sub is not None else slice(0, chunk.height)
        cols = x_sub if x_sub is not None else slice(0, chunk.width)
        s2_masks = mask_bundle.mask[:, rows, cols]
        s2_obs_count = mask_bundle.obs_count[rows, cols]
        s2_doys = mask_bundle.doys
        abs_kept = mask_bundle.abs_indices
        y_slice = _resolve_y_slice(chunk, y_sub)

        # obs_count > 0 ⟺ the pruned mask has a valid entry (pruning drops only
        # all-False planes), and the (H, W) scan avoids walking the strided
        # (T_kept, H, W) mask view (~100 ms/chunk on dense multi-strip chunks).
        if not s2_obs_count.any():
            # No valid S2 pixel anywhere in this strip: bucketing would select
            # zero pixels, so the (expensive, 20 B/px) band read is pure waste.
            # Return a T=0 band stack — the dataset sees no candidates and the
            # strip short-circuits — while obs counts keep full fidelity from
            # the bundle. Common on sparse/edge chunks whose valid sliver lies
            # in other strips; the timestep prune can't help them because it is
            # chunk-global (one sliver anywhere keeps the timestep).
            h = y_slice.stop - y_slice.start
            w = x_slice.stop - x_slice.start
            logger.info("Strip rows %s have no valid S2 pixels — skipping band read", rows)
            empty = np.empty((0, h, w, len(S2_BAND_ORDER)), dtype=root[S2_BAND_ORDER[0]].dtype)
            return empty, s2_masks[:0], s2_doys[:0], s2_obs_count

        s2_bands = _load_s2_bands(
            root, time_indices=abs_kept, y_slice=y_slice, x_slice=x_slice, reserve_cpus=reserve_cpus
        )
        logger.info("Loaded S2 bands shape %s (sliced shared mask)", s2_bands.shape)
        return s2_bands, s2_masks, s2_doys, s2_obs_count

    window_indices, s2_doys_full = _filter_times_from_zarr(root, time_window)
    y_slice = _resolve_y_slice(chunk, y_sub)

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

    s2_bands = _load_s2_bands(root, time_indices=abs_kept, y_slice=y_slice, x_slice=x_slice, reserve_cpus=reserve_cpus)
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
    s1_orbit: Literal["ascending", "descending", "both"] = "both",
    y_sub: slice | None = None,
    store_opener: StoreOpener | None = None,
    mask_bundle: S2MaskBundle | None = None,
    reserve_cpus: int = 0,
    x_sub: slice | None = None,
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
            (see ``make_store_opener``) for all strips of a chunk so each strip
            reuses one repo handle instead of re-paying the repo open per strip.
        y_sub: Optional chunk-relative northing slice bounding the resident
            input working set. When given, only that horizontal strip of the
            chunk is read (full easting width) and the returned ``ChunkData``
            describes a self-contained strip — its ``height`` reflects the
            strip, ``width`` stays full, and pruning / obs counts are computed
            on the strip's own pixels (so a strip may keep a different T_kept
            than its neighbour). ``None`` reads the whole chunk, reproducing
            the unstriped behaviour byte-for-byte.
        mask_bundle: Optional precomputed full-chunk S2 SCL bundle (see
            :func:`load_s2_mask_bundle`). When supplied, S2 SCL is sliced from
            it per strip instead of re-read, and timestep pruning uses the
            chunk-level decision baked into the bundle. The strip loop loads the
            bundle once and passes it to every strip; ``None`` loads SCL inline.
        reserve_cpus: Cores to leave free during the S2 band decompression —
            background loads running alongside GPU inference reserve cores for
            the batch-prep workers (see :func:`_band_read_workers`); foreground
            loads use the default 0.
        x_sub: Optional chunk-relative EASTING window (the S2 valid-pixel
            bounding box on sparse/edge chunks). S2 bands — the 20 B/px cost —
            are read only for these columns and the returned grid is cropped
            to them (``width`` shrinks, mirroring how ``y_sub`` crops rows).
            SAR is still READ full-width so the saved obs-count layers keep
            full-extent fidelity: the pre-crop counts are returned in
            ``s1_*_obs_count_full`` while the grid-shaped ``s1_*_obs_count``
            are cropped like everything else. Pixels outside the box have zero
            valid S2 observations, so they could never be inferred — cropping
            them changes which bytes are read, not any output.

    Returns:
        ChunkData with all S2 timesteps that have any valid pixel in the
        (sub-)region and all SAR timesteps within the time window.
    """
    active = set(_active_orbits(s1_orbit))
    height = chunk.height if y_sub is None else (y_sub.stop - y_sub.start)
    width = chunk.width if x_sub is None else (x_sub.stop - x_sub.start)
    logger.info(
        "Loading chunk %s from %s (s1_orbit=%s, time_window=%s, y_sub=%s)",
        chunk.label,
        mosaic_base,
        s1_orbit,
        f"{time_window.months[0]}-{time_window.months[-1]}",
        y_sub,
    )

    s2_bands, s2_masks, s2_doys, s2_obs_count = _load_s2(
        mosaic_base,
        chunk,
        time_window,
        y_sub=y_sub,
        store_opener=store_opener,
        mask_bundle=mask_bundle,
        reserve_cpus=reserve_cpus,
        x_sub=x_sub,
    )

    # Skipped orbits get FULL-width placeholders: every SAR array must be
    # full-width here because the x_sub block below crops them all uniformly
    # (after capturing the full-width obs-count side channel).
    s1_asc_bands, s1_asc_doys = (
        _load_sar_orbit(
            mosaic_base, chunk, "ascending", time_window=time_window, y_sub=y_sub, store_opener=store_opener
        )
        if "ascending" in active
        else _empty_sar_arrays(height, chunk.width)
    )
    s1_desc_bands, s1_desc_doys = (
        _load_sar_orbit(
            mosaic_base, chunk, "descending", time_window=time_window, y_sub=y_sub, store_opener=store_opener
        )
        if "descending" in active
        else _empty_sar_arrays(height, chunk.width)
    )

    s1_asc_obs_count = np.any(s1_asc_bands != 0, axis=-1).sum(axis=0).astype(np.uint16)
    s1_desc_obs_count = np.any(s1_desc_bands != 0, axis=-1).sum(axis=0).astype(np.uint16)

    s1_asc_obs_count_full: np.ndarray | None = None
    s1_desc_obs_count_full: np.ndarray | None = None
    if x_sub is not None:
        # SAR was read full-width (see the x_sub docstring): keep the full-width
        # counts for the saved obs layers, then crop the grid-shaped arrays to
        # match the S2 grid. ascontiguousarray drops the full-width parents.
        s1_asc_obs_count_full = s1_asc_obs_count
        s1_desc_obs_count_full = s1_desc_obs_count
        s1_asc_obs_count = np.ascontiguousarray(s1_asc_obs_count[:, x_sub])
        s1_desc_obs_count = np.ascontiguousarray(s1_desc_obs_count[:, x_sub])
        s1_asc_bands = np.ascontiguousarray(s1_asc_bands[:, :, x_sub, :])
        s1_desc_bands = np.ascontiguousarray(s1_desc_bands[:, :, x_sub, :])

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
        s1_asc_obs_count_full=s1_asc_obs_count_full,
        s1_desc_obs_count_full=s1_desc_obs_count_full,
    )
