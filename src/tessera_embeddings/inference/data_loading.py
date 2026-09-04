"""Load and prepare data from Icechunk/Zarr stores for inference.

Reads spatial chunks from three stores (reflectance, sar_ascending, sar_descending),
stacks bands in the correct order, derives masks from SCL, and extracts DOY values.

Under Tessera v1.1 the model uses every valid observation per pixel (bucketed resampling at
the dataset layer), so all S2 timesteps with any valid pixel in the chunk are loaded.
Timesteps with zero coverage across the chunk are pruned — the sampler could never draw them.

SAR is ~10x smaller than S2 (2 bands, fewer timesteps); both orbits are loaded eagerly.
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

from tessera_embeddings.config.inference import S1_ORBIT_NONE, S2_BAND_ORDER, SCL_VALID_CLASSES
from tessera_embeddings.config.store_layout import MONTHS_IN_YEAR
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.storage.time_axis import compute_doy
from tessera_embeddings.storage.zarr_store import ASSESSED_WINDOW_ATTR, is_missing_repo, open_store_as_zarr_group

logger = logging.getLogger(__name__)


# Worker cap for the concurrent S2 band read (see ``_load_s2_bands``). Unlike the latency-bound
# ROI probe (``chunk_spec._ROI_PROBE_WORKERS``, oversubscribed x4), this read is
# DECOMPRESSION-bound: with time chunked at 1, each band's ``oindex`` already issues its
# per-timestep S3 GETs concurrently, so what fanning out removes is the 10 serial decompression
# waves, not GET latency. Decompression runs in Rust with the GIL released, so it scales to cores
# and only to cores — hence the cap at the allocated CPU count, never above the band count.
# ``sched_getaffinity`` where available (Linux actors) so a cgroup-pinned worker sees its real
# allocation, not the host's.
def _band_read_workers(reserve_cpus: int = 0) -> int:
    """Decompression pool size for the concurrent S2 band read.

    ``reserve_cpus`` leaves that many cores free for concurrent CPU work. Callers loading in
    the background while the GPU runs (the intra-chunk strip prefetch) reserve cores for the
    batch-prep workers that feed it — on the 4-vCPU g6e.xlarge, four decompression threads
    competing with batch prep produced ~500 ms get_batch spikes that starved the GPU.
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


# Maps a store path to an open zarr group. Default ``open_store_as_zarr_group`` (a fresh repo open
# per call); the strip loop passes a memoizing opener (``make_store_opener``) so every strip of a
# chunk reuses one repo handle instead of re-paying the repo-open and manifest load.
StoreOpener = Callable[[str], zarr.Group]


def make_store_opener(region: str | None = None) -> StoreOpener:
    """Return a store opener that opens each distinct path once and reuses it.

    Intended to live for one chunk's strip loop: each strip of a split chunk would otherwise
    reopen the same three stores. It does NOT amortise chunk *data* re-reads — a dense strip's
    working set far exceeds icechunk's chunk cache, so cross-strip reuse does not hit (see
    ``zarr_store._default_repo_config``).

    ``region`` is the S3 region for the mosaic repos (credentials are injected separately, via
    the actor's :func:`credentials_provider` context); a non-default-region fill must thread it
    so the actor's reads open the store in the same region as the preflight/assembly paths.
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

    The strip loop reads SCL for the whole chunk once, then slices this bundle per strip. SCL
    chunks on disk are ``(time=1, 4000, 4000)``, so any sub-region read decompresses the whole
    chunk anyway; loading once turns per-strip SCL re-reads into in-memory views. The mask also
    sizes the strip height (``actors._strip_height_for_density``): its ``T_kept`` is the true
    post-pruning timestep count for *this* chunk, so sparse chunks get tall strips and only
    genuinely dense chunks split.

    Timesteps are pruned at the *chunk* level — any timestep with no valid pixel anywhere in the
    chunk is dropped, since the per-pixel resampler can never draw it. A strip may therefore
    carry a timestep empty within its own rows, wasting a little band memory but keeping pruning
    one whole-chunk decision.

    Attributes:
        mask: Binary SCL validity, shape (T_kept, chunk_height, W), bool.
        doys: Day-of-year per kept timestep, shape (T_kept,), int32.
        abs_indices: Absolute store-level time indices of kept timesteps, (T_kept,).
        obs_count: Per-pixel valid-timestep count from the full (pre-prune) mask,
            shape (chunk_height, W), uint16.
        month_covered: Which calendar months each pixel was seen in, shape
            (12, chunk_height, W), bool, month 0 = January. From the same
            pre-prune mask as ``obs_count``, so the twelve flags partition
            exactly the observations that count totals.
    """

    mask: np.ndarray
    doys: np.ndarray
    abs_indices: np.ndarray
    obs_count: np.ndarray
    month_covered: np.ndarray


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
    # Full-chunk-width SAR obs counts, populated ONLY for x-cropped loads (x_sub): the saved
    # obs-count layers must keep full-extent fidelity, which the cropped grid above cannot carry.
    # SAR is read full-width regardless (~10x smaller than S2); these hold the pre-crop counts.
    s1_asc_obs_count_full: np.ndarray | None = None  # (H, full W), uint16
    s1_desc_obs_count_full: np.ndarray | None = None  # (H, full W), uint16
    # Which months each pixel was seen in per orbit, (MONTHS_IN_YEAR, H, W) bool, paired with the
    # counts above and cropped alongside them. S2's equivalent rides on the mask bundle instead,
    # because optical SCL is loaded whole-chunk once and shared across strips.
    s1_asc_month_covered: np.ndarray | None = None
    s1_desc_month_covered: np.ndarray | None = None
    s1_asc_month_covered_full: np.ndarray | None = None  # (MONTHS_IN_YEAR, H, full W), bool
    s1_desc_month_covered_full: np.ndarray | None = None


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
    # ``result[..., i]`` column, so the reads share no mutable state. Fanning them out collapses
    # 10 serial decompression waves into one bounded by the allocated cores
    # (``_band_read_workers``). Concurrent reads on one readonly icechunk session store (an
    # immutable snapshot view) are safe — only concurrent *commits* are not; verified on a real
    # store.
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


def _filter_times_from_zarr(root: zarr.Group, window: TimeWindow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter the time coordinate of a raw zarr group to a TimeWindow.

    Returns:
        Tuple of (window_indices, doys, months): absolute store-level integer indices of
        the matching timesteps, int32 DOY values for those timesteps, and their calendar
        months as 1-12.

    Months are taken from the timestamps themselves rather than derived from the DOYs, which
        would need the year to place a boundary and would be a day out for every month after
        February of a leap year.
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
    return indices, compute_doy(times[indices]), months_arr[indices].astype(np.int16)


def _active_orbits(s1_orbit: str) -> tuple[str, ...]:
    """Return the orbit directions active for a given ``s1_orbit`` setting."""
    if s1_orbit == "both":
        return ("ascending", "descending")
    if s1_orbit == S1_ORBIT_NONE:
        return ()
    if s1_orbit in {"ascending", "descending"}:
        return (s1_orbit,)
    raise ValueError(f"Invalid s1_orbit: {s1_orbit!r}. Must be 'ascending', 'descending', 'both', or 'none'.")


def resolve_s1_orbit(
    mosaic_base: str,
    s1_orbit: str,
    *,
    allow_none: bool = True,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> str:
    """Downgrade ``s1_orbit="both"`` to a single orbit when only one store exists.

    ``"both"`` is a request, not a guarantee: if upstream ingestion wrote only one SAR store,
    inference falls back to single-orbit rather than failing. Probing once at flow entry keeps
    every downstream callsite on the same effective orbit. ``"ascending"`` / ``"descending"``
    are returned unchanged without probing; a missing store then is an error downstream.

    ``allow_none`` defaults to **True**, because parts of the globe are radar-free in principle
    and a global product cannot refuse them: over ice Sentinel-1 flies Extra Wide swath with
    HH/HV, which the dual-pol query correctly discards, so a zone can be permanently radar-free
    while its catalogue holds a hundred thousand granules. Requiring a SAR store there fails the
    cell forever, on every retry.

    Pass ``allow_none=False`` where radar is a *demand* — a single run over terrain known to be
    imaged, where an absent store means something upstream broke and embedding without radar
    would hide it. Callers reach this through the flows' ``require_s1`` parameter.

    The SAR stores are opened with the SAME credential callback / region the runner uses for the
    rest of the fill; without them the probe falls back to the default Icechunk credential chain
    and fails at orbit resolution wherever the callback is needed.
    """
    if s1_orbit != "both":
        _active_orbits(s1_orbit)  # validates
        if s1_orbit == S1_ORBIT_NONE and not allow_none:
            # `none` is meant as a RESOLVED value — what "both" becomes over radar-free terrain —
            # but it is a plain string on a public flow parameter, so a caller can pass it in.
            # Returned unchecked it skips the checks below, and `InferenceConfig` then forces
            # `allow_s2_only`, so a run that DEMANDED radar publishes optical-only embeddings
            # instead of failing. The contradiction is refused rather than read either way.
            raise InsufficientCoverageError(
                f"s1_orbit={S1_ORBIT_NONE!r} was requested for {mosaic_base} while radar was demanded "
                "(require_s1). Those cannot both hold: drop require_s1 to embed this cell optical-only, "
                "or name the orbit(s) the run must have."
            )
        return s1_orbit

    present = []
    for orbit in ("ascending", "descending"):
        path = f"{mosaic_base}/sar_{orbit}.zarr"
        try:
            open_store_as_zarr_group(path, get_credentials=get_credentials, region=s3_region)
            present.append(orbit)
        except zarr.errors.GroupNotFoundError:
            # PRESENT but rootless (created, then crashed before the schema commit).
            # GroupNotFoundError subclasses FileNotFoundError, so without this clause a damaged
            # orbit reads as an absent one — and with `both` and a healthy sibling the campaign
            # quietly resolves to a single orbit, permanently publishing half the radar it was
            # asked for. Fail closed.
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
        if not allow_none:
            msg = (
                f"s1_orbit='both' but no SAR stores found under {mosaic_base}, and radar was "
                "demanded (require_s1). Either the ingest lost an orbit, or this ROI is "
                "genuinely radar-free — the ingest's per-orbit item counts say which."
            )
            raise InsufficientCoverageError(msg)
        # A consumer reading a finished mosaic cannot distinguish a radar-free ROI from a lost
        # orbit, so this warning is the only record it happened — and the confirmation lives in a
        # different run's log: the INGEST queried both orbits and its per-orbit item count is the
        # authority (`items_seen=0` means the source offers nothing here, which is terrain).
        logger.warning(
            "s1_orbit='both' and NO SAR store exists under %s — resolving to %r. "
            "Legitimate where the ROI has no dual-pol VV+VH coverage; check the ingest's "
            "per-batch item counts to confirm the source offered nothing usable.",
            mosaic_base,
            S1_ORBIT_NONE,
        )
        return S1_ORBIT_NONE
    if len(present) == 1:
        logger.warning("s1_orbit='both' requested but only %s store is present — falling back", present[0])
        return present[0]
    return "both"


def _months_within_assessed(months: list[tuple[int, int]], assessed: object) -> set[tuple[int, int]]:
    """Of ``months``, those lying ENTIRELY inside an ``assessed_window`` attribute.

    Returns an empty set for any unusable attribute — absent, malformed, unparseable — so a
    damaged record makes the gate STRICTER. Over-excusing a month publishes a mosaic with a hole
    in it; under-excusing one costs a re-ingest.
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
) -> dict:
    """Verify that source stores span the requested time window; report what they hold.

    Opens each store with the caller's credential callback / region (same as the
    rest of the fill); ``skip_coverage_check=True`` still hard-fails an EMPTY
    store (no in-window data at all) but skips the month-span check, the escape
    hatch for a legitimately partial window (e.g. an arctic-only edge zone).

    **Returns what it measured, so a fill can record it, and the RELAXED path is why that is
    worth doing.** A cell filled under ``skip_coverage_check`` is published from an input the
    strict rule would have refused, and mosaics are deleted once a cell lands, so afterwards
    nothing can tell. The summary is per store label — months present of those required, the
    first/last in-window date — plus ``relaxed``, recording which rule was in force.
    :func:`~tessera_embeddings.storage.shard_writer.run_provenance` lands it on the year.

    Raises:
        InsufficientCoverageError: If any required store does not span the window.
    """
    summary: dict = {
        "window_months": len(window.months),
        "relaxed": bool(skip_coverage_check),
        "stores": {},
    }
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

        # Recorded before any raise, and measured against the REQUIRED months rather than
        # the store's own extent: a store may hold dates outside the window, and what a
        # published year needs to state is how much of ITS OWN window the input covered.
        in_window = times[[(y, m) in required_months for y, m in zip(years.tolist(), months.tolist(), strict=True)]]
        summary["stores"][label] = {
            "months_present": len(present_months & required_months),
            "dates_in_window": len(in_window),
            "first": str(in_window.min())[:10] if len(in_window) else None,
            "last": str(in_window.max())[:10] if len(in_window) else None,
        }

        # At least one timestamp INSIDE the window, whichever mode this is. A store holding only
        # out-of-window dates is non-empty but useless: the loaders filter to the window and raise
        # on an empty index, so without this the preflight passes, a GPU fleet is provisioned, and
        # the run fails at the first read. Not redundant with STRICT mode's every-month rule
        # either — an assessed window can explain every absent month away (see below) and empty
        # the missing list entirely, so a store the ingest examined and wrote nothing into would
        # sail through the one gate that exists to fail before provisioning.
        if not (present_months & required_months):
            msg = (
                f"{label} store at {path} has no timestamps within the window "
                f"{earliest[0]}-{earliest[1]:02d}..{latest[0]}-{latest[1]:02d}"
            )
            raise InsufficientCoverageError(msg)

        # Require EVERY month of the window, not just that the min/max span it: a mosaic with only
        # January + December (or out-of-window dates bracketing the year) would otherwise pass
        # despite missing every intervening month, and the write-once tag makes that partial year
        # permanent. Month granularity matches the campaign's calendar-year window.
        missing = sorted(required_months - present_months)

        # Split the absence into EXPLAINED and UNEXPLAINED for the record, before any decision to
        # raise or skip, and for BOTH paths. A month the ingest examined and found empty is
        # legitimately absent — common for radar, where a zone's orbit may reach its land on no
        # date of a month — and a count of present months cannot tell it from a hole. Anything
        # alarming on this must key on the UNEXPLAINED count, or it fires on healthy cells.
        _assessed = root.attrs.get(ASSESSED_WINDOW_ATTR)
        # A month wholly inside the assessed window is EXPLAINED: the ingest looked, and WHY a day
        # is absent is not a distinction this gate can act on — an unreadable day and a cloudy one
        # are the same absence downstream, and the published `*_month_covered` masks record the
        # month as uncovered per pixel either way.
        #
        # The time axis only grows, so a month below the store's newest date is closed for good. A
        # TRAILING month is not (a resume starts at the newest date plus one and would re-offer
        # it), and is excused anyway: two loss paths can leave one, neither worth blocking a cell
        # for.
        #
        # * A date given up because `is_unreadable_source` says the bytes are permanently bad. That
        #   verdict RECOMPUTES, so re-offering recovers nothing the provider has not republished.
        # * A date refused as `producer-conflict`, where no single BOA-offset decision fits the
        #   day. That one is OURS to fix — classify the bucket in `ingest/asset_locations.py` — but
        #   a whole absent month means EVERY date in it was refused, a catalogue-wide event that
        #   announces itself: the leg ends with a DATA LOSS SUMMARY naming the remedy. The fix is a
        #   code change and a re-ingest either way, which blocking one cell's inference does not
        #   surface.
        _explained = _months_within_assessed(missing, _assessed) if missing else set()
        # The ingest's own account of what it looked at and did not keep, carried onto the year
        # because it lives on the MOSAIC and the mosaic is deleted once a cell lands. Empty dates
        # are the cloud-and-footprint answer — dates examined that yielded nothing — which turns
        # "why is this year thin?" into a read.
        summary["stores"][label].update(
            assessed_window=list(_assessed) if isinstance(_assessed, (list, tuple)) else None,
            assessed_empty_dates=int(root.attrs.get("assessed_empty_dates") or 0),
            months_absent=len(missing),
            months_absent_examined=len(_explained),
            months_absent_unexplained=len(missing) - len(_explained),
        )

        if skip_coverage_check:
            continue

        # A month absent because the ingest EXAMINED it and found nothing reachable is a finding;
        # absent because the ingest never covered it is a gap, and only the second is an error.
        # `assessed_window` tells them apart: the ingest records the range it processed in full, so
        # a month wholly inside that range was looked at. (A satellite pass covers a swath, not a
        # whole UTM zone, and some zones have an orbit reaching their land on no date of the year.)
        # COVERED ENTIRELY is required — a partially-assessed month could hide unexamined days —
        # which costs nothing on the campaign's calendar-year windows.
        if missing:
            # Reuses the split computed above rather than re-deriving it: the record and the gate
            # must never differ about which months were examined.
            assessed, examined = _assessed, _explained
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

    # Honest about WHICH rule passed. The relaxed path reaches here too, having verified
    # only that each store holds something in the window — claiming "every month present"
    # there would assert exactly the thing that was not checked.
    logger.info(
        "Time window coverage verified (%s): %d-%02d through %d-%02d — %s",
        "non-empty only, month rule RELAXED" if skip_coverage_check else "every month present",
        earliest[0],
        earliest[1],
        latest[0],
        latest[1],
        ", ".join(f"{k} {v['months_present']}/{len(required_months)} mo" for k, v in summary["stores"].items()),
    )
    return summary


# ---------------------------------------------------------------------------
# Main loading
# ---------------------------------------------------------------------------


def _empty_sar_arrays(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return empty SAR band, DOY and month arrays for a skipped orbit direction.

    Length zero rather than absent, so a skipped orbit flows through the same derivation as a
    present one: :func:`coverage_from_validity` over an empty mask yields a zero count and an
    all-False month mask — what a pixel that orbit never saw should read.
    """
    return (
        np.empty((0, height, width, 2), dtype=np.uint16),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int16),
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

    Reads SCL for the entire chunk extent, computes the per-pixel obs count from the UN-PRUNED
    mask, then drops timesteps with no valid pixel anywhere in the chunk. The result is shared
    by every strip's band load (sliced, not re-read) and sizes the strip height.

    SCL is 1 byte/pixel against reflectance's 20, so loading the whole chunk's mask up front is
    a small fixed cost that removes per-strip SCL re-decompression. See :class:`S2MaskBundle`.
    """
    if store_opener is None:
        store_opener = open_store_as_zarr_group
    root = store_opener(f"{mosaic_base}/reflectance.zarr")
    window_indices, doys_full, months_full = _filter_times_from_zarr(root, time_window)
    y_slice = slice(chunk.y_start, chunk.y_stop)
    x_slice = slice(chunk.x_start, chunk.x_stop)

    mask_full = _load_scl_mask(root, window_indices, y_slice, x_slice)
    t_full = mask_full.shape[0]
    # Both from the full (PRE-PRUNE) mask, so pixels aren't under-counted, and both from the SAME
    # mask, so "how many" and "which months" cannot disagree about what counted.
    obs_count, month_covered = coverage_from_validity(mask_full, months_full)
    logger.info("Loaded full-chunk SCL for %d S2 timesteps", t_full)

    # Prune timesteps with no valid pixel anywhere in the chunk — the v1.1 per-pixel resampler
    # only draws from valid indices, so no strip would ever read them.
    nonempty_t = mask_full.any(axis=(1, 2))
    kept = np.where(nonempty_t)[0]
    if len(kept) < t_full:
        logger.info("Pruned %d/%d empty S2 timesteps (chunk-level)", t_full - len(kept), t_full)

    return S2MaskBundle(
        mask=mask_full[kept],
        doys=doys_full[kept],
        abs_indices=window_indices[kept],
        obs_count=obs_count,
        month_covered=month_covered,
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
    :func:`load_s2_mask_bundle`). When supplied the SCL is sliced from it rather than re-read —
    chunk-level pruning has already happened, so the band load reads exactly the bundle's kept
    timesteps. ``None`` (the unstriped path) loads and prunes SCL inline.

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

        # obs_count > 0 ⟺ the pruned mask has a valid entry (pruning drops only all-False
        # planes), and the (H, W) scan avoids walking the strided (T_kept, H, W) mask view
        # (~100 ms/chunk on dense multi-strip chunks).
        if not s2_obs_count.any():
            # No valid S2 pixel anywhere in this strip: bucketing would select none, so the
            # 20 B/px band read is pure waste. Return a T=0 band stack — the dataset sees no
            # candidates and the strip short-circuits — while obs counts keep full fidelity
            # from the bundle. Common on sparse/edge chunks whose valid sliver lies in other
            # strips; the chunk-global timestep prune cannot help them (one sliver anywhere
            # keeps the timestep).
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

    window_indices, s2_doys_full, _months = _filter_times_from_zarr(root, time_window)
    y_slice = _resolve_y_slice(chunk, y_sub)

    abs_indices = window_indices
    s2_masks_full = _load_scl_mask(root, abs_indices, y_slice, x_slice)
    t_full = s2_masks_full.shape[0]
    # Compute obs_count from the full mask before pruning so pixels aren't under-counted.
    s2_obs_count = s2_masks_full.sum(axis=0).astype(np.uint16)
    logger.info("Loaded SCL for %d S2 timesteps", t_full)

    # Prune timesteps with no valid pixel anywhere in the chunk — the v1.1 per-pixel resampler
    # only draws from valid indices, so they would never be read.
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

    # Mask stays bool (1 byte/elem) rather than widening to int32: it is resident in ChunkData for
    # the whole chunk, and both consumers (np.nonzero in resampling, .sum for obs_count) take bool.
    return s2_bands, s2_masks, s2_doys, s2_obs_count


def coverage_from_validity(valid: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(obs_count, month_covered)`` from one per-timestep validity mask.

    **The pair is derived together so it cannot disagree, for every sensor.** A count and a month
    mask answer different questions about the same evidence — how many usable observations a pixel
    had, and how they were spread — and the only way they can contradict each other is by resting
    on two different notions of "usable". Taking one validity mask and returning both makes that
    structural: whatever a caller counts as valid is what both outputs describe.

    Args:
        valid: ``(T, H, W)`` boolean-ish mask, True where timestep ``t`` is usable at that pixel.
            Optical passes its SCL mask; radar passes a non-zero test over the polarisation axis.
        months: ``(T,)`` calendar month of each timestep, 1 = January.

    Returns:
        ``obs_count`` ``(H, W)`` uint16, and ``month_covered`` ``(MONTHS_IN_YEAR, H, W)`` bool with
        January first. A month with no timestep at all is False everywhere, which is the same value
        an unwritten pixel reads — absence of evidence and evidence of absence are not distinguished
        here, exactly as a count of 0 does not distinguish them.
    """
    # `dtype=` rather than a trailing `.astype`: summing a bool array defaults to int64, which for
    # a 2048-px chunk is a 33 MB temporary thrown away on the next line.
    obs_count = valid.sum(axis=0, dtype=np.uint16)
    month_covered = np.zeros((MONTHS_IN_YEAR, *obs_count.shape), dtype=bool)

    # Accumulated per TIMESTEP rather than gathered per month: `valid[months == m]` copies that
    # month's timesteps into a fresh array before reducing, so a pass over twelve months copies the
    # whole mask — tens of MB per chunk, times three sensors. OR-ing each timestep into its month's
    # plane touches the same bytes with no temporary, still one vectorised op per iteration.
    if valid.shape[0]:
        index = np.asarray(months, dtype=np.intp) - 1
        if index.min() < 0 or index.max() >= MONTHS_IN_YEAR:
            # Guarded because the failure would otherwise be SILENT: a month of 0 indexes -1, which
            # is December, so a mislabelled timestep would mark the wrong end of the year.
            # `_filter_times_from_zarr` derives these modulo 12, so it should never happen; this
            # says so rather than assuming it.
            raise ValueError(f"month labels out of range 1..{MONTHS_IN_YEAR}: {np.unique(months)}")
        for timestep, month_index in enumerate(index):
            month_covered[month_index] |= valid[timestep]
    return obs_count, month_covered


def _load_sar_orbit(
    mosaic_base: str,
    chunk: ChunkSpec,
    orbit: str,
    time_window: TimeWindow,
    y_sub: slice | None = None,
    store_opener: StoreOpener | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAR bands, DOYs and calendar months for a single orbit direction.

    ``y_sub`` is an optional chunk-relative northing slice (full easting width); ``None`` reads
    the full chunk. The months come free from ``_filter_times_from_zarr`` alongside the DOYs, and
    returning them is what lets radar say WHEN a pixel was observed and not only how often.

    Returns:
        Tuple of (bands, doys, months) with shapes (T, H, W, 2) uint16, (T,) int32 and (T,) int16.
    """
    if store_opener is None:
        store_opener = open_store_as_zarr_group
    root = store_opener(f"{mosaic_base}/sar_{orbit}.zarr")
    window_indices, doys_full, months_full = _filter_times_from_zarr(root, time_window)
    y_slice = _resolve_y_slice(chunk, y_sub)
    x_slice = slice(chunk.x_start, chunk.x_stop)
    bands, kept = _load_sar_bands_from_zarr(root, window_indices, y_slice, x_slice)
    return bands, doys_full[kept], months_full[kept]


def load_chunk(
    chunk: ChunkSpec,
    mosaic_base: str,
    time_window: TimeWindow,
    # Includes "none": `_active_orbits` returns an empty tuple for it, and the S2-only path
    # (ADR-013) reaches here with exactly that.
    s1_orbit: Literal["ascending", "descending", "both", "none"] = "both",
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
        store_opener: Maps a store path to an open zarr group. Defaults to a fresh repo open
            per call; the strip loop passes one shared opener (``make_store_opener``) so all
            strips of a chunk reuse one repo handle.
        y_sub: Optional chunk-relative northing slice bounding the resident input working
            set. Only that horizontal strip is read (full easting width) and the returned
            ``ChunkData`` is a self-contained strip — ``height`` is the strip's, ``width``
            stays full, and pruning / obs counts are computed on the strip's own pixels (so a
            strip may keep a different T_kept than its neighbour). ``None`` reads the whole
            chunk, reproducing the unstriped behaviour byte-for-byte.
        mask_bundle: Optional precomputed full-chunk S2 SCL bundle (see
            :func:`load_s2_mask_bundle`). When supplied, S2 SCL is sliced from it per strip
            instead of re-read and timestep pruning uses the bundle's chunk-level decision.
            ``None`` loads SCL inline.
        reserve_cpus: Cores to leave free during the S2 band decompression — background loads
            running alongside GPU inference reserve cores for the batch-prep workers (see
            :func:`_band_read_workers`); foreground loads use the default 0.
        x_sub: Optional chunk-relative EASTING window (the S2 valid-pixel bounding box on
            sparse/edge chunks). S2 bands — the 20 B/px cost — are read only for these columns
            and the returned grid is cropped to them, mirroring how ``y_sub`` crops rows. SAR
            is still READ full-width so the saved obs-count layers keep full-extent fidelity:
            pre-crop counts come back in ``s1_*_obs_count_full`` while the grid-shaped
            ``s1_*_obs_count`` are cropped like everything else. Pixels outside the box have
            zero valid S2 observations and could never be inferred, so cropping them changes
            which bytes are read, not any output.

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
    s1_asc_bands, s1_asc_doys, s1_asc_months = (
        _load_sar_orbit(
            mosaic_base, chunk, "ascending", time_window=time_window, y_sub=y_sub, store_opener=store_opener
        )
        if "ascending" in active
        else _empty_sar_arrays(height, chunk.width)
    )
    s1_desc_bands, s1_desc_doys, s1_desc_months = (
        _load_sar_orbit(
            mosaic_base, chunk, "descending", time_window=time_window, y_sub=y_sub, store_opener=store_opener
        )
        if "descending" in active
        else _empty_sar_arrays(height, chunk.width)
    )

    # A non-zero sample in either polarisation is the radar validity test; passing that one
    # expression to `coverage_from_validity` makes the count and the month mask two views of one
    # decision rather than two decisions.
    s1_asc_obs_count, s1_asc_month_covered = coverage_from_validity(np.any(s1_asc_bands != 0, axis=-1), s1_asc_months)
    s1_desc_obs_count, s1_desc_month_covered = coverage_from_validity(
        np.any(s1_desc_bands != 0, axis=-1), s1_desc_months
    )

    s1_asc_obs_count_full: np.ndarray | None = None
    s1_desc_obs_count_full: np.ndarray | None = None
    s1_asc_month_covered_full: np.ndarray | None = None
    s1_desc_month_covered_full: np.ndarray | None = None
    if x_sub is not None:
        # SAR was read full-width (see the x_sub docstring): keep the full-width counts for the
        # saved obs layers, then crop the grid-shaped arrays to the S2 grid (ascontiguousarray
        # drops the full-width parents). The month masks travel with their counts — same crop, one
        # axis further right because month leads their shape.
        s1_asc_obs_count_full = s1_asc_obs_count
        s1_desc_obs_count_full = s1_desc_obs_count
        s1_asc_month_covered_full = s1_asc_month_covered
        s1_desc_month_covered_full = s1_desc_month_covered
        s1_asc_obs_count = np.ascontiguousarray(s1_asc_obs_count[:, x_sub])
        s1_desc_obs_count = np.ascontiguousarray(s1_desc_obs_count[:, x_sub])
        s1_asc_month_covered = np.ascontiguousarray(s1_asc_month_covered[:, :, x_sub])
        s1_desc_month_covered = np.ascontiguousarray(s1_desc_month_covered[:, :, x_sub])
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
        s1_asc_month_covered=s1_asc_month_covered,
        s1_desc_month_covered=s1_desc_month_covered,
        s1_asc_month_covered_full=s1_asc_month_covered_full,
        s1_desc_month_covered_full=s1_desc_month_covered_full,
    )
