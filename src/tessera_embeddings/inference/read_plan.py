"""How a chunk is tiled into strips, cropped, and prefetched — the arithmetic, alone.

Split out of :mod:`~tessera_embeddings.inference.actors` because it is a self-contained
subject with no Ray, no torch and no actor state: given a chunk and its SCL mask, decide the
northing strip height, the easting crop, and whether the next chunk is worth prefetching.

It was already being treated as a separate thing from four other modules, which reach for
these names in prose — ``config/inference.py`` and ``data_loading.py`` both cite
``_strip_height_for_density``, ``resource_monitor.py`` cites ``_S2_STRIP_BYTE_BUDGET`` and
``scheduling.py`` cites ``_XCHUNK_PREFETCH_CAP_BYTES``. A block four modules point at by name
is not an implementation detail of the actor.

The budgets here are derived, not chosen. Do not raise one without re-deriving the arithmetic
in ``context_docs/inference/inference-on-gpus.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from tessera_embeddings.config.inference import S2_BAND_ORDER
from tessera_embeddings.inference.chunk_spec import ChunkSpec

if TYPE_CHECKING:
    from tessera_embeddings.inference.data_loading import S2MaskBundle

logger = logging.getLogger(__name__)

# Host-RAM budget (bytes) for one resident S2 band set (a strip's bands + its full-chunk SCL
# mask). The intra-chunk 1-deep strip prefetch keeps TWO such sets resident, so the steady-state
# S2 ceiling is the PAIR: 5.75 GiB/set => ~11.5 GiB. The operating target is peak host RAM under
# 60% of the 30.9 GB usable on a g6e.xlarge:
#   pair ~11.5 GiB (~12.35 GB) + SAR ~1.5 (unmodelled; spikes on dense-S1 chunks) + whole-chunk
#   int8 output buffers ~0.6 + model/torch/misc baseline ~3.3 => ~17.7 GB ~ 57%, leaving ~0.9 GB
#   for sub-30s spikes the RESOURCES poll misses. Measured at 4.75 GiB/set: 34% avg / 51% peak;
#   5.75 lets the whole T<=71 full-width band run as a SINGLE strip (saving a ~13 s fixed read
#   per chunk) while dense T=114 still splits. The cross-chunk starter prefetch adds at most
#   _XCHUNK_PREFETCH_CAP_BYTES (~2 GiB) during the last strip: measured +~6 pts (45-47% -> ~52%).
# The non-hideable strategy (_strip_plan regime 3) may size a strip to the PAIR budget, but only
# with prefetch OFF, so at most ONE such set is resident — the pair ceiling bounds peak either way.
# Do NOT raise this, or reintroduce whole-chunk cross-chunk prefetch, without re-deriving the
# arithmetic above: context_docs/inference/inference-on-gpus.md.
_S2_STRIP_BYTE_BUDGET = int(5.75 * 1024**3)


# Per-(timestep, pixel) byte cost of resident S2 bands: 10 bands x uint16.
_S2_BYTES_PER_OBS_PX = len(S2_BAND_ORDER) * 2


# The foreground S2 band read fans out one decompression thread per core (capped at the 10-band count) — 4 on the
# 4-vCPU g6e.xlarge. Each thread holds a transient single-band (T, strip_h, W) uint16 array ON TOP of the resident
# stacked result, so a read's momentary peak is (10 + readers) bands' worth, not 10. The dense path reads in the
# BACKGROUND and its measured peak already includes that transient; only the pair-budget FOREGROUND read (_strip_plan
# regime 3) must charge it so its momentary peak stays inside the pair ceiling.
_S2_FOREGROUND_DECODE_READERS = min(len(S2_BAND_ORDER), 4)


# Apply the S2 easting-bbox crop only when it removes at least this fraction of the chunk's width. Near-full boxes
# (interior chunks) skip it, keeping the mainline path byte-for-byte identical to the uncropped code rather than
# paying SAR column-copies for a few saved columns.
_X_CROP_MIN_SAVING = 0.10


# Floor on derived strip height. Below this, per-strip fixed overhead (zarr open, SCL slice, dataset bucketing)
# dominates and read amplification climbs without meaningfully lowering peak RAM. A pathologically dense chunk bottoms
# out here — deliberately breaching the byte budget, and logged — rather than degenerating into hundreds of tiny
# reads.
_MIN_STRIP_H = 256


# Strategy-only estimator constants, calibrated from real run logs (a two-strip chunk: 4.85 GB in 19.4 s and 0.19 GB
# in 14.5 s => BW ~950 MB/s, fixed ~14 s/read from re-decompressing the shared 4000^2 storage chunks + zarr open +
# bucketing; inference ~16K px/s, GPU-bound). These pick a strip STRATEGY only — never a correctness value and never a
# RAM bound. A wrong estimate's worst case is the old always-prefetch behaviour, since every strategy respects the
# same resident-pair ceiling (see _strip_plan).
_EST_PX_PER_SEC = 16_000.0


_EST_READ_BYTES_PER_SEC = 900e6


_EST_FIXED_READ_S = 13.0


# Starter strip: a deliberately small first strip so the GPU starts inferring after only the fixed read cost instead
# of a full budget-sized load, with the remainder hiding behind it. Bounded upside (~one budget / BW ~= 7 s); applied
# only when inference comfortably hides the extra read.
_STARTER_STRIP_H = 256


# Resident cap for a stashed payload (mask bundle + starter ChunkData/dataset).
_XCHUNK_PREFETCH_CAP_BYTES = int(2.0 * 1024**3)


# Flat allowance for the starter strip's full-width SAR stack (read full-width for obs-layer fidelity; see
# load_chunk's x_sub docs).
_XCHUNK_SAR_STARTER_EST_BYTES = int(0.4 * 1024**3)


def _strip_height_for_density(
    t_kept: int,
    width: int,
    height: int,
    budget: int = _S2_STRIP_BYTE_BUDGET,
    mask_width: int | None = None,
    decode_readers: int = 0,
) -> int:
    """Largest northing strip height (rows) whose resident S2 working set fits ``budget``.

    Every resident set is charged ``bands(strip_h) + a full SCL mask``; the RAM model lives at
    :data:`_S2_STRIP_BYTE_BUDGET`. ``width`` sizes the (possibly easting-cropped) band read;
    ``mask_width`` sizes the mask, which stays full-chunk-width even when bands are cropped
    (defaults to ``width``). ``decode_readers`` (>0) additionally charges the concurrent per-band
    decode transient (:data:`_S2_FOREGROUND_DECODE_READERS`), so the strip's momentary read peak
    — not just its resident set — fits ``budget``; the default 0 leaves the background-read dense
    path (already measured-safe) unchanged. A chunk dense enough to drive the height below
    ``_MIN_STRIP_H`` bottoms out there and breaches the budget, logged.
    """
    t = max(1, t_kept)
    mask_bytes = t * height * (mask_width if mask_width is not None else width)
    band_budget = budget - mask_bytes
    # Resident stacked result (_S2_BYTES_PER_OBS_PX) plus decode_readers transient single-band arrays (2 B/px each)
    # held concurrently during the read.
    per_row = t * width * (_S2_BYTES_PER_OBS_PX + 2 * decode_readers)
    budget_h = max(0, band_budget) // per_row
    if budget_h < _MIN_STRIP_H:
        # Worst-case resident pair: two floor-height band sets, each charged a full mask (conservative; the
        # intra-chunk pair actually shares one).
        pair_gib = 2 * (_MIN_STRIP_H * per_row + mask_bytes) / 1024**3
        logger.warning(
            "S2 density (T_kept=%d, W=%d, H=%d) drives strip_h=%d below floor "
            "%d; using %d. Resident bands+mask pair ~%.1f GiB exceeds the "
            "budget (2 x %.1f GiB) — raise _S2_STRIP_BYTE_BUDGET or expect high "
            "host RAM.",
            t_kept,
            width,
            height,
            budget_h,
            _MIN_STRIP_H,
            _MIN_STRIP_H,
            pair_gib,
            budget / 1024**3,
        )
        return _MIN_STRIP_H
    return min(height, budget_h)


def _strip_slices(height: int, strip_h: int, start: int = 0) -> list[slice]:
    """Tile ``[start, height)`` into chunk-relative northing strips of ``strip_h`` rows.

    The final strip is shorter when the span is not a multiple of ``strip_h``.
    ``strip_h >= height - start`` yields a single strip; ``start >= height``
    yields ``[]``. A non-zero ``start`` tiles the body after a starter strip.
    """
    return [slice(s, min(s + strip_h, height)) for s in range(start, height, strip_h)]


@dataclass
class _StripPlan:
    """How to tile a chunk into northing strips, and whether to prefetch them.

    ``prefetch`` gates the intra-chunk 1-deep load pipeline: True only when inference is
    estimated long enough to hide a background strip load; otherwise loads run serially (never
    two strips co-resident) and strips may use the PAIR budget. ``strategy`` is a log label and
    ``strip_h`` the logged body height — neither drives behaviour.
    """

    strips: list[slice]
    prefetch: bool
    strategy: str
    strip_h: int
    # True when a strip was sized to the PAIR budget (~2x): the last strip then holds a near-2x set, NOT a RAM trough,
    # so the cross-chunk prefetch must be skipped for this chunk.
    pair_budget: bool = False
    # True when strips[0] is already the small starter strip.
    starter_first: bool = False


def _strip_plan(t_kept: int, height: int, width: int, valid_px: int, mask_width: int | None = None) -> _StripPlan:
    """Choose a strip tiling + prefetch strategy for a chunk.

    Bytes scale with ``t_kept x height x width``; inference time scales with valid pixels — they
    diverge, so the plan is per chunk: fits-one-budget -> single strip; split + hideable (dense)
    -> balanced strips + prefetch, with a small starter strip when inference absorbs the extra
    fixed read; split + not hideable (wide but few valid px) -> prefetch OFF at the PAIR budget
    (only one set ever resident). RAM safety does NOT depend on the estimates: every branch
    respects the pair ceiling (arithmetic at :data:`_S2_STRIP_BYTE_BUDGET`), so a mis-estimate
    costs only speed. ``mask_width`` is the full-chunk mask width when bands are easting-cropped.
    """
    budget = _S2_STRIP_BYTE_BUDGET
    mw = mask_width if mask_width is not None else width
    mask_bytes = t_kept * height * mw
    bytes_total = t_kept * height * width * _S2_BYTES_PER_OBS_PX
    per_set_1x = bytes_total + mask_bytes

    # Regime 1: the whole chunk fits one budget — one strip, nothing to prefetch.
    if per_set_1x <= budget:
        return _StripPlan([slice(0, height)], prefetch=False, strategy="single", strip_h=height)

    # Split needed. Will inference hide the loads? Compare estimated total inference time against estimated total load
    # time (fixed per-read cost x number of 1x-budget strips, plus bytes / bandwidth).
    n_dense = (per_set_1x + budget - 1) // budget
    t_infer = valid_px / _EST_PX_PER_SEC
    t_load = n_dense * _EST_FIXED_READ_S + bytes_total / _EST_READ_BYTES_PER_SEC

    if t_infer >= t_load:
        # Regime 2: hideable (dense). Balanced strips at one budget, prefetch on.
        strip_h = _strip_height_for_density(t_kept, width, height, budget, mask_width=mw)
        strips = _strip_slices(height, strip_h)
        # P2 starter strip: only when there is a real body to hide behind and inference comfortably absorbs the one
        # extra fixed read it introduces.
        starter = len(strips) >= 2 and strip_h > _STARTER_STRIP_H and t_infer >= t_load + _EST_FIXED_READ_S
        if starter:
            strips = [slice(0, _STARTER_STRIP_H), *_strip_slices(height, strip_h, start=_STARTER_STRIP_H)]
        return _StripPlan(
            strips,
            prefetch=True,
            strategy="dense/prefetch+starter" if starter else "dense/prefetch",
            strip_h=strip_h,
            starter_first=starter,
        )

    # Regime 3: not hideable. Prefetch off => only one set resident, read in the FOREGROUND (GPU idle), so the
    # momentary peak is that set's bands + its concurrent decode transient + mask. Size so THAT fits the pair budget —
    # the same measured-safe ceiling the dense resident pair sits at — rather than sizing the resident set alone to
    # the pair and overshooting mid-read.
    pair = 2 * budget
    readers = _S2_FOREGROUND_DECODE_READERS
    decode_transient = bytes_total * 2 * readers // _S2_BYTES_PER_OBS_PX
    if bytes_total + decode_transient + mask_bytes <= pair:
        return _StripPlan(
            [slice(0, height)], prefetch=False, strategy="single/wide-budget", strip_h=height, pair_budget=True
        )
    strip_h = _strip_height_for_density(t_kept, width, height, pair, mask_width=mw, decode_readers=readers)
    return _StripPlan(
        _strip_slices(height, strip_h), prefetch=False, strategy="no-prefetch", strip_h=strip_h, pair_budget=True
    )


def _chunk_read_plan(chunk: ChunkSpec, mask_bundle: S2MaskBundle) -> tuple[slice | None, int, _StripPlan]:
    """Crop bbox, valid-pixel count, and strip plan derived from the SCL mask.

    Shared by the serial prologue and the cross-chunk starter prefetch so both make identical
    decisions from the same inputs — a prefetched chunk must tile and crop exactly as it would
    have serially.
    """
    t_kept = int(mask_bundle.mask.shape[0])
    # (H, W): pixels with >=1 valid S2 observation. Drives the easting crop bbox and the strip plan's inference-time
    # estimate. Equivalent to mask.any(axis=0) — obs_count sums the pre-prune mask and pruning drops only all-False
    # timestep planes — but reads (H, W) instead of scanning the full (T_kept, H, W) mask (up to ~1 GB) once per
    # chunk.
    valid_any = mask_bundle.obs_count > 0

    # S2 valid-pixel bounding box in easting. On sparse/edge chunks (a coastline sliver, a UTM zone boundary) the
    # valid columns can be a small fraction of the width, and the S2 band read (20 B/px) shrinks to the box. Columns
    # outside it have zero valid S2 observations and could never be inferred; the saved obs layers keep full extent
    # via the bundle (S2) and full-width SAR reads (see load_chunk's x_sub docs). Near-full boxes skip the crop so
    # interior chunks stay on the byte-identical uncropped path.
    x_sub: slice | None = None
    valid_cols = np.flatnonzero(valid_any.any(axis=0))
    if valid_cols.size:
        box = slice(int(valid_cols[0]), int(valid_cols[-1]) + 1)
        if (box.stop - box.start) <= (1 - _X_CROP_MIN_SAVING) * chunk.width:
            x_sub = box
            logger.info(
                "Chunk %s: S2 valid bbox covers columns %d-%d (%.0f%% of width) — cropping reads",
                chunk.label,
                box.start,
                box.stop,
                100.0 * (box.stop - box.start) / chunk.width,
            )

    effective_width = chunk.width if x_sub is None else (x_sub.stop - x_sub.start)
    valid_px = int(valid_any[:, x_sub].sum()) if x_sub is not None else int(valid_any.sum())
    # Bands read at effective_width (possibly cropped); the SCL mask stays full chunk width, so charge it at
    # chunk.width in the budget.
    plan = _strip_plan(t_kept, chunk.height, effective_width, valid_px, mask_width=chunk.width)
    return x_sub, valid_px, plan


def _xchunk_rung(chunk: ChunkSpec, t_kept: int, x_sub: slice | None, valid_px: int, plan: _StripPlan) -> str:
    """Choose the prefetch rung for a hinted next chunk: "starter" or "mask-only".

    The starter rung requires ``plan.strips[0]`` to be the SMALL 256-row starter: either the plan
    already starts small (``starter_first``, no extra read), or it is a one-budget single strip
    convertible to starter+body — which PAYS one extra fixed read, so it is worth it only when
    the starter's own inference hides a meaningful slice of the body read (the net-gain check).
    Everything else gets the mask: budget-sized first strips are over the cap by construction,
    and pair-budget plans have too few valid pixels to hide the extra read. Finally the byte cap:
    the resident mask + starter estimate must fit ``_XCHUNK_PREFETCH_CAP_BYTES``.
    """
    if plan.starter_first:
        candidate = True
    elif len(plan.strips) == 1 and not plan.pair_budget and chunk.height > _STARTER_STRIP_H:
        starter_infer_s = (_STARTER_STRIP_H / chunk.height) * (valid_px / _EST_PX_PER_SEC)
        candidate = starter_infer_s >= _EST_FIXED_READ_S
    else:
        candidate = False
    if not candidate:
        return "mask-only"

    effective_width = chunk.width if x_sub is None else (x_sub.stop - x_sub.start)
    mask_bytes = t_kept * chunk.height * chunk.width
    starter_bytes = t_kept * _STARTER_STRIP_H * effective_width * _S2_BYTES_PER_OBS_PX + _XCHUNK_SAR_STARTER_EST_BYTES
    if mask_bytes + starter_bytes > _XCHUNK_PREFETCH_CAP_BYTES:
        return "mask-only"
    return "starter"
