"""Ray actor for distributed GPU inference.

Each actor loads the model once, then processes multiple spatial chunks sequentially,
amortizing model loading and GPU warm-up across chunks.

torch and the modules that import it (models.builder, dataset, inference) are deferred to
``__init__`` / method bodies so this module imports on the Fargate flow runner, which has ray
but not torch. Ray serializes only the actor class reference; torch is needed on GPU workers.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fsspec
import numpy as np
import ray
import requests

from tessera_embeddings.config.inference import (
    EMBEDDING_DIM,
    OPTICAL_MIN_OBS,
    PREFETCH_DEPTH,
    RADAR_THIN_MAX_OBS,
    S1_ORBIT_NONE,
    S2_BAND_ORDER,
    TUNED_GPU_GIB,
    TUNED_TOKENS_PER_PIXEL,
    InferenceConfig,
    S1Orbit,
    batch_size_for_gpu,
)
from tessera_embeddings.config.store_layout import MONTH_COVERED_VARS, MONTHS_IN_YEAR
from tessera_embeddings.config.time_windows import TimeWindow
from tessera_embeddings.inference.assembly import OBS_COUNT_VARS, ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.data_loading import load_chunk, load_s2_mask_bundle, make_store_opener
from tessera_embeddings.inference.progress import chunk_uid
from tessera_embeddings.inference.resource_monitor import ResourceMonitor
from tessera_embeddings.storage.zarr_store import credentials_provider

if TYPE_CHECKING:
    import types
    from collections.abc import Callable

    import icechunk
    import torch

    from tessera_embeddings.inference.data_loading import ChunkData, S2MaskBundle, StoreOpener
    from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
    from tessera_embeddings.inference.models.ssl_model import MultimodalBTInferenceModel


@dataclass
class _ChunkPrologue:
    """Everything ``_process_chunk`` needs before its first forward pass.

    Built serially (GPU idle) by :meth:`InferenceActor._load_chunk_prologue`, or ahead of time
    by the bounded cross-chunk prefetch (the ``_XCHUNK_*`` constants) — in which case
    ``first_strip`` may still be None (mask-only rung) and the consumer loads it serially.
    """

    store_opener: StoreOpener
    mask_bundle: S2MaskBundle
    plan: _StripPlan
    # Chunk-relative easting window of the S2 valid-pixel bounding box, or None for full width. Sparse/edge chunks
    # read (and infer) only these columns; outputs are placed at this offset in the whole-chunk buffers.
    x_sub: slice | None
    # (ChunkData, dataset) for plan.strips[0]; None until loaded.
    first_strip: tuple[ChunkData, MosaicChunkInferenceDataset] | None
    # Prefetch rung ("starter" | "mask-only") for the hit log; None = serial.
    rung: str | None = None


logger = logging.getLogger(__name__)


def _chunk_summary_line(**fields: Any) -> str:  # noqa: ANN401 — heterogeneous JSON payload
    """One machine-readable per-chunk line for the profiling tools.

    ``te-observe-cluster --report`` parses these rather than prose log lines, whose wording
    drifts across branches and silently breaks regex parsers. Keep the keys stable, or update
    that parser in the same change.
    """
    return "CHUNK_SUMMARY: " + json.dumps(fields, sort_keys=True)


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
# arithmetic above: context_docs/design/inference_gpu_saturation_profile_2026_07.md.
_S2_STRIP_BYTE_BUDGET = int(5.75 * 1024**3)

# Per-(timestep, pixel) byte cost of resident S2 bands: 10 bands x uint16.
_S2_BYTES_PER_OBS_PX = len(S2_BAND_ORDER) * 2

# The foreground S2 band read fans out one decompression thread per core (capped at the 10-band count) — 4 on the
# 4-vCPU g6e.xlarge. Each thread holds a transient single-band (T, strip_h, W) uint16 array ON TOP of the resident
# stacked result, so a read's momentary peak is (10 + readers) bands' worth, not 10. The dense path reads in the
# BACKGROUND and its measured peak already includes that transient; only the pair-budget FOREGROUND read (_strip_plan
# regime 3) must charge it so its momentary peak stays inside the pair ceiling.
_S2_FOREGROUND_DECODE_READERS = min(len(S2_BAND_ORDER), 4)

# Cores left free for the inference loop's batch-prep workers while a BACKGROUND strip load decompresses bands: one
# per prep worker, tied BY DEFINITION to the prep pool size so a PREFETCH_DEPTH change cannot silently re-introduce
# the ~500 ms get_batch starvation this prevents on the 4-vCPU g6e.xlarge. Foreground (prologue) loads reserve nothing
# — the GPU is idle and wants the fastest load.
_BACKGROUND_LOAD_RESERVED_CPUS = PREFETCH_DEPTH

# Hard ceiling on any in-actor wait for a background I/O future — a strip / starter prefetch load, or the prior
# chunk's deferred staging write. Normal case is seconds of S3/zarr I/O (a full strip is ~6.5 s at measured BW), so a
# wait this long means the client is wedged with no socket timeout. Bounding it lets process_chunk fail out (or fall
# back to a serial load) so Ray surfaces the error and the scheduler kills + replaces the actor and requeues —
# recovery that flush_writes() provides for an IDLE actor but that a wedge INSIDE process_chunk never reaches (Ray
# serialises actor calls; a 1-2 actor run never hits the >=3-stall abort). Matches the scheduler's flush_writes() RPC
# timeout.
_BACKGROUND_IO_TIMEOUT_S = 600.0

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

# ---------------------------------------------------------------------------
# Bounded cross-chunk starter prefetch ("interleaving lite")
# ---------------------------------------------------------------------------
# The scheduler reserves each actor's next chunk and passes it as ``prefetch_hint``; during the
# CURRENT chunk's last strip (its final load is complete by then — the RAM trough, temporally
# separated from the mid-chunk two-strip peak), the actor prefetches a hard-capped payload for
# the next chunk: its SCL mask bundle and, when the cap and a net-gain check allow, its 256-row
# starter strip. This is NOT the removed full interleaving, which co-resided an entire next
# working set (5-7+ GiB) and OOM-killed a worker at 92-95% RAM; this co-resides <= ~2 GiB by
# construction, and every failure (cap exceeded, label mismatch after a steal, credential-window
# expiry, load error) degrades to the serial prologue — slower, never bigger. Escape hatch:
# TESSERA_DISABLE_XCHUNK_PREFETCH=1.
_XCHUNK_DISABLE_ENV = "TESSERA_DISABLE_XCHUNK_PREFETCH"
# Resident cap for a stashed payload (mask bundle + starter ChunkData/dataset).
_XCHUNK_PREFETCH_CAP_BYTES = int(2.0 * 1024**3)
# The full-SCL read transiently holds t_all x H x W bytes before pruning; skip prefetching entirely for chunks where
# even that transient is outsized.
_XCHUNK_MASK_TRANSIENT_CAP_BYTES = int(1.5 * 1024**3)
# Upper-bound estimate of pre-prune S2 timesteps in a 12-month window (~5-day revisit x tile overlap); used only for
# the transient precheck above.
_XCHUNK_T_ALL_EST = 230
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


def _fetch_ec2_instance_id() -> str:
    """Fetch EC2 instance ID from instance metadata (IMDSv2).

    Returns "unknown" if metadata is unavailable (e.g., local development).
    """
    try:
        token = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=2,
        ).text

        return requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        ).text
    except Exception:
        return "unknown"


def _configure_actor_logging() -> None:
    """Configure logging for inference actors.

    Sets root logger to INFO with a standard format, silences noisy third-party
    loggers, and enables DEBUG for inference profiling modules.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    for noisy in ("zarr", "zarr.group", "icechunk", "botocore", "s3transfer", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # Derived from this module's package so the names survive renames: a hardcoded, stale "src.inference.*" prefix
    # once silently disabled every DEBUG diagnostic (TIMING, EFFECTIVE TFLOPS, autocast probe).
    pkg = __name__.rsplit(".", 1)[0]  # tessera_embeddings.inference
    for name in (
        f"{pkg}.inference",
        f"{pkg}.profiling",
        f"{pkg}.models.modules",
        f"{pkg}.models.ssl_model",
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)


def _select_device(torch_mod: types.ModuleType, instance_id: str) -> torch.device:
    """Detect CUDA availability and return the appropriate torch device.

    Args:
        torch_mod: The torch module (passed to avoid top-level import).
        instance_id: EC2 instance ID for log messages.
    """
    if torch_mod.cuda.is_available():
        device = torch_mod.device("cuda")
        gpu_name = torch_mod.cuda.get_device_name(0)
        logger.info("InferenceActor on instance %s, GPU: %s", instance_id, gpu_name)
    else:
        device = torch_mod.device("cpu")
        logger.warning("InferenceActor on instance %s: no GPU available, using CPU", instance_id)
    return device


def _gpu_total_gib(torch_mod: types.ModuleType, device: torch.device) -> float | None:
    """This actor's card size in GiB, or ``None`` when there is no card to measure.

    Args:
        torch_mod: The torch module (passed to avoid top-level import).
        device: The device this actor selected.
    """
    if device.type != "cuda":
        return None
    return float(torch_mod.cuda.get_device_properties(0).total_memory) / 1024**3


def _log_vram_breakdown(model: MultimodalBTInferenceModel, torch_mod: types.ModuleType) -> None:
    """Log VRAM usage breakdown after model loading."""
    allocated = torch_mod.cuda.memory_allocated() / 1024**3
    reserved = torch_mod.cuda.memory_reserved() / 1024**3
    total = torch_mod.cuda.get_device_properties(0).total_memory / 1024**3
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    logger.info(
        "VRAM after model load: allocated=%.2f GB, reserved=%.2f GB, total=%.2f GB, "
        "model params=%.2f GB, model buffers=%.2f GB",
        allocated,
        reserved,
        total,
        param_bytes / 1024**3,
        buffer_bytes / 1024**3,
    )


def _coverage_record(
    chunk: ChunkSpec,
    *,
    refused: Mapping[str, int],
    radar_rule_enforced: bool | None,
    obs_buffers: Mapping[str, np.ndarray],
    x_sub: slice | None,
    optical_min_obs: int,
) -> dict[str, Any]:
    """One shard's refusal reasons and observation depth, in the shape the registry consumes.

    Built here rather than at each of the two call sites — a wholly-refused shard writes it into
    its skip marker, a shard that embedded something returns it with its result — because two
    copies of these expressions is how two rows in one registry stop being comparable. Comparing
    them is the point: a shard 60% refused for thin optical is a better revisit candidate than
    many that were refused outright.

    Every count is over the footprint the reasons were COUNTED over (``eligible_px``), not over
    the embedded pixels: an infill planner asks about the land, and a denominator that shrinks
    with the answer cannot support the comparison. ``chunk_px`` sits beside it so a reader can
    see how much of the shard was never evaluated at all.

    ``x_sub`` IS THAT COLUMN RANGE, which is why this takes the slice rather than its width, and
    every buffer here is sliced by it. The obs buffers are deliberately allocated and filled at
    FULL chunk width on both paths so the published arrays keep full-extent fidelity; summing
    them whole would describe a footprint the refusal counts do not cover, inflating every
    statistic and potentially pushing a count above its own ``eligible_px`` denominator — telling
    a consumer more imagery could repair a shard when that imagery was never in question.
    """
    cols = x_sub if x_sub is not None else slice(None)
    eff_w = int(chunk.width) if x_sub is None else int(x_sub.stop - x_sub.start)
    s2_obs = obs_buffers["s2_obs_count"][:, cols]
    any_obs = s2_obs > 0
    return {
        "refused": dict(refused),
        "eligible_px": int(chunk.height) * int(eff_w),
        "chunk_px": int(chunk.height) * int(chunk.width),
        # The grid this record was produced on. An assembly-only resume of an all-skipped run has no staged tile to
        # read a chunk size off, and falling back to the CURRENT default re-enumerates the labels — after which valid
        # old markers read as unexpected.
        "chunk_side_px": int(chunk.width),
        # None when no strip ran far enough to say — unknown rather than guessed from the config, whose value the
        # per-cell orbit downgrade can override.
        "radar_rule_enforced": radar_rule_enforced,
        "s2_obs": {
            "px_with_any": int(any_obs.sum()),
            "max": int(s2_obs.max()),
            "mean_where_any": round(float(s2_obs[any_obs].mean()), 2) if any_obs.any() else 0.0,
            # MEDIAN beside the mean: a cleanup planner asks "would another scene or two cross the line for most of
            # this tile", and a mean is pulled by a bright patch of deep pixels next to a dark majority. Over pixels
            # that saw ANYTHING, not over the land — including never-imaged ones drags it to zero and it then
            # describes neither population.
            "median_where_any": (round(float(np.median(s2_obs[any_obs])), 1) if any_obs.any() else 0.0),
            # AND THE DEPTH OF THE PIXELS THAT FELL SHORT, which is what an infill planner ranks by.
            # `median_where_any` covers the whole evaluated footprint, so on a PARTLY refused shard it is dominated by
            # the pixels that passed: 09S/2021 published tiles with a median of 50 against a line of 15 while still
            # refusing several thousand pixels each, so ranking by it put marginal pixels in deep imagery above
            # uniformly thin shards. Taken over pixels strictly below the line and above zero — the thin population by
            # the rule's own definition. None when the shard has none, which is different from a shard whose thin
            # pixels sit at zero.
            "median_where_thin": (
                round(float(np.median(s2_obs[thin])), 1)
                if (thin := (any_obs & (s2_obs < optical_min_obs))).any()
                else None
            ),
        },
        # RADAR PRESENCE: a tile that is thin AND radar-free is a different cleanup candidate from one that is merely
        # thin, since more optical will not fix the first. Either orbit counts — which one it was is per-pixel in the
        # coverage arrays.
        "px_with_any_radar": int(
            ((obs_buffers["s1_asc_obs_count"][:, cols] > 0) | (obs_buffers["s1_desc_obs_count"][:, cols] > 0)).sum()
        ),
    }


def _accelerator_index() -> str | None:
    """This actor's own GPU index on its host, from Ray's assignment. ``None`` if unassigned.

    Ray sets ``CUDA_VISIBLE_DEVICES`` per actor, so TORCH already sees only this actor's device
    and calls it 0. ``nvidia-smi`` does NOT honour that variable — it reports every GPU on the
    host — so anything shelling out to it needs the real host-level index. On a one-GPU host the
    answer is always "0", which is why the bug this prevents survived until a 4-GPU host was
    tried.

    Returns the FIRST assigned id as a string: an ``InferenceActor`` reserves exactly one GPU
    (``num_gpus=1``), so a second id would mean the reservation changed and a single-GPU reader
    would be wrong anyway.
    """
    try:
        ids = ray.get_runtime_context().get_accelerator_ids().get("GPU", [])
    except Exception:  # No Ray runtime (local runner, unit tests)
        return None
    return str(ids[0]) if ids else None


# Memory env vars, all on runtime_env rather than the decorator process, so Ray exports them into the worker
# environment BEFORE this module imports torch / instantiates the C allocator — each is read once at init and ignored
# if set later. On the decorator so they cover every creation site: the initial pool, the ActorPool replacement path
# (scheduling.py), and the local CPU runner.
@ray.remote(
    runtime_env={
        "env_vars": {
            # Caps the cuBLAS host-side workspace. The model runs two CUDA streams (one per backbone), so the default
            # workspace is reserved twice and inflates the per-chunk host-RAM plateau. cuBLAS reads this only when the
            # first CUDA handle is created.
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
            # glibc heap retention: the per-sub-batch prefetch thread churns numpy/torch CPU buffers every ~250 ms,
            # and across multiple arenas the freed regions sit on per-arena free lists at the high-water mark instead
            # of returning to the OS (visible as dirty-anon RSS in an smaps rollup). Two arenas (main + prefetch) with
            # trim-on-free returns that churn.
            "MALLOC_ARENA_MAX": "2",
            "MALLOC_TRIM_THRESHOLD_": "0",
            # Segment-backed allocation, so the caching allocator GROWS a segment instead of reserving a fresh larger
            # one and stranding the old. On the L40S it stops the reserved pool drifting to ~95% of the card to hold a
            # <9 GB working set — measured 42.2 GB reserved for a chunk whose real need was 8.99 GB. It is also what
            # opens the deeper t_s2 rungs on 22.4 GiB cards; see context_docs/design/gpu-card-choice-2026_08.md.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    }
)
class InferenceActor:
    """Ray actor that runs embedding inference on a single GPU or CPU.

    Loads the model checkpoint once at initialization, then processes chunks on demand via
    :meth:`process_chunk`.

    Resource reservations (``num_gpus``, ``num_cpus``, ...) are NOT set on the decorator; callers
    pass them via ``InferenceActor.options(...)`` at ``.remote()`` time so one class serves both
    GPU and CPU deployments::

        # GPU worker (production / CUDA host)
        InferenceActor.options(num_gpus=1).remote(config, ckpt)

        # CPU-only worker (local runner, smoke tests)
        InferenceActor.options(num_gpus=0).remote(config, ckpt)

    The optional ``get_credentials`` callback is the icechunk S3 credential provider used for
    every store open in :meth:`process_chunk`. The cloud-aware caller injects it (the AWS provider
    passes ``iam_icechunk_credentials``) so this domain actor stays free of any cloud-SDK import.
    ``None`` falls back to icechunk's default AWS credential chain — fine for local/moto runs, but
    on long-lived cloud workers that chain can fail to refresh the instance-profile token; see
    ``providers.aws.credentials.iam_icechunk_credentials``.
    """

    def __init__(
        self,
        config: InferenceConfig,
        checkpoint_path: str,
        get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
        s3_region: str | None = None,
    ) -> None:
        """Initialize actor: download checkpoint (if S3) and load model onto GPU.

        Args:
            config: Inference configuration.
            checkpoint_path: S3 URI or local path to the model checkpoint.
            get_credentials: Optional icechunk S3 credential provider applied
                (via :func:`credentials_provider`) for the duration of every
                :meth:`process_chunk` call. Injected by the cloud-aware caller;
                ``None`` uses icechunk's default credential chain.
            s3_region: Optional S3 region for the mosaic repos, applied to every
                store open. ``None`` uses icechunk's default region — a fill in a
                non-default region must thread it so the actor's reads match the
                region the preflight/assembly paths use.
        """
        import torch as _torch

        from tessera_embeddings.inference.models.builder import build_inference_model

        _configure_actor_logging()

        self.config = config
        self._get_credentials = get_credentials
        self._s3_region = s3_region
        self.instance_id = _fetch_ec2_instance_id()
        self.device = _torch.device("cpu") if self.config.num_gpus == 0 else _select_device(_torch, self.instance_id)

        # NARROWED HERE AND NOWHERE ELSE. `batch_size` is read by both the sub-batch split and the pinned host
        # buffers, in different functions; scaling it at either site leaves the other on the tuned value, which is how
        # a card ends up with buffers it cannot fill or batches it cannot hold. One rewrite, before anything reads it.
        gpu_gib = _gpu_total_gib(_torch, self.device)
        fitted = batch_size_for_gpu(
            config.batch_size,
            gpu_gib,
            num_obs_checkpoints=config.num_obs_checkpoints,
            gpu_fraction=config.num_gpus,
        )
        if fitted != config.batch_size:
            logger.info(
                "Batch size %d -> %d on instance %s: %.1f GiB card at %.2f GPU reserved, "
                "deepest bucket %d tokens/px; the default was tuned on %.0f GiB at %d.",
                config.batch_size,
                fitted,
                self.instance_id,
                gpu_gib,
                config.num_gpus,
                2 * max(config.num_obs_checkpoints),
                TUNED_GPU_GIB,
                TUNED_TOKENS_PER_PIXEL,
            )
            self.config = replace(config, batch_size=fitted)

        local_ckpt = download_checkpoint(checkpoint_path) if _is_remote_uri(checkpoint_path) else checkpoint_path
        self.model: MultimodalBTInferenceModel = build_inference_model(
            self.config,
            self.device,
            checkpoint_path=local_ckpt,
        )

        if self.device.type == "cuda":  # No-op on CPU
            _log_vram_breakdown(self.model, _torch)

        # This actor's GPU index ON THE HOST, which is not always 0: Ray packs one actor per GPU, so a 4-GPU host runs
        # four actors holding indices 0-3. Without it the monitor queries every GPU, rejects the multi-row answer, and
        # a packed actor emits no GPU statistics.
        self.gpu_index = _accelerator_index()
        self._resource_monitor = ResourceMonitor(interval_sec=30, gpu_index=self.gpu_index)
        self._resource_monitor.start()
        logger.info("InferenceActor ready on instance %s", self.instance_id)

    def _open_and_plan(
        self, chunk: ChunkSpec, mosaic_base: str, time_window: TimeWindow
    ) -> tuple[StoreOpener, S2MaskBundle, slice | None, int, _StripPlan]:
        """Open stores, load the SCL bundle, and derive the chunk read plan."""
        store_opener = make_store_opener(region=self._s3_region)
        mask_bundle = load_s2_mask_bundle(mosaic_base, chunk, time_window, store_opener=store_opener)
        x_sub, valid_px, plan = _chunk_read_plan(chunk, mask_bundle)
        return store_opener, mask_bundle, x_sub, valid_px, plan

    def _load_strip_dataset(
        self,
        chunk: ChunkSpec,
        mosaic_base: str,
        time_window: TimeWindow,
        s1_orbit: S1Orbit,
        *,
        y_sub: slice,
        store_opener: StoreOpener,
        mask_bundle: S2MaskBundle,
        x_sub: slice | None,
        reserve_cpus: int = 0,
    ) -> tuple[ChunkData, MosaicChunkInferenceDataset]:
        """Load one strip's bands and build its bucketed dataset.

        The single loader shared by the serial prologue, the cross-chunk prefetch, and the strip
        loop — the load/dataset kwargs must stay identical across all three for bit-identity.
        """
        from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset

        # Derived from the ONE threaded orbit, never re-resolved per path: a value each caller computed for itself is
        # exactly how the three would drift out of bit-identity.
        #
        # A radar-free cell has zero S1 observations at every pixel, so the default gate would skip all of them and
        # the cell would write nothing while reporting success. The gate is therefore opened BY the orbit rather than
        # asked of the caller — the rule InferenceConfig applies, applied per cell because the orbit is now per cell.
        allow_s2_only = self.config.allow_s2_only or s1_orbit == S1_ORBIT_NONE
        data = load_chunk(
            chunk,
            mosaic_base,
            time_window=time_window,
            s1_orbit=s1_orbit,
            y_sub=y_sub,
            store_opener=store_opener,
            mask_bundle=mask_bundle,
            reserve_cpus=reserve_cpus,
            x_sub=x_sub,
        )
        dataset = MosaicChunkInferenceDataset(
            data,
            num_obs_checkpoints=self.config.num_obs_checkpoints,
            s1_orbit=s1_orbit,
            allow_s2_only=allow_s2_only,
            optical_min_obs=self.config.optical_min_obs,
        )
        return data, dataset

    def _load_chunk_prologue(
        self, chunk: ChunkSpec, mosaic_base: str, time_window: TimeWindow, s1_orbit: S1Orbit
    ) -> _ChunkPrologue:
        """Load everything _process_chunk needs before its first forward pass.

        Runs inline (serially, GPU idle) at the top of every chunk, UNLESS the bounded cross-chunk
        prefetch staged some of it during the previous chunk's tail (the ``_XCHUNK_*`` constants):
        a label-matching stash supplies the mask bundle + read plan and possibly the first strip,
        and whatever is missing loads serially here.
        """
        prologue = self._take_prefetched(chunk.label)
        if prologue is not None:
            logger.info("xchunk prefetch: hit (%s) for %s", prologue.rung, chunk.label)
            # The stash's opener was created on the prefetch thread; if that thread's store opens outlived the PRIOR
            # chunk's credential scope (the icechunk provider is a process-wide global, not thread-local) its repo
            # handles are bound to the default AWS chain, which can fail to refresh on long-lived workers. The
            # prefetched mask/strip are already-materialised numpy and safe to keep, but the LIVE opener is rebuilt
            # inside THIS call's credential scope so body-strip reads never inherit stale credentials. Worst case: one
            # repo re-open, matching the serial path.
            prologue.store_opener = make_store_opener(region=self._s3_region)
        else:
            store_opener, mask_bundle, x_sub, _valid_px, plan = self._open_and_plan(chunk, mosaic_base, time_window)
            prologue = _ChunkPrologue(store_opener, mask_bundle, plan, x_sub, first_strip=None)
        if prologue.first_strip is None:
            prologue.first_strip = self._load_strip_dataset(
                chunk,
                mosaic_base,
                time_window,
                s1_orbit,
                y_sub=prologue.plan.strips[0],
                store_opener=prologue.store_opener,
                mask_bundle=prologue.mask_bundle,
                x_sub=prologue.x_sub,
            )
        return prologue

    # ------------------------------------------------------------------
    # Bounded cross-chunk starter prefetch
    # ------------------------------------------------------------------
    # Stash keyed by chunk label; lazily initialised so test harnesses building bare actors via
    # object.__new__ keep working without __init__.

    def _prefetch_state(self) -> tuple[ThreadPoolExecutor, dict[str, Future[_ChunkPrologue]]]:
        pool = getattr(self, "_xchunk_prefetch_pool", None)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xchunk-prefetch")
            self._xchunk_prefetch_pool = pool
            self._xchunk_prefetched: dict[str, Future[_ChunkPrologue]] = {}
        return self._xchunk_prefetch_pool, self._xchunk_prefetched

    def _load_prefetched_starter(
        self, chunk: ChunkSpec, mosaic_base: str, time_window: TimeWindow, s1_orbit: S1Orbit
    ) -> _ChunkPrologue:
        """Load the capped prefetch payload for ``chunk`` (prefetch thread).

        Store opens normally land inside the calling process_chunk's scoped credential provider;
        a prefetch that outlives its originating call may open through icechunk's default chain
        instead. If that fails the consumer falls back to an inline, in-scope reload, so a
        credential-window miss degrades to the unprefetched behaviour.
        """
        store_opener, mask_bundle, x_sub, valid_px, plan = self._open_and_plan(chunk, mosaic_base, time_window)
        rung = _xchunk_rung(chunk, int(mask_bundle.mask.shape[0]), x_sub, valid_px, plan)

        first_strip = None
        if rung == "starter":
            if not plan.starter_first:
                # One-budget single-strip plan: convert to starter+body (both <= one budget) so the prefetched piece
                # is small; the body loads behind the starter's inference via the normal strip pipeline. The rung's
                # net-gain check priced the extra read.
                plan = _StripPlan(
                    strips=[slice(0, _STARTER_STRIP_H), slice(_STARTER_STRIP_H, chunk.height)],
                    prefetch=True,
                    strategy="single+xstarter",
                    strip_h=chunk.height - _STARTER_STRIP_H,
                    starter_first=True,
                )
            first_strip = self._load_strip_dataset(
                chunk,
                mosaic_base,
                time_window,
                s1_orbit,
                y_sub=plan.strips[0],
                store_opener=store_opener,
                mask_bundle=mask_bundle,
                x_sub=x_sub,
                reserve_cpus=_BACKGROUND_LOAD_RESERVED_CPUS,
            )
        return _ChunkPrologue(store_opener, mask_bundle, plan, x_sub, first_strip, rung=rung)

    def _start_chunk_prefetch(
        self, chunk: ChunkSpec, mosaic_base: str, time_window: TimeWindow, s1_orbit: S1Orbit
    ) -> None:
        """Kick off the next chunk's capped prefetch on the prefetch thread.

        Called from the strip loop at the top of the CURRENT chunk's last strip — its load is
        complete by then and no body loads remain, so the stash's bytes land on the RAM trough,
        temporally separated from the mid-chunk two-strip peak.
        """
        if os.environ.get(_XCHUNK_DISABLE_ENV):
            logger.info("xchunk prefetch: disabled by env — skipping %s", chunk.label)
            return
        # Transient precheck: the full-SCL read briefly holds t_all x H x W bytes before pruning.
        if _XCHUNK_T_ALL_EST * chunk.height * chunk.width > _XCHUNK_MASK_TRANSIENT_CAP_BYTES:
            logger.info("xchunk prefetch: skipped (cap) — %s mask transient too large", chunk.label)
            return
        pool, stash = self._prefetch_state()
        if chunk.label in stash:
            return
        logger.info("xchunk prefetch: starting for %s", chunk.label)
        stash[chunk.label] = pool.submit(self._load_prefetched_starter, chunk, mosaic_base, time_window, s1_orbit)

    def _take_prefetched(self, label: str) -> _ChunkPrologue | None:
        """Consume the stash for ``label``; evict stale entries.

        Blocks on an in-flight matching prefetch (partial overlap still wins). Returns None — the
        serial prologue — when there is no matching stash or the prefetch ERRORED (worker free,
        degrade gracefully). RAISES if a prefetch TIMED OUT: the prefetch pool is a single
        persistent worker, so a wedged task never frees and would poison every later prefetch
        with a 600 s wait; failing the chunk gets the actor and its pool replaced.
        """
        _, stash = self._prefetch_state()
        for stale in [k for k in stash if k != label]:
            # A steal/requeue changed our next chunk. The stale prefetch's load may still be running; DRAIN it (wait,
            # then drop the result) so its capped stash frees before the reassigned chunk loads — otherwise the stale
            # set and the new load briefly co-reside. Rare (reassignment only); a failed stale prefetch drains to a
            # no-op.
            logger.info("xchunk prefetch: draining stale stash for %s", stale)
            try:
                stash.pop(stale).result(timeout=_BACKGROUND_IO_TIMEOUT_S)
            except TimeoutError as exc:
                raise RuntimeError(f"xchunk prefetch pool wedged draining stale {stale}") from exc
            except Exception as exc:
                logger.warning("xchunk prefetch: stale drain for %s did not complete (%s)", stale, exc)
        future = stash.pop(label, None)
        if future is None:
            return None
        try:
            return future.result(timeout=_BACKGROUND_IO_TIMEOUT_S)
        except TimeoutError as exc:
            raise RuntimeError(f"xchunk prefetch pool wedged loading {label}") from exc
        except Exception as exc:
            # The load errored (worker is free) — fall back to the in-scope serial prologue.
            logger.warning("xchunk prefetch for %s failed (%s); loading inline", label, exc)
            return None

    # ------------------------------------------------------------------
    # Deferred staging writes
    # ------------------------------------------------------------------
    # A chunk's staging upload (~seconds of pure I/O) runs on a single-slot writer thread so it
    # overlaps the NEXT chunk's serial prologue load instead of adding GPU-idle time. Durability
    # protocol: the deferring result carries ``write_deferred=True`` and is NOT counted complete
    # by the scheduler until the write's outcome arrives — piggybacked as ``prior_write`` on this
    # actor's next result, or via ``flush_writes()`` when the actor idles. RAM cost: one chunk's
    # output buffers (~0.6 GB) held until the upload lands. Lazily initialised for bare-actor
    # tests.

    def _writer_pool_handle(self) -> ThreadPoolExecutor:
        pool = getattr(self, "_writer_pool", None)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="staging-write")
            self._writer_pool = pool
            self._pending_write: tuple[str, Future[str]] | None = None
        return pool

    def _collect_prior_write(self) -> dict[str, Any] | None:
        """Resolve the outstanding deferred write, if any.

        Blocks until the upload finishes — by construction it started at least one chunk ago, so
        it is almost always done. Returns ``{"label", "ok", "error"}``, or ``None`` if nothing
        was pending.
        """
        self._writer_pool_handle()
        pending = self._pending_write
        if pending is None:
            return None
        label, future = pending
        self._pending_write = None
        wait_start = time.monotonic()
        try:
            # Bounded: an unbounded wait on a wedged upload hangs process_chunk, so the scheduler never reaches its
            # flush_writes() recovery. On timeout (or any error) return ok=False so the scheduler requeues the
            # deferred chunk on a healthy actor.
            future.result(timeout=_BACKGROUND_IO_TIMEOUT_S)
        except TimeoutError:
            # A bare TimeoutError stringifies to "" — give the scheduler's requeue log a legible reason.
            # timed_out=True flags that the upload is still WEDGED in the single-slot writer pool rather than merely
            # failed: the actor must be replaced, not reused, or later writes queue behind it and time out too.
            err = f"staging upload did not complete within {_BACKGROUND_IO_TIMEOUT_S:.0f}s"
            logger.warning("Deferred staging write for %s FAILED: %s", label, err)
            return {"label": label, "ok": False, "error": err, "timed_out": True}
        except Exception as exc:
            logger.warning("Deferred staging write for %s FAILED: %s", label, exc)
            return {"label": label, "ok": False, "error": str(exc)}
        blocked = time.monotonic() - wait_start
        if blocked > 0.5:
            # The upload did NOT fully hide under the next chunk's prologue. Persistent blocking here means S3 write
            # throughput, not GPU work, is pacing the actor.
            logger.info(
                "Deferred write for %s blocked collection for %.1fs (upload slower than prologue)", label, blocked
            )
        return {"label": label, "ok": True, "error": None}

    def _collect_prior_write_checked(self) -> dict[str, Any] | None:
        """`_collect_prior_write`, but RAISE if the prior upload timed out.

        A timeout means the single-slot writer pool is wedged on a hung upload, so deferring this
        chunk's write behind it would queue another doomed task. Raising fails the whole chunk,
        which gets the scheduler to kill + replace this actor (reaping the wedged writer thread)
        and requeue both this chunk and the orphaned prior write via the standard failure path. A
        normal write error (`ok=False` without `timed_out`) leaves the healthy actor in rotation
        and requeues just the one chunk. Hot path only; `flush_writes` (idle path) keeps the
        non-raising variant so the scheduler's flush handler can replace the slot itself.
        """
        outcome = self._collect_prior_write()
        if outcome is not None and outcome.get("timed_out"):
            msg = f"prior staging write for {outcome['label']} timed out; writer pool wedged"
            raise RuntimeError(msg)
        return outcome

    def flush_writes(self) -> dict[str, Any] | None:
        """Drain the outstanding deferred write (scheduler calls this when the
        actor idles or at end-of-run, when no further result can carry it).
        """
        return self._collect_prior_write()

    def ping(self) -> bool:
        """No-op health check: blocks until ``__init__`` completes (model loaded, GPU ready)."""
        return True

    def get_instance_id(self) -> str:
        """Return the EC2 instance ID this actor is running on."""
        return self.instance_id

    def process_chunk(
        self,
        chunk: ChunkSpec,
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None = None,
        prefetch_hint: ChunkSpec | None = None,
        time_window: TimeWindow | None = None,
        s1_orbit: S1Orbit | None = None,
    ) -> dict[str, Any]:
        """Process a single spatial chunk: load data, run inference, write output.

        When a ``get_credentials`` provider was injected at construction, reads and the staging
        write run inside a scoped :func:`credentials_provider` so every icechunk S3 open resolves
        through it. The AWS provider passes ``iam_icechunk_credentials``, a botocore-backed
        callback carrying an ``expires_after`` so icechunk periodically re-invokes it and
        botocore refreshes the instance-profile token. Without an injected provider, icechunk
        falls back to the Rust AWS SDK's default chain, which resolves the instance-profile
        credential once and — on a long-lived actor — can fail to refresh it, failing this chunk
        and every subsequent one in the process with "no providers in chain provided
        credentials".

        Args:
            chunk: Spatial chunk specification.
            mosaic_base: Base path for the mosaic stores
                (e.g., "s3://bucket/mosaics/small_minnesota").
            staging_base: Base path for staging output.
            run_id: Unique run identifier.
            tracker: Optional ProgressTracker actor handle for batch-level progress.
            prefetch_hint: The chunk the scheduler has reserved as this actor's next assignment;
                its capped starter payload is prefetched during this chunk's last strip (see the
                ``_XCHUNK_*`` constants).
            s1_orbit: This CELL's resolved orbit, overriding the actor's own config: a chained
                session's cells may resolve different orbits because parts of the globe are
                radar-free in principle, and an actor built for ``"both"`` could not otherwise
                serve one. ``None`` uses the actor's config. A ``"none"`` cell also opens the
                S2-only pixel gate, since every one of its pixels has zero S1 observations.
            time_window: This CELL's inference window, overriding the actor's own config.
                Required for a chained session whose cells span campaign years: an actor is built
                once with one config, so reading the window from it would make every cell of a
                different year read the wrong months — and the session's only mismatch check is
                on ``s1_orbit``, so nothing would catch it. ``None`` uses the actor's config,
                which is what the single-ROI path does.

        Returns:
            Result dict with chunk label, status, pixel count, and timing.
        """
        cred_scope = (
            credentials_provider(self._get_credentials)
            if self._get_credentials is not None
            else contextlib.nullcontext()
        )
        with cred_scope:
            return self._process_chunk(
                chunk, mosaic_base, staging_base, run_id, tracker, prefetch_hint, time_window, s1_orbit
            )

    def _process_chunk(
        self,
        chunk: ChunkSpec,
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None = None,
        prefetch_hint: ChunkSpec | None = None,
        time_window: TimeWindow | None = None,
        s1_orbit: S1Orbit | None = None,
    ) -> dict[str, Any]:
        """Run the load → inference → write pipeline for one chunk.

        Always invoked through :meth:`process_chunk`, which establishes the scoped icechunk
        credential provider this body's S3 opens depend on.
        """
        # Resolved ONCE and passed down. The three loader call sites (serial prologue, cross-chunk prefetch, strip
        # loop) must receive identical kwargs for bit-identity, so neither fallback may be re-evaluated at each of
        # them.
        window = time_window if time_window is not None else self.config.time_window
        # A chained session's cells may resolve DIFFERENT orbits, because parts of the globe are radar-free in
        # principle.
        orbit = s1_orbit if s1_orbit is not None else self.config.s1_orbit
        from tessera_embeddings.inference.inference import run_inference

        t0 = time.monotonic()
        # Per CHUNK, not per actor: host RAM scales with a chunk's optical depth, so a peak taken over the actor's
        # life describes its deepest chunk and would be attached to every later one.
        self._resource_monitor.reset_peak_host_ram()
        self._resource_monitor.set_context("work", f"{chunk.label}:prologue")

        # Tracked by the run-qualified uid, not the bare label: labels repeat across zones, and a shared multi-zone
        # session (chained fill) can have two zones' same-labelled chunks in flight at once (see chunk_uid).
        uid = chunk_uid(run_id, chunk.label)

        # Report loading phase so stall detection has visibility before batch 50
        if tracker:
            tracker.report.remote(uid, 0, 0, "loading")  # type: ignore[union-attr]

        try:
            # The prologue — repo handles, full-chunk SCL mask (which sizes the density-based strips), and the first
            # strip's bands + dataset — loads serially here (GPU idle) unless the bounded cross-chunk prefetch staged
            # part of it during the PREVIOUS chunk's tail; see _load_chunk_prologue and the _XCHUNK_* constants.
            prologue = self._load_chunk_prologue(chunk, mosaic_base, window, orbit)
            store_opener = prologue.store_opener
            mask_bundle = prologue.mask_bundle
            plan = prologue.plan
            strips = plan.strips
            x_sub = prologue.x_sub
            # Column window the cropped grids map to in the whole-chunk output buffers (full width when uncropped).
            cols = x_sub if x_sub is not None else slice(0, chunk.width)
            # (chunk_data, dataset) for strips[0]; handed to iteration 0 of the strip loop below, then dropped so at
            # most two strips stay resident.
            first_strip: tuple[ChunkData, MosaicChunkInferenceDataset] | None = prologue.first_strip
            # Captured for the CHUNK_SUMMARY line — mask_bundle and prologue are both deleted before the completion
            # site.
            t_kept = int(mask_bundle.mask.shape[0])
            # Radar sequence lengths, also for the CHUNK_SUMMARY line, and NOT redundant with ``t_kept``: that counts
            # OPTICAL timesteps only, so the token identity built on it (``t_kept x valid_px``) cannot see the two
            # further sequences a radar-bearing chunk's forward pass encodes. Measured at equal optical depth, one
            # orbit costs ~1.3x and both orbits ~2.0x the per-chunk inference time of a radar-free chunk. Without
            # these fields that gap shows only by comparing whole runs.
            #
            # Zero is a real answer, not a missing one: for a radar-free cell it is every chunk, which is what makes
            # the comparison possible.
            t_s1_asc = 0
            t_s1_desc = 0
            rung = prologue.rung or "serial"
            prologue_s = time.monotonic() - t0
            logger.info(
                "Chunk %s: T_kept=%d -> strip_h=%d -> %d strip(s) [%s, prefetch=%s]",
                chunk.label,
                t_kept,
                plan.strip_h,
                len(strips),
                plan.strategy,
                plan.prefetch,
            )
            del prologue

            # Whole-chunk output buffers, allocated once and held for the chunk: only INPUTS are sub-tiled. save_dim
            # mirrors run_inference — the canonical 128-D slice, or the full representation width for smaller (test)
            # models.
            save_dim = min(EMBEDDING_DIM, self.config.representation_dim)
            embeddings = np.zeros((chunk.height, chunk.width, save_dim), dtype=np.int8)
            scales = np.full((chunk.height, chunk.width), np.nan, dtype=np.float32)

            writer = ZarrWriter(staging_base, embedding_dim=save_dim)

            # mask_bundle is an explicit submit() arg rather than a capture, so the loop can `del` the only strong
            # reference once the last strip has loaded (a closure capture cannot be deleted). BACKGROUND loads
            # (prefetch on) reserve cores for the batch-prep workers feeding the GPU; SERIAL loads reserve nothing —
            # the GPU is idle.
            def _load_strip(
                y_sub: slice, bundle: S2MaskBundle, reserve_cpus: int
            ) -> tuple[ChunkData, MosaicChunkInferenceDataset]:
                return self._load_strip_dataset(
                    chunk,
                    mosaic_base,
                    window,
                    orbit,
                    y_sub=y_sub,
                    store_opener=store_opener,
                    mask_bundle=bundle,
                    x_sub=x_sub,
                    reserve_cpus=reserve_cpus,
                )

            on_batch = (
                (lambda b, t: tracker.report.remote(uid, b, t, "inference"))  # type: ignore[union-attr]
                if tracker
                else None
            )

            # Obs counts accumulate per strip into whole-chunk buffers so the single write_chunk carries the
            # full-chunk obs maps.
            obs_buffers: dict[str, np.ndarray] = {
                var: np.zeros((chunk.height, chunk.width), dtype=np.uint16) for var in OBS_COUNT_VARS
            }
            # Which months each pixel was seen in, PER SENSOR, accumulated per strip and keyed the same way. Gates
            # nothing — published so a reader can apply their own view of sufficiency without the imagery, which is
            # deleted after the fill. Month axis first so a strip assigns as `[:, strip, :]`, matching how the mask
            # bundle is sliced. Keyed off the layout rather than named here, so the buffers and the arrays on disk
            # cannot come to disagree about which sensors exist.
            month_buffers: dict[str, np.ndarray] = {
                var: np.zeros((MONTHS_IN_YEAR, chunk.height, chunk.width), dtype=bool) for var in MONTH_COVERED_VARS
            }

            total_valid = 0
            # Refusals accumulate across STRIPS: one dataset is built per strip, so its counters describe a slice
            # rather than the shard, and the per-shard record's invariant is that the parts reach the shard's eligible
            # total.
            refused = {"thin": 0, "no_optical": 0, "no_radar": 0}
            # WAS THE RADAR RULE IN FORCE? Without it `no_radar: 0` means two different things: no tile was refused
            # for missing radar, or that rule was switched off. `allow_s2_only` defaults to FALSE in the library —
            # refusing radar-free land is the default — while the global campaign registers True to embed it, so under
            # campaign settings the count is zero by construction and a reader who took it for a measurement would
            # conclude the campaign found radar everywhere. None until a strip has run, since the effective value
            # includes the per-cell orbit downgrade and only the dataset knows it.
            radar_rule_enforced: bool | None = None
            infer_s = 0.0  # summed wall-clock of the per-strip inference calls
            # Strip pipeline. With prefetch on (dense/hideable), strip i+1 loads and buckets on the background thread
            # while strip i runs inference, so at most two S2 sets are co-resident. With it off (sparse/non-hideable),
            # each strip loads serially in the loop body with the prior set dropped first, so only ONE is ever
            # resident and a strip may safely use the larger pair budget. Strip 0 arrives loaded with the prologue in
            # both modes. Managed explicitly, NOT with `with`: a timed-out strip load raises below, and
            # `ThreadPoolExecutor.__exit__` calls shutdown(wait=True), which re-joins the SAME hung worker and
            # swallows the escape. The finally shuts down non-blocking so the raise reaches the scheduler; the wedged
            # worker leaks but dies with the actor the scheduler then replaces. On the normal path nothing is
            # outstanding at loop exit.
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="strip-prefetch")
            try:
                next_future: Future[tuple[ChunkData, MosaicChunkInferenceDataset]] | None = None

                # Bound to None so the first iteration's `del` has a target; each iteration rebinds both before use.
                chunk_data: ChunkData | None = None
                dataset: MosaicChunkInferenceDataset | None = None
                for i, strip in enumerate(strips):
                    # Split the strip window into :load then :infer (below) so a RESOURCES spike is attributable to
                    # the band decode versus the forward — two different RAM events worth telling apart.
                    self._resource_monitor.set_context("work", f"{chunk.label}:s{i + 1}/{len(strips)}:load")
                    # The previous strip reported "inference"; flip back to "loading" before blocking on this strip's
                    # read so _poll_tracker does not misclassify a slow multi-strip load as an inference stall. (Strip
                    # 0's was reported above.)
                    if tracker and i > 0:
                        tracker.report.remote(uid, 0, 0, "loading")  # type: ignore[union-attr]

                    # Release the previous strip's input arrays BEFORE blocking on this strip's load and submitting
                    # the next prefetch. The dataset retains chunk_data.s2_bands (dataset.py), so without this drop
                    # the prior strip's dataset, the current strip and the prefetched next strip could all be resident
                    # at once — three strips, on a budget sized for two.
                    del chunk_data, dataset

                    if i == 0:
                        assert first_strip is not None
                        chunk_data, dataset = first_strip
                        first_strip = None
                    elif plan.prefetch:
                        assert next_future is not None
                        # Bounded: a wedged background strip read would hang the actor inside process_chunk with no
                        # scheduler recourse. Raising on timeout (with strip context) fails the chunk out so the actor
                        # is replaced and the chunk requeued.
                        try:
                            chunk_data, dataset = next_future.result(timeout=_BACKGROUND_IO_TIMEOUT_S)
                        except TimeoutError as exc:
                            msg = (
                                f"Background strip {i} load for {chunk.label} exceeded {_BACKGROUND_IO_TIMEOUT_S:.0f}s"
                            )
                            raise RuntimeError(msg) from exc
                    else:
                        # Serial: GPU idle, take all cores, one set resident.
                        chunk_data, dataset = _load_strip(strip, mask_bundle, 0)

                    # With prefetch on, kick off the next strip's BACKGROUND load (reserving prep cores) before
                    # inferring this one.
                    if plan.prefetch:
                        next_future = (
                            pool.submit(_load_strip, strips[i + 1], mask_bundle, _BACKGROUND_LOAD_RESERVED_CPUS)
                            if i + 1 < len(strips)
                            else None
                        )

                    # Last strip: its load is complete (bound above) and no body loads remain. For <=1x-budget plans
                    # this is the RAM trough, so prefetch the NEXT chunk's capped starter behind this strip's
                    # inference. Pair-budget plans hold a near-2x set here — no room for the stash — so they take a
                    # serial prologue.
                    if i + 1 == len(strips) and prefetch_hint is not None and not plan.pair_budget:
                        self._start_chunk_prefetch(prefetch_hint, mosaic_base, window, orbit)

                    # The chunk's radar sequence lengths. Every strip sees the same ones — SAR is read full-width
                    # regardless of the crop — so any strip's value is the chunk's and the last assignment wins. A
                    # skipped orbit gets an EMPTY array rather than None (``load_chunk``'s full-width placeholders),
                    # so a length is always defined.
                    t_s1_asc = len(chunk_data.s1_asc_doys)
                    t_s1_desc = len(chunk_data.s1_desc_doys)

                    # SAR month masks: the strip's own when the grid is uncropped, the full-width side channel when it
                    # is — the same choice the SAR obs counts make, so the saved provenance layers keep full extent
                    # even where the inferred grid is narrower. Either way the array is (month, strip rows, full
                    # width), so both assign through the same slice below.
                    sar_months = (
                        (chunk_data.s1_asc_month_covered, chunk_data.s1_desc_month_covered)
                        if x_sub is None
                        else (chunk_data.s1_asc_month_covered_full, chunk_data.s1_desc_month_covered_full)
                    )
                    if x_sub is None:
                        for var in OBS_COUNT_VARS:
                            arr = getattr(chunk_data, var)
                            if arr is not None:
                                obs_buffers[var][strip] = arr
                        if mask_bundle is not None:
                            # Sliced, not assigned whole: the bundle is a WHOLE-CHUNK array loaded once and shared by
                            # every strip, unlike the strip-shaped SAR arrays.
                            month_buffers["s2_month_covered"][:, strip, :] = mask_bundle.month_covered[:, strip, :]
                    else:
                        # Cropped grid: the saved obs layers keep full extent — S2 counts from the (full-width) mask
                        # bundle, SAR counts from the full-width side channel (SAR is read full-width regardless; see
                        # load_chunk).
                        obs_buffers["s2_obs_count"][strip] = mask_bundle.obs_count[strip, :]
                        month_buffers["s2_month_covered"][:, strip, :] = mask_bundle.month_covered[:, strip, :]
                        for var, full in (
                            ("s1_asc_obs_count", chunk_data.s1_asc_obs_count_full),
                            ("s1_desc_obs_count", chunk_data.s1_desc_obs_count_full),
                        ):
                            if full is not None:
                                obs_buffers[var][strip] = full

                    for var, arr in zip(("s1_asc_month_covered", "s1_desc_month_covered"), sar_months, strict=True):
                        if arr is not None:
                            month_buffers[var][:, strip, :] = arr

                    if len(dataset) == 0:
                        # Empty strip: leave its output rows at the initialised zero embeddings / NaN scale, mirroring
                        # run_inference's handling of fully-invalid chunks and the NaN convention for "no embedding
                        # here" (see #39). The strip still contributes its (zero) obs counts, written above. Its
                        # refusals still count too: an empty strip is the case the record most needs to explain, and
                        # skipping the tally here would make a fully refused shard report zero refusals.
                        refused["thin"] += dataset.refused_thin
                        refused["no_optical"] += dataset.refused_no_optical
                        refused["no_radar"] += dataset.refused_no_radar
                        radar_rule_enforced = not dataset.allow_s2_only
                        logger.info("Chunk %s strip %s: no valid pixels, leaving zero-filled", chunk.label, strip)
                        continue

                    self._resource_monitor.set_context("work", f"{chunk.label}:s{i + 1}/{len(strips)}:infer")
                    t_inf = time.monotonic()
                    result = run_inference(self.model, dataset, self.config, self.device, on_batch=on_batch)
                    infer_s += time.monotonic() - t_inf
                    # Cropped grids land at their column offset; outside the box the buffers keep their initial
                    # zero/NaN fill — the values those never-valid pixels get.
                    embeddings[strip, cols] = result.embeddings
                    scales[strip, cols] = result.scales
                    total_valid += len(dataset)
                    refused["thin"] += dataset.refused_thin
                    refused["no_optical"] += dataset.refused_no_optical
                    refused["no_radar"] += dataset.refused_no_radar
                    radar_rule_enforced = not dataset.allow_s2_only
            finally:
                pool.shutdown(wait=False, cancel_futures=True)

            # All strips done: the last strip's inputs and the full-chunk SCL mask are no longer needed (obs counts
            # and embeddings already live in the whole-chunk output buffers). Freed before the S3 write so peak RAM
            # does not carry a dead strip + mask through it.
            del chunk_data, dataset, mask_bundle

            if total_valid == 0:
                # Every strip was empty. The ROI pre-filter means a chunk reaches here only if it intersects the ROI,
                # so this fires only when every pixel in a live chunk fails the SCL/S1 validity thresholds. Assembly
                # fills the footprint with zeros/NaN from the Dask graph, so no placeholder zarr is needed — but a
                # zero-byte skip marker still goes down so verify_staged_completeness can tell a legitimate skip from
                # a silently-failed chunk (Ray worker crash, etc.).
                #
                # THE REASON, RECORDED. Discarding these three counts made a thin-depth refusal indistinguishable from
                # land that was never imaged, and left 43 of 40S's 58 live shards attributed to "optical skips" when
                # every one was refused for having no radar. The cell is write-once and its mosaic is deleted when it
                # lands, so a reason not written here is not recoverable. The observation summary rides along because
                # the counts alone cannot say HOW thin: obs counts accumulate per strip regardless of validity, so
                # they are populated even where nothing passed. The crop is passed as the slice itself so the record
                # can restrict its own buffers to the footprint the reasons actually cover. Shared with the success
                # path so a refused and a partly-refused shard are described in the same terms — see
                # `_coverage_record`.
                skip_record = {"label": chunk.label} | _coverage_record(
                    chunk,
                    refused=refused,
                    radar_rule_enforced=radar_rule_enforced,
                    obs_buffers=obs_buffers,
                    x_sub=x_sub,
                    optical_min_obs=self.config.optical_min_obs or OPTICAL_MIN_OBS,
                )
                logger.info(
                    "Chunk %s has no valid pixels, skipping (assembly will fill) — refused %s",
                    chunk.label,
                    ", ".join(f"{k}={v}" for k, v in sorted(refused.items()) if v),
                )
                # COVERAGE FIRST, MARKER LAST — the order IS the atomicity: the marker's presence vouches for the
                # coverage tile being complete, so a crash between them leaves a tile nothing points at rather than a
                # marker promising data that is not there. Best-effort: losing the coverage degrades provenance, while
                # losing the MARKER turns a benign skip into a failed chunk and wedges the cell on every retry under
                # the stable run id.
                try:
                    writer.write_coverage_only(chunk, run_id, obs_buffers, month_buffers)
                except Exception:
                    logger.exception(
                        "Chunk %s: coverage-only tile could not be staged; its counts will publish as "
                        "fill. The refusal reasons in the marker are unaffected.",
                        chunk.label,
                    )
                    # AND REMOVE WHAT IT LEFT, which is the half that matters. A failure AFTER `to_zarr` created the
                    # array metadata but BEFORE `staged_complete` was set leaves a tile that reads back as fill and
                    # that `_open_staged_tile` refuses — while the marker written just below makes every resume skip
                    # this chunk, so nothing ever repairs it. Under the stable, input-fingerprinted run id that
                    # refusal repeats on every retry and wedges the cell until someone deletes the prefix by hand.
                    # Guarded at the CALL as well, even though `discard_coverage` is written not to raise: nothing on
                    # this path may stop the marker below being written, and an exception here (an older writer
                    # without the method, anything) would turn a benign skip into a failed chunk — the very failure
                    # this cleanup exists to prevent.
                    with contextlib.suppress(Exception):
                        writer.discard_coverage(chunk, run_id)
                writer.write_skip_marker(chunk, run_id, skip_record)  # small marker: keep synchronous
                # Collect the prior deferred write BEFORE snapshotting elapsed: the wait is actor-occupancy this chunk
                # owns, as on the success path below, and leaving it out under-reports the phase table.
                prior_write = self._collect_prior_write_checked()
                elapsed = time.monotonic() - t0
                logger.info(
                    "%s",
                    _chunk_summary_line(
                        label=chunk.label,
                        run=run_id,
                        status="skipped",
                        valid_px=0,
                        total_s=round(elapsed, 1),
                        prologue_s=round(prologue_s, 1),
                        infer_s=0.0,
                        overhead_s=round(elapsed, 1),
                        strips=len(strips),
                        strip_h=plan.strip_h,
                        strategy=plan.strategy,
                        t_kept=t_kept,
                        rung=rung,
                        x_crop_w=(x_sub.stop - x_sub.start) if x_sub is not None else None,
                    ),
                )
                return {
                    "chunk": chunk.label,
                    "status": "skipped",
                    "valid_pixels": 0,
                    "elapsed_sec": elapsed,
                    "instance_id": self.instance_id,
                    "prior_write": prior_write,
                }

            # Resolve the PREVIOUS chunk's deferred write first (queued a full chunk ago on the single-slot writer, so
            # normally long done) so its outcome rides this result back to the scheduler.
            prior_write = self._collect_prior_write_checked()

            # Defer the whole-chunk staging write to the writer thread: it overlaps the next chunk's serial prologue
            # load instead of adding GPU-idle time. This chunk is only counted complete once the write's outcome is
            # confirmed (see the section comment).
            self._writer_pool_handle()

            # Radar coverage for this chunk, from buffers already in memory — no extra read. `scales` is NaN-filled
            # and written only where a pixel was embedded, so it IS the embedded mask; counting over the whole chunk
            # would score every out-of-ROI pixel as radar-free and swamp the answer. Summing only over embedded pixels
            # also keeps the intermediate ROI-sized rather than chunk-sized.
            embedded = ~np.isnan(scales)
            asc, desc = obs_buffers["s1_asc_obs_count"], obs_buffers["s1_desc_obs_count"]
            s1_free_px = int((embedded & (asc == 0) & (desc == 0)).sum())
            obs_at_embedded = asc[embedded].astype(np.uint32) + desc[embedded].astype(np.uint32)
            s1_thin_px = int(((obs_at_embedded > 0) & (obs_at_embedded < RADAR_THIN_MAX_OBS)).sum())
            # Optical depth, from the same mask and the same principle. Without it only the EXTREME is visible: a tile
            # where nothing survived the validity filter is recorded as a skip, so every depth above zero publishes as
            # ordinary data even though a year seen a handful of times and one seen weekly are not the same embedding.
            # No optical-free count is needed — that case IS the skip, and a skipped chunk never reaches here.
            s2_at_embedded = obs_buffers["s2_obs_count"][embedded]
            # The rule this RUN applied, not the module default. The dataset gates on `config.optical_min_obs`, which
            # comes from the store root, so counting against the constant would report a thin share measured by a line
            # the fill never used — and would read as a non-zero count on a store whose rule already refuses every
            # pixel it names.
            thin_below = self.config.optical_min_obs or OPTICAL_MIN_OBS
            s2_thin_px = int((s2_at_embedded < thin_below).sum())
            del embedded, obs_at_embedded, s2_at_embedded

            # THE SAME RECORD A WHOLLY-REFUSED SHARD WRITES, for a shard that embedded something. `refused`
            # accumulates over this shard's strips on BOTH paths, so a shard where the depth gate removed part of the
            # land knows how much. Without this the registry described such a shard as embedded with no refusals
            # recorded, reading as "covered" for ground that is partly holes — and those holes are the larger half of
            # the infill work list, not a refinement of it.
            coverage = _coverage_record(
                chunk,
                refused=refused,
                radar_rule_enforced=radar_rule_enforced,
                obs_buffers=obs_buffers,
                x_sub=x_sub,
                optical_min_obs=thin_below,
            )

            def _timed_write() -> str:
                # "write" is a separate context slot: the upload overlaps the NEXT chunk's prologue on the main
                # thread, so both phases must be attributable on the same RESOURCES line.
                self._resource_monitor.set_context("write", chunk.label)
                try:
                    t_w = time.monotonic()
                    path = writer.write_chunk(
                        chunk,
                        embeddings,
                        run_id,
                        embeddings_std=None,
                        scales=scales,
                        obs_counts=obs_buffers,
                        month_covered=month_buffers,
                    )
                    # One line per chunk: how long the backgrounded upload took. The phase table's write_s is ~0 by
                    # design, so this is the off-critical-path cost, for post-run upload health checks.
                    logger.info(
                        "Staging write for %s completed in %.1fs (backgrounded)", chunk.label, time.monotonic() - t_w
                    )
                    return path
                finally:
                    self._resource_monitor.set_context("write", None)

            self._pending_write = (chunk.label, self._writer_pool.submit(_timed_write))

            elapsed = time.monotonic() - t0
            logger.info(
                "Chunk %s complete: %d valid pixels, %.1fs",
                chunk.label,
                total_valid,
                elapsed,
            )
            logger.info(
                "%s",
                _chunk_summary_line(
                    label=chunk.label,
                    # The per-CELL run id (e.g. "49S-2021-f1fa65fc"). Without it this line cannot be attributed to a
                    # zone: `label` is chunk_<row>_<col>, grid-local, so every cell restarts at chunk_0_0 and labels
                    # collide across zones and across concurrent fills sharing a log group. Attributing by time window
                    # instead produced two confidently wrong findings — see
                    # context_docs/design/campaign_inference_profile_2026_08.md.
                    run=run_id,
                    # "success" = inference finished and outputs are staged for upload. write_confirmed=False flags
                    # that the deferred S3 write is NOT yet durably confirmed (that happens a chunk later via
                    # prior-write chain-confirmation), so a phase tool should treat a later duplicate label as the
                    # retry of a failed write and keep the last, not double-count it.
                    status="success",
                    write_confirmed=False,
                    valid_px=total_valid,
                    total_s=round(elapsed, 1),
                    prologue_s=round(prologue_s, 1),
                    infer_s=round(infer_s, 1),
                    # The write is deferred to the background writer, so the honest critical-path overhead is
                    # everything that is not inference.
                    overhead_s=round(elapsed - infer_s, 1),
                    strips=len(strips),
                    strip_h=plan.strip_h,
                    strategy=plan.strategy,
                    t_kept=t_kept,
                    # Optical depth alone does not say how much the forward pass did — see the capture site. Additive
                    # keys, and every consumer reads by name.
                    t_s1_asc=t_s1_asc,
                    t_s1_desc=t_s1_desc,
                    rung=rung,
                    x_crop_w=(x_sub.stop - x_sub.start) if x_sub is not None else None,
                ),
            )

            return {
                "chunk": chunk.label,
                "status": "success",
                "valid_pixels": total_valid,
                # Per-chunk radar coverage, aggregated per YEAR at assembly. Reported here because this is the only
                # place the observation maps and the embedded mask are both in memory; recomputing it later would mean
                # reading the grid.
                "s1_free_pixels": s1_free_px,
                "s1_thin_pixels": s1_thin_px,
                "s2_thin_pixels": s2_thin_px,
                # The line the count was measured against, so the year record states the rule this fill applied rather
                # than the module default.
                "s2_thin_below_obs": thin_below,
                # Per-shard refusal reasons and optical depth, for the published registry. Keyed under one name so
                # assembly can merge it by label without knowing its contents.
                "coverage": coverage,
                "elapsed_sec": elapsed,
                "instance_id": self.instance_id,
                "write_deferred": True,
                "prior_write": prior_write,
            }

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.exception("Chunk %s failed after %.1fs on instance %s", chunk.label, elapsed, self.instance_id)
            # Minimal summary: the prologue may not have completed, so per-plan fields can be unbound here. Parsers
            # key on status.
            logger.info(
                "%s",
                _chunk_summary_line(
                    label=chunk.label,
                    run=run_id,
                    status="failed",
                    total_s=round(elapsed, 1),
                    error=str(e)[:200],
                ),
            )
            return {
                "chunk": chunk.label,
                "status": "failed",
                "error": str(e),
                "elapsed_sec": elapsed,
                "instance_id": self.instance_id,
            }
        finally:
            # Between chunks the main thread is idle; any live "write" slot is owned by the background writer and
            # cleared there.
            self._resource_monitor.set_context("work", "idle")


# Schemes that mean "fetch this from somewhere else first". A local filesystem path (or an explicit file:// URI) is
# loaded in place by torch.
_REMOTE_CKPT_SCHEMES = ("s3://", "http://", "https://", "gs://", "az://", "abfs://")


def _is_remote_uri(path: str) -> bool:
    """True if ``path`` must be downloaded before torch.load can open it."""
    return path.startswith(_REMOTE_CKPT_SCHEMES)


def _default_checkpoint_cache() -> str:
    """Pick a download cache dir that exists on the running host.

    On AWS DLAMI GPU boxes the NVMe instance store (~1.5 GB/s) is the right target: the root EBS
    volume (~42 MB/s) is too slow and torch.load with mmap hangs on it. Off that path (laptops,
    CI, non-AWS GPUs) the NVMe mount does not exist, so fall back to the system temp dir.
    """
    nvme = Path("/opt/dlami/nvme")
    if nvme.is_dir():
        return str(nvme / "tessera-checkpoints")
    return str(Path(tempfile.gettempdir()) / "tessera-checkpoints")


def download_checkpoint(remote_path: str, local_dir: str | None = None) -> str:
    """Download a model checkpoint from a remote URI to local storage.

    Handles any fsspec-supported remote scheme — ``s3://``, ``https://`` (e.g. a HuggingFace
    ``resolve/main`` URL), ``gs://``. The file is staged locally because torch.load wants a real
    path and reads it twice.

    Args:
        remote_path: Remote URI (e.g. ``"s3://bucket/path/model.pt"`` or
            ``"https://huggingface.co/.../tessera_v1_1_aws_encoder.pt"``).
        local_dir: Local directory for downloads. Defaults to the NVMe instance store on AWS
            DLAMI hosts, else a system temp dir.

    Returns:
        Local file path.

    Concurrency: many actors on one host may call this with the same ``remote_path`` and shared
    cache dir at once (cold cache, hundreds of actors). The download writes to a unique temp file
    and is published with an atomic rename, so a concurrent reader never sees a partially-written
    checkpoint and concurrent writers cannot corrupt each other — the last rename wins and every
    byte is identical.
    """
    filename = remote_path.rsplit("/", 1)[-1]

    local = Path(local_dir or _default_checkpoint_cache())
    local.mkdir(parents=True, exist_ok=True)
    local_path = local / filename

    if local_path.exists():
        logger.info("Checkpoint already cached: %s", local_path)
        return str(local_path)

    logger.info("Downloading checkpoint: %s → %s", remote_path, local_path)
    # Checkpoints are ~200 MB, so reading the whole file into memory is fine.
    with fsspec.open(remote_path, "rb") as remote:
        data = remote.read()
    # Staged into a unique temp file in the SAME dir, so the rename stays on one filesystem and is atomic: concurrent
    # actors publishing the same checkpoint cannot see a half-written file.
    with tempfile.NamedTemporaryFile(dir=local, prefix=f"{filename}.", suffix=".part", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(local_path)

    downloaded_size = local_path.stat().st_size
    logger.info("Download complete: %s (%.1f MB)", local_path, downloaded_size / 1024 / 1024)

    return str(local_path)
