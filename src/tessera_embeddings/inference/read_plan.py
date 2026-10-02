"""How a chunk is tiled into strips and cropped — the arithmetic, alone.

Split out of :mod:`~tessera_embeddings.inference.actors` because it is a self-contained
subject with no Ray, no torch and no actor state: given a chunk and its SCL mask, decide the
northing strip height and the easting crop.

It was already being treated as a separate thing from other modules, which reach for these
names in prose — ``config/inference.py`` and ``data_loading.py`` both cite
``_strip_height_for_density`` and ``resource_monitor.py`` cites ``_S2_STRIP_BYTE_BUDGET``. A
block other modules point at by name is not an implementation detail of the actor.

The budget here is derived, not chosen. Do not raise it without re-deriving the arithmetic in
``context_docs/inference/inference-on-gpus.md``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from tessera_embeddings.config.inference import S2_BAND_ORDER
from tessera_embeddings.inference.chunk_spec import ChunkSpec

if TYPE_CHECKING:
    from tessera_embeddings.inference.data_loading import S2MaskBundle

logger = logging.getLogger(__name__)

# Host-RAM budget (bytes) for one resident S2 band set (a strip's bands + its full-chunk SCL
# mask). The actor's load pipeline is one strip deep and runs across chunk boundaries — strip
# i+1 loads while strip i infers, and during a chunk's last strip the "next strip" is the next
# chunk's first — so at most TWO such sets are ever resident and the S2 ceiling is the PAIR:
# 3.5 GiB/set => ~7 GiB. The ceiling is peak host RAM under 60% of the 30.9 GB usable on a
# g6e.xlarge, and the margin under it is what absorbs the memory spikes a global run does hit:
#   pair ~7 GiB (~7.5 GB) + SAR ~1.5 (unmodelled; spikes on dense-S1 chunks) + whole-chunk int8
#   output buffers ~0.6 + model/torch/misc baseline ~3.3 => ~12.9 GB ~ 42%.
# Measured on the Iowa year in 2048^2 mosaics (2026-10-02, four paired arms): 3.5 GiB peaked at
# 12.7 GB (41%) against 16.4 GB (53%) at the 4000^2-era 5.75 GiB, at equal throughput; a 1 GiB
# budget bottomed out at 12.4 GB with twice the strips, so below ~3.5 GiB the budget no longer
# sets the peak, and 3.5 GiB leaves ~19 points under the ceiling for spikes. Strips are cheap at
# 2048^2 because a strip reads row slices of whole storage chunks: each extra strip costs a
# background decompression, not the ~13 s fixed read 4000^2 storage chunks charged. Do NOT raise
# this without re-deriving the arithmetic above: context_docs/inference/inference-on-gpus.md.
_S2_STRIP_BYTE_BUDGET = int(3.5 * 1024**3)


# Per-(timestep, pixel) byte cost of resident S2 bands: 10 bands x uint16.
_S2_BYTES_PER_OBS_PX = len(S2_BAND_ORDER) * 2


# Apply the S2 easting-bbox crop only when it removes at least this fraction of the chunk's width. Near-full boxes
# (interior chunks) skip it, keeping the mainline path byte-for-byte identical to the uncropped code rather than
# paying SAR column-copies for a few saved columns.
_X_CROP_MIN_SAVING = 0.10


# Floor on derived strip height. Below this, per-strip fixed overhead (zarr open, SCL slice, dataset bucketing)
# dominates and read amplification climbs without meaningfully lowering peak RAM. A pathologically dense chunk bottoms
# out here — deliberately breaching the byte budget, and logged — rather than degenerating into hundreds of tiny
# reads.
_MIN_STRIP_H = 256


def _strip_height_for_density(
    t_kept: int,
    width: int,
    height: int,
    budget: int = _S2_STRIP_BYTE_BUDGET,
    mask_width: int | None = None,
) -> int:
    """Largest northing strip height (rows) whose resident S2 working set fits ``budget``.

    Every resident set is charged ``bands(strip_h) + a full SCL mask``; the RAM model lives at
    :data:`_S2_STRIP_BYTE_BUDGET`. ``width`` sizes the (possibly easting-cropped) band read;
    ``mask_width`` sizes the mask, which stays full-chunk-width even when bands are cropped
    (defaults to ``width``). A chunk dense enough to drive the height below ``_MIN_STRIP_H``
    bottoms out there and breaches the budget, logged.
    """
    t = max(1, t_kept)
    mask_bytes = t * height * (mask_width if mask_width is not None else width)
    per_row = t * width * _S2_BYTES_PER_OBS_PX
    budget_h = max(0, budget - mask_bytes) // per_row
    if budget_h < _MIN_STRIP_H:
        # Worst-case resident pair: two floor-height band sets, each charged a full mask.
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


def _strip_slices(height: int, strip_h: int) -> list[slice]:
    """Tile ``[0, height)`` into chunk-relative northing strips of ``strip_h`` rows.

    The final strip is shorter when ``height`` is not a multiple of ``strip_h``;
    ``strip_h >= height`` yields a single strip.
    """
    return [slice(s, min(s + strip_h, height)) for s in range(0, height, strip_h)]


def _strip_plan(t_kept: int, height: int, width: int, mask_width: int | None = None) -> list[slice]:
    """Tile a chunk into budget-sized northing strips.

    ``mask_width`` is the full-chunk mask width when bands are easting-cropped. A chunk whose
    bands and mask fit one budget is a single strip.
    """
    return _strip_slices(height, _strip_height_for_density(t_kept, width, height, mask_width=mask_width))


def _chunk_read_plan(chunk: ChunkSpec, mask_bundle: S2MaskBundle) -> tuple[slice | None, list[slice]]:
    """Easting crop and northing strips derived from the SCL mask.

    Shared by the serial prologue and the cross-chunk prefetch so both make identical decisions
    from the same inputs — a prefetched chunk must tile and crop exactly as it would have serially.
    """
    t_kept = int(mask_bundle.mask.shape[0])
    # (H, W): pixels with >=1 valid S2 observation. Equivalent to mask.any(axis=0) — obs_count sums the pre-prune mask
    # and pruning drops only all-False timestep planes — but reads (H, W) instead of scanning the full (T_kept, H, W)
    # mask (up to ~1 GB) once per chunk.
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
    # Bands read at effective_width (possibly cropped); the SCL mask stays full chunk width, so charge it at
    # chunk.width in the budget.
    return x_sub, _strip_plan(t_kept, chunk.height, effective_width, mask_width=chunk.width)
