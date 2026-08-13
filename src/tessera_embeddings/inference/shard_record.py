"""The per-shard record: what a shard contained, and what it refused, in one row.

One inference chunk is one 2048-px shard, so this is per-shard accounting even though it is computed
per chunk. It exists because a minimum-depth rule makes "what is not here" a thing the product has to
be able to explain: a refused pixel and an ocean pixel are the same bytes in the store, and only a
record separates them.

**Everything here is a count, never a percentage.** A stored percentage is a second copy of a truth
that can drift from its numerator, and correcting one place and not the others is a failure this
repository has already had. Formulas belong in the reader.

**Three categories, never two.** Live shards include the land mask's sea buffer, so a coastal shard
can be majority ocean. "Percent skipped" against the full 2048² is meaningless, and ``n_eligible_px``
— pixels the mask calls land — is the only honest denominator. The parts must satisfy

    n_embedded_px + n_refused_thin_px + n_refused_no_optical_px + n_refused_no_radar_px == n_eligible_px

and :func:`build` checks it rather than trusting the caller, because the three refusal reasons are
computed in a different module from the eligibility mask and nothing else would notice them drifting
apart.

**The histogram is for the top-up, not for reporting.** Refused pixels are expected to be revisited
once more imagery is published, and that pass has to choose where to spend without scanning a
petabyte. How CLOSE a shard's refusals were to the line is what makes that choosable, and a mean
cannot express it: a shard whose refusals sit just under the line is rescued by a small backfill,
one whose refusals sit near zero is not.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from tessera_embeddings.config.store_layout import INNER_PX, SHARD_PX

#: Inner chunks along one shard edge — 8 for a 2048-px shard of 256-px chunks.
CHUNKS_PER_EDGE = SHARD_PX // INNER_PX
CHUNKS_PER_SHARD = CHUNKS_PER_EDGE**2

#: Upper edges of the depth bins the THIN refusals are counted into, exclusive of zero-observation
#: pixels (which are a different refusal). Five equal bins to 25 plus an open top bin, so the shape
#: survives the line moving: the bins describe the depths, not the rule.
DEPTH_BIN_EDGES: tuple[int, ...] = (5, 10, 15, 20, 25, 1_000_000)

STATUS_WRITTEN = "written"
STATUS_SKIPPED = "skipped"
STATUS_RESUMED_UNKNOWN = "resumed_unknown"


@dataclasses.dataclass(frozen=True)
class ShardRecord:
    """One shard's coverage, in counts, with the rule that produced it."""

    zone: str
    year: int
    tile_row: int
    tile_col: int
    n_eligible_px: int
    n_embedded_px: int
    n_refused_thin_px: int
    n_refused_no_optical_px: int
    n_refused_no_radar_px: int
    s2_obs_mean: float
    s2_obs_median: float
    s2_obs_p10: float
    refused_depth_hist: tuple[int, ...]
    chunks_skipped_mask: int
    n_chunks_eligible: int
    n_chunks_skipped: int
    status: str
    optical_min_obs: int | None
    mosaic_identity: str | None = None

    def as_row(self) -> dict:
        """A flat dict for the registry parquet, with the bitmask as a signed-safe string.

        The mask is 64 bits and parquet's integer types are signed, so a shard with its top chunk
        refused would round-trip as a negative number. Written as a fixed-width hex string, which
        is unambiguous and still sorts.
        """
        row = dataclasses.asdict(self)
        row["refused_depth_hist"] = list(self.refused_depth_hist)
        row["chunks_skipped_mask"] = f"{self.chunks_skipped_mask:016x}"
        return row


def depth_histogram(depths: np.ndarray, edges: tuple[int, ...] = DEPTH_BIN_EDGES) -> tuple[int, ...]:
    """Count refused depths into ``edges``, which must cover every value they can take.

    Zero-observation pixels are NOT here: they are ``n_refused_no_optical_px``, a fact about the
    imagery rather than about this campaign's rule, and counting them in both would break the
    invariant that the bins sum to the thin refusals.
    """
    if depths.size == 0:
        return tuple(0 for _ in edges)
    counts = []
    low = 1  # a thin pixel has at least one observation; zero is the other refusal
    for high in edges:
        counts.append(int(((depths >= low) & (depths < high)).sum()))
        low = high
    return tuple(counts)


def chunk_skip_mask(refused: np.ndarray, eligible: np.ndarray) -> tuple[int, int, int]:
    """Which inner chunks were refused ENTIRELY, as a bitmask, with the two counts beside it.

    Bit *i* is set when inner chunk *i* — row-major, 8 by 8 — had eligible pixels and every one of
    them was refused. A chunk with no eligible pixels is not "skipped": it is ocean, and marking it
    would make the count of skipped chunks a measure of coastline.

    Per-chunk *rows* would be 193 million for the campaign. This is chunk fidelity in eight bytes.
    """
    if refused.shape != eligible.shape:
        raise ValueError(f"refused {refused.shape} and eligible {eligible.shape} describe different areas")
    side = refused.shape[0] // CHUNKS_PER_EDGE
    if side * CHUNKS_PER_EDGE != refused.shape[0] or refused.shape[0] != refused.shape[1]:
        raise ValueError(f"{refused.shape} is not a square shard of {CHUNKS_PER_EDGE} inner chunks a side")
    grid = (CHUNKS_PER_EDGE, side, CHUNKS_PER_EDGE, side)
    eligible_per = eligible.reshape(grid).sum(axis=(1, 3))
    refused_per = refused.reshape(grid).sum(axis=(1, 3))
    fully = (eligible_per > 0) & (refused_per == eligible_per)
    mask = 0
    for index, flag in enumerate(fully.reshape(-1)):
        if flag:
            mask |= 1 << index
    return mask, int((eligible_per > 0).sum()), int(fully.sum())


def build(
    *,
    zone: str,
    year: int,
    tile_row: int,
    tile_col: int,
    eligible: np.ndarray,
    embedded: np.ndarray,
    s2_obs: np.ndarray,
    refused_thin: np.ndarray,
    refused_no_optical: np.ndarray,
    refused_no_radar: np.ndarray,
    optical_min_obs: int | None,
    status: str = STATUS_WRITTEN,
    mosaic_identity: str | None = None,
) -> ShardRecord:
    """Assemble one shard's record from masks that all describe the same footprint.

    Every mask is boolean over the shard. ``eligible`` is the land mask; the other four partition it,
    and that is checked here rather than assumed — the refusal masks come from the inference dataset
    and the eligibility mask from the land mask, so nothing else in the system would notice them
    disagreeing.

    Depth statistics are over ELIGIBLE pixels, not embedded ones. Over embedded pixels they would
    describe only the pixels that passed, which is the one population whose depth is already known
    to clear the line — and the question the record exists to answer is what the refused ones looked
    like.
    """
    shapes = {a.shape for a in (eligible, embedded, s2_obs, refused_thin, refused_no_optical, refused_no_radar)}
    if len(shapes) != 1:
        raise ValueError(f"masks describe different footprints: {sorted(shapes)}")

    n_eligible = int(eligible.sum())
    parts = (
        int(embedded.sum()),
        int(refused_thin.sum()),
        int(refused_no_optical.sum()),
        int(refused_no_radar.sum()),
    )
    if sum(parts) != n_eligible:
        raise ValueError(
            f"{zone}/{year} shard ({tile_row},{tile_col}): embedded + refused = {sum(parts)} but "
            f"{n_eligible} pixels are eligible. The eligibility mask and the refusal masks disagree, "
            "and every share derived from this row would be wrong in a way nothing downstream checks."
        )

    depths = s2_obs[eligible].astype(np.float32)
    refused_depths = s2_obs[refused_thin].astype(np.float32)
    refused_any = refused_thin | refused_no_optical | refused_no_radar
    mask, chunks_eligible, chunks_skipped = chunk_skip_mask(refused_any, eligible)
    return ShardRecord(
        zone=zone,
        year=year,
        tile_row=int(tile_row),
        tile_col=int(tile_col),
        n_eligible_px=n_eligible,
        n_embedded_px=parts[0],
        n_refused_thin_px=parts[1],
        n_refused_no_optical_px=parts[2],
        n_refused_no_radar_px=parts[3],
        s2_obs_mean=float(depths.mean()) if depths.size else float("nan"),
        s2_obs_median=float(np.median(depths)) if depths.size else float("nan"),
        s2_obs_p10=float(np.percentile(depths, 10)) if depths.size else float("nan"),
        refused_depth_hist=depth_histogram(refused_depths),
        chunks_skipped_mask=mask,
        n_chunks_eligible=chunks_eligible,
        n_chunks_skipped=chunks_skipped,
        status=status,
        optical_min_obs=optical_min_obs,
        mosaic_identity=mosaic_identity,
    )
