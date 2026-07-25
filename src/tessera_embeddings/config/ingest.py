"""Configuration for the satellite data ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

# Spatial chunk size for storage (written at ingest). 4096 aligns with the
# global store's 2048-px shard grid: one ingest chunk is exactly 2x2 shards
# (and 16x16 of the 256-px inner chunks). Still larger than the inference
# read-tile size to keep the satellite-ingest Dask graph small. Inference reads
# sub-tiles out of these chunks via .oindex, so storage chunk size and inference
# tile size stay independent: the global fill uses a 2048-px tile (2x2 per ingest
# chunk), the single-ROI default is 2000 to divide that output's 500-px chunks
# (see config.inference.INFERENCE_CHUNK_SIZE).
#
# NOTE: this changed 4000 -> 4096. Existing 4000-px stores stay readable, but
# appends to them under this config are rejected by the RoiManifest chunk-size
# check (structural-param mismatch) — finish an old campaign on the old config
# or re-ingest.
INGEST_CHUNK_SIZE = 4096

INGEST_CHUNKS = {"time": 1, "northing": INGEST_CHUNK_SIZE, "easting": INGEST_CHUNK_SIZE}

# Dask block size for the LOAD side of ingest — deliberately DECOUPLED from the
# storage chunk size above, and the reason the two exist separately.
#
# Graph task count is what limits ingest, and it scales with the number of blocks
# the read path builds. Coarsening the load blocks shrinks that count without
# touching the store's read geometry, which matters because the inference side is
# tuned around the STORED chunk size: zarr decompresses whole stored chunks to
# serve any part of one, so a coarser store adds a fixed per-chunk read cost to
# the GPU path (measured, and the reason the store stays at 4096 — see
# context_docs/design/ingest-graph-and-stac-budget.md).
#
# Must be a positive multiple of INGEST_CHUNK_SIZE: the write rechunks load blocks
# down to store chunks, and a non-multiple would make that a cross-block shuffle
# instead of a pure split. Set equal to INGEST_CHUNK_SIZE to disable the decoupling.
INGEST_LOAD_CHUNK_SIZE = 8192

if INGEST_LOAD_CHUNK_SIZE % INGEST_CHUNK_SIZE:
    raise ValueError(
        f"INGEST_LOAD_CHUNK_SIZE ({INGEST_LOAD_CHUNK_SIZE}) must be a multiple of "
        f"INGEST_CHUNK_SIZE ({INGEST_CHUNK_SIZE}) so the write is a pure split"
    )

INGEST_LOAD_CHUNKS = {"time": 1, "northing": INGEST_LOAD_CHUNK_SIZE, "easting": INGEST_LOAD_CHUNK_SIZE}

# Manifest sharding for the campaign mosaics. Without it every commit rewrites the
# whole array manifest, so the bytes rewritten grow with the number of dates already
# written — the shape that makes a zone-YEAR disproportionately worse than a month.
#
# TIME ONLY, and that is the load-bearing detail. One commit here is one DATE, and a
# date writes every live window — i.e. essentially the whole live area. So spatial
# shards cannot localise a commit: every commit touches nearly all of them, and all
# they add is object count. A spatial split of 4x4 on a 6-degree zone means ~4,000
# tiny manifest objects rewritten per commit instead of ~14, and the latency of those
# PUTs costs far more than the bytes saved. A time split is what localises a per-date
# commit: it rewrites only the dates sharing its shard rather than every date so far.
#
# The size trades write cost against read cost: smaller shards rewrite less per commit
# but a reader spanning a long window opens more of them. 8 caps the rewrite at eight
# dates' references while keeping a year to ~32 shards.
INGEST_MANIFEST_SPLIT = {"time": 8}

# S2 per-solar-day keep threshold for the GLOBAL CAMPAIGN ingest path, as a
# PERCENT of ROI pixels that must be valid (cloud-screened) to keep a solar
# day — the same percent scale as ingest/roi_processing.py's same-named
# constant, but tuned to 0.1% because a whole-zone ROI is mostly ocean/edge
# (the single-ROI default of 5% would drop nearly every date). The duplicate
# constants should eventually be unified under one name with per-path
# defaults.
DEFAULT_MIN_VALID_COVERAGE = 0.1


class IngestSettings(BaseModel):
    """Shared ingest tuning knobs, grouped for Prefect flow signatures.

    One model consumed by every campaign flow that carries ingest tuning —
    ``ingest-zone-year`` (which fans the values out to the base S1/S2 ingest
    flows), ``run-global-campaign``, and ``fill-zones-sequential`` — so the
    knobs stay one nested object in dispatch parameter dicts and render as one
    collapsible group in the Prefect UI, instead of four flat copies per
    signature.

    Defaults are campaign-scale. The single-ROI path (``tessera-full-pipeline``
    and the base ingest flows) keeps its own flat knobs for now: its worker
    bounds are ``None``-means-auto-size (auto-derived from ROI chunk count)
    and its coverage default is 5% vs the campaign's 0.1% (see
    :data:`DEFAULT_MIN_VALID_COVERAGE`), so adopting this model there first
    requires reconciling those defaults.
    """

    # Dask worker bounds for one (zone, year) ingest.
    min_workers: int = Field(default=1, ge=1)
    max_workers: int = Field(default=50, ge=1)
    # S2 per-solar-day keep threshold (percent; see DEFAULT_MIN_VALID_COVERAGE).
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE
    # S1 CMR query batch window, in days. Must be >= 1: the S1 loop advances
    # batch_start by this much, so 0 never advances (an ingest cluster billing
    # forever) and a negative walks it backwards.
    batch_days: int = Field(default=30, ge=1)
    # Optional base URI (an fsspec target, e.g. s3://.../perf/) for capturing a
    # Dask ``distributed.performance_report`` per child ingest. Default None =
    # off (normal runs pay nothing); set it only on a probe rung — ingest-zone-year
    # composes a unique per-child filename (``s2.html``, ``s1-<orbit>.html``) under it.
    perf_report_uri: str | None = None
    # Restrict mosaic writes (and the SCL coverage reduce) to the chunk-aligned
    # windows that intersect the ROI mask (ingest.live_windows). Default False
    # until the cropped path is validated end to end.
    crop_to_live_windows: bool = False

    @model_validator(mode="after")
    def _worker_bounds_ordered(self) -> IngestSettings:
        """Reject max_workers < min_workers.

        The cropped path's fleet sizing clamps into ``[max(min_workers, floor),
        max_workers]``, so inverted bounds make the floor win and the derived cap
        silently EXCEEDS the configured maximum — the one number an operator sets
        to bound spend. The uncropped path just hands the provider a nonsense
        range. Cheaper to refuse the config than to reconcile it downstream.
        """
        if self.max_workers < self.min_workers:
            raise ValueError(
                f"max_workers ({self.max_workers}) is below min_workers ({self.min_workers}); "
                "the worker bounds must be orderable."
            )
        return self
