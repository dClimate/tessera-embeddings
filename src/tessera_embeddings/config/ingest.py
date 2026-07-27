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

# The load side deliberately uses the SAME block size as the store. A coarser load
# block (8192) was shipped once to shrink graph task counts for the scheduler's
# single-threaded dispatch, and was removed: fewer, larger read tasks cap a date's
# parallel width at blocks x bands, which starves any fleet wider than that. The
# width cost is paid on every compact ROI while the dispatch saving only binds on
# the densest zones at wide fleets. History, measurements and the cost model are in
# context_docs/design/ingest_optimization_campaign_2026_07.md (section 3.5 and its
# 2026-07-27 correction); do not reintroduce a load/store split without re-reading
# that record.

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
    # S2 per-solar-day keep threshold (PERCENT, so 0-100; see
    # DEFAULT_MIN_VALID_COVERAGE). Bounded, not just typed: a negative threshold is
    # satisfied by every date including one with zero valid pixels, which then counts
    # toward the month-presence gate and lets an empty year be filled and permanently
    # tagged complete. Above 100 nothing is ever kept.
    min_valid_coverage: float = Field(default=DEFAULT_MIN_VALID_COVERAGE, ge=0.0, le=100.0)
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
