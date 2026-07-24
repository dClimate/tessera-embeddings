"""Configuration for the satellite data ingestion pipeline."""

from __future__ import annotations

from pydantic import BaseModel

# Spatial chunk size for storage (written at ingest). 4096 aligns with the
# global store's 2048-px shard grid: one ingest chunk is exactly 2x2 shards
# (and 16x16 of the 256-px inner chunks). Still larger than the inference
# read-tile size to keep the satellite-ingest Dask graph small. Inference reads
# sub-tiles out of these chunks via .oindex, so storage chunk size and inference
# tile size stay independent; the inference tile moves to 2048 in the aligned
# design (see config.inference.INFERENCE_CHUNK_SIZE).
#
# NOTE: this changed 4000 -> 4096. Existing 4000-px stores stay readable, but
# appends to them under this config are rejected by the RoiManifest chunk-size
# check (structural-param mismatch) — finish an old campaign on the old config
# or re-ingest.
INGEST_CHUNK_SIZE = 4096

INGEST_CHUNKS = {"time": 1, "northing": INGEST_CHUNK_SIZE, "easting": INGEST_CHUNK_SIZE}

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
    min_workers: int = 1
    max_workers: int = 50
    # S2 per-solar-day keep threshold (percent; see DEFAULT_MIN_VALID_COVERAGE).
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE
    # S1 CMR query batch window, in days.
    batch_days: int = 30
    # Optional base URI (an fsspec target, e.g. s3://.../perf/) for capturing a
    # Dask ``distributed.performance_report`` per child ingest. Default None =
    # off (normal runs pay nothing); set it only on a probe rung — ingest-zone-year
    # composes a unique per-child filename (``s2.html``, ``s1-<orbit>.html``) under it.
    perf_report_uri: str | None = None
    # Restrict mosaic writes (and the SCL coverage reduce) to the chunk-aligned
    # windows that intersect the ROI mask (ingest.live_windows). Default False
    # until the cropped path is validated end to end.
    crop_to_live_windows: bool = False
