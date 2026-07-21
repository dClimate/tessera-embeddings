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

# S2 per-solar-day keep threshold for the GLOBAL CAMPAIGN ingest path: a solar
# day is kept when at least this FRACTION of the zone's live tiles has valid
# (cloud-screened) coverage. NOTE the single-ROI path has a same-named,
# different-scaled constant (ingest/roi_processing.py, 5.0 — a PERCENT of the
# ROI); the two paths' semantics must be reconciled before one constant can
# serve both.
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
    bounds are ``None``-means-auto-size and its coverage threshold is
    percent-scaled (see :data:`DEFAULT_MIN_VALID_COVERAGE`), so adopting this
    model there first requires unifying those semantics.
    """

    # Dask worker bounds for one (zone, year) ingest.
    min_workers: int = 1
    max_workers: int = 50
    # S2 per-solar-day keep threshold (fraction; see DEFAULT_MIN_VALID_COVERAGE).
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE
    # S1 CMR query batch window, in days.
    batch_days: int = 30
