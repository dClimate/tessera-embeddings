"""Configuration for the satellite data ingestion pipeline."""

from __future__ import annotations

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
