"""Configuration for the satellite data ingestion pipeline."""

from __future__ import annotations

# Spatial chunk size for storage (written at ingest). Deliberately larger than
# the inference read-tile size (config.inference.INFERENCE_CHUNK_SIZE) to keep the
# satellite-ingest Dask graph small: 4000x4000 yields 1/4 the spatial tasks per
# date/batch. Inference reads 2000x2000 sub-tiles out of these chunks via
# .oindex, so storage chunk size and inference tile size are independent.
INGEST_CHUNK_SIZE = 4000

INGEST_CHUNKS = {"time": 1, "northing": INGEST_CHUNK_SIZE, "easting": INGEST_CHUNK_SIZE}
