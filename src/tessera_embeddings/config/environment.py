"""Environment configuration for GDAL and logging.

Must be called BEFORE importing rasterio/odc.stac to take effect.

Usage:
    from tessera_embeddings.config.environment import configure_gdal_environment

    # Must be called before importing rasterio, odc.stac, etc.
    configure_gdal_environment()
"""

import logging
import os

IO_WORKFLOW_DASK_CLUSTER_SPECS = {"DASK_WORKER_CPU": "1024", "DASK_WORKER_MEMORY": "3072"}


def configure_gdal_environment() -> None:
    """Configure GDAL/Rasterio environment variables for optimal COG access.

    Must be called BEFORE importing rasterio/odc.stac to take effect.
    These settings optimize Cloud Optimized GeoTIFF (COG) access from S3/HTTP.
    """
    # EXTRACTION SETTINGS

    # Retry settings for transient network failures (DNS, connection, timeout)
    os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")  # Default: 0 (no retries)
    os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "5")  # Default: 30 seconds
    os.environ.setdefault("GDAL_HTTP_TIMEOUT", "120")  # Connection timeout in seconds

    # Kill hung connections that stall for 60 seconds
    os.environ.setdefault("GDAL_HTTP_LOW_SPEED_LIMIT", "1")  # bytes/sec threshold
    os.environ.setdefault("GDAL_HTTP_LOW_SPEED_TIME", "60")  # seconds below threshold

    # PROCESSING SETTINGS

    # Use all available CPU cores for GDAL operations (decompression, resampling)
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

    # Enable HTTP/2 multiplexing - allows multiple range requests over a single
    # connection, reducing connection overhead when reading many COG chunks
    os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")

    # Merge consecutive byte range requests into single requests - reduces the
    # number of HTTP round-trips when reading adjacent chunks
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")

    # Increase the GDAL block cache to 1GB (default is 5% of RAM, often too small)
    # This cache stores decompressed raster blocks, reducing repeated decompression
    os.environ.setdefault("GDAL_CACHEMAX", "1024")

    # Disable GDAL's directory listing on open - speeds up access to remote files
    # by skipping sidecar file checks (.aux.xml, .ovr, etc.)
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")

    # # Useful for local dev - prevents intermittent failures on MacOS
    # os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "NO")
    # os.environ.setdefault("GDAL_HTTP_VERSION", "1.1")
    # os.environ.setdefault("CPL_VSIL_CURL_CHUNK_SIZE", "YES")

    # LOGGING
    # Configure the `tessera_embeddings` package logger so all module-level
    # loggers (getLogger(__name__) in stac.py, zarr_store.py, etc.) emit to
    # stderr, which the ECS awslogs driver captures. The package logger name
    # must match the module loggers' parent — they are getLogger(__name__),
    # i.e. "tessera_embeddings.*" — or the level/handler never apply and the
    # loggers fall through to the root WARNING default.
    # Reads SRC_LOG_LEVEL env var; defaults to INFO.
    level = os.environ.get("SRC_LOG_LEVEL", "INFO").upper()
    pkg_logger = logging.getLogger("tessera_embeddings")
    pkg_logger.setLevel(getattr(logging, level, logging.INFO))
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s - %(message)s"))
        pkg_logger.addHandler(handler)
