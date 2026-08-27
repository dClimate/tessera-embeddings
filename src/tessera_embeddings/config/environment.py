"""Environment configuration for GDAL and logging.

Must be called BEFORE importing rasterio/odc.stac to take effect.

Usage:
    from tessera_embeddings.config.environment import configure_gdal_environment

    # Must be called before importing rasterio, odc.stac, etc.
    configure_gdal_environment()
"""

import logging
import os


def configure_gdal_environment() -> None:
    """Configure GDAL/Rasterio environment variables for optimal COG access.

    Must be called BEFORE importing rasterio/odc.stac to take effect.
    These settings optimize Cloud Optimized GeoTIFF (COG) access from S3/HTTP.
    """
    # EXTRACTION SETTINGS

    # Retry settings for transient network failures (DNS, connection, timeout).
    #
    # RETRY_DELAY is the BASE of an exponential ladder, not a fixed wait: GDAL multiplies it
    # by ~2 after each failure and does not cap it, so give-up time scales with BOTH values
    # and is dominated by the last rung. Measured 2026-08-27 server-side at base 0.5:
    # 0.5, 1.01, 2.07, 4.92, 10.96, 24.82, 52.36, 105.94 s. READ THE PAIR TOGETHER: these five
    # retries at base 5 give ~3 min per unreadable object, while odc's ten at base 0.5 give ~14
    # min — odc's read path is ~5x MORE patient than these values, not less. Ten at base 5 would
    # be ~2.5 h, and the S2 coverage gate adds seven further attempts (SOURCE_READ_ATTEMPTS = 8)
    # while max_leg_wall_clock_s cannot interrupt a running leg.
    #
    # NOTE the odc read path does not use these two, nor GDAL_DISABLE_READDIR_ON_OPEN:
    # `odc.loader.capture_rio_env()` applies its own values as an EXPLICIT rasterio Env, which
    # beats the process environment. GDAL falls back to the environment for every option odc
    # does NOT name, so the rest of this function does reach it. See
    # `context_docs/design/gdal-read-config-2026_08.md`.
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
    configure_logging()


def configure_logging() -> None:
    """Make the package's module loggers emit, wherever the process got its handlers.

    Idempotent, and safe to call in any process. Two callers need it: process
    entry points (via :func:`configure_gdal_environment`), and spawned worker
    processes — a ``spawn`` child inherits NO logging configuration, so its
    module loggers fall through to the root WARNING default with no handler and
    every INFO record is silently dropped. A worker that reports progress must
    call this first or its reports never exist.
    """
    # Set the LEVEL on the `tessera_embeddings` package logger so all module-level
    # loggers (getLogger(__name__) in stac.py, zarr_store.py, etc.) are not dropped.
    # The package logger name must match the module loggers' parent — they are
    # getLogger(__name__), i.e. "tessera_embeddings.*" — or the level never applies
    # and the loggers fall through to the root WARNING default.
    # Reads SRC_LOG_LEVEL env var; defaults to INFO.
    level = os.environ.get("SRC_LOG_LEVEL", "INFO").upper()
    pkg_logger = logging.getLogger("tessera_embeddings")
    pkg_logger.setLevel(getattr(logging, level, logging.INFO))

    # Attach a stderr handler ONLY when nothing upstream will emit these records.
    # Under Prefect, `setup_logging()` puts a PrefectConsoleHandler on the ROOT
    # logger, and our records reach it by propagation — so adding a handler here
    # too emitted EVERY line twice to CloudWatch, in two formats (ours renders
    # `2026-07-25 00:33:35,395 | ...`, Prefect's `00:33:35.395 | ...`). That is a
    # straight 2x on log ingest at campaign scale, and it silently inflates any
    # analysis that counts log lines. Root-handler presence is the right test
    # rather than a Prefect import check: it equally covers a caller who ran
    # logging.basicConfig() themselves. In a spawned worker neither logger has
    # handlers, so the worker gets this handler and its records reach the
    # container's log stream through the inherited stderr.
    #
    # This does NOT affect the Prefect UI. UI logs come from the APILogHandler on
    # `prefect.flow_runs` (what get_run_logger() returns) — never from this logger.
    if not pkg_logger.handlers and not logging.getLogger().handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s - %(message)s"))
        pkg_logger.addHandler(handler)


def code_identity() -> dict | None:
    """Which build of this package is running, read from its own install metadata.

    Recorded on a published cell so "what code produced this?" is a read rather than an
    inference. It was inferred once, and only worked by luck: a fill was found to have run an
    image predating a provenance field, and the only evidence was that the field it should
    have written was absent. That proxy exists exactly once, for exactly that field.

    Needs no build argument and no environment variable, because a git install already
    records its commit: ``direct_url.json`` carries the resolved ``commit_id`` alongside the
    revision that was asked for. The distinction matters — a branch name says which line of
    development, the commit says which build — and both go in.

    **RECORD ONLY. Nothing compares this, and nothing may.** A mid-campaign change is a
    normal event, so a value that differs between cells is information for whoever is
    diagnosing something and inert the rest of the time. Turning it into a gate would let a
    routine edit refuse a dispatch, which is a decision an operator makes, not a property the
    architecture should assert.

    Returns ``None`` rather than raising, for the same reason: this is a nice-to-have on a
    record, and no fill should ever fail because its own metadata was unreadable. A wheel
    install (no VCS info) legitimately has no commit and returns the version alone.
    """
    try:
        import importlib.metadata as md

        dist = md.distribution("tessera-embeddings")
        identity: dict = {"version": dist.version}
        raw = dist.read_text("direct_url.json")
        if raw:
            import json

            record = json.loads(raw) or {}
            vcs = record.get("vcs_info") or {}
            if vcs.get("commit_id"):
                identity["commit"] = str(vcs["commit_id"])
            if vcs.get("requested_revision"):
                identity["revision"] = str(vcs["requested_revision"])
            # An editable install has no commit and is worth SAYING so, not omitting: it means
            # the cell was filled from a working copy rather than a pinned build, which is the
            # single most useful thing to know about a result nobody can reproduce.
            if (record.get("dir_info") or {}).get("editable"):
                identity["editable"] = True
        return identity
    except Exception:
        return None
