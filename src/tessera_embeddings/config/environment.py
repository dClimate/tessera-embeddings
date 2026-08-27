"""Environment configuration for GDAL and logging.

Usage:
    from tessera_embeddings.config.environment import configure_gdal_environment

    # Must be called before importing rasterio, odc.stac, etc.
    configure_gdal_environment()

**Setting these in the environment is NOT sufficient for the odc read path**, which is where
almost every source byte is read. ``odc.loader``'s ``capture_rio_env()`` builds the GDAL
environment it ships to its readers from its own config object and the *active rasterio Env* —
it never consults ``os.environ`` — and when both are empty it returns ``GDAL_CLOUD_DEFAULTS``:
just ``GDAL_DISABLE_READDIR_ON_OPEN``, ``GDAL_HTTP_MAX_RETRY=10`` and
``GDAL_HTTP_RETRY_DELAY=0.5``. Every other option below was silently absent from optical reads,
and the two that were present ran at odc's values rather than ours.

:data:`GDAL_READ_OPTIONS` is therefore the single source of truth, and
:func:`configure_odc_rio` is what makes it reach the reader. See
``context_docs/design/gdal-read-config-2026_08.md``.
"""

import logging
import os

#: GDAL options that must be in force wherever SOURCE IMAGERY is read.
#:
#: One definition, consumed twice: :func:`configure_gdal_environment` puts it in the process
#: environment (for direct rasterio use), and :func:`configure_odc_rio` hands the same dict to
#: odc, which does not read the environment. Two lists kept in step is how the read path came to
#: run a configuration nobody had written down.
GDAL_READ_OPTIONS: dict[str, str] = {
    # Retries for transient network failures. RETRY_DELAY matters more than it looks: odc's
    # default is 0.5 s, so a burst of HTTP 500s outlasting ~5 s exhausted the budget and failed
    # the date. Measured 2026-08-27: 2,120 HTTP 500 and 174 HTTP 503 from S3 us-west-2 in 90
    # minutes, every one retried at 0.5 s.
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "5",
    "GDAL_HTTP_TIMEOUT": "120",
    # Kill hung connections that stall for 60 s. Absent from the odc path entirely until now,
    # which is why no `Operation too slow` or CURL error ever appeared in a read failure.
    "GDAL_HTTP_LOW_SPEED_LIMIT": "1",
    "GDAL_HTTP_LOW_SPEED_TIME": "60",
    # HTTP/2 multiplexing: many range requests over one connection.
    "GDAL_HTTP_MULTIPLEX": "YES",
    # Merge adjacent byte ranges into one request.
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    # Skip sidecar probes (.aux.xml, .ovr) on open.
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
}


def configure_gdal_environment() -> None:
    """Configure GDAL/Rasterio environment variables for optimal COG access.

    Must be called BEFORE importing rasterio/odc.stac to take effect.
    These settings optimize Cloud Optimized GeoTIFF (COG) access from S3/HTTP.
    """
    # The read-path options, from the one definition above.
    for name, value in GDAL_READ_OPTIONS.items():
        os.environ.setdefault(name, value)

    # PROCESSING SETTINGS

    # Use all available CPU cores for GDAL operations (decompression, resampling)
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

    # Increase the GDAL block cache to 1GB (default is 5% of RAM, often too small)
    # This cache stores decompressed raster blocks, reducing repeated decompression
    os.environ.setdefault("GDAL_CACHEMAX", "1024")

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


def configure_odc_rio() -> None:
    """Make :data:`GDAL_READ_OPTIONS` reach the odc reader, which ignores the environment.

    ``odc.loader``'s ``capture_rio_env()`` builds the GDAL environment for its readers from its
    own config object and the active rasterio ``Env``, and falls back to ``GDAL_CLOUD_DEFAULTS``
    when both are empty. Nothing in that path looks at ``os.environ``, so
    :func:`configure_gdal_environment` alone left optical reads running three options instead of
    eight — with ``GDAL_HTTP_RETRY_DELAY`` at odc's 0.5 s rather than ours.

    ``configure_rio(cloud_defaults=True, **opts)`` merges as ``{**GDAL_CLOUD_DEFAULTS, **opts}``,
    so our values win and odc's remain as the base for anything we do not name.

    Idempotent, and safe to call before or after a Dask cluster exists: odc propagates the captured
    environment to its workers itself.
    """
    # Imported HERE rather than at module scope, and not as an oversight: this module's whole
    # contract is that `configure_gdal_environment()` runs BEFORE rasterio/odc.stac are imported,
    # and `odc.stac` pulls in rasterio. A top-level import would make importing the config module
    # import rasterio, breaking the ordering the file exists to enforce.
    import odc.stac

    # `configure_rio(*, cloud_defaults, verbose, aws, **params)` declares typed keyword
    # parameters beside its catch-all, so mypy cannot prove a `dict[str, str]` splat will not
    # land on `verbose: bool` or `aws: dict | None`. Every key here is a GDAL_* option name, so
    # the collision it is guarding against cannot occur.
    odc.stac.configure_rio(cloud_defaults=True, **GDAL_READ_OPTIONS)  # type: ignore[arg-type]
