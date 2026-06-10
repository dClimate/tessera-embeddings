"""Zarr store management utilities for reflectance and cloudmask data.

This module provides functions for creating, reading, and appending to Zarr stores
that hold preprocessed Sentinel-2 reflectance data and cloud masks.

Uses Icechunk for transactional writes with atomic commit semantics.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import dask.array as da
import fsspec
import icechunk
import numpy as np
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk

from tessera_embeddings.errors import CorruptedStoreError
from tessera_embeddings.storage.manifest import IngestManifest, extract_manifest
from tessera_embeddings.utils import utcnow_iso

logger = logging.getLogger(__name__)

# Default chunking for Zarr storage arrays.
# time=1 keeps each date separate for efficient single-date reads and appends.
# Full spatial chunks reasonable with icechunk and simplifies task graph
DEFAULT_CHUNKS = {"time": 1, "northing": 10980, "easting": 10980}

# Standard time encoding for all stores
TIME_ENCODING = {"units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian"}


# =============================================================================
# Store Cleanup Utilities
# =============================================================================


def _delete_store(store_path: str) -> bool:
    """Delete a store at the given path. Returns True if deleted, False if not found."""
    try:
        fs = fsspec.filesystem(fsspec.utils.get_protocol(store_path))
        if fs.exists(store_path):
            fs.rm(store_path, recursive=True)
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to delete store {store_path}: {e}")
        return False


def cleanup_on_failure[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """Decorator that deletes the store at store_path (first arg) if the function fails.

    Use on store creation functions to prevent leaving corrupted/partial stores.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        store_path: str = args[0] if args else kwargs.get("store_path")  # type: ignore[assignment]
        if not store_path:
            raise ValueError("cleanup_on_failure requires store_path as first argument")
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.warning(f"Store creation failed, cleaning up {store_path}")
            _delete_store(store_path)
            raise

    return wrapper


# =============================================================================
# Core Repository Helpers
# =============================================================================


def _parse_s3_url(url: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix into (bucket, prefix)."""
    path = url[5:]
    parts = path.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


# NOTE: Hardcoding a default region is normally an anti-pattern in this
# package — paths and credentials are caller-supplied so the code works for
# any deployment. We make a deliberate exception here: the Sentinel-1 OPERA
# RTC archive lives in us-west-2 and authentication to it only succeeds
# from in-region clients (cross-region requests are rejected, not just
# charged for). Any pipeline running this code against OPERA must therefore
# be in us-west-2, so defaulting the region here keeps that path zero-config
# while still letting callers override via the ``region`` argument.
_DEFAULT_S3_REGION = "us-west-2"


# Process-wide fallback credential provider for icechunk S3 storage. When set,
# _create_storage uses it for any S3 open that didn't receive an explicit
# get_credentials. The S1 ingest path registers an IAM-resolving callback here
# (via the AWS provider) so icechunk writes keep using IAM-role creds even
# after set_s3_credentials overwrites the AWS_* env vars with OPERA-scoped STS
# tokens. Stays None in the open-source layer, keeping it cloud-agnostic.
_default_credentials_provider: "Callable[[], icechunk.S3StaticCredentials] | None" = None


def set_credentials_provider(
    provider: "Callable[[], icechunk.S3StaticCredentials] | None",
) -> None:
    """Register a process-wide fallback credential provider for icechunk S3 opens.

    Used by :func:`_create_storage` whenever an S3 open has no explicit
    ``get_credentials``. Pass ``None`` to clear. See the module-level
    ``_default_credentials_provider`` note for why the S1 ingest path needs this.
    """
    global _default_credentials_provider
    _default_credentials_provider = provider


def _create_storage(
    store_path: str,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> icechunk.Storage:
    """Create Icechunk storage for local or S3 paths.

    Args:
        store_path: Local path or S3 URI.
        get_credentials: Optional credential callback for Icechunk's S3
            client. The callable returns a fresh
            :class:`icechunk.S3StaticCredentials` on each invocation; setting
            ``expires_after`` on the returned object tells Icechunk when to
            call back for refresh. When ``None`` on an S3 path, Icechunk
            falls back to the default AWS credential chain (env, instance
            profile, etc.), which is suitable for local testing with moto.
            The AWS provider in the closed-source repo supplies a
            botocore-backed callback that adapts boto's
            ``RefreshableCredentials`` into ``S3StaticCredentials``.
        region: Optional S3 region. Defaults to ``us-west-2`` because the
            Sentinel-1 OPERA RTC archive only authenticates in-region.
            Override for stores outside us-west-2.
        scatter_initial_credentials: When True, Icechunk eagerly calls
            ``get_credentials`` once and caches the result so that pickled
            copies of the storage (e.g. shipped to Ray actors or Dask workers
            during ``to_icechunk``) don't all stampede the credential
            provider on deserialisation. Set True for distributed assembly.
    """
    if store_path.startswith("s3://"):
        bucket, prefix = _parse_s3_url(store_path)
        if _s3_config_override:
            return _s3_config_override.make_storage(prefix_override=prefix)
        # Fall back to a globally-registered credential provider when the
        # caller didn't pass one explicitly. This is how the S1 ingest path
        # keeps icechunk on IAM-role creds: set_s3_credentials overwrites the
        # AWS_* env vars with OPERA-scoped STS tokens for GDAL reads, and
        # icechunk's default AWS chain would otherwise pick those up and get
        # AccessDenied writing our own store. The AWS provider registers an
        # IAM-resolving callback via set_credentials_provider(). See
        # tessera_embeddings.providers.aws.credentials.
        if get_credentials is None:
            get_credentials = _default_credentials_provider
        s3_kwargs: dict = {
            "bucket": bucket,
            "prefix": prefix,
            "region": region if region is not None else _DEFAULT_S3_REGION,
        }
        if get_credentials is not None:
            s3_kwargs["get_credentials"] = get_credentials
            s3_kwargs["scatter_initial_credentials"] = scatter_initial_credentials
        return icechunk.s3_storage(**s3_kwargs)
    # icechunk.local_filesystem_storage takes a plain path, not a URI.
    # Strip the file:// scheme that fsspec-style configs commonly use.
    local_path = store_path[7:] if store_path.startswith("file://") else store_path
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    return icechunk.local_filesystem_storage(local_path)


def _default_repo_config(max_concurrent_requests: int | None = None) -> icechunk.RepositoryConfig | None:
    """RepositoryConfig overrides. Returns None when no overrides are set.

    When ``max_concurrent_requests`` is provided, caps per-repo HTTP
    concurrency. Assembly at cornbelt scale fans out thousands of concurrent
    PUTs to one zarr prefix, blowing past S3's ~3.5K/s per-prefix limit and
    triggering 503 SlowDown. Icechunk's default is 256 concurrent HTTP
    requests per repo; callers that fan out across many workers should set
    this lower (e.g. 64) so aggregate request rate stays under S3's ceiling.
    Retry/backoff is left at icechunk's defaults — SlowDown is already
    classified as retriable by the underlying AWS SDK.
    """
    if max_concurrent_requests is None:
        return None
    config = icechunk.RepositoryConfig.default()
    config.max_concurrent_requests = max_concurrent_requests
    return config


def _open_repo(
    store_path: str,
    max_concurrent_requests: int | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> icechunk.Repository:
    """Open an existing Icechunk repository."""
    return icechunk.Repository.open(
        _create_storage(
            store_path,
            get_credentials=get_credentials,
            region=region,
            scatter_initial_credentials=scatter_initial_credentials,
        ),
        config=_default_repo_config(max_concurrent_requests),
    )


def _create_repo(
    store_path: str,
    max_concurrent_requests: int | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> icechunk.Repository:
    """Create a new Icechunk repository."""
    try:
        return icechunk.Repository.create(
            _create_storage(
                store_path,
                get_credentials=get_credentials,
                region=region,
                scatter_initial_credentials=scatter_initial_credentials,
            ),
            config=_default_repo_config(max_concurrent_requests),
        )
    except icechunk.IcechunkError as e:
        if "repositories can only be created in clean prefixes" in str(e):
            raise CorruptedStoreError(
                f"Store {store_path} appears corrupted. Delete it or use a different path."
            ) from e
        raise


def open_or_create_repo(
    store_path: str,
    max_concurrent_requests: int | None = None,
    get_credentials: "Callable[[], icechunk.S3StaticCredentials] | None" = None,
    region: str | None = None,
    scatter_initial_credentials: bool = False,
) -> tuple[icechunk.Repository, bool]:
    """Open existing or create new Icechunk repository.

    Args:
        store_path: Local path or S3 URI.
        max_concurrent_requests: Optional cap on concurrent S3 requests per repo.
        get_credentials: Optional credential callback for Icechunk's S3 client.
            See :func:`_create_storage` for details.
        region: Optional S3 region override.
        scatter_initial_credentials: When True with a distributed writer,
            cache the initial credentials so pickled storage copies don't
            re-invoke the callback on each worker. See :func:`_create_storage`.

    Returns:
        Tuple of ``(repository, is_new)``. ``is_new`` is True if the repo was
        just created.
    """
    try:
        return _open_repo(
            store_path,
            max_concurrent_requests,
            get_credentials=get_credentials,
            region=region,
            scatter_initial_credentials=scatter_initial_credentials,
        ), False
    except (FileNotFoundError, icechunk.IcechunkError):
        return _create_repo(
            store_path,
            max_concurrent_requests,
            get_credentials=get_credentials,
            region=region,
            scatter_initial_credentials=scatter_initial_credentials,
        ), True


# =============================================================================
# Session Helpers (used by all read/write operations)
# =============================================================================


def _open_readonly(store_path: str) -> xr.Dataset:
    """Open store for reading. Returns xarray Dataset."""
    repo = _open_repo(store_path)
    session = repo.readonly_session(branch="main")
    return xr.open_zarr(session.store, consolidated=False)


@cleanup_on_failure
def _write_new(
    store_path: str,
    data: xr.Dataset,
    encoding: dict[str, Any] | None,
    message: str,
) -> None:
    """Create a new store with data. Cleans up on failure."""
    repo = _create_repo(store_path)
    session = repo.writable_session("main")
    to_icechunk(data, session, mode="w", encoding=encoding, align_chunks=True)
    session.commit(message)
    logger.info(f"Created {store_path}")


def _write_append(
    store_path: str,
    data: xr.Dataset,
    message: str,
    update_attrs: dict[str, Any] | None = None,
) -> None:
    """Append data to existing store."""
    repo = _open_repo(store_path)
    session = repo.writable_session("main")

    # Snapshot root attrs before to_icechunk, which overwrites them with
    # the (typically empty) data.attrs — destroying crs, _manifest, etc.
    root = zarr.open_group(session.store, mode="r")
    preserved_attrs = dict(root.attrs)

    to_icechunk(data, session, mode="a", append_dim="time", align_chunks=True)

    root = zarr.open_group(session.store, mode="r+")
    root.attrs.update(preserved_attrs)
    if update_attrs:
        root.attrs.update(update_attrs)
    session.commit(message)
    logger.info(f"Appended to {store_path}")


# =============================================================================
# Public API
# =============================================================================


def open_store(store_path: str) -> xr.Dataset:
    """Open an Icechunk store for reading as an xarray Dataset."""
    return _open_readonly(store_path)


def open_store_as_zarr_group(store_path: str) -> zarr.Group:
    """Open an Icechunk store for reading as a raw zarr Group.

    Bypasses xarray/dask entirely. Use when you need to read large arrays into
    numpy without the dask task-graph overhead — e.g., loading a chunk of
    reflectance bands for inference, where each ``.values`` call on the xarray
    path would build and execute a fresh dask graph per variable and hold
    scheduler state until the dataset handle dies.
    """
    repo = _open_repo(store_path)
    session = repo.readonly_session(branch="main")
    return zarr.open_group(session.store, mode="r")


def get_existing_dates(store_path: str) -> set[str]:
    """Get dates already present in a store. Returns empty set if store doesn't exist."""
    t0 = time.monotonic()
    logger.debug(f"Opening store: {store_path}")
    try:
        ds = _open_readonly(store_path)
        dates = {str(t.values)[:10] for t in ds.time}
        ds.close()
        logger.debug(f"Store has {len(dates)} existing dates ({time.monotonic() - t0:.1f}s)")
        return dates
    except (FileNotFoundError, icechunk.IcechunkError):
        logger.debug(f"Store not found ({time.monotonic() - t0:.1f}s): {store_path}")
        return set()
    except Exception as e:
        logger.warning(f"Could not read dates from {store_path}: {e}")
        return set()


def compute_doy(timestamps: np.ndarray) -> np.ndarray:
    """Compute day-of-year from datetime64 timestamps.

    Returns (N,) array of int32 DOY values (1-366).
    """
    years = timestamps.astype("datetime64[Y]")
    return ((timestamps.astype("datetime64[D]") - years).astype(int) + 1).astype(np.int32)


def _write_dataset_impl(
    store_path: str,
    data: xr.Dataset,
    tile_id: str,
    baselines: dict[str, int],
    chunks: dict[str, int] | None = None,
    manifest: IngestManifest | None = None,
    crs: str | None = None,
) -> None:
    """Shared implementation for write_dataset and write_reflectance."""
    effective_chunks = chunks if chunks is not None else DEFAULT_CHUNKS
    existing_dates = get_existing_dates(store_path)

    # Normalize time to nanosecond resolution to match TIME_ENCODING.
    # Newer pandas/xarray versions may produce datetime64[us]; coerce
    # so the zarr encoding round-trips correctly.
    if data.time.dtype != np.dtype("datetime64[ns]"):
        data = data.assign_coords(time=data.time.values.astype("datetime64[ns]"))

    # Compute DOY coordinate
    doy = compute_doy(data.time.values)

    if existing_dates:
        # Single store open: read manifest + baseline attrs together
        existing_ds = _open_readonly(store_path)
        if manifest:
            manifest.validate_against(extract_manifest(existing_ds.attrs), store_path)
        merged_baselines = dict(existing_ds.attrs.get("baselines_applied", {}))
        merged_baselines.update(baselines)
        existing_doy = list(existing_ds.attrs.get("doy", []))
        existing_ds.close()

        _write_append(
            store_path,
            data,
            message=f"Append {data.sizes['time']} dates",
            update_attrs={
                "baselines_applied": merged_baselines,
                "doy": existing_doy + doy.tolist(),
                "last_appended": utcnow_iso(),
            },
        )
    else:
        # Create new store with encoding
        chunk_sizes = (
            effective_chunks["time"],
            min(effective_chunks["northing"], data.sizes["northing"]),
            min(effective_chunks["easting"], data.sizes["easting"]),
        )
        encoding: dict[str, Any] = {str(var): {"chunks": chunk_sizes} for var in data.data_vars}
        encoding["time"] = TIME_ENCODING

        store_attrs: dict[str, Any] = {
            "tile_id": tile_id,
            "baselines_applied": baselines,
            "doy": doy.tolist(),
            "created_at": utcnow_iso(),
            "last_appended": utcnow_iso(),
        }
        if crs:
            store_attrs["crs"] = crs
        if manifest:
            store_attrs["_manifest"] = manifest.to_dict()
            logger.info("Writing _manifest to %s", store_path)

        data.attrs.update(store_attrs)
        _write_new(store_path, data, encoding, f"Create with {data.sizes['time']} dates")


def write_dataset(
    store_path: str,
    data: xr.Dataset,
    tile_id: str,
    baselines: dict[str, int],
    chunks: dict[str, int] | None = None,
    manifest: IngestManifest | None = None,
    *,
    crs: str,
) -> None:
    """Write dataset to Icechunk Zarr store, creating or appending as needed.

    Args:
        store_path: Path to the Zarr store (local or s3://).
        data: xarray Dataset with (time, northing, easting) dimensions.
        tile_id: Tile identifier for store metadata.
        baselines: Dict mapping date strings to baseline integers.
        chunks: Optional chunk sizes dict. Defaults to DEFAULT_CHUNKS.
        manifest: Typed manifest for append-safety validation.
            Written on create, validated on append.
        crs: CRS authority code (e.g. ``"EPSG:32615"``). Stored in root
            attrs so downstream consumers can determine the projection.
    """
    _write_dataset_impl(store_path, data, tile_id, baselines, chunks, manifest, crs=crs)


def write_reflectance(
    store_path: str,
    data: xr.Dataset,
    tile_id: str,
    baselines: dict[str, int],
    chunks: dict[str, int] | None = None,
    manifest: IngestManifest | None = None,
    *,
    crs: str | None = None,
) -> None:
    """Backward-compatible wrapper: writes reflectance without requiring CRS.

    Prefer ``write_dataset`` for new code — it enforces CRS.
    """
    _write_dataset_impl(store_path, data, tile_id, baselines, chunks, manifest, crs=crs)


def write_cloudmask(
    store_path: str,
    mask: np.ndarray | da.Array,
    timestamps: list[np.datetime64],
    tile_id: str | None = None,
    model_version: str | None = None,
    chunks: dict[str, int] | None = None,
    *,
    northing: np.ndarray,
    easting: np.ndarray,
) -> None:
    """Write cloud mask data, creating store if needed or appending if exists.

    Args:
        store_path: Path to the Zarr store.
        mask: Cloud mask array with shape (time, northing, easting).
        timestamps: Full timestamps for each time slice. Using full timestamps
            (not just dates) preserves distinct acquisitions on the same day.
        tile_id: Tile ID for new stores (required for creation).
        model_version: Model version for new stores (required for creation).
        chunks: Optional chunk sizes dict. Defaults to DEFAULT_CHUNKS.
        northing: 1-D UTM northing coordinate array (metres). Required — a
            store without spatial coordinates does not satisfy the GeoZarr contract.
        easting: 1-D UTM easting coordinate array (metres, paired with ``northing``).
    """
    effective_chunks = chunks if chunks is not None else DEFAULT_CHUNKS

    if mask.ndim != 3 or mask.shape[0] != len(timestamps):
        raise ValueError(f"mask shape {mask.shape} doesn't match {len(timestamps)} timestamps")

    # Build dataset with full timestamps to handle multiple acquisitions per day
    mask_ds = xr.DataArray(
        mask,
        dims=["time", "northing", "easting"],
        coords={"time": timestamps, "northing": northing, "easting": easting},
        name="mask",
    ).to_dataset()

    chunk_sizes = {
        "time": 1,
        "northing": min(effective_chunks["northing"], mask.shape[1]),
        "easting": min(effective_chunks["easting"], mask.shape[2]),
    }
    mask_ds = mask_ds.chunk(chunk_sizes)

    if get_existing_dates(store_path):
        _write_append(store_path, mask_ds, f"Append {len(timestamps)} cloudmask timestamps")
    else:
        if not tile_id or not model_version:
            raise ValueError("tile_id and model_version required for new cloudmask store")
        mask_ds.attrs.update(
            {
                "tile_id": tile_id,
                "model_version": model_version,
                "created_at": utcnow_iso(),
            }
        )
        _write_new(store_path, mask_ds, {"time": TIME_ENCODING}, f"Create with {len(timestamps)} timestamps")


# =============================================================================
# Test Helpers (for verifying Icechunk transactional behavior)
# =============================================================================


def _open_writable_session(store_path: str) -> tuple[icechunk.Session, icechunk.IcechunkStore]:
    """Open a writable session. Caller must commit."""
    repo = _open_repo(store_path)
    session = repo.writable_session("main")
    return session, session.store


# =============================================================================
# S3 Configuration (for testing with moto)
# =============================================================================


@dataclass
class S3Config:
    """S3 configuration for testing with moto or custom endpoints."""

    bucket: str
    prefix: str = ""
    endpoint_url: str | None = None
    allow_http: bool = False
    access_key_id: str | None = None
    secret_access_key: str | None = None
    region: str | None = None

    def make_storage(self, prefix_override: str | None = None) -> icechunk.Storage:
        """Create Icechunk Storage from this configuration."""
        kwargs: dict[str, Any] = {
            "bucket": self.bucket,
            "prefix": prefix_override if prefix_override is not None else self.prefix,
        }
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.allow_http:
            kwargs["allow_http"] = True
        if self.access_key_id and self.secret_access_key:
            kwargs["access_key_id"] = self.access_key_id
            kwargs["secret_access_key"] = self.secret_access_key
        if self.region:
            kwargs["region"] = self.region
        return icechunk.s3_storage(**kwargs)


_s3_config_override: S3Config | None = None


def set_s3_config(config: S3Config | None) -> None:
    """Set a global S3 configuration override for testing."""
    global _s3_config_override
    _s3_config_override = config


def get_s3_config() -> S3Config | None:
    """Get the current S3 configuration override."""
    return _s3_config_override
