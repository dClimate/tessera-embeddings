"""Embedding output writers.

Staged writes (one Zarr per chunk) use raw uncompressed bytes for zero CPU
overhead on GPU actors. The final Icechunk store uses PCodec for long-term
storage efficiency.

PCodec is an array-to-bytes codec (serializer) in the Zarr v3 codec pipeline,
not a bytes-to-bytes compressor. It must be passed via the `serializer` encoding
key when writing via xarray, or via the `serializer` parameter in zarr.create_array.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import subprocess
import warnings
from collections.abc import Callable, Mapping

import dask.array as da
import fsspec
import icechunk
import numpy as np
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk
from zarr.codecs.numcodecs import PCodec as PCodecZarr3

from tessera_embeddings.config.inference import EMBEDDING_DIM, TimeWindow
from tessera_embeddings.inference.chunk_spec import ChunkSpec, filter_chunks_by_roi_mask
from tessera_embeddings.inference.conventions import build_convention_attrs
from tessera_embeddings.storage.manifest import EmbeddingManifest, extract_manifest
from tessera_embeddings.storage.zarr_store import manifest_split, open_or_create_repo, open_store

# PCodec is intentionally used as a Zarr v3 serializer; silence the
# "not in the Zarr version 3 specification" warning it emits on every instantiation.
warnings.filterwarnings("ignore", message="Numcodecs codecs are not in the Zarr version 3")

logger = logging.getLogger(__name__)


STAGED_READ_CONFIG_KWARGS = {"retries": {"max_attempts": 10, "mode": "adaptive"}}
"""botocore retry config for staged-Zarr GETs during assembly.

Staged reads fan out across every Dask worker at once, so a momentary GET burst
can trip S3 into a 503 SlowDown even well under the per-prefix ceiling. botocore's
default ``legacy`` mode makes 5 attempts, has no client-side rate limiting, and
doesn't recognize ``SlowDown`` in its throttle set, so the burst exhausts retries
and fails the block. ``adaptive`` mode adds explicit ``SlowDown`` handling plus a
token-bucket rate limiter that self-throttles when S3 pushes back — that feedback
loop, not the higher attempt count, is the actual fix.
"""


def _staged_storage_options(path: str) -> dict | None:
    """Return s3fs storage options for a staged read, or ``None`` for non-S3 paths.

    The retry config is a botocore client setting and only applies to the S3
    backend; local staging paths (``/tmp/...``) get ``None`` and open normally.
    """
    if fsspec.utils.get_protocol(path) == "s3":
        return {"config_kwargs": STAGED_READ_CONFIG_KWARGS}
    return None


def _fs_for(uri: str, storage_options: dict | None = None) -> fsspec.AbstractFileSystem:
    """Return an fsspec filesystem inferred from the URI scheme.

    Args:
        uri: Any fsspec-compatible URI (``s3://``, ``gs://``, ``file://``,
            absolute local path, etc.).
        storage_options: Extra kwargs forwarded to
            :func:`fsspec.filesystem` (e.g. ``{"anon": False}``).
    """
    opts = storage_options or {}
    protocol = fsspec.utils.get_protocol(uri)
    return fsspec.filesystem(protocol, **opts)


class IncompleteStageError(RuntimeError):
    """Raised when assembly is attempted before all chunks have been staged."""


@dataclasses.dataclass
class SpatialCoords:
    """Projected coordinate arrays and CRS from an input store."""

    northing: np.ndarray
    easting: np.ndarray
    crs: str | None = None


OBS_COUNT_VARS = ("s2_obs_count", "s1_asc_obs_count", "s1_desc_obs_count")

BAND_CHUNK_DIVISOR = 32
"""
Divisor applied to embedding_dim to set the band chunk size for staged and final Zarr stores.
NOTE this ended up not being necessary after cutting spatial chunks down to 500x500, leaving it here
in case it's useful down the road.
"""

TARGET_AGGREGATE_S3_CONCURRENCY = 100
"""Fleet-wide ceiling on concurrent S3 PUTs during assembly, divided across workers.

icechunk's ``max_concurrent_requests`` is per-Repository-instance, and every Dask
worker forks a fresh Repository when ``to_icechunk`` ships the session out. So
the effective aggregate concurrency is ``n_workers * per_worker_cap``, not the
per-worker cap alone. We target ~100 concurrent PUTs fleet-wide (roughly 1/35 of
S3's ~3500 req/s/prefix ceiling at ~100 ms PUT latency) to leave headroom for
retries and avoid 503 SlowDown.

icechunk writes chunk objects to a flat ``chunks/<random-id>`` keyspace, so the
keys spread across S3 partitions well — but partition splitting is adaptive and a
hard burst overruns the per-prefix rate before (and even after) S3 adapts, which
is exactly the SlowDown observed at 800 concurrent PUTs. The invariant that keeps
us under target is ``AssemblyConfig.max_workers <= TARGET_AGGREGATE_S3_CONCURRENCY``:
since ``per_worker_cap`` floors at 1, aggregate is >= n_workers, so the worker cap
must not exceed the target. Keep these two constants in sync.
"""


def pcodec_serializer() -> PCodecZarr3:
    """Create a PCodec serializer for Zarr v3."""
    return PCodecZarr3()


def _assemble_var_block(
    template: np.ndarray,
    block_id: tuple[int, ...] | None = None,
    *,
    var_name: str,
    live_lookup: dict[tuple[int, int], str],
) -> np.ndarray:
    """Produce one output block: read from a staged Zarr if live, else return the template.

    Called once per dask block via ``map_blocks``. Dask blocks are
    ChunkSpec-sized (one block per ChunkSpec spatially, full band axis as one
    block) — decoupled from the on-disk sub-chunk size used for output zarr
    encoding. This keeps the scheduler graph O(n_chunks) instead of
    O(n_chunks * sub_chunks_per_chunk), which is necessary at cornbelt scale
    where per-sub-chunk tasks would produce millions of TaskStates and OOM
    the scheduler.

    ``block_id`` is ``(time_idx, row, col[, band_idx=0])`` — the spatial
    indices correspond to ChunkSpec ``(row, col)`` directly since dask blocks
    match ChunkSpec extent.

    ``live_lookup`` maps ``(row, col) -> staged_path`` for chunks that have a
    staged Zarr. Missing entries return the fill template unchanged.
    """
    assert block_id is not None
    row = block_id[1]
    col = block_id[2]
    path = live_lookup.get((row, col))
    if path is None:
        return template
    group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
    try:
        # template shape is (1, H, W[, D]); the staged array is (H, W[, D]).
        arr = np.asarray(group[var_name][...])  # type: ignore[index]
        return arr[np.newaxis, ...]
    finally:
        # Drop the group reference so the underlying store (and its file handles /
        # S3 connections) becomes eligible for immediate GC. Without this, thousands
        # of uncollected store objects accumulate on Dask workers at scale.
        del group


def _build_live_lookup(
    live_chunks: list[ChunkSpec],
    staging_path_fn: Callable[[ChunkSpec], str],
) -> dict[tuple[int, int], str]:
    """Map ``(row, col)`` -> staged path for every live ChunkSpec.

    Keyed by ChunkSpec grid position because dask blocks are ChunkSpec-sized;
    block_id directly indexes this dict.
    """
    return {(chunk.row, chunk.col): staging_path_fn(chunk) for chunk in live_chunks}


def _s5cmd_rm(s3_url: str, log: logging.Logger | logging.LoggerAdapter[logging.Logger]) -> None:
    """Delete all S3 objects under *s3_url* using s5cmd.

    Raises:
        FileNotFoundError: If the s5cmd binary is not on PATH.
        RuntimeError: If s5cmd exits with a non-zero status.
    """
    cmd = ["s5cmd", "rm", f"{s3_url.rstrip('/')}/*"]
    log.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        msg = "s5cmd binary not found — install it or add it to PATH"
        raise FileNotFoundError(msg) from None

    if result.returncode != 0:
        msg = f"s5cmd failed (rc={result.returncode}): {result.stderr.strip()}"
        raise RuntimeError(msg)

    if result.stdout.strip():
        lines = result.stdout.strip().splitlines()
        log.info("s5cmd deleted %d objects from %s", len(lines), s3_url)
    else:
        log.info("s5cmd: no objects found under %s", s3_url)


def read_spatial_coords(mosaic_base: str) -> SpatialCoords:
    """Read projected x/y coordinates and CRS from the reflectance store.

    Args:
        mosaic_base: Base path for the mosaic stores (e.g., "s3://bucket/mosaics/roi").

    Returns:
        SpatialCoords with projected northing and easting arrays.
    """
    reflectance_path = f"{mosaic_base}/reflectance.zarr"
    logger.info("Reading projected coordinates from %s", reflectance_path)
    ref_ds = open_store(reflectance_path)
    coords = SpatialCoords(
        northing=ref_ds["northing"].values,
        easting=ref_ds["easting"].values,
        crs=ref_ds.attrs.get("crs"),
    )
    ref_ds.close()
    return coords


class ZarrWriter:
    """Write embeddings to Zarr stores with pcodec compression.

    Phase 1: Each chunk is written as a standalone Zarr store at a staging location.
    Phase 2: Staged chunks are assembled in parallel via Dask into a single
    Icechunk store using ``to_icechunk()``.
    """

    def __init__(self, staging_base: str, embedding_dim: int = EMBEDDING_DIM) -> None:
        """Initialize the writer.

        Embeddings are always int8-quantized with per-pixel float32 scale factors.

        Args:
            staging_base: Base path for staging chunk Zarrs
                (e.g., "s3://bucket/staging" or "/tmp/staging").
            embedding_dim: Number of embedding dimensions (default 128).
        """
        self.staging_base = staging_base.rstrip("/")
        self.embedding_dim = embedding_dim

    def _staging_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Get the staging path for a chunk."""
        return f"{self.staging_base}/{run_id}/{chunk.label}.zarr"

    def _skip_marker_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Get the skip-marker path for a chunk.

        A zero-byte object whose presence records that the chunk was
        intentionally skipped during inference (every pixel failed the
        validity filter). Distinct from "no state at all", which indicates
        a chunk was never processed or was silently dropped by a crashing
        worker.
        """
        return f"{self.staging_base}/{run_id}/{chunk.label}.skipped"

    def write_skip_marker(self, chunk: ChunkSpec, run_id: str) -> str:
        """Write a zero-byte skip marker for a chunk.

        Called instead of ``write_chunk`` when a live (ROI-intersecting)
        chunk has no valid pixels. The marker lets assembly-only runs
        distinguish a legitimate skip from a silently-failed chunk.
        """
        path = self._skip_marker_path(run_id, chunk)
        fs = _fs_for(path)
        with fs.open(path, "wb") as f:
            f.write(b"")
        logger.info("Wrote skip marker for %s to %s", chunk.label, path)
        return path

    def write_chunk(
        self,
        chunk: ChunkSpec,
        embeddings: np.ndarray,
        run_id: str,
        scales: np.ndarray,
        embeddings_std: np.ndarray | None = None,
        obs_counts: Mapping[str, np.ndarray | None] | None = None,
    ) -> str:
        """Write one chunk's embeddings to a staged intermediate (non-Icechunk) Zarr store.

        Creates an xarray Dataset with an 'embedding' variable (mean) and
        optionally 'embedding_std' and 'scale' variables.

        Args:
            chunk: Chunk specification.
            embeddings: Array of shape (H, W, embedding_dim), float32 or int8.
            run_id: Unique run identifier.
            scales: Per-pixel scale factors of shape (H, W), float32.
                Used to dequantize int8 embeddings.
            embeddings_std: Optional std array, same shape as embeddings, float32.
            obs_counts: Optional dict mapping obs count variable names to (H, W)
                uint16 arrays. Keys should be from ``OBS_COUNT_VARS``.

        Returns:
            Path to the staged Zarr store.
        """
        expected_shape = (chunk.height, chunk.width, self.embedding_dim)
        if embeddings.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {embeddings.shape}"
            raise ValueError(msg)

        path = self._staging_path(run_id, chunk)
        # Sub-chunk staged files so assembly tasks operate on small, not monolithic, chunks
        staged_chunks = (min(500, chunk.height), min(500, chunk.width), self.embedding_dim // BAND_CHUNK_DIVISOR)
        staged_chunks_2d = (min(500, chunk.height), min(500, chunk.width))

        data_vars: dict[str, tuple[list[str], np.ndarray]] = {
            "embeddings": (["northing", "easting", "band"], embeddings),
        }
        encoding: dict[str, dict] = {
            "embeddings": {
                "chunks": staged_chunks,
                "compressors": None,
            },
        }

        if embeddings_std is not None:
            if embeddings_std.shape != expected_shape:
                msg = f"Expected std shape {expected_shape}, got {embeddings_std.shape}"
                raise ValueError(msg)
            data_vars["embedding_std"] = (["northing", "easting", "band"], embeddings_std)
            encoding["embedding_std"] = {
                "chunks": staged_chunks,
                "compressors": None,
            }

        expected_scale_shape = (chunk.height, chunk.width)
        if scales.shape != expected_scale_shape:
            msg = f"Expected scale shape {expected_scale_shape}, got {scales.shape}"
            raise ValueError(msg)
        data_vars["scales"] = (["northing", "easting"], scales)
        encoding["scales"] = {
            "chunks": staged_chunks_2d,
            "compressors": None,
        }

        if obs_counts is not None:
            expected_2d = (chunk.height, chunk.width)
            for var_name in OBS_COUNT_VARS:
                arr = obs_counts.get(var_name)
                if arr is not None:
                    if arr.shape != expected_2d:
                        msg = f"Expected {var_name} shape {expected_2d}, got {arr.shape}"
                        raise ValueError(msg)
                    data_vars[var_name] = (["northing", "easting"], arr)
                    encoding[var_name] = {"chunks": staged_chunks_2d, "compressors": None}

        ds = xr.Dataset(
            data_vars,
            coords={
                "northing": np.arange(chunk.y_start, chunk.y_stop),
                "easting": np.arange(chunk.x_start, chunk.x_stop),
                "band": np.arange(self.embedding_dim),
            },
        )

        ds.to_zarr(path, mode="w", encoding=encoding)
        logger.info("Wrote %s to %s", chunk.label, path)
        return path

    def detect_staged_chunk_size(self, run_id: str) -> int:
        """Detect the inference chunk_size from staged Zarrs.

        Opens the first available staged chunk and returns ``max(height, width)``
        from its shape.  This recovers the ``chunk_size`` used during inference,
        which is needed to re-enumerate the chunk grid for assembly-only or
        resume runs.

        Uses :meth:`_list_staged_labels` rather than assuming ``chunk_0_0``
        exists — sparse ROIs may not have staged that chunk.

        Args:
            run_id: Run identifier whose staged chunks to inspect.

        Returns:
            The chunk_size (pixels) used during inference.

        Raises:
            FileNotFoundError: If no staged chunks exist for the run.
        """
        labels = self._list_staged_labels(run_id)
        if not labels:
            raise FileNotFoundError(f"No staged chunks found for run '{run_id}' under {self.staging_base}")
        path = f"{self.staging_base}/{run_id}/{labels[0]}.zarr"
        try:
            group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
        except Exception as exc:
            raise FileNotFoundError(f"Cannot open {path}: {exc}") from exc
        arr: zarr.Array = group["embeddings"]  # type: ignore[assignment]
        h, w, _ = arr.shape
        return max(h, w)

    def verify_staged_completeness(
        self,
        run_id: str,
        expected_chunks: list[ChunkSpec],
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    ) -> None:
        """Verify all expected chunks have either a staged zarr or a skip marker.

        A live (ROI-intersecting) chunk can resolve in two ways: inference
        produced embeddings (staged zarr), or every pixel failed the validity
        filter (skip marker). Any live chunk with neither indicates a silent
        failure — Ray worker died mid-chunk, a bug in the actor, etc. — and
        assembly must fail loudly rather than produce a zero-filled output
        that looks identical to a legitimate skip.

        Args:
            run_id: Run identifier (locates the staging directory).
            expected_chunks: Chunks that are expected to resolve (i.e. ROI-
                intersecting chunks). Non-intersecting chunks are filled in
                at assembly time and should not be passed here.
            log: Optional logger.

        Raises:
            IncompleteStageError: If any live chunk has neither a staged
                zarr nor a skip marker, or if there are extra labels that
                don't match any expected chunk.
        """
        _log = log or logger
        staged = set(self._list_staged_labels(run_id))
        skipped = set(self._list_skip_marker_labels(run_id))
        resolved = staged | skipped
        expected = {c.label for c in expected_chunks}

        missing = expected - resolved
        extra = resolved - expected
        both = staged & skipped

        if missing or extra or both:
            parts = [f"Staged chunks for run '{run_id}' do not match the expected chunk grid."]
            parts.append(f"Expected {len(expected)} chunks, found {len(staged)} staged + {len(skipped)} skipped.")
            if missing:
                sample = sorted(missing)[:10]
                parts.append(
                    f"{len(missing)} missing (neither zarr nor skip marker): "
                    f"{sample}{'...' if len(missing) > 10 else ''}"
                )
            if extra:
                sample = sorted(extra)[:10]
                parts.append(f"{len(extra)} unexpected: {sample}{'...' if len(extra) > 10 else ''}")
            if both:
                sample = sorted(both)[:10]
                parts.append(
                    f"{len(both)} chunks have BOTH a staged zarr and a skip marker: "
                    f"{sample}{'...' if len(both) > 10 else ''}"
                )
            raise IncompleteStageError("\n".join(parts))

        _log.info(
            "Staged completeness verified: %d staged + %d skipped = %d/%d live chunks resolved",
            len(staged),
            len(skipped),
            len(resolved),
            len(expected),
        )

    def _list_staged_labels(self, run_id: str) -> list[str]:
        """List chunk labels that have staged Zarrs in the run directory.

        Returns:
            List of chunk label strings (e.g. ``["chunk_0_0", "chunk_0_1"]``).
            Empty list if the staging directory doesn't exist.
        """
        return self._list_labels_by_suffix(run_id, ".zarr")

    def _list_skip_marker_labels(self, run_id: str) -> list[str]:
        """List chunk labels that have skip markers in the run directory."""
        return self._list_labels_by_suffix(run_id, ".skipped")

    def _list_labels_by_suffix(self, run_id: str, suffix: str) -> list[str]:
        staging_dir = f"{self.staging_base}/{run_id}"
        fs = _fs_for(staging_dir)
        try:
            # refresh=True forces a fresh S3 LIST instead of returning the
            # cached dir listing. Staged artifacts (zarrs + skip markers) are
            # written by Ray actor processes, not the flow runner, so the
            # runner's fsspec dircache never sees them invalidated and would
            # otherwise return a stale snapshot from an earlier ls() call.
            entries = fs.ls(staging_dir, detail=False, refresh=True)
        except FileNotFoundError:
            return []
        labels = []
        for entry in entries:
            name = entry.rstrip("/").rsplit("/", 1)[-1]
            if name.endswith(suffix):
                labels.append(name.removesuffix(suffix))
        return labels

    def _validate_staged_chunk(
        self,
        run_id: str,
        chunk: ChunkSpec,
        required_vars: list[str],
    ) -> str | None:
        """Validate a single staged chunk Zarr.

        Returns:
            ``None`` if valid, or a string describing the problem.
        """
        path = self._staging_path(run_id, chunk)
        expected_shape = (chunk.height, chunk.width, self.embedding_dim)
        try:
            group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
            for var in required_vars:
                if var not in group:
                    return f"missing variable '{var}'"
                arr: zarr.Array = group[var]  # type: ignore[assignment]
                if arr.shape != expected_shape:
                    return f"'{var}' shape {arr.shape} != expected {expected_shape}"
                expected_dtype = np.int8 if var == "embeddings" else np.float32
                if arr.dtype != expected_dtype:
                    return f"'{var}' dtype {arr.dtype} != expected {expected_dtype}"
        except Exception as exc:
            return f"failed to open: {exc}"
        return None

    def scan_existing_staged_chunks(
        self,
        run_id: str,
        chunks: list[ChunkSpec],
        *,
        compute_std: bool = False,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    ) -> set[str]:
        """Scan staging for already-written chunks, validate each, return valid labels.

        Args:
            run_id: Run identifier (locates the staging directory).
            chunks: Full list of chunk specs for this ROI.
            compute_std: If True, also require ``embedding_std`` in each staged Zarr.
            log: Optional logger.

        Returns:
            Set of ``chunk.label`` strings for chunks already staged and valid.

        Raises:
            RuntimeError: If any staged Zarrs are invalid. Lists all invalid
                paths so the user can remove them before retrying.
        """
        _log = log or logger
        staged_labels = self._list_staged_labels(run_id)
        skip_marker_labels = set(self._list_skip_marker_labels(run_id))
        if not staged_labels and not skip_marker_labels:
            _log.info("No staged chunks or skip markers found for run %s — starting fresh", run_id)
            return set()

        _log.info(
            "Found %d staged chunk Zarrs and %d skip markers — validating",
            len(staged_labels),
            len(skip_marker_labels),
        )

        chunk_by_label = {c.label: c for c in chunks}
        required_vars = ["embeddings"] + (["embedding_std"] if compute_std else [])
        valid: set[str] = set()
        invalid_paths: list[tuple[str, str]] = []

        for label in staged_labels:
            chunk = chunk_by_label.get(label)
            if chunk is None:
                invalid_paths.append((f"{label}.zarr", "no matching ChunkSpec (stale chunk from a different grid?)"))
                continue
            reason = self._validate_staged_chunk(run_id, chunk, required_vars)
            if reason:
                invalid_paths.append((f"{label}.zarr", reason))
            else:
                valid.add(label)

        for label in skip_marker_labels:
            if label not in chunk_by_label:
                invalid_paths.append((f"{label}.skipped", "skip marker has no matching ChunkSpec"))
                continue
            if label in valid:
                invalid_paths.append((f"{label}.skipped", "chunk has BOTH a staged zarr and a skip marker"))
                continue
            valid.add(label)

        if invalid_paths:
            details = "\n".join(
                f"  {self.staging_base}/{run_id}/{suffix} — {reason}" for suffix, reason in invalid_paths
            )
            msg = (
                f"{len(invalid_paths)} invalid staged chunk artifact(s) found. Remove them before retrying:\n{details}"
            )
            raise RuntimeError(msg)

        _log.info(
            "Validated %d existing artifacts (%d zarrs + %d skip markers) — will skip during inference",
            len(valid),
            len(valid & set(staged_labels)),
            len(valid & skip_marker_labels),
        )
        return valid

    def _staged_vars_present(self, run_id: str, chunks: list[ChunkSpec], var_names: tuple[str, ...]) -> set[str]:
        """Check which variables exist in the staged Zarrs.

        Probes the largest chunk (same logic as _detect_staged_chunk_size).
        Opens the zarr group once and checks all variable names. Callers must
        pass only chunks that have staged files.
        """
        probe = max(chunks, key=lambda c: c.height * c.width)
        path = self._staging_path(run_id, probe)
        try:
            group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
            return {v for v in var_names if v in group}
        except (FileNotFoundError, KeyError, zarr.errors.GroupNotFoundError):
            return set()

    def _detect_staged_chunk_size(self, run_id: str, chunks: list[ChunkSpec], var_name: str) -> tuple[int, ...]:
        """Read on-disk chunk shape from the largest staged Zarr.

        Used only to choose output-zarr encoding — dask task granularity is
        independent (one task per ChunkSpec, regardless of sub-chunk size).

        Probes the largest chunk by pixel count to avoid edge chunks whose
        on-disk chunk shape is smaller than interior chunks.

        Args:
            run_id: Run identifier for locating staged files.
            chunks: Chunk specs (largest is probed).
            var_name: Variable name to inspect.

        Returns:
            On-disk chunk shape tuple (length matches the variable's dimensionality).
        """
        largest = max(chunks, key=lambda c: c.height * c.width)
        probe_path = self._staging_path(run_id, largest)
        group = zarr.open_group(probe_path, mode="r", storage_options=_staged_storage_options(probe_path))
        return tuple(group[var_name].metadata.chunk_grid.chunk_shape)  # type: ignore[union-attr]

    def _chunk_grid_sizes(self, chunks: list[ChunkSpec]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Return per-row heights and per-col widths for the ChunkSpec grid.

        The ChunkSpec grid is rectangular but edge chunks may be smaller than
        interior chunks, so dask needs a tuple-per-axis rather than a scalar
        chunk size. Assumes each (row, col) exists exactly once and row-r
        chunks share a height, col-c chunks share a width.
        """
        by_row: dict[int, int] = {}
        by_col: dict[int, int] = {}
        for c in chunks:
            by_row[c.row] = c.height
            by_col[c.col] = c.width
        row_heights = tuple(by_row[r] for r in sorted(by_row))
        col_widths = tuple(by_col[c] for c in sorted(by_col))
        return row_heights, col_widths

    def _build_var_grid(
        self,
        chunks: list[ChunkSpec],
        live_labels: set[str],
        run_id: str,
        var_name: str,
    ) -> da.Array:
        """Build a lazy 4D Dask array at ChunkSpec granularity.

        One dask block per ChunkSpec spatially, full band axis as one block.
        Block count equals ``n_chunks`` — much smaller than
        ``n_chunks * sub_chunks_per_chunk`` and the only scale the Dask
        scheduler can handle at cornbelt ROI size (millions of TaskStates
        OOM an 8 GB scheduler).

        Output zarr sub-chunking is controlled independently via encoding at
        :meth:`assemble` write time; zarr handles the per-block fan-out into
        small on-disk chunks internally when ``align_chunks=True``.

        The graph is two unmaterialized Blockwise layers: a ``da.full``
        template and a ``map_blocks`` dispatcher that either reads the staged
        Zarr for live positions or returns the fill template unchanged.

        Returns:
            Lazy 4D Dask array of shape ``(1, total_y, total_x, embedding_dim)``.
        """
        total_y = max(c.y_stop for c in chunks)
        total_x = max(c.x_stop for c in chunks)
        row_heights, col_widths = self._chunk_grid_sizes(chunks)

        live_chunks = [c for c in chunks if c.label in live_labels]
        logger.info(
            "Building %s with %d blocks (%d rows x %d cols, band=1 block)",
            var_name,
            len(chunks),
            len(row_heights),
            len(col_widths),
        )

        emb_dtype = np.int8 if var_name == "embeddings" else np.float32
        fill = 0 if var_name == "embeddings" else float("nan")

        live_lookup = _build_live_lookup(live_chunks, lambda c: self._staging_path(run_id, c))

        chunks_spec = ((1,), row_heights, col_widths, (self.embedding_dim,))
        template = da.full((1, total_y, total_x, self.embedding_dim), fill, chunks=chunks_spec, dtype=emb_dtype)
        return template.map_blocks(
            _assemble_var_block,
            var_name=var_name,
            live_lookup=live_lookup,
            dtype=emb_dtype,
        )

    def _build_var_grid_2d(
        self,
        chunks: list[ChunkSpec],
        live_labels: set[str],
        run_id: str,
        var_name: str,
        dtype: np.dtype | type = np.float32,
    ) -> da.Array:
        """Build a lazy 3D Dask array (time=1, y, x) for a spatial-only variable.

        Same ChunkSpec-granularity design as :meth:`_build_var_grid`.

        Returns:
            Lazy 3D Dask array of shape ``(1, total_y, total_x)``.
        """
        total_y = max(c.y_stop for c in chunks)
        total_x = max(c.x_stop for c in chunks)
        row_heights, col_widths = self._chunk_grid_sizes(chunks)

        live_chunks = [c for c in chunks if c.label in live_labels]
        logger.info(
            "Building %s with %d blocks (%d rows x %d cols)",
            var_name,
            len(chunks),
            len(row_heights),
            len(col_widths),
        )

        fill = float("nan") if np.issubdtype(np.dtype(dtype), np.floating) else 0
        live_lookup = _build_live_lookup(live_chunks, lambda c: self._staging_path(run_id, c))

        chunks_spec = ((1,), row_heights, col_widths)
        template = da.full((1, total_y, total_x), fill, chunks=chunks_spec, dtype=dtype)
        return template.map_blocks(
            _assemble_var_block,
            var_name=var_name,
            live_lookup=live_lookup,
            dtype=dtype,
        )

    def _build_mosaic(
        self,
        chunks: list[ChunkSpec],
        live_labels: set[str],
        run_id: str,
        compute_std: bool,
        run_started_at: datetime.datetime,
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        time_window: TimeWindow | None = None,
    ) -> xr.Dataset:
        """Build a lazy xarray Dataset from staged chunks at 500x500 granularity.

        Uses ``_build_var_grid`` to create a direct dask graph with one task
        per on-disk sub-chunk. No ``combine_by_coords``, no concatenation tasks.

        Args:
            chunks: Full chunk grid with row/col positions.
            live_labels: Labels of chunks with a staged Zarr; others are
                filled with zeros/NaN in the Dask graph.
            run_id: Run identifier for locating staged files.
            compute_std: Whether to include embedding_std variable.
            run_started_at: Flow trigger time used as the time coordinate.
                Ignored when *time_window* is provided.
            y_coords: Projected y coordinates from the input store. Falls back
                to integer pixel indices if not provided.
            x_coords: Projected x coordinates from the input store. Falls back
                to integer pixel indices if not provided.
            time_window: If provided, use the window end month as the time
                coordinate and attach window metadata as dataset attributes.

        Returns:
            Lazy xarray Dataset with dims ``(time, northing, easting, band)``.
        """
        embedding = self._build_var_grid(chunks, live_labels, run_id, "embeddings")

        if time_window:
            time_date = np.datetime64(time_window.window_end_label, "ns")
        else:
            time_date = np.datetime64(run_started_at.date(), "ns")

        coords: dict[str, object] = {
            "time": [time_date],
            "northing": y_coords if y_coords is not None else np.arange(embedding.shape[1]),
            "easting": x_coords if x_coords is not None else np.arange(embedding.shape[2]),
            "band": np.arange(self.embedding_dim),
        }

        ds = xr.Dataset(
            {"embeddings": (["time", "northing", "easting", "band"], embedding)},
            coords=coords,
        )

        if time_window:
            ds.attrs["time_convention"] = "12mo_window_end"

        if compute_std:
            ds["embedding_std"] = (
                ["time", "northing", "easting", "band"],
                self._build_var_grid(chunks, live_labels, run_id, "embedding_std"),
            )

        ds["scales"] = (
            ["time", "northing", "easting"],
            self._build_var_grid_2d(chunks, live_labels, run_id, "scales"),
        )

        live_chunks = [c for c in chunks if c.label in live_labels]
        for obs_var in self._staged_vars_present(run_id, live_chunks, OBS_COUNT_VARS):
            ds[obs_var] = (
                ["time", "northing", "easting"],
                self._build_var_grid_2d(chunks, live_labels, run_id, obs_var, dtype=np.uint16),
            )

        return ds

    def assemble(
        self,
        chunks: list[ChunkSpec],
        total_y: int,
        total_x: int,
        run_id: str,
        output_path: str,
        *,
        roi_zarr_path: str,
        compute_std: bool = False,
        run_started_at: datetime.datetime | None = None,
        mosaic_base: str | None = None,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
        time_window: TimeWindow | None = None,
        tile_id: str | None = None,
        model_version: str | None = None,
        encoder_version: str | None = None,
        manifest: EmbeddingManifest | None = None,
        n_workers: int,
        get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
        s3_region: str | None = None,
    ) -> str:
        """Assemble staged chunk Zarrs into the output Icechunk store.

        Builds a lazy Dask-backed mosaic from staged sub-chunks and writes via
        ``to_icechunk()``. Appends along the time dimension if the output store
        already exists. Creates a new store otherwise.

        Inference only runs on chunks that intersect the ROI mask, so staged
        files exist only for those chunks. Assembly re-runs the same ROI
        filter to determine which chunks have staged files; non-intersecting
        chunks are filled in the output Dask graph with zeros (integer dtypes)
        or NaN (float dtypes) at the same spatial position.

        When ``mosaic_base`` is provided, projected x/y coordinates are copied
        from the input reflectance store so the output is georeferenced.

        Args:
            chunks: Full chunk grid (both live and non-live).
            total_y: Total mosaic height.
            total_x: Total mosaic width.
            run_id: Run identifier.
            output_path: Final output Icechunk store path.
            roi_zarr_path: Path to the ROI boolean zarr. Assembly re-enumerates
                live chunks from this to avoid marshaling the list through Prefect.
            compute_std: Whether staged chunks contain embedding_std data.
            run_started_at: Flow trigger time for the time coordinate.
                Falls back to now if not provided. Ignored when *time_window*
                is provided.
            mosaic_base: Base path for the input mosaic stores. If provided, the
                reflectance store is opened to copy projected coordinates and CRS.
            log: Optional logger (e.g., Prefect's run logger). Falls back to module logger.
            time_window: If provided, use the window end month as the time
                coordinate and store window metadata in dataset attributes.
            tile_id: Sentinel-2 MGRS tile ID (e.g. ``"37PBM"``). Used to derive
                the EPSG code for ``proj:`` convention attributes when the
                mosaic store does not carry a ``crs`` attr.
            model_version: Checkpoint filename stem, recorded as ``checkpoint_id``.
            encoder_version: Model FAMILY (e.g. ``"v2-large"``) selecting the public
                ``geoemb:model`` URL. Omitting it stamps the default model's URL.
            manifest: Typed manifest for append-safety validation.
                Written on create, validated on append.
            n_workers: Max Dask worker count for this assembly. Used to divide
                ``TARGET_AGGREGATE_S3_CONCURRENCY`` across workers so the
                fleet-wide PUT rate stays under S3's per-prefix ceiling.
            get_credentials: Optional credential callback forwarded to
                Icechunk for the output store. See
                :func:`tessera_embeddings.storage.zarr_store._create_storage`.
            s3_region: Optional S3 region override for the output store.
                Defaults to us-west-2 (see ``zarr_store._DEFAULT_S3_REGION``).

        Returns:
            Path to the assembled output store.
        """
        _log = log or logger

        # Determine which chunks have staged zarrs. A chunk that intersects the
        # ROI but was skipped during inference (all pixels failed validity) has
        # a skip marker instead of a zarr — exclude those from live_labels so
        # their positions fall through to the Dask fill template rather than
        # triggering a GroupNotFoundError on the missing zarr.
        roi_live_chunks = filter_chunks_by_roi_mask(chunks, roi_zarr_path)
        skipped_labels = set(self._list_skip_marker_labels(run_id))
        live_chunks = [c for c in roi_live_chunks if c.label not in skipped_labels]
        live_labels = {c.label for c in live_chunks}
        _log.info(
            "Assembling %d chunks (%d live with staged zarr, %d skipped, %d fill) into %s",
            len(chunks),
            len(live_labels),
            len(roi_live_chunks) - len(live_chunks),
            len(chunks) - len(roi_live_chunks),
            output_path,
        )

        # Validate that total_y/total_x match the actual chunk grid extent.
        # A mismatch would produce dask graph entries with no corresponding data.
        actual_y = max(c.y_stop for c in chunks)
        actual_x = max(c.x_stop for c in chunks)
        if actual_y != total_y or actual_x != total_x:
            msg = f"total_y/total_x ({total_y}, {total_x}) doesn't match chunk grid extent ({actual_y}, {actual_x})"
            raise ValueError(msg)

        started = run_started_at or datetime.datetime.now(datetime.UTC)

        # Extract projected coordinates and CRS from the input reflectance store.
        spatial: SpatialCoords | None = None
        if mosaic_base:
            spatial = read_spatial_coords(mosaic_base)
            _log.info("Using projected coordinates from %s", mosaic_base)

        # Build lazy Dask-backed mosaic: task granularity matches on-disk chunk size.
        mosaic = self._build_mosaic(
            chunks,
            live_labels,
            run_id,
            compute_std,
            run_started_at=started,
            y_coords=spatial.northing if spatial else None,
            x_coords=spatial.easting if spatial else None,
            time_window=time_window,
        )
        _log.info("Mosaic built: %s", dict(mosaic.sizes))

        # Divide the fleet-wide S3 concurrency target across workers. icechunk's
        # cap is per-Repository (per-worker after session fork), so we must
        # pre-divide by n_workers to keep aggregate under S3's per-prefix
        # ceiling. See TARGET_AGGREGATE_S3_CONCURRENCY for the rationale.
        #
        # The floor is 1, not a larger number: every forked Repository issues at
        # least 1 concurrent PUT, so aggregate >= n_workers regardless of the
        # divisor. A floor above 1 only holds the target while n_workers is small
        # enough that the division still lands >= the floor; past that it pins at
        # the floor and aggregate grows linearly with n_workers, blowing the
        # target (e.g. floor=4 at 200 workers -> 800 concurrent PUTs -> SlowDown).
        # AssemblyConfig.max_workers is capped at the target so n_workers never
        # drives aggregate over it.
        per_worker_cap = max(1, TARGET_AGGREGATE_S3_CONCURRENCY // n_workers)

        # Split each spatial axis's manifest into 32-chunk shards. Embeddings are
        # written in 500-px spatial chunks, so a 32-chunk shard is ~16k px/axis —
        # matching DEFAULT_MANIFEST_SPLIT_SIZES' ~16k-px target.
        #
        # Time is split at 1 shard per timestep. An assembly append writes one
        # full-spatial-extent timestep, so it touches every spatial shard — and
        # since manifest objects are immutable, each touched shard is rewritten in
        # full. With "time": 1 each timestep is its own shard, so an append writes a
        # new shard and rewrites zero prior ones; the per-append manifest rewrite
        # (and the peak worker RAM building it) stays bounded by a single timestep
        # rather than the whole time series. We never region-write across time on
        # this store (appends only add whole timesteps) and the series tops out at
        # ~30 dates, so the extra shard objects and read fan-out are negligible.
        #
        # This wraps appends to pre-existing stores too, which is intentional and
        # safe. A store created before splitting was introduced has one unsplit
        # manifest; the first append under this block re-shards the touched array
        # in that commit (a one-time migration — existing chunk data is untouched
        # and reads back identically), and every later commit rewrites only the
        # shards it touches. icechunk merges the split config into the store's
        # persisted config on open, so the layout follows the array forward with no
        # manual migration. Verified against icechunk 2.0.4.
        #
        # The context is scoped tightly here — around the output store open/write
        # only — so that read-only opens of other stores (e.g. the mosaic base for
        # spatial coords) are not inadvertently opened under the split config.
        with manifest_split({"northing": 32, "easting": 32, "time": 1}):
            # scatter_initial_credentials: to_icechunk pickles the session out to
            # every Dask worker. With a credential callback, eager scatter
            # caches one credential set on the driver so workers don't all stampede
            # the credential provider on deserialisation.
            repo, is_new = open_or_create_repo(
                output_path,
                max_concurrent_requests=per_worker_cap,
                get_credentials=get_credentials,
                region=s3_region,
                scatter_initial_credentials=get_credentials is not None,
            )
            # Persist the cap into the repo's config blob. to_icechunk ships the
            # session out and each worker re-opens the Repository — without
            # save_config, those workers read the persisted config (defaults) and
            # the runtime override is silently dropped, so concurrency stays at
            # icechunk's 256 default regardless of what we pass here.
            repo.save_config()
            session = repo.writable_session("main")

            # Validate manifest before appending to an existing store
            if not is_new and manifest:
                prev_root = zarr.open_group(session.store, mode="r")
                manifest.validate_against(extract_manifest(prev_root.attrs), output_path)

            # Snapshot existing time_windows BEFORE to_icechunk, which may
            # overwrite root attrs on append.
            prev_time_windows: dict = {}
            if not is_new and time_window:
                prev_root = zarr.open_group(session.store, mode="r")
                prev_time_windows = dict(prev_root.attrs.get("time_windows", {}))  # type: ignore[arg-type]

            if is_new:
                # Output zarr sub-chunking is read from the staged files directly,
                # not from dask block size: dask blocks are ChunkSpec-sized (one
                # task per ChunkSpec, keeps the scheduler graph small), but on-disk
                # zarr chunks must remain 500x500x(embedding_dim/BAND_CHUNK_DIVISOR)
                # for downstream partial-read performance. to_icechunk handles the
                # per-dask-block fan-out into smaller zarr chunks via align_chunks=True.
                sub_h, sub_w, sub_band = self._detect_staged_chunk_size(run_id, live_chunks, "embeddings")
                _log.info("Output zarr sub-chunks: (%d, %d, %d) from staged files", sub_h, sub_w, sub_band)
                encoding_ic: dict[str, dict] = {
                    "embeddings": {"chunks": (1, sub_h, sub_w, sub_band)},
                    "time": {"units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian"},
                }
                # fill_value=NaN matches the out-of-ROI fill used in the mosaic
                # graph for these float vars, so all-empty sub-chunks collapse to
                # no object on write (zarr's write_empty_chunks=False default) and
                # don't inflate the S3 prefix. The default 0.0 fill would never
                # match the NaN footprint, forcing a real object per empty chunk.
                if compute_std:
                    encoding_ic["embedding_std"] = {
                        "chunks": (1, sub_h, sub_w, sub_band),
                        "fill_value": float("nan"),
                        "serializer": pcodec_serializer(),
                        "compressors": None,
                    }
                encoding_ic["scales"] = {
                    "chunks": (1, sub_h, sub_w),
                    "fill_value": float("nan"),
                    "serializer": pcodec_serializer(),
                    "compressors": None,
                }
                for obs_var in OBS_COUNT_VARS:
                    if obs_var in mosaic:
                        encoding_ic[obs_var] = {"chunks": (1, sub_h, sub_w), "compressors": None}
                to_icechunk(mosaic, session, mode="w", encoding=encoding_ic, align_chunks=True, split_every=8)
                _log.info("Wrote new store at %s", output_path)
            else:
                # Existing store: append along time (no encoding — existing metadata preserved)
                to_icechunk(mosaic, session, mode="a", append_dim="time", align_chunks=True, split_every=8)
                _log.info("Appended to existing store at %s", output_path)

            # Set root attrs on same session
            root = zarr.open_group(session.store, mode="r+")
            root.attrs["run_id"] = run_id
            root.attrs["total_y"] = total_y
            root.attrs["total_x"] = total_x
            root.attrs["embedding_dim"] = self.embedding_dim
            root.attrs["run_started_at"] = started.isoformat()
            root.attrs["run_completed_at"] = datetime.datetime.now(datetime.UTC).isoformat()

            # GeoZarr convention attributes (proj:, spatial:, geoemb:).
            # Set on every write (create and append) so they survive to_icechunk attr overwrites.
            conv_attrs = build_convention_attrs(
                tile_id=tile_id,
                epsg_code=spatial.crs if spatial else None,
                total_y=total_y,
                total_x=total_x,
                embedding_dim=self.embedding_dim,
                y_coords=spatial.northing if spatial else None,
                x_coords=spatial.easting if spatial else None,
                model_version=model_version,
                encoder_version=encoder_version,
            )
            if conv_attrs:
                # Drop any retired tessera:* attrs first: appending to a store
                # created before the geoemb switch would otherwise leave the old
                # keys behind (dict.update only overwrites the keys it carries),
                # so the store would advertise both conventions. zarr_conventions
                # itself is replaced wholesale by the update.
                for stale in [k for k in root.attrs if str(k).startswith("tessera:")]:
                    del root.attrs[stale]
                root.attrs.update(conv_attrs)

            if manifest:
                root.attrs["_manifest"] = manifest.to_dict()
                _log.info("Wrote _manifest to %s", output_path)

            # Merge time_windows: combine previously-snapshotted entries with the
            # current window, then write back to root attrs. This ensures each
            # appended time step retains its provenance.
            if time_window:
                window_meta: dict[str, list[str]] = {
                    "range": [
                        f"{time_window.window_start[0]}-{time_window.window_start[1]:02d}",
                        f"{time_window.window_end[0]}-{time_window.window_end[1]:02d}",
                    ],
                }
                prev_time_windows[time_window.window_end_label] = window_meta
                root.attrs["time_windows"] = prev_time_windows
                root.attrs["time_convention"] = "12mo_window_end"

            session.commit(f"Run {run_id}: {len(chunks)} chunks assembled")
        _log.info("Assembly complete: %s", output_path)
        return output_path

    def cleanup_staging(
        self,
        run_id: str,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    ) -> None:
        """Delete the staging directory for a completed run.

        S3 paths are removed with ``s5cmd rm``, falling back to fsspec's
        ``rm`` if s5cmd is unavailable or fails. Non-S3 paths use fsspec.

        Args:
            run_id: Run identifier whose staging artifacts should be deleted.
            log: Optional logger. Falls back to the module logger.
        """
        _log = log or logger
        target = f"{self.staging_base}/{run_id}"

        _log.info("Cleaning up staging: %s", target)

        # For S3, prefer s5cmd: it deletes far faster than fsspec's per-key
        # serial DELETEs. Fall back to fsspec if s5cmd is unavailable or errors.
        if fsspec.utils.get_protocol(target) == "s3":
            try:
                _s5cmd_rm(target, _log)
                return
            except (FileNotFoundError, RuntimeError) as exc:
                _log.warning("s5cmd cleanup of %s failed (%s) — falling back to fsspec", target, exc)

        try:
            fs = _fs_for(target)
            if fs.exists(target):
                fs.rm(target, recursive=True)
        except Exception:
            _log.warning("Failed to clean up staging directory %s", target, exc_info=True)
