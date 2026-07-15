"""Embedding output writers.

Staged writes (one Zarr per chunk) use raw uncompressed bytes for zero CPU
overhead on GPU actors; the on-disk staged pieces are inner-chunk-sized
(``INNER_PX`` = 256 px, full band) so a staged tile is exactly the inner-chunk
grid of the output region it becomes. The final Icechunk store's geometry
(chunks, shards, codecs) comes from a :class:`StoreLayout` preset — ``LEGACY``
for single-ROI stores, ``GLOBAL_V1`` for the global campaign's zone groups.

Assembly is raw-zarr, not Dask: the coordinator forks an icechunk session,
worker processes write staged-tile pixels straight into the output arrays via
plain zarr assignment, and the coordinator merges the forks and commits once
(the cooperative fork/merge model shared with
:mod:`tessera_embeddings.storage.shard_writer`). No task graph is ever built
over the store, so assembly cost scales with the *live* pixels, not the grid.

Two write-conflict regimes, one engine:

* **Single-ROI** (``assemble``): output chunks (500 px) don't align with the
  2048-px inference tiles, so workers partition the mosaic into *northing bands
  aligned to the output write granularity* — no two forks ever touch the same
  output chunk, and x-boundary partial chunks are read-modify-written
  sequentially inside one fork (icechunk sessions are read-your-writes).
* **Global** (``assemble_global``): one inference tile == one 2048-px shard
  (ADR-008 D3), so whole tiles round-robin across workers via
  :func:`~tessera_embeddings.storage.shard_writer.write_year_shards` — every
  shard object is emitted once, lean, with ocean inner chunks elided.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import multiprocessing
import subprocess
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Any, cast

import fsspec
import icechunk
import numpy as np
import xarray as xr
import zarr

from tessera_embeddings.config.inference import EMBEDDING_DIM, TimeWindow
from tessera_embeddings.config.store_layout import INNER_PX, LEGACY, OBS_COUNT_VARS, StoreLayout
from tessera_embeddings.inference.chunk_spec import ChunkSpec, filter_chunks_by_roi_mask
from tessera_embeddings.inference.conventions import build_convention_attrs
from tessera_embeddings.storage.empty_store import _write_coord_arrays
from tessera_embeddings.storage.global_store import open_global_repo
from tessera_embeddings.storage.manifest import EmbeddingManifest, extract_manifest
from tessera_embeddings.storage.shard_writer import CommitGate, commit_with_rebase, write_year_shards
from tessera_embeddings.storage.zarr_store import (
    manifest_split,
    open_or_create_repo,
    open_store,
    read_time_values,
)

logger = logging.getLogger(__name__)


STAGED_READ_CONFIG_KWARGS = {"retries": {"max_attempts": 10, "mode": "adaptive"}}
"""botocore retry config for staged-Zarr GETs during assembly.

Staged reads fan out across every worker process at once, so a momentary GET burst
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


TARGET_AGGREGATE_S3_CONCURRENCY = 100
"""Fleet-wide ceiling on concurrent S3 PUTs during assembly, divided across workers.

icechunk's ``max_concurrent_requests`` is per-Repository-instance, and each
assembly worker process carries its own pickled fork of the session (and thus
its own request pool). So the effective aggregate concurrency is
``n_workers * per_worker_cap``, not the per-worker cap alone. We target ~100
concurrent PUTs fleet-wide (roughly 1/35 of S3's ~3500 req/s/prefix ceiling at
~100 ms PUT latency) to leave headroom for retries and avoid 503 SlowDown.

The coordinator opens the repo with ``max_concurrent_requests = target //
n_workers`` and the forks inherit it — no ``save_config`` persistence needed,
because forks travel by pickle rather than re-opening the repo from its URI.

icechunk writes chunk objects to a flat ``chunks/<random-id>`` keyspace, so the
keys spread across S3 partitions well — but partition splitting is adaptive and a
hard burst overruns the per-prefix rate before (and even after) S3 adapts, which
is exactly the SlowDown observed at 800 concurrent PUTs. Since ``per_worker_cap``
floors at 1, aggregate is >= n_workers — keep worker counts at or under this
target (``AssemblyConfig`` caps them far lower).
"""


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


# =============================================================================
# Raw-zarr assembly engine (module-level so spawn workers can unpickle them)
# =============================================================================


def _partition_bands(total_y: int, granularity: int, n_workers: int) -> list[tuple[int, int]]:
    """Split ``[0, total_y)`` into at most ``n_workers`` bands aligned to ``granularity``.

    Bands start and end on multiples of the output's northing write granularity
    (shard height when sharded, chunk height otherwise), so no two workers ever
    touch the same output chunk — the fork/merge write-conflict invariant.
    Whole granularity units are spread as evenly as possible; the last band
    absorbs the ragged tail.
    """
    n_units = -(-total_y // granularity)
    n_bands = max(1, min(n_workers, n_units))
    base, extra = divmod(n_units, n_bands)
    bands: list[tuple[int, int]] = []
    y = 0
    for i in range(n_bands):
        units = base + (1 if i < extra else 0)
        y_stop = min(total_y, y + units * granularity)
        bands.append((y, y_stop))
        y = y_stop
    return bands


def _write_granularity(node: zarr.Group, variables: Iterable[str]) -> int:
    """The output's northing write granularity: shard height if sharded, else chunk height.

    All data variables must agree (both presets do); a disagreement would let
    two bands share an output object, so it is an error rather than a max().
    """
    sizes = {}
    for var in variables:
        arr = cast(zarr.Array, node[var])
        sizes[var] = (arr.shards or arr.chunks)[1]
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Data variables disagree on northing write granularity: {sizes}")
    return next(iter(sizes.values()))


def _fill_band_worker(payload: dict[str, Any]) -> Any:  # noqa: ANN401 — returns a ForkSession
    """Write one northing band's staged-tile slices into a forked session.

    For each staged tile overlapping the band, reads the tile's overlapping
    y-slice (one slice in flight per worker — peak RAM is bounded by one tile,
    ~0.5 GB int8 at 2048 px) and writes it into every output array with a plain
    zarr assignment. Partial output chunks at tile x-boundaries are
    read-modify-written sequentially within this fork (icechunk sessions are
    read-your-writes), so the merged result is exact. ``payload["clear"]``
    tiles (skip-marked on a same-date overwrite — see :meth:`ZarrWriter.assemble`)
    get the fill value written over their footprint so a prior run's data
    can't survive under a rerun's skip marker. Returns the fork for the
    coordinator to merge.
    """
    fork = payload["fork"]
    t = int(payload["time_index"])
    y0b, y1b = payload["band"]
    root = zarr.open_group(fork.store, mode="a")
    arrays = {var: cast(zarr.Array, root[var]) for var in payload["variables"]}
    for tile in payload["clear"]:
        y0, y1 = max(tile.y_start, y0b), min(tile.y_stop, y1b)
        for arr in arrays.values():
            shape = (1, y1 - y0, tile.width, *arr.shape[3:]) if arr.ndim == 4 else (1, y1 - y0, tile.width)
            block = np.full(shape, arr.fill_value, dtype=arr.dtype)
            if arr.ndim == 4:
                arr[t : t + 1, y0:y1, tile.x_start : tile.x_stop, :] = block
            else:
                arr[t : t + 1, y0:y1, tile.x_start : tile.x_stop] = block
    for tile, path in payload["tiles"]:
        y0, y1 = max(tile.y_start, y0b), min(tile.y_stop, y1b)
        staged = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
        for var, arr in arrays.items():
            block = np.asarray(cast(zarr.Array, staged[var])[y0 - tile.y_start : y1 - tile.y_start])[np.newaxis]
            if arr.ndim == 4:
                arr[t : t + 1, y0:y1, tile.x_start : tile.x_stop, :] = block
            else:
                arr[t : t + 1, y0:y1, tile.x_start : tile.x_stop] = block
        # Drop the group reference so its file handles / S3 connections are
        # immediately collectable before the next tile's read.
        del staged
    return fork


def _extend_time_axis(node: zarr.Group, time_date: np.datetime64) -> int:
    """Grow every time-dimmed array by one step and write the new timestamp.

    An append IS a resize plus a write at the new index — doing it explicitly
    replaces the old engine's ``to_icechunk(mode="a", append_dim="time")`` and
    keeps raw zarr the only write path. Returns the new timestep's index.
    """
    nt = cast(zarr.Array, node["time"]).shape[0]
    for name, arr in node.arrays():
        # Zarr v2 metadata has no dimension_names; every engine-written store is v3.
        dims = getattr(arr.metadata, "dimension_names", None)
        if name != "time" and dims and dims[0] == "time":
            arr.resize((nt + 1, *arr.shape[1:]))
    time_arr = cast(zarr.Array, node["time"])
    time_arr.resize((nt + 1,))
    time_arr[nt] = time_date.astype("datetime64[ns]").astype("int64")
    return nt


def _label_to_grid(label: str) -> tuple[int, int]:
    """Parse a staged ``chunk_{row}_{col}`` label into ``(row, col)``."""
    parts = label.split("_")
    if len(parts) != 3 or parts[0] != "chunk":
        raise ValueError(f"Staged label {label!r} is not of the form 'chunk_<row>_<col>'")
    return int(parts[1]), int(parts[2])


@dataclasses.dataclass(frozen=True)
class StagedShardSource:
    """:class:`~tessera_embeddings.storage.shard_writer.ShardSource` over staged tiles.

    The global write path's 1:1 mapping (ADR-008 D3): one staged 2048-px
    inference tile is exactly one output shard, so ``live_shards`` is the staged
    tile grid positions and ``load`` returns each staged variable whole, with a
    leading time axis. Frozen dataclass of plain strings/tuples so the shard
    writer can pickle it to spawned workers.
    """

    staging_base: str
    run_id: str
    shards: tuple[tuple[int, int], ...]
    variables: tuple[str, ...]

    def live_shards(self) -> list[tuple[int, int]]:
        """The staged ``(row, col)`` tile positions — shard indices, 1:1."""
        return list(self.shards)

    def load(self, shard: tuple[int, int]) -> dict[str, np.ndarray]:
        """Read one staged tile whole and return ``{var: (1, h, w[, band]) block}``."""
        sy, sx = shard
        path = f"{self.staging_base}/{self.run_id}/chunk_{sy}_{sx}.zarr"
        group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
        return {var: np.asarray(cast(zarr.Array, group[var])[:])[np.newaxis] for var in self.variables}


class ZarrWriter:
    """Write embeddings to Zarr stores.

    Phase 1: each chunk is written as a standalone raw Zarr at a staging
    location (:meth:`write_chunk`, on GPU actors). Phase 2: staged chunks are
    assembled into the final Icechunk store with raw-zarr fork/merge writes —
    :meth:`assemble` for standalone single-ROI stores, :meth:`assemble_global`
    for a pre-allocated zone group of the global store.
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
        # A run whose FIRST artifact is a skip marker has no staging dir yet on
        # directory-backed filesystems (write_chunk's to_zarr creates it as a
        # side effect; a bare open() does not). No-op on object stores.
        fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
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
        # Sub-chunk staged files at inner-chunk size, full band (ADR-008 D2: the
        # band axis is never split). A staged 2048-px tile is then exactly the
        # 8x8 inner-chunk grid of the shard it becomes, and assembly's banded
        # y-slice reads stay aligned to whole staged pieces.
        staged_chunks = (min(INNER_PX, chunk.height), min(INNER_PX, chunk.width), self.embedding_dim)
        staged_chunks_2d = (min(INNER_PX, chunk.height), min(INNER_PX, chunk.width))

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

    def _create_schema(
        self,
        root: zarr.Group,
        layout: StoreLayout,
        variables: Iterable[str],
        total_y: int,
        total_x: int,
        time_date: np.datetime64,
        spatial: SpatialCoords | None,
    ) -> None:
        """Create the output arrays (schema only, no chunk data) and coordinates.

        Geometry comes from the :class:`StoreLayout` preset — the single source
        of truth shared with the global store's seeder — not from the staged
        files, so the on-disk output is identical regardless of how inference
        tiled the mosaic. Cost is independent of extent (no pixels written).
        """
        sizes = {"time": 1, "northing": total_y, "easting": total_x, "band": self.embedding_dim}
        for var in variables:
            array_layout = layout.for_var(var)
            shape = tuple(sizes[d] for d in array_layout.dims)
            root.create_array(var, **array_layout.create_kwargs(shape))
        _write_coord_arrays(
            root,
            {
                "time": np.asarray([time_date], dtype="datetime64[ns]"),
                "northing": spatial.northing if spatial else np.arange(total_y),
                "easting": spatial.easting if spatial else np.arange(total_x),
                "band": np.arange(self.embedding_dim),
            },
        )

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
        manifest: EmbeddingManifest | None = None,
        n_workers: int,
        get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
        s3_region: str | None = None,
        layout: StoreLayout = LEGACY,
        gate: CommitGate | None = None,
    ) -> str:
        """Assemble staged chunk Zarrs into a standalone Icechunk store.

        Raw-zarr fork/merge engine (no Dask): worker processes write staged-tile
        pixels for disjoint, granularity-aligned northing bands straight into
        the output arrays; the coordinator merges the forks, sets root attrs,
        and commits once via
        :func:`~tessera_embeddings.storage.shard_writer.commit_with_rebase`.

        Create-or-extend semantics on the time axis:

        * fresh store → schema + coords from ``layout`` (``LEGACY`` by default —
          today's single-ROI geometry, D8) in a first metadata-only commit;
        * existing store, new time value → every time-dimmed array is resized by
          one step (explicitly — this replaces ``mode="a"`` appends) in a first
          commit;
        * existing store, time value already present → its index is overwritten
          in place (idempotent resume: a crashed assembly re-run lands on the
          same index instead of appending a duplicate timestep). Live positions
          are rewritten and skip-marked chunks' footprints are reset to fill (a
          prior run's data must not survive under a rerun's skip marker);
          positions **outside the current ROI mask** are not touched — a
          same-date re-assembly assumes the same grid and mask, so shrink the
          ROI only with a fresh date (or a fresh store).

        The fill itself is always the second, single data commit, so a crash
        mid-fill leaves only an all-fill timestep that the re-run overwrites.

        Inference only stages chunks that intersect the ROI mask and had valid
        pixels; everything else is never written, so those positions read back
        as the layout's fill (0 for int8, NaN for floats) exactly as the old
        engine's fill templates produced.

        Args:
            chunks: Full chunk grid (both live and non-live).
            total_y: Total mosaic height.
            total_x: Total mosaic width.
            run_id: Run identifier (locates staged files).
            output_path: Final output Icechunk store path.
            roi_zarr_path: Path to the ROI boolean zarr. Assembly re-enumerates
                live chunks from this to avoid marshaling the list through
                Prefect.
            compute_std: Whether staged chunks contain embedding_std data.
            run_started_at: Flow trigger time for the time coordinate. Falls
                back to now. Ignored when *time_window* is provided.
            mosaic_base: Base path for the input mosaic stores. If provided,
                the reflectance store supplies projected coordinates and CRS.
            log: Optional logger (e.g. Prefect's run logger).
            time_window: If provided, the window end month is the time
                coordinate and window metadata lands in dataset attributes.
            tile_id: Sentinel-2 MGRS tile ID for ``proj:`` convention attrs
                when the mosaic store carries no ``crs`` attr.
            model_version: Model version string for ``tessera:model_version``.
            manifest: Typed manifest for append-safety validation. Written on
                create, validated before extending an existing store.
            n_workers: Worker *process* count. Also divides
                ``TARGET_AGGREGATE_S3_CONCURRENCY`` into the per-fork request
                cap so fleet-wide PUT concurrency stays under S3's ceiling.
            get_credentials: Optional icechunk credential callback for the
                output store (see ``zarr_store._create_storage``).
            s3_region: Optional S3 region override for the output store.
            layout: Output geometry preset. ``LEGACY`` (default) reproduces
                today's single-ROI stores exactly; only new stores consult it.
            gate: Optional commit gate when many assemblies share a process.

        Returns:
            Path to the assembled output store.
        """
        _log = log or logger

        # Determine which chunks have staged zarrs. A chunk that intersects the
        # ROI but was skipped during inference (all pixels failed validity) has
        # a skip marker instead of a zarr — its footprint stays at fill.
        roi_live_chunks = filter_chunks_by_roi_mask(chunks, roi_zarr_path)
        skipped_labels = set(self._list_skip_marker_labels(run_id))
        live_chunks = [c for c in roi_live_chunks if c.label not in skipped_labels]
        if not live_chunks:
            raise IncompleteStageError(
                f"Run {run_id!r} has no staged chunks to assemble under {self.staging_base} "
                f"({len(roi_live_chunks)} ROI chunks, all skipped or unstaged)."
            )
        _log.info(
            "Assembling %d chunks (%d live with staged zarr, %d skipped, %d fill) into %s",
            len(chunks),
            len(live_chunks),
            len(roi_live_chunks) - len(live_chunks),
            len(chunks) - len(roi_live_chunks),
            output_path,
        )

        # Validate that total_y/total_x match the actual chunk grid extent.
        actual_y = max(c.y_stop for c in chunks)
        actual_x = max(c.x_stop for c in chunks)
        if actual_y != total_y or actual_x != total_x:
            msg = f"total_y/total_x ({total_y}, {total_x}) doesn't match chunk grid extent ({actual_y}, {actual_x})"
            raise ValueError(msg)

        started = run_started_at or datetime.datetime.now(datetime.UTC)

        # Projected coordinates and CRS from the input reflectance store.
        spatial: SpatialCoords | None = None
        if mosaic_base:
            spatial = read_spatial_coords(mosaic_base)
            _log.info("Using projected coordinates from %s", mosaic_base)

        if time_window:
            time_date = np.datetime64(time_window.window_end_label, "ns")
        else:
            time_date = np.datetime64(started.date(), "ns")

        staged_obs = self._staged_vars_present(run_id, live_chunks, OBS_COUNT_VARS)
        variables = ["embeddings", "scales"]
        if compute_std:
            variables.append("embedding_std")
        variables += [v for v in OBS_COUNT_VARS if v in staged_obs]

        # Divide the fleet-wide S3 concurrency target across worker forks; see
        # TARGET_AGGREGATE_S3_CONCURRENCY. Forks inherit the repo config through
        # the pickled session, so no save_config round-trip is needed.
        per_worker_cap = max(1, TARGET_AGGREGATE_S3_CONCURRENCY // max(1, n_workers))

        # Manifest split (same tiling as the old engine): each spatial axis at
        # 32 chunks/shard (~16k px at 500-px chunks) and time at 1 shard per
        # timestep, so a one-timestep write rewrites no prior year's manifests.
        with manifest_split({"northing": 32, "easting": 32, "time": 1}):
            repo, is_new = open_or_create_repo(
                output_path,
                max_concurrent_requests=per_worker_cap,
                get_credentials=get_credentials,
                region=s3_region,
                scatter_initial_credentials=get_credentials is not None,
            )

        # --- Phase 1: schema (create) or time-axis placement (extend) --------
        session = repo.writable_session("main")
        root = zarr.open_group(session.store, mode="a")
        overwrite = False
        # "time" absent means a created-but-never-seeded repo (e.g. a crash
        # between repo creation and the schema commit) — treat as fresh.
        if is_new or "time" not in root:
            self._create_schema(root, layout, variables, total_y, total_x, time_date, spatial)
            session.commit(f"Run {run_id}: create schema ({layout.name})")
            time_index = 0
            _log.info("Created %s with layout %s", output_path, layout.name)
        else:
            if manifest:
                manifest.validate_against(extract_manifest(root.attrs), output_path)
            times = read_time_values(root)
            hits = np.flatnonzero(times == time_date)
            if hits.size:
                time_index = int(hits[0])
                overwrite = True
                _log.warning(
                    "Time %s already exists at index %d in %s — overwriting in place "
                    "(live positions rewritten, skip-marked footprints reset to fill)",
                    time_date,
                    time_index,
                    output_path,
                )
            else:
                time_index = _extend_time_axis(root, time_date)
                session.commit(f"Run {run_id}: extend time axis to {time_date}")
                _log.info("Extended %s time axis to index %d (%s)", output_path, time_index, time_date)

        # --- Phase 2: banded parallel fill (fork/merge) -----------------------
        session = repo.writable_session("main")
        granularity = _write_granularity(zarr.open_group(session.store, mode="r"), variables)
        bands = _partition_bands(total_y, granularity, n_workers)
        fork = session.fork()
        # On a same-date overwrite, ROI-live chunks that this run skip-marked
        # must be reset to fill — a prior run may have written real data there.
        clear_chunks = [c for c in roi_live_chunks if c.label in skipped_labels] if overwrite else []
        payloads = []
        for band in bands:
            y0b, y1b = band
            tiles = [(c, self._staging_path(run_id, c)) for c in live_chunks if c.y_start < y1b and c.y_stop > y0b]
            clear = [c for c in clear_chunks if c.y_start < y1b and c.y_stop > y0b]
            if tiles or clear:
                payloads.append(
                    {
                        "fork": fork,
                        "band": band,
                        "time_index": time_index,
                        "variables": tuple(variables),
                        "tiles": tiles,
                        "clear": clear,
                    }
                )
        _log.info(
            "Filling %d band(s) (granularity %d px) across %d process(es)",
            len(payloads),
            granularity,
            min(len(payloads), n_workers),
        )
        if len(payloads) == 1:
            forks = [_fill_band_worker(payloads[0])]
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as ex:
                forks = list(ex.map(_fill_band_worker, payloads))
        session.merge(*forks)

        # --- Phase 3: root attrs + one data commit ----------------------------
        node = zarr.open_group(session.store, mode="a")
        attrs: dict[str, Any] = {
            "run_id": run_id,
            "total_y": total_y,
            "total_x": total_x,
            "embedding_dim": self.embedding_dim,
            "run_started_at": started.isoformat(),
            "run_completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        conv_attrs = build_convention_attrs(
            tile_id=tile_id,
            epsg_code=spatial.crs if spatial else None,
            total_y=total_y,
            total_x=total_x,
            embedding_dim=self.embedding_dim,
            y_coords=spatial.northing if spatial else None,
            x_coords=spatial.easting if spatial else None,
            model_version=model_version,
            n_tiles=len(chunks),
        )
        if conv_attrs:
            attrs.update(conv_attrs)
        if manifest:
            attrs["_manifest"] = manifest.to_dict()
            _log.info("Wrote _manifest to %s", output_path)
        if time_window:
            # Raw writes never clobber attrs, so prior windows are read straight
            # off the store (no pre-write snapshot dance) and merged.
            windows: dict[str, Any] = dict(node.attrs.get("time_windows", {}))  # type: ignore[arg-type]
            windows[time_window.window_end_label] = {
                "range": [
                    f"{time_window.window_start[0]}-{time_window.window_start[1]:02d}",
                    f"{time_window.window_end[0]}-{time_window.window_end[1]:02d}",
                ],
            }
            attrs["time_windows"] = windows
            attrs["time_convention"] = "12mo_window_end"
        node.attrs.update(attrs)

        with gate if gate is not None else nullcontext():
            commit_with_rebase(session, f"Run {run_id}: {len(chunks)} chunks assembled")
        _log.info("Assembly complete: %s", output_path)
        return output_path

    def assemble_global(
        self,
        store_path: str,
        zone: str,
        *,
        year: int,
        run_id: str,
        n_workers: int = 8,
        gate: CommitGate | None = None,
        get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
        s3_region: str | None = None,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    ) -> str:
        """Assemble a run's staged tiles into one (zone, year) of the global store.

        The global write path (ADR-008 D3/D6): every staged tile is exactly one
        output shard, written whole and lean by
        :func:`~tessera_embeddings.storage.shard_writer.write_year_shards` —
        fork/merge across ``n_workers`` processes, one commit per (zone, year)
        behind ``gate``, ``years_complete`` and per-year run provenance updated
        in the same commit. The zone group must already be seeded
        (:func:`~tessera_embeddings.storage.global_store.seed_zone_groups`);
        nothing is ever created or resized here (D1).

        The caller (the zone-fill runner) is responsible for staged-completeness
        verification against the campaign land mask
        (:meth:`verify_staged_completeness`) — this method writes whatever tiles
        are staged. ``embedding_std`` is never staged under v1.1.

        Args:
            store_path: URI of the global Icechunk repo
                (``BucketPaths.global_store()``).
            zone: Zone group name (EPSG code string, e.g. ``"32601"``).
            year: Campaign calendar year to fill — must be on the group's
                pre-allocated time axis.
            run_id: Run identifier (locates staged files).
            n_workers: Worker process count; also divides
                ``TARGET_AGGREGATE_S3_CONCURRENCY`` into the per-fork cap.
            gate: Optional commit gate shared across the zone-year fills this
                process drives (fleet-wide gating is the orchestrator's job).
            get_credentials: Optional icechunk credential callback.
            s3_region: Optional S3 region override.
            log: Optional logger.

        Returns:
            The commit snapshot id.
        """
        _log = log or logger

        labels = self._list_staged_labels(run_id)
        if not labels:
            raise IncompleteStageError(f"Run {run_id!r} has no staged chunks under {self.staging_base}")
        shards = tuple(sorted(_label_to_grid(label) for label in labels))

        per_worker_cap = max(1, TARGET_AGGREGATE_S3_CONCURRENCY // max(1, n_workers))
        repo = open_global_repo(
            store_path,
            get_credentials=get_credentials,
            region=s3_region,
            max_concurrent_requests=per_worker_cap,
        )

        # One readonly probe: year index, shard pitch, variables, run provenance.
        probe = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[zone]
        node = cast(zarr.Group, probe)
        times = read_time_values(node)
        hits = np.flatnonzero(times == np.datetime64(f"{year}-01-01", "ns"))
        if hits.size == 0:
            raise ValueError(
                f"Year {year} is not on {zone}'s pre-allocated time axis "
                f"({np.datetime_as_string(times, unit='D').tolist()}) — the axis is fixed at seeding (ADR-008 D1)."
            )
        year_index = int(hits[0])

        emb = cast(zarr.Array, node["embeddings"])
        shard_px = (emb.shards or emb.chunks)[1]
        staged_px = self.detect_staged_chunk_size(run_id)
        if staged_px != shard_px:
            raise ValueError(
                f"Staged tiles are {staged_px} px but {zone} shards are {shard_px} px — "
                "the global write path requires 1 inference tile == 1 shard (ADR-008 D3)."
            )

        probe_path = f"{self.staging_base}/{run_id}/{labels[0]}.zarr"
        staged_group = zarr.open_group(probe_path, mode="r", storage_options=_staged_storage_options(probe_path))
        missing = [v for v in ("embeddings", "scales") if v not in staged_group]
        if missing:
            raise IncompleteStageError(
                f"Staged tile {probe_path} is missing required variable(s) {missing} — refusing to "
                "mark the year complete over a corrupt or partial staged run."
            )
        variables = tuple(
            v for v in ("embeddings", "scales", "embedding_std", *OBS_COUNT_VARS) if v in staged_group and v in node
        )

        runs: dict[str, Any] = dict(node.attrs.get("runs", {}))  # type: ignore[arg-type]
        runs[str(year)] = {
            "run_id": run_id,
            "assembled_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }

        _log.info(
            "Assembling %d staged tiles into %s/%s year %d (index %d, vars %s, %d workers)",
            len(shards),
            store_path,
            zone,
            year,
            year_index,
            list(variables),
            n_workers,
        )
        source = StagedShardSource(staging_base=self.staging_base, run_id=run_id, shards=shards, variables=variables)
        snapshot = write_year_shards(
            repo,
            zone,
            year_index,
            source,
            n_workers=n_workers,
            gate=gate,
            shard_px=shard_px,
            commit_msg=f"Run {run_id}: fill {zone} year {year}",
            extra_attrs={"runs": runs},
        )
        _log.info("Global assembly complete: %s/%s year %d (snapshot %s)", store_path, zone, year, snapshot)
        return snapshot

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
