"""Embedding output writers.

Staged writes (one Zarr per chunk) use raw uncompressed bytes for zero CPU
overhead on GPU actors; the on-disk staged pieces are inner-chunk-sized
(``INNER_PX`` = 256 px, full band) so a staged tile is exactly the inner-chunk
grid of the output region it becomes. The final Icechunk store's geometry
(chunks, shards, codecs) comes from a :class:`StoreLayout` preset — ``SINGLE``
for single-ROI stores, ``GLOBAL`` for the global campaign's zone groups.

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

import bisect
import dataclasses
import datetime
import itertools
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, cast

import fsspec
import icechunk
import numpy as np
import xarray as xr
import zarr

from tessera_embeddings.config.inference import EMBEDDING_DIM, TimeWindow
from tessera_embeddings.config.store_layout import INNER_PX, OBS_COUNT_VARS, SINGLE, StoreLayout
from tessera_embeddings.inference.chunk_spec import ChunkSpec, chunk_label, filter_chunks_by_roi_mask, parse_chunk_label
from tessera_embeddings.inference.conventions import build_convention_attrs
from tessera_embeddings.storage.empty_store import _write_coord_arrays
from tessera_embeddings.storage.global_store import create_layout_arrays, open_global_repo
from tessera_embeddings.storage.manifest import EmbeddingManifest, extract_manifest
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.shard_writer import (
    CommitGate,
    commit_with_rebase,
    run_forked,
    shard_pitch,
    write_year_shards,
)
from tessera_embeddings.storage.zarr_store import (
    manifest_split,
    open_or_create_repo,
    open_store_as_zarr_group,
    read_time_values,
    time_index_of,
)
from tessera_embeddings.storage.zone_grid import PIXEL_M, year_timestamp

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
is exactly the SlowDown observed at 800 concurrent PUTs. Because
``per_worker_cap`` floors at 1, :func:`_s3_budget_split` also caps the worker
count at the budget so the ``n_workers * per_worker_cap`` product can never
exceed it (``AssemblyConfig`` caps workers far lower in the common case).
"""


def _s3_budget_split(s3_concurrency: int | None, n_workers: int) -> tuple[int, int]:
    """``(effective_workers, per_worker_cap)`` honoring the fleet S3-PUT budget.

    Each fork worker opens its own repo capped at ``per_worker_cap``, so a fill
    issues up to ``effective_workers * per_worker_cap`` concurrent requests. When
    the budget is smaller than the requested worker count, flooring the cap at 1
    would let ``n_workers`` alone exceed it (e.g. budget 5, 8 workers → 8 > 5), so
    the worker count is capped at the budget too — fewer concurrent forks, but the
    fleet target holds. ``s3_concurrency=None`` uses the full aggregate target (a
    lone fill). Both returned values are ``>= 1``, and their product ``<= budget``.
    """
    budget = s3_concurrency if s3_concurrency is not None else TARGET_AGGREGATE_S3_CONCURRENCY
    workers = min(n_workers, max(1, budget))
    return workers, max(1, budget // workers)


def read_store_spatial_coords(
    store_path: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> SpatialCoords:
    """Read projected x/y coordinates and CRS from any mosaic child store.

    Reads via the raw-zarr opener — the 1-D coord arrays and root attrs off
    metadata, never an xarray/dask graph over the store's data chunks.
    ``get_credentials``/``s3_region`` are threaded to the opener so an S3 store
    that authenticates only through the campaign's credential callback (or lives
    outside the default region) is readable.

    Args:
        store_path: Full path to a single mosaic store (reflectance or SAR).
        get_credentials: Optional icechunk credential callback for the store.
        s3_region: Optional S3 region override.

    Returns:
        SpatialCoords with projected northing and easting arrays.
    """
    logger.info("Reading projected coordinates from %s", store_path)
    root = open_store_as_zarr_group(store_path, get_credentials=get_credentials, region=s3_region)
    return SpatialCoords(
        northing=np.asarray(cast(zarr.Array, root["northing"])),
        easting=np.asarray(cast(zarr.Array, root["easting"])),
        crs=cast("str | None", root.attrs.get("crs")),
    )


def read_spatial_coords(
    mosaic_base: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> SpatialCoords:
    """Read projected x/y coordinates and CRS from the reflectance store.

    Thin wrapper over :func:`read_store_spatial_coords` for the reflectance store
    under ``mosaic_base`` — the fill's canonical grid reference (runs once per
    zone-year fill).

    Args:
        mosaic_base: Base path for the mosaic stores (e.g., "s3://bucket/mosaics/roi").
        get_credentials: Optional icechunk credential callback for the mosaic store.
        s3_region: Optional S3 region override.

    Returns:
        SpatialCoords with projected northing and easting arrays.
    """
    return read_store_spatial_coords(
        f"{mosaic_base}/reflectance.zarr", get_credentials=get_credentials, s3_region=s3_region
    )


# =============================================================================
# Raw-zarr assembly engine (module-level so spawn workers can unpickle them)
# =============================================================================


def _partition_bands(
    total_y: int,
    granularity: int,
    n_workers: int,
    weights: Sequence[int] | None = None,
) -> list[tuple[int, int]]:
    """Split ``[0, total_y)`` into at most ``n_workers`` bands aligned to ``granularity``.

    Bands start and end on multiples of the output's northing write granularity
    (shard height when sharded, chunk height otherwise), so no two workers ever
    touch the same output chunk — the fork/merge write-conflict invariant.

    Without ``weights``, whole granularity units are spread as evenly as
    possible. With ``weights`` (work per unit — live tiles overlapping it),
    boundaries balance total work per band instead of raw height: an ROI mask
    clusters live tiles spatially, and equal-height bands would leave most
    workers idle while one drags the assembly. Either way the bands tile
    ``[0, total_y)`` exactly and the last band absorbs the ragged tail.
    """
    n_units = -(-total_y // granularity)
    n_bands = max(1, min(n_workers, n_units))
    if weights is None or sum(weights) == 0:
        base, extra = divmod(n_units, n_bands)
        unit_counts = [base + (1 if i < extra else 0) for i in range(n_bands)]
        cuts = list(itertools.accumulate(unit_counts))
    else:
        prefix = list(itertools.accumulate(weights))
        cuts = []
        start = 0
        for b in range(1, n_bands):
            # First unit index whose cumulative weight reaches the band's share;
            # +1 because prefix[i] includes unit i, and the cut is exclusive.
            cut = bisect.bisect_left(prefix, prefix[-1] * b / n_bands, lo=start) + 1
            # Every band keeps at least one unit so the tiling stays exact.
            cut = min(max(cut, start + 1), n_units - (n_bands - b))
            cuts.append(cut)
            start = cut
        cuts.append(n_units)
    bands: list[tuple[int, int]] = []
    y = 0
    for cut in cuts:
        y_stop = min(total_y, cut * granularity)
        bands.append((y, y_stop))
        y = y_stop
    return bands


def _write_granularity(node: zarr.Group, variables: Iterable[str]) -> int:
    """The output's northing write granularity: shard height if sharded, else chunk height.

    All data variables must agree (both presets do); a disagreement would let
    two bands share an output object, so it is an error rather than a max().
    """
    sizes = {var: shard_pitch(cast(zarr.Array, node[var])) for var in variables}
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
    tiles (every chunk this run does not write, on a same-date overwrite — see
    :meth:`ZarrWriter.assemble`) get the fill value written over their footprint
    so a prior run's data — under a rerun's skip marker OR outside a changed ROI —
    can't survive. Returns the fork for the coordinator to merge.
    """
    fork = payload["fork"]
    t = int(payload["time_index"])
    y0b, y1b = payload["band"]
    root = zarr.open_group(fork.store, mode="a")
    arrays = {var: cast(zarr.Array, root[var]) for var in payload["variables"]}
    # Trailing dims (band) not indexed are written in full, so each assignment
    # below covers both the 3-D and 4-D arrays.
    for tile in payload["clear"]:
        y0, y1 = max(tile.y_start, y0b), min(tile.y_stop, y1b)
        for arr in arrays.values():
            # Scalar assignment: zarr broadcasts per-chunk without materializing
            # the selection (a full-band float32 block here would be ~2 GB).
            arr[t : t + 1, y0:y1, tile.x_start : tile.x_stop] = arr.fill_value
    for tile, path in payload["tiles"]:
        y0, y1 = max(tile.y_start, y0b), min(tile.y_stop, y1b)
        staged = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
        for var, arr in arrays.items():
            staged_arr = cast(zarr.Array, staged[var])
            # Validate dtype before assigning: zarr would silently CAST a float
            # staged tile into an int8 destination (corrupting it) while the
            # assembly still commits successfully. Every tile is checked, not
            # just the probe (a mixed/corrupt run can differ per tile).
            if staged_arr.dtype != arr.dtype:
                raise ValueError(
                    f"Staged tile {path} variable {var!r} has dtype {staged_arr.dtype} but the destination "
                    f"array is {arr.dtype} — a silent cast would corrupt the output."
                )
            # Validate the full shape too: a singleton spatial or band dim would
            # broadcast over the destination region and commit repeated values.
            # 4-D destination (embeddings/std) → staged tile is (h, w, band);
            # 3-D (scales/obs) → (h, w).
            spatial_hw = (tile.y_stop - tile.y_start, tile.x_stop - tile.x_start)
            expected_shape = (*spatial_hw, arr.shape[3]) if arr.ndim == 4 else spatial_hw
            if staged_arr.shape != expected_shape:
                raise ValueError(
                    f"Staged tile {path} variable {var!r} has shape {staged_arr.shape}, expected "
                    f"{expected_shape} — a singleton/off-grid dim would broadcast and corrupt the output."
                )
            block = np.asarray(staged_arr[y0 - tile.y_start : y1 - tile.y_start])[np.newaxis]
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


@dataclasses.dataclass(frozen=True)
class StagedShardSource:
    """:class:`~tessera_embeddings.storage.shard_writer.ShardSource` over staged tiles.

    The global write path's 1:1 mapping (ADR-008 D3): one staged 2048-px
    inference tile is exactly one output shard, so ``live_shards`` is the staged
    tile grid positions and ``load`` returns each staged variable whole, with a
    leading time axis. Every tile is validated as it is read — every expected
    variable present, each exactly one shard tall/wide, and matching the
    probe's dtype — so a truncated, mixed-version, or corrupt staged run fails
    loudly (naming the tile and variable) instead of silently casting or
    leaving fill inside a shard of a year that then gets tagged complete. Frozen
    dataclass of plain strings/tuples so the shard writer can pickle it to
    spawned workers.
    """

    staging_base: str
    run_id: str
    shards: tuple[tuple[int, int], ...]
    variables: tuple[str, ...]
    shard_px: int
    dtypes: tuple[tuple[str, str], ...] = ()  # (var, dtype) expectations from the probe
    embedding_dim: int = 0  # band width for 4-D vars; 0 = skip the band check
    # Optional vars present in the DESTINATION. A tile carrying one that is NOT in
    # `variables` (the assembled set) is a heterogeneous stage whose var would be
    # silently dropped — checked per tile in load(), so EVERY tile is validated.
    optional_present: tuple[str, ...] = ()

    def live_shards(self) -> list[tuple[int, int]]:
        """The staged ``(row, col)`` tile positions — shard indices, 1:1."""
        return list(self.shards)

    def load(self, shard: tuple[int, int]) -> dict[str, np.ndarray]:
        """Read one staged tile whole and return ``{var: (1, h, w[, band]) block}``."""
        sy, sx = shard
        path = f"{self.staging_base}/{self.run_id}/{chunk_label(sy, sx)}.zarr"
        group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
        # Require the completion marker at ASSEMBLY read too, not only on resume: the
        # listing paths (verify_staged_completeness / _list_staged_labels) count a
        # crash-partial .zarr as a valid staged tile, and to_zarr can write array
        # metadata before all chunk objects. Without this, an assembly-only run (or a
        # partial that slipped past inference) would read missing chunks as fill and
        # commit+tag silent holes. The open already happened, so this is free.
        if not group.attrs.get("staged_complete"):
            raise IncompleteStageError(
                f"Staged tile {path} lacks the staged_complete marker — a crashed write_chunk left partial "
                "chunks (read as fill). Assembling it would publish silent holes; re-infer the tile first."
            )
        expected_dtypes = dict(self.dtypes)
        blocks: dict[str, np.ndarray] = {}
        for var in self.variables:
            if var not in group:
                raise ValueError(
                    f"Staged tile {path} is missing variable {var!r} (present in the probed tile) — "
                    "heterogeneous staged run; re-stage or fix before assembling."
                )
            block = np.asarray(cast(zarr.Array, group[var])[:])[np.newaxis]
            want = expected_dtypes.get(var)
            if want is not None and str(block.dtype) != want:
                raise ValueError(
                    f"Staged tile {path} has {var} dtype {block.dtype}, expected {want} — a raw-zarr "
                    "write would silently C-cast (wraparound); mixed-version or corrupt staged run."
                )
            h, w = block.shape[1:3]
            if (h, w) != (self.shard_px, self.shard_px):
                raise ValueError(
                    f"Staged tile {path} has {var} extent {h} x {w} px but shards are {self.shard_px} px — "
                    "truncated or off-grid tile (ADR-008 D3 requires whole tiles)."
                )
            # Also validate the trailing band axis of 4-D vars (embeddings /
            # embedding_std): a (1, px, px, 1) block would broadcast across all
            # destination bands and commit repeated values under a shape check
            # that only looked at [1:3].
            if self.embedding_dim and block.ndim == 4 and block.shape[3] != self.embedding_dim:
                raise ValueError(
                    f"Staged tile {path} has {var} band width {block.shape[3]} but expected "
                    f"{self.embedding_dim} — a singleton band would broadcast over the output."
                )
            blocks[var] = block
        # Detect a heterogeneous stage on EVERY tile (not just a first/last sample):
        # an optional var present in the destination but absent from the assembled
        # `variables` (because the probe tile lacked it) would be silently dropped
        # from this — and every — shard write, then the year tagged complete.
        extras = [v for v in self.optional_present if v not in self.variables and v in group]
        if extras:
            raise ValueError(
                f"Staged tile {path} carries optional var(s) {extras} absent from the assembled set "
                f"{list(self.variables)} — heterogeneous staged run; re-stage with one config before assembling."
            )
        return blocks


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
        # Completion marker, written LAST in a SEPARATE op after to_zarr returns:
        # to_zarr can create every array's metadata before all its chunk objects, so
        # a crash mid-write leaves a tile with correct vars/shape/dtype but MISSING
        # chunks (read back as fill). The resume scan requires this marker, so such a
        # partial tile is rejected + re-inferred instead of skipped and silently
        # assembled with holes. A crash before this line leaves no marker.
        marker = zarr.open_group(path, mode="a", storage_options=_staged_storage_options(path))
        marker.attrs["staged_complete"] = True
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
    ) -> set[str]:
        """Verify all expected chunks have either a staged zarr or a skip marker.

        Returns the labels that have staged zarrs (an empty set means every
        live chunk resolved to a skip marker), so callers don't re-LIST the
        staging prefix to learn what verification already read.

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
        staged_list, skipped_list = self._list_run_labels(run_id)
        staged, skipped = set(staged_list), set(skipped_list)
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
        return staged

    def _list_run_names(self, run_id: str) -> list[str]:
        """One listing of the run's staging directory, as bare entry names.

        The single owner of the staging LIST (a full prefix scan on S3 — at
        zone scale ~15k objects, so callers derive everything from one pass).
        """
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
        # Sorted: fs.ls order is backend-dependent (S3 lexicographic, local
        # filesystems arbitrary), and downstream probes take labels[0] — the
        # probe tile must not depend on which OS listed the directory.
        return sorted(entry.rstrip("/").rsplit("/", 1)[-1] for entry in entries)

    def _list_run_labels(self, run_id: str) -> tuple[list[str], list[str]]:
        """``(staged zarr labels, skip-marker labels)`` from one staging LIST."""
        names = self._list_run_names(run_id)
        staged = [n.removesuffix(".zarr") for n in names if n.endswith(".zarr")]
        skipped = [n.removesuffix(".skipped") for n in names if n.endswith(".skipped")]
        return staged, skipped

    def _list_staged_labels(self, run_id: str) -> list[str]:
        """List chunk labels that have staged Zarrs in the run directory.

        Returns:
            List of chunk label strings (e.g. ``["chunk_0_0", "chunk_0_1"]``).
            Empty list if the staging directory doesn't exist.
        """
        return self._list_run_labels(run_id)[0]

    def _list_skip_marker_labels(self, run_id: str) -> list[str]:
        """List chunk labels that have skip markers in the run directory."""
        return self._list_run_labels(run_id)[1]

    def _validate_staged_chunk(
        self,
        run_id: str,
        chunk: ChunkSpec,
        required_vars: list[str],
    ) -> tuple[str, str] | None:
        """Validate a single staged chunk Zarr.

        Returns ``None`` if the tile is complete and valid, else ``(kind, reason)``:

        - ``("incomplete", ...)`` — a crash artifact: genuinely absent
          (``FileNotFoundError``), or missing the ``staged_complete`` marker (to_zarr
          can create array metadata before all chunk objects, so a crash leaves gaps
          read as fill). The caller RE-INFERS it (``write_chunk``'s ``mode="w"``
          overwrites) rather than raising — with the stable, input-fingerprinted
          run_id a raise would wedge every retry on the same partial artifact.
        - ``("invalid", ...)`` — a COMPLETE tile (marker present) that is
          structurally wrong (missing var / shape / dtype). The caller raises: this
          is a real anomaly (e.g. a stale wrong-grid store), not a resumable crash.

        Any OTHER open error (auth, throttling, transient network, corrupt metadata)
        propagates — a valid completed tile must not be silently re-inferred because
        of a transient read failure.
        """
        path = self._staging_path(run_id, chunk)
        try:
            group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
        except FileNotFoundError:
            # The artifact is genuinely gone (partial/removed) → self-heal by re-inference.
            # Other open errors (auth, throttling, transient network, corrupt metadata)
            # must NOT be treated as "partial" — silently excluding + re-inferring a valid
            # completed tile on a transient read failure is expensive and wrong; propagate.
            return ("incomplete", "staged zarr not found (partial or removed) — re-infer")
        # The completion marker (written last by write_chunk) is the authority on
        # whether every chunk landed. Absent → a crashed write → re-infer.
        if not group.attrs.get("staged_complete"):
            return ("incomplete", "no staged_complete marker — a crash mid-write_chunk left partial chunks")
        for var in required_vars:
            if var not in group:
                return ("invalid", f"missing variable '{var}'")
            arr: zarr.Array = group[var]  # type: ignore[assignment]
            # scales is a 2-D per-pixel factor (H, W); embeddings/embedding_std carry
            # the band dim (H, W, band). Validating scales against the 3-D shape
            # would false-reject every valid tile.
            expected_shape = (
                (chunk.height, chunk.width) if var == "scales" else (chunk.height, chunk.width, self.embedding_dim)
            )
            if arr.shape != expected_shape:
                return ("invalid", f"'{var}' shape {arr.shape} != expected {expected_shape}")
            expected_dtype = np.int8 if var == "embeddings" else np.float32
            if arr.dtype != expected_dtype:
                return ("invalid", f"'{var}' dtype {arr.dtype} != expected {expected_dtype}")
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
        staged_labels, skip_list = self._list_run_labels(run_id)
        skip_marker_labels = set(skip_list)
        if not staged_labels and not skip_marker_labels:
            _log.info("No staged chunks or skip markers found for run %s — starting fresh", run_id)
            return set()

        _log.info(
            "Found %d staged chunk Zarrs and %d skip markers — validating",
            len(staged_labels),
            len(skip_marker_labels),
        )

        chunk_by_label = {c.label: c for c in chunks}
        # `scales` is mandatory (dequantization needs it) and staged Zarr writes are
        # NOT atomic across arrays, so a crash mid-write_chunk can leave an
        # embeddings-only tile. Requiring scales here rejects that partial artifact
        # up front — otherwise the resume scan counts it valid, run_inference skips
        # it, and the fill only fails (permanently, on every retry) at assembly.
        required_vars = ["embeddings", "scales"] + (["embedding_std"] if compute_std else [])
        valid: set[str] = set()
        invalid_paths: list[tuple[str, str]] = []

        for label in staged_labels:
            chunk = chunk_by_label.get(label)
            if chunk is None:
                invalid_paths.append((f"{label}.zarr", "no matching ChunkSpec (stale chunk from a different grid?)"))
                continue
            result = self._validate_staged_chunk(run_id, chunk, required_vars)
            if result is None:
                valid.add(label)
            elif result[0] == "incomplete":
                # A crash artifact (no completion marker / unopenable): EXCLUDE it so
                # run_inference regenerates it (write_chunk's mode="w" overwrites the
                # partial). Do NOT raise — with the stable, input-fingerprinted run_id
                # a raise would re-fire on the same artifact every retry, wedging the
                # cell until manual deletion.
                _log.warning("Staged tile %s is incomplete (%s) — will re-infer (overwrite)", label, result[1])
            else:  # "invalid" — a COMPLETE tile that is structurally wrong: needs attention.
                invalid_paths.append((f"{label}.zarr", result[1]))

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
        """Which of *var_names* exist in the staged Zarrs — checked across EVERY tile.

        A sampled probe (even first+last) can miss a tile that alone carries an
        obs-count variable, silently dropping it from the whole output. So every
        staged chunk is opened and required to agree on the optional-variable set;
        any disagreement is a hard stop. Callers must pass only chunks that have
        staged files. (One metadata open per tile; the band fill re-opens them for
        data — acceptable for the single-ROI path.)
        """
        present: set[str] | None = None
        first_label: str | None = None
        for chunk in chunks:
            path = self._staging_path(run_id, chunk)
            try:
                group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
            except FileNotFoundError as exc:  # GroupNotFoundError subclasses this
                # A silent empty set here would quietly drop every obs-count
                # variable from the output; a chunk that should exist but can't be
                # opened is a corrupt/partial stage and must be loud.
                raise IncompleteStageError(f"Cannot open staged chunk {path}: {exc}") from exc
            found = {v for v in var_names if v in group}
            if present is None:
                present, first_label = found, chunk.label
            elif found != present:
                raise IncompleteStageError(
                    f"Heterogeneous staged run {run_id}: chunk {first_label} has optional vars {sorted(present)} "
                    f"but chunk {chunk.label} has {sorted(found)} — re-stage with a single inference config."
                )
        return present or set()

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
        create_layout_arrays(root, layout, variables, sizes)
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
        layout: StoreLayout = SINGLE,
    ) -> str:
        """Assemble staged chunk Zarrs into a standalone Icechunk store.

        Raw-zarr fork/merge engine (no Dask): worker processes write staged-tile
        pixels for disjoint, granularity-aligned northing bands straight into
        the output arrays; the coordinator merges the forks, sets root attrs,
        and commits once via
        :func:`~tessera_embeddings.storage.shard_writer.commit_with_rebase`.

        Create-or-extend semantics on the time axis:

        * fresh store → schema + coords from ``layout`` (``SINGLE`` by default —
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
        mid-fill leaves only an all-fill timestep that the re-run overwrites —
        provided the re-run lands on the same time value. With the default
        date-derived coordinate (no ``time_window``), retry the same day or
        pass the original ``run_started_at`` explicitly; otherwise the crashed
        date's all-fill timestep persists as a ghost.

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
            model_version: Encoder checkpoint identifier, recorded as the
                ``checkpoint_id`` provenance attr (``geoemb:model`` is the public
                encoder URL, derived separately).
            manifest: Typed manifest for append-safety validation. Written on
                create, validated before extending an existing store.
            n_workers: Worker *process* count. Also divides
                ``TARGET_AGGREGATE_S3_CONCURRENCY`` into the per-fork request
                cap so fleet-wide PUT concurrency stays under S3's ceiling.
            get_credentials: Optional icechunk credential callback for the
                output store (see ``zarr_store._create_storage``).
            s3_region: Optional S3 region override for the output store.
            layout: Output geometry preset. ``SINGLE`` (default) reproduces
                today's single-ROI stores exactly; only new stores consult it.

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
            # Every ROI-intersecting chunk was skipped (no valid pixels). Publish
            # an all-fill timestep anyway rather than aborting: a create/append
            # writes fill, and a same-date overwrite clears the prior data (its
            # skip-marked footprints are reset in Phase 2). This matches the old
            # engine, which could still publish an all-fill output / clear a retry.
            _log.info(
                "Run %r: all %d ROI chunk(s) skipped — publishing an all-fill timestep to %s",
                run_id,
                len(roi_live_chunks),
                output_path,
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
            spatial = read_spatial_coords(mosaic_base, get_credentials=get_credentials, s3_region=s3_region)
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
            # Persist the split on create so a later COLD writer (one that opens
            # the store outside this manifest_split block) keeps splitting rather
            # than reverting to O(store) manifest rewrites. Mirrors
            # create_global_repo; forks in THIS session already inherit it via
            # the pickled session, so this only matters for future opens.
            if is_new:
                repo.save_config()

        # Publish atomically: run the whole schema/extend + data lifecycle on a
        # private work branch and fast-forward `main` only once the timestep's
        # data has landed. Committing Phase 1 to `main` directly would advertise
        # a resized array + new time coordinate (or, for a fresh store, an empty
        # schema) BEFORE any worker writes — so a crash in Phase 2/3 would leave
        # `main` serving an all-fill timestep. On the work branch `main` never
        # observes the half-written state; a failed run leaves it untouched. One
        # writer per single-ROI store (the campaign uses assemble_global), so a
        # fixed branch name is safe; a stale ref from a crashed prior attempt is
        # reset here rather than left to accumulate.
        base_snapshot = repo.lookup_branch("main")
        work_branch = "_assemble-wip"
        if work_branch in repo.list_branches():
            repo.delete_branch(work_branch)
        repo.create_branch(work_branch, base_snapshot)

        # --- Phase 1: schema (create) or time-axis placement (extend) --------
        session = repo.writable_session(work_branch)
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
            # The store's own grid is authoritative: raw region writes would
            # silently land in a corner (or be clamp-truncated) on a mismatched
            # extent — the loud check xarray's append used to provide.
            emb = cast(zarr.Array, root["embeddings"])
            if emb.shape[1] != total_y or emb.shape[2] != total_x or emb.shape[3] != self.embedding_dim:
                raise ValueError(
                    f"Mosaic extent ({total_y} x {total_x} x {self.embedding_dim} bands) does not match "
                    f"existing store {output_path} ({emb.shape[1]} x {emb.shape[2]} x {emb.shape[3]} bands) "
                    "— wrong output path, ROI grid, or model width."
                )
            # Shape alone can't catch a shifted/reversed mosaic on the same grid
            # size (the manifest omits origin), which would be appended under the
            # existing coordinates and silently misgeoreferenced. Compare CRS +
            # coordinate endpoints against the stored grid (half-pixel absolute
            # tolerance, as in the zone-fill runner) when mosaic coords are known.
            #
            # CRS is checked only when present. Atomic publish (the work-branch
            # fast-forward below) means `main` only ever exposes a COMPLETE store
            # — CRS included — so a normal re-append always sees it; the None
            # guard is kept defensive, covering a legacy store left partial by a
            # pre-atomic-publish engine. The coordinate ARRAYS are the real check
            # here: comparing endpoints catches a shifted/reversed retry mosaic
            # that would otherwise write positionally over the existing grid.
            if spatial is not None:
                stored_crs = root.attrs.get("crs")
                if stored_crs is not None and spatial.crs != stored_crs:
                    raise ValueError(
                        f"Mosaic CRS {spatial.crs!r} does not match existing store {output_path} "
                        f"CRS {stored_crs!r} — appending would misgeoreference the timestep."
                    )
                z_north = cast(zarr.Array, root["northing"])
                z_east = cast(zarr.Array, root["easting"])
                atol = PIXEL_M / 2
                if not (
                    np.isclose(spatial.northing[0], z_north[0], rtol=0.0, atol=atol)
                    and np.isclose(spatial.northing[-1], z_north[-1], rtol=0.0, atol=atol)
                    and np.isclose(spatial.easting[0], z_east[0], rtol=0.0, atol=atol)
                    and np.isclose(spatial.easting[-1], z_east[-1], rtol=0.0, atol=atol)
                ):
                    raise ValueError(
                        f"Mosaic coordinates do not lie on existing store {output_path}'s grid "
                        "(shifted or reversed axes) — appending would silently misgeoreference the timestep."
                    )
            # Variables staged by this run but absent from the store (e.g. a
            # store created before obs counts, or compute_std newly on) are
            # created schema-only at the current time extent; prior timesteps
            # read back as the layout fill. The old engine silently wrote such
            # variables MISALIGNED with the time axis — creating them properly
            # is the loud-and-correct replacement.
            missing = [v for v in variables if v not in root]
            if missing:
                nt = cast(zarr.Array, root["time"]).shape[0]
                sizes = {"time": nt, "northing": total_y, "easting": total_x, "band": self.embedding_dim}
                create_layout_arrays(root, layout, missing, sizes)
                _log.info("Created missing variable(s) %s in %s from layout %s", missing, output_path, layout.name)
            time_index_found = time_index_of(root, time_date)
            if time_index_found is not None:
                time_index = time_index_found
                overwrite = True
                # A time-dependent array the store carries but THIS run does not
                # write (e.g. a store seeded with embedding_std, now filled with
                # std off; or the other S1 orbit's obs count on a single-orbit
                # fill) would otherwise keep its PRIOR values at this timestep
                # while embeddings/scales are overwritten — stale metadata
                # describing data that no longer exists. Reset those slices to
                # fill so the overwritten timestep is internally consistent.
                untouched = [
                    name
                    for name, arr in root.arrays()
                    if name != "time"
                    and name not in variables
                    and (getattr(arr.metadata, "dimension_names", None) or ("",))[0] == "time"
                ]
                for name in untouched:
                    arr = cast(zarr.Array, root[name])
                    arr[time_index] = arr.fill_value if arr.fill_value is not None else 0
                # Newly created vars and the untouched-reset must be committed
                # before the Phase 2 fork, which opens a fresh session here.
                if missing or untouched:
                    parts = ([f"add {missing}"] if missing else []) + (
                        [f"reset untouched {untouched}"] if untouched else []
                    )
                    session.commit(f"Run {run_id}: overwrite {time_date} — {'; '.join(parts)}")
                _log.warning(
                    "Time %s already exists at index %d in %s — overwriting in place "
                    "(live positions rewritten, skip-marked footprints reset to fill%s)",
                    time_date,
                    time_index,
                    output_path,
                    f"; untouched vars reset to fill: {untouched}" if untouched else "",
                )
            else:
                time_index = _extend_time_axis(root, time_date)
                session.commit(
                    f"Run {run_id}: extend time axis to {time_date}"
                    + (f" (adding variables {missing})" if missing else "")
                )
                _log.info("Extended %s time axis to index %d (%s)", output_path, time_index, time_date)

        # --- Phase 2: banded parallel fill (fork/merge) -----------------------
        session = repo.writable_session(work_branch)
        granularity = _write_granularity(zarr.open_group(session.store, mode="r"), variables)
        # Weight bands by live tiles so clustered ROIs don't starve most workers.
        unit_weights = [0] * (-(-total_y // granularity))
        for c in live_chunks:
            for unit in range(c.y_start // granularity, -(-c.y_stop // granularity)):
                unit_weights[unit] += 1
        bands = _partition_bands(total_y, granularity, n_workers, weights=unit_weights)
        # On a same-date overwrite, clear EVERY destination chunk this run does not
        # write, not just the ROI-live chunks it skip-marked: a prior run under a
        # LARGER or shifted ROI may have written real data into chunks that fall
        # outside THIS run's ROI, and those would otherwise survive as stale
        # published embeddings. Clearing the full non-live footprint (scalar fill,
        # elided for already-empty ocean chunks) makes the overwritten timestep
        # exactly this run's ROI regardless of how the ROI changed between runs.
        clear_chunks = [c for c in chunks if c.label not in {lc.label for lc in live_chunks}] if overwrite else []
        payloads: list[dict[str, Any]] = []
        for band in bands:
            y0b, y1b = band
            tiles = [(c, self._staging_path(run_id, c)) for c in live_chunks if c.y_start < y1b and c.y_stop > y0b]
            clear = [c for c in clear_chunks if c.y_start < y1b and c.y_stop > y0b]
            if tiles or clear:
                payloads.append(
                    {
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
        # No payloads = an all-skipped run with nothing to write or clear (a fresh
        # or appended all-fill timestep); the schema/timestep from Phase 1 already
        # stands. run_forked would spawn a zero-worker pool, so skip it.
        if payloads:
            run_forked(session, _fill_band_worker, payloads)

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
        # Drop any retired tessera:* attrs before writing geoemb: appending to a
        # store created before the geoemb switch would otherwise leave the old
        # keys behind (update only overwrites the keys it carries), so the store
        # would advertise both conventions. zarr_conventions is replaced wholesale.
        for stale in [k for k in node.attrs if str(k).startswith("tessera:")]:
            del node.attrs[stale]
        node.attrs.update(attrs)

        commit_with_rebase(session, f"Run {run_id}: {len(chunks)} chunks assembled")
        # Atomic publish: fast-forward `main` to the fully-written tip. The guard
        # fails loudly if another writer advanced `main` since we branched (two
        # processes assembling the same ROI store — unsupported). Then drop the
        # work ref; `main` now retains the snapshot.
        repo.reset_branch("main", repo.lookup_branch(work_branch), from_snapshot_id=base_snapshot)
        repo.delete_branch(work_branch)
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
        staged_labels: Iterable[str] | None = None,
        s3_concurrency: int | None = None,
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
        (:meth:`verify_staged_completeness`) — pass its returned label set as
        ``staged_labels`` so the verified inventory is, by construction, the
        assembled inventory (and the staging prefix isn't re-LISTed); when
        omitted, the staging prefix is listed here. The variable set is probed
        from one tile (staged runs are homogeneous — one code version stages
        one variable set); per-tile shape/variable validation happens as tiles
        are read. ``embedding_std`` is never staged under v1.1.

        Refill caveat: only staged tiles are written. A deliberate refill of a
        landed year (an exceptional, manual operation — the zone-year tag is
        write-once) whose new run skips or drops tiles the previous run staged
        leaves the prior data in those shards; a refill must re-stage every
        previously-live tile, or the operator must clear the year first.

        Args:
            store_path: URI of the global Icechunk repo
                (``BucketPaths.global_store()``).
            zone: Zone group name — UTM common name, e.g. ``"01N"``/``"60S"``.
            year: Campaign calendar year to fill — must be on the group's
                pre-allocated time axis.
            run_id: Run identifier (locates staged files).
            n_workers: Worker process count; also divides
                ``TARGET_AGGREGATE_S3_CONCURRENCY`` into the per-fork cap.
            gate: Optional commit gate shared across the zone-year fills this
                process drives (fleet-wide gating is the orchestrator's job).
            staged_labels: Pre-listed staged tile labels (e.g. the return of
                :meth:`verify_staged_completeness`); ``None`` lists the prefix.
            s3_concurrency: This fill's slice of the fleet S3-PUT budget (divided
                across ``n_workers`` for the per-fork cap); ``None`` uses the full
                ``TARGET_AGGREGATE_S3_CONCURRENCY`` (a lone fill).
            get_credentials: Optional icechunk credential callback.
            s3_region: Optional S3 region override.
            log: Optional logger.

        Returns:
            The commit snapshot id.
        """
        _log = log or logger

        labels = sorted(staged_labels) if staged_labels is not None else self._list_staged_labels(run_id)
        if not labels:
            raise IncompleteStageError(f"Run {run_id!r} has no staged chunks under {self.staging_base}")
        shards = tuple(sorted(parse_chunk_label(label) for label in labels))

        # S3 budget: divide the FLEET target across concurrent fills, not just this
        # fill's forks. TARGET_AGGREGATE_S3_CONCURRENCY // n_workers alone bounds one
        # fill to ~target, so K concurrent fills burst K times the target PUTs (the
        # 800-req SlowDown). The campaign passes `s3_concurrency = target //
        # max_parallel_zones` so the fleet stays near target; None = full target.
        n_workers, per_worker_cap = _s3_budget_split(s3_concurrency, n_workers)
        repo = open_global_repo(
            store_path,
            get_credentials=get_credentials,
            region=s3_region,
            max_concurrent_requests=per_worker_cap,
        )

        # One readonly probe of the zone group: year index, shard pitch, variables.
        node = cast(zarr.Group, zarr.open_group(repo.readonly_session(branch="main").store, mode="r")[zone])
        year_index = time_index_of(node, year_timestamp(year))
        if year_index is None:
            raise ValueError(
                f"Year {year} is not on {zone}'s pre-allocated time axis "
                f"({np.datetime_as_string(read_time_values(node), unit='D').tolist()}) — "
                "the axis is fixed at seeding (ADR-008 D1)."
            )
        shard_px = shard_pitch(cast(zarr.Array, node["embeddings"]))
        # Destination band width is the grid authority for the staged-tile band
        # check — NOT ZarrWriter.embedding_dim, which defaults to 128 and needn't
        # match a zone seeded at a different band.
        dest_band = int(cast(zarr.Array, node["embeddings"]).shape[-1])

        # One probe of a staged tile: required vars, dtypes, variable set, and
        # tile pitch (exact — a truncated half-tile must not pass; every other
        # tile is re-validated by StagedShardSource.load as it is read).
        probe_path = f"{self.staging_base}/{run_id}/{labels[0]}.zarr"
        staged_group = zarr.open_group(probe_path, mode="r", storage_options=_staged_storage_options(probe_path))
        missing = [v for v in ("embeddings", "scales") if v not in staged_group]
        if missing:
            raise IncompleteStageError(
                f"Staged tile {probe_path} is missing required variable(s) {missing} — refusing to "
                "mark the year complete over a corrupt or partial staged run."
            )
        for var, want in (("embeddings", np.int8), ("scales", np.float32)):
            got = cast(zarr.Array, staged_group[var]).dtype
            if got != want:
                raise IncompleteStageError(
                    f"Staged tile {probe_path} has {var} dtype {got}, expected {np.dtype(want)} — "
                    "a raw-zarr write would silently C-cast (wraparound); re-stage correctly."
                )
        staged_shape = tuple(cast(zarr.Array, staged_group["embeddings"]).shape[:2])
        if staged_shape != (shard_px, shard_px):
            raise ValueError(
                f"Staged tiles are {staged_shape[0]} x {staged_shape[1]} px but {zone} shards are "
                f"{shard_px} px — the global write path requires 1 inference tile == 1 shard (ADR-008 D3)."
            )
        missing_dst = [v for v in ("embeddings", "scales") if v not in node]
        if missing_dst:
            raise ValueError(
                f"Zone group {zone} lacks required array(s) {missing_dst} — a fill without them "
                "could not be dequantized; the group must be seeded with a full GLOBAL layout."
            )
        variables = tuple(
            v for v in ("embeddings", "scales", "embedding_std", *OBS_COUNT_VARS) if v in staged_group and v in node
        )
        # Optional vars the destination CAN hold. StagedShardSource.load checks
        # every tile for one of these that's absent from `variables` (i.e. present
        # only in some tiles, so silently dropped) — full-coverage homogeneity, not
        # a first/last sample.
        dest_optional = tuple(v for v in ("embedding_std", *OBS_COUNT_VARS) if v in node)

        # Validate each staged variable's dtype against its DESTINATION array, not
        # just embeddings/scales (asserted int8/float32 above): a raw-zarr shard
        # write does a silent C-cast, so a uniformly int64/uint32 observation
        # count — which agrees with the probe, so StagedShardSource's tile-vs-probe
        # check passes — would be narrowed into a seeded uint16 without this guard.
        for v in variables:
            if v in ("embeddings", "scales"):
                continue
            staged_dt = cast(zarr.Array, staged_group[v]).dtype
            dest_dt = cast(zarr.Array, node[v]).dtype
            if staged_dt != dest_dt:
                raise IncompleteStageError(
                    f"Staged tile {probe_path} has {v} dtype {staged_dt} but {zone}/{v} is seeded "
                    f"{dest_dt} — a raw-zarr write would silently C-cast; re-stage at the seeded dtype."
                )

        probe_dtypes = tuple((v, str(cast(zarr.Array, staged_group[v]).dtype)) for v in variables)

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
        source = StagedShardSource(
            staging_base=self.staging_base,
            run_id=run_id,
            shards=shards,
            variables=variables,
            shard_px=shard_px,
            dtypes=probe_dtypes,
            embedding_dim=dest_band,
            optional_present=dest_optional,
        )
        snapshot = write_year_shards(
            repo,
            zone,
            year_index,
            source,
            n_workers=n_workers,
            gate=gate,
            shard_px=shard_px,
            commit_msg=f"Run {run_id}: fill {zone} year {year}",
            run_id=run_id,
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
        # Shared prefix delete: s5cmd --all-versions (so a versioned bucket doesn't
        # keep the staged tiles as non-current versions), fsspec fallback.
        delete_prefix(target, log=_log)
