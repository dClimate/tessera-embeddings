"""Embedding output writers.

Staged writes (one Zarr per chunk) use raw uncompressed bytes for zero CPU overhead on GPU
actors, in inner-chunk-sized pieces (``INNER_PX`` = 256 px, full band), so a staged tile is
exactly the inner-chunk grid of the output region it becomes. The final Icechunk store's geometry
(chunks, shards, codecs) comes from a :class:`StoreLayout` preset — ``SINGLE`` for single-ROI
stores, ``GLOBAL`` for the global campaign's zone groups.

Assembly is raw-zarr, not Dask: the coordinator forks an icechunk session, worker processes write
staged-tile pixels straight into the output arrays by plain zarr assignment, and the coordinator
merges the forks and commits once (the cooperative fork/merge model shared with
:mod:`tessera_embeddings.storage.shard_writer`). No task graph is built over the store, so cost
scales with the *live* pixels, not the grid.

One inference tile is one 2048-px output shard on both paths (ADR-008 D3), so nothing rechunks.
Two write strategies:

* **Single-ROI** (``assemble``): workers partition the mosaic into *northing bands aligned to the
  output write granularity*, so no two forks touch the same output object and each tile is read
  by exactly one band. A mosaic whose extent is not a whole number of shards has ragged edge
  tiles; their partial chunks are read-modify-written sequentially inside one fork (icechunk
  sessions are read-your-writes).
* **Global** (``assemble_global``): the zone grid is seeded to whole shards, so there are no
  ragged edges and no banding — whole tiles round-robin across workers via
  :func:`~tessera_embeddings.storage.shard_writer.write_year_shards`, and every shard object is
  emitted once, lean, with ocean inner chunks elided.

Both paths emit one machine-readable ``ASSEMBLY_SUMMARY`` log record per assembly — per-worker
read/write phase timings with a CPU/wall split, worker counts, byte and object counts — so a slow
assembly can be attributed to staged reads, compression or the object store without re-running
it. See :func:`_assembly_summary_line` for the fields and their exact claims.
"""

from __future__ import annotations

import bisect
import contextlib
import dataclasses
import datetime
import itertools
import json
import logging
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import fsspec
import icechunk
import numpy as np
import xarray as xr
import zarr

from tessera_embeddings.config.environment import code_identity
from tessera_embeddings.config.fault_injection import ArmedFault
from tessera_embeddings.config.inference import (
    EMBEDDING_DIM,
    OPTICAL_MIN_OBS,
    RADAR_THIN_MAX_OBS,
    TimeWindow,
)
from tessera_embeddings.config.store_layout import (
    CARRIED_VARS,
    INNER_PX,
    MONTH_COORD,
    MONTH_COVERED_VARS,
    MONTHS_IN_YEAR,
    OBS_COUNT_VARS,
    REQUIRED_VARS,
    SINGLE,
    StoreLayout,
    trailing_extent,
)
from tessera_embeddings.inference.chunk_spec import ChunkSpec, chunk_label, filter_chunks_by_roi_mask, parse_chunk_label
from tessera_embeddings.storage import zone_grid
from tessera_embeddings.storage.conventions import build_convention_attrs
from tessera_embeddings.storage.empty_store import _write_coord_arrays
from tessera_embeddings.storage.global_store import create_layout_arrays, open_global_repo
from tessera_embeddings.storage.icechunk_logging import traced_commit
from tessera_embeddings.storage.manifest import EmbeddingManifest, extract_manifest
from tessera_embeddings.storage.object_store import delete_prefix
from tessera_embeddings.storage.registry import part_uri, registry_rows, write_registry_part
from tessera_embeddings.storage.shard_writer import (
    PhaseTimer,
    commit_with_rebase,
    run_forked,
    shard_pitch,
    write_year_shards,
)
from tessera_embeddings.storage.zarr_store import (
    manifest_split,
    open_or_create_repo,
    open_store_as_zarr_group,
    plain_zarr_storage_options,
    read_time_values,
    time_index_of,
)
from tessera_embeddings.storage.zone_grid import PIXEL_M, year_timestamp

logger = logging.getLogger(__name__)


#: botocore retry config for staged-Zarr GETs during assembly. Staged reads fan out across every
#: worker process at once, so a momentary GET burst trips S3 into a 503 SlowDown well under the
#: per-prefix ceiling. botocore's default ``legacy`` mode has no client-side rate limiting and
#: does not recognise ``SlowDown`` as throttling, so a burst just exhausts its attempts and fails
#: the block. ``adaptive`` mode recognises ``SlowDown`` and adds a token-bucket rate limiter that
#: self-throttles when S3 pushes back — that feedback loop, not the higher attempt count, is the
#: actual fix.
STAGED_READ_CONFIG_KWARGS = {"retries": {"max_attempts": 10, "mode": "adaptive"}}
_COVERAGE_SUFFIX = ".coverage.zarr"
"""Suffix of the coverage-only tile a refused chunk stages.

Named once because two places must agree on it: the writer that creates such a tile, and the
listing that must NOT read it as a chunk label.
"""


def _staged_storage_options(path: str) -> dict | None:
    """Return s3fs storage options for a staged read or write, or ``None`` off S3.

    The retry config is a botocore client setting and applies only to the S3 backend; local
    staging paths (``/tmp/...``) get ``None`` and open normally.

    Deliberately carries no credentials and no region. Staging is written by an inference actor
    on an instance profile and fsspec resolves AND REFRESHES that itself; the Icechunk credential
    callback exists only because Icechunk's Rust client does not refresh, a problem staging does
    not have. A non-default-region STAGING bucket would need a region here, but the deployment
    keeps staging under ``paths.outputs`` in the store's own region, so the parameter would only
    add four call sites passing None.
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


def _open_staged_tile(path: str) -> zarr.Group:
    """Open a staged tile, refusing one whose write never finished.

    The single in-store completeness check. ``write_chunk`` sets ``staged_complete`` only after
    ``to_zarr`` returns, so its absence means a crash left array metadata over missing data
    chunks — which Zarr reads back as fill values rather than as an error, the one failure that
    would otherwise reach the published store looking like legitimate data.

    The ``.done``-marker listing gate (:meth:`ZarrWriter._done_marker_path`) normally excludes
    such a tile first, and this check is free for a reader that must open the group anyway. It
    earns its place on the one case the listing cannot cover: the listing is taken once in the
    driver, while tiles are read minutes to hours later in worker processes. Two attempts at one
    zone-year share a staging prefix (the run id derives from the inputs), so a tile REWRITTEN in
    that window keeps its old ``.done`` marker throughout, and only the in-store attribute —
    absent until the rewrite finishes — reports the tile as it is now.

    Raises:
        IncompleteStageError: If the tile's write never completed.
        FileNotFoundError: If there is no tile at *path* (zarr's ``GroupNotFoundError`` is a
            subclass). Any other open error — auth, throttling, transient network — propagates
            untouched, so a bad moment on S3 never makes a valid tile look partial.
    """
    group = zarr.open_group(path, mode="r", storage_options=_staged_storage_options(path))
    if not group.attrs.get("staged_complete"):
        raise IncompleteStageError(
            f"Staged tile {path} lacks the staged_complete marker — a crashed write_chunk left partial "
            "chunks, which read back as fill values. Using it would publish silent holes; re-infer the tile."
        )
    return group


class AllChunksSkippedError(RuntimeError):
    """Every chunk of a run resolved to a skip marker, so no staged tile exists.

    Distinct from ``FileNotFoundError``: the run IS real, it just has nothing to measure a chunk
    size from. Callers resuming such a run fall back to their configured chunk size; ``assemble``
    publishes an all-fill timestep.
    """

    def __init__(self, run_id: str) -> None:
        super().__init__(f"Every chunk of run '{run_id}' was skipped — no staged tile to size from.")


@dataclasses.dataclass
class SpatialCoords:
    """Projected coordinate arrays and CRS from an input store."""

    northing: np.ndarray
    easting: np.ndarray
    crs: str | None = None


TARGET_AGGREGATE_S3_CONCURRENCY = 100
"""Fleet-wide ceiling on concurrent S3 PUTs during assembly, divided across workers.

icechunk's ``max_concurrent_requests`` is per-Repository-instance and each assembly worker
process carries its own pickled fork of the session, so aggregate concurrency is
``n_workers * per_worker_cap``, not the per-worker cap alone. ~100 fleet-wide is roughly 1/35 of
S3's ~3500 req/s/prefix ceiling at ~100 ms PUT latency, leaving headroom for retries. The
coordinator opens the repo with ``max_concurrent_requests = target // n_workers`` and the forks
inherit it; no ``save_config`` persistence is needed because forks travel by pickle rather than
re-opening the repo from its URI.

icechunk's flat ``chunks/<random-id>`` keyspace spreads across S3 partitions well, but partition
splitting is adaptive and a hard burst overruns the per-prefix rate before (and even after) S3
adapts — the SlowDown observed at 800 concurrent PUTs. Because ``per_worker_cap`` floors at 1, a
fill's aggregate is ``max(budget, n_workers)`` rather than ``<= budget``: the floor and the
ceiling cannot both hold and the ceiling gives way. ``AssemblyConfig.max_workers`` bounds the
overshoot — see :func:`_s3_budget_split` for why that is the right way round.
"""


def _s3_budget_split(s3_concurrency: int | None, n_workers: int) -> tuple[int, int]:
    """``(effective_workers, per_worker_cap)`` honoring the fleet S3-PUT budget.

    Each fork worker opens its own repo capped at ``per_worker_cap``, so a fill issues up to
    ``effective_workers * per_worker_cap`` concurrent requests. ``s3_concurrency=None`` uses the
    full aggregate target (a lone fill). Both returned values are ``>= 1``.

    **The worker count is NOT reduced to fit the budget.** A per-fill budget is the fleet target
    divided by the cluster count, so on a wide campaign it lands well below the requested worker
    count, and clamping to it would silently cost most of the fork pool on the campaign's longest
    stage. Because ``per_worker_cap`` floors at 1, a worker count above the budget makes the
    aggregate ``max(budget, n_workers)``: the ceiling gives way, not the floor, because the costs
    are asymmetric. Overshooting risks 503s, which retry, and the target sits far below the
    concurrency at which SlowDown was actually observed; holding the target by dropping forks
    costs wall-clock unconditionally on every cell. The overshoot is bounded by
    ``AssemblyConfig.max_workers`` times the fleet's cluster count.

    Measured cost of the clamp and the concurrency evidence:
    ``context_docs/storage/writing-to-the-global-store.md``.
    """
    budget = s3_concurrency if s3_concurrency is not None else TARGET_AGGREGATE_S3_CONCURRENCY
    workers = max(1, n_workers)
    return workers, max(1, budget // workers)


def _assembly_summary_line(**fields: Any) -> str:  # noqa: ANN401 — heterogeneous JSON payload
    """One machine-readable per-assembly record for the profiling tools.

    The assembly-phase counterpart of ``actors._chunk_summary_line``: one
    ``ASSEMBLY_SUMMARY: {json}`` line per assembly, never per tile — at zone scale that would be
    thousands of lines and would perturb the I/O being measured. Same prefix-plus-JSON convention,
    so keep the keys stable or update the parsers in the same change.

    What the fields mean, and what they deliberately do not claim:

    * ``read_s``/``read_cpu_s`` — staged-tile fetches, summed across workers;
      ``write_s``/``write_cpu_s`` — raw-zarr region assignments, summed. Compression and upload
      are FUSED inside one assignment this code cannot see into, so there is no separate compress
      or upload timer: ``*_cpu_s`` bounds in-process compute (for writes, the compression) and
      ``*_s - *_cpu_s`` is time blocked on the object store. ``fused_compress_put`` says so
      in-band, so a reader of the record alone cannot mistake ``write_s`` for upload time.
    * ``workers`` — per-worker stats in payload order (band order for the single-ROI engine,
      round-robin partition order for the global one). ``wall_s - read_s - write_s`` per worker is
      time outside both phases (validation, partitioning, interpreter start); the slowest worker's
      ``wall_s`` bounds the fill, since every payload gets its own process.
    * ``workers_requested`` vs ``workers_used`` — what was asked for vs how many forks ran. The S3
      budget and the work partition can each cap the count, and this pair is what exposes a fill
      quietly running below its requested width.
    * ``per_worker_s3_cap`` — each fork's concurrent-request cap (``budget // workers``, see
      :data:`TARGET_AGGREGATE_S3_CONCURRENCY`).
    * ``tiles``/``writes``/``bytes`` — tile loads, region assignments and uncompressed bytes handed
      to zarr, summed across workers, so rates derive without a store listing.
      ``tiles_staged``/``tiles_cleared`` are the caller's intent: real data vs fill-over-skip.
    * ``fill_wall_s``/``merge_s`` — the fork-to-merge span and the merge alone;
      ``commit_s``/``attrs_commit_s`` — the data and attrs commits; ``total_s`` — the whole call.
    """
    return "ASSEMBLY_SUMMARY: " + json.dumps(fields, sort_keys=True)


#: Worker-stats keys summed into the ASSEMBLY_SUMMARY totals. ``wall_s``/``cpu_s`` are summed as
#: ``worker_wall_s``/``worker_cpu_s``: a SUM of per-worker walls measures aggregate occupancy, not
#: the fill's duration (``fill_wall_s``), the two differing by exactly the parallelism achieved.
_WORKER_SUM_KEYS: tuple[str, ...] = (
    "tiles",
    "cleared",
    "writes",
    "bytes",
    "read_s",
    "read_cpu_s",
    "write_s",
    "write_cpu_s",
    "wall_s",
    "cpu_s",
)


def _sum_worker_stats(workers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum the per-worker stats into record-level totals (missing keys count as absent, not zero)."""
    totals: dict[str, Any] = {}
    for key in _WORKER_SUM_KEYS:
        values = [w[key] for w in workers if key in w]
        if values:
            total = sum(values)
            name = f"worker_{key}" if key in ("wall_s", "cpu_s") else key
            totals[name] = round(total, 3) if isinstance(total, float) else total
    return totals


def read_store_spatial_coords(
    store_path: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    s3_region: str | None = None,
) -> SpatialCoords:
    """Read projected x/y coordinates and CRS from any mosaic child store.

    Reads via the raw-zarr opener: the 1-D coord arrays and root attrs off metadata, never an
    xarray/dask graph over the store's data chunks. ``get_credentials``/``s3_region`` are threaded
    through so a store authenticating only through the campaign's credential callback, or living
    outside the default region, is readable.

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

    Thin wrapper over :func:`read_store_spatial_coords` for the reflectance store under
    ``mosaic_base``, the fill's canonical grid reference. Runs once per zone-year fill.

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

    Bands start and end on multiples of the output's northing write granularity (shard height
    when sharded, chunk height otherwise), so no two workers touch the same output chunk — the
    fork/merge write-conflict invariant.

    Without ``weights``, whole granularity units spread as evenly as possible. With ``weights``
    (live tiles overlapping each unit), boundaries balance work rather than height: an ROI mask
    clusters live tiles spatially, so equal-height bands would idle most workers while one drags
    the assembly. Either way the bands tile ``[0, total_y)`` exactly, the last absorbing the
    ragged tail.
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

    All data variables must agree (both presets do); a disagreement would let two bands share an
    output object, so it raises rather than taking a max().
    """
    sizes = {var: shard_pitch(cast(zarr.Array, node[var])) for var in variables}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Data variables disagree on northing write granularity: {sizes}")
    return next(iter(sizes.values()))


def _layout_matching_store(root: zarr.Group, layout: StoreLayout, variables: Iterable[str]) -> StoreLayout:
    """*layout*, with each missing variable's geometry taken from the store it joins.

    A variable added to an EXISTING store must adopt that store's chunk and shard geometry, not
    the current preset's. All data variables must agree on a write granularity
    (:func:`_write_granularity` raises otherwise, since disagreeing arrays would let separate
    forks share an output object), so an array created at a different pitch from its siblings
    does not merely make the store mixed — it makes it unassemblable. A store written before a
    preset changed still accepts appends, and gaining a variable must not be what breaks it.

    Geometry comes from an existing array of the same rank; there always is one, since
    ``embeddings`` and ``scales`` are mandatory and cover 4-D and 3-D. Only chunks and shards are
    copied — dtype, fill value and codec are properties of the variable, not the store's tiling.
    A rank nothing in the store shares falls through to the layout unchanged.
    """
    donors: dict[int, tuple[tuple[int, ...], tuple[int, ...] | None]] = {}
    for name, array in root.arrays():
        if name not in layout.arrays:
            continue  # coordinates and time_bnds tile on their own terms
        donors.setdefault(array.ndim, (array.chunks, array.shards))

    adjusted = dict(layout.arrays)
    for var in variables:
        original = layout.for_var(var)
        donor = donors.get(len(original.dims))
        if donor is None:
            continue
        chunks, shards = donor
        adjusted[var] = dataclasses.replace(original, chunks=chunks, shards=shards)
    return dataclasses.replace(layout, arrays=adjusted)


def _fill_band_worker(payload: dict[str, Any]) -> Any:  # noqa: ANN401 — returns (ForkSession, stats)
    """Write one northing band's staged-tile slices into a forked session.

    For each staged tile overlapping the band, reads the tile's overlapping y-slice — one slice
    in flight per worker, so peak RAM is bounded by one tile, ~0.5 GB int8 at 2048 px — and writes
    it into every output array by plain zarr assignment. Partial output chunks at tile
    x-boundaries are read-modify-written sequentially within this fork (icechunk sessions are
    read-your-writes), so the merged result is exact. ``payload["clear"]`` tiles — every chunk
    this run does not write, on a same-date overwrite, see :meth:`ZarrWriter.assemble` — get the
    fill value written over their footprint so no prior run's data survives under a rerun's skip
    marker or outside a changed ROI.

    Returns ``(fork, stats)``: the fork for the coordinator to merge, plus this band's phase
    timings and counts. ``read`` covers the staged opens and slice fetches, ``write`` the zarr
    assignments (encode and upload fused, see ``_assembly_summary_line``); clear-to-fill
    assignments count as writes too, since they emit output objects like any other.
    """
    fork = payload["fork"]
    t = int(payload["time_index"])
    y0b, y1b = payload["band"]
    root = zarr.open_group(fork.store, mode="a")
    arrays = {var: cast(zarr.Array, root[var]) for var in payload["variables"]}
    timer = PhaseTimer()
    tiles = writes = nbytes = 0
    # Unindexed trailing dims (band) are written in full, so each assignment below covers both
    # the 3-D and 4-D arrays.
    for tile in payload["clear"]:
        y0, y1 = max(tile.y_start, y0b), min(tile.y_stop, y1b)
        for arr in arrays.values():
            # Scalar assignment: zarr broadcasts per chunk without materialising the selection,
            # which as a full-band float32 block would be ~2 GB.
            with timer.phase("write"):
                arr[t : t + 1, y0:y1, tile.x_start : tile.x_stop] = arr.fill_value
            writes += 1
    for tile, path in payload["tiles"]:
        y0, y1 = max(tile.y_start, y0b), min(tile.y_stop, y1b)
        with timer.phase("read"):
            staged = _open_staged_tile(path)
        tiles += 1
        for var, arr in arrays.items():
            staged_arr = cast(zarr.Array, staged[var])
            # Validate dtype before assigning: zarr would silently CAST a float staged tile into
            # an int8 destination and still commit. Every tile, not just the probe — a corrupt
            # run can differ per tile.
            if staged_arr.dtype != arr.dtype:
                raise ValueError(
                    f"Staged tile {path} variable {var!r} has dtype {staged_arr.dtype} but the destination "
                    f"array is {arr.dtype} — a silent cast would corrupt the output."
                )
            # Shape too: a singleton spatial or band dim would broadcast over the destination
            # region and commit repeated values. 4-D destination (embeddings/std) takes a
            # (h, w, band) staged tile; 3-D (scales/obs) takes (h, w).
            spatial_hw = (tile.y_stop - tile.y_start, tile.x_stop - tile.x_start)
            expected_shape = (*spatial_hw, arr.shape[3]) if arr.ndim == 4 else spatial_hw
            if staged_arr.shape != expected_shape:
                raise ValueError(
                    f"Staged tile {path} variable {var!r} has shape {staged_arr.shape}, expected "
                    f"{expected_shape} — a singleton/off-grid dim would broadcast and corrupt the output."
                )
            with timer.phase("read"):
                block = np.asarray(staged_arr[y0 - tile.y_start : y1 - tile.y_start])[np.newaxis]
            with timer.phase("write"):
                arr[t : t + 1, y0:y1, tile.x_start : tile.x_stop] = block
            writes += 1
            nbytes += block.nbytes
        # Drop the group reference so its file handles / S3 connections are collectable before
        # the next tile's read.
        del staged
    return fork, {
        "tiles": tiles,
        "cleared": len(payload["clear"]),
        "writes": writes,
        "bytes": nbytes,
        **timer.stats(),
    }


def _extend_time_axis(node: zarr.Group, time_date: np.datetime64) -> int:
    """Grow every time-dimmed array by one step and write the new timestamp.

    An append IS a resize plus a write at the new index; doing it explicitly keeps raw zarr the
    only write path. Returns the new timestep's index.
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
class StagedListing:
    """What one LIST of a run's staging prefix says about each chunk's artifacts.

    The three states are mutually exclusive and each has exactly one remedy, which is what lets
    every caller agree on how to treat a resumed run:

    * ``complete`` — staged ``.zarr`` and ``.done`` marker both landed, so the write finished.
      **Skip it**, after :meth:`ZarrWriter._validate_staged_chunk` confirms shape and dtype.
    * ``interrupted`` — one of the pair is missing, so a crash caught the write part-way and the
      ``.zarr``'s data chunks may be absent, which Zarr reads back silently as fill values.
      **Re-infer it**; ``write_chunk``'s ``mode="w"`` overwrites, so no cleanup first.
    * ``skipped`` — a ``.skipped`` marker: a previous attempt found no valid pixels at all.
      **Skip it**, but report it as a skip rather than a success (see :class:`StagedResume`).

    Labels are sorted because ``fs.ls`` order is backend-dependent and downstream probes take
    ``complete[0]`` — the probe tile must not depend on the filesystem.
    """

    complete: list[str]
    interrupted: list[str]
    skipped: list[str]


@dataclasses.dataclass(frozen=True)
class StagedResume:
    """What a resume scan found in staging, with the two artifact kinds kept apart.

    ``done`` is every label that must NOT be re-inferred — staged tiles and skip markers
    together, which is all an inference loop needs. ``skipped`` is the subset from skip markers:
    tiles a previous attempt found had no pixels to write. Callers reporting per-tile outcomes
    need the split, because counting a restored skip as a success makes a resumed zone's tally
    disagree with the same zone's tally on a fresh run.
    """

    done: set[str]
    skipped: set[str]


@dataclasses.dataclass(frozen=True)
class StagedShardSource:
    """:class:`~tessera_embeddings.storage.shard_writer.ShardSource` over staged tiles.

    The global write path's 1:1 mapping (ADR-008 D3): one staged 2048-px inference tile is
    exactly one output shard, so ``live_shards`` is the staged tile grid positions and ``load``
    returns each staged variable whole with a leading time axis. Every tile is validated as it is
    read — each expected variable present, exactly one shard tall and wide, matching the probe's
    dtype — so a truncated, mixed-version or corrupt staged run fails loudly, naming the tile and
    variable, rather than silently casting or leaving fill inside a shard of a year that then
    gets tagged complete. A frozen dataclass of plain strings and tuples, so the shard writer can
    pickle it to spawned workers.
    """

    staging_base: str
    run_id: str
    shards: tuple[tuple[int, int], ...]
    variables: tuple[str, ...]
    shard_px: int
    dtypes: tuple[tuple[str, str], ...] = ()  # (var, dtype) expectations from the probe
    embedding_dim: int = 0  # band width for 4-D vars; 0 = skip the band check
    # Optional vars present in the DESTINATION. A tile carrying one that is NOT in `variables`
    # (the assembled set) is a heterogeneous stage whose var would be silently dropped — checked
    # per tile in load(), so EVERY tile is validated.
    optional_present: tuple[str, ...] = ()
    # Live tiles this run RESOLVED TO A SKIP (no valid pixels), so nothing was staged for them.
    # Written as fill rather than left alone — see `live_shards`.
    cleared: tuple[tuple[int, int], ...] = ()
    fill_values: tuple[tuple[str, float], ...] = ()  # (var, fill) for the cleared tiles

    def live_shards(self) -> list[tuple[int, int]]:
        """Every ``(row, col)`` this run is responsible for — staged AND skipped.

        Skipped tiles are included so the run WRITES its whole footprint, not just the part it
        has data for. A year is filled in two commits (shards, then the completion attrs), so a
        crash between them leaves shards on an unmarked year that the campaign re-dispatches. If
        the retry's live set has shrunk — a re-ingested mosaic can turn a tile with valid pixels
        into one that skips — leaving skipped tiles untouched would preserve the previous
        attempt's data under the new attempt's completion mark, mixing two inputs in one
        write-once year. Writing fill over them makes the published year exactly this run's.

        Still bounded by the land mask, so ocean tiles are never written at all, and an all-fill
        int8 shard is nearly free once zstd has seen it. The single-ROI `assemble` clears its
        non-live footprint for the same reason.
        """
        return [*self.shards, *self.cleared]

    def load(self, shard: tuple[int, int]) -> dict[str, np.ndarray]:
        """Read one staged tile whole and return ``{var: (1, h, w[, band]) block}``."""
        if shard in self.cleared:
            return self._fill_block(shard)
        sy, sx = shard
        path = f"{self.staging_base}/{self.run_id}/{chunk_label(sy, sx)}.zarr"
        group = _open_staged_tile(path)
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
            # Trailing axis of 4-D vars too: a (1, px, px, 1) block would broadcast across every
            # destination band/month and commit repeated values under a check that only looked at
            # [1:3]. The expected width is per-variable (`trailing_extent`) because the two 4-D
            # axes differ — bands are as wide as the embedding, months are twelve.
            want_trailing = trailing_extent(var, self.embedding_dim)
            if want_trailing and block.ndim == 4 and block.shape[3] != want_trailing:
                raise ValueError(
                    f"Staged tile {path} has {var} trailing width {block.shape[3]} but expected "
                    f"{want_trailing} — a singleton axis would broadcast over the output."
                )
            blocks[var] = block
        # Heterogeneous-stage check on EVERY tile, not a first/last sample: an optional var
        # present in the destination but absent from the assembled `variables` (the probe tile
        # lacked it) would be dropped from every shard write, then the year tagged complete.
        extras = [v for v in self.optional_present if v not in self.variables and v in group]
        if extras:
            raise ValueError(
                f"Staged tile {path} carries optional var(s) {extras} absent from the assembled set "
                f"{list(self.variables)} — heterogeneous staged run; re-stage with one config before assembling."
            )
        return blocks

    def _fill_block(self, shard: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
        """A whole tile of fill for a position this run skipped — with any coverage it DID measure.

        Built from the DESTINATION's fill values, which the caller reads off the seeded arrays, so
        a cleared tile reads back identical to one never written — not zero for a float array whose
        fill is NaN.

        The coverage variables are then overlaid from the refused chunk's own tile where it staged
        one. A fully refused chunk measures real observation counts and month coverage before it
        fails the gate, so filling those alongside the embeddings would publish zeros for a tile
        that had been looked at, while a MIXED tile published true counts because one pixel in it
        happened to embed — provenance that depends on a neighbour, and nothing marks it.

        Absence stays ordinary and silent: an ocean position never staged coverage, and neither did
        a refused chunk from an older run. Both fill. What is NOT tolerated is a tile that exists
        and disagrees with the destination, because a raw-zarr write would C-cast it.
        """
        dtypes = dict(self.dtypes)
        fills = dict(self.fill_values)
        blocks: dict[str, np.ndarray] = {}
        for var in self.variables:
            shape: tuple[int, ...] = (1, self.shard_px, self.shard_px)
            trailing = trailing_extent(var, self.embedding_dim)
            if trailing:
                shape = (*shape, trailing)
            blocks[var] = np.full(shape, fills.get(var, 0), dtype=np.dtype(dtypes.get(var, "float32")))
        if shard is None:
            return blocks
        sy, sx = shard
        path = f"{self.staging_base}/{self.run_id}/{chunk_label(sy, sx)}{_COVERAGE_SUFFIX}"
        try:
            group = _open_staged_tile(path)
        except (FileNotFoundError, IncompleteStageError):
            # ABSENT and HALF-WRITTEN are the same answer for an OPTIONAL artifact: fill, and
            # publish the year. The skip marker means no resume will ever re-stage this tile, so
            # refusing here would wedge the cell on every retry under the stable run id — over
            # provenance, not data. A crash between `to_zarr` and `staged_complete` leaves exactly
            # this state, with no handler having run.
            return blocks
        for var in self.variables:
            if var not in group:
                continue
            block = np.asarray(cast(zarr.Array, group[var])[:])[np.newaxis]
            want = dtypes.get(var)
            if want is not None and str(block.dtype) != want:
                raise ValueError(
                    f"Coverage tile {path} has {var} dtype {block.dtype}, expected {want} — a raw-zarr "
                    "write would silently C-cast (wraparound); mixed-version or corrupt staged run."
                )
            if block.shape != blocks[var].shape:
                raise ValueError(
                    f"Coverage tile {path} has {var} shape {block.shape}, expected {blocks[var].shape} — "
                    "truncated or off-grid tile (ADR-008 D3 requires whole tiles)."
                )
            blocks[var] = block
        return blocks


def _tile_bboxes_wgs84(zone: str, labels: Iterable[str]) -> dict[str, tuple[float, float, float, float]]:
    """WGS84 ``(west, south, east, north)`` per tile label, for the registry's bbox columns.

    A label's ``(row, col)`` indexes the SHARD grid — the same interpretation
    ``StagedShardSource.load`` uses — so the box is that shard's footprint and no new assumption
    about chunk size enters here.

    Best-effort per label: a malformed label, or a row/col outside the zone, yields no entry rather
    than a wrong box or an exception. This runs after the cell has committed, so nothing here may
    raise, and a missing box publishes as null, reading as "not recorded" instead of pointing a
    consumer at the wrong ground.
    """
    try:
        spec = zone_grid.zone(zone)
    except Exception:
        logger.warning("Zone %s has no grid spec; registry rows will carry no bounding box", zone)
        return {}
    boxes: dict[str, tuple[float, float, float, float]] = {}
    for label in labels:
        try:
            row, col = parse_chunk_label(label)
            boxes[label] = zone_grid.tile_range_bbox_wgs84(spec, row, row + 1, col, col + 1)
        except Exception:
            logger.warning("Could not derive a bounding box for tile %s of zone %s", label, zone)
    return boxes


class ZarrWriter:
    """Write embeddings to Zarr stores.

    Phase 1: each chunk is written as a standalone raw Zarr at a staging location
    (:meth:`write_chunk`, on GPU actors). Phase 2: staged chunks are assembled into the final
    Icechunk store with raw-zarr fork/merge writes — :meth:`assemble` for standalone single-ROI
    stores, :meth:`assemble_global` for a pre-allocated zone group of the global store.
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
        # This assembly's skip records, read from the markers by `_skip_summary` and consumed by
        # `publish_registry_part` afterwards. Always present rather than appearing only once a
        # summary has run, so a reader can tell "no skips" from "a previous cell's skips";
        # `assemble_global` clears it at the start of every assembly.
        self._last_skip_records: dict[str, dict] = {}

    def _staging_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Get the staging path for a chunk."""
        return f"{self.staging_base}/{run_id}/{chunk.label}.zarr"

    def _coverage_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Where a REFUSED chunk stages the coverage it measured but could not embed.

        A third artifact class beside the staged tile and the skip marker, because neither can
        carry this. A fully refused chunk still has real observation counts and month coverage —
        accumulated per strip regardless of validity — so discarding them published zero counts for
        the tile while a MIXED tile published true ones: whether a pixel's provenance survived
        depended on whether a neighbouring pixel happened to embed.

        Named ``.coverage.zarr`` and excluded explicitly from the ``.zarr`` match in
        :meth:`_list_staged`; without that it parses as a chunk label of its own, lands in
        ``staged`` with no ``.done`` beside it, and every refused chunk reads as an interrupted
        write.

        NOT vouched for by its own ``.done``: it is written BEFORE the skip marker, so the marker's
        presence already implies it finished — the same ordering trick ``.done`` plays for a staged
        tile, reusing an artifact that has to exist anyway.
        """
        return f"{self.staging_base}/{run_id}/{chunk.label}{_COVERAGE_SUFFIX}"

    def _skip_marker_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Get the skip-marker path for a chunk.

        A zero-byte object recording that the chunk was intentionally skipped during inference
        (every pixel failed the validity filter). Distinct from "no state at all", which means the
        chunk was never processed or was silently dropped by a crashing worker.
        """
        return f"{self.staging_base}/{run_id}/{chunk.label}.skipped"

    def _done_marker_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Get the completion-marker path for a staged chunk.

        A zero-byte object written by :meth:`write_chunk` **after** the staged Zarr is fully
        uploaded. A staged ``.zarr`` is many objects (group and array metadata plus one per data
        chunk) with no atomic multi-object commit, so a crash mid-upload leaves valid metadata over
        missing data chunks — which Zarr reads back as *fill values*, not an error.

        This marker is the LIST-visible completeness signal: every consumer that learns what a run
        staged from one prefix listing (:meth:`_list_staged`, and through it
        :meth:`verify_staged_completeness` and :meth:`scan_existing_staged_artifacts`) keys on it,
        so an interrupted write is recognised without opening anything. The in-store
        ``staged_complete`` attribute written just before it is the same fact in a form a reader
        that already has the group open can check for free — see :meth:`_validate_staged_chunk`.
        """
        return f"{self.staging_base}/{run_id}/{chunk.label}.done"

    def _skip_summary(
        self,
        run_id: str,
        staged: list[str],
        skipped: list[str],
    ) -> dict:
        """The year's optical-skip summary, with the per-shard reasons read from their markers.

        Read HERE and not later because this is the last moment the reasons exist: the markers go
        with the staging prefix when it is cleaned up, the mosaic they came from goes when the cell
        lands, and a published cell is write-once.

        ``detail_uri`` is where the PER-TILE records are persisted, which is what makes a future
        infill campaign possible: the summary in the store's attributes is pooled over the year, so
        it can say a tile somewhere reached fourteen observations against a cutoff of fifteen but
        not WHICH tile, and ranking candidates by how close they came is the whole of that planning
        problem.

        Does NOT publish them — it keeps them on the writer for :meth:`publish_registry_part`,
        which the caller runs only after the cell has committed. Registry consumers are told they
        need not open the store, so a part written before the commit advertises a cell that may
        never land.
        """
        records, unreadable = read_skip_records(self.staging_base, run_id, skipped)
        # KEPT for the caller to publish AFTER the commit — see `publish_registry_part`. Reading
        # here is not optional: this is the last moment the markers exist.
        self._last_skip_records = dict(records)
        summary = summarise_optical_skips(staged=staged, skipped=skipped, records=records)
        if unreadable:
            # SURFACED, because "no records" and "every read failed" are the same empty dict:
            # without it a systematic failure (expired credentials, a wrong prefix) publishes a
            # provenance entry quietly saying no reason was recorded for anything.
            summary["unreadable_markers"] = unreadable
            logger.warning(
                "Run %s: %d of %d skip marker(s) could not be read, so their refusal reasons are "
                "absent from the year's record",
                run_id,
                unreadable,
                len(skipped),
            )
        return summary

    @staticmethod
    def _stage_month_masks(
        data_vars: dict[str, tuple[list[str], np.ndarray]],
        encoding: dict[str, dict],
        month_covered: Mapping[str, np.ndarray | None] | None,
        chunk: ChunkSpec,
        staged_chunks_2d: tuple[int, int],
    ) -> None:
        """Add every present month mask to a staged tile's ``data_vars``/``encoding``, in place.

        Shared by :meth:`write_chunk` and :meth:`write_coverage_only`, which must stage this
        identically — same transpose, same chunking, same deliberate absence of a dtype;
        ``test_coverage_only_staging`` guards the equivalence.

        Month axis LAST in the staged file, matching the destination's
        ``(northing, easting, month)`` so nothing transposes on the way in. The actor keeps it
        FIRST, because that is the axis the mask bundle slices on.

        No dtype in the encoding, deliberately: ``to_zarr`` stores a bool array as int8 with
        ``attrs dtype="bool"`` and IGNORES an encoding dtype asking for bool, so the staged tile is
        int8 either way. The destination is seeded int8 to match, because assembly reads staged
        tiles with RAW zarr and compares against what is on disk rather than what xarray hands
        back — a bool destination refuses every staged month tile on the dtype guard.
        """
        if month_covered is None:
            return
        expected = (MONTHS_IN_YEAR, chunk.height, chunk.width)
        for var in MONTH_COVERED_VARS:
            arr = month_covered.get(var)
            if arr is None:
                continue
            if arr.shape != expected:
                raise ValueError(f"Expected {var} shape {expected}, got {arr.shape}")
            data_vars[var] = (
                ["northing", "easting", "month"],
                np.ascontiguousarray(arr.transpose(1, 2, 0)),
            )
            encoding[var] = {"chunks": (*staged_chunks_2d, MONTHS_IN_YEAR), "compressors": None}

    def write_coverage_only(
        self,
        chunk: ChunkSpec,
        run_id: str,
        obs_counts: Mapping[str, np.ndarray | None],
        month_covered: Mapping[str, np.ndarray | None] | None,
    ) -> str:
        """Stage the coverage arrays of a chunk that embedded nothing. Call BEFORE the marker.

        Only the provenance variables — observation counts and the month-coverage mask — with the
        same dtypes, axis order and chunking :meth:`write_chunk` gives them, because assembly reads
        both with raw zarr and compares against what is on disk. ``test_coverage_only_staging``
        asserts that equivalence rather than trusting the two paths to stay in step.

        No embeddings and no scales, which is the point: those are what the chunk failed to
        produce, and this avoids staging a 128-band fill array per refused tile. Assembly fills
        them and copies these — see :meth:`StagedShardSource._fill_block`.
        """
        path = self._coverage_path(run_id, chunk)
        staged_chunks_2d = (min(INNER_PX, chunk.height), min(INNER_PX, chunk.width))
        data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
        encoding: dict[str, dict] = {}
        expected_2d = (chunk.height, chunk.width)
        for var_name in OBS_COUNT_VARS:
            arr = obs_counts.get(var_name)
            if arr is None:
                continue
            if arr.shape != expected_2d:
                raise ValueError(f"Expected {var_name} shape {expected_2d}, got {arr.shape}")
            data_vars[var_name] = (["northing", "easting"], arr)
            encoding[var_name] = {"chunks": staged_chunks_2d, "compressors": None}
        self._stage_month_masks(data_vars, encoding, month_covered, chunk, staged_chunks_2d)
        if not data_vars:
            raise ValueError(f"Chunk {chunk.label}: no coverage arrays to stage")
        ds = xr.Dataset(
            data_vars,
            coords={
                "northing": np.arange(chunk.y_start, chunk.y_stop),
                "easting": np.arange(chunk.x_start, chunk.x_stop),
            },
        )
        ds.to_zarr(path, mode="w", encoding=encoding, storage_options=_staged_storage_options(path))
        # `staged_complete`, for the same reason write_chunk sets it and read through the same
        # gate (`_open_staged_tile`). A crash between the array metadata and the chunk objects
        # leaves a tile that reads back as FILL rather than as an error, which here would publish
        # zeroed counts looking like measured ones.
        marker = zarr.open_group(path, mode="a", storage_options=_staged_storage_options(path))
        marker.attrs["staged_complete"] = True
        logger.info("Staged coverage-only tile for refused chunk %s to %s", chunk.label, path)
        return path

    def publish_registry_part(
        self,
        registry_root: str,
        zone: str,
        year: int,
        run_id: str,
        *,
        embedded: list[str],
        refused: list[str],
        optical_min_obs: int | None = None,
        embedded_records: Mapping[str, dict] | None = None,
        get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
        s3_region: str | None = None,
    ) -> str | None:
        """Publish this cell's registry part. Call ONLY after the cell has committed.

        The ordering is the point. The registry is a sibling of the store, not part of it, and
        nothing reconciles the two — so a part written while assembly was still running advertises
        embedded and refused tiles for a zone-year that may never land, permanently. Publishing
        after the commit makes the part's existence mean the cell exists.

        Returns the URI on success, ``None`` on failure. Best-effort in this direction only: losing
        a part costs a future infill campaign some precision, while raising would fail a cell that
        has already landed. A lost part is recoverable anyway, since every column is derivable from
        the store the cell just wrote.
        """
        # Refused shards' records come from their markers, embedded shards' from the actors'
        # results. Merged because a row's measurements mean the same thing either way (see
        # `_coverage_record`); the marker side wins on a label in both, since a marker is written
        # at the end of a shard that refused everything.
        records = {**(embedded_records or {}), **self._last_skip_records}
        uri = part_uri(registry_root, zone, year, run_id)
        boxes = _tile_bboxes_wgs84(zone, [*embedded, *refused])
        # The build publishing this part IS the build that produced the cell — same process.
        # Returns None rather than raising, so a wheel install with no VCS information just leaves
        # the columns null.
        build = code_identity() or {}
        try:
            # CREDENTIALLED, like every other write: the registry sits in the store's bucket, so
            # against a partner-owned store it needs the same assumed role, and `_fs_for` with no
            # options falls back to fsspec's ambient chain (the task role, which cannot write
            # there). The failure is caught below and the cell is still tagged, so the symptom is
            # not a failed run but a campaign that looks healthy and publishes no registry at all.
            # Resolved per call, so a fill outliving its credential picks up a fresh one.
            fs = _fs_for(uri, plain_zarr_storage_options(uri, get_credentials, s3_region))
            fs.makedirs(uri.rsplit("/", 1)[0], exist_ok=True)
            written = write_registry_part(
                uri,
                registry_rows(
                    run_id,
                    datetime.datetime.now(datetime.UTC).isoformat(),
                    embedded=embedded,
                    refused=refused,
                    records=records,
                    optical_min_obs=optical_min_obs,
                    bboxes=boxes,
                    code=build,
                ),
                open_output=lambda target: fs.open(target, "wb"),
                zone=zone,
                year=year,
                # Also in the key-value block, so a part read on its own, without the dataset's
                # partitioning, still states the rule its rows were judged against.
                extra_metadata={
                    "optical_min_obs": "" if optical_min_obs is None else str(optical_min_obs),
                    "code_commit": str(build.get("commit") or ""),
                    "code_version": str(build.get("version") or ""),
                },
            )
        except Exception:
            logger.exception(
                "Run %s: registry part for %s year %d could not be published to %s. The cell itself "
                "is committed and unaffected, and every column is derivable from it, so this is a "
                "rebuildable index rather than lost data. READ THIS AS A CONFIGURATION FAULT IF IT "
                "REPEATS: a permission or credential failure here is identical on every cell, so the "
                "campaign completes green having published NO registry — the one artifact a consumer "
                "is told to read instead of the store.",
                run_id,
                zone,
                year,
                uri,
            )
            return None
        logger.info("Published registry part with %d tile row(s) to %s", written, uri)
        return uri

    def discard_coverage(self, chunk: ChunkSpec, run_id: str) -> None:
        """Remove a chunk's coverage-only tile, complete or partial. Never raises.

        Called when :meth:`write_coverage_only` failed partway. A tile whose ``staged_complete``
        was never set reads back as fill and is REFUSED by :func:`_open_staged_tile`, while the skip
        marker written immediately afterwards makes every resume omit the chunk — so nothing repairs
        it and, under the stable run id, the refusal wedges the cell on every retry. Best-effort
        because it runs on an already-failing path: another exception here would turn a degraded
        provenance entry into a lost chunk.
        """
        path = self._coverage_path(run_id, chunk)
        try:
            fs = _fs_for(path)
            with contextlib.suppress(FileNotFoundError):
                fs.rm(path, recursive=True)
        except Exception:
            logger.exception(
                "Chunk %s: could not remove the partial coverage tile at %s. Assembly tolerates an "
                "incomplete tile as absent, so this is untidy rather than blocking.",
                chunk.label,
                path,
            )

    def write_skip_marker(self, chunk: ChunkSpec, run_id: str, record: dict | None = None) -> str:
        """Write a skip marker for a chunk, carrying WHY it was refused.

        Called instead of ``write_chunk`` when a live (ROI-intersecting) chunk has no valid pixels.
        The marker's PRESENCE lets assembly-only runs distinguish a legitimate skip from a
        silently-failed chunk; its CONTENT is the per-shard registry.

        The content is load-bearing because a bare marker cannot tell a thin-depth refusal from no
        coverage at all. The dataset computes three refusal reasons per strip and keeps them apart —
        no optical input, too little of it, no radar — and the actor sums them over the chunk. With
        only an "optical skips" count surviving, 43 of 40S's 58 live shards were named for the wrong
        cause on 2026-08-18 (they were refused for having no RADAR), indistinguishable from land
        never imaged. A published cell is write-once and the mosaic it derives from is deleted when
        the cell lands, so a reason not recorded here is not recoverable later.

        ``record`` is written as JSON. ``None``, and an unreadable or empty marker, stays legal and
        reads back as "no reason recorded" — a marker from an older run is not an error, see
        :func:`read_skip_records`.
        """
        path = self._skip_marker_path(run_id, chunk)
        fs = _fs_for(path)
        # A run whose FIRST artifact is a skip marker has no staging dir yet on directory-backed
        # filesystems (write_chunk's to_zarr creates it as a side effect; a bare open() does not).
        # No-op on object stores.
        fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        # Drop any staged zarr for this chunk and its completion marker first. Both can only be
        # crash or stale artifacts, since the chunk is skipping, and leaving the pair makes
        # verify_staged_completeness raise "BOTH a staged zarr and a skip marker" on every retry
        # under the stable run_id, wedging the cell until someone deletes it by hand. The .done
        # goes FIRST so an interruption here cannot leave one vouching for a .zarr already gone.
        for stale in (self._done_marker_path(run_id, chunk), self._staging_path(run_id, chunk)):
            with contextlib.suppress(FileNotFoundError):
                fs.rm(stale, recursive=True)
        # THE MARKER'S PRESENCE IS LOAD-BEARING; ITS CONTENT IS NOT. `verify_staged_completeness`
        # tells a legitimate skip from a crashed worker by this file existing, so a record that
        # cannot be serialised must cost the RECORD, never the marker: losing the reason degrades
        # provenance, while losing the marker turns a benign skip into a failed chunk and wedges
        # the cell on every retry. An un-cast numpy scalar reaching `json.dumps` is how that
        # happens.
        body = b""
        if record is not None:
            try:
                # Sorted keys and a trailing newline: read by eye during an investigation at least
                # as often as by the summariser.
                body = json.dumps(record, sort_keys=True, indent=2).encode() + b"\n"
            except (TypeError, ValueError):
                logger.exception("Chunk %s: skip record could not be serialised; writing a bare marker", chunk.label)
        with fs.open(path, "wb") as f:
            f.write(body)
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
        month_covered: Mapping[str, np.ndarray | None] | None = None,
    ) -> str:
        """Write one chunk's embeddings to a staged intermediate (non-Icechunk) Zarr store.

        Creates an xarray Dataset with an 'embedding' variable (mean) and optionally
        'embedding_std' and 'scale' variables.

        Records completion **after** the store is fully written — the ``staged_complete`` in-store
        attribute, then the ``<label>.done`` sibling marker (:meth:`_done_marker_path`) — so an
        interrupted upload leaves a tile resume re-infers, rather than one assembly trusts with
        fill-valued holes.

        Args:
            chunk: Chunk specification.
            embeddings: Array of shape (H, W, embedding_dim), float32 or int8.
            run_id: Unique run identifier.
            scales: Per-pixel float32 scale factors of shape (H, W), used to dequantize int8
                embeddings.
            embeddings_std: Optional std array, same shape as embeddings, float32.
            month_covered: Optional dict of month-mask variable names to (12, H, W) bool arrays —
                which calendar months each pixel was seen in per sensor, month 0 = January. Keys
                from ``MONTH_COVERED_VARS``. Staged transposed to (H, W, month) so each lands in
                its destination's axis order and assembly writes it without reordering.
            obs_counts: Optional dict of obs count variable names to (H, W) uint16 arrays. Keys
                from ``OBS_COUNT_VARS``.

        Returns:
            Path to the staged Zarr store.
        """
        expected_shape = (chunk.height, chunk.width, self.embedding_dim)
        if embeddings.shape != expected_shape:
            msg = f"Expected shape {expected_shape}, got {embeddings.shape}"
            raise ValueError(msg)

        path = self._staging_path(run_id, chunk)
        # Sub-chunk staged files at inner-chunk size, full band (ADR-008 D2: the band axis is
        # never split). A staged 2048-px tile is then exactly the 8x8 inner-chunk grid of the shard
        # it becomes, so assembly's banded y-slice reads stay aligned to whole staged pieces.
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

        self._stage_month_masks(data_vars, encoding, month_covered, chunk, staged_chunks_2d)

        ds = xr.Dataset(
            data_vars,
            coords={
                "northing": np.arange(chunk.y_start, chunk.y_stop),
                "easting": np.arange(chunk.x_start, chunk.x_stop),
                "band": np.arange(self.embedding_dim),
            },
        )

        # Retract the previous attempt's completion marker BEFORE overwriting the tile: to_zarr
        # replaces the tile in place, so an older marker keeps vouching for it throughout the
        # rewrite and a listing taken in that window calls an incomplete tile complete. Retracting
        # first makes the window read as interrupted, which is the truth. No-op on a first write.
        #
        # The SKIP marker goes too, for a different reason: a chunk that skipped on an earlier
        # attempt and produces pixels on this one would carry both, and verification reads that
        # pair as an inconsistent artifact and refuses to assemble — repeating on every retry under
        # the stable, input-fingerprinted run_id until someone deletes the marker by hand.
        # `write_skip_marker` clears a stale tile in the opposite direction; whichever outcome a
        # chunk reaches, it leaves no trace of the other.
        done_path = self._done_marker_path(run_id, chunk)
        done_fs = _fs_for(done_path)
        for stale in (done_path, self._skip_marker_path(run_id, chunk)):
            with contextlib.suppress(FileNotFoundError):
                done_fs.rm(stale)
        # And the COVERAGE-ONLY tile of a previous refusal, the third artifact under this rule. It
        # carries real counts, so a stale one beside a successful write measures a footprint this
        # attempt has just replaced. Recursive — it is a directory, not a marker.
        with contextlib.suppress(FileNotFoundError):
            done_fs.rm(self._coverage_path(run_id, chunk), recursive=True)

        # The same S3 client config the staged READS use: the write is the larger and more
        # failure-prone of the two, since it uploads the tile.
        ds.to_zarr(path, mode="w", encoding=encoding, storage_options=_staged_storage_options(path))
        # --- Completion markers, written LAST, in SEPARATE ops after to_zarr returns. to_zarr can
        # create every array's metadata before all its chunk objects, so a crash mid-write leaves a
        # tile with correct vars/shape/dtype but MISSING chunks (read back as fill, not as an
        # error). Completeness is therefore its own signal, recorded twice for two consumers:
        #
        #   1. `staged_complete` in-store attribute — free for a reader that already has the group
        #      open (assembly's per-tile reads), and the only signal reflecting the tile as it is
        #      NOW rather than as the last listing saw it (see _open_staged_tile).
        #   2. `<label>.done` sibling object — visible in the staging prefix LISTING, so
        #      verification and the resume scan classify a whole run from one listing.
        #
        # ORDER MATTERS and is the invariant both checks rely on: .done last, so `.done present`
        # implies `attribute set` implies `to_zarr returned`. A crash anywhere earlier leaves a tile
        # the listing reports as interrupted, which the resume scan re-infers (mode="w" overwrites).
        marker = zarr.open_group(path, mode="a", storage_options=_staged_storage_options(path))
        marker.attrs["staged_complete"] = True
        with done_fs.open(done_path, "wb") as f:
            f.write(b"")
        logger.info("Wrote %s to %s", chunk.label, path)
        return path

    def detect_staged_chunk_size(self, run_id: str) -> int:
        """Detect the inference chunk_size from staged Zarrs.

        Opens the first available staged chunk and returns ``max(height, width)`` from its shape,
        recovering the ``chunk_size`` used during inference so an assembly-only or resume run can
        re-enumerate the chunk grid. Uses :meth:`_list_staged_labels` rather than assuming
        ``chunk_0_0`` exists — a sparse ROI may not have staged that chunk.

        Args:
            run_id: Run identifier whose staged chunks to inspect.

        Returns:
            The chunk_size (pixels) used during inference.

        Raises:
            AllChunksSkippedError: If the run exists but every chunk skipped.
            FileNotFoundError: If the run has no staged chunks and no skip markers.
        """
        labels = self._list_staged_labels(run_id)
        if not labels:
            if skipped := self._list_skip_marker_labels(run_id):
                # A real run in which every chunk skipped. ASK THE MARKERS what grid they were
                # produced on before giving up: they record it precisely because there is no tile
                # to measure, and falling back to the CURRENT configured size re-enumerates the
                # grid under different labels — after which verification calls the valid old
                # markers unexpected and its own new labels missing, defeating the all-skipped
                # assembly path. The configured side has changed before (2000 -> 2048), so this is
                # not hypothetical for older runs.
                records, _unreadable = read_skip_records(self.staging_base, run_id, skipped)
                sides = {int(r["chunk_side_px"]) for r in records.values() if isinstance(r.get("chunk_side_px"), int)}
                if len(sides) == 1:
                    return sides.pop()
                # Either no marker records the grid, or they disagree — a heterogeneous stage no
                # single size describes. Both are "unknown", leaving the caller's configured size.
                raise AllChunksSkippedError(run_id)
            # A run_id matching nothing at all still raises, so a typo cannot become a silent re-run.
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
        """Verify all expected chunks have either a completed staged zarr or a skip marker.

        Returns the labels with completed staged zarrs — an empty set means every live chunk
        resolved to a skip marker — so callers need not re-LIST the staging prefix to learn what
        verification already read.

        A live (ROI-intersecting) chunk resolves two ways: inference produced embeddings (a staged
        zarr whose ``.done`` marker confirms the many-object write finished, see
        :meth:`_done_marker_path`), or every pixel failed the validity filter (skip marker). A live
        chunk with neither — including one left half-written by a crash — is a silent failure and
        must stop assembly rather than produce output indistinguishable from a legitimate skip.

        Interrupted tiles are their own category, not folded into "missing", because reaching
        assembly is what makes them anomalous: the resume scan re-infers an interrupted tile, so
        one surviving to here means inference never ran (an assembly-only run) or crashed the same
        way twice, and the remedy differs from a chunk never attempted.

        Args:
            run_id: Run identifier (locates the staging directory).
            expected_chunks: Chunks expected to resolve, i.e. ROI-intersecting ones.
                Non-intersecting chunks are filled at assembly time and must not be passed here.
            log: Optional logger.

        Raises:
            IncompleteStageError: If any live chunk has neither a completed staged zarr nor a skip
                marker, or if extra labels match no expected chunk.
        """
        _log = log or logger
        listing = self._list_staged(run_id)
        staged, skipped = set(listing.complete), set(listing.skipped)
        resolved = staged | skipped
        expected = {c.label for c in expected_chunks}

        # A skip marker resolves the chunk on its own, so a leftover half-written pair beside one
        # is inert (write_skip_marker clears both, but a crash between the removals leaves one).
        interrupted = set(listing.interrupted) - resolved
        missing = expected - resolved - interrupted
        extra = (resolved | interrupted) - expected
        both = staged & skipped

        if missing or extra or both or interrupted:
            parts = [f"Staged chunks for run '{run_id}' do not match the expected chunk grid."]
            parts.append(f"Expected {len(expected)} chunks, found {len(staged)} staged + {len(skipped)} skipped.")
            if missing:
                sample = sorted(missing)[:10]
                parts.append(
                    f"{len(missing)} missing (neither zarr nor skip marker): "
                    f"{sample}{'...' if len(missing) > 10 else ''}"
                )
            if interrupted:
                sample = sorted(interrupted)[:10]
                parts.append(
                    f"{len(interrupted)} interrupted (staged zarr and .done marker did not both land — a crash "
                    f"mid-write, whose data chunks would read as fill): "
                    f"{sample}{'...' if len(interrupted) > 10 else ''}"
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

        The single owner of the staging LIST — a full prefix scan, ~15k objects at zone scale, so
        callers derive everything from one pass.
        """
        staging_dir = f"{self.staging_base}/{run_id}"
        fs = _fs_for(staging_dir)
        try:
            # refresh=True forces a fresh S3 LIST rather than the cached dir listing: staged
            # artifacts are written by Ray actor processes, not the flow runner, so the runner's
            # fsspec dircache is never invalidated and would return an earlier ls()'s snapshot.
            entries = fs.ls(staging_dir, detail=False, refresh=True)
        except FileNotFoundError:
            return []
        # Sorted: fs.ls order is backend-dependent (S3 lexicographic, local filesystems
        # arbitrary) and downstream probes take labels[0].
        return sorted(entry.rstrip("/").rsplit("/", 1)[-1] for entry in entries)

    def _list_staged(self, run_id: str) -> StagedListing:
        """Classify every artifact of a run from one staging LIST.

        The single place the three-way split is derived, so verification and the resume scan can
        never disagree about what a run staged. See :class:`StagedListing` for what each bucket
        means and :meth:`_done_marker_path` for why completeness is its own signal.
        """
        names = self._list_run_names(run_id)
        # The coverage-only tile of a REFUSED chunk must be excluded FIRST: it ends in ".zarr", so
        # the match below would take "<label>.coverage" for a chunk label of its own, find no
        # ".done" beside it, and report every refused chunk as an interrupted write.
        staged = {n.removesuffix(".zarr") for n in names if n.endswith(".zarr") and not n.endswith(_COVERAGE_SUFFIX)}
        done = {n.removesuffix(".done") for n in names if n.endswith(".done")}
        skipped = sorted(n.removesuffix(".skipped") for n in names if n.endswith(".skipped"))
        # Symmetric difference, not `staged - done`: a .done whose .zarr is absent is equally a
        # half-landed pair (interrupted cleanup, or a marker outliving its tile) with the same
        # remedy — re-infer, which rewrites both.
        return StagedListing(
            complete=sorted(staged & done),
            interrupted=sorted(staged ^ done),
            skipped=skipped,
        )

    def _list_staged_labels(self, run_id: str) -> list[str]:
        """List chunk labels whose staged write COMPLETED in the run directory.

        Keyed on the ``.done`` completion marker, not the ``.zarr`` directory: a ``.zarr`` without
        its marker is an interrupted write, excluded so resume re-runs it rather than assembling
        its partial, fill-valued contents. See :meth:`_done_marker_path`.

        Returns:
            List of chunk label strings (e.g. ``["chunk_0_0", "chunk_0_1"]``).
            Empty list if the staging directory doesn't exist.
        """
        return self._list_staged(run_id).complete

    def _list_skip_marker_labels(self, run_id: str) -> list[str]:
        """List chunk labels that have skip markers in the run directory."""
        return self._list_staged(run_id).skipped

    def _validate_staged_chunk(
        self,
        run_id: str,
        chunk: ChunkSpec,
        required_vars: list[str],
    ) -> tuple[str, str] | None:
        """Validate a single staged chunk Zarr.

        Returns ``None`` if the tile is complete and valid, else ``(kind, reason)``:

        - ``("incomplete", ...)`` — a crash artifact: genuinely absent (``FileNotFoundError``), or
          rejected by :func:`_open_staged_tile` because its write never finished. The caller
          RE-INFERS it (``write_chunk``'s ``mode="w"`` overwrites) rather than raising, because
          under the stable input-fingerprinted run_id a raise would wedge every retry on the same
          partial artifact. This is the one caller for which an unfinished tile is the expected
          input rather than an anomaly, hence catching what the shared opener raises.
        - ``("invalid", ...)`` — a COMPLETE tile that is structurally wrong (missing var, shape or
          dtype). The caller raises: a real anomaly such as a stale wrong-grid store, not a
          resumable crash.

        Any OTHER open error (auth, throttling, transient network, corrupt metadata) propagates —
        a transient read failure must not silently re-infer a valid completed tile.
        """
        path = self._staging_path(run_id, chunk)
        try:
            group = _open_staged_tile(path)
        except FileNotFoundError:
            # Genuinely gone (partial or removed) — self-heal by re-inference. Other open errors
            # propagate: see the docstring.
            return ("incomplete", "staged zarr not found (partial or removed) — re-infer")
        except IncompleteStageError as exc:
            return ("incomplete", str(exc))
        for var in required_vars:
            if var not in group:
                return ("invalid", f"missing variable '{var}'")
            arr: zarr.Array = group[var]  # type: ignore[assignment]
            # scales is a 2-D per-pixel factor (H, W); embeddings/embedding_std carry the band dim
            # (H, W, band). Checking scales against the 3-D shape false-rejects every valid tile.
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
            Set of ``chunk.label`` strings already staged and valid. Use
            :meth:`scan_existing_staged_artifacts` to learn WHICH of those came from skip markers.

        Raises:
            RuntimeError: If any staged Zarrs are invalid, listing every invalid path so they can
                be removed before retrying.
        """
        return self.scan_existing_staged_artifacts(run_id, chunks, compute_std=compute_std, log=log).done

    def scan_existing_staged_artifacts(
        self,
        run_id: str,
        chunks: list[ChunkSpec],
        *,
        compute_std: bool = False,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    ) -> StagedResume:
        """:meth:`scan_existing_staged_chunks`, keeping skip markers distinguishable.

        Both artifact kinds mean "do not re-infer this tile", which is why the set-returning form
        merges them, but they are different OUTCOMES: a staged zarr produced pixels, a skip marker
        recorded that the tile had none. A resumed zone reporting its skips as successes misstates
        what the run did.
        """
        _log = log or logger
        listing = self._list_staged(run_id)
        staged_labels = listing.complete
        skip_marker_labels = set(listing.skipped)
        if not staged_labels and not skip_marker_labels and not listing.interrupted:
            _log.info("No staged chunks or skip markers found for run %s — starting fresh", run_id)
            return StagedResume(done=set(), skipped=set())

        _log.info(
            "Found %d completed staged chunk Zarrs and %d skip markers — validating",
            len(staged_labels),
            len(skip_marker_labels),
        )
        if listing.interrupted:
            # Not an error — this is what resume exists for. They are absent from `done`, so
            # run_inference re-runs them (mode="w" overwrites the partial). Logged because a tile
            # appearing here run after run means the crash is reproducible, not incidental.
            _log.warning(
                "%d staged tile(s) are interrupted (staged zarr and .done marker did not both land) — "
                "re-inferring: %s%s",
                len(listing.interrupted),
                listing.interrupted[:10],
                "..." if len(listing.interrupted) > 10 else "",
            )

        chunk_by_label = {c.label: c for c in chunks}
        # `scales` is mandatory (dequantization needs it) and staged Zarr writes are NOT atomic
        # across arrays, so a crash mid-write_chunk can leave an embeddings-only tile. Requiring
        # scales rejects that up front; otherwise the resume scan counts it valid, run_inference
        # skips it, and the fill fails permanently at assembly on every retry.
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
                # A crash artifact: EXCLUDE it so run_inference regenerates it (mode="w"
                # overwrites). Do NOT raise — under the stable, input-fingerprinted run_id a raise
                # re-fires on the same artifact every retry and wedges the cell.
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
        return StagedResume(done=valid, skipped=valid & skip_marker_labels)

    def _staged_vars_present(self, run_id: str, chunks: list[ChunkSpec], var_names: tuple[str, ...]) -> set[str]:
        """Which of *var_names* exist in the staged Zarrs — checked across EVERY tile.

        A sampled probe, even first+last, can miss a tile that alone carries an obs-count
        variable and silently drop it from the whole output. So every staged chunk is opened and
        must agree on the optional-variable set; any disagreement is a hard stop. Callers must
        pass only chunks that have staged files. Costs one metadata open per tile, and the band
        fill re-opens them for data — acceptable on the single-ROI path.
        """
        present: set[str] | None = None
        first_label: str | None = None
        for chunk in chunks:
            path = self._staging_path(run_id, chunk)
            try:
                # Rejects a crash-partial tile before the schema commit — the same check the band
                # fill makes on read, but cheaper than failing part-way through a created store.
                group = _open_staged_tile(path)
            except FileNotFoundError as exc:  # GroupNotFoundError subclasses this
                # A silent empty set here would drop every obs-count variable from the output; a
                # chunk that should exist but cannot be opened is a partial stage and must be loud.
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

        Geometry comes from the :class:`StoreLayout` preset — the single source of truth, shared
        with the global store's seeder — not from the staged files, so the on-disk output is
        identical however inference tiled the mosaic. Cost is independent of extent, since no
        pixels are written.
        """
        variables = list(variables)
        sizes = {
            "time": 1,
            "northing": total_y,
            "easting": total_x,
            "band": self.embedding_dim,
            "month": MONTHS_IN_YEAR,
        }
        create_layout_arrays(root, layout, variables, sizes)
        coords: dict[str, np.ndarray] = {
            "time": np.asarray([time_date], dtype="datetime64[ns]"),
            "northing": spatial.northing if spatial else np.arange(total_y),
            "easting": spatial.easting if spatial else np.arange(total_x),
            "band": np.arange(self.embedding_dim),
        }
        if any(var in variables for var in MONTH_COVERED_VARS):
            # Only when an array using the axis is created: a bare `month` coordinate in a store
            # with no month dimension surprises readers. All three sensors share the axis, so it is
            # written once. 1..12, so `cov.sel(month=7)` means July rather than the eighth index,
            # mirroring the global store's seeder.
            coords["month"] = np.asarray(MONTH_COORD, dtype="int16")
        _write_coord_arrays(root, coords)

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

        Raw-zarr fork/merge engine (no Dask): worker processes write staged-tile pixels for
        disjoint, granularity-aligned northing bands straight into the output arrays; the
        coordinator merges the forks, sets root attrs and commits once via
        :func:`~tessera_embeddings.storage.shard_writer.commit_with_rebase`. Emits one
        ``ASSEMBLY_SUMMARY`` record (:func:`_assembly_summary_line`) with the fill's per-worker
        phase timings and counts.

        Create-or-extend semantics on the time axis:

        * fresh store → schema and coords from ``layout`` (``SINGLE`` by default: today's
          single-ROI geometry, D8) in a first metadata-only commit;
        * existing store, new time value → every time-dimmed array is explicitly resized by one
          step in a first commit;
        * existing store, time value already present → that index is overwritten in place, so a
          crashed assembly's re-run lands on the same index rather than appending a duplicate
          timestep. Live positions are rewritten and **every destination chunk this run does not
          rewrite as live is cleared to fill** — skip-marked footprints AND positions outside this
          run's ROI mask, so no prior run's data survives under a rerun's skip marker or is
          stranded as stale embeddings when the ROI shrinks. A same-date re-assembly with a
          different mask is therefore safe: the timestep is exactly this run's live footprint.

        The fill is always the second, single data commit, so a crash mid-fill leaves only an
        all-fill timestep that the re-run overwrites — provided the re-run lands on the same time
        value. With the default date-derived coordinate (no ``time_window``), retry the same day
        or pass the original ``run_started_at``; otherwise the crashed date's all-fill timestep
        persists as a ghost.

        Inference stages only chunks that intersect the ROI mask and had valid pixels; everything
        else is never written and reads back as the layout's fill (0 for int8, NaN for floats).

        Args:
            chunks: Full chunk grid (both live and non-live).
            total_y: Total mosaic height.
            total_x: Total mosaic width.
            run_id: Run identifier (locates staged files).
            output_path: Final output Icechunk store path.
            roi_zarr_path: Path to the ROI boolean zarr. Assembly re-enumerates live chunks from
                this rather than marshaling the list through Prefect.
            compute_std: Whether staged chunks contain embedding_std data.
            run_started_at: Flow trigger time for the time coordinate; defaults to now, and is
                ignored when *time_window* is given.
            mosaic_base: Base path for the input mosaic stores. When given, the reflectance store
                supplies projected coordinates and CRS.
            log: Optional logger (e.g. Prefect's run logger).
            time_window: When given, the window end month is the time coordinate and window
                metadata lands in dataset attributes.
            tile_id: Sentinel-2 MGRS tile ID for ``proj:`` convention attrs when the mosaic store
                carries no ``crs`` attr.
            model_version: Encoder checkpoint identifier, recorded as the ``checkpoint_id``
                provenance attr (``geoemb:model`` is the public encoder URL, derived separately).
            manifest: Typed manifest for append-safety validation. Written on create, validated
                before extending an existing store.
            n_workers: Worker *process* count. Also divides ``TARGET_AGGREGATE_S3_CONCURRENCY``
                into the per-fork request cap, keeping fleet-wide PUT concurrency under S3's
                ceiling.
            get_credentials: Optional icechunk credential callback for the output store (see
                ``zarr_store._create_storage``).
            s3_region: Optional S3 region override for the output store.
            layout: Output geometry preset. ``SINGLE`` (default) reproduces today's single-ROI
                stores exactly; only new stores consult it.

        Returns:
            Path to the assembled output store.
        """
        _log = log or logger
        t0 = time.monotonic()

        # Which chunks have COMPLETED staged zarrs. A chunk that intersects the ROI but skipped
        # during inference has a skip marker instead of a zarr; its footprint stays at fill.
        #
        # The gate runs here, not only in the calling flow, because this method derives its own
        # live set from the ROI mask and is called directly (assembly-only re-runs, tests).
        # Otherwise the only thing between a crash-partial tile and a published timestep of fill
        # would be the per-tile check inside the workers, which fires after the schema commit.
        roi_live_chunks = filter_chunks_by_roi_mask(
            chunks,
            roi_zarr_path,
            storage_options=plain_zarr_storage_options(roi_zarr_path, get_credentials, s3_region),
        )
        staged_labels = self.verify_staged_completeness(run_id, roi_live_chunks, log=_log)
        live_chunks = [c for c in roi_live_chunks if c.label in staged_labels]
        # SKIP-MARKED CHUNKS ARE DROPPED HERE AND THEIR COVERAGE ARTIFACTS GO UNREAD — a DELIBERATE
        # asymmetry with the global path, recorded because it reads as an oversight otherwise.
        # Global assembly consumes each skipped chunk's `.coverage.zarr`, so its observation counts
        # and month coverage are real over refused footprints; this path publishes fill there, so a
        # mixed output carries zeros where the global one carries measurements and an all-skipped
        # fresh output can omit the coverage arrays entirely.
        #
        # It stays that way because the consumers differ: those arrays feed the registry a targeted
        # repair campaign ranks from, and only the global path writes that registry. Closing the gap
        # is tracked with the repair-campaign follow-ups (issue #103) — doing it properly means
        # reading the artifacts the way `StagedShardSource._fill_block` does, and this path has no
        # coverage test that would catch getting it wrong.
        if not live_chunks:
            # Every ROI-intersecting chunk skipped (no valid pixels). Publish an all-fill timestep
            # rather than aborting: a create/append writes fill, and a same-date overwrite clears
            # the prior data (skip-marked footprints are reset in Phase 2).
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

        # `embedding_std` follows the caller's `compute_std`, not what the tiles happen to hold;
        # every other carried var is included when the tiles staged it.
        staged_extra = self._staged_vars_present(run_id, live_chunks, CARRIED_VARS)
        variables = [*REQUIRED_VARS]
        if compute_std:
            variables.append("embedding_std")
        variables += [v for v in CARRIED_VARS if v != "embedding_std" and v in staged_extra]

        # Divide the fleet-wide S3 concurrency target across worker forks (see
        # TARGET_AGGREGATE_S3_CONCURRENCY). Forks inherit the repo config through the pickled
        # session, so no save_config round-trip is needed.
        per_worker_cap = max(1, TARGET_AGGREGATE_S3_CONCURRENCY // max(1, n_workers))

        # Time-only, matching the global store (zarr_store.global_store_config) and the rule
        # documented there: split the axis along which a single commit is NARROW. An assemble
        # writes ONE timestep across the whole spatial extent, so time@1 keeps the write off every
        # prior timestep's manifests, while a spatial split would only shard the axis this commit
        # rewrites in full — more manifest objects for the same refs.
        with manifest_split({"time": 1}):
            repo, is_new = open_or_create_repo(
                output_path,
                max_concurrent_requests=per_worker_cap,
                get_credentials=get_credentials,
                region=s3_region,
                scatter_initial_credentials=True,  # see assemble_global's call site
            )
            # Persist the split on create so a later COLD writer — one opening the store outside
            # this manifest_split block — keeps splitting rather than reverting to O(store)
            # manifest rewrites. Mirrors create_global_repo; only matters for future opens, since
            # forks in THIS session inherit it via the pickled session.
            if is_new:
                repo.save_config()

        # Publish atomically: run the whole schema/extend plus data lifecycle on a private work
        # branch and fast-forward `main` only once the timestep's data has landed. Committing
        # Phase 1 to `main` would advertise a resized array and new time coordinate — or, for a
        # fresh store, an empty schema — BEFORE any worker writes, so a crash in Phase 2/3 would
        # leave `main` serving an all-fill timestep. One writer per single-ROI store (the campaign
        # uses assemble_global), so a fixed branch name is safe; a stale ref from a crashed prior
        # attempt is reset here rather than left to accumulate.
        base_snapshot = repo.lookup_branch("main")
        work_branch = "_assemble-wip"
        if work_branch in repo.list_branches():
            repo.delete_branch(work_branch)
        repo.create_branch(work_branch, base_snapshot)

        # --- Phase 1: schema (create) or time-axis placement (extend) --------
        session = repo.writable_session(work_branch)
        root = zarr.open_group(session.store, mode="a")
        overwrite = False
        # "time" absent means a created-but-never-seeded repo (a crash between repo creation and
        # the schema commit) — treat as fresh.
        if is_new or "time" not in root:
            self._create_schema(root, layout, variables, total_y, total_x, time_date, spatial)
            traced_commit(session, f"Run {run_id}: create schema ({layout.name})")
            time_index = 0
            _log.info("Created %s with layout %s", output_path, layout.name)
        else:
            if manifest:
                manifest.validate_against(extract_manifest(root.attrs), output_path)
            # The store's own grid is authoritative: on a mismatched extent a raw region write
            # would silently land in a corner, or be clamp-truncated.
            emb = cast(zarr.Array, root["embeddings"])
            if emb.shape[1] != total_y or emb.shape[2] != total_x or emb.shape[3] != self.embedding_dim:
                raise ValueError(
                    f"Mosaic extent ({total_y} x {total_x} x {self.embedding_dim} bands) does not match "
                    f"existing store {output_path} ({emb.shape[1]} x {emb.shape[2]} x {emb.shape[3]} bands) "
                    "— wrong output path, ROI grid, or model width."
                )
            # Shape alone cannot catch a shifted or reversed mosaic on the same grid size (the
            # manifest omits origin), which would append under the existing coordinates and be
            # silently misgeoreferenced. So compare CRS and coordinate endpoints against the
            # stored grid, at half-pixel absolute tolerance as in the zone-fill runner.
            #
            # CRS is checked only when present: atomic publish means `main` only ever exposes a
            # COMPLETE store, so a normal re-append always sees one, and the None guard covers
            # only a legacy store left partial by a pre-atomic-publish engine. The coordinate
            # ARRAYS are the real check.
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
                # Extent, CRS and endpoints still do not pin an axis: a reordered or non-affine
                # interior satisfies all three, and this phase writes staged pixels POSITIONALLY
                # against the store's existing coordinate arrays, so such an append publishes real
                # pixels at the wrong coordinates with nothing to signal it. Hence the full-vector
                # comparison the zone-fill runner also makes — two 1-D reads per append. A
                # per-step spacing test cannot substitute: whatever slack it admits per step, an
                # axis that wanders and returns accumulates it while still matching the length and
                # both endpoints. Reachable on the supported hand-provided mosaic path.
                for axis, values, stored in (
                    ("northing", spatial.northing, z_north),
                    ("easting", spatial.easting, z_east),
                ):
                    got = np.asarray(values, dtype="float64")
                    want = np.asarray(stored[:], dtype="float64")
                    # A float round-trip allowance, not a geometric one: coordinates round-trip
                    # through float32 at ~1 m near a 9.3e6 m northing, while any real displacement
                    # is a whole pixel.
                    bad = ~np.isclose(got, want, rtol=0.0, atol=1.0)
                    if bad.any():
                        i = int(np.argmax(bad))
                        raise ValueError(
                            f"Mosaic {axis} does not match existing store {output_path}'s axis: "
                            f"index {i} is {got[i]} where the store says {want[i]} "
                            f"({int(bad.sum())} of {got.size} coordinates differ). Extent, CRS and "
                            "endpoints all match, so this is a reordered or non-affine interior — "
                            "appending would write real pixels at the wrong coordinates."
                        )
            # Variables staged by this run but absent from the store (a store created before obs
            # counts, or compute_std newly on) are created schema-only at the current time extent;
            # prior timesteps read back as the layout fill.
            missing = [v for v in variables if v not in root]
            if missing:
                nt = cast(zarr.Array, root["time"]).shape[0]
                sizes = {
                    "time": nt,
                    "northing": total_y,
                    "easting": total_x,
                    "band": self.embedding_dim,
                    "month": MONTHS_IN_YEAR,
                }
                create_layout_arrays(root, _layout_matching_store(root, layout, missing), missing, sizes)
                if any(var in missing for var in MONTH_COVERED_VARS) and "month" not in root:
                    # The axis this array introduces to a store that predates it; without the
                    # coordinate a reader gets a positional month axis and must know it is 0-based
                    # (see `_create_schema`).
                    _write_coord_arrays(root, {"month": np.asarray(MONTH_COORD, dtype="int16")})
                _log.info("Created missing variable(s) %s in %s from layout %s", missing, output_path, layout.name)
            time_index_found = time_index_of(root, time_date)
            if time_index_found is not None:
                time_index = time_index_found
                overwrite = True
                # A time-dependent array the store carries but THIS run does not write — a store
                # seeded with embedding_std now filled with std off, or the other S1 orbit's obs
                # count on a single-orbit fill — would keep its PRIOR values at this timestep while
                # embeddings/scales are overwritten: stale metadata describing data that no longer
                # exists. Reset those slices to fill so the timestep is internally consistent.
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
                # Newly created vars and the untouched-reset must be committed before the Phase 2
                # fork, which opens a fresh session.
                if missing or untouched:
                    parts = ([f"add {missing}"] if missing else []) + (
                        [f"reset untouched {untouched}"] if untouched else []
                    )
                    traced_commit(session, f"Run {run_id}: overwrite {time_date} — {'; '.join(parts)}")
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
                traced_commit(
                    session,
                    f"Run {run_id}: extend time axis to {time_date}"
                    + (f" (adding variables {missing})" if missing else ""),
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
        # On a same-date overwrite, clear EVERY destination chunk this run does not write, not
        # just the ROI-live ones it skip-marked: a prior run under a LARGER or shifted ROI may have
        # written real data outside THIS run's ROI, which would survive as stale published
        # embeddings. Clearing the full non-live footprint (scalar fill, elided for already-empty
        # ocean chunks) makes the timestep exactly this run's ROI however the ROI changed.
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
        # No payloads = an all-skipped run with nothing to write or clear; Phase 1's
        # schema/timestep already stands, and run_forked would spawn a zero-worker pool.
        fill: dict[str, Any] = {"workers": [], "wall_s": 0.0, "merge_s": 0.0}
        if payloads:
            # `_log` is the caller's logger (the flow's run logger under Prefect), so the
            # coordinator's progress lines reach the orchestrator too. No catch-up on this path,
            # so no timer and nothing that can wedge; the session comes back unchanged.
            fill, _ = run_forked(session, _fill_band_worker, payloads, unit="band writes", log=_log)

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
            # Raw writes never clobber attrs, so prior windows are read straight off the store
            # and merged.
            windows: dict[str, Any] = dict(node.attrs.get("time_windows", {}))  # type: ignore[arg-type]
            windows[time_window.window_end_label] = {
                "range": [
                    f"{time_window.window_start[0]}-{time_window.window_start[1]:02d}",
                    f"{time_window.window_end[0]}-{time_window.window_end[1]:02d}",
                ],
            }
            attrs["time_windows"] = windows
            attrs["time_convention"] = "12mo_window_end"
        # Drop retired tessera:* attrs before writing geoemb:. `update` only overwrites the keys
        # it carries, so appending to a store created before the geoemb switch would leave the old
        # keys behind and advertise both conventions. zarr_conventions is replaced wholesale.
        for stale in [k for k in node.attrs if str(k).startswith("tessera:")]:
            del node.attrs[stale]
        node.attrs.update(attrs)

        t_commit = time.monotonic()
        commit_with_rebase(session, f"Run {run_id}: {len(chunks)} chunks assembled")
        commit_s = round(time.monotonic() - t_commit, 3)
        # Atomic publish: fast-forward `main` to the fully-written tip. The guard fails loudly if
        # another writer advanced `main` since we branched (two processes assembling the same ROI
        # store — unsupported). Then drop the work ref; `main` retains the snapshot.
        repo.reset_branch("main", repo.lookup_branch(work_branch), from_snapshot_id=base_snapshot)
        repo.delete_branch(work_branch)
        _log.info(
            "%s",
            _assembly_summary_line(
                run=run_id,
                tiles_staged=len(live_chunks),
                tiles_cleared=len(clear_chunks),
                workers_requested=n_workers,
                workers_used=len(payloads),
                per_worker_s3_cap=per_worker_cap,
                fill_wall_s=fill["wall_s"],
                merge_s=fill["merge_s"],
                commit_s=commit_s,
                total_s=round(time.monotonic() - t0, 3),
                fused_compress_put=True,
                workers=fill["workers"],
                **_sum_worker_stats(fill["workers"]),
            ),
        )
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
        staged_labels: Iterable[str] | None = None,
        skipped_labels: Iterable[str] | None = None,
        s3_concurrency: int | None = None,
        radar_coverage: dict | None = None,
        empty: bool = False,
        get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
        s3_region: str | None = None,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
        fault: ArmedFault | None = None,
        input_coverage: dict | None = None,
        # Where the PER-TILE refusal detail is persisted, from `BucketPaths.refusal_detail`. None
        # writes no sidecar and changes nothing the store commits — see `_skip_summary`.
        registry_root: str | None = None,
        optical_min_obs: int | None = None,
        embedded_records: Mapping[str, dict] | None = None,
    ) -> str:
        """Assemble a run's staged tiles into one (zone, year) of the global store.

        The global write path (ADR-008 D3/D6): every staged tile is exactly one output shard,
        written whole and lean by
        :func:`~tessera_embeddings.storage.shard_writer.write_year_shards` — fork/merge across
        ``n_workers`` processes, one commit per (zone, year), with ``years_complete`` and per-year
        run provenance updated in the same commit. The zone group must already be seeded
        (:func:`~tessera_embeddings.storage.global_store.seed_zone_groups`); nothing is created or
        resized here (D1). Emits one ``ASSEMBLY_SUMMARY`` record (:func:`_assembly_summary_line`)
        with the fill's per-worker phase timings and counts.

        The caller (the zone-fill runner) owns staged-completeness verification against the
        campaign land mask (:meth:`verify_staged_completeness`) — pass its returned label set as
        ``staged_labels`` so the verified inventory is by construction the assembled inventory and
        the staging prefix is not re-LISTed; when omitted, the prefix is listed here. The variable
        set is probed from one tile, since a staged run is homogeneous (one code version stages one
        variable set), and per-tile shape/variable validation happens as tiles are read.
        ``embedding_std`` is never staged under v1.1.

        Refill caveat: only staged tiles are written. A deliberate refill of a landed year — an
        exceptional manual operation, since the zone-year tag is write-once — whose new run skips
        or drops tiles the previous run staged leaves the prior data in those shards. A refill must
        re-stage every previously-live tile, or the operator must clear the year first.

        Args:
            store_path: URI of the global Icechunk repo (``BucketPaths.global_store()``).
            zone: Zone group name — UTM common name, e.g. ``"01N"``/``"60S"``.
            year: Campaign calendar year to fill; must be on the group's pre-allocated time axis.
            run_id: Run identifier (locates staged files).
            registry_root: Root of the published registry dataset
                (:meth:`BucketPaths.optical_registry`). One Parquet part per cell lands under it,
                a row per live tile, because the year's summary in the store is pooled and cannot
                say WHICH tile came closest to the depth cutoff — the question a cleanup campaign
                asks. ``None`` writes nothing and changes nothing the store commits.
            optical_min_obs: The depth rule this cell was filled under, stamped on every registry
                row. The registry's ``obs_max``/``median_obs_where_any`` are distances from this
                line and unreadable without it. The store's root is the authority, and the runner
                asserts the config matches it before any of this runs.
            embedded_records: Per-shard coverage for shards that DID embed something, keyed by
                label, as the actors reported it. Merged with the refused shards' marker records so
                a partly-refused tile reports what the depth gate removed; a label in both takes
                the marker, written at the end of a wholly refused shard.
            n_workers: Worker process count; also divides ``TARGET_AGGREGATE_S3_CONCURRENCY`` into
                the per-fork cap.
            staged_labels: Pre-listed staged tile labels (e.g. the return of
                :meth:`verify_staged_completeness`); ``None`` lists the prefix.
            radar_coverage: This YEAR's radar-coverage summary, from
                :func:`summarise_radar_coverage` over the run's chunk results, recorded on the
                year's ``runs`` entry. Per year rather than per zone because coverage is a property
                of what was acquired: one year of a zone can be radar-free where another is not.
            skipped_labels: Live tiles this run resolved to a SKIP, written as fill so the
                published year is exactly this run's output — see
                :meth:`StagedShardSource.live_shards` for the mixed-year hazard leaving them
                untouched creates. Also recorded with ``staged_labels`` as the year's
                ``optical_skips`` summary (:func:`summarise_optical_skips`), so it must be the
                staging prefix's resolution of the live set, never the finishing leg's own tally —
                a resumed run's markers were written by earlier legs. An empty sequence MEANS
                "resolved, none skipped" and records a zero summary; ``None`` means the caller
                resolved no live set, writes only staged tiles and records no summary, because a
                zero it did not establish would read as measured. When EVERY live tile skipped this
                is the whole footprint, ``staged_labels`` is empty, and ``empty=True`` goes with it.
            empty: Record the year as holding no data. For the all-skipped case, where the fill
                write and the completion mark must agree the year is empty — marking it without
                the write leaves a previous attempt's shards readable underneath.
            s3_concurrency: This fill's slice of the fleet S3-PUT budget (divided
                across ``n_workers`` for the per-fork cap); ``None`` uses the full
                ``TARGET_AGGREGATE_S3_CONCURRENCY`` (a lone fill).
            get_credentials: Optional icechunk credential callback.
            s3_region: Optional S3 region override.
            log: Optional logger.
            input_coverage: How much of the requested window the INPUT mosaics actually
                held, from the fill's preflight, recorded on the year by
                :func:`~tessera_embeddings.storage.shard_writer.run_provenance`. The
                mosaics are deleted once a cell lands, so this is the only lasting record.
            fault: Supervised-drill hook, forwarded to
                :func:`~tessera_embeddings.storage.shard_writer.write_year_shards` for
                the gap between its two commits. Inert unless the run was armed for
                that fault and this cell
                (:mod:`tessera_embeddings.config.fault_injection`).

        Returns:
            The commit snapshot id.
        """
        _log = log or logger
        t0 = time.monotonic()
        workers_requested = n_workers
        # THIS ASSEMBLY'S records only. `_skip_summary` populates them during the shard write and
        # `publish_registry_part` reads them afterwards, but `skipped_labels=None` is a supported
        # default that skips the summary entirely — so a reused writer would carry the PREVIOUS
        # cell's refusal measurements into these registry rows. Chunk labels are grid-local
        # (`chunk_<row>_<col>`) and repeat across every zone and year, so stale entries match by
        # name and attach one cell's refusals to another's tiles, silently and with plausible
        # numbers. Reset here rather than in the summary, because the bug is the summary NOT
        # running.
        self._last_skip_records = {}

        labels = sorted(staged_labels) if staged_labels is not None else self._list_staged_labels(run_id)
        # Zero staged tiles is legitimate in exactly one case: every live tile of the year
        # resolved to a skip and the caller is clearing the footprint before marking the year
        # empty. With no skipped tiles either there is nothing to do, and an empty staged prefix is
        # the corruption this exists to catch.
        if not labels and not skipped_labels:
            raise IncompleteStageError(f"Run {run_id!r} has no staged chunks under {self.staging_base}")
        shards = tuple(sorted(parse_chunk_label(label) for label in labels))

        # S3 budget: divide the FLEET target across concurrent fills, not just this fill's forks.
        # TARGET_AGGREGATE_S3_CONCURRENCY // n_workers alone bounds ONE fill to ~target, so K
        # concurrent fills burst K times the target PUTs (the 800-req SlowDown). The campaign
        # passes `s3_concurrency = target // max_parallel_zones`; None = full target.
        #
        # The budget sets the per-fork REQUEST CAP only and never reduces the fork count. Since
        # that cap floors at 1, the fleet may exceed the target by up to
        # `max_workers * n_clusters`; see `_s3_budget_split`.
        n_workers, per_worker_cap = _s3_budget_split(s3_concurrency, n_workers)
        repo = open_global_repo(
            store_path,
            get_credentials=get_credentials,
            region=s3_region,
            max_concurrent_requests=per_worker_cap,
            # `write_year_shards` PICKLES this session to spawned children; without this each
            # deserialises with no credential and calls back per S3 request for the life of the
            # fork (icechunk#2077). Opt-in per call site because the pickle carries a live secret —
            # safe across a local spawn pipe, not over a network transport. True unconditionally,
            # since `_create_storage` substitutes a default provider when `get_credentials` is None
            # and omits the option when there is no provider at all.
            scatter_initial_credentials=True,
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
        # Destination band width is the authority for the staged-tile band check, NOT
        # ZarrWriter.embedding_dim, which defaults to 128 and need not match a zone seeded wider.
        dest_band = int(cast(zarr.Array, node["embeddings"]).shape[-1])

        missing_dst = [v for v in ("embeddings", "scales") if v not in node]
        if missing_dst:
            raise ValueError(
                f"Zone group {zone} lacks required array(s) {missing_dst} — a fill without them "
                "could not be dequantized; the group must be seeded with a full GLOBAL layout."
            )
        # Optional vars the destination CAN hold. StagedShardSource.load checks EVERY tile for one
        # of these that is absent from `variables` — present in only some tiles, so silently
        # dropped — rather than sampling first and last.
        dest_optional = tuple(v for v in CARRIED_VARS if v in node)

        if labels:
            # One probe of a staged tile: required vars, dtypes, variable set, and exact tile
            # pitch (a truncated half-tile must not pass). Every other tile is re-validated by
            # StagedShardSource.load as it is read.
            probe_path = f"{self.staging_base}/{run_id}/{labels[0]}.zarr"
            staged_group = zarr.open_group(probe_path, mode="r", storage_options=_staged_storage_options(probe_path))
            missing = [v for v in REQUIRED_VARS if v not in staged_group]
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
            variables = tuple(v for v in (*REQUIRED_VARS, *CARRIED_VARS) if v in staged_group and v in node)
            # Check every staged variable's dtype against its DESTINATION array, not just the
            # embeddings/scales asserted above: a raw-zarr shard write C-casts silently, so a
            # uniformly int64/uint32 observation count — which agrees with the probe, so
            # StagedShardSource's tile-vs-probe check passes — would be narrowed into a seeded
            # uint16 without this guard.
            for v in variables:
                if v in REQUIRED_VARS:
                    continue
                staged_dt = cast(zarr.Array, staged_group[v]).dtype
                dest_dt = cast(zarr.Array, node[v]).dtype
                if staged_dt != dest_dt:
                    raise IncompleteStageError(
                        f"Staged tile {probe_path} has {v} dtype {staged_dt} but {zone}/{v} is seeded "
                        f"{dest_dt} — a raw-zarr write would silently C-cast; re-stage at the seeded dtype."
                    )
            probe_dtypes = tuple((v, str(cast(zarr.Array, staged_group[v]).dtype)) for v in variables)
        else:
            # Nothing staged, so there is no tile to probe and the write is fill only. The
            # DESTINATION answers both questions the probe usually does: which variables to write
            # (every one the zone holds, since any could carry a previous attempt's data) and at
            # what dtype (the seeded one, which a fill cannot disagree with).
            variables = (*REQUIRED_VARS, *dest_optional)
            probe_dtypes = tuple((v, str(cast(zarr.Array, node[v]).dtype)) for v in variables)

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
        # Materialised once so the clearing writes and the year's provenance summary can never
        # describe different tile sets.
        skipped = sorted(skipped_labels or ())
        # Fill values come off the SEEDED arrays, so a cleared tile reads back exactly as an
        # unwritten one does (0 for int8 embeddings, NaN for float scales).
        cleared = tuple(sorted(parse_chunk_label(label) for label in skipped))
        fill_values = tuple((v, float(cast(zarr.Array, node[v]).fill_value or 0)) for v in variables)
        if cleared:
            _log.info(
                "Clearing %d skipped tile(s) to fill: this run staged nothing for them, and an "
                "earlier unmarked attempt at this year may have.",
                len(cleared),
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
            cleared=cleared,
            fill_values=fill_values,
        )
        telemetry: dict[str, Any] = {}
        # Captured before the write, so the registry part reports what THIS call published rather
        # than whatever a later listing happens to see.
        registry_embedded, registry_refused = list(labels), list(skipped)
        snapshot = write_year_shards(
            repo,
            zone,
            year_index,
            source,
            n_workers=n_workers,
            shard_px=shard_px,
            commit_msg=f"Run {run_id}: fill {zone} year {year}",
            run_id=run_id,
            radar_coverage=radar_coverage,
            # Derived from what THIS call publishes as fill, so record and write agree by
            # construction; run_provenance drops it on an empty year. `skipped_labels=None` is a
            # caller that resolved no live set at all, so there is nothing to summarise and no
            # ZERO to assert.
            optical_skips=(self._skip_summary(run_id, labels, skipped) if skipped_labels is not None else None),
            empty=empty,
            telemetry=telemetry,
            # Coordinator progress goes through the caller's logger: under Prefect that is the
            # flow's run logger, the only route to the Prefect API. The module logger reaches only
            # the process log stream.
            log=_log,
            fault=fault,
            input_coverage=input_coverage,
        )
        # THE CELL IS COMMITTED. Only now may the registry say so: nothing reconciles it against
        # the store, and consumers are told they need not open the store, so a part written any
        # earlier could advertise a cell that never landed.
        if registry_root:
            self.publish_registry_part(
                registry_root,
                zone,
                year,
                run_id,
                embedded=registry_embedded,
                refused=registry_refused,
                optical_min_obs=optical_min_obs,
                embedded_records=embedded_records,
                # The store's own callback: the registry is its sibling in the same bucket, so the
                # credential that opened the store is the one that can write beside it.
                get_credentials=get_credentials,
                s3_region=s3_region,
            )
        workers = telemetry.get("workers", [])
        _log.info(
            "%s",
            _assembly_summary_line(
                run=run_id,
                zone=zone,
                year=year,
                tiles_staged=len(shards),
                tiles_cleared=len(cleared),
                workers_requested=workers_requested,
                workers_used=len(workers),
                per_worker_s3_cap=per_worker_cap,
                fill_wall_s=telemetry.get("fill_wall_s"),
                merge_s=telemetry.get("merge_s"),
                commit_s=telemetry.get("commit_s"),
                attrs_commit_s=telemetry.get("attrs_commit_s"),
                # The catch-up tally. This record is the ONLY place it reaches an operator, and a
                # healthy commit looks identical whether the session was kept current or merely got
                # lucky — without it the mechanism has no evidence it ran at all.
                catch_ups=telemetry.get("catch_ups"),
                total_s=round(time.monotonic() - t0, 3),
                fused_compress_put=True,
                workers=workers,
                **_sum_worker_stats(workers),
            ),
        )
        _log.info("Global assembly complete: %s/%s year %d (snapshot %s)", store_path, zone, year, snapshot)
        return snapshot

    def cleanup_staging(
        self,
        run_id: str,
        log: logging.Logger | logging.LoggerAdapter[logging.Logger] | None = None,
    ) -> None:
        """Delete the staging directory for a completed run.

        S3 paths are removed with ``s5cmd rm``, falling back to fsspec's ``rm`` if s5cmd is
        unavailable or fails. Non-S3 paths use fsspec.

        Args:
            run_id: Run identifier whose staging artifacts should be deleted.
            log: Optional logger. Falls back to the module logger.
        """
        _log = log or logger
        target = f"{self.staging_base}/{run_id}"
        _log.info("Cleaning up staging: %s", target)
        # Shared prefix delete: s5cmd --all-versions, so a versioned bucket does not keep the
        # staged tiles as non-current versions; fsspec fallback.
        delete_prefix(target, log=_log)


#: The refusal reasons a skip record may carry, in the order a reader should weigh them: a fact
#: about the imagery, then this campaign's quality rule, then a coverage fact. An explicit tuple
#: because the summary reports each one's total, and a missing key must read as zero rather than
#: vanish from the record.
REFUSAL_REASONS: tuple[str, ...] = ("no_optical", "thin", "no_radar")


def read_skip_records(
    staging_base: str, run_id: str, labels: Iterable[str], *, workers: int = 32
) -> tuple[dict[str, dict], int]:
    """``({label: record}, n_unreadable)`` for every skip marker that carries one.

    Concurrent, because these are independent one-object GETs and a zone can refuse hundreds — 40S
    refused 43 of 58, and the largest zones hold 556 live tiles — so serial reads would add a round
    trip each to the critical path between the last chunk and the commit. The filesystem is
    resolved once rather than per label for the same reason.

    **The failure count is RETURNED, not swallowed.** A marker that is empty, unreadable or not
    JSON yields no entry, which is right for the ordinary cases (a marker predating the registry is
    zero bytes, and a resume across that change must still assemble). But "no records" and "every
    read failed" are the same empty dict and mean opposite things: a systematic failure — expired
    credentials, a wrong prefix — would otherwise publish a provenance entry quietly saying nothing
    was recorded. The caller reports the count.
    """
    labels = list(labels)
    if not labels:
        return {}, 0
    base = f"{staging_base.rstrip('/')}/{run_id}"
    try:
        fs = _fs_for(base)
    except Exception:
        # RESOLVING the filesystem can fail on its own — bad credentials, a malformed URI — and
        # unguarded that raises out of the year's provenance construction, failing a cell at
        # ASSEMBLY after all its inference is paid for. Same asymmetry as the marker itself: these
        # reasons are a diagnostic, and losing them must never cost the cell that earned them.
        # Reported as every marker unreadable, which the caller surfaces loudly, rather than as no
        # reasons recorded.
        logger.exception(
            "Run %s: could not open the staging filesystem to read %d skip marker(s); their refusal "
            "reasons will be absent from the year's record",
            run_id,
            len(labels),
        )
        return {}, len(labels)

    def one(label: str) -> tuple[str, dict | None, bool]:
        """``(label, record, unreadable)``. Absent and empty are ordinary; a read error is not."""
        try:
            with fs.open(f"{base}/{label}.skipped", "rb") as f:
                raw = f.read()
        except FileNotFoundError:
            return label, None, False
        except OSError:
            return label, None, True
        if not raw.strip():
            return label, None, False
        try:
            return label, json.loads(raw), False
        except (json.JSONDecodeError, UnicodeDecodeError):
            # UnicodeDecodeError is NOT a JSONDecodeError: a partially written or byte-corrupt
            # marker raises it from `json.loads`'s decode step, and letting it escape `pool.map`
            # aborts the whole assembly on one bad object. Provenance is fail-soft by design.
            return label, None, True

    out: dict[str, dict] = {}
    unreadable = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(labels))) as pool:
        for label, record, bad in pool.map(one, labels):
            if bad:
                unreadable += 1
            elif isinstance(record, dict):
                out[label] = record
    return out, unreadable


def summarise_optical_skips(
    *, staged: Iterable[str], skipped: Iterable[str], records: dict[str, dict] | None = None
) -> dict:
    """One year's optical-skip summary: which live tiles published as fill, and how many.

    A skipped tile — every pixel failed the validity filter, nothing staged — is written as fill,
    which is also what ocean reads as, so without this record a consumer of a completed year cannot
    tell "no valid optical data" from "not land". The count says how much was lost, the labels say
    WHERE so a consumer can mask the area, and the live total is the denominator that makes the
    count interpretable without fetching the land mask.

    Both inputs must be the STAGING PREFIX's resolution of the run's live tiles — the two halves
    the staged-completeness scan establishes, staged zarrs and skip markers — never one leg's own
    tally. Skip markers persist across resumes, so the leg that finishes a run may have staged
    nothing for tiles an earlier leg skipped and would report them as resumed successes: a summary
    built from that leg's results records zero skips while publishing fill over them.

    The label list needs no size cap: the one case where it would span a whole zone, every live
    tile skipped, is recorded by the ``empty`` flag instead, and
    :func:`~tessera_embeddings.storage.shard_writer.run_provenance` drops this summary from an
    ``empty`` year's record.
    """
    skipped_list = sorted(skipped)
    summary = {
        "tiles_skipped": len(skipped_list),
        "tiles_live": sum(1 for _ in staged) + len(skipped_list),
        "labels": skipped_list,
    }
    if records is None:
        return summary
    # THE PER-SHARD REGISTRY, folded into the year's provenance.
    #
    # ORGANISED BY REASON rather than by shard: it answers what a reader asks ("why is this land
    # empty, and which parts") without walking a dict, it is self-describing since every key is a
    # reason name, and it is an order of magnitude smaller — a nested record per shard puts
    # ~200 bytes x hundreds of shards into a zarr ATTRIBUTE that every reader of the zone group
    # pays for on every open, and the largest zones hold 556 live tiles.
    #
    # UNITS ARE IN THE NAMES: `tiles_skipped` counts TILES, the refusal totals count PIXELS.
    per_reason: dict[str, list[str]] = {}
    refused_px = dict.fromkeys(REFUSAL_REASONS, 0)
    obs_max = 0
    obs_px_with_any = 0
    mixed = 0
    inconsistent: list[str] = []
    # Pixels inside a published tile the dataset never evaluated, because the read plan cropped the
    # chunk to the columns holding valid pixels. They are filled like refused pixels but are NOT
    # refusals, so folding them into a reason misattributes them, while omitting them makes the
    # reason totals look short of the footprint they explain.
    not_evaluated_px = 0
    # Whether the radar rule was in force, pooled over the records. Otherwise `no_radar: 0` means
    # two things at once — no tile refused for missing radar, or the rule switched off.
    # `allow_s2_only` defaults to FALSE in the library and the global campaign registers True, so
    # under campaign settings that zero is structural, not a finding about radar coverage.
    radar_rule: set[bool] = set()
    for label in skipped_list:
        record = records.get(label)
        if record is None:
            continue
        refused = {r: int((record.get("refused") or {}).get(r) or 0) for r in REFUSAL_REASONS}
        for reason, n in refused.items():
            refused_px[reason] += n
        total = sum(refused.values())
        if not total:
            # A record refusing nothing cannot explain a shard holding nothing. Named rather than
            # dropped: the producer and this summary disagree, a defect in one of them, and it must
            # not read as a shard with no reason recorded.
            inconsistent.append(label)
            continue
        # THE INVARIANT: a fully refused shard refused every pixel the dataset EVALUATED. The
        # three reasons partition those pixels by construction, so a mismatch means the strips did
        # not cover what was loaded or a count was double-added — either way the pixel totals below
        # are wrong, and saying so beats a plausible number.
        #
        # Against the EVALUATED footprint, not the tile: the read plan crops a chunk to the columns
        # holding valid pixels, so comparing cropped counts against the whole tile flags every
        # cropped shard as a defect. Those columns are counted separately below as unevaluated — a
        # third thing from refused and from embedded.
        eligible = int(record.get("eligible_px") or 0)
        if eligible and total != eligible:
            inconsistent.append(label)
        # A record predating `chunk_px` has `eligible_px` equal to the whole tile by construction,
        # so the difference is zero rather than unknown.
        not_evaluated_px += max(int(record.get("chunk_px") or eligible) - eligible, 0)
        if isinstance(record.get("radar_rule_enforced"), bool):
            radar_rule.add(bool(record["radar_rule_enforced"]))
        if sum(1 for n in refused.values() if n) > 1:
            mixed += 1
        dominant = max(refused, key=lambda r: refused[r])
        per_reason.setdefault(dominant, []).append(label)
        obs = record.get("s2_obs") or {}
        obs_max = max(obs_max, int(obs.get("max") or 0))
        obs_px_with_any += int(obs.get("px_with_any") or 0)

    summary["refused_px_by_reason"] = refused_px
    # A shard appears under the reason that refused the MOST of its pixels, so the lists partition
    # the shards and sum to the recorded total. `shards_mixed` counts those with more than one
    # reason, which a bare dominant label would hide.
    summary["shards_by_reason"] = {r: sorted(v) for r, v in sorted(per_reason.items())}
    summary["shards_mixed"] = mixed
    # HOW THIN, pooled over the refused shards: per-shard depth answers a question nobody has once
    # the mosaic is deleted, and it is what made the record large.
    summary["s2_obs_at_refused"] = {"max": obs_max, "px_with_any": obs_px_with_any}
    summary["unrecorded"] = [label for label in skipped_list if label not in records]
    # Emitted only when non-zero, so an ordinary uncropped year carries no field of zeros and the
    # field's presence is itself the signal that tiles were cropped. With the per-reason totals it
    # accounts for the whole filled footprint: refused plus never evaluated.
    if not_evaluated_px:
        summary["not_evaluated_px"] = not_evaluated_px
    # "enforced" / "disabled" / "mixed" — never a bare zero standing in for either. Absent when no
    # record says, which is honestly unknown rather than assumed.
    if radar_rule:
        summary["radar_refusal_rule"] = (
            "mixed" if len(radar_rule) > 1 else ("enforced" if next(iter(radar_rule)) else "disabled")
        )
    if inconsistent:
        summary["inconsistent"] = sorted(inconsistent)
    return summary


def summarise_radar_coverage(results: Iterable[dict]) -> dict | None:
    """Aggregate per-chunk radar counts into one year's coverage summary.

    Returns ``None`` when ANY embedded tile reported no counts, not only when none did: a resumed
    tile is a synthetic success with no counters, and dropping it from both sides of the ratio
    leaves a figure describing only the tiles this run redid — whatever the previous attempt failed
    to finish, which says nothing about the year it would be stored as. ``None`` likewise when no
    chunk reported counts, so a run from an older build records no summary rather than a wrong one
    of zeros.

    Reduces what the actors already reported: the counts are computed where the observation maps
    and the embedded mask are both in memory, so nothing here reads pixels.

    Percentages are of EMBEDDED area, not of the zone. A zone is mostly ocean and mostly
    unembedded, so a fraction of the grid would be dominated by area no radar was ever expected
    over.
    """
    embedded = free = light = s2_thin = 0
    reported = silent = fully_free = 0
    thin_below: int | None = None
    for r in results:
        if r.get("status") != "success":
            continue
        if "s1_free_pixels" not in r:
            silent += 1
            continue
        reported += 1
        embedded += int(r.get("valid_pixels", 0))
        free += int(r["s1_free_pixels"])
        light += int(r.get("s1_thin_pixels", 0))
        s2_thin += int(r.get("s2_thin_pixels", 0))
        thin_below = thin_below or r.get("s2_thin_below_obs")
        # In THIS loop: `results` is an Iterable, and a generator is exhausted by the time any
        # second pass runs, which would silently report zero fully-free tiles rather than fail.
        if r["s1_free_pixels"] and r["s1_free_pixels"] == r.get("valid_pixels"):
            fully_free += 1
    if silent:
        # A RESUMED tile is a synthetic success carrying no counters. Dropping it from both sides
        # of the ratio still yields a figure, computed over whatever this run happened to redo,
        # which would then be written on the year's provenance as the year's — wrong by an
        # unknowable margin with nothing saying so. No summary is the honest answer; the per-pixel
        # observation counts in the store remain the authority.
        logger.info(
            "No radar-coverage summary for this year: %d of %d embedded tile(s) reported no counts "
            "(a resume reuses tiles staged by an earlier attempt, which did not report them). "
            "The per-pixel s1_*_obs_count arrays in the store are unaffected.",
            silent,
            silent + reported,
        )
        return None
    if not reported or embedded == 0:
        return None
    return {
        "embedded_px": embedded,
        "s1_free_px": free,
        "s1_free_pct": round(100.0 * free / embedded, 3),
        "s1_thin_px": light,
        "s1_thin_pct": round(100.0 * light / embedded, 3),
        "s1_thin_below_obs": RADAR_THIN_MAX_OBS,
        # The optical counterpart, on the same denominator so the two are comparable. There is no
        # optical-FREE figure: a tile with no valid optical pixel is a skip and never produces a
        # result, so `optical_skips` is where that case lives.
        "s2_thin_px": s2_thin,
        "s2_thin_pct": round(100.0 * s2_thin / embedded, 3),
        # Reported by the actors, because the fill applies the STORE's rule and not the module
        # default: stamping the constant here would label the year with a threshold no pixel was
        # judged against. Captured in the loop above because `results` is consumed once.
        "s2_thin_below_obs": thin_below or OPTICAL_MIN_OBS,
        "chunks_reporting": reported,
        # Where, coarsely: an ENTIRELY radar-free tile localises the gap without a per-tile grid,
        # separating a concentrated absence (whole tiles, e.g. an ice margin) from a diffuse one (a
        # swath edge crossing many tiles). Exact locations are in the per-pixel obs-count arrays.
        "tiles_fully_s1_free": fully_free,
        "tiles_reporting": reported,
    }
