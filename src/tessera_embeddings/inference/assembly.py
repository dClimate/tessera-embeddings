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

One inference tile is one 2048-px output shard on both paths (ADR-008 D3), so
nothing rechunks. Two write strategies:

* **Single-ROI** (``assemble``): workers partition the mosaic into *northing
  bands aligned to the output write granularity*, so no two forks ever touch the
  same output object and each tile is read by exactly one band. A mosaic whose
  extent is not a whole number of shards has ragged edge tiles; their partial
  chunks are read-modify-written sequentially inside one fork (icechunk sessions
  are read-your-writes).
* **Global** (``assemble_global``): the zone grid is seeded to whole shards, so
  there are no ragged edges and no banding is needed — whole tiles round-robin
  across workers via
  :func:`~tessera_embeddings.storage.shard_writer.write_year_shards`, and every
  shard object is emitted once, lean, with ocean inner chunks elided.

Both paths emit one machine-readable ``ASSEMBLY_SUMMARY`` log record per
assembly — per-worker read/write phase timings with a CPU/wall split, worker
counts, and byte/object counts — so a slow assembly can be attributed to staged
reads, to compression, or to the object store without re-running it. See
:func:`_assembly_summary_line` for the fields and their exact claims.
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
from tessera_embeddings.inference.conventions import build_convention_attrs
from tessera_embeddings.storage import zone_grid
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


STAGED_READ_CONFIG_KWARGS = {"retries": {"max_attempts": 10, "mode": "adaptive"}}
#: Suffix of the coverage-only tile a refused chunk stages. Named once because two places
#: depend on it agreeing: the writer that creates it, and the listing that must NOT read it as
#: a chunk label.
_COVERAGE_SUFFIX = ".coverage.zarr"
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
    """Return s3fs storage options for a staged read or write, or ``None`` off S3.

    The retry config is a botocore client setting and only applies to the S3
    backend; local staging paths (``/tmp/...``) get ``None`` and open normally.

    Deliberately carries no credentials and no region. Unlike the ROI mask — a plain zarr
    the Icechunk callback never reaches — staging is written by an inference actor on an
    instance profile, and fsspec resolves AND REFRESHES that itself. The callback exists
    because Icechunk's Rust client does not refresh, which is a problem staging does not
    have. A non-default-region STAGING bucket would need a region here; the deployment
    keeps staging under ``paths.outputs`` in the store's own region, so nothing threads one
    today and inventing the parameter would leave four call sites passing None.
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

    The single in-store completeness check. ``write_chunk`` sets
    ``staged_complete`` only after ``to_zarr`` returns, so its absence means a
    crash left array metadata over missing data chunks — which Zarr reads back as
    fill values rather than as an error, the one failure that would otherwise
    reach the published store looking like legitimate data.

    The listing gate (``.done`` markers, see :meth:`ZarrWriter._done_marker_path`)
    normally excludes such a tile long before any reader sees it, and this check is
    free for a reader that has to open the group anyway. It earns its place on one
    case the listing cannot cover: a listing is taken once, in the driver, while
    tiles are read minutes to hours later in worker processes. A tile REWRITTEN in
    that window keeps its old ``.done`` marker throughout — the run identifier is
    derived from the inputs and so is stable across attempts, which means two
    attempts at one zone-year share a staging prefix — and only the in-store
    attribute, absent until the rewrite finishes, reports the tile as it is now
    rather than as the listing last saw it.

    Raises:
        IncompleteStageError: If the tile's write never completed.
        FileNotFoundError: If there is no tile at *path* (also raised by zarr as
            ``GroupNotFoundError``, a subclass). Any other open error — auth,
            throttling, transient network — propagates untouched, so a valid tile
            is never mistaken for a partial one because of a bad moment on S3.
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

    Distinct from ``FileNotFoundError`` (nothing there at all): the run IS real, it
    just has nothing to measure a chunk size from. Callers resuming such a run should
    fall back to their configured chunk size — ``assemble`` publishes an all-fill
    timestep for this case.
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
``per_worker_cap`` floors at 1, a fill's aggregate is ``max(budget, n_workers)``
rather than ``<= budget``: the floor and the ceiling cannot both hold, and the
ceiling is the one that gives way. ``AssemblyConfig.max_workers`` bounds the
overshoot -- see :func:`_s3_budget_split` for why that is the right way round.
"""


def _s3_budget_split(s3_concurrency: int | None, n_workers: int) -> tuple[int, int]:
    """``(effective_workers, per_worker_cap)`` honoring the fleet S3-PUT budget.

    Each fork worker opens its own repo capped at ``per_worker_cap``, so a fill
    issues up to ``effective_workers * per_worker_cap`` concurrent requests.
    ``s3_concurrency=None`` uses the full aggregate target (a lone fill). Both
    returned values are ``>= 1``.

    **The worker count is NOT reduced to fit the budget**, and that is the whole
    point of this function. A per-fill budget is the fleet target divided by the
    cluster count, so on a wide campaign it lands well below the requested worker
    count — and clamping to it silently costs most of the fork pool on every
    assembly, which is the campaign's longest stage.

    Because ``per_worker_cap`` floors at 1, a worker count above the budget makes
    the aggregate ``max(budget, n_workers)``. The floor and the ceiling cannot
    both hold; the ceiling gives way, because the costs are asymmetric.
    Overshooting the target risks 503s, which retry, and the target sits far below
    the concurrency at which SlowDown was actually observed. Holding the target by
    dropping forks costs wall-clock unconditionally, on every cell, whether or not
    the service was ever going to complain. The overshoot is bounded by
    ``AssemblyConfig.max_workers`` times the fleet's cluster count.

    The measured cost of the clamp, and the concurrency evidence:
    ``context_docs/design/assembly-worker-clamp-2026_08.md``.
    """
    budget = s3_concurrency if s3_concurrency is not None else TARGET_AGGREGATE_S3_CONCURRENCY
    workers = max(1, n_workers)
    return workers, max(1, budget // workers)


def _assembly_summary_line(**fields: Any) -> str:  # noqa: ANN401 — heterogeneous JSON payload
    """One machine-readable per-assembly record for the profiling tools.

    The assembly-phase counterpart of ``actors._chunk_summary_line``: a single
    ``ASSEMBLY_SUMMARY: {json}`` line per assembly (never per tile — at zone
    scale a per-tile line would be thousands of lines and would perturb the very
    I/O being measured), parsed by the same prefix-plus-JSON convention. Keep the
    keys stable — or update the parsers in the same change.

    What the fields mean, and what they deliberately do not claim:

    * ``read_s``/``read_cpu_s`` — staged-tile fetches, summed across workers.
      ``write_s``/``write_cpu_s`` — the raw-zarr region assignments, summed.
      Compression and upload are FUSED inside the write call (zarr encodes and
      icechunk uploads within one assignment this code cannot see into), so
      there is no separate compress or upload timer; the honest decomposition is
      the CPU/wall split per phase — ``*_cpu_s`` bounds the in-process compute
      (for writes, that is the compression), ``*_s - *_cpu_s`` is time blocked
      on the object store. ``fused_compress_put`` states this in-band so a
      reader of the record alone cannot mistake ``write_s`` for upload time.
    * ``workers`` — the per-worker stats dicts in payload order (band order for
      the single-ROI engine, round-robin partition order for the global one).
      Per worker, ``wall_s - read_s - write_s`` is time outside both phases
      (validation, partitioning, interpreter start), and ``wall_s`` of the
      slowest worker bounds the fill: every payload gets its own process, so
      the fill is as long as its slowest worker, not the sum.
    * ``workers_requested`` vs ``workers_used`` — what the caller asked for vs
      how many forks actually ran; the S3 budget and the work partition can both
      cap the count, and a fill quietly running below its requested width is
      exactly what this pair exposes.
    * ``per_worker_s3_cap`` — each fork's concurrent-request cap
      (``budget // workers``, see :data:`TARGET_AGGREGATE_S3_CONCURRENCY`).
    * ``tiles``/``writes``/``bytes`` — tile loads, region assignments, and
      uncompressed bytes handed to zarr, summed across workers; rates derive
      from these without a store listing. ``tiles_staged``/``tiles_cleared``
      are the caller's intent (real data vs fill-over-skip footprints).
    * ``fill_wall_s``/``merge_s`` — the fork-to-merge span and the merge alone;
      ``commit_s``/``attrs_commit_s`` — the data and attrs commits;
      ``total_s`` — the whole assembly call.
    """
    return "ASSEMBLY_SUMMARY: " + json.dumps(fields, sort_keys=True)


#: Worker-stats keys summed into the ASSEMBLY_SUMMARY totals. ``wall_s``/``cpu_s``
#: are summed as ``worker_wall_s``/``worker_cpu_s`` — a SUM of per-worker walls,
#: which measures aggregate occupancy, not the fill's duration (that is
#: ``fill_wall_s``; the two differ by exactly the parallelism achieved).
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


def _layout_matching_store(root: zarr.Group, layout: StoreLayout, variables: Iterable[str]) -> StoreLayout:
    """*layout*, with each missing variable's geometry taken from the store it joins.

    A variable added to an EXISTING store must adopt that store's chunk and shard
    geometry, not the current preset's. Every data variable has to agree on a write
    granularity — :func:`_write_granularity` raises otherwise, because two disagreeing
    arrays would let separate forks share an output object — so creating one array at a
    different pitch from its siblings does not produce a merely mixed store, it makes
    the store unassemblable. A store written before a preset changed still accepts
    appends, and gaining a variable must not be what breaks it.

    Geometry is copied from an existing array of the same rank (there always is one:
    ``embeddings`` and ``scales`` are mandatory, covering 4-D and 3-D). Only chunks and
    shards are taken; dtype, fill value and codec stay the layout's, since those are
    properties of the variable rather than of the store's tiling. Falls through to the
    layout unchanged for a rank nothing in the store shares.
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

    For each staged tile overlapping the band, reads the tile's overlapping
    y-slice (one slice in flight per worker — peak RAM is bounded by one tile,
    ~0.5 GB int8 at 2048 px) and writes it into every output array with a plain
    zarr assignment. Partial output chunks at tile x-boundaries are
    read-modify-written sequentially within this fork (icechunk sessions are
    read-your-writes), so the merged result is exact. ``payload["clear"]``
    tiles (every chunk this run does not write, on a same-date overwrite — see
    :meth:`ZarrWriter.assemble`) get the fill value written over their footprint
    so a prior run's data — under a rerun's skip marker OR outside a changed ROI —
    can't survive. Returns ``(fork, stats)``: the fork for the coordinator to
    merge, and this band's phase timings and counts — ``read`` covers the staged
    opens and slice fetches, ``write`` the zarr assignments (encode and upload
    fused, see ``_assembly_summary_line``; clear-to-fill assignments count in
    ``write`` too, since they emit output objects like any other write).
    """
    fork = payload["fork"]
    t = int(payload["time_index"])
    y0b, y1b = payload["band"]
    root = zarr.open_group(fork.store, mode="a")
    arrays = {var: cast(zarr.Array, root[var]) for var in payload["variables"]}
    timer = PhaseTimer()
    tiles = writes = nbytes = 0
    # Trailing dims (band) not indexed are written in full, so each assignment
    # below covers both the 3-D and 4-D arrays.
    for tile in payload["clear"]:
        y0, y1 = max(tile.y_start, y0b), min(tile.y_stop, y1b)
        for arr in arrays.values():
            # Scalar assignment: zarr broadcasts per-chunk without materializing
            # the selection (a full-band float32 block here would be ~2 GB).
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
            with timer.phase("read"):
                block = np.asarray(staged_arr[y0 - tile.y_start : y1 - tile.y_start])[np.newaxis]
            with timer.phase("write"):
                arr[t : t + 1, y0:y1, tile.x_start : tile.x_stop] = block
            writes += 1
            nbytes += block.nbytes
        # Drop the group reference so its file handles / S3 connections are
        # immediately collectable before the next tile's read.
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
class StagedListing:
    """What one LIST of a run's staging prefix says about each chunk's artifacts.

    The three states are mutually exclusive and each has exactly one remedy, which
    is what lets every caller agree on how to treat a resumed run:

    * ``complete`` — a staged ``.zarr`` and its ``.done`` marker both landed, so
      the write finished. **Skip it** (after :meth:`ZarrWriter._validate_staged_chunk`
      confirms its shape and dtype).
    * ``interrupted`` — one of the pair is missing, so a crash caught the write
      part-way and the ``.zarr``'s data chunks may be absent (Zarr reads those back
      as fill values, silently). **Re-infer it** — ``write_chunk``'s ``mode="w"``
      overwrites, so no cleanup is needed first.
    * ``skipped`` — a ``.skipped`` marker: a previous attempt found the chunk had no
      valid pixels at all. **Skip it**, but report it as a skip rather than a
      success (see :class:`StagedResume`).

    Labels are sorted: ``fs.ls`` order is backend-dependent and downstream probes
    take ``complete[0]``, so the probe tile must not depend on the filesystem.
    """

    complete: list[str]
    interrupted: list[str]
    skipped: list[str]


@dataclasses.dataclass(frozen=True)
class StagedResume:
    """What a resume scan found in staging, with the two artifact kinds kept apart.

    ``done`` is every label that must NOT be re-inferred — staged tiles and skip
    markers together, which is all an inference loop needs. ``skipped`` is the
    subset that came from skip markers, i.e. tiles a previous attempt determined
    had no pixels to write. Callers that report per-tile outcomes need the split:
    counting a restored skip as a success makes a resumed zone's tally disagree
    with the same zone's tally on a fresh run.
    """

    done: set[str]
    skipped: set[str]


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
    # Live tiles this run RESOLVED TO A SKIP (no valid pixels), so it staged nothing
    # for them. Written as fill rather than left alone — see `live_shards`.
    cleared: tuple[tuple[int, int], ...] = ()
    fill_values: tuple[tuple[str, float], ...] = ()  # (var, fill) for the cleared tiles

    def live_shards(self) -> list[tuple[int, int]]:
        """Every ``(row, col)`` this run is responsible for — staged AND skipped.

        Skipped tiles are included so the run WRITES its whole footprint instead of
        only the part it has data for. A year is filled in two commits (shards, then
        the completion attrs), so a crash between them leaves shards on an unmarked
        year that the campaign re-dispatches. If the retry's live set has shrunk — a
        re-ingested mosaic can turn a tile that had valid pixels into one that skips
        — then leaving skipped tiles untouched preserves the previous attempt's data
        under the new attempt's completion mark, mixing two inputs in one write-once
        year. Writing fill over them makes the published year exactly this run's.

        Bounded by the land mask, so ocean tiles are still never written at all, and
        an all-fill int8 shard is nearly free once zstd has seen it. The single-ROI
        `assemble` clears its non-live footprint for the same reason.
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
            # Also validate the trailing axis of 4-D vars: a (1, px, px, 1) block would
            # broadcast across every destination band/month and commit repeated values
            # under a shape check that only looked at [1:3]. The expected width is
            # per-variable (`trailing_extent`) because the two 4-D axes differ — bands
            # are as wide as the embedding, months are twelve.
            want_trailing = trailing_extent(var, self.embedding_dim)
            if want_trailing and block.ndim == 4 and block.shape[3] != want_trailing:
                raise ValueError(
                    f"Staged tile {path} has {var} trailing width {block.shape[3]} but expected "
                    f"{want_trailing} — a singleton axis would broadcast over the output."
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

    def _fill_block(self, shard: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
        """A whole tile of fill for a position this run skipped — with any coverage it DID measure.

        Built from the DESTINATION's fill values (passed in by the caller, which read
        them off the seeded arrays) so a cleared tile reads back identical to one that
        was never written — not zero for a float array whose fill is NaN.

        **Then the coverage variables are overlaid from the refused chunk's own tile, when it staged
        one.** A fully refused chunk measures real observation counts and real month coverage before
        it fails the gate, and filling those alongside the embeddings published zeros for a tile that
        had been looked at — while a MIXED tile published true counts, because one pixel in it
        happened to embed. Provenance that depends on a neighbour is worse than provenance that is
        merely absent, because nothing marks it.

        Absence stays ordinary and silent: an ocean position never staged coverage, and neither did a
        refused chunk from a run predating this. Both fill, exactly as before. What is NOT tolerated
        is a tile that exists and disagrees with the destination — that raises, because a raw-zarr
        write would C-cast it.
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
            # ABSENT or HALF-WRITTEN are the same answer for an OPTIONAL artifact: fill, and publish
            # the year. An incomplete coverage tile must not fail assembly — the skip marker means no
            # resume will ever re-stage it, so refusing here wedges the cell on every retry under the
            # stable run id, and it wedges it over provenance rather than over data. A crash between
            # `to_zarr` and `staged_complete` leaves exactly this, with no handler having run.
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
    ``StagedShardSource.load`` uses to place a tile — so the box is that shard's footprint and no
    new assumption about the chunk size is introduced here.

    Best-effort per label: a malformed label, or a row/col outside the zone, yields no entry rather
    than a wrong box or an exception. This runs after the cell has committed, so nothing here may
    raise; and a missing box publishes as null, which reads as "not recorded" instead of pointing a
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
        # This assembly's skip records, read from the markers by `_skip_summary` and consumed by
        # `publish_registry_part` afterwards. Declared here so the attribute always exists rather
        # than appearing only once a summary has run — a reader that had to guard for its absence
        # was also a reader that could not tell "no skips" from "a previous cell's skips", which is
        # the confusion `assemble_global` now clears at the start of every assembly.
        self._last_skip_records: dict[str, dict] = {}

    def _staging_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Get the staging path for a chunk."""
        return f"{self.staging_base}/{run_id}/{chunk.label}.zarr"

    def _coverage_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Where a REFUSED chunk stages the coverage it measured but could not embed.

        A third artifact class beside the staged tile and the skip marker, and it exists because
        the other two cannot carry this. A fully refused chunk has real observation counts and real
        month coverage — they are accumulated per strip regardless of validity — and throwing them
        away published zero counts for the tile while a MIXED tile published true ones. Whether a
        pixel's provenance survived therefore depended on whether some neighbouring pixel happened
        to embed, which is not a property anyone should have to reason about.

        Named ``.coverage.zarr`` and excluded explicitly from the ``.zarr`` match in
        :meth:`_list_staged` — without that it parses as a chunk label of its own, lands in
        ``staged`` with no ``.done`` beside it, and every refused chunk reads as an interrupted
        write.

        NOT vouched for by its own ``.done``: it is written BEFORE the skip marker, so the marker
        being present already implies this finished. That is the same ordering trick ``.done`` plays
        for a staged tile, reusing an artifact that has to exist anyway.
        """
        return f"{self.staging_base}/{run_id}/{chunk.label}{_COVERAGE_SUFFIX}"

    def _skip_marker_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Get the skip-marker path for a chunk.

        A zero-byte object whose presence records that the chunk was
        intentionally skipped during inference (every pixel failed the
        validity filter). Distinct from "no state at all", which indicates
        a chunk was never processed or was silently dropped by a crashing
        worker.
        """
        return f"{self.staging_base}/{run_id}/{chunk.label}.skipped"

    def _done_marker_path(self, run_id: str, chunk: ChunkSpec) -> str:
        """Get the completion-marker path for a staged chunk.

        A zero-byte object written by :meth:`write_chunk` **after** the staged
        Zarr is fully uploaded. A staged ``.zarr`` is many objects (group and
        array metadata plus one per data chunk) written with no atomic
        multi-object commit, so a crash mid-upload leaves a ``.zarr`` with valid
        metadata but missing data chunks — which Zarr reads back as *fill
        values*, not an error.

        This marker is the LIST-visible completeness signal: every consumer that
        learns what a run staged from one prefix listing (:meth:`_list_staged`,
        and through it :meth:`verify_staged_completeness` and
        :meth:`scan_existing_staged_artifacts`) keys on it, so an interrupted
        write is recognised without opening anything. The in-store
        ``staged_complete`` attribute written just before it (see
        :meth:`write_chunk`) is the same fact in a form a reader that already has
        the group open can check for free — see :meth:`_validate_staged_chunk`.
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
        with the staging prefix when it is cleaned up, and the mosaic they were derived from goes when
        the cell lands. A published cell is write-once.

        ``detail_uri`` is where the PER-TILE records are persisted, and passing one is what makes a
        future infill campaign possible: the summary in the store's attributes is pooled over the
        year, so it can say a tile somewhere reached fourteen observations against a cutoff of
        fifteen but not WHICH tile — and ranking candidates by how close they came is the whole of
        that planning problem.

        This method does NOT publish them. It keeps them on the writer for
        :meth:`publish_registry_part`, which the caller runs only after the cell has committed: the
        registry is read by consumers who are explicitly told they need not open the store, so a part
        written before the commit advertises a cell that may never land.
        """
        records, unreadable = read_skip_records(self.staging_base, run_id, skipped)
        # KEPT for the caller to publish AFTER the commit — see `publish_registry_part`. Reading the
        # markers here is not optional: this is the last moment they exist.
        self._last_skip_records = dict(records)
        summary = summarise_optical_skips(staged=staged, skipped=skipped, records=records)
        if unreadable:
            # SURFACED, because "no records" and "every read failed" are the same empty dict. Without
            # this a systematic failure — expired credentials, a wrong prefix — would publish a
            # provenance entry that quietly says no reason was recorded for anything.
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

        **One implementation because there were already two.** :meth:`write_chunk` and
        :meth:`write_coverage_only` staged this identically — same transpose, same chunking, same
        deliberate absence of a dtype — and ``test_coverage_only_staging`` exists precisely because
        the two could drift. Three sensors would have made it three copies. Sharing the code retires
        the drift rather than continuing to test for it.

        Month axis LAST in the staged file, matching the destination's ``(northing, easting, month)``
        so nothing has to transpose on the way in; the actor keeps it FIRST because that is the axis
        order the mask bundle slices on.

        No dtype in the encoding, deliberately. ``to_zarr`` stores a bool array as int8 with
        ``attrs dtype="bool"`` — its own boolean representation — and IGNORES an encoding dtype
        asking for bool, so the staged tile is int8 either way. The destination is seeded int8 to
        match, because assembly reads staged tiles with RAW zarr and so compares against what is on
        disk rather than what xarray hands back. A bool destination refused every staged month tile
        on the dtype guard; asking for bool here only looks like a fix.
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

        Only the provenance variables: the observation counts and the month-coverage mask, with the
        same dtypes, axis order and chunking :meth:`write_chunk` gives them, because assembly reads
        both with raw zarr and compares against what is on disk. ``test_coverage_only_staging``
        asserts that equivalence directly rather than trusting these two code paths to stay in step.

        No embeddings and no scales, which is the point: those are what the chunk failed to produce,
        and a fill array of 128 bands per refused tile is the cost this avoids. Assembly fills them
        and copies these — see :meth:`StagedShardSource._fill_block`.
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
        # `staged_complete`, for the same reason write_chunk sets it and read through the same gate
        # (`_open_staged_tile`). A crash between the array metadata and the chunk objects leaves a
        # tile that reads back as FILL rather than as an error — which here would publish zeroed
        # counts while looking like measured ones, the exact failure this whole change removes.
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

        The ordering is the point. The registry exists so a consumer need not open the store, and it
        is a sibling of the store rather than part of it — so nothing reconciles the two. A part
        written while the assembly was still running therefore advertises embedded and refused tiles
        for a zone-year that may never land: a failed worker, a failed merge, an injected fault, and
        the claim is permanent. Publishing after the commit makes the part's existence mean the cell
        exists.

        Returns the URI on success, ``None`` on failure. Best-effort, and only in this direction:
        losing a part costs a future infill campaign some precision, while raising here would fail a
        cell that has already landed — the worst trade available, since the data is fine and only its
        index is missing. A lost part is also recoverable, because every column is derivable from the
        store the cell just wrote.
        """
        # Refused shards' records come from their markers; embedded shards' come from the actors'
        # results. Merged into one map because a row's measurements mean the same thing either way —
        # see `_coverage_record` — and the marker side wins on a label in both, since a marker is
        # written at the end of a shard that refused everything.
        records = {**(embedded_records or {}), **self._last_skip_records}
        uri = part_uri(registry_root, zone, year, run_id)
        boxes = _tile_bboxes_wgs84(zone, [*embedded, *refused])
        # The build publishing this part IS the build that produced the cell — same process. Reads
        # install metadata and returns None rather than raising, so a wheel install with no VCS
        # information simply leaves the columns null.
        build = code_identity() or {}
        try:
            # CREDENTIALLED, like every other write. The registry is a sibling of the store in the
            # same bucket, so when that store is the partner-owned one it needs the same assumed
            # role — and `_fs_for` with no options falls back to fsspec's ambient chain, which is
            # the task role and cannot write there. That failure is caught below and the cell is
            # still tagged, so the symptom is not a failed run: it is a campaign that looks healthy
            # and publishes no registry at all, which is the artifact the access request promises.
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
                # Also in the key-value block, so a part read on its own — without the dataset's
                # partitioning — still states the rule its rows were judged against.
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

        Called when :meth:`write_coverage_only` failed partway. A tile whose ``staged_complete`` was
        never set reads back as fill and is REFUSED by :func:`_open_staged_tile`, and the skip marker
        written immediately afterwards makes every resume omit the chunk — so nothing repairs it, and
        under the stable run id the refusal repeats on every retry and wedges the cell. Best-effort
        because it runs on a path that is already failing: another exception here would replace a
        degraded provenance entry with a lost chunk.
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

        Called instead of ``write_chunk`` when a live (ROI-intersecting) chunk has no valid
        pixels. The marker's PRESENCE lets assembly-only runs distinguish a legitimate skip
        from a silently-failed chunk; its CONTENT is the per-shard registry.

        **The content exists because the marker used to be zero bytes, and that made a
        thin-depth refusal indistinguishable from no coverage at all.** The dataset computes
        three refusal reasons per strip and deliberately keeps them apart — no optical input,
        too little of it, no radar — the actor sums them over the chunk, and then a fully
        refused chunk threw all of it away. What survived was a count of "optical skips",
        which on 2026-08-18 named the wrong cause for 43 of 40S's 58 live shards (they were
        refused for having no RADAR) and left no way to tell those from land that was never
        imaged. A published cell is write-once, so a reason not recorded here is not
        recoverable later: the mosaic it was derived from is deleted when the cell lands.

        ``record`` is written as JSON. ``None`` (and an unreadable or empty marker) stays
        legal and reads back as "no reason recorded", because a marker from an older run is
        not an error — see :func:`read_skip_records`.
        """
        path = self._skip_marker_path(run_id, chunk)
        fs = _fs_for(path)
        # A run whose FIRST artifact is a skip marker has no staging dir yet on
        # directory-backed filesystems (write_chunk's to_zarr creates it as a
        # side effect; a bare open() does not). No-op on object stores.
        fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
        # Drop any staged zarr for this chunk first, and its completion marker with
        # it. Both can only be crash/stale artifacts — the chunk is skipping, so
        # nothing valid can be staged for it — and leaving the pair behind makes
        # verify_staged_completeness raise "BOTH a staged zarr and a skip marker" on
        # every retry under the stable run_id, wedging the cell until someone deletes
        # it by hand. The marker goes FIRST so an interruption here can never leave a
        # .done vouching for a .zarr that is already gone.
        for stale in (self._done_marker_path(run_id, chunk), self._staging_path(run_id, chunk)):
            with contextlib.suppress(FileNotFoundError):
                fs.rm(stale, recursive=True)
        # THE MARKER'S PRESENCE IS LOAD-BEARING; ITS CONTENT IS NOT. `verify_staged_completeness`
        # distinguishes a legitimate skip from a crashed worker by this file existing, so a record
        # that cannot be serialised must cost the RECORD, never the marker — losing the reason is a
        # degraded provenance entry, while failing to write the marker turns a benign skip into a
        # failed chunk and wedges the cell on every retry. A numpy scalar reaching `json.dumps` is
        # exactly how that would happen, and it is one careless `int()` away at all times.
        body = b""
        if record is not None:
            try:
                # Sorted keys and a trailing newline: the marker is read by eye during an
                # investigation at least as often as by the summariser.
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

        Creates an xarray Dataset with an 'embedding' variable (mean) and
        optionally 'embedding_std' and 'scale' variables.

        Records completion **after** the store is fully written — the
        ``staged_complete`` in-store attribute and then the ``<label>.done``
        sibling marker (see :meth:`_done_marker_path`) — so an interrupted upload
        leaves a tile that resume re-infers rather than assembly trusting its
        fill-valued holes.

        Args:
            chunk: Chunk specification.
            embeddings: Array of shape (H, W, embedding_dim), float32 or int8.
            run_id: Unique run identifier.
            scales: Per-pixel scale factors of shape (H, W), float32.
                Used to dequantize int8 embeddings.
            embeddings_std: Optional std array, same shape as embeddings, float32.
            month_covered: Optional dict mapping month-mask variable names to (12, H, W) bool
                arrays — which calendar months each pixel was seen in per sensor, month 0 = January.
                Keys should be from ``MONTH_COVERED_VARS``. Staged transposed to (H, W, month) so
                each lands in the same axis order as its destination array, which the assembly graph
                writes without reordering.
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

        self._stage_month_masks(data_vars, encoding, month_covered, chunk, staged_chunks_2d)

        ds = xr.Dataset(
            data_vars,
            coords={
                "northing": np.arange(chunk.y_start, chunk.y_stop),
                "easting": np.arange(chunk.x_start, chunk.x_stop),
                "band": np.arange(self.embedding_dim),
            },
        )

        # Retract the previous attempt's completion marker BEFORE overwriting the tile.
        # to_zarr replaces the tile in place, so a marker left from an earlier write
        # would keep vouching for it all the way through the rewrite, and a listing
        # taken in that window would report a tile that is currently incomplete as
        # complete. Retracting first makes the window read as interrupted, which is
        # the truth. No-op on a first write.
        # The SKIP marker goes too, and for a different reason: a chunk that had no
        # valid pixels on an earlier attempt and produces some on this one would
        # otherwise end up carrying both, and verification reads that pair as an
        # inconsistent artifact and refuses to assemble. Under the stable,
        # input-fingerprinted run_id that refusal repeats on every retry, wedging the
        # cell until someone deletes the marker by hand. `write_skip_marker` already
        # clears a stale tile in the opposite direction; this is the other half of the
        # same rule — whichever outcome a chunk reaches, it leaves no trace of the
        # other one.
        done_path = self._done_marker_path(run_id, chunk)
        done_fs = _fs_for(done_path)
        for stale in (done_path, self._skip_marker_path(run_id, chunk)):
            with contextlib.suppress(FileNotFoundError):
                done_fs.rm(stale)
        # And the COVERAGE-ONLY tile of a previous refusal, the third artifact this rule now covers.
        # That tile carries real counts, so a stale one beside a successful write is worse than
        # untidy: it measures a footprint this attempt has just replaced. Recursive — it is a
        # directory, not a marker.
        with contextlib.suppress(FileNotFoundError):
            done_fs.rm(self._coverage_path(run_id, chunk), recursive=True)

        # The same S3 client config the staged READS use. The write is the larger and
        # more failure-prone of the two — it is the one uploading the tile — and it was
        # the only staging op going out without the retry configuration.
        ds.to_zarr(path, mode="w", encoding=encoding, storage_options=_staged_storage_options(path))
        # --- Completion markers, written LAST, in SEPARATE ops after to_zarr returns.
        # to_zarr can create every array's metadata before all its chunk objects, so a
        # crash mid-write leaves a tile with correct vars/shape/dtype but MISSING
        # chunks (read back as fill, not as an error). Completeness is therefore its
        # own signal, and it is recorded twice because two kinds of consumer need it:
        #
        #   1. `staged_complete` in-store attribute — free to check for a reader that
        #      already has the group open (assembly's per-tile reads), and the only
        #      signal that reflects the tile as it is NOW rather than as the last
        #      listing saw it (see _open_staged_tile).
        #   2. `<label>.done` sibling object — visible in the staging prefix LISTING,
        #      so verification and the resume scan classify every tile of a run from
        #      one listing without opening any of them.
        #
        # ORDER MATTERS and is the invariant the two checks rely on: .done last, so
        # `.done present` implies `attribute set` implies `to_zarr returned`. A crash
        # anywhere earlier leaves a tile the listing reports as interrupted, which the
        # resume scan re-infers (write_chunk's mode="w" overwrites it).
        marker = zarr.open_group(path, mode="a", storage_options=_staged_storage_options(path))
        marker.attrs["staged_complete"] = True
        with done_fs.open(done_path, "wb") as f:
            f.write(b"")
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
            AllChunksSkippedError: If the run exists but every chunk skipped.
            FileNotFoundError: If the run has no staged chunks and no skip markers.
        """
        labels = self._list_staged_labels(run_id)
        if not labels:
            if skipped := self._list_skip_marker_labels(run_id):
                # A real run in which every chunk skipped. ASK THE MARKERS what grid they were
                # produced on before giving up: they record it precisely because there is no tile
                # here to measure, and falling back to the CURRENT configured size re-enumerates the
                # chunk grid under different labels — after which verification reports the valid old
                # markers as unexpected and its own new labels as missing, defeating the all-skipped
                # assembly path this branch exists to serve. The default moved from 2000 to 2048, so
                # this is not hypothetical for any run staged before that.
                records, _unreadable = read_skip_records(self.staging_base, run_id, skipped)
                sides = {int(r["chunk_side_px"]) for r in records.values() if isinstance(r.get("chunk_side_px"), int)}
                if len(sides) == 1:
                    return sides.pop()
                # Either no marker records the grid (every run before this field) or they disagree,
                # which is a heterogeneous stage no single size can describe. Both are "unknown", and
                # the caller's configured size is the only answer left.
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

        Returns the labels that have completed staged zarrs (an empty set means
        every live chunk resolved to a skip marker), so callers don't re-LIST the
        staging prefix to learn what verification already read.

        A live (ROI-intersecting) chunk can resolve in two ways: inference
        produced embeddings (a staged zarr whose ``.done`` marker confirms the
        many-object write finished — see :meth:`_done_marker_path`), or every
        pixel failed the validity filter (skip marker). Any live chunk with
        neither — including one left half-written by a crash — indicates a silent
        failure and must stop assembly here rather than produce an output that
        looks identical to a legitimate skip.

        Interrupted tiles are reported as their own category rather than folded
        into "missing". Reaching assembly is what makes them anomalous: the resume
        scan re-infers an interrupted tile, so one that survives to here means
        either inference did not run at all (an assembly-only run) or it crashed
        the same way twice — and the remedy differs from a chunk that was never
        attempted.

        Args:
            run_id: Run identifier (locates the staging directory).
            expected_chunks: Chunks that are expected to resolve (i.e. ROI-
                intersecting chunks). Non-intersecting chunks are filled in
                at assembly time and should not be passed here.
            log: Optional logger.

        Raises:
            IncompleteStageError: If any live chunk has neither a completed
                staged zarr nor a skip marker, or if there are extra labels that
                don't match any expected chunk.
        """
        _log = log or logger
        listing = self._list_staged(run_id)
        staged, skipped = set(listing.complete), set(listing.skipped)
        resolved = staged | skipped
        expected = {c.label for c in expected_chunks}

        # A skip marker resolves the chunk on its own, so a leftover half-written
        # pair alongside one is inert (write_skip_marker clears both, but a crash
        # between the two removals could leave one behind).
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

    def _list_staged(self, run_id: str) -> StagedListing:
        """Classify every artifact of a run from one staging LIST.

        The single place the three-way split is derived, so verification and the
        resume scan can never disagree about what a run staged. See
        :class:`StagedListing` for what each bucket means and
        :meth:`_done_marker_path` for why completeness is its own signal.
        """
        names = self._list_run_names(run_id)
        # The coverage-only tile of a REFUSED chunk is excluded FIRST, and must be: it ends in
        # ".zarr", so the match below would take "<label>.coverage" for a chunk label of its own,
        # find no ".done" beside it, and report every refused chunk as an interrupted write.
        staged = {n.removesuffix(".zarr") for n in names if n.endswith(".zarr") and not n.endswith(_COVERAGE_SUFFIX)}
        done = {n.removesuffix(".done") for n in names if n.endswith(".done")}
        skipped = sorted(n.removesuffix(".skipped") for n in names if n.endswith(".skipped"))
        # Symmetric difference, not `staged - done`: a .done whose .zarr is absent is
        # equally a half-landed pair (an interrupted cleanup, or a marker that outlived
        # the tile it vouches for), and it has the same remedy — re-infer, which
        # rewrites both. Collapsing the two cases keeps one bucket with one fix.
        return StagedListing(
            complete=sorted(staged & done),
            interrupted=sorted(staged ^ done),
            skipped=skipped,
        )

    def _list_staged_labels(self, run_id: str) -> list[str]:
        """List chunk labels whose staged write COMPLETED in the run directory.

        Keyed on the ``.done`` completion marker, not the ``.zarr`` directory: a
        ``.zarr`` present without its marker is an interrupted write, deliberately
        excluded so resume re-runs it rather than assembling its partial
        (fill-valued) contents. See :meth:`_done_marker_path`.

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

        - ``("incomplete", ...)`` — a crash artifact: genuinely absent
          (``FileNotFoundError``), or rejected by :func:`_open_staged_tile` because
          its write never finished. The caller RE-INFERS it (``write_chunk``'s
          ``mode="w"`` overwrites) rather than raising — with the stable,
          input-fingerprinted run_id a raise would wedge every retry on the same
          partial artifact. This is the one caller that treats an unfinished tile
          as routine, which is why it catches what the shared opener raises instead
          of letting it through: on resume an unfinished tile is the expected input,
          not an anomaly.
        - ``("invalid", ...)`` — a COMPLETE tile that is structurally wrong (missing
          var / shape / dtype). The caller raises: this is a real anomaly (e.g. a
          stale wrong-grid store), not a resumable crash.

        Any OTHER open error (auth, throttling, transient network, corrupt metadata)
        propagates — a valid completed tile must not be silently re-inferred because
        of a transient read failure.
        """
        path = self._staging_path(run_id, chunk)
        try:
            group = _open_staged_tile(path)
        except FileNotFoundError:
            # The artifact is genuinely gone (partial/removed) → self-heal by re-inference.
            # Other open errors (auth, throttling, transient network, corrupt metadata)
            # must NOT be treated as "partial" — silently excluding + re-inferring a valid
            # completed tile on a transient read failure is expensive and wrong; propagate.
            return ("incomplete", "staged zarr not found (partial or removed) — re-infer")
        except IncompleteStageError as exc:
            return ("incomplete", str(exc))
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
            Use :meth:`scan_existing_staged_artifacts` when the caller needs to
            know WHICH of those came from skip markers rather than staged tiles.

        Raises:
            RuntimeError: If any staged Zarrs are invalid. Lists all invalid
                paths so the user can remove them before retrying.
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

        Both artifact kinds mean "do not re-infer this tile", which is why the
        set-returning form merges them. But they are different OUTCOMES: a staged
        zarr produced pixels, a skip marker recorded that the tile had none. A
        resumed zone that reports its skips as successes misstates what the run did.
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
            # Not an error: these are exactly what resume exists for. They are simply
            # absent from `done`, so run_inference re-runs them (write_chunk's
            # mode="w" overwrites the partial). Logged because a tile appearing here
            # run after run means the crash is reproducible, not incidental.
            _log.warning(
                "%d staged tile(s) are interrupted (staged zarr and .done marker did not both land) — "
                "re-inferring: %s%s",
                len(listing.interrupted),
                listing.interrupted[:10],
                "..." if len(listing.interrupted) > 10 else "",
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
        return StagedResume(done=valid, skipped=valid & skip_marker_labels)

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
                # Rejects a crash-partial tile up front, before the schema commit —
                # the same check the band fill will make when it reads the data, but
                # cheaper to fail here than part-way through a created store.
                group = _open_staged_tile(path)
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
            # Only when an array that uses the axis is created — a bare `month` coordinate in a
            # store with no month dimension is a schema surprise for readers. ANY of the three
            # sensors introduces the axis and they share it, so the coordinate is written once.
            # 1..12, so `cov.sel(month=7)` means July rather than the eighth index, mirroring what
            # the global store's seeder writes.
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

        Raw-zarr fork/merge engine (no Dask): worker processes write staged-tile
        pixels for disjoint, granularity-aligned northing bands straight into
        the output arrays; the coordinator merges the forks, sets root attrs,
        and commits once via
        :func:`~tessera_embeddings.storage.shard_writer.commit_with_rebase`.
        Emits one ``ASSEMBLY_SUMMARY`` record (:func:`_assembly_summary_line`)
        with the fill's per-worker phase timings and counts.

        Create-or-extend semantics on the time axis:

        * fresh store → schema + coords from ``layout`` (``SINGLE`` by default —
          today's single-ROI geometry, D8) in a first metadata-only commit;
        * existing store, new time value → every time-dimmed array is resized by
          one step (explicitly — this replaces ``mode="a"`` appends) in a first
          commit;
        * existing store, time value already present → its index is overwritten
          in place (idempotent resume: a crashed assembly re-run lands on the
          same index instead of appending a duplicate timestep). Live positions
          are rewritten and **every destination chunk this run does not rewrite
          as live is cleared to fill** — skip-marked footprints AND positions
          outside this run's ROI mask (a prior run's data must not survive under
          a rerun's skip marker, nor be stranded as stale embeddings when the ROI
          shrinks). A same-date re-assembly with a different mask is therefore
          safe; the timestep reflects exactly this run's live footprint.

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
        t0 = time.monotonic()

        # Determine which chunks have COMPLETED staged zarrs. A chunk that intersects
        # the ROI but was skipped during inference (all pixels failed validity) has a
        # skip marker instead of a zarr — its footprint stays at fill.
        #
        # The gate runs here rather than only in the calling flow, because this method
        # derives its own live set from the ROI mask and is called directly (an
        # assembly-only re-run, a test). Without it, the only thing standing between a
        # crash-partial tile and a published timestep of fill values would be the
        # per-tile check inside the workers — a correct backstop, but one that fires
        # after the schema commit rather than before any work starts.
        roi_live_chunks = filter_chunks_by_roi_mask(
            chunks,
            roi_zarr_path,
            storage_options=plain_zarr_storage_options(roi_zarr_path, get_credentials, s3_region),
        )
        staged_labels = self.verify_staged_completeness(run_id, roi_live_chunks, log=_log)
        live_chunks = [c for c in roi_live_chunks if c.label in staged_labels]
        # SKIP-MARKED CHUNKS ARE DROPPED HERE AND THEIR COVERAGE ARTIFACTS GO UNREAD. That is a
        # DELIBERATE asymmetry with the global path, recorded because it reads as an oversight
        # otherwise: global assembly consumes each skipped chunk's `.coverage.zarr`, so its published
        # observation counts and month coverage are real over refused footprints. This path publishes
        # fill there instead — a mixed output carries zeros where the global one carries measurements,
        # and an all-skipped fresh output can omit the coverage arrays entirely.
        #
        # It stays that way because the two paths have different consumers: those arrays exist to feed
        # the registry a targeted repair campaign ranks from, and only the global path writes that
        # registry. Closing the gap is tracked with the repair-campaign follow-ups (issue #103) rather
        # than done here, because doing it properly means reading the artifacts the way
        # `StagedShardSource._fill_block` does, and this path has no coverage test that would catch
        # getting it wrong. Anyone reading the single-ROI output for coverage analysis wants that issue
        # first.
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

        # `embedding_std` is decided by the caller's `compute_std`, not by what the tiles
        # happen to hold; every other carried var is included when the tiles staged it.
        staged_extra = self._staged_vars_present(run_id, live_chunks, CARRIED_VARS)
        variables = [*REQUIRED_VARS]
        if compute_std:
            variables.append("embedding_std")
        variables += [v for v in CARRIED_VARS if v != "embedding_std" and v in staged_extra]

        # Divide the fleet-wide S3 concurrency target across worker forks; see
        # TARGET_AGGREGATE_S3_CONCURRENCY. Forks inherit the repo config through
        # the pickled session, so no save_config round-trip is needed.
        per_worker_cap = max(1, TARGET_AGGREGATE_S3_CONCURRENCY // max(1, n_workers))

        # Time-only, matching the global store (zarr_store.global_store_config)
        # and the rule documented there: split the axis along which a single
        # commit is NARROW. An assemble writes ONE timestep across the whole
        # spatial extent, so time@1 keeps a write off every prior timestep's
        # manifests, while a spatial split would only shard the axis this commit
        # rewrites in full — more manifest objects for the same refs.
        with manifest_split({"time": 1}):
            repo, is_new = open_or_create_repo(
                output_path,
                max_concurrent_requests=per_worker_cap,
                get_credentials=get_credentials,
                region=s3_region,
                scatter_initial_credentials=True,  # see assemble_global's call site
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
            traced_commit(session, f"Run {run_id}: create schema ({layout.name})")
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
                # Extent, CRS and endpoints STILL do not pin an axis: a reordered or
                # non-affine interior satisfies all three, and this phase writes staged
                # pixels POSITIONALLY while the store keeps its existing coordinate arrays.
                # Such an append publishes real pixels at the wrong coordinates with nothing
                # to signal it.
                #
                # The same comparison the zone-fill runner makes, for the same reason and at
                # the same cost: the complete vectors against the stored axes, two 1-D reads
                # once per append. A per-step spacing test cannot substitute — whatever slack
                # it admits per step, an axis that wanders and returns can accumulate while
                # still matching the length and both endpoints.
                #
                # Reachable on the hand-provided mosaic path, which is supported.
                for axis, values, stored in (
                    ("northing", spatial.northing, z_north),
                    ("easting", spatial.easting, z_east),
                ):
                    got = np.asarray(values, dtype="float64")
                    want = np.asarray(stored[:], dtype="float64")
                    # A float round-trip allowance, not a geometric one: coordinates
                    # round-trip through float32 at ~1 m near a 9.3e6 m northing, while any
                    # real displacement is a whole pixel.
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
            # Variables staged by this run but absent from the store (e.g. a
            # store created before obs counts, or compute_std newly on) are
            # created schema-only at the current time extent; prior timesteps
            # read back as the layout fill. The old engine silently wrote such
            # variables MISALIGNED with the time axis — creating them properly
            # is the loud-and-correct replacement.
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
                    # The axis this array introduces to a store that predates it; without
                    # the coordinate a reader gets a positional month axis and has to know
                    # it is 0-based (see `_create_schema`).
                    _write_coord_arrays(root, {"month": np.asarray(MONTH_COORD, dtype="int16")})
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
        fill: dict[str, Any] = {"workers": [], "wall_s": 0.0, "merge_s": 0.0}
        if payloads:
            # These payloads genuinely are northing-band writes — name them so.
            # `_log` is the caller's logger (the flow's run logger under Prefect),
            # so the coordinator's progress lines reach the orchestrator too.
            fill = run_forked(session, _fill_band_worker, payloads, unit="band writes", log=_log)

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

        t_commit = time.monotonic()
        commit_with_rebase(session, f"Run {run_id}: {len(chunks)} chunks assembled")
        commit_s = round(time.monotonic() - t_commit, 3)
        # Atomic publish: fast-forward `main` to the fully-written tip. The guard
        # fails loudly if another writer advanced `main` since we branched (two
        # processes assembling the same ROI store — unsupported). Then drop the
        # work ref; `main` now retains the snapshot.
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

        The global write path (ADR-008 D3/D6): every staged tile is exactly one
        output shard, written whole and lean by
        :func:`~tessera_embeddings.storage.shard_writer.write_year_shards` —
        fork/merge across ``n_workers`` processes, one commit per (zone, year),
        ``years_complete`` and per-year run provenance updated in the same commit. The zone group must already be seeded
        (:func:`~tessera_embeddings.storage.global_store.seed_zone_groups`);
        nothing is ever created or resized here (D1). Emits one
        ``ASSEMBLY_SUMMARY`` record (:func:`_assembly_summary_line`) with the
        fill's per-worker phase timings and counts.

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
            registry_root: Root of the published registry dataset
                (:meth:`BucketPaths.optical_registry`). One Parquet part per cell lands under it,
                a row per live tile, because the year's summary in the store is pooled and so cannot
                say WHICH tile came closest to the depth cutoff — the question a cleanup campaign
                asks. ``None`` writes nothing and changes nothing the store commits.
            optical_min_obs: The depth rule this cell was filled under, stamped on every registry
                row. The registry's ``obs_max``/``median_obs_where_any`` are distances from this
                line, so without it they cannot be read; the store's root is the authority and the
                runner asserts the config matches it before any of this runs.
            embedded_records: Per-shard coverage for shards that DID embed something, keyed by
                label, as the actors reported it. Merged with the refused shards' marker records so
                a partly-refused tile reports what the depth gate removed from it; a label present
                in both takes the marker, which is written at the end of a wholly refused shard.
            n_workers: Worker process count; also divides
                ``TARGET_AGGREGATE_S3_CONCURRENCY`` into the per-fork cap.
            staged_labels: Pre-listed staged tile labels (e.g. the return of
                :meth:`verify_staged_completeness`); ``None`` lists the prefix.
            radar_coverage: This YEAR's radar-coverage summary, from
                :func:`summarise_radar_coverage` over the run's chunk results. Recorded on
                the year's ``runs`` entry. Per year rather than per zone because radar
                coverage is a property of what was acquired: one year of a zone can be
                radar-free where another is not.
            skipped_labels: Live tiles this run resolved to a SKIP. Written as fill,
                so the published year is exactly this run's output — see
                :meth:`StagedShardSource.live_shards` for the mixed-year hazard that
                leaving them untouched creates. ``None`` writes only staged tiles.
                Also recorded, with ``staged_labels``, as the year's ``optical_skips``
                provenance summary (:func:`summarise_optical_skips`) — so it must be
                the staging prefix's resolution of the live set (what the
                staged-completeness scan established), never the finishing leg's own
                tally: a resumed run's markers were written by earlier legs. An empty
                sequence therefore MEANS "resolved, none skipped" and records a zero
                summary, while ``None`` means the caller resolved no live set and
                records no summary — a zero it did not establish would read as measured.
                When EVERY live tile skipped, this is the whole footprint and
                ``staged_labels`` is empty; pass ``empty=True`` with it.
            empty: Record the year as holding no data. For the all-skipped case, where
                the fill write and the completion mark must agree that the year is
                empty — marking it without the write would leave a previous attempt's
                shards readable underneath.
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
        # THIS ASSEMBLY'S records only. `_skip_summary` populates this during the shard write, and
        # `publish_registry_part` reads it afterwards — but `skipped_labels=None` is a supported
        # default that skips the summary entirely, so a reused writer would carry the PREVIOUS cell's
        # refusal measurements into these registry rows. Chunk labels are grid-local
        # (`chunk_<row>_<col>`) and repeat across every zone and year, so the stale entries would
        # match by name and attach one cell's refusals to another's tiles — silently, and with
        # plausible numbers. Reset here rather than in the summary, because the bug is the summary
        # NOT running.
        self._last_skip_records = {}

        labels = sorted(staged_labels) if staged_labels is not None else self._list_staged_labels(run_id)
        # Zero staged tiles is legitimate in exactly one case: every live tile of the
        # year resolved to a skip, and the caller is clearing the footprint before
        # marking the year empty. Without any skipped tiles to write there is nothing
        # for this call to do, and an empty staged prefix is the corruption it exists
        # to catch.
        if not labels and not skipped_labels:
            raise IncompleteStageError(f"Run {run_id!r} has no staged chunks under {self.staging_base}")
        shards = tuple(sorted(parse_chunk_label(label) for label in labels))

        # S3 budget: divide the FLEET target across concurrent fills, not just this
        # fill's forks. TARGET_AGGREGATE_S3_CONCURRENCY // n_workers alone bounds one
        # fill to ~target, so K concurrent fills burst K times the target PUTs (the
        # 800-req SlowDown). The campaign passes `s3_concurrency = target //
        # max_parallel_zones`; None = full target.
        #
        # The budget sets the per-fork REQUEST CAP only -- it does not reduce the fork
        # count, which is why a wide campaign no longer runs its assemblies on a
        # fraction of their workers. Since that cap floors at 1, the fleet may exceed
        # the target by up to `max_workers * n_clusters`; see `_s3_budget_split`.
        n_workers, per_worker_cap = _s3_budget_split(s3_concurrency, n_workers)
        repo = open_global_repo(
            store_path,
            get_credentials=get_credentials,
            region=s3_region,
            max_concurrent_requests=per_worker_cap,
            # `write_year_shards` PICKLES this session to spawned children; without it each
            # deserialises with no credential and calls back per S3 request for the life of the
            # fork (icechunk#2077). Opt-in per call site because the pickle carries a live
            # secret -- safe across a local spawn pipe, not over a network transport. True
            # unconditionally, since `_create_storage` substitutes a default provider when
            # `get_credentials` is None and omits the option when there is no provider at all.
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
        # Destination band width is the grid authority for the staged-tile band
        # check — NOT ZarrWriter.embedding_dim, which defaults to 128 and needn't
        # match a zone seeded at a different band.
        dest_band = int(cast(zarr.Array, node["embeddings"]).shape[-1])

        missing_dst = [v for v in ("embeddings", "scales") if v not in node]
        if missing_dst:
            raise ValueError(
                f"Zone group {zone} lacks required array(s) {missing_dst} — a fill without them "
                "could not be dequantized; the group must be seeded with a full GLOBAL layout."
            )
        # Optional vars the destination CAN hold. StagedShardSource.load checks
        # every tile for one of these that's absent from `variables` (i.e. present
        # only in some tiles, so silently dropped) — full-coverage homogeneity, not
        # a first/last sample.
        dest_optional = tuple(v for v in CARRIED_VARS if v in node)

        if labels:
            # One probe of a staged tile: required vars, dtypes, variable set, and
            # tile pitch (exact — a truncated half-tile must not pass; every other
            # tile is re-validated by StagedShardSource.load as it is read).
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
            # Validate each staged variable's dtype against its DESTINATION array, not
            # just embeddings/scales (asserted int8/float32 above): a raw-zarr shard
            # write does a silent C-cast, so a uniformly int64/uint32 observation
            # count — which agrees with the probe, so StagedShardSource's tile-vs-probe
            # check passes — would be narrowed into a seeded uint16 without this guard.
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
            # Nothing staged, so there is no tile to probe and nothing to validate
            # against — the write is fill and only fill. The DESTINATION is then the
            # authority on both questions the probe usually answers: which variables to
            # write (every one the zone holds, since any of them could carry a previous
            # attempt's data) and at what dtype (the seeded one, which a fill cannot
            # disagree with).
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
        # Materialised once: the same labels drive the clearing writes AND the year's
        # provenance summary, so the two can never describe different tile sets.
        skipped = sorted(skipped_labels or ())
        # Fill values come off the SEEDED arrays, so a cleared tile reads back exactly
        # as an unwritten one does (0 for int8 embeddings, NaN for float scales).
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
        # The labels the registry part will describe, captured before the write so the part reports
        # what THIS call published rather than whatever a later listing happens to see.
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
            # Derived from what THIS call publishes as fill, so the record and the
            # write agree by construction; run_provenance drops it on an empty year.
            # `skipped_labels=None` is a caller that resolved no live set at all, so
            # there is nothing to summarise and no ZERO to assert (see below).
            optical_skips=(self._skip_summary(run_id, labels, skipped) if skipped_labels is not None else None),
            empty=empty,
            telemetry=telemetry,
            # The fill's coordinator progress goes through the caller's logger —
            # under Prefect that is the flow's run logger, the only route to the
            # Prefect API; the module logger reaches only the process log stream.
            log=_log,
            fault=fault,
            input_coverage=input_coverage,
        )
        # THE CELL IS COMMITTED. Only now is the registry allowed to say so — it is a sibling of the
        # store that nothing reconciles against it, and consumers are told they need not open the
        # store, so a part written any earlier could advertise a cell that never landed.
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
                # The store's own callback: the registry is its sibling in the same bucket, so
                # whatever credential opened the store is the one that can write beside it.
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


#: A refusal reason a skip record may carry, in the order a reader should weigh them: the first is a
#: fact about the imagery, the second this campaign's quality rule, the third a coverage fact. Kept as
#: an explicit tuple because the summary reports each one's total and a missing key must read as zero
#: rather than vanish from the record.
REFUSAL_REASONS: tuple[str, ...] = ("no_optical", "thin", "no_radar")


def read_skip_records(
    staging_base: str, run_id: str, labels: Iterable[str], *, workers: int = 32
) -> tuple[dict[str, dict], int]:
    """``({label: record}, n_unreadable)`` for every skip marker that carries one.

    **Concurrent, because this is one small GET per refused shard and they are independent.** A
    zone can refuse hundreds — 40S refused 43 of 58, and the largest zones hold 556 live tiles — so
    read serially this would add a round trip each to the critical path between the last chunk and
    the commit. One filesystem is resolved once rather than per label, for the same reason.

    **The failure count is RETURNED, not swallowed.** A marker that is empty, unreadable or not JSON
    yields no entry, which is correct for the ordinary cases: markers written before the registry
    existed are zero bytes, and a resume across that change must still assemble. But "no records" and
    "every read failed" are the same empty dict and mean opposite things — a systematic failure
    (expired credentials, a wrong prefix) would otherwise publish a provenance entry that quietly
    says nothing was recorded. The caller reports the count.
    """
    labels = list(labels)
    if not labels:
        return {}, 0
    base = f"{staging_base.rstrip('/')}/{run_id}"
    try:
        fs = _fs_for(base)
    except Exception:
        # RESOLVING the filesystem can fail on its own — bad credentials, a malformed URI — and this
        # sat outside every guard, so one such failure raised out of the year's provenance
        # construction and would have failed a cell at ASSEMBLY, after all its inference was paid
        # for. Same asymmetry as the marker itself: these reasons are a diagnostic, and losing them
        # must never cost the cell that earned them. Reported as every marker unreadable, which the
        # caller surfaces loudly, rather than as no reasons recorded.
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
            # UnicodeDecodeError is NOT a JSONDecodeError — a partially written or
            # byte-corrupt marker raises it out of `json.loads`'s own decode step, and
            # letting it escape `pool.map` aborts the whole assembly on one bad object.
            # Provenance is fail-soft by design: count it unreadable and carry on.
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

    A skipped tile — every pixel failed the validity filter, nothing staged — is
    written as fill, which is also what ocean reads as, so a consumer of a completed
    year cannot tell "no valid optical data" from "not land" without this record. The
    count answers how much was lost; the labels answer WHERE, which is what lets a
    consumer mask the area; the live total is the denominator that makes the count
    interpretable without fetching the land mask.

    Both inputs must be the STAGING PREFIX's resolution of the run's live tiles —
    the two halves the staged-completeness scan establishes (staged zarrs and skip
    markers) — never one leg's own tally: skip markers persist across resumes, and
    the leg that finishes a run may have staged nothing for tiles an earlier leg
    skipped, reporting them as resumed successes. A summary built from that leg's
    results would record zero skips while publishing fill over them.

    The label list needs no size cap: the one case where it would span a whole zone —
    every live tile skipped — is recorded by the ``empty`` flag instead, and
    :func:`~tessera_embeddings.storage.shard_writer.run_provenance` drops this summary
    from an ``empty`` year's record.
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
    # ORGANISED BY REASON rather than by shard, for three reasons. It is what a reader asks — "why is
    # this land empty, and which parts" — answered without walking a dict. It is self-describing:
    # every key is a reason name, so no legend is needed. And it is an order of magnitude smaller: a
    # nested record per shard put ~200 bytes x hundreds of shards into a zarr ATTRIBUTE, which every
    # reader of the zone group then pays on every open. The largest zones hold 556 live tiles.
    #
    # UNITS ARE IN THE NAMES. `tiles_skipped` counts TILES and the refusal totals count PIXELS; the
    # earlier `by_reason` said neither, and a reader comparing the two numbers would be comparing a
    # census to a census of something else.
    per_reason: dict[str, list[str]] = {}
    refused_px = dict.fromkeys(REFUSAL_REASONS, 0)
    obs_max = 0
    obs_px_with_any = 0
    mixed = 0
    inconsistent: list[str] = []
    # Pixels inside a published tile that the dataset never evaluated, because the read plan cropped
    # the chunk to the columns holding valid pixels. They are filled like the refused ones and are
    # NOT refusals, so folding them into a reason would misattribute them — but leaving them out
    # entirely makes the reason totals look short of the footprint they explain.
    not_evaluated_px = 0
    # Whether the radar rule was in force, pooled over the records. `no_radar: 0` otherwise means
    # two different things — no tile was refused for missing radar, or that rule was switched off —
    # and the count alone cannot say which. `allow_s2_only` defaults to FALSE in the library, and the
    # global campaign registers True, so under campaign settings the zero is structural. Reported so
    # nobody reads it as a finding about radar coverage.
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
            # A record that refuses nothing cannot explain a shard that holds nothing. Named rather
            # than dropped: it means the producer and this summary disagree, which is a defect in one
            # of them and must not read as a shard with no reason recorded.
            inconsistent.append(label)
            continue
        # THE INVARIANT: a fully refused shard refused every pixel the dataset EVALUATED. The three
        # reasons partition those pixels by construction, so a mismatch means the strips did not
        # cover what was loaded or a count was double-added — either way the pixel totals below are
        # wrong and saying so is worth more than a plausible number.
        #
        # Against the EVALUATED footprint, not the tile: the read plan crops a chunk to the columns
        # holding valid pixels, and comparing cropped counts against the whole tile flagged every
        # cropped shard as a defect. The cropped-out columns are accounted for separately below —
        # they are unevaluated, which is a third thing from refused and from embedded.
        eligible = int(record.get("eligible_px") or 0)
        if eligible and total != eligible:
            inconsistent.append(label)
        # A record predating this field carries no `chunk_px`; `eligible_px` is then the whole tile
        # by construction, so the difference is zero rather than unknown.
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
    # the shards and their lengths sum to the recorded total. `shards_mixed` says how many had more
    # than one reason, which is what a bare dominant label would hide.
    summary["shards_by_reason"] = {r: sorted(v) for r, v in sorted(per_reason.items())}
    summary["shards_mixed"] = mixed
    # HOW THIN, pooled over the refused shards. Per-shard depth would answer a question nobody has
    # once the mosaic is deleted, and it is what made the record large.
    summary["s2_obs_at_refused"] = {"max": obs_max, "px_with_any": obs_px_with_any}
    summary["unrecorded"] = [label for label in skipped_list if label not in records]
    # Emitted only when non-zero, so the ordinary uncropped year does not carry a field of zeros —
    # and so its presence is itself the signal that some tiles were cropped. Together with the
    # per-reason totals this accounts for the whole filled footprint: refused + never evaluated.
    if not_evaluated_px:
        summary["not_evaluated_px"] = not_evaluated_px
    # "enforced" / "disabled" / "mixed" — never a bare zero standing in for either. Absent when no
    # record says (every run before the field), which is honestly unknown rather than assumed.
    if radar_rule:
        summary["radar_refusal_rule"] = (
            "mixed" if len(radar_rule) > 1 else ("enforced" if next(iter(radar_rule)) else "disabled")
        )
    if inconsistent:
        summary["inconsistent"] = sorted(inconsistent)
    return summary


def summarise_radar_coverage(results: Iterable[dict]) -> dict | None:
    """Aggregate per-chunk radar counts into one year's coverage summary.

    ``None`` when ANY embedded tile reported no counts, not just when none did. A resumed
    tile is a synthetic success with no counters, and dropping it from both sides of the
    ratio leaves a figure describing only the tiles this run redid — which is whatever the
    previous attempt failed to finish, and so says nothing about the year it would be
    stored as. See the branch itself.

    Reduces what the actors already reported — the counts are computed where the
    observation maps and the embedded mask are both in memory, so nothing here reads
    pixels. ``None`` when no chunk reported the counts, which is how a run from an older
    build records no summary rather than a wrong one of zeros.

    Percentages are of EMBEDDED area, not of the zone: a zone is mostly ocean and mostly
    unembedded, so a fraction of the grid would be dominated by area no radar was ever
    expected over and would say nothing about the data.
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
        # In THIS loop, for the reason stated two fields below: `results` is an Iterable and
        # a generator is exhausted by the time any second pass runs, which would silently
        # report zero fully-free tiles rather than fail.
        if r["s1_free_pixels"] and r["s1_free_pixels"] == r.get("valid_pixels"):
            fully_free += 1
    if silent:
        # A RESUMED tile is a synthetic success carrying no counters, and dropping it from
        # both sides of the ratio still leaves a figure — computed over the tiles this run
        # happened to redo, then written on the year's provenance and logged as the year's.
        # A resume's redone set is whatever the previous attempt failed to finish, which
        # has no relationship to the year's radar coverage, so the number would be wrong by
        # an unknowable margin and carry nothing saying so. No summary is the honest
        # answer; the per-pixel observation counts in the store remain the authority.
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
        # The optical counterpart, on the same denominator so the two are directly
        # comparable. There is no optical-FREE figure: a tile with no valid optical pixel is
        # a skip and never produces a result, so `optical_skips` is where that case lives.
        "s2_thin_px": s2_thin,
        "s2_thin_pct": round(100.0 * s2_thin / embedded, 3),
        # Reported by the actors, because the fill applies the STORE's rule and not the
        # module default — stamping the constant here would label the year with a threshold
        # no pixel was actually judged against. Captured in the loop above rather than by a
        # second pass, because `results` is an Iterable and this function consumes it once.
        "s2_thin_below_obs": thin_below or OPTICAL_MIN_OBS,
        "chunks_reporting": reported,
        # Where, coarsely: a tile that is ENTIRELY radar-free localises the gap without
        # storing a per-tile grid, and distinguishes a concentrated absence (whole tiles,
        # e.g. an ice margin) from a diffuse one (a swath edge crossing many tiles). Exact
        # locations are already in the store's per-pixel observation-count arrays.
        "tiles_fully_s1_free": fully_free,
        "tiles_reporting": reported,
    }
