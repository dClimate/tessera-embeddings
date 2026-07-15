"""Create 100%-empty (all-fill) stores over a grid extent and a set of dates.

Pre-allocate an Icechunk store with the correct shape, coordinates, chunking,
dtype, and root attrs for a given grid and set of dates — but with every pixel
the store's fill value (no data computed). Downstream population (a real ingest
region-write, an append, or a batch region-merge) then aligns to this grid
exactly.

The intended pairing: seed an empty **master** store here from the union of a
collection of stores' dates, then merge each store into it with a batch
region-merge (the process-parallel, no-Dask merge path landing in a stacked
follow-up PR). Seed such a master with ``time`` chunk size 1 so that merge's
disjointness invariant holds (every merged date lands in its own time chunk —
the merge validates this and raises otherwise).

**Why the direct zarr path is used for large stores.** The naive approach
(``dask.array.full → to_icechunk``) appears cheap but is not: zarr evaluates
every dask chunk to call ``all_equal(fill_value)`` before deciding to skip the
write, materializing all chunks on the local scheduler. At CONUS scale
(289707x461427 px, 4000-px chunks) that is ~8 500 spatial chunks x bands x
dates of ``np.full`` evaluation — measured at 750+ seconds on the flow runner.

Instead ``create_empty_store`` creates the icechunk repo and zarr arrays
directly, without any dask graph over the pixel dimensions. The data variables
are declared with shape+dtype+fill_value but zero chunks written; coordinate
arrays (time, northing, easting) are written as plain numpy (they are 1-D and
small). Root attrs are written to match what ``write_dataset``'s create path
produces, so the append and region-write paths read them back correctly.

**Fill value.** Integer stores use ``0`` as the nodata sentinel (the value both
ingest paths write for absent/masked pixels — ``amplitude_to_db`` emits 0 for
zero-amplitude; S2 masks with ``.where(any_valid, other=0)``; the S2 ``scl``
class 0 is "no data"), so an empty integer store is correctly all-zero. ``NaN``
is not representable in an integer dtype, so the fill is resolved per-dtype: 0
for integer, NaN for float. The batch region-merge resolves fill the same way,
so a merge copies only a store's real pixels.

Domain-layer rules apply: stdlib logging only (no orchestrator imports), storage
via the fsspec-backed helpers (no boto3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import fsspec
import numpy as np
import zarr

from tessera_embeddings.config.ingest import INGEST_CHUNKS
from tessera_embeddings.config.store_layout import clamp_chunks_and_shards
from tessera_embeddings.storage.zarr_store import TIME_ENCODING, _create_repo, compute_doy
from tessera_embeddings.utils import utcnow_iso

if TYPE_CHECKING:
    from icechunk import Repository

    from tessera_embeddings.ingest.roi import ROIMetadata
    from tessera_embeddings.storage.manifest import IngestManifest

logger = logging.getLogger(__name__)


def _fill_for_dtype(dtype: np.dtype) -> float | int:
    """Return the fill value for *dtype*: NaN for float, 0 for integer.

    ``0`` is the established nodata sentinel for the integer stores (see module
    docstring), and NaN is not representable in an integer dtype — a
    ``da.full(..., np.nan, dtype="uint16")`` would silently cast NaN to 0 with a
    RuntimeWarning. Resolving per-dtype keeps the value meaningful and the build
    warning-free.
    """
    return np.nan if np.issubdtype(dtype, np.floating) else 0


def daily_times(start: str, end: str) -> np.ndarray:
    """Return one ``datetime64[ns]`` timestamp per day in ``[start, end]`` (inclusive).

    A contiguous daily time axis over a date range. ``2025-01-01`` → ``2025-12-31``
    yields 365 timestamps. Both bounds are ``YYYY-MM-DD`` strings; ``end`` must not
    precede ``start``.

    A daily axis pre-seeds every day a later infill might land on, so a
    region-overwrite of any single date aligns to an existing coordinate without
    an append. To instead seed a store from the union of a collection of input
    stores' time axes, the batch region-merge's date-union helper owns that path.
    """
    start_d = np.datetime64(start, "D")
    end_d = np.datetime64(end, "D")
    if end_d < start_d:
        msg = f"end {end!r} precedes start {start!r}"
        raise ValueError(msg)
    days = np.arange(start_d, end_d + np.timedelta64(1, "D"), dtype="datetime64[D]")
    return days.astype("datetime64[ns]")


# Sentinel for "no explicit fill" so a caller can request an integer fill of 0
# without it being confused with "use the per-dtype default" — and so a None in
# a VarSpec means "resolve from dtype" rather than "no fill value".
_DEFAULT_FILL = object()


@dataclass(frozen=True)
class VarSpec:
    """Schema for one direct-created (all-fill) data variable.

    The contract a later real region-write must align to: dims, dtype, on-disk
    chunking, fill value, and zarr encoding. ``dims`` name axes that must each
    appear in the ``coords`` mapping passed to ``create_empty_store_from_coords``;
    ``chunks`` is the per-dim on-disk chunk size in the same order as ``dims``.

    ``fill_value`` defaults to the per-dtype fill (NaN for float, 0 for integer);
    pass an explicit value to override. ``serializer`` / ``compressors`` are
    forwarded to ``zarr.Group.create_array`` verbatim — the default ``"auto"``
    matches what xarray's zarr backend would pick, while a caller can pin e.g.
    a PCodec serializer + ``compressors=None`` for a float array. (Sharded
    layouts belong to the global store's ``ArrayLayout``/``StoreLayout`` path —
    see ``config.store_layout`` and ``storage.global_store``.)
    """

    dims: tuple[str, ...]
    dtype: np.dtype
    chunks: tuple[int, ...]
    fill_value: Any = _DEFAULT_FILL
    serializer: Any = "auto"
    compressors: Any = "auto"

    def resolved_fill(self) -> float | int:
        """The fill value to write: the explicit ``fill_value`` or the per-dtype default."""
        if self.fill_value is _DEFAULT_FILL:
            return _fill_for_dtype(self.dtype)
        return self.fill_value


def _write_coord_arrays(node: zarr.Group, coords: dict[str, np.ndarray]) -> None:
    """Write 1-D coordinate arrays onto a zarr group node.

    ``time`` is special-cased: stored as int64 nanoseconds with
    :data:`TIME_ENCODING` (matching ``write_dataset``); other coords are written
    as their values. Shared by the single-store and multi-group seed paths.
    """
    for name, values in coords.items():
        if name == "time":
            time_int = np.asarray(values, dtype="datetime64[ns]").astype("int64")
            time_arr = node.create_array("time", data=time_int, chunks=(len(time_int),), dimension_names=("time",))
            time_arr.attrs.update(TIME_ENCODING)
        else:
            arr = np.asarray(values)
            node.create_array(name, data=arr, chunks=(len(arr),), dimension_names=(name,))


def _write_group_schema(
    node: zarr.Group,
    coords: dict[str, np.ndarray],
    var_specs: dict[str, VarSpec],
    attrs: dict | None,
) -> None:
    """Create schema-only data arrays + coord arrays + attrs on a group node.

    Data variables are created with shape/chunks but no chunk data (all fill),
    so cost is independent of spatial extent.
    """
    for name, spec in var_specs.items():
        shape = tuple(len(coords[d]) for d in spec.dims)
        # One clamp implementation for on-disk geometry, shared with
        # ArrayLayout.create_kwargs (config.store_layout).
        chunks, _ = clamp_chunks_and_shards(shape, spec.chunks, None)
        node.create_array(
            name,
            shape=shape,
            chunks=chunks,
            dtype=spec.dtype,
            fill_value=spec.resolved_fill(),
            dimension_names=spec.dims,
            serializer=spec.serializer,
            compressors=spec.compressors,
        )
    _write_coord_arrays(node, coords)
    if attrs:
        node.attrs.update(attrs)


def create_empty_store_from_coords(
    store_path: str,
    *,
    coords: dict[str, np.ndarray],
    var_specs: dict[str, VarSpec],
    commit_msg: str,
    attrs: dict | None = None,
    repo: Repository | None = None,
) -> None:
    """Create an all-fill Icechunk store from explicit coords and var schemas.

    The low-level primitive the higher-level helpers build on: it creates the
    icechunk repo and the zarr arrays directly — data vars as schema only (no
    chunks written, so creation cost is independent of spatial extent),
    coordinate arrays written in full as 1-D numpy. No dask graph over the pixel
    dimensions, so none of the ``da.full -> to_icechunk`` fill-equality
    materialization the module docstring warns about.

    The store must not already exist. Populate it afterward with a real
    region-write / append, which aligns to this grid exactly.

    Args:
        store_path: Destination Icechunk store URI.
        coords: Mapping of coord name -> 1-D values. A ``"time"`` entry is
            special-cased: stored as int64 nanoseconds with ``TIME_ENCODING``
            (matching ``write_dataset``). Every dim named by a ``VarSpec`` must
            have a matching coord here.
        var_specs: Mapping of data-var name -> :class:`VarSpec`.
        commit_msg: Commit message for the single create commit.
        attrs: Optional root attrs. Written verbatim; pass ``None`` to write a
            store whose attrs the caller fills in a later commit.
        repo: Optional already-created Icechunk repository for ``store_path``.
            Pass this when the caller created the repo itself (e.g. via
            ``open_or_create_repo``) — icechunk forbids creating a repo twice in
            the same prefix. When given, cleanup-on-failure is skipped (the
            caller owns the repo lifecycle); when ``None``, the repo is created
            here and a partial store is removed on error.
    """
    for name, spec in var_specs.items():
        missing = [d for d in spec.dims if d not in coords]
        if missing:
            raise ValueError(f"VarSpec {name!r} names dims {missing} absent from coords {list(coords)}")
        if len(spec.dims) != len(spec.chunks):
            raise ValueError(f"VarSpec {name!r} has {len(spec.dims)} dims but {len(spec.chunks)} chunk sizes")

    owns_repo = repo is None
    if repo is None:
        repo = _create_repo(store_path)
    session = repo.writable_session("main")
    store = session.store
    try:
        root = zarr.open_group(store, mode="w")
        _write_group_schema(root, coords, var_specs, attrs)
        session.commit(commit_msg)
    except Exception:
        # Mirror write_dataset's cleanup_on_failure — delete partial store on
        # error, but only when we created the repo. A caller-supplied repo owns
        # its own lifecycle (and may already hold data we must not delete).
        if owns_repo:
            logger.warning("Empty store creation failed, cleaning up %s", store_path)
            try:
                fs = fsspec.filesystem(fsspec.utils.get_protocol(store_path))
                if fs.exists(store_path):
                    fs.rm(store_path, recursive=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up partial store %s: %s", store_path, cleanup_err)
        raise


def create_empty_store(
    store_path: str,
    *,
    roi: ROIMetadata,
    times: np.ndarray,
    var_dtypes: dict[str, np.dtype],
    tile_id: str,
    crs: str,
    chunks: dict[str, int] | None = None,
    baselines: dict[str, int] | None = None,
    manifest: IngestManifest | None = None,
) -> None:
    """Create a 100%-empty store over the ROI extent and given dates.

    Computes no pixels: every value is the dtype's fill (0 for integer, NaN for
    float). Creates the icechunk repo and zarr arrays directly — no dask graph
    over the pixel dimensions — so creation time is independent of the spatial
    extent. Coordinate arrays (time, northing, easting) are small 1-D numpy
    arrays and are written normally.

    Root attrs match what ``write_dataset``'s create path produces, so the
    append and region-write paths read them back correctly.

    The store must not already exist. Use a region-write / append (real ingest)
    or a batch region-merge to populate.

    Args:
        store_path: Destination Icechunk store URI.
        roi: ROI metadata (grid authority) from ``read_roi_metadata``.
        times: 1-D ``datetime64[ns]`` coordinate values — the dates to pre-seed.
        var_dtypes: Mapping of data-var name -> stored dtype.
        tile_id: Tile/ROI identifier for store metadata.
        crs: CRS authority code (e.g. ``"EPSG:32615"``); ``roi.native_crs``.
        chunks: Chunk sizes (``time``/``northing``/``easting``). Defaults to the
            OSS ``INGEST_CHUNKS`` so chunking matches a real ingest. A store that
            will receive a batch region-merge should use ``time`` chunk
            size 1 (the default does).
        baselines: Baseline map written to root attrs. Defaults to empty (no
            baseline correction is meaningful for an unpopulated store).
        manifest: Optional ingest manifest for provenance / append-safety.
    """
    chunks = chunks if chunks is not None else INGEST_CHUNKS
    times = np.asarray(times, dtype="datetime64[ns]")

    coords = roi.geobox.coordinates
    northing = coords["y"].values.astype("float64")
    easting = coords["x"].values.astype("float64")
    ny, nx = roi.height, roi.width
    nt = len(times)

    if (len(northing), len(easting)) != (ny, nx):
        raise ValueError(
            f"ROI geobox coords ({len(northing)}, {len(easting)}) disagree with ROI (height, width) = ({ny}, {nx})"
        )

    chunk_t = chunks["time"]
    chunk_y = min(chunks["northing"], ny)
    chunk_x = min(chunks["easting"], nx)

    logger.info(
        "Creating empty store %s: %d dates x %d x %d, vars=%s",
        store_path,
        nt,
        ny,
        nx,
        list(var_dtypes),
    )

    # Root attrs matching write_dataset's create path exactly.
    doy = compute_doy(times)
    store_attrs: dict = {
        "tile_id": tile_id,
        "baselines_applied": baselines or {},
        "doy": doy.tolist(),
        "created_at": utcnow_iso(),
        "last_appended": utcnow_iso(),
        "crs": crs,
    }
    if manifest is not None:
        store_attrs["_manifest"] = manifest.to_dict()

    create_empty_store_from_coords(
        store_path,
        coords={"time": times, "northing": northing, "easting": easting},
        var_specs={
            name: VarSpec(
                dims=("time", "northing", "easting"),
                dtype=dtype,
                chunks=(chunk_t, chunk_y, chunk_x),
            )
            for name, dtype in var_dtypes.items()
        },
        commit_msg=f"Create empty store with {nt} dates",
        attrs=store_attrs,
    )


__all__ = [
    "VarSpec",
    "create_empty_store",
    "create_empty_store_from_coords",
    "daily_times",
]
