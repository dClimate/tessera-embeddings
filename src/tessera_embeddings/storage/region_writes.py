"""Region-write geometry: chunk alignment, padding, and disjointness checks.

Pure data-transform helpers shared by the region-overwrite paths in
:mod:`tessera_embeddings.storage.zarr_store`. Everything here operates on
in-memory xarray/dask/zarr objects and the opened ``existing`` store view — none
of it opens repos, sessions, or commits. The orchestration (open repo, fork,
merge, commit) lives next to the public ``write_region`` / ``write_regions``
wrappers in ``zarr_store``; this module is the geometry those wrappers lean on.
"""

import itertools
from typing import Any, cast

import dask.array as da
import icechunk
import numpy as np
import xarray as xr
import zarr
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph


def _read_zarr_block(zarr_arr: zarr.Array, region: tuple[slice, ...]) -> np.ndarray:
    """Read one chunk-block out of a raw zarr array. The unit of work per task."""
    # zarr v3 stubs type __getitem__ as a wide union; a basic-slice index always
    # returns an ndarray at runtime.
    return zarr_arr[region]  # type: ignore[return-value]


def _slab_array_from_zarr(
    zarr_arr: zarr.Array,
    slices: tuple[slice, ...],
    name: str,
    *,
    band_single_chunk: bool,
) -> da.Array:
    """Build a Dask array reading ONLY the zarr chunks overlapping ``slices``.

    The store's zarr layer enumerates every ``(time, y, x[, band])`` chunk —
    millions of keys on a continental master. ``xr.open_zarr`` + ``isel`` leaves
    that whole layer reachable under the slice, so the shell each region item
    backfills drags the master's entire chunk layer and any cull/optimize over it
    is ``O(total store chunks)`` — the scaling that OOM'd the scheduler as the
    master's time axis grew (independent of how many dates a batch writes).

    Instead we enumerate, in closed form, only the chunk blocks the ``slices``
    touch and emit one read task per block. The resulting graph is
    ``O(slab chunks)``, independent of store size, and sits on a fresh HLG with
    no parent dependency — there is no whole-store layer left to cull.

    ``band_single_chunk`` collapses the band axis (innermost, for a 4-D store)
    into one chunk. The reflectance region path is 3-D, so callers pass ``False``
    and every axis is split on the store's own chunk grid.
    """
    ndim = zarr_arr.ndim
    band_axis = ndim - 1 if (band_single_chunk and ndim == 4) else None

    # Per-dim list of (source_slice, length) for each output block along that axis.
    dim_blocks: list[list[tuple[slice, int]]] = []
    for axis in range(ndim):
        sl = slices[axis]
        start = sl.start or 0
        stop = zarr_arr.shape[axis] if sl.stop is None else sl.stop
        if axis == band_axis:
            dim_blocks.append([(slice(start, stop), stop - start)])
            continue
        chunk = zarr_arr.chunks[axis]
        blocks: list[tuple[slice, int]] = []
        pos = start
        while pos < stop:
            chunk_lo = (pos // chunk) * chunk
            chunk_hi = min(chunk_lo + chunk, zarr_arr.shape[axis])
            lo, hi = max(pos, chunk_lo), min(stop, chunk_hi)
            blocks.append((slice(lo, hi), hi - lo))
            pos = hi
        dim_blocks.append(blocks)

    out_chunks = tuple(tuple(length for _, length in axis_blocks) for axis_blocks in dim_blocks)
    dsk: dict[tuple, tuple] = {}
    for block_idx in itertools.product(*(range(len(axis_blocks)) for axis_blocks in dim_blocks)):
        region = tuple(dim_blocks[axis][block_idx[axis]][0] for axis in range(ndim))
        dsk[(name, *block_idx)] = (_read_zarr_block, zarr_arr, region)
    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=[])  # type: ignore[arg-type]
    return da.Array(hlg, name, out_chunks, dtype=zarr_arr.dtype)


def _store_chunk_sizes(existing: xr.Dataset, dims: "tuple[str, ...]") -> dict[str, int]:
    """Read the on-disk chunk size per dimension from the existing store.

    The store is authoritative for chunking (config may have drifted), so we
    read it off the opened arrays rather than INGEST_CHUNKS. All data_vars in
    our stores share identical chunking on the spatial/temporal dims; we assert
    that here so the single widened region we compute is valid for every var.
    """
    sizes: dict[str, int] = {}
    for dim in dims:
        for var in existing.data_vars:
            chunks = existing[var].chunks
            if chunks is None or dim not in existing[var].dims:
                continue
            axis = existing[var].dims.index(dim)
            size = chunks[axis][0]  # first block = nominal chunk size
            if dim in sizes and sizes[dim] != size:
                raise ValueError(
                    f"Store data_vars disagree on chunk size for {dim!r}: {sizes[dim]} vs {size}. "
                    "Region writes require uniform chunking across variables."
                )
            sizes[dim] = size
    return sizes


def _assert_zarr_layout_matches_dims(zarr_arr: zarr.Array, var: xr.DataArray, name: str) -> None:
    """Assert the raw zarr array's physical layout matches the xarray view's.

    The zarr-direct shell indexes ``zarr_arr`` POSITIONALLY by axis, while the
    slice bounds and chunk grid are computed from xarray's *named* dims. The two
    only line up if the array's physical axis order and per-axis chunk grid match
    the xarray view's dim order and chunking. They do for stores written through
    this package (xarray records ``_ARRAY_DIMENSIONS`` in dim order, ``open_zarr``
    preserves it, and both views share one snapshot) — but nothing enforces it, so
    a transposed write, a per-var dim permutation, or a chunk grid sourced from
    config instead of the store would silently slice the wrong axes and corrupt the
    backfill. Fail loudly here instead.
    """
    dims = tuple(str(d) for d in var.dims)
    if zarr_arr.ndim != len(dims):
        raise ValueError(
            f"Zarr array {name!r} has {zarr_arr.ndim} axes but the store view has dims {dims}; "
            "the zarr-direct padding shell cannot map named dims onto physical axes."
        )
    # zarr v3 records dim names in physical-axis order; check they match the view's.
    # (v2 metadata has no dimension_names — getattr yields None, skipping the check.)
    zarr_dim_names = getattr(zarr_arr.metadata, "dimension_names", None)
    if zarr_dim_names is not None and tuple(zarr_dim_names) != dims:
        raise ValueError(
            f"Zarr array {name!r} axis order {tuple(zarr_dim_names)} does not match store view dims {dims}; "
            "the zarr-direct padding shell indexes axes positionally and would slice the wrong axes."
        )
    # Per-axis nominal chunk size must agree between the raw array (used to build the
    # shell blocks) and the xarray view (used to widen the region).
    view_chunks = var.chunks
    if view_chunks is not None:
        for axis, dim in enumerate(dims):
            zarr_chunk = zarr_arr.chunks[axis]
            view_chunk = view_chunks[axis][0]  # first block = nominal chunk size
            if zarr_chunk != view_chunk:
                raise ValueError(
                    f"Chunk grid for {name!r} dim {dim!r} disagrees: zarr {zarr_chunk} vs store view "
                    f"{view_chunk}. The widened region (from the view) would not align to the shell blocks."
                )


def _match_region_shapes(
    existing: xr.Dataset,
    data: xr.Dataset,
    normalized: dict[str, slice],
    region: dict[str, slice],
) -> xr.Dataset:
    """Transpose each var to store dim order and assert it covers exactly the region.

    ``region`` is the caller's original (for the error message); ``normalized``
    holds the resolved integer bounds. A written variable must match the region's
    length along every region dim and the store's full length along the rest —
    dask/NumPy broadcasting would otherwise silently repeat a smaller (or missing
    one of the selected axes) array across the whole region instead of rejecting
    it. Returns a dataset with every var transposed to the store's dim order so a
    transposed input can't overlay onto the wrong axes.
    """
    matched: dict[str, xr.DataArray] = {}
    for var in data.data_vars:
        name = str(var)
        dims = existing[name].dims  # store dimension order is authoritative
        incoming = data[name].transpose(*dims)
        expected = tuple(
            (normalized[d].stop - normalized[d].start) if d in normalized else existing.sizes[d] for d in dims
        )
        if incoming.shape != expected:
            raise ValueError(
                f"Region data for {name!r} has shape {incoming.shape}, expected {expected} for region {region}."
            )
        matched[name] = incoming
    return xr.Dataset(matched, coords=data.coords)


def _blend_block(
    zarr_arr: zarr.Array,
    store_slices: tuple[slice, ...],
    incoming_block: np.ndarray,
    dst_slices: tuple[slice, ...],
) -> np.ndarray:
    """Read one boundary store chunk and overlay the incoming overlap window onto it.

    The unit of work for a grid chunk that straddles the region edge: the store
    chunk is read whole (so the pad cells outside the region round-trip unchanged)
    and the incoming sub-window is written over the cells the region covers
    (``dst_slices``, positions relative to the chunk). Incoming wins on overlap.
    The read is copied because the zarr read may alias a buffer we must not mutate.
    """
    base = np.array(zarr_arr[store_slices], copy=True)
    base[dst_slices] = incoming_block
    return base


def _grid_tiled_region_source(
    zarr_arr: zarr.Array,
    incoming: da.Array,
    dims: "tuple[str, ...]",
    normalized: dict[str, slice],
    widened: dict[str, slice],
    chunk_sizes: dict[str, int],
    store_sizes: dict[str, int],
    name: str,
) -> da.Array:
    """Build a Dask array whose blocks ARE the store grid chunks over ``widened``.

    The write source for one variable, constructed in closed form as exactly one
    HLG task per output grid chunk so the source lands block-for-chunk on the store
    with no realignment. Each grid chunk falls into one of three cases against the
    (pre-widening) ``normalized`` region:

    * **Pure pad** — no overlap with the region: read that store chunk straight
      from the raw zarr array. Backfills a widening cell unchanged.
    * **Fully inside** — the whole grid chunk lies within the region: reference the
      incoming block covering it directly, no store read.
    * **Boundary** — partial overlap: read the store chunk and overlay the incoming
      overlap window on top (:func:`_blend_block`), preserving read-modify-write
      semantics. Only boundary chunks touch the store, so the store IO stays on the
      region's perimeter — the same frame-only read the prior edge-strip approach
      gave, without the off-grid intermediate a downstream rechunk would undo.

    ``widened``'s origin is chunk-aligned by construction, so grid chunks tile
    ``[widened.start, widened.stop)`` at the store chunk size, the last possibly
    short (store edge). ``incoming`` is indexed by ``coord - normalized.start``; the
    per-overlap window is forced to a single block (``rechunk(-1)``) so it is one
    dependency key per inside/boundary task. The incoming array is a genuine lazy
    dependency (its read graph composes in via :func:`HighLevelGraph.from_collections`);
    nothing is materialized at build time.

    The graph sits on a fresh HLG: the only parent layers are the incoming source's
    own (unavoidable — it is the data being written), never the master's whole
    chunk layer, so the source stays ``O(grid chunks)`` in the master's size.
    """
    # Per axis: the grid-chunk start coordinates (absolute store coords), the chunk
    # size, the widened stop (to clamp a short final chunk), and the region bounds.
    # A dim absent from the region spans its full axis and has no pad/overlap split.
    axis_bounds: list[list[int]] = []
    axis_meta: list[tuple[int, int, int, int]] = []  # (chunk, widened_stop, region_lo, region_hi)
    for dim in dims:
        cs = chunk_sizes[dim]
        if dim in widened:
            w_start, w_stop = widened[dim].start, widened[dim].stop
            r_lo, r_hi = normalized[dim].start, normalized[dim].stop
        else:
            w_start, w_stop = 0, store_sizes[dim]
            r_lo, r_hi = 0, store_sizes[dim]
        axis_bounds.append(list(range(w_start, w_stop, cs)))
        axis_meta.append((cs, w_stop, r_lo, r_hi))

    out_chunks = tuple(
        tuple(min(b + cs, w_stop) - b for b in bounds)
        for bounds, (cs, w_stop, _r_lo, _r_hi) in zip(axis_bounds, axis_meta, strict=True)
    )

    dsk: dict[tuple, Any] = {}
    deps: list[da.Array] = []
    for block_idx in itertools.product(*(range(len(bounds)) for bounds in axis_bounds)):
        store_slices: list[slice] = []
        # Per axis: (overlap_lo, overlap_hi, chunk_lo, chunk_hi, region_lo) in absolute coords.
        ov: list[tuple[int, int, int, int, int]] = []
        for axis, bi in enumerate(block_idx):
            cs, w_stop, r_lo, r_hi = axis_meta[axis]
            b_lo = axis_bounds[axis][bi]
            b_hi = min(b_lo + cs, w_stop)
            store_slices.append(slice(b_lo, b_hi))
            ov.append((max(b_lo, r_lo), min(b_hi, r_hi), b_lo, b_hi, r_lo))
        store_slices_t = tuple(store_slices)

        pure_pad = any(o_lo >= o_hi for o_lo, o_hi, *_ in ov)
        fully_inside = all(o_lo == b_lo and o_hi == b_hi for o_lo, o_hi, b_lo, b_hi, _ in ov)

        if pure_pad:
            dsk[(name, *block_idx)] = (_read_zarr_block, zarr_arr, store_slices_t)
            continue

        # Incoming sub-window for this chunk's overlap, in incoming-local coords
        # (region origin maps to incoming index 0). Forced to a single block so the
        # task references one dependency key.
        inc_window = incoming[tuple(slice(o_lo - r_lo, o_hi - r_lo) for o_lo, o_hi, _b_lo, _b_hi, r_lo in ov)]
        inc_window = inc_window.rechunk(-1)
        deps.append(inc_window)
        inc_keys = list(da.core.flatten(inc_window.__dask_keys__()))
        inc_key = inc_keys[0]

        if fully_inside:
            dsk[(name, *block_idx)] = inc_key
        else:
            dst_slices = tuple(slice(o_lo - b_lo, o_hi - b_lo) for o_lo, o_hi, b_lo, _b_hi, _ in ov)
            dsk[(name, *block_idx)] = (_blend_block, zarr_arr, store_slices_t, inc_key, dst_slices)

    hlg = HighLevelGraph.from_collections(name, dsk, dependencies=deps)  # type: ignore[arg-type]
    return da.Array(hlg, name, out_chunks, dtype=zarr_arr.dtype)


def _pad_region_to_chunks(
    existing: xr.Dataset,
    data: xr.Dataset,
    region: dict[str, slice],
    group: zarr.Group,
) -> "tuple[xr.Dataset, dict[str, slice]]":
    """Widen ``region`` to whole-chunk bounds, backfilling the shell from the store.

    Icechunk's ``mode="r+"`` rejects a region whose edges fall mid-chunk
    (xarray ``allow_partial_chunks=False``), and ``align_chunks=True`` only
    remaps the *producer's* dask blocks — it never reads neighbouring on-disk
    data. So an unaligned region must be padded out to its enclosing chunk
    boundaries, with the newly-included-but-unchanged cells read back from the
    store so they round-trip unchanged through the write.

    Returns ``(padded, widened)``. When ``region`` is already chunk-aligned this
    returns ``(data, widened)`` with the data passed through unchanged — no store
    read. The ``padded`` block is built grid-tiled (see
    :func:`_grid_tiled_region_source`): one dask block per store chunk over the
    widened bounds, so it lands block-for-chunk on the store with no realignment.
    Each block is either an incoming block (cells the region covers), a store chunk
    read whole (a pure widening cell), or a boundary chunk read whole with the
    incoming overlap window overlaid — so the store IO stays on the region's
    *perimeter*, not its area.

    Every store read is taken **straight from the raw zarr ``group``** (the readonly
    session pinned to the write base snapshot, same view ``existing`` opens): one
    read task per chunk, on a fresh graph with no whole-store parent layer. Reading
    instead through the lazy ``existing`` view (``isel``) carries the master's
    *entire* chunk layer into every read — a graph sized ``O(total store chunks)``
    regardless of how much the batch writes, which the flow-runner cannot build at
    continental scale. The grid-tiled source keeps the graph ``O(grid chunks)``.
    """
    # Normalize region slices up front: resolve open bounds (slice(None)) to
    # concrete indices, reject non-unit steps, and reject empty slices, so the
    # offset arithmetic below is always well-defined.
    normalized: dict[str, slice] = {}
    for dim, sl in region.items():
        if sl.step not in (None, 1):
            raise ValueError(f"Region slice for {dim!r} must have step 1, got {sl.step!r}.")
        start, stop, _ = sl.indices(existing.sizes[dim])
        if start >= stop:
            raise ValueError(f"Region slice for {dim!r} is empty: {sl!r}.")
        normalized[dim] = slice(start, stop)

    # Assert every written variable covers exactly the region before any write.
    # Done for both the aligned and padded paths: on the aligned path
    # ``to_icechunk`` would otherwise reject (or, for a missing axis, broadcast)
    # a wrongly shaped array deep inside its writer with a cryptic error; here we
    # fail early with a clear message. Returns data matched to the store's dim
    # order (a transposed input would otherwise overlay onto the wrong axes).
    matched = _match_region_shapes(existing, data, normalized, region)

    chunk_sizes = _store_chunk_sizes(existing, tuple(normalized))

    widened: dict[str, slice] = {}
    needs_pad = False
    for dim, sl in normalized.items():
        cs = chunk_sizes[dim]
        size = existing.sizes[dim]
        new_start = (sl.start // cs) * cs
        new_stop = min(((sl.stop + cs - 1) // cs) * cs, size)
        widened[dim] = slice(new_start, new_stop)
        if new_start != sl.start or new_stop != sl.stop:
            needs_pad = True

    if not needs_pad:
        return matched, widened

    # Grid chunk sizes for EVERY store dim (a dim absent from the region is written
    # in full and must still tile on the store grid, not coalesce its whole axis).
    grid_sizes = _store_chunk_sizes(existing, tuple(str(d) for d in existing.sizes))
    store_sizes = {str(d): existing.sizes[d] for d in existing.sizes}

    padded_vars: dict[str, Any] = {}
    for name, incoming in matched.data_vars.items():
        name = str(name)
        dims = tuple(str(d) for d in existing[name].dims)  # store dimension order is authoritative
        # group[name] is a data-var array; zarr v3 stubs widen __getitem__ to
        # Array | Group, so narrow it for the grid tiler.
        zarr_arr = cast("zarr.Array", group[name])
        # The grid tiler indexes the raw zarr array POSITIONALLY by axis, but the
        # slice bounds/widened/chunk grid are all keyed by xarray's named dims. That
        # crossing is only safe if the named-dim order and per-axis chunk grid match
        # the raw array's physical layout. They do today (xarray writes
        # _ARRAY_DIMENSIONS in dim order and open_zarr preserves it; both views come
        # from the same snapshot), but it's an unasserted invariant — a transposed
        # write, mixed per-var dim orders, or sourcing chunk sizes from config would
        # silently slice the wrong axes / mis-align the backfill. Assert it so that
        # failure mode is loud, not silent.
        _assert_zarr_layout_matches_dims(zarr_arr, existing[name], name)
        block = _grid_tiled_region_source(
            zarr_arr,
            incoming.transpose(*dims).data,
            dims,
            normalized,
            widened,
            grid_sizes,
            store_sizes,
            f"gridsrc-{name}-{tokenize(zarr_arr, widened, normalized)}",
        )
        padded_vars[name] = (dims, block)

    # Coords come from the lazy store view (authoritative). They're 1-D and dropped
    # before the region write anyway, but carrying them keeps the dataset
    # well-formed. Slice only the coordinate variables: ``existing.isel(widened)``
    # would index every data var too — building (and immediately discarding) a
    # per-region dask slice of the master's whole chunk layer for each one. Slicing
    # the coords-only dataset never touches the heavy data-var graph.
    coords = existing.coords.to_dataset().isel(widened).coords
    padded = xr.Dataset(padded_vars, coords=coords)
    return padded, widened


def _drop_region_coords(data: xr.Dataset, region_dims: "set[str]") -> xr.Dataset:
    """Drop coords xarray won't accept on a region write.

    xarray rejects writing a region-dim's own coordinate (the store's coords are
    authoritative), and a region write requires every written variable to share
    a dim with the region — so coords spanning only non-region dims must go too.
    """
    drop = [name for name in data.coords if name in region_dims or not region_dims.intersection(data[name].dims)]
    return data.drop_vars(drop)


def _aligned_region_sources(
    existing: xr.Dataset,
    region_items: "list[tuple[xr.Dataset, dict[str, slice]]]",
    fork: "icechunk.ForkSession",
    group: zarr.Group,
) -> "tuple[list[da.Array], list[zarr.Array], list[tuple[slice, ...]], list[dict[str, slice]]]":
    """Build the ``(sources, targets, regions, widened)`` lists for one distributed store.

    Each ``(data, region)`` is padded to whole-chunk bounds (reusing
    :func:`_pad_region_to_chunks`'s read-modify-write backfill for unaligned edges),
    then every written variable is transposed to the store's dim order and rechunked
    to the store's on-disk chunk grid so each dask block maps onto whole Zarr chunks
    — the alignment ``write_region`` gets from ``align_chunks=True``, done explicitly
    here because the raw ``store_dask`` path performs no realignment. The unaligned
    path returns an already grid-tiled source, so the ``rechunk`` is a no-op there;
    on the aligned fast path it tiles the pass-through incoming onto the grid. Either
    way the widened region's start is chunk-aligned by construction, so the source
    lands block-for-chunk on the store.

    Targets are opened against ``fork.store`` (the pickleable fork shipped to
    workers); the same variable appearing in several items yields several target
    entries pointing at the same array, which ``store_dask`` writes into the
    distinct, non-overlapping regions. ``widened`` returns each item's
    chunk-aligned region (one entry per item, for logging/inspection).
    """
    sources: list[da.Array] = []
    targets: list[zarr.Array] = []
    regions: list[tuple[slice, ...]] = []
    widened_per_item: list[dict[str, slice]] = []
    # Chunk sizes for EVERY store dim, not just the region's: a dim omitted from
    # a region is written in full, and it must still land on the store chunk grid
    # rather than coalescing the whole axis into one dask block (a full-axis block
    # would destroy store-grid parallelism and can OOM a worker on large stores).
    chunk_sizes = _store_chunk_sizes(existing, tuple(str(d) for d in existing.sizes))
    for data, region in region_items:
        padded, widened = _pad_region_to_chunks(existing, data, region, group)
        widened_per_item.append(widened)
        for name in padded.data_vars:
            name = str(name)
            dims = existing[name].dims  # store dimension order is authoritative
            arr = padded[name].transpose(*dims).data
            grid = tuple(chunk_sizes.get(str(d), existing.sizes[d]) for d in dims)
            # The blended insert + backfilled-shell block, rechunked to the grid.
            aligned = da.from_array(arr, chunks=grid) if not isinstance(arr, da.Array) else arr.rechunk(grid)
            sources.append(aligned)
            targets.append(zarr.open_array(fork.store, path=name, mode="a"))
            regions.append(tuple(widened.get(str(d), slice(None)) for d in dims))
    return sources, targets, regions, widened_per_item


def _assert_regions_chunk_disjoint(widened_per_item: "list[dict[str, slice]]") -> None:
    """Reject items whose widened (chunk-aligned) regions share a Zarr chunk.

    ``store_dask`` merges item changesets with no conflict resolution, so two
    items overlapping in chunk space race (last-writer-wins) and silently drop
    data. The widened regions are chunk-aligned rectangular boxes, so chunk-space
    overlap is exactly index-space overlap: two boxes are disjoint iff separated
    along at least one dim. A dim absent from a region spans the full axis, so it
    never separates — overlap there is decided by the other dims.
    """
    for i in range(len(widened_per_item)):
        for j in range(i + 1, len(widened_per_item)):
            a, b = widened_per_item[i], widened_per_item[j]
            separated = False
            for dim in set(a) | set(b):
                sa, sb = a.get(dim), b.get(dim)
                if sa is None or sb is None:
                    continue  # full-axis on this dim → cannot separate the boxes
                if sa.stop <= sb.start or sb.stop <= sa.start:
                    separated = True
                    break
            if not separated:
                raise ValueError(
                    f"write_regions items {i} and {j} are not chunk-disjoint: widened regions "
                    f"{a} and {b} share a Zarr chunk. Batched regions must not overlap in chunk space "
                    "(the merge does no conflict resolution, so an overlap would silently drop data)."
                )
