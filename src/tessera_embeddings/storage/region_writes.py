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
    store and the incoming values overlaid on top.

    Returns ``(padded, widened)``. When ``region`` is already chunk-aligned this
    returns ``(data, region)`` unchanged — no store read.

    The shell is read **straight from the raw zarr ``group``** (the readonly
    session pinned to the write base snapshot, same view ``existing`` opens), via
    :func:`_slab_array_from_zarr`: one read task per overlapping chunk block, on a
    fresh graph with no whole-store parent layer. ``existing.isel(widened)`` would
    instead carry the master's *entire* time-chunk layer into every shell — a
    graph sized ``O(master time * spatial)``, independent of how many dates the
    batch writes — which is what OOM'd the scheduler as the master grew. A
    positional ``setitem`` then overlays the incoming data on the grid-chunked
    shell.

    Note this keeps every overlapped chunk — including interior chunks fully
    covered by the region — dependent on the store read, so a large unaligned
    write fetches the whole widened slab, not just its edge chunks. That extra IO
    is bounded by the padding (at most one chunk per region face) and is
    acceptable at current tile sizes; reading only the boundary chunks would need
    an edge-concatenation rebuild and is deferred.
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

    padded_vars: dict[str, Any] = {}
    for name, incoming in matched.data_vars.items():
        name = str(name)
        dims = existing[name].dims  # store dimension order is authoritative
        # group[name] is a data-var array; zarr v3 stubs widen __getitem__ to
        # Array | Group, so narrow it for the slab builder.
        zarr_arr = cast("zarr.Array", group[name])
        # The shell is built by indexing the raw zarr array POSITIONALLY by axis,
        # but slab_slices/widened/chunk grid are all keyed by xarray's named dims.
        # That crossing is only safe if the named-dim order and per-axis chunk grid
        # match the raw array's physical layout. They do today (xarray writes
        # _ARRAY_DIMENSIONS in dim order and open_zarr preserves it; both views come
        # from the same snapshot), but it's an unasserted invariant — a transposed
        # write, mixed per-var dim orders, or sourcing chunk sizes from config would
        # silently slice the wrong axes / mis-align the backfill. Assert it so that
        # failure mode is loud, not silent.
        _assert_zarr_layout_matches_dims(zarr_arr, existing[name], name)
        # Per-axis slices for the widened region in store dim order; a dim absent
        # from the region spans its full axis.
        slab_slices = tuple(widened[d] if d in widened else slice(None) for d in dims)
        # Shell read straight from zarr: one task per overlapping chunk block on a
        # fresh graph, sized by the widened slab — NOT existing.isel(widened),
        # which would carry the master's whole time-chunk layer into every item.
        shell = _slab_array_from_zarr(
            zarr_arr, slab_slices, f"shell-{name}-{tokenize(zarr_arr, slab_slices)}", band_single_chunk=False
        )
        idx = tuple(
            slice(normalized[d].start - widened[d].start, normalized[d].stop - widened[d].start)
            if d in normalized
            else slice(None)
            for d in dims
        )
        shell[idx] = incoming.data  # positional overlay; incoming wins
        padded_vars[name] = (dims, shell)

    # Coords come from the lazy store view (authoritative). They're 1-D and dropped
    # before the region write anyway, but carrying them keeps the dataset
    # well-formed; isel on coords is cheap and (unlike the data-var shell) never
    # touches the heavy chunk layer.
    padded = xr.Dataset(padded_vars, coords=existing.isel(widened).coords)
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
    :func:`_pad_region_to_chunks`'s read-modify-write shell for unaligned edges),
    then every written variable is transposed to the store's dim order and
    rechunked to the store's on-disk chunk grid so each dask block maps onto whole
    Zarr chunks — the alignment ``write_region`` gets from ``align_chunks=True``,
    done explicitly here because the raw ``store_dask`` path performs no
    realignment. The widened region's start is chunk-aligned by construction, so a
    grid-rechunked source lands block-for-chunk on the store.

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
