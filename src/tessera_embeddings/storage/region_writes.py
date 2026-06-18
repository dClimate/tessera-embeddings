"""Region-write geometry: chunk alignment, padding, and disjointness checks.

Pure data-transform helpers shared by the region-overwrite paths in
:mod:`tessera_embeddings.storage.zarr_store`. Everything here operates on
in-memory xarray/dask/zarr objects and the opened ``existing`` store view — none
of it opens repos, sessions, or commits. The orchestration (open repo, fork,
merge, commit) lives next to the public ``write_region`` / ``write_regions``
wrappers in ``zarr_store``; this module is the geometry those wrappers lean on.
"""

from typing import Any

import dask.array as da
import icechunk
import xarray as xr
import zarr


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


def _strip_region(
    dims: "tuple[str, ...]",
    cur: dict[str, slice],
    pad_dim: str,
    *,
    side: str,
    widened: dict[str, slice],
) -> dict[str, slice]:
    """The ``isel`` indexer for a store edge strip on the ``pad_dim`` face.

    Along ``pad_dim`` the strip is the pad margin between the current (pre-pad)
    extent and the widened bound on the requested ``side`` (``"lo"`` / ``"hi"``).
    Along every other axis it spans whatever extent the growing block currently
    has: ``widened`` for axes already padded, the region extent for axes not yet
    padded, and the full axis for dims absent from the region (omitted, so ``isel``
    takes all). Matching the block's current span on the other axes is what lets a
    plain ``da.concatenate`` fill the corner cells from the store.
    """
    region: dict[str, slice] = {}
    for d in dims:
        if d == pad_dim:
            region[d] = slice(widened[d].start, cur[d].start) if side == "lo" else slice(cur[d].stop, widened[d].stop)
        elif d in cur:
            region[d] = cur[d]
    return region


def _pad_region_to_chunks(
    existing: xr.Dataset,
    data: xr.Dataset,
    region: dict[str, slice],
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

    Padding wraps the incoming block in store-read edge strips: for each unaligned
    face, a thin slab covering only the pad margin (``widened`` minus
    ``normalized`` on that face) is read from the store and ``da.concatenate``-d
    onto the block, one axis at a time. The interior — every cell the incoming
    data covers — is never read or copied; only the boundary frame touches the
    store. Concatenating axis by axis, with each strip spanning the
    already-extended extent on previously-padded axes, fills the corner cells from
    the store too. This is far cheaper to build than overlaying the incoming data
    onto a full-window shell via dask ``__setitem__`` (which rewrites the task
    graph over every chunk in the window): for a region inset within a few chunks,
    the cost scales with the perimeter, not the area.
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

    # Per-dim low/high pad margins (cells of widening on each face). A dim with
    # no margin contributes no edge strip.
    pad_lo = {d: normalized[d].start - widened[d].start for d in normalized}
    pad_hi = {d: widened[d].stop - normalized[d].stop for d in normalized}

    padded_vars: dict[str, Any] = {}
    for name, incoming in matched.data_vars.items():
        name = str(name)
        dims = tuple(str(d) for d in existing[name].dims)  # store dim order is authoritative
        block = incoming.transpose(*dims).data
        # Grow the incoming block to the widened bounds one axis at a time. After
        # axis k is padded the block already spans widened on axes < k, so the
        # strip read for axis k covers that extended extent — which backfills the
        # corner cells from the store without a separate corner read.
        cur = {d: normalized[d] for d in normalized}  # current extent per padded dim
        for axis, dim in enumerate(dims):
            if dim not in normalized:
                continue
            lo, hi = pad_lo[dim], pad_hi[dim]
            if lo:
                strip = existing[name].isel(_strip_region(dims, cur, dim, side="lo", widened=widened))
                block = da.concatenate([strip.transpose(*dims).data, block], axis=axis)
            if hi:
                strip = existing[name].isel(_strip_region(dims, cur, dim, side="hi", widened=widened))
                block = da.concatenate([block, strip.transpose(*dims).data], axis=axis)
            cur[dim] = widened[dim]  # this axis now spans the widened extent
        padded_vars[name] = (dims, block)

    # Coords come from the store's widened slab (authoritative); they're dropped
    # before the region write anyway, but carrying them keeps the dataset well-formed.
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
        padded, widened = _pad_region_to_chunks(existing, data, region)
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
