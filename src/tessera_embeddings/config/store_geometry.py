"""What a store's own metadata says about its geometry, and the chunk arithmetic every writer shares.

Its own module, apart from the presets in :mod:`~tessera_embeddings.config.store_layout`, because
the mosaic writer and the generic readers use these and sit inside the ingest code identity: a
new or changed preset must not move that identity (ADR 031 §5).

Everything here reads the store rather than the model, so one code path serves the band-last
v1.1 stores, which hold their arrays in the zone group itself, and the band-first v2 stores,
which hold them in a level group ``0`` beneath it (ADR 031 §1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import zarr

if TYPE_CHECKING:
    import xarray as xr

#: The Y and X names the presets spell, and the fallback for a group that declares none.
PRESET_SPATIAL_DIMS: tuple[str, str] = ("northing", "easting")

#: The child group a store keeps its 10 m arrays in, when it has one: the first level of a
#: multiscales pyramid. Coarser levels, once built, sit beside it as ``1``, ``2`` and so on.
LEVEL0 = "0"


def level0(group: zarr.Group) -> zarr.Group:
    """The group holding *group*'s 10 m arrays: its ``0`` child group if it has one, else itself.

    *group* is a zone group of a global store or the root of a single-area store. Its attributes
    (CRS, completed years, update records, ``proj:``, ``spatial:``, ``multiscales``) stay on
    *group*; only the arrays and their coordinates are reached through this.
    """
    child = group.get(LEVEL0)
    return child if isinstance(child, zarr.Group) else group


def dims_of(array: object) -> tuple[str, ...]:
    """*array*'s dimension names: xarray's on the variable, zarr's in its metadata; empty if it has none."""
    names = getattr(array, "dims", None) or getattr(getattr(array, "metadata", None), "dimension_names", None)
    return tuple(names or ())


def band_axis(array: zarr.Array | xr.DataArray) -> int | None:
    """The position of ``band`` among *array*'s dimensions, or ``None`` if it has no band axis.

    Read from the array's own dimension names, never inferred from the model or the store, so the
    band-last v1.1 store and a band-first v2 one are indexed by the same code.
    """
    names = dims_of(array)
    return names.index("band") if "band" in names else None


def spatial_dims_of(group: zarr.Group | xr.Dataset) -> tuple[str, str]:
    """The Y and X dimension names an existing output store uses.

    Its ``spatial:dimensions`` when it declares them, as every georeferenced store does; otherwise
    its required ``scales`` array, ``(time, Y, X)``, which a store with pixel-index coordinates
    still has; the presets' names for anything else (a mosaic, a malformed group), so an audit
    reports it rather than raises. Takes a zarr group, a zone group whose arrays sit in its level
    group, or the xarray dataset opened from one.
    """
    declared = cast("list[str] | None", group.attrs.get("spatial:dimensions"))
    if declared and len(declared) == 2:
        return declared[0], declared[1]
    holder = level0(group) if isinstance(group, zarr.Group) else group
    scales = holder.get("scales")
    dims = dims_of(scales) if scales is not None else ()
    return (dims[1], dims[2]) if len(dims) == 3 else PRESET_SPATIAL_DIMS


def clamp_chunks_and_shards(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Clamp nominal chunk/shard sizes to an array's ``shape``.

    Chunks are clamped to the shape; shards are clamped to the shape and then floored to a whole
    multiple of the clamped chunks (Zarr v3 requires shards to be exact chunk multiples), never
    below one chunk, so a store smaller than one nominal chunk/shard still creates cleanly. The
    single implementation of this geometry math: both
    :meth:`~tessera_embeddings.config.store_layout.ArrayLayout.create_kwargs` and the empty-store
    seeder call it.
    """
    # max(..., 1): a zero-extent axis is legal (a mosaic seeded with an EMPTY time axis, appended per date), but a
    # zero chunk edge is not — keep the nominal chunk so the axis grows into it.
    clamped = tuple(max(min(c, s), 1) for c, s in zip(chunks, shape, strict=True))
    if shards is None:
        return clamped, None
    clamped_shards = tuple(max(c, (min(sh, s) // c) * c) for sh, s, c in zip(shards, shape, clamped, strict=True))
    return clamped, clamped_shards
