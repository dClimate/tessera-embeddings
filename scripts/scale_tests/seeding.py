"""Multi-group empty-store seeding (the primitive the library still lacks).

The library's ``create_empty_store_from_coords`` opens the *root* group with
``mode="w"`` and so cannot seed sibling groups (it clobbers). These helpers do
the group-aware seed the 120-UTM-zone layout needs: ``require_group`` per zone,
schema-only data arrays (no chunk objects written — creation cost is independent
of extent, ADR D1), fully-written 1-D coordinate arrays, and per-group attrs.

This is script code that *demonstrates* the needed primitive; the real one will
land in ``tessera_embeddings.storage`` once the tests justify the design.
"""

from __future__ import annotations

import dataclasses

import icechunk
import numpy as np
import zarr

from scale_tests import variants as V
from scale_tests.zone_geometry import YEARS, MockZone, coords
from tessera_embeddings.storage.zarr_store import TIME_ENCODING


@dataclasses.dataclass(frozen=True)
class ArraySpec:
    """One data-var array to create schema-only (no chunk data written)."""

    name: str
    create_kwargs: dict


@dataclasses.dataclass(frozen=True)
class GroupSpec:
    """One zone group: its coordinate arrays, data-var schemas, and attrs."""

    coords: dict[str, np.ndarray]
    arrays: list[ArraySpec]
    attrs: dict


def embedding_group_spec(
    zone: MockZone,
    variant: V.Variant,
    *,
    years: tuple[int, ...] = YEARS,
    extra_attrs: dict | None = None,
) -> GroupSpec:
    """Build the ``embeddings`` + ``scales`` group schema for one zone/variant.

    Arrays span the *full* year axis up front (pre-allocation, ADR D1); unwritten
    years cost nothing until filled. ``years_complete`` starts empty and is
    advanced by the writer in the same commit as each year's data.
    """
    nt, ny, nx = len(years), zone.height, zone.width
    emb_shape = (nt, ny, nx, V.BAND)
    scl_shape = (nt, ny, nx)
    attrs = {
        "crs": zone.epsg,
        "years_complete": [],
        "dataset_version": "scale-test",
        "variant": variant.name,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    return GroupSpec(
        coords=coords(zone, years),
        arrays=[
            ArraySpec("embeddings", V.embeddings_array_kwargs(variant, emb_shape)),
            ArraySpec("scales", V.scales_array_kwargs(variant, scl_shape)),
        ],
        attrs=attrs,
    )


def seed_groups(repo: icechunk.Repository, groups: dict[str, GroupSpec], *, commit_msg: str) -> str:
    """Seed one or more groups into ``repo`` in a single commit; return its id.

    Data arrays are created schema-only (metadata, zero chunk objects).
    Coordinate arrays are written in full; ``time`` is stored as int64 ns with
    :data:`TIME_ENCODING` (matching the library's create path).
    """
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")  # "a": create root if new, never clobber
    for gname, spec in groups.items():
        group = root.require_group(gname)
        for arr in spec.arrays:
            group.create_array(arr.name, **arr.create_kwargs)
        for cname, cvals in spec.coords.items():
            if cname == "time":
                tint = np.asarray(cvals, dtype="datetime64[ns]").astype("int64")
                time_arr = group.create_array("time", data=tint, chunks=(len(tint),), dimension_names=("time",))
                time_arr.attrs.update(TIME_ENCODING)
            else:
                vals = np.asarray(cvals)
                group.create_array(cname, data=vals, chunks=(len(vals),), dimension_names=(cname,))
        group.attrs.update(spec.attrs)
    return session.commit(commit_msg)
