"""Global embeddings store: seed the 120 zone groups; open/create the repo.

The composition layer for the global campaign (ADR-008). One Icechunk repo holds
120 Zarr groups (one per UTM zone); this module seeds them as pre-allocated,
metadata-only empty stores — full 2017-2025 time axis, per-zone coordinates and
CRS, ``GLOBAL`` sharded array schemas — so the campaign can then fill years as
region/shard writes without ever resizing (D1).

Lives above the low-level primitives on purpose: it pulls together
:mod:`~tessera_embeddings.storage.zone_grid` (the grid), the ``GLOBAL`` layout
(:mod:`~tessera_embeddings.config.store_layout`), the coord/schema writers in
:mod:`~tessera_embeddings.storage.empty_store`, and GeoZarr convention attrs
(:func:`~tessera_embeddings.inference.conventions.build_convention_attrs`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.store_layout import GLOBAL, StoreLayout
from tessera_embeddings.inference.conventions import build_convention_attrs
from tessera_embeddings.storage.empty_store import _write_coord_arrays
from tessera_embeddings.storage.zarr_store import _create_storage, global_store_config
from tessera_embeddings.storage.zone_grid import (
    CAMPAIGN_YEARS,
    ZONE_SCHEME,
    ZONES,
    ZoneSpec,
    calendar_year_times,
    easting_coords,
    northing_coords,
)


def create_global_repo(
    store_path: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    region: str | None = None,
) -> icechunk.Repository:
    """Create the global-store repo with :func:`global_store_config` and persist it.

    ``save_config`` bakes the manifest-split + preload config into the repo so
    every later open (and every forked worker) inherits it without re-passing.
    """
    repo = icechunk.Repository.create(
        _create_storage(store_path, get_credentials=get_credentials, region=region),
        config=global_store_config(),
    )
    repo.save_config()
    return repo


def open_global_repo(
    store_path: str,
    *,
    get_credentials: Callable[[], icechunk.S3StaticCredentials] | None = None,
    region: str | None = None,
    max_concurrent_requests: int | None = None,
) -> icechunk.Repository:
    """Open the global-store repo with the global config layered on."""
    return icechunk.Repository.open(
        _create_storage(store_path, get_credentials=get_credentials, region=region),
        config=global_store_config(max_concurrent_requests),
    )


def create_layout_arrays(
    node: zarr.Group,
    layout: StoreLayout,
    variables: Iterable[str],
    sizes: dict[str, int],
) -> None:
    """Create schema-only arrays on ``node`` for each variable, sized from ``sizes``.

    The one loop that applies a :class:`StoreLayout` to a group — shared by the
    zone-group seeder here and the single-ROI schema creator in
    :mod:`tessera_embeddings.inference.assembly`, so the two write paths cannot
    diverge in how a layout becomes on-disk arrays. ``sizes`` maps dim name →
    extent (``{"time": ..., "northing": ..., "easting": ..., "band": ...}``).
    """
    for var in variables:
        array_layout = layout.for_var(var)
        shape = tuple(sizes[d] for d in array_layout.dims)
        node.create_array(var, **array_layout.create_kwargs(shape))


def _layout_band(layout: StoreLayout) -> int:
    """The band extent for a layout — the embeddings array's full-band chunk.

    Zone seeding requires a *sharded* layout (D3 — the whole global write path
    is shard-aligned) whose band axis is never split (D2), so the band chunk
    size IS the band count. Both are enforced here: an unsharded layout
    (e.g. ``SINGLE``, whose band chunk of 4 would otherwise silently seed
    4-band zones) and a band-split sharded layout are rejected loudly.
    """
    emb = layout.arrays["embeddings"]
    band = emb.chunks[-1]
    if emb.shards is None:
        raise ValueError(
            f"Layout {layout.name!r} is unsharded — zone seeding requires a sharded, "
            "full-band layout like GLOBAL (ADR-008 D2/D3)."
        )
    if emb.shards[-1] != band:
        raise ValueError(
            f"Layout {layout.name!r} splits the band axis (chunk {band} != shard {emb.shards[-1]}) — "
            "zone seeding requires full-band chunks (ADR-008 D2)."
        )
    return band


def _zone_attrs(
    spec: ZoneSpec, north: np.ndarray, east: np.ndarray, layout: StoreLayout, model_version: str | None
) -> dict:
    """Per-zone group attrs: campaign markers + GeoZarr conventions."""
    attrs: dict = {
        "crs": spec.crs,
        "years_complete": [],
        "zone_scheme": ZONE_SCHEME,
        "time_convention": "calendar_year",
        "layout": layout.name,
    }
    attrs.update(
        build_convention_attrs(
            epsg_code=spec.crs,  # canonical "EPSG:NNNNN" so proj:code is authority-qualified
            total_y=spec.height,
            total_x=spec.width,
            embedding_dim=_layout_band(layout),
            y_coords=north,
            x_coords=east,
            model_version=model_version,
        )
    )
    return attrs


def seed_zone_groups(
    repo: icechunk.Repository,
    specs: Iterable[ZoneSpec],
    *,
    years: tuple[int, ...] = CAMPAIGN_YEARS,
    layout: StoreLayout = GLOBAL,
    model_version: str | None = None,
    commit_msg: str | None = None,
) -> str:
    """Seed one or more zone groups (metadata-only) in a single commit.

    Each group gets the full ``years`` time axis, per-zone coordinate arrays, the
    ``layout``'s sharded array schemas (no chunk data — cost is independent of
    extent, D1), and per-zone attrs (CRS, ``years_complete: []``, zone scheme,
    conventions). Returns the commit snapshot id.
    """
    specs = list(specs)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    times = calendar_year_times(years)
    nt = len(times)
    band = _layout_band(layout)
    for spec in specs:
        node = root.require_group(spec.group_name)
        north = northing_coords(spec)
        east = easting_coords(spec)
        sizes = {"time": nt, "northing": spec.height, "easting": spec.width, "band": band}
        create_layout_arrays(node, layout, layout.arrays, sizes)
        _write_coord_arrays(node, {"time": times, "northing": north, "easting": east, "band": np.arange(band)})
        node.attrs.update(_zone_attrs(spec, north, east, layout, model_version))
    return session.commit(commit_msg or f"seed {len(specs)} zone group(s)")


def seed_all_zones(
    repo: icechunk.Repository,
    *,
    years: tuple[int, ...] = CAMPAIGN_YEARS,
    layout: StoreLayout = GLOBAL,
    model_version: str | None = None,
) -> str:
    """Seed all 120 UTM-zone groups in one commit."""
    return seed_zone_groups(
        repo, ZONES.values(), years=years, layout=layout, model_version=model_version, commit_msg="seed all 120 zones"
    )
