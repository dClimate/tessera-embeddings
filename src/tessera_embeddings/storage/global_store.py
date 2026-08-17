"""Global embeddings store: seed the 120 zone groups; open/create the repo.

The composition layer for the global campaign (ADR-008). One Icechunk repo holds
120 Zarr groups (one per UTM zone); this module seeds them as pre-allocated,
metadata-only empty stores — full 2017-2025 time axis, per-zone coordinates and
CRS, ``GLOBAL`` sharded array schemas — so the campaign can then fill years as
region/shard writes without ever resizing (D1).

Lives above the low-level primitives on purpose: it pulls together
:mod:`~tessera_embeddings.storage.zone_grid` (the grid), the ``GLOBAL`` layout
(:mod:`~tessera_embeddings.config.store_layout`), the coord/schema writers in
:mod:`~tessera_embeddings.storage.empty_store`, and GeoZarr convention attrs.

Convention placement follows the geoembeddings ``utm_zones`` layout: the encoder
and quantization provenance (``geoemb:``) is written ONCE on the root group
(:func:`~tessera_embeddings.inference.conventions.build_geoemb_root_attrs`, with
``spatial_layout="utm_zones"``), while each zone group carries only its own
``proj:``/``spatial:`` (``build_convention_attrs(..., include_geoemb=False)``) —
their CRS and grid differ by zone.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import cast

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.store_layout import GLOBAL, MONTH_COORD, MONTHS_IN_YEAR, StoreLayout
from tessera_embeddings.inference.conventions import build_convention_attrs, build_geoemb_root_attrs
from tessera_embeddings.storage.empty_store import _write_coord_arrays
from tessera_embeddings.storage.zarr_store import (
    TIME_ENCODING,
    _create_storage,
    global_store_config,
    read_time_values,
)
from tessera_embeddings.storage.zone_grid import (
    CAMPAIGN_YEARS,
    PIXEL_M,
    ZONE_SCHEME,
    ZoneSpec,
    calendar_year_times,
    easting_coords,
    northing_coords,
    year_of,
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


def _zone_attrs(spec: ZoneSpec, north: np.ndarray, east: np.ndarray, layout: StoreLayout) -> dict:
    """Per-zone group attrs: campaign markers + per-zone ``proj:``/``spatial:``.

    Encoder/quantization provenance (``geoemb:``) is NOT here — in the geoembeddings
    ``utm_zones`` layout it lives once on the root group (see :func:`_root_attrs`);
    each zone group carries only its own CRS/grid conventions.
    """
    attrs: dict = {
        "crs": spec.crs,
        "years_complete": [],
        "zone_scheme": ZONE_SCHEME,
        # Calendar-year slots are a GUARANTEE: each `time` point is Jan 1 of its year
        # (the START of the exact Jan-Dec window it holds — fixed and uniform across
        # zones at seeding), and the zone-fill gate rejects any window that is not
        # exactly that calendar year, so the label always matches the data. The seeded
        # `time_bnds` CF-bounds variable states each slot's true interval, half-open
        # ([Jan 1 of y, Jan 1 of y+1)). Non-calendar 12-month windows belong in a store whose
        # time points ARE the windows (single-ROI `12mo_window_end`, or the ADR-011
        # windowed-variant design at zone scale) — never in this store.
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
            include_geoemb=False,  # geoemb: is stated once on the root, not per zone
        )
    )
    return attrs


def _root_attrs(layout: StoreLayout, model_version: str | None, optical_min_obs: int | None) -> dict:
    """Root-group attrs for the multi-zone campaign store: geoemb: stated once.

    The geoembeddings ``utm_zones`` layout puts the encoder/quantization provenance
    on the root (identical across all zones), with ``spatial_layout="utm_zones"``.

    ``optical_min_obs`` is the minimum valid optical observations a pixel needed to be
    embedded at all, or ``None`` for a store that embedded every pixel it had input for. It
    lives on the ROOT because it is a property of the whole product rather than of a zone:
    a user asking "what rule produced this dataset" must be able to answer it from the store
    without reading provenance per cell, and a fill must be able to check that the rule it
    is about to apply is the rule the store advertises. Absent rather than zero when there
    was no rule — zero is a threshold that refuses nothing, which is a different statement
    from never having had one.
    """
    attrs = build_geoemb_root_attrs(
        embedding_dim=_layout_band(layout),
        spatial_layout="utm_zones",
        gsd=float(PIXEL_M),  # every zone is the same fixed-metre grid; the root has none to derive from
        model_version=model_version,
    )
    if optical_min_obs is not None:
        attrs["optical_min_obs"] = int(optical_min_obs)
    return attrs


def _fill_equal(have: object, want: object) -> bool:
    """Compare two fill values across the numpy-scalar / Python-literal boundary.

    An existing array reports ``np.int8(0)`` where the layout says ``0``, and NaN is
    never equal to itself — so neither ``==`` nor ``repr`` alone works. Values are
    compared numerically, with NaN treated as equal to NaN (both mean "no data").
    """
    h, w = np.asarray(have), np.asarray(want)
    if h.dtype.kind == "f" and w.dtype.kind == "f" and bool(np.isnan(h)) and bool(np.isnan(w)):
        return True
    return bool(h == w)


def _codec_id(arr: zarr.Array) -> str:
    """The layout ``codec`` key an existing array corresponds to.

    Mirrors :meth:`ArrayLayout.create_kwargs`'s three cases: a PCodec serializer is
    ``"pcodec"``; otherwise ``"zstd"`` when a compressor is configured and ``"raw"``
    when none is.
    """
    if type(getattr(arr, "serializer", None)).__name__ == "PCodec":
        return "pcodec"
    return "zstd" if arr.compressors else "raw"


def _check_layout_matches(grp: zarr.Group, gname: str, layout: StoreLayout) -> None:
    """Reject an incremental seed whose layout differs from the seeded groups'.

    The time axis is validated against existing groups but the GEOMETRY was not, so a
    store could be half-seeded with one shard pitch and half with another. That is not
    a cosmetic split: ``plan_zone_inference`` requires the inference tile to equal the
    group's shard pitch, so the later zones are rejected at fill time — after seeding
    has already committed them — and the store needs hand repair. Cheaper to refuse
    the mixed seed here, alongside the axis check it belongs with.

    Compared per-variable against what THIS layout would have created at the existing
    array's own shape, so shape-clamping (a zone smaller than one nominal shard) is not
    mistaken for a layout change.
    """
    for var, spec in layout.arrays.items():
        if var not in grp:
            raise ValueError(
                f"Refusing to seed: existing group {gname!r} has no {var!r} array, which layout "
                f"{layout.name!r} requires — the store was seeded with a different layout."
            )
        arr = cast("zarr.Array", grp[var])
        want = spec.create_kwargs(tuple(arr.shape))
        # The WHOLE schema, not only the geometry. Matching chunks and shards with a
        # different dtype is the nastier case: same pitch, so a geometry-only check
        # passes, and the new zones are then created as (say) float32 while the
        # staging writer emits int8 and refuses to fill them — a heterogeneous store
        # that only reveals itself at fill time. Fill value and codec go the same way.
        checks: tuple[tuple[str, object, object, bool], ...] = (
            ("chunks", tuple(arr.chunks), tuple(want["chunks"]), tuple(arr.chunks) == tuple(want["chunks"])),
            (
                "shards",
                tuple(arr.shards or ()),
                tuple(want.get("shards") or ()),
                tuple(arr.shards or ()) == tuple(want.get("shards") or ()),
            ),
            ("dtype", arr.dtype, want["dtype"], arr.dtype == want["dtype"]),
            (
                "fill_value",
                arr.fill_value,
                want["fill_value"],
                _fill_equal(arr.fill_value, want["fill_value"]),
            ),
            ("codec", _codec_id(arr), spec.codec, _codec_id(arr) == spec.codec),
        )
        mismatch = {k: (have, wanted) for k, have, wanted, ok in checks if not ok}
        if mismatch:
            raise ValueError(
                f"Refusing to seed: layout {layout.name!r} disagrees with existing group {gname!r} on "
                f"{var!r} — "
                + ", ".join(f"{k}: have {h!r}, layout wants {w!r}" for k, (h, w) in mismatch.items())
                + ". One store cannot mix schemas: the fill's tile size is pinned to the shard pitch and the "
                "staging writer emits one dtype, so the new zones would be unfillable."
            )


def seed_zone_groups(
    repo: icechunk.Repository,
    specs: Iterable[ZoneSpec],
    *,
    years: tuple[int, ...] = CAMPAIGN_YEARS,
    layout: StoreLayout = GLOBAL,
    model_version: str | None = None,
    optical_min_obs: int | None = None,
    commit_msg: str | None = None,
) -> str:
    """Seed one or more zone groups (metadata-only) in a single commit.

    Each group gets the full ``years`` time axis, per-zone coordinate arrays, the
    ``layout``'s sharded array schemas (no chunk data — cost is independent of
    extent, D1), and per-zone attrs (CRS, ``years_complete: []``, zone scheme,
    conventions). Returns the commit snapshot id.
    """
    specs = list(specs)
    # The time axis is created from `years` verbatim and is fixed forever after
    # (ADR-008 D1), so a malformed tuple is not recoverable by reseeding. Duplicates
    # are the dangerous shape: years=(2025, 2025) makes two identical coordinates,
    # `time_index_of` always resolves to the first, and `years_complete` then marks
    # the pair done while the second slot is never written — permanently empty, and
    # invisible to the campaign because status reports it complete. A non-monotonic
    # tuple additionally produces a CF-invalid time coordinate.
    if not years or list(years) != sorted(set(years)):
        raise ValueError(
            f"years must be a non-empty, strictly increasing tuple, got {tuple(years)} — the time axis is "
            "fixed at seeding and a duplicate or unordered year cannot be repaired afterwards."
        )
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    # geoemb: provenance is stated once on the root (utm_zones layout). It is
    # WRITE-ONCE: a later fill's model gate (fill_zone_year) trusts it to decide
    # which encoder may write, so an incremental seed must NOT silently re-stamp it
    # with a different encoder — that would let already-seeded/filled zones (encoder
    # A) be mixed with a new one (encoder B) under a root now advertising B. The
    # first seed stamps it; a matching reseed is a no-op; a changed identity is
    # rejected. (Software build_version may drift and is not part of the identity.)
    if optical_min_obs is not None and optical_min_obs <= 0:
        # Zero cannot be a rule: it refuses nothing while presenting as a configured line, and it
        # would then be stamped as one, permanently, on a store that has no rule at all.
        raise ValueError(
            f"optical_min_obs={optical_min_obs} refuses nothing — pass None for a store with no "
            "minimum-depth rule, or a positive number of observations."
        )
    new_root = _root_attrs(layout, model_version, optical_min_obs)
    if "geoemb:model" not in root.attrs:
        root.attrs.update(new_root)
    else:
        # optical_min_obs is in the identity for the same reason the encoder is: a fill reads it to
        # decide which pixels it may embed, so re-stamping it would let zones filled under one rule
        # sit beside zones filled under another, under a root advertising only the second. The
        # consequence to know is that it makes the rule UNCHANGEABLE for this store — moving the
        # line means a new store, not a migration.
        identity = ("geoemb:model", "geoemb:dimensions", "geoemb:data_type", "checkpoint_id", "optical_min_obs")
        mismatched = {k: (root.attrs.get(k), new_root.get(k)) for k in identity if root.attrs.get(k) != new_root.get(k)}
        if mismatched:
            raise ValueError(
                f"Refusing to reseed: the store root's published identity is write-once, but this seed "
                f"would change {mismatched}. Seed a fresh store for a new encoder or a new minimum-depth "
                "rule (mixing either under one store would corrupt already-published zones)."
            )
    # The time axis is fixed and UNIFORM across all zone groups (ADR-008 D1):
    # campaign status/fill code assumes every group shares one axis. A direct
    # incremental seed passing a different `years` than the groups already present
    # would silently leave a mixed-axis store, so validate against any seeded group
    # here — not only in the seed_global_store flow's retry guard, which a
    # lower-level caller bypasses.
    for gname in root.group_keys():
        grp = cast("zarr.Group", root[gname])
        if "time" not in grp:
            continue
        existing_years = tuple(year_of(t) for t in read_time_values(grp))
        if existing_years != tuple(years):
            raise ValueError(
                f"years {tuple(years)} differ from the store's existing axis {existing_years} — the time axis "
                "is fixed at seeding (ADR-008 D1); seeding more groups with a different axis would corrupt it."
            )
        _check_layout_matches(grp, gname, layout)
        break  # all groups share one axis and one layout (invariant) — one check suffices
    times = calendar_year_times(years)
    nt = len(times)
    band = _layout_band(layout)
    # CF time bounds: each year slot covers its whole calendar year, half-open as CF
    # intends — [Jan 1 00:00 of y, Jan 1 00:00 of y+1). The upper bound is the instant
    # the year ENDS, not the start of Dec 31: at ns precision the latter would drop
    # Dec 31 and leave a one-day hole between consecutive slots. So years are
    # contiguous, each cell is a full 365/366 days, and the `time` point (Jan 1) lies
    # inside its own cell (CF §7.1 containment). Written once here and never touched
    # by fills — the calendar-year guarantee makes these each slot's true extent.
    default_bnds = np.stack(
        [
            np.array([np.datetime64(f"{y}-01-01", "ns") for y in years], dtype="datetime64[ns]"),
            np.array([np.datetime64(f"{y + 1}-01-01", "ns") for y in years], dtype="datetime64[ns]"),
        ],
        axis=1,
    ).astype("int64")
    for spec in specs:
        node = root.require_group(spec.group_name)
        north = northing_coords(spec)
        east = easting_coords(spec)
        sizes = {
            "time": nt,
            "northing": spec.height,
            "easting": spec.width,
            "band": band,
            "month": MONTHS_IN_YEAR,
        }
        create_layout_arrays(node, layout, layout.arrays, sizes)
        _write_coord_arrays(
            node,
            {
                "time": times,
                "northing": north,
                "easting": east,
                "band": np.arange(band),
                # 1..12, not 0..11, so ``cov.sel(month=7)`` means July. A 0-based axis would put
                # every reader's selection one off from the calendar they are thinking in.
                "month": np.asarray(MONTH_COORD, dtype="int16"),
            },
        )
        bnds = node.create_array("time_bnds", data=default_bnds, chunks=(nt, 2), dimension_names=("time", "bnds"))
        bnds.attrs.update(TIME_ENCODING)  # same int64-ns encoding as `time`
        node["time"].attrs["bounds"] = "time_bnds"  # CF: this coordinate represents an interval
        node.attrs.update(_zone_attrs(spec, north, east, layout))
    return session.commit(commit_msg or f"seed {len(specs)} zone group(s)")
