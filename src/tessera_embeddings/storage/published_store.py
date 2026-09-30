"""Reading the published global store from the outside: layout conformance and shard discovery.

:func:`layout_departures` is deliberately separate from
:func:`~tessera_embeddings.storage.global_store.check_destination_types`, which checks the same
group as a write preflight and skips chunk geometry on purpose, because a variable joining an
existing store must take that store's granularity. An audit wants geometry under test.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from tessera_embeddings.config.store_layout import GLOBAL, SHARD_PX, StoreLayout, clamp_chunks_and_shards
from tessera_embeddings.storage.global_store import _codec_id, missing_seeded_arrays

if TYPE_CHECKING:
    import icechunk
    import zarr

    from tessera_embeddings.storage.zone_grid import ZoneSpec

#: The array whose initialized chunks define a zone-year's shard coverage. ``scales`` rather than
#: ``embeddings``: both carry the same shard set, but its NaN fill makes absence unambiguous where
#: int8 zero does not (ADR 008 D1), and it is a thirty-second of ``embeddings`` to then read.
COVERAGE_VAR = "scales"


def layout_departures(group: zarr.Group, layout: StoreLayout = GLOBAL) -> list[str]:
    """Every way ``group`` departs from ``layout``, as one sentence each; empty means it conforms.

    An array the layout declares and the group lacks is a departure; the reverse is not, because
    the coordinate arrays live beside the data arrays and no layout describes them. Expectations
    are clamped to each array's own shape by the writer's own
    :func:`~tessera_embeddings.config.store_layout.clamp_chunks_and_shards`, or every zone narrower
    than one nominal shard reports a false departure.
    """
    present = dict(group.arrays())
    out: list[str] = []
    # The COORDINATE arrays too: losing `northing`, `month` or `time_bnds` would otherwise only
    # skip that dimension's extent check, so the audit reports nothing while a labelled read of the
    # zone is incomplete.
    for name in sorted(missing_seeded_arrays(group, layout)):
        out.append(f"{name}: a seed of the {layout.name} layout writes it and the group does not have it")
    for var in layout.arrays:
        array = present.get(var)
        if array is None:
            out.append(f"{var}: declared by the {layout.name} layout and absent from the group")
            continue
        expected = layout.for_var(var)
        if array.ndim != len(expected.dims):
            # Everything below indexes by dimension, so the rest would compare unrelated axes.
            out.append(f"{var}: has {array.ndim} dimensions, the layout declares {len(expected.dims)} {expected.dims}")
            continue
        # The NAMES: an array labelled `easting, northing` has the right rank, dtype and — both
        # spatial chunks being 256 — chunk geometry, so every other check passes while every
        # labelled read comes back transposed. `getattr` because Zarr v2 metadata has no names at
        # all, and reporting that beats raising on it.
        names = getattr(array.metadata, "dimension_names", None)
        if names is None:
            out.append(f"{var}: has no dimension names, the layout declares {expected.dims}")
        elif tuple(names) != tuple(expected.dims):
            out.append(f"{var}: dimension names are {tuple(names)}, the layout declares {tuple(expected.dims)}")
        if array.dtype != np.dtype(expected.dtype):
            out.append(f"{var}: dtype is {array.dtype}, the {layout.name} layout declares {expected.dtype}")
        if not _fill_values_match(array.fill_value, expected.fill_value):
            # Load-bearing, not cosmetic: everything here reads a finite `scales` as "written", so
            # a finite fill hides never-written pixels behind the right dtype, chunks and shards.
            out.append(f"{var}: fill value is {array.fill_value!r}, the layout declares {expected.fill_value!r}")
        for name, value in expected.attrs:
            # Part of the array's TYPE: `dtype="bool"` on an int8 array is how xarray knows to
            # present booleans, and without it a labelled reader gets 0 and 1.
            if array.attrs.get(name) != value:
                out.append(f"{var}: attribute {name}={array.attrs.get(name)!r}, the layout declares {value!r}")
        # The codec decides the store's size and read speed, so `scales` without its PCodec
        # serializer is a real departure. `_codec_id` is the write side's own mapping, reused
        # because a second copy of that rule is how the two drift apart.
        actual_codec = _codec_id(array)
        if actual_codec != expected.codec:
            out.append(f"{var}: codec is {actual_codec!r}, the layout declares {expected.codec!r}")
        chunks, shards = clamp_chunks_and_shards(tuple(array.shape), expected.chunks, expected.shards)
        if tuple(array.chunks) != chunks:
            out.append(f"{var}: inner chunks are {tuple(array.chunks)}, the layout declares {chunks}")
        actual_shards = tuple(array.shards) if array.shards else None
        if actual_shards != shards:
            out.append(f"{var}: shards are {actual_shards}, the layout declares {shards}")
        # Extents against the COORDINATE arrays, because the expectations above come from this
        # array's own shape and cannot see it being short: truncating it by a whole shard keeps the
        # nominal chunk and shard sizes and passes every check above.
        for axis, dim in enumerate(expected.dims):
            coord = present.get(dim)
            if coord is None:
                continue  # already reported above, from `missing_seeded_arrays`
            if coord.ndim != 1:
                # The name exists, so `missing_seeded_arrays` is happy, and a coordinate of the
                # wrong rank still indexes nothing.
                out.append(f"{var}: coordinate {dim} has {coord.ndim} dimensions, a coordinate must have one")
            elif coord.shape[0] != array.shape[axis]:
                out.append(f"{var}: dimension {dim} is {array.shape[axis]}, its coordinate array is {coord.shape[0]}")
    return out


def _fill_values_match(actual: object, expected: object) -> bool:
    """Whether two fill values agree, treating NaN as equal to NaN.

    ``float("nan") != float("nan")`` and NaN is exactly the fill ``scales`` relies on, so the
    obvious comparison calls every conforming store broken; zarr also hands back a NUMPY scalar,
    so an ``isinstance(..., float)`` guard misses it and the NaN case never fires.
    """
    try:
        left = np.asarray(actual, dtype="float64")
        right = np.asarray(expected, dtype="float64")
    except (TypeError, ValueError):
        return bool(actual == expected)
    if bool(np.isnan(left)) and bool(np.isnan(right)):
        return True
    return bool(left == right)


def coordinate_departures(group: zarr.Group, spec: ZoneSpec) -> list[str]:
    """Every way the group's spatial coordinates depart from the grid its zone spec defines.

    A zone can pass :func:`layout_departures` completely and still be geolocated wrongly —
    `northing` reversed, shifted by a pixel, or at the wrong spacing — with no attribute to recover
    the true positions from.

    Origin, step and far end only: those three pin a shift, a reversal and a wrong spacing, and
    reading both full axes of 120 zones would move about a gigabyte to rule out a non-uniform
    interior that :func:`numpy.arange` cannot produce. Compared against ``zone_grid``'s own
    builders, so the convention has one definition.
    """
    from tessera_embeddings.config.store_layout import MONTH_COORD
    from tessera_embeddings.storage.zone_grid import easting_coords, northing_coords

    out: list[str] = []
    present = dict(group.arrays())
    # `month` by VALUE: 0..11, a reordering or a duplicate has the right rank and extent while
    # `sel(month=7)` selects the wrong plane. Small enough to read whole, unlike the spatial axes.
    month = present.get("month")
    if month is not None and month.ndim == 1:
        months = [int(v) for v in np.asarray(month[:])]
        if months != list(MONTH_COORD):
            out.append(f"month: holds {months}, the writer seeds {list(MONTH_COORD)}")
    for name, expected in (("northing", northing_coords(spec)), ("easting", easting_coords(spec))):
        array = present.get(name)
        if array is None:
            out.append(f"{name}: absent, so its grid cannot be checked")
            continue
        actual = cast("zarr.Array", array)
        if actual.ndim != 1:
            out.append(f"{name}: has {actual.ndim} dimensions, a coordinate axis must have one")
            continue
        if actual.shape[0] != expected.size:
            out.append(f"{name}: has {actual.shape[0]} values, the zone grid defines {expected.size}")
            continue
        if expected.size < 2:
            continue
        got = np.asarray(actual[[0, 1, -1]], dtype="float64")
        want = expected[[0, 1, -1]]
        if not np.allclose(got, want, rtol=0.0, atol=1e-6):
            out.append(
                f"{name}: starts {got[0]}, steps {got[1] - got[0]}, ends {got[2]}; "
                f"the zone grid says starts {want[0]}, steps {want[1] - want[0]}, ends {want[2]}"
            )
    return out


def calendar_years(group: zarr.Group) -> list[int]:
    """The calendar year of each time slot, from the group's own ``time`` coordinate.

    The coordinate is int64 nanoseconds, so it must be VIEWED as ``datetime64[ns]`` before being
    truncated to years; casting the raw integers straight to ``datetime64[Y]`` reads each
    nanosecond count as a year offset. The range check makes a future change of units a loud
    failure rather than a silently wrong time-index-to-year mapping.
    """
    time_coord = cast("zarr.Array", group["time"])
    stamps = np.asarray(time_coord[:]).astype("datetime64[ns]")
    years = [int(y) for y in stamps.astype("datetime64[Y]").astype(int) + 1970]
    if not all(1970 <= y <= 2200 for y in years):
        raise ValueError(f"time coordinate did not decode to plausible years: {years[:4]}")
    return years


def live_shards(session: icechunk.Session, zone: str, var: str = COVERAGE_VAR) -> dict[int, frozenset[tuple[int, int]]]:
    """Shard-grid coordinates holding data in ``zone``'s ``var``, keyed by time index.

    The ``(shard_y, shard_x)`` pairs index the SHARD grid, not the inner-chunk grid — which is what
    makes this a cheap coverage map rather than a billion-entry one. See :func:`shard_pixel_window`
    for pixels and :func:`calendar_years` for the time index. Reads manifests only.

    Not callable from inside a running event loop — use :func:`live_shards_async` there.
    """
    return asyncio.run(live_shards_async(session, zone, var))


async def live_shards_async(
    session: icechunk.Session, zone: str, var: str = COVERAGE_VAR
) -> dict[int, frozenset[tuple[int, int]]]:
    """:func:`live_shards`, for a caller that already has an event loop."""
    by_year: dict[int, set[tuple[int, int]]] = {}
    async for coord in session.chunk_coordinates(f"/{zone}/{var}"):
        # A 4-D array appends a band/month index that is always 0, one shard spanning the whole
        # non-spatial axis, so take the spatial pair by position rather than a fixed arity.
        time_index, shard_y, shard_x = int(coord[0]), int(coord[1]), int(coord[2])
        by_year.setdefault(time_index, set()).add((shard_y, shard_x))
    return {year: frozenset(shards) for year, shards in sorted(by_year.items())}


def shard_pixel_window(
    shard: tuple[int, int], shape: Sequence[int], *, shard_px: int = SHARD_PX
) -> tuple[int, int, int, int]:
    """The ``(y0, y1, x0, x1)`` pixel window of one shard, clamped to a zone of ``shape``.

    ``shape`` is the array's full shape. Clamping matters at a zone's south and east margins, where
    the extent is a whole number of shards only because the grid was built that way.
    """
    shard_y, shard_x = shard
    y0, x0 = shard_y * shard_px, shard_x * shard_px
    return y0, min(y0 + shard_px, int(shape[1])), x0, min(x0 + shard_px, int(shape[2]))


def sample_live_pixels(
    group: zarr.Group,
    time_index: int,
    shards: Iterable[tuple[int, int]],
    count: int,
    *,
    seed: int = 0,
    oversample: int = 4,
    coverage_var: str = COVERAGE_VAR,
) -> list[tuple[int, int]]:
    """Up to ``count`` ``(y, x)`` pixels inside ``shards`` that provably hold embeddings.

    The filter is the point: a live shard's ocean inner chunks are elided, and Icechunk answers a
    read of an absent chunk from the manifest without a request, so an unfiltered sample measures a
    mix of real reads and free ones and reports the mixture.

    Returns fewer than ``count`` if the shards are mostly elided; raise ``oversample`` rather than
    treating that as an error, because the right factor depends on how much of the zone is land.
    """
    shards = list(shards)
    if not shards or count <= 0:
        return []
    rng = np.random.default_rng(seed)
    array = cast("zarr.Array", group[coverage_var])
    candidates: list[tuple[int, int]] = []
    for index in rng.integers(len(shards), size=count * oversample):
        y0, y1, x0, x1 = shard_pixel_window(shards[int(index)], array.shape)
        candidates.append((int(rng.integers(y0, y1)), int(rng.integers(x0, x1))))
    # Per candidate rather than one bounding-box slice: the candidates are scattered, so their
    # bounding box IS the zone and slicing it would pull gigabytes.
    live = [(y, x) for y, x in candidates if bool(np.isfinite(np.asarray(array[time_index, y, x], dtype="float64")))]
    return live[:count]
