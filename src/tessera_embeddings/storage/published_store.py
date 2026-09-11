"""Reading the published global store from the outside — layout conformance and live-shard discovery.

The write path already has everything it needs to know about the store's geometry. What it does
not have is the *reader's* three questions, which this module answers:

* **Does a zone group actually have the geometry the architecture promised?**
  :func:`layout_departures` compares a live group against :data:`~tessera_embeddings.config.store_layout.GLOBAL`
  — dims, dtype, inner chunks and shards, per array. Deliberately separate from
  :func:`~tessera_embeddings.storage.global_store.check_destination_types`, which checks the same
  group for a different purpose: that one is a write preflight and skips chunk geometry on purpose,
  because a variable joining an existing store must take that store's granularity. An audit of what
  was published wants the opposite — geometry is the thing under test.

* **Which shards of a zone-year hold data?** :func:`live_shards` asks Icechunk for the array's
  initialized chunk coordinates. On a sharded array those coordinates are **shard-grid**, not
  inner-chunk-grid, which is what makes this a cheap shard-level coverage map rather than a
  billion-entry one. It reads manifests only; no chunk bytes move.

  **Ask ``scales``, not ``embeddings``.** Both carry the same shard set, but ``scales`` is float32
  with a NaN fill, so "no chunk here" and "chunk of zeros here" can never be confused (ADR 008 D1
  makes NaN the never-written sentinel precisely so a reader has an unambiguous question to ask).
  ``scales`` is also a thirty-second of ``embeddings`` on the wire if a caller goes on to read
  values, and the obs-count arrays are a different question — a shard can hold observation counts
  and no embeddings at all, which is how a wholly-refused tile looks.

* **Where can a reader put a probe that will hit real data?** A live shard is 2048 px square and
  its ocean inner chunks are elided, so a pixel picked anywhere inside one may still resolve to
  fill — and a read that resolves to fill never leaves the process, which silently turns a latency
  measurement into a measurement of nothing. :func:`sample_live_pixels` picks candidates inside
  live shards and then keeps only those whose ``scales`` value is finite, so what comes back is
  pixels that provably hold embeddings.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from tessera_embeddings.config.store_layout import GLOBAL, SHARD_PX, StoreLayout, clamp_chunks_and_shards

if TYPE_CHECKING:
    import icechunk
    import zarr

#: The array whose initialized chunks define a zone-year's shard coverage. See the module
#: docstring: NaN fill makes absence unambiguous, and it is the cheapest array to then read.
COVERAGE_VAR = "scales"


def layout_departures(group: zarr.Group, layout: StoreLayout = GLOBAL) -> list[str]:
    """Every way ``group`` departs from ``layout``, as one sentence each; empty means it conforms.

    Returns departures rather than raising so an audit across 120 groups reports all of them
    instead of stopping at the first. A variable the layout declares and the group lacks is a
    departure; a variable the group has and the layout does not is NOT, because coordinate arrays
    (``time``, ``band``, ``northing``, ``easting``, ``time_bnds``, ``month``) legitimately live
    beside the data arrays and no layout describes them.

    **Expectations are clamped to each array's own shape** by the same
    :func:`~tessera_embeddings.config.store_layout.clamp_chunks_and_shards` the writer used. Without
    that, every zone narrower or shorter than one nominal shard reports a chunk-geometry departure
    for every array it holds — a wrong answer produced at scale, and the audit's whole value is
    that a departure it reports is worth investigating.
    """
    present = dict(group.arrays())
    out: list[str] = []
    for var in layout.arrays:
        array = present.get(var)
        if array is None:
            out.append(f"{var}: declared by the {layout.name} layout and absent from the group")
            continue
        expected = layout.for_var(var)
        if array.ndim != len(expected.dims):
            # Everything below indexes by dimension, so a shape of the wrong arity is the only
            # departure worth reporting for this array — the rest would compare unrelated axes.
            out.append(f"{var}: has {array.ndim} dimensions, the layout declares {len(expected.dims)} {expected.dims}")
            continue
        if array.dtype != np.dtype(expected.dtype):
            out.append(f"{var}: dtype is {array.dtype}, the {layout.name} layout declares {expected.dtype}")
        chunks, shards = clamp_chunks_and_shards(tuple(array.shape), expected.chunks, expected.shards)
        if tuple(array.chunks) != chunks:
            out.append(f"{var}: inner chunks are {tuple(array.chunks)}, the layout declares {chunks}")
        actual_shards = tuple(array.shards) if array.shards else None
        if actual_shards != shards:
            out.append(f"{var}: shards are {actual_shards}, the layout declares {shards}")
    return out


def live_shards(session: icechunk.Session, zone: str, var: str = COVERAGE_VAR) -> dict[int, frozenset[tuple[int, int]]]:
    """Shard-grid coordinates holding data in ``zone``'s ``var``, keyed by time index.

    The returned ``(shard_y, shard_x)`` pairs index the SHARD grid — multiply by
    :data:`~tessera_embeddings.config.store_layout.SHARD_PX` for pixel origins, or see
    :func:`shard_pixel_window`. Time is the array's own index, not a calendar year; the group's
    ``time`` coordinate converts.

    Not callable from inside a running event loop: Icechunk's chunk enumeration is async and this
    drives it with :func:`asyncio.run`. A caller that already has a loop should use
    :func:`live_shards_async`.
    """
    return asyncio.run(live_shards_async(session, zone, var))


async def live_shards_async(
    session: icechunk.Session, zone: str, var: str = COVERAGE_VAR
) -> dict[int, frozenset[tuple[int, int]]]:
    """:func:`live_shards`, for a caller that already has an event loop."""
    by_year: dict[int, set[tuple[int, int]]] = {}
    async for coord in session.chunk_coordinates(f"/{zone}/{var}"):
        # A 4-D array (embeddings, the month-covered arrays) appends a band/month index that is
        # always 0 — one shard spans the whole non-spatial axis — so take the spatial pair by
        # position from the front rather than unpacking a fixed arity.
        time_index, shard_y, shard_x = int(coord[0]), int(coord[1]), int(coord[2])
        by_year.setdefault(time_index, set()).add((shard_y, shard_x))
    return {year: frozenset(shards) for year, shards in sorted(by_year.items())}


def shard_pixel_window(
    shard: tuple[int, int], shape: Sequence[int], *, shard_px: int = SHARD_PX
) -> tuple[int, int, int, int]:
    """The ``(y0, y1, x0, x1)`` pixel window of one shard, clamped to a zone of ``shape``.

    ``shape`` is the array's full shape; only its northing and easting extents are read, from the
    positions they occupy in every store array (``(time, northing, easting, ...)``). Clamping
    matters at a zone's south and east margins, where the extent is a whole number of shards only
    because the grid was built that way — an edge shard of a future layout need not be full.
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

    Picks ``count * oversample`` candidates uniformly across the given shards, reads
    ``coverage_var`` at each, and keeps the finite ones. The filter is the point: a live shard's
    ocean inner chunks are elided, so a candidate can land on fill — and Icechunk answers a read
    of an absent chunk from the manifest without a request, so an unfiltered sample measures a mix
    of real reads and free ones and reports the mixture as latency.

    Returns fewer than ``count`` pixels if the shards are mostly elided, and an empty list if
    ``shards`` is empty. Callers that need a fixed sample size should raise ``oversample`` rather
    than treat a short list as an error, because the right factor depends on how much of a zone's
    land the shard grid actually covers.
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
    # Read the sentinel per candidate rather than as one bounding-box slice: the candidates are
    # scattered over the whole zone, so their bounding box is the zone, and slicing it would pull
    # gigabytes to decide where a hundred probes may go.
    live = [(y, x) for y, x in candidates if bool(np.isfinite(np.asarray(array[time_index, y, x], dtype="float64")))]
    return live[:count]
