"""The model half of the provenance record: which granule supplied each pixel the model's input carried.

A mosaic built with the provenance switch on carries a ``source_index`` band beside its imagery
(``ingest/provenance.py``): per pixel and slot, the fusion position plus one of the granule the
fuser took the pixel from. The actor turns that into the per-shard record the registry publishes
(``context_docs/storage/registry-provenance-plan.md`` §3, §4):

* **Pieces, read where the mask is read.** Wherever a loader reads a sensor's validity — the
  whole-shard SCL mask for optical, each strip's VV and VH for radar — it also reads the source
  index over the same dates and window, zeroes it wherever the model's own validity test fails,
  and keeps it compressed per date (:func:`read_source_pieces`). Nothing is read twice, and a
  shard holds kilobytes per date rather than a full plane.
* **Planes, ranked at the staging write.** :class:`SourcePlanes` collects a shard's pieces across
  strips. At the write, on the actor's background writer, each date's pieces are stitched into one
  plane, cut for a holes update to the pixels it embedded, and its values replaced by their rank in
  the shard-date's ``sources``: the distinct fusion positions that supplied a pixel, ascending. A day's
  fusion order is zone-wide and can exceed 255, while a shard has a handful of contributing
  granules on a date, so the rank keeps the published plane ``uint8``. The plane and its
  ``sources`` and valid-pixel count are staged with the shard.

A non-zero value means the observation was valid for the model, by its own test, and names the
granule that supplied it. For a pixel with an embedding those are the observations its input drew
on; a pixel deeper than the deepest bucket is resampled to a deterministic subset of them
(``inference/sampling.py``). A fill or a shards update records every pixel it was asked to embed,
refused ones included, which is how a refused shard still has rows. A holes update records only the
pixels it embedded, and stages them as its ``update_pixels`` mask: a reader takes those pixels from
the update and every other pixel from the run before it (plan §5).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numcodecs
import numpy as np
import zarr
from zarr.codecs import BloscCodec

from tessera_embeddings.config.ingest import SOURCE_INDEX_VAR
from tessera_embeddings.config.store_layout import SENSORS

#: The staged and published planes' codec: Blosc zstd 3 with bit shuffle, measured at 1.17 MB a
#: shard-year where plain zstd took 2.83 MB (the plan's Appendix A).
PLANE_COMPRESSORS = (BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),)

#: The same codec for the in-memory pieces, which are 16-bit positions rather than 8-bit ranks.
_PIECE_CODEC = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

#: Dates read per batch when building pieces: a whole shard's index for every kept date at once
#: would be gigabytes, one date at a time is a request each.
_READ_BATCH = 8

#: The staged arrays, per sensor: the planes, and the mosaic time index of each.
STAGED_PLANES = "pixel_sources_{sensor}"
STAGED_TIMES = "pixel_sources_{sensor}_time"

#: The staged shard's attribute holding its ``shard_dates`` summary, beside the planes it describes.
PROVENANCE_ATTR = "provenance"

#: A holes update's staged pixel mask: the pixels the run embedded, which the planes are cut to and
#: which publication copies to ``update_pixels``.
STAGED_UPDATE_PIXELS = "update_pixels"


@dataclass(frozen=True)
class SourcePiece:
    """One date's source positions over one window of a shard, compressed.

    ``time`` is the mosaic time index; ``rows`` and ``cols`` are shard-relative. Values are the
    fusion position plus one, and zero wherever the model's validity test failed.
    """

    time: int
    rows: slice
    cols: slice
    data: bytes


def read_array(group: zarr.Group, name: str) -> np.ndarray:
    """``group[name]`` read whole, as numpy."""
    arr = group[name]
    assert isinstance(arr, zarr.Array)
    return np.asarray(arr[...])


def read_source_pieces(
    root: zarr.Group,
    times: np.ndarray,
    y_slice: slice,
    x_slice: slice,
    valid: np.ndarray,
    *,
    rows: slice,
    cols: slice,
) -> list[SourcePiece]:
    """The source index at ``times`` over the window, zeroed where ``valid`` is False, per date.

    ``valid`` is ``(len(times), h, w)``, the test the model applies to the same reads. Empty when
    the mosaic carries no source index, which is how the record follows the mosaics: a store built
    with provenance off gives the actor nothing to stage.
    """
    if SOURCE_INDEX_VAR not in root or len(times) == 0:
        return []
    index = root[SOURCE_INDEX_VAR]
    assert isinstance(index, zarr.Array)
    pieces: list[SourcePiece] = []
    for start in range(0, len(times), _READ_BATCH):
        batch = np.asarray(times[start : start + _READ_BATCH])
        positions = np.asarray(index.oindex[batch, y_slice, x_slice])
        positions[~valid[start : start + len(batch)]] = 0
        pieces.extend(
            SourcePiece(int(t), rows, cols, _PIECE_CODEC.encode(np.ascontiguousarray(plane)))
            for t, plane in zip(batch, positions, strict=True)
        )
    return pieces


@dataclass(frozen=True)
class ShardDate:
    """One shard, sensor and date: what the record publishes about it."""

    time: int
    plane: np.ndarray
    #: The fusion positions (0-based) that supplied at least one pixel, ascending.
    sources: list[int]
    valid_px: int


class SourcePlanes:
    """One shard's pieces for every sensor, collected across strips, ranked at the staging write.

    Pieces arrive from the loader's thread and are read on the writer's, so ``add`` is locked even
    though a shard's strips load one at a time.
    """

    def __init__(self, height: int, width: int) -> None:
        """An empty record for a ``height`` by ``width`` shard."""
        self.height, self.width = height, width
        self._pieces: dict[str, dict[int, list[SourcePiece]]] = {sensor: {} for sensor in SENSORS}
        self._lock = threading.Lock()

    def add(self, sensor: str, pieces: list[SourcePiece]) -> None:
        """Keep ``pieces`` for ``sensor``."""
        with self._lock:
            for piece in pieces:
                self._pieces[sensor].setdefault(piece.time, []).append(piece)

    def __bool__(self) -> bool:
        """Whether any sensor's mosaic carried a source index."""
        return any(self._pieces.values())

    def sensors(self) -> list[str]:
        """The sensors whose mosaic carried a source index, in :data:`SENSORS` order."""
        return [sensor for sensor in SENSORS if self._pieces[sensor]]

    def dates(self, sensor: str, pixels: np.ndarray | None = None) -> Iterator[ShardDate]:
        """Each date's ranked plane for ``sensor``, in time order, cut to ``pixels`` when given.

        A date with no valid pixel left after the cut is dropped: the shard did not see it.

        Raises:
            ValueError: a shard-date with more than 255 contributing granules, which the ``uint8``
                plane cannot rank. Refused rather than wrapped.
        """
        for time in sorted(self._pieces[sensor]):
            full = np.zeros((self.height, self.width), dtype=np.uint16)
            for piece in self._pieces[sensor][time]:
                shape = (piece.rows.stop - piece.rows.start, piece.cols.stop - piece.cols.start)
                full[piece.rows, piece.cols] = np.frombuffer(_PIECE_CODEC.decode(piece.data), np.uint16).reshape(shape)
            if pixels is not None:
                full[~pixels] = 0
            present = np.flatnonzero(np.bincount(full.ravel(), minlength=1))
            present = present[present > 0]
            if not len(present):
                continue
            if len(present) > np.iinfo(np.uint8).max:
                raise ValueError(f"{sensor} time {time}: {len(present)} granules supplied one shard; the cap is 255")
            rank = np.zeros(int(present[-1]) + 1, dtype=np.uint8)
            rank[present] = np.arange(1, len(present) + 1, dtype=np.uint8)
            yield ShardDate(time, rank[full], [int(p) - 1 for p in present], int(np.count_nonzero(full)))


def stage_planes(group: zarr.Group, planes: SourcePlanes, pixels: np.ndarray | None) -> dict[str, Any]:
    """Write each sensor's ranked planes into a staged shard; return its ``shard_dates`` summary.

    The planes go in as ``(time, northing, easting)`` arrays chunked one date by the shard, with
    their mosaic time indices beside them. The summary is ``{sensor: {"time", "sources",
    "valid_px"}}``; the caller stores it as the shard's ``provenance`` attribute. A sensor whose
    mosaic had no index is absent from both, so the record follows the mosaics sensor by sensor.
    ``pixels``, given only for a holes update, is the pixels the run embedded; the planes are cut to
    it and it is staged too, as the update's ``update_pixels`` mask.
    """
    summary: dict[str, Any] = {}
    if pixels is not None:
        group.create_array(STAGED_UPDATE_PIXELS, shape=pixels.shape, chunks=pixels.shape, dtype=bool, overwrite=True)[
            :
        ] = pixels
    for sensor in planes.sensors():
        dates = list(planes.dates(sensor, pixels))
        arr = group.create_array(
            STAGED_PLANES.format(sensor=sensor),
            shape=(len(dates), planes.height, planes.width),
            chunks=(1, planes.height, planes.width),
            dtype=np.uint8,
            fill_value=0,
            compressors=PLANE_COMPRESSORS,
            dimension_names=("time", "northing", "easting"),
            overwrite=True,
        )
        for i, date in enumerate(dates):
            arr[i] = date.plane
        times = np.array([d.time for d in dates], dtype=np.int32)
        group.create_array(
            STAGED_TIMES.format(sensor=sensor),
            shape=times.shape,
            chunks=(max(len(times), 1),),
            dtype=np.int32,
            overwrite=True,
        )[:] = times
        summary[sensor] = {
            "time": times.tolist(),
            "sources": [d.sources for d in dates],
            "valid_px": [d.valid_px for d in dates],
        }
    return summary


def staged_planes(group: zarr.Group, sensor: str) -> tuple[np.ndarray, zarr.Array] | None:
    """A staged shard's ``(time indices, planes)`` for ``sensor``, or ``None`` if it staged none.

    ``group`` must be opened without consolidated metadata: the tile's listing was consolidated when
    its embeddings were written, before these arrays existed.
    """
    name = STAGED_PLANES.format(sensor=sensor)
    if name not in group:
        return None
    arr = group[name]
    assert isinstance(arr, zarr.Array)
    return read_array(group, STAGED_TIMES.format(sensor=sensor)), arr


def provenance_summary(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    """A staged shard's ``shard_dates`` summary, or ``None`` if it staged no planes."""
    summary = attrs.get(PROVENANCE_ATTR)
    return summary if isinstance(summary, dict) else None
