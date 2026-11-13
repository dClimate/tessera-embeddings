"""What an ingest records about the granules behind each mosaic slot: the ingest half of provenance.

Two records, both written in the slot's own commit and only when the ingest runs with
``provenance`` on (``context_docs/storage/registry-provenance-plan.md`` §4), and a third, kept
whatever the switch, for the days it skipped whole (:class:`SkippedDays`):

* **The granule list.** One JSON array per slot naming the day's granules in fusion order, with
  the identifiers, acquisition time, footprint and exact object URLs read, then the copies the
  ingest dropped before the fuse and why (:func:`day_granules`, :func:`mark_dropped`).
* **The source index.** Per pixel, the fusion position plus one of the granule that supplied it,
  0 where none had data.

**How the index is made without a second read or a second fuser.** odc's fuser keeps the first
source with data at each pixel and reports nothing about which source that was. So one band per
sensor — SCL for optical, VV for radar — is loaded through a reader that packs each granule's
fusion position into the high bits of every pixel it returns (:func:`pack`). The packed value is
zero exactly where the granule's own pixel is nodata, so odc's unchanged fuser, which fills only
where the destination is still zero, keeps the first granule with data, as it always has. Splitting
the fused result (:func:`unpack`) gives back the band, identical to an unpacked load, and the index.
One read and the fuser's own rule; the split and the index's writes are the only new work.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from tessera_embeddings.config.ingest import (
    CATALOG_ASSETS,
    DROPPED,
    DUPLICATE,
    FUSED,
    LOW_COVERAGE,
    MISSING_BANDS,
    PRODUCER_CONFLICT,
    UNREADABLE,
)
from tessera_embeddings.ingest.duplicates import image_identity
from tessera_embeddings.ingest.opera_query import BURST_ID
from tessera_embeddings.ingest.solar_days import solar_day_of
from tessera_embeddings.storage.zarr_store import record_skipped_days

#: Item property: the item's position in its solar day's fusion order, stamped just before a load.
FUSION_POSITION = "tessera:fusion_position"

#: Item property: this copy replaced one that would not read (the duplicate-copy fallback ladder).
STEPPED_DOWN = "tessera:stepped_down"

#: Bits the packed pixel gives the position. A day's fusion order is zone-wide, so a long land zone
#: can exceed 255 optical scenes or radar bursts in one day; sixteen bits leave ample room.
_POSITION_BITS = 16
MAX_GRANULES_PER_DAY = (1 << _POSITION_BITS) - 1

_PLATFORM_RE = re.compile(r"(?:^|_)(S[12][A-D])_")


def stamp_fusion_positions(items: Iterable[Any], loaded: list[Any]) -> None:
    """Stamp each of ``loaded`` with its position in its solar day's fusion order, in place.

    ``loaded`` is in fusion order already, the order the loader keeps. Every other item of
    ``items`` loses any stamp an earlier load left, so a stamped item is exactly one this load read
    and :func:`day_granules` cannot name a copy the duplicate selection pruned.

    Raises:
        ValueError: a solar day holds more granules than the packed pixel can number.
    """
    for item in items:
        item.properties.pop(FUSION_POSITION, None)
    counts: dict[str, int] = {}
    for item in loaded:
        item.properties.pop(DROPPED, None)
        day = solar_day_of(item)
        position = counts.get(day, 0)
        if position >= MAX_GRANULES_PER_DAY:
            raise ValueError(f"{day}: more than {MAX_GRANULES_PER_DAY} granules in one solar day")
        item.properties[FUSION_POSITION] = position
        counts[day] = position + 1


def mark_dropped(items: Iterable[Any], reason: str) -> None:
    """Mark ``items`` as dropped before the fuse for ``reason``, in place.

    Clears any fusion position an earlier load stamped, so a copy that read once and was then
    replaced is listed as dropped, not fused. A dropped copy that a later load reads is fused again
    (:func:`stamp_fusion_positions` clears the mark).
    """
    for item in items:
        item.properties.pop(FUSION_POSITION, None)
        item.properties[DROPPED] = reason


def mark_duplicates(supplied: Iterable[Any], kept: Iterable[Any]) -> dict[str, list[Any]]:
    """Mark the copies of ``supplied`` that duplicate selection did not keep; return them by solar day."""
    kept_ids = {id(item) for item in kept}
    by_day: dict[str, list[Any]] = {}
    for item in supplied:
        if id(item) not in kept_ids:
            by_day.setdefault(solar_day_of(item), []).append(item)
    mark_dropped((item for day in by_day.values() for item in day), DUPLICATE)
    return by_day


def mark_stepped_down(previous: Sequence[Any], stepped: Sequence[Any]) -> list[Any]:
    """Mark one step down the fallback ladder: each copy standing in, and each it replaced. Returns those.

    A copy in ``stepped`` and not in ``previous`` is marked :data:`STEPPED_DOWN`; one in ``previous``
    and not in ``stepped`` would not read, and is marked dropped as unreadable.
    """
    before, after = {id(item) for item in previous}, {id(item) for item in stepped}
    for item in stepped:
        if id(item) not in before:
            item.properties[STEPPED_DOWN] = True
    replaced = [item for item in previous if id(item) not in after]
    mark_dropped(replaced, UNREADABLE)
    return replaced


def packed_dtype(native: np.dtype) -> np.dtype:
    """The unsigned dtype that holds a ``native`` pixel's bits below a fusion position."""
    return np.dtype(np.uint32 if native.itemsize * 8 + _POSITION_BITS <= 32 else np.uint64)


def pack(pixels: np.ndarray, position: int, missing: np.ndarray) -> np.ndarray:
    """``pixels`` with ``position + 1`` in the high bits, and 0 wherever ``missing``.

    ``missing`` must be the fuser's own nodata test for the native band, which is what makes the
    packed fuse keep the same granule the native fuse would.
    """
    bits = pixels.itemsize * 8
    raw = np.ascontiguousarray(pixels).view(f"u{pixels.itemsize}")
    out = raw.astype(packed_dtype(pixels.dtype)) | (packed_dtype(pixels.dtype).type(position + 1) << bits)
    out[missing] = 0
    return out


def unpack(packed: xr.DataArray, native: np.dtype, fill: float) -> tuple[xr.DataArray, xr.DataArray]:
    """Split a fused packed band into ``(band, source_index)``, lazily.

    ``band`` is ``fill`` wherever no granule had data, the value an unpacked load leaves there.
    """
    bits = native.itemsize * 8
    data = packed.data
    index = (data >> bits).astype(np.uint16)
    raw = (data & ((1 << bits) - 1)).astype(f"u{native.itemsize}").view(native)
    band = da.where(index > 0, raw, native.type(fill)) if isinstance(data, da.Array) else np.where(index > 0, raw, fill)
    return packed.copy(data=band.astype(native)), packed.copy(data=index)


def _platform(item: Any) -> str | None:  # noqa: ANN401
    match = _PLATFORM_RE.search(str(item.id))
    return match.group(1) if match else None


def _image_key(item: Any) -> str | None:  # noqa: ANN401
    """Which image the copy is of: S2 pass and datastrip (#240), or the S1 burst (#250)."""
    if (burst := item.properties.get(BURST_ID)) is not None:
        return str(burst)
    identity = image_identity(item)
    return "/".join(identity) if identity else None


def _href(asset: Any) -> str | None:  # noqa: ANN401
    return asset.get("href") if isinstance(asset, dict) else getattr(asset, "href", None)


def _record(item: Any, status: str, **source: str) -> dict[str, Any]:  # noqa: ANN401
    """One granule's row, less the columns assembly adds (zone, year, run, sensor, day, order)."""
    properties = item.properties
    return {
        "granule_id": item.id,
        "status": status,
        "image_key": _image_key(item),
        "platform": _platform(item),
        "acquired_at": properties.get("datetime"),
        "cloud_cover": properties.get("eo:cloud_cover"),
        "processing_baseline": properties.get("s2:processing_baseline"),
        "product_uri": properties.get("s2:product_uri"),
        "provider": source["provider"],
        "collection": source["collection"],
        # Rebuilt rather than kept: the query prunes links to bound the driver's memory.
        "stac_item_url": f"{source['catalog_url'].rstrip('/')}/collections/{source['collection_id']}/items/{item.id}",
        # The catalogue's own links where the item was stamped with them: radar's are rewritten for
        # reading, to S3 URIs or signed URLs that carry a credential and expire.
        "assets": dict(
            sorted((properties.get(CATALOG_ASSETS) or {k: _href(a) for k, a in item.assets.items()}).items())
        ),
        "stepped_down": bool(properties.get(STEPPED_DOWN, False)),
        "geometry": item.geometry,
        "bbox": list(item.bbox) if item.bbox is not None else None,
    }


def day_granules(
    items: Iterable[Any],
    *,
    provider: str,
    collection: str,
    catalog_url: str,
    collection_id: str,
) -> dict[str, str]:
    """Per solar day, the JSON granule list of ``items``: the fused ones, then the dropped ones.

    The fused are the items the last load stamped (:func:`stamp_fusion_positions`), in fusion
    order, so their positions are the source index's numbering. The dropped are those marked by
    :func:`mark_dropped`, by id, each with its reason as ``status``. An item passed twice is listed
    once, and any other item not at all.
    """
    source = {
        "provider": provider,
        "collection": collection,
        "catalog_url": catalog_url,
        "collection_id": collection_id,
    }
    by_day: dict[str, dict[int, Any]] = {}
    for item in items:
        if FUSION_POSITION in item.properties or DROPPED in item.properties:
            by_day.setdefault(solar_day_of(item), {})[id(item)] = item
    out: dict[str, str] = {}
    for day, unique in by_day.items():
        fused = sorted(
            (it for it in unique.values() if FUSION_POSITION in it.properties),
            key=lambda it: it.properties[FUSION_POSITION],
        )
        dropped = sorted(
            (it for it in unique.values() if FUSION_POSITION not in it.properties), key=lambda it: str(it.id)
        )
        records = [_record(it, FUSED, **source) for it in fused]
        records += [_record(it, it.properties[DROPPED], **source) for it in dropped]
        out[day] = json.dumps(records, separators=(",", ":"), default=str)
    return out


#: The ingest's reasons for skipping a whole day, as the status its copies are listed under. A day
#: whose imagery reached no live window is absent: none of it covered a pixel the store embeds.
_SKIP_STATUS = {
    "coverage": LOW_COVERAGE,
    "producer-conflict": PRODUCER_CONFLICT,
    "unreadable": UNREADABLE,
    "missing-bands": MISSING_BANDS,
}


class SkippedDays:
    """The granule lists of the days one ingest leg skipped whole, committed to its mosaic.

    A skipped day has no slot, so no list beside its pixels; it is listed off the time axis instead
    (plan §3), every copy dropped for the day's reason unless already dropped for its own. Each day
    is committed when it is skipped, or, while the mosaic does not exist yet, right after the leg's
    first write creates it. A retry rebuilds and overwrites it.
    """

    def __init__(self, store_path: str, source: Mapping[str, str], *, exists: bool, s3_region: str | None) -> None:
        """A recorder for the mosaic at ``store_path``; ``source`` is :func:`day_granules`'s keywords."""
        self.store_path, self.source, self.exists, self.s3_region = store_path, source, exists, s3_region
        self._pending: dict[str, str] = {}

    def record(self, day: str, items: Iterable[Any], reason: str | None) -> None:
        """List ``day``'s copies among ``items`` under the status for ``reason``, and commit them."""
        self.record_days([day], items, reason)

    def record_days(self, days: Iterable[str], items: Iterable[Any], reason: str | None) -> None:
        """:meth:`record` for each of ``days``, in one commit."""
        status, items = _SKIP_STATUS.get(reason or ""), list(items)
        if status is None:
            return
        for day in days:
            day_items = [item for item in items if solar_day_of(item) == day]
            if day_items:
                mark_dropped((item for item in day_items if DROPPED not in item.properties), status)
                self._pending[day] = day_granules(day_items, **self.source)[day]
        self._commit()

    def written(self) -> None:
        """The leg committed a date, so the mosaic exists: commit any day held until it did."""
        self.exists = True
        self._commit()

    def _commit(self) -> None:
        if self.exists and self._pending:
            record_skipped_days(self.store_path, self._pending, s3_region=self.s3_region)
            self._pending = {}
