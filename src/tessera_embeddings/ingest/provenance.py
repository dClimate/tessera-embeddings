"""What an ingest records about the granules behind each mosaic slot: the ingest half of provenance.

Two records, both written in the slot's own commit and only when the ingest runs with
``provenance`` on (``context_docs/storage/registry-provenance-plan.md`` §4):

* **The granule list.** One JSON array per slot naming the day's granules in fusion order, with
  the identifiers, acquisition time, footprint and exact object URLs read (:func:`day_granules`).
* **The source index.** Per pixel, the fusion position plus one of the granule that supplied it,
  0 where none had data.

**How the index is made without a second read or a second fuser.** odc's fuser keeps the first
source with data at each pixel and reports nothing about which source that was. So one band per
sensor — SCL for optical, VV for radar — is loaded through a reader that packs each granule's
fusion position into the high bits of every pixel it returns (:func:`pack`). The packed value is
zero exactly where the granule's own pixel is nodata, so odc's unchanged fuser, which fills only
where the destination is still zero, keeps the first granule with data, as it always has. Splitting
the fused result (:func:`unpack`) gives back the band, identical to an unpacked load, and the index.
One read, the fuser's own rule, no extra tasks in the graph.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

import dask.array as da
import numpy as np
import xarray as xr

from tessera_embeddings.ingest.duplicates import image_identity
from tessera_embeddings.ingest.opera_query import BURST_ID
from tessera_embeddings.ingest.solar_days import solar_day_of

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
        day = solar_day_of(item)
        position = counts.get(day, 0)
        if position >= MAX_GRANULES_PER_DAY:
            raise ValueError(f"{day}: more than {MAX_GRANULES_PER_DAY} granules in one solar day")
        item.properties[FUSION_POSITION] = position
        counts[day] = position + 1


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


def _record(item: Any, *, provider: str, collection: str, catalog_url: str, collection_id: str) -> dict[str, Any]:  # noqa: ANN401
    """One granule's row, less the columns assembly adds (zone, year, run, sensor, day, order)."""
    properties = item.properties
    return {
        "granule_id": item.id,
        "image_key": _image_key(item),
        "platform": _platform(item),
        "acquired_at": properties.get("datetime"),
        "cloud_cover": properties.get("eo:cloud_cover"),
        "processing_baseline": properties.get("s2:processing_baseline"),
        "product_uri": properties.get("s2:product_uri"),
        "provider": provider,
        "collection": collection,
        # Rebuilt rather than kept: the query prunes links to bound the driver's memory.
        "stac_item_url": f"{catalog_url.rstrip('/')}/collections/{collection_id}/items/{item.id}",
        "assets": {key: _href(asset) for key, asset in sorted(item.assets.items())},
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
    """Per solar day, the JSON granule list of the stamped ``items``, in fusion order.

    Only items the last load stamped are listed (:func:`stamp_fusion_positions`), so the list is
    the granules the fused slot was built from and its positions are the source index's numbering.
    """
    by_day: dict[str, list[Any]] = {}
    for item in items:
        if FUSION_POSITION in item.properties:
            by_day.setdefault(solar_day_of(item), []).append(item)
    out: dict[str, str] = {}
    for day, day_items in by_day.items():
        day_items.sort(key=lambda it: it.properties[FUSION_POSITION])
        records = [
            _record(it, provider=provider, collection=collection, catalog_url=catalog_url, collection_id=collection_id)
            for it in day_items
        ]
        out[day] = json.dumps(records, separators=(",", ":"), default=str)
    return out
