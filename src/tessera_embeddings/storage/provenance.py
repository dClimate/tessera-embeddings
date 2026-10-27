"""The provenance record beside the registry: where its pieces live and the shape of its tables.

Everything sits under ``<registry>/provenance/`` and is written once per run, never rewritten by
another run (``context_docs/storage/registry-provenance-plan.md`` §3, Appendix A)::

    provenance/
    ├── granules/      zone=<Z>/year=<Y>/<run_id>.parquet  one row per granule per solar day per sensor
    ├── shard_dates/   zone=<Z>/year=<Y>/<run_id>.parquet  one row per shard, sensor and date a run used
    ├── update_pixels/ zone=<Z>/year=<Y>/<run_id>.zarr     holes updates only: the pixels the run embedded
    └── pixel_sources.icechunk/                          one Icechunk repository, a group per run
        └── <Z>/<Y>/<run_id>/{s2,s1_asc,s1_desc}         (date, northing, easting) uint8 ranks

As in ``registry``, ``zone`` and ``year`` are the hive partition keys and not columns: carrying them
as columns too makes pyarrow refuse the dataset (see that module's docstring). Every reader that
uses hive partitioning, which is every Parquet engine's default for this layout, gets them as
columns anyway. ``run_id`` is a column, so files from several runs join without parsing paths.
"""

from __future__ import annotations

import datetime
import io
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pyarrow as pa

#: The record's tables, each a hive-partitioned Parquet dataset.
GRANULES, SHARD_DATES = "granules", "shard_dates"


def _root(registry_root: str) -> str:
    return f"{registry_root.rstrip('/')}/provenance"


def table_uri(registry_root: str, table: str, zone: str, year: int, run_id: str) -> str:
    """One run's file of ``table`` for one zone-year."""
    return f"{_root(registry_root)}/{table}/zone={zone}/year={year}/{run_id}.parquet"


def update_pixels_uri(registry_root: str, zone: str, year: int, run_id: str) -> str:
    """A holes update's per-shard pixel masks for one zone-year."""
    return f"{_root(registry_root)}/update_pixels/zone={zone}/year={year}/{run_id}.zarr"


def pixel_sources_uri(registry_root: str) -> str:
    """The Icechunk repository holding every run's per-pixel source planes."""
    return f"{_root(registry_root)}/pixel_sources.icechunk"


def pixel_sources_group(zone: str, year: int, run_id: str) -> str:
    """One run's group inside :func:`pixel_sources_uri`."""
    return f"{zone}/{year}/{run_id}"


def granules_schema() -> pa.Schema:
    """One row per granule per solar day per sensor, in the day's fusion order (the plan's Appendix A).

    ``geometry`` is the footprint as the catalogue published it, WKB in WGS84 (GeoParquet 1.1); the
    four ``bbox_*`` columns let any Parquet reader filter without decoding it.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("sensor", pa.string(), nullable=False),
            pa.field("solar_day", pa.date32(), nullable=False),
            pa.field("fusion_order", pa.int16(), nullable=False),
            pa.field("granule_id", pa.string(), nullable=False),
            pa.field("image_key", pa.string()),
            pa.field("platform", pa.string()),
            pa.field("acquired_at", pa.timestamp("us", tz="UTC")),
            pa.field("cloud_cover", pa.float64()),
            pa.field("processing_baseline", pa.string()),
            pa.field("product_uri", pa.string()),
            pa.field("provider", pa.string()),
            pa.field("collection", pa.string()),
            pa.field("stac_item_url", pa.string()),
            pa.field("assets", pa.map_(pa.string(), pa.string())),
            pa.field("stepped_down", pa.bool_()),
            pa.field("geometry", pa.binary()),
            pa.field("bbox_west", pa.float64()),
            pa.field("bbox_south", pa.float64()),
            pa.field("bbox_east", pa.float64()),
            pa.field("bbox_north", pa.float64()),
        ]
    )


def shard_dates_schema() -> pa.Schema:
    """One row per shard, sensor and date a run used (the plan's Appendix A).

    ``tile`` names the shard as the registry does (``chunk_<row>_<col>``). ``sources`` are the
    ``fusion_order`` values that supplied at least one of the shard's pixels that date, ascending;
    a pixel's ``pixel_sources`` value k means ``sources[k - 1]``.
    """
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("tile", pa.string(), nullable=False),
            pa.field("sensor", pa.string(), nullable=False),
            pa.field("solar_day", pa.date32(), nullable=False),
            pa.field("valid_px", pa.int32(), nullable=False),
            pa.field("embedded_px", pa.int32(), nullable=False),
            pa.field("sources", pa.list_(pa.int16()), nullable=False),
        ]
    )


def granule_rows(run_id: str, sensor: str, days: Sequence[datetime.date], lists: Sequence[str]) -> list[dict[str, Any]]:
    """The ``granules`` rows for one mosaic store: one per entry of each slot's JSON list.

    ``days`` is the store's time axis and ``lists`` its ``granules`` array, slot for slot.
    """
    import shapely

    rows: list[dict[str, Any]] = []
    for day, raw in zip(days, lists, strict=True):
        for order, granule in enumerate(json.loads(raw)):
            bbox = granule.get("bbox") or [None] * 4
            geometry = granule.get("geometry")
            rows.append(
                {
                    "run_id": run_id,
                    "sensor": sensor,
                    "solar_day": day,
                    "fusion_order": order,
                    "granule_id": granule["granule_id"],
                    "image_key": granule.get("image_key"),
                    "platform": granule.get("platform"),
                    "acquired_at": _instant(granule.get("acquired_at")),
                    "cloud_cover": granule.get("cloud_cover"),
                    "processing_baseline": granule.get("processing_baseline"),
                    "product_uri": granule.get("product_uri"),
                    "provider": granule.get("provider"),
                    "collection": granule.get("collection"),
                    "stac_item_url": granule.get("stac_item_url"),
                    "assets": granule.get("assets") or {},
                    "stepped_down": bool(granule.get("stepped_down")),
                    "geometry": shapely.to_wkb(shapely.geometry.shape(geometry)) if geometry else None,
                    "bbox_west": bbox[0],
                    "bbox_south": bbox[1],
                    "bbox_east": bbox[2],
                    "bbox_north": bbox[3],
                }
            )
    return rows


def _instant(value: object) -> datetime.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))


def shard_date_rows(
    run_id: str,
    tile: str,
    sensor: str,
    summary: Mapping[str, Sequence[Any]],
    days: Sequence[datetime.date],
    embedded_px: int,
) -> list[dict[str, Any]]:
    """The ``shard_dates`` rows of one shard and sensor from its staged summary.

    ``summary`` is ``{"time", "sources", "valid_px"}`` as the actor staged it; ``days`` maps a
    mosaic time index to its solar day.
    """
    return [
        {
            "run_id": run_id,
            "tile": tile,
            "sensor": sensor,
            "solar_day": days[int(time)],
            "valid_px": int(valid),
            "embedded_px": int(embedded_px),
            "sources": [int(s) for s in sources],
        }
        for time, sources, valid in zip(summary["time"], summary["sources"], summary["valid_px"], strict=True)
    ]


def write_table(
    uri: str,
    rows: Iterable[dict[str, Any]],
    schema: pa.Schema,
    *,
    open_output: Callable[[str], AbstractContextManager[Any]],
    metadata: Mapping[str, str],
    geometry: bool = False,
) -> int:
    """Write ``rows`` as one zstd Parquet file at ``uri``; returns the rows written.

    Buffered and written in one shot, as registry parts are: a partial Parquet file is unreadable.
    ``metadata`` lands in the file's key-value block; ``geometry`` adds GeoParquet's ``geo`` entry
    for the WKB ``geometry`` column.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(list(rows), schema=schema)
    meta = {k.encode(): v.encode() for k, v in metadata.items()}
    if geometry:
        meta[b"geo"] = json.dumps(
            {
                "version": "1.1.0",
                "primary_column": "geometry",
                "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon", "MultiPolygon"]}},
            }
        ).encode()
    table = table.replace_schema_metadata(meta)
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    with open_output(uri) as f:
        f.write(buf.getvalue())
    return table.num_rows


def days_of(times: np.ndarray) -> list[datetime.date]:
    """A mosaic time axis (int64 nanoseconds or datetime64) as solar days."""
    return [datetime.date.fromisoformat(str(t)[:10]) for t in np.asarray(times).astype("datetime64[ns]")]
