"""Read the provenance record: a day's granules, a shard's dates and granules, a pixel's granules.

The three questions of ``context_docs/storage/registry-provenance-plan.md`` §2, each answered as a
polars frame from the record beside a registry (layout in ``storage/provenance.py``). A shard's
pixels may come from several runs, listed oldest first in the registry's ``source_runs``. The pixel
readers walk them newest first: a holes update claims a pixel only where its ``update_pixels`` mask
marks it, and a fill or a shards update claims every pixel (plan §5).

``registry_root`` is a local path or an ``s3://`` URI; ``get_credentials`` and ``region`` are the
credential callback and region the rest of the store's readers take.
"""

from __future__ import annotations

import datetime
import math
from collections.abc import Callable
from typing import NamedTuple, cast

import icechunk
import numpy as np
import polars as pl
import zarr

from tessera_embeddings.config.store_layout import SENSORS, SHARD_PX
from tessera_embeddings.inference.chunk_spec import chunk_label, parse_chunk_label
from tessera_embeddings.ingest.land_mask import zone_for_cell
from tessera_embeddings.storage import zone_grid
from tessera_embeddings.storage.provenance import (
    GRANULES,
    SHARD_DATES,
    days_of,
    pixel_sources_group,
    pixel_sources_uri,
    table_dir,
    update_pixels_uri,
)
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group, plain_zarr_storage_options

_Creds = Callable[[], icechunk.S3StaticCredentials] | None

#: What a pixel's answer says about each date it was observed on.
PIXEL_COLUMNS = (
    "run_id",
    "sensor",
    "solar_day",
    "fusion_order",
    "granule_id",
    "platform",
    "acquired_at",
    "cloud_cover",
    "assets",
)

#: The columns ``shard_dates`` and ``granules`` join on.
_GRANULE_KEY = ["sensor", "snapshot_id", "solar_day", "fusion_order"]


class PixelLocation(NamedTuple):
    """A store pixel: its zone, its shard as the registry names it, and its row and column in it."""

    zone: str
    tile: str
    row: int
    col: int


def locate(lat: float, lon: float) -> PixelLocation:
    """The store pixel holding a WGS84 point, on the store's own zone grid (``zone_grid``).

    Raises:
        ValueError: the point is outside its zone's grid, beyond UTM's reach near the poles.
    """
    spec = zone_grid.zone(zone_for_cell(lon, lat))
    east, north = zone_grid.to_wgs84(int(spec.epsg)).transform(lon, lat, direction="INVERSE")
    row = math.floor((spec.northing[1] - north) / zone_grid.PIXEL_M)
    col = math.floor((east - spec.easting[0]) / zone_grid.PIXEL_M)
    if not (0 <= row < spec.height and 0 <= col < spec.width):
        raise ValueError(f"({lat}, {lon}) is outside zone {spec.group_name}'s grid")
    return PixelLocation(spec.group_name, chunk_label(row // SHARD_PX, col // SHARD_PX), row % SHARD_PX, col % SHARD_PX)


def day_granules(
    registry_root: str,
    zone: str,
    year: int,
    sensor: str,
    day: datetime.date,
    *,
    run_id: str | None = None,
    get_credentials: _Creds = None,
    region: str | None = None,
) -> pl.DataFrame:
    """Every granule fused into one sensor's mosaic of ``day``, in fusion order.

    One list per mosaic snapshot that holds the day, since each run that re-ingests builds its own
    mosaics; ``run_id`` keeps the list of the snapshot that run read.
    """
    scan = _Scanner(registry_root, zone, year, get_credentials, region)
    rows = scan(GRANULES).filter(pl.col("sensor") == sensor, pl.col("solar_day") == day)
    if run_id is not None:
        read = scan(SHARD_DATES).filter(pl.col("run_id") == run_id, pl.col("sensor") == sensor).select("snapshot_id")
        rows = rows.join(read.unique(), on="snapshot_id", how="semi")
    return rows.sort("snapshot_id", "fusion_order").collect()


def shard_granules(
    registry_root: str,
    zone: str,
    year: int,
    tile: str,
    run_id: str,
    *,
    get_credentials: _Creds = None,
    region: str | None = None,
) -> pl.DataFrame:
    """Per sensor and date ``run_id`` used in shard ``tile``: its valid pixels, and each granule that supplied any."""
    scan = _Scanner(registry_root, zone, year, get_credentials, region)
    return (
        scan(SHARD_DATES)
        .filter(pl.col("run_id") == run_id, pl.col("tile") == tile)
        .explode("sources")
        .rename({"sources": "fusion_order"})
        .join(scan(GRANULES), on=_GRANULE_KEY)
        .sort("sensor", "solar_day", "fusion_order")
        .collect()
    )


def pixel_granules(
    registry_root: str,
    lat: float,
    lon: float,
    year: int,
    *,
    get_credentials: _Creds = None,
    region: str | None = None,
) -> pl.DataFrame:
    """Per sensor and date in ``year``, the granule that supplied the store pixel at a WGS84 point."""
    return pixel_granules_at(registry_root, locate(lat, lon), year, get_credentials=get_credentials, region=region)


def pixel_granules_at(
    registry_root: str,
    where: PixelLocation,
    year: int,
    *,
    get_credentials: _Creds = None,
    region: str | None = None,
) -> pl.DataFrame:
    """Per sensor and date in ``year``, the granule that supplied the pixel ``where``, as :data:`PIXEL_COLUMNS`.

    Read from the run that claims the pixel among the shard's ``source_runs``. A date is listed when
    the pixel's observation was valid for the model; refused pixels have them too.

    Raises:
        LookupError: the registry holds no complete record for the shard (its ``source_runs`` is null).
    """
    scan = _Scanner(registry_root, where.zone, year, get_credentials, region)
    run = _claiming_run(registry_root, where, year, _source_runs(scan, where.tile), get_credentials, region)
    group = open_store_as_zarr_group(
        pixel_sources_uri(registry_root),
        group=pixel_sources_group(where.zone, year, run),
        get_credentials=get_credentials,
        region=region,
    )
    shard_px = cast(int, group.attrs["shard_px"])
    shard_row, shard_col = parse_chunk_label(where.tile)
    y, x = shard_row * shard_px + where.row, shard_col * shard_px + where.col
    seen: list[tuple[str, datetime.date, int]] = []
    for sensor in (s for s in SENSORS if s in group):
        ranks = np.asarray(cast(zarr.Array, group[sensor])[:, y, x])
        days = days_of(np.asarray(cast(zarr.Array, group[f"{sensor}_date"])[:]))
        seen += [(sensor, days[i], int(ranks[i]) - 1) for i in np.flatnonzero(ranks)]
    return (
        pl.LazyFrame(seen, schema={"sensor": pl.String, "solar_day": pl.Date, "rank": pl.Int64}, orient="row")
        .join(
            scan(SHARD_DATES).filter(pl.col("run_id") == run, pl.col("tile") == where.tile), on=["sensor", "solar_day"]
        )
        .with_columns(fusion_order=pl.col("sources").list.get(pl.col("rank")))
        .join(scan(GRANULES), on=_GRANULE_KEY)
        .select(PIXEL_COLUMNS)
        .sort("sensor", "solar_day")
        .collect()
    )


def _source_runs(scan: _Scanner, tile: str) -> list[str]:
    """The runs whose pixels ``tile`` holds, oldest first, from the registry's ``master/``.

    Raises:
        LookupError: the registry has no complete record for the shard, or no row for it.
    """
    rows = scan.master().filter(pl.col("tile") == tile).select("source_runs").collect()
    if rows.height != 1 or rows.item() is None:
        raise LookupError(f"{scan.zone}/{scan.year} {tile}: the registry holds no complete provenance record")
    return list(rows.item())


def _claiming_run(
    registry_root: str, where: PixelLocation, year: int, runs: list[str], get_credentials: _Creds, region: str | None
) -> str:
    """Newest first, the run whose pixel this is: a holes update where its mask marks it, else the run below.

    A run with no ``update_pixels`` is a fill or a shards update, which wrote every pixel of the shard.
    """
    for run in reversed(runs):
        uri = update_pixels_uri(registry_root, where.zone, year, run)
        try:
            masks = zarr.open_group(
                uri, mode="r", storage_options=plain_zarr_storage_options(uri, get_credentials, region)
            )
        except FileNotFoundError:
            return run
        if where.tile in masks and bool(cast(zarr.Array, masks[where.tile])[where.row, where.col]):
            return run
    raise LookupError(f"{where}: none of {runs} wrote this pixel")


class _Scanner:
    """Lazy scans of one zone-year's record tables and registry master, with the store's credentials."""

    def __init__(self, registry_root: str, zone: str, year: int, get_credentials: _Creds, region: str | None) -> None:
        self.root, self.zone, self.year = registry_root.rstrip("/"), zone, year
        self.options: dict[str, str] | None = None
        if self.root.startswith("s3://"):
            creds = get_credentials() if get_credentials is not None else None
            options = {
                "aws_region": region,
                "aws_access_key_id": creds and creds.access_key_id,
                "aws_secret_access_key": creds and creds.secret_access_key,
                "aws_session_token": creds and creds.session_token,
            }
            self.options = {k: v for k, v in options.items() if v} or None

    def __call__(self, table: str) -> pl.LazyFrame:
        return self._scan(f"{table_dir(self.root, table, self.zone, self.year)}/*.parquet")

    def master(self) -> pl.LazyFrame:
        return self._scan(f"{self.root}/master/zone={self.zone}/year={self.year}/*.parquet")

    def _scan(self, pattern: str) -> pl.LazyFrame:
        return pl.scan_parquet(pattern, hive_partitioning=False, storage_options=self.options)
