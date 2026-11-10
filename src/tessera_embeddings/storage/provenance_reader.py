"""Read the provenance record: a day's granules, a shard's dates and granules, a pixel's granules.

The three questions of ``context_docs/storage/registry-provenance-plan.md`` §2, each answered as a
polars frame from the record beside a registry (layout in ``storage/provenance.py``). A shard's
pixels may come from several runs, listed oldest first in the registry's ``source_runs``. The pixel
readers walk them newest first: a holes update claims a pixel only where its ``update_pixels`` mask
marks it, and a fill or a shards update claims every pixel (plan §5).

:class:`ProvenanceRecord` reads the record beside one registry; :func:`locate` finds a WGS84 point's
store pixel.
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

#: What a pixel's answer says about each date a granule had data there: whether the model used it,
#: and which granule it was.
PIXEL_COLUMNS = (
    "run_id",
    "sensor",
    "solar_day",
    "valid",
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


class ProvenanceRecord:
    """The record beside one registry, read as polars frames.

    ``registry_root`` is a local path or an ``s3://`` URI; ``get_credentials`` and ``region`` are the
    credential callback and region the rest of the store's readers take.
    """

    def __init__(self, registry_root: str, get_credentials: _Creds = None, region: str | None = None) -> None:
        """A reader of the record beside ``registry_root``."""
        self.root, self.get_credentials, self.region = registry_root.rstrip("/"), get_credentials, region
        self._options: dict[str, str] | None = None
        if self.root.startswith("s3://"):
            options: dict[str, str | None] = {"aws_region": region}
            if get_credentials is not None:
                creds = get_credentials()
                options |= {
                    "aws_access_key_id": creds.access_key_id,
                    "aws_secret_access_key": creds.secret_access_key,
                    "aws_session_token": creds.session_token,
                }
            self._options = {k: v for k, v in options.items() if v is not None} or None

    def day_granules(
        self, zone: str, year: int, sensor: str, day: datetime.date, run_id: str | None = None
    ) -> pl.DataFrame:
        """Every granule fused into one sensor's mosaic of ``day``, in fusion order, then the copies dropped.

        A dropped copy has no ``fusion_order`` and says why in ``status``: a losing duplicate, or a copy
        that would not read and was replaced. One list per mosaic snapshot that holds the day, since
        each run that re-ingests builds its own mosaics; ``run_id`` keeps the list of the snapshot that
        run read.
        """
        rows = self._table(zone, year, GRANULES).filter(pl.col("sensor") == sensor, pl.col("solar_day") == day)
        if run_id is not None:
            read = self._table(zone, year, SHARD_DATES).filter(pl.col("run_id") == run_id, pl.col("sensor") == sensor)
            rows = rows.join(read.select("snapshot_id").unique(), on="snapshot_id", how="semi")
        return rows.sort("snapshot_id", "fusion_order", "granule_id", nulls_last=True).collect()

    def shard_granules(self, zone: str, year: int, tile: str, run_id: str) -> pl.DataFrame:
        """Per sensor and date ``run_id`` used in shard ``tile``: its valid pixels, and each granule with data there."""
        return (
            self._table(zone, year, SHARD_DATES)
            .filter(pl.col("run_id") == run_id, pl.col("tile") == tile)
            .explode("sources")
            .rename({"sources": "fusion_order"})
            .join(self._table(zone, year, GRANULES), on=_GRANULE_KEY)
            .sort("sensor", "solar_day", "fusion_order")
            .collect()
        )

    def pixel_granules(self, lat: float, lon: float, year: int) -> pl.DataFrame:
        """Per sensor and date in ``year``, the granule that covered the store pixel at a WGS84 point."""
        return self.pixel_granules_at(locate(lat, lon), year)

    def pixel_granules_at(self, where: PixelLocation, year: int) -> pl.DataFrame:
        """Per sensor and date in ``year``, the granule that covered the pixel ``where``, as :data:`PIXEL_COLUMNS`.

        Read from the run that claims the pixel among the shard's ``source_runs``. A date is listed when
        a granule had data at the pixel; ``valid`` says whether the model used the observation or
        rejected it, a cloud for instance. A date not listed had no data there. Refused pixels have
        their dates too.

        Raises:
            LookupError: the registry holds no complete record for the shard (its ``source_runs`` is null).
        """
        run = self._claiming_run(where, year, self._source_runs(where, year))
        group = open_store_as_zarr_group(
            pixel_sources_uri(self.root),
            group=pixel_sources_group(where.zone, year, run),
            get_credentials=self.get_credentials,
            region=self.region,
        )
        shard_px = cast(int, group.attrs["shard_px"])
        shard_row, shard_col = parse_chunk_label(where.tile)
        y, x = shard_row * shard_px + where.row, shard_col * shard_px + where.col
        # A plane value k names `sources[|k| - 1]`; positive means the model used the observation.
        seen: list[tuple[str, datetime.date, bool, int]] = []
        for sensor in (s for s in SENSORS if s in group):
            ranks = np.asarray(cast(zarr.Array, group[sensor])[:, y, x]).astype(int)
            days = days_of(np.asarray(cast(zarr.Array, group[f"{sensor}_date"])[:]))
            seen += [(sensor, days[i], bool(ranks[i] > 0), abs(int(ranks[i])) - 1) for i in np.flatnonzero(ranks)]
        schema = {"sensor": pl.String, "solar_day": pl.Date, "valid": pl.Boolean, "rank": pl.Int64}
        rows = self._table(where.zone, year, SHARD_DATES).filter(pl.col("run_id") == run, pl.col("tile") == where.tile)
        return (
            pl.LazyFrame(seen, schema=schema, orient="row")
            .join(rows, on=["sensor", "solar_day"])
            .with_columns(fusion_order=pl.col("sources").list.get(pl.col("rank")))
            .join(self._table(where.zone, year, GRANULES), on=_GRANULE_KEY)
            .select(PIXEL_COLUMNS)
            .sort("sensor", "solar_day")
            .collect()
        )

    def _source_runs(self, where: PixelLocation, year: int) -> list[str]:
        """The runs whose pixels the shard holds, oldest first, from the registry's ``master/``.

        Raises:
            LookupError: the registry has no complete record for the shard, or no row for it.
        """
        master = self._scan(f"{self.root}/master/zone={where.zone}/year={year}/*.parquet")
        rows = master.filter(pl.col("tile") == where.tile).select("source_runs").collect()
        if rows.height != 1 or rows.item() is None:
            raise LookupError(f"{where.zone}/{year} {where.tile}: the registry holds no complete provenance record")
        return list(rows.item())

    def _claiming_run(self, where: PixelLocation, year: int, runs: list[str]) -> str:
        """Newest first, the run whose pixel this is: a holes update where its mask marks it, else the run below.

        A run with no ``update_pixels`` is a fill or a shards update, which wrote every pixel of the shard.
        """
        for run in reversed(runs):
            uri = update_pixels_uri(self.root, where.zone, year, run)
            options = plain_zarr_storage_options(uri, self.get_credentials, self.region)
            try:
                masks = zarr.open_group(uri, mode="r", storage_options=options)
            except FileNotFoundError:
                return run
            if where.tile in masks and bool(cast(zarr.Array, masks[where.tile])[where.row, where.col]):
                return run
        raise LookupError(f"{where}: none of {runs} wrote this pixel")

    def _table(self, zone: str, year: int, table: str) -> pl.LazyFrame:
        return self._scan(f"{table_dir(self.root, table, zone, year)}/*.parquet")

    def _scan(self, pattern: str) -> pl.LazyFrame:
        return pl.scan_parquet(pattern, hive_partitioning=False, storage_options=self._options)
