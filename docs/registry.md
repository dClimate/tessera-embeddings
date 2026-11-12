# The TESSERA v1.1 registry

A small Parquet table beside the global TESSERA embeddings store
(`s3://tessera-embeddings/v1.1/dclimate.icechunk/`). It has **one row per tile per year**: a tile
is 2048 × 2048 pixels, about 20 km on a side. Each row says whether the tile holds embeddings, how
much of its land the quality rule left empty and why, and how much imagery there was. Use it to
find out whether an area is covered, and how well, before opening the store. It is about 150 MB
and anyone can read it anonymously.

This page is published as `README.md` in `s3://tessera-embeddings/v1.1/dclimate.registry/`. Its
source is [`docs/registry.md`][self] in the tessera-embeddings repository, which built the store
and maintains this table.

## What is in this folder

```
  dclimate.registry/
  ├── README.md     this page
  ├── master/       READ THIS: one row per tile per year, describing the store as it is now
  │   ├── _common_metadata                          the dataset's schema
  │   └── zone=<ZONE>/year=<YEAR>/part-0.parquet    one file per zone-year
  └── runs/         internal: one file per run, named by its run ID
      └── zone=<ZONE>/year=<YEAR>/<run_id>.parquet
```

`<ZONE>` is one of the store's 120 groups, a UTM zone and hemisphere from `01N` to `60S`, and
`<YEAR>` runs from 2017 to 2025. 112 groups hold land. On 2026-10-12 `master/` held 3,247,410 rows
in 994 zone-year files: every one of those groups in 2022–2025, and all but 14 zone-years in
2017–2021, which were never filled and so have no file.

**Read `master/`, not `runs/`.** `runs/` is the raw record the master is built from. It can hold
a tile more than once, one row per run that wrote it, and 47% of its v1.1 rows lack measurements
that `master/` carries. A copy of the registry taken before the folder was renamed has `parts/`
where this has `runs/`, with the same files inside; [`rename_registry_parts_to_runs.py`][rename]
moves them across.

### Runs, and how `runs/` relates to `master/`

A *run* is one fill of one zone-year under one set of inputs. Its ID is
`<ZONE>-<YEAR>-<fingerprint>`, such as `43S-2020-090ca792`, where the fingerprint hashes
everything that decides the run's pixels: the fill's settings, its code and the imagery it read.
Each run writes one file, named by its ID, into its zone-year's folder, and never touches another
run's file:

```
  runs/zone=43S/year=2020/
  ├── 43S-2020-090ca792.parquet    the first fill: a row for every tile it evaluated
  └── 43S-2020-<later-id>.parquet  a later run, such as an update: rows for the tiles it wrote
```

- **A re-run with unchanged inputs keeps its ID**, such as a retry after an interruption, so it
  replaces its own file rather than adding one.
- **A run with any input changed gets a new ID** and a new file beside the old one, so `runs/`
  keeps every run's own account of the tiles it wrote.
- **`master/` combines a zone-year's runs tile by tile**, taking each tile's row from the newest
  run that wrote it. The first fill's file goes on describing every tile no later run touched.
- **v1.1 has one run per zone-year**, so `runs/` holds 994 files, one for each zone-year file in
  `master/`.

A master row's `run_id` names the file it came from, with one exception. The 1,519,045 re-derived
rows (`measured_by = "rebuild"`) carry the re-derivation's ID, `rebuild-…`, which has no file;
their fill's file is the one in the same zone-year whose rows' `assembled_at` equals the row's
`filled_at`.

## Reading it

### Python

```python
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyarrow.fs import S3FileSystem

fs = S3FileSystem(region="us-west-2", anonymous=True)
master = "tessera-embeddings/v1.1/dclimate.registry/master"
schema = pq.read_schema(f"{master}/_common_metadata", filesystem=fs)
registry = ds.dataset(master, filesystem=fs, schema=schema, partitioning="hive")

# Every 2024 tile that touches a box around Rome, which lies in UTM zone 33 north.
west, south, east, north = 12.3, 41.7, 12.7, 42.0
f = ds.field
rows = registry.to_table(filter=(
    (f("zone") == "33N") & (f("year") == 2024)
    & (f("bbox_west") <= east) & (f("bbox_east") >= west)
    & (f("bbox_south") <= north) & (f("bbox_north") >= south)
)).to_pandas()
```

Pass the schema as shown. Filtering on `zone` and `year` reads only that zone-year's file, which
takes about 1.7 s in `us-west-2`, against about 14 s for the whole table. To find the zone, number
`NN` covers longitudes from `6·NN − 186` to `6·NN − 180` degrees, and `N` or `S` is the
hemisphere. Or drop the `zone` filter and let the box do the work at the cost of reading every
file.

### SQL

With [DuckDB](https://duckdb.org):

```sql
INSTALL httpfs; LOAD httpfs;
SET s3_region = 'us-west-2';
SELECT year,
       count(*)                                   AS tiles,
       count(*) FILTER (WHERE embedded)           AS embedded,
       round(100 * sum(refused_px) / sum(coalesce(eligible_px, chunk_px)), 2) AS pct_refused
FROM read_parquet('s3://tessera-embeddings/v1.1/dclimate.registry/master/*/*/part-0.parquet',
                  hive_partitioning = true)
WHERE zone = '33N'
GROUP BY year ORDER BY year;
```

### From a row to its pixels

`tile` is `chunk_<row>_<col>`, the tile's position in its zone group's grid of 2048-pixel shards,
and each tile is exactly one shard in the store. Its embeddings for a year are

```python
store_group.isel(time=year - 2017,
                 northing=slice(row * 2048, (row + 1) * 2048),
                 easting=slice(col * 2048, (col + 1) * 2048))
```

where `store_group` is the zone's group opened with xarray, as in the store's
[reading guide][store].

## The columns

| column | what it says |
|---|---|
| `zone`, `year`, `tile` | which tile: the store group, the year, and the tile's position in the group's shard grid |
| `bbox_west`, `bbox_south`, `bbox_east`, `bbox_north` | the tile's footprint in WGS84 degrees |
| `embedded` | whether the tile holds any embeddings at all, not whether it is whole |
| `chunk_px` | the tile's pixel count, 4,194,304 |
| `eligible_px` | the pixels the quality rule was applied to: fewer than `chunk_px` where the fill evaluated only part of the tile, and null on re-derived rows |
| `refused_px` | pixels the quality rule left empty: the sum of the next three columns |
| `refused_no_optical_px` | refused with no usable optical observation all year |
| `refused_thin_px` | refused with some optical observations, but fewer than `optical_min_obs` |
| `refused_no_radar_px` | refused for having no radar; always 0 in v1.1, where that rule was off |
| `px_with_any_optical` | pixels with at least one usable optical observation |
| `obs_max` | the most usable optical observations any pixel in the tile had |
| `median_obs_where_any` | the median optical observation count over pixels with at least one |
| `median_obs_where_thin` | the median over the refused thin pixels only, null when there are none |
| `px_with_any_radar` | pixels with at least one radar observation |
| `optical_min_obs` | the quality rule's line this row was judged against: 15 for v1.1's campaign fill, the update's line for a tile an update rewrote |
| `radar_rule_enforced` | whether pixels without radar were refused; false in v1.1 |
| `measured_by` | `"fill"` if the fill measured this row's numbers, `"rebuild"` if they were re-derived from the store later |
| `filled_at` | when the fill whose pixels the row describes landed |
| `run_id`, `assembled_at` | the run the row came from, and when it was written: the fill, or for a re-derived row the re-derivation |
| `code_version`, `code_commit` | the build of the tessera-embeddings repository that ran the fill |
| `update_name` | the targeted update that last rewrote the tile ([ADR 027](../context_docs/decisions/027-targeted-updates.md)), null where the tile is still the campaign's fill |
| `source_runs` | on `master/` only: the runs whose pixels the tile holds, oldest first, for finding them in the provenance record below; null for v1.1 and wherever the tile has no complete record |

The quality rule is the store's: a pixel is embedded only if it had at least `optical_min_obs`
usable optical observations in the year: the store root's value, or a lower one a targeted update
used for its tiles, and each row's own `optical_min_obs` says which. The campaign's line is one, at
which `refused_thin_px` is always 0; a v1.1 row still at 15 is a tile no update has yet reached at a
lower line
([ADR 028](../context_docs/decisions/028-embed-every-pixel-with-an-optical-observation.md)). Every
count above comes from the store's own per-pixel observation-count arrays, so any row can be checked
against the store.

## What catches people out

- **`embedded` is not the same as complete.** 1,188,227 of v1.1's 3,229,545 embedded tile-years,
  37%, have some refused pixels. Read `refused_px` alongside `embedded`.
- **A missing row means the tile was never evaluated**: open sea beyond the land mask's margin of
  about 11 km, or Antarctica. A row with `embedded = false`, 17,865 of them in v1.1, is land that
  was evaluated and refused in full.
- **Rows crossing the antimeridian have `bbox_west > bbox_east`.** These are 675 rows in zones 01
  and 60. A test of `bbox_west <= lon <= bbox_east` silently drops them; for those rows use
  `lon >= bbox_west or lon <= bbox_east`.
- **Divide refusals by `eligible_px` where it is set, and by `chunk_px` otherwise.** Re-derived rows
  leave `eligible_px` null and count over the whole tile. The two differ on only 237 tiles.
- **To ask where more imagery would help, rank by `median_obs_where_thin`.** `median_obs_where_any`
  also counts the pixels that passed, so on a partly refused tile it hides how thin the rest is.
  A tile whose refusals are mostly `refused_no_optical_px` was never imaged usably at all, which
  more of the same imagery will not fix.
- **Neighbouring zones overlap by up to one tile**, so a box near a zone boundary matches tiles in
  both groups. When combining zones, take each point from the zone whose 6° band contains it.

## How it is kept current

Every run, whether a zone-year's first fill, a refill or a later update of some of its tiles,
reaches the registry right after its pixels are committed to the store:

```
  a run commits to the store ──▶ writes its rows as a new file in runs/
                             ──▶ merges them into master/zone=Z/year=Y/part-0.parquet
```

- **A tile's row comes from the newest run that wrote that tile.** A full refill replaces every
  row in its zone-year. An update of some tiles replaces only the tiles it rewrote, and every other
  tile keeps its row. An update that filled a tile's holes merged two runs' pixels, so its row is
  re-derived from the tile as committed rather than taken from either run. The master follows the
  store, which also keeps each tile as the last run that wrote it left it.
- **Each zone-year file is replaced in a single write**, so a reader sees the old file or the new
  one, never a mixture, and zone-years never wait on each other.
- **`_common_metadata` follows the schema.** When a column is added, the next update rewrites it, so
  a reader taking the dataset's schema from it sees the new column; cell files written before it
  read it as null.
- **`measured_by` records where the numbers came from.** Fills that resumed after an interruption
  published 1,519,045 of v1.1's rows without their measurements. Those were re-derived from the
  store's observation counts on 2026-10-12, by a method that matched the fill's own numbers
  exactly on every uncropped tile of a 39,578-row sample the fill had measured. Fills now keep
  their measurements across a resume.
- **The registry is checked against the store.** Each merge compares the master's embedded
  tiles with the shards the store actually holds, and a census checks `master/` against `runs/`
  zone-year by zone-year. Either names any zone-year that fails, for repair from `runs/` and the
  store. Every column can be derived from the store, so the registry is a convenience, never a
  second source of truth.

## Which imagery is behind an embedding (v2 onward)

From v2, stores filled for publication also carry a provenance record under `provenance/`,
beside `master/` and `runs/`. It names the satellite images behind every embedding, down to the
one image that supplied each pixel on each date. v1.1 has none: its imagery was deleted before
the record existed. A zone-year filled without the record has no files under `provenance/`.

```
  provenance/
  ├── granules/      zone=<ZONE>/year=<YEAR>/<sensor>-<snapshot_id>.parquet  each mosaic's images, per day
  ├── shard_dates/   zone=<ZONE>/year=<YEAR>/<run_id>.parquet   per run, tile, sensor and date
  ├── update_pixels/ zone=<ZONE>/year=<YEAR>/<run_id>.zarr      holes updates only: the pixels they embedded
  └── pixel_sources.icechunk/                                  per run, per pixel and date: which image
```

A tile here is the same 2048-pixel unit as a registry row, and the `tile` column names it the same
way. The sensors are `s2` (optical), `s1_asc` and `s1_desc` (radar by orbit direction).

- **`granules`**: one row per image per day per sensor, in the order the day's mosaic was built
  (`fusion_order`, 0 first; the first image with data at a pixel supplies it). A file holds one
  mosaic snapshot's lists, named on every row as `snapshot_id`, and is written once, by the first
  run that read that snapshot. Each row carries the image's catalogue ID and link, acquisition
  time, cloud cover, footprint (GeoParquet) and the exact object URLs that were read. A URL that no
  longer resolves is itself information: the catalogue entry has moved since. After a day's fused
  images come the copies the ingest set aside before building the mosaic, with no `fusion_order`
  and a `status` saying why: `duplicate`, another copy of an image it used; `unreadable`, a copy
  that would not read and was replaced; or `missing_bands`, a radar image whose catalogue entry
  lacks its VV or VH file. Fused rows have `status` `fused`. A day the ingest skipped whole has
  no fused rows, only its images, each marked with the day's reason: `low_coverage`,
  `producer_conflict` or `unreadable`.
- **`shard_dates`**: one row per tile, sensor and date on which an image had data at the tile, with
  `valid_px`, the tile's pixels whose observation the model used that day; `sources`, the
  `fusion_order` of every image with data at at least one of its pixels, used or not; and
  `snapshot_id`, the mosaic the run read, which is the key into `granules`. It answers questions
  about a whole tile; a pixel's lookup does not need it.
- **`pixel_sources`**: one Icechunk repository with a group `<ZONE>/<YEAR>/<run_id>`, holding a
  `(date, northing, easting)` `int16` array per sensor on the zone group's grid, and its dates as
  `<sensor>_date`. The group's attributes name the mosaic each sensor's array describes, as
  `snapshot_<sensor>`. A value `k` names the image at `fusion_order = |k| - 1` in that mosaic's
  `granules` list for the date. Positive means the model used that observation; negative means the
  image had data there and the model rejected it, a cloud for instance; `0` means no image had
  data.

### Reading the record with the helpers

`tessera_embeddings.storage.provenance_reader` answers the three usual questions as polars tables.
`REGISTRY` is the registry's root, beside the store it describes; `anonymous=True` reads a public
registry without an AWS account.

```python
import datetime as dt
from tessera_embeddings.storage.provenance_reader import ProvenanceRecord, locate

record = ProvenanceRecord(REGISTRY, region="us-west-2", anonymous=True)

# A pixel: one row per sensor and date an image had data there, from the run that holds the pixel.
record.pixel_granules(45.5071, 14.3761, 2024)
locate(45.5071, 14.3761)   # PixelLocation(zone='33N', tile='chunk_209_14', row=1908, col=70)

# A tile: per sensor and date one run recorded, its valid pixels and every image with data there.
record.shard_granules("33N", 2024, "chunk_209_14", run_id)

# A day: every image in the optical mosaic of 9 May, in fusion order, then the copies set aside.
record.day_granules("33N", 2024, "s2", dt.date(2024, 5, 9))
```

| call | one row per | columns |
|---|---|---|
| `pixel_granules(lat, lon, year)`, or `pixel_granules_at(locate(...), year)` | sensor and date with data at the pixel | `run_id`, `sensor`, `solar_day`, `valid` (did the model use it), `fusion_order`, `granule_id`, `platform`, `acquired_at`, `cloud_cover`, `assets` (the URLs read) |
| `shard_granules(zone, year, tile, run_id)` | sensor, date and image with data in the tile | `shard_dates`' columns and the image's `granules` row |
| `day_granules(zone, year, sensor, day, run_id=None)` | image in the day's mosaic, then each copy set aside | the `granules` row; `run_id` keeps the mosaic that run read |

The pixel calls raise `LookupError` where the tile has no complete record (below), and
`ValueError` for a point outside its zone's grid.

### Which run holds a pixel

A tile's pixels can come from more than one run: a holes update fills only the pixels the run
before it left empty. The master row's `source_runs` lists the runs whose pixels the tile holds,
oldest first. Walk it from the newest: a holes update holds the pixels its `update_pixels` mask
marks, the pixels it embedded, and a fill or a shards update holds every pixel; the first run that
holds the pixel has its record. Every merge derives `source_runs` afresh from the store's shard
commits, in the order they landed: a fill or a shards update starts the list again, and a holes
update adds itself.

`source_runs` is null where the tile has no complete record, such as a run from before the record
existed. One more case has none: a refused tile whose run staged no record for it, which that
run's `shard_dates` file lists in its `unrecorded_tiles` metadata.

### Reading the record by hand

Without the package, with polars, zarr and icechunk. `TILE`, `ROW` and `COL` are what `locate`
returns; the store's [reading guide][store] converts a point to a pixel without it.

```python
import icechunk, numpy as np, polars as pl, zarr

REGISTRY = "s3://<bucket>/<store>.registry"
ZONE, YEAR, TILE, ROW, COL = "33N", 2024, "chunk_209_14", 1908, 70
S3 = {"aws_region": "us-west-2", "skip_signature": "true"}   # anonymous reads
prov = f"{REGISTRY}/provenance"

# 1. The runs whose pixels the tile holds, oldest first. None: the tile has no complete record.
master = pl.read_parquet(f"{REGISTRY}/master/zone={ZONE}/year={YEAR}/part-0.parquet", storage_options=S3)
runs = master.filter(pl.col("tile") == TILE)["source_runs"].item()

# 2. The run that holds the pixel: newest first, a holes update where its mask marks the pixel,
#    else a fill or a shards update, which has no mask.
for run_id in reversed(runs):
    try:
        mask = zarr.open_group(f"{prov}/update_pixels/zone={ZONE}/year={YEAR}/{run_id}.zarr",
                               mode="r", storage_options={"anon": True})
    except FileNotFoundError:
        break
    if TILE in mask and mask[TILE][ROW, COL]:
        break
meta = pl.read_parquet_metadata(f"{prov}/shard_dates/zone={ZONE}/year={YEAR}/{run_id}.parquet", storage_options=S3)
assert TILE not in meta.get("unrecorded_tiles", "").split(","), "a refused tile its run recorded nothing for"

# 3. The pixel's value on each date, per sensor, from that run's group.
bucket, key = prov.removeprefix("s3://").split("/", 1)
repo = icechunk.Repository.open(icechunk.s3_storage(
    bucket=bucket, prefix=f"{key}/pixel_sources.icechunk", region="us-west-2", anonymous=True))
run = zarr.open_group(repo.readonly_session("main").store, path=f"{ZONE}/{YEAR}/{run_id}", mode="r")
r, c = (int(n) for n in TILE.removeprefix("chunk_").split("_"))
y, x = r * run.attrs["shard_px"] + ROW, c * run.attrs["shard_px"] + COL   # on the zone's grid
seen = []
for sensor in ("s2", "s1_asc", "s1_desc"):
    if sensor not in run:
        continue
    k = run[sensor][:, y, x].astype(np.int64)        # 0: no image; > 0 used; < 0 rejected
    days = run[f"{sensor}_date"][:].astype("datetime64[ns]").astype("datetime64[D]")
    seen.append(pl.DataFrame({
        "sensor": sensor, "snapshot_id": run.attrs[f"snapshot_{sensor}"], "solar_day": days[k != 0],
        "valid": k[k != 0] > 0, "fusion_order": (np.abs(k[k != 0]) - 1).astype(np.int16),
    }))
seen = pl.concat(seen)

# 4. Each value, its date and the run's snapshot are a `granules` key: the image itself.
granules = pl.read_parquet(f"{prov}/granules/zone={ZONE}/year={YEAR}/*.parquet", storage_options=S3)
pixel = seen.join(granules, on=["sensor", "snapshot_id", "solar_day", "fusion_order"]).sort("sensor", "solar_day")
```

`pixel` holds the same rows as `pixel_granules`. The tables are plain Parquet, so SQL reads them
too; with [DuckDB](https://duckdb.org), the pixel joins the `seen` frame from step 3:

```python
import duckdb

con = duckdb.connect()
con.sql("INSTALL httpfs; LOAD httpfs; SET s3_region = 'us-west-2'")
table = f"{prov}/{{}}/zone={ZONE}/year={YEAR}/*.parquet"

# A day: the optical mosaic of 9 May, its images in fusion order, then the copies set aside.
# One set per mosaic snapshot that holds the day.
con.sql(f"""
SELECT snapshot_id, fusion_order, status, granule_id, platform, acquired_at, cloud_cover
FROM read_parquet('{table.format("granules")}')
WHERE sensor = 's2' AND solar_day = DATE '2024-05-09'
ORDER BY snapshot_id, fusion_order NULLS LAST, granule_id""")

# A tile: per sensor and date the run recorded, every image with data at one of its pixels.
con.sql(f"""
SELECT d.sensor, d.solar_day, d.valid_px, g.fusion_order, g.granule_id, g.acquired_at, g.cloud_cover
FROM read_parquet('{table.format("shard_dates")}') AS d, UNNEST(d.sources) AS s(fusion_order)
JOIN read_parquet('{table.format("granules")}') AS g
  ON g.sensor = d.sensor AND g.snapshot_id = d.snapshot_id AND g.solar_day = d.solar_day
 AND g.fusion_order = s.fusion_order
WHERE d.run_id = '{run_id}' AND d.tile = '{TILE}'
ORDER BY d.sensor, d.solar_day, g.fusion_order""")

# The pixel.
con.sql(f"""
SELECT p.sensor, p.solar_day, p.valid, g.fusion_order, g.granule_id, g.acquired_at, g.cloud_cover, g.assets
FROM seen AS p
JOIN read_parquet('{table.format("granules")}') AS g
  ON g.sensor = p.sensor AND g.snapshot_id = p.snapshot_id AND g.solar_day = p.solar_day
 AND g.fusion_order = p.fusion_order
ORDER BY p.sensor, p.solar_day""")
```

Join `granules` through a snapshot, never through `run_id`: a mosaic's list is written once and
shared by every run that read that mosaic.

The record says what went in, not how much each image mattered: the model has no attribution. For
a pixel observed more often than the model's longest sequence, the model reads a fixed subset of
these dates, chosen by a deterministic rule from the dates alone.

## Going deeper

- Reading the embeddings themselves: [`global-store.md`][store].
- The merge rule, its checks, repair and rollback: [`registry-master.md`][master].
- How the v1.1 measurements were re-derived, and the proof that they match:
  [`registry-backfill.md`][backfill].
- The registry audited against the store tile for tile:
  [`reading-the-published-store.md`, section 5][audit].
- How the provenance record is made, sized and checked: [`registry-provenance-plan.md`][provenance].
- Why the provenance record is shaped as it is: [ADR 035][provenance-adr].

[self]: https://github.com/dClimate/tessera-embeddings/blob/main/docs/registry.md
[rename]: https://github.com/dClimate/tessera-embeddings/blob/main/scripts/maintenance/rename_registry_parts_to_runs.py
[store]: https://github.com/dClimate/tessera-embeddings/blob/main/docs/global-store.md
[master]: https://github.com/dClimate/tessera-embeddings/blob/main/context_docs/storage/registry-master.md
[backfill]: https://github.com/dClimate/tessera-embeddings/blob/main/context_docs/storage/registry-backfill.md
[audit]: https://github.com/dClimate/tessera-embeddings/blob/main/context_docs/storage/reading-the-published-store.md#5-the-registry-beside-the-store
[provenance]: https://github.com/dClimate/tessera-embeddings/blob/main/context_docs/storage/registry-provenance-plan.md
[provenance-adr]: https://github.com/dClimate/tessera-embeddings/blob/main/context_docs/decisions/035-a-provenance-record-for-the-published-store.md
