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
  └── parts/        internal: one file per fill run, named by its run ID
      └── zone=<ZONE>/year=<YEAR>/<run_id>.parquet
```

`<ZONE>` is one of the store's 120 groups, a UTM zone and hemisphere from `01N` to `60S`, and
`<YEAR>` runs from 2017 to 2025. 112 groups hold land. On 2026-10-12 `master/` held 3,247,410 rows
in 994 zone-year files: every one of those groups in 2022–2025, and all but 14 zone-years in
2017–2021, which were never filled and so have no file.

**Read `master/`, not `parts/`.** `parts/` is the raw record the master is built from. It can hold
a tile more than once, one row per run that wrote it, and 47% of its v1.1 rows lack measurements
that `master/` carries.

### Runs, and how `parts/` relates to `master/`

A *run* is one fill of one zone-year under one set of inputs. Its ID is
`<ZONE>-<YEAR>-<fingerprint>`, such as `43S-2020-090ca792`, where the fingerprint hashes
everything that decides the run's pixels: the fill's settings, its code and the imagery it read.
Each run writes one file, named by its ID, into its zone-year's folder, and never touches another
run's file:

```
  parts/zone=43S/year=2020/
  ├── 43S-2020-090ca792.parquet    the first fill: a row for every tile it evaluated
  └── 43S-2020-<later-id>.parquet  a later run, such as an update: rows for the tiles it wrote
```

- **A re-run with unchanged inputs keeps its ID**, such as a retry after an interruption, so it
  replaces its own file rather than adding one.
- **A run with any input changed gets a new ID** and a new file beside the old one, so `parts/`
  keeps every run's own account of the tiles it wrote.
- **`master/` combines a zone-year's runs tile by tile**, taking each tile's row from the newest
  run that wrote it. The first fill's file goes on describing every tile no later run touched.
- **v1.1 has one run per zone-year**, so `parts/` holds 994 files, one for each zone-year file in
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
| `optical_min_obs` | the quality rule's line, 15 observations in v1.1 |
| `radar_rule_enforced` | whether pixels without radar were refused; false in v1.1 |
| `measured_by` | `"fill"` if the fill measured this row's numbers, `"rebuild"` if they were re-derived from the store later |
| `filled_at` | when the fill whose pixels the row describes landed |
| `run_id`, `assembled_at` | the run the row came from, and when it was written: the fill, or for a re-derived row the re-derivation |
| `code_version`, `code_commit` | the build of the tessera-embeddings repository that ran the fill |
| `update_name` | the targeted update that last rewrote the tile ([ADR 027](../context_docs/decisions/027-targeted-updates.md)), null where the tile is still the campaign's fill |

The quality rule is the store's: a pixel is embedded only if it had at least 15 usable optical
observations in the year. Every count above comes from the store's own per-pixel observation-count
arrays, so any row can be checked against the store.

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
  a run commits to the store ──▶ writes its rows as a new file in parts/
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
- **The registry is checked against the store.** Each update compares the master's embedded
  tiles with the shards the store actually holds, and a census checks `master/` against `parts/`
  zone-year by zone-year. Either names any zone-year that fails, for repair from `parts/` and the
  store. Every column can be derived from the store, so the registry is a convenience, never a
  second source of truth.

## Going deeper

- Reading the embeddings themselves: [`global-store.md`][store].
- The merge rule, its checks, repair and rollback: [`registry-master.md`][master].
- How the v1.1 measurements were re-derived, and the proof that they match:
  [`registry-backfill.md`][backfill].
- The registry audited against the store tile for tile:
  [`reading-the-published-store.md`, section 5][audit].

[self]: https://github.com/dClimate/tessera-embeddings/blob/main/docs/registry.md
[store]: https://github.com/dClimate/tessera-embeddings/blob/main/docs/global-store.md
[master]: https://github.com/dClimate/tessera-embeddings/blob/main/context_docs/storage/registry-master.md
[backfill]: https://github.com/dClimate/tessera-embeddings/blob/main/context_docs/storage/registry-backfill.md
[audit]: https://github.com/dClimate/tessera-embeddings/blob/main/context_docs/storage/reading-the-published-store.md#5-the-registry-beside-the-store
