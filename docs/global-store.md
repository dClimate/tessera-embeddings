# The global TESSERA store

A published dataset of satellite embeddings covering the world's land, one value per 10-metre
pixel per year. Each section links onward to the reference that goes deeper.

## What it is

[TESSERA](https://github.com/ucam-eo/tessera) is a geospatial foundation model. It reads a
year of satellite imagery over a pixel and returns 128 numbers summarising what that pixel
looked like across the year. Those numbers work as input features for classification,
regression or change detection, so you can train against them without handling raw imagery.

| | |
|---|---|
| resolution | 10 m per pixel |
| embedding | 128 dimensions |
| time steps | 9 — one per calendar year, 2017 to 2025 |
| each step covers | 1 January to 31 December |
| extent | land between 59.45°S and 83.65°N, poles and Antarctica excluded |
| land mask | GeoTESSERA's, delivered as 0.1° tiles (see below) |
| model | TESSERA v1.1, the AWS-optimised checkpoint (`tessera_v1_1_aws_encoder.pt`) |

The store records all of this in its root attributes, so you can check it against the data:
`geoemb:model` (`https://geotessera.org/model/1.1`), `checkpoint_id`, `geoemb:gsd`,
`geoemb:dimensions`, `geoemb:source_data` listing the Sentinel-2 and OPERA archives it read,
and `optical_min_obs` (`15`), the quality rule described below. Each zone group adds its own
`crs`, `proj:*` and `spatial:*` metadata, plus `years_complete`.

Sentinel-2 L2A imagery only became generally available partway through 2017, so that year rests
on fewer observations than the rest. It is usable, but check the observation counts before
leaning on it, particularly outside of Europe.

### What counts as land

The mask arrives from GeoTESSERA as about 1.59 million GeoTIFFs, one per 0.1° cell, at
`s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/` alongside a `registry.txt` listing
them. Every listed tile is entirely ones, so the listing itself is the mask and there is no
per-pixel land signal to read.

Coverage extends about one cell, roughly 11 km, into the sea, since some users wanted a
generous ocean margin. A 2048-pixel tile is treated as live when any cell's footprint touches
it, and at 10 m that tile is 20.5 km across against an 11 km margin, so expect embedded ocean
near coastlines.

## How the data is organised

The store is an [Icechunk](https://icechunk.io) repository. Icechunk, from
[Earthmover](https://earthmover.io), is an open-source storage engine for
[Zarr](https://zarr.dev) arrays that adds database-style ACID transactions: a write either
lands completely or not at all, and any past state can be read back by snapshot. At this size
that matters, since a half-finished write to a plain Zarr hierarchy on object storage leaves
no way to tell which parts are good.

Inside the repository the data splits into 120 Zarr *groups*, one per
[UTM zone](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system)
north and south: `01N` through `60N`, `01S` through `60S`. Each group is projected in its own
zone's coordinate reference system, so zone 22 north is EPSG:32622 and so on. Two things
follow. A query spanning several zones has to reproject, since the groups share no grid. And
there are slight discontinuities along zone boundaries, because each side was projected
independently.

Each group carries the attributes and metadata required by the
[Zarr Spatial](https://github.com/zarr-conventions/spatial) and
[GeoEmbeddings](https://github.com/geo-embeddings/embeddings-zarr-convention) conventions, so
a reader can discover the projection and grid from the data itself.

### Chunks and shards

The atomic unit is a chunk of 256 × 256 pixels × 128 dimensions × 1 year. Chunks are grouped
8 × 8 into a shard of 2048 × 2048 pixels, which is one object in the store.

| | pixels | size |
|---|---|---|
| inner chunk | 256 × 256 × 128 × 1 year | 8.4 MB |
| shard | 2048 × 2048 × 128 × 1 year | 537 MB |

Both figures describe the `embeddings` array. It is stored as int8 with a separate float32
scale per pixel, and that quantisation is what holds a shard to 537 MB; dequantised in memory
the same data is four times the size. The remaining arrays are much smaller — a `scales`
chunk is 0.26 MB, an observation-count chunk 0.13 MB.

Writing whole 2048-pixel shards keeps the object count manageable across the globe, while the
256-pixel chunks inside them keep small reads small: a reader fetches the shard's index, then
only the byte ranges of the chunks it overlaps.

### What a read costs

Measured on a compute instance in the bucket's own region, and on one across the continent.

| | same region | cross-region |
|---|---|---|
| opening the store | 169 ms | 902 ms |
| one pixel, all 128 dimensions | 325 ms | 486 ms |
| a 1000 × 1000 tile | 0.4 s | 1.4 s |
| a 4096 × 4096 block (2.1 GB) | 2.8 s | 9.8 s |

A single pixel costs about 8.65 MB on the wire. The smallest fetchable unit is one full-depth
chunk, and a usable read needs two of them: the embeddings chunk and the `scales` chunk that
dequantises it. Scattered point lookups are the expensive access pattern; a window costs far
less per pixel. Running in `us-west-2` alongside the bucket is worth roughly
three times on bulk reads.

## Reading it

You need the Icechunk library; xarray and Zarr alone cannot resolve an Icechunk snapshot. No
AWS account is needed, because the bucket allows anonymous reads.

### With Icechunk

To open group `33N`

```python
import icechunk, xarray as xr

storage = icechunk.s3_storage(
    bucket="tessera-embeddings", prefix="v1.1/dclimate.icechunk",
    region="us-west-2", anonymous=True,
)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store, group="33N", consolidated=False,
                  decode_coords="all", chunks=None)
```

Pass no configuration object. The store carries settings already tuned for readers, and
supplying your own replaces them wholesale.

Note that `chunks=None` skips building a Dask task graph over all 8.67 million chunks in the
array, which is usually much faster. Slice down to your area of interest with `.sel` or
`.isel` before reading any values: without Dask the unsliced array has no lazy wrapper, so
touching it directly attempts the full 66 TiB.

### Without Icechunk

A plain Zarr v3 hierarchy is hosted on Source Coop at
<insert_source_coop_url_here> and opens with xarray or Zarr alone. It is an identical
copy of the dataset published above.


```python
import xarray as xr

ds = xr.open_zarr(
    "s3://tessera-embeddings/v1.1/cambridge.zarr", group="33N",
    storage_options={"anon": True}, consolidated=False, chunks=None,
)
```

Omit `group=` and the call still succeeds, but it hands back an empty dataset.

## Input data

Each yearly embedding is computed from 10 bands of Sentinel-2 L2A optical imagery and 2 bands of
Sentinel-1 RTC OPERA radar, using every observation available that year. Coverage is uneven for
three reasons, and all three show up in the per-pixel counts described below.

Sentinel-2 L2A arrives partway through 2017, beginning with Europe. Radar coverage becomes
spottier over much of the world from 2022 to 2024. Sentinel-1B failed in December 2021; its
replacement Sentinel-1C launched on 5 December 2024 but [opened to users only on 26 March
2025][s1c], with commissioning taking priority until that May. About a fifth of the land has no
radar for those three years. The store shows the gap and the recovery: sampling a live tile in
zone 33N gives roughly 24 to 30 ascending radar observations a year through 2021, exactly zero
for 2022, 2023 and 2024, then 17.5 in 2025 — around 60% of the earlier rate, for the nine months
the satellite was publishing.

[s1c]: https://dataspace.copernicus.eu/news/2025-3-25-sentinel-1c-user-data-opening-26th-march

Equatorial and remote areas get less imagery in the first place: overpasses are less
frequent at low latitudes and over small islands, and equatorial cloud cover is persistently
high. Expect weaker embeddings there.

## Quality rules

One rule decides whether a pixel is embedded: it needs at least 15 valid, sufficiently
cloud-free optical observations across the year. Below that the pixel is left empty and reads
back as the array's fill value.

There is no radar threshold. A pixel with zero Sentinel-1 observations, ascending or
descending, is still embedded, and is fed a neutral radar input in place of the missing data.
Global radar availability is too unpredictable to require: insisting on it would have left
large parts of the world with no embeddings for several years. Radar-free pixels stay
identifiable afterwards.

## Telling good coverage from bad

Every zone group ships the evidence alongside the data, so a pixel can be judged on its own
observations.

Three arrays count usable observations per pixel per year: `s2_obs_count` for optical,
`s1_asc_obs_count` and `s1_desc_obs_count` for radar on ascending and descending passes.
Three more record which months of the year held any observation: `s2_month_covered`,
`s1_asc_month_covered` and `s1_desc_month_covered`. These separate a pixel observed steadily
through the year from one observed in a single burst.

A radar-free pixel is exactly one with a finite `scales` value and
`s1_asc_obs_count + s1_desc_obs_count == 0`.

### The registry

Coverage questions across a wide area are better answered without opening the store. A
Parquet dataset sits beside it:

```
s3://tessera-embeddings/v1.1/dclimate.registry/parts/zone=<ZONE>/year=<YEAR>/<run_id>.parquet
```

994 parts, 142 MB, 3,247,410 rows — one per 2048-pixel tile per year. Each row carries a
WGS84 bounding box, whether the tile was embedded, how many of its pixels the depth rule
refused and why, and how deep the imagery was where it fell short.

The rows are partitioned by zone and year and carry bounding boxes, which makes "is my area
covered, and how well" a 1.7-second query in-region against 14 seconds to read the whole
dataset. Use it to screen areas before committing to a read. It is also what tells us where a
backfill would pay off if more imagery appears later.

Read `refused_px` alongside `embedded`, since a tile can be marked embedded and still be
largely holes.

## Going deeper

- The model and the method: [`ucam-eo/tessera`](https://github.com/ucam-eo/tessera), the
  University of Cambridge repository and papers.
- One area versus the whole world, and when to use which:
  [`single-vs-global.md`](single-vs-global.md).
- Reading this store in detail, including the method behind every measurement above and the
  audit confirming all 120 zones match the declared design:
  [`reading-the-published-store.md`](../context_docs/storage/reading-the-published-store.md).
- Ingesting imagery: [`ingest/README.md`](../src/tessera_embeddings/ingest/README.md).
- Running inference: [`inference/README.md`](../src/tessera_embeddings/inference/README.md).
