<p align="center">
  <img src="../../../docs/images/sacramento-delta-tessera-2025-plain.webp" width="100%"
       alt="TESSERA embeddings of the Sacramento-San Joaquin Delta in false colour: the reclaimed islands in green and terracotta, the sloughs and channels threading between them in dark blue.">
</p>

<p align="center">
  <em>The Sacramento&ndash;San Joaquin Delta, 28&nbsp;km of it, as 2025 TESSERA embeddings.
  Every island reads as its own colour because each one was farmed, flooded or left alone
  differently across the year.</em>
</p>

# Satellite Ingestion

Modules for querying, authenticating, and loading satellite data from STAC catalogs and CMR
into Icechunk/Zarr stores. Used by the Tessera ingestion flows (`ingest_s1_roi_sar`,
`ingest_s2_roi_reflectance`).

---

## Contents

- [Overview](#overview)
- [Module map](#module-map)
- [ROI Workflow](#roi-workflow) · zone masks: [ingest-zone-rois.md](../../../docs/ingest-zone-rois.md)
- [Basic Ingestion Process](#basic-ingestion-process)
- [Detailed Ingestion Process](#detailed-ingestion-process)
  - [STAC query strategy](#stac-query-strategy) and [the fusion order](#the-fusion-order-decides-which-source-wins-a-pixel)
  - [Radar: OPERA RTC-S1 through CMR](#radar-opera-rtc-s1-through-cmr) · [ingest-radar.md](../../../docs/ingest-radar.md)
  - [Which day an image belongs to](#which-day-an-image-belongs-to-solar_dayspy) · [ingest-solar-days.md](../../../docs/ingest-solar-days.md)
  - [Sizing requests to Earth Search's response cap](#sizing-requests-to-earth-searchs-response-cap) · [ingest-earth-search-response-cap.md](../../../docs/ingest-earth-search-response-cap.md)
  - [Date deduplication before loading](#date-deduplication-before-loading)
  - [Choosing between duplicate copies of a tile-date](#choosing-between-duplicate-copies-of-a-tile-date) · [ingest-duplicate-copies.md](../../../docs/ingest-duplicate-copies.md)
  - [Data transformations](#data-transformations)
  - [Sentinel-2 baseline correction](#sentinel-2-baseline-correction-applied-during-the-read) · [ingest-baseline-correction.md](../../../docs/ingest-baseline-correction.md)
  - [Recording the window an ingest examined](#recording-the-window-an-ingest-examined)
- [Authentication (EDL / OPERA data)](#authentication-edl--opera-data) · [ingest-authentication.md](../../../docs/ingest-authentication.md)
- [Error handling](#error-handling) · [ingest-error-handling.md](../../../docs/ingest-error-handling.md)
- [Performance](#performance) · [ingest-performance.md](../../../docs/ingest-performance.md)
- [Accessing the Dask Dashboard](#accessing-the-dask-dashboard)

Read the overview below; the rest of this file is a map. Every subject with machinery of its own
is summarised here in a paragraph and explained in a standalone document under
[`docs/`](../../../docs/README.md). The rationale behind these choices — what was measured, and
what was tried and abandoned — is in [`context_docs/ingest/`](../../../context_docs/ingest/).

---

## Overview

Ingestion turns a region and a date range into a mosaic: a Zarr store holding one
cloud-screened, analysis-ready pixel stack per date, which inference then reads.

```text
ROI ──────▶ query ──────▶ filtering ──────▶ mosaic writing
 │            │              │                  │
 a boolean    which          which of those     the surviving pixels,
 mask on a    catalogue      items to keep,     written per date into
 UTM grid     items touch    and which pixels   the live windows the
 saying       it in this     inside them are    ROI marks as land
 where to     window         usable
 look
```

**ROI.** Everything starts from a region of interest: a boolean mask on a UTM grid marking
which pixels matter. A single-area run rasterises one from a GeoJSON polygon; the global
campaign reads a pre-built per-zone mask instead. The mask is not only a filter — it
determines the *windows* that get loaded and written, so a sparse ROI means proportionally
less work rather than the same work discarded at the end.

**Query.** For each date range, ask a catalogue what imagery touches the ROI's bounding box.
Sentinel-2 comes from a STAC API; OPERA radar comes from a native granule search, because
STAC cannot filter by orbit direction server-side. Queries are streamed month by month and
per orbit, because a single whole-year request over a large zone is large enough that
catalogues refuse it.

**Filtering.** Whole items are set aside first — when their date is already in the store, when a
reprocessed granule duplicates one already chosen, or when an optional caller hook rejects them.
A duplicate is kept rather than discarded: it becomes the next rung the read falls back to.
What survives is then sorted — optical clearest scene first, radar most-acquired burst first — and
that sort decides which source supplies a pixel where two of one solar day overlap.

**Mosaic writing.** What survives is loaded lazily through `odc.stac.load` into a
Dask-backed array, corrected at load time where a producer's reflectance offset requires it,
converted where radar amplitude needs decibels, and written one date at a time into the
ROI's live windows. Writing per date rather than per year keeps the Dask graph small enough
for a scheduler to hold. Each date is stored as `INGEST_CHUNKS` blocks, one object each, which is
also the unit ingest writes. Each block is a shard of smaller inner chunks
(`INGEST_INNER_CHUNKS`: 512 rows of one 2048-px inference tile), so a reader decompresses only the
rows and tile it asks for rather than the whole block, and every inner chunk is compressed with
Blosc-LZ4 (`INGEST_COMPRESSORS`), which decompresses 2.6× faster than zarr's default zstd while
storing 6% less.

**Where the rest of the machinery lives.** Those four stages are the whole pipeline. Everything
else exists because catalogues rate-limit and refuse, granules get reprocessed and duplicated,
objects fail to read while reporting success, credentials expire mid-run, and a task graph over a
whole zone-year exhausts a scheduler's memory before reading a byte. This README follows the
happy path and links each of those subjects to its own document; the failure modes are
[ingest-error-handling.md](../../../docs/ingest-error-handling.md) and the scheduler's limits
[ingest-performance.md](../../../docs/ingest-performance.md).

---
## Module map

| Module | What it does |
|---|---|
| `s2_roi.py` | The Sentinel-2 ingest: queries month by month, keeps one copy of each image, then loads and writes one solar day at a time. |
| `s1_roi.py` | The OPERA radar ingest: walks the window in `batch_days` batches, builds the burst census once per orbit leg, and renews ASF credentials on a timer. |
| `stac.py` | STAC querying and loading through `odc.stac.load` — providers, the date-window worklist that keeps Earth Search queries under its response cap, date filtering, and the load-time machinery that applies the Sentinel-2 reflectance offset. [§ Earth Search's response cap](../../../docs/ingest-earth-search-response-cap.md), [§ Sentinel-2 baseline correction](../../../docs/ingest-baseline-correction.md) |
| `opera_query.py` | OPERA RTC-S1 queries: bounding boxes, items built from CMR's native Granule Search with orbit direction filtered server-side, UTM EPSG derivation, asset preparation, and the burst census and sort that decide which burst wins an overlap. [§ Radar ingest](../../../docs/ingest-radar.md) |
| `solar_days.py` | The one place the solar offset is applied: stamps each item with noon UTC of its solar day, and splits a window into chunks that own solar days and query UTC dates. [§ Solar days](../../../docs/ingest-solar-days.md) |
| `boa_offset.py` | The one place the reflectance-offset question is answered. Asked per ASSET, so an item whose bands come from two producers is corrected band by band rather than refused. [§ Sentinel-2 baseline correction](../../../docs/ingest-baseline-correction.md) |
| `item_baselines.py` | The one reader of `s2:processing_baseline`. Reports integer hundredths (`04.00` -> `400`) and `None` for every kind of unreadable. |
| `asset_locations.py` | Where an item's assets live, keeping two questions apart: whether the read is cheap (the bucket's REGION) and whether the offset is already removed (the PRODUCER). `AssetSources` reports the keys it could not resolve instead of dropping them. |
| `duplicates.py` | Keeps one copy of each image of a tile-date — the catalogue lists an image again whenever it is reprocessed — and every image. Rejected copies are kept as a fallback. [§ Duplicate copies](../../../docs/ingest-duplicate-copies.md) |
| `loader_failures.py` | Keeps what a failed load knows — which object, and why — neither of which reaches the caller on its own. One `install_capture_everywhere` call covers every current and future worker. [§ When a source object will not read](../../../docs/ingest-error-handling.md#when-a-source-object-will-not-read) |
| `auth.py` | Earthdata Login for ASF-hosted OPERA data: S3 direct access on roughly hourly credentials, plus legacy signed URLs. Renewal is timer-driven, because the credentials expire on their own clock. [§ Authentication](../../../docs/ingest-authentication.md) |
| `transforms.py` | Post-load lazy Dask transforms. Currently `amplitude_to_db`. [§ Amplitude to decibels](../../../docs/ingest-radar.md#amplitude-to-decibels) |
| `roi.py` | ROI utilities: read an existing Zarr ROI store, rasterise a GeoJSON polygon to a boolean mask on a UTM grid, load Sentinel-2 tile footprints. [§ Generating an ROI](#generating-an-roi) |
| `roi_processing.py` | Higher-level ROI helpers used by the `generate_roi` flow. |
| `land_mask.py` | Builds per-zone coverage bitmaps from the partner's land mask, and writes and validates zone ROIs (`export_zone_roi`, `validate_zone_roi`). [§ Zone ingestion](#zone-ingestion-the-global-campaign--adr-011) |
| `source_coverage.py` | Optical preflight: does the catalogue publish anything reaching a zone's live land in this window, answered before any cluster is provisioned. Three-valued — only a positive finding of absence refuses. [§ Zone ingestion (the global campaign) — ADR-011](#zone-ingestion-the-global-campaign--adr-011) |
| `catalogue_refusal.py` | Tells a catalogue that is BUSY apart from one that cannot serve this REQUEST, and names the request either way. [§ When the catalogue refuses](../../../docs/ingest-error-handling.md#when-the-catalogue-refuses-naming-the-request-and-telling-the-two-refusals-apart) |
| `live_windows.py` | Derives the chunk-aligned live windows every ingest loads and writes, and narrows them per date to the land that date's imagery reaches. [§ Cropping to live windows](../../../docs/ingest-performance.md#cropping-to-live-windows-unconditional) |
| `_http.py` | Shared HTTP helpers for catalogue and granule queries: retries that log each attempt, calls the caller can abandon, and a guard for replies that claim success but are not JSON. |
| `_pipeline.py` | A prepare/consume pipeline with a look-ahead depth, so the next item is prepared while the current one is consumed. Buys buffering, never concurrency. S2 dates go through it automatically on every fleet up to 140 workers. [§ Pipelining a date's preparation](../../../docs/ingest-performance.md#pipelining-a-dates-preparation-automatic) |

---

## ROI Workflow

The Tessera pipeline uses a spatial ROI mask — a chunked boolean Zarr array stored on S3 — to
define the area of interest for all downstream ingestion and inference.

### Generating an ROI

`generate_roi` flow calls `roi.rasterize_roi_zarr`. Steps:

1. **Load geometry** — from a local or S3 GeoJSON (`input_path`) or a pre-loaded list of
   Shapely geometries. Alternatively, `load_s2_tile_geometry` fetches MGRS tile footprints
   from the S3 tile index.
2. **WGS84 bbox** — computed from the *original* geometry **before** reprojection. Using
   the post-projection axis-aligned bounds would inflate the bbox significantly at oblique
   UTM zone edges.
3. **CRS selection** — `determine_target_crs` picks (in order): user-specified `force_crs`,
   the input CRS if it is already projected, or the best UTM zone derived from the
   geometry centroid. Geographic CRS output is rejected with an error.
4. **Grid computation** — `compute_grid` converts projected bounds + resolution into pixel
   dimensions and an Affine transform.
5. **Chunk-at-a-time rasterization** — `rasterize_roi_zarr` iterates over `chunk_size × chunk_size`
   blocks, calling `rasterio.features.rasterize` per chunk with a chunk-local transform.
   The full boolean grid is never held in memory.
6. **Zarr attrs** — `crs`, `transform` (6-element Affine list), `resolution`, `bbox_wgs84`,
   and a `_manifest` written atomically after all chunks succeed.

### Reading an ROI

Ingestion flows call:

- `read_roi_metadata(roi_path)` — returns `ROIMetadata`: WGS84 bbox (for STAC queries),
  native CRS string, `odc.geo.GeoBox` (for `geobox=` kwarg to `odc.stac.load` so output
  grids align exactly), width/height.
- `read_roi_mask(roi_path, chunks)` — returns a lazy Dask boolean array for masking.

### Applying the ROI Mask

`roi_processing.apply_roi_mask` broadcasts the 2D mask over the time dimension and sets
out-of-ROI pixels to `fill_value` (default 0) across all dataset variables.

`roi_processing.filter_low_coverage_dates` then drops time steps where fewer than
`min_valid_coverage` percent of ROI pixels are valid (default 5%). Only the per-date valid pixel
counts are computed eagerly — one scalar per time step, from SCL for S2 or VV for S1 — so band
arrays remain lazy and cloud-covered or off-ROI scenes are dropped before any band data is read.

`identify_low_coverage_ds` is the lazy alternative: instead of dropping dates it attaches a
`valid_coverage` boolean coordinate. Evaluating it still reads the quality band over the whole
ROI — it defers that read and skips the imagery bands, rather than answering from metadata.

### Zone ingestion (the global campaign) — ADR-011

The global campaign reuses this engine per zone, but does not rasterise a polygon: a bounding-box
fit cannot reproduce the fixed, shard-snapped zone grid its fill validates against, so
`land_mask.export_zone_roi` writes each zone's mask directly from the zone's grid and its land
coverage bitmap. The same S1/S2 flows then ingest it into per-(zone, year) mosaics, with the S2
coverage bar lowered from 5% to about 0.1% because one solar day's swath covers only a sliver of
a 6° zone. The `export-zone-rois` flow builds and validates many zones' masks ahead of the
campaign. Where a zone's ingest sits in the campaign, and what validation proves, are in
**[ingest-zone-rois.md](../../../docs/ingest-zone-rois.md)**.

---

## Basic Ingestion Process

The high-level entry point is `ingest_tile()` in `stac.py`. It runs five stages in sequence:

```text
1. Item query        — find items for the tile/bbox + date range (S1: native CMR
                       granule query, orbit-filtered server-side)
2. Item filtering    — optional item_filter_fn pre-filter hook
3. Date dedup        — drop items whose dates are already in the Zarr store
4. odc.stac.load     — lazy-load COGs into a Dask-backed xarray Dataset. The S2
                       baseline correction happens here, per source, inside the read
5. Corrections       — dB conversion (S1)
```

`query_stac_items` and `load_stac_items` expose these stages separately so a flow could
check for new data before spinning up a Dask cluster (see `has_new_stac_dates`, which is
not yet wired into any flow — [issue #47](https://github.com/dClimate/tessera-embeddings/issues/47)).

### STAC Providers and Collections

Provider configs live in
[`config/providers.py`](../config/providers.py) (`PROVIDERS`, `CollectionConfig`). Two
catalogues are queried in production:

| Catalogue | Collection in production | Whose pixels arrive |
|---|---|---|
| Earth Search (Element 84) | Sentinel-2 L2A | Element 84's harmonised COGs (`sentinel-cogs`, `e84-earth-search-sentinel-data`) **and** ESA's originals (`sentinel-s2-l2a`) |
| CMR-STAC (NASA/ASF) | OPERA RTC-S1 | ASF (`asf-cumulus-prod-opera-products`) |

Earth Search's `sentinel-1-grd`, `sentinel-2-l1c` and `landsat-c2-l2` entries are configured and
**have no caller**: radar comes from OPERA, and `ingest_s1_roi_sar` names the `cmr-asf` provider
directly rather than taking it as a parameter.

**ESA's own products reach us through Earth Search, not through a separate catalogue.** The
`sentinel-2-l2a` collection there indexes both Element 84's reprocessed COGs and items whose
assets point straight at ESA's archive, and a query returns them mixed. The two producers differ
on one thing that matters — Element 84 has already subtracted the post-baseline-04.00 reflectance
offset and ESA has not — so which producer served an asset is decided per asset from where that
asset lives, never from the collection. That decision is the subject of
[ingest-baseline-correction.md](../../../docs/ingest-baseline-correction.md), and choosing between
an ESA copy and an Element 84 copy of the same tile-date is the subject of
[ingest-duplicate-copies.md](../../../docs/ingest-duplicate-copies.md).

Each `CollectionConfig` records: collection ID, band list, native resolution, tile ID property
(for S2/Landsat property-based queries), and correction parameters. For OPERA RTC-S1 on
CMR-STAC there is no tile ID property, so the query falls back to a WGS84 bbox.

`PROVIDERS` also carries a `planetary-computer` entry. **Nothing has ever run against it** — it
is a sketch of a future path rather than a supported one, and the Cambridge TESSERA team's report
that Microsoft throttles heavy outbound traffic from Planetary Computer is why the OPERA RTC route
was built instead. Treat it as unvalidated configuration.

---
## Detailed Ingestion Process

The happy path, stage by stage. Failure modes live in
[ingest-error-handling.md](../../../docs/ingest-error-handling.md) instead.

Six things happen between a region and a written mosaic, and they are the order of this section:

1. **Ask the catalogue** what imagery touches the region in this window.
2. **Settle which day** each image belongs to, since a catalogue speaks UTC and a mosaic is
   indexed by local solar day.
3. **Size the request** so the catalogue will actually answer it.
4. **Filter what came back** — dates already written, and duplicate copies of one image.
5. **Transform and write** — the reflectance offset during the read, the radar conversion after.
6. **Record what was examined**, so a month with no imagery reads as a finding and not a gap.

### STAC query strategy

S2 and Landsat are queried by tile ID property (e.g., `grid:code = T33UUP`), which returns
only items for that specific MGRS tile. OPERA RTC-S1 on CMR-STAC lacks an equivalent
property, so queries use a bbox derived from the MGRS tile via `mgrs_tile_to_bbox()`.

#### The fusion order decides which source wins a pixel

Where two sources of one solar day overlap, the loader keeps the first valid one. `odc.loader`'s
default fuser is `nodata_fuser` — `np.copyto(dst, src, where=dst_is_nodata)` — so it writes only
where the destination is still empty, and this package configures no fuser of its own. The order
items reach the loader in is therefore a quality decision. Each collection names its **fusion
order** (`CollectionConfig.fusion_order`), and the same field sets the loader's
`preserve_original_order`, so the loader keeps exactly the orders this package imposes. A
collection naming none, such as Landsat, keeps odc's own `(time, id)` order.

**Optical: the clearest scene first.** Cloud cover is intentionally **not** used as a filter at the
STAC query stage — pixel-level cloud classification is handled later (SCL for S2, ML model for
inference). For S2, items are sorted by `(date, eo:cloud_cover)` ASCENDING, so the clearest scene
covering a pixel wins it and a hole in it falls through to the next-clearest rather than to
nothing. An item declaring no `eo:cloud_cover` sorts after every measured value, where it can only
fill gaps. `solar_day_sort_key` gives it infinity rather than 100, since 100 is itself a real
reading and the two would otherwise tie and be reordered by `id`, letting an unmeasured scene take
ground from one known to be fully clouded.

**Radar: the most-acquired burst first.** Overlapping bursts of one day go to the burst acquired
most often over the window, so a pixel's series keeps one look angle wherever the archive allows —
see [Which burst wins an overlap](../../../docs/ingest-radar.md#which-burst-wins-an-overlap).

#### Streaming the query month by month (S2)

`ingest_s2_roi_reflectance` queries **one month at a time** by default
(`stream_stac_monthly`), prefetching the next while the current one is ingested. Querying the
whole window up front retains every returned item for the run's duration, and a zone-year's worth
does not fit alongside the ingest on one worker; streaming bounds retention to the month in hand
plus the one being fetched.

The prefetch runs on a daemon thread rather than a pooled worker: an in-flight catalogue walk
cannot be interrupted from outside, so abandoning it is the only way to stop waiting, and a
daemon thread does not hold the process open when a run is cancelled mid-query.

Month ranges **partition** the window — each month owns a half-open slice and items are
filtered to their owner — so a date cannot be ingested twice or skipped at a boundary.
Set `stream_stac_monthly=False` to issue one query for the whole window; the per-date work
is byte-identical either way.

#### Antimeridian queries

UTM zones 01 and 60 straddle ±180°, and the ROI's WGS84 bounding box is written in the
GeoJSON crossing convention (`west > east`) so it stays narrow instead of spanning the
globe. Neither catalog can be relied on to read that form — the native CMR path is known
to reject it — so both query paths split the box at ±180° into two ordinary west-to-east
queries and deduplicate the results by item id. A granule straddling the line is returned
by both halves and must be loaded once.

### Radar: OPERA RTC-S1 through CMR

OPERA RTC-S1 is not queried through a STAC search. `make_s1_item_provider` builds ready-to-load
items straight from NASA's CMR Granule Search, which filters orbit direction and dual polarisation
(VV+VH) server-side, where CMR-STAC pages unreliably at this scale and silently ignores the orbit
filter. Each item's UTM CRS is derived from its MGRS tile, since OPERA items carry no projection
metadata, and overlapping bursts are ranked by a once-per-leg census of how often each was
acquired. A zone whose land is ice has granules but no usable dual-pol radar; it resolves to
`S1_ORBIT_NONE` and is embedded from Sentinel-2 alone, unless the caller passes `require_s1`. All
of it, with the decibel conversion applied after the load, is in
**[ingest-radar.md](../../../docs/ingest-radar.md)**.

### Which day an image belongs to (`solar_days.py`)

A catalogue speaks UTC; a mosaic is indexed by local solar day, and where a zone's offset crosses
UTC midnight the two disagree. One rule keeps them apart: `normalize_to_solar_day` applies the
solar offset exactly once, at the catalogue chokepoint, after which an item's `datetime` is noon
UTC of its solar day and every downstream date is a plain `YYYY-MM-DD` string. An architecture
rule fails CI if `solar_day_offset_seconds` is called anywhere else. A query can only be bounded
in UTC, so each chunk of a window **owns** a range of solar days and **queries** one UTC day wider
on each side, and items are filtered to their owner before the loader groups them:

```text
                 own:            2024-01-31 .............. 2024-02-29
                 query:   2024-01-30 ........................... 2024-03-01
                          └─ pad ─┘                              └─ pad ─┘

  a solar day landing on the cut is drawn from BOTH sides by the batch that owns it,
  and is owned by exactly one batch, so it is written once and written whole
```

The full table of date conventions, what enforces them, how OPERA's per-burst timestamps become
one slice, and which day a slice is labelled with are in
**[ingest-solar-days.md](../../../docs/ingest-solar-days.md)**.

### Sizing requests to Earth Search's response cap

Earth Search refuses any answer larger than about 6 MB, AWS Lambda's synchronous response limit,
with a fast 502 that repeating cannot fix, and a Sentinel-2 query over a zone-year asks for far
more. So the query is a worklist of date windows. A refused window is cut into two shorter ones,
which regroups its scenes into pages that fit; a refusal no shorter window can reach, a first page
or a single day, is retried with a smaller page. Windows meet on a shared instant, so together they
cover exactly the window asked for, and the results are re-sorted afterwards, so the order of the
walk never reaches the mosaic. The mechanism in plain language, then in full, is in
**[ingest-earth-search-response-cap.md](../../../docs/ingest-earth-search-response-cap.md)**.

### Date deduplication before loading

`_filter_existing_dates` removes items whose dates are already in the Zarr store before calling
`odc.stac.load`, so no task graph is built and no COG read is issued for data that would be
discarded. It is keyed by solar day, the way the store was written — see
[Dates already in the store](../../../docs/ingest-solar-days.md#dates-already-in-the-store).

The filter is an optimisation, not the guarantee. On S1 the queries are built one batch ahead of
the writes, so the set they filter against is frozen before the run began; the write loop tracks
what it actually wrote and is the authority.

### Choosing between duplicate copies of a tile-date

A Sentinel-2 tile-date often has several catalogue items. Some are copies of one image: a
reprocessing at a newer baseline, or one granule served by two producers. Others are different
images: successive orbits on one day, or one pass split across datastrips. `duplicates.py` keeps
every image and one copy of each, before the loader sees them, telling them apart by pass
(`s2:datatake_id`) and datastrip (`s2:datastrip_id`) and merging copies only on proof. Among
copies of one image a single context-free sort key decides: first whether a copy can be processed
at all, then the newest baseline, then cost. The copies it rejects are kept, in that order, as the
fallback ladder a read failure steps down. The seven terms of the key and what each protects are
in **[ingest-duplicate-copies.md](../../../docs/ingest-duplicate-copies.md)**.

### Data transformations

What happens to the data between a catalogue item and a written pixel, in the order it happens:
items are rewritten before the load, the load itself is configured to produce the grid and the
groups we want, then the radar amplitude conversion runs after it. The one correction that does
not fit that sequence is the Sentinel-2 reflectance offset, which happens *inside* the read and
has its own section after this one.

#### Pre-load (STAC items)

These happen before `odc.stac.load` is called:

| Transform | Where | What |
|---|---|---|
| **Date dedup** | `stac._filter_existing_dates` | Drops STAC items whose date is already written to the store. Keyed on the SOLAR day when the caller passes `mid_longitude`, matching how the store was written. |
| **Item sort** | `stac.query_stac_items` | Sorts by the collection's fusion order, so the loader's first-valid-source fuser keeps the right source. S2: `(date, cloud_cover)` ascending, clearest tile FIRST. Radar: most-acquired burst FIRST. |
| **Item provider** | `opera_query.make_s1_item_provider` | Builds orbit-filtered OPERA items directly from the native CMR granule API (bypasses CMR-STAC search). |
| **URL rewriting** | `auth.rewrite_assets_to_s3` | Rewrites HTTPS datapool/earthdatacloud URLs to `s3://` URIs. |
| **Timestamp normalisation** | `solar_days.normalize_to_solar_day` | Stamps every item with noon UTC of its **solar day**. The single place the solar offset is applied; also what makes `odc.stac.load` mosaic OPERA's per-burst granules into one time slice. |

#### Load-time (`odc.stac.load`)

`_load_from_stac` configures `odc.stac.load` with:

- **Resampling** — bilinear for primary spectral bands. Extra bands (e.g., S2 SCL) always
  use nearest-neighbour regardless of the primary resampling, enforced via a per-band dict.
- **Resolution override** — S1 loads at 10 m to share a grid with S2 although the native OPERA
  product is 30 m; resampling uses COG overviews during the read, not as a post-process.
- **CRS override** — OPERA RTC-S1 items on CMR-STAC lack `proj:` extension metadata; an
  explicit `crs=` (e.g. `EPSG:32633`) must be passed so `odc.stac.load` knows the output
  projection.
- **GeoBox alignment** — a `GeoBox` from `read_roi_metadata` makes the output grid match the ROI
  exactly in CRS, transform and shape, overriding bbox, CRS and resolution.
- **groupby** — `"solar_day"` merges items from adjacent MGRS tiles acquired on the same local
  calendar day into one mosaic, which is required for ROI queries crossing tile boundaries. Which
  source wins a pixel is decided by the sort order; see
  [the fusion order](#the-fusion-order-decides-which-source-wins-a-pixel).
- **Dimension rename** — `normalize_odc_dims` maps `odc.stac.load`'s `y`/`x` output
  dimensions to the project-wide `northing`/`easting` convention and drops `spatial_ref`.
- **The Sentinel-2 reflectance offset** is applied inside this read, per source — see
  [Sentinel-2 baseline correction](#sentinel-2-baseline-correction-applied-during-the-read)
  below.

#### Post-load

One transform runs after `odc.stac.load` returns: `transforms.amplitude_to_db` converts OPERA's
linear float32 amplitude to scaled decibels stored as uint16. It is lazy, so nothing is
materialised until the Zarr write. The formula and its constants are in
[Amplitude to decibels](../../../docs/ingest-radar.md#amplitude-to-decibels).

### Sentinel-2 baseline correction (applied during the read)

ESA's processing baseline 04.00 (January 2022) added +1000 to every Sentinel-2 L2A reflectance
value. Element 84's harmonised COGs have it removed and ESA's originals do not, and Earth Search
serves both under one collection, so whether to subtract it is decided per **asset**, from where
that asset lives (`boa_offset.source_decision`), and applied inside the read, per source, before a
solar day's tiles are fused together. The amount is a constant, `-1000`, floored at DN 1 so a dark
pixel never becomes nodata; only whether it applies is decided. A source at or above the threshold
whose producer cannot be determined refuses rather than guess, because both wrong answers are
silent. The rest, including the limits recorded rather than fixed, is in
**[ingest-baseline-correction.md](../../../docs/ingest-baseline-correction.md)**.

### Recording the window an ingest examined

Both paths write `assessed_window` — the date range processed in full — onto the store. A month
absent from the time axis but wholly inside that range was **examined and found to hold nothing
reachable**, which is a finding; a month outside it is a gap. Without the record those are
indistinguishable, and the coverage gate has to fail on both.

The attribute belongs on the repo the gate opens — `reflectance.zarr` or `sar_<orbit>.zarr` —
not on the mosaic directory that contains them.

**It is written whenever the store exists, not only when the run wrote a date**, because the
case needing it most writes nothing. A run interrupted between its last date commit and this
record leaves every date present and the attribute absent; the retry dedupes those dates away,
takes the zero-write path and, keyed on what *this* invocation wrote, skips the record again.
Every later retry does the same, so a legitimately empty month stays indistinguishable from a gap
and the zone-year can never complete. Keyed on the store, the resume repairs the attribute. The
extra existence probe runs only when nothing was written, so a normal run pays nothing for it. A
genuinely absent store is still left alone: there is no repo to annotate, and that case was never
ambiguous — no store means the orbit is absent and callers downgrade.

Every uncertain path is strict: an absent, malformed or unparseable attribute excuses nothing,
and a partially-covered month stays an error because it could hide unexamined days. Failing to
write the attribute is logged, never raised — the gate simply falls back to requiring every
month. `assessed_empty_dates` is recorded alongside for observability, separating "sparse region"
from "the footprints are wrong".

---
## Authentication (EDL / OPERA data)

OPERA radar hosted by ASF needs NASA Earthdata Login (EDL) credentials; Sentinel-2 is publicly
readable. Set:

```bash
export EARTHDATA_USERNAME=your-username
export EARTHDATA_PASSWORD=your-password
```

and approve the **ASF Cumulus** application at
[urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) → Authorized Apps.

The pipeline exchanges those for roughly hourly AWS STS credentials scoped to ASF's OPERA bucket
and reads it directly over S3, renewing them on a timer and broadcasting them to every Dask worker.
Because those credentials occupy the `AWS_*` environment variables, every read and write on the
project's own bucket resolves the deployment's IAM role instead. Renewal, the per-thread session
patch, the IAM identity and why it is cached, URL rewriting and the signed-URL fallback are in
**[ingest-authentication.md](../../../docs/ingest-authentication.md)**.

---

## Error handling

Everything that goes wrong between a catalogue request and a written pixel, and what the pipeline
does about it, is in **[ingest-error-handling.md](../../../docs/ingest-error-handling.md)** — twelve diagnosed causes, the
evidence each one leaves, and the single classifier that tells them apart. The short version: a
date given up is given up for good, so the pipeline spends time rather than dates.

## Performance

What bounds an ingest and the levers against it are in
**[ingest-performance.md](../../../docs/ingest-performance.md)**. The short version: the Dask scheduler is one
process holding the whole task graph, so every lever either keeps the graph small enough to fit
or keeps the fleet busy once it does, and the two pull against each other.

## Accessing the Dask Dashboard

Ingestion flows run a Dask cluster on ECS Fargate. The scheduler is in a private subnet.
Use SSM port forwarding to reach the Bokeh dashboard:

```bash
# Look up TASK_ID and RUNTIME_ID for the Dask scheduler task in the ECS console
aws ssm start-session \
  --target ecs:yield-cluster_${TASK_ID}_${RUNTIME_ID} \
  --document-name AWS-StartPortForwardingSession \
  --region <aws-region> \
  --parameters '{"portNumber":["8787"],"localPortNumber":["8787"]}'
```

Then open http://localhost:8787 in your browser. `log_dashboard_ssm_command` (in
`providers/aws/dask.py`) logs the same command with the target and region already filled in
once the cluster is up.

`AWS-StartPortForwardingSession` forwards a port on the scheduler container itself. The
`AWS-StartPortForwardingSessionToRemoteHost` variant is for hosts reachable *from* the
container (RDS, an internal ALB); current SSM agents refuse loopback destinations for it and
fail with `Forwarding to IP address localhost is forbidden`.

Requires the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (`brew install session-manager-plugin`).
