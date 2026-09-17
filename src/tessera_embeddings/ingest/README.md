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
- [ROI Workflow](#roi-workflow)
- [Basic Ingestion Process](#basic-ingestion-process)
- [Detailed Ingestion Process](#detailed-ingestion-process)
- [Authentication (EDL / OPERA data)](#authentication-edl--opera-data)
- [Error handling](../../../docs/ingest-error-handling.md)
- [Performance](../../../docs/ingest-performance.md)
- [Accessing the Dask Dashboard](#accessing-the-dask-dashboard)
- [Appendix A — the Earth Search response cap in detail](#appendix-a--the-earth-search-response-cap-in-detail)

Read the overview below; the rest of this file is meant to be searched rather than read
through. The rationale behind these choices — what was measured, and what was tried and
abandoned — is in [`context_docs/ingest/`](../../../context_docs/ingest/), not here.

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
What survives is then sorted clearest-first, and that sort decides which pixel wins where two
scenes of a solar day overlap.

**Mosaic writing.** What survives is loaded lazily through `odc.stac.load` into a
Dask-backed array, corrected at load time where a producer's reflectance offset requires it,
converted where radar amplitude needs decibels, and written one date at a time into the
ROI's live windows. Writing per date rather than per year keeps the Dask graph small enough
for a scheduler to hold.

**Where the length of this file comes from.** Those four stages are the whole pipeline.
Everything else here exists because catalogues rate-limit and refuse, granules get reprocessed
and duplicated, objects fail to read while reporting success, credentials expire mid-run, and a
task graph over a whole zone-year exhausts a scheduler's memory before reading a byte. The
sections up to [Authentication](#authentication-edl--opera-data) describe the happy path.
[ingest-error-handling.md](../../../docs/ingest-error-handling.md) and
[ingest-performance.md](../../../docs/ingest-performance.md) are the rest.

---
## Module map

| Module | What it does |
|---|---|
| `stac.py` | STAC loading through `odc.stac.load` — providers, date filtering, and the load-time machinery that applies the Sentinel-2 reflectance offset. [§ Sentinel-2 baseline correction](#sentinel-2-baseline-correction-applied-during-the-read) |
| `opera_query.py` | OPERA RTC-S1 queries: bounding boxes, items built from CMR's native Granule Search with orbit direction filtered server-side, UTM EPSG derivation, asset preparation. [§ OPERA-Specific Query Quirks](#opera-specific-query-quirks) |
| `boa_offset.py` | The one place the reflectance-offset question is answered. Asked per ASSET, so an item whose bands come from two producers is corrected band by band rather than refused. [§ Sentinel-2 baseline correction](#sentinel-2-baseline-correction-applied-during-the-read) |
| `item_baselines.py` | The one reader of `s2:processing_baseline`. Reports integer hundredths (`04.00` -> `400`) and `None` for every kind of unreadable. |
| `asset_locations.py` | Where an item's assets live, keeping two questions apart: whether the read is cheap (the bucket's REGION) and whether the offset is already removed (the PRODUCER). `AssetSources` reports the keys it could not resolve instead of dropping them. |
| `duplicates.py` | Chooses between duplicate catalogue items for one tile-date, which Element 84 publishes whenever a granule is reprocessed. Rejected copies are kept as a fallback. [§ Choosing between duplicate copies of a tile-date](#choosing-between-duplicate-copies-of-a-tile-date) |
| `loader_failures.py` | Keeps what a failed load knows — which object, and why — neither of which reaches the caller on its own. One `install_capture_everywhere` call covers every current and future worker. [§ When a source object will not read](../../../docs/ingest-error-handling.md#when-a-source-object-will-not-read) |
| `auth.py` | Earthdata Login for ASF-hosted OPERA data: S3 direct access on roughly hourly credentials, plus legacy signed URLs. Renewal is timer-driven, because the credentials expire on their own clock. [§ Authentication (EDL / OPERA data)](#authentication-edl--opera-data) |
| `transforms.py` | Post-load lazy Dask transforms. Currently `amplitude_to_db`. [§ OPERA RTC-S1 Amplitude-to-dB Conversion](#opera-rtc-s1-amplitude-to-db-conversion) |
| `roi.py` | ROI utilities: read an existing Zarr ROI store, rasterise a GeoJSON polygon to a boolean mask on a UTM grid, load Sentinel-2 tile footprints. [§ Generating an ROI](#generating-an-roi) |
| `roi_processing.py` | Higher-level ROI helpers used by the `generate_roi` flow. |
| `source_coverage.py` | Optical preflight: does the catalogue publish anything reaching a zone's live land in this window, answered before any cluster is provisioned. Three-valued — only a positive finding of absence refuses. [§ Zone ingestion (the global campaign) — ADR-011](#zone-ingestion-the-global-campaign--adr-011) |
| `catalogue_refusal.py` | Tells a catalogue that is BUSY apart from one that cannot serve this REQUEST, and names the request either way. [§ When the catalogue refuses](../../../docs/ingest-error-handling.md#when-the-catalogue-refuses-naming-the-request-and-telling-the-two-refusals-apart) |
| `live_windows.py` | Derives the chunk-aligned live windows every ingest loads and writes, and narrows them per date to the land that date's imagery reaches. [§ Cropping to live windows](../../../docs/ingest-performance.md#cropping-to-live-windows-unconditional) |
| `_http.py` | Shared HTTP helpers for catalogue and granule queries: retries that log each attempt, calls the caller can abandon, and a guard for replies that claim success but are not JSON. |
| `_pipeline.py` | A prepare/consume pipeline with a look-ahead depth, so the next item is prepared while the current one is consumed. Buys buffering, never concurrency. [§ Pipelining a date's preparation](../../../docs/ingest-performance.md#pipelining-a-dates-preparation-pipeline_dates) |

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
   UTM zone edges (see `docs/bbox-projection-inflation.md`).
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

The global campaign reuses this exact ROI engine to produce its per-zone mosaics: it
**synthesizes a zone-shaped ROI** instead of rasterizing a GeoJSON, then dispatches the same
S1/S2 ingest flows. `generate_roi`'s `compute_grid` bbox-fits geometry and cannot reproduce the
fixed, shard-snapped `zone_grid.ZoneSpec` extent the fill validates against — so
`land_mask.export_zone_roi` writes the ROI mask directly from `ZoneSpec` (mask = the zone's
`tile_live_2048` coverage bitmap upsampled ×2048; WGS84 bbox tight to the live tiles).

```text
run_global_campaign  (per pending (zone, year), zone-parallel within a year)
   │
   ├─ ingest-zone-year ──► export_zone_roi(zone)         {inputs}/rois/zarrs/zone_33N.zarr
   │      │                  (ZoneSpec grid + tile_live mask; ocean-tile skip)
   │      ├─ marker probe (ingest_marker fingerprint; stale/partial ⇒ clear+rebuild)
   │      ├─ (live-chunk count ⇒ max_workers)
   │      ├─ ingest_s1_roi_sar × orbit ┐  concurrent, onto
   │      ├─ ingest_s2_roi_reflectance ┘  {inputs}/mosaics/33N/2025/
   │      ├─ check_time_window_coverage (strict span; allow_partial_window escape)
   │      └─ write ingest_marker  (last — crash before this ⇒ clean rebuild on re-run)
   │
   ├─ fill-zone-year  ──► coverage + SAR-grid + model gates (pre-Ray) ──► inference ──► assemble ──► tag
   │
   └─ delete mosaics/33N/2025  (s5cmd --all-versions; transient input)
```

The S2 `min_valid_coverage` bar is lowered far below the ROI default (5 % → ~0.1 %): a single
solar-day's swath covers only a sliver of a whole 6° zone, so a high bar would drop nearly
every date. Mosaics are per `(zone, year)` and deleted after the fill is tagged — they are
re-derivable inputs at ~TB scale (ADR-011). Zones are named by UTM common name (`33N`/`07S`),
not EPSG (see `storage/zone_grid.canonicalize_zone`).

#### Pre-generating the zone masks — `export-zone-rois`

`ingest-zone-year` exports the mask it needs on the fly, so the campaign is self-sufficient.
The `export-zone-rois` flow does the same work for many zones **ahead of the campaign**, and
adds the check the per-cell path has no reason to run:

```text
export-zone-rois  (one task per zone, max_parallel_zones in flight, no barrier)
   └─ per zone ─► live_chunk_count(zone)        coverage bitmap, one ~KB GET
                  ├─ 0 live chunks ⇒ all_ocean (no mask by design; nothing written)
                  ├─ export_zone_roi(zone)      skipped when already current
                  └─ validate_zone_roi(zone)    grid · completion · placement · layout
```

`validate_zone_roi` is the reason to run this early. Its load-bearing check is **placement**:
the count of stored chunk objects must equal `live_chunk_count`. That holds because the writer
skips non-live blocks and Zarr elides all-fill chunks, so the set of stored chunks *is* the set
of live cells — one listing asserts for the whole zone that the mask is live exactly where the
coverage bitmap is, and nowhere else. Note that the bitmap is buffered *coverage*, not a
coastline: it reaches about 11 km offshore and a live tile is written whole, ocean pixels
included. It also confirms the chunk grid is recoverable
from the keys, which the cropped ingest's fast path depends on
(`live_windows.live_chunk_grid_from_keys`). Alongside that: shape, CRS and affine equal the
zone's `ZoneSpec`, since a wrong origin otherwise surfaces hours later as data on the wrong
ground; and `coverage_sha256` matches the coverage group's `registry_sha256`, stamped last by
the writer, so it is the only evidence every pixel landed for *this* land-mask delivery.

Safe to run before or alongside campaign work. `export_zone_roi` is idempotent on that sha, so
a pre-generated mask is what the campaign would have written and the campaign skips it; a new
delivery changes the sha and both paths rebuild. `validate_only=true` re-checks without writing,
and any invalid zone **fails the run**, so a green run rather than a log line is the evidence.

```bash
# all zones, then re-assert the gate without writing
--param zones=null --param max_parallel_zones=16
--param validate_only=true
```

Cost is S3 request latency, roughly one PUT per live ingest chunk (~100 k campaign-wide across
the 112 land zones), so it fans out per zone and running it in-region matters:
the same export measured ~4 chunk-writes/second from a laptop.

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
[§ Sentinel-2 baseline correction](#sentinel-2-baseline-correction-applied-during-the-read), and choosing
between an ESA copy and an Element 84 copy of the same tile-date is the subject of
[§ Choosing between duplicate copies](#choosing-between-duplicate-copies-of-a-tile-date).

Each `CollectionConfig` records: collection ID, band list, native resolution, tile ID property
(for S2/Landsat property-based queries), and correction parameters. For OPERA RTC-S1 on
CMR-STAC there is no tile ID property, so the query falls back to a WGS84 bbox.

`PROVIDERS` also carries a `planetary-computer` entry. **Nothing has ever run against it** — it
is a sketch of a future path rather than a supported one, and the Cambridge TESSERA team's report
that Microsoft throttles heavy outbound traffic from Planetary Computer is why the OPERA RTC route
was built instead. Treat it as unvalidated configuration.

---
## Detailed Ingestion Process

The happy path in full. Failure modes live in
[ingest-error-handling.md](../../../docs/ingest-error-handling.md) instead.

Six things happen between a region and a written mosaic, and they are the order of this section:

1. **Ask the catalogue** what imagery touches the region in this window.
2. **Settle which day** each image belongs to, since a catalogue speaks UTC and a mosaic is
   indexed by local solar day.
3. **Size the request** so the catalogue will actually answer it.
4. **Filter what came back** — dates already written, and duplicate copies of one acquisition.
5. **Transform and write** — the reflectance offset during the read, the radar conversion after.
6. **Record what was examined**, so a month with no imagery reads as a finding and not a gap.

### STAC query strategy

S2 and Landsat are queried by tile ID property (e.g., `grid:code = T33UUP`), which returns
only items for that specific MGRS tile. OPERA RTC-S1 on CMR-STAC lacks an equivalent
property, so queries use a bbox derived from the MGRS tile via `mgrs_tile_to_bbox()`.

#### Cloud cover decides which scene wins a pixel

Cloud cover is intentionally **not** used as a filter at the STAC query stage — pixel-level
cloud classification is handled later (SCL for S2, ML model for inference). For S2, items
are sorted by `(date, eo:cloud_cover)` ASCENDING, so the clearest tile of a solar day comes FIRST.

First, because that is the one the loader keeps. `odc.loader`'s default fuser is `nodata_fuser`
— `np.copyto(dst, src, where=dst_is_nodata)` — so it writes only where the destination is still
empty, and this package configures no fuser of its own. The first valid source of a group supplies
a pixel and later ones fill its gaps, so the clearest scene covering a pixel wins it and a hole in
it falls through to the next-clearest rather than to nothing.

An item declaring no `eo:cloud_cover` sorts after every measured value, where it can only fill
gaps. `solar_day_sort_key` gives it infinity rather than 100, since 100 is itself a real reading
and the two would otherwise tie and be reordered by `id`, letting an unmeasured scene take ground
from one known to be fully clouded.

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

### Timestamp handling (`solar_days.py`)

Three things about a date have to be decided before a query is sent, and they are one subject:
which day an acquisition belongs to, which day a group of them is labelled with, and which day
range a catalogue is asked for. Getting any of them from a different convention than the others
loses imagery silently.

**One rule, and everything else follows: the solar offset is applied exactly once, by
`normalize_to_solar_day`, at the catalogue chokepoint. After that an item's `datetime` IS
its solar day (at noon UTC), and every date derivation downstream is a plain
`strftime("%Y-%m-%d")` with no offset.**

Applied at more than one site the conventions drift: keyed on the UTC date while the loader
groups by solar day, a group straddling UTC midnight is not sorted as intended and half its
baseline entries never match.

**Every date modality in the pipeline, and which convention it uses.** The point of one
chokepoint is that this table has no exceptions:

| where a date lives | form | convention |
|---|---|---|
| catalogue item, before the chokepoint | real acquisition instant | UTC — the only place a raw timestamp exists |
| catalogue item, after `normalize_to_solar_day` | **noon UTC of its solar day** | solar |
| loaded array coordinate (`odc.stac.load` output) | inherits the item stamp | solar (noon) |
| written store coordinate | `np.datetime64(solar_day)` | solar (midnight) |
| `existing_dates`, `written_dates`, baseline keys, `assessed_window` | `YYYY-MM-DD` strings | solar |
| chunk `own_start` / `own_end` | `YYYY-MM-DD` strings | solar |
| chunk `query_start` / `query_end` | `YYYY-MM-DD` strings | **UTC** — the only thing a catalogue understands |

The two timestamp forms differ (noon in flight, midnight in the store) and never meet as
numbers: everything crossing that boundary compares `YYYY-MM-DD` strings. The one deliberately
UTC row is the query bound, because a catalogue has no other vocabulary — which is why
ownership, not the query bound, decides what gets written. Noon rather than midnight is what
makes the stamp read as the solar day both directly and after `odc.stac.load` groups on it: noon
leaves half a day of margin, and no offset the grid produces (±11 h nearest the antimeridian)
crosses midnight.

**Three things enforce this rather than describing it:**

- `solar_day_of` **raises** on an item that is not stamped noon UTC. A raw item's UTC date
  is a plausible-looking wrong answer — right at central longitudes, wrong only where the
  offset crosses midnight — so a path that skips the chokepoint would look correct
  everywhere anyone checks interactively.
- An **architecture rule** (`solar-offset-applied-only-in-solar-days`) fails CI if
  `solar_day_offset_seconds` is called outside this module. One application is the
  invariant; a second is a bug in the opposite direction.
- Every consumption point **re-normalises defensively** rather than trusting call order, because
  every supplier (`query_fn`, `item_provider_fn`) is injectable and `normalize_to_solar_day` is
  idempotent.

A catalogue query is bounded in **UTC**; an ingest window and every chunk of it is a range of
**solar** days. Where a zone's offset crosses UTC midnight the two do not line up, and both ways
of ignoring that have been in this codebase: querying the chunk's own range and writing what
comes back splits a straddling solar day, so the earlier chunk writes it from its half, the later
chunk's half is dropped as an already-written date, and the day lands looking complete while
missing acquisitions; padding the query but clamping the pad to the window loses the first and
last solar day of a zone-year, since the padding vanishes at the window's own edges. Both are
silent — `assessed_window` still covers the days and the coverage gate still passes.

A chunk **owns** a range of solar days and **queries** a wider range of UTC dates:

```text
                 own:            2024-01-31 .............. 2024-02-29
                 query:   2024-01-30 ........................... 2024-03-01
                          └─ pad ─┘                              └─ pad ─┘

  a solar day landing on the cut is drawn from BOTH sides by the batch that owns it,
  and is owned by exactly one batch, so it is written once and written whole
```

Owned ranges tile the window exactly, so nothing is processed twice and nothing outside the
window is written — **ownership is what guarantees that, never the query bound**, which cannot
see solar days at all. The pads are deliberately *not* clamped to the window, because a solar day
owned at its very edge still draws on the UTC day beyond it. One day of padding is always enough:
the offset is a whole number of hours in `[-12, +12]`, so solar day `D` lies inside UTC
`[D-1, D+1]`.

`owned_items` is applied **between the query and the loader**, never after loading. Filtered
there, the loader builds a group only for an owned day and that group holds every image of it.
Filtered afterwards, a straddling day has already been split into two partial groups and what is
needed to rejoin them is gone.

Three chunkings share it today, and a new provider adds a fourth by writing a span producer and
nothing else:

| producer | used by | chunk |
|---|---|---|
| `month_ranges` | S2, streaming (default) | one calendar month, to bound item retention |
| `fixed_day_ranges` | S1 | `batch_days` solar days, to bound Dask graph size |
| `whole_window_range` | S2, `stream_stac_monthly=False` | the window in one query |

This rests on our offset arithmetic agreeing exactly with the loader's — both truncate
longitude over fifteen to whole hours. If they diverged an image could be filtered out as another
chunk's while the loader would have grouped it into this one, dropping it from the run entirely.
`solar_day_offset_seconds` is the single definition, and it is why `solar_grouping_longitude`
prefers the geobox centroid over a bbox midpoint, why `group_items_by_date` takes a
`mid_longitude`, and why the pre-sort uses the same key: the sort carries the fusion contract
(clearest tile FIRST within a group), so sorting on a different notion of "day" than the grouping
would silently let a cloudier pixel win. Central longitudes image far from UTC midnight and are
unaffected, which kept it latent.

#### Fusing OPERA's per-burst timestamps

A single MGRS tile bbox query returns ~10 burst granules per date, each with a slightly
different sub-second UTC timestamp (reflecting actual acquisition time). If passed to
`odc.stac.load` as-is, each burst becomes a separate time step instead of being mosaicked
together.

`normalize_opera_timestamps` delegates to `solar_days.normalize_to_solar_day`, grouping bursts
by **solar day** and setting every timestamp in a group to noon UTC of that day. `odc.stac.load`
then treats them as concurrent acquisitions and mosaics them into a single time slice.

#### Which day a slice is called, and why it is not an item's timestamp

A mosaic slice represents one **solar day**, and it is labelled with that day — taken from the
grouping key, not from the loaded dataset's own time coordinate.

`odc.stac.load` stamps each group from `group[0]`, tying the
label to whichever item the sort left first — which can disagree with the solar day wherever the
offset crosses UTC midnight, so two consecutive solar days collide on the time axis: the batched
write rejects them as not strictly increasing, the unbatched write rejects the second as a
duplicate slot.

Taking the day from the grouping key instead makes three things true by construction: labels are
unique per slice, monotonic across them (so the batched write needs no sorting), and stable
against the catalogue revising its cloud estimates. This only decides WHICH day — pixels, ordering
and which tile wins are untouched, and at mid longitudes the value is unchanged anyway.

### Adjusting Request Sizes to Accommodate EarthSearch (S2) Lambda Response Limits

Earth Search will not return more than about 6 MB in one answer, and a Sentinel-2 query over a
zone-year asks for far more than that. Sizing requests around the refusal is most of the query
code and all of its complexity, so this section is one plain-language pass over the mechanism;
[Appendix A](#appendix-a--the-earth-search-response-cap-in-detail) holds the implementation.

Four terms. A **page** is one request and the hundred results it returns. A **cursor** is a
bookmark the catalogue hands back, which the next request must carry — not a page number, so
there is no way to ask for the twentieth page directly and a refused request leaves no bookmark
for the one after it. A **date window** is a from-date and a to-date. A **worklist** holds one job
per date window; an impossible job is crossed off and replaced by two shorter ones.

The problem this shape exists for: Earth Search refuses **any request whose answer would exceed
about 6 MB**, AWS Lambda's synchronous response limit. The refusal arrives in 1.3 seconds, as fast
as a success, so nothing is overloaded and repeating cannot help — the remedy is always to ask for
a smaller answer.

```text
WHY A REQUEST GETS REFUSED -- the whole mechanism, in one line

   Earth Search will not return more than about 6 MB in one answer.
   That is AWS Lambda's limit on a single synchronous response, and the search API
   sits behind one.

   scenes are not all the same size, so a hundred of them is 4.6 MB on average
   but anywhere from 4.2 to 5.3 MB in practice -- and sometimes over the line:

      100 scenes from this bookmark  ->  would be ~6.2 MB  ->  REFUSED
       90 scenes from this bookmark  ->            5.6 MB  ->  answered
       75 scenes from this bookmark  ->            4.6 MB  ->  answered

   Same bookmark, same dates, character for character. Only the number asked for
   changed. So nothing is wrong with the bookmark, the depth, or the service --
   the reply was simply too big to send.

   Which hundred scenes you land on is decided by your bookmark and your date
   window, which is why it looks like the service has taken against one specific
   request. It hasn't. It is doing arithmetic on the size of the answer.

   THE MARGIN IS THE RISK: 4.6 MB average against a 6 MB ceiling is ~30% of room.
   Fatter scenes -- a newer processing baseline, a provider-side change -- push
   FIRST pages over the line, and a first page is the one refusal that shortening
   the dates cannot fix. `max_page_size` is the lever if that ever happens.


HOW THE QUERY IS SHAPED AROUND IT

   28 Feb ---------------------------------------------------------------- 1 Apr
       |  the catalogue reports the total beside the first page, so a window too big to
       |  walk is cut before the walk starts rather than hundreds of requests in
       v
    +--------+--------+--------+--------+--------+--------+
    |  job 1 |  job 2 |  job 3 |  job 4 |  job 5 |  job 6 |   jobs meet on a shared
    +--------+--------+--------+--------+--------+--------+   INSTANT, so nothing falls
        ok       ok       ok    refused     ok       ok       between them and nothing
                                  |                          is asked for twice
                                  |  cross it off, write two shorter jobs. Shorter dates
                                  |  regroup the scenes into different hundreds, so the
                                  |  fat group is split and every answer fits. The other
                                  v  five jobs are untouched and still running.
                                     If the dates cannot be shortened -- a single day, or
                                     a FIRST request, which a shorter window asks the same
                                     way -- ask for fewer scenes at a time instead.
                            16-22 March
                              +-- 16-19 March   ok
                              +-- 19-22 March   refused
                                    +-- 19-21 March   ok
                                    +-- 21-22 March   refused
                                          +-- 21 March   ok   <- one day is as far
                                          +-- 22 March   ok      as this can go

   Up to six jobs run at once. Almost all of a request is spent waiting for the catalogue
   to think -- 86% of it, before a single byte arrives -- so overlapping the waits is the
   only thing that moves the clock. Same query, same scenes, same order: 39 minutes as
   first written, 3.5 minutes now.
```

Two properties are tested. The jobs must add up to exactly the window asked for, no day missed
and no day added. And the results must come back in the same **order**, not merely the same set —
two scenes taken on the same day with the same cloud cover are separated only by which arrived
first, and that decides which one supplies an overlapping pixel.

Everything behind that picture — which months carry the heavy entries, why the page size is 100,
how a refused window is re-cut, and how the windows are walked concurrently — is in
[Appendix A: the Earth Search response cap in detail](#appendix-a--the-earth-search-response-cap-in-detail).

### OPERA-specific query quirks

OPERA RTC-S1 is queried through NASA's CMR rather than a STAC API, and three things about it have
no counterpart on the optical path: the query is built against CMR's native granule endpoint, one
date arrives as ten separate burst granules, and the items carry no projection metadata for the
loader to read.

#### Native granule query (orbit filtering + item construction)

`make_s1_item_provider` builds an `item_provider_fn` that returns ready-to-load OPERA items
**without calling CMR-STAC `client.search()` at all**. CMR-STAC's cursor pagination
intermittently 500s on CONUS-scale queries (nasa/cmr-stac#408) and pages internally at ~100
items regardless of the requested `limit` (#411); it also **silently ignores** the `query`
extension for CMR additional attributes such as `ASCENDING_DESCENDING`. The native CMR
Granule Search API has none of these problems.

The provider queries the granule API directly:

```text
GET https://cmr.earthdata.nasa.gov/search/granules.json
    ?short_name=OPERA_L2_RTC-S1_V1
    &attribute[]=string,ASCENDING_DESCENDING,ASCENDING
    &bounding_box=...
    &temporal=...
    &page_size=2000
```

Orbit direction is filtered **server-side** via `attribute[]`, so the response holds only the
desired orbit — no separate STAC search and no local granule-ID intersection. Each granule's data
links (`rel` ending `/data#`, href ending `_VV.tif` / `_VH.tif`) map onto the `S1_OPERA_BANDS`
asset keys (`0_VV`, `0_VH`) to build `pystac.Item`s shape-compatible with the rest of the
pipeline, with `title`, `time_start` and `polygons` supplying id, datetime and geometry.
Pagination uses the `CMR-Search-After` header, which pages cleanly at 2000. See
[ADR 009](../../../context_docs/decisions/009-native-cmr-granule-query.md).

#### Polarisation filtered server-side

A cost fix rather than a correctness one. Ingest needs dual-pol VV+VH, and the query carries
`attribute[]=string,POLARIZATION,VV` alongside the orbit filter. It discards nothing reachable,
since CMR matches a multi-valued attribute if ANY value matches, so VV admits every VV+VH granule.
The client-side check remains as a safety net, and its warning means a catalogue inconsistency —
metadata advertising VV without the bands published — rather than a regional fact.

#### When a zone has no usable radar

**That is a finding rather than a failure.** Over ice Sentinel-1 images in Extra Wide swath with
HH/HV polarisation, and the OPERA query discards anything that is not dual-pol VV+VH, so an ROI
whose land is ice has a catalogue full of granules and not one the ingest can use. Zone 23N
(Greenland) returns ~183,000 granules for 2021 and **zero** usable items. (They are not EW-mode
either: the Greenland ones report `BEAM_MODE=IW`, and a `BEAM_MODE=EW` query there returns
nothing.) Requiring a SAR store there failed the cell permanently, so `"both"` resolves to
`S1_ORBIT_NONE`, which activates no orbit and leaves the coverage gate checking reflectance alone.

**Permissive by default, refusable on demand.** A global product cannot reject terrain that is
radar-free as a matter of geography, so `resolve_s1_orbit`'s `allow_none` defaults to True. A
single run over terrain known to be imaged is the opposite case — there an absent store means
something upstream broke, and resolving to `none` would embed without radar and hide it — so
those callers pass the flows' `require_s1`, which reaches the resolver as `allow_none=False`. An
operator who names one orbit is never downgraded either. Which case it is comes from the ingest's
per-orbit item count, since it has just queried both orbits: `items_seen=0` means the source
offers nothing here, which is terrain rather than a gap. A consumer reading a finished mosaic
cannot distinguish the two, so its warning names the mosaic and points at that count. Accepting a
radar-free ROI necessarily means embedding S2-only pixels, and `InferenceConfig` derives
`allow_s2_only` for that case rather than asking the caller, because the alternative is a fill
that writes nothing and reports success.

#### UTM CRS derivation

CMR-STAC OPERA items lack the `proj:` extension, so `odc.stac.load` cannot infer the output
CRS. `mgrs_tile_to_utm_epsg` derives the correct UTM EPSG from the tile's zone number and
latitude band (C–M = southern hemisphere, N–X = northern hemisphere), e.g. `33UUP` → EPSG:32633.

### Date deduplication before loading

`_filter_existing_dates` removes items whose dates are already in the Zarr store before calling
`odc.stac.load`, so no task graph is built and no COG read is issued for data that would be
discarded.

The filter must be keyed the way the store was written. Both sensors load with
`groupby="solar_day"`, so their time axes hold solar days, and an acquisition's UTC date is not
its solar day where the offset crosses midnight. Callers grouping by solar day pass
`mid_longitude` down through `ingest_tile` / `query_stac_items`; matching UTC dates against a
solar-day set would filter only the half of a committed group on the near side of midnight, and
the surviving half would reload, regroup onto the day already present, and be written twice.

The filter is an optimisation, not the guarantee. On S1 the queries are built one batch ahead of
the writes, so the set they filter against is frozen before the run began; the write loop tracks
what it actually wrote and is the authority.

### Data transformations

What happens to the data between a catalogue item and a written pixel, in the order it happens:
items are rewritten before the load, the load itself is configured to produce the grid and the
groups we want, then the radar amplitude conversion runs after it. The one correction that does
not fit that sequence is the Sentinel-2 reflectance offset, which happens *inside* the read and
is large enough to have its own section below.

#### Pre-load (STAC items)

These happen before `odc.stac.load` is called:

| Transform | Where | What |
|---|---|---|
| **Date dedup** | `stac._filter_existing_dates` | Drops STAC items whose date is already written to the store. Keyed on the SOLAR day when the caller passes `mid_longitude`, matching how the store was written. |
| **Item sort** | `stac.query_stac_items` | For S2: sorts by `(date, cloud_cover)` ascending, so the clearest tile comes FIRST and the loader's first-valid-source fuser keeps it. |
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
  scene wins a pixel is decided by the sort order; see *Cloud cover decides which scene wins a
  pixel*.
- **Dimension rename** — `normalize_odc_dims` maps `odc.stac.load`'s `y`/`x` output
  dimensions to the project-wide `northing`/`easting` convention and drops `spatial_ref`.
- **The Sentinel-2 reflectance offset** is applied inside this read, per source — see
  [§ Sentinel-2 baseline correction](#sentinel-2-baseline-correction-applied-during-the-read)
  next.

#### Post-load

These happen after `odc.stac.load` returns:

##### OPERA RTC-S1 amplitude-to-dB conversion

OPERA products store linear amplitude (float32). `transforms.amplitude_to_db` converts to a
compact scaled uint16 suitable for storage and model inference:

```text
dB = 20 × log10(amplitude) + 50
scaled = dB × 200
result = clip(scaled, 0, 32767).astype(uint16)
```

Constants (`S1_DB_SHIFT = 50`, `S1_DB_SCALE = 200`) are ported from
`tessera_preprocessing/s1_fast_processor.py`. Zero/negative amplitudes are masked to `1e-10`
before `log10` to avoid domain errors; they are written back as 0 (nodata) after conversion.
This is a fully lazy Dask operation — no data is materialised until the Zarr write.

### Sentinel-2 baseline correction (applied during the read)

ESA changed the S2 L2A processing baseline at version 04.00 (January 2022), adding +1000 to all
surface reflectance values. Whether that offset has to be subtracted is a property of **who
served the pixels**, not of the collection: Element 84 harmonises its own COGs and subtracts it
for you, while ESA's originals carry it. Since Earth Search indexes both kinds, reading the
collection alone exempts or corrects them together — and one of those is always wrong and always
silent, a skipped correction leaving plausible pixels 1000 too high and a doubled one shifting
every value by 1000.

So `baseline_threshold` **is** set for Earth Search, and the decision is made per ASSET from where
that asset lives (`boa_offset.source_decision`). Three properties of that decision:

- **Judged over the reflectance bands only.** A real Element 84 item carries the original JP2s
  as extra assets beside its COG bands, so judging every asset reports a straddle for an item
  that is wholly harmonised where it matters. `scl` is excluded even though it *is* read: it is
  categorical and never corrected, so its producer cannot make the reflectance ambiguous. This
  is a different asset set from the one locality is judged over — see the duplicate selector,
  which uses the full read set including `scl`.
- **Decided per SOURCE, and applied as each image is read.** `odc.stac.load` fuses a solar day
  into one time slice, so a correction applied to its OUTPUT hits every tile at once, and a day
  whose tiles disagree has no correct answer — 347 days of one region-year were refused that way.
  The decision is stamped per reflectance asset at parse time (`stac.BoaOffsetParser`) and applied
  inside the read (`stac._BoaCorrectingReader`) before anything is resampled together, so no pixel
  is both corrected and uncorrected. It is purely additive, the amount being a constant: days the
  pipeline loaded before are bit-identical and only previously-refused days change. See
  [ADR 021](../../../context_docs/decisions/021-correct-the-boa-offset-per-image.md) §3.
- **Thresholded on the item's own declared baseline, and an unreadable one refuses.** An absent
  or malformed `s2:processing_baseline` parses as nothing rather than 0: read as zero it falls
  under the threshold and exempts pixels that may carry the offset, while correcting it takes 1000
  off pre-04.00 pixels that never had it. Both are wrong by the same amount in opposite directions
  and both are silent, so the source refuses. `item_baselines` is the only reader of that
  property.

**Consulting the assets at all is scoped to the collections that need it**, via
`CollectionConfig.harmonisation_varies_by_item`. Where the producer cannot vary between items the
**collection's configuration supplies the answer** instead: a correction threshold on such a
collection says every item is unharmonised, which is what the threshold is there to correct, and
`source_decision` takes that as `known_harmonisation` without consulting the bucket. That is what
lets a catalogue serving its bands under native asset keys — `B02`, `SCL`, which an item-level
read keyed on the configured band names finds nothing under — be decided here and corrected
rather than refused on every modern item. One decision serves both shapes of collection, so they
cannot disagree about a producer.

Which assets carry the reflectance bands is resolved through **odc's own alias table**
(`stac._reflectance_asset_keys`), not from the configured band names, for the same reason. `scl`
is excluded structurally — it is simply not among the resolved keys — rather than by a list the
corrector is told to skip.

The correction VALUE is a **constant**, `S2_BASELINE_OFFSET` of `-1000`; the baseline decides only
*whether* the offset is removed, never how much. `extract_baselines` remains separate: it records
what each item declared and is what reaches the store's `baselines_applied`, but it is **not** a
correction input. One integer per date is provenance, and correcting from it left raw post-04.00
pixels uncorrected whenever it omitted a date, carried the zero an unreadable baseline collapses
to, or named an arbitrary item's baseline on a multi-item date.

Duplicate selection has **two owners**, and neither is the shared query. `query_stac_items`
deliberately does not prune: `s2_roi` runs its own selection over that output and keeps the
rejected copies as the ladder `step_down_copies` walks when a source object will not read, so
pruning upstream would leave it nothing to step down to. `load_stac_items` prunes for everyone
else — both the documented `query_stac_items` -> `load_stac_items` workflow and `ingest_tile`,
which leaves it to the loader: the loader realigns `baselines` in place and that is the same dict
`ingest_tile` returns, so the map still describes the copy that was kept. Selecting over an
already-selected set is a no-op, so more than one owner is safe.

**One ambiguous shape survives, and it refuses** (`HeterogeneousProducerError`): a reflectance
source at or above the threshold whose producer cannot be determined — served from a bucket nobody
has classified, or belonging to an unharmonised copy that declares no readable baseline. The
refusal is a property of that one source and is gated on the threshold, since below it no producer
changes a pixel. It is skipped alone and counted rather than failing the leg, and duplicate
selection routes around it: a copy that would refuse is ranked last *and* withheld from the
fallback ladder, since a refusal is not a read failure and nothing retries one. "Alone" is a
property of `s2_roi`, which loads one solar day per call — a caller pairing `query_stac_items`
with `load_stac_items` over a multi-day list forfeits every day in it, so pass a day at a time.

**An item not exposing EVERY reflectance band under the configured names is `UNKNOWN`, not
`RAW`**, and the safe direction inverts here. A non-empty subset is not enough: nothing in this
module sees the alias table mapping a band name to an asset key, so a band absent under `blue` may
be served under `B02`, and `_prune_item_dict` preserves exactly those partially aliased items.
Letting one visible harmonised band speak for a hidden native-keyed one would silently subtract
1000 from pixels that may already be harmonised.

**How far the decision reached is reported after every load**, from counters the parser keeps:
how many reflectance sources it stamped, and how many of those were owed the offset. Reaching
*none* of them is the one way this can still go wrong quietly — an empty or mistaken
`reflectance_assets` set corrects nothing and produces plausible pixels 1000 too high — so a zero
count is a WARNING naming the assets it resolved and any other count is an INFO line. A warning
rather than an error, because a caller that replaces the loader also reports zero; what makes that
enough is that every other way of getting this wrong is loud, since an unclassifiable source
refuses and an unresolvable band name fails the load.

`baselines_applied` is correspondingly **lossy** on a day whose tiles declare different baselines.
The value is the first item's — the clearest tile, since the query sorts cloud-ascending — and the
day's other vintages are recorded nowhere. Deliberate: nothing in this package reads the value.

`correct_boa_dn` does the arithmetic, once, on one source's pixels. **The offset applies to every
valid DN, not only to bright ones**: ESA adds it across the whole reflectance range precisely so
that negative surface reflectance, routine over water and deep shadow, is representable in an
unsigned type. So the correction is `DN - 1000` for every DN from 1 upward, floored at the lowest
VALID code, which is **1**. Not zero, because zero is the nodata code and flooring a real dark
observation there makes it indistinguishable from no observation, which every downstream mask then
drops. Element 84's harmonised COGs floor at 1 for the same reason, and reproducing them exactly
is what makes a corrected raw copy comparable with a harmonised one. DN 0 itself never reaches the
arithmetic — the reader applies the result only where the source was valid — so nodata survives as
nodata.

**That the nodata code IS 0 is hard-coded, and that is an unchecked assumption.** `_NODATA = 0`
is a constant in `stac.py`, and the correction rests on it twice: DN 0 is excluded from the
arithmetic, and the floor of 1 is what stops a corrected pixel from looking like nodata. The real
answer belongs to the catalogue, which `odc` derives per band from `raster:bands` and hands the
reader as `RasterLoadParams`, but the driver is installed on any collection whose config sets
`requires_baseline_correction`, so nothing ties the constant to what that collection declares.
Under a marker of 65535 every gap pixel would test as valid — the correction shifts them to 64535,
the fuser's `dst == fill_value` test stops recognising them as empty, and the mosaic gains
data-looking pixels where there was no observation. Both Sentinel-2 providers declare 0 today, so
this is latent; resolving the marker from `cfg` and refusing anything else is owed.

The arithmetic widens to `int32` and casts back to the INPUT dtype, so nothing wraps and the
store's unsigned arrays are unaffected: the offset is negative and the floor is positive, so an
unsigned input stays representable. Adding a negative Python int to a `uint16` array raises under
numpy 2, which the widening exists for.

**The floor still acts on resampled values, and that is a recorded limit.** `odc.stac.load`
reads and resamples in one step, so the wrapped reader sees already-warped pixels, and six of the
ten configured bands are natively 20 m on a 10 m grid. A pixel whose kernel spans the DN-1000
boundary is floored where Element 84, flooring each source first, would not have been. Fixing it
means taking over the read-and-warp step and would rewrite pixels in every existing store, so it
is owed separately — measured in
[ADR 021](../../../context_docs/decisions/021-correct-the-boa-offset-per-image.md) §6, alongside
`context_docs/decisions/020-boa-offset-applies-to-every-valid-dn.md`.

### Choosing between duplicate copies of a tile-date

A catalogue indexes the same tile-date more than once whenever a granule is reprocessed, and
sometimes from more than one region. An **acquisition** is one real pass of the satellite over
the tile — a high-latitude tile can have two in a day, and each can be published several times as
ESA reprocesses it — so the job is to reduce each tile-date to one copy per acquisition, keeping
every distinct pass. `duplicates.py` does that before the loader sees it, because `odc.stac.load`
fuses a solar-day group: two copies of one acquisition would be blended into one pixel stack at
two different processing baselines, and the baseline recorded on the store would match neither.

The copies it rejects are not discarded. They become the **fallback ladder**: if the chosen
copy's object will not read, the recovery steps down to the next rung rather than losing the
date. That is why several terms below rank for "can this copy be processed at all" ahead of
"is this copy the best", and it is described in full at the end of this section.

Preference is **one sort key** (`_preference_key`), and what makes it work is that it is
**context-free**: no term means "best in my group", so the same tuple orders two copies of one
acquisition and two copies from different ones. A term relative to the group's own best baseline
would make cross-acquisition comparison meaningless and force a second key alongside this one. Add
a signal here and nowhere else.

The eight terms, and what each is there to stop. The first four ask whether a copy can be
processed at all and the last four which of the processable ones is best — a copy that cannot be
processed must never outrank one that can, however good its pixels would have been:

| # | term | what it protects |
|---|---|---|
| 1 | read-set completeness | a copy missing a band cannot deliver the date, and the ladder does not recognise that as a read failure |
| 2 | producer decidable | an undecidable producer refuses the date, and the ladder cannot step down on a refusal |
| 3 | belongs to this acquisition | a copy naming no pass was clustered by guesswork, so it must not displace one that says which pass it came from |
| 4 | baseline readable | an unreadable baseline refuses the date too |
| 5 | processing baseline, descending | newer reprocessing first, by value, so every rung of the ladder stays in baseline order |
| 6 | owes no offset correction | an uncorrected copy cannot be wrong by the offset; a corrected one is only right if the bucket lists are |
| 7 | locality, among equal baselines | cheaper egress, but never at the price of an older pixel |
| 8 | `s2:sequence`, then item id | a total order, so a rerun cannot silently produce a different mosaic |

In full, in that order:

1. **Read-set completeness**, judged over the assets *this* load will request — the configured
   bands plus the caller's `extra_bands`, not a fixed list and not the broader pruning set, which
   keeps `scl` regardless. First, because a copy missing one cannot deliver the tile-date at any
   baseline, and a missing band is not a failure the fallback ladder recognises.
2. **Whether the producer is decidable**, where it would change a pixel. An undecidable producer
   refuses its date at or above the correction threshold, and a refusal is not something the
   ladder can step down on. A copy spanning a harmonised and a raw producer is not undecidable:
   each source is decided on its own bucket and corrected band by band. Inert below the threshold.
3. **Whether the copy demonstrably belongs to the acquisition it is ranked in.** A copy naming
   neither an observation nor an instant was clustered arbitrarily, so it must not displace one
   that says which pass it came from.
4. **Whether the baseline is readable**, for producers whose correction depends on it, unknown
   sorting last. An unreadable baseline refuses the whole date, for the reason given under the
   baseline correction above, so an older reprocessing that can be corrected beats a newer one
   that cannot be processed — and it outranks every term below because the ladder recovers from a
   read error but not from a refusal. Already-harmonised copies are exempt.
5. **Processing baseline, descending.** Ordered by value rather than "is it best", so every rung
   of the fallback ladder stays in descending baseline order — a read failure can skip a 04.00
   copy and hand out a 03.00 one.
6. **Whether the copy owes an offset correction at all**, where the producer is an item's own
   property. Below the baseline, for two pixel-level reasons: an already-harmonised copy had its
   floor applied before resampling where one we correct is floored after, so the two disagree on
   very dark pixels; and a copy owing nothing cannot be wrong by the offset, while a corrected one
   is right only if the bucket lists and declared baseline are both honest. A quality-versus-
   quality preference must not buy a better pixel with an older reprocessing. Inert below the
   threshold — though not universally: zone 01N in 2017 carries 15 at baseline 05.00, so it does
   fire on real data. Also inert where the producer is the collection's answer.
7. **Locality, among equal baselines only.** A copy whose read assets sit in a preferred bucket is
   cheaper to read, so it wins a tie. Restricting it to ties stops it buying cheaper egress with
   an older pixel. This is why there are two bucket lists: harmonisation is a **pixel** claim,
   locality a **cost** claim. They name the same buckets today but their key sets differ by `scl`,
   so a copy can be harmonised without being local.
8. **`s2:sequence`, descending, then item id.** The id keeps the choice independent of catalogue
   response order, so a rerun cannot silently produce a different mosaic — and it makes the key a
   total order, so no comparison ever falls back to input order.

Two properties of that ordering are easy to get wrong and are held by tests:

- **Locality is judged over the read set, not over every asset.** A real Element 84 item carries
  its COG bands *and* the original JP2s, across two buckets, so requiring every asset disables
  the preference altogether. It is judged over the *whole* read set: one local band among many
  remote ones is not locality, and an item exposing none of them is remote, because absence of
  evidence is not evidence of locality.
- **An unreadable baseline sorts LAST, and makes locality inert for that copy.** A missing
  baseline is an absence of evidence rather than a tie: as a tie it let a copy with no baseline
  displace a raw copy at 05.00, taking an older reprocessing *and* skipping the correction. Such
  a copy also refuses its whole date downstream, and the ladder recovers from a read error but
  not a refusal. An already-harmonised copy is exempt, since no offset decision rests on its
  baseline and penalising it would hand the tile-date to an older raw reprocessing.

**Which copies are the same acquisition is decided by identity, not by a timestamp.** Two
reprocessings of one granule share a datatake — mission, sensing start and absolute orbit, in
`s2:datatake_id` — and differ only in the baseline suffix. They do **not** agree on the catalogue
`datetime`, which is per-copy and has been seen to differ by more than three minutes between two
copies of one granule, so a tolerance around that timestamp cannot separate "two reprocessings"
from "two passes" without getting one wrong. The timestamp window survives only as the fallback
for a copy naming no datatake. Splitting on a real acquisition protects genuine same-day
coverage: successive orbits revisit a high-latitude tile the same day, and keying on `(tile,
solar day)` alone dropped 493 of 2,733 distinct acquisitions as duplicates.

A copy naming **no** datatake joins an identified acquisition its timestamp places it in, before
it is allowed to start one, and it is matched against *any* member of that acquisition — members
of one observation do not agree on the timestamp, which is the whole reason identity is primary,
so closeness to any of them is the available evidence. Without that, one reprocessing declaring
the datatake while its sibling omitted it were never compared however close their timestamps, and
both survived to be fused.

**The tile key is read from whichever property the catalogue populates**, `grid:code` or
`s2:mgrs_tile`, then the item id — all canonicalised to one form, so two catalogues naming one
tile produce one grouping key. Reading only Earth Search's property would leave every item from a
catalogue naming the tile elsewhere unkeyable, which makes duplicate selection a silent no-op for
that whole provider rather than an error.

**Where the producer cannot be read from an item's assets, the collection supplies it**, through
`known_harmonisation` on `select_preferred_duplicates` — the same value
`stac.collection_harmonisation` gives the correction path, so the two cannot disagree. This is
load-bearing rather than an optimisation: a spare judged only on visible assets looks harmless,
is offered to the ladder, and aborts the ingest when a read failure steps down to it, because the
recovery loop steps down on a read failure and not on a refusal.

The **fallback ladder** — the rejected copies, in the order a read failure steps down them — is
built by one global sort over the whole tile-date, using a key with no notion of "best in my
group". Ranking each acquisition separately and concatenating the results is wrong further down
the ladder: with one acquisition holding 05.00 and 01.00 spares and another holding 04.00 it
yields `[05, 01, 04]`, and since the unattributed recovery consumes the head on each retry, the
second retry takes 01.00 and never reaches 04.00. In the global order an unreadable baseline
simply sorts last, which is the same protection by a more direct route: a copy whose baseline
cannot be read is the one whose correction will silently be skipped.

Buckets are compared by parsing the href's host and path rather than by substring, so a lookalike
host cannot be mistaken for a preferred one. The baseline is matched as a version string rather
than parsed as a number, so `"NaN"`, `"Infinity"` and every other value that is numeric without
being a version read as unknown — see `item_baselines.py`.

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
and the zone-year can never complete. Keyed on the store, the
resume repairs the attribute. The extra existence probe runs only when nothing was written, so a
normal run pays nothing for it. A genuinely absent store is still left alone: there is no repo to
annotate, and that case was never ambiguous — no store means the orbit is absent and callers
downgrade.

Every uncertain path is strict: an absent, malformed or unparseable attribute excuses nothing,
and a partially-covered month stays an error because it could hide unexamined days. Failing to
write the attribute is logged, never raised — the gate simply falls back to requiring every
month. `assessed_empty_dates` is recorded alongside for observability, separating "sparse region"
from "the footprints are wrong".

---
## Authentication (EDL / OPERA data)

OPERA RTC-S1 data hosted by ASF requires NASA Earthdata Login (EDL) credentials because ASF
uses NASA's OAuth2/URS system for access control. Unlike commercial cloud data (S2, Landsat),
OPERA data is not publicly readable from S3.

### Setup

```bash
export EARTHDATA_USERNAME=your-username
export EARTHDATA_PASSWORD=your-password
```

You must also approve the **ASF Cumulus** application at
[urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) → Authorized Apps.

### S3 Direct Access (preferred)

`auth.get_s3_credentials` exchanges EDL credentials for temporary AWS STS credentials:

1. `GET https://urs.earthdata.nasa.gov/api/users/tokens` — reuse an existing EDL bearer
   token (EDL accounts have a maximum token limit; creating a new one unnecessarily can hit
   that limit).
2. If no token exists, `POST .../api/users/token` to create one.
3. `GET https://cumulus.asf.alaska.edu/s3credentials` with `Authorization: Bearer <token>` —
   returns `accessKeyId`, `secretAccessKey`, `sessionToken` (valid 1 hour) for the
   `asf-cumulus-prod-opera-products` bucket in `us-west-2`.

`set_s3_credentials` then injects these onto both the orchestrator process and all current and
future Dask workers via a `WorkerPlugin`. It sets `AWS_*` environment variables (consumed by
boto3 when `odc.loader` builds an `AWSSession`) and resets the cached per-thread session so the
next `/vsis3/` open picks up the new credentials.

**Renewal runs on a timer, not on the work loop.** `s1_roi.credential_ticker` re-checks the
credential's remaining life every `CRED_TICK_INTERVAL_SEC` while batches are being consumed; the
loop's per-batch and per-date checks remain as a fallback, on `ingest_s1_roi_sar`'s
`cred_refresh_interval_sec`. Loop-driven renewal can only fire
*between* units of work, so a unit outliving its margin cannot renew from inside itself — and that
coupling is self-reinforcing, since failing reads stop the progress that would trigger renewal.
What a worker receives is also a **snapshot**: the plugin freezes the credential at construction,
so a worker joining N minutes after the last broadcast starts with only the remaining TTL and past
it with none. Under adaptive scaling workers join throughout a leg, so the broadcast cadence is a
correctness condition and the ticker is what bounds N. Every broadcast logs the advertised expiry
(`S3 credentials broadcast to workers`).

**Per-thread AWSSession cache**: `odc.loader` caches a boto3 `AWSSession` per thread in
`threading.local` on first use and ignores subsequent env var updates for that thread's lifetime.
Dask task pool threads are long-lived, so the initial 1 h STS token was being pinned across
refreshes and expiring mid-read. `auth.py` patches `odc.loader._rio.ThreadSession` at module
import time so each thread self-detects `AWS_ACCESS_KEY_ID` drift and rebuilds its cached session
from current env vars; `rasterio.env.Env`, entered by `odc.loader.rio_env()` on every `/vsis3/`
open, then hands GDAL the refreshed session's frozen credentials, so no `gdal.SetConfigOption` or
`VSICurlClearCache` is needed. This reaches into private `odc.loader` internals
(`_OdcThreadSession`, `_local`): if odc renames those symbols the import fails loudly, and
`tests/unit/ingest/test_auth.py` catches it in CI.

OPERA asset STS credentials are intentionally **never cleaned up** from env vars, which avoids
one Dask task's cleanup removing credentials another still needs. The consequence: once
`set_s3_credentials` runs, the `AWS_*` env vars hold OPERA-scoped tokens granting access **only**
to `asf-cumulus-prod-opera-products`, so any S3 access to the project's own bucket resolving from
those env vars — every icechunk `Repository.open`/`create` in the S1 write path — fails with
`AccessDenied`.

Icechunk/Zarr operations on the project's own bucket therefore must resolve **IAM-role**
credentials, bypassing the env vars. The mechanism:

- `providers/aws/credentials.py::iam_icechunk_credentials` resolves credentials from the
  botocore chain with the `env` provider **removed**, so it always lands on the deployment's
  IAM role (instance-metadata / ECS task role / local SSO) regardless of what STS tokens the
  env vars hold. It returns `icechunk.S3StaticCredentials`.
- `storage.zarr_store` exposes a `credentials_provider(provider)` **context manager**.
  `_create_storage` uses the registered provider as the `get_credentials` callback for any S3
  open lacking an explicit one, for the duration of the block — scoped rather than permanent, so a
  reused Dask worker is not left pinned to it, and the previous provider is restored even if the
  body raises. The storage layer ships this as `None` and never imports botocore, per the
  `no-botocore-outside-aws-provider` architecture rule; only the AWS provider supplies it.
- The `process_roi_sar` Prefect task registers `iam_icechunk_credentials` through that hook when
  `use_s3_direct=True`. **In the task shell, not the flow body** — under the Dask task runner the
  domain function and its store writes execute in a *worker* process, so a provider registered in
  the flow-runner process would never reach them.

The plain-Zarr side needs the same identity and one property beyond it. An ROI mask is read
through fsspec rather than icechunk, and
`providers/aws/credentials.py::iam_s3_storage_options` is the fsspec counterpart — the same
env-stripped chain in the shape fsspec takes as `storage_options`. The ingest is handed the
**callable**, not its result, and `read_roi_mask` resolves it inside each block read rather than
once at graph build.

That matters because the mask array is LAZY: its block reads happen inside a later
`write_day_windows` compute, which on the radar path spans a whole 30-day batch. One credential
resolved at graph-build time would be presented by every one of those reads and, once expired,
fail with `ExpiredToken` on a bucket the role can always read — a lifetime problem wearing a
permissions problem's error message. Two consequences: each block read pays its own store open
and metadata round trip where the old construction paid one for the whole array, and the returned
array is cloudpickle-only because the closure is nested. Both are measured in
`context_docs/decisions/022-resolve-the-roi-mask-credential-at-read-time.md`, and both argue
against handing this array to a plain-pickle boundary or reading a zone grid you do not need.

**IMDS throttling — why `_resolve_iam_credentials` is `lru_cache`d (gotcha).** The credential
machinery has two distinct TTLs, and conflating them overwhelms the EC2 Instance Metadata
Service (IMDS):

- `iam_icechunk_credentials` sets `expires_after=15min` on the returned `S3StaticCredentials`,
  which is how often **icechunk** re-invokes the callback per repo client — not how often we
  should touch IMDS.
- `_resolve_iam_credentials` is `@lru_cache(maxsize=1)`, so the botocore session and the live
  `RefreshableCredentials` it returns are built **once per process**. Those refresh themselves in
  the background, and `get_frozen_credentials()` is a pure expiry check that only re-hits IMDS
  inside botocore's refresh window (advisory ~15 min before the ~6 h token expiry), lock-guarded
  against stampedes.

Without the cache every callback built a fresh session and did a **cold IMDS resolve**, which
under many concurrent workers bursts IMDS past its per-instance rate limit and surfaces as
`failed to load IMDS session token / invalid token` or `no providers in chain provided
credentials` — transient, but enough to fail a run of chunks. Caching decouples how often
icechunk asks from how often we hit IMDS: the former stays at 15 min, the latter drops to roughly
once per token lifetime. `lru_cache` does not cache exceptions, so a failed cold resolve retries.
The same provider is injected into inference workers, where long-lived Ray actors made this the
dominant failure mode.

S3 direct access also removes a 5-hop OAuth redirect chain per OPERA COG tile: one round trip
buys an hour of STS credentials and GDAL then reads `s3://asf-cumulus-prod-opera-products`
without per-file HTTPS redirects, which at batch scale is the dominant latency reduction.

### URL Rewriting

CMR-STAC returns HTTPS asset URLs in two formats depending on satellite vintage:

| Format | Example |
|---|---|
| **datapool** (older S1A) | `https://datapool.asf.alaska.edu/RTC/OPERA-S1/<filename>` |
| **earthdatacloud** (newer S1C) | `https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/<dir>/<file>` |

`auth.rewrite_assets_to_s3` converts both to `s3://asf-cumulus-prod-opera-products/...` by pure
string manipulation, no HTTP calls. For the datapool format the granule directory name is
reconstructed by stripping the band suffix (`_VV.tif`, `_VH.tif`, `_mask.tif`) from the flat
filename.

### Legacy CloudFront Signed URLs (fallback)

`_EDLSession` is a `requests.Session` subclass that preserves the `Authorization` header
across cross-domain redirects. Python `requests` strips this header when following a redirect
to a different domain. The ASF download chain goes: `datapool.asf.alaska.edu` → 
`urs.earthdata.nasa.gov` (OAuth exchange) → CloudFront CDN. Because the header is stripped
at the first hop, it is missing by the time URS sees the request. `_EDLSession.rebuild_auth`
re-injects credentials whenever the redirect target URL contains `urs.earthdata.nasa.gov`.

`resolve_item_assets` follows the full redirect chain per asset and mutates the STAC item's
asset HREFs to CloudFront signed URLs before `odc.stac.load` reads them. This path is kept
for out-of-region access where S3 direct is not available, but is significantly slower.

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

---

## Appendix A — the Earth Search response cap in detail

**Earth Search refuses any request whose response would exceed roughly 6 MB** — AWS Lambda's
limit on a synchronous response, and the search API sits behind one. Every 502 this campaign has
seen from that provider is this and nothing else.

What makes it bite unevenly is that items are not all the same size. Catalogue entries from
roughly **November 2018 to March 2019** are about 100× larger than normal: an entry usually
carries the tile's bounding rectangle, a couple of hundred bytes, while these carry the outline of
where the imagery actually falls, and since Sentinel-2 builds an image from twelve detectors that
edge is a fine sawtooth — 98 KB in one measured entry against 0.2 KB. The asset list is ~18 KB
either way, so the outline is what makes them heavy. The band is a reprocessing gap: ESA
reprocessed most of the archive to a version carrying the simple rectangle, and those months are
the stretch where only the original 02.11 is on offer. It cannot spread, since no current
processing version produces these outlines, and it could disappear if that stretch is ever
reprocessed.

Two consequences in this code: a hundred entries is ~2.2 MB outside the band and at or over the
ceiling inside it, so page refusals are a 2019 phenomenon; and a month of these entries holds an
order of magnitude more bytes than the same month in 2024, which is part of why the query streams
month by month. Any per-item figure — page size, retained bytes, query timing — reads differently
here than anywhere else in the archive.

The page size is `STACProvider.max_page_size` (the `limit` per page request), defaulting to 250
and set to 100 for Earth Search. It applies only to providers queried through `client.search()`,
which in production means Earth Search; the OPERA `cmr-asf` path queries the native CMR Granule
API instead — see [ADR 009](../../../context_docs/decisions/009-native-cmr-granule-query.md).

**Why 100 for Earth Search, and what to watch.** Not throughput but the cap: 250 items of
`sentinel-2-l2a` is always over it. A hundred averages 4.6 MB, yet the largest page ever served
was 96% of the cap, so the margin between fine and refused is about a quarter of a megabyte and
the average is the wrong number to reason from. Lowering it further is not the answer — six months
of a ten-year archive is a concentrated problem, and the page-size fallback below handles it where
it happens rather than taxing every query in every year. If first pages ever start returning 502
this margin is the first thing to check, since a first-page refusal is the one case no date-window
re-cut can route around.

**The same cap also refuses pages deep in a walk**, which looks like a separate defect and is
not. Because item sizes vary the refusal is deterministic in the *request* — cursor and date
window together — rather than in how deep the walk has got, which is why the same refusal has
appeared at page 289 of one window and page 14 of a shorter one sharing its late bound.

What clears it is either a smaller response or a regrouping that produces one, and `stac.py`
tries them in cost order.

**A shorter window first.** Its halves between them walk about as many pages as the parent would
have, where a smaller page re-walks the whole window at twice the requests. What matters is the
window's **end** date: the catalogue pages newest-first, so the late bound fixes the whole cursor
sequence and shortening only the start does not help. `_query_stac_items` re-queries as shorter
windows on any upstream-error refusal past the first page, recursing until a window completes or
reaches a single day. Separately, it reads the match count reported beside the first page and
cuts a window matching more than `_MAX_QUERY_ITEMS` to size — that bounds *cost* rather than
fixing the defect.

**A smaller page as the fallback**, halved down to `_MIN_PAGE_SIZE`, because shortening cannot
reach two refusals: a **first page**, which a shorter window asks identically, and a **single
day**, the re-cut's floor. A stated overload (429, 503) is never answered with a smaller page:
that means the provider is busy, and more requests is the wrong direction. A refusal neither
lever can route around still raises the classified `CatalogueQueryError` with its token.

**Concurrency.** The windows are independent searches, so `_fill_window_tree` walks up to
`_QUERY_WINDOW_WORKERS` (2) of them at once. The worklist is driven from the calling thread and
tasks only ever walk — they never submit and never wait — so deadlock is structurally impossible
rather than merely unobserved. Each thread gets its own `Client`, because `StacApiIO` wraps a
`requests.Session` that is not documented thread-safe. Output order comes from
`_WindowWalk.preorder()` on the finished tree, and the `id` dedupe runs at that assembly step
rather than as pages arrive, so first-occurrence-wins means first in the **walk** and not first
off the wire. Two, not more, even though more is faster: the campaign runs tens of cells against
this one provider at once, so the setting multiplies the concurrent search streams Element 84
sees. At 6 they answered the fleet with 403; 2 ran clean for a whole campaign. A failure does not
stop the other windows — every window is walked, all failures collected, and the
depth-first-earliest raised, so which failure surfaces is a function of the query rather than of
which task finished first.

The re-partition is a pure re-cut, never a narrowing, and it is exact in both directions: the
outer bounds are the caller's own date strings handed straight back, and every interior boundary
is a single **instant** (`T00:00:00Z`) shared by the window that ends there and the window that
starts there. The catalogue's range is inclusive at both ends, so the union is the input window
with no gap and no overhang, and the only overlap is that one instant. An instant rather than a
date because the client expands a bare date end to `T23:59:59Z`, so windows abutting on
consecutive DATES would leave the last second of each seam's earlier day unasked for; sharing the
whole boundary DAY closes that gap too, but makes every seam re-fetch a full day for the dedupe
to discard. Items are deduped by `id` across every search a query runs, which absorbs the
boundary instant and the antimeridian overlap alike.

It does change the order items are *walked* in — one walk returns the window newest-first, the
worklist returns window by window in date order — which is safe only because `query_stac_items`
re-sorts with `solar_day_sort_key`, making the final sequence a function of the items rather than
of the order the walk produced (see
[§ Cloud cover decides which scene wins a pixel](#cloud-cover-decides-which-scene-wins-a-pixel)
for what that sort decides).

The cap, the measured margins, the sampling behind the heavy band and the levers that are closed
are all derived in `context_docs/ingest/campaign-ingest-measurements.md` §7c.

