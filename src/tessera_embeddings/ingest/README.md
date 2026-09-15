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
- [Error handling](#error-handling)
- [Performance Optimizations](#performance-optimizations)
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

**Filtering.** Whole items are dropped first — when their date is already in the store, when a
reprocessed granule duplicates one already chosen, or when an optional caller hook rejects them.
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
sections up to [Authentication](#authentication-edl--opera-data) describe the happy path;
[Error handling](#error-handling) and [Performance Optimizations](#performance-optimizations)
are the rest.

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
| `loader_failures.py` | Keeps what a failed load knows — which object, and why — neither of which reaches the caller on its own. One `install_capture_everywhere` call covers every current and future worker. [§ When a source object will not read](#when-a-source-object-will-not-read) |
| `auth.py` | Earthdata Login for ASF-hosted OPERA data: S3 direct access on roughly hourly credentials, plus legacy signed URLs. Renewal is timer-driven, because the credentials expire on their own clock. [§ Authentication (EDL / OPERA data)](#authentication-edl--opera-data) |
| `transforms.py` | Post-load lazy Dask transforms. Currently `amplitude_to_db`. [§ OPERA RTC-S1 Amplitude-to-dB Conversion](#opera-rtc-s1-amplitude-to-db-conversion) |
| `roi.py` | ROI utilities: read an existing Zarr ROI store, rasterise a GeoJSON polygon to a boolean mask on a UTM grid, load Sentinel-2 tile footprints. [§ Generating an ROI](#generating-an-roi) |
| `roi_processing.py` | Higher-level ROI helpers used by the `generate_roi` flow. |
| `source_coverage.py` | Optical preflight: does the catalogue publish anything reaching a zone's live land in this window, answered before any cluster is provisioned. Three-valued — only a positive finding of absence refuses. [§ Zone ingestion (the global campaign) — ADR-011](#zone-ingestion-the-global-campaign--adr-011) |
| `catalogue_refusal.py` | Tells a catalogue that is BUSY apart from one that cannot serve this REQUEST, and names the request either way. [§ When the catalogue refuses: naming the request, and telling the two refusals apart](#when-the-catalogue-refuses-naming-the-request-and-telling-the-two-refusals-apart) |
| `live_windows.py` | Derives the chunk-aligned live windows every ingest loads and writes, and narrows them per date to the land that date's imagery reaches. [§ Cropping to live windows (unconditional)](#cropping-to-live-windows-unconditional) |
| `_http.py` | Shared HTTP helpers for catalogue and granule queries: retries that log each attempt, calls the caller can abandon, and a guard for replies that claim success but are not JSON. |
| `_pipeline.py` | A prepare/consume pipeline with a look-ahead depth, so the next item is prepared while the current one is consumed. Buys buffering, never concurrency. [§ Pipelining a date's preparation (`pipeline_dates`)](#pipelining-a-dates-preparation-pipeline_dates) |

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

The happy path in full. Failure modes are collected under
[Error handling](#error-handling) instead.

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
UTC row is the query bound, because a catalogue has no other vocabulary — which is exactly why
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

The mechanism is one idea. A chunk **owns** a range of solar days and **queries** a wider range
of UTC dates:

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

That distinction matters. `odc.stac.load` stamps each group from `group[0]`, tying the
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
a smaller answer. The diagram is the whole mechanism; everything after it is detail.

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
that asset lives (`boa_offset.source_decision`). Three properties of it are worth stating:

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
does about it.

**One fact governs every decision here.** A store's dates are append-only — a date can only be
added after the newest one the store already holds — so a date the pipeline gives up on is given
up permanently, and no later run can fill it in. That makes the two failure modes cost wildly
different amounts. Giving up too early costs a hole in the dataset that nothing can repair.
Giving up too late costs wall clock on a job that will be dispatched again anyway. So the whole
section is built to spend time rather than dates, and the mechanism behind the constraint is in
*Where a resumed run starts* at the end.

Three things can fail, and they are answered differently:

- **The catalogue will not answer a query.** Nothing has been read yet, so nothing is at risk
  beyond the time spent asking again.
- **A read will not produce pixels.** Here the decision is expensive, because one answer gives a
  date up and another waits for it.
- **Whatever was lost has to be reckoned with by the next run**, which starts from what the
  store already holds rather than from what the last run intended.

### How a failure is decided

Reading one satellite image can fail for very different reasons, and the right answer to each is
different — sometimes opposite. So a failed read is asked one question, once, and gets exactly one
answer.

```
                            a read fails
                                 |
                                 v
                  +-------------------------------+
                  |  What does the error say the  |
                  |  problem actually IS?         |
                  +-------------------------------+
                                 |
   our own login is wrong  <-----+-----> the provider said "no"
   (bad key, expired token)      |       (busy, throttling, 500s, "access denied")
        |                        |            |
        v                        |            v
   STOP the job.                 |       WAIT, then ask again. Their bad day,
   No retry and no other         |       not our data. Minutes, not seconds.
   copy fixes our own key.       |
                                 |
   the request is wrong  <-------+-------> the file is damaged
   (400, 401, a bad URL)         |         (won't decompress, truncated,
        |                        |          missing a band we need)
        v                        |            |
   STOP the job.                 |            v
   Every copy is fetched         |       Try ANOTHER COPY of the same
   the same way, so no           |       image. Providers often publish
   copy will read either.        |       the same scene twice. Only if no
                                 |       copy is left, skip this one date.
                                 |
                                 +-----> the file was never published
                                 |       (404, "no such key")
                                 |            |
                                 |            v
                                 |       Same as damaged: another copy first,
                                 |       then skip the date if there is none.
                                 |
                                 +-----> WE CANNOT TELL
                                              |
                                              v
                                         Fail the job so it runs again later.
                                         Never skip a date on a guess.
```

The last branch is the important one. **A date is only ever abandoned on positive evidence that the
image itself is unusable.** Anything we cannot explain fails the job instead, which costs time and
is recoverable, rather than costing a date, which is not.

#### Waiting: where it happens changes what it costs

Two different budgets, for one reason:

```
  waiting INSIDE a running job     ~ minutes    the machines it rented sit idle, so this is expensive
  waiting BETWEEN attempts          ~ tens of   the machines are already released, so this is nearly
                                      minutes   free — patience goes here
```

A provider having a bad minute is ridden out inside the job. A provider having a bad half-hour is
better handled by letting the job fail, releasing the machines, and trying again later — a restart
begins the day after the newest date the store holds, so nothing is redone (see *Where a resumed
run starts*).

One extra guard: the long wait is only granted after a job has already read something successfully.
"Access denied" looks identical whether the provider is misbehaving or our permissions are simply
wrong — but wrong permissions fail the very first image, while a provider wobble arrives after the
job has already been served. So the first successful read is what earns the patience.

### The catalogue would not answer

Both of these happen before a single pixel is read, so the remedy is always to ask again, ask
differently, or stop the leg — never to abandon a date.

#### When the catalogue refuses: naming the request, and telling the two refusals apart

`catalogue_refusal.py` is where a refused query stops being anonymous. Two things about the
client stack make that necessary:

- **The request is discarded on the way up.** `StacApiIO.request` catches every transport failure
  and re-raises `APIError(str(err))`, which names only the host and endpoint path. A STAC search
  is a request **body**, so the collection, window, bbox and page are gone — and without them a
  refusal cannot be narrowed to a month or a page, reproduced, or reported upstream.
- **Our layer sits ABOVE a retry ladder, and only partly behind it.** For a force-listed status
  what escapes is the ladder reporting its own exhaustion, a much stronger statement than one
  error response; for a status kept out of that list (502) the first refusal arrives directly.
  `CatalogueRefusal.exhausted` records which.

So `_query_stac_items` pages explicitly (`pages_as_dicts`) and wraps **only the page fetch** in a
`CatalogueQueryError` carrying a `CatalogueRequest` — wrapping the page body too would classify
our own validation failures as someone else's outage. Opening the catalogue is page 0, named
separately so a root outage is not attributed to a window never asked for.

```text
CATALOGUE REFUSED collection=sentinel-2-l2a window=2021-09-01/2021-10-02
                  bbox=-3.0000,50.0000,-2.0000,51.0000 page 3
                  with HTTP 502 without being retried after 500 item(s)
                  — classified upstream-error:502
```

The **classification** separates refusals that arrive as one exception type from one endpoint
and need opposite responses:

| refusal | statuses | what it claims | response |
|---|---|---|---|
| `LOAD` | 429, 503 | the upstream names ITSELF as the constraint | wait, however often it recurs — an upstream naming its own load is the one refusal patience actually fixes |
| `UPSTREAM_ERROR` | 500, 502, 504 | the upstream failed to PRODUCE an answer | retry once; a repeat settles it as deterministic |
| `UNKNOWN` | anything else | no readable status | behave as the default does: retry |

A `LOAD` verdict draws the **expansive retry**: the leg-retry ladder's long, doubling delays,
granted without counting against the attempt budget for as long as the upstream keeps naming
itself. Everything else gets the ordinary attempt limit. So the two named sets must jointly cover
the ladder's `status_forcelist`, or a status the ladder retries but the taxonomy does not name
falls to `UNKNOWN` and keeps that expansive retry forever; a unit test asserts the containment.
The converse is deliberate: the taxonomy names 502, which the ladder does **not** retry, and a
second test pins that exclusion.

**The ladder it must cover.** `_query_stac_items` configures retries at the HTTP layer via a
custom `urllib3.Retry` built by `make_logging_retry()` (`_http.py`, shared with the CMR Granule
query) and passed into `StacApiIO` (`total=8, backoff_factor=2, status_forcelist=(429, 500, 503,
504)`). The subclass logs each attempt at WARNING — urllib3 otherwise retries silently inside the
`HTTPAdapter`, making a slow query indistinguishable from a hang. Because `search.items()`
paginates lazily each page fetch is a separate HTTP call, so retrying at the adapter recovers a
transient 5xx on page N in place instead of throwing away prior pages and restarting the whole
query. `StacApiIO`'s own default `max_retries=5` passes a bare int to urllib3, whose empty
`status_forcelist` means 5xx is **not** retried, so the explicit `Retry` object is required.

**Why 502 sits outside it.** An Earth Search page refusal then arrives unretried, and the
date-window re-cut described in
[Appendix A](#appendix-a--the-earth-search-response-cap-in-detail)
starts immediately rather than after the ladder's backoff; a transient 502 is absorbed by the
attempt budget owning the leg. The CMR Granule query keeps its own ladder
(`opera_query._CMR_RETRY`), 502 included, because nothing has measured a response-size cap there.

The status is read from the exception **chain**, not the message: `pystac_client` re-raises
without `from`, so the evidence sits under `__context__` on urllib3's exception. The message is a
documented fallback for a refusal that crossed a boundary carrying no chain.

**A status is necessary and not sufficient.** A gateway can fail for minutes and recover, so one
exhaustion is not proof of a defect. What settles it is a REPEAT — the identical request refused
the identical way on a later attempt — and that belongs to whoever holds the attempt budget,
`ingest_zone_year`'s leg loop: this module classifies, the budget holder supplies the repeat. The
two live in separate deployment runs, so the only thing crossing between them is failure text —
hence one whitespace-free token under a stable name (`CATALOGUE_REFUSAL=`), matched by name and
never by position, covering exactly the fields that decide the answer (collection, window, area,
page) and nothing that varies between attempts. A counter or timestamp inside it would make every
refusal unique and the repeat check dead code.

**Attempts are the only thing those budgets count; elapsed time has exactly one bound.** Each
page fetch gets 9 HTTP attempts across 364 s of backoff before anything above sees a failure, and
every budget above it — leg, cell, zone round — treats the layer below as one try. None reads a
clock, and expansive backoff makes the clock the axis that grows without limit.
`IngestSettings.max_leg_wall_clock_s` bounds it in the leg loop: once the deadline passes, the
loop refuses to START another attempt. A running leg is never measured against it, so the worst
case is the deadline plus one final attempt. Failing the cell this way costs latency, not work —
the cell returns to the work list and a later dispatch resumes from the dates already committed.
The derivation is in `context_docs/ingest/source-read-failures.md` (cause 3).

**Two things stop that bound refusing an attempt a leg had the budget for.**

The retry ladder DESCENDS rather than ending the retry. Backoff doubles per attempt, so the rung
an attempt has escalated to can be longer than the deadline has left even while a shorter rung
fits easily. The rungs beneath are the same policy applied one escalation earlier, so the loop
takes the longest rung that FITS and only a leg with no room for even the base rung is refused.
It does not cap the wait to the REMAINDER: waiting exactly what is left makes the next dispatch
land on the deadline every time, turning a race into a guarantee of the thing the deadline
forbids.

A leg that is still COMMITTING DATES earns more deadline, by
`IngestSettings.leg_progress_extension_s`, because a deadline counted from the first dispatch
charges a leg for every prior attempt's productive work and cannot tell a pathological cell from
one working steadily. Progress is read from the leg's own child store through the same
`get_existing_dates` the ingest resumes from, so parent and leg cannot disagree, and a store that
cannot be read earns nothing. A grant is a FIXED size and each must be PAID FOR by dates
committed since the previous grant, which also limits the rate: every ask sits after a failed
attempt and the asks within one attempt compete for the same growth, so at most one is paid. The
ceiling is `max_leg_wall_clock_s + (max_leg_attempts - 1) * leg_progress_extension_s`; a leg that
commits nothing never leaves `max_leg_wall_clock_s`, and an extension of 0 restores the plain
deadline.

`source_coverage.py`'s preflight probe deliberately does **not** use any of this. Every failure
of that probe is already INCONCLUSIVE by design, which is the right answer for both refusals at
once, so telling them apart would buy nothing.

#### When the archive says "success" but sends something that is not JSON

Asking the archive for a page of radar granules normally returns a success code and a JSON
document. Occasionally it returns a success code and a body that is not JSON at all — an error
page, or a document cut off partway through.

**This slips past every defence we have.** Everything that decides whether to retry a request
looks at the response's status code, and here the status code is fine: it says success, and by the
only measure those checks apply it *was* a success. Only the body is wrong, and nothing inspects
the body. So the request sails through the retry logic untouched and fails later, when something
tries to read it as JSON, with a message that says only:

```
Expecting value: line 1 column 1 (char 0)
```

That line names no address, no status, nothing about what arrived, and not even which of the
several services we query was the one that broke. In production it ended a radar run that had been
working for half an hour.

**So the page is simply asked for again**, a small fixed number of times. This is deliberately
narrow: every other kind of failure is left exactly as it was, and a server error is not re-asked
here, because the ordinary retry logic has already waited and tried for that one. The re-ask uses
the position marker the archive itself gave us, so it asks for the same page rather than the next
one and cannot step over granules.

If the retries are used up, the failure now describes itself: which address answered, what status
it gave, what kind of document it claimed to be sending, and how big it was. A document claiming
to be JSON alongside a body that will not parse means it was cut off; one claiming to be a web
page means an error page was substituted.

**The body that arrived is written to the log, and deliberately kept out of the error message.**
Whether to run a failed leg again is decided by searching the failure's text for certain words,
and the body is text the provider chose rather than us — so an error page containing one of those
words could flip a leg that should have been retried into one treated as permanently dead, costing
a whole zone-year. In the log it is just as readable and steers nothing.

**The credential requests never log their body at all.** They use the same helper, because they can
fail the same way, but with the body capture switched off: a credential document cut off partway
through is precisely the one that fails to parse, and its opening characters are the credential.

### A read would not produce pixels

This is where the cost asymmetry bites, and where most of the machinery is. The sections below
follow one failure outwards: where the retry sits, how a corrupt object is told apart from a
provider having a bad hour, what to do when GDAL declines to say which it was, and what the radar
path does differently because it has no second copy to fall back on.

#### GDAL network tuning

`configure_gdal_environment()` (in [`config/environment.py`](../config/environment.py)) must be
called before importing `rasterio` or `odc.stac`. It sets GDAL config options for network
resilience (retry counts, timeouts, connection pooling) that affect all subsequent COG reads.

#### Where the retry sits, and how a failed date is attributed

`roi_processing.source_read_retrying` wraps the point where a date's graph is first *computed*.
S1's read happens inside its write's `compute()` and is already covered by the write retry; S2's
fires earlier, in its coverage gate. Scoped per date deliberately: a task-level retry would re-run
the whole multi-day loop, so `tasks/ingest.py` refuses `@task(retries=...)`. Unlike the write
policy it is **not** narrowed by exception type — reads fail through rasterio, GDAL/CPL, botocore
and bare socket timeouts, a read is idempotent, and enumerating those surfaces risks a new
transient class becoming fatal.

**A failed date must say which date, and on which ROI.** Per-date telemetry is emitted *after* a
date commits, so the furthest date in a log is the last one that WORKED and a failure otherwise
leaves no trace. `roi_processing.read_failure_context` emits `READ FAILED roi=… date=… items=…
first=…` with the traceback on both sensors' per-date paths. `roi=` is what makes it attributable:
the exception is raised on a Dask worker whose log stream is an ECS task id, so without it the
same text appears for every zone and belongs to none. The traceback recovers rasterio's cause,
which reports only `Read failed. See previous exception for details.` — GDAL's actual reason is
discarded unless the chain is logged. It is also where the reason GDAL never raised is attached;
see *When GDAL logs the reason instead of raising it*.

#### When a source object will not read

Some published objects are corrupt: a tile of the COG will not inflate, and no retry of any
length recovers it. That is a different condition from a throttle or an expired credential,
which look similar coming out of the loader — `rasterio` wraps both in a
`WarpOperationError` that discards the cause — so `is_unreadable_source` inspects the whole
exception chain and matches only the codec-level signatures, excluding the credential and
throttle markers explicitly. It fails CLOSED: anything unrecognised propagates rather than
being treated as bad data, because responding to a bad minute by reading worse imagery is
the one outcome the recovery must never produce.

**An intact chain is still only what the reader chose to RAISE.** GDAL states some refusals in
its own log and raises something else entirely, and the section *When GDAL logs the reason instead
of raising it* below is what closes that.

**The chain only exists if something kept it**, and a leg **refuses to start** unless every
worker confirms it can: a job that cannot explain its own failures can quietly ruin a dataset. The
read fails on a Dask worker, and rasterio's GDAL error classes cannot be serialised out of one by
default — Dask substitutes a plain `Exception` holding the wrapper's repr, so what arrives is one
line with no cause and every predicate here has nothing to read.
`loader_failures.keep_causes_picklable`, installed on every worker by the same plugin as the
object capture, is what makes the cause arrive. It is best effort, so `cause_was_flattened`
recognises a failure that arrived without one and `read_failure_context` logs `READ CAUSE LOST`.

**An object that was never published counts too, and needs its own markers.** Every
codec-level signature comes from a BLOCK READ, and a missing object fails at open before any
block is requested. `ObjectNotFound`, `NoSuchKey` and `The specified key does not exist` cover
the three layers that surface it, matched only alongside the source reader's own vocabulary
(`RasterioIOError`, `WarpOperationError`, `CPLE_`, `HTTP response code:`). That pairing is what
makes them mean SOURCE: those strings belong to the S3 layer that every S3 client in the process
shares — `icechunk`'s error enum carries two verbatim — so unpaired they would record a hole in
the destination store as provider data loss. `NoSuchBucket` is deliberately excluded: a vanished
bucket is systemic and must fail the leg on its first date.

Nothing counts or caps these skips on the OPTICAL path: they are rare enough per granule that a
ceiling would only fire on a fault of another kind, and every date given up is logged, restated
in the end-of-run summary, and written to the store. The radar skip below does carry a ceiling,
because it answers a provider refusal, which arrives fleet-wide and all at once.

Past that point the response is a ladder, in `s2_roi.py`'s consume path:

1. **Attribute.** Ask the cluster which objects the loader gave up on
   (`loader_failures.collect_aborted_hrefs`) and map them back to tile-dates.
2. **Step down** those tile-dates to their next catalogue copy and re-prepare the date. The
   copy is older reprocessing, so this trades processing baseline for a date that reads.
3. **Give up, loudly,** when the implicated tile-dates have no copies left: the date is
   skipped rather than the leg failed, and a `DATA LOSS` line names the date, the objects and
   the scope. Nothing is written to the store — see *Why nothing records what was missed*.

Two properties are what the attribution step buys, and tests hold them rather than comments.
**Blast radius**: with attribution one bad object steps one tile-date, where without it every
duplicated tile-date steps together, downgrading hundreds of tiles on a wide ROI that read
perfectly well. **Termination**: a bad object whose tile-date has no alternate is given up
immediately, where without attribution the ladder first walks every *other* tile's alternates, at
a full re-read of the date per rung, to reach the same answer.

Attribution can fail — a worker that died with the read, a cluster already gone, a loader that
words its message differently — and the unattributed behaviour above is then the fallback. The
record says which happened: `scope=attributed` means the named objects are the ones that failed,
`scope=whole-date` means the failing object was not identified and the tiles listed are every tile
in the date.

The batched write path cannot reach the ladder — a batch is one graph and one commit — so it
isolates first: an unreadable source anywhere in a batch re-runs the batch's dates one at a
time, each then getting the per-date recovery. That isolation is what stops one corrupt
object from failing a zone-year identically on every retry.

#### When the provider refuses the read

An authorization refusal, a throttle and a server error are a different finding again. They say
nothing about the imagery — the same object read minutes earlier and reads again once the service
recovers — so no fallback copy helps and no date should be given up for one. That verdict is
reached by the same `is_unreadable_source` the section above describes: it answers one question
for both cases, returning true for codec-level damage and **false** for a refusal, which it tests
for first. There is no second predicate to ask.

**What a positive verdict buys is TIME, and nothing else.** It is passed to the shared write retry
as `wait_out`, and the policy re-attempts that one failure until it has spent `WAIT_OUT_BACKOFF_S`
of backoff. It is never spent on giving up a date, for the append-only reason above: a date
abandoned now cannot be written later. If the wait is not enough the write fails, the leg fails
with its time axis unmoved, and the leg's own retry re-offers the date in order.

The in-leg budget is `WAIT_OUT_BACKOFF_S` and the between-attempt one is
`leg_refusal_backoff_s`, per the two budgets above.

Carrying the verdict between the two takes a type: the leg-retry layer sees only a failure DETAIL
string, and no marker on it can separate a refused read from a crash, since the wrapper discarded
the cause. So a radar write that exhausts its in-leg budget on a refusal raises
`errors.ProviderRefusedReadsError`, whose name reaches the detail and is what `_leg_backoff_s`
keys the long delay on. Nothing else about the failure changes.

#### When GDAL logs the reason instead of raising it

Everything above reads the exception chain. Some of a read failure's reason never reaches it.

A refused object is not empty: S3 answers the range request with an XML error document, and GDAL
hands it to the TIFF decompressor, which fails on it — `ZIPDecode: Decoding error at scanline 0`,
sometimes `unknown compression method`. That is what gets raised. GDAL states the refusal as a
warning in its own log and raises nothing about it. So the chain says the bytes are bad and the
log says the service refused, and those verdicts are opposites: bad bytes gives the date up,
refused waits and gives up nothing.

`loader_failures` closes it with a second handler on `rasterio._env`, the logger rasterio's CPL
error handler forwards to. `carry_logged_refusal` attaches what it collected to the failing
exception as a note, `_exception_chain_text` reads notes with the rest of the chain, and
`classify_read_failure` decides from all of it. Both sensors reach this through
`roi_processing.read_failure_context`, so it is one classifier over one set of evidence.

That handler only hears what reaches a logger, and most of it does not: rasterio installs its CPL
handler with `CPLPushErrorHandler`, which GDAL keeps **per thread**, so a message from one of
GDAL's own fetch threads goes to the process-wide handler and out to stderr, where no
`logging.Handler` can reach it. `hear_gdal_from_every_thread` gives that process-wide handler
somewhere to forward to, and `install_capture` installs it alongside the two log handlers. It
chains to the existing handler rather than replacing it, so GDAL's stderr line still appears and a
fatal error still aborts through it, and GDAL consults the reporting thread's own handler first so
nothing rasterio already forwards is duplicated.

Four properties are what make it safe to add evidence at all:

- **Only refusals are recorded.** A line is kept only if the classifier reads that line alone as
  `PROVIDER_REFUSED` or `OUR_CREDENTIAL`, so there is no second vocabulary to drift. Everything
  else GDAL says is dropped, which matters because GDAL probes for sidecars that were never
  published — a kept `HTTP response code: 404` is the marker for an ABSENT source object, and
  gives a date up.
- **The direction is bounded.** Refusal is tested before any statement about the bytes, so an
  attached line can only move a verdict into those two, never into `UNREADABLE` or `ABSENT`. A
  wrong attribution therefore costs patience rather than a date: the write spends its refusal
  budget, the leg fails with its time axis unmoved, and the date is judged alone on re-dispatch.
- **Only onto a source read failure**, gated by `is_source_read_failure` on the same
  `_SOURCE_READER_MARKERS` every other corroboration uses. A store conflict raised while some
  other read is being refused is still a store conflict, and a wait fixes nothing.
- **Reading the refusals does not consume them**, and they sit in a separate buffer from the
  aborted hrefs, which is still drained destructively. Two reads are in flight whenever the
  optical path pipelines a date, each inside its own `read_failure_context`, and a destructive
  collection let whichever failed first take the other's evidence. What keeps a stale line out is
  therefore its AGE: a read that logs a refusal and then succeeds drains nothing, so without an
  age bound its line would be inherited by whatever failed next and a genuinely corrupt object
  would read as a refusal and never step down the copy ladder. Each worker reports its lines' age
  by its own clock and the caller applies the cutoff **after** the round trip against its own, so
  nothing depends on the fleet's clocks agreeing or on the collection being quick.

The evidence is ATTACHED rather than answered: a caller that classified and discarded would hand
the next reader of the same exception the opposite verdict, and the radar path has two readers —
the retry policy that spends patience, and the handler that decides whether the date is lost.

Radar asks for it twice, and the second ask is what arms `wait_out`. `refusal_wait_out(client)` is
`is_provider_refusal` over evidence gathered at the moment the policy asks; a predicate reading
only the exception declines the outage the budget exists to outlast and spends the ordinary three
attempts on it.

**The evidence window is the WRITE, not the attempt.** The policy asks once per failed attempt,
but what it asks is whether this WRITE is being refused, and an outage states its refusal while it
is refusing rather than on the ladder's schedule. Re-armed per attempt, the question becomes "was
anything refused in the last few seconds" — which a recovering provider answers NO one attempt
before the write would have succeeded, withdrawing patience exactly when it was about to pay off.
One window per write also makes the two readings of one failure agree, since
`read_failure_context` judges the same failure over the whole write when the retry gives up.
Nothing is remembered: the window is derived, so evidence is re-read on every attempt, a write
whose window holds no refusal never waits, and `WAIT_OUT_BACKOFF_S` bounds one that keeps seeing
one. Each ask logs its verdict and how many refusal lines it read, because an attempt count cannot
separate "no refusal was logged" from "a refusal was logged and not read".

Optical does not pass `wait_out` at all: its answer to a refusal is a leg failure with the axis
unmoved, because its per-date remedy is the copy ladder and a long wait per rung would multiply
with it. The evidence still reaches optical through the same context manager, so a refusal there
is declined by `is_unreadable_source` and fails the leg rather than stepping copies.

The prior-success guard lives here too: `s1_roi` keeps one per-leg flag, set on the first
committed date, and the long wait is withheld until it is set.

Three shapes are excluded, each costing only the ordinary attempt limit. A credential fault on
THIS side, being repairable here. A refusal nothing attributes to the source reader —
`AccessDenied`, `SlowDown` and `InternalError` are S3's words, so they count only alongside
GDAL's own vocabulary, by the same pairing rule as the not-found markers above. And a refusal
carrying neither a name nor a status: a transport failure with no code, or a cause destroyed
crossing the worker boundary. `cause_was_flattened` says which of those last two happened, so a
leg reading without a decidable cause is visible rather than silent.

The two predicates are disjoint by construction, and stay so by sharing one classification rather
than keeping two lists in step: both read the same markers and the same HTTP status RANGES, so a
status nobody enumerated cannot be a refusal to one predicate and bad data to the other, and a
caller that knows only one of them cannot misclassify.

#### The radar bounded skip (`s1_roi.py`)

Every OPERA read on the radar path happens inside a date's write, so a failed read raises out of
the per-date loop. Until this skip, one refused read cost every LATER date in the window too: a
source refusing reads for thirteen minutes emptied 178 zone-years that had already committed
months of sound data.

The radar response is the tail of the optical one without the copy ladder, which radar has no use
for: OPERA publishes one copy of a granule, so there is nothing to step down to.

1. **Retry**, through the shared `store_write_retrying` policy — and for a provider refusal that
   arrived after a successful read, retry past the attempt limit, because waiting is the only
   response a refusal has. Radar is the one caller that asks for this.
2. **Fail the leg under a name the cell can act on** if that wait was not enough
   (`ProviderRefusedReadsError`), so the re-dispatch waits on the long schedule. No date is
   skipped and the time axis does not move.
3. **Give up the date** once that retry is exhausted, if and only if the failure is one the source
   is answerable for AND recomputes. One scope, `unreadable`, one remedy: a reprocessed copy at
   the provider. A refusal is deliberately not a second recoverable scope — giving up a date and
   then committing a later one puts the earlier one permanently below the append-only maximum, so
   the re-run meant to recover it is refused instead.
4. **Name it in the log**, per date and again in an end-of-leg summary. Nothing durable: the day
   is below the store's newest date by the time the next date commits.
5. **Stop past `MAX_GIVEN_UP_DATES`**, and stopping is TERMINAL.
   `TooManyGivenUpDatesError` is in the leg-retry classifier's non-retryable set, because nothing
   counted toward the ceiling can clear: every date reaching that counter failed for a cause that
   recomputes, so a re-dispatch would re-read the same objects to reach the identical answer.

A date offered by two consecutive batches is given up ONCE. Batch queries are padded a day either
side, so a boundary solar day comes back from two queries and would otherwise be listed twice and
cost twice.

### What a resumed run does about it

A leg that failed is dispatched again, and what it does next is decided by the store rather than
by anything the failed run recorded. This is the constraint the rest of the section is written
around.

#### Where a resumed run starts

A store's dates can only be added in order, newest last. Slotting one into the middle would mean
shifting every chunk after it, and a Zarr store's chunks sit at fixed positions — there is nowhere
to shift them to. So **every day at or before the newest date a store already holds is closed to
that store for good**, whatever the imagery for that day later turns out to be.

Most runs are resumes: a leg dispatched for a calendar year fails part way through and is
dispatched again, so everything below the line it reached is settled and only the days above it
are open. **A run therefore starts the day after the newest date its own store holds**
(`resume_window_start` in `solar_days.py`). Three questions, asked before the catalogue is
queried and before any date is prepared:

1. **Does the window end before it begins?** That is a caller mistake and always was, so the run
   refuses it. Asked first, so a misconfigured leg can never be reported as a successful skip.
2. **Is the store's newest date already at or past the window's last day?** Then nothing in this
   window is open to it. The run reports a skip and stops without querying anything. The
   comparison is against the raw date, not against anything derived from it: a value floored to
   its month first can precede the window's end while the date it came from does not.
3. **Otherwise, begin the day after that newest date.**

**The day after, rather than the first of its month.** Starting at the month boundary still
offers the earlier days of that month, which is exactly where an old gap sits — a day an earlier
attempt could not write while later days landed above it. Offering one to the writer is fatal: the
append is refused, the leg dies, and it dies again on every retry because the imagery is the same.
A drop-and-log guard sits in front of the writer too; it should never fire, and exists because the
failure it prevents leaves a store with no remedy but deletion.

Nothing is lost by the tighter start. What a run may *write* is the span it **owns**, and every
catalogue query is padded a day either side, because a solar day's imagery can carry the adjacent
UTC date (see *Timestamp handling*). One day is provably enough, since a solar offset
is a whole number of hours within ±12.

Each store works this out for itself: a cell has up to three, and they advance at different rates,
so a shared start would skip days a lagging store never reached. The saving matters as much as the
safety — searching below the line cannot write anything, and searching is most of what a resumed
run does.

#### Why nothing records what was missed

Once a day is closed, what happened on it stops mattering: an image that would not read this
morning and reads this afternoon still cannot be written. Readability can change; the outcome
cannot. So there is no ledger of missed days — it would unlock no action, would have to stay in
step with the store, and would be deleted along with the mosaic it was written on.

**The published product already answers the question a reader actually has.** A mosaic is an
intermediate, deleted once embeddings are computed from it. What survives carries per-pixel
coverage layers: `s2_obs_count`, `s1_asc_obs_count` and `s1_desc_obs_count` count usable
observations, and `s2_month_covered`, `s1_asc_month_covered` and `s1_desc_month_covered` give one
boolean per pixel per month (`config/store_layout.py`). "Does this pixel have data for August" is
answered by the published data rather than by a note attached to something deleted. And downstream
every absence is the same absence: a day the satellite did not pass over, a day too cloudy to
keep, and a day whose files would not read all put no pixel in the mosaic, and nothing consuming a
mosaic tells them apart.

There used to be a ledger: `assessed_unreadable_dates` named every day a leg gave up on, and the
coverage gate subtracted the months holding those days from the months an assessed window excuses.
That subtraction refuses a month that can never be filled — nothing can be written below the line
— so it deadlocked the cell rather than protecting anything, and the only way out was to delete
the store, which is a judgement a person makes from an audit.

What remains on the store is `assessed_window`, which is not a loss record. It says which range a
leg examined, so a month holding no dates reads as "we looked and there was nothing" rather than
as "no run reached this month". It works at month granularity and unblocks a cell rather than
blocking one. `assessed_empty_dates` sits beside it as a count, for observability only.

A lost day still produces a `DATA LOSS` line naming the date, the cause and the objects, and a
summary at the end of the leg. What it does not produce is a record anything later reads.

---

## Performance Optimizations

One thing bounds an ingest, and it is not the imagery. The Dask scheduler is a single process
that expands the whole task graph into memory before a worker reads a byte, and dispatches every
task through one event loop — so a graph big enough to describe a zone-year exhausts it, and a
graph small enough to fit can still leave the fleet idle. Every lever below is one of two moves:
**keep the graph small**, or **keep the fleet busy while it runs**. They pull against each other,
which is why several of them are priced rather than switched on.

### What bounds the run

Three facts a reader needs before any of the levers make sense: what the scheduler actually
spends memory on, how the chunk grids line up, and what the write contract is that every lever
has to preserve.

#### Background: how Dask task graphs consume scheduler RAM

"Dask is lazy" means workers do not read data until `.compute()`; it does not mean the
scheduler avoids work. Before the first worker task executes, `dask.distributed` expands the full
`HighLevelGraph` into a flat dictionary of `TaskState` objects in the scheduler process, each
holding one task's function, arguments and dependency set. The cost:

```text
    scheduler RAM used ≈ n_tasks × 1.5 KB
```

This is fully predictable and independent of data size. A graph with 1 million tasks consumes
~1.5 GB of scheduler RAM before any worker reads a single byte. Nothing is read until the Zarr
write triggers the compute, so a date's whole pipeline — load, correct, mask, write — is one
graph execution and the scheduler holds all of it at once.

The HLG itself is compact — it stores *layer dicts* rather than expanded objects. The
expansion to TaskStates happens only when the graph is submitted to the scheduler:

```text
  Python process (builds the HLG)          Dask distributed scheduler process
  ─────────────────────────────────         ─────────────────────────────────────
  Layer "zarr_read"                         TaskState("zarr_read",(t=0,y= 0,x= 0))
    { (t=0,y=0,x=0): read_fn, ... }         TaskState("zarr_read",(t=0,y= 0,x= 1))
  Layer "baseline_corr"                     TaskState("zarr_read",(t=0,y= 0,x= 2))
    { (t=0,y=0,x=0): corr_fn, ... }         ...
  Layer "roi_mask"                          TaskState("baseline_corr",(t=0,y=0,x=0))
    { (t=0,y=0,x=0): mask_fn, ... }         ...
  Layer "zarr_write"
    { (t=0,y=0,x=0): write_fn, ... }       one TaskState object per task,
                                            all held in scheduler RAM simultaneously
  4 compact Python dicts ≈ tens of MB      n_tasks × 1.5 KB of scheduler RAM
```

##### How task count multiplies: operations × chunk dimensions

Each Dask operation (read, transform, write) adds a new layer. Each layer has one task per
combination of *chunk coordinates* across all chunked dimensions. Ingest writes 4096×4096 px
storage chunks (`INGEST_CHUNK_SIZE`), deliberately larger than the 2048×2048 inference
read-tile size, precisely to keep this task count down. For S2 ingestion over a large ROI
(e.g. cornbelt scale: ~38×25 grid of 4096×4096 px spatial chunks), the S2 flow processes one
date at a time, so the time dimension is always 1:

```text
  Dimensions per single S2 date (ingest_s2_roi_reflectance):
    spatial chunks:     ~950  (38×25 grid of 4096×4096 px)
    dates:                 1  (one date per loop iteration)
    band variables:       10  (each S2 band is a separate xarray variable)

  Operation            tasks per layer                          notes
  ────────────────────────────────────────────────────────────────────────
  odc.stac read       950 × 1 × 10 =   9,500   one task per (chunk, band var)
  baseline corr.      950 × 1 × 10 =   9,500
  ROI mask            950 × 1 × 10 =   9,500
  zarr write          950 × 1 × 10 =   9,500
                                    ─────────
        per-date total:                38,000 tasks ≈ 0.06 GB scheduler RAM   ✓

  Full year (all ~100 S2 dates in one graph, hypothetically):
  950 × 100 × 10 × 4 = 3,800,000 tasks ≈ 5.6 GB scheduler RAM                ✗ OOM
```

Had ingest reused the 2048×2048 inference tile size, every count above would be 4× higher —
that 4× on the satellite-ingest graph is the reason storage and inference chunk sizes are
decoupled.

The two flows keep task count bounded via different mechanisms — per-date iteration for S2,
time-windowed batching for S1 — described in the sections below. The same scheduler-RAM
discipline reappears in inference assembly; see
[`inference/README.md`](../inference/README.md#1-deciding-which-tiles-to-run) for the
ChunkSpec-vs-sub-chunk decoupling that makes assembly survive on the same budget.

#### Chunk alignment

The ROI Zarr mask is generated with `chunk_size` matching `INGEST_CHUNKS` so that
`da.from_zarr` reads are zero-copy — each Dask partition maps to exactly one Zarr chunk.
The same chunk sizes are passed to `odc.stac.load` (after translating `northing`/`easting`
to `y`/`x`) so band arrays and the mask share the same partition boundaries for aligned
Dask operations. (Inference reads 2048×2048 sub-tiles out of these 4096×4096 chunks via
`zarr.Array.oindex`, which needs no such alignment — see
[`inference/README.md`](../inference/README.md).)

#### Writing a date: one session, one commit

`storage.write_day_windows` owns it. A missing store is seeded all-fill (schema only — creation
cost independent of extent), then each date appends its time slot atomically WITH its windows in
**one commit**:

```text
per passing date (one writable session ── one commit)
   ├─ append time slot            (metadata-only resize; duplicate date = loud error)
   ├─ to_icechunk(region=window₁) ┐  pixels flow from the Dask workers that
   ├─ to_icechunk(region=window₂) │  computed them — never materialised on
   ├─ ...                         ┘  the flow runner
   ├─ merge attrs                 (baselines ∪, doy ++, last_appended)
   └─ commit                      (crash before here ⇒ nothing visible; retry is clean)
```

The empty-axis seed matters: the time axis only ever contains dates whose pixels
committed, keeping `get_existing_dates` (the STAC dedupe),
`check_time_window_coverage`, and the empty-timestep prunes truthful.
**The retry must not retry a second writer** — the one exception to "a failed write commits
nothing, so retrying is safe". One store has exactly one writer: these commits pass no
`rebase_with`, so a concurrent commit is *refused* rather than merged
(`icechunk.ConflictError`), and a date the other writer reached first is refused by the append
guard (`DuplicateDateError`). Retrying would re-open the session from the tip that writer
moved, turning the refusal into a success and letting two writers interleave dates onto one
axis. Both errors are excluded by type in `storage.zarr_store.store_write_retrying`, the
single policy all three write sites use (S1 per-date, S2 per-date, S2 per-batch).

### Keeping the graph small

The two flows start from different baselines, and the two cropping steps then cut what either
of them has to build.

#### S2: per-date iteration

`ingest_s2_roi_reflectance` queries STAC for the full date range upfront, groups items by **local
solar day** via `group_items_by_date`, then processes one day at a time in a Python loop: each
iteration builds a single-date Dask graph, calls `odc.stac.load` for that day, filters coverage,
and writes before moving on.

```text
Full year (don't build at once):
┌──────────────────────────── 365 days ────────────────────────────────┐
│ tiles × dates × bands = O(millions of tasks) → scheduler OOM         │
└──────────────────────────────────────────────────────────────────────┘

Per-date iteration (what ingest_s2_roi_reflectance actually does):
 2024-03-01    2024-03-06    2024-03-11    ...
┌────────────┐ ┌────────────┐ ┌────────────┐
│ build      │ │ build      │ │ build      │
│ SCL check  │ │ SCL check  │ │ SCL check  │
│ compute    │ │ compute    │ │ compute    │
│ write      │ │ write      │ │ write      │
│ discard ◄──┼─┼── graph freed after each date
└────────────┘ └────────────┘ └────────────┘
```

Each single-date graph is small: `spatial_chunks × bands` tasks, with no date dimension to
multiply through, and the per-date overhead of one Python loop iteration and one Zarr append is
negligible beside the Dask compute for a large spatial ROI.

The grouping key must match the loader's, per *Timestamp handling* above. Grouping
here by UTC calendar date lets the two disagree, and a group we believe is one day then loads as
TWO time slices against a cloud mask reduced to one:

```text
   UTC:      ... 23:00 | 00:00  01:00 ...      ONE UTC date
   solar:        day N |  day N+1              TWO solar days   (at a +10 h offset)
                       ^ far-eastern zones image right here
```

#### S1: time-windowed batching

`ingest_s1_roi_sar` uses a different approach: it splits the full date range into
`batch_days`-wide windows (default 30) and runs one `build → compute → write → discard`
cycle per window, which bounds how large any one task graph gets.

```text
Batched approach (batch_days=30, ingest_s1_roi_sar only):
 Jan 1–30        Feb 1–28        Mar 1–30       ...
┌────────────┐  ┌────────────┐  ┌────────────┐
│ build      │  │ build      │  │ build      │
│ compute    │  │ compute    │  │ compute    │
│ write      │  │ write      │  │ write      │
│ discard ◄──┼──┼── graph freed
└────────────┘  └────────────┘  └────────────┘
```

A batch boundary is **not** a credential checkpoint, and treating it as one is unsafe: the STS
credential's roughly one-hour life is unrelated to how long a batch takes, so a batch that outruns
it cannot renew at its own boundary. Renewal is owned by a timer — see "Renewal runs on a timer"
above.

`batch_days` is a parameter on `ingest_s1_roi_sar` and is absent from the S2 flow. The formula
in the background section above estimates how many tasks a given window width produces; the
30-day default keeps each batch inside the scheduler's RAM budget at cornbelt scale.

Batch windows are inclusive at both ends and do not overlap: each spans `batch_days` calendar
days and the loop advances `batch_start` to the day *after* `batch_end`. Since CMR and STAC also
treat their end date as inclusive, each day is queried by exactly one batch — a boundary landing
on the next batch's start day would page it twice, wasteful where each day is many pages of
bursts.

#### Cropping to live windows (unconditional)

Ingest cost scales with the **extent it computes, not the land it keeps**, and a mosaic load
covers the whole ROI grid even where the mask is entirely ocean or out of footprint. So every
load and write is restricted to the chunk-aligned windows that intersect the ROI mask,
unconditionally and with no flag: on a zone where land is 0.238% of the extent, uncropped is
~420× the array volume, which exhausts a worker's disk. Across the campaign's land zones roughly
three-quarters of the compute would otherwise go to ocean.

```text
zone / ROI extent (declared grid — UNCHANGED, the fill validates it)
┌──────────────────────────────────────────────┐
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │   · ocean / out-of-footprint:
│ · · · ┌────────────────────┐ ·  ·  ·  ·  ·   │     never loaded, never computed,
│ · · · │████████████████████│ ·  ·  ·  ·  ·   │     never written (reads back as
│ · · · └────────────────────┘ ·  ·  ·  ·  ·   │     fill — Zarr elides all-fill
│ · · · · · · ┌────────┐ ·  ·  ·  ·  ·  ·  ·   │     chunks anyway)
│ · · · · · · │████████│ ·  ·  ·  ·  ·  ·  ·   │
│ · · · · · · └────────┘ ·  ·  ·  ·  ·  ·  ·   │   █ live windows: chunk-aligned,
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │     4096 px, derived below
└──────────────────────────────────────────────┘
```

Windows are derived in **two stages**, and the second is where most of the win is. Each cell
below is one 4096-px ingest chunk; `#` is live, `.` is ocean.

**Stage 1 — row bands.** One window per live chunk-row, spanning that row's first to last live
column. A row's interior gaps are included; nothing above or below it is. This is the
minimum-*area* answer.

**Stage 2 — grouping.** Vertically adjacent bands are unioned into taller windows. That
computes some dead chunks and is still a large net win, because the two costs differ:

```text
  chunk area                          a window boundary
  ──────────                          ─────────────────
  computed in PARALLEL across the     a BLOCKING region write: the whole
  whole fleet — adding chunks         fleet waits while one completes.
  widens the graph, which is           N windows = N serial stalls.
  what the fleet wants.
```

So the objective is `least n_windows × price + area`, where the price is one window expressed
as the chunk area costing the same (`WINDOW_COST_IN_CHUNKS`). That price is large, so grouping
pays whenever it is geometrically sane.

The merge runs **twice** — once over the run's live grid, once over each date's narrowed grid —
and both must use the same price, so `windows_for_date` takes it as a parameter. Priced
at the sequential default while the run used the overlapped rate, the per-date re-merge would buy
dead area back to save boundaries the write path had already made cheap.

```text
     live chunk grid        stage 1: row bands        stage 2: grouped
     c0 c1 c2 c3 c4 c5      c0 c1 c2 c3 c4 c5         c0 c1 c2 c3 c4 c5
r0    .  #  #  #  .  .       .  A  A  A  .  .          .  a  a  a  +  .
r1    .  #  #  #  #  .       .  B  B  B  B  .          .  a  a  a  a  .
r2    .  #  #  #  #  .       .  C  C  C  C  .          .  a  a  a  a  .
r3    .  .  #  #  #  .       .  .  D  D  D  .          .  +  a  a  a  .
r4    .  #  #  #  #  .       .  E  E  E  E  .          .  a  a  a  a  .
r5    .  .  .  .  .  .       .  .  .  .  .  .          .  .  .  .  .  .
r6    #  #  .  .  .  .       F  F  .  .  .  .          b  b  .  .  .  .
r7    #  #  .  .  .  .       G  G  .  .  .  .          b  b  .  .  .  .
r8    .  .  .  .  .  .       .  .  .  .  .  .          .  .  .  .  .  .

     # live chunk          7 windows, 7 stalls       2 windows, 2 stalls
     . never written       A..G one per live row     a = r0-r4 x c1-c4
                           r5/r8 have no live        b = r6-r7 x c0-c1
                           chunk, so no window       + = dead chunk the
                                                         union pulled in
```

Group `a` covers r0–r4 at the cost of two dead chunks (`+`), buying four fewer stalls. Whether
`b` joins it is the same question, and at the production price it would — real masks group harder
than this illustration suggests. Two things stop a group: the price ceasing to justify the area,
and `MAX_TASKS_PER_WINDOW`, which bounds one window's graph. That cap is expressed in TASKS and
converted to a chunk area through `DEFAULT_TASKS_PER_CHUNK`, because the scheduler dispatches on
a single-threaded event loop — past its throughput, extra area stops being cheap.

Grouping is solved **exactly**, by a dynamic program over consecutive bands rather than a greedy
rule: a heuristic bound on wasted area cannot express "extra area is nearly free", so it
under-merges precisely on the sparse ROIs where the waste is trivial. Windows stay chunk-aligned
and mutually chunk-disjoint either way, letting one session write a whole date and commit once.

Across the campaign's zones this lands every one in 2–5 windows, saving between 2× and 66× of
the stage-1 write boundaries for between 0% and 50% added area, and cutting predicted per-date
ingest cost by about **11×** summed over all land zones. Sparse ROIs group *harder* in relative
terms, which is the point: a large fraction of a tiny area is still a tiny area. Calibration of
the price, the per-zone table and the cap sweep are in
`context_docs/ingest/ingest-performance.md` §13.

- **Windows** come from `live_windows.py`, from the boolean ROI mask that both
  `rasterize_roi_zarr` and `export_zone_roi` write. The mask is coarsened to the ingest chunk
  grid — normally from its chunk keys in one listing, else by scanning one chunk block at a
  time (~16 MB peak, no Dask) — then row-banded and grouped as above.
- **Writes** go through `storage.write_day_windows`, one commit per date — see
  *Writing a date: one session, one commit* above.
- **Reads retry, per date**, and a failed date says which date and which ROI. See
  [§ Where the retry sits, and how a failed date is attributed](#where-the-retry-sits-and-how-a-failed-date-is-attributed).
- **Each date narrows further, to the land its own imagery reaches**, via `windows_for_date`.
  See *Narrowing a date's windows, and skipping dates that reach none*.

```text
   run windows (where the ROI has land)   one date's items      that date writes
     c0 c1 c2 c3 c4                        c0 c1 c2 c3 c4        c0 c1 c2 c3 c4
r0    a  a  a  a  .                         .  F  F  .  .    r0   .  a  a  .  .
r1    a  a  a  a  .          ∩              .  F  F  F  .  = r1   .  a  a  a  .
r2    a  a  a  a  .                         .  .  .  .  .    r2   .  .  .  .  .
r3    b  b  .  .  .                         .  .  .  .  .    r3   .  .  .  .  .

     2 windows, every date                F = the swath           1 window; window b
                                            this date covers        is skipped entirely
```

- **The declared grid stays full-extent**, so the zone fill's exact-grid validation is
  unaffected — Zarr/Icechunk arrays are sparse and unwritten chunks read back as fill. That
  validation checks the declared grid COMPLETELY rather than just its corners, because matching
  length, CRS and endpoints still admit a reordered or non-affine interior, and inference writes
  positionally, so such a mosaic would publish real pixels at wrong coordinates silently. Uniform
  10 m spacing is asserted on both axes. Nothing this ingest produces can fail it, since odc
  builds every load against the zone geobox, but a fill run with `ingest=False` accepts a mosaic
  staged by hand.
- **The SCL coverage phase is cropped too**: its reduce runs over the windows only
  (identical total — the mask is False outside them) and the validity mask stays lazy,
  so no full-extent array is ever persisted.
- **Worker sizing follows**: with cropping on, `ingest-zone-year` sizes `max_workers`
  from the cell's live-chunk count (0.5 workers/chunk, clamped) instead of granting a
  4-tile zone the same fleet as a dense one.

Unconditional at every layer; the full-extent write path is gone from both modules along with
every branch that tested for it. S1 and S2 share the mechanism, differing only in that S1's
multi-date batches loop per date, since non-contiguous dates cannot share a region write, each
keeping its own atomic commit and retry scope.

#### Narrowing a date's windows, and skipping dates that reach none

A run's live windows are the same on every date; one date is not, since a satellite images a
fraction of a wide ROI per pass, so most windows hold nothing for it. `windows_for_date`
intersects the run's windows with that date's own STAC footprints — reprojected onto the ingest
grid and padded one cell, so a curved reprojection cannot under-cover — then re-bands and
re-groups. Tasks over the removed windows would run, find nothing and write nothing, so this
cannot change what a mosaic contains. When a footprint cannot be determined the full window list
is returned unchanged, so the conservative path is the fallback. Both sensors do it
(`narrow_windows_per_date` on S1, always on S2): six times fewer windows per date on the S1 zones
measured, worth 7–20% of per-date wall clock.

**A date whose imagery reaches NO live window is skipped entirely**, on both paths and
unconditionally. Writing it builds a full graph to store nothing. On S1 this is not a rare case:
one zone skipped 13 of 58 dates, and some zones have an orbit that reaches land on *no* date of
the year. Skipping those creates no store, letting `resolve_s1_orbit` downgrade to single-orbit
rather than publishing a store of fill that inference would read as real signal.

**The safety rule, and it is the whole design.** A footprint that is too LARGE only costs
computed area that would have been discarded; one that is too SMALL drops imagery and nothing
downstream notices. So every uncertain path widens rather than narrows: an unreadable footprint
returns the full window set, and on S1 a time slice that cannot be matched to its items writes
everything. "Reaches nothing" and "we cannot tell" are separate branches — only the first skips.

S1's match is on an **exact timestamp** rather than a date string, because odc sets a slice's
time coordinate to its group's earliest item timestamp. Keying by solar day instead would
disagree with the loader wherever the offset crosses UTC midnight.

### Keeping the fleet busy

A graph that fits can still under-use the fleet: windows written one after another leave most
slots idle, and the client-side work between dates is time no worker spends on anything. These
three fill that, and the last of them is not a straight win.

#### Overlapping a date's window writes (`overlap_window_writes`)

A date is written as several chunk-disjoint windows. Writing them one at a time dominates the
cost of a date: each window's compute completes before the next begins, so the date costs the
**sum** of the windows' critical paths while the fleet idles through most of each.

```text
Sequential windows — the fleet sees one window at a time:
 │◄─ window 1 ─►│◄─ window 2 ─►│◄─ window 3 ─►│◄ w4 ►│◄─ window 5 ─►│
 └─ the date costs the SUM of these, and most slots idle within each ─┘

Overlapped (overlap_window_writes, the default) — one graph, one commit:
 │◄─ window 1 ─►│
 │◄─ window 2 ──►│     all submitted together, so the fleet packs them and
 │◄─ window 3 ─►│      the date costs roughly the LONGEST window plus
 │◄ w4 ►│              whatever the total work itself requires
 │◄─ window 5 ──►│
 └── one merge, one commit for the date (contract unchanged) ──┘
```

Mechanism: icechunk's dask path already forks a session, stores lazily and merges changesets,
and writing per window runs that sequence once per window. Overlapping lifts it one level — fork
once, collect every window's lazy stored arrays, run one merge reduction — so every window's
loads, masks and chunk writes occupy a single graph.

The resulting store is identical either way, because the windows are chunk-disjoint: that is what
makes the merged changesets conflict-free, and the same property that lets a date commit exactly
once. Should icechunk's internals move, the write falls back to the sequential loop with a
warning.

Default **on** for both S2 and S1. `write_day_windows` itself still defaults to the
sequential path: a storage-layer default should not decide write strategy for its callers,
so each ingest path opts in explicitly.

The gain is **2.4–3.9×** on per-date write time and varies with neither window count nor fleet
width. Why it is that size is not explained — three accounts were proposed and all three refuted
by their own predictions — so rely on the measured range, do not model it, and do not extrapolate
far outside the widths measured. `context_docs/ingest/ingest-performance.md` §3.11 and §4.9.

#### Pipelining a date's preparation (`pipeline_dates`)

A date's wall clock splits into **preparation** — building the load graph, running the coverage
gate, narrowing the footprint, constructing the masks — and the **write**. Preparation is part
client-side CPU, independent of fleet width, and part cluster compute, since the coverage gate
reads SCL on the workers. Only the client-side part is serial residual a wider fleet cannot
shrink.

**The overlap's payoff is therefore not symmetric.** Hiding the client-side part behind the write
is free; hiding the gate is not, because it is fleet work and on a saturated fleet competes for the
same slots regardless of scheduling order. So the overlap pays in proportion to the spare capacity
the write leaves, making it **more** valuable on narrow fleets than wide ones.

`pipeline_dates` prepares date N+1 on one background thread while date N is written
(`ingest/_pipeline.py`). The write stays serial: icechunk commits are sequential on a branch and
one commit per date is the contract, so the store has exactly one writer either way. Preparation
must be **side-effect-free**, touching nothing but the dataset it hands back, and that is what
makes the two modes produce identical stores — pinned by a parity test including a date that fails
the coverage gate mid-run.

Depth is 1, intrinsically rather than by tuning: preparation is a small fraction of a write, so
buffering more would hold graphs in memory to hide nothing. The pipeline lives inside one `_drive`
call and drains at each streamed month boundary, leaving one unhidden preparation per month.

Each written date logs `Pipeline date=…: prepare=… hidden=… stall=…` in both modes. `stall` is
the preparation the write could not cover, and is the health metric: near zero when preparation
hides fully, rising toward the whole preparation when the gate is starved behind the write's own
tasks. Serially every date stalls for its full preparation, so the two modes compare from one
line. **`hidden` is not a saving** — pipelined, `prepare` is background-thread wall time spanning
the whole concurrent write, so it inflates with contention; take any A/B's expected saving from
the control arm's `prepare` and treat `hidden` as a contention diagnostic.

Default **off**, and the flag threads from the outer flow through the task shell to the
domain function. S1 has no coverage gate and a different batch loop; it is deliberately
untouched.

#### Batching dates into one compute (`batch_dates`)

**Sized per ROI, and NOT a straight win.** `batch_dates=None` (the default) derives the batch size
from the ROI's covered window area via `config.ingest.auto_batch_dates`; an explicit integer forces
one, which pins an A/B arm. Batching helps small ROIs, is roughly neutral on large
ones, and **costs about 29% on mid-sized ones** — so one global value is wrong for part of the
range.

The arithmetic behind that shape:

```
per-date wall clock  ≈  max( W , P )  +  commit / k

    W = the batch's write, per date          k = dates fused into one graph
    P = the preparation running alongside it, per date
```

Batching divides the commit by `k` and does nothing else. It cannot make the write faster,
since the fleet is already the constraint, so commit amortisation is its only gain — and it LOSES
wherever the larger write graph crowds out the preparation overlapping it. On a mid-sized ROI,
preparation at `k=1` already fitted inside the write with zero stall.

So batching pays only where the fleet has idle capacity to fill, and the threshold sits at the
top of the range where that was measured to hold — widening it means measuring an ROI in between.
Denominating it in covered window area also couples it to the merge exchange rate above: a finer
merge covers less area, so more ROIs drift below the threshold and batch. Recalibrate against
runs, never an offline sweep at a different merge cost. Figures in
`context_docs/ingest/ingest-performance.md` §3.16.

When it is on, k consecutive PASSING dates compute as ONE graph: their work packs the fleet
together, one date's straggling reads backfill with another's writes, and the drain tail and
commit gap are paid once per batch.

The commit unit becomes the batch, forced rather than chosen: every date's append resizes the
time axis, so per-date sessions forked from one snapshot would conflict on array metadata even
with disjoint chunk data (`storage.zarr_store.write_days_windows`). A mid-batch failure commits
none of the batch's dates, and a retry re-ingests exactly the uncommitted ones. Stores are
byte-identical to the per-date path (pinned by a parity test whose gate-failing date
sits mid-batch).

Skipped dates do not occupy batch slots, so batches stay full exactly where the gate
filters most; the trailing partial batch flushes at each streamed month boundary. In
batched mode the per-date `Stage timings` line is replaced by one `Batch timings` line
per batch (build/gate are sums of real per-date values; the write is one shared compute
and has no per-date decomposition). Default 1 — the one-commit-per-date path — is unchanged.

**Composing with `pipeline_dates`.** The two are complementary: batching removes fleet idleness
*within* a date's write, pipelining removes serial preparation *between* writes. Composed, the
look-ahead is sized to the batch rather than one date, since a batch's write is one long consume
and a depth-1 buffer would hide one date's preparation out of k. Preparation stays
single-threaded at any depth, so its side-effect-free contract is unchanged; the cost is up to k
prepared dates buffered while k more are written. `Batch timings` reports
`prepare`/`hidden`/`stall` per batch, with the same caveat as the per-date line.

### has_new_stac_dates pre-check

**Not yet wired into any flow** — this section describes something unbuilt, kept because the
reasons are worth having written down. `has_new_stac_dates` is meant to run before provisioning a
Dask cluster: it queries the STAC catalog and checks for new dates without reading any raster data
or starting Fargate tasks, so a flow could exit early when nothing is new.

An overlapping date range is no longer a correctness problem:
each ROI ingest begins the day after the newest date its store holds, and a window wholly below
that line returns a skip without querying (see *Where a resumed run starts*). What the pre-check
would still buy is avoiding the cluster, since the skip is decided inside the ingest and the
caller reaches it only after provisioning one. Tracked in
[issue #47](https://github.com/dClimate/tessera-embeddings/issues/47); when wiring it, do not
share one OPERA `item_provider_fn` between the pre-check and the real query, because the provider
re-queries CMR on every call.

---

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
`_QUERY_WINDOW_WORKERS` (6) of them at once. The worklist is driven from the calling thread and
tasks only ever walk — they never submit and never wait — so deadlock is structurally impossible
rather than merely unobserved. Each thread gets its own `Client`, because `StacApiIO` wraps a
`requests.Session` that is not documented thread-safe. Output order comes from
`_WindowWalk.preorder()` on the finished tree, and the `id` dedupe runs at that assembly step
rather than as pages arrive, so first-occurrence-wins means first in the **walk** and not first
off the wire. Six rather than eight, even though eight is faster: the campaign runs tens of cells
against this one provider at once, so the setting multiplies the concurrent search streams
Element 84 sees, and per-page latency degrades with width. A failure does not stop the other
windows — every window is walked, all failures collected, and the depth-first-earliest raised, so
which failure surfaces is a function of the query rather than of which task finished first.

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
are all derived in `context_docs/ingest/ingest-performance.md` §7c.

