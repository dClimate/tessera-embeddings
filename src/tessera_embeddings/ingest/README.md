# Satellite Ingestion

Modules for querying, authenticating, and loading satellite data from STAC catalogs and CMR
into Icechunk/Zarr stores. Used by the Tessera ingestion flows (`ingest_s1_roi_sar`,
`ingest_s2_roi_reflectance`) and the BarlowTwins tile-based ingestion flows.

---

## Module Overview

| Module | Purpose |
|---|---|
| `stac.py` | STAC-based data loading via `odc.stac.load`. Handles multiple providers (Earth Search, Planetary Computer), S2 baseline correction, and date filtering. |
| `opera_query.py` | OPERA RTC-S1 query utilities: spatial bbox construction, orbit-direction filtering via native CMR Granule Search API (CMR-STAC silently ignores the `query` extension), UTM EPSG derivation, and asset preparation. |
| `auth.py` | NASA Earthdata Login (EDL) authentication for ASF-hosted OPERA data. Provides S3 direct access (temporary STS credentials) and legacy CloudFront signed URL resolution. |
| `transforms.py` | Post-load lazy Dask transforms. Currently: `amplitude_to_db` for converting OPERA RTC-S1 linear amplitude to scaled uint16 dB. |
| `roi.py` | ROI (Region of Interest) utilities: reading existing Zarr ROI stores (WGS84 bbox, CRS, grid dims), rasterizing GeoJSON polygons to chunked boolean Zarr masks on UTM grids, and loading S2 MGRS tile footprints from S3. |
| `roi_processing.py` | Higher-level ROI processing helpers used by the `generate_roi` flow. |

---

## Basic Ingestion Process

The high-level entry point is `ingest_tile()` in `stac.py`. It runs five stages in sequence:

```text
1. STAC query        — find items for the tile/bbox + date range
2. Item filtering    — apply sensor-specific pre-filters (e.g. orbit direction for S1)
3. Date dedup        — drop items whose dates are already in the Zarr store
4. odc.stac.load     — lazy-load COGs into a Dask-backed xarray Dataset
5. Corrections       — baseline correction (S2) or dB conversion (S1)
```

`query_stac_items` and `load_stac_items` expose these stages separately for flows that need
to check for new data before spinning up a Dask cluster (see `has_new_stac_dates`).

### STAC Providers and Collections

Provider configs live in
[`config/providers.py`](../config/providers.py) (`PROVIDERS`, `CollectionConfig`). Supported:

| Provider | Collections |
|---|---|
| Earth Search (AWS) | Sentinel-2 L2A, Sentinel-1 GRD |
| Planetary Computer | Sentinel-2 L2A, Landsat (**untested**) |
| CMR-STAC (NASA) | OPERA RTC-S1 |

Each `CollectionConfig` records: collection ID, band list, native resolution, tile ID property
(for S2/Landsat property-based queries), and correction parameters. For OPERA RTC-S1 on
CMR-STAC there is no tile ID property, so the query falls back to a WGS84 bbox.

**NOTE** Planetary Computer is an untested provider. Feedback from Cambridge's TESSERA team
indicates however that Microsoft throttles heavy outbound traffic from Planetary Computer and
hence it's not an ideal provider. For this reason we jumped through all the OPERA RTC hoops.

### STAC Query Strategy

S2 and Landsat are queried by tile ID property (e.g., `grid:code = T33UUP`), which returns
only items for that specific MGRS tile. OPERA RTC-S1 on CMR-STAC lacks an equivalent
property, so queries use a bbox derived from the MGRS tile via `mgrs_tile_to_bbox()`.

`_query_stac_items` is wrapped with `tenacity` retry logic: up to 4 attempts with
exponential back-off (max 120 s) on `APIError`, which handles transient catalog outages
without failing the whole flow.

Cloud cover is intentionally **not** used as a filter at the STAC query stage — pixel-level
cloud classification is handled later (SCL for S2, ML model for inference). For S2, items
are sorted by `(date, eo:cloud_cover)` so that within a solar-day mosaic the clearest tile
is placed first, which is what `odc.stac.load`'s groupby mosaicking relies on.

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
`min_valid_coverage` percent of ROI pixels are valid (default 5%). Only the per-date valid
pixel counts are computed eagerly — band arrays remain lazy until the Zarr write.

`identify_low_coverage_ds` is the lazy alternative: instead of dropping dates it attaches a
`valid_coverage` boolean coordinate that downstream tasks can check without reading band data.

---

## Data Transformations

### Pre-load (STAC items)

These happen before `odc.stac.load` is called:

| Transform | Where | What |
|---|---|---|
| **Date dedup** | `stac._filter_existing_dates` | Drops STAC items whose date is already written to the store. |
| **Item sort** | `stac.query_stac_items` | For S2: sorts by `(date, cloud_cover)` so mosaicking picks the clearest tile. |
| **Orbit filter** | `opera_query.make_s1_item_rewriter` | Keeps only OPERA bursts matching the desired orbit direction. |
| **URL rewriting** | `auth.rewrite_assets_to_s3` | Rewrites HTTPS datapool/earthdatacloud URLs to `s3://` URIs. |
| **Timestamp normalisation** | `opera_query.normalize_opera_timestamps` | Sets all OPERA burst timestamps on the same date to noon UTC so `odc.stac.load` groups them into a single mosaic. |

### Load-time (`odc.stac.load`)

`_load_from_stac` configures `odc.stac.load` with:

- **Resampling** — bilinear for primary spectral bands. Extra bands (e.g., S2 SCL) always
  use nearest-neighbour regardless of the primary resampling, enforced via a per-band dict.
- **Resolution override** — S1 is loaded at 10 m to share a common grid with S2, even though
  the native OPERA product is 30 m. Resampling to target resolution uses COG overviews and
  happens during read rather than as a post-processing step.
- **CRS override** — OPERA RTC-S1 items on CMR-STAC lack `proj:` extension metadata; an
  explicit `crs=` (e.g. `EPSG:32633`) must be passed so `odc.stac.load` knows the output
  projection.
- **GeoBox alignment** — when a `GeoBox` derived from `read_roi_metadata` is supplied, the
  output grid matches the ROI exactly (same CRS, transform, shape). This overrides bbox,
  CRS, and resolution.
- **groupby** — `"solar_day"` merges items from adjacent MGRS tiles that were acquired on
  the same local calendar day into a single mosaic using the painter's algorithm: items are
  rendered in order and later items overwrite earlier ones where they overlap. Items are
  sorted cloudiest-first so the clearest tile paints last and wins. This is the standard
  approach for multi-tile STAC mosaicking and is required for ROI queries that cross tile
  boundaries; not needed for single-tile BarlowTwins ingestion.
- **Dimension rename** — `normalize_odc_dims` maps `odc.stac.load`'s `y`/`x` output
  dimensions to the project-wide `northing`/`easting` convention and drops `spatial_ref`.

### Post-load

#### Sentinel-2 Baseline Correction

ESA changed the S2 L2A processing baseline at version 04.00, adding +1000 to all pixel values.
The correction logic in `_apply_baseline_corrections_by_date` is enabled per-collection via
the `requires_baseline_correction` flag in `CollectionConfig`. In practice, the Earth Search
catalog (used by the Tessera ROI ingestion flow) already delivers pre-corrected values, so
the correction is **not applied** in the current production flow.

The code is retained for collections or catalog providers that still require it. When active,
it builds a per-date correction mask vectorised across the time dimension. Two modes:

- **Default** (`preserve_low_values=False`) — subtracts the offset from all pixels. Values
  below 1000 go negative, which acts as a nodata/dark-pixel signal in the cloud-mask model.
- **Tessera mode** (`preserve_low_values=True`) — only subtracts from pixels where
  `value >= abs(offset)`, leaving dark pixels unchanged. Matches Tessera's `harmonize_arr()`
  behaviour for inference.

Values above `65535 + offset` (i.e. 64535) are clamped before subtraction to prevent uint16
overflow. The correction is applied only to spectral bands; SCL and other extra bands are
passed through unchanged.

#### OPERA RTC-S1 Amplitude-to-dB Conversion

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

---

## Performance Optimizations

### Lazy evaluation throughout

`odc.stac.load` returns a Dask-backed xarray Dataset with no raster data read yet. All
post-load transformations (baseline correction, dB conversion, ROI masking) chain additional
Dask operations without computing. Data is read and written in a single Dask graph execution
triggered by the Zarr write step.

### Coverage pre-filtering before compute

`filter_low_coverage_dates` eagerly computes only the per-date valid pixel counts (one
scalar per time step) from the quality band (SCL for S2, VV for S1) to decide which dates to
keep. All spectral bands remain lazy. Dropping low-coverage dates before `.compute()` avoids
reading band data for cloud-covered or off-ROI scenes.

### Date deduplication before loading

`_filter_existing_dates` removes items whose dates are already in the Zarr store before
calling `odc.stac.load`. This avoids building Dask task graphs for data that will be
discarded, and prevents unnecessary COG reads from S3.

### S3 direct access for OPERA

S3 direct access (`get_s3_credentials`) bypasses the 5-hop OAuth redirect chain for each
OPERA COG tile. One HTTP round trip (~0.5 s) fetches temporary STS credentials valid for
1 hour, enabling GDAL to read directly from `s3://asf-cumulus-prod-opera-products` without
per-file HTTPS redirects. At batch scale this is the dominant latency reduction.

### GDAL network tuning

`configure_gdal_environment()` (in [`config/environment.py`](../config/environment.py)) must be
called before importing `rasterio` or `odc.stac`. It sets GDAL config options for network
resilience (retry counts, timeouts, connection pooling) that affect all subsequent COG reads.

### Chunk alignment

The ROI Zarr mask is generated with `chunk_size` matching `TESSERA_CHUNKS` so that
`da.from_zarr` reads are zero-copy — each Dask partition maps to exactly one Zarr chunk.
The same chunk sizes are passed to `odc.stac.load` (after translating `northing`/`easting`
to `y`/`x`) so band arrays and the mask share the same partition boundaries for aligned
Dask operations.

### has_new_stac_dates pre-check

Flows call `has_new_stac_dates` before provisioning a Dask cluster. It queries the STAC
catalog and checks for new dates without reading any raster data or starting Fargate tasks.
If nothing is new the flow exits early.

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
future Dask workers via a `WorkerPlugin`. It sets both environment variables (for boto3/rasterio
session creation) and GDAL config options (which bypass GDAL's internal credential cache that
can hold stale values).

**Why GDAL config options are needed in addition to env vars**: GDAL's `/vsis3/` handler caches
credentials resolved from env vars in a static C++ member. `VSICurlClearCache()` flushes curl
handles and file-property caches but does **not** flush the credential cache.
`gdal.SetConfigOption` bypasses the cache entirely because GDAL checks config options before
env vars and re-reads them on every file open.

**Per-thread AWSSession cache**: `odc.loader` caches a boto3 `AWSSession` per thread in
`threading.local` on first use and ignores subsequent env var updates for that thread's
lifetime. Dask task pool threads are long-lived, so the initial 1hr STS token was getting
pinned across refreshes and expiring mid-read. `auth.py` patches `odc.loader._rio.ThreadSession`
at module import time so each thread self-detects `AWS_ACCESS_KEY_ID` drift and rebuilds its
cached session. This reaches into private `odc.loader` internals (`_OdcThreadSession`, `_local`)
and is a version-sensitive hook — if odc renames those symbols, the import fails loudly and the
test in `tests/unit/ingestion/test_edl_auth.py::TestOdcThreadSessionEnvDriftPatch` catches the
break in CI before it hits a 1hr cloud run.

OPERA asset STS credentials are intentionally **never cleaned up** from env vars. This avoids
a race condition where one Dask task's cleanup could remove credentials another task still
needs. Icechunk/Zarr write operations on the project's own S3 bucket stay isolated by being
configured with an explicit credential callback (see ``storage.zarr_store._create_storage``
``get_credentials``); the AWS-provider implementation passes a callback that resolves
deployment credentials directly from the IAM role, so it is unaffected by whatever
``AWS_*`` env vars OPERA's STS session has set.

### URL Rewriting

CMR-STAC returns HTTPS asset URLs in two formats depending on satellite vintage:

| Format | Example |
|---|---|
| **datapool** (older S1A) | `https://datapool.asf.alaska.edu/RTC/OPERA-S1/<filename>` |
| **earthdatacloud** (newer S1C) | `https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/<dir>/<file>` |

`auth.rewrite_assets_to_s3` converts both to `s3://asf-cumulus-prod-opera-products/...` via
pure string manipulation (no HTTP calls). For the datapool format, the granule directory name
is reconstructed by stripping the band suffix (`_VV.tif`, `_VH.tif`, `_mask.tif`) from the
flat filename.

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

## OPERA-Specific Query Quirks

### Orbit Direction Filtering

`make_s1_item_rewriter` builds an `item_filter_fn` that filters OPERA bursts by ascending or
descending orbit. CMR-STAC **silently ignores** the `query` extension for CMR additional
attributes such as `ASCENDING_DESCENDING`, so passing orbit direction to a STAC search has
no effect.

The workaround queries the native CMR Granule Search API directly:

```text
GET https://cmr.earthdata.nasa.gov/search/granules.json
    ?short_name=OPERA_L2_RTC-S1_V1
    &attribute[]=string,ASCENDING_DESCENDING,ASCENDING
    &bounding_box=...
    &temporal=...
```

This returns a paginated list of matching `producer_granule_id` values. The STAC items are
then filtered by matching their `id` against this set. CMR pagination is handled via the
`CMR-Search-After` response header.

### Burst Timestamp Normalisation

A single MGRS tile bbox query returns ~10 burst granules per date, each with a slightly
different sub-second UTC timestamp (reflecting actual acquisition time). If passed to
`odc.stac.load` as-is, each burst becomes a separate time step instead of being mosaicked
together.

`normalize_opera_timestamps` groups bursts by calendar date and sets all timestamps in each
group to noon UTC of that date. `odc.stac.load` then treats them as concurrent acquisitions
and spatially mosaics them into a single time slice.

### UTM CRS Derivation

CMR-STAC OPERA items lack the `proj:` extension, so `odc.stac.load` cannot infer the output
CRS. `mgrs_tile_to_utm_epsg` derives the correct UTM EPSG from the tile's zone number and
latitude band (C–M = southern hemisphere, N–X = northern hemisphere), e.g. `33UUP` → EPSG:32633.

---

## Accessing the Dask Dashboard

Ingestion flows run a Dask cluster on ECS Fargate. The scheduler is in a private subnet.
Use SSM port forwarding to reach the Bokeh dashboard:

```bash
# Look up TASK_ID and RUNTIME_ID for the Dask scheduler task in the ECS console
aws ssm start-session \
  --target ecs:yield-cluster_${TASK_ID}_${RUNTIME_ID} \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters '{"host":["localhost"],"portNumber":["8787"],"localPortNumber":["8787"]}'
```

Then open http://localhost:8787 in your browser.

Requires the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (`brew install session-manager-plugin`).
