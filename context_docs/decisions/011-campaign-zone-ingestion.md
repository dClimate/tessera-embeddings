# 011 — Campaign-triggered per-zone ingestion

**Status:** **Accepted (2026-07-15).** Implemented as
`ingest.land_mask.export_zone_roi` (zone ROI synthesis), the
`ingest-zone-year` Prefect flow, and campaign wiring in `run_global_campaign`
(ingest → fill → mosaic cleanup). Builds on ADR-008 (global store), ADR-010
(coverage bitmaps), and the existing ROI ingest engine.

## Context

The global campaign (ADR-008) drove inference + assembly per `(zone, year)` but
treated the input mosaics under `{inputs}/mosaics/{zone}` as "a separate upstream
step" — nothing in-repo produced them. All satellite ingestion was **ROI-based**
(`ingest_s2_roi_reflectance` / `ingest_s1_roi_sar`), keyed off a rasterized ROI
zarr whose grid/CRS/bbox is the sole authority. We want the campaign to produce
its own mosaics without a second ingestion engine.

Two hard constraints shaped the design:

1. **The fill validates mosaics against the seeded zone grid** exactly (shape /
   CRS / coordinate endpoints ±½ px, `runners/zone_fill.py`). `generate_roi`'s
   `compute_grid` bbox-fits the input geometry and cannot reproduce the fixed,
   shard-snapped `zone_grid.ZoneSpec` extent — so a GeoJSON detour is not viable.
2. **A UTM zone is huge** (~676 km × up to ~934 km per hemisphere). The S2
   per-solar-day coverage filter is a percentage of the *whole ROI*, so the ROI
   default (5 %) would drop almost every date over a whole 6° band.

## Decision

**Reuse the ROI ingest engine unchanged; feed it a synthesized zone-shaped ROI.**

- **Zone ROI synthesis** (`export_zone_roi`): write the exact ROI-mask artifact
  (`ingest.roi.read_roi_metadata`'s contract) directly from `ZoneSpec` — a bool
  zarr on the fixed zone grid, mask = the zone's `tile_live_2048` coverage bitmap
  upsampled ×2048, with a WGS84 bbox tight to the live tiles (so the STAC/CMR
  query never scans ocean-only latitudes). Ocean skipping is tile-granular via
  the coverage mask; within a live coastal tile, water pixels are ingested by
  design (the fill embeds whole live tiles, ADR-010), and masked ocean chunks are
  elided on write (empirically pinned in `test_zarr_store`).
- **`ingest-zone-year` flow**: ocean-skip → synthesize ROI → per-store completion
  marker probe → dispatch S1/S2 ingest deployments onto
  `{inputs}/mosaics/{zone}/{year}` → verify temporal coverage → write the marker.
  A low `min_valid_coverage` default (0.1 %) fits the zone-scale denominator.
- **Campaign wiring**: per cell, ingest (own concurrency cap) → fill → delete the
  transient mosaic. `ingest=False` bypasses for pre-existing mosaics.

**Per `(zone, year)` mosaics** (`mosaics/{zone}/{year}/…`), not one multi-year
store per zone: isolates windows, avoids out-of-order multi-year time-axis
appends, and makes deletion + idempotency per-cell.

**Idempotency / crash-repair** rests on a per-store **completion marker** (root
attr `ingest_window`): a matching marker on every required store short-circuits;
a crash mid-ingest is repaired by a plain re-run because the ROI flows dedupe
already-present dates (incremental append) and the marker is written **only after
coverage is verified**. No deletion is needed for repair.

**Mosaics are transient** (`cleanup_mosaics`, default on): they are re-derivable
inputs at ~5–15 TB per zone-year, so retention across 120 zones × 9 years is
untenable. Deletion uses `s5cmd rm --all-versions` (shared
`storage.object_store.delete_prefix`) so a versioned bucket does not accumulate
non-current versions; staging cleanup was upgraded to the same helper.

**Coverage gate** (`check_time_window_coverage`): a zone-wide mosaic spans all
latitudes, so a whole-month gap is an ingest-failure signal. The **fill** hard-
fails on a partial mosaic **before provisioning Ray** (the write-once zone-year
tag would otherwise make partial embeddings permanent); the **ingest** verifies
the same before marking done. `allow_partial_window` relaxes both to "non-empty"
for the rare arctic-only edge zone; an empty store always fails.

**Zone identity = UTM common name** (`"33N"`, `"07S"`) everywhere — group names,
mosaic paths, tags, flow params — with EPSG retained only as the CRS
(`ZoneSpec.epsg` / `proj:code`). This is a **deliberate deviation from the
geoembeddings `utm_zones` spec**, whose `utm{NN}` group name cannot express the
hemisphere (33N vs 33S). A hemisphere amendment is a candidate to propose
upstream. `canonicalize_zone` is the single parser.

## Alternatives considered

- **A zone-native (non-Dask) ingest engine** — rejected: it would duplicate the
  STAC/CMR + `odc.stac.load` + write machinery. The ROI engine's per-date/per-
  batch graphs are already the accepted Dask pattern; the "avoid unnecessary Dask
  graphs" rule targets *metadata* reads, not the ingest data plane.
- **A zone GeoJSON fed to `generate_roi`** — rejected: `compute_grid` bbox-fits
  and cannot pin the shard-snapped zone grid the fill requires.
- **One multi-year mosaic per zone** — rejected: out-of-order appends break
  `resolve_region`, and per-cell deletion/idempotency is cleaner.

## Consequences

- The campaign is self-contained: seed → build mask → (per cell) ingest → fill →
  cleanup. Ingestion and fill have independent concurrency caps.
- Peak input storage is bounded by in-flight cells, not the whole campaign.
- A deliberate refill re-ingests (hours, STAC/ASF re-pull) — acceptable given the
  storage saving; the coverage store + zone ROI regenerate in seconds.
- The zone-fill chain gained its first temporal-coverage gate; a missing-months
  mosaic now fails loudly instead of silently tagging partial embeddings.
