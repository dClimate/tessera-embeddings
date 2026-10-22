# Zone ROIs: the masks the global campaign ingests through

Every ingest starts from a region of interest: a boolean mask on a UTM grid saying which pixels
matter. A single-area run rasterises one from a GeoJSON polygon. The global campaign cannot,
because its fill validates against a fixed, shard-snapped zone grid that a polygon's bounding box
will not reproduce, so it writes each zone's mask directly from the zone's grid and its land
coverage.

This document is for anyone running the global campaign, pre-generating zone masks ahead of it,
or changing `land_mask.export_zone_roi` or `validate_zone_roi`. It covers how a zone's mask is
made and where it sits in a zone-year's ingest, and the `export-zone-rois` flow that builds and
checks many zones' masks ahead of time. How a single-area ROI is generated, read and applied is in
the [ingest README](../src/tessera_embeddings/ingest/README.md#roi-workflow).

## How a zone's mask is made, and where it is used

The global campaign reuses the single-area ROI engine to produce its per-zone mosaics: it
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
re-derivable inputs at ~TB scale
([ADR 011](../context_docs/decisions/011-campaign-zone-ingestion.md)). Zones are named by UTM common name (`33N`/`07S`),
not EPSG (see `storage/zone_grid.canonicalize_zone`).

## Pre-generating the zone masks: `export-zone-rois`

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
