# 010 — Land mask v1.1: registry-derived coverage bitmaps

**Status:** **Accepted (2026-07-15).** Implemented in
`src/tessera_embeddings/ingest/land_mask.py` (builder), the
`build-land-mask-coverage` Prefect flow, and `scripts/build_landmask_coverage.py`.
Consumed by the zone-fill runner (ADR-008 W5). Verified end-to-end against the
full delivery registry (120 zones, 112 with cells, 1,593,479 cells, 360,953 live
tiles; build ~5 s, validate ~1 s on a laptop).

## Context

The partner delivers the global land mask as ~1.59M per-0.1°-cell GeoTIFFs
(`grid_{lon}_{lat}.tiff`) plus a `registry.txt` and `SHA256SUM` under
`s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/`. This mirrors their
HPC TESSERA workflow, which tiles the globe for every step. The question was
whether to consume that format directly, or convert to GPKGs / a master TIFF /
per-zone TIFFs / some other form optimised for our per-zone, worker-parallel
campaign.

**Decisive partner semantics (v1.1):** every registry-listed tile is **all-1s**.
v1 followed the coastline per-pixel but *cropped land*, so v1.1 dropped
per-pixel masking; coverage also **extends ~1 cell (~11 km) into the sea**
because some users wanted a generous ocean margin. Therefore *tile presence /
the registry listing IS the mask* — there is no per-pixel land signal to read.

This was confirmed empirically on the delivery sample: registry membership
tracks all-1s exactly, and v1-era all-zero tiles (e.g. `grid_-0.05_-16.05.tiff`)
are absent from the registry. The registry is the authority.

## Decision

**D1 — The mask is the registry, materialised as per-zone coverage bitmaps.**
We build, from `registry.txt` alone (no TIFF pixels), two boolean arrays per UTM
zone:

- `tile_live_2048` — `(n_tile_rows, n_tile_cols)`, one bit per 2048-px tile,
  `True` where ≥1 land cell's projected footprint intersects it. This is what
  the zone-fill runner reads to select live tiles.
- `chunk_live_256` — the same coverage at 256-px inner-chunk granularity, kept
  for a future coverage-edge shard-leaning optimisation. By construction
  `tile_live_2048 == 8x8-block-any(chunk_live_256)`.

Globally this is ~4 MB (vs ~2–8 GB for per-zone pixel masks). Build is pure
geometry: project each cell's footprint into its zone CRS with pyproj, snap to
our zone-grid pixel indices, OR the covered rectangles into the bitmaps. No
data-plane reads; sub-minute for the whole globe.

**D2 — One Icechunk repo, 120 zone groups** (`BucketPaths.land_mask_store()`),
mirroring the global store (ADR-008 D5). The consumer read is exactly
`open_store_as_zarr_group(path, group=zone)` — the same helper, storage
timeout/retry hardening, and credential path the rest of the campaign uses. A
re-delivery is a new commit whose message carries the registry sha256 (real
provenance history). This was chosen over plain Zarr after weighing it: plain
Zarr would have meant a second, unhardened I/O pathway and a hand-rolled
done-marker convention to save ~5 lines of session ceremony in a builder that
runs once for seconds — a net loss.

**D3 — Cell → zone is filename math on the NOMINAL 6° UTM band**
(`utm_6deg_nominal`, matching ADR-008), never the MGRS width exceptions in
`ingest.roi.determine_utm_zone`. Cells never straddle a 6° band edge (a band is
exactly 60 0.1° cells; centres sit 0.05° off every boundary), so the assignment
is exact. Confirmed against the delivery TIFFs' embedded CRS.

**D4 — Pixel-level land masking is moot, not deferred.** v1.1 has no per-pixel
signal (all-1s + buffer), so within a live tile every observation-valid pixel is
embedded, and water is SCL-valid (as before). The ADR-008-era "revisit coastal
pixel masking" question is closed. If a future v1.2 reintroduces per-pixel
coastline data, the builder's geometry path is the regeneration hook.

**D5 — Verification guards the load-bearing assumption.** The build itself reads
no pixels, so `verify` (pre-build) samples ~500 delivery TIFFs and asserts
all-1s, CRS == filename-derived zone, and bounds ≈ projected cell bbox; any
violation hard-stops. It also reconciles registry ⊆ bucket (presence-based; the
~126k `SHA256SUM` extras are the expected v1-era leftovers). Post-build
`validate` checks bitmap shape/consistency (`tile_live == block-any(chunk_live)`)
and known land/ocean points.

## Consequences

- The zone-fill mask read drops from ~15k windowed GETs per fill to one ~1 KB
  GET — ~16.3M GETs saved across the 120-zone × 9-year campaign.
- The delivery bucket is archived as-is (the partner's canonical copy); we build
  no GPKG/COG/master-TIFF artifacts and hash no pixels.
- **Cost model note:** what we embed/store is the *buffered coverage*
  (~1.96 Tpx) not the ~1.49 Tpx land-only estimate in ADR-008 — ~30% more
  pixels. Storage and compute scale with this; flagged for campaign budgeting.

## Partner questions — confirmed (2026-07-15)

Both confirmed "yes, for our purposes" by the maintainer:

- The **registry** (not a bucket listing) is the authority; the `SHA256SUM` /
  bucket extras are v1-era leftovers to be ignored. Our reconciliation
  (`registry ⊆ bucket`, extras reported but non-authoritative) is correct.
- The **~1-cell sea buffer width is stable** across versions, so the coverage
  extents and the buffered-coverage cost model (~1.96 Tpx) hold.
