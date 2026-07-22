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
attr `ingest_marker`, a fingerprint of window + `min_valid_coverage` + requested
`s1_orbit` + `allow_partial_window` + coverage-delivery sha): a matching marker on every required store
short-circuits, and the marker is written **only after coverage is verified**, so
it can never bless an incomplete mosaic. The probe runs over the **maximal**
candidate set (reflectance + both SAR orbits) keyed on physical existence, not
just the resolved-orbit set — so a half-written prior attempt (one store written,
crashed before any marker, or a SAR crash the orbit-resolver can't see) is
**cleared and rebuilt** rather than appended onto, which would dedupe against
stale dates and then stamp the new fingerprint over mixed inputs. A changed input
(rebuilt coverage, new threshold, ascending-only → both) changes the fingerprint
and likewise forces a clean rebuild. The clearing delete is `strict` so a failed
delete aborts rather than ingesting onto stale data. An unreadable store (transient
/ auth `IcechunkError`) re-raises rather than being mistaken for "absent".

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
for the rare arctic-only edge zone; an empty store always fails. Because the
marker short-circuits before coverage validation, `allow_partial_window` is part
of the fingerprint — a mosaic accepted under the relaxed policy never satisfies a
later strict run (which would otherwise reuse it and then fail its fill's strict
preflight forever); the differing fingerprint forces a re-ingest instead.

**Grid gate** (fill, pre-Ray): reflectance **and every active SAR store** are
validated against the seeded zone grid (shape / CRS / coordinate endpoints).
SAR is read by positional slice with no coords of its own, so a stale child SAR
store or a hand-provided `mosaic_base` on a different grid than reflectance would
otherwise be mixed in and silently misgeoreference the fill.

**Model gate** (fill, pre-Ray): the store root's seeded `geoemb:model` (encoder
version) must match the running build's, else the fill refuses — a mid-campaign
model upgrade would otherwise write new-encoder embeddings under a store still
advertising the old one and tag them permanently. `allow_model_mismatch` is the
deliberate-override escape hatch. This gate is only sound because the root
provenance is **write-once**: `seed_zone_groups` stamps the encoder identity on the
first seed and rejects any incremental reseed that would change it (a partial
reseed with a different encoder would otherwise re-stamp the root and let the gate
wave the new encoder through onto already-published zones).

**Zone identity = UTM common name** (`"33N"`, `"07S"`) everywhere — group names,
mosaic paths, tags, flow params — with EPSG retained only as the CRS
(`ZoneSpec.epsg` / `proj:code`). This is a **deliberate deviation from the
geoembeddings `utm_zones` spec**, whose `utm{NN}` group name cannot express the
hemisphere (33N vs 33S). A hemisphere amendment is a candidate to propose
upstream. `canonicalize_zone` is the single parser.

**Staging `run_id`** (fill, per cell): the campaign derives each cell's staging
`run_id` as `{zone}-{year}-{hash}` over the acceptance config (threshold / orbit /
window / checkpoint), the **immutable code artifact** the fill will run, and the
per-`(zone, year)` mosaic identity (post-ingest `ingest_marker`). A retry with
identical inputs resumes the same staging prefix; any change starts a fresh one, so
tiles staged by old inputs are never resumed under new ones. The code artifact is the
**resolved AMI ID plus (when a source tarball overlays it) that object's ETag**, not
the mutable `code_suffix` label — re-baking the AMI under the same SSM name or
overwriting `code/src{suffix}.tar.gz` would otherwise leave the fingerprint unchanged
and let a retry publish a permanently-tagged mixed-code year. An **all-ocean cell**
(no live tiles) produces no mosaic and the fill marks it empty with no staging, so it
takes a stable `-empty` `run_id` and skips both mosaic fingerprinting (which would
raise) and cleanup. The campaign's `s3_region` is threaded through ingest's Icechunk
metadata opens as well as the fill's, so a non-default-region store is read
consistently (the ROI-engine mosaic write remains us-west-2-only, a pre-existing
limitation moot while all campaign data lives there).

**Time convention — calendar-year DEFAULT, not a guarantee.** Each zone group's time
axis is indexed by calendar year (slot coordinate = Jan 1 of the year, fixed at seeding,
D1), and the campaign always fills a strict Jan–Dec window — so the group advertises
`time_convention="calendar_year"`. But that is the DEFAULT labeling, not a hard promise
(`time_convention_strict=False`): a non-campaign consumer may deliberately drive a
non-calendar 12-month window (e.g. a rolling Feb–Jan) into a year slot for non-standard
processing. The `fill_zone_year` runner therefore requires only that the inference window
OVERLAP the target year (a fully-disjoint window is still rejected as operator error), and
the **actual window is recorded per year** in the group's `runs` provenance (its
`window_end_label`, written by `assemble_global` / `mark_zone_year_empty` via
`run_provenance`). A deviation is thus legible rather than a silent mislabel under the
calendar-year slot. (Considered and rejected: hard-requiring an exact Jan–Dec window —
it would break the deliberate rolling-window capability for a marginal guarantee the
per-slot provenance already provides.)

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
