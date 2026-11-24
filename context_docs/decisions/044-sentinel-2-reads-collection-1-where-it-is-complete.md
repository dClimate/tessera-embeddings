# 044 — Sentinel-2 is read from Collection 1 wherever Earth Search holds all of it

**Status:** Proposed (PR open, 2026-11-19)

## Context

Earth Search serves Sentinel-2 L2A as two collections:

- **`sentinel-2-l2a`** (the legacy collection) holds products at whatever processing baseline ESA
  used when each was made, from 02.xx to 05.xx. It mixes two producers: Element 84's COGs, which
  have the post-04.00 reflectance offset subtracted, and items pointing at ESA's originals, which
  keep it. That mix is why the offset is decided per asset from the bucket
  ([ADR 021](021-correct-the-boa-offset-per-image.md)), why copies of one image compete in
  duplicate selection ([`docs/ingest/duplicate-copies.md`](../../docs/ingest/duplicate-copies.md)),
  and why its `raster:bands` offset metadata cannot be trusted.
- **`sentinel-2-c1-l2a`** (Collection 1) is built from ESA's reprocessing of the archive at baseline
  05.00 or later. It has one producer, Element 84, whose COGs keep the offset. It also uses the
  newer Sen2Cor, the Copernicus DEM and improved co-registration.

The live catalogue was checked on 2026-11-19.

**Pixels.** For the same scene, Collection 1 reads exactly 1,000 above the legacy collection's
harmonised COG: a median of 1,387 against 387 on 15TVF on 2023-07-10, and 1,690 against 690 on
15TVF on 2020-07-10. Every item sampled declares `raster:bands` scale 0.0001 and offset −0.1, and
baselines 05.00 to 05.13.

**Bucket.** The COGs live in `e84-earth-search-sentinel-data`, which `asset_locations` lists as
**harmonised** because the legacy collection's COGs there are harmonised. Classifying Collection 1
by its bucket would exempt it and leave every pixel 1,000 too high.

**Coverage.** These are unique acquisition dates per MGRS tile per year. Collection 1 never has a
date the legacy collection lacks.

| Tile | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| 15TVF (Iowa) | 2/38 | 55/113 | 114/146 | 146/146 | 145/145 | 10/142 | 145/145 | 146/146 | 158/157 |
| 33UUP (Germany) | 0/33 | 61/168 | — | 217/217 | 218/218 | 18/215 | 216/216 | 215/216 | 243/243 |
| 36MZE (Tanzania) | 0/102 | 63/144 | 112/145 | 144/144 | — | 12/144 | — | 144/145 | 173/173 |
| 55HBU (Australia) | 4/33 | 37/109 | 106/137 | 128/128 | 140/140 | 9/139 | 141/141 | 141/141 | 154/153 |
| 23KLP (Brazil) | 3/54 | 46/132 | 113/145 | 145/145 | 146/146 | 12/144 | 146/146 | 146/146 | 158/155 |
| 44QKE (India) | 4/65 | 56/132 | 113/145 | 145/145 | — | 12/146 | — | 146/146 | 159/159 |

Each cell is Collection 1 dates over legacy dates. A dash means not measured; 33UUP's 2019 query
failed with a 502. 2026 to mid-November matched on three tiles.

The gaps are two periods Element 84 has not yet reprocessed, not scattered losses. Measured on
2026-11-23 across five of the tiles:

- **2019:** Collection 1's first date is 2019-03-23 or 2019-03-24 on every tile; before it there
  is nothing in 2019 and about half of 2018.
- **2022:** its first date in late 2022 is between 2022-12-06 and 2022-12-08; from there it holds
  every legacy date into 2023.
- **Isolated dates** are missing outside those periods: 2024-01-11 on 33UUP and 36MZE (but not on
  15TVF, 55HBU, 23KLP or 44QKE), 2019-06-25 on 33UUP and 2019-04-04 on 44QKE. This accounts for the
  215/216 and 144/145 in the 2024 column, under 1% of a year.
- **Near the antimeridian** Collection 1 is no worse than the legacy collection. Element 84
  [documents](https://github.com/Element84/earth-search/blob/main/docs/collections/sentinel-2-l2a.md#known-issues)
  that it excludes most scenes crossing 180°. In 2023 it held every legacy date on 60KYF and 60KXF
  (Fiji), and 51 of 53 on 60VXQ (Kamchatka, June to August). On 01KAB, 01KBB and 01VCC the legacy
  collection returns nothing for the tile while Collection 1 returns 25 to 28 dates.

Collection 1 can also hold several items for one tile and day: two datastrips of one pass, or two
satellites. They carry the same `s2:datatake_id` and `s2:datastrip_id` fields as the legacy items, so
`duplicates.image_identity` keeps each image, as it does for the legacy collection.

**Planetary Computer** offers no Collection 1. Its only Sentinel-2 collection is `sentinel-2-l2a`.
On 15TVF it holds all of 2019 and 2022, at the baselines originally produced: 148 of 150 items in
2019 are baseline 02.12, and 2022 mixes 03.00, 04.00 and 05.10.

## Decision

1. **Collection 1 is the default Sentinel-2 collection.** `DEFAULT_S2_COLLECTION` in
   `config/providers.py` is the default for the S2 ingest, its Prefect task and flow, and the optical
   preflight.
2. **Its offset rule is settled for the whole collection.** `harmonisation_varies_by_item` is off, so
   `collection_harmonisation` answers RAW and `boa_offset.source_decision` never consults the
   bucket. The threshold stays at 04.00, so every Collection 1 item is corrected.
3. **The periods it lacks read the legacy collection.** Its config sets `incomplete` to 2015-01-01
   through 2019-03-31 and 2022-01-01 through 2022-12-15, each about a week past the measured edge,
   with `fallback="sentinel-2-l2a"`. `stac.collection_for_window` routes a window that touches any
   of those days to the fallback **for the whole window**, so one run never mixes producers. The
   ingest and the preflight both resolve through it, so the preflight probes the collection the
   ingest will read.
   - The test starts a day before the window, because the catalogue query does: an image dated
     that UTC day can belong to the window's first solar day. It ends on the window's last day,
     because the query's padding day after the window holds none of the window's images at the
     one place it could matter, the day before 2022-01-01. Testing that day too would send every
     window ending on 2021-12-31 to the legacy collection.
   - Measured on 2026-11-23 with the ingest's own solar-day rule (`solar_days.solar_day_of_instant`).
     In 2023, 6,141 Collection 1 images between 180° and 60°W were dated the UTC day after their
     solar day, all of them in the first hour after midnight UTC:
     - zones 01–02 above 67.5°N (Chukotka and the Chukchi Sea);
     - zones 07–11 above 79°N (the Canadian Arctic islands).

     All were taken between late February and October, when those latitudes have daylight. The
     same region of the legacy collection from 2021-11-01 to 2022-02-28 held 47 such images, all
     on 2022-02-28, and none between 2021-12-20 and 2022-01-10. South of 55°S, between 180° and
     120°W, there were none in 2023. Sentinel-2's 10:30 local overpass time holds at mid-latitudes
     only; this far north the overpass is near local noon or later.
   - This makes the rule depend on the ranges. A future `incomplete` range starting between March
     and October would need the trailing day tested too, for ROIs in those zones.
   - Isolated missing dates are accepted rather than routed. Falling back for one date would send
     the whole year to the legacy collection to recover under 1% of its images.
4. **The store records both collections** in `geoemb:source_data`.

## Dev test

Run on 2026-11-19 in global-tessera-dev, through YE branch `dev/global-tessera-s2-c1` pinned to this
PR, with the `ingest-s2-roi-reflectance` flow. The area was `rois/zarrs/tiny_epsg5070.zarr`, about
4.5 × 6 km in western Minnesota spanning MGRS tiles 15TUM and 14TQS. Each run used 4 workers with
provenance on. Three runs went in parallel, each taking about 2.5–3 minutes:

| Run | Window | `collection` | Collection recorded in provenance | Dates |
|---|---|---|---|---|
| A | July 2023 | default | `sentinel-2-c1-l2a` | 12 |
| B | July 2023 | `sentinel-2-l2a` (control) | `sentinel-2-l2a` | 12 |
| C | July 2022 | default | `sentinel-2-l2a` (fallback) | 7 |

- **The offset is removed correctly.** A and B hold the same 12 dates, and every pixel came from the
  same source scene (`source_index` agrees everywhere). On every band, the median and mean difference
  is 0. At least 99.9% of pixels agree to within ±10 and at least 99.8% exactly; none differ by
  500 or more.
- **The remaining 0.15% of pixels** (5,007 of 3.4 million) are dark: a median of 32 in A against 51
  in B on `nir08`, where the band's median is 3,778. Reading one scene's files directly, Collection 1
  minus 1,000 equals the legacy COG on every pixel except a handful whose raw value is below 1,000,
  that is, negative reflectance. The two producers floor those differently
  ([ADR 020](020-boa-offset-applies-to-every-valid-dn.md)), and bilinear resampling of the 20 m
  bands spreads each difference to its neighbours.
- **The fallback works end to end.** Collection 1 has no July 2022 imagery over this area, and run C
  read the legacy collection and found 7 dates passing the coverage filter.

Provenance was on in every run, but no run had it off, so this test does not measure the cost of
provenance.

## Consistency checks

These ran on 2026-11-24, to test whether the switch changes what the ingest produces, beyond the
single month of the dev test above.

**Scene survey: what the ingest would read.** The survey covered six tiles (15TVF, 33UUP, 36MZE,
55HBU, 23KLP and 44QKE) in January and July of 2021 and 2023. In each tile-month it took up to four
datatakes: the two clearest, and two with 20–70% cloud. Each Collection 1 scene was paired with the
legacy copy `duplicates.select_preferred_duplicates` keeps, which is the copy the legacy ingest
reads. It compared SCL, red and nir over a 20 km square centred on the scene's data, 81 pairs in all.

- **In 2021 the legacy collection already serves Collection 1's pixels.** Element 84 added a
  baseline 05.00 copy of each 2021 scene to the legacy collection in 2023, beside the original
  (02.14 or 03.01). Duplicate selection prefers the newer baseline, so the legacy ingest reads the
  05.00 copy. That copy matches Collection 1 except on near-black pixels and in the one case below. The original copies differ by 400 DN in
  mean absolute red, but are never chosen.
- **79 of 81 pairs agree on SCL exactly**, and no pair changes a date's verdict at the 5%
  valid-coverage gate. Their red and nir match except on near-black pixels, where Collection 1
  minus 1,000 goes negative and the legacy COG is floored. `correct_boa_dn` floors both alike
  ([ADR 020](020-boa-offset-applies-to-every-valid-dn.md)).
- **One genuine difference favours Collection 1.** On 15TVF on 2021-01-21 the legacy collection
  holds two 05.00 reprocessings of one image, sequence `_1` and sequence `_2`. Duplicate selection
  keeps `_2`, which sits one pixel west of Collection 1: a mean absolute red difference of 568 DN,
  falling to 176 after a one-column shift. `_1` is identical to Collection 1. Collection 1 holds a
  single copy, so the ambiguity disappears.
- **The other mismatch was a pairing artefact.** 55HBU on 2023-07-19 has two datastrips, and both
  collections hold both.

**Dev ingest, 2021.** Same set-up as the dev test above (`tiny_epsg5070.zarr`, 4 workers,
provenance on), with Collection 1 and legacy arms for January and July 2021:

| Month | Dates (each arm) | `source_index` equal | Pixels exactly equal, per band | Pixels differing by ≥ 100 |
|---|---|---|---|---|
| January 2021 | 6 | 100% | ≥ 99.998% | 0 |
| July 2021 | 11 | 100% | ≥ 99.7% | ≤ 0.006% |

SCL is identical in both months. July's residue is the near-black floor difference described
above, spread by resampling, as in the 2023 test. Both arms record baseline 500 for every date.

Taken together, the switch leaves 2021 and 2023 effectively unchanged, apart from scenes like the
15TVF shift above. It does the same in any other
year where the legacy collection already holds Element 84's 05.xx copies. No embedding comparison was run, so the effect on
embeddings is unmeasured. It was judged unnecessary because at most 0.006% of the dev test area's
mosaic pixels differ by 100 DN or more.

## Consequences

- From April 2019 onward, except 2022 before mid-December, mosaics come from one baseline and one
  producer. Annual windows for 2019 and 2022 still read the legacy collection. Duplicate
  selection still runs, but in Collection 1 it only separates images; there are no competing copies.
- Mosaics barely change where the legacy collection already holds Element 84's 05.xx copies, as it
  did in every 2021 and 2023 scene checked, because duplicate selection already reads them (see
  Consistency checks). Other years were not checked.
  Where it held only a pre-04.00 original, pixels and scene classification differ slightly (686
  against 690 on 15TVF on 2020-07-10). The change therefore lands at a store boundary, like
  other input changes. The ingest code identity changes with this code, so an existing store will
  not resume under it without `allow_ingest_code_mismatch`.
- The `incomplete` ranges are a measured snapshot. When Element 84 backfills 2022 or the early
  years, rerun the date comparison and shrink the ranges.
- A Collection 1 window can lack an isolated date the legacy collection holds, such as 2024-01-11
  on some tiles.
- `provider="planetary-computer"` has no Collection 1 entry. It must be given
  `collection="sentinel-2-l2a"` explicitly, as with every unvalidated use of that provider.
