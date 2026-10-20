# Split datastrips: one pass, two granules, one kept

**Status: the selection rule is fixed in `ingest/split-datastrips`. The dev fills and the remediation
decision are open (§8).**

Sentinel-2 duplicate selection (`ingest/duplicates.py`) treated two granules of one pass as two copies
of one image whenever a datastrip boundary crossed the tile, kept one, and dropped the other's ground
for that day. Every selection rule the ingest has had since 2026-08-05 did this, so every published
v1.1 cell is exposed. Measured from the public catalogue, it cost about **1.2% of clear optical
observations** worldwide, concentrated where a datastrip boundary crosses a tile on most passes.

## 1. The terms

- **Datatake**: one continuous imaging pass of one satellite, named by `s2:datatake_id` (mission,
  sensing start, absolute orbit, then the processing baseline).
- **Datastrip**: the segment of a datatake that ESA processes as one unit, named by `s2:datastrip_id`
  (mission, processing centre, processing time, the datastrip's **sensing start**, the baseline). A
  datatake is cut into several datastrips.
- **Granule**: the part of one datastrip falling on one 110 km MGRS tile. Earth Search publishes one
  item per granule.
- **Reprocessing**: ESA reprocessing a datastrip at a newer processing baseline. Earth Search then
  holds both items for the tile-date.

Where a datastrip boundary crosses a tile, the tile receives one granule per datastrip on the same
pass. Both share the datatake and the baseline and cover different parts of the tile.

## 2. The defect

`acquisition_identity` named an acquisition by the datatake alone, so the two granules of a split pass
clustered as one acquisition. At equal baselines the ranking falls through to `s2:sequence`, which is
the order Earth Search published the items in and says nothing about coverage. The other granule
became a fallback rung, read only if the chosen one failed to read.

The three cases that surfaced it, from Earth Search on 2026-10-20, run through the library's own
selector with the zone ROI's grouping longitude and the S2 driver's read keys:

| tile, day | kept | dropped | ground only the dropped granule covered |
|---|---|---|---|
| 18NUN, 2025-06-19 | `_1_`, 94% no-data | `_0_`, 0% no-data | 94% of the tile |
| 18PUQ, 2025-02-14 | `_1_`, 53% no-data | `_0_`, 29% no-data | 53% |
| 18FWK, 2024-09-18 | `_1_`, 97% no-data | `_0_`, 53% no-data | 44% |

**How it surfaced.** While sizing the Latin America update, a measurement agent rebuilt per-date SCL
validity for 4,096-px windows by querying each window's own bounding box, and found production's
`s2_obs_count` one observation short over whole tiles. On 18N/2025 `chunk_419_10` all 183,764 holes
trace to 2025-06-19, the 18NUN pair, and that date was their only clear observation (the cell was
filled at an optical line of 1). Six of the nine other dates the agent found missing are also splits.

**Why the replication disagreed: the loss depends on the query's extent.** The selector sees only
the items a query returns. A window query that misses the sliver returns only the large granule, which
is then kept. The zone-wide query production issues returns both, and the sliver wins. So a small
single-area ROI could keep a date the campaign lost on the same ground.

**Since when.** The two granules' catalogue timestamps are 7–8 s apart, so every rule since
`b2982b1f` (2026-08-05) clustered them: (tile, solar day) from `b2982b1f`, timestamps within 120 s
from `d0451836`, the datatake from `a5015376`. Before `b2982b1f` there was no selection and both
granules were fused. Every published v1.1 cell ran later code.

## 3. The census

**Sample.** One point per 5° land bin, taken from the optical census grid
(`scripts/scoping/census_s2_coverage.py`, one registry land cell per 1° bin): 1,127 points, each
weighted by its bin's land area (registry cells × cos latitude). For each point, every
`sentinel-2-l2a` item whose footprint covers it; for every MGRS tile those items belong to (1,506),
every item 2017–2025. **3,352,894 items**, fetched with the `fields` extension (no assets, no
geometry) at three requests a second.

**Selection.** Each tile's nine years go through the S2 ingest's own steps: solar-day normalisation
with the zone ROI's grouping longitude, then `select_preferred_duplicates` with the S2 driver's read
keys and harmonisation. Selecting over a whole tile reproduces what a zone-wide query presents.

**Control for the missing assets.** Without assets every copy ranks as equally incomplete and remote,
so the asset terms tie. Ten tile-years were run on full items and on asset-stripped copies. Eight gave
identical decisions. In the other two (17QRE and 25XEM, 2018) the asset terms picked a different one
of two identical copies of the same datastrip (same sensing start, baseline and no-data), so coverage
is unaffected.

**Loss at a point.** A solar day is lost at a point when some granule covering the point was published
that day but the rule kept none of the granules covering it. A day's clear-sky weight is 1 − the mean
`eo:cloud_cover` of the granules covering the point, as in the optical census. Scene cloud cover is an
estimate of the pixel's chance of being clear, not an SCL measurement.

## 4. What the census found

**Every item carries a readable `s2:datastrip_id` and `s2:datatake_id`.**

**Shapes.** Per tile, datatake and baseline: 3,123,588 groups with one datastrip, 87,035 with two, 25
with three. 52,967 groups hold one datastrip published twice.

**Splits are everywhere.** 3.3% of tile-dates hold a split, and the share barely moves by year
(2.2–3.1%), mission (S2A 2.6%, S2B 2.8%, S2C 3.5%), processing centre, region or latitude.

**Reprocessing keeps each datastrip's sensing start.** Comparing each older baseline's datastrip
sensing starts with the newest baseline's, per tile and datatake:

| shape | pairs | meaning |
|---|---|---|
| stable | 877,083 (99.66%) | the same starts |
| lacks | 2,622 (0.30%) | the newest reprocessing has fewer datastrips and misses one; 1,699 of them in 2018 |
| recut | 182 | the sets differ without the newest having fewer; all but five cover visibly different ground (median no-data difference 53 points), so they are a different datastrip rather than a re-cut |
| superset | 209 | the newest has extra datastrips |

**The loss under the old rule**, land-weighted:

| | |
|---|---|
| share of clear observations lost | 1.2% |
| lost dates per pixel-year | 1.16 |
| lost clear observations per pixel-year | 0.62, of about 51 |
| land losing at least one date in a year | 50% |
| land losing more than 1% / 5% / 10% / 25% of its clear observations | 35% / 4.2% / 1.1% / 0.1% |

By year the share lost runs 0.8% (2017) to 1.4%; by region 1.0%
(Asia) to 1.9% (Oceania); by latitude band 0.4% (north of 80°N) to 1.7% (20–40°S).

**The tail is systematic, not random.** Where a datastrip boundary crosses a tile on most passes, the
loss repeats every year. On 59WNQ (Chukotka) 76 of the 210 passes in 2022 are split, and on those the
rule kept the near-empty granule. The worst sampled points lose 16–21% of their clear observations every
year: near 62°N 57°E (the Urals), the Pribilof Islands, Chukotka. 16% of all lost observations come
from point-years losing more than 10%.

**The line of 15.** From expected clear counts rather than per-pixel ones, the loss pushes 0.05% of
land below 15 clear observations. The store measurement (§7) replaces this.

## 5. The fix

`acquisition_identity` now appends the datastrip's sensing start to the datatake. Everything that
asks whether two items are one acquisition goes through it — clustering, ranking within an
acquisition, and the fallback ladder's attribution and placement — so nothing else changed.

- A split pass keeps every granule.
- Copies of one datastrip still compete, and the newest baseline still wins (the stable row).
- One datastrip published twice still reduces to one copy.
- Where a reprocessing omits a datastrip (the lacks row), the older copy of that datastrip is kept
  beside the newer ones. The day then mixes baselines across datastrips, never within one: coverage
  outranks one baseline per pass ([ADR 029](../decisions/029-an-acquisition-is-one-datastrip-of-one-pass.md)).
- An item whose datastrip cannot be read is named by its datatake alone, as before. None was found.

**Equivalence gate.** Over the whole corpus the fixed rule decides differently on 64,811 tile-dates,
and every one of them holds a datatake with more than one datastrip. The branch's own code agrees with
the simulated rule on all 3,352,894 items. It never drops an item the old rule kept; it adds 65,325.

**What remains is not this defect.** The fixed rule still loses 0.075% of clear observations at the
sample points, almost all of it one datastrip at two baselines whose newer footprint is a few hundred
metres narrower at a swath edge (no-data differing by about 0.25 points). That comes from preferring
the newest baseline and is unchanged by this fix.

**Tests** (`tests/unit/ingest/test_duplicate_granules.py`, `TestADatastripSplitIsTwoAcquisitions`) use
the catalogue's own values for 18NUN, a 33UUP split that survived reprocessing, and a 17QRE pass whose
reprocessing omitted a datastrip. Five of the seven fail on the old rule.

## 6. Radar is not affected

The OPERA path (`ingest/opera_query.py`, `ingest/s1_roi.py`) does not call the selector. Its only
item-level deduplication is by exact granule id across the two halves of an antimeridian query, so
distinct bursts of a date are never collapsed.

## 7. What it did to the published store

Measured exactly at the 1,127 sample points rather than estimated. For every day the old rule lost at a
point, the scene classification was read at the point from each granule the fix restores (12,959 reads
of public SCL COGs, 14 unreadable); the published store's count and embedded state were read at the
point's pixel, and the tile's own optical line from the published registry.

| land-weighted, all nine years | |
|---|---|
| valid optical observations lost per pixel-year | 0.62 (the cloud-weighted estimate in §4 was 0.62) |
| embedded pixel-years whose sequence is one or more observations short | 36% |
| pixel-years refused only because of the loss (count below the tile's line, count plus the lost observations at or above it) | 6 of 10,121 sampled, about 0.03% |

So the loss mostly changed embeddings, slightly, rather than coverage: almost every published tile ran
at the line of 15, and a lost date pushes a pixel across it rarely. Three of the six refusals are in
2017, the thinnest year. Six events bound the refusal figure only loosely.

## 8. Dev verification

Paired arms on global-tessera-dev, `dev/global-tessera-dupsel-control` (YE main, TE `26136f5b`) and
`dev/global-tessera-dupsel-fix` (TE `24a9c92d`), through the campaign path on one tile: 18N
`chunk_421_8`, 2025, chosen to straddle the edge of the 18NUN sliver so that its query returns both
granules. Run locally before launch, the year's query for that tile drops three granules under the old
rule and none under the fix. Each mosaic's ingest code identity matched its arm's code computed locally.

**Mosaics.** The same 81 dates in both arms, and every band byte-identical on the dates without a
split. On 2025-06-19 the control holds data on 19% of the tile and the fix on all of it: 3,388,172
pixels gained, 1,314,548 of them clear, every one inside a dropped granule's footprint, none lost.

**Where both granules cover the ground, the fix can take the pixel from the other one.** Fusion is
clearest first (`solar-day-fusion-order.md`), so the restored granule wins wherever it sorts ahead. Two
datastrips of one pass do not agree exactly, because atmospheric correction runs per datastrip: on
clear ground the median red difference is 19 DN (0.19% reflectance), the 90th percentile 100. On
2025-10-27 the differing pixels are all cloud or cirrus in the control; cirrus is valid for the model,
so those reach it.

**Fills: not yet run.** No g6e.xlarge capacity was available in us-west-2b for twenty minutes, and the
arms were cancelled. Open: the fills' counts and embeddings against the mosaics, the size of the
embedding change, and the remediation decision (fix forward only, a `holes` update, or a `shards`
rewrite of the worst cells).
