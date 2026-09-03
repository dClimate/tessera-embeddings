# How much of the product is optically thin, and where — 2026-08-12

The optical half of the coverage picture, measured at shard resolution so
`OPTICAL_THIN_MAX_OBS` can be argued about with numbers. The radar half is
[`radar_source_coverage_2026_08.md`](radar_source_coverage_2026_08.md); the
observation-count model this refines is
[`campaign-cost-model.md`](campaign-cost-model.md) §6.

Instrument: [`scripts/census_s2_coverage.py`](../../scripts/census_s2_coverage.py).

---

## 1. The headline

Every live shard the campaign will write, scored against four candidate lines.
**A cell is a zone-year** (1,008 of them); a cell is *majority below* when more
than half its live shards are.

15 and 40 are the lines that were asked about. 20 and 30 are carried at equal
weight throughout because the repo's own measured quality change — crisp at 30+,
noisy below about 20, over 15 cells — falls between them.

| | **15** | **20** | **30** | **40** |
|---|---:|---:|---:|---:|
| **cells** majority below | **5.6%** | **9.3%** | **20.5%** | **36.5%** |
| — count, of 1,008 zone-years | 56 | 94 | 207 | 368 |
| — weighted by zone footprint | 0.6% | 4.3% | 9.8% | 20.9% |
| — excluding 2017, of 896 | 3.0% | 4.8% | 12.9% | 28.9% |
| **shard-years** below | **2.95%** | **6.76%** | **18.38%** | **34.46%** |
| — count, of 3,248,577 | 95,927 | 219,453 | 597,013 | 1,119,452 |
| — chunk-years below (256 px) | 2.83% | 6.63% | 18.15% | 34.19% |
| — excluding 2017 | 0.72% | 2.76% | 12.46% | 28.51% |
| **shards** below in ALL nine years | 0.18% | 0.70% | 5.15% | 15.23% |
| — count, of 360,953 | 645 | 2,513 | 18,586 | 54,967 |
| shards below in **2017 alone** | 20.9% | 38.7% | 65.7% | 82.1% |
| shards below in **2025 alone** | 0.5% | 1.6% | 7.6% | 20.1% |

Shards and chunks agree to within 0.3 points, so the unit does not move the
answer. **Shards are the reportable unit** — the census resolves ~110 km, and a
2.56 km chunk figure would imply a precision the instrument does not have.

**The curve is steepest exactly where the quality evidence sits.** Moving the line
15 → 20 adds 3.8 points of flagged product; 20 → 30 adds 11.6; 30 → 40 adds 16.1.
So a line anywhere in the 20–30 band is the most sensitive to the census being
slightly wrong, and a line at 15 is the least.

## 2. At the 15 line this is mostly a 2017 problem

Sentinel-2B reached routine operations partway through 2017, so that year carries
roughly half the acquisitions of every year after it and behaves like a separate
dataset.

| year | <15 | <20 | <30 | <40 | shards with zero obs |
|---|---:|---:|---:|---:|---:|
| **2017** | **20.85%** | **38.70%** | **65.69%** | **82.05%** | 8,468 |
| 2018 | 0.78% | 3.24% | 14.06% | 30.73% | 842 |
| 2019 | 0.59% | 2.16% | 11.67% | 26.98% | 479 |
| 2020 | 0.76% | 2.27% | 11.10% | 26.43% | 523 |
| 2021 | 0.55% | 2.50% | 12.13% | 26.02% | 493 |
| 2022 | 0.84% | 3.64% | 14.21% | 29.76% | 238 |
| 2023 | 1.23% | 4.28% | 16.79% | 38.66% | 238 |
| 2024 | 0.48% | 2.41% | 12.15% | 29.38% | 238 |
| 2025 | 0.47% | 1.61% | 7.58% | 20.12% | 238 |

**Two thirds of the headline 2.95% is 2017 alone.** The 40 line does not
decompose the same way — dropping 2017 moves it only 34.5% → 28.5% — because
there the driver is ordinary cloud climatology over the tropics and the boreal
belt, not a commissioning gap.

## 3. Where

**Persistently below 15** (all nine years): 645 shards in 25 contiguous clusters.

| place | shards | centroid | zones |
|---|---:|---|---|
| Gabon – Republic of Congo, Ogooué basin | 151 | 2.6°S 12.2°E | 32S 33S |
| N Greenland & Canadian Arctic islands † | 107 | 82–84°N 20–80°W | 17N–27N |
| Coastal Ecuador, western Andean foothills | 96 | 1.5°S 79.0°W | 17S |
| Equatorial Guinea – northern Gabon | 96 | 0.5°N 11.5°E | 32N 33N |
| **Eastern Chukotka** *(see §5)* | 67 | 67°N, at 180° | 01N 60N |
| Papua New Guinea highlands | 30 | 7.0°S 143.5°E | 54S |
| Southern Cameroon | 30 | 2.5°N 12.0°E | 32N 33N |
| **Western Aleutians** *(see §5)* | 20 | 52°N 179°E | 60N |

† Not one cluster but every shard above 80°N, which the clustering splits across a
dozen small ice-cap fragments; the rest of the table is single contiguous clusters.

**Persistently below 40**: 54,967 shards, concentrated in three places that are
two thirds of the total — insular and mainland Southeast Asia (16,760), the
Amazon basin (13,783), and the Congo basin with the Gulf of Guinea (7,948). The
zones carrying the most are `48N` (3,195 shards, 36% of its footprint) and `49N`
(2,652, 36%), neither of which is majority-below.

## 4. Do not read the cell figures as area

**A cell is a unit of work, so counting cells gives every zone one vote
regardless of size.** Zone footprints run from **4** live shards to **9,132**;
the 43 smallest zones are 38% of the zone count and **3.9%** of the product.
Weighted by footprint, the cell figures fall from 5.6% → 0.6% and 36.5% → 20.9%.

The same trap bites the zone lists. Nineteen zones are majority-below-40 in every
year, but fourteen of them hold under 500 live shards and all nineteen together
are 2.8% of the product. At the 15 line the single zone that qualifies, `31S`, has
**six** live shards — four of six is not a finding.

*Quote the shard figures for how much of the product is thin, and the cell figures
for how many units of work carry the label.*

## 5. Zones 01 and 60 are not measurable this way

earth-search stores antimeridian-crossing granules as clipped fragments, so a
point query in zones `01N/01S/60N/60S` under-reports. Those four zones hold
**0.46%** of live shards but **20%** of the persistently-below-15 population —
the Chukotka and Aleutian rows in §3. At the 40 line the contamination is 1%.

**This may be a real thinness rather than an artifact.** The ingest path queries
the same catalogue the same way, so if the fragments defeat a point query they may
also defeat an ingest query. Settling it is cheap: compare a `grid:code` query for
an affected MGRS tile against a bbox query over the same ground.

Separately, **211 shards (0.06%) have no Sentinel-2 acquisitions at all**, 116 of
them antimeridian-suspect. Those produce no embeddings rather than thin ones, so
they are a coverage question, not a quality one.

## 6. Method, and what it was checked against

The quantity is the one the threshold is applied to: `s2_obs_count`, the number of
dates a pixel survives the SCL mask in a calendar year. Estimated as the campaign
cost model's optical census estimates it — per location, the sum over **distinct
acquisition dates** of `1 − eo:cloud_cover` — at ~110× its spatial resolution.

- **18,819 point queries**, one per 1° land bin, each returning the whole
  2017–2025 history in a single response. Bins come from the land-mask registry,
  so small islands get their own measurement instead of inheriting a mainland one.
- **Same catalogue as the ingest path** (earth-search `sentinel-2-l2a`). Item
  geometries there are true granule footprints — verified: swath-edge granules
  carry clipped polygons — so a point query returns only acquisitions that cover
  the point.
- **360,953 live shards** enumerated with `ingest.land_mask.build_zone_coverage`,
  the function the land-mask store is built with, so these are exactly the tiles
  the campaign will write. 21,439,830 live 256 px chunks, mean 59.4 of 64 per shard.
- Each shard takes the nearest census point: median **39 km**, p95 **62 km**.

**Checked against the store, not asserted.** 1,079 chunks read from the dev global
store across 22 zone-years in 15 zones from 40°S to 85°N, comparing written
`s2_obs_count` with the census at the same place:

| | store / census |
|---|---:|
| median | **1.04** |
| quartiles | 0.95 – 1.19 |
| by band, −20…0 / 0…20 / 20…40 / 40…60 / 60…85 | 1.23 / 1.04 / 1.03 / 1.07 / 1.01 |

So the census sits within a few percent of what the pipeline writes, and runs
slightly **low** — which makes every percentage above a mild over-estimate rather
than an under-estimate.

The agreement is not guaranteed by construction and the direction was not
predictable: `eo:cloud_cover` is a granule average that cannot see cloud
clustering; the mask drops SCL 2 and 3 (dark area, cloud shadow) which
`eo:cloud_cover` does not count, pushing the store down; and it keeps SCL 10 and
11 (thin cirrus, snow) which `eo:cloud_cover` does count as cloudy for cirrus,
pushing the store up. The measurement is what settles it.

> **Noted in passing, not fixed here:** `config/satellites.py` describes
> `S2_SCL_INVALID_CLASSES = {0, 1, 2, 3, 8, 9}` as "(nodata, saturated, cloud
> shadow, cloud, snow/ice)", but class 11 (snow) is **not** in the set and is
> therefore kept. The comment is wrong about snow; the code is what the validation
> above measured.

## 7. What this does not settle

- **The measurement resolves ~110 km, not 20 km.** Shard-level figures inherit a
  1° census value and are not independent per shard. The aggregates are sound; a
  single named shard is not.
- **The orbit sidelap is aliased.** Revisit roughly doubles in the ~50 km overlap
  strips, visible as diagonal banding across the boreal belt at the 40 line. A 1°
  census samples that coarsely, so cluster *edges* there are approximate.
- **Within-shard spread is unmeasured.** The validation bounds the aggregate bias,
  not the distribution of pixels inside a shard around its granule-mean estimate.
- **This is optical only.** Radar absence alone does not visibly degrade the
  embeddings ([`final-data-validation-plan.md`](final-data-validation-plan.md) §4).

## 8. What the numbers frame

Nothing here is fixable by re-running: the observations do not exist, and the
per-pixel counts are already written beside every embedding. The decision is about
labelling. The four lines are compared in §1; that table is the only copy, so it
cannot drift out of step with a second one.

15 sits deep in the left tail, where moving it changes almost nothing; 40 sits just
below the mode, where the distribution is steepest and a line is least stable.
That is the quantitative form of *"a label that covers a third of the product is
not a label"* ([`final-data-validation-plan.md`](final-data-validation-plan.md) §4)
— and it measures the claim, which had not been: at 40 it is 34.5% of shard-years,
28.5% excluding 2017.

Open questions, cheapest first: whether 2017 should carry a different label from
the other eight years; whether the 645 permanently-thin shards are few enough to
name individually; and §5's antimeridian question, which is the only one that could
change a published number.
