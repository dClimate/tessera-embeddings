# The published optical registry: what it holds, and what the first live run corrected

**Status: the schema and the producer are built and exercised against a live campaign run. The
consumer side — a pipeline that can act on the work list this produces — is not built.**

The registry is a Parquet dataset beside the embeddings store, one row per 2048-pixel tile per year,
written as each campaign cell lands. It exists so two questions can be answered without opening a
petabyte-scale store: a user's *"is my area covered, and how well"*, and a later infill campaign's
*"which tiles were short of imagery, and would more of it help"*.

Location and layout: `<store-stem>.registry/parts/zone=<Z>/year=<Y>/<run_id>.parquet`, a **sibling**
of the store rather than a path inside it, because Icechunk owns every key under its own prefix and
its garbage collection reconciles that prefix against its own manifests.

---

## 1. What the first live run found

Dispatched 2026-08-19 against `tessera-radar` in `global-tessera-dev`: zone 16S, years 2022 then
2021, 17 live tiles each, mosaics already present. Both cells landed and both published a part. The
run exposed four defects, and the most serious one was invisible in every log except the one naming
the path it wrote to.

**The registry was published beside the wrong store.** Parts landed at
`.../global/tessera.registry/...` while the run was filling `tessera-radar.icechunk`.
`BucketPaths.optical_registry()` called `global_store()` with no argument, so it derived from the
*default* repo basename whatever store the run was writing. The part was valid, 17 rows, and the log
said it had been published — it simply described a zone-year that the store beside it does not
contain. That is worse than an overwrite: two stores' parts merge into one dataset under the same
`zone=`/`year=` keys, and no column can tell them apart, so the error is unfalsifiable from the
artifact a consumer was told to read *instead of* the store. Fixed by making the method take the
resolved store URI as a **required** argument — a default would reintroduce the same silent failure
the first time a caller forgot it.

**The depth rule never reached the rows.** Every published row carried `optical_min_obs = None`
while the store root declares 15. `fill_zones_sequential` is a second entry into
`assemble_zone_year`, and it never forwarded the value. The columns that exist to say *how close a
refusal came* — `obs_max`, `median_obs_where_any` — are distances from that line, so without it they
cannot be read at all.

**Nothing was recorded for tiles that were only partly refused.** All 17 rows read
`embedded = true` with every measurement null. The actor accumulates refusal reasons over a shard's
strips on the success path as well as the refused one, so a shard that lost part of its land to the
depth gate has always known how much — but only the wholly-refused branch wrote a record. A tile 60%
refused therefore published as covered ground. Those partial refusals are the **bulk** of an infill
work list, not a refinement of it.

**No row said where its tile was.** The access request promises a bounding box per row and none was
produced.

## 2. What the same run confirmed works

- The dataset opens and reads **across parts** with hive partitioning; `zone` (string) and `year`
  (int32) come from the path with no type collision — the two red-team defects found before launch
  stayed fixed.
- One row per live tile, 17 of 17, with `run_id` and `assembled_at` present on every row.
- Every consistency invariant held: reasons summing to `refused_px`, `eligible_px <= chunk_px`,
  `px_with_any_optical <= eligible_px`.
- `tag_year_complete` correctly **refused** to mint `year-2021-complete` from a 9-zone subset, and
  `year-2022-complete` from a 1-zone subset. No globally-named completion tag exists.

## 3. Why 16S could not exercise the refusal columns

16S/2022 refused **nothing**: all 67,728,022 embedded pixels met the depth-15 rule, and
`s1_free_px` was 0 despite the ascending orbit carrying only 4 of 12 months, because the descending
orbit covered all 12. An empty infill work list is the *correct* answer for such a cell — but it
means the run could not distinguish a registry that measures and finds zero from one that measures
nothing. That distinction is now explicit: a measured zero is written as `0`, and only a tile whose
coverage record never arrived stays null.

A second verification run was dispatched against 09S/2021 (10 live tiles, western Amazon) and
03S/2024 (4 live tiles) for exactly that reason — small, fresh cells over persistently cloudy land.

## 3a. The second run: verified, and it found one more thing

09S/2021 (10 tiles, Pitcairn) and 03S/2024 (4 tiles) landed against the fixed code. The dataset now
reads as one table across both parts — 14 rows, two zones, two years, partition keys typed from the
path, and **zero consistency violations**: refusal reasons sum to `refused_px`, `eligible_px` never
exceeds `chunk_px`, `px_with_any_optical` never exceeds `eligible_px`, latitudes ordered.

Four of the fourteen tiles carry refused pixels while being embedded — the case that previously
recorded nothing — and the worst is 03S/2024 `chunk_59_11`: 42,212 refused of 4,194,304 eligible,
40,852 of them imaged but thin and 1,360 never imaged at all. Both zones are wholly radar-free
(`px_with_any_radar = 0`, `radar_rule_enforced = false`), which is correct: neither has OPERA RTC
coverage, and the campaign embeds radar-free land by policy.

**That run also showed why `median_obs_where_any` had to be replaced as the ranking key.**
`chunk_59_11`'s footprint median is 37 against a line of 15, because the median describes the pixels
that PASSED. Its `median_obs_where_thin` is 5.0 — the refused pixels are about ten observations
short. Ranked by footprint median it would have sat below tiles with no refusals at all; ranked by
thin median it is correctly first.

**And it confirmed the schema-evolution hazard on live data.** The two parts were written by
different builds — 09S by `10f31a1`, 03S by `22b2352` — and `median_obs_where_thin` exists in only
the newer one. The merged read shows the column *because* `zone=03S` sorts before `zone=09S`, so the
newer part's schema is the one inferred. Had the older zone sorted first the column would have
vanished from the read with no error. That is why `dataset_schema()` exists and why any whole-dataset
read must pass it. The `code_commit` column is what made the diagnosis possible, which is its first
real use.

## 3b. One failure that was not a registry defect

03S/2024's first attempt lost all four tiles to
`TypeError: _coverage_record() missing 1 required keyword-only argument`. No commit on the branch
contains that inconsistency — the function and both its call sites move together in every one — and
the tarball on S3 afterwards was correct. A pin bump pushed **while that cell was mid-ingest** had
CI replace the Ray code tarball between its ingest and its fill. Re-dispatched against a stable
tarball, the same cell completed.

Two things worth keeping. The failure direction was right: a required keyword-only argument with no
default produced four loud tile failures instead of a registry of rows measured against a line
nobody recorded. And a green test suite cannot see this class at all — the tree was internally
consistent; the inconsistency was created by deployment.

## 4. Decisions worth keeping

**Null means "not measured", never zero.** A resumed tile is a synthetic success carrying no
coverage record, and an unreadable marker leaves none. Writing zero there would assert a measurement
nobody took, which is the misreading the whole artifact exists to prevent.

**`embedded` is not coverage on its own.** It says the tile holds embeddings *at all*. "Is my area
covered" is `embedded` together with `refused_px`, and a reader using the first alone will overstate
coverage.

**The schema is declared, never inferred**, and `zone`/`year` are partition keys only — both because
the failures live in the merge, not in any one part.

**One `_coverage_record`, two call sites.** A refused shard's marker and an embedded shard's result
are built by the same function, because comparing the two rows is the entire point and two copies of
those expressions is how they stop meaning the same thing.

**The bounding box is not derived in the registry module.** It comes from
`zone_grid.tile_range_bbox_wgs84`, whose densified 64-sample perimeter is *measured* to contain the
true envelope to within one pixel and which handles the antimeridian. The ingest's catalogue
preflight already depended on that guarantee; the function moved so there is one implementation
rather than two copies of a grid convention. Rows for zones 01 and 60 carry `west > east` per
GeoJSON and STAC — a consumer filtering `west <= lon <= east` silently drops them.

**A part is published only after its cell commits**, and the write is quiet on failure. Before the
commit, failing loudly costs nothing; after it, raising would fail complete work over its index. Every
column is derivable from the store, which is what makes a lost part recoverable and that trade
correct rather than lazy.

## 5. What is still missing

- **The pipeline cannot consume the work list.** Threading an optional set of tile labels through a
  fill is a real change, not a small one: the year record is replaced wholesale and the skipped set
  is derived as live-minus-staged.
- **Multi-contribution year provenance.** A refilled cell should append a contribution — run id,
  timestamp, code identity, input coverage, and the tile labels it wrote — rather than replace the
  original. Sketched and agreed; not built. Safe to do now, since no campaign has published.
- **Compaction.** Parts are keyed by run so a refill adds one rather than replacing the original.
  Reading the raw parts double-counts a twice-filled cell unless the reader keeps the latest
  `assembled_at` per `(zone, year, tile)`.
