# Data validation — the campaign's per-cell gate and closing sweep

**Status: PLAN, not yet executed.** Revised after review. Delete or fold into a measurement record
once it has run.

Every rung of the pre-campaign programme measured throughput, cost, or recovery. **None of them
asked whether the published pixels are right.** This is what does.

The instrument exists and has been exercised on 15 zone-years of the dev store. It is what found the
observation-count seams at tile edges, a dimension check that would have failed every healthy cell,
and a cell published from six months of optical input. Two things do not exist yet: the in-campaign
gate of §3, and the roll-up of §7.

---

## 1. What "exhaustive" can and cannot mean

Agree this first, because the word does not survive contact with the scale.

The campaign publishes roughly **112 land zones × 9 years ≈ 1,000 zone-years**, each about 63
gigapixels with 128 dimensions per pixel. Reading every stored byte is not a budget question, it is
an impossibility — the whole product, several petabytes, through one process.

So validation is exhaustive along the axes where that is achievable, and explicitly sampled where it
is not:

| axis | coverage | why |
|---|---|---|
| **cells** | **every one** | a per-cell defect is invisible in an aggregate |
| **coverage / placement** | **exact, every shard** | METADATA — §2. No pixel reads at all |
| **provenance reconciliation** | **exact, every cell** | attribute reads |
| **pixels** | **sampled — one 256-px chunk per 2048-px shard, 1/64 = 1.6% of each shard** | the only axis where the budget binds |
| **seam boundaries** | sampled per axis | each is a pixel read on both sides |
| **native-resolution windows** | a handful per cell | for a human to look at |

**Do not describe the result as "every pixel checked".** The honest claim is: *every cell, every
shard's existence and placement exactly, and 1.6% of every shard's pixels systematically sampled.*

## 2. Why the exact half is free

**Which shards exist is metadata, not data.** The arrays are sharded, so one chunk-manifest entry is
one 2048-px shard. Walking the manifest answers "which shards hold data" exactly, for a whole
zone-year, in about a second with no pixel reads.

That is what makes the two strongest checks nearly free — the coverage cross-tabulation against the
frozen land mask, and **placement in its unambiguous half**: a shard written where the mask says no
land is misplacement, full stop.

The pixel budget is then aimed only at shards already known to exist.

## 3. WHEN it runs: a per-cell gate, then one closing sweep

Two passes, and the first is the one that changes the campaign's code.

### 3a. Per-cell, inside the campaign, BEFORE the tag

**Run the exact half as the last step of every cell, and raise on a blocking finding.** A defect
found on cell 3 of 1,000 is a fix; the same defect found at the end is 1,000 refills.

**Where exactly: in the gap between the year-attribute commit and `tag_zone_year`.** That placement
is deliberate and is worth more than the convenience of a separate script:

* The data and provenance are committed, so validation can read the real thing.
* **A cell that fails is never tagged.** Pendingness is decided by the completion mark and the tag,
  so an untagged cell reads as unfinished to the dispatch queue and is retried — rather than entering
  the published set with a defect nobody blocked.
* **We know that state is recoverable, because F2 drilled it.** A cell left committed-but-untagged is
  exactly the mid-commit crash state, and the recovery path re-lands it. This gate reuses a state
  whose recovery is measured rather than inventing one.

Requirements on the mechanism:

* **It raises.** Not a log line, not a warning — a deterministic error, so the failure is visible in
  the run's state and in the campaign's own failed-run monitoring.
* **The error names the cell and the finding**, so triage needs no log archaeology.
* **It must not loop.** A deterministic defect will fail every retry, so the gate must sit under the
  existing per-cell attempt budget; a cell that exhausts it is failed and left for a human, not
  re-dispatched forever.
* **Only the exact half runs here.** Coverage cross-tab, reconciliation, placement. No pixel reads,
  so it adds about a second to a multi-hour cell.

### 3b. Whole-globe, once, at the end

The full instrument — pixels, seams, detail windows, dimension health — over every published cell,
producing the roll-up of §7. One consistent snapshot of the finished product, and the pass that
catches anything the cheap gate cannot see.

## 4. Per-cell pass criteria

**The four that gate a cell in 3a** — all exact, all metadata:

1. **The coverage reconciliation closes**: `written + skipped == live`.
2. **Zero shards written outside the land mask.**
3. **`input_coverage` shows a full window and `relaxed: false`.** A cell published from a partial
   input through the coverage relaxation is the failure class P7 produced by accident.
4. **The completion mark and the run record agree** — the year is marked and carries a record.

**The additional ones in the closing sweep of 3b**, which need pixels:

5. **No constant embedding dimension**, and the **scale-setting shares sum to 1.0** — the invariant
   confirming the data was quantized the way every reader assumes.
6. **No seam on the embeddings**, on either axis — see §5 for why this one and not the other.
7. **The detail windows look like the Earth.** Not reducible to a statistic, and the check that
   caught the thin cells. Judge against that cell's own optical depth — §6.

## 5. The two seam rows are not the same finding

This was raised in review and the distinction is the whole point, so it is stated here rather than
left implicit.

**The observation-count seams are upstream and out of our hands.** Neighbouring tiles keep different
date sets after cloud masking, so the count genuinely steps at a tile edge. It appeared on every cell
measured, uniformly, and it is a property of what the archive delivered. **Do not report it per cell
and do not treat it as a defect** — it will otherwise fire a thousand times and bury everything else.

**The embedding seams are ours.** The reason is the grid: **shard boundaries are OUR grid**, not the
provider's. Our 2048-px shards are our inference tiles, written by our own index arithmetic over a
fork/merge write. An upstream defect would align to MGRS tiles or to swath edges — it has no reason
to align to a boundary we invented. So a discontinuity that appears **at our shard boundaries and
nowhere else** implicates our assembly, and this is the only detector for it.

That is why criterion 6 covers the embeddings only, and why an embedding seam blocks while a count
seam is ignored.

Two further guards against over-reading it, both already in the instrument: the verdict rests on the
**median across many boundaries**, because a single boundary falling on a coastline or field edge is
genuinely a discontinuity in the world; and the **edge-distance profile** separates a placement defect
(a spike at the boundary only) from a model given less context near its tile edge (a rise over
several pixels), which need opposite responses.

## 6. A thin cell is not a broken cell

Optical depth varies by an order of magnitude across the globe, and **picture quality tracks optical
depth and nothing else** — measured across 15 cells: crisp, field-boundary-sharp imagery at 30+ valid
observations per pixel; indistinguishable from noise below about 20. Radar absence alone does not
visibly degrade the embeddings.

A sub-Antarctic or atoll cell can be legitimately noise-like. `OPTICAL_LIGHT_MAX_OBS` records that on
the cell — see §8 for exactly what it measures — so the answer is on the record rather than in
someone's reading of a picture. **Publish thin cells; label them.**

## 7. THE MISSING PIECE: a roll-up for the closing sweep

**A thousand per-cell verdicts is not a validation, it is a pile.** The tool prints a human-readable
verdict and exits non-zero on any disagreement; it has **no machine-readable mode**. Without one the
closing sweep gets skimmed or skipped.

What is needed, in the cheapest form that works:

* **A `--json` mode on `embedding_reality_check.py`** — per check a stable slug, its status, and its
  named subjects. `campaign_health.py` already implements exactly this projection, so it is a pattern
  to copy rather than a design to invent.
* **A driver** over every completed cell, emitting:
  * a **one-line-per-cell table** — written, absent, reconciliation, optical range, seam medians;
  * an **exception list**, only cells with a finding, worst first;
  * **aggregate distributions** for what is only interpretable in bulk — seam median across all cells,
    optical-light share, absent-tile fraction.

The aggregate half matters as much as the exceptions, with one caution: **a consistent offset is not
evidence of an effect until the noise floor is known.** The current set leans about 3% above 1.0 on
seam medians in 7 of 9 cells, far inside tolerance, with no noise floor measured. The roll-up should
surface that kind of pattern for judgement, not grade it.

## 8. What `OPTICAL_LIGHT_MAX_OBS` actually measures

Asked in review; recorded because the granularity is easy to assume wrongly.

* **Applied per PIXEL.** In the inference actor, over embedded pixels only:
  `s2_obs_count < OPTICAL_LIGHT_MAX_OBS`. Currently **40**.
* **Counted per CHUNK** — one inference chunk is one 2048-px shard — and returned as
  `s2_light_pixels` on that chunk's result.
* **Published per CELL.** Aggregated into the year's `radar_coverage` record as `s2_light_px`,
  `s2_light_pct` (share of the cell's embedded area), and `s2_light_below_obs` recording the
  threshold in force, so a later change of the number does not silently reinterpret old records.
* **Nothing is marked at shard or chunk level in the store.** The per-pixel `s2_obs_count` array is
  already published, so any granularity can be recomputed by a consumer at any time.

**Consequence, and the one gap.** A cell that is half thin reads about 50% — which correctly captures
a sub-zonal split, so the figure does not hide the case where a zone spans the tropics into temperate
latitudes. But it does **not localise**: the percentage says how much, never where. Radar has a
coarse locator for this (`tiles_fully_s1_free`); optical has no equivalent. If localisation is wanted,
the cheapest addition is a per-cell count of tiles whose embedded area is *entirely* optical-light,
mirroring the radar field. Not proposed here — flagged as the available option.

**It is a reporting line only.** Nothing refuses a cell for being under it, and raising or lowering it
changes what future summaries say and never what is published, so it is a config change rather than a
migration.

## 9. Triage: what blocks

| finding | disposition |
|---|---|
| shard outside the land mask | **BLOCKS at 3a.** Placement defect; cell stays untagged |
| reconciliation does not close | **BLOCKS at 3a** |
| `relaxed: true` or unexplained absent months | **BLOCKS at 3a.** The input was not what the cell claims |
| completion mark without a run record | **BLOCKS at 3a** |
| seam median outside 0.80–1.25 on **embeddings** | **BLOCKS at 3b.** Our grid, our defect — §5 |
| scale-setting shares far from 1.0 | **BLOCKS at 3b.** Not quantized as readers assume |
| constant embedding dimension | **BLOCKS at 3b.** Dead output |
| high optical-light share | **RECORD and PUBLISH.** A property of the input, not a fault |
| absent land tiles explained by skips | **RECORD.** Expected; the skip record is the evidence |
| **observation-count seams** | **IGNORE.** Upstream, on every cell — §5 |

A corrected cell needs a **fresh tag name**: icechunk tags are write-once forever, so a refill cannot
re-pin the canonical name. `scripts/reopen_zone_year.py` clears the completion mark and pins a fresh
tag; `scripts/drop_run_field.py` removes a provenance field known to be wrong.

## 10. Known limitations of the instrument

State these in whatever report the sweep produces, so nobody reads more into a pass than it carries:

* **The pixel sample is 1.6% of each shard**, systematically placed (one inner chunk per shard, aimed
  at land). A defect confined to a smaller area can escape it.
* **The seam test cannot separate terrain from misplacement on a single boundary.** It rests on the
  median over many, and on the edge profile.
* **It validates against the land mask and the store's own provenance, not against ground truth.** It
  can show the data is self-consistent and plausibly Earth; it cannot show the embeddings are
  *correct* for the surface.
* **A cell with too few written shards yields too few boundaries to decide** — reported as
  undetermined rather than as a pass.
* **One unexplained reproducible outlier exists.** 40S/2024 shows pixel differences ~50% larger at
  shard boundaries than 128 px inside, across two independent assembly runs, while 12 other cells sit
  at 0.91–1.10. Its medians pass. Cause unknown; the input-depth hypothesis was tested and refuted.
  The roll-up would show whether it is a class or a one-off.

## 11. Tooling inventory

| what | where |
|---|---|
| the reader and renderer | `yield-embeddings/scripts/embedding_reality_check.py` |
| the decisions and thresholds (unit-tested against synthetic arrays) | `yield-embeddings/src/yield_embeddings/domain/embedding_audit.py` |
| per-cell structural + placement validation | `yield-embeddings/scripts/validate_zone_group.py` |
| campaign-day monitoring, incl. published-input-coverage and shard-placement checks | `yield-embeddings/scripts/campaign_health.py` |
| clear a completion mark / pin a fresh tag | `yield-embeddings/scripts/reopen_zone_year.py` |
| remove a provenance field known to be wrong | `yield-embeddings/scripts/drop_run_field.py` |

## 12. Cost

Measured on the dev store, in-region where transfer is free:

| cell | written shards | cost | wall |
|---|---:|---:|---:|
| 37N/2021 (largest measured) | 8,714 | **$0.028** | 3 min |
| 60N/2020 | 527 | $0.002 | 52 s |
| ten-cell sweep (6 to 9,050 shards) | — | **$0.048 total** | — |

**The closing sweep over ~1,000 cells: order $15–25 and a few hours of wall clock, in-region.**

**The per-cell gate of 3a is effectively free** — metadata only, about a second per cell against a
multi-hour fill.

Two operational notes:

* **Run in-region.** The same reads from outside cost about 6x more, because egress dominates: 37N is
  $0.028 in-region and $0.176 over the internet. At campaign scale that is twenty dollars against a
  hundred and twenty.
* **Raising the pixel depth scales linearly in chunks read.** Reading all 64 chunks per shard instead
  of one would be ~64x the pixel cost and wall clock, which is the boundary this plan declines to
  cross — the sample is systematic rather than random, so a defect large enough to matter is already
  visible in it.

## 13. Open questions for review

1. **Is the 1.6% pixel sample the accepted definition of the closing sweep**, or should it raise
   `--max-shards`/`--seam-boundaries` and pay the extra?
2. **Who reads the roll-up, and against what deadline?** The sweep is worth little if its output
   arrives after the product is committed to.
3. **On a 3a failure, does the campaign continue with other cells?** A blocking finding on one cell is
   probably local; the same finding on the first three is systemic and should stop the run. That
   threshold is a policy choice, not a technical one.
