# Final data validation — the campaign's closing gate

**Status: PLAN, not yet executed.** For review. Delete or fold into a measurement record once it
has run.

Every rung of the pre-campaign programme measured throughput, cost, or recovery. **None of them
asked whether the published pixels are right.** This is the gate that does, and it runs after the
campaign over everything it produced.

The instrument already exists and has been exercised on 15 zone-years of the dev store. It is what
found the observation-count seams at tile edges, a dimension check that would have failed every
healthy cell, and a cell published from six months of optical input. What does *not* exist is the
roll-up that makes a thousand of its verdicts readable by a person — see §6, the one piece of
tooling this plan asks for.

---

## 1. What "exhaustive" can and cannot mean

This is the first thing to agree, because the word does not survive contact with the scale.

The campaign publishes roughly **112 land zones × 9 years ≈ 1,000 zone-years**, each about 63
gigapixels with 128 dimensions per pixel. Reading every stored byte is not a budget question, it is
an impossibility — it is the whole product, several petabytes, pulled through one process.

So the gate is exhaustive along the axes where exhaustive is achievable, and explicitly sampled
where it is not:

| axis | coverage | why |
|---|---|---|
| **cells** | **every one** | a per-cell defect is invisible in an aggregate; there is no reason to sample cells |
| **coverage / placement** | **exact, every shard** | this is METADATA — see §2. Costs no pixel reads at all |
| **provenance reconciliation** | **exact, every cell** | attribute reads |
| **pixels** | **sampled — one 256-px chunk per 2048-px shard, i.e. 1/64 = 1.6% of each shard** | the only axis where the budget binds |
| **seam boundaries** | sampled per axis, raised for this pass | each boundary is a pixel read on both sides |
| **native-resolution windows** | a handful per cell | for a human to look at |

**Do not describe the result as "every pixel checked".** The honest claim is: *every cell, every
shard's existence and placement exactly, and 1.6% of every shard's pixels systematically sampled.*
That is a strong claim. Overstating it is worse than making it.

## 2. Why the exact half is free

The design fact that makes this affordable at all: **which shards exist is metadata, not data.**
The store's arrays are sharded, so one chunk-manifest entry is one 2048-px shard. Walking the
manifest answers "which shards hold data" exactly, for a whole zone-year, in about a second and with
no pixel reads.

That is what makes the two strongest checks nearly free:

* **Coverage cross-tabulation** — the frozen land mask against the manifest. Yields the exact count
  of land tiles written, land tiles absent, and **shards written where the mask says no land**.
* **Placement, in its unambiguous half** — a shard outside the mask is misplacement, full stop.
  Nothing legitimate produces one.

The pixel budget is then aimed *only* at shards already known to exist, which is why a dense
zone-year costs cents rather than dollars.

## 3. Measured cost, and the wall clock

From the dev store, in-region where transfer is free:

| cell | written shards | cost | wall |
|---|---:|---:|---:|
| 37N/2021 (largest measured) | 8,714 | **$0.028** | 3 min |
| 60N/2020 | 527 | $0.002 | 52 s |
| ten-cell sweep (6 to 9,050 shards) | — | **$0.048 total** | — |

Extrapolating to ~1,000 cells at the observed mix: **order $15–25 and a few hours of wall clock,
in-region.**

**Run it in-region.** The same reads from a laptop cost about 6x more, because egress dominates: 37N
is $0.028 in-region and $0.176 over the internet. At campaign scale that is the difference between
twenty dollars and a hundred and twenty.

Raising the pixel depth scales roughly linearly in chunks read. Reading all 64 chunks per shard
instead of one would be ~64x the pixel cost and the wall clock, which is the boundary this plan
declines to cross — the sample is systematic, not random, so a defect large enough to matter is
already visible in it.

## 4. Per-cell pass criteria

Six, and each names the failure it exists to catch:

1. **The coverage reconciliation closes**: `written + skipped == live`. This is what `optical_skips`
   exists to make answerable. On a campaign cell it must **PASS**, not read UNAVAILABLE — an
   UNAVAILABLE means the cell was filled by code predating the field, which is itself a finding
   about the run rather than about the data.
2. **Zero shards written outside the land mask.** Unambiguous misplacement.
3. **No seam on either axis**, and the edge-distance profile flat. The median ratio must sit inside
   0.80–1.25; a median outside it is a placement defect, and a median below 0.80 specifically means
   a duplicated write. Individual flagged boundaries with a median near 1.0 are **scene structure,
   not a defect** — a coastline or field edge that happens to fall on a boundary genuinely is a
   discontinuity.
4. **No constant embedding dimension**, and the **scale-setting shares sum to 1.0** — the invariant
   that confirms the data was quantized the way every reader assumes. A total near zero means the
   scale was not derived per pixel from these values; well above one means several dimensions of one
   pixel hit a rail, which is the only reading under which range was genuinely lost.
5. **`input_coverage` shows a full window and `relaxed: false`.** A cell published from a partial
   input through the coverage relaxation is the failure class P7 produced by accident.
6. **The detail windows look like the Earth.** Not reducible to a statistic, and the check that
   caught the thin cells. Judge it against that cell's own optical depth — see §5.

## 5. Two readings that are NOT problems

Both will appear on many cells. Grading either as a defect would bury the real findings.

**The observation-count arrays step at tile boundaries.** Median ratio exactly 1.000 with most
boundaries flagged, driven by the effect-size term rather than by magnitude. Neighbouring tiles keep
different date sets after cloud masking, so the count genuinely steps at a tile edge. It is upstream
of assembly, it appeared uniformly on every cell measured, and the embeddings built from those dates
show no matching seam.

**A thin cell is not a broken cell.** Optical depth varies by an order of magnitude across the globe
and **picture quality tracks optical depth and nothing else** — measured across 15 cells: crisp,
field-boundary-sharp imagery at 30+ observations per pixel; indistinguishable-from-noise below about
20. Radar absence alone does not visibly degrade the embeddings. A sub-Antarctic or atoll cell can be
legitimately noise-like, and `OPTICAL_LIGHT_MAX_OBS` (40) records that per pixel as a share of
embedded area — so the answer is on the record rather than in someone's reading of a picture.

## 6. THE MISSING PIECE: a roll-up

**A thousand per-cell verdicts is not a validation, it is a pile.** The tool prints a human-readable
verdict and exits non-zero on any disagreement; it has **no machine-readable mode**. Without one,
this gate either gets skimmed or gets skipped.

What is needed, in the cheapest form that works:

* **A `--json` mode on `embedding_reality_check.py`**, emitting per check a stable slug, its status,
  and its named subjects — the same projection `campaign_health.py` already implements, so there is a
  pattern to copy rather than a design to invent.
* **A driver** that runs every completed cell, collects those objects, and emits:
  * a **one-line-per-cell table** — written, absent, reconciliation, optical range, seam medians;
  * an **exception list**, only the cells with a finding, sorted by severity;
  * **aggregate distributions** for the quantities that are only interpretable in bulk: the seam
    median across all cells, the optical-light share, the absent-tile fraction.

The aggregate half matters as much as the exceptions. A small consistent offset is invisible per cell
and obvious across a thousand — and the reverse trap applies too: **a consistent offset is not
evidence of an effect until the noise floor is known.** The current set shows seam medians leaning
about 3% above 1.0 in 7 of 9 cells, far inside tolerance, with no noise floor measured. The roll-up
should surface that kind of pattern for judgement, not grade it.

## 7. Triage: what blocks publication

| finding | disposition |
|---|---|
| shard outside the land mask | **BLOCKS.** Placement defect; localise and re-fill the cell |
| seam median outside 0.80–1.25 on embeddings | **BLOCKS.** Same |
| scale-setting shares far from 1.0 | **BLOCKS.** The data is not quantized as readers assume |
| constant embedding dimension | **BLOCKS.** Dead output |
| `relaxed: true` or unexplained absent months | **HOLD** — decide per cell; re-ingest and re-fill is the remedy |
| reconciliation UNAVAILABLE | **RECORD** — a run-vintage finding, not a data defect |
| high optical-light share | **RECORD and PUBLISH.** A real property of the input, not a fault |
| absent land tiles explained by skips | **RECORD.** Expected; the skip record is the evidence |
| observation-count seams | **EXPECTED.** Do not report per cell |

A refill needs a **fresh tag name**: icechunk tags are write-once forever, so a corrected cell cannot
re-pin the canonical name. `scripts/reopen_zone_year.py` clears the completion mark and pins a fresh
tag; `scripts/drop_run_field.py` removes a provenance field that is known wrong.

## 8. Known limitations of the instrument

State these in whatever report the gate produces, so nobody reads more into a pass than it carries:

* **The pixel sample is 1.6% of each shard**, systematically placed (one inner chunk per shard, aimed
  at land). A defect confined to a smaller area than that can escape it.
* **The seam test cannot separate a real terrain discontinuity from misplacement on a single
  boundary.** It rests on the median over many boundaries and on the edge profile.
* **It validates against the land mask and the store's own provenance**, not against ground truth.
  It can show the data is self-consistent and plausibly Earth; it cannot show the embeddings are
  *correct* for the surface.
* **A cell with too few written shards yields too few boundaries to decide** — reported as
  undetermined rather than as a pass.
* **One unexplained reproducible outlier exists**: 40S/2024 shows pixel differences ~50% larger at
  shard boundaries than 128 px inside, across two independent assembly runs, while 12 other cells sit
  at 0.91–1.10. Its medians pass. Cause unknown; the depth hypothesis was tested and refuted. Worth
  watching for at campaign scale, where the roll-up would show whether it is a class or a one-off.

## 9. Tooling inventory

| what | where |
|---|---|
| the reader and renderer | `yield-embeddings/scripts/embedding_reality_check.py` |
| the decisions and thresholds (unit-tested against synthetic arrays) | `yield-embeddings/src/yield_embeddings/domain/embedding_audit.py` |
| per-cell structural + placement validation | `yield-embeddings/scripts/validate_zone_group.py` |
| campaign-day monitoring, incl. published-input-coverage and shard-placement checks | `yield-embeddings/scripts/campaign_health.py` |
| clear a completion mark / pin a fresh tag | `yield-embeddings/scripts/reopen_zone_year.py` |
| remove a provenance field known to be wrong | `yield-embeddings/scripts/drop_run_field.py` |

## 10. Open questions for review

1. **Is the 1.6% pixel sample acceptable as the definition of the gate**, or should the final pass
   raise `--max-shards`/`--seam-boundaries` and pay the extra? The knobs exist; the cost scales
   linearly.
2. **Who reads the roll-up, and against what deadline?** The gate is worth little if its output
   arrives after the product is committed to.
3. **Should the gate run per cell as the campaign proceeds, rather than only at the end?** Per-cell
   is cheaper to act on — a defect found on cell 3 of 1,000 is a fix, the same defect found at the
   end is 1,000 refills. The argument for the end is a single consistent snapshot. Both is possible:
   the cheap exact half per cell, the full pass at the end.
