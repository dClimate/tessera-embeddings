# Data validation — the campaign's per-cell gate and closing sweep

Every rung of the pre-campaign programme measured throughput, cost, or recovery. **None asked
whether the published pixels are right.** This is what does.

The instrument exists and has run on 15 zone-years of the dev store; it found the observation-count
seams, a dimension check that would have failed every healthy cell, and a cell published from six
months of optical input. **One thing is unbuilt: the per-cell gate of §1.1.** Everything it needs
exists, so what remains is wiring.

---

## 1. Two passes

### 1.1 Per cell, inside the campaign, before the tag — **NOT YET BUILT**

**Run the full check as the last step of every cell and raise on a blocking finding.** A defect found
on cell 3 of 1,000 is a fix; the same defect at the end is 1,000 refills.

**Where: on the trailing assembly thread, between the year-attribute commit and `tag_zone_year`.** One
placement satisfies everything at once —

* **already off the critical path** — the runner assembles cell N while cell N+1 infers, so this delays
  only cell N's tag and no GPU fleet waits on it;
* **a failing cell is never tagged**, so it reads as unfinished to the dispatch queue and is retried
  rather than entering the published set;
* **that state is known-recoverable**, because F2 drilled exactly it — committed-but-untagged is the
  mid-commit crash state, and the recovery path re-lands it.

**Cost: 3 minutes on the largest cell we have** (8,714 shards), under a minute on small ones — about
1.5% added to that cell's ~196-minute trailing budget, which the 1.21x trailing margin absorbs.

Four requirements, each guarding a distinct way this goes wrong:

1. **It raises, naming the cell and the finding** — not a log line, so triage needs no archaeology.
2. **A finding fails the CELL, not the RUN.** The campaign keeps moving and the cluster's other cells
   continue. It sits under the existing per-cell attempt budget, since a deterministic defect fails
   every retry and must not loop.
3. **The error is DISTINGUISHABLE to monitoring, not merely present.** A recognisable signal, because
   the requirement is that monitoring surfaces it *with urgency* — a published-data defect outranks
   every throughput signal the campaign-day view reports.
4. **"Found a defect" and "could not run" are different, and only the first blocks.** A validator that
   cannot read the store logs loudly and lets the cell tag. A guard that fails good data because the
   guard broke is worse than no guard.

### 1.2 Closing sweep — **BUILT**

`scripts/validate_all_cells.py` over every published cell, for final peace of mind rather than as a
gate the campaign waits on. It re-runs the free exact half globally, rolls up the stored per-cell
verdicts, and re-reads pixels where warranted, with `--max-shards` and `--seam-boundaries` raised above
the defaults and the values recorded.

**It renders NEW windows, not a refresh** — a different `--label` and `--seed`, so each cell ends with
two disjoint sampled sets and the inspectable total doubles. Refreshing the same windows would buy
nothing.

Its behaviour is documented in the script. What matters here: it sorts findings **by slug, not prose**,
and counts the expected-and-upstream ones once rather than listing them per cell.

### 1.3 Output

**`s3://global-tessera-embeddings/windows/<zone>/`**, from both passes. Filenames carry zone and year,
so a zone's whole history shares one browsable folder, and the machine-readable verdict sits beside the
figures — which is what lets the sweep roll up stored results instead of re-reading the product.

## 2. What blocks, and what does not

| finding | disposition |
|---|---|
| shard written outside the land mask | **BLOCKS** — unambiguous misplacement |
| coverage reconciliation does not close (`written + skipped == live`) | **BLOCKS** |
| `input_coverage` shows a partial window or `relaxed: true` | **BLOCKS** — the input was not what the cell claims |
| completion mark with no run record | **BLOCKS** |
| seam median outside 0.80–1.25 on **embeddings** | **BLOCKS** — §3 |
| a constant embedding dimension, or scale-setting shares far from 1.0 | **BLOCKS** — a dead channel; or not quantized as readers assume |
| high optical-thin share | **RECORD and PUBLISH** — a property of the input, not a fault |
| absent land tiles explained by the skip record | **RECORD** — expected |
| **observation-count seams** | **IGNORE** — §3 |
| a window the AI review calls suspicious | **FLAG for a human, never block** — §4 |

A **constant dimension** is one of the 128 numbers with zero variance across the sample: a dead output
channel, invisible in any picture because three components make the image. What makes it blocking is
less the wasted dimension than what it implies about the rest.

A corrected cell needs a **fresh tag name** — icechunk tags are write-once forever.
`scripts/reopen_zone_year.py` clears the completion mark and pins a fresh tag;
`scripts/drop_run_field.py` removes a provenance field known wrong.

## 3. The two seam rows are different findings

**Observation-count seams are upstream and ignored.** Neighbouring tiles keep different date sets after
cloud masking, so the count genuinely steps at a tile edge. Present on every cell measured; reporting it
per cell would fire a thousand times and bury everything else.

**Embedding seams are ours, and the reason is the grid.** Shard boundaries are a boundary *we* invented;
an upstream defect would align to MGRS tiles or swath edges, not to ours. So a discontinuity at our
shard boundaries and nowhere else implicates our assembly, and this is its only detector.

Two guards against over-reading it, both already in the instrument: the verdict rests on the **median
across many boundaries**, since one boundary on a coastline genuinely is a discontinuity; and the
**edge-distance profile** separates a placement defect (a spike at the boundary only) from a model given
less context near its tile edge (a rise over several pixels).

## 4. A thin cell is not a broken cell — and how the AI review works

Optical depth varies by an order of magnitude globally, and **picture quality tracks optical depth and
nothing else** — measured across 15 cells: crisp at 30+ valid observations per pixel, noise below about
20. Radar absence alone does not visibly degrade the embeddings. `OPTICAL_THIN_MAX_OBS` records this per
pixel as a share of embedded area. **Publish thin cells; label them.**

**An AI reviews the windows before a human sees them**, as part of the monitoring round — **10 windows
per cell**. This is the one check that otherwise does not scale: nobody eyeballs a thousand cells, so in
practice it would be sampled and the majority would go unlooked-at.

**It reports named observations, not a quality score.** A 1-to-5 rating would drift — two cells scored
"3" by separate invocations are not comparable, the number tells a human nothing about where to look,
and worst of all it collapses *thin input* and *defect* into one axis when they need opposite responses.
So the review answers three concrete questions per window instead:

* **Which real-world features are present?** Field boundaries, drainage networks, coastline, topography,
  settlement. Presence of a nameable thing is comparable across cells in a way a felt quality is not.
* **Which artifacts are present?** Blocking, a visible straight edge crossing the window, repeated or
  mirrored texture, uniform noise. These are the shapes a placement or quantization defect would make.
* **A verdict of plausible / cannot-tell / suspicious**, where **cannot-tell is mandatory** — a
  thin cell genuinely cannot be judged, and forcing a verdict there is what would flag every atoll and
  sub-Antarctic cell for being correct.

Two constraints on the review: it is given **that cell's own optical depth** and told to judge against
it; and **suspicious flags for a human, never blocks**, because "looks wrong to a model" is not evidence
any other check would confirm. Its value is ranking a thousand cells so the human reads the worst
twenty.

## 5. Scope and cost

**"Exhaustive" cannot mean every pixel.** ~1,000 zone-years at 63 Gpx and 128 dimensions is several
petabytes — reading it all is impossible, not expensive. So:

* **every cell** — a per-cell defect is invisible in an aggregate;
* **coverage and placement exact, every shard** — this is *metadata*: one chunk-manifest entry is one
  2048-px shard, so the written map is exact for a whole zone-year with no pixel reads, and the pixel
  budget is then aimed only where data is known to be;
* **pixels sampled at 1/64 of each shard**, systematically placed.

**Do not describe the result as "every pixel checked."** The honest claim: *every cell, every shard's
existence and placement exactly, 1.6% of each shard's pixels systematically sampled.*

Measured, in-region: **$0.028 and 3 min** for the largest cell; $0.048 for a ten-cell sweep. The closing
sweep over ~1,000 cells is **order $15–25 and a few hours**. **Run it in-region** — the same reads cost
~6x more from outside, where egress dominates. The per-cell gate is effectively free.

## 6. What a pass does not carry

* **A defect smaller than the 1.6% sample can escape it.**
* **It validates self-consistency, not ground truth** — that the data agrees with the land mask and its
  own provenance and looks like Earth, never that the embeddings are *correct* for the surface.
* **A cell with too few written shards** yields too few boundaries to decide, and reports undetermined.
* **One unexplained reproducible outlier**: 40S/2024 shows pixel differences ~50% larger at shard
  boundaries than 128 px inside, across two independent assembly runs, where 12 other cells sit at
  0.91–1.10. Its medians pass. The input-depth hypothesis was tested and refuted; cause unknown.

## 7. Tools

| what | where (all in `yield-embeddings`) |
|---|---|
| per-cell check and renderer — `--json`, `--out <fsspec>`, `--label` | `scripts/embedding_reality_check.py` |
| its decisions and thresholds, unit-tested on synthetic arrays | `src/yield_embeddings/domain/embedding_audit.py` |
| the closing sweep and its roll-up | `scripts/validate_all_cells.py` |
| per-cell structural and placement validation | `scripts/validate_zone_group.py` |
| campaign-day monitoring | `scripts/campaign_health.py` |
| reopen a cell / pin a fresh tag | `scripts/reopen_zone_year.py` |
| drop a provenance field known wrong | `scripts/drop_run_field.py` |
