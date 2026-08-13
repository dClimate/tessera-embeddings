# Data validation — the campaign's per-cell gate and closing sweep

Every rung of the pre-campaign programme measured throughput, cost, or recovery. **None asked
whether the published pixels are right.** This is what does.

The instrument exists, has run on 15 zone-years of the dev store, and is now wired into the campaign.
It found the observation-count seams, a dimension check that would have failed every healthy cell, and
a cell published from six months of optical input. **Both passes below are built.**

---

## 1. Two passes

### 1.1 Per cell, inside the campaign, as each cell lands — **BUILT**

**Each fill dispatches one `validate-zone-year` run per cell the moment that cell is tagged, and does
not wait for it.** A defect found on cell 3 of 1,000 is a fix; the same defect at the end is 1,000
refills.

**A separate flow, dispatched from the trailing assembly thread.** Cell N is validated while cell N+1
is still being inferred and assembled — the same pipelining assembly itself already has — so the check
never delays a GPU fleet. It runs on the ingestion image: no Ray, no GPU, and one run per cell, so a
failure names a coordinate instead of being a line inside a multi-hour fill's log.

**The cell is already tagged when the check runs, and this is the one place the plan changed.** An
earlier draft withheld the tag until the check passed. That is wrong once the check is a separate run
the campaign does not wait for: campaign progress would then depend on a run nothing joins, and a
validation that never ran would look exactly like a cell that never landed. So the two facts get two
records — **the tag says the cell LANDED, the verdict says it is SOUND** — and a blocking finding
surfaces as a FAILED validation run rather than as a withheld tag.

**Cost: about 3 minutes and under three cents on the largest cell we have** (8,714 shards), under a
minute on small ones, and none of it on the campaign's critical path.

Four requirements, each guarding a distinct way this goes wrong:

1. **It raises `CellValidationFailed`, naming the cell and the slugs** — and the verdict is written to
   S3 *before* the raise. A record that exists only when the news is good is not a record.
2. **A finding fails the CELL, not the RUN.** One validation run per cell, so the campaign and every
   other cell keep moving. Nothing retries or reopens the cell automatically: remediation is a
   decision, because a false positive would otherwise trigger a multi-hour, real-money refill.
3. **The error is DISTINGUISHABLE to monitoring, not merely present.** Its own exception type, and
   `campaign_health.py`'s **cell validation** check is last in the checklist so its PROBLEM wins a
   verdict tie — a published-data defect outranks every throughput signal there.
4. **"Found a defect" and "could not run" are different.** A cell that cannot be read raises
   `CellNotAuditable` and leaves NO verdict on file; a dispatch that failed leaves none either
   (the dispatch is best-effort by design — a landed, tagged cell must not be undone by an unreachable
   API). Monitoring reads published cells against the verdicts beside them, so an unchecked cell
   reports as *unvalidated* rather than as passed. **The absence is the record.**

**Cells marked empty are excluded**, by both passes and by the monitoring check. Nothing was embedded,
so every pixel-level check has no subject, and the coverage question that remains — *is it right that
this is empty?* — is settled by the fill's own `written + skipped == live` reconciliation. All three
read one definition of the validated set (`auditable_cells`), so the exclusion cannot read as a
missing verdict.

### 1.2 Closing sweep — **BUILT**

`scripts/validate_all_cells.py` over every published cell, for final peace of mind rather than as a
gate the campaign waits on. It **re-runs the whole check per cell**, one subprocess each and
concurrently, with `--max-shards` and `--seam-boundaries` raised above the defaults and the values
recorded — then rolls the results into an exception list, a per-cell table, and aggregate
distributions.

**It re-runs rather than reading the stored verdicts, and that is the point.** A different `--label`
and `--seed` sample different windows and different boundaries, so each cell ends with **two disjoint
sampled sets and two verdicts** — the second an independent read of the cell rather than a copy of the
first. The inspectable total doubles. Rolling up the per-cell verdicts alone would cost nothing and
add nothing.

Each pass writes its verdict beside its figures, labelled like them, so the two never overwrite each
other and a disagreement between them is legible: same cell, same snapshot, two independent samples.

Its behaviour is documented in the script. What matters here: it sorts findings **by slug, not prose**,
and counts the expected-and-upstream ones once rather than listing them per cell.

### 1.3 Output

Two prefixes in the outputs bucket, one per zone, written by both passes:
**`windows/<zone>/`** for the figures and **`verdicts/<zone>/`** for the machine-readable verdict.
Filenames carry zone and year, so a zone's whole history shares one browsable folder in each. The
split is what keeps the sweep cheap: it rolls up stored verdicts rather than re-reading the product,
and it can list a directory of verdicts without paging past every PNG.

## 2. What blocks, and what does not

| finding | disposition |
|---|---|
| shard written outside the land mask | **BLOCKS** — unambiguous misplacement |
| coverage reconciliation does not close (`written + skipped == live`) | **BLOCKS** |
| `input_coverage` shows a month gap **nothing accounts for** | **BLOCKS** — the input was not what the cell claims |
| `input_coverage` shows `relaxed: true` but a whole window | **RECORD** — the guard was off for that *dispatch*, which is graded at the campaign level, where the other cells of it are. Failing this cell for its neighbours' risk would be a finding about a different cell |
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

**The seam test was not reproducible before 2026-08-13, and a borderline row could appear or vanish
between runs of the same snapshot.** Its boundary offsets came from one generator that the worker
threads also drew from, so how many draws the first axis's threads made decided which boundaries the
second axis tested. Four runs of one cell at one seed gave four different northing results and flipped
an embedding-seam row between "seam" and "no seam"; the easting axis, drawn before any thread started,
was identical every time. Each axis now seeds from the seed and the axis, and each boundary's interior
sample from that boundary's own identity, so three consecutive runs agree exactly. Two consequences
worth holding: **a seam row in a verdict written before that date is not reproducible**, which matters
because only dev verdicts exist so far and none is a record of the product; and a nondeterministic
check is the one thing that could manufacture the campaign's systemic signal, since four cells failing
the same check is what escalates.

## 4. A thin cell is not a broken cell — and how the AI review works

Optical depth varies by an order of magnitude globally, and **picture quality tracks optical depth and
nothing else** — measured across 15 cells: crisp at 30+ valid observations per pixel, noise below about
20. Radar absence alone does not visibly degrade the embeddings. `OPTICAL_THIN_MAX_OBS` records this per
pixel as a share of embedded area. **Publish thin cells; label them.**

> **That measurement needs re-checking, and is not withdrawn (2026-08-13).** It was read off figures
> drawn with a **per-window** contrast stretch, which expanded a low-variation window's noise floor to
> full contrast — so "looks like noise" was partly a property of the rendering rather than of the
> depth. The stretch is now shared across a cell's windows (see below), and a cell at 46.9
> observations per pixel that previously showed three noise-like windows out of four now shows three
> uniform surfaces. Which way this pushes the 30/20 boundary is unknown until the comparison is
> re-run: the mechanism behind it is independently plausible, and finding a flaw in a measurement does
> not tell you the direction of its error. What is safe to keep using is the *relationship* — deeper
> input, crisper picture — not the two numbers.

**The line is 15 observations, set 2026-08-12 (it was 40).** That is deliberately *below* the depth at
which pictures start to degrade, so the label names cells thin enough to be worth calling out rather
than every cell that is merely not crisp — at 40 a large share of the globe would carry it, and a label
that covers a third of the product is not a label. It is only a reporting line: nothing refuses a cell
for being under it, and the per-pixel counts are in the store either way. The verdict publishes the
cell's own mean depth (`s2_obs_mean`) beside the share, which is what a reviewer judges a window
against — so lowering the threshold changed what is *flagged*, not what is *known*.

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

**What the review is actually handed, decided 2026-08-13 by looking at a real cell's figures.**

* **One PNG per window, at the resolution it was read**, beside the composite panel. The panel is for a
  person who can zoom into it; an image pipeline that bounds an image's long edge shrinks a composite
  by however many windows are in it, so a reader that cannot zoom would judge a 512-pixel window at
  150 pixels — and the artifacts in the list above are the first thing a downsample destroys. The
  panel remains the human figure.
* **One contrast stretch shared by all of a cell's windows**, rather than each window stretched to its
  own percentiles. This was a real defect: a window whose ground varies little had its noise floor
  expanded to full contrast and arrived looking broken, next to a window with real structure, while
  the figure's caption claimed the colours were comparable. Uniform ground now looks uniform, which
  is the only way the "uniform noise" artifact above can mean anything.
* **Per-window facts as text**, in the verdict's `windows` list: how much of the window the **land
  mask** calls land, and how flat it is relative to the widest window in the cell. Land share is not
  the embedded share — inference fills every pixel of a live shard, so a window over open water is
  100% embedded and entirely uninformative. Without these two numbers the correct reading of a flat
  picture is indistinguishable from a defect, which would push the review toward `cannot-tell` on
  exactly the cells it is supposed to rank.
* **Not the other three figures.** Coverage, bands and seams are renderings of numbers that are
  already in the verdict, and the verdict is the better input for those.

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

**How the verdicts reach a person** — the 5-minute monitoring round, what goes to Slack, which
findings justify stopping the campaign rather than shelving the zone, and the AI window review's
place in it: `campaign-monitoring-plan.md`.

## 7. Tools

| what | where (all in `yield-embeddings`) |
|---|---|
| the instrument: `audit_cell`, one implementation for both passes | `src/yield_embeddings/domain/embedding_validation.py` |
| the operator CLI over it — `--json`, `--out <fsspec>`, `--label` | `scripts/validate_cell.py` |
| the campaign's per-cell gate, dispatched by each fill | `src/yield_embeddings/orchestration/prefect/flows/validate_zone_year.py` |
| its decisions and thresholds, unit-tested on synthetic arrays | `src/yield_embeddings/domain/embedding_validation_rules.py` |
| the closing sweep and its roll-up | `scripts/validate_all_cells.py` |
| per-cell structural and placement validation | `scripts/validate_zone_group.py` |
| campaign-day monitoring, incl. the per-cell **cell validation** check | `scripts/campaign_health.py` |
| reopen a cell / pin a fresh tag | `scripts/reopen_zone_year.py` |
| drop a provenance field known wrong | `scripts/drop_run_field.py` |
