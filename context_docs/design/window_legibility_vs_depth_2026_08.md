# Does picture quality track optical depth? A blind re-measurement — 2026-08-13

**Why this exists.** The campaign's records have carried one sentence about embedding quality since
July: *crisp at 30 or more valid observations per pixel, noisy below about 20*, measured across 15
cells. [`minimum-optical-depth-plan.md`](minimum-optical-depth-plan.md) proposes to make that
sentence into a **refusal**: a pixel with fewer than 30 observations would not be embedded at all.
That decision is irreversible — a refused pixel does not exist, so recovery is a re-run — and on
2026-08-13 two defects were found in the instrument that produced the original sentence.

**Result, stated first: picture quality does not track optical depth, and in this corpus the
gradient runs the other way.** Across 671 fully-land windows reviewed blind, legibility falls from 92%
at 15–18 observations per pixel to 61% at 35–45. The cause is not depth: it is that depth is a proxy
for climate and climate is a proxy for how much there is to see — deserts are clear overhead and
uniform on the ground, while the cloudiest tropics are rivers, clearings and smallholder farmland.
**At every candidate cutoff between 15 and 35 observations, about 85% of what the rule would refuse
shows recognisable ground**, and raising the cutoff leaves a *higher* share of unreadable windows in
the product, because the unreadable ones are mostly the deep ones.

---

## 1. What was wrong with the original measurement

**The renderer manufactured noise.** The window figures were drawn with a **per-window** contrast
stretch — each window scaled to its own 2nd–98th percentile — so a window whose ground varies little
had its noise floor expanded to full contrast *whatever its depth*. Some of the "noise below 20" was
the picture, not the data. The stretch is now shared across a cell's windows.

**Depth was read per cell.** A zone spans most of a hemisphere and depth falls with latitude and with
cloud, so a cell mean describes a mixed population. The first cell re-measured has a mean of 46.9 and
windows at 34.5 and 61.0 — and its *thinnest* window is the one with the most structure. Each window
now records its own mean and tenth percentile.

## 2. Method

**Two passes.** A pilot of 73 windows from 9 cells, sampled by latitude the way the audit samples, and
then the pass this document reports: **713 windows from 40 cells**, sampled by depth so the bands that
decide a cutoff are populated on purpose rather than by accident. The pilot is why the second pass
exists — it had one window separating a cutoff of 20 from one of 30.

**Blinded in code, not by instruction** (`scripts/depth_legibility_probe.py` in `yield-embeddings`).
The reviewer received opaque filenames in shuffled order and one fact per window — how much of the
frame the land mask calls land, without which an ocean window cannot be told from a failed land one.
**Depth was withheld**, and so was the flatness figure: it is computed from the same pixels, so a
reviewer handed it could answer "spread 0.12, therefore illegible" without looking, and the result
would be a correlation between two derived numbers.

**One question per window:** can you name a real ground feature — field boundaries, drainage, valley,
coastline, settlement, dune or glacial forms, a boundary between distinct covers? Answer `legible`,
`uncertain` or `illegible`, with `illegible` explicitly not a criticism, since open ocean, unbroken
desert and unbroken canopy are correctly illegible. Every verdict carries a one-line note naming what
was seen, so a verdict can be audited rather than trusted.

## 3. The result — 671 fully-land windows

The first pass sampled 73 windows by latitude and had **one window** separating a cutoff of 20 from
one of 30, so it could not choose between them. This pass sampled **by depth**: 713 windows from 40
cells, harvested by probing one observation-count chunk per shard and then reading windows only where
the estimate landed in a target band, preferring shards whose window is wholly land
(`scripts/harvest_depth_windows.py`). Reviewed blind by six independent sessions over disjoint ranges,
with depth and flatness both withheld. **All 713 were scored**; 671 of them are wholly land.

**The six agents agree on level**, which is what makes the pooling legitimate: on fully-land windows
they returned 69%, 74%, 75%, 78%, 79% and 81% legible against matched median depths of 25 to 28
observations. (Two of them reached that agreement by different routes — one upgraded verdicts after a
per-image restretch revealed organised structure, another refused to upgrade on anything not visible
at the delivered contrast. The seven-point spread is the size of that disagreement.)

| depth band | n | % legible |
|---|---:|---:|
| under 12 | 15 | 60% |
| 12–15 | 39 | **90%** |
| 15–18 | 59 | **92%** |
| 18–21 | 69 | 86% |
| 21–24 | 71 | 86% |
| 24–27 | 72 | 86% |
| 27–30 | 79 | 77% |
| 30–35 | 111 | **66%** |
| 35–45 | 140 | **61%** |
| 45+ | 16 | 69% |

**Legibility FALLS as optical depth rises**, from 92% in the 15–18 band to 61% at 35–45. The July claim
is not merely unsupported; the gradient in this corpus runs the other way. The one band where refusal
looks defensible is the very bottom — **under 12 observations, legibility is 60%**, the lowest of any
band and the only one where a cutoff removes roughly as much bad as good.

### Why it runs backwards, tested rather than asserted

Not latitude: legibility by absolute latitude is flat within noise (76%, 78%, 67%, 84%, 72% across five
bands). The controlled test is depth **within** a latitude band, and it splits:

| | under 20 obs | 20–30 | 30–45 |
|---|---:|---:|---:|
| tropics, 0–25° (n=401) | **87%** | 85% | **61%** |
| mid-latitudes, 25–55° (n=112) | 77% | 82% | 82% |
| high latitudes, 55–90° (n=43) | 75% | 67% | 67% |

So the effect is real inside the tropics and absent elsewhere, which points at **land cover
correlated with cloudiness** rather than at depth doing anything. In the tropics the wettest places
are the most persistently clouded — and they are river networks, forest-and-clearing mosaics and
smallholder agriculture, which are legible. The drier tropics are clear overhead and uniform on the
ground. Depth is a proxy for climate, and climate is a proxy for how much there is to see.

The per-cell spread says the same thing more bluntly. Legibility by cell runs from **11%** (23N/2021,
which is Saharan desert, deep and uniform) to **100%** (38N/2021, 47S/2021, 59S/2022). Which zone a
window came from predicts its legibility far better than how many observations it had.

## 4. What each candidate cutoff costs

| cutoff | windows refused | legible among refused | illegible among kept |
|---|---:|---:|---:|
| 12 | 15 | 9 (**60%**) | 121 (18%) |
| 15 | 54 | 44 (81%) | 118 (19%) |
| 18 | 113 | 98 (87%) | 115 (21%) |
| **20** | 155 | 132 (**85%**) | 109 (21%) |
| **25** | 280 | 240 (**86%**) | 99 (25%) |
| **30** | 404 | 341 (**84%**) | 81 (**30%**) |

**The transferable number is the second column, and it is flat: whatever line is drawn between 15 and
30, roughly 85% of what it refuses shows recognisable ground.** Only at 12 does that fall to 60%. The
corpus is deliberately depth-stratified, so the counts describe the corpus rather than the product —
the product-level volume comes from the census (10.9% of pixels below 30 in its sample, 18.4%
globally).

**The third column is the finding nobody expected.** Raising the cutoff makes the *kept* product
proportionally worse: the illegible share of what survives climbs monotonically from 18% at a cutoff
of 12 to 30% at 30, because the illegible windows are mostly the deep ones. A refusal line does not
concentrate on poor data. It removes good data and leaves the poor data behind.

## 5. What this measurement cannot settle

**Thin data and its geography remain entangled, and no sample fixes that.** Thin optical data occurs
only where it is cloudy or dark for months, so a corpus of thin windows is a corpus of particular
places. Forty cells and a within-latitude control make the confound visible and partly separable —
that is what §3's second table does — but the campaign will publish zones this corpus has no example
of, and the dev store holds no 2017, which is the year where two thirds of shard-years fall below 30.

**Below 12 observations is thinly sampled** — 13 windows — so the shape of the curve at the very
bottom is the one part of it that more data would still change. It is also the one part where the
refusal case is strongest: legibility there is 62%, the lowest of any band.

**Legibility is not utility, and this is the limit that matters most.** The picture is a
three-component projection of a 128-dimensional embedding, so a window a reader cannot interpret may
still carry signal a downstream model uses; the first three components explain about 67%, 8% and 6% of
variance, and nothing about the other 122 dimensions appears in the frame. This measurement therefore
bounds one thing only — whether the *pictures* justify the sentence the plan quotes. **It cannot
establish that thin embeddings are useless, and the plan's reputational argument rests on a claim
about usefulness that no measurement here or before it has tested.**

## 6. What follows for the plan

**No cutoff between 15 and 35 is supportable on picture evidence, 30 least of all.** Each one refuses
about 85% legible ground, and the higher the line the more unreadable windows it leaves behind. If a
refusal ships, it ships as a product judgement with this document cited against it, not as a
conclusion the pictures support.

One further observation from the reviewers, which weakens any inference from "illegible" to
"worthless": one session measured seven of its twenty-five illegible windows as carrying real,
coherent ground structure — region boundaries, discrete patches, in one case a full drainage network —
at roughly a tenth of the amplitude the shared contrast stretch renders visible, three to eight grey
levels out of 255 against twenty or more for a typical legible window. **"Illegible" here means "low
contrast relative to the rest of its cell", not "carries no information."**

Three options, in the order I would consider them:

1. **Publish everything with the quality field, and refuse nothing.** The reversible branch. The
   reputational goal — a user not silently handed poor data — is served by the per-pixel
   `s2_obs_count` already in the store, a documented quality band, and the registry's per-shard
   summary. Everything else in the plan is worth building unchanged.
2. **Refuse at a line low enough that nothing legible is known to fall under it.** Below 12.8 is
   unmeasured, so this means measuring the 5–13 range first, in more than one biome.
3. **Refuse at 30 anyway**, as a deliberate quality-over-coverage judgement, with this document cited
   in the ADR as the evidence against it. Defensible as a product decision; not defensible as a
   claim that the pictures support the line.

**What would change the answer:** a corpus with thin windows from more biomes — high-latitude winter,
monsoon Asia, coastal west Africa — and, more decisively, any measurement of whether a thin embedding
degrades a downstream model. The second is the question the rule is really about, and it has never
been measured.

---

## 7. What explains the inversion — and why legibility was the wrong measurement

The inverted gradient in §3 is real in the data but it is **not a fact about information content**.
Three measurements, each computed over the 671 fully-land windows.

**Legibility is a function of rendered contrast, not of depth.** Grouping by `variation_share` — each
window's range as a share of the widest window in its cell — legibility runs 18%, 77%, 94%, 92%, 91%
across five bands. There is a cliff below about 0.15 and a plateau above 0.3. Depth barely enters:
hold contrast fixed in the middle band and legibility is 94%, 96%, 94% across three depth bands.

**Spatial organisation does NOT fall with depth.** Measuring the share of each window's variance that
survives 8×8 block averaging — per-pixel noise averages away, ground structure does not — the medians
by depth band are **0.69, 0.70, 0.74, 0.72, 0.69, 0.72, 0.79** from under 15 observations to over 45.
Flat. What falls with depth is amplitude: median 23.9 at 15–20 observations against 17.4 at 35–45.

**So the same structure is present and rendered fainter.** The split that shows it: windows with high
organisation but low amplitude are called legible **44%** of the time, against **91%** for equally
organised but brighter windows. The faint group's median depth is 29.9 — deep.

> **The conclusion this forces.** The legibility measurement, in both its July form and this one, is
> measuring **contrast in a rendering choice**. The July per-window stretch amplified thin windows and
> made them look noisy; the corrected shared stretch compresses deep-but-uniform windows and makes
> them look empty. Neither reading is about whether the embedding carries information, and on the one
> amplitude-invariant measure available — spatial organisation — **depth makes no difference across
> 12 to 45 observations.** No cutoff in that range can be justified from pictures, in either
> direction, because the pictures do not measure the thing.

## 8. What DOES depend on observation count: a measurable bias

Found while attributing the straight edges of §9, and it is the best-founded depth effect in this
whole investigation. At an observation-count boundary — same ground, same day, a step in how many
valid looks each side received — **the embedding steps too**. Measured on three windows, comparing the
mean 128-dimensional vector of equal strips either side against a control pair of adjacent strips
wholly on one side:

Swept across all 40 harvested cells (`scripts/seam_drift.py`), **148 seams**, with the seam located
in the COUNT array rather than in the picture so the selection cannot be made on the outcome:

| observation step | n | across the seam | beside it | ratio |
|---|---:|---:|---:|---:|
| 0–1 | 65 | 0.00470 | 0.00298 | 1.6× |
| 1–2 | 35 | 0.00808 | 0.00337 | 2.4× |
| 2–4 | 34 | 0.01088 | 0.00246 | **4.4×** |
| 4–8 | 12 | 0.02481 | 0.00406 | **6.1×** |
| 8+ | 2 | 0.08461 | 0.00609 | **13.9×** |

**Monotonic over five bins, and 123 of the 148 seams move further across than beside.** An eight-
observation step displaces the embedding about fourteen times more than neighbouring ground does on
its own.

The two methods measure against different baselines and that is why their per-observation slopes
differ — 0.00264 here against 0.00053 for the overlaps. The seam's control is *adjacent ground within
one scene*, which genuinely varies; the overlap's floor is *the same ground twice*, which does not. The
overlap figure is therefore the cleaner estimate of the pure depth effect, and the seam figure is the
one that says how the effect compares with ordinary spatial variation. Both agree on sign and on
monotonicity, from 229 measurements sharing no code and no data.

Unambiguous in direction: **observation count shifts the embedding
systematically, as a bias rather than as noise.** That is a far better-founded worry than "thin data
looks noisy", and it points somewhere uncomfortable for the rule: **a refusal does not remove a
depth-induced discontinuity, it creates a sharper one** — between refused pixels, which are absent,
and kept pixels at the cutoff. The measurement worth having is how the embedding drifts with depth
over the whole range; it is cheap, since every ingredient is already in the store, and nobody has
taken it.

## 9. The straight-edge artefacts: upstream, not ours



Raised unprompted by four of six reviewer sessions that had no knowledge of each other, then
attributed by measurement. Across the 713 notes, **33 mention a straight edge, a seam or an
axis-aligned artefact**; most are legitimate ground (roads, field boundaries, forestry blocks), but
roughly a dozen are cases where the reviewer judged the frame's only structure to be an artefact.
Three carry measurements:

* one seam pinned to a single pixel column in **373 of 512 rows**;
* one straight to within **0.70 pixels of a fitted line over the full 512-row height**, which nothing
  on the ground achieves over five kilometres;
* one where an apparent directional signal **vanished entirely** once a single-pixel vertical seam was
  masked out, so the seam had produced the whole effect.

One reviewer described a case as "a cluster of axis-aligned rectangular blocks — structure or chunk
artefact", which is the reading that matters: **axis-aligned means aligned to OUR raster.**

**The attribution, run 2026-08-13 over all 713 windows.** Taking the strongest full-width step in each
axis profile, **45 windows (6.3%) carry one at eight sigma or more above their own frame's noise**.
Where they fall settles it:

* **none at our chunk boundary.** A window is 512 pixels starting at a chunk boundary, so ours sits at
  exactly 256. **Zero of fifty** strong steps land within two pixels of it, and the positions show no
  periodicity in our chunk grid.
* **two zones repeat a position across DIFFERENT YEARS** — column 30 four times in 30S over 2022 and
  2023, row 432 three times in 26S over 2021 and 2022 — so the line is fixed in geography, not in our
  processing.
* **the embedding step coincides exactly with a step in the input observation count.** Checked on
  three windows: embedding column 30 against observation column 30; embedding row 432 against
  observation row 432, at comparable strength each time.

So these are **Sentinel-2 swath and orbit-overlap boundaries**, which the campaign already knows about
and ignores at the count level. What is new is that they **propagate into the published embeddings**,
visibly, in about one window in sixteen. Two consequences: the validator's seam test examines *shard*
boundaries only and cannot see them, so they will not appear in any verdict; and they are the
mechanism behind §8.

---

## 10. The measurement that finally isolates depth — zone overlaps, 2026-08-13

Adjacent UTM zones overlap by about 200 km and each embeds that ground independently: its own mosaic,
its own ingest, its own per-pixel observation count. So the overlap gives the same ground twice, and —
this is the part every earlier attempt lacked — **blocks where the two runs saw the same number of
observations measure the pipeline's own reproducibility**, which is the yardstick everything else has
to be judged against. 81 blocks across four adjacent pairs (`scripts/zone_overlap_drift.py`).

**The noise floor: 0.00050.** Two completely independent embeddings of the same ground, at matched
depth, differ by that much in cosine distance (n=59, 90th percentile 0.00232).

**Depth of difference moves the embedding**, and two independent methods agree on the slope:

| observation difference | n | median distance | against the floor |
|---|---:|---:|---:|
| 0–1 | 59 | 0.00050 | 1.0× |
| 1–3 | 18 | 0.00140 | **2.8×** |
| 3–6 | 4 | 0.00259 | **5.2×** |

That is 0.00053 of excess distance per observation of difference. The seam measurement, which uses
Sentinel-2 swath boundaries and shares no code or data with this one, gives **0.00060**.

**And depth ITSELF predicts reproducibility** — measured on matched-depth blocks only, so this is not
the difference effect:

| shallower side | n | median distance | against the floor |
|---|---:|---:|---:|
| under 20 obs | 4 | 0.00164 | **3.3×** |
| 20–25 | 4 | 0.00137 | 2.7× |
| 25–30 | 4 | 0.00157 | 3.2× |
| 30–40 | 14 | 0.00020 | 0.4× |
| 40+ | 33 | 0.00032 | 0.6× |

**Thin ground is about five times less reproducible than deep ground**, and the effect is not one
zone's: it appears in all four pairs, thin against deep within each — 1.2×, 2.1×, 3.1× and 6.1×.

> **This is the first evidence in the whole investigation that supports the rule's premise**, and it
> arrives by a completely different route from legibility: not "thin pictures look worse" but "thin
> embeddings are not reproducible". Two runs of the same thin ground disagree several times more than
> two runs of the same deep ground.
>
> **What it cannot yet do is choose a number.** Twelve blocks below 30 observations, four per bin, is
> enough to establish a direction and nowhere near enough to locate where the curve breaks — which is
> exactly what separating 20 from 25 from 30 requires. The bin edges above are mine, not the data's.
> More sampling is the fix and it is cheap: this run took about fifteen minutes and a dollar, and only
> four of seven zone pairs yielded comparable blocks at all.

---

## 11. Where the curve breaks: 741 matched blocks, 2026-08-13

The first overlap sweep established the direction with twelve blocks below thirty observations and
could not locate anything. This one stratifies by depth — a block's depth is screened from the count
array before either embedding is read, and sixteen blocks are taken from each qualifying shard — so
the budget goes where the question is. **813 blocks across five adjacent zone pairs, 741 of them at
matched depth.**

**The floor is 0.00040**, the median distance between two independent embeddings of the same ground
where both runs saw forty or more observations. That is as well as this pipeline ever agrees with
itself, and every figure below is against it.

| shallower side | n | median distance | against the floor |
|---|---:|---:|---:|
| under 15 | 36 | 0.00088 | 2.21× |
| 15–20 | 76 | 0.00152 | **3.83×** |
| 20–25 | 114 | 0.00135 | **3.40×** |
| **25–30** | 114 | 0.00077 | **1.93×** |
| 30–35 | 159 | 0.00057 | 1.43× |
| 35–40 | 22 | 0.00048 | 1.22× |
| 40+ | 220 | 0.00040 | 1.00× |

**The elbow is at 25.** Reproducibility runs between three and four times the floor everywhere below
25 observations, then **halves** crossing into the 25–30 band, and closes on the floor slowly after
that. The single largest improvement available from any cutoff in this range is the one at 25.

> **The line shipped is 15, not this elbow (decided 2026-08-17).** Coverage was chosen over
> reproducibility deliberately: 15 retains 94% of pixels against 79% at 25. A line at 15 admits the
> **worst-reproducing band measured here** — 15–19 at 3.83× the floor, worse than under-15 at 2.21× —
> and the decision was taken with that in view. This section is not withdrawn: it remains the best
> available account of how agreement varies with depth, and it is the reason the trade has a price at
> all. See [`minimum-optical-depth-plan.md`](minimum-optical-depth-plan.md).

> **This elbow is stratified by observation COUNT, and count is confounded with temporal SPREAD
> (added 2026-08-17).** Measured over 919M pixels in 24 cells, mean months covered climbs 5.1 → 9.9
> → 11.5 across the count bands below 40, so the bands here differ in *how much of the year they
> see* as well as in how many looks they got. The elbow may therefore be a spread elbow reported in
> count's units. Two details point that way: only **64.3%** of pixels at 20–24 observations cover ten
> or more months, and the **40+ band is worse-distributed than 30–39** — which is also the shape that
> would explain the anomaly noted below, that the bottom bin reproduces better than 15–20. Full
> figures and limits in
> [`minimum-optical-depth-plan.md`](minimum-optical-depth-plan.md) §1. **Nothing here is withdrawn:**
> the count elbow is a real, measured feature of the data and remains the best available basis for a
> count rule. What is not established is that count is the right *variable*. Settling that means
> re-stratifying this measurement by spread at fixed count.

Read as a decision, cutting at each candidate:

| cutoff | blocks refused | refused side, against floor | kept side, against floor |
|---|---:|---:|---:|
| 20 | 112 | 3.11× | 1.64× |
| **25** | 226 | **3.15×** | **1.33×** |
| 30 | 340 | 2.94× | 1.22× |
| 35 | 499 | 2.36× | 1.06× |

**Choosing 30 over 25 buys little and costs roughly double.** The kept side improves from 1.33× the
floor to 1.22× while the refused population grows by half again — and by 35 the cutoff is eating data
that is already near the floor, which is what the falling refused-side column shows.

**Three things this does not say.**

The bottom bin is not the worst: under 15 observations reproducibility is *better* than at 15–20
(2.21× against 3.83×). One coherent explanation is that agreement needs either the same few inputs or
enough inputs that the sample stops mattering, and the middle is where two runs most easily see
different halves of a moderate set. It is an observation from 36 blocks, not a finding.

**The effect is not universal.** Thin against deep within each pair: 3.0×, 2.8×, 2.1× — and then 1.1×
and 1.1× for the two pairs that both involve zone 48S, which are poor at *every* depth (about 0.0016
either side). Whatever is wrong there is not depth, and it is worth a look on its own.

And this is reproducibility, not accuracy. Two runs agreeing tells you the pipeline is stable at that
depth; it does not tell you the embedding is right. That question still has no measurement, and with
no downstream model in existence it cannot get one.
