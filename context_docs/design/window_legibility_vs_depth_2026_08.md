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

## 7. A separate finding: straight-edge artefacts, raised independently by four reviewers

Not part of the depth question, and recorded because it arrived unprompted from sessions that had no
knowledge of each other. Across the 713 notes, **33 mention a straight edge, a seam or an
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

**The test that would settle it is cheap and has not been run.** A window is 512 pixels starting at a
chunk boundary, so an internal chunk boundary sits at exactly column or row 256. Locate the strongest
straight edge in each flagged window and look at where it falls: positions clustering at 256 implicate
our chunking, scattered positions implicate acquisition geometry upstream — and the one measured
position so far, column 77, is not a chunk boundary. Worth doing before the campaign, since the seam
test in the validator only examines *shard* boundaries and would not see any of these.
