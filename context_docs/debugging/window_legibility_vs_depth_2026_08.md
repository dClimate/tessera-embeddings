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

## 4-9. What the legibility route concluded, and why it was abandoned

**Six sections here priced candidate cutoffs by legibility, and §7's own conclusion is that
legibility was the wrong measure.** They are cut rather than kept because the line was then decided
against them twice over — the picture is a three-component projection of 128 dimensions, one reviewer
measured illegible windows carrying coherent structure at a tenth of the rendered amplitude, and the
rule's argument rests on usefulness, which no measurement here tested. What survives:

**Legibility does not track depth, and the gradient runs backwards.** Deeper windows are *less*
often legible, because depth correlates with cloud-prone geography and the contrast stretch is shared
across a cell. At every cutoff from 15 to 30 about 85% of what it refuses is legible, and the
illegible share of what a cutoff KEEPS rises monotonically to 30% at 30 — refusing more made the
surviving product proportionally worse.

**What does depend on observation count is a measurable bias**, not legibility: the straight-edge
artefacts visible in some windows are upstream, present in the source imagery rather than introduced
by assembly, which was checked rather than assumed.

**The reason this matters beyond this document:** the original 30 line rested on how pictures looked,
and the blind re-measurement is what dislodged it. The failure mode was that the renderer's
per-window contrast stretch expanded a low-variance window's noise floor to full contrast, so
"noisy below 20" was partly the renderer. That is why §10-11 below abandon pictures entirely and
measure agreement between two independent embeddings of the same ground.

Full working in git history.

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
