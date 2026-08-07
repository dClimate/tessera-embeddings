# Revised inference cost basis for the global TESSERA campaign

**Dated 2026-08-07. APPLIED, and its central figure then superseded the same day.** This is the
working note behind [`campaign-cost-model.md`](campaign-cost-model.md) §6b — kept for the
reasoning, not for the numbers. Two things a reader needs before using any figure below:

1. **It has been applied.** This header once read "a derivation only — nothing in
   `tessera-embeddings` was edited … somebody else applies" these constants. They were applied,
   to `campaign-cost-model.md` §6b and to `tests/unit/test_gpu_starvation.py`, where the
   measurement tables are now typed constants that derive the rate and the cost line live.
2. **Its central line is superseded by §6c.** This note's highest-value recommendation was to
   complete the 37N/2021 fill — "the single measurement to repeat". That was done hours later.
   The completion moved the pooled rate from **2.273 M to 2.127 M** tok/s, the campaign depth
   from **170 to 167** tok/px, the central line from **$527,000 to $553,000**, and it reversed
   which term sets the interval's width. Every rate, depth and dollar figure below is the
   pre-completion value. **The method is current; the numbers are not.**

The two errors the completion exposed are both instances of a mechanism this note itself warns
about — a partial run's whole-cell summary is *biased*, not merely imprecise. Its 37N row read a
30–32° span and a 146.8 tok/px radar depth; the completed cell reads **0.1–32.1°** and **89.9**.

**What this fixes.** The campaign's inference line divides a token census counting **Sentinel-2
plus Sentinel-1** observations by a throughput rate measured in **optical tokens only**. Both
sides are now measurable on one basis, because `CHUNK_SUMMARY` gained `t_s1_asc` and
`t_s1_desc` on 2026-08-06 and four cells have since run under that telemetry.

---

## 1. The headline

| | GPU-hours | cost at $1.861/GPU-h |
|---|---:|---:|
| current model (145 combined tokens/px ÷ 1.90 M optical tok/s) | 288,940 | **$537,700** |
| **revised central** | **283,200** | **$527,000** |
| revised low | 242,700 | $452,000 |
| revised high | 308,200 | $573,000 |

**The line survives the correction: the revised central is 0.98× the current one.** That is not
because the mismatch was small. It is because there were **two** errors of similar size pointing
opposite ways, and fixing only the one that was named would have moved the line 19% in the wrong
direction.

| | current | revised | direction |
|---|---:|---:|---|
| rate per actor | 1.90 M tok/s (optical) | **2.27 M tok/s (combined)** | +19% — line falls |
| campaign depth | 145 tokens/px | **170 tokens/px** | +17% — line rises |
| **pixels per second per actor** | **13,103** | **13,371** | **+2.0%** |

Because the capacity-planning rate barely moves, **fleet sizing, the 85%-provisioning policy and
the starvation bank are all unaffected.** Only the dollar line and the token census change, and
the dollar line barely.

**Two findings worth more than the revised number itself.**

1. **The combined-token rate is invariant to radar status; the optical rate is not.** Within a
   single run the combined rate is the same for both-orbit, one-orbit and radar-free chunks to
   within 3%, while the optical rate over the same chunks varies by 1.25–1.60×. That is the
   empirical proof that combined tokens are the right unit — the same shape of evidence that
   settled tokens-versus-pixels (tok/s flat to ±1% while px/s varied 2.2×).
2. **Radar depth does not vary with latitude.** Optical depth does, strongly and monotonically.
   The census's *total* is roughly right; its optical/radar *split* is close to inverted at high
   latitude. See §3 and §4.

---

## 2. The measurements, re-derived

Every figure below was re-run from CloudWatch `CHUNK_SUMMARY` records over a 96-hour window
(`/ecs/global-tessera-dev`, 29,886 successful chunks, 1,633 of them carrying radar telemetry),
using the truncation-bisecting query in `yield-embeddings/scripts/inference_profile.py`.

**The transcriptions in the brief all check out.** No figure differed.

| run | cell | chunks | `t_kept` med | asc / desc med | optical tok/s | $/chunk | radar basis |
|---|---|---:|---:|---:|---:|---:|---|
| `p4d2-47S-2020-w16` | 47S/2020 | 235 / 239 live | 61 | 30 / 31 | 1.301 M | $0.118 | both on 228 of 235 |
| `p4d2b-60N-2020-w16-run2` | 60N/2020 | 527 / 620 live | 145 | 22 / 30 | 1.693 M | $0.177 | 326 both, 199 one, 2 free |
| `assembly-baseline-37N-2021-stagekept` | 37N/2021 | 3,855 / 8,731 live | 121 | — | 1.085 M | $0.232 | telemetry predates the fields |
| `p6-duty-38N-2021-60a` | 38N/2021 | 9,051 / 9,100 live | 73 | — | 1.227 M | $0.177 | telemetry predates the fields |

38N also reproduces exactly: 859.9 GPU-hours, $1,600.

**Two runs the brief did not know about, and they matter.** Both carry radar telemetry and both
are 37N/2021, so the programme has **four** both-orbit cells, not two:

| run | chunks | \|lat\| of its chunks | optical | radar | combined |
|---|---:|---|---:|---:|---:|
| `assembly-dense-37N-2021-w160` | 141 (45 both-orbit) | 65–69° | 164 | 100 | 264 |
| `assembly-dense-37N-2021-resume` | 203 (163 both-orbit) | 30–32° | 100 | 147 | 246 |

`assembly-dense-37N-2021-resume` was **still running** when this was written (last chunk
18:16:58Z against a 18:29Z clock), and it is 203 chunks into a 8,731-tile cell. It is used here
**only per latitude band**, never as a whole-cell median — which is what makes an in-flight
sample admissible. The bias that ruined earlier 37N and 38N figures is a bias in *whole-cell*
medians, caused by a run sweeping north to south while depth falls with latitude. Attributing
each chunk to its own latitude removes that mechanism rather than tolerating it.

**Latitudes are derived, never guessed from the zone name.** Each chunk's `chunk_<row>_<col>`
label gives a tile row; the campaign mask's per-zone `grid_shape` and `tile_px` turn that into a
northing and then a latitude. The zones involved are nothing like their numbers suggest:

| zone | \|lat\| of its live tiles | what it actually is |
|---|---|---|
| 47S | 0.1 – 12.4° | equatorial: Sumatra and Christmas Island, not "far south" |
| 60N | 51.1 – 71.5° | Chukotka and the western Aleutians |
| 37N | 0.1 – 70°+ | Turkey to East Africa to Arctic Russia — **spans every latitude** |
| 38N | 0.1 – 81° | likewise |

37N spanning 0–70° is why its three runs report `t_kept` medians of 72, 121 and 149: they are
sampling different latitudes of one zone, not disagreeing.

**One duplicate.** Streams `p4d2b-60N-2020-w16` and `p4d2b-60N-2020-w16-run2` each carry the
same 527 chunks with identical depths — the cell was filled twice against one mosaic. Only
`-run2` is counted. Their combined rates differ by **0.3%**, which is a free measurement of the
run-to-run noise floor on this instrument and is worth holding: any effect smaller than about
1% is noise.

---

## 3. Task 1 — 60N's rate, decomposed by radar basis

60N/2020's 1.69 M is an average over 326 both-orbit, 199 one-orbit and 2 radar-free chunks. The
profiler refuses to quote one rate for it, correctly. Split:

| basis | n | optical depth | radar depth | combined depth | **optical tok/s** | **combined tok/s** | $/chunk |
|---|---:|---:|---:|---:|---:|---:|---:|
| both orbits | 326 | 137.4 | 76.4 | 213.9 | **1.562 M** | **2.431 M** | $0.187 |
| one orbit | 199 | 158.0 | 44.1 | 202.1 | 1.946 M | 2.489 M | $0.162 |
| radar-free | 2 | 138.6 | 0 | 138.6 | — (n=2) | — (n=2) | — |
| whole cell | 527 | 144.9 | 64.7 | 209.6 | 1.693 M | 2.449 M | $0.177 |

Depths are pixel-weighted (`Σ depth × valid_px ÷ Σ valid_px`), not medians, because the rate is
a pixel-weighted quantity and mixing the two invites a mismatch of exactly the kind being
corrected.

**60N's both-orbit rate is 1.562 M optical, comparable with 47S's 1.292 M.** The gap between
them is real and is what a single optical rate cannot describe.

**The decisive observation is the last column but one.** On the combined basis the three
populations agree — 2.431, 2.489 and 2.449 M — while on the optical basis they span 1.562 to
1.946 M, a 1.25× spread. The same holds in every cell that has the telemetry:

| cell | combined tok/s: both | one orbit | spread | optical tok/s: both | one orbit | spread |
|---|---:|---:|---:|---:|---:|---:|
| 47S/2020 | 2.499 M | 2.650 M | 1.06× | 1.292 M | 1.747 M | 1.35× |
| 60N/2020 | 2.431 M | 2.489 M | 1.02× | 1.562 M | 1.946 M | 1.25× |
| 37N w160 | 2.579 M | 2.555 M | 1.01× | 1.601 M | 2.051 M | 1.28× |
| 37N resume | 1.870 M | 1.729 M | 1.08× | 0.756 M | 0.937 M | 1.24× |

**Four cells, four times, the combined basis collapses a 1.24–1.35× spread to 1.01–1.08×.** This
is a within-run comparison, so geography, day, code version and fleet are all held constant.

**And the two radar-free cells corroborate it from the other end.** For a radar-free cell the
optical rate *is* the combined rate by definition, and 23N/2021 (1,395 chunks) runs at 2.934 M
while 57S/2022 (267 chunks) runs at 2.258 M — bracketing the both-orbit combined figures rather
than sitting far above them, which is what the optical basis made them look like.

### The pooled rate, and its one outlier

| population | n | combined tok/s per actor |
|---|---:|---:|
| all radar-telemetry chunks | 1,106 | 2.307 M |
| **both orbits, pooled** | **762** | **2.273 M** |
| one orbit | 342 | 2.399 M |

Per cell, both-orbit only: **2.499 M (47S), 2.431 M (60N), 2.579 M (37N at 66–69°), 1.870 M
(37N at 30–32°).** Three of four agree within 6%. The fourth is 30% below and it is not
explained:

- **Not the strip plan.** 37N-resume runs 2 strips per chunk at `strip_h` 1792; 60N runs 3 at 944.
  The slower cell has the *simpler* plan.
- **Not bucket fragmentation.** Median buckets per strip: 13 for 37N-resume against 12 for 60N
  and 12 for 37N-w160. The hypothesis was that heterogeneous per-pixel observation counts in a
  densely-overlapped SAR region would fragment the batches; the log line
  `MosaicChunkInferenceDataset: … in %d buckets` refutes it directly.
- **Not per-chunk overhead.** Inference is 91% of its chunk wall-clock, and on inference time
  alone it is still the outlier (2.07 M against 2.82–3.12 M for the other three).
- **A radar-token premium is consistent with it but contradicted elsewhere.** 37N-resume is the
  only radar-dominated cell measured (radar 1.47× optical; every other cell has radar ≤ optical).
  But *within* 60N the bands with the highest radar share are the *cheapest* per combined token
  (0.356 µs/token at 48% radar share against 0.419 at 37%), which is the opposite sign.

A two-coefficient regression pricing optical and radar tokens separately **is not identifiable
from this data** and must not be used: the radar-to-optical cost ratio comes out 0.26×, 0.39×,
0.74× and 1.42× on the four cells — it changes sign between them. Optical and radar depth are
collinear within a run, so the regression has no leverage. This is worth stating explicitly
because the fit reports R² 0.79 and looks respectable.

---

## 4. Task 2 — radar depth against latitude

**Pooled both-orbit chunks only, by 5° band of \|latitude\|:**

| \|lat\| | % land | zones | n | optical | radar | combined | radar share | µs/valid px |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 0–5 | 7.4% | 47S | 228 | 71.0 | 66.3 | 137.3 | 48% | 54.9 |
| 30–35 | 7.9% | 37N | 163 | 99.6 | **146.8** | 246.4 | 60% | 131.7 |
| 50–55 | 6.1% | 60N | 19 | 100.1 | 93.5 | 193.6 | 48% | 69.0 |
| 60–65 | 5.4% | 60N | 211 | 135.8 | 70.3 | 206.1 | 34% | 83.1 |
| 65–70 | 4.8% | 37N, 60N | 141 | 152.9 | 91.0 | 243.9 | 37% | 102.1 |

**The finding: radar depth is not a function of latitude.** Across these five bands radar depth
is 66, 147, 93, 70, 91 — a 2.21× spread with **Pearson r = +0.009** against band midpoint. Over
the same five bands optical depth gives **r = +0.912**. One relationship is real; the other is
absent.

**This is stronger than the brief's expected answer and it is a different answer.** The brief
anticipated "radar depth is approximately flat in absolute terms, so its share falls as optical
rises". Flatness is not what the four cells show — radar depth varies 2.2× — but it varies
**regionally, not latitudinally**. The deepest radar measured anywhere is at 30–32°N in zone 37N
(longitudes 36–42°E: Israel, Jordan, north-west Saudi Arabia) and the shallowest is at 60–70°N.
The plausible mechanism is the Sentinel-1 observation plan, which tasks the Middle East and
Europe far more densely than remote tropics or Arctic tundra. **Two consequences:**

1. **Do not build a latitude model for radar.** Any such model is fitting noise. The defensible
   form is a constant with a stated spread: **radar depth for a both-orbit chunk is 90 tokens/px
   (measured pooled mean), range 66–147 across the five measured bands.** No curve, no
   extrapolation.
2. **Radar's *share* of the sequence does fall with latitude** — 48% at the equator to 34–37%
   above 60° — but that is the optical denominator rising, not the radar numerator falling. Two
   different mechanisms, and only one of them is understood.

**One-orbit chunks carry about half a both-orbit chunk's radar.** Measured per cell:
31 / 66, 44 / 76, 39 / 100, 72 / 147 → ratios 0.47, 0.58, 0.39, 0.49, mean **0.48**. "Half" is
supported, not assumed.

**A radar-free chunk carries 8 tokens/px, and that is a code fact rather than a measurement.**
`DEFAULT_NUM_OBS_CHECKPOINTS` is `range(8, 257, 8)`, so the smallest S1 bucket is 8, and
`resample_s1_bucket` hands the model an all-zeros slice of that length. It is not zero, and at
0.068 of pixel-years it is immaterial either way (0.5 tokens/px of the campaign mean).

**Uncertainty on the radar finding.** Five band-observations from four cells across three zones.
The 30–35° point is a single 203-chunk in-flight sample and the 50–55° point is **19 chunks**.
The absence of a latitude relationship is well supported (a 2.2× spread with r ≈ 0 cannot be
rescued into a trend), but the *level* — 90 tokens/px — rests on very little, and it is the
single largest driver of the interval in §7.

---

## 5. Task 3a — the census's 145 is COMBINED, established

**This is the crux and it is settled: 145 counts optical plus radar.** Three independent
confirmations:

1. **The census table's own arithmetic.** Land-weighting the seven latitude bands of
   `campaign-cost-model.md` §6 by their land area gives S2 = 52.0 and S1 = 90.7 observations per
   pixel, matching the printed 52 and 91; land-weighting the `tokens/px` column the same way
   gives **152.1**, matching the printed 152. And 52 + 91 = 143 ≈ 152. The `tokens/px` column is
   the sum of the two halves plus a small per-band bucket allowance. **145 is 152 after the
   optical-only-cell discount**, so 145 is a combined figure.
2. **The document says so, repeatedly and in its own correction note.** "The census above counts
   **S2 + S1**: 52 optical plus 91 radar observations per pixel, land-weighted." The 9%
   agreement between `t_kept × valid_px` and the pipeline's own token metric is an
   optical-against-optical instrument self-check, not evidence that 145 is optical.
3. **60N's `t_kept` of exactly 145 is a coincidence, and it is now demonstrable rather than
   suspected.** 60N's *combined* depth is 210, not 145. Its band's census figure is 208. The
   coincidence is between an optical measurement and a combined census figure that happen to
   collide at 145, and reading it as agreement would have inverted the correction.

### The census's total is close; its split is nearly inverted

| | census for that band | measured (pixel-weighted) |
|---|---|---|
| **60N/2020** (\|lat\| 51–70°, census band +60 to +80) | optical 48, radar 156, **total 208** | optical 145, radar 65, **total 210** |
| **47S/2020** (\|lat\| 0–4°, census band −20 to 0) | optical 44, radar 49, `tokens/px` column **104** | optical 71, radar 65, **total 136** |

**The combined totals agree to 1% at 60N and are 31% low at 47S. The splits are wrong in both
places, and at 60N they are close to swapped.** Across the whole corpus the same picture holds:
the census's land-weighted optical 52 against a measured 103, and its radar 91 against a
measured composition-weighted 67.

**Two consequences.**

- **For this revision, only the total matters,** so the numerator survives being built on a
  badly-split census. But it must be built on the *measured* total, not the censused one — see
  the convention point below.
- **For anything that uses the split, the census is not fit for purpose.** `zone_work_weight`
  weights the zone-to-cluster partition by each latitude band's censused observation count
  (`campaign-cost-model.md` §9 item 3). That balancing needs only the *ratios between bands*,
  which is the robust part, and the measured dynamic range (123 → 243 across bands) matches the
  censused one (104 → 208) — so the fix is probably not urgent. **It has not been checked and
  should be.**

### The convention that must not be mixed — a third unit, and it is why 170 ≠ 145

There are **three** distinct definitions of "observations per pixel" in play, and pairing the
wrong two reintroduces exactly the class of error being corrected:

| | what it counts |
|---|---|
| **census** (145) | per-pixel observations: distinct acquisition dates × mean clear fraction, granules ÷ cos(lat) |
| **chunk-array depth** (`t_kept + t_s1_asc + t_s1_desc`) | the depth of the chunk's arrays — a date is present if *any* pixel in the chunk kept it |
| **what the model processes** | each pixel's *own* valid count, rounded up to a multiple of 8 |

Chunk-array depth is an **upper bound** on what the model processes, so `depth × valid_px`
overstates true tokens. **That overstatement cancels exactly when the numerator and denominator
are both on the chunk-array convention** — which is the case for `measured depth ÷ measured
rate`, since the rate is `chunk-array tokens ÷ measured seconds` over the same chunks.

**So the only self-consistent pair available is measured-depth ÷ measured-rate.** Pairing the
census's 145 with a chunk-array rate of 2.27 M would understate GPU-hours; it is a different
mismatch of the same kind, one level down. That is the reason the revised depth is 170 rather
than 145, and most of that difference is convention rather than disagreement.

The convention-free statement of the same result, which carries no definitional risk at all:
**74.8 microseconds of busy GPU time per valid pixel, land-weighted.**

---

## 6. Task 3b — the revised census and the line, with the arithmetic

### The optical half: land-weighted depth 103.1

Per-chunk latitudes over the whole 96-hour corpus — **22,343 chunks across 13 zones, and every
one of the 17 populated 5° bands now has data**, against 2,779 chunks and 74% band coverage in
the existing stratification:

| \|lat\| | % land | live tiles/yr | n | optical depth (px-wtd) | median | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–5 | 7.4% | 26,811 | 1,172 | 74.0 | 70 | 53 | 117 |
| 5–10 | 7.7% | 27,940 | 1,532 | 77.1 | 69 | 57 | 134 |
| 10–15 | 7.4% | 26,679 | 742 | 80.3 | 72 | 64 | 142 |
| 15–20 | 8.3% | 29,785 | 963 | 87.2 | 72 | 70 | 144 |
| 20–25 | 8.8% | 31,732 | 851 | 89.2 | 73 | 72 | 145 |
| 25–30 | 8.7% | 31,429 | 811 | 96.6 | 73 | 70 | 145 |
| 30–35 | 7.9% | 28,399 | 1,751 | 95.7 | 72 | 67 | 141 |
| 35–40 | 6.4% | 23,189 | 1,698 | 99.6 | 72 | 65 | 139 |
| 40–45 | 6.4% | 23,160 | 1,922 | 103.2 | 121 | 62 | 134 |
| 45–50 | 6.3% | 22,833 | 2,030 | 107.8 | 120 | 60 | 130 |
| 50–55 | 6.1% | 21,887 | 1,811 | 120.5 | 122 | 112 | 130 |
| 55–60 | 4.9% | 17,653 | 1,612 | 133.7 | 122 | 112 | 181 |
| 60–65 | 5.4% | 19,358 | 2,820 | 142.0 | 150 | 105 | 169 |
| 65–70 | 4.8% | 17,168 | 2,397 | 154.6 | 147 | 135 | 190 |
| 70–75 | 2.1% | 7,529 | 436 | 164.5 | 167 | 138 | 184 |
| 75–80 | 1.1% | 4,032 | 235 | 175.3 | 174 | 165 | 187 |
| 80–85 | 0.4% | 1,369 | 87 | 175.5 | 176 | 167 | 182 |

**Land-weighted optical depth = 103.1** (per-band p25 85.4, p75 115.6). The band table's live
tiles sum to **360,953**, which is the coverage census's own figure to the tile — so the land
weighting is the campaign's real geography, not a proxy.

The monotonic rise, 74.0 to 175.5 with no reversal in 17 bands, confirms the existing step-function
finding and supersedes its 106 central (75–139) with a better-supported 103.

### The radar half: composition-weighted 67

Radar composition from `campaign-cost-model.md` §6, area-weighted per pixel-year over the nine
campaign years: dual-orbit **0.55**, single-orbit **0.38**, radar-free **0.068**.

```
  radar tokens/px  =  0.55 × 90        (dual, measured pooled mean)
                   +  0.38 × 0.48 × 90 (single, at the measured 0.48 of dual)
                   +  0.068 × 8        (radar-free, the smallest S1 bucket)
                   =  49.5 + 16.4 + 0.5
                   =  66.5  ->  67
```

### The line

```
  campaign combined depth  =  103.1 optical  +  66.5 radar  =  169.6  ->  170 tokens/px
  campaign tokens          =  1.363e13 px  x  170            =  2.317e15
  rate                     =  2.273e6 combined tok/s per actor  (762 both-orbit chunks)

  GPU-hours  =  2.317e15  /  2.273e6  /  3600  =  283,200
  cost       =  283,200  x  $1.861               =  $527,000

  px/s equivalent  =  2.273e6 / 170  =  13,371   (current basis: 1.90e6 / 145 = 13,103)
  seconds per valid pixel  =  170 / 2.273e6  =  74.8 microseconds, land-weighted
```

### Cross-checks

**Independent route: measured cost per chunk, using neither the token census nor the rate.**
Both-orbit dollars per chunk by band: $0.119 (0–5°), $0.286 (30–35°), $0.150 (50–55°), $0.180
(60–65°), $0.221 (65–70°). Those five bands hold 31% of land and land-weight to **88.8
microseconds per valid pixel**, which across 3.25 M chunk-years is **$625,000 for a campaign that
were entirely both-orbit**. The central model evaluated on **the same five bands** gives 78.6 µs,
1.13× lower. That gap is the right sign and roughly the right size: the measured column is
**both-orbit only**, while the model is composition-weighted over a land base that is 45%
single-orbit or radar-free. Two routes that share no inputs agreeing within 13% once composition
is allowed for is the strongest evidence here that the central figure is sound.

**Pixel count.** 13 cells completed to ≥90% give a tile-weighted valid-pixel yield of
`Σ valid_px ÷ (live tiles × 2048²)` = **0.984**, so the 1.363e13 pixel census is about 1.6%
optimistic. Immaterial, and **not** applied above. The spread is worth noting though: 47S yields
0.983 while 60N yields 0.811, because 93 of 60N's 620 live tiles produced zero valid pixels.
Whether high-latitude cells systematically yield less is unmeasured and would reduce the line if
so.

---

## 7. Task 4 — the interval and what sets its width

**$452,000 – $573,000, central $527,000.** Against the current $537,700 that is 0.84× to 1.07×,
central 0.98×. Well inside the factor-of-two escalation bound.

The bounds are **coherent constructions, not products of independent extremes.** This matters:
the 30–32°N band supplies both the deepest radar (147) and the slowest rate (1.87 M), and those
are the *same 163 chunks*. Multiplying "max radar" by "worst rate" double-counts one
observation and produces $845,000, which nothing supports. The primitive quantity is seconds per
pixel, and it is measured directly.

| construction | what it assumes | GPU-hours | cost |
|---|---|---:|---:|
| **low** — three-cell consensus | 37N/30–32° is unrepresentative; radar 74, rate 2.465 M everywhere | 242,700 | **$452,000** |
| **central** — pooled both-orbit | the 762 measured both-orbit chunks are the population; radar 90, rate 2.273 M | 283,200 | **$527,000** |
| **high** — band-attributed | each band takes its nearest measured anchor's own (radar, rate) pair, which extends 37N/30–32° over the 20–45° bands (38% of land) | 308,200 | **$573,000** |

Each of the four cells taken alone, its own (radar depth, rate) pair applied to the land-weighted
optical depth: **$429,000 (47S), $462,000 (60N), $483,000 (37N at 66–69°), $796,000 (37N at
30–32°).**

### What drives the width, largest first

1. **The 37N/30–32° cell, and it is most of the width.** One 203-chunk in-flight sample carries
   both the deepest radar and the slowest rate, and both are unexplained after ruling out the
   strip plan, bucket fragmentation and per-chunk overhead. It alone separates $483,000 from
   $796,000. Its band holds 7.9% of land and the 20–45° span it anchors holds **38%** — so it
   cannot be dismissed as a corner case. **This is the single measurement to repeat.**
2. **Radar depth's level, not its shape.** 90 tokens/px rests on five band-observations from
   three zones, spanning 66–147. Holding everything else at central, radar 66 gives $471,000 and
   radar 147 gives $655,000 (both at the central rate). The *absence of a latitude relationship* is solid; the level is not.
3. **The 45–60° assignment rests on 19 chunks.** The 50–55° anchor — which the band-attributed
   construction extends over 17.3% of land — is 19 both-orbit chunks from 60N, and it happens to
   carry the *fastest* rate measured (2.807 M). That makes the high bound optimistic in one place
   while pessimistic in another.
4. **Optical depth is now the smallest term.** Swinging it to its per-band p25 or p75 (85.4 or
   115.6) moves the line to $471,000 or $564,000 — a ±9% band on 22,343 chunks covering 100% of
   land. It used to be the largest unvalidated input; it is now the best-measured one.
5. **No both-orbit rate has been measured at campaign fleet width.** The four cells ran at 20, 20,
   55 and 95 actors against a campaign 228 per cluster. There is no actor-count trend in the four
   (55 actors gave the fastest rate and 95 the slowest), so there is no evidence of a problem —
   but there is also no measurement, and the direction is unknown.
6. **Convention risk, direction known.** Chunk-array depth exceeds what the model processes, and
   the cancellation in §5 is exact only if the overstatement factor is the same in the projection
   as in the rate measurement. It is not exactly the same — the projection's optical depth comes
   from a 30-run corpus and the rate from four cells. Any residual pushes the line **down**,
   because a smaller true depth divided by an unchanged measured rate buys fewer GPU-seconds.

### On elasticity

The existing model's warning that cost is only ~0.3 elastic to the token census **does not apply
to this revision, and assuming it did would understate the change threefold.** That elasticity
describes varying tokens at a *fixed all-in per-chunk cost*, where fixed overhead dilutes the
effect. Here the rate is re-measured on the same chunks as the depth, so the two move together
and the elasticity is 1.0 by construction — a 17% depth rise is a 17% cost rise before the rate
correction offsets it. The fixed per-chunk term is small in the both-orbit population anyway:
inference is **90.3%** of both-orbit chunk wall-clock, and a pooled two-term fit puts the fixed
intercept at 3–15% of a mean chunk depending on the cell.

**What remains true is the underlying advice**: per-chunk overhead, not tokens, is the lever
worth engineering, because it pays on every chunk and pays most on token-poor land.

---

## 8. The constants to change in `tests/unit/test_gpu_starvation.py`

All five live in the `--- inference rate ---` block, lines 116–135. **The behavioural
consequence is small: `RATE_CAPACITY_PLANNING` moves +2.0%, so every starvation, bank and
fleet-sizing assertion in the file should still pass unchanged.** Run it and confirm rather than
assuming — `BANK_WORK_HOURS = 2.0` was set *at* its threshold, so a 2% shift is not guaranteed to
be free.

| constant | old | **new** | why |
|---|---|---|---|
| `TOK_PER_SEC` | `1_900_000.0` | **`2_273_000.0`** | pooled combined-token rate over 762 both-orbit chunks (§3). **Rename it** — `COMBINED_TOK_PER_SEC` — so the basis is in the name and a future edit cannot silently reintroduce an optical rate. |
| `CAMPAIGN_TOK_PER_PX` | `145.0` | **`170.0`** | 103.1 land-weighted optical (22,343 chunks, 17/17 bands) + 66.5 composition-weighted radar (§6). Chunk-array convention, matching `TOK_PER_SEC`. |
| `RATE_CAPACITY_PLANNING` | derived, `~13_103` px/s | **derived, `~13_371` px/s** | `2_273_000 / 170`. Update the trailing comment. |
| `IOWA_TOK_PER_PX` | `136.0` | **leave at 136.0, but see below** | Iowa has never been measured under radar telemetry, so 136 is a *census* figure in a different convention from the new 170. Comparing them is a convention mix. |
| `RATE_MID_WHILE_PROCESSING` | `22_500.0` | **delete it** | Dead: defined at line 135 and referenced nowhere in the repo. If a while-processing rate is wanted, the combined-basis equivalent is `2_517_000 / 170 = 14_806` px/s (inference-only rate, excluding the 9.7% overhead). |

**One assertion must change with them.** `test_the_campaign_costs_more_per_pixel_than_the_site_every_rate_came_from`
(line 532) asserts `CAMPAIGN_TOK_PER_PX / IOWA_TOK_PER_PX == approx(1.066, abs=0.01)`. At 170 the
ratio is **1.25**, so the test fails as written.

**The recommendation is to drop the exact figure rather than update it to 1.25.** The test's
purpose is to guard the *inequality* — that the campaign is dearer per pixel than the single-orbit
ROI every historical rate came from — and that guard is now stronger, not weaker. The exact ratio
is not defensible because its two sides are in different conventions (§5). Keep
`assert IOWA_TOK_PER_PX < CAMPAIGN_TOK_PER_PX` and the two rate assertions, which all still hold;
replace the `approx(1.066)` line with a comment recording that Iowa's combined depth is
un-measured and naming the run that would fix it.

**Also worth updating, though not a constant:** the docstring at lines 117–127 states "1.9M
tok/sec is the reference per-worker rate; 145 tok/px is the campaign's land-weighted observation
count". Both halves change, and the docstring should say which *unit* the rate is in — its
absence is what allowed the mismatch to sit in the model's central division for three revisions.
Per the repo's documentation convention, the docstring states the concept and the invariant
("both sides of this division must count the same modalities"); the numbers and their provenance
belong in `campaign-cost-model.md`.

---

## 9. What further measurement would most narrow the interval

In order of interval reduction per unit of effort.

1. **Complete 37N/2021, or fill any zone whose live tiles sit in 20–40°N, under radar
   telemetry.** This is worth more than everything else combined: it is the difference between
   $483,000 and $796,000, it anchors 38% of land, and 37N is already ingested and mid-fill. A
   completed 37N also settles whether the 30–32° rate deficit is real or an artefact of a
   203-chunk in-flight sample. **If one measurement is taken, take this one.**
2. **Read `s1_asc_obs_count` and `s1_desc_obs_count` off a completed store.** This is the only
   way to close the convention gap in §5, and the store already writes both per pixel. It would
   convert the chunk-array depth into the per-pixel depth the model actually processes, retire
   the "cancellation is exact" assumption, and let the census be compared to the measurement
   like-for-like for the first time. It costs a read, not a run.
3. **One both-orbit cell at campaign fleet width — 228 actors on one cluster.** Every rate we
   hold comes from 20–95 actors. There is no evidence of an actor-count effect and no measurement
   of one, and the whole line scales inversely with this number.
4. **A both-orbit cell in the 45–60° band with more than 19 chunks.** That band anchors 17.3% of
   land on 19 chunks that happen to carry the fastest rate measured anywhere.
5. **Whether the radar-depth spread is regional, as hypothesised.** Two both-orbit cells at the
   same latitude in different longitudes — one in a densely-tasked Sentinel-1 region, one remote —
   would confirm or kill the observation-plan mechanism. This does not narrow the cost interval
   much on its own, but it decides whether radar depth can ever be *modelled* or must always be
   measured per region, which changes how much measurement the campaign owes in total.

**What would NOT help:** another high-latitude cell. 60–85° is anchored on 5,975 chunks across
seven zones and the bands agree.

---

## 10. Provenance

- **Source**: CloudWatch Logs Insights, log group `/ecs/global-tessera-dev`, 96-hour window
  ending approximately 2026-08-07T18:19Z, `CHUNK_SUMMARY` records deduplicated on
  (stream, timestamp, label). 29,886 successful chunks over 31 streams.
- **Query machinery**: `yield-embeddings/scripts/inference_profile.py` `_query` / `_chunks` /
  `_stream_labels`, imported rather than reimplemented, so the truncation bisection and the
  double-logging dedup are the same code the profiler reports through.
- **Latitudes**: the campaign land mask (`branch_scoped_dev_buckets(...).land_mask_store("global")`),
  per-zone `grid_shape[0]` and `tile_px`, tile-row centre northing ÷ 111,320 m/deg. The mask's
  live tiles sum to 360,953, matching the coverage census.
- **GPU price**: $1.861/h, `g6e.xlarge` on-demand us-west-2, as `inference_profile.py`'s
  `GPU_HOURLY`.
- **Attribution**: by log stream, never by chunk label — `CHUNK_SUMMARY` carries no zone and its
  `chunk_<row>_<col>` label is grid-local and collides across cells.
- Runs still in flight are used **per latitude band only**, never as whole-cell medians.
