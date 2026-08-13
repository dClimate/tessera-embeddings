# Does picture quality track optical depth? A blind re-measurement — 2026-08-13

**Why this exists.** The campaign's records have carried one sentence about embedding quality since
July: *crisp at 30 or more valid observations per pixel, noisy below about 20*, measured across 15
cells. [`minimum-optical-depth-plan.md`](minimum-optical-depth-plan.md) proposes to make that
sentence into a **refusal**: a pixel with fewer than 30 observations would not be embedded at all.
That decision is irreversible — a refused pixel does not exist, so recovery is a re-run — and on
2026-08-13 two defects were found in the instrument that produced the original sentence.

**Result, stated first: the claim is not supported, and a 30 line does not sit where the plan
assumes.** The thinnest legible window in the corpus is at **12.8** observations per pixel, and a
fully-land window at **41.6** observations is structureless confetti. Both were confirmed by eye
after the blind pass, because they are the two cases the decision turns on.

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

**73 windows from 9 completed dev cells**, spanning **13 to 98** observations per pixel, 18 of them
below the 30 line. Thin windows are not evenly available: they come from equatorial cloud (32S/2022,
Congo basin, latitude −0.3 to −4.9) and from the sub-Antarctic (26S/2021, latitude −56 to −59).

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

## 3. The result

All 73 windows reviewed. **Restricted to the 54 fully-land windows**, which is the only comparison
that isolates depth from how much of the frame is water:

| depth band | n | legible | uncertain | illegible | % legible |
|---|---:|---:|---:|---:|---:|
| under 20 | 10 | 5 | 2 | 3 | **50%** |
| 20–30 | 2 | 2 | 0 | 0 | 100% |
| **30–45** | 9 | 3 | 0 | 6 | **33%** |
| 45+ | 33 | 32 | 1 | 0 | **97%** |

| split at the proposed line | n | % legible |
|---|---:|---:|
| below 30 — **would be refused** | 12 | **58%** |
| 30 and above — would be kept | 42 | 83% |

**Three readings, in descending order of how well the data supports them.**

**The claim as written fails in both halves.** Half the fully-land windows below 20 observations are
legible, and only a third of those between 30 and 45 are. The relationship is **not monotonic**: the
band immediately above the proposed cutoff is the worst-performing band in the corpus, worse than the
thinnest one.

**Depth does still carry signal, but the only clean break is at about 45**, not 30 — 97% legible above
it against 33% just below it. A line drawn at 45 would refuse a quarter of the product's pixels or
more, which nobody has proposed and which the 2017 figures make untenable.

**Land cover explains more than depth does.** Land fraction alone accounts for 15 of the 24 illegible
verdicts: of the 19 windows the mask calls less than fully land, 15 are illegible, and that is the
correct output for a frame that is mostly water. Within fully-land windows, how structured a window is
separates legible from illegible better than its depth does.

## 4. The two cases the decision turns on

Both re-examined directly rather than accepted from the blind pass.

**`32S/2022` window 7 — 12.8 observations per pixel, fully land, LEGIBLE.** Sharp lobed boundaries
between strongly distinct covers, dark channel-like forms wrapping a bright lobe, and a corridor with
parallel margins running about 5 km down the frame: a river or valley corridor with adjoining wetland.
At less than half the proposed cutoff, this is recognisable ground, and the rule as specified would
discard it.

**`16S/2021` window 1 — 41.6 observations per pixel, fully land, ILLEGIBLE.** Structureless per-pixel
confetti with no organisation at any scale. Above the cutoff, so the rule would keep it.

## 5. What this measurement cannot settle

**Thin data and its geography are entangled, and the corpus is small.** Eight of the twelve fully-land
windows below 30 come from one cell. You cannot obtain thin optical data except in places that are
cloudy or dark for months, and those places have their own land cover — so "below 30 is 58% legible" is
substantially a statement about the Congo basin in 2022. Nine cells cannot separate depth from biome.

**Nothing here is measured below 12.8 observations**, which is the corpus floor. A refusal line low
enough to sit under every legible window in this corpus is therefore unproven rather than supported.

**Legibility is not utility, and this is the limit that matters most.** The picture is a
three-component projection of a 128-dimensional embedding, so a window a reader cannot interpret may
still carry signal a downstream model uses; the first three components explain about 67%, 8% and 6% of
variance, and nothing about the other 122 dimensions appears in the frame. This measurement therefore
bounds one thing only — whether the *pictures* justify the sentence the plan quotes. **It cannot
establish that thin embeddings are useless, and the plan's reputational argument rests on a claim
about usefulness that no measurement here or before it has tested.**

## 6. What follows for the plan

**Do not ship 30.** On this evidence it refuses demonstrably legible ground while keeping
demonstrably structureless windows, which is the opposite of its stated purpose in both directions.

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
