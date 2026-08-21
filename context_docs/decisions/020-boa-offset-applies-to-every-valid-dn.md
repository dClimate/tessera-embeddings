# 020 — Removing the Sentinel-2 brightness offset, and choosing which copy to read

**Status:** Accepted 2026-08-20. Built on `solutions/prefer-in-region-duplicate-assets` (PR #107).
Three limits are live and unfixed, and they are one piece of follow-up work — see section 5.

This document is written to be read by someone who has not seen the code. Terms are defined at
first use, and there is a glossary at the end.

---

## 1. The problem

Sentinel-2 measures how much light each patch of ground reflects. That measurement is stored as a
whole number — 0.15 reflectance becomes 1500, and so on.

Reflectance can legitimately come out **negative** after the atmosphere is subtracted, which
happens routinely over deep water and in shadow. Whole numbers in these files are unsigned, so
negatives cannot be stored. In January 2022 ESA solved this by adding a fixed **+1000 to every
value**, giving negative reflectance somewhere to live:

```
                  true reflectance      what ESA stores      what we must store
                  ────────────────      ───────────────      ──────────────────
  bright snow            0.90       ──►      10000       ──►       9000
  vegetation             0.15       ──►       2500       ──►       1500
  dark water             0.01       ──►       1100       ──►        100
  darker water          -0.05       ──►        500       ──►          1   ← floored
  no observation          --        ──►          0       ──►          0   ← untouched
```

If we store ESA's numbers without subtracting the 1000, every pixel is too bright by 1000 — a
large error, and a silent one. Nothing crashes; the imagery just becomes wrong.

---

## 2. Why this isn't simple: one catalogue, two sources

We find imagery through Earth Search, a search service run by Element 84. A single search of a
single collection returns files from **two different places**:

```
                     Earth Search catalogue
                  (one search, one collection)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
      s3://sentinel-cogs                s3://sentinel-s2-l2a
      Element 84's own copies           ESA's original files
      offset ALREADY REMOVED            offset STILL PRESENT
      the large majority                a small minority
              │                                │
              └───────────────┬────────────────┘
                              ▼
                  These need OPPOSITE treatment,
                  and the catalogue's own metadata
                  does not reliably say which is which.
```

Get it backwards in either direction and the error is the same size and equally silent: subtract
1000 from a file that already had it removed, or fail to subtract it from one that still has it.

Before this change, the code did not distinguish them at all — it left the whole collection alone,
which is right for the majority and wrong for the minority.

---

## 3. What we decided

### Decision 1 — Subtract from every real measurement, and floor the result at 1

The offset is on **every** value from 1 upward, not only on bright ones. The single exception is 0,
which is the code for "no observation here"; it carries no offset and is left alone.

Values that were negative before the offset was added come out at or below zero when we subtract
it. Those are floored — and the floor is **1, not 0**:

```
  a real but very dark observation, stored as 500

    floor at 0  ──►  0    now indistinguishable from "no observation".
                          Every cloud and quality mask will throw it away.

    floor at 1  ──►  1    still the darkest valid value, but visibly
                          an observation.        ✓ this is what we do
```

This detail was not obvious and we got it wrong first. Flooring at zero is what the widely
circulated code snippets for this correction do. We only found the difference by comparing our
output against Element 84's, pixel by pixel.

### Decision 2 — Tell the two sources apart by where the file is stored, not by the metadata

Membership of a list of storage locations decides which treatment a file gets. The catalogue's own
metadata is not usable for this, for three separate reasons:

1. The field that should say whether the offset was removed is **missing on exactly the files where
   the question matters** — present on Element 84's copies, absent on ESA's.
2. A second metadata field states an offset that contradicts what is actually in the pixels.
3. Some file locations listed in the metadata **do not exist**. Following one returns "no such
   file"; the real file is in a different folder.

Point 3 also explains a design choice that otherwise looks arbitrary. We judge a file's source by
looking only at the colour bands we actually read, not at everything the catalogue lists alongside
them. Those extra listings include the broken paths above — so judging everything would conclude
"this file comes from both sources at once" for a file that is entirely from one, on the strength of
paths that lead nowhere.

The owner of this data has also confirmed from operational experience that every Element 84 copy is
already corrected, and that Element 84's own documentation is misleading on the point.

### Decision 3 — Work out the correction from the imagery, not from a table passed in

There is a table of processing versions that travels alongside the imagery. It exists to record
*what was used*, and it is unsuitable for *deciding what to do*. Deciding from it meant a needed
correction could be skipped in three different ways, all silent:

- the day has no entry in the table at all, and a missing entry reads as "nothing to do";
- the version could not be read, which the table records as 0 — again "nothing to do";
- the day has several images, and the table keeps only one entry, so it records whichever image
  happened to be processed last.

For providers where every file comes from the same source, the collection's own configuration
supplies the answer instead. This is what lets Microsoft Planetary Computer be handled at all: it
names its files differently, so inspecting individual files there finds nothing.

### Decision 4 — Two copies are the same photograph when they name the same datatake

The same photograph is often published more than once, reprocessed with a newer version. We must
keep one, not blend both together. The obvious way to spot them — a shared timestamp — does not
work:

```
  One photograph of tile 33TWM, 19 December 2017. Published twice.

    copy A   processed 2017   catalogue time 09:54:10   version 02.06
    copy B   processed 2023   catalogue time 09:57:39   version 05.00
                              └──── 209 seconds apart ────┘

  Grouping by timestamp (within 2 minutes) concludes: TWO photographs.
    ──► both kept, both loaded, blended into one image.

  But both carry the same "datatake" name — the satellite pass they came from:

      GS2B_20171219T095409_004109_N02.06
      GS2B_20171219T095409_004109_N05.00
      └──────── identical ───────────┘└ differs ┘

  Grouping by that concludes: ONE photograph, published twice.
    ──► one kept, the other held in reserve as a fallback.
```

The timestamp is a per-copy field, so no tolerance around it can separate "two reprocessings" from
"two genuinely different passes" without getting one of them wrong. The datatake name needs no
tolerance at all. Timestamps remain the fallback for files that do not name one.

### Decision 5 — Prefer a copy needing no correction, but never a copy that is merely cheaper to read

When several copies of one photograph exist, they are ranked by asking questions in a fixed order.
The first question that separates two copies decides between them:

```
   1  Can we read every band we need?                 ← unusable beats everything
   2  Can we tell which source produced it?
   3  Does it say which photograph it is?
   4  Can we read its processing version?
  ─────────────────────────────────────────────  above: "will this work at all?"
   5  Does it avoid needing the offset removed?       ← may cost a newer version
   6  Is its processing version newer?
  ─────────────────────────────────────────────  below: "which is the better image?"
   7  Is it cheaper for us to read?                   ← must never override 6
   8  Publication sequence, then name (tie-break)
```

Questions 5 and 7 both come down to where a file is stored, and they sit on opposite sides of
question 6 deliberately:

- **Question 5 is about how much imagery survives.** The correction decision is made once per day
  across every tile in that day, so one file needing correction can cost the whole day (see limit 1
  below). Avoiding that is worth accepting an older processing version.
- **Question 7 is only about our own costs.** Files in the same cloud region as our computers are
  cheaper and faster to read. That must never buy us a worse image, so it ranks below version.

### Decision 6 — Reduce duplicates for every collection that makes an offset decision

Two entry points into the loader previously skipped duplicate reduction unless a collection's
source could vary file by file. That excluded the one provider that most needed it. Measured on the
live Planetary Computer catalogue across six tiles and 3,585 files: **1,000 redundant copies of the
same photograph** were being loaded and blended together, and **12 days** would have been refused.

This deliberately stops at collections that have no offset to remove. Landsat files can be grouped
by the same machinery, so removing the condition entirely would start reducing Landsat imagery too —
a change nothing here has measured.

### Decision 7 — "Both sources at once" requires actually seeing both

A file whose bands come from a known-corrected location and an unclassified one is reported as
*unknown*, not as *mixed*. Nothing in it is known to be uncorrected, so the useful advice is
"classify that location", which is what the unknown message says. The previous message told the
operator to split the day apart, for a conflict that might not exist.

---

## 4. The evidence

One Sentinel-2 product is published in both places at once: Element 84's corrected copy, and ESA's
original, same processing version, same pixel grid. Element 84's copy is therefore a **reference
answer** — whatever a correct correction produces, it should reproduce that file exactly.

We ran the real correction code over whole bands read from both places and compared every pixel:

| product | version | band | pixels | of which "no observation" | of which very dark | our result | if we floored at 0 | if we skipped dark values |
|---|---|---|---|---|---|---|---|---|
| `S2A_33TWM_20221128_0_L2A` | 04.00 | B02 @10 m | 120,560,400 | 13,862,094 | 12,187 | **100.000000%** | 99.989891% | 99.989893% |
| `S2B_29SMC_20230919_0_L2A` | 05.09 | B02 @10 m | 120,560,400 | 24,095,834 | 19,834 | **100.000000%** | 99.983548% | 99.987026% |
| `S2B_29SMC_20230919_0_L2A` | 05.09 | B12 @20 m | 30,140,100 | 6,023,948 | 187 | **100.000000%** | 99.999380% | 99.999380% |
| `S2B_29SMC_20230919_0_L2A` | 05.09 | B01 @60 m | 3,348,900 | 668,358 | 2,643 | **100.000000%** | 99.921079% | 99.921079% |

**274,609,800 pixels. Every one identical.** Two products, two processing versions, four bands,
three resolutions.

The last two columns matter as much as the result. They are what the two rejected alternatives would
have scored, and they are below 100% — so the agreement is a real test rather than a comparison of
something with itself. Both alternatives fail on exactly the very dark pixels, which is the
population the decision was about.

**How much does this affect?** The very dark pixels are between 0.0006% and 0.08% of these scenes.
This is a correctness fix for water and shadow, not a change to most of the imagery.

---

## 4b. What the end-to-end run found

The pixel comparison above tests the arithmetic. It cannot test what happens when the whole pipeline
runs over a year of real imagery, so three zone-years were ingested on the development account:

| arm | zone | year | why this one | outcome |
|---|---|---|---|---|
| A | 01N | 2017 | densest ESA-original coverage found anywhere | ran clean |
| B | 57S | 2017 | most duplicate copies of one photograph | ran clean |
| C | 01N | 2024 | same ground, later year — do the dynamics hold? | **failed** |

**Arms A and B behaved as intended.** Duplicate reduction ran and reported itself: zone 01N pruned
15 tile-days that had more than one copy, with 11 winners in our own cloud region and 4 elsewhere;
zone 57S pruned 64 tile-days, 68 copies rejected and all 68 retained as fallbacks. No day was
refused in either. One Sentinel-1 read failed with a permission error and its retry succeeded — an
infrastructure hiccup, unrelated.

**Arm C failed, and that is the useful result.** The optical leg died three times, once per retry,
every time on the same day, 2 January 2024, with: *"fuses a raw item owed the offset correction with
an already harmonised one."* Because a refusal is deterministic, each retry reached the same day and
failed identically, so the whole zone-year was lost to one day's metadata.

**The later year is much worse than the early one**, which is the opposite of what we assumed:

| zone | days refused, late 2017 | days refused, early 2024 |
|---|---|---|
| 01N | 2 of 54 (3.7%) | **26 of 60 (43.3%)** |
| 33N (Europe) | 0 of 38 | 5 of 60 (8.3%) |
| 59N | 0 of 50 | 2 of 60 (3.3%) |
| 02N | 0 of 46 | 2 of 60 (3.3%) |

In 2017 the refusals come from ESA originals spanning two processing eras. In 2024 they come from
ESA originals at a current version sitting beside already-corrected copies — a normal, ongoing state
of the archive, and present in Europe as well as at the antimeridian.

**What changed as a result.** A refused day is now skipped on its own, loudly, and counted, instead
of failing the leg. The correction decision is untouched — a day that cannot be decided is still not
corrected — but the loss is one day instead of a year. This is mitigation, not a fix; see limit 1.

---

## 5. What this costs — four honest limits, and three of them are one problem

### Limit 1 — A whole day of imagery can be refused, and this happens for real

This is the most serious consequence and it is currently unfixed.

The correction is applied **once per day, with one value**. That is fine when everything in a day
needs the same treatment. It breaks when a day contains ESA originals from two different processing
eras:

```
  One day, zone 01N, 16 November 2017 — 45 images across many tiles

    27 images   Element 84 copies      already corrected        ✓ fine as they are
     9 images   ESA originals, v00.01  no offset was ever added ✓ fine as they are
     9 images   ESA originals, v05.00  offset IS present        ✗ needs -1000

  Only one correction value can be applied to the whole day:

    correct the day  ──►  the nine v00.01 images become 1000 too LOW
    skip the day     ──►  the nine v05.00 images stay  1000 too HIGH

  Neither is right. So the day is refused — and the 27 perfectly good
  Element 84 images are thrown away with it.
```

Choosing one copy per photograph does not help here: the conflict is between **different tiles**,
not between duplicate copies of the same tile. We confirmed this — reducing that day from 45 images
to 33 still leaves it refused. The same happens on 21 December 2017.

Refusing is still the right answer over silently shifting some tiles the wrong way by 1000. But this
costs real days, and an earlier version of this document said the situation did not occur. That was
an inference from a small sample and it was wrong.

**How common is it?** ESA originals at version 04.00 or above are concentrated near the
antimeridian — the Bering Sea and Aleutians, map zones 1 and 60 — and the population is growing
rather than shrinking. Counted over November and December of each year, so these are lower bounds
for full years:

| year | ESA originals needing correction | tiles affected |
|---|---:|---:|
| 2017 | 15 | 10 |
| 2018 | 10 | 10 |
| 2019 | **210** | 15 |

A single case was also found outside that region, in map zone 59 in 2019. So this is a recurring
and increasing population, not a one-off in a single year.

**How often, and how bad.** Measured on the live catalogue: 26 of 60 days in early 2024 refuse in
zone 01N, and 5 of 60 even in Europe. A refused day used to fail the whole optical leg — the run
that found this lost a whole zone-year to one day — so a refused day is now skipped alone and
counted. The loss is bounded at one day per affected day, and it is announced.

**The fix** is to apply the correction to each image before they are combined, rather than to the
combined result. That removes the conflict instead of isolating it, because different tiles occupy
different ground: there is no pixel that is both corrected and uncorrected. It is a change to how
imagery is loaded, and it is owed as separate work.

### Limit 2 — The floor is applied after images are resized

Images are read and resized to a common grid in a single step, and the correction runs afterwards.
Six of the ten bands we use are natively half-resolution and get enlarged, so this is the normal
case rather than an edge case.

Flooring does not survive resizing intact:

```
  two neighbouring raw values:        500        1500

  resize first, then correct:   average 1000 ──► 1000-1000 = 0 ──► floored to 1
  correct first, then resize:   1 and 500     ──► average    ≈ 251

                                        1   vs   251
```

Not changed here, for two reasons. It is the order `main` already uses, so changing it changes the
pixels every existing store was written with, and would need justifying as an improvement rather
than a bug fix. And the affected pixels are the immediate neighbourhood of the very dark population
above — well under 0.1% of a scene. But this does have live exposure, because ESA originals needing
correction do occur (limit 1), so it is a real if small error and not a theoretical one.

### Limit 3 — A fallback copy may be withheld

When the copy we chose cannot be read, we fall back to another copy of the same photograph. A
fallback that itself needs the offset removed is now **withheld**, because swapping it in beside the
already-corrected tiles of the same day would refuse that day (limit 1) — and the recovery machinery
knows how to handle a file that will not read, not a day that refuses. One unreadable file would
otherwise stop the whole run.

The cost is real: on a day where everything needs correcting, that fallback would have worked, and
withholding it loses the day instead. We accept that because a lost day is bounded and recorded,
whereas a stopped run is neither. Checking the fallback against the whole day would be better than
either, and is part of the same owed work as limits 1 and 2.

### Limit 4 — If Element 84 ever publishes an uncorrected file, we will not notice

Our rule is "files in these locations are already corrected". If an uncorrected file ever appears
there, we will leave it alone and it will stay 1000 too bright, silently. Nothing detects this.

We considered refusing whenever the metadata explicitly says the offset was *not* applied — using
that field only in the direction where a value is a positive statement, never treating its absence
as one. We did not add it, because the same field is known to be wrong in that direction too, and a
false negative would refuse imagery that is perfectly good.

---

### Limits 1, 2 and 3 are one piece of work

Worth stating plainly, because they arrived as three separate review findings and read as three
separate caveats. They are not. All three come from the same design choice — **the correction is
applied to the combined day-image, after the individual images have been merged and resized** — and
all three dissolve if it is applied to each image beforehand instead:

```
  today:     read + resize + merge  ────►  correct once per day
                                              │
             limit 1: one value for the whole day, so a day mixing eras refuses
             limit 2: the floor lands on resized values, not original ones
             limit 3: a fallback cannot be judged without the whole day

  the fix:   read ─► correct each image ─► resize + merge
```

That is a change to how imagery is loaded, affecting the shape and memory profile of every ingest,
so it is not a side effect of this change. It is the single most valuable follow-up here.

## 6. What we chose not to do

**Floor at zero.** What the widely circulated snippets do, and what this branch shipped first.
Measurement rejected it: zero means "no observation", so flooring there turns real dark
observations into gaps. Wrong on 12,187 and 19,834 pixels in the two full-resolution comparisons
above.

**Leave dark values alone.** What `main` does, on the reasoning that they are "below the offset".
They are not below anything — the offset was added to the whole range precisely so that
below-zero reflectance survives.

**Trust the metadata field about the offset.** Discussed under limit 3.

**Widen the tolerance on timestamps** past the 209 seconds observed. A tolerance around a per-copy
timestamp cannot separate two reprocessings from two genuinely different passes without getting one
of them wrong — satellite passes over one tile are about fifty minutes apart, and collapsing them
would discard real imagery. We changed which field is used instead.

**Reduce every collection's duplicates.** Discussed under decision 6.

**Pass the collection's configuration down through every call site** that needs the tile name.
Five call sites across three modules do not carry it. A short list of the two known properties,
each checked for the right shape, closes the same gap without the plumbing.

---

## 7. Glossary

| term | meaning |
|---|---|
| **reflectance** | the fraction of light a patch of ground reflects; stored as a whole number, 1500 meaning 0.15 |
| **the offset** | the fixed +1000 ESA added to every stored value from January 2022, so negative reflectance could be stored |
| **processing version** | ESA's version number for how a product was generated, e.g. `05.00`. Same photograph, reprocessed, gets a newer one |
| **the threshold** | version 04.00, the point at which the offset started being added |
| **no-observation code** | the stored value 0, meaning nothing was measured. Carries no offset, and must not be confused with a real dark measurement |
| **datatake** | one continuous strip of imaging by the satellite; names the photograph independently of how it was processed |
| **tile** | a fixed square of ground, about 110 km across, that Sentinel-2 imagery is cut into |
| **solar day** | the local day an image belongs to; imagery is combined one solar day at a time |
| **Earth Search** | Element 84's search service, through which we find Sentinel-2 imagery |
| **Planetary Computer** | Microsoft's equivalent service; serves the same imagery, uncorrected, under different file names |

---

## 8. Related

- [`ingest/README.md`](../../src/tessera_embeddings/ingest/README.md) — how the code implements all
  of the above
- [ADR 004](004-duck-typed-providers.md) — the per-provider configuration this relies on
