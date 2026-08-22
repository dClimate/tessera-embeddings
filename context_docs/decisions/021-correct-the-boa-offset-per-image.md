# 021 — Correcting the Sentinel-2 brightness offset per image, before the mosaic

**Status:** Accepted 2026-08-21. Built on `solutions/correct-boa-offset-per-image`.

**Supersedes decision 5 of [ADR 020](020-boa-offset-applies-to-every-valid-dn.md)** and closes its
limits 1 and 3. Limit 2 stays open, and section 5 explains why it is a different piece of work
rather than the rest of this one.

This document is written to be read by someone who has not seen the code. Terms are defined at
first use; ADR 020's glossary still applies.

---

## 1. The problem this fixes

ESA adds a fixed **+1000** to every stored brightness value in imagery processed at version 04.00
or later. Element 84 removes it from its own copies; ESA's originals keep it. ADR 020 established
how to tell them apart and what arithmetic to apply, and verified the arithmetic against Element
84's own files over 274 million pixels.

What it did **not** change is *when* the correction happens. Imagery is read, resized onto a common
grid, and merged into one image per day — and only then was the offset removed. So one decision had
to cover a whole day's worth of ground.

That works while a day's imagery agrees. When it does not, there is no correct answer:

```
  One day. Three tiles, three different situations.

    tile A   ESA original, version 05.00   offset present     needs -1000
    tile B   ESA original, version 00.01   never had it       needs nothing
    tile C   Element 84 copy               already removed    needs nothing

    correct the day  ->  B and C become 1000 too LOW
    skip the day     ->  A stays          1000 too HIGH
```

So the day was **refused** — not silently mis-corrected, which is the right call, but refused
entirely, taking every sound image on it. And the condition is not rare, because it only takes
*one* affected tile anywhere in the region. A region covers hundreds of tiles; roughly fifteen of
them carry such originals; each is imaged every two or three days. Re-running one region for 2024,
the pipeline skipped **347 days of the year**.

## 2. The change

Remove the offset from **each image as it is read**, before the images are merged.

```
  before:   read+resize each image ─► merge ─► correct the merged day, once
  now:      read+resize each image ─► correct each image ─► merge
                                      └── each image decided on its own ──┘
```

Note where "resize" sits: a source is read and reprojected onto the output grid in one step, and the
correction is applied to what that returns. So the correction moves ahead of the **merge**, which is
what dissolves the per-day conflict, but not ahead of the **resize**. That distinction is the whole
of limit 2 in section 6, and it is why this change closes two limits and not three.

Different tiles occupy different ground, so there is no pixel that is both corrected and
uncorrected. The three-tile day above simply loads: A is corrected, B and C are not.

The question also gets asked of one **file** rather than one item, which resolves a second case for
free. An item whose bands are served from two different places used to have no correct answer
either; now each band is judged on its own.

## 3. Why this is safe to merge without re-processing anything

**No pixel that exists today changes.** This is the part worth checking rather than trusting, and
it follows from one fact: the amount subtracted is a **fixed constant**. The processing version
decides *whether* the offset is removed, never *how much*. So:

| the day contains | before | now |
|---|---|---|
| only already-corrected imagery | nothing subtracted | nothing subtracted |
| only imagery needing correction | 1000 subtracted from all of it | 1000 subtracted from each image |
| a mixture | **refused — no imagery at all** | each image corrected on its own terms |

The first two rows are identical arithmetic. The third had no output to preserve. The change is
therefore **purely additive**: it produces imagery for days that produced none, and leaves every
other day bit-for-bit as it was.

This is what makes it mergeable without re-ingesting completed stores, and it is the reason this
change and limit 2 below must not be bundled together.

*(In-flight stores still cannot be appended to, because the ingest code fingerprint changes. That
is the ordinary cost of any ingest change and is unrelated to pixel values.)*

## 4. What this does to the choice of which copy to read

When the same photograph exists in more than one copy, they are ranked by asking questions in a
fixed order. ADR 020's decision 5 put **"does it avoid needing the offset removed?"** *above*
**"is its processing version newer?"** — meaning we would deliberately accept an older
reprocessing to get a copy needing no correction.

The reason given was coverage: one copy needing correction could refuse a whole day, so avoiding
that was worth an older image. **That reason no longer exists** — no copy can refuse another tile's
imagery any more.

Two reasons to prefer such a copy do survive, and both are about image quality rather than coverage:

- An Element 84 copy had its floor applied by the producer *before* any resizing. One we correct
  ourselves is floored *after* (see limit 2). On very dark ground the two differ, and theirs is the
  better of the two.
- A copy needing no correction cannot be wrong by 1000. One we correct is right only if our
  classification of where files live, and the version the file declares, are both honest.

Those are quality arguments, and this codebase already has a rule for quality arguments: **they may
not buy a better image with an older reprocessing.** That is exactly why "is it cheaper to read?"
sits below "is the version newer?". Harmonisation is now held to the same rule.

```
   1  Can we read every band we need?
   2  Can we tell which source produced it?
   3  Does it say which photograph it is?
   4  Can we read its processing version?
  ─────────────────────────────────────────────  above: "will this work at all?"
   5  Is its processing version newer?
   6  Does it avoid needing the offset removed?   ← moved DOWN from position 5
   7  Is it cheaper for us to read?
   8  Publication sequence, then name (tie-break)
```

**And a copy needing correction is no longer withheld as a spare.** ADR 020's limit 3: when the
chosen copy could not be read, a replacement needing correction was refused, because swapping it in
could make the whole day refuse. It cannot any more, so it is offered — which returns the days that
exclusion was costing.

## 5. What still refuses, and it is not a day

Three situations remain where no answer can be justified, and all are properties of a **single
file** rather than of a day:

- its file lives somewhere nobody has classified as either corrected or uncorrected, and it is at
  version 04.00 or later;
- it is uncorrected and declares no readable processing version;
- the collection's own configuration names no single producer for it (`MIXED` or `UNKNOWN`).

**Measured on the live catalogue, 2026-08-22: none of them fires.** Seven real zone-months, chosen
to span the campaign's years and four continents — 33N for June 2017, January 2018, February 2022
and June 2024; 15S and 45N for June 2024; 10N for June 2025 — queried through the production path
and put through the production duplicate selection, then every reflectance asset of every surviving
item classified by the same call the loader makes:

| | |
|---|---|
| items returned | 36,183 |
| items after duplicate selection | 34,307 |
| reflectance assets classified | 343,070 |
| assets whose bucket was unclassified | **0** |
| items declaring no readable processing version | **0** |
| assets refused | **0** |
| items refused | **0** |

Every href resolved to `sentinel-cogs` or `sentinel-s2-l2a`, and the mixed case is genuinely
exercised rather than absent: 741 assets across four of the seven months are served from ESA's
bucket inside an otherwise-Element-84 item, and each is decided from its own location — exempt
below version 04.00, owed at or above it.

The third situation is **unreachable from any caller today**, and that is structural rather than
lucky. `source_decision` is asked only by the loader's parser, which passes
`collection_harmonisation(config)`, and that returns `RAW` or nothing — never `MIXED`, never
`UNKNOWN`. Those two values are produced by `item_harmonisation`, which feeds duplicate *ranking*
and never the correction. Closing the enum is defensive: it stops a future caller falling through
to "correction owed" on an answer nobody resolved.

All three still refuse their day rather than guessing, which is unchanged behaviour — the machinery
that records and announces a refused day is untouched.

**Correcting and exempting are wrong by the same amount in opposite directions, and both are
silent**, which is why neither is chosen as a default.

## 6. Limit 2 stays open, and this is deliberate

The floor is still applied *after* resizing. Six of the ten bands we use are natively
half-resolution and get enlarged, so this is the normal case rather than an edge case:

```
  two neighbouring stored values:     500        1500

  resize then correct (today):  average 1000 ──► 1000-1000 = 0 ──► floored to 1
  correct then resize (E84):     1 and 500   ──► average    ≈ 251
```

Measured on a synthetic worst case: correcting after resizing yields `{1, 250, 500}` where
correcting before yields `{1, 126, 375, 500}`.

**It is not bundled here, and section 3 is the reason.** Reading and resizing happen in a single
step, so fixing this means taking over that step. More importantly it **would change imagery every
existing store already holds**, on those six bands — so it has to be justified as an improvement
and paid for with a re-processing decision, where this change costs neither.

ADR 020 section 5 grouped limits 1, 2 and 3 as "one piece of work". **That was wrong**, and the
constant offset in section 3 is why: 1 and 3 dissolve without touching a single existing pixel,
while 2 rewrites them.

One related detail worth recording: where imagery is read at a coarser resolution than its native
one, the reader may use a pre-summarised version of the file, whose values were averaged while the
offset was still present. That pushes the same error one step further back. It does not arise on
the 10 m path we use.

## 7. What we chose not to do

**Carry a table of which files need correcting.** The natural first design, and it does not scale:
the object holding it is copied into every parallel task, so at region scale a 3,081-entry table
cost **14.6 MB** of task graph across 64 tasks. The decision instead travels on the file
description that was being sent anyway — **158 bytes**, and flat in the number of files. It also
removes a whole class of bug, because there is no lookup key to get wrong.

**Refuse when the reflectance files cannot be identified.** Tempting as a guard against a
correction that silently reaches nothing. Declined: if those names cannot be resolved, the *load*
fails a moment later on the same resolution, with an error the pipeline already recognises and
recovers from by trying another copy. Refusing first would replace a recoverable failure with an
unrecoverable one — the precise shape of the two-changes-that-only-break-together bug ADR 020
records. It warns instead, and the real protection is that an unclassifiable file refuses rather
than being exempted.

**Drop the file instead of refusing its day**, for the two cases in section 5. That would lose one
tile instead of one day, which is better — but it is a coverage-versus-correctness choice, it needs
its own record of what was lost, and it would buy back coverage only for situations never seen in
the archive. Not worth it for this change.

**Delete the harmonisation preference entirely.** Section 4: two of its three justifications
survive. Demoted, not removed.

---

## 8. Related

- [ADR 020](020-boa-offset-applies-to-every-valid-dn.md) — the arithmetic, how the two sources are
  told apart, and the limits this closes
- [`ingest/README.md`](../../src/tessera_embeddings/ingest/README.md) — how the code implements it
- [ADR 004](004-duck-typed-providers.md) — the per-provider configuration this relies on
