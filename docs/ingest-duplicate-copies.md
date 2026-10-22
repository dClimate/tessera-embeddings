# Duplicate copies: which copy of a Sentinel-2 image is read

A tile-date often has several catalogue items, and they are not all the same kind of thing. Some are
**copies of one image**: ESA reprocesses a granule at a newer baseline, a product is published twice,
or one granule is served by two producers. Others are **different images**: a high-latitude tile is
imaged by successive orbits on one day, and a pass crossing a datastrip boundary reaches the tile once
per datastrip, each covering part of it. `duplicates.py` keeps every image and one copy of each,
before the loader sees them, because `odc.stac.load` fuses a solar-day group: two copies of one image
would both be read, and one unreadable copy would fail the whole date with nothing to fall back to.

This document is for anyone changing `duplicates.py`, or asking why a tile-date was read from the
copy it was, or why a date survived a read failure. The offset correction that several of the
ranking terms below protect is its own subject, in
[ingest-baseline-correction.md](ingest-baseline-correction.md).

## Which image a copy is of

It is read from two identifiers (`image_identity`): the pass, from
`s2:datatake_id`, and the datastrip's sensing start, from `s2:datastrip_id`. Copies of one image agree
on both, and different images differ in one. Reprocessing changes neither. The catalogue `datetime` is
not used: it is per copy, reprocessings of one image disagree on it by minutes, and two datastrips of
one pass are seconds apart.

```text
   a SPLIT pass: two images                     a REPROCESSING: one image twice
   ┌──────────────────────────┐                 ┌──────────────────────────┐
   │ datastrip S…T153625      │ granule _0_     │ datastrip S…T101549      │ _0_ at 02.13
   │ most of the tile         │                 │                          │ _2_ at 05.00
   ├──────────────────────────┤                 └──────────────────────────┘
   │ datastrip S…T154121      │ granule _1_     same pass, same datastrip:
   └──────────────────────────┘                 they compete and the newest baseline wins
   same pass and baseline, different
   datastrips: both are kept
```

## Copies are merged only on proof

A copy whose pass or datastrip cannot be read is kept as an
image of its own, with no spares. The two possible mistakes are not alike. Keeping a copy that was in
fact a duplicate costs a second read and a missing fallback rung, and its pixels are still right
because the offset is corrected per image
([ADR 021](../context_docs/decisions/021-correct-the-boa-offset-per-image.md)). Merging two
different images loses ground for the day, silently. So under doubt the cheap mistake is made, and the run logs how many copies it kept
that way. Where a reprocessing omits one of a pass's datastrips, the older copy of that datastrip is
kept beside the newer ones for the same reason
([ADR 029](../context_docs/decisions/029-an-acquisition-is-one-datastrip-of-one-pass.md)). The
census behind these choices is in
[`context_docs/ingest/split-datastrips.md`](../context_docs/ingest/split-datastrips.md).

The copies it rejects are not discarded. They become the **fallback ladder**: if the chosen
copy's object will not read, the
[recovery](ingest-error-handling.md#when-a-source-object-will-not-read) steps down to the next
rung rather than losing the date. That is why several terms below rank for "can this copy be
processed at all" ahead of "is this copy the best", and it is described in full under
[The fallback ladder](#the-fallback-ladder).

## One preference key

Preference is **one sort key** (`_preference_key`), and what makes it work is that it is
**context-free**: no term means "best in my group", so the same tuple orders two copies of one
image and two copies of different ones. A term relative to the group's own best baseline would make
cross-image comparison meaningless and force a second key alongside this one. Add a signal here and
nowhere else.

The seven terms, and what each is there to stop. The first three ask whether a copy can be
processed at all and the last four which of the processable ones is best — a copy that cannot be
processed must never outrank one that can, however good its pixels would have been:

| # | term | what it protects |
|---|---|---|
| 1 | read-set completeness | a copy missing a band cannot deliver the date, and the ladder does not recognise that as a read failure |
| 2 | producer decidable | an undecidable producer refuses the date, and the ladder cannot step down on a refusal |
| 3 | baseline readable | an unreadable baseline refuses the date too |
| 4 | processing baseline, descending | newer reprocessing first, by value, so every rung of the ladder stays in baseline order |
| 5 | owes no offset correction | an uncorrected copy cannot be wrong by the offset; a corrected one is only right if the bucket lists are |
| 6 | locality, among equal baselines | cheaper egress, but never at the price of an older pixel |
| 7 | `s2:sequence`, then item id | a total order, so a rerun cannot silently produce a different mosaic |

In full, in that order:

1. **Read-set completeness**, judged over the assets *this* load will request — the configured
   bands plus the caller's `extra_bands`, not a fixed list and not the broader pruning set, which
   keeps `scl` regardless. First, because a copy missing one cannot deliver the tile-date at any
   baseline, and a missing band is not a failure the fallback ladder recognises.
2. **Whether the producer is decidable**, where it would change a pixel. An undecidable producer
   refuses its date at or above the correction threshold, and a refusal is not something the
   ladder can step down on. A copy spanning a harmonised and a raw producer is not undecidable:
   each source is decided on its own bucket and corrected band by band. Inert below the threshold.
3. **Whether the baseline is readable**, for producers whose correction depends on it, unknown
   sorting last. An unreadable baseline refuses the whole date, for the reason given under
   [the baseline correction](ingest-baseline-correction.md#who-served-the-pixels-decides), so an older reprocessing that can be corrected beats a newer one
   that cannot be processed — and it outranks every term below because the ladder recovers from a
   read error but not from a refusal. Already-harmonised copies are exempt.
4. **Processing baseline, descending.** Ordered by value rather than "is it best", so every rung
   of the fallback ladder stays in descending baseline order — a read failure can skip a 04.00
   copy and hand out a 03.00 one.
5. **Whether the copy owes an offset correction at all**, where the producer is an item's own
   property. Below the baseline, for two pixel-level reasons: an already-harmonised copy had its
   floor applied before resampling where one we correct is floored after, so the two disagree on
   very dark pixels; and a copy owing nothing cannot be wrong by the offset, while a corrected one
   is right only if the bucket lists and declared baseline are both honest. A quality-versus-
   quality preference must not buy a better pixel with an older reprocessing. Inert below the
   threshold — though not universally: zone 01N in 2017 carries 15 at baseline 05.00, so it does
   fire on real data. Also inert where the producer is the collection's answer.
6. **Locality, among equal baselines only.** A copy whose read assets sit in a preferred bucket is
   cheaper to read, so it wins a tie. Restricting it to ties stops it buying cheaper egress with
   an older pixel. This is why there are two bucket lists: harmonisation is a **pixel** claim,
   locality a **cost** claim. They name the same buckets today but their key sets differ by `scl`,
   so a copy can be harmonised without being local.
7. **`s2:sequence`, descending, then item id.** The id keeps the choice independent of catalogue
   response order, so a rerun cannot silently produce a different mosaic — and it makes the key a
   total order, so no comparison ever falls back to input order.

## Two properties held by tests

Two properties of that ordering are easy to get wrong and are held by tests:

- **Locality is judged over the read set, not over every asset.** A real Element 84 item carries
  its COG bands *and* the original JP2s, across two buckets, so requiring every asset disables
  the preference altogether. It is judged over the *whole* read set: one local band among many
  remote ones is not locality, and an item exposing none of them is remote, because absence of
  evidence is not evidence of locality.
- **An unreadable baseline sorts LAST, and makes locality inert for that copy.** A missing
  baseline is an absence of evidence rather than a tie: as a tie it let a copy with no baseline
  displace a raw copy at 05.00, taking an older reprocessing *and* skipping the correction. Such
  a copy also refuses its whole date downstream, and the ladder recovers from a read error but
  not a refusal. An already-harmonised copy is exempt, since no offset decision rests on its
  baseline and penalising it would hand the tile-date to an older raw reprocessing.

## Where the tile key and the producer come from

**The tile key is read from whichever property the catalogue populates**, `grid:code` or
`s2:mgrs_tile`, then the item id — all canonicalised to one form, so two catalogues naming one
tile produce one grouping key. Reading only Earth Search's property would leave every item from a
catalogue naming the tile elsewhere unkeyable, which makes duplicate selection a silent no-op for
that whole provider rather than an error.

**Where the producer cannot be read from an item's assets, the collection supplies it**, through
`known_harmonisation` on `select_preferred_duplicates` — the same value
`stac.collection_harmonisation` gives the correction path, so the two cannot disagree. This is
load-bearing rather than an optimisation: a spare judged only on visible assets looks harmless,
is offered to the ladder, and aborts the ingest when a read failure steps down to it, because the
recovery loop steps down on a read failure and not on a refusal.

## The fallback ladder

The **fallback ladder** — the rejected copies, in the order a read failure steps down them — is
built by one global sort over the whole tile-date, using a key with no notion of "best in my
group". Ranking each image separately and concatenating the results is wrong further down the
ladder: with one image holding 05.00 and 01.00 spares and another holding 04.00 it
yields `[05, 01, 04]`, and since the unattributed recovery consumes the head on each retry, the
second retry takes 01.00 and never reaches 04.00. In the global order an unreadable baseline
simply sorts last, which is the same protection by a more direct route: a copy whose baseline
cannot be read is the one whose correction will silently be skipped.

## Who runs the selection

Duplicate selection has **two owners**, and neither is the shared query. `query_stac_items`
deliberately does not prune: `s2_roi` runs its own selection over that output and keeps the
rejected copies as the ladder `step_down_copies` walks when a source object will not read, so
pruning upstream would leave it nothing to step down to. `load_stac_items` prunes for everyone
else — both the documented `query_stac_items` -> `load_stac_items` workflow and `ingest_tile`,
which leaves it to the loader: the loader realigns `baselines` in place and that is the same dict
`ingest_tile` returns, so the map still describes the copy that was kept. Selecting over an
already-selected set is a no-op, so more than one owner is safe.

## Parsing buckets and baselines

Buckets are compared by parsing the href's host and path rather than by substring, so a lookalike
host cannot be mistaken for a preferred one. The baseline is matched as a version string rather
than parsed as a number, so `"NaN"`, `"Infinity"` and every other value that is numeric without
being a version read as unknown — see `item_baselines.py`.
