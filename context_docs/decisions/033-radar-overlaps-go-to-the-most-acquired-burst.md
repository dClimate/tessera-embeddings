# 033 — Radar overlaps go to the most-acquired burst

**Status:** Proposed 2026-10-22. Replaces the radar half of
[`ingest/solar-day-fusion-order.md`](../ingest/solar-day-fusion-order.md), which recorded the radar
order as undisciplined; that document holds the loader mechanism this rests on.

## Context

A radar mosaic holds one slice per solar day, and the loader fills each pixel from the first item of
the day that has a valid value there. Later items only fill holes, so the order of a day's items
decides which one supplies a pixel that two of them cover.

OPERA publishes Sentinel-1 radar as bursts: strips about 20 km by 85 km, each named by its relative
orbit (track), ESA burst number and sub-swath, with a footprint fixed by ESA's burst grid. Ascending
and descending passes go to separate stores, so the overlaps inside one store are:

```text
  one pass, adjacent bursts    one pass, adjacent sub-swaths    one day, two tracks (high latitude)

  ┌───────────┐                ┌──────────┬──────────┐          ┌────────────┐
  │ T029  b1  │                │ T029   ▒▒│▒▒  T029  │          │ T029       │
  ├───────────┤                │ IW1    ▒▒│▒▒  IW2   │          │      ┌─────┼──────┐
  │ T029  b2  │                │        ▒▒│▒▒        │          │      │▒▒▒▒▒│      │
  └───────────┘                └──────────┴──────────┘          └──────┼─────┘ T131 │
                                                                       └────────────┘
  same geometry                nearly the same geometry         different look angles: the case
                                                                this decision is about
```

Only the third case changes what a pixel means. Backscatter depends on the angle the radar looks
from, and two tracks see a high-latitude pixel from different angles on the same day.

Until this decision nothing ordered radar items. The loader ordered each day by `(time, id)`, every
item of a day carries the same normalised noon timestamp, so the granule title decided: the lowest
track number won. That is reproducible, but unrelated to which track supplies most of the pixel's
year, so a pixel's time series could switch angle on every day both tracks flew.

## Decision

1. **Where two bursts of one solar day cover a pixel, the burst acquired most often over the ingest
   window supplies it.** `opera_query.burst_fusion_sort_key` orders a day's items by acquisitions,
   most first. The loader keeps that order, because `preserve_original_order` is now set for every
   collection that names a `fusion_order` in its configuration (`CollectionConfig.fusion_order`),
   rather than for collections with a cloud band.
2. **This is the per-pixel rule, computed per item.** A burst's footprint is fixed, so the bursts that
   can cover a given pixel are fixed, and each covers it once per acquisition. The burst with the most
   acquisitions in the year is therefore the look geometry most of that pixel's year comes from.
   Putting it first gives it every contested day, so contested days agree with the pixel's majority.
   No raster is read to decide it.
3. **Ties have a fixed order.** Equal counts go to the burst id, whose fixed-width fields order it by
   track, then burst number, then sub-swath. After that come the raw acquisition time and the granule
   id, so the order is total and a rerun builds the same pixels. Ties are the common case: most bursts
   are acquired on the same 12-day cycle. A tie still picks the same burst on every contested day,
   which is the consistency that matters.
4. **The count is a census of the whole window, taken once per leg.** `opera_query.burst_acquisitions`
   counts distinct acquisitions per burst id from granule titles, through CMR's reference format.
   `s1_roi` builds one `BurstCensus` per orbit leg over the requested window, before any batch runs.
   - It counts acquisitions rather than granules, because a reprocessed acquisition is published
     again under a new production time, and counting granules would rank a burst by how often it
     was reprocessed.
   - It uses the requested window, not the window a resumed leg searches, so a resumed leg ranks
     bursts exactly as the run it resumes did.
   - A burst's count does not depend on the query box, as long as the burst touches it. So a run
     over a few tiles of a zone ranks the bursts it shares with the whole zone the same way.
5. **A title that does not parse sorts after every named burst of its day**, so it can only fill
   gaps, and each query logs how many there were.

## What it costs

Measured against CMR from a laptop, zone 35N (24–30°E, 0–72°N), 2024:

| query | granules | bursts | pages | bytes | time |
|---|---|---|---|---|---|
| census, ascending, whole year | 100,179 | 2,780 | 51 | 25.5 MB | 49 s |
| census, descending, whole year | 115,142 | 3,284 | 58 | 29.3 MB | 50 s |
| the batch query's format, ascending, January only | 10,853 | 2,774 | 6 | 157.9 MB | 41 s |
| census format, ascending, January only | 10,853 | 2,774 | 6 | 2.8 MB | 6 s |

A year of batch queries is about twelve months like that one. The census therefore adds roughly a
tenth to a leg's catalogue time and about 1.4% to its catalogue bytes. The first batch's provider
starts the census before its own query and waits for it afterwards, so the two overlap. The time
the census adds to a leg is therefore what it runs beyond the first batch's query, about 10 s at
35N.

The same query over the high-latitude part of that zone (55–72°N, ascending) found 1,181 of 1,192
bursts acquired 29 to 31 times, which is why the tie order matters. It also found 8,039 granules
that were reprocessed copies of an acquisition already counted.

## Rejected alternatives

- **A count per batch.** It ranks a burst by a month of acquisitions, so it moves with `batch_days`
  and with the batch boundaries. A burst could win one month and lose the next, which brings back
  the switching this decision removes.
- **A per-pixel count from the rasters.** It gives the same answer as decision 2 but needs every
  acquisition read twice.
- **Ranking by track instead of burst.** A track's coverage of a pixel is its bursts' coverage, and
  bursts of one track can differ in count where a segment of acquisitions starts or stops. The burst
  is the unit whose footprint is fixed.
- **Keeping odc's `(time, id)` order.** It is reproducible, but it hands every contested day to the
  lowest track number, whatever the pixel's year is mostly made of.

## Not done

- **No per-pixel quality layers.** OPERA also publishes layover and shadow masks, and the inputs for
  a number-of-looks or incidence-angle preference. Choosing a burst per pixel by those is a different
  decision, unmeasured here, and would need those layers read for every pixel.
- **Reprocessed copies of one acquisition are not judged.** Two granules of the same burst and
  acquisition tie on everything up to the id, and the id, which ends in the production time, keeps
  the earlier production first, as before. Preferring the newest production is a separate decision.
- **The size of the effect is unmeasured.** How much ground two tracks contest on one day, and how
  far the change moves embeddings, has not been measured.

## Consequences

- `s1_roi.py` changed, so the ingest code identity (`config/ingest.py:ingest_code_identity`) moves
  for every mosaic, optical included. A part-built mosaic needs `allow_ingest_code_mismatch` to
  resume, so this deploys with no ingest or update running.
- It applies to mosaics ingested after the deploy. Mosaics are deleted after inference, so no
  existing mosaic needs migrating, and nothing already published is rebuilt by this change.
- Each orbit leg makes one extra CMR query, measured above.
