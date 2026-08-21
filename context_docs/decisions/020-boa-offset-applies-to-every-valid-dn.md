# 020 — Correct the BOA offset on every valid DN, and decide it from the items

**Status:** Accepted (measured 2026-08-20). Built, on
`solutions/prefer-in-region-duplicate-assets`.

## Context

ESA processing baseline 04.00 (January 2022) added a fixed `+1000` to Sentinel-2 L2A surface
reflectance, so that negative reflectance — routine over water and deep shadow — is representable
in an unsigned type. Reading such a product without removing the offset yields reflectance 1000 DN
too high.

Earth Search indexes two things under one collection. Most items serve Element 84's own COGs from
`sentinel-cogs`, which already have the offset removed. Some serve ESA's originals from
`sentinel-s2-l2a`, which do not. So the correction cannot be a property of the collection there,
and `main` set no threshold for Earth Search at all — it never corrected that collection.

This ADR records the decisions taken while making that correction right, and the measurement that
settled the one nobody could settle by reading documentation.

## Decision

**The offset applies to every valid DN, and the corrected value floors at the lowest VALID code
rather than at zero.** DN 0 is the nodata code and is the one value carrying no offset. Everything
from DN 1 upward loses 1000, and a result at or below zero becomes 1, not 0.

That floor is the part that could only be measured. Flooring at 0 is the arithmetic the community's
harmonisation snippets use, and it is wrong here: 0 is nodata, so it makes a real dark observation
indistinguishable from no observation, and every downstream mask drops it. Element 84 floors at 1,
and matching them is the whole point — a corrected raw copy has to be interchangeable with a
harmonised one.

**Element 84's COGs are harmonised, and the bucket is the signal.** Measured, not assumed: reading
the same granule from both hosts, the COG is exactly the ESA original minus 1000, floored at 1,
with nodata preserved. Two granules, two post-04.00 baselines, four bands, three resolutions,
**274,609,800 pixels, every one identical.**

| granule | baseline | band | pixels | nodata | DN 1–1000 | ours | subtract only ≥1000 | floor at 0 |
|---|---|---|---|---|---|---|---|---|
| `S2A_33TWM_20221128_0_L2A` | 04.00 | B02 @10m | 120,560,400 | 13,862,094 | 12,187 | 100.000000% | 99.989893% | 99.989891% |
| `S2B_29SMC_20230919_0_L2A` | 05.09 | B02 @10m | 120,560,400 | 24,095,834 | 19,834 | 100.000000% | 99.987026% | 99.983548% |
| `S2B_29SMC_20230919_0_L2A` | 05.09 | B12 @20m | 30,140,100 | 6,023,948 | 187 | 100.000000% | 99.999380% | 99.999380% |
| `S2B_29SMC_20230919_0_L2A` | 05.09 | B01 @60m | 3,348,900 | 668,358 | 2,643 | 100.000000% | 99.921079% | 99.921079% |

The last two columns are the point: both rejected variants score below 100% on exactly the dark
population, so the 100% is falsifiable rather than the vacuous result of comparing something with
itself. The magnitude is honest too — the DN 1–1000 band is 0.0006% to 0.079% of these scenes, so
this is a correctness fix over dark surfaces and not a fix affecting most pixels.

**Item-level metadata cannot substitute for the bucket, because it is missing or wrong exactly
where a decision is needed.** `earthsearch:boa_offset_applied` is `True` on the COGs and **absent**
on the ESA-hosted copies — present only where it is not needed. `raster:bands` carries an offset
that contradicts the data (sertit/eoreader#120). And the `-jp2` extra assets on post-04.00 items
point at keys that do not exist: `s3://sentinel-s2-l2a/tiles/33/T/WM/2022/11/28/0/B02.jp2` returns
`NoSuchKey`, the real object being under `R10m/`. That dangling href is also why harmonisation is
judged over the configured reflectance bands and not over every asset — judging all of them reports
a straddle for a wholly harmonised item, on the strength of hrefs that do not resolve.

**The correction is derived from the ITEMS on every provider; the caller's baseline map is
provenance and decides no pixel.** `extract_baselines` is last-wins per date and reports 0 for any
unreadable baseline, so correcting from it skipped raw post-04.00 pixels silently for three
separate reasons: a missing entry, an unreadable baseline, or an arbitrary item's value on a
multi-item date. A figure derived from the items is not evidence about the items.

Where the producer cannot vary between items, the collection's own configuration supplies it: a
correction threshold on such a collection says every item is unharmonised, which is what the
threshold exists to correct. `dates_exempt_from_correction` and `correction_baselines_by_date` take
that as `known_harmonisation` and skip the asset read, which is what lets a provider serving its
bands under native asset keys (`B02`, `SCL`) be judged from its items at all. One derivation now
serves both providers.

**Two copies are the same acquisition when they name the same datatake, not when their timestamps
are close.** The catalogue `datetime` is a per-copy field and reprocessings do not agree on it: the
committed 2017-12-19 pair — one granule, baselines 02.06 and 05.00, same sensing time, orbit and
tile — is timestamped 208.6 seconds apart. A 120-second window therefore kept both as separate
acquisitions and handed both to the loader to mosaic. `s2:datatake_id` names mission, sensing start
and absolute orbit, and only its baseline suffix changes between reprocessings, so identity needs
no tolerance at all. The timestamp window survives as the fallback for a copy naming no datatake.

**A copy owing no offset correction outranks the baseline VALUE; locality does not.** These are two
different claims about where an item's assets live, and they sit on opposite sides of the baseline
deliberately. The correction is decided per solar day over every tile fused into it, so one raw
copy at or above the threshold can refuse a whole day — avoiding that is a coverage argument and is
allowed to cost a reprocessing. Locality is only about egress and must never buy a cheaper read
with a worse pixel, so it stays below the baseline.

**The tile key reads whichever property the catalogue populates.** `grid:code`, then
`s2:mgrs_tile`, then the item id, canonicalised to one form. Planetary Computer ids carry the tile
in a field the Element 84 pattern does not match, so without its own property every PC item was
unkeyable and duplicate selection was a no-op for the entire provider.

**`MIXED` requires both known producer classes to be present.** Harmonised bands beside a bucket
nobody has classified is `UNKNOWN`: nothing there is known to be raw, so the actionable remedy is
to classify the bucket, which is what the `UNKNOWN` message says. Both states still refuse the
date; only the diagnostic changes.

## Consequences

**Two new loud failures on the collection-wide raw path**, where the behaviour used to be silent.
An unreadable baseline now raises `HeterogeneousProducerError` instead of reading as 0 and skipping
the correction. And a solar day whose raw items straddle the threshold now refuses, which is the
rule Earth Search already followed — previously the map's last-wins pick decided it and one side of
the straddle was quietly wrong. Both are loud in place of silent, which is the trade being made.

**The store's dtype is preserved by the default mode**, which is why it is the default. A ROI
store's arrays are seeded from the first date's dataset, and a date that skips correction keeps its
unsigned input dtype, so a mode returning signed values makes the store's dtype depend on which
date landed first and casts every date of the other kind. `preserve_low_values` is renamed
`clamp_negatives`, which is what the flag now selects.

**The ranking term that outranks the baseline is not observed to fire.** Every copy served from the
ESA bucket in the archive as indexed reports a pre-04.00 baseline, and below the threshold the term
is inert. So it engages only in the combination the ingest already logs at WARNING as unexpected —
a raw ESA copy at or above the threshold — where preferring the harmonised copy is the conservative
choice. It is recorded here because a reader will otherwise read it as buying cheaper egress with
older imagery, which is the rule directly below it and the opposite of this one.

**The floor is applied after resampling, and does not commute with it.** `odc.stac.load` reads and
resamples in one step, so the correction acts on resampled values. Six of the ten configured bands
are natively 20 m loaded onto a 10 m grid, so this is the normal case rather than an edge one. Two
raw neighbours at DN 500 and 1500 average to 1000 and then floor to 1, where Element 84 — flooring
each source pixel first — would average 1 and 500 to about 251.

Not fixed here, for three reasons, and recorded so the trade is visible rather than implied. The
ordering is **inherited**: the reference pipeline in `yield-modeling` also applies
`_apply_baseline_corrections_by_date` after `odc.stac.load`, and this store feeds that pipeline, so
correcting on the native grid would diverge from the reference it is meant to match. The affected
pixels are the resampling neighbourhood of the DN-below-1000 population, which is 0.0006%–0.08% of a
real scene. And the parity claim it weakens has no live exposure: the correction never fires on Earth
Search data, because every ESA-hosted copy in the archive is pre-04.00, while on Planetary Computer
it fires on most dates but there is no harmonised counterpart there to be in parity with.

It becomes real the day an ESA-hosted post-04.00 copy appears — the same case term 5 of the ranking
key exists for. Fixing it means loading at native resolution, correcting, then resampling
separately, which changes the graph shape and the memory profile of every ingest; that is separate
work, not a side effect of this one.

**The residual risk on the bucket list is real.** If an unharmonised post-04.00 COG ever appears in
`sentinel-cogs`, it is exempted and stays 1000 DN high, silently. Nothing detects that today.

## Rejected alternatives

**Floor the corrected value at 0.** The arithmetic in the widely-circulated harmonisation snippets,
and what this branch shipped first. Measurement rejected it: Element 84 floors at 1, and 0 is the
nodata code, so flooring there converts valid dark observations into no-data. Wrong on 12,187 and
19,834 pixels in the two 10 m arms above.

**Leave DN 1–999 unchanged.** The behaviour on `main`, on the reasoning that dark pixels are
"below the offset". They are not below anything — ESA applies the offset across the whole range
precisely so that sub-zero reflectance survives the trip. Wrong on the same population.

**Refuse on `earthsearch:boa_offset_applied: false`**, using the metadata only in the direction
where a value is a positive claim and never treating absence as one. Not added: the same field is
known to be wrong in that direction too, so a false `false` would refuse loadable data. This was
considered because two HIGH review comments argued the bucket list is unsafe; the answer is the
measurement above, and this alternative is recorded so the next reader sees it was weighed.

**Widen `_SAME_ACQUISITION_S` past the observed 209-second skew.** A tolerance around a per-copy
timestamp cannot separate "two reprocessings" from "two passes" without getting one of them wrong —
successive orbits revisit a high-latitude tile about 50 minutes apart, and keying on
`(tile, solar day)` alone dropped 493 of 2,733 items that were distinct acquisitions. The field
changed instead of the constant. The old comment claimed reprocessings agree to sub-second
precision; that claim is now corrected in place rather than replaced by a new number.

**Collapse the whole tile-date to one copy.** Would prevent the same fusion by discarding coverage
from passes that identified themselves perfectly well.

**Thread `CollectionConfig.tile_id_property` through to `item_tile`.** The configuration-driven
option, and it needs the config at five call sites across three modules that do not carry it today.
An ordered list of the two known MGRS properties, each validated lexically, closes the same defect
without the plumbing.

**Keep the caller's map as the correction evidence for collection-wide raw providers.** The
narrower fix the review asked for — refuse when a loaded item's baseline is unreadable — leaves two
providers on two kinds of evidence and leaves the last-wins defect in place on one of them.

## Related

- [`ingest/README.md`](../../src/tessera_embeddings/ingest/README.md) — the duplicate ranking order,
  the correction modes, and the three selection owners
- [ADR 004](004-duck-typed-providers.md) — the provider abstraction whose per-collection
  configuration carries `harmonisation_varies_by_item` and `band_names_are_asset_keys`
