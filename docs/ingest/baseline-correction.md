# Sentinel-2 baseline correction: removing the reflectance offset during the read

ESA changed the Sentinel-2 L2A processing baseline at version 04.00 (January 2022), adding +1000
to every surface reflectance value. Some copies of an image have that offset removed and some do
not, Earth Search serves both kinds under one collection, and both ways of getting it wrong are
silent: a skipped correction leaves plausible pixels 1000 too high, a doubled one shifts every
value by 1000.

This document is for anyone changing `boa_offset.py`, `asset_locations.py`, `item_baselines.py` or
the loader in `stac.py`, or checking why a Sentinel-2 pixel has the value it does. It covers how
the decision is made per source and applied inside the read, the arithmetic, and the limits that
are recorded rather than fixed. Which of several copies of one image gets read is a neighbouring
subject, in [duplicate-copies.md](duplicate-copies.md). The reasoning and
measurements are in [ADR 020](../../context_docs/decisions/020-boa-offset-applies-to-every-valid-dn.md)
and [ADR 021](../../context_docs/decisions/021-correct-the-boa-offset-per-image.md).

## Who served the pixels decides

Whether the offset has to be subtracted is a property of **who served the pixels**, not of the
collection: Element 84 harmonises its own COGs and subtracts it for you, while ESA's originals
carry it. Since Earth Search indexes both kinds, reading the collection alone exempts or corrects
them together, and one of those is always wrong.

So `baseline_threshold` **is** set for Earth Search, and the decision is made per ASSET from where
that asset lives (`boa_offset.source_decision`). Three properties of that decision:

- **Judged over the reflectance bands only.** A real Element 84 item carries the original JP2s
  as extra assets beside its COG bands, so judging every asset reports a straddle for an item
  that is wholly harmonised where it matters. `scl` is excluded even though it *is* read: it is
  categorical and never corrected, so its producer cannot make the reflectance ambiguous. This
  is a different asset set from the one locality is judged over — see the
  [duplicate selector](duplicate-copies.md#two-properties-held-by-tests), which uses the
  full read set including `scl`.
- **Decided per SOURCE, and applied as each image is read.** `odc.stac.load` fuses a solar day
  into one time slice, so a correction applied to its OUTPUT hits every tile at once, and a day
  whose tiles disagree has no correct answer — 347 days of one region-year were refused that way.
  The decision is stamped per reflectance asset at parse time (`stac.BoaOffsetParser`) and applied
  inside the read (`stac._SourceReader`) before anything is resampled together, so no pixel
  is both corrected and uncorrected. It is purely additive, the amount being a constant: days the
  pipeline loaded before are bit-identical and only previously-refused days change. See
  [ADR 021](../../context_docs/decisions/021-correct-the-boa-offset-per-image.md) §3.
- **Thresholded on the item's own declared baseline, and an unreadable one refuses.** An absent
  or malformed `s2:processing_baseline` parses as nothing rather than 0: read as zero it falls
  under the threshold and exempts pixels that may carry the offset, while correcting it takes 1000
  off pre-04.00 pixels that never had it. Both are wrong by the same amount in opposite directions
  and both are silent, so the source refuses. `item_baselines` is the only reader of that
  property.

## Where the producer cannot vary, the collection answers

**Consulting the assets at all is scoped to the collections that need it**, via
`CollectionConfig.harmonisation_varies_by_item`. Where the producer cannot vary between items the
**collection's configuration supplies the answer** instead: a correction threshold on such a
collection says every item is unharmonised, which is what the threshold is there to correct, and
`source_decision` takes that as `known_harmonisation` without consulting the bucket. That is what
lets a catalogue serving its bands under native asset keys — `B02`, `SCL`, which an item-level
read keyed on the configured band names finds nothing under — be decided here and corrected
rather than refused on every modern item. One decision serves both shapes of collection, so they
cannot disagree about a producer.

Which assets carry the reflectance bands is resolved through **odc's own alias table**
(`stac._reflectance_asset_keys`), not from the configured band names, for the same reason. `scl`
is excluded structurally — it is simply not among the resolved keys — rather than by a list the
corrector is told to skip.

## The amount is a constant

The correction VALUE is a **constant**, `S2_BASELINE_OFFSET` of `-1000`; the baseline decides only
*whether* the offset is removed, never how much. `extract_baselines` remains separate: it records
what each item declared and is what reaches the store's `baselines_applied`, but it is **not** a
correction input. One integer per date is provenance, and correcting from it left raw post-04.00
pixels uncorrected whenever it omitted a date, carried the zero an unreadable baseline collapses
to, or named an arbitrary item's baseline on a multi-item date.

## What still refuses

**One ambiguous shape survives, and it refuses** (`HeterogeneousProducerError`): a reflectance
source at or above the threshold whose producer cannot be determined — served from a bucket nobody
has classified, or belonging to an unharmonised copy that declares no readable baseline. The
refusal is a property of that one source and is gated on the threshold, since below it no producer
changes a pixel. It is skipped alone and counted rather than failing the leg, and duplicate
selection routes around it: a copy that would refuse is ranked last *and* withheld from the
fallback ladder, since a refusal is not a read failure and nothing retries one. "Alone" is a
property of `s2_roi`, which loads one solar day per call — a caller pairing `query_stac_items`
with `load_stac_items` over a multi-day list forfeits every day in it, so pass a day at a time.

**An item not exposing EVERY reflectance band under the configured names is `UNKNOWN`, not
`RAW`**, and the safe direction inverts here. A non-empty subset is not enough: nothing in
`asset_locations.py` sees the alias table mapping a band name to an asset key, so a band absent under `blue` may
be served under `B02`, and `_prune_item_dict` preserves exactly those partially aliased items.
Letting one visible harmonised band speak for a hidden native-keyed one would silently subtract
1000 from pixels that may already be harmonised.

## Reporting how far the decision reached

It is reported after every load, from counters the parser keeps:
how many reflectance sources it stamped, and how many of those were owed the offset. Reaching
*none* of them is the one way this can still go wrong quietly — an empty or mistaken
`reflectance_assets` set corrects nothing and produces plausible pixels 1000 too high — so a zero
count is a WARNING naming the assets it resolved and any other count is an INFO line. A warning
rather than an error, because a caller that replaces the loader also reports zero; what makes that
enough is that every other way of getting this wrong is loud, since an unclassifiable source
refuses and an unresolvable band name fails the load.

`baselines_applied` is correspondingly **lossy** on a day whose tiles declare different baselines.
The value is the first item's — the clearest tile, since the query sorts cloud-ascending — and the
day's other vintages are recorded nowhere. Deliberate: nothing in this package reads the value.

## The arithmetic

`correct_boa_dn` does the arithmetic, once, on one source's pixels. **The offset applies to every
valid DN, not only to bright ones**: ESA adds it across the whole reflectance range precisely so
that negative surface reflectance, routine over water and deep shadow, is representable in an
unsigned type. So the correction is `DN - 1000` for every DN from 1 upward, floored at the lowest
VALID code, which is **1**. Not zero, because zero is the nodata code and flooring a real dark
observation there makes it indistinguishable from no observation, which every downstream mask then
drops. Element 84's harmonised COGs floor at 1 for the same reason, and reproducing them exactly
is what makes a corrected raw copy comparable with a harmonised one. DN 0 itself never reaches the
arithmetic — the reader applies the result only where the source was valid — so nodata survives as
nodata.

**That the nodata code IS 0 is hard-coded, and that is an unchecked assumption.** `_NODATA = 0`
is a constant in `stac.py`, and the correction rests on it twice: DN 0 is excluded from the
arithmetic, and the floor of 1 is what stops a corrected pixel from looking like nodata. The real
answer belongs to the catalogue, which `odc` derives per band from `raster:bands` and hands the
reader as `RasterLoadParams`, but the driver is installed on any collection whose config sets
`requires_baseline_correction`, so nothing ties the constant to what that collection declares.
Under a marker of 65535 every gap pixel would test as valid — the correction shifts them to 64535,
the fuser's `dst == fill_value` test stops recognising them as empty, and the mosaic gains
data-looking pixels where there was no observation. Both Sentinel-2 providers declare 0 today, so
this is latent; resolving the marker from `cfg` and refusing anything else is owed.

The arithmetic widens to `int32` and casts back to the INPUT dtype, so nothing wraps and the
store's unsigned arrays are unaffected: the offset is negative and the floor is positive, so an
unsigned input stays representable. Adding a negative Python int to a `uint16` array raises under
numpy 2, which the widening exists for.

## A recorded limit: the floor acts on resampled values

`odc.stac.load`
reads and resamples in one step, so the wrapped reader sees already-warped pixels, and six of the
ten configured bands are natively 20 m on a 10 m grid. A pixel whose kernel spans the DN-1000
boundary is floored where Element 84, flooring each source first, would not have been. Fixing it
means taking over the read-and-warp step and would rewrite pixels in every existing store, so it
is owed separately — measured in
[ADR 021](../../context_docs/decisions/021-correct-the-boa-offset-per-image.md) §6, alongside
[ADR 020](../../context_docs/decisions/020-boa-offset-applies-to-every-valid-dn.md).
