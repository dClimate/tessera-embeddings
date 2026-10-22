# Which scene wins a pixel: the solar-day fusion order

**Status: the optical ordering was corrected in `fix/clearest-scene-wins-the-solar-day`. The
blast-radius measurement is in progress and is the input to the restart decision. The radar path
now has a fusion order of its own, the most-acquired burst first
([ADR 033](../decisions/033-radar-overlaps-go-to-the-most-acquired-burst.md)).**

This document exists because a belief about a third-party library sat unchecked in this codebase
for its whole life, silently inverted a quality decision on every published optical pixel, and was
recorded as fact in eighteen places. The mechanism is simple. The way it survived is the useful
part.

## The terms

**A solar day group.** Sentinel-2 delivers imagery as 100 km tiles. One region of interest usually
spans several, and adjacent tiles overlap. The loader (`odc.stac.load`, called with
`groupby="solar_day"`) collects every item acquired on the same local calendar day into one time
slice — a *mosaic* — so a region reads as one observation per day rather than several fragments.

**Fusion.** Building that one slice out of several overlapping sources means deciding, for each
pixel that more than one source covers, which source supplies it. The function that decides is the
*fuser*.

**Why it is a quality decision.** Sentinel-2 items carry `eo:cloud_cover`, a scene-level percentage.
Where two scenes of one solar day overlap, one is usually far clearer than the other. Choosing
between them is choosing whether the published pixel is cloud or ground.

## The mechanism

`odc` ships one fuser and this package configures no other:

```python
def nodata_fuser(dst, src, nodata) -> None:
    """Default fuser - only fill where src is nodata"""
    missing = nodata_mask(dst, nodata)
    np.copyto(dst, src, where=missing)
```

The mask is computed on `dst`, the destination. `np.copyto` writes `src` into `dst` **only where
`dst` is still empty**. So the first source to supply a pixel keeps it, and every later source can
only fill the holes the earlier ones left.

**First valid source wins. Later sources cannot overwrite.**

Which source is "first" is the order the items are handed to the loader, because this package sets
`preserve_original_order=True` for every collection that names a fusion order. Without that flag
`odc` sorts each group by `(time, id)`; with it, the caller's order decides.

## What was believed instead

That `odc` mosaics by a painter's algorithm — sources rendered in sequence, each overwriting the
ones before it, so the **last** source wins. Under that belief the correct sort is cloudiest-first,
leaving the clearest scene last to paint over the others. That is what both ingest paths did.

Since the fuser is first-wins, cloudiest-first handed every same-day overlap to the **cloudiest**
scene available.

## Why it survived

Three things had to line up, and all three did.

**The library's own docstring corroborates the wrong reading.** `nodata_fuser`'s docstring says
"only fill where **src** is nodata". The code masks on `dst`. Anyone checking the docstring rather
than the two lines beneath it had the false belief confirmed.

**Every test asserted the order, not the outcome.** There were tests for the sort direction, for
grouping preserving order, for `preserve_original_order` being set. All of them checked what was
handed *to* the loader. None checked what came *out*. The half that was assumed was the half that
was wrong.

**The output looks correct.** A cloudiest-wins mosaic has the right shape, the right dimensions, the
right dates, and plausible pixel values. Nothing downstream can tell that a pixel is cloud where it
could have been ground. There is no error, no warning, and no shape mismatch — only worse imagery.

## The correction

| Site | Was | Now |
|---|---|---|
| `stac.query_stac_items` sort | `(solar date, −cloud, id)` | `(solar date, +cloud, id)` |
| `s2_roi` re-sort | `(solar date, −cloud)` | `(solar date, +cloud, id)` |
| `stac.extract_baselines` | last item of a date wins | first item of a date wins |

The baseline map moved with the sort for a reason that is easy to miss. It records one processing
baseline per date and is persisted as the store's `baselines_applied`. That is **provenance only**:
the radiometric correction is decided per source as each source is read, and nothing derives a
correction from this map. What the move fixes is therefore the honesty of the record — under the
corrected order the first item is the clearest, the scene that actually supplied most of the day's
pixels, so the stored provenance names the scene that contributed. Keeping last-wins would have
named the cloudiest scene, the one contributing least.

Both paths now sort on one function, `solar_day_sort_key`, which owns the key and its reasoning.
They each built their own before, and the defect that replaced was those two disagreeing about the
direction.

## What now pins it

`tests/unit/ingest/test_solar_day_fusion.py` refuses to mock the loader: two real GeoTIFFs, two real STAC
items on one solar day, through the production path, asserting the pixels that come out.

It asserts first-wins **in both directions**, which pins the mechanism rather than today's outcome —
a future reversal of the sort cannot pass it. It also asserts a hole in the clearest scene falls
through to the next-clearest, so the day keeps its coverage instead of trading coverage for clarity.

The radar test in the same file does the same for bursts: two granules from adjacent tracks over
the same ground, through `ingest_tile`, with the census favouring first one and then the other. The
winner swaps with the census, whatever the supply order or the granule ids.

## Blast radius

Being wrong on overlaps only matters where scenes actually contest ground, and a day carrying two
acquisitions need not have a single contested pixel — two passes over a 6° strip at different
latitudes never touch. **The defect is confirmed; its extent is unmeasured.** The measurement that
answers it computes the overlap area between different acquisitions in an equal-area projection,
with the cloud gap across each contesting pair. No count of multi-acquisition days is a substitute,
and none should be quoted as the impact.

## Radar: the most-acquired burst first

Radar has no cloud cover, but it has an overlap whose winner matters. At high latitude two adjacent
relative orbits image the same ground on one solar day, from different look angles, and backscatter
depends on that angle. [ADR 033](../decisions/033-radar-overlaps-go-to-the-most-acquired-burst.md)
gives such a pixel to the burst acquired most often over the ingest window, so the pixel's time
series follows one geometry wherever the archive allows. Like the optical order it is an item sort,
`opera_query.burst_fusion_sort_key`, ranked by a census of the window's bursts taken once per leg.

## The loader flag follows the fusion order

`CollectionConfig.fusion_order` names the order a collection's items are handed to the loader in:
`"clearest_scene"` for Sentinel-2 on Earth Search, `"most_acquired_burst"` for OPERA radar. One table
in `stac.py` maps each name to its sort, and the same field sets `preserve_original_order`. So a
collection is sorted exactly when the loader keeps the order, and a collection that names none
(Landsat, Planetary Computer's Sentinel-2) keeps odc's deterministic `(time, id)`.

**What this replaced.** The flag was first set for every collection, which gave radar the flag's
effect with no sort behind it: the winner of an overlap was whatever order the CMR granule query
returned. It was then keyed on `has_scl`, the condition gating the cloud sort, which restored odc's
`(time, id)` for radar and made the result reproducible. The reasoning recorded with that change
held that radar had no quality ordering, so neither of two overlapping bursts was the worse
observation and a radar sort would buy nothing. It missed the two-track case above. With every
item of a day on one noon timestamp, `(time, id)` resolves by granule title, which hands every
contested day to the lowest track number, whatever track supplies most of the pixel's year.

The radar footprint join is unaffected by any of this: `normalize_to_solar_day` gives every item
in a group the same canonical noon timestamp and odc reads `item.datetime`, so that join key never
depended on the order.

## Trade-offs accepted

**An item with no `eo:cloud_cover` sorts after every measured value.** `solar_day_sort_key` gives
it infinity rather than 100, because 100 is itself a real reading the two would otherwise tie on,
and the `id` term would then decide — letting an unmeasured scene take ground from one known to be
fully clouded. Under first-wins the sentinel means an unknown scene can only fill gaps: unknown
never displaces measured. That is a change in kind from the previous order, where an unknown-cloud
item sorted first and won everything it covered.

**Clearest-first plus first-wins is the cheapest pairing that keeps coverage, not the only one.**
A custom valid-aware last-wins fuser — one that overwrites the destination only where the later
source is valid — reaches the same result from the opposite order, and keeps gap-filling too. So
coverage is not an argument against that design and should not be quoted as one. The arguments
are cost and blast radius: it means writing and configuring a fuser where the library's default
already suffices, and the fuser applies to every collection the loader serves rather than only
this one.
