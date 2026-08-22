# Which scene wins a pixel: the solar-day fusion order

**Status: the ordering was corrected in `fix/clearest-scene-wins-the-solar-day`. The blast-radius
measurement is in progress and is the input to the restart decision. The radar path's ordering is
identified as undisciplined and is not yet fixed.**

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
`preserve_original_order=True`. Without that flag `odc` sorts each group by `(time, id)`; with it,
the caller's order decides.

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

`s2_roi`'s sort gained the `id` key that `query_stac_items` already had. Two scenes of one solar day
can tie on cloud cover — `0.0` is a common value — and a stable sort then preserves whatever order
the supplier produced. With a first-wins fuser that lets the supplier decide published pixels. The
third key makes the sequence a function of the items alone. On the production path this changes
nothing, because the items arrive already tie-broken on `id` and the re-sort is stable; it closes
the case where a different supplier is injected.

## What now pins it

`tests/unit/test_solar_day_fusion.py` refuses to mock the loader. It writes two real single-band
GeoTIFFs on one grid, builds two real STAC items on one solar day, and loads them through the
production path with the production load kwargs. The assertions are on the pixels that come out.

Three properties are pinned, and the second is the one that matters most:

1. Handed clearest-first, the clear scene's values are what survive.
2. **Handed either way round, the first source wins** — asserted in both directions, so the
   mechanism itself is pinned rather than today's outcome. A future reversal of the sort cannot
   pass this test.
3. A hole in the clearest scene falls through to the next-clearest rather than to nothing, so the
   day keeps its coverage instead of trading coverage for clarity.

The test also asserts `ds.sizes["time"] == 1`, because two items landing in two separate slices
would make the whole test vacuous.

## Blast radius

Being wrong on overlaps only matters where scenes actually contest ground. First figures, zone 33N,
June 2024: all 30 days carry more than one distinct acquisition, and the median cloud-cover gap
between acquisitions on a day is 70 percentage points. Zone 15S has none — 207 items in the month,
a sparse tropical strip.

**That overstates contested ground and must not be quoted as the impact.** Two passes over a 6°
strip at different latitudes never touch, so a day with two acquisitions need not have a single
contested pixel. The measurement that answers the question computes the actual overlap *area*
between different acquisitions in an equal-area projection, and the cloud gap across each
contesting pair. Until that lands, the honest statement is that the defect is confirmed and its
extent is unmeasured.

## The radar path is not covered by this fix

`preserve_original_order=True` is set unconditionally in `_load_from_stac`, and the cloud sort runs
only for collections with an SCL band. So for radar the library's deterministic `(time, id)` default
is switched off and nothing replaces it: the winner of an overlap is whatever order the CMR granule
query returned.

This is not the same defect. Radar has no quality ordering analogous to cloud cover — two
overlapping bursts are both valid observations at comparable geometry, so there is no worse one to
prefer by mistake and nothing is being silently degraded. The exposure is **reproducibility**: which
burst wins depends on an external service's return order that nothing here pins.

The asymmetry is what makes it worth fixing. Setting the flag globally to serve the one collection
that has its own sort removes the library's determinism guarantee from every collection that does
not. Setting it only where a deliberate order exists would restore that for free. Verified safe for
the footprint join: `normalize_to_solar_day` assigns every item in a group the same canonical noon
timestamp and `odc` reads `item.datetime`, so the join key is order-independent either way. It would
change which burst wins an overlap, which is a pixel change — so it belongs in a window where the
store is being rebuilt, or not for a long time.

## Trade-offs accepted

**An item with no `eo:cloud_cover` reads as 100 and sorts last.** Under first-wins that means it can
only fill gaps: unknown never displaces measured. This is the cautious end, and it is a change in
kind from the previous order, where an unknown-cloud item sorted first and won everything it
covered.

**Clearest-first plus first-wins is the cheapest pairing that keeps coverage, not the only one.**
A custom valid-aware last-wins fuser — one that overwrites the destination only where the later
source is valid — reaches the same result from the opposite order, and keeps gap-filling too. So
coverage is not an argument against that design and should not be quoted as one. The arguments
are cost and blast radius: it means writing and configuring a fuser where the library's default
already suffices, and the fuser applies to every collection the loader serves rather than only
this one.

**One integer per date remains a lossy record** of a day whose tiles declare different baselines.
That is unchanged by this fix and is deliberate; nothing in the package reads the attribute yet.
