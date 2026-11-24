# 045 — Sentinel-2 sources are chosen per calendar year, from one table

**Status:** Proposed (2026-11-24). Changes how [ADR 044](044-sentinel-2-reads-collection-1-where-it-is-complete.md)'s
decision 3 is carried out; its measurements and its choice of collections stand.

## Context

ADR 044 made Earth Search's Collection 1 the default Sentinel-2 collection and sent the periods it
holds only part of to the legacy collection. It did that inside the Collection 1 configuration:
`CollectionConfig.incomplete` listed the periods (2015-01-01 to 2019-03-31, and 2022-01-01 to
2022-12-15) and `fallback` named the collection to read instead.

Three things now need more than that mechanism gives:

- **Earth Search has stopped growing.** ESA has cut it off from new data, so its backfill of 2017
  and 2018 and its updates from late 2026 are frozen. Those years will have to be read from
  another catalogue, Microsoft Planetary Computer. A fallback can name only a collection of the
  same catalogue.
- **A year must be read from one source.** The periods ended mid-year, so a calendar-year run of
  2019 read the legacy collection while a run of June 2019 alone read Collection 1. Re-running
  part of a year could then put two producers into one year of a store.
- **A control arm must be able to name its source.** An explicit request for Collection 1 in 2022
  was silently rerouted to the legacy collection, so a comparison of the two could not be run
  through the ingest.

## Decision

1. **One table decides the source of every period.** `S2_SOURCE_BY_PERIOD` in
   `config/providers.py` lists `(first day, last day, (provider, collection))` rows, first match
   wins; a window no row claims reads `DEFAULT_S2_SOURCE`, Earth Search's Collection 1.
   `CollectionConfig.incomplete` and `fallback` are removed.
2. **Every row is whole calendar years.** The legacy collection's rows become 2015 through 2019,
   and 2022. A window touching a row reads that row's source for the whole window, as before, so
   no window inside one year can read a different source from that year's own run.
3. **The window's own days are tested**, not the padding day the catalogue query adds either side.
   Testing the padding day would send a run starting on 1 January to the previous year's row,
   which matters once adjacent years have different sources. It is safe because:
   - the leading padding day of a year is 31 December of the previous one, and each year's source
     holds it: the legacy collection holds December 2018, and Collection 1 holds December 2019
     and the end of December 2022 (ADR 044's coverage measurements);
   - an image dated the UTC day after its solar day exists only in the far-western high Arctic
     between late February and October (ADR 044, decision 3), so the trailing padding day of a
     year never holds one. ADR 044's exception for that day goes away.
4. **A `provider` and `collection` named together are read as given.** Named neither, the table
   decides. Naming one without the other is refused, because half a source has no meaning. The
   arguments stay separate on `ingest_s2_roi_reflectance`, `preflight_optical_source`, the Prefect
   task and the flow, and the routing lives in `stac.s2_source_for_window`.

## Consequences

- Calendar-year windows, which are all the campaign runs, read the same collection as before.
- A window shorter than a year that starts after 2019-03-31 in 2019, or after 2022-12-15 in 2022,
  now reads the legacy collection instead of Collection 1, matching its year.
- A caller that named only `provider="earth-search"` or only a collection must now name both, or
  neither.
- Adding Planetary Computer for 2017, 2018 and 2026 is a matter of adding rows once that catalogue
  is readable and tested.
- The ingest code identity changes with this code, so an existing store resumes under it only with
  `allow_ingest_code_mismatch`, as with ADR 044.
