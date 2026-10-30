# Radar ingest: OPERA RTC-S1 through NASA's CMR

Radar comes from OPERA RTC-S1, ASF's terrain-corrected Sentinel-1 product, and five things about
reading it have no counterpart on the optical path. It is found through NASA's CMR rather than a
STAC search. One date arrives as ten separate burst granules. Overlapping bursts are ranked by how
often each was acquired, not by cloud cover. The items carry no projection metadata for the loader
to read. And amplitude is converted to decibels before it is stored.

This document is for anyone changing `opera_query.py`, `s1_roi.py` or `transforms.py`, or asking
why a radar pixel came from the burst it did, or why a zone has no radar at all. Two neighbouring
subjects have their own documents: the credentials for ASF's bucket are in
[authentication.md](authentication.md), and how ten burst timestamps become one
day's slice is in [solar-days.md](solar-days.md#fusing-operas-per-burst-timestamps).

## Native granule query (orbit filtering + item construction)

`make_s1_item_provider` builds an `item_provider_fn` that returns ready-to-load OPERA items
**without calling CMR-STAC `client.search()` at all**. CMR-STAC's cursor pagination
intermittently 500s on CONUS-scale queries (nasa/cmr-stac#408) and pages internally at ~100
items regardless of the requested `limit` (#411); it also **silently ignores** the `query`
extension for CMR additional attributes such as `ASCENDING_DESCENDING`. The native CMR
Granule Search API has none of these problems.

The provider queries the granule API directly:

```text
GET https://cmr.earthdata.nasa.gov/search/granules.json
    ?short_name=OPERA_L2_RTC-S1_V1
    &attribute[]=string,ASCENDING_DESCENDING,ASCENDING
    &bounding_box=...
    &temporal=...
    &page_size=2000
```

Orbit direction is filtered **server-side** via `attribute[]`, so the response holds only the
desired orbit — no separate STAC search and no local granule-ID intersection. Each granule's data
links (`rel` ending `/data#`, href ending `_VV.tif` / `_VH.tif`) map onto the `S1_OPERA_BANDS`
asset keys (`0_VV`, `0_VH`) to build `pystac.Item`s shape-compatible with the rest of the
pipeline, with `title`, `time_start` and `polygons` supplying id, datetime and geometry. The title
also names the burst (`opera:burst_id`, `sat:relative_orbit`, `opera:subswath` on the item), and
`time_start` is kept in `properties["datetime"]` for solar-day normalisation to read.
Pagination uses the `CMR-Search-After` header, which pages cleanly at 2000. See
[ADR 009](../../context_docs/decisions/009-native-cmr-granule-query.md).

## Polarisation filtered server-side

A cost fix rather than a correctness one. Ingest needs dual-pol VV+VH, and the query carries
`attribute[]=string,POLARIZATION,VV` and `attribute[]=string,POLARIZATION,VH` alongside the orbit
filter. CMR matches a multi-valued attribute if ANY value matches and ANDs repeated attributes, so
naming both is exact: it admits no single-pol granule and drops no dual-pol one. The client-side
check remains as a safety net, and its warning means a catalogue inconsistency — metadata
advertising the pair without the bands published — rather than a regional fact.

## When a zone has no usable radar

**That is a finding rather than a failure.** Over ice Sentinel-1 images in Extra Wide swath with
HH/HV polarisation, and the OPERA query discards anything that is not dual-pol VV+VH, so an ROI
whose land is ice has a catalogue full of granules and not one the ingest can use. Zone 23N
(Greenland) returns ~183,000 granules for 2021 and **zero** usable items. (They are not EW-mode
either: the Greenland ones report `BEAM_MODE=IW`, and a `BEAM_MODE=EW` query there returns
nothing.) Requiring a SAR store there failed the cell permanently, so `"both"` resolves to
`S1_ORBIT_NONE`, which activates no orbit and leaves the coverage gate checking reflectance alone.

**Permissive by default, refusable on demand.** A global product cannot reject terrain that is
radar-free as a matter of geography, so `resolve_s1_orbit`'s `allow_none` defaults to True. A
single run over terrain known to be imaged is the opposite case — there an absent store means
something upstream broke, and resolving to `none` would embed without radar and hide it — so
those callers pass the flows' `require_s1`, which reaches the resolver as `allow_none=False`. An
operator who names one orbit is never downgraded either. Which case it is comes from the ingest's
per-orbit item count, since it has just queried both orbits: `items_seen=0` means the source
offers nothing here, which is terrain rather than a gap. A consumer reading a finished mosaic
cannot distinguish the two, so its warning names the mosaic and points at that count. Accepting a
radar-free ROI necessarily means embedding S2-only pixels, and `InferenceConfig` derives
`allow_s2_only` for that case rather than asking the caller, because the alternative is a fill
that writes nothing and reports success.

## Which burst wins an overlap

Bursts of one solar day overlap, and the loader keeps the first valid one, so their order decides
which supplies a pixel — radar's half of
[the fusion order](../../src/tessera_embeddings/ingest/README.md#the-fusion-order-decides-which-source-wins-a-pixel).
Inside one store (each orbit direction has its own) the overlaps are adjacent bursts and
sub-swaths of one pass, which share a look geometry, and at high latitude two adjacent tracks on
the same day, which do not:

```text
   ┌──────────────────┐
   │ burst of T029    │                 on a day both tracks fly, the strip ▒ that both
   │ (29 this year)   │                 cover goes to the burst acquired MORE OFTEN over
   │          ┌───────┼──────────┐      the window — here T131, though its track number is
   │          │▒▒▒▒▒▒▒│          │      higher — so the pixel's series keeps one look angle
   └──────────┼───────┘          │      wherever the archive allows
              │ burst of T131    │
              │ (31 this year)   │
              └──────────────────┘
```

`opera_query.burst_fusion_sort_key` orders each day's items by the burst's acquisitions over the
whole ingest window, most first, with ties broken by burst id so a rerun builds the same pixels; a
title naming no burst sorts last. The counts come from one census per orbit leg
(`opera_query.BurstCensus`), read from granule titles in CMR's light reference format over the
same solar days the batches own. It is recorded on the orbit's store as the root attribute
`burst_census`, with the window and longitude it covers, in every date's commit. A resumed leg
ranks by that record instead of counting again, and a leg asking for a different window is
refused; a different window goes into its own store.
[ADR 033](../../context_docs/decisions/033-radar-overlaps-go-to-the-most-acquired-burst.md) has
the reasoning, the measured cost, and what was left out, such as layover and shadow masks.

## UTM CRS derivation

CMR-STAC OPERA items lack the `proj:` extension, so `odc.stac.load` cannot infer the output
CRS. `mgrs_tile_to_utm_epsg` derives the correct UTM EPSG from the tile's zone number and
latitude band (C–M = southern hemisphere, N–X = northern hemisphere), e.g. `33UUP` → EPSG:32633.

## Amplitude to decibels

OPERA products store linear amplitude (float32). `transforms.amplitude_to_db` converts to a
compact scaled uint16 suitable for storage and model inference:

```text
dB = 20 × log10(amplitude) + 50
scaled = dB × 200
result = clip(scaled, 0, 32767).astype(uint16)
```

Constants (`S1_DB_SHIFT = 50`, `S1_DB_SCALE = 200`) are ported from
`tessera_preprocessing/s1_fast_processor.py`. Zero/negative amplitudes are masked to `1e-10`
before `log10` to avoid domain errors; they are written back as 0 (nodata) after conversion.
This is a fully lazy Dask operation — no data is materialised until the Zarr write.
