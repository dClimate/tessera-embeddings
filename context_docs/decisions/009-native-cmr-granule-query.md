# 007 — Native CMR Granule query for OPERA (not CMR-STAC search)

**Status:** Accepted (v0.1.0)

## Context

OPERA RTC-S1 ingest over large (CONUS-scale) ROIs intermittently
failed with HTTP 500 from `https://cmr.earthdata.nasa.gov/stac/ASF`
during the synchronous query phase — inside `client.search().items()`
in `_query_stac_items` (`ingest/stac.py`). The first page succeeded;
a later page fetch returned:

```text
Oops! Something has gone wrong. We have been alerted and are working to resolve the problem.
```

Because the whole query was one lazy pagination chain, a single failed
page failed the entire batch's query — there was no per-page resume.

Our first instinct made it worse. We raised the STAC search page size
(`limit`) from 250 to 2000 (`STACProvider.max_page_size`), reasoning
that fewer page requests = fewer chances of a per-request 500. **It
made the 500s more consistent, not less** — the model was wrong. On
CMR-STAC the failure rate is not a fixed per-request probability that
paging less reduces.

The root cause, from CMR-STAC maintainer issues:

- **[nasa/cmr-stac #408](https://github.com/nasa/cmr-stac/issues/408)** —
  "Error during pagination" (open, Dec 2025). Reports our exact
  symptom: the initial request succeeds, then following the `next`
  cursor link returns HTTP 500. The fault is in CMR-STAC's
  **cursor-based pagination**, not backend load. Larger / different
  pagination chains expose the bug more, not less — which is why
  bumping `limit` made it worse.
- **[nasa/cmr-stac #411](https://github.com/nasa/cmr-stac/issues/411)** —
  "Item search is very slow" (open, Mar 2026). Documents that CMR-STAC
  pages at ~100 items internally regardless of the `limit` we pass, and
  is ~28× slower than the native CMR API (14s vs ~500ms for 742 items).
  So `limit=2000` never produced 2000-item pages — it just reshaped the
  fragile cursor chain.

Separately, CMR-STAC **silently ignores** the `query` extension for CMR
additional attributes such as `ASCENDING_DESCENDING`, so orbit-direction
filtering could not be pushed to the server. The pre-refactor path
worked around this by querying the native CMR Granule API *for orbit
IDs* and then running a *second* CMR-STAC bbox search whose unwanted
orbit was discarded locally — paying the 500-prone, 28× path on top of a
redundant double query.

## Decision

**Bypass CMR-STAC search entirely for the OPERA `cmr-asf` path and query
the native CMR Granule Search API
(`https://cmr.earthdata.nasa.gov/search/granules.json`), constructing
the pystac `Item`s directly from that response.**

- `opera_query._query_cmr_granules` queries the native granule API and
  pages cleanly at `page_size=2000` against the same host with no 500s —
  the native API does not share the CMR-STAC pagination bug.
- Orbit direction is filtered **server-side** via the `attribute[]`
  parameter, so the response already contains only the desired orbit.
  `_granule_to_item` maps each granule's data download links onto the
  `S1_OPERA_BANDS` asset keys (`0_VV`, `0_VH`).
- **Polarisation is filtered server-side too, on BOTH values.** See
  "Requiring one polarisation is not requiring the pair" below.
- `make_s1_item_provider` returns an `item_provider_fn` (a zero-arg
  callable); `stac._query_stac_items` short-circuits to it and never
  calls `client.search()` for this provider.

This eliminated the 500-prone code path, the redundant double query, and
the 28× CMR-STAC latency penalty in one move.

Validated implementation notes:

- **Asset HREF format.** `_granule_to_item` maps each granule's `/data#`
  download links (href ending `_VV.tif` / `_VH.tif`) onto the
  `S1_OPERA_BANDS` asset keys. These are the same datapool HTTPS hrefs
  the CMR-STAC `0_VV` / `0_VH` assets carried, so `rewrite_assets_to_s3`
  / `resolve_item_assets` work unchanged. Verified against the live
  granule API: constructed items are shape-identical to the CMR-STAC
  items they replace.
- **`proj:` / projection metadata.** CMR-STAC OPERA items already lacked
  the `proj:` extension (we pass an explicit `crs=` / geobox), so the
  granule path is no worse — `odc.stac.load` reads dtype, nodata, and
  CRS from the COGs.
- **Timestamp normalisation** (`normalize_opera_timestamps`) and the
  noon-UTC mosaicking behaviour are applied to the constructed items
  inside the item provider, exactly as before.
- **Pagination resume.** The native API pages at 2000 reliably; a failed
  page retries in place via the urllib3 retry on the granule session.
  Per-batch checkpointing (`get_existing_dates` + the per-batch write)
  bounds the blast radius of any failure to one `batch_days` window.

### Requiring one polarisation is not requiring the pair — corrected 2026-08-23

The polarisation filter shipped as a single `attribute[]=string,POLARIZATION,VV`, and its comment
argued that adding VH "would be redundant" because requiring VH instead "would admit nothing that
VV does not". **That reasoning was wrong, and measurably so.** It considered exactly two
populations — dual VV+VH granules (wanted) and cross-pol HH/HV ones (excluded) — and missed a
third: genuinely single-polarisation VV-only granules. Those include VV, so they passed the
server-side filter, were paged in full, and were then rejected one at a time by the client-side
band check in `_granule_to_item`.

The cost was almost entirely logging. In one 45-minute window of the global campaign the skip
WARNING produced **115,276 lines, 94% of all application log output**. Separately measured:
211,786 skip events over only 49,629 distinct granules — a multiplicity of 4.27, from overlapping
monthly windows, both orbit directions, and overlapping zone boxes.

**The fix rests on CMR's own semantics: repeated `attribute[]` entries on one attribute name are
ANDed, while a multi-valued attribute matches if ANY value matches.** So `VV` alone is a strictly
weaker filter than `VV` and `VH` together. Verified directly against the granule API:

| query | result | what it shows |
|---|---|---|
| Greenland, Feb 2017, `HH` alone | 979 | HH is universal there |
| same box, `VV` alone | 0 | VV is absent there |
| same box, `VV` + `HH` | 0 | ANDed — an OR would return 979 |
| same box, `HH` + `HV` | 372 | equals `HV` alone, the true cross-pol count |

And the two-value filter is **exact**, not merely tighter. Three complete censuses — every entry
walked, no sampling, published band links classified per granule:

| box, window | passes `VV` | passes `VV`+`VH` | publishes both bands | filter == dual set |
|---|---|---|---|---|
| Alaska + BC, Jun 2016 | 2,483 | 0 | 0 | yes |
| N. Europe, Feb 2017 | 27,976 | 27,976 | 27,976 | yes |
| S. America, Sep 2016 | 10,487 | 7,279 | 7,279 | yes |

The S. America box is the discriminating one: genuinely mixed, and the filter admits no
single-pol granule and drops no dual-pol one. Alaska/BC is the clean demonstration of the missed
population — 2,483 granules passed `VV` and not one of them published a VH band.

**This recovers no data.** A single-polarisation granule is unusable either way; ingesting half a
pair would write a fabricated all-nodata band the encoder reads as a confident physical signal
(ADR-013). It stops us paging and logging them, and it restores the client-side WARNING to being
a real alarm.

Two defects in that WARNING were fixed with it. It interpolated **our own** required-polarisation
constant into a "CMR reported %s" slot, so it could only ever print that constant back — and
`granules.json` carries no polarisation field of any kind (verified: the entry keys are
title/links/polygons/times/ids and nothing else), so the line cannot quote the catalogue at all
and now names what the *query* required instead. Its verdict was also wrong for this population:
a granule reporting VV and publishing `['VV']` is self-consistent and single-polarisation, not
"inconsistent with its own metadata". Its comment additionally told the reader that a burst of
these "means something changed upstream rather than that this region is cross-pol" — nothing had
changed upstream; single-pol VV-only granules are a fixed property of the 2016 and early-2017
archive. That claim only holds now that both polarisations are required server-side.

## Rejected alternatives

- **Raise `max_page_size` to 2000 on CMR-STAC.** Tried first; made the
  500s *more* consistent (see Context). Reverted — the `cmr-asf`
  provider no longer sets `max_page_size`, inheriting the default 250,
  which now only affects the `client.search()` providers (Earth Search,
  Planetary Computer) the OPERA path no longer uses.
- **Lower `batch_days` for CONUS ingest (e.g. 30 → 7).** Shrank the
  per-batch query and the work lost when a batch's STAC query 500'd, but
  only mitigated the symptom. With the native granule API it is no
  longer needed to avoid 5xx, though smaller batches still usefully
  bound per-batch Dask graph size.
- **Keep CMR-STAC search + native orbit-ID intersection.** The
  pre-refactor state. Retained the 500-prone path and the 28× latency
  penalty, and paid for two CMR round-trips per batch. Superseded
  wholesale.

## Consequences

- **Pro:** the 500-prone CMR-STAC cursor path is gone for OPERA; CONUS
  ingest queries complete reliably.
- **Pro:** one CMR round-trip per batch instead of two, at ~28× lower
  latency.
- **Con:** items are now hand-constructed from granule JSON rather than
  parsed by pystac-client. The mapping (`_granule_to_item`) is covered
  by `tests/unit/ingest/test_opera_query.py`, but we own its correctness.
- **Con:** the provider re-queries CMR on every invocation (it no longer
  caches at construction). The current S1 flow invokes it once per
  batch, so this is fine; sharing one provider across `has_new_stac_dates`
  and `query_stac_items` would double the query cost. Documented at the
  call sites and tracked in
  [issue #47](https://github.com/dClimate/tessera-embeddings/issues/47).
- **Con (test debt):** `tests/parity/test_ingest_s1_roi_parity.py` replays
  a VCR cassette recorded against the old CMR-STAC-search path, so it no
  longer matches on replay. Re-recording produces a ~116 MB cassette
  (COG bodies + EDL/ASF auth redirects) that trips the credential-safety
  guard and is too large to commit. The test is excluded from the default
  suite (`-m 'not parity'`) and gated behind EDL credentials, so this
  does not block CI. Tracked in
  [issue #45](https://github.com/dClimate/tessera-embeddings/issues/45).

## References

- nasa/cmr-stac #408 — pagination 500: https://github.com/nasa/cmr-stac/issues/408
- nasa/cmr-stac #411 — slow item search / 100-vs-2000 paging: https://github.com/nasa/cmr-stac/issues/411
- CMR-STAC docs: https://cmr.earthdata.nasa.gov/stac/docs/index.html
