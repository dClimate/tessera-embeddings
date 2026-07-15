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
  by `tests/unit/test_opera_query.py`, but we own its correctness.
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
