# Satellite Ingestion

Modules for querying, authenticating, and loading satellite data from STAC catalogs and CMR
into Icechunk/Zarr stores. Used by the Tessera ingestion flows (`ingest_s1_roi_sar`,
`ingest_s2_roi_reflectance`).

---

## Module Overview

| Module | Purpose |
|---|---|
| `stac.py` | STAC-based data loading via `odc.stac.load`. Handles multiple providers (Earth Search, Planetary Computer), date filtering, and the load-time machinery that applies the BOA offset: `BoaOffsetParser` stamps each source with its decision and `_BoaCorrectingReader` applies it inside the read. |
| `opera_query.py` | OPERA RTC-S1 query utilities: spatial bbox construction, item construction from the native CMR Granule Search API (bypasses CMR-STAC search; orbit-direction filtered server-side), UTM EPSG derivation, and asset preparation. |
| `boa_offset.py` | The ONE place the BOA offset question is answered. `source_decision` takes a bucket and a declared baseline and returns owed, exempt, or undecidable. Asked per ASSET, which is what lets an item whose bands straddle two producers be corrected band by band instead of refused. Imports no odc on purpose: the GDAL environment has to be configured before `odc.stac` is imported, so the module holding the decision must not be the one that pulls odc in. |
| `item_baselines.py` | The ONE reader of `s2:processing_baseline`. Reports an integer hundredth (`04.00` -> `400`), the same scale `S2_BASELINE_THRESHOLD` is expressed in, and `None` for every kind of unreadable. Exists because there were two readers on two scales with two notions of unreadable, so each numeric edge case had to be fixed twice — and they had drifted before anyone noticed. |
| `asset_locations.py` | Answers the two questions asked of where an item's assets live, and keeps them apart: **is the read cheap** (a property of the bucket's REGION) and **has the reflectance offset already been removed** (a property of the PRODUCER). They have the same answer today and would not the first time anyone mirrors unharmonised data in region, so two bucket lists make that impossible to conflate. Also holds `AssetSources`, which reports the keys it could NOT resolve rather than dropping them — the primitive that generated the same defect three times when it returned only what it found. Its item-level harmonisation answer feeds duplicate RANKING; the correction itself is decided per asset in `boa_offset.py`. |
| `duplicates.py` | Chooses between DUPLICATE catalogue items for one tile-date — Element 84 publishes more than one whenever a granule is reprocessed, distinguished by `s2:sequence`. Usable first, then the newest baseline, then the copy owing no offset correction; the rejected copies are retained as a fallback the write steps down when a source object will never read. Reducing to one copy before the loader is what makes a fallback possible at all, since `odc.stac.load` FUSES a solar-day group — and it is also what makes the recorded baseline match the pixels written. |
| `loader_failures.py` | Keeps what a failed load knows — WHICH object, and WHY — since neither reaches the caller on its own and both need code running on the reader before the read fails. One `install_capture_everywhere` call per ingest installs both on every current and future worker. Names the source object a failed load could not read. `odc.stac.load` reports it in its OWN log record and raises an exception that does not carry it, so a logging handler on every reader process records each aborted href, and the caller collects them after a failure and maps them back to tile-dates. That name is what lets the duplicate ladder step down ONE copy instead of every duplicated tile in the date. Attribution is best effort by design: an empty answer means "attribute nothing", never "nothing was at fault", and the recovery it sharpens still works without it. |
| `auth.py` | NASA Earthdata Login (EDL) authentication for ASF-hosted OPERA data. Provides S3 direct access (temporary AWS credentials minted by ASF, ~1 hour) and legacy CloudFront signed URL resolution. Those credentials expire on their OWN clock, unrelated to any unit of work, so renewal must be driven by a timer and by the advertised expiry — never by the work loop, which can only renew between units and so cannot renew inside one that outlives the margin. |
| `transforms.py` | Post-load lazy Dask transforms. Currently: `amplitude_to_db` for converting OPERA RTC-S1 linear amplitude to scaled uint16 dB. |
| `roi.py` | ROI (Region of Interest) utilities: reading existing Zarr ROI stores (WGS84 bbox, CRS, grid dims), rasterizing GeoJSON polygons to chunked boolean Zarr masks on UTM grids, and loading S2 MGRS tile footprints from S3. |
| `source_coverage.py` | Optical-source preflight: whether the catalogue publishes ANYTHING reaching a zone's live land in a window, answered by limit-1 existence probes over live-tile block envelopes (window padded per the solar-day convention) before any cluster is provisioned. The verdict is three-valued — only a positive finding of absence refuses; inconclusive and provisional-present both pass the cell through, because a wrong refusal loses campaign coverage while a pass-through only costs the late failure it would have hit anyway. Deliberately OUTSIDE the mosaic-content fingerprint closure (its probe is built here, not in `stac.py`), so shipping preflight changes never invalidates in-flight mosaics. Called by the chained fill driver's pre-cluster triage. |
| `catalogue_refusal.py` | Tells a catalogue that is BUSY apart from one that cannot serve a given REQUEST, and names the request either way. The client stack discards the search body when it wraps a transport failure, so a refusal arrives identifying the host and the endpoint path and nothing about what was asked. Both refusals are one exception type from one endpoint and need opposite responses: waiting is the entire remedy for a stated overload and pure waste against a request that is refused every time. The status separates them but does not settle it — what settles it is an identical REPEAT, which only the layer holding the attempt budget can observe, so this module classifies and that layer supplies the repeat. |
| `roi_processing.py` | Higher-level ROI processing helpers used by the `generate_roi` flow. |
| `_pipeline.py` | A prepare/consume pipeline with a configurable look-ahead `depth`: overlaps the preparation of the next item with the consumption of the current one on one background thread, and reports the preparation time the consumer had to wait for. Used by the S2 date loop (`pipeline_dates`), with `depth` sized to `batch_dates` so a batch's whole preparation can hide behind the previous batch's write. Depth buys BUFFERING, never concurrency — preparation stays on one thread in order, so the side-effect-free contract holds at any depth. |
| `live_windows.py` | (Merge exchange rate is caller-owned: pass `WINDOW_COST_IN_CHUNKS_OVERLAPPED` when a date's windows share one graph, the higher `WINDOW_COST_IN_CHUNKS` when each is a blocking write. Both S2 and S1 select it from how the run writes, so the rate cannot drift from the write strategy it prices.) Derives the chunk-aligned live windows every ingest loads and writes: row bands over the ROI mask's live chunk-rows, then grouped into fewer, taller windows, and narrowed per date to the land that date's imagery reaches. Grouping was originally justified by each window being a serial blocking write — `overlap_window_writes` has since removed most of that serial cost, so the grouping bounds graph size and merge work rather than serial time. Serves single-ROI and campaign runs identically. |

---

## Basic Ingestion Process

The high-level entry point is `ingest_tile()` in `stac.py`. It runs five stages in sequence:

```text
1. Item query        — find items for the tile/bbox + date range (S1: native CMR
                       granule query, orbit-filtered server-side)
2. Item filtering    — optional item_filter_fn pre-filter hook
3. Date dedup        — drop items whose dates are already in the Zarr store
4. odc.stac.load     — lazy-load COGs into a Dask-backed xarray Dataset. The S2
                       baseline correction happens here, per source, inside the read
5. Corrections       — dB conversion (S1)
```

`query_stac_items` and `load_stac_items` expose these stages separately so a flow could
check for new data before spinning up a Dask cluster (see `has_new_stac_dates`, which is
not yet wired into any flow — [issue #47](https://github.com/dClimate/tessera-embeddings/issues/47)).

### STAC Providers and Collections

Provider configs live in
[`config/providers.py`](../config/providers.py) (`PROVIDERS`, `CollectionConfig`). Supported:

| Provider | Collections |
|---|---|
| Earth Search (AWS) | Sentinel-2 L2A, Sentinel-1 GRD |
| Planetary Computer | Sentinel-2 L2A, Landsat (**untested**) |
| CMR-STAC (NASA) | OPERA RTC-S1 |

Each `CollectionConfig` records: collection ID, band list, native resolution, tile ID property
(for S2/Landsat property-based queries), and correction parameters. For OPERA RTC-S1 on
CMR-STAC there is no tile ID property, so the query falls back to a WGS84 bbox.

**NOTE** Planetary Computer is an untested provider. Feedback from Cambridge's TESSERA team
indicates however that Microsoft throttles heavy outbound traffic from Planetary Computer and
hence it's not an ideal provider. For this reason we jumped through all the OPERA RTC hoops.

### STAC Query Strategy

S2 and Landsat are queried by tile ID property (e.g., `grid:code = T33UUP`), which returns
only items for that specific MGRS tile. OPERA RTC-S1 on CMR-STAC lacks an equivalent
property, so queries use a bbox derived from the MGRS tile via `mgrs_tile_to_bbox()`.

#### How one query runs, in plain terms

Four words do all the work in this section. A **page** is one request and the hundred results it
returns; the catalogue will not hand over more at once. A **cursor** is a bookmark — the catalogue
returns a slip of paper meaning "you got as far as here", and the next request must carry it. It is
not a page number: there is no way to ask for the twentieth page directly, and a refused request
leaves no slip for the one after it, so a walk cannot step over a gap. A **date window** is a
from-date and a to-date. A **worklist** is a to-do list of jobs, one job per date window; when a job
turns out to be impossible it is crossed off and two shorter jobs are written in its place.

The problem this shape exists for: Earth Search refuses **a request whose answer would be bigger
than about 6 MB** — AWS Lambda's synchronous response limit. Item sizes vary, so whether a given
hundred scenes clears the cap depends on which hundred the bookmark and date window select, which is
why one request out of hundreds is refused every time while all the others are served. It is not
overloaded — the refusal comes back in 1.3 seconds, as fast as a success — and it is not the
bookmark's fault: the identical bookmark is answered when the date window is shorter, and answered
when only ninety results are asked for instead of a hundred. So the remedy is to ask for a *smaller
answer*, never the same one again.

```text
WHY A REQUEST GETS REFUSED -- the whole mechanism, in one line

   Earth Search will not return more than about 6 MB in one answer.
   That is AWS Lambda's limit on a single synchronous response, and the search API
   sits behind one.

   scenes are not all the same size, so a hundred of them is 4.6 MB on average
   but anywhere from 4.2 to 5.3 MB in practice -- and sometimes over the line:

      100 scenes from this bookmark  ->  would be ~6.2 MB  ->  REFUSED
       90 scenes from this bookmark  ->            5.6 MB  ->  answered
       75 scenes from this bookmark  ->            4.6 MB  ->  answered

   Same bookmark, same dates, character for character. Only the number asked for
   changed. So nothing is wrong with the bookmark, the depth, or the service --
   the reply was simply too big to send.

   Which hundred scenes you land on is decided by your bookmark and your date
   window, which is why it looks like the service has taken against one specific
   request. It hasn't. It is doing arithmetic on the size of the answer.

   THE MARGIN IS THE RISK: 4.6 MB average against a 6 MB ceiling is ~30% of room.
   Fatter scenes -- a newer processing baseline, a provider-side change -- push
   FIRST pages over the line, and a first page is the one refusal that shortening
   the dates cannot fix. `max_page_size` is the lever if that ever happens.


HOW THE QUERY IS SHAPED AROUND IT

   28 Feb ---------------------------------------------------------------- 1 Apr
       |  the catalogue reports the total beside the first page, so a window too big to
       |  walk is cut before the walk starts rather than hundreds of requests in
       v
    +--------+--------+--------+--------+--------+--------+
    |  job 1 |  job 2 |  job 3 |  job 4 |  job 5 |  job 6 |   jobs meet on a shared
    +--------+--------+--------+--------+--------+--------+   INSTANT, so nothing falls
        ok       ok       ok    refused     ok       ok       between them and nothing
                                  |                          is asked for twice
                                  |  cross it off, write two shorter jobs. Shorter dates
                                  |  regroup the scenes into different hundreds, so the
                                  |  fat group is split and every answer fits. The other
                                  v  five jobs are untouched and still running.
                                     If the dates cannot be shortened -- a single day, or
                                     a FIRST request, which a shorter window asks the same
                                     way -- ask for fewer scenes at a time instead.
                            16-22 March
                              +-- 16-19 March   ok
                              +-- 19-22 March   refused
                                    +-- 19-21 March   ok
                                    +-- 21-22 March   refused
                                          +-- 21 March   ok   <- one day is as far
                                          +-- 22 March   ok      as this can go

   Up to six jobs run at once. Almost all of a request is spent waiting for the catalogue
   to think -- 86% of it, before a single byte arrives -- so overlapping the waits is the
   only thing that moves the clock. Same query, same scenes, same order: 39 minutes as
   first written, 3.5 minutes now.
```

Two properties are load-bearing and both are tested. The jobs must add up to exactly the window
asked for, no day missed and no day added. And the results must come back in the same **order**, not
merely the same set — two scenes taken on the same day with the same cloud cover are separated only
by which arrived first, and that decides which one supplies an overlapping pixel.

The rest of this section is the mechanism behind that picture.

#### The months where the catalogue entries are heavy

Late 2018 and early 2019 are the awkward part of the archive, and it is worth knowing why before
it surprises you somewhere else.

Every scene's catalogue entry includes the shape of the ground it covers. Normally that is a
rectangle — four corners, a couple of hundred bytes. But entries from roughly **November 2018 to
March 2019** give the shape of where the imagery *actually* is instead, tracing the edge of the
data rather than the tile boundary. Sentinel-2 builds each image from twelve separate detectors,
so that edge is a fine sawtooth, and drawing it takes thousands of points. One entry we measured
runs to 2,497 points and 98 KB, against 0.2 KB for an ordinary one. The list of files attached to
the scene is about 18 KB either way, so it is the outline, not the imagery, that makes these
entries heavy.

The band exists because of a gap in reprocessing rather than anything about that season. ESA
reprocessed most of the archive to a later version whose entries carry the simple rectangle, and
the catalogue serves that version when it exists. For those few months it does not exist, so the
original products are what you get. Sampling one day a month across the boundary:

| month | typical points per entry | heavy entries | versions available |
|---|---|---|---|
| Oct 2018 | 6 | 0 of 50 | 02.09 |
| **Nov 2018** | **580** | **31 of 50** | **02.11 only** |
| **Dec 2018** | **531** | **33 of 50** | **02.11 only** |
| **Feb 2019** | **579** | **30 of 50** | **02.11 only** |
| Apr 2019 | 6 | 0 of 50 | 02.11, 05.00 |
| Jun 2023 | 5 | 0 of 50 | 05.09 |

The heavy months are exactly the ones where version 02.11 is the only one on offer. Why that
version drew outlines this way when the versions either side did not is an ESA processing decision
and not something we have chased.

Two things follow. It **cannot spread** — no current processing version produces these outlines,
so no future data brings the problem back. And it **could disappear on its own**, if that stretch
is ever reprocessed.

What it presses against, in this code:

- **The response cap.** A hundred entries is about 2.2 MB outside the band and at or over the
  ~6 MB ceiling inside it, which is what makes a page refusal a 2019 phenomenon. See the page-size
  discussion below.
- **What a query holds in memory.** A month's worth of these entries is an order of magnitude
  more bytes than the same month in 2024, which is part of why the query streams month by month
  rather than fetching a year at once.

Anything that measures per-item cost — page sizes, retained bytes, query timings — will read
differently in these months than anywhere else in the archive, so a figure taken here is not a
figure about the campaign generally, and vice versa.

`_query_stac_items` configures retries at the HTTP layer via a custom `urllib3.Retry`
built by `make_logging_retry()` (`_http.py`, shared with the CMR Granule query) and passed
into `StacApiIO` (`total=8, backoff_factor=2, status_forcelist=(429, 500, 503, 504)`).
The subclass logs each retry attempt at WARNING — urllib3 otherwise retries silently inside
the `HTTPAdapter`, making a slow query indistinguishable from a hang. Because
`search.items()` paginates lazily, each page fetch is a separate HTTP call — configuring
retries on the underlying `HTTPAdapter` means a transient 5xx on page N is retried in
place, instead of throwing away prior pages and restarting the whole query.

**502 is deliberately absent from that force-list**, so an Earth Search page refusal
arrives unretried and the date-window re-cut below can start immediately instead of after
the ladder's backoff. A 502 that really was transient is absorbed by the attempt budget
that owns the leg, which is where a refusal is settled anyway — it takes an identical
repeat to prove one deterministic. The CMR Granule query keeps its own ladder
(`opera_query._CMR_RETRY`), 502 included, and is unaffected. Note that
`StacApiIO`'s default `max_retries=5` passes a bare int to urllib3, which has an empty
`status_forcelist` and therefore does **not** retry 5xx responses — the explicit `Retry`
object is required.

The page size is `STACProvider.max_page_size` (the `limit` per page request), defaulting
to 250 and set to 100 for Earth Search. This applies only to providers queried through
`client.search()` (Earth Search, Planetary Computer) — the OPERA `cmr-asf` path bypasses
CMR-STAC search entirely and queries the native CMR Granule API. See
[ADR 009](../../../context_docs/decisions/009-native-cmr-granule-query.md) for the full rationale.

**Why 100 for Earth Search, and what to watch.** It is not a throughput choice — it is the
response cap. Earth Search refuses any request whose response would exceed roughly **6 MB**,
AWS Lambda's synchronous response limit, and 250 items of `sentinel-2-l2a` is always over it.
A hundred items averages 4.6 MB, but the biggest page ever **served** was **5.73 MB — 96% of the
cap** — and the refused one works out to 5.96 MB. So the gap between fine and refused is about a
quarter of a megabyte, and the average is the wrong number to reason from.

What makes an item heavy is the shape of the ground it covers rather than the files attached to
it, and the heavy items are confined to a few months of 2018 and 2019 — see *The months where the
catalogue entries are heavy* above. Outside that band a hundred items is about 2.2 MB, a third of
the cap.

Lowering this number is therefore not the answer — see the campaign record for the measurements.
Six months of a ten-year archive is a concentrated problem, and the page-size fallback below handles
it where it happens rather than taxing every query in every year. The cap tracks **bytes, not the `limit` value**: measured directly, 130
items are served at 5.99 MB and 150 refused, while 150 are served at 4.77 MB once unneeded
assets are excluded server-side. If first pages ever start returning 502, this margin is the
first thing to check and lowering this number is the lever — a first-page refusal is the one
case no date-window re-cut can route around.

**A page request deep in a walk is refused by the SAME ~6 MB response cap, and there are two
levers for it.** Some individual page requests are refused with a 502 while
the rest of the same walk is served, which looks like a separate defect and is not: item sizes
vary, so whether a given hundred items clears the cap depends on which hundred the cursor and
date window select. That is why the refusal is deterministic in the *request* — cursor and date
window together — rather than in how deep the walk has got, and why it has been seen at page 289
of one window and page 14 of a shorter one sharing its late bound. Re-sending the byte-identical
cursor and window at `limit=90` is served, returning 5.60 MB where the hundred would have been
about 6.2 MB. So waiting cannot absorb
it — which is why 502 is kept out of the retry ladder above. **It is the same cap
`max_page_size` was lowered for**, reached from the other direction: that one refuses a first
page because 250 items are always too many, this one refuses a later page because those
particular hundred items happen to be. A smaller page from the refused cursor IS served, and
is the most direct remedy; it is not used here only because `pystac_client` bakes the limit
into a search and cannot resume from a cursor at a different size.

What clears it is either a smaller response or a regrouping that produces one. `stac.py` tries
them in cost order.

**A shorter window first.** Its halves between them walk about as many pages as the parent
would have, where a smaller page re-walks the whole window at twice the requests. What matters
is the window's **end** date; shortening only the start does not help, because the catalogue
pages newest-first and the late bound fixes the whole cursor sequence. So `_query_stac_items` re-queries the window as shorter windows on any
upstream-error refusal past the first page, recursing until a window completes or is down to
a single day. Separately and proactively, it reads the match count the catalogue reports
beside the first page and cuts a window matching more than `_MAX_QUERY_ITEMS` to size — that
bounds *cost*, keeping a refusal from discarding a long walk, and is not what fixes the
defect. **A smaller page as the fallback**, halved each time down to `_MIN_PAGE_SIZE`. Shortening
cannot reach two refusals, and both failed a leg outright before this existed: a **first page**,
which a shorter window asks identically, and a **single day**, which is the re-cut's own floor.
Verified live — a 250-item page is over the cap and refused outright, and the recovery stepped
250 → 125 → 62, interleaving with a window cut, and returned all 2,512 items with the post-sort
order and the extracted baselines unchanged, for twice the requests.

A stated overload (429, 503) is never answered with a smaller page: that means the provider is
busy, and more requests is the wrong direction. A refusal neither lever can route around still
raises the classified `CatalogueQueryError` with its token.

**Concurrency.** The windows are independent searches, so `_fill_window_tree` walks up to
`_QUERY_WINDOW_WORKERS` (6) of them at once. The worklist is driven from the calling thread and
tasks only ever walk — they never submit and never wait — which is what makes deadlock
structurally impossible rather than merely unobserved. Each thread gets its own `Client`, because
`StacApiIO` wraps a `requests.Session` that is not documented thread-safe. Output order comes
from `_WindowWalk.preorder()` on the finished tree, and the `id` dedupe runs at that assembly step
rather than as pages arrive, so first-occurrence-wins means first in the **walk** and not first
off the wire.

Six rather than eight, even though eight is faster: the campaign runs tens of cells against this
one provider at once, so the setting is a multiplier on the concurrent search streams Element 84
sees, and measured per-page latency degrades with width. A failure no longer stops the other
windows either — every window is walked, all failures are collected, and the depth-first-earliest
is raised, because which failure surfaces must be a function of the query rather than of which
task finished first. The attempt-budget layer above decides a refusal is deterministic by seeing
the identical signature again.

The re-partition is a pure re-cut of the window, never a narrowing, and it is exact in both
directions: the outer bounds are the caller's own date strings handed straight back, and every
interior boundary is a single **instant** (`T00:00:00Z`) shared by the window that ends there
and the window that starts there. The catalogue's range is inclusive at both ends, so the union
is the input window with no gap and no overhang, and the only overlap is that one instant.

The boundary is an instant rather than a date because the client expands a bare date end to
`T23:59:59Z`: windows abutting on consecutive DATES would leave the last second of each seam's
earlier day unasked for, a second the unsplit window covers. Sharing the whole boundary DAY
also closes that gap, and was how this first shipped, but it makes every seam re-fetch a full
day for the dedupe to discard — about 13 page requests each at Earth Search. Items are still
deduped by `id` across every search a query runs, which absorbs the boundary instant and the
antimeridian overlap alike.

**Item order.** The re-partition does change the order items are *walked* in — one walk returns
the window newest-first, the worklist returns window by window in date order — and that matters
because `query_stac_items` sorts with `solar_day_sort_key`, which orders a solar day's items
clearest-first and settles equal-cloud ties on `id` — so the sequence is a function of the items
rather than of the order the walk produced, and the loader's fuser keeps the first valid source of
each group. Verified against an unsplit walk at several part counts; see
[the ingest campaign record](../../../context_docs/design/ingest_optimization_campaign_2026_07.md),
which also records the measurements and the two optimisations that are closed (a larger page,
and server-side field selection).

#### When the catalogue refuses: naming the request, and telling the two refusals apart

`catalogue_refusal.py` is where a refused query stops being anonymous. Two things about the
client stack make that necessary:

- **The request is discarded on the way up.** `StacApiIO.request` catches every transport
  failure and re-raises `APIError(str(err))`. What survives names the host and the endpoint
  path; a STAC search is a request **body**, so the collection, the window, the bbox and the
  page are all gone. Without them a refusal cannot be narrowed to a month or a page,
  reproduced, or reported to whoever runs the archive.
- **Our layer sits ABOVE a retry ladder, and only partly behind it.** For a status the
  `urllib3.Retry` above force-lists, what escapes is the ladder reporting its own
  exhaustion — a much stronger statement than one error response, and one that must not be
  mistaken for a first attempt. For a status kept out of that list (502) the first refusal
  arrives directly. `CatalogueRefusal.exhausted` records which, so no caller has to assume.

So `_query_stac_items` pages explicitly (`pages_as_dicts`, which is what `items_as_dicts`
iterates internally) and wraps **only the page fetch** in a `CatalogueQueryError` carrying a
`CatalogueRequest`. Wrapping the page body as well would classify our own validation
failures as someone else's outage. Opening the catalogue is page 0, named separately so a
root outage is not attributed to a window that was never asked for.

```text
CATALOGUE REFUSED collection=sentinel-2-l2a window=2021-09-01/2021-10-02
                  bbox=-3.0000,50.0000,-2.0000,51.0000 page 3
                  with HTTP 502 without being retried after 500 item(s)
                  — classified upstream-error:502
```

The **classification** separates two refusals that arrive as one exception type from one
endpoint and need opposite responses:

| refusal | statuses | what it claims | response |
|---|---|---|---|
| `LOAD` | 429, 503 | the upstream names ITSELF as the constraint | wait — this is what the expansive retry exists for, however often it recurs |
| `UPSTREAM_ERROR` | 500, 502, 504 | the upstream failed to PRODUCE an answer | retry once; a repeat settles it as deterministic |
| `UNKNOWN` | anything else | no readable status | behave as the default does: retry |

The two named sets must jointly cover the ladder's `status_forcelist` — a status the ladder
retries but the taxonomy does not name falls to `UNKNOWN` and keeps its expansive retry
forever. A unit test asserts that containment rather than leaving it to care. The converse
is allowed and deliberate: the taxonomy names 502, which the ladder does **not** retry, and
a second test pins that exclusion so re-adding it cannot quietly restore the backoff the
window re-cut exists to avoid.

The status is read from the exception **chain**, not the message: `pystac_client` re-raises
without `from`, so the evidence sits under `__context__` on urllib3's own exception, and the
top-level text is only a stringification of it. The message is a documented fallback for a
refusal that crossed a boundary carrying no chain.

**A status is necessary and not sufficient.** A gateway can fail for minutes and recover, so
one exhaustion is not proof of a defect. What settles it is a REPEAT — the identical request
refused the identical way on a later attempt — and that observation belongs to whoever holds
the attempt budget, which is `ingest_zone_year`'s leg loop (see
[its retry policy](../orchestration/prefect/flows/ingest_zone_year.py)). The two halves are
deliberately split: this module classifies, the budget holder supplies the repeat, and
neither is a verdict alone.

That split forces the signature's design. The leg that queries and the layer that counts
attempts are separate deployment runs, so the only thing crossing between them is failure
text — hence one whitespace-free token under a stable name (`CATALOGUE_REFUSAL=`), matched
by name and never by position. And the signature covers exactly the fields that decide the
answer (collection, window, area, page) and nothing that varies between attempts: a counter
or a timestamp inside it would make every refusal unique and the repeat check dead code that
always reports "not repeated".

**Attempts are the only thing those budgets count; elapsed time has exactly one bound.**
Each page fetch already gets 9 HTTP attempts across 364 s of exponential backoff before
anything above the ladder sees a failure, and every attempt budget above it — leg, cell,
zone round — treats the whole layer below as one try. None of them reads a clock, and
expansive backoff makes the clock the axis that can grow without limit.
`IngestSettings.max_leg_wall_clock_s` bounds it in the leg loop, at the one place that
cannot defeat the patience it serves: once the deadline has passed, the loop refuses to
START another attempt. A leg that is running is never measured against it — a
slow-but-succeeding leg cannot be why the loop stopped — so the loop's worst case is the
deadline plus one final attempt. And failing the cell this way is not surrender: the cell
returns to the campaign's work list, and a later dispatch RESUMES from the dates already
committed (Icechunk commits a date's time slot atomically with its pixels), so the bound
costs latency, never work. The default's derivation against measured leg durations is in
`context_docs/design/ingest_read_failure_causes_2026_08.md` (cause 3).

`source_coverage.py`'s preflight probe deliberately does **not** use any of this. Every
failure of that probe is already INCONCLUSIVE by design, which is the right answer for both
refusals at once, and the module sits outside the mosaic-content fingerprint closure.

Cloud cover is intentionally **not** used as a filter at the STAC query stage — pixel-level
cloud classification is handled later (SCL for S2, ML model for inference). For S2, items
are sorted by `(date, eo:cloud_cover)` ASCENDING, so the clearest tile of a solar day comes FIRST.

First, because that is the one the loader keeps. `odc.loader`'s default fuser is `nodata_fuser` —
`np.copyto(dst, src, where=dst_is_nodata)` — so it writes only where the destination is still empty,
and this package configures no fuser of its own. The first valid source of a group therefore
supplies a pixel and later ones fill the gaps it left, which is the behaviour wanted: the clearest
scene covering a pixel wins it, and a hole in the clearest scene falls through to the next-clearest
rather than to nothing.

An item declaring no `eo:cloud_cover` sorts after every measured value, where it can only fill
gaps: unknown never displaces measured. `solar_day_sort_key` gives it infinity rather than 100 —
100 is itself a real reading, and the two would otherwise tie and be reordered by `id`, letting an
unmeasured scene take ground from one known to be fully clouded.

#### Streaming the query month by month (S2)

`ingest_s2_roi_reflectance` queries **one month at a time** by default
(`stream_stac_monthly`), prefetching the next month while the current one is being
ingested. Querying a whole window up front is simpler, but it retains every returned item
for the run's duration, and a zone-year's worth does not fit alongside the ingest on one
worker. Streaming bounds retention to the month in hand plus the one being fetched.

The prefetch runs on a daemon thread rather than a pooled worker: an in-flight catalog
walk cannot be interrupted from outside, so abandoning it is the only way to stop waiting,
and a daemon thread does not hold the process open when a run is cancelled mid-query.

Month ranges **partition** the window — each month owns a half-open slice and items are
filtered to their owner — so a date cannot be ingested twice or skipped at a boundary.
Set `stream_stac_monthly=False` to issue one query for the whole window; the per-date work
is byte-identical either way.

#### Antimeridian queries

UTM zones 01 and 60 straddle ±180°, and the ROI's WGS84 bounding box is written in the
GeoJSON crossing convention (`west > east`) so it stays narrow instead of spanning the
globe. Neither catalog can be relied on to read that form — the native CMR path is known
to reject it — so both query paths split the box at ±180° into two ordinary west-to-east
queries and deduplicate the results by item id. A granule straddling the line is returned
by both halves and must be loaded once.

---

## ROI Workflow

The Tessera pipeline uses a spatial ROI mask — a chunked boolean Zarr array stored on S3 — to
define the area of interest for all downstream ingestion and inference.

### Generating an ROI

`generate_roi` flow calls `roi.rasterize_roi_zarr`. Steps:

1. **Load geometry** — from a local or S3 GeoJSON (`input_path`) or a pre-loaded list of
   Shapely geometries. Alternatively, `load_s2_tile_geometry` fetches MGRS tile footprints
   from the S3 tile index.
2. **WGS84 bbox** — computed from the *original* geometry **before** reprojection. Using
   the post-projection axis-aligned bounds would inflate the bbox significantly at oblique
   UTM zone edges (see `docs/bbox-projection-inflation.md`).
3. **CRS selection** — `determine_target_crs` picks (in order): user-specified `force_crs`,
   the input CRS if it is already projected, or the best UTM zone derived from the
   geometry centroid. Geographic CRS output is rejected with an error.
4. **Grid computation** — `compute_grid` converts projected bounds + resolution into pixel
   dimensions and an Affine transform.
5. **Chunk-at-a-time rasterization** — `rasterize_roi_zarr` iterates over `chunk_size × chunk_size`
   blocks, calling `rasterio.features.rasterize` per chunk with a chunk-local transform.
   The full boolean grid is never held in memory.
6. **Zarr attrs** — `crs`, `transform` (6-element Affine list), `resolution`, `bbox_wgs84`,
   and a `_manifest` written atomically after all chunks succeed.

### Reading an ROI

Ingestion flows call:

- `read_roi_metadata(roi_path)` — returns `ROIMetadata`: WGS84 bbox (for STAC queries),
  native CRS string, `odc.geo.GeoBox` (for `geobox=` kwarg to `odc.stac.load` so output
  grids align exactly), width/height.
- `read_roi_mask(roi_path, chunks)` — returns a lazy Dask boolean array for masking.

### Applying the ROI Mask

`roi_processing.apply_roi_mask` broadcasts the 2D mask over the time dimension and sets
out-of-ROI pixels to `fill_value` (default 0) across all dataset variables.

`roi_processing.filter_low_coverage_dates` then drops time steps where fewer than
`min_valid_coverage` percent of ROI pixels are valid (default 5%). Only the per-date valid
pixel counts are computed eagerly — band arrays remain lazy until the Zarr write.

`identify_low_coverage_ds` is the lazy alternative: instead of dropping dates it attaches a
`valid_coverage` boolean coordinate that downstream tasks can check without reading band data.

### Zone ingestion (the global campaign) — ADR-011

The global campaign reuses this exact ROI engine to produce its per-zone mosaics: it
**synthesizes a zone-shaped ROI** instead of rasterizing a GeoJSON, then dispatches the same
S1/S2 ingest flows. `generate_roi`'s `compute_grid` bbox-fits geometry and cannot reproduce the
fixed, shard-snapped `zone_grid.ZoneSpec` extent the fill validates against — so
`land_mask.export_zone_roi` writes the ROI mask directly from `ZoneSpec` (mask = the zone's
`tile_live_2048` coverage bitmap upsampled ×2048; WGS84 bbox tight to the live tiles).

```text
run_global_campaign  (per pending (zone, year), zone-parallel within a year)
   │
   ├─ ingest-zone-year ──► export_zone_roi(zone)         {inputs}/rois/zarrs/zone_33N.zarr
   │      │                  (ZoneSpec grid + tile_live mask; ocean-tile skip)
   │      ├─ marker probe (ingest_marker fingerprint; stale/partial ⇒ clear+rebuild)
   │      ├─ (live-chunk count ⇒ max_workers)
   │      ├─ ingest_s1_roi_sar × orbit ┐  concurrent, onto
   │      ├─ ingest_s2_roi_reflectance ┘  {inputs}/mosaics/33N/2025/
   │      ├─ check_time_window_coverage (strict span; allow_partial_window escape)
   │      └─ write ingest_marker  (last — crash before this ⇒ clean rebuild on re-run)
   │
   ├─ fill-zone-year  ──► coverage + SAR-grid + model gates (pre-Ray) ──► inference ──► assemble ──► tag
   │
   └─ delete mosaics/33N/2025  (s5cmd --all-versions; transient input)
```

The S2 `min_valid_coverage` bar is lowered far below the ROI default (5 % → ~0.1 %): a single
solar-day's swath covers only a sliver of a whole 6° zone, so a high bar would drop nearly
every date. Mosaics are per `(zone, year)` and deleted after the fill is tagged — they are
re-derivable inputs at ~TB scale (ADR-011). Zones are named by UTM common name (`33N`/`07S`),
not EPSG (see `storage/zone_grid.canonicalize_zone`).

#### Pre-generating the zone masks — `export-zone-rois`

`ingest-zone-year` exports the mask it needs on the fly, so the campaign is self-sufficient.
The `export-zone-rois` flow does the same work for many zones **ahead of the campaign**, and
adds the check the per-cell path has no reason to run:

```text
export-zone-rois  (one task per zone, max_parallel_zones in flight, no barrier)
   └─ per zone ─► live_chunk_count(zone)        coverage bitmap, one ~KB GET
                  ├─ 0 live chunks ⇒ all_ocean (no mask by design; nothing written)
                  ├─ export_zone_roi(zone)      skipped when already current
                  └─ validate_zone_roi(zone)    grid · completion · placement · layout
```

`validate_zone_roi` is the reason to run this early. Its load-bearing check is **placement**:
the count of stored chunk objects must equal `live_chunk_count`. That equality holds because
the writer skips all-ocean blocks and Zarr elides all-fill chunks, so *the set of stored chunks
is the set of live cells* — one listing asserts, for the whole zone, that the mask marks land
where the coverage bitmap says land is and nowhere else. It also confirms the chunk grid is
recoverable from the keys at all, which is the property the cropped ingest's fast path depends
on (see `live_windows.live_chunk_grid_from_keys`). Alongside that: shape/CRS/affine equal the
zone's `ZoneSpec` (a wrong origin otherwise surfaces hours later, as data on the wrong ground
position), and `coverage_sha256` matches the coverage group's `registry_sha256` — stamped last
by the writer, so it is the only evidence every pixel landed and that the mask is current for
*this* land-mask delivery.

Safe to run before, or alongside, campaign work. `export_zone_roi` is idempotent on that same
sha, so a pre-generated mask is what the campaign would have written and the campaign skips it;
a new delivery changes the sha and both paths rebuild. `validate_only=true` re-checks without
writing. Any invalid zone **fails the run**, so a green run — not a log line — is the evidence
that every mask is right.

```bash
# all zones, then re-assert the gate without writing
--param zones=null --param max_parallel_zones=16
--param validate_only=true
```

Cost is S3 request latency, roughly one PUT per live ingest chunk (~100 k campaign-wide across
the 112 land zones), which is why it fans out per zone and why running it in-region matters:
the same export measured ~4 chunk-writes/second from a laptop.

---

## Data Transformations

### Pre-load (STAC items)

These happen before `odc.stac.load` is called:

| Transform | Where | What |
|---|---|---|
| **Date dedup** | `stac._filter_existing_dates` | Drops STAC items whose date is already written to the store. Keyed on the SOLAR day when the caller passes `mid_longitude`, matching how the store was written. |
| **Item sort** | `stac.query_stac_items` | For S2: sorts by `(date, cloud_cover)` ascending, so the clearest tile comes FIRST and the loader's first-valid-source fuser keeps it. |
| **Item provider** | `opera_query.make_s1_item_provider` | Builds orbit-filtered OPERA items directly from the native CMR granule API (bypasses CMR-STAC search). |
| **URL rewriting** | `auth.rewrite_assets_to_s3` | Rewrites HTTPS datapool/earthdatacloud URLs to `s3://` URIs. |
| **Timestamp normalisation** | `solar_days.normalize_to_solar_day` | Stamps every item with noon UTC of its **solar day**. The single place the solar offset is applied; also what makes `odc.stac.load` mosaic OPERA's per-burst granules into one time slice. |

### Load-time (`odc.stac.load`)

`_load_from_stac` configures `odc.stac.load` with:

- **Resampling** — bilinear for primary spectral bands. Extra bands (e.g., S2 SCL) always
  use nearest-neighbour regardless of the primary resampling, enforced via a per-band dict.
- **Resolution override** — S1 is loaded at 10 m to share a common grid with S2, even though
  the native OPERA product is 30 m. Resampling to target resolution uses COG overviews and
  happens during read rather than as a post-processing step.
- **CRS override** — OPERA RTC-S1 items on CMR-STAC lack `proj:` extension metadata; an
  explicit `crs=` (e.g. `EPSG:32633`) must be passed so `odc.stac.load` knows the output
  projection.
- **GeoBox alignment** — when a `GeoBox` derived from `read_roi_metadata` is supplied, the
  output grid matches the ROI exactly (same CRS, transform, shape). This overrides bbox,
  CRS, and resolution.
- **groupby** — `"solar_day"` merges items from adjacent MGRS tiles that were acquired on
  the same local calendar day into a single mosaic. `odc.loader`'s default fuser writes only
  where the destination is still empty, so the FIRST valid source of a group supplies a pixel
  and later ones fill its gaps. Items are therefore sorted clearest-first: the clearest scene
  wins the ground it covers, and its holes fall through to the next-clearest rather than to
  nothing. Required for ROI queries that cross tile boundaries.
- **Dimension rename** — `normalize_odc_dims` maps `odc.stac.load`'s `y`/`x` output
  dimensions to the project-wide `northing`/`easting` convention and drops `spatial_ref`.

### Sentinel-2 Baseline Correction (load-time)

ESA changed the S2 L2A processing baseline at version 04.00 (January 2022), adding +1000 to all
surface reflectance values. Whether that offset has to be subtracted is a property of **who
served the pixels**, not of the collection: Element 84 harmonises its own COGs and subtracts it
for you, while ESA's originals carry it.

Leaving `baseline_threshold` unset for Earth Search assumes every item is a harmonised COG, and
that fails once the collection also indexes items whose assets point at ESA's archive. Reading the
collection alone exempts or corrects both kinds together, and one of those is always wrong and
always silent: a skipped correction leaves plausible pixels 1000 too high, a doubled one shifts
every value by 1000.

So the threshold **is** set for Earth Search, and the decision is made per ASSET from where that
asset lives (`boa_offset.source_decision`). Three properties of it are worth stating:

- **Judged over the reflectance bands only.** A real Element 84 item carries the original JP2s
  as extra assets beside its COG bands, so judging every asset reports a straddle for an item
  that is wholly harmonised where it matters. `scl` is excluded even though it *is* read: it is
  categorical and never corrected, so its producer cannot make the reflectance ambiguous. This
  is a different asset set from the one locality is judged over — see the duplicate selector,
  which uses the full read set including `scl`.
- **Decided per SOURCE, and applied as each image is read.** `odc.stac.load` fuses a solar day
  into one time slice, so a correction applied to its OUTPUT is applied to every tile at once —
  and a day whose tiles disagree then has no correct answer and was refused, at a measured cost of
  347 days of one region-year. The decision is now made per reflectance asset
  (`boa_offset.source_decision`), stamped onto each source at parse time (`stac.BoaOffsetParser`)
  and applied inside the read (`stac._BoaCorrectingReader`), before anything is resampled together.
  Different tiles occupy different ground, so no pixel is both corrected and uncorrected.

  This is **purely additive**: the amount subtracted is a constant, so a day the pipeline loaded
  before produces bit-identical pixels, and only previously-refused days change. See
  [ADR 021](../../../context_docs/decisions/021-correct-the-boa-offset-per-image.md) §3.
- **Thresholded on the item's own declared baseline, and an unreadable one refuses.**
  An absent or malformed `s2:processing_baseline` parses as nothing rather than as 0: reading it as
  zero puts it under the threshold and exempts pixels that may carry the offset, while correcting
  it takes 1000 off pre-04.00 pixels that never had it. Both are wrong by the same amount in
  opposite directions and both are silent, so the source refuses. `item_baselines` is the only
  reader of that property, on one scale, with one notion of unreadable.

**Consulting the assets at all is scoped to the collections that need it**, via
`CollectionConfig.harmonisation_varies_by_item`. That read looks assets up under the keys named in
`bands`, which is how Earth Search keys its assets and is NOT how every provider does: Planetary
Computer serves the same imagery under native keys (`B02`, `SCL`) and relies on the loader
resolving the common names, so the read finds nothing there. On Planetary Computer it therefore
classified every modern item as undeterminable and refused every date at baseline 04.00 or above.

So where the producer cannot vary between items, the **collection's own configuration supplies the
answer**: a correction threshold on such a collection says every item is unharmonised, which is
what the threshold is there to correct. `source_decision` takes that as `known_harmonisation` and
does not consult the bucket at all — which is what lets a provider serving its bands under native
asset keys be decided here, and is why every Planetary Computer source is corrected rather than
refused. One decision then serves both providers, so they cannot disagree about a producer.

Which assets carry the reflectance bands is resolved through **odc's own alias table**
(`stac._reflectance_asset_keys`), not from the configured band names: Planetary Computer serves
`blue` as an asset called `B02`, so deciding against the configured names would match none of its
assets and correct nothing. `scl` is excluded structurally — it is simply not among the resolved
keys — rather than by a list the corrector is told to skip.

The correction VALUE is a **constant** — `S2_BASELINE_OFFSET`, `-1000`. The baseline decides only
*whether* the offset is removed, never how much, which is what makes the move from a per-date to a
per-source decision produce bit-identical pixels on every day the pipeline already loaded.
`extract_baselines` remains separate and untouched: it records what each item declared, is what
reaches the store's `baselines_applied`, and is **not** a correction input — one integer per date is
provenance, and correcting from it left raw post-04.00 pixels uncorrected whenever it omitted a
date, carried the zero an unreadable baseline collapses to, or named an arbitrary item's baseline on
a multi-item date.

Duplicate selection has **two owners**, and neither is the shared query. `query_stac_items`
deliberately does not prune: `s2_roi` runs its own selection over that output and keeps the
rejected copies as the ladder `step_down_copies` walks when a source object will not read, so
pruning upstream would leave it nothing to step down to. `load_stac_items` prunes for everyone
else — both the documented `query_stac_items` -> `load_stac_items` workflow, which passes through
neither of the others, and `ingest_tile`, which leaves it to the loader: the loader realigns
`baselines` in place and that is the same dict `ingest_tile` returns, so the map still describes
the copy that was kept.

Selecting over an already-selected set is a no-op, which is what makes more than one owner safe.

**One ambiguous shape survives, and it refuses** (`HeterogeneousProducerError`): a reflectance
source at or above the threshold whose producer cannot be determined — served from a bucket nobody
has classified, or belonging to an unharmonised copy that declares no readable baseline. Correcting
it and exempting it are wrong by the same amount in opposite directions and both are silent, so
nothing guesses. The refusal is a property of that one source, and it is gated on the threshold:
below it no producer changes a pixel, so there is nothing to decide.

**The shapes that cost whole days are gone**, because a day is no longer decided as a whole. Tiles
straddling the threshold, and a raw item owed the offset fused with an already-harmonised one, are
each corrected on their own ground, so no pixel is both corrected and uncorrected — the 347 days a
2024 region-year lost to that conflict now load. An item whose own bands are served from two
different producers is likewise corrected band by band. See
[ADR 021](../../../context_docs/decisions/021-correct-the-boa-offset-per-image.md), and
`context_docs/decisions/020-boa-offset-applies-to-every-valid-dn.md` for the frequency table the
change was justified against.

A refusal that does still happen is skipped alone and counted rather than failing the leg, and
duplicate selection routes around it: a copy that would refuse is ranked last *and* withheld from
the fallback ladder, because a refusal is not a read failure and nothing retries one.

"Alone" is a property of `s2_roi`, which loads ONE solar day per call — not of the refusal, which
is raised while `odc.stac.load` parses its item list synchronously and so abandons whatever list it
was given. A caller pairing `query_stac_items` with `load_stac_items` over a multi-day list
forfeits every day in it, so pass a day at a time wherever one undecidable day should not cost the
window. Production does not reach this: the only `ingest_tile` caller is the S1 path, whose
collections have no baseline threshold and therefore no offset decision to refuse.

The surviving case is worth naming, because the safe direction inverts. An item that does not expose
EVERY reflectance band under the configured names is `UNKNOWN`, not `RAW`. A non-empty subset is not
enough: nothing in this module can see the alias table that maps a band name to an asset key, so a
band absent under `blue` may be served under `B02`, and `_prune_item_dict` preserves exactly those
partially aliased items. Letting one visible harmonised band speak for a hidden native-keyed one
would subtract 1000 from pixels that may already be harmonised, silently. The live Element 84
catalogue keys its assets by the configured band names, so this does not arise there today; a
catalogue that changed its naming would stop the ingest rather than halve a season's reflectance.

**How far the decision reached is reported after every load**, from counters the parser keeps:
how many reflectance sources it stamped, and how many of those were owed the offset. Reaching
*none* of them is the one way this can still go wrong quietly — an empty or mistaken
`reflectance_assets` set corrects nothing and produces plausible pixels 1000 too high — so a zero
count is a WARNING naming the assets it resolved, and any other count is an INFO line. A warning
rather than an error, because a caller that replaces the loader also reports zero; what makes that
enough is that every other way of getting this wrong is loud, since an unclassifiable source
refuses and an unresolvable band name fails the load.

One integer per date is a **lossy** record of a day whose tiles declare different baselines, and
those days now load — the three-tile day in ADR 021 is exactly that shape. The value is the first
item's, the clearest tile, because the query sorts a date's items cloud-ascending; the day's other
vintages are recorded nowhere. Left that way deliberately: the attribute is written and merged
forward on append, and nothing in this package reads a value from it, so widening its type would
charge a future reader for a record nobody reads yet.

`correct_boa_dn` does the arithmetic, once, on one source's pixels:

**The offset applies to every valid DN, not only to bright ones.** ESA adds it across the whole
reflectance range precisely so that negative surface reflectance — routine over water and deep
shadow — is representable in an unsigned type. So the correction is `DN - 1000` for every DN from 1
upward, floored at the lowest VALID code, which is **1**.

Not zero: zero is the nodata code, so flooring a real dark observation there makes it
indistinguishable from no observation and every downstream mask drops it. Element 84's harmonised
COGs floor at 1 for the same reason, and reproducing them exactly is what makes a corrected raw copy
comparable with a harmonised one. DN 0 itself never reaches the arithmetic — the reader applies the
result only where the source was valid — so nodata survives as nodata.

**That the nodata code IS 0 is hard-coded, and that is an unchecked assumption.** `_NODATA = 0` is
a constant in `stac.py`, and the correction rests on it twice over: DN 0 is the value excluded from
the arithmetic, and the floor of 1 is what stops a corrected pixel from *looking* like nodata. The
real answer belongs to the catalogue — `odc` derives it per band from `raster:bands`, and the reader
has the resolved `RasterLoadParams` in hand when it corrects — but the driver is installed on any
collection whose config sets `requires_baseline_correction`, so nothing ties the constant to what
the collection actually declares. Under a marker of, say, 65535 every gap pixel would test as
valid: the correction shifts them to 64535, the fuser's `dst == fill_value` test stops recognising
them as empty, and the mosaic gains data-looking pixels 1000 below the nodata code where there was
no observation. Both Sentinel-2 providers declare 0 today, so this is latent rather than live — the
fix is to resolve the marker from `cfg` and refuse anything else, and it is owed.

The arithmetic widens to `int32` and casts back to the INPUT dtype, so nothing wraps and the store's
unsigned arrays are unaffected: the offset is negative and the floor is positive, so an unsigned
input stays representable. Adding a negative Python int to a `uint16` array raises under numpy 2,
which is what the widening is for.

**The floor still acts on resampled values, and that is a recorded limit.** `odc.stac.load` reads
and resamples in one step, so the wrapped reader sees pixels that have already been warped — and six
of the ten configured bands are natively 20 m on a 10 m grid. A pixel whose resampling kernel spans
the DN-1000 boundary is floored where Element 84, flooring each source pixel first, would not have
been. Fixing it means taking over the read-and-warp step, and unlike the per-source move it would
**rewrite pixels in every existing store**, so it is owed separately. Measured in
[ADR 021](../../../context_docs/decisions/021-correct-the-boa-offset-per-image.md) §6.

SCL is never corrected. It is not among the resolved reflectance asset keys, so it carries no
decision at all — a stronger exclusion than a band list, which could go stale.

### Post-load

These happen after `odc.stac.load` returns:

#### OPERA RTC-S1 Amplitude-to-dB Conversion

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

---

## Performance Optimizations

### Cropping to live windows (unconditional)

Ingest cost scales with the **extent it computes, not the land it keeps**: the mosaic
loads cover the whole ROI grid even where the mask is entirely ocean/out-of-footprint.
On a UTM zone that is mostly sea this is the difference between a graph that fits and one
that does not — the extreme cells hold a few live tiles out of many thousands, and across
the campaign's land zones roughly three-quarters of the compute would go to ocean.

Every load and write is restricted to the chunk-aligned windows that intersect the ROI
mask. **This is not optional and has no flag.** It was `crop_to_live_windows`, defaulting
OFF while the path was validated; the validation passed, the default was never flipped, and
the consequence of running without it is catastrophic rather than merely slower — on a zone
where land is 0.238% of the extent, uncropped is ~420x the array volume, which exhausts a
worker's disk and invalidates every measured campaign figure. No caller wanted it off, so
the parameter is gone rather than defaulted.

```text
zone / ROI extent (declared grid — UNCHANGED, the fill validates it)
┌──────────────────────────────────────────────┐
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │   · ocean / out-of-footprint:
│ · · · ┌────────────────────┐ ·  ·  ·  ·  ·   │     never loaded, never computed,
│ · · · │████████████████████│ ·  ·  ·  ·  ·   │     never written (reads back as
│ · · · └────────────────────┘ ·  ·  ·  ·  ·   │     fill — Zarr elides all-fill
│ · · · · · · ┌────────┐ ·  ·  ·  ·  ·  ·  ·   │     chunks anyway)
│ · · · · · · │████████│ ·  ·  ·  ·  ·  ·  ·   │
│ · · · · · · └────────┘ ·  ·  ·  ·  ·  ·  ·   │   █ live windows: chunk-aligned,
│ ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │     4096 px, derived below
└──────────────────────────────────────────────┘
```

Windows are derived in **two stages**, and the second is where most of the win is. Each cell
below is one 4096-px ingest chunk; `#` is live (the mask has land in it), `.` is ocean.

**Stage 1 — row bands.** One window per live chunk-row, spanning that row's first to last live
column. A row's interior gaps are included; nothing above or below it is. This is the
minimum-*area* answer.

**Stage 2 — grouping.** Vertically adjacent bands are unioned into taller windows. That
computes some dead chunks, and is still a large net win, because the two costs are nothing
alike:

```text
  chunk area                          a window boundary
  ──────────                          ─────────────────
  computed in PARALLEL across the     a BLOCKING region write: the whole
  whole fleet — adding chunks         fleet waits while one completes.
  widens the graph, which is           N windows = N serial stalls.
  what the fleet wants.
```

So the objective is not "least area" but "least `n_windows × price + area`", where the price is
one window expressed in the chunk area that costs the same (`WINDOW_COST_IN_CHUNKS`). That
price is *large*, so grouping pays almost whenever it is geometrically sane.

The merge runs **twice** — once over the run's live grid, once over each date's narrowed grid —
and both must use the same price. `windows_for_date` takes it as a parameter for that reason:
priced at the sequential default while the run used the overlapped rate, the per-date re-merge
would buy dead area back to save boundaries the write path has already made cheap, undoing the
calibration for every narrowed date.

```text
     live chunk grid        stage 1: row bands        stage 2: grouped
     c0 c1 c2 c3 c4 c5      c0 c1 c2 c3 c4 c5         c0 c1 c2 c3 c4 c5
r0    .  #  #  #  .  .       .  A  A  A  .  .          .  a  a  a  +  .
r1    .  #  #  #  #  .       .  B  B  B  B  .          .  a  a  a  a  .
r2    .  #  #  #  #  .       .  C  C  C  C  .          .  a  a  a  a  .
r3    .  .  #  #  #  .       .  .  D  D  D  .          .  +  a  a  a  .
r4    .  #  #  #  #  .       .  E  E  E  E  .          .  a  a  a  a  .
r5    .  .  .  .  .  .       .  .  .  .  .  .          .  .  .  .  .  .
r6    #  #  .  .  .  .       F  F  .  .  .  .          b  b  .  .  .  .
r7    #  #  .  .  .  .       G  G  .  .  .  .          b  b  .  .  .  .
r8    .  .  .  .  .  .       .  .  .  .  .  .          .  .  .  .  .  .

     # live chunk          7 windows, 7 stalls       2 windows, 2 stalls
     . never written       A..G one per live row     a = r0-r4 x c1-c4
                           r5/r8 have no live        b = r6-r7 x c0-c1
                           chunk, so no window       + = dead chunk the
                                                         union pulled in
```

Group `a` covers r0–r4 at the cost of two dead chunks (`+`), buying four fewer stalls. Whether
`b` should join it is likewise a cost question, not a shape one: that union would span
`r0–r7 × c0–c4`, adding 16 chunks to save one stall — worth it at the production price, which
is why real masks group harder than this small illustration suggests. Two things stop a group:
the price ceasing to justify the added area, and `MAX_TASKS_PER_WINDOW`, which bounds one
window's graph so memory and scheduler load stay inside a proven envelope. That cap is
expressed in TASKS, converted to a chunk area through `DEFAULT_TASKS_PER_CHUNK`, because
the scheduler dispatches tasks on a single-threaded event loop — past its throughput extra
area stops being cheap, which is how an unbounded objective over-merges into a saturated
scheduler and gets slower.

Grouping is solved **exactly**, by a dynamic program over consecutive bands rather than a
greedy rule — a heuristic bound on wasted area cannot express "extra area is nearly free", and
so under-merges precisely on the sparse ROIs where the absolute waste is trivial. Windows stay
chunk-aligned and mutually chunk-disjoint either way, which is what lets one session write a
whole date and commit once.

Effect on the campaign's zones, smallest to largest:

| zone live chunks | stage 1 windows | grouped | writes saved | added area |
|---|---|---|---|---|
| 4     | 2   | 1 | 2.0×  | +0%  |
| 22    | 5   | 1 | 5.0×  | +45% |
| 26    | 12  | 2 | 6.0×  | +50% |
| 2,415 | 197 | 3 | 65.7× | +20% |

Summed over all land zones the grouping cuts predicted per-date ingest cost by about **11×**,
landing every zone in 2–5 windows. Sparse ROIs group *harder* in relative terms, which is the
point: a large fraction of a tiny area is still a tiny area.

Calibration of the price, the campaign-wide table, and the cap sweep are in
`context_docs/design/ingest-live-tile-cropping.md`.

- **Windows** come from `live_windows.py`, from the boolean ROI mask that both
  `rasterize_roi_zarr` and `export_zone_roi` write. The mask is coarsened to the ingest chunk
  grid — normally from its chunk keys in one listing, else by scanning one chunk block at a
  time (~16 MB peak, no Dask) — then row-banded and grouped as above.
- **Writes** go through `storage.write_day_windows`: a missing store is seeded all-fill
  with an **empty** time axis (schema only — creation cost independent of extent), then
  each date appends its time slot atomically WITH its windows in **one commit**:

```text
per passing date (one writable session ── one commit)
   ├─ append time slot            (metadata-only resize; duplicate date = loud error)
   ├─ to_icechunk(region=window₁) ┐  pixels flow from the Dask workers that
   ├─ to_icechunk(region=window₂) │  computed them — never materialised on
   ├─ ...                         ┘  the flow runner
   ├─ merge attrs                 (baselines ∪, doy ++, last_appended)
   └─ commit                      (crash before here ⇒ nothing visible; retry is clean)
```

  The empty-axis seed matters: the time axis only ever contains dates whose pixels
  committed, which is what keeps `get_existing_dates` (the STAC dedupe),
  `check_time_window_coverage`, and the empty-timestep prunes truthful.
- **The retry must not retry a second writer** — the one exception to "a failed write
  commits nothing, so retrying is safe". One store has exactly one writer: these commits
  pass no `rebase_with`, so a concurrent commit is *refused* rather than merged
  (`icechunk.ConflictError`), and a date the other writer reached first is refused by the
  append guard (`DuplicateDateError`). A retry re-opens the session from the tip that
  writer moved, which turns the refusal into a success and lets two writers interleave
  dates onto one axis — the outcome the no-rebase choice exists to prevent. Both errors are
  excluded by type in `storage.zarr_store.store_write_retrying`, the single policy all
  three write sites use (S1 per-date, S2 per-date, S2 per-batch). It is shared because it
  was not: each site had built its own `Retrying`, and the exclusion was missing from all
  three at once.
- **Polarisation is filtered SERVER-SIDE, and that is a cost fix rather than a correctness
  one.** Ingest needs dual-pol VV+VH, so a granule whose `POLARIZATION` lacks VV was always
  rejected client-side — but only after being fetched and parsed. The query now carries
  `attribute[]=string,POLARIZATION,VV` alongside the orbit filter, which discards nothing
  reachable (CMR matches a multi-valued attribute if ANY value matches, so VV admits every
  VV+VH granule) and stops us paying to page the rest. Measured: the all-Greenland zone 23N
  went from 7.9 s paging 138,276 rejected granules to 0.6 s returning none, and mixed zone 25N
  from 11.6 s to 2.7 s returning the SAME 2,799 usable items — the unchanged count is what
  makes it safe. The client-side check is now a safety net, and its warning means a catalogue
  inconsistency (metadata advertising VV, bands not published) rather than a regional fact.
  **Cross-pol granules are NOT EW-mode**: that label was a guess and was wrong — the Greenland
  granules report `BEAM_MODE=IW`, and a `BEAM_MODE=EW` query over that region returns nothing.

- **A zone can have NO usable radar, and that is a finding rather than a failure.** Over ice
  Sentinel-1 images in Extra Wide swath with HH/HV polarisation, and the OPERA query discards
  anything that is not dual-pol VV+VH — so an ROI whose land is ice has a catalogue full of
  granules and not one the ingest can use. Zone 23N (Greenland) returns ~45,000 ascending and
  ~138,000 descending granules for 2021 and **zero** usable items. Requiring a SAR store there
  failed the cell permanently, so `"both"` resolves to `S1_ORBIT_NONE`, which activates no orbit
  and leaves the coverage gate checking reflectance alone.

  **Permissive by default, refusable on demand.** A global product cannot reject terrain that is
  radar-free as a matter of geography, so `resolve_s1_orbit`'s `allow_none` defaults to True for
  every caller. A single run over terrain known to be imaged is the opposite case — there an
  absent store means something upstream broke, and resolving to `none` would embed without radar
  and hide it — so those callers pass the flows' `require_s1`, which reaches the resolver as
  `allow_none=False`. An operator who names one orbit is never downgraded either.

  **The ingest's per-orbit item count is the authority on which case it is.** It has just queried
  both orbits, so `items_seen=0` means the source offers nothing here, which is terrain rather
  than a gap. A consumer reading a finished mosaic cannot distinguish the two, which is why its
  warning names the mosaic and points at that count.

  Accepting a radar-free ROI necessarily means embedding S2-only pixels, since every pixel there
  has zero S1 observations. `InferenceConfig` derives `allow_s2_only` for that case rather than
  asking the caller, because the alternative is a fill that writes nothing and reports success.

- **Reads retry too, and did not used to.** `roi_processing.source_read_retrying` wraps the
  point where a date's graph is first *computed*. The two sensors reach that point
  differently, which is why only one of them needed the fix: S1's read happens inside its
  write's `compute()`, already covered by the write retry, while S2's fires earlier in its
  coverage gate and so sat outside every retry. One transient failure reading one granule
  therefore propagated out of S2's per-date loop and failed the whole zone-year, discarding
  the months the run had already committed. Scoped per date on purpose — a retry at the task
  level would re-run the entire multi-day loop, which is why `tasks/ingest.py` refuses
  `@task(retries=...)`. Unlike the write policy it is **not** narrowed by exception type:
  reads fail through rasterio, GDAL/CPL, botocore and bare socket timeouts, a read is
  idempotent, and enumerating those surfaces only risks a new transient class becoming fatal.
- **A failed date must say which date, and on which ROI.** Per-date telemetry is emitted
  *after* a date commits, so the furthest date in a log is the last one that WORKED — a
  failure leaves no trace of what was being attempted, and the log reads as progress right up
  to the point of death. `roi_processing.read_failure_context` closes that on both sensors'
  per-date paths, emitting `READ FAILED roi=… date=… items=… first=…` with the traceback.
  The `roi=` field is what makes it attributable: the exception is raised on a Dask worker
  whose log stream is an ECS task id, so without it the same error text appears for every
  zone in the fleet and belongs to none of them. The traceback is what recovers rasterio's
  cause — it reports `Read failed. See previous exception for details.` and that previous
  exception is GDAL's, discarded unless the chain is logged.
- **Each date narrows further, to the land its own imagery reaches.** A run's windows
  say where the ROI has land, and are the same on every date; a single date is not, because
  an optical satellite images a fraction of a wide ROI per pass. Windows a date does not
  reach still built tasks that ran, found nothing, and wrote nothing. `windows_for_date`
  intersects the run's windows with the footprint of that date's own STAC items
  (reprojected onto the ingest grid, padded outward by one cell so a curved reprojection
  cannot under-cover), then re-bands and re-groups the remainder. This cannot change what
  a mosaic holds — it only removes work whose result was already discarded — but it shrinks
  both the graph and the count of serial region writes. When the footprint cannot be
  determined the run's full window list is returned unchanged, so the conservative
  behaviour is the fallback rather than something a caller must opt into.

```text
   run windows (where the ROI has land)   one date's items      that date writes
     c0 c1 c2 c3 c4                        c0 c1 c2 c3 c4        c0 c1 c2 c3 c4
r0    a  a  a  a  .                         .  F  F  .  .    r0   .  a  a  .  .
r1    a  a  a  a  .          ∩              .  F  F  F  .  = r1   .  a  a  a  .
r2    a  a  a  a  .                         .  .  .  .  .    r2   .  .  .  .  .
r3    b  b  .  .  .                         .  .  .  .  .    r3   .  .  .  .  .

     2 windows, every date                F = the swath           1 window; window b
                                            this date covers        is skipped entirely
```

- **The declared grid stays full-extent**, so the zone fill's exact-grid validation is
  unaffected — Zarr/Icechunk arrays are sparse and unwritten chunks read back as fill.
- **The SCL coverage phase is cropped too**: its reduce runs over the windows only
  (identical total — the mask is False outside them) and the validity mask stays lazy,
  so no full-extent array is ever persisted.
- **Worker sizing follows**: with cropping on, `ingest-zone-year` sizes `max_workers`
  from the cell's live-chunk count (0.5 workers/chunk, clamped) instead of granting a
  4-tile zone the same fleet as a dense one.

Unconditional at every layer — see "Cropping to live windows (unconditional)" above for
why the flag was removed rather than defaulted. The full-extent write path it used to
select is gone from both modules, along with every branch that tested for it. S1 and S2
share the mechanism; the one difference is that S1's multi-date batches loop per date
(non-contiguous dates cannot share a region write), each date keeping its own atomic
commit and retry scope.

The zone fill's grid validation checks the declared grid COMPLETELY, not just its
corners: matching length, CRS and endpoints still admit a reordered or non-affine
interior, and inference writes positionally, so such a mosaic would publish real pixels
at wrong coordinates in silence. Uniform 10 m spacing is asserted on both axes. Nothing
this ingest produces can fail that — odc builds every load against the zone geobox — but
a fill run with `ingest=False` accepts a mosaic staged by hand, and that path is
supported.

### Background: how Dask task graphs consume scheduler RAM

"Dask is lazy" means workers don't read data until `.compute()` is called. It does *not*
mean the scheduler avoids work. Before the first worker task executes, `dask.distributed`
must expand the full `HighLevelGraph` (HLG) — the compact Python-side description of the
computation — into a flat dictionary of `TaskState` objects in the scheduler process. Each
`TaskState` holds the function, arguments, and dependency set for one task. The cost:

```text
    scheduler RAM used ≈ n_tasks × 1.5 KB
```

This is fully predictable and independent of data size. A graph with 1 million tasks
consumes ~1.5 GB of scheduler RAM before any worker reads a single byte.

The HLG itself is compact — it stores *layer dicts* rather than expanded objects. The
expansion to TaskStates happens only when the graph is submitted to the scheduler:

```text
  Python process (builds the HLG)          Dask distributed scheduler process
  ─────────────────────────────────         ─────────────────────────────────────
  Layer "zarr_read"                         TaskState("zarr_read",(t=0,y= 0,x= 0))
    { (t=0,y=0,x=0): read_fn, ... }         TaskState("zarr_read",(t=0,y= 0,x= 1))
  Layer "baseline_corr"                     TaskState("zarr_read",(t=0,y= 0,x= 2))
    { (t=0,y=0,x=0): corr_fn, ... }         ...
  Layer "roi_mask"                          TaskState("baseline_corr",(t=0,y=0,x=0))
    { (t=0,y=0,x=0): mask_fn, ... }         ...
  Layer "zarr_write"
    { (t=0,y=0,x=0): write_fn, ... }       one TaskState object per task,
                                            all held in scheduler RAM simultaneously
  4 compact Python dicts ≈ tens of MB      n_tasks × 1.5 KB of scheduler RAM
```

#### How task count multiplies: operations × chunk dimensions

Each Dask operation (read, transform, write) adds a new layer. Each layer has one task per
combination of *chunk coordinates* across all chunked dimensions. Ingest writes 4096×4096 px
storage chunks (`INGEST_CHUNK_SIZE`), deliberately larger than the 2048×2048 inference
read-tile size, precisely to keep this task count down. For S2 ingestion over a large ROI
(e.g. cornbelt scale: ~38×25 grid of 4096×4096 px spatial chunks), the S2 flow processes one
date at a time, so the time dimension is always 1:

```text
  Dimensions per single S2 date (ingest_s2_roi_reflectance):
    spatial chunks:     ~950  (38×25 grid of 4096×4096 px)
    dates:                 1  (one date per loop iteration)
    band variables:       10  (each S2 band is a separate xarray variable)

  Operation            tasks per layer                          notes
  ────────────────────────────────────────────────────────────────────────
  odc.stac read       950 × 1 × 10 =   9,500   one task per (chunk, band var)
  baseline corr.      950 × 1 × 10 =   9,500
  ROI mask            950 × 1 × 10 =   9,500
  zarr write          950 × 1 × 10 =   9,500
                                    ─────────
        per-date total:                38,000 tasks ≈ 0.06 GB scheduler RAM   ✓

  Full year (all ~100 S2 dates in one graph, hypothetically):
  950 × 100 × 10 × 4 = 3,800,000 tasks ≈ 5.6 GB scheduler RAM                ✗ OOM
```

Had ingest reused the 2048×2048 inference tile size, every count above would be 4× higher —
that 4× on the satellite-ingest graph is the reason storage and inference chunk sizes are
decoupled.

The two flows keep task count bounded via different mechanisms — per-date iteration for S2,
time-windowed batching for S1 — described in the sections below. The same scheduler-RAM
discipline reappears in inference assembly; see
[`inference/README.md`](../inference/README.md#three-layer-chunk-anatomy) for the
ChunkSpec-vs-sub-chunk decoupling that makes assembly survive on the same budget.

### S2: per-date iteration (task graph management)

`ingest_s2_roi_reflectance` queries STAC for the full date range upfront, groups items by
**local solar day** via `group_items_by_date`, then processes one day at a time in a Python loop.

The grouping key is load-bearing and must match the loader's. `odc.stac.load(groupby="solar_day")`
shifts every timestamp by ONE longitude — its geobox extent's centroid in WGS84, truncated to whole
hours — and groups on the result. Grouping here by UTC calendar date instead lets the two disagree,
and a group we believe is one day then loads as TWO time slices against a cloud mask reduced to one:

```text
   UTC:      ... 23:00 | 00:00  01:00 ...      ONE UTC date
   solar:        day N |  day N+1              TWO solar days   (at a +10 h offset)
                       ^ far-eastern zones image right here
```

### Solar days versus UTC queries (`solar_days.py`)

**One rule, and everything else follows: the solar offset is applied exactly once, by
`normalize_to_solar_day`, at the catalogue chokepoint. After that an item's `datetime` IS
its solar day (at noon UTC), and every date derivation downstream is a plain
`strftime("%Y-%m-%d")` with no offset.**

That rule exists because the alternative was tried and drifted. The offset used to be
applied independently at six sites, and two of them disagreed with the rest: the
cloud pre-sort and the baseline map both keyed on the UTC date while the loader grouped by
solar day. On a day straddling UTC midnight that meant the group was not sorted as intended,
and half its baseline entries never matched. A seventh
application would have been one more chance to disagree; applying it once cannot.

**Every date modality in the pipeline, and which convention it uses.** The whole point of
one chokepoint is that this table has no exceptions:

| where a date lives | form | convention |
|---|---|---|
| catalogue item, before the chokepoint | real acquisition instant | UTC — the only place a raw timestamp exists |
| catalogue item, after `normalize_to_solar_day` | **noon UTC of its solar day** | solar |
| loaded array coordinate (`odc.stac.load` output) | inherits the item stamp | solar (noon) |
| written store coordinate | `np.datetime64(solar_day)` | solar (midnight) |
| `existing_dates`, `written_dates`, baseline keys, `assessed_window` | `YYYY-MM-DD` strings | solar |
| chunk `own_start` / `own_end` | `YYYY-MM-DD` strings | solar |
| chunk `query_start` / `query_end` | `YYYY-MM-DD` strings | **UTC** — the only thing a catalogue understands |

The two timestamp forms differ (noon in flight, midnight in the store) and never meet as
numbers: everything that crosses that boundary compares `YYYY-MM-DD` strings. The one row
that is deliberately UTC is the query bound, because a catalogue has no other vocabulary —
which is exactly why ownership, not the query bound, decides what gets written.

**Three things enforce this rather than describing it:**

- `solar_day_of` **raises** on an item that is not stamped noon UTC. A raw item's UTC date
  is a plausible-looking wrong answer — right at central longitudes, wrong only where the
  offset crosses midnight — so a path that skips the chokepoint would look correct
  everywhere anyone checks interactively.
- An **architecture rule** (`solar-offset-applied-only-in-solar-days`) fails CI if
  `solar_day_offset_seconds` is called outside this module. One application is the
  invariant; a second is a bug in the opposite direction.
- Every consumption point **re-normalises defensively** rather than trusting call order,
  because every supplier (`query_fn`, `item_provider_fn`) is injectable.

Two consequences worth knowing:

- **`normalize_to_solar_day` is idempotent**, so the consumption points (`stream_stac_months`,
  `has_new_stac_dates`) call it defensively rather than trusting whoever supplied their
  items — `query_fn` and `item_provider_fn` are both injectable.
- **Noon, not midnight.** The canonical timestamp has to read as the solar day both directly
  and after `odc.stac.load` groups on it. Noon has half a day of margin either side, so
  neither reading crosses midnight for any offset the grid produces (±11 h at the zones
  nearest the antimeridian).

The same disagreement decides how every query window is bounded, on every path, which is why
it lives in one module — `ingest/solar_days.py` — instead of being re-derived per sensor.

A catalogue query is bounded in **UTC**. An ingest window, and every chunk of it, is a range of
**solar** days. Wherever a zone's offset crosses UTC midnight the two do not line up, and both
ways of ignoring that have been in this codebase:

- **Query the chunk's own range and write whatever comes back.** A solar day straddling the cut
  is split: the earlier chunk writes it from its half, and the later chunk's half is then dropped
  as an already-written date. The day lands looking complete and is missing acquisitions. The S1
  batch loop did this at *every* batch boundary.
- **Pad the query, but clamp the pad to the window.** The padding then vanishes at the window's
  own two edges, so the first and last solar day of a zone-year lose whatever imagery was dated
  the adjacent UTC day. The S2 month slicing did this.

Both are silent: `assessed_window` still covers the days, and the month coverage gate still
passes. Only a comparison against the catalogue would reveal them.

The mechanism is one idea. A chunk **owns** a range of solar days and **queries** a wider range
of UTC dates:

```text
                 own:            2024-01-31 .............. 2024-02-29
                 query:   2024-01-30 ........................... 2024-03-01
                          └─ pad ─┘                              └─ pad ─┘

  a solar day landing on the cut is drawn from BOTH sides by the batch that owns it,
  and is owned by exactly one batch, so it is written once and written whole
```

Owned ranges tile the window exactly, so nothing is processed twice and nothing outside the
window is written — **ownership is what guarantees that, never the query bound**, which cannot
see solar days at all. The pads are deliberately *not* clamped to the window, because a solar day
owned at its very edge still draws on the UTC day beyond it. One day of padding is always enough:
the offset is a whole number of hours in `[-12, +12]`, so solar day `D` lies inside UTC
`[D-1, D+1]`.

`owned_items` is applied **between the query and the loader**, never after loading. Filtered
there, the loader builds a group only for an owned day and that group holds every image of it.
Filtered afterwards, a straddling day has already been split into two partial groups and what is
needed to rejoin them is gone.

Three chunkings share it today, and a new provider adds a fourth by writing a span producer and
nothing else:

| producer | used by | chunk |
|---|---|---|
| `month_ranges` | S2, streaming (default) | one calendar month, to bound item retention |
| `fixed_day_ranges` | S1 | `batch_days` solar days, to bound Dask graph size |
| `whole_window_range` | S2, `stream_stac_monthly=False` | the window in one query |

This rests on our offset arithmetic agreeing exactly with the loader's — both truncate
longitude over fifteen to whole hours. If they diverged, an image could be filtered out as
another chunk's while the loader would have grouped it into this one, dropping it from the run
entirely. `solar_day_offset_seconds` is the single definition, and it is the reason
`solar_grouping_longitude` prefers the geobox centroid over a bbox midpoint.

That is why `group_items_by_date` takes a `mid_longitude`, and why the pre-sort uses the same key —
the sort carries the fusion contract (clearest tile FIRST within a group), so sorting on
a different notion of "day" than the grouping would silently let a cloudier pixel win. Central
longitudes image far from UTC midnight and are unaffected, which is what kept this latent.
Each iteration builds a single-date Dask graph, calls `odc.stac.load` for that day, filters
coverage, and writes before moving to the next date:

```text
Full year (don't build at once):
┌──────────────────────────── 365 days ────────────────────────────────┐
│ tiles × dates × bands = O(millions of tasks) → scheduler OOM         │
└──────────────────────────────────────────────────────────────────────┘

Per-date iteration (what ingest_s2_roi_reflectance actually does):
 2024-03-01    2024-03-06    2024-03-11    ...
┌────────────┐ ┌────────────┐ ┌────────────┐
│ build      │ │ build      │ │ build      │
│ SCL check  │ │ SCL check  │ │ SCL check  │
│ compute    │ │ compute    │ │ compute    │
│ write      │ │ write      │ │ write      │
│ discard ◄──┼─┼── graph freed after each date
└────────────┘ └────────────┘ └────────────┘
```

Each single-date graph is small: `spatial_chunks × bands` tasks, with no date dimension to
multiply through. The per-date overhead (one Python loop iteration, one Zarr append) is
negligible compared to the Dask compute time for a large spatial ROI.

### Overlapping a date's window writes (`overlap_window_writes`)

A date is written as several chunk-disjoint windows. The
obvious implementation writes them one at a time, and that turns out to dominate the cost of
a date: each window's compute runs to completion before the next begins, so the date costs
the **sum** of the windows' critical paths while the fleet works on one window and idles
through the rest of each.

```text
Sequential windows — the fleet sees one window at a time:
 │◄─ window 1 ─►│◄─ window 2 ─►│◄─ window 3 ─►│◄ w4 ►│◄─ window 5 ─►│
 └─ the date costs the SUM of these, and most slots idle within each ─┘

Overlapped (overlap_window_writes, the default) — one graph, one commit:
 │◄─ window 1 ─►│
 │◄─ window 2 ──►│     all submitted together, so the fleet packs them and
 │◄─ window 3 ─►│      the date costs roughly the LONGEST window plus
 │◄ w4 ►│              whatever the total work itself requires
 │◄─ window 5 ──►│
 └── one merge, one commit for the date (contract unchanged) ──┘
```

Mechanism: icechunk's dask path already forks a session, stores lazily, and merges
changesets; writing per window merely runs that whole sequence once per window. Overlapping
lifts it one level — fork once, collect every window's lazy stored arrays, run one merge
reduction — so all the windows' loads, masks and chunk writes occupy a single graph.

The resulting store is identical either way, and the reason is the windows'
chunk-disjointness: that is what makes the merged changesets conflict-free, and it is the
same property that lets a date commit exactly once. Should icechunk's internals move, the
write falls back to the sequential loop with a warning rather than failing.

Default **on** for both S2 and S1. `write_day_windows` itself still defaults to the
sequential path: a storage-layer default should not decide write strategy for its callers,
so each ingest path opts in explicitly.

S1 was measured before it opted in. The gain is **2.4–3.9×** on per-date write time and does
not vary with either quantity we can vary: not with window count (23, 9 and 7 windows gave
2.79×, 2.86× and 2.40×) and not with fleet width (30 and 60 workers gave 3.67× and 3.85×,
inside the noise floor). **Why it is that size is not explained** — three accounts have been
proposed for the family of S1 write effects and all three were refuted by their own
predictions. Rely on the measured range; do not model it, and do not extrapolate far outside
the widths measured. The campaign record's §4.9 has each refutation.

### Which day a slice is called (and why it is not an item's timestamp)

A mosaic slice represents one **solar day**, and it is labelled with that day — taken from the
grouping key, not from the loaded dataset's own time coordinate.

That distinction is load-bearing. `odc.stac.load` stamps each group from `group[0]`, which ties the
label to whichever item the sort left first. A label taken that way can disagree with the solar day
wherever the solar offset crosses UTC midnight, and two consecutive solar days then collide on the
time axis — the batched write rejects the dates as not strictly increasing, the unbatched write
rejects the second as a duplicate time slot. Taking the day from the grouping key removes the
dependence on order entirely.

Taking the day from the grouping makes three things true by construction rather than by care:
labels are unique per slice, monotonic across them (consecutive solar days differ by exactly one
day, so the batched write needs no sorting), and stable against the catalogue revising its cloud
estimates. The store's axis is day-granular either way, so this only decides WHICH day — pixels,
ordering and which tile wins are untouched, and at mid longitudes the value is unchanged because
the solar day IS the UTC date there.

### Recording the window an ingest examined

Both paths write `assessed_window` — the date range processed in full — onto the store. A month
absent from the time axis but wholly inside that range was **examined and found to hold nothing
reachable**, which is a finding; a month outside it is a gap. Without the record those are
indistinguishable, and the coverage gate has to fail on both.

The attribute belongs on the repo the gate opens — `reflectance.zarr` or `sar_<orbit>.zarr` —
not on the mosaic directory that contains them.

**It is written whenever the store exists, not only when the run wrote a date.** The case that
needs it most writes nothing. A run interrupted between its last date commit and this record
leaves every date present and the attribute absent; the retry then dedupes all of those dates
away, takes the zero-write path, and — keyed on what *this* invocation wrote — skipped the record
again. Every retry after it did the same, so a legitimately empty month stayed permanently
indistinguishable from a gap and the zone-year could never complete. Keyed on the store, the
resume repairs the attribute. The extra existence probe runs only when nothing was written, so a
normal run pays nothing for it. A genuinely absent store is still left alone: there is no repo to
annotate, and that case was never ambiguous — no store means the orbit is absent and callers
downgrade.

Every uncertain path is strict: an absent, malformed or unparseable attribute excuses nothing,
and a partially-covered month stays an error because it could hide unexamined days. Failing to
write the attribute is logged, never raised — the gate simply falls back to requiring every
month. `assessed_empty_dates` is recorded alongside for observability, separating "sparse region"
from "the footprints are wrong".

### Narrowing a date's windows, and skipping dates that reach none

A run's live windows describe where the ROI has land, so they are the same on every date. One
date is not: a satellite images a fraction of a wide ROI per pass, so most of those windows hold
nothing for a given date. `windows_for_date` removes them. Tasks over them would run, find no
data and write nothing — an all-fill chunk is never stored — so this cannot change what a mosaic
contains. Both sensors now do it (`narrow_windows_per_date` on S1, always on S2): six times fewer
windows per date on the S1 zones measured, worth 7–20% of per-date wall clock.

**A date whose imagery reaches NO live window is skipped entirely**, on both paths and
unconditionally. Writing it builds a full graph to store nothing. On S1 this is not a rare case:
one zone skipped 13 of 58 dates, and some zones have an orbit that reaches land on *no* date of
the year. Skipping those means no store is created, which is what lets `resolve_s1_orbit`
correctly downgrade to single-orbit instead of publishing a store full of fill that inference
would read as real signal.

**The safety rule, and it is the whole design.** A footprint that is too LARGE only costs
computed area that would have been discarded; one that is too SMALL drops imagery and nothing
downstream notices. So every uncertain path widens rather than narrows: an unreadable footprint
returns the full window set, and on S1 a time slice that cannot be matched to its items writes
everything. "Reaches nothing" and "we cannot tell" are separate branches — only the first skips.

S1's match is on an **exact timestamp** rather than a date string, because odc sets a slice's
time coordinate to its group's earliest item timestamp. Keying by solar day instead would
disagree with the loader wherever the offset crosses UTC midnight.

### Pipelining a date's preparation (`pipeline_dates`)

A date's wall clock splits into **preparation** — building the load graph, running the
coverage gate, narrowing the footprint, constructing the masks — and the **write**.
Preparation is part client-side CPU (graph building, genuinely independent of fleet width)
and part cluster compute (the coverage gate reads SCL on the workers, so it scales with
width like any other fleet work). Only the client-side part is serial residual that a wider
fleet cannot shrink.

**The overlap's payoff is therefore not symmetric between the two parts.** Hiding the
client-side part behind the write is free. Hiding the gate is not: it is fleet work, so on a
fleet the write already saturates it competes for the same slots and is additive regardless
of scheduling order — and task priorities cannot change that, since they reorder a queue
without creating capacity. The overlap pays off in proportion to the spare capacity the
write leaves, which makes it **more** valuable on narrow fleets than on wide ones.

`pipeline_dates` prepares date N+1 on one background thread while date N is being written
(`ingest/_pipeline.py`). What stays serial is the write: icechunk commits are sequential on
a branch, one commit per date is the contract, and the store therefore has exactly one
writer either way. Preparation is required to be **side-effect-free** — it may touch nothing
but the dataset it hands back — which is what makes the two modes produce identical stores
(pinned by a parity test that includes a date failing the coverage gate mid-run).

Depth is 1 and that is intrinsic rather than tuned: preparation is a small fraction of a
write, so buffering more dates would hold graphs in memory to hide nothing. The pipeline
lives inside one `_drive` call, so it drains naturally at each streamed month boundary,
leaving one unhidden preparation per month.

Each written date logs a `Pipeline date=…: prepare=… hidden=… stall=…` line in both modes.
`stall` is the preparation the write could not cover and is the health metric: near zero
when preparation hides fully, and rising toward the whole preparation when the gate is
starved behind the write's own tasks. Serially every date stalls for its full preparation
and hides none of it, so the two modes are comparable from one line.

> **`hidden` is not a saving, and reading it as one overstates the benefit several-fold.**
> When pipelined, `prepare` is wall time on a background thread that spans the whole
> concurrent write, so it inflates with contention: the same preparation reports a small
> number serially and a large one pipelined, because it is being *queued behind the write*,
> not because more of it was avoided. The ceiling on what the overlap can save is what
> preparation costs when nothing competes with it — i.e. the serial mode's own
> `prepare` — so any A/B must take its expected saving from the **control** arm and treat
> `hidden` as a diagnostic of contention only.

Default **off**, and the flag threads from the outer flow through the task shell to the
domain function. S1 has no coverage gate and a different batch loop; it is deliberately
untouched.

### Batching dates into one compute (`batch_dates`)

**Sized per ROI, and NOT a straight win.** `batch_dates=None` (the default) derives the batch size
from the ROI's covered window area via `config.ingest.auto_batch_dates`; an explicit integer forces
one, which is how an A/B arm is pinned. Batching helps small ROIs, is roughly neutral on large
ones, and **costs about 29% on mid-sized ones** — so one global value is wrong for part of the
range.

The arithmetic behind that shape:

```
per-date wall clock  ≈  max( W , P )  +  commit / k

    W = the batch's write, per date          k = dates fused into one graph
    P = the preparation running alongside it, per date
```

Batching divides the commit by `k` and does nothing else. It **cannot** make the write faster,
because the fleet is already the constraint — so commit amortisation is its only gain, and it
LOSES wherever the larger write graph crowds out the preparation overlapping it. On a mid-sized
ROI, preparation at `k=1` already fitted exactly inside the write with zero stall, and batching
disturbed an already-optimal overlap.

So batching pays only where the fleet has idle capacity for the extra work to fill. The threshold
sits at the top of the range where that was *measured* to hold, not at an estimated crossover, so
widening it means measuring an ROI in between. Being denominated in covered window area also couples
it to the merge exchange rate above: a finer merge covers less area, so ROIs drift below the
threshold and more of them batch. Recalibrate against runs, never an offline sweep at a different
merge cost. Figures in `context_docs/design/ingest_optimization_campaign_2026_07.md` §3.16.

When it is on, k consecutive PASSING dates compute as ONE graph: their work packs the fleet
together, one date's straggling reads backfill with another's writes, and the drain tail and commit
gap are paid once per batch.

The commit unit becomes the batch, and that is forced rather than chosen: every date's
append resizes the time axis, so per-date sessions forked from one snapshot would
conflict on array metadata even though their chunk data is disjoint
(`storage.zarr_store.write_days_windows`). A mid-batch failure therefore commits none
of the batch's dates, and a retry — or a fresh run — re-ingests exactly the uncommitted
dates; `get_existing_dates` sees only committed dates either way. Stores are
byte-identical to the per-date path (pinned by a parity test whose gate-failing date
sits mid-batch).

Skipped dates do not occupy batch slots, so batches stay full exactly where the gate
filters most; the trailing partial batch flushes at each streamed month boundary. In
batched mode the per-date `Stage timings` line is replaced by one `Batch timings` line
per batch (build/gate are sums of real per-date values; the write is one shared compute
and has no per-date decomposition). Default 1 — the one-commit-per-date path — is unchanged.

**Composing with `pipeline_dates`.** The two are complementary and compose: batching removes
the fleet idleness *within* a date's write, pipelining removes the serial preparation
*between* writes. Composed, the pipeline's look-ahead is sized to the batch rather than to
one date — a batch's write is one long consume, so a depth-1 buffer would hide only one
date's preparation out of k. Preparation stays single-threaded at any depth, so its
side-effect-free contract is unchanged; the extra cost is that up to k prepared dates are
buffered while k more are written. `Batch timings` reports `prepare`/`hidden`/`stall` for the
batch so the two modes are comparable from one log line, with the same caveat as the per-date
line: `hidden` is bounded by the SERIAL preparation cost, never by a pipelined `prepare`
figure that contention has inflated.

### S1: time-windowed batching (task graph management)

`ingest_s1_roi_sar` uses a different approach: it splits the full date range into
`batch_days`-wide windows (default 30) and runs one `build → compute → write → discard`
cycle per window, which bounds how large any one task graph gets.

```text
Batched approach (batch_days=30, ingest_s1_roi_sar only):
 Jan 1–30        Feb 1–28        Mar 1–30       ...
┌────────────┐  ┌────────────┐  ┌────────────┐
│ build      │  │ build      │  │ build      │
│ compute    │  │ compute    │  │ compute    │
│ write      │  │ write      │  │ write      │
│ discard ◄──┼──┼── graph freed
└────────────┘  └────────────┘  └────────────┘
```

A batch boundary is **not** a credential checkpoint, and treating it as one is unsafe: the STS
credential's roughly one-hour life is unrelated to how long a batch takes, so a batch that outruns
it cannot renew at its own boundary. Renewal is owned by a timer — see "Renewal runs on a timer"
below.

`batch_days` is a parameter on `ingest_s1_roi_sar`. It is not present on the S2 flow. The
formula in the background section above applies to estimating how many tasks a given
window width will produce; the 30-day default keeps each batch well within the scheduler's
RAM budget at cornbelt scale.

Batch windows are inclusive on both ends and do not overlap: each batch spans `batch_days`
calendar days, and the loop advances `batch_start` to the day *after* `batch_end`. Because
CMR/STAC also treat their query end date as inclusive, each day is queried by exactly one
batch — a batch boundary that landed on the same day as the next batch's start would page
that day twice, wasteful at CONUS scale where each day is many pages of bursts. The overall
`[start_date, end_date]` range is inclusive of `end_date`.

### Lazy evaluation throughout

`odc.stac.load` returns a Dask-backed xarray Dataset with no raster data read yet. The BOA offset
is applied inside the read itself; the post-load transformations (dB conversion, ROI masking) chain
further Dask operations without computing. Data is read and written in a single Dask graph
execution triggered by the Zarr write step.

### Coverage pre-filtering before compute

`filter_low_coverage_dates` eagerly computes only the per-date valid pixel counts (one
scalar per time step) from the quality band (SCL for S2, VV for S1) to decide which dates to
keep. All spectral bands remain lazy. Dropping low-coverage dates before `.compute()` avoids
reading band data for cloud-covered or off-ROI scenes.

### Date deduplication before loading

`_filter_existing_dates` removes items whose dates are already in the Zarr store before
calling `odc.stac.load`. This avoids building Dask task graphs for data that will be
discarded, and prevents unnecessary COG reads from S3.

The filter must be keyed the same way the store was written. Both S1 and S2 load with
`groupby="solar_day"`, so their time axes hold solar days — and an acquisition's UTC date
is not its solar day wherever the offset crosses midnight (the far-eastern and far-western
zones). Callers that group by solar day pass `mid_longitude` down through `ingest_tile` /
`query_stac_items`; matching UTC dates against a solar-day set instead would filter only the
half of a committed group that falls on the near side of midnight, and the surviving half
would reload, regroup onto the day already present, and be written a second time.

The filter is an optimisation, not the guarantee. On S1 the queries are built one batch
ahead of the writes, so the set they filter against is a snapshot frozen before the run
began; the write loop tracks what it has actually written and is the authority, on the
cropped and full-extent branches alike.

### Choosing between duplicate copies of a tile-date

A catalogue indexes the same tile-date more than once whenever a granule is reprocessed, and
sometimes from more than one region. `duplicates.py` reduces each tile-date to one copy per
*acquisition* before the loader sees it, because `odc.stac.load` fuses a solar-day group: two
copies of one acquisition would be blended into one pixel stack at two different processing
baselines, and the baseline recorded on the store would match neither.

Preference is expressed as **one sort key** (`_preference_key`), and the property that makes it
work is that it is **context-free**: no term means "best in my group", so the same tuple orders two
copies of one acquisition and two copies from different ones. A term relative to the group's own
best baseline makes a cross-acquisition comparison meaningless and forces a second key alongside
this one, where a signal added to either is easily missed from the other. If you add a signal, add
it here and nowhere else.

The key reads these signals, in this order:

1. **Read-set completeness**, judged over the assets *this* load will request — the configured
   bands plus the caller's `extra_bands`, not a fixed list and not the broader pruning set, which
   keeps `scl` whether or not the call asks for it. First, because a copy missing one of them
   cannot deliver the tile-date at any baseline, and the generic paths have no recovery for it: a
   missing band is not one of the read failures the fallback ladder recognises, so an incomplete
   winner fails the acquisition outright.
2. **Whether the producer is decidable**, where it would change a pixel. A copy whose producer
   cannot be identified at all refuses its date at or above the correction threshold — and a
   refusal is not something the fallback ladder can step down on. A copy whose reflectance bands
   span a harmonised and a raw producer is *not* among them: each of its sources is decided on its
   own bucket, so it is corrected band by band rather than refused. Below the threshold the term is
   inert, because the producer changes no pixel there.
3. **Whether the copy demonstrably belongs to the acquisition it is ranked in.** A copy naming
   neither an observation nor an instant was attached to a cluster arbitrarily, so it must not
   displace one that says which pass it came from — a known member at an older baseline still
   represents that pass, while a possibly-unrelated newer one may duplicate another and drop this
   one's coverage.
4. **Whether the baseline is readable**, for producers whose correction depends on it, with
   unknown sorting last. An absent baseline is an absence of evidence, and such a copy refuses its
   whole date downstream, so an older reprocessing that can be corrected beats a newer one that
   cannot be processed at all. Above every term below it because a refusal has no recovery. An
   already-harmonised copy is exempt: its pixels need no correction whatever the baseline says.
5. **Processing baseline, descending.** The signal that carries data vintage. Ordered by value
   rather than by "is it the best", so every rung of the fallback ladder stays in descending
   baseline order. Collapsing the non-best baselines into one tier lets a read failure skip a
   04.00 copy and hand out a 03.00 one.
6. **Whether the copy owes an offset correction at all**, where the producer is an item's own
   property. **Below the baseline**, and there are two reasons to prefer a copy owing nothing —
   both about the PIXEL rather than about coverage. An already-harmonised copy had its floor
   applied by its producer *before* any resampling, where one we correct is floored *after*, so on
   the very dark population the two disagree; and a copy owing nothing cannot be wrong by the
   offset at all, while one we correct is right only if the bucket lists and the declared baseline
   are both honest. Those are quality claims, and a quality-versus-quality preference must not buy
   a better pixel with an older reprocessing — the same rule that keeps locality below the
   baseline. It outranks locality only because a pixel argument beats a cost one. Inert below the
   threshold, where no producer changes a pixel — which is most but NOT all ESA-hosted copies: zone
   01N in 2017 carries 15 at baseline 05.00, so the term does fire on real data. Also inert where
   the producer is the COLLECTION's answer: every copy then has the same producer, so a term that
   compares producers would discriminate on the threshold alone, which is the baseline term ranked
   above it, in the opposite direction.
7. **Locality, among equal baselines only.** A copy whose read assets all sit in a preferred
   bucket is cheaper to read, so it wins a baseline tie. Restricting locality to ties is what
   stops it buying cheaper egress with an older pixel, and it is inert where the baseline is
   unreadable, so it cannot decide a comparison the baseline could not enter. This is the
   distinction the two bucket lists exist for: harmonisation is a **pixel** claim and locality is a
   **cost** claim, so both sit below the baseline, harmonisation above locality. The lists name
   the same buckets today, but the key sets differ by `scl`, so a copy can be harmonised without
   being local and the terms are not interchangeable.
8. **`s2:sequence`, descending, then item id.** The id keeps the choice independent of catalogue
   response order, so a rerun cannot silently produce a different mosaic — and it makes the key a
   total order, so no comparison ever falls back to input order.

Two properties of that ordering are easy to get wrong and are held by tests:

- **Locality is judged over the read set, not over every asset.** A real Element 84 item
  carries its COG bands *and* the original JP2s, across two buckets. Requiring all of them
  disables the preference altogether. It is also judged over the *whole* read set: one local band among many remote
  ones is not locality, and an item exposing none of them is remote, because absence of
  evidence is not evidence of locality.
- **An unreadable baseline sorts LAST, and makes locality inert for that copy.** A missing
  baseline is an absence of evidence rather than a tie: treating it as a tie let a copy with no
  baseline displace a raw copy at 05.00, selecting an older reprocessing *and* skipping the offset
  correction. Such a copy also refuses its whole date downstream, and the read-failure ladder
  recovers from a read error but not from a refusal, so a reprocessing that can be corrected beats
  a newer one that cannot be processed at all. An already-harmonised copy is exempt: no offset
  decision rests on its baseline, so penalising it there would hand the tile-date to an OLDER raw
  reprocessing, which is the opposite of the usable-first rule the key starts with.

**Which copies are the same acquisition is decided by identity, not by a timestamp.** Two
reprocessings of one granule share a datatake — mission, sensing start and absolute orbit, in
`s2:datatake_id` — and differ only in the processing-baseline suffix. They do **not** agree on the
catalogue `datetime`, which is a per-copy field: on the committed 2017-12-19 cassette, the 02.06
and 05.00 copies of one granule are timestamped more than three minutes apart. A tolerance around
that timestamp therefore cannot separate "two reprocessings" from "two passes" without getting one
of them wrong, and the pair was kept as two acquisitions and mosaicked together. Identity needs no
tolerance: the timestamp window survives only as the fallback for a copy naming no datatake.
Splitting on a real acquisition is what protects genuine same-day coverage — successive orbits
revisit a high-latitude tile the same day, and keying on `(tile, solar day)` alone dropped 493 of
2,733 items as duplicates when they were distinct acquisitions.

A copy naming **no** datatake joins an identified acquisition its timestamp places it in, before it
is allowed to start one, and it is matched against *any* member of that acquisition — members of
one observation do not agree on the timestamp, which is the whole reason identity is primary, so
closeness to any of them is the available evidence. Without that, one reprocessing declaring the
datatake while its sibling omitted it were never compared however close their timestamps, and both
survived to be fused.

**The tile key is read from whichever property the catalogue populates**, `grid:code` or
`s2:mgrs_tile`, then the item id — all canonicalised to one form, so two catalogues naming one
tile produce one grouping key. Planetary Computer needs its own property: its ids carry the tile
in a field the Element 84 pattern does not match, so without it every item was unkeyable and
duplicate selection was a no-op for the whole provider.

**Where the producer cannot be read from an item's assets, the collection supplies it**, through
`known_harmonisation` on `select_preferred_duplicates` — the same value `stac.collection_harmonisation`
gives the correction path, so the two cannot disagree. This is load-bearing rather than an
optimisation, and the two changes above are why: making Planetary Computer items keyable gave that
provider a fallback ladder for the first time, and deriving the correction from the items made its
unreadable baselines refuse. A spare judged only on visible assets therefore looked harmless, would
be offered to the ladder, and would abort the ingest when a read failure stepped down to it, since
the recovery loop steps down on a read failure and not on a refusal.

The **fallback ladder** — the rejected copies, in the order a read failure steps down them — is
built by one global sort over the whole tile-date instead, using a key that has no notion of
"best in my group". Two other constructions were wrong. Ranking the tile-date with the same
function used for winners let one malformed item's suspension revert every acquisition's ladder
to sequence order. Ranking each acquisition separately and concatenating them behind their
sorted heads is wrong further down: with one acquisition holding 05.00 and 01.00 spares and
another holding 04.00, it yields `[05, 01, 04]`, and the unattributed recovery consumes the head
on each retry, so the second retry takes 01.00 and never tries the 04.00. In the global order an
unreadable baseline sorts last rather than suspending anything, which is the same protection by a
more direct route: a copy whose baseline cannot be read is the one whose correction will silently
be skipped.

Buckets are compared by parsing the href's host and path rather than by substring, so a
lookalike host cannot be mistaken for a preferred one. The baseline is matched as a version
string rather than parsed as a number, so `"NaN"`, `"Infinity"` and every other value that is
numeric without being a version read as unknown — see `item_baselines.py`.

### When a source object will not read

Some published objects are corrupt: a tile of the COG will not inflate, and no retry of any
length recovers it. That is a different condition from a throttle or an expired credential,
which look similar coming out of the loader — `rasterio` wraps both in a
`WarpOperationError` that discards the cause — so `is_unreadable_source` inspects the whole
exception chain and matches only the codec-level signatures, excluding the credential and
throttle markers explicitly. It fails CLOSED: anything unrecognised propagates rather than
being treated as bad data, because responding to a bad minute by reading worse imagery is
the one outcome the recovery must never produce.

**The chain only exists if something kept it.** The read fails on a Dask worker, and rasterio's
GDAL error classes cannot be serialised out of it by default — Dask detects that and substitutes
a plain `Exception` holding the wrapper's repr, so what arrives is one line with no cause and
every predicate here has nothing to read. `loader_failures.keep_causes_picklable`, installed on
every worker by the same plugin as the object capture, is what makes the cause arrive. It is best
effort, so `cause_was_flattened` recognises a failure that arrived without one and
`read_failure_context` logs `READ CAUSE LOST` — the signal that a verdict was reached from
nothing rather than from evidence.

**An object that was never published counts too, and needs its own markers.** Every
codec-level signature is emitted by a BLOCK READ, and a missing object fails at open, before
any block is requested — so a catalogue item naming an href the provider never wrote used to
match nothing and fail the whole leg. `ObjectNotFound`, `NoSuchKey` and `The specified key
does not exist` cover the three layers that can surface it, and they are matched only
alongside the source reader's own vocabulary (`RasterioIOError`, `WarpOperationError`, `CPLE_`,
`HTTP response code:` — the same set the refusal predicate below pairs against). That pairing
is what makes them mean SOURCE: those strings belong to the S3 layer and every S3 client in
the process shares it — `icechunk`'s error enum carries two of them verbatim — so unpaired
they would let a hole in the destination store, or in the ROI mask, be recorded durably as
provider data loss. GDAL is used here only to read source imagery, so its name beside the
not-found text is the discriminator. `NoSuchBucket` is deliberately excluded: a vanished
bucket is systemic and must fail the leg on its first date rather than be skipped date by
date.

Nothing counts or caps these skips on the OPTICAL path. A source object that will never read
is rare enough per granule that a ceiling would only ever fire on a fault of some other kind,
and every date given up is already logged, restated in the end-of-run summary, and written to
the store. The radar skip below does carry a ceiling, for the different reason given there: it
answers a provider refusal, which arrives fleet-wide and all at once.

Past that point the response is a ladder, in `s2_roi.py`'s consume path:

1. **Attribute.** Ask the cluster which objects the loader gave up on
   (`loader_failures.collect_aborted_hrefs`) and map them back to tile-dates.
2. **Step down** those tile-dates to their next catalogue copy and re-prepare the date. The
   copy is older reprocessing, so this trades processing baseline for a date that reads.
3. **Give up, loudly,** when the implicated tile-dates have no copies left: the date is
   skipped rather than the leg failed, and it is recorded on the store as
   `assessed_unreadable_dates` so the absence reads as a finding rather than an unexamined
   gap.

Two properties are worth stating because they are what the attribution step buys, and they
are held by tests rather than by comment:

- **Blast radius.** With attribution, one bad object steps one tile-date. Without it, every
  duplicated tile-date in the date steps together — which on a wide ROI is most of the date,
  so a single bad object downgrades the baseline of hundreds of tiles that read perfectly
  well.
- **Termination.** A bad object whose tile-date has no alternate is given up immediately.
  Without attribution the ladder first walks every *other* tile's alternates, at a full
  re-read of the date per rung, before reaching the same answer.

Attribution can fail — a worker that died with the read, a cluster already gone, a loader
that words its message differently. When it does, the unattributed behaviour above is the
fallback, and the record says which of the two happened: `scope=attributed` means the named
objects are the ones that failed, `scope=whole-date` means the failing object was not
identified and the tiles listed are every tile in the date.

The batched write path cannot reach the ladder — a batch is one graph and one commit — so it
isolates first: an unreadable source anywhere in a batch re-runs the batch's dates one at a
time, each then getting the per-date recovery. That isolation is what stops one corrupt
object from failing a zone-year identically on every retry.

### When the provider refuses the read

An authorization refusal, a throttle and a server error are a different finding again. They say
nothing about the imagery — the same object read minutes earlier and reads again once the service
recovers — so no fallback copy helps and no date should be given up for one. That verdict is
reached inside `is_unreadable_source` in `duplicates.py`, which declines them before any
bad-data marker is consulted; there is no separate predicate to ask.

**What a positive verdict buys is TIME, and nothing else.** It is passed to the shared write
retry as `wait_out`, and the policy then keeps re-attempting that one failure until it has spent
`WAIT_OUT_BACKOFF_S` of backoff. It may never be spent on giving up a date: a date given up and a
later date committed puts the earlier one permanently below the store's append-only maximum, so
the re-run meant to recover it is refused instead. If the wait is not enough, the write fails, the
leg fails with its time axis unmoved, and the leg's own retry re-offers the date in order.

**How much time depends on WHERE the waiting happens**, and the two places cost very different
things. A write waits with its leg's whole Dask fleet held idle behind it, so the in-leg budget is
minutes. A leg that has FAILED has released its fleet, so waiting before re-dispatching it costs
latency and nothing else — that budget is `leg_refusal_backoff_s`, and it is tens of minutes. The
patience goes where it is cheap, and the in-leg wait covers only the ordinary wobble.

Carrying the verdict from one place to the other takes a type. The leg-retry layer sees a failure
DETAIL string, and no marker on that string can separate a refused read from a crash — the wrapper
discarded the cause long before. So a radar write that exhausts its in-leg budget on a refusal
raises `errors.ProviderRefusedReadsError`, whose name reaches the detail and is what
`_leg_backoff_s` keys the long delay on. Nothing about the failure changes: it fails the leg
exactly as it did, and skips exactly as much, which is nothing.

**And only a refusal that arrives AFTER a successful read earns the expensive wait.** An
authorization verdict on a valid credential is either the provider misbehaving or our permissions
being genuinely wrong, in the same words. What separates them is not the message but when it
arrives: a permissions fault is total and deterministic, so it refuses the leg's FIRST date, where
a provider wobble arrives after the leg has already been served. `s1_roi` keeps one per-leg flag,
set on the first committed date, and withholds the long wait until it is set — so a leg whose
access is genuinely wrong fails promptly and releases its fleet instead of idling on it.

It fails closed three ways, and each closed door costs only the ordinary attempt limit. A
credential fault on THIS side is excluded first, because it is repairable here and no waiting
fixes it. A refusal nothing attributes to the source reader is excluded too: `AccessDenied`,
`SlowDown` and `InternalError` are S3's words and every S3 client shares them, so those markers
only count alongside GDAL's own vocabulary (`RasterioIOError`, `WarpOperationError`, `CPLE_`) —
GDAL reads source imagery here and nothing else. And anything unrecognised is excluded, which is
what a failure whose cause was stripped crossing the worker boundary looks like: it draws no long
wait on suspicion, and it is not given up either.

The two predicates were once overlapping, and a caller's ORDER of asking decided the verdict.
They are now disjoint by construction, and by sharing one classification rather than keeping two
lists in step: both read the same markers and the same HTTP status RANGES, so a status nobody
enumerated cannot be a refusal to one predicate and bad data to the other. A caller that knows
only one of them cannot misclassify.

That leaves one honest gap, and it is the fail-closed direction. A refusal carrying neither a
name nor a status — a transport failure with no code, or a cause destroyed crossing the worker
boundary — is declined for want of evidence rather than named as a refusal. It draws no long wait
on suspicion and is not given up either. `cause_was_flattened` is what says which of those two
happened, so a leg reading without a decidable cause is visible rather than silent.

### The radar bounded skip (`s1_roi.py`)

Every OPERA read on the radar path happens inside a date's write, so a failed read raises out of
the per-date loop. Until this skip, one refused read cost every LATER date in the window too: a
source refusing reads for thirteen minutes emptied 178 zone-years that had already committed
months of sound data.

The radar response is the tail of the optical one without the copy ladder, which radar has no use
for — OPERA publishes one copy of a granule:

1. **Retry**, through the shared `store_write_retrying` policy — and for a provider refusal that
   arrived after a successful read, retry past the attempt limit, because waiting is the only
   response a refusal has. Radar is the one caller that asks for this: OPERA publishes one copy of
   a granule, so there is nothing to step down to, and the optical path's answer is the copy ladder
   instead. A long wait per rung of that ladder would multiply with it.
1. **Fail the leg under a name the cell can act on** if that wait was not enough
   (`ProviderRefusedReadsError`), so the re-dispatch waits on the long schedule rather than the
   short one. No date is skipped and the time axis does not move.
2. **Give up the date** once that retry is exhausted, if and only if the failure is one the
   source is answerable for AND recomputes. There is one scope, `unreadable`, and one remedy: a
   reprocessed copy at the provider. A refusal used to be a second, recoverable scope
   (`provider-refused`); it is not, because giving up a date and then committing a later one puts
   the earlier one permanently below the append-only maximum, so the re-run meant to recover it is
   refused instead.
3. **Record it on the store** in the same `assessed_unreadable_dates` attribute the optical path
   writes, so the coverage gate refuses to excuse a month that lost dates.
4. **Stop past `MAX_GIVEN_UP_DATES`**, and stopping is TERMINAL.
   `TooManyGivenUpDatesError` is IN the leg-retry classifier's non-retryable set, because nothing
   counted toward the ceiling can clear: a provider refusal re-raises and is retried in order, so
   every date reaching that counter failed for a cause that recomputes, and a re-dispatch would
   re-read the same objects, spend the per-read ladder on each, and hold a fleet to reach the
   identical answer. The remedy is a reprocessed copy, not another attempt.

A date offered by two consecutive batches is given up ONCE. Batch queries are padded a day either
side, so a boundary solar day comes back from two queries and would otherwise be listed twice and
cost twice.

### S3 direct access for OPERA

S3 direct access (`get_s3_credentials`) bypasses the 5-hop OAuth redirect chain for each
OPERA COG tile. One HTTP round trip (~0.5 s) fetches temporary STS credentials valid for
1 hour, enabling GDAL to read directly from `s3://asf-cumulus-prod-opera-products` without
per-file HTTPS redirects. At batch scale this is the dominant latency reduction.

### GDAL network tuning

`configure_gdal_environment()` (in [`config/environment.py`](../config/environment.py)) must be
called before importing `rasterio` or `odc.stac`. It sets GDAL config options for network
resilience (retry counts, timeouts, connection pooling) that affect all subsequent COG reads.

### Chunk alignment

The ROI Zarr mask is generated with `chunk_size` matching `INGEST_CHUNKS` so that
`da.from_zarr` reads are zero-copy — each Dask partition maps to exactly one Zarr chunk.
The same chunk sizes are passed to `odc.stac.load` (after translating `northing`/`easting`
to `y`/`x`) so band arrays and the mask share the same partition boundaries for aligned
Dask operations. (Inference reads 2048×2048 sub-tiles out of these 4096×4096 chunks via
`zarr.Array.oindex`, which needs no such alignment — see
[`inference/README.md`](../inference/README.md).)

### has_new_stac_dates pre-check

`has_new_stac_dates` is meant to run before provisioning a Dask cluster: it queries the STAC
catalog and checks for new dates without reading any raster data or starting Fargate tasks,
so a flow can exit early when nothing is new.

**Not yet wired into any flow** — until it is, the caller is responsible for not passing date
ranges that overlap what's already in the store (otherwise we spin up a cluster and iterate
batches only to discover there's nothing to write). Wiring it into the S1/S2 ROI flows is
tracked in [issue #47](https://github.com/dClimate/tessera-embeddings/issues/47). When doing
so, avoid sharing one OPERA `item_provider_fn` between the pre-check and the real query — the
provider re-queries CMR on every call, so reuse would double the query cost.

---

## Authentication (EDL / OPERA data)

OPERA RTC-S1 data hosted by ASF requires NASA Earthdata Login (EDL) credentials because ASF
uses NASA's OAuth2/URS system for access control. Unlike commercial cloud data (S2, Landsat),
OPERA data is not publicly readable from S3.

### Setup

```bash
export EARTHDATA_USERNAME=your-username
export EARTHDATA_PASSWORD=your-password
```

You must also approve the **ASF Cumulus** application at
[urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) → Authorized Apps.

### S3 Direct Access (preferred)

`auth.get_s3_credentials` exchanges EDL credentials for temporary AWS STS credentials:

1. `GET https://urs.earthdata.nasa.gov/api/users/tokens` — reuse an existing EDL bearer
   token (EDL accounts have a maximum token limit; creating a new one unnecessarily can hit
   that limit).
2. If no token exists, `POST .../api/users/token` to create one.
3. `GET https://cumulus.asf.alaska.edu/s3credentials` with `Authorization: Bearer <token>` —
   returns `accessKeyId`, `secretAccessKey`, `sessionToken` (valid 1 hour) for the
   `asf-cumulus-prod-opera-products` bucket in `us-west-2`.

`set_s3_credentials` then injects these onto both the orchestrator process and all current and
future Dask workers via a `WorkerPlugin`. It sets `AWS_*` environment variables (consumed by
boto3 when `odc.loader` builds an `AWSSession`) and resets the cached per-thread session so the
next `/vsis3/` open picks up the new credentials.

**Renewal runs on a timer, not on the work loop.** `s1_roi.credential_ticker` re-checks the
credential's remaining life every `CRED_TICK_INTERVAL_SEC` for as long as batches are being
consumed; the loop's own per-batch and per-date checks remain as a fallback. The timer is what makes
this correct rather than merely usual: renewal driven only by the loop can fire only *between* units
of work, so any unit that outlives the remaining margin cannot renew from inside itself. That
coupling is self-reinforcing — slow work renews less often, an expired credential fails every read,
failing reads stop progress, and no progress means no further renewal.

**What a worker receives is a snapshot.** The plugin freezes the credential at construction, so a
worker joining N minutes after the last broadcast starts life with only the remaining TTL, and past
the TTL starts with none. Under adaptive scaling workers join throughout a leg, which makes the
broadcast **cadence** a correctness condition rather than a tidiness one — the ticker is what bounds
N. Every broadcast logs the credential's advertised expiry (`S3 credentials broadcast to workers`),
so the cadence is auditable from a leg's own log.

**Per-thread AWSSession cache**: `odc.loader` caches a boto3 `AWSSession` per thread in
`threading.local` on first use and ignores subsequent env var updates for that thread's
lifetime. Dask task pool threads are long-lived, so the initial 1hr STS token was getting
pinned across refreshes and expiring mid-read. `auth.py` patches `odc.loader._rio.ThreadSession`
at module import time so each thread self-detects `AWS_ACCESS_KEY_ID` drift and rebuilds its
cached session from current env vars. `rasterio.env.Env` (entered by `odc.loader.rio_env()` on
every `/vsis3/` open) then hands the refreshed `AWSSession`'s frozen credentials to GDAL — so
no `gdal.SetConfigOption` or `VSICurlClearCache` is needed. This reaches into private
`odc.loader` internals (`_OdcThreadSession`, `_local`) and is a version-sensitive hook — if odc
renames those symbols, the import fails loudly and the regression tests in
`tests/unit/test_auth.py` catch the break in CI before it hits a 1hr cloud run.

This was empirically verified on a us-west-2 EC2 box (2026-05-20): a four-month S1 ingest
run at `cred_refresh_interval_sec=60` over a persistent local Dask cluster forced multiple
mid-run STS refreshes; every batch's `/vsis3/` reads succeeded, confirming the env-drift
patch plus orchestrator-side `_local.reset()` are sufficient without explicit GDAL
credential-cache calls.

OPERA asset STS credentials are intentionally **never cleaned up** from env vars. This avoids
a race condition where one Dask task's cleanup could remove credentials another task still
needs. The consequence is subtle: once `set_s3_credentials` runs, the `AWS_*` env vars hold
OPERA-scoped STS tokens that grant access **only** to `asf-cumulus-prod-opera-products`. Any
S3 access to the project's *own* bucket that resolves credentials from those env vars — every
icechunk `Repository.open`/`create` in the S1 write path, not just the initial create — then
fails with `AccessDenied`.

Icechunk/Zarr operations on the project's own bucket therefore must resolve **IAM-role**
credentials, bypassing the env vars. The mechanism:

- `providers/aws/credentials.py::iam_icechunk_credentials` resolves credentials from the
  botocore chain with the `env` provider **removed**, so it always lands on the deployment's
  IAM role (instance-metadata / ECS task role / local SSO) regardless of what STS tokens the
  env vars hold. It returns `icechunk.S3StaticCredentials`.
- `storage.zarr_store` exposes a `credentials_provider(provider)` **context manager**.
  `_create_storage` uses the registered provider as the `get_credentials` callback for any S3
  open lacking an explicit one, for the duration of the block — scoped rather than permanent
  so a reused process (a Dask worker) is not left pinned to it for later, unrelated opens,
  and the previous provider is restored even if the body raises. The storage layer ships this as `None` and never imports
  botocore (it must stay cloud-agnostic, per the `no-botocore-outside-aws-provider`
  architecture rule); only the AWS provider supplies the concrete callback.
- The `process_roi_sar` Prefect task registers `iam_icechunk_credentials` via that hook when
  `use_s3_direct=True`. **This must happen in the task shell, not the flow body** — with the
  Dask task runner the domain function (and its store writes) execute in a *worker* process,
  so a provider registered in the flow-runner process would never reach them.

The plain-Zarr side needs the same identity, and one property beyond it. An ROI mask is not an
Icechunk store, so it is read through fsspec, and
`providers/aws/credentials.py::iam_s3_storage_options` is the fsspec counterpart: the same
env-stripped chain, returned in the shape fsspec takes as `storage_options`. The ingest is handed
the **callable**, not its result — and `read_roi_mask` resolves it inside each block read rather
than once when it builds the graph.

That last part is load-bearing, because the mask array is LAZY: its block reads happen inside a
later `write_day_windows` compute, which on the radar path spans a whole 30-day batch's writes. One
credential resolved at graph-build time would be presented by every one of those reads, and once it
expired the read would fail with `ExpiredToken` on a bucket the role can always read — a lifetime
problem wearing a permissions problem's error message. Opening the store per block keeps the
credential no older than the read that uses it.

Two consequences. Each block read costs its own store open, so its own metadata round trip, where
the old construction paid one for the whole array; and the returned array is cloudpickle-only,
because the closure is a nested function. Both are measured in
`context_docs/decisions/022-resolve-the-roi-mask-credential-at-read-time.md`, and both are reasons
not to hand this array to a plain-pickle boundary, or to read a whole zone grid you do not need.

**IMDS throttling — why `_resolve_iam_credentials` is `lru_cache`d (gotcha).** The credential
machinery has two distinct TTLs, and conflating them overwhelms the EC2 Instance Metadata
Service (IMDS):

- `iam_icechunk_credentials` sets `expires_after=15min` on the returned `S3StaticCredentials`.
  This is how often **icechunk** re-invokes our callback per repo client — it is *not* how
  often we should touch IMDS.
- `_resolve_iam_credentials` is `@lru_cache(maxsize=1)`, so the botocore session — and the
  live `RefreshableCredentials` it returns — is built **once per process**. For an IAM role
  botocore hands back a `RefreshableCredentials` that serves its in-memory credential and
  refreshes itself in the background; `get_frozen_credentials()` is a pure expiry-time check
  that only re-hits IMDS inside botocore's refresh window (~advisory 15 min before the ~6h
  token expiry), lock-guarded so concurrent callers don't stampede.

Without the cache, every callback built a *fresh* session and did a **cold IMDS resolve**.
Under many concurrent workers/threads that bursts IMDS past its per-instance rate limit, and
the SDK surfaces it as `failed to load IMDS session token / invalid token` or
`no providers in chain provided credentials` — transient, but enough to fail a run of chunks
before recovering. Caching the session decouples "how often icechunk asks" from "how often we
hit IMDS": the former stays at 15 min, the latter drops to roughly once per token lifetime.
`lru_cache` does not cache exceptions, so a failed cold resolve still retries next call. The
same provider is injected into inference workers (see `inference/README.md`), where long-lived
Ray actors made this the dominant failure mode.

### URL Rewriting

CMR-STAC returns HTTPS asset URLs in two formats depending on satellite vintage:

| Format | Example |
|---|---|
| **datapool** (older S1A) | `https://datapool.asf.alaska.edu/RTC/OPERA-S1/<filename>` |
| **earthdatacloud** (newer S1C) | `https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/<dir>/<file>` |

`auth.rewrite_assets_to_s3` converts both to `s3://asf-cumulus-prod-opera-products/...` via
pure string manipulation (no HTTP calls). For the datapool format, the granule directory name
is reconstructed by stripping the band suffix (`_VV.tif`, `_VH.tif`, `_mask.tif`) from the
flat filename.

### Legacy CloudFront Signed URLs (fallback)

`_EDLSession` is a `requests.Session` subclass that preserves the `Authorization` header
across cross-domain redirects. Python `requests` strips this header when following a redirect
to a different domain. The ASF download chain goes: `datapool.asf.alaska.edu` → 
`urs.earthdata.nasa.gov` (OAuth exchange) → CloudFront CDN. Because the header is stripped
at the first hop, it is missing by the time URS sees the request. `_EDLSession.rebuild_auth`
re-injects credentials whenever the redirect target URL contains `urs.earthdata.nasa.gov`.

`resolve_item_assets` follows the full redirect chain per asset and mutates the STAC item's
asset HREFs to CloudFront signed URLs before `odc.stac.load` reads them. This path is kept
for out-of-region access where S3 direct is not available, but is significantly slower.

---

## OPERA-Specific Query Quirks

### Native granule query (orbit filtering + item construction)

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

Orbit direction is filtered **server-side** via `attribute[]`, so the response already
contains only the desired orbit — no separate STAC search and no local granule-ID
intersection. Each granule entry's data download links (`rel` ending `/data#`, href ending
`_VV.tif` / `_VH.tif`) are mapped onto the `S1_OPERA_BANDS` asset keys (`0_VV`, `0_VH`) to
construct `pystac.Item`s shape-compatible with the rest of the pipeline. The granule's
`title`, `time_start`, and `polygons` supply the item id, datetime, and geometry. CMR
pagination is handled via the `CMR-Search-After` response header, which pages cleanly at
2000 against the same host. See
[ADR 009](../../../context_docs/decisions/009-native-cmr-granule-query.md) for the full rationale.

### Burst Timestamp Normalisation

A single MGRS tile bbox query returns ~10 burst granules per date, each with a slightly
different sub-second UTC timestamp (reflecting actual acquisition time). If passed to
`odc.stac.load` as-is, each burst becomes a separate time step instead of being mosaicked
together.

`normalize_opera_timestamps` delegates to `solar_days.normalize_to_solar_day`: it groups
bursts by **solar day** and sets all timestamps in each group to noon UTC of that day.
It grouped by UTC *date* until 2026-07-30, which made the whole solar-day apparatus on the
S1 path inert — everything downstream derived its "solar day" from a timestamp already
flattened to the UTC date, so radar was labelled in UTC while optical was labelled in solar
days. `odc.stac.load` then treats them as concurrent acquisitions
and spatially mosaics them into a single time slice.

### UTM CRS Derivation

CMR-STAC OPERA items lack the `proj:` extension, so `odc.stac.load` cannot infer the output
CRS. `mgrs_tile_to_utm_epsg` derives the correct UTM EPSG from the tile's zone number and
latitude band (C–M = southern hemisphere, N–X = northern hemisphere), e.g. `33UUP` → EPSG:32633.

---

## Accessing the Dask Dashboard

Ingestion flows run a Dask cluster on ECS Fargate. The scheduler is in a private subnet.
Use SSM port forwarding to reach the Bokeh dashboard:

```bash
# Look up TASK_ID and RUNTIME_ID for the Dask scheduler task in the ECS console
aws ssm start-session \
  --target ecs:yield-cluster_${TASK_ID}_${RUNTIME_ID} \
  --document-name AWS-StartPortForwardingSession \
  --region <aws-region> \
  --parameters '{"portNumber":["8787"],"localPortNumber":["8787"]}'
```

Then open http://localhost:8787 in your browser. `log_dashboard_ssm_command` (in
`providers/aws/dask.py`) logs the same command with the target and region already filled in
once the cluster is up.

`AWS-StartPortForwardingSession` forwards a port on the scheduler container itself. The
`AWS-StartPortForwardingSessionToRemoteHost` variant is for hosts reachable *from* the
container (RDS, an internal ALB); current SSM agents refuse loopback destinations for it and
fail with `Forwarding to IP address localhost is forbidden`.

Requires the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (`brew install session-manager-plugin`).
