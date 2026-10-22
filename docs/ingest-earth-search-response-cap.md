# Earth Search's response cap: sizing Sentinel-2 queries so they are answered

Earth Search will not return more than about 6 MB in one answer, and a Sentinel-2 query over a
zone-year asks for far more than that. Sizing requests around that refusal is most of the query
code in `stac.py` and all of its complexity.

This document is for anyone changing that code, or looking at a 502 from Earth Search and asking
whether something is wrong. The first half is one plain-language pass over the mechanism;
[In detail](#in-detail) is the implementation.

## Four terms

A **page** is one request and the hundred results it returns. A **cursor** is a
bookmark the catalogue hands back, which the next request must carry — not a page number, so
there is no way to ask for the twentieth page directly and a refused request leaves no bookmark
for the one after it. A **date window** is a from-date and a to-date. A **worklist** holds one job
per date window; an impossible job is crossed off and replaced by two shorter ones.

## Why a request is refused, and how the query is shaped around it

The problem this shape exists for: Earth Search refuses **any request whose answer would exceed
about 6 MB**, AWS Lambda's synchronous response limit. Every 502 the campaign has seen from that
provider is this and nothing else. The refusal arrives in 1.3 seconds, as fast as a success, so
nothing is overloaded and repeating cannot help — the remedy is always to ask for a smaller
answer.

```text
WHY A REQUEST GETS REFUSED -- the whole mechanism, in one line

   Earth Search will not return more than about 6 MB in one answer.
   That is AWS Lambda's limit on a single synchronous response, and the search API
   sits behind one.

   scenes are not all the same size: in the heavy months a hundred of them is
   4.6 MB on average, the largest page ever served was 5.7 MB -- and some go over:

      100 scenes from this bookmark  ->  would be ~6.2 MB  ->  REFUSED
       90 scenes from this bookmark  ->            5.6 MB  ->  answered
       75 scenes from this bookmark  ->            4.6 MB  ->  answered

   Same bookmark, same dates, character for character. Only the number asked for
   changed. So nothing is wrong with the bookmark, the depth, or the service --
   the reply was simply too big to send.

   Which hundred scenes you land on is decided by your bookmark and your date
   window, which is why it looks like the service has taken against one specific
   request. It hasn't. It is doing arithmetic on the size of the answer.

   THE MARGIN IS THE RISK: the largest page served is 96% of the ceiling.
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
                                  v  five jobs are untouched.
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

   Jobs run two at a time. Almost all of a request is spent waiting for the catalogue
   to think -- 86% of it, before a single byte arrives -- so overlapping the waits is the
   only thing that moves the clock. Same query, same scenes, same order: 39 minutes as
   first written, 3.5 minutes with six at once. The campaign runs two, because six
   drew 403s from Earth Search across the fleet.
```

## What the tests hold

Two properties are tested. The jobs must add up to exactly the window asked for, no day missed
and no day added. And the results must come back in the same **order**, not merely the same set —
two scenes taken on the same day with the same cloud cover are separated only by which arrived
first, and that decides which one supplies an overlapping pixel.

## In detail

### Which entries are heavy, and when

What makes it bite unevenly is that items are not all the same size. Catalogue entries from
roughly **November 2018 to March 2019** are about 100× larger than normal: an entry usually
carries the tile's bounding rectangle, a couple of hundred bytes, while these carry the outline of
where the imagery actually falls, and since Sentinel-2 builds an image from twelve detectors that
edge is a fine sawtooth — 98 KB in one measured entry against 0.2 KB. The asset list is ~18 KB
either way, so the outline is what makes them heavy. The band is a reprocessing gap: ESA
reprocessed most of the archive to a version carrying the simple rectangle, and those months are
the stretch where only the original 02.11 is on offer. It cannot spread, since no current
processing version produces these outlines, and it could disappear if that stretch is ever
reprocessed.

Two consequences in this code: a hundred entries is ~2.2 MB outside the band and at or over the
ceiling inside it, so page refusals are a 2019 phenomenon; and a month of these entries holds an
order of magnitude more bytes than the same month in 2024, which is part of why the query streams
month by month. Any per-item figure — page size, retained bytes, query timing — reads differently
here than anywhere else in the archive.

### The page size

The page size is `STACProvider.max_page_size` (the `limit` per page request), defaulting to 250
and set to 100 for Earth Search. It applies only to providers queried through `client.search()`,
which in production means Earth Search; the OPERA `cmr-asf` path queries the native CMR Granule
API instead — see [ADR 009](../context_docs/decisions/009-native-cmr-granule-query.md).

**Why 100 for Earth Search, and what to watch.** Not throughput but the cap: 250 items of
`sentinel-2-l2a` is always over it. A hundred averages 4.6 MB, yet the largest page ever served
was 96% of the cap, so the margin between fine and refused is about a quarter of a megabyte and
the average is the wrong number to reason from. Lowering it further is not the answer — six months
of a ten-year archive is a concentrated problem, and the page-size fallback below handles it where
it happens rather than taxing every query in every year. If first pages ever start returning 502
this margin is the first thing to check, since a first-page refusal is the one case no date-window
re-cut can route around.

### Refusals deep in a walk, and the two levers against them

**The same cap also refuses pages deep in a walk**, which looks like a separate defect and is
not. Because item sizes vary the refusal is deterministic in the *request* — cursor and date
window together — rather than in how deep the walk has got, which is why the same refusal has
appeared at page 289 of one window and page 14 of a shorter one sharing its late bound.

What clears it is either a smaller response or a regrouping that produces one, and `stac.py`
tries them in cost order.

**A shorter window first.** Its halves between them walk about as many pages as the parent would
have, where a smaller page re-walks the whole window at twice the requests. What matters is the
window's **end** date: the catalogue pages newest-first, so the late bound fixes the whole cursor
sequence and shortening only the start does not help. `_query_stac_items` re-queries as shorter
windows on any upstream-error refusal past the first page, recursing until a window completes or
reaches a single day. Separately, it reads the match count reported beside the first page and
cuts a window matching more than `_MAX_QUERY_ITEMS` to size — that bounds *cost* rather than
fixing the defect.

**A smaller page as the fallback**, halved down to `_MIN_PAGE_SIZE`, because shortening cannot
reach two refusals: a **first page**, which a shorter window asks identically, and a **single
day**, the re-cut's floor. A stated overload (429, 503) is never answered with a smaller page:
that means the provider is busy, and more requests is the wrong direction. A refusal neither
lever can route around still raises the classified `CatalogueQueryError` with its token.

### Concurrency

The windows are independent searches, so `_fill_window_tree` walks up to
`_QUERY_WINDOW_WORKERS` (2) of them at once. The worklist is driven from the calling thread and
tasks only ever walk — they never submit and never wait — so deadlock is structurally impossible
rather than merely unobserved. Each thread gets its own `Client`, because `StacApiIO` wraps a
`requests.Session` that is not documented thread-safe. Output order comes from
`_WindowWalk.preorder()` on the finished tree, and the `id` dedupe runs at that assembly step
rather than as pages arrive, so first-occurrence-wins means first in the **walk** and not first
off the wire. Two, not more, even though more is faster: the campaign runs tens of cells against
this one provider at once, so the setting multiplies the concurrent search streams Element 84
sees. At 6 they answered the fleet with 403; 2 ran clean for a whole campaign. A failure does not
stop the other windows — every window is walked, all failures collected, and the
depth-first-earliest raised, so which failure surfaces is a function of the query rather than of
which task finished first.

### The re-cut is exact

The re-partition is a pure re-cut, never a narrowing, and it is exact in both directions: the
outer bounds are the caller's own date strings handed straight back, and every interior boundary
is a single **instant** (`T00:00:00Z`) shared by the window that ends there and the window that
starts there. The catalogue's range is inclusive at both ends, so the union is the input window
with no gap and no overhang, and the only overlap is that one instant. An instant rather than a
date because the client expands a bare date end to `T23:59:59Z`, so windows abutting on
consecutive DATES would leave the last second of each seam's earlier day unasked for; sharing the
whole boundary DAY closes that gap too, but makes every seam re-fetch a full day for the dedupe
to discard. Items are deduped by `id` across every search a query runs, which absorbs the
boundary instant and the antimeridian overlap alike.

### The order of the walk does not reach the mosaic

It does change the order items are *walked* in — one walk returns the window newest-first, the
worklist returns window by window in date order — which is safe only because `query_stac_items`
re-sorts with `solar_day_sort_key`, making the final sequence a function of the items rather than
of the order the walk produced (see
[the fusion order](../src/tessera_embeddings/ingest/README.md#the-fusion-order-decides-which-source-wins-a-pixel)
for what that sort decides).

The cap, the measured margins, the sampling behind the heavy band and the levers that are closed
are all derived in `context_docs/ingest/campaign-ingest-measurements.md` §7c.
