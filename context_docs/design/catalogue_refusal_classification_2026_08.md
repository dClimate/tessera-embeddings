# Catalogue refusals: naming the request, and separating load from a deterministic refusal

Design + investigation record, 2026-08-10, branch `global-tessera-scoping`.

Companion to `ingest_read_failure_causes_2026_08.md`, whose "Coupling to the leg retry" section
states the general problem this instance is a second case of: a failure class that consumes the
whole attempt budget and can never succeed. That record is about **source-object reads**
(`WarpOperationError`); this one is about the **catalogue search** that precedes them.

## The failure

A campaign cell's optical ingest leg failed three times, hours apart, with the same error:

```
APIError: HTTPSConnectionPool(host='earth-search.aws.element84.com', port=443):
Max retries exceeded with url: /v1/search
(Caused by ResponseError('too many 502 error responses'))
```

Three attempts hours apart failing identically is deterministic for that query, not congestion.
A neighbouring year built successfully from the same coverage, so the archive publishes the data.

## What the code did about it, before this change

### 1. It could not say which query failed

Nothing in the log named the search. Two independent reasons:

- **`pystac_client` discards the request when it wraps a transport failure.**
  `StacApiIO.request` is `except Exception as err: raise APIError(str(err))`. A STAC search is
  a POST **body**; the surviving message carries the host and the endpoint path and nothing
  else. The `status_code` attribute is annotated on `APIError` but only assigned by
  `from_response`, so on the transport path it does not exist at all.
- **Our own retry logging names the URL, which is the same URL for every search.**
  `_http.make_logging_retry`'s `increment` logs `method` and `url` per attempt — `/v1/search`
  in all cases.

So "one month, one page, or the whole year" was unanswerable from the log, and no reproduction
could be handed to the archive's operator.

### 2. Our layer sat above a retry ladder and treated its exhaustion as a first attempt

Measured from the configured `Retry` object (`stac._STAC_RETRY`: `total=8, backoff_factor=2,
status_forcelist=(429, 500, 502, 503, 504), respect_retry_after_header=True`), driven to
exhaustion:

| | value |
|---|---|
| HTTP attempts per **page fetch** | **9** (1 initial + 8 retries) |
| backoff sleeps | 0, 4, 8, 16, 32, 64, 120, 120 s (`Retry.DEFAULT_BACKOFF_MAX = 120`) |
| total wait per exhausted page fetch | **364 s** |

The ladder is mounted on the session by `StacApiIO(max_retries=...)`, so it applies per page
request — a transient 5xx on page N is retried in place rather than restarting the query.
Nothing in our code wrapped the query above that. The exception that escapes is therefore the
ladder's own exhaustion report, and every layer above it treated the whole thing as one failed
attempt.

### 3. The attempt budget was spent exactly as if it were a data failure

Three nested budgets, none of which knew anything about the failure's class:

| layer | knob | default | how a catalogue 502 is treated |
|---|---|---|---|
| leg | `IngestSettings.max_leg_attempts` | 3 | retryable — `_NON_RETRYABLE_LEG_MARKERS` names no HTTP failure, and the default is TRUE |
| cell | `attempts_per_cell_in_cluster` (`sequential_fill` / `fill_zones_sequential`) | 2 | eligible — retry eligibility is by **phase**, and this is `inputs/prepare`, which additionally gets `discard()` + `start()`, re-dispatching the whole ingest |
| zone round | `max_dispatch_rounds` (`run_global_campaign`) | 2 | re-dispatched; the zero-progress guard only breaks *after* a whole round has made none |

So one deterministically-502ing query could cost up to **3 × 2 × 2 = 12 dispatches of the
optical leg**, each preceded by ~6 minutes of ladder per failing page fetch, and each
provisioning a Dask fleet.

The polarity that produces this is deliberate and, in general, right: a re-dispatch RESUMES
(committed dates are skipped, not rewritten), so a wasted retry is cheap while a missed retry
leaves a mosaic incomplete until a human notices. It is only wrong where retrying cannot
succeed — which is exactly the case a message alone cannot identify.

### 4. Leg → parent propagation: already correct, and left alone

`ingest_zone_year` gathers the legs with `return_exceptions=True`, and any surviving entry in
`errors` raises, failing the cell even when siblings completed. In the observed case both radar
legs succeeded and the cell was lost.

**This was examined and found correct. Nothing was changed.** The reasons:

- The completion marker is written **per cell**, after the coverage gate, and `s1_orbit="both"`
  resolving to whichever orbit happened to ingest would stamp a "both" marker over half the
  radar — after which every later run reads the marker and skips the cell. The code comment at
  the `if terminal: break` branch already states this.
- **The successful siblings' work is not discarded.** The unmarked-store branch RESUMES rather
  than clears, and Icechunk commits a date's time slot atomically with its pixels, so a
  re-dispatched complete leg re-queries its catalogue and writes nothing. "The cell was lost"
  means it did not complete, not that its radar was thrown away.

The residual inefficiency is real but separate: the leg list is rebuilt from `_active_orbits`
and never consulted against the marker probe, so a **cell**-level retry re-dispatches a leg that
had fully completed. It pays a catalogue query and a fleet spin-up, not a re-ingest. Not
addressed here.

## The change

Three pieces, each the narrowest thing that makes the distinction possible.

### `ingest/catalogue_refusal.py` (new)

- `CatalogueRequest(collection, window, area, page)` — the request's identity in the fields that
  decide its answer. `page` is a real ordinal, which required `_query_stac_items` to page
  explicitly via `pages_as_dicts()` (behaviour-identical: `items_as_dicts()` is defined as
  `for page in pages_as_dicts(): yield from page["features"]`). Page 0 is the catalogue root.
- `classify_refusal(exc)` → `LOAD` (429, 503) / `UPSTREAM_ERROR` (500, 502, 504) / `UNKNOWN`.
  Read from the exception **chain**, not the message: `pystac_client` re-raises without `from`,
  so the evidence is `APIError.__context__ → RetryError.__context__ → MaxRetryError.reason`, a
  `urllib3.ResponseError` whose text comes from urllib3's own `SPECIFIC_ERROR` template. The
  message is a documented fallback for a refusal that crossed a boundary carrying no chain.
- `CatalogueQueryError` — carries the request and the refusal, and leads its message with one
  whitespace-free token (`CATALOGUE_REFUSAL=<kind>:<status>|<collection>@<window>@<area>@p<page>`).
- `repeat_is_deterministic(kind)` — the policy, kept with the taxonomy.
- `refusal_in(text)` — recovers the token from failure text, by token NAME.

### `ingest/stac.py`

`_query_stac_items` wraps **only** the page fetch (and, separately, `Client.open`). The page
body stays outside: wrapping it would classify our own missing-`id` validation failure as a
catalogue refusal and hand it a retry policy meant for someone else's outage.

### `orchestration/prefect/flows/ingest_zone_year.py`

The leg loop remembers each leg's previous refusal token. A token that repeats **and** classifies
as `UPSTREAM_ERROR` becomes terminal — it ends the loop instead of spending the remaining
attempts. A `LOAD` refusal stays retryable however often it recurs, and says so at WARNING.

## Why a REPEAT, and not the status alone

The status is necessary and not sufficient. A gateway can fail for minutes and recover, so one
exhaustion — even at 364 s — is not proof of a permanent defect, and the existing polarity
("retry unless deterministic in the INPUT") is right to refuse that inference. The repeat is
what earns it: the identical request, refused the identical way, on a later attempt.

Cost of requiring the repeat: one extra leg dispatch versus classifying on first sight.
Benefit: a transient gateway outage still gets its retry.

The signature must therefore be stable across attempts, which is what excludes progress counters
and timestamps from it — those go in the log line only. It holds across attempts in practice
because the search parameters are identical between them: `existing_dates` filters *after* the
query, and `month_ranges` is a pure function of the window, so a resumed leg re-issues the same
searches in the same order and fails on the same page.

**Residual amplification, stated honestly.** The repeat is observed within one
`ingest_zone_year` run, so the cell and zone-round budgets above it still get their turns:
worst case falls from **12 leg dispatches to 8**, not to 2. Closing the rest would mean teaching
`sequential_fill`'s phase-based eligibility about failure classes, which is a different module
and a larger change. Not done. Dispatches were also never the only unbounded axis — elapsed
time was bounded at NO layer, which the next section closes.

## Bounding elapsed time: `max_leg_wall_clock_s`

Everything above counts ATTEMPTS. Elapsed time was bounded nowhere:

- every single page fetch gets **9 HTTP attempts across 364 s** of exponential backoff
  (`stac._STAC_RETRY`: `total=8, backoff_factor=2`, urllib3 caps each sleep at 120 s) before
  anything of ours sees a failure;
- the three budgets above it — `max_leg_attempts` (3), `attempts_per_cell_in_cluster` (2),
  `max_dispatch_rounds` (2) — each treat that whole ladder as one try, and none of the three
  reads a clock.

The campaign's decided policy is to back off and retry expansively when an upstream refuses
reads under load, and to fail a cell only under serious duress — but "duress" had never been
expressed as a number, which is how a single cell could quietly consume hours.
`IngestSettings.max_leg_wall_clock_s` is that number.

### Where the bound sits, and why exactly there

It gates **the decision to start another attempt** in `ingest_zone_year`'s leg loop — a
monotonic anchor before the first attempt, checked only at the point where the loop is about
to re-dispatch a failed leg. A leg that is running is never measured against it, so a
slow-but-succeeding leg can never be why the loop stopped: expansive backoff is the decision,
and the bound exists so patience cannot become *unbounded*, not to make the system give up
early. Consequently the leg loop's true worst case is the deadline **plus one final attempt**
(the last attempt may start just inside the deadline and run to its own natural length).

When it fires, the loop breaks exactly as a terminal failure does — `errors` survives and the
cell raises — and the ERROR line names the elapsed time, the configured budget, and the fact
that the cell returns to the campaign work list and will RESUME from committed dates.

### The default (36 h = 129,600 s), derived from measured leg durations

Two facts pin it, both from `ingest_optimization_campaign_2026_07.md`:

1. **It must comfortably exceed the longest legitimate single leg at the default width.**
   The densest measured zone-year (35N, 2,415 live chunks) ran **175.6 s/date at ~60
   workers** (the five-region k=1 column, §3.16), and a zone-year is **365 dates** — so the
   S2 leg alone is **~17.8 h**. Per-zone residuals on that fit run **±35%**, and a legitimate
   leg can additionally carry in-leg catalogue patience (364 s ladders that eventually
   succeed), so the legitimate band edge is **roughly a day**. This fact is load-bearing for
   the multi-leg case: if S2 succeeds slowly while an S1 orbit fails transiently, the elapsed
   at the re-dispatch decision is S2's whole runtime — a deadline inside the legitimate band
   would refuse that S1 orbit its *first* retry on every dense cell.
2. **A stuck cell must release its campaign slot promptly.** No bound can release it in less
   time than the longest legitimate leg without cutting legs that are merely slow, so the
   best achievable is "within about a working day of outliving the legitimate band":
   36 h ≈ the ~24 h band edge plus ~12 h.

Re-derive the number if per-date cost or the default fleet width changes; both live in the
ingest optimisation record.

### What a legitimately slow-but-recovering source loses

A leg that fails after the deadline loses its remaining leg attempts **in this flow run
only**. The cell fails back to the campaign work list, the cell/zone budgets above still
re-dispatch it, and the re-dispatch resumes from the dates already committed (Icechunk
commits a date's time slot atomically with its pixels) — so the loss is one campaign
round-trip of latency, never work, and never the retry policy as a whole. Worst case across
the stack: up to 4 leg-loop entries (2 cell × 2 zone attempts), each now bounded at the
deadline plus one attempt, where before each was unbounded.

## A hazard the page-by-page walk introduced, and how it was caught

Advancing the cursor by hand (`next(pages)`) is what makes per-page attribution possible, and
it also changed what a *stubbed* search does. The previous `for raw in search.items_as_dicts()`
terminated against a `MagicMock` because `iter(MagicMock())` yields nothing; `next(MagicMock())`
answers forever. One existing test mocks the whole client, and the loop became unbounded there —
a worker reached 17 GB RSS and the unit suite went from ~25 s to over 25 minutes without ever
failing.

Two things fixed it, and the second is the one worth keeping:

- the test now names `pages_as_dicts` and hands it an empty iterator;
- the production walk asks for an iterator (`pages = iter(search.pages_as_dicts())`) instead of
  assuming it was given one, which also accepts a plain list of pages rather than raising
  `TypeError`. A regression test covers both shapes.

The general shape: **replacing an implicit `for` with an explicit `next()` changes the protocol
you depend on** — from `__iter__` to `__next__` — and a test double that satisfied the first can
be pathological under the second. The symptom was a slow suite, not a red one, which is the
failure mode that survives a green CI run.

## Statuses the taxonomy deliberately does not name

`400` and other client errors classify as `UNKNOWN` and keep today's expansive retry. They are
deterministic in the request by definition, so this is arguably wrong — but no observed failure
has that shape, and a general-purpose HTTP classifier was explicitly out of scope. A unit test
asserts the two named sets jointly cover `_STAC_RETRY.status_forcelist`, which is the set that
can actually reach us as an exhaustion.

## Fingerprint consequence — a real cost, paid once

`stac.py` is inside the mosaic-content fingerprint closure (`config.ingest._MOSAIC_CONTENT_SOURCES`
seeds on `ingest/s1_roi.py` and `ingest/s2_roi.py` and follows imports; the closure was 30
modules and gains `catalogue_refusal.py`). So `ingest_code_identity()` moves:

```
before  ingcode-e3a8b30295d5859f
after   ingcode-5c9a0eb3c1e42de4
```

(An earlier draft of this record listed `after ingcode-7c9992d48c189cc1`; that was the value
before `max_leg_wall_clock_s` was added to `config/ingest.py`, which is itself inside the
closure and moved the identity again. Both changes land together on this branch, so the pair
above is the only before/after any store will ever observe.)

That check is validated on **append**, not on the completion marker, so:

- **Finished mosaics are unaffected.** Nothing re-ingests.
- **A mosaic that is mid-ingest right now will refuse its next append** with
  `ConfigMismatchError`, which is a non-retryable leg marker; the resolution is a human deleting
  the interrupted store.

There is no way to add request naming to the catalogue query without this, because the query is
inside the closure. It is the cost every ingest change pays.

## What is worth reporting to the archive's operator

`earth-search.aws.element84.com` `POST /v1/search` returns **502 Bad Gateway repeatedly and
reproducibly for at least one specific search**, while adjacent searches over the same area
succeed:

- The 502s defeat a 9-attempt ladder spanning 364 s of exponential backoff, and recurred
  identically on three attempts hours apart — so this is not a thundering herd from us and not a
  momentary gateway blip.
- A neighbouring year's search over the same coverage succeeded, so the data is published and
  the fault is in serving that request.
- This is the **second independent observation** of earth-search 502s under repeated querying;
  the first is in `ingest_read_failure_causes_2026_08.md` ("Caveat on the sample"), where it
  prevented a duplicate-item survey from converging.

**What we cannot yet give them is the search body**, because the code that records it is the
change described above. The next occurrence will log the collection, the exact temporal window,
the bbox and the page ordinal, which is a complete reproduction. Until then the report is a
symptom without a repro, and should be sent as such rather than with a guessed query.
