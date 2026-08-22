# Ingest source-read failures: five causes, and the retry budget they share

Investigation record. Causes 1 and 2 traced 2026-08-04/05 on `global-tessera-dev`; cause 3 absorbed
2026-08-18 from its own document, for the reason given at that section; cause 4 traced 2026-08-20
and cause 5 on 2026-08-21, both from live campaign legs. **Corrected in place: this document said
"three causes" until cause 4 was added, and "four causes" until cause 5 joined it.** Traces
`WarpOperationError('Chunk and warp failed')` and `PermissionError: The provided token has
expired` to their exact causes, so the guards can be aimed at the right terms.

Companion to `ingest_optimization_campaign_2026_07.md` (the authoritative ingest record) and
`ingest_concurrency_investigation_2026_08.md` (the fleet-width contention work).

## Headline

`WarpOperationError('Chunk and warp failed')` is **not a cause**. It is rasterio's wrapper around
whatever GDAL failed at, and it **discards the reason**. Behind it are **two unrelated causes**
that need opposite responses — and a third joined them on 2026-08-21, see Cause 5:

| | source | preceded by | retryable? |
|---|---|---|---|
| **Expired read credential** | `asf-cumulus-prod-opera-products` (radar) | `ERROR 1: Request for <url> range X-Y failed with response_code=403` | **yes** — the next attempt with a fresh credential succeeds |
| **Corrupt source object** | `sentinel-cogs` (optical) | nothing; no HTTP error at all | **never** — the object is broken at the provider |

Both surface as the identical exception text, so **the leg-retry classifier cannot tell them
apart from the message**, and one of them makes retrying pure waste. See "Coupling to the leg
retry" below.

**Correction, in place.** `docs/runbooks/incident-response.md` previously attributed this error to
"a Dask worker task definition missing a `ulimit` the library needed", citing the
`mirror-a-pinned-resource-field-for-field` episode. **That attribution was wrong for these
failures.** A missing ulimit did cause a warp failure once, which is why the guess was available;
it is not what caused any of the 226 aborted loads examined here. The worker task definitions were
not implicated, and checking them first sends the reader to the wrong place.

## How the wrapper hides the cause

Every one of the 214 pickled `WarpOperationError` payloads in the window carries the same inner
GDAL exception:

```
rasterio._err.CPLE_AppDefinedError: ZIPDecode:Decoding error at scanline <N>
rasterio._err.CPLE_AppDefinedError: <asset>, band 1: IReadBlock failed at X offset i, Y offset j:
                                    TIFFReadEncodedTile() failed.
```

`ZIPDecode` is libtiff failing to inflate a tile's DEFLATE stream. It is what GDAL reports
whenever the bytes it hands the codec are not a valid deflate stream — **including when the range
GET that should have produced them failed.** GDAL logs the HTTP failure to its error handler and
then proceeds into the codec with an unusable buffer, so the decode error is the one that
propagates and the access error is the one that is lost.

The consequence for diagnosis: the words "token has expired" and `ExpiredToken` **never appear on
this path**. Bucketing warp failures against those strings finds no overlap, because the two
populations are disjoint by construction, not because the causes are unrelated. The string that
does correlate is **`response_code=403`**.

## Cause 1 — the read credential expires mid-run (radar)

### How each cause was identified

**Cause 1 — the credential.** The logs showed `PermissionError: The provided token has expired` on
radar reads mid-run, and the credential's own record showed refreshes stopping partway. The mechanism
has two limbs: the credential is resolved once per leg, and the plugin distributes a SNAPSHOT of it
to workers rather than a refreshable handle — so a leg outliving the token expires everywhere at
once. What is *not* the cause, checked rather than assumed: clock skew, IAM policy, and the object
itself, all of which were ruled out before the guard was written.

### The guard

**Drive the refresh from a timer, not from the work loop.** A background ticker on the runner that
re-fetches and re-broadcasts every ~30 minutes regardless of what the loop is doing removes the
coupling that produced every failure above, and fixes both limbs at once:

- a long date write can no longer outlive its credential, because renewal no longer waits for the
  write to finish
- a late-joining worker inherits a snapshot at most one tick old, so it always starts with ≥30
  minutes of credential

Keep the existing per-date check as the belt to the timer's braces — it costs nothing when the
timer is already keeping the credential fresh.

A reader-side guard (bounding how long an open dataset handle may live across a renewal, or
re-opening on credential change) is the deeper fix, since `rasterio.env.Env` passes the
`AWSSession`'s frozen credentials into GDAL on entry and an already-open handle keeps them. It is
not needed if the timer holds the credential fresh for longer than any single read, and it is the
harder change; the timer comes first.

### What the per-read refresh costs

The optical path took the change one step further — the ROI mask's credentials resolve per READ
rather than freezing once per leg — which is a throughput question, not a correctness one.
Measured 2026-08-05 against an acceptance bar of no worse than 4–5%.

**No regression at the bar: the measured upper bound is 1.33%, and the realistic figure is
lower.** Two components, and only one of them exists:

| component | cost |
|---|---|
| resolving the credential | **3.8 µs per read** — irrelevant, and it was the part under suspicion |
| rebuilding the mask graph per date | **~340 ms per date**, against per-date totals of 200–380 s |

Credential resolution is free because `_resolve_iam_credentials` caches a live refreshable
credential, so each call copies an already-resolved one and touches no network. The only real
cost is the per-date mask graph rebuild, and 340 ms against a date that takes minutes is the
1.33% upper bound.

## Cause 2 — the source object is corrupt (optical)

### How cause 2 was identified

The logs named a `WarpOperationError` with no reason attached, and the object was reproduced
independently outside the pipeline — which is what separated "our reader is wrong" from "the bytes
are bad". The discriminator is below.

### The duplicate item is the discriminator

The STAC catalogue holds **two items per affected tile-date**, and only the higher-sequence one is
broken. Two independent cases, different satellites, different bands, different UTM zones:

| tile-date | item | `s2:sequence` | baseline | tiles ok / bad |
|---|---|---:|---|---|
| 34WFA 2021-09-08 | `S2B_..._0_L2A` | 0 | 03.01 | **121 / 0** |
| 34WFA 2021-09-08 | `S2B_..._1_L2A` | 1 | 05.00 | 41 / **80** |
| 11VPD 2021-08-04 | `S2A_..._0_L2A` | 0 | — | **121 / 0** |
| 11VPD 2021-08-04 | `S2A_..._1_L2A` | 1 | — | 16 / **105** |

The `_0_` sibling is **intact in both cases**, and covers the same tile fully — so the pixels are
not lost, and dropping the broken duplicate costs nothing.

**Caveat on the sample.** Both cases were found *by their corruption*, so this does not establish
that higher-sequence items are broadly corrupt — only that where corruption occurred, the
lower-sequence sibling was the healthy one. An unbiased survey (sampling duplicates not selected on
failure) was attempted and did not converge: reading 121 tiles per asset over HTTPS is slow, and the
`earth-search` STAC API returned 502s under repeated querying.

### A second, independent defect the duplicates expose

The two items carry **different processing baselines** — 03.01 and 05.00, i.e. **either side of
`S2_BASELINE_THRESHOLD = 400`**, which decides whether `S2_BASELINE_OFFSET = -1000` is applied.

`stac._extract_baselines` keys by date:

```python
baselines[date_str] = _extract_baseline(item)   # last item for a date WINS
```

Nothing dedupes items per tile-date, and `odc.stac.load` fuses both items into one solar-day slice
(`fuse_nd_slices` is in the failing traceback). So a duplicated tile-date can hold pixels processed
under both baselines while **one** correction is applied to the fused result — and which one is
decided by item order, not by which item's pixels are present. Part of that date's reflectance is
then wrong by 1000 DN.

This is a correctness defect independent of the corruption, and it is **not fixed** by anything in
this record. Fixing it requires choosing one item per (tile, solar day), which is the same choice
the corruption forces.

### The guard

Deduplicating to one item per (tile, solar day) addresses both the corruption and the baseline
mixing at once, and loses no pixels. **Which item to keep is a data-quality decision, not a
reliability one**, and the two candidate rules disagree:

- **keep the lower sequence** — matches the corruption evidence (2/2), but discards the newer
  processing baseline, which is normally the better data
- **keep the higher sequence** — the usual data-quality preference, and the one that fails here

`odc.stac.load(fail_on_error=False)` is **not** a safe substitute. It would tolerate a corrupt
object, but it would equally swallow the credential 403s of cause 1 — turning a credential outage
into silent, large-scale missing data on dates that then commit and read as complete. If it is ever
used it needs a bound on how much of a date may fail before the date is rejected. The credential
fix must land first regardless.

### It kills legs, and the leg retry feeds them back in

Observed live on 2026-08-05, after the automatic leg retry shipped: **four optical leg attempts
died on these two objects within one hour** — two attempts per zone, the retry re-running each leg
into the same permanently-broken object.

```
17:01, 17:03  almond-parrot, lean-anteater   Failed  WarpOperationError   (11VPD)
17:35, 17:36  knowing-narwhal, straight-aardwolf  Failed  WarpOperationError   (34WFA)
```

Only **four distinct objects** appeared in eight hours of fleet logs, two of them these — so this is
rare per granule and expensive per occurrence, which is the profile that justifies a guard rather
than tolerance.


---

# Cause 3 — the catalogue refuses the query (optical)

**Absorbed 2026-08-18 from `catalogue_refusal_classification_2026_08.md`**, whose own preamble named
this document as stating "the general problem this instance is a second case of": a failure class that
consumes the retry budget as though it were a data failure. Two documents describing one problem, one
of them explicitly a case of the other, is the shape that should be one file — so the third cause now
sits beside the two it shares a mechanism with, and the "Coupling to the leg retry" section below
covers all three.

The distinction that makes it a separate cause rather than a third symptom: causes 1 and 2 are about
an OBJECT the loader could not read, and this one is about a QUERY the catalogue would not answer. The
guard is therefore a different shape — a taxonomy and a wall-clock bound rather than a credential
refresh or a duplicate step-down.
### The failure

A campaign cell's optical ingest leg failed three times, hours apart, with the same error:

```
APIError: HTTPSConnectionPool(host='earth-search.aws.element84.com', port=443):
Max retries exceeded with url: /v1/search
(Caused by ResponseError('too many 502 error responses'))
```

Three attempts hours apart failing identically is deterministic for that query, not congestion.
A neighbouring year built successfully from the same coverage, so the archive publishes the data.

### What the code did about it, before this change

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

### The change

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

### Why a REPEAT, and not the status alone

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

### Bounding elapsed time: `max_leg_wall_clock_s`

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

### A hazard the page-by-page walk introduced, and how it was caught

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

### Statuses the taxonomy deliberately does not name

`400` and other client errors classify as `UNKNOWN` and keep today's expansive retry. They are
deterministic in the request by definition, so this is arguably wrong — but no observed failure
has that shape, and a general-purpose HTTP classifier was explicitly out of scope. A unit test
asserts the two named sets jointly cover `_STAC_RETRY.status_forcelist`, which is the set that
can actually reach us as an exhaustion.

### Fingerprint consequence — a real cost, paid once

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

**Corrected in place on integration.** The claim just above — that this pair "is the only
before/after any store will ever observe" — held only for the branch that wrote it. Every later
change inside the closure moves the identity again, and two have: `main` now computes
`ingcode-48e673cac9b274c3`, and the combined campaign-night branch (causes 4 and 5, the
re-partitioning query, the identity override) computes `ingcode-54f78c9283c72ae3`. Read the pair
above as what THAT change cost, not as a standing value.

That check is validated on **append**, not on the completion marker, so:

- **Finished mosaics are unaffected.** Nothing re-ingests.
- **A mosaic that is mid-ingest right now will refuse its next append** with
  `ConfigMismatchError`, which is a non-retryable leg marker; the resolution is either
  `allow_ingest_code_mismatch` on the resuming run — off by default, and it relaxes only the
  code-identity term — or a human deleting the interrupted store. See
  `staging-identity-and-resume.md` section 5.

There is no way to add request naming to the catalogue query without this, because the query is
inside the closure. It is the cost every ingest change pays.

### What is worth reporting to the archive's operator

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

---
## Cause 4 — the object was never published (optical), 2026-08-20

Traced 2026-08-20 from a live campaign leg. A separate cause rather than a variant of cause 2
because the two differ in exactly the way the guard reads: cause 2 is an object whose BYTES will
not decode, and this is an object that is not there at all. The remedy is the same (step down to
another catalogue copy, and give the date up if there is none); the detection is not.

### The failure

One optical leg abandoned an entire zone-year because **one object out of the 6,424 that day
needed did not exist**:

```
s3://sentinel-s2-l2a/tiles/33/P/WM/2018/1/24/0/B02.jp2
item S2A_33PWM_20180124_0_L2A, zone 33N, 2018-01-24
```

Upstream mechanism, and it is not our bug: Element 84 published no B01/B02 COG for that granule,
so earth-search fills those asset slots with ESA JPEG-2000 hrefs — and those hrefs omit the
`R10m/` path segment, so the href names a key one level above the file that exists. The date is
recoverable by the ladder; the leg was not, because the failure never reached the ladder.

### Why the existing markers missed it

Every marker in `_UNREADABLE_MARKERS` before this change is emitted by a BLOCK READ:
`ZIPDecode`, `TIFFReadEncodedTile`, `IReadBlock failed`, and the two wrappers around them. A
missing object fails at `rasterio.open`, before any block is requested, so none of them appears.
`is_unreadable_source` therefore answered no, `s2_roi` re-raised, and the whole zone-year died
where one date should have been skipped.

What the driver receives is:

```
Exception("RasterioIOError('ObjectNotFound: The specified key does not exist.')")
```

A plain `Exception` carrying a repr — the read runs on a Dask worker and tblib cannot reconstruct
rasterio's GDAL-backed classes across the boundary. **An `isinstance` check against a rasterio or
GDAL type is therefore not available here**, however much it would be the better test. Text is the
only evidence that survives, which is why the fix is three markers — `ObjectNotFound` (GDAL and
rasterio), `NoSuchKey` (the S3 API error code), and `The specified key does not exist` (the
response body) — covering the three layers that can be the one to surface it.

### Not-found alone does not say whose object it was

Those three strings belong to the S3 layer, and **every S3 client in the process shares that
layer**. Checked against the installed wheel: `icechunk`'s error enum carries `ObjectNotFound` and
`NoSuchKey` verbatim, so the DESTINATION store speaks the same vocabulary the source does. Matched
on their own they would let a hole in the destination repository, or in the ROI-mask store, be
classified as source data loss — recorded durably on the store as a finding about the provider, and
a date given up for a fault that giving it up cannot route around.

So the missing-object markers only count when the text ALSO names the source reader
(`RasterioIOError`, or a `CPLE_` class). GDAL is used here only to read source imagery: the
destination is Zarr and Icechunk, and the ROI mask is read through `da.from_zarr`, neither of which
goes through it. Unpaired, the exception is re-raised — the behaviour that predates this marker
class, and the safe direction to fail in. The decode markers above need no such pairing; only GDAL
emits them at all.

**Corrected in place on integration.** This section described the pairing set as `RasterioIOError`
and `CPLE_`; cause 5 needed the same pairing for a different marker class and described it as those
two plus `WarpOperationError`. They are now **one** `_SOURCE_READER_MARKERS` tuple, shared by
both predicates, holding only the reader's own vocabulary: `RasterioIOError`,
`WarpOperationError`, `CPLE_`.

**CORRECTED 2026-08-22.** That tuple briefly also held `HTTP response code:`, and this paragraph
claimed the benefit was that a missing object reported as `HTTP response code: 404` would then
pair. It was a defect, not a benefit: the refusal markers spell 403, 500, 502, 503 and 504 with
that same prefix, so one message became both the refusal and its own corroboration, and a fault
from the destination store carrying that text was attributed to the source. The prefix is out of
the reader markers. The 404 case it was reaching for is handled where it belongs instead —
`HTTP response code: 404` is now one of the MISSING-OBJECT markers, which still needs an
independent reader marker beside it. That matters more than it sounds: the optical assets are
plain `https://` hrefs, so GDAL reads them through its HTTP driver and reports a missing object
as a bare status rather than as `ObjectNotFound`, and without the 404 form the optical step-down
would not have fired on the shape that path actually produces.

### The leg was not even retried, for an unrelated reason

`_NON_RETRYABLE_LEG_MARKERS` in `ingest_zone_year.py` already contained `ObjectNotFound`, listed
there for a completely different meaning: **Prefect's** `ObjectNotFound`, raised when a child
deployment is not registered. The two names collide, so a source object's 404 matched the marker
and the zone-year was classified deterministic and never re-dispatched. The classification was
right by accident — a retry would have read the same absent object — but nothing in the code knew
that, and the collision is still live for any future S3 404 reaching that classifier.

### The skip, and what the downstream gate does and does not catch

**Corrected in place 2026-08-20/21. Two claims withdrawn.** This section previously said that "one
recorded unreadable date **excuses its whole month** from the downstream coverage gate, so k
recorded dates can excuse k months", and derived a ceiling of 6 from it. It then said the opposite
overshoot: that a leg losing many dates **would inevitably fail** that gate. Neither is right, and
the ceiling built on the first is gone from the code.

Read `inference/data_loading.py` around the gate rather than the helper in isolation:

```python
missing      = sorted(required_months - present_months)
_lost_months = _months_holding_unreadable_dates(_raw_unreadable)
_explained   = _months_within_assessed(missing, _assessed) if missing else set()
_explained   = (_explained - _lost_months) if _lost_months is not None else set()
```

Two facts, and both matter:

1. **The gate only ever looks at WHOLLY ABSENT months.** `missing` is
   `required_months - present_months`. A month that keeps even one acquisition is never examined,
   so dates lost from it are invisible to the gate — its depth is quietly reduced and nothing
   downstream objects.
2. **Where the gate does look, a recorded loss makes it STRICTER.** `_lost_months` is *subtracted*
   from the excused set, so a given-up date disqualifies its month from being excused, what remains
   in `missing` raises `InsufficientCoverageError`, and the comment above that code says why:
   "Excusing it would let a write-once year publish a whole-month data-loss hole labelled a
   legitimate absence." `tests/unit/test_time_window.py::test_a_month_lost_to_unreadable_imagery_is_not_excused`
   had pinned that all along.

So the two cases land differently:

- **every date in a month lost** — the month is empty AND holds given-up dates, so it is not
  excused, stays in `missing`, and the cell fails with `InsufficientCoverageError` before
  publishing anything.
- **some dates lost from a month that keeps others** — nothing per-month sees it. The only rule
  that does is per pixel: `config.inference.OPTICAL_MIN_OBS` refuses a pixel with too few valid
  optical observations in the year rather than publishing it thin.

A vanished bucket reaches neither case, because `NoSuchBucket` is deliberately NOT matched: a
bucket or prefix that has gone is systemic by construction and every remaining date fails
identically, so it fails the leg on the first date rather than being skipped date by date.

**The owner's decision, 2026-08-20/21: skip the date, keep the leg, accept the loss of depth the
second case implies, and add nothing else.** No count, no ceiling, no new failure mode. At the rate
below, any bound would only ever fire on a fault of some other kind — and every date given up is
already logged at error level and written to the store's `assessed_unreadable_dates`, so a leg that
lost an implausible number of them says so on the record without a guard to announce it.

### The observed rate — why this is low priority

Measured 2026-08-20/21 on the live campaign, and the reason the fix is deferred rather than rushed:

| measurement | value |
|---|---|
| zone-years abandoned by this defect | **1** |
| zone-years attempted | ~244 |
| recurrence in the 91 minutes after, at 60 concurrent cells | **none** |
| required files probed across ten sample days | 100,320, of which **exactly one** missing |

The existing shelf-and-second-wave machinery absorbs a failure at that rate, which is why the
campaign was left to run rather than stopped for it, and why this fix waited for a batch rather
than going in on its own.

### A docstring that is broader than its code

`_months_holding_unreadable_dates`'s prose says "a month whose EVERY acquisition was skipped as
unreadable", while the implementation adds the month of **ANY** given-up date. Recorded, not
changed. The breadth is currently inert: the set is only ever intersected with, or subtracted from,
months already known to be WHOLLY absent, so a month that kept a date is never in play. A future
caller using the helper on its own would inherit the wider meaning without the docstring warning
it.

## Cause 5 — the provider refuses reads for a while (radar), 2026-08-21

Absorbed here rather than given its own document, because it is a fifth thing the same
`WarpOperationError('Chunk and warp failed')` text hides.

| | source | preceded by | retryable? |
|---|---|---|---|
| **Provider refusing reads** | `asf-cumulus-prod-opera-products` (radar) | `HTTP response code: 403` / `AccessDenied`, fleet-wide and simultaneous | **yes** — by waiting, not by trying harder |

### The failure

Every Sentinel-1 leg in the global campaign failed: **178 failed, zero completed**, Sentinel-2
unaffected throughout. 1,565 403 / AccessDenied exceptions landed in a single ten-minute bin
(22:50 UTC) with zero in the bins either side. ASF refused reads for about six minutes at full
rate and roughly seven more decaying, then recovered on its own. Of the 178, 86 died inside the
window and 92 in its aftermath — re-dispatches walking back into a source still refusing.

**The refusal is not the defect. The absence of any containment around it is.** Every OPERA read
on the radar path happens inside a date's write, so an exhausted read raises out of the per-date
loop and out of the leg: a leg ninety dates into its year, holding sound committed data, lost
every remaining date because one date's read was refused. The optical path had had this
containment since the corrupt-object work; radar had none of it (`is_unreadable_source`: four
occurrences in `s2_roi.py`, zero in `s1_roi.py`).

Nothing was corrupted. All 120 radar stores: time axis strictly monotonic, zero duplicate
instants, median gap 1 day, 10,840 committed dates, each stopping cleanly at day 60–125. Every
one of those dates was resumable. The legs simply stopped.

### Two traps worth not re-hitting

- **Counting GDAL's `HTTP error code for ...: NNN.` warning finds zero 403s.** GDAL
  logs-and-retries a 503 but raises immediately on a 403 *without emitting that line*. Count
  `HTTP response code: 403` and `AccessDenied` instead.
- **The Prefect flow-run logs hold only cluster setup.** The work logs are in CloudWatch log
  group `/ecs/global-tessera-prod`. Sampling Prefect logs for 403s finds nothing, and the nothing
  means nothing.

It was also **not credential pinning**, which was the first hypothesis: the credential in force
across the wave was valid with a wide margin, the refusals were simultaneous fleet-wide (one
upstream service, not 178 independent credential events), and ASF recovering unaided is not
something a pinning defect explains.

### The change

`ingest/duplicates.py` gains `is_provider_refusal`, beside `is_unreadable_source` rather than
inside it — an unreadable object wants a different copy or a give-up, a refusal wants only to be
waited out. `ingest/s1_roi.py` gains the bounded skip: retry, then give up the one date, record
it on the store as `assessed_unreadable_dates` with `scope=provider-refused` or
`scope=unreadable`, and stop past `MAX_GIVEN_UP_DATES = 10`.

Two properties carry the design:

- **A refusal is only the provider's if the SOURCE READER was refused.** `AccessDenied`,
  `SlowDown` and `InternalError` are S3's words and every S3 client shares them — our own store
  answers `AccessDenied` when icechunk picks up the OPERA-scoped token instead of the role, which
  `storage/zarr_store.py` documents. Since the read and the write both happen inside
  `write_day_windows`, the text is all there is to separate them, so the refusal markers only
  count alongside GDAL's own vocabulary (`RasterioIOError`, `WarpOperationError`, `CPLE_`,
  `HTTP response code:`). GDAL reads source imagery here and nothing else. Unpaired, the
  exception is re-raised, so a destination fault fails the leg on its first date rather than
  being absorbed one date at a time.
- **The ceiling is RETRYABLE**, and deliberately absent from
  `ingest_zone_year._NON_RETRYABLE_LEG_MARKERS`. This is the opposite verdict from the one an
  unreadable object deserves. A refusal clears, so every date given up while one lasts is a date
  a later run would have written — and a leg that *finishes* returns success, so nothing comes
  back for it. Stopping is a request to be re-dispatched, and the dates it names are then
  written rather than lost. That is also why ten is small: a radar zone-year holds a couple of
  hundred dates per orbit.

### An asymmetry in `is_unreadable_source`, found while testing and NOT fixed here

It excludes refusals by NAME (`AccessDenied`, `SlowDown`), so a refusal reported as a bare status
code carries none of those words and the wrapper's decode marker decides — a
`WarpOperationError('Chunk and warp failed')` caused by `HTTP response code: 403` returns
**True**. On the optical path that predicate gates the copy ladder, so a transient numeric 403
can step a tile-date down to an older baseline: exactly what the predicate documents itself as
preventing. The radar path resolves it by ORDER — `_give_up_date` asks about the refusal first.

Left alone because widening that tuple changes optical behaviour during a live campaign and the
direction is not obviously safe: a 403 that currently steps down and succeeds would instead
propagate. It wants its own change with the optical ladder's blast radius measured first.

### A credential refresh strips its own task's GDAL configuration — measured, and FIXED here

Reproduced deterministically against the installed `odc.loader` and `rasterio`.
`auth._patch_odc_thread_session_for_env_drift` responds to a credential change by calling
`ThreadSession.reset()`, which calls `rasterio.env.delenv()` — from inside `rio_env()`, which
`_rio_read` enters *while a dataset is open*, inside the outer `rio_env` that
`RioDriver.restore_env` wraps around a whole Dask chunk task. The inner `Env` then believes it is
outermost and on exit restores a freshly-defaulted option set.

`rasterio.open` self-heals its credentials from `os.environ`, so this is not itself an
authorization failure. The cost is the lost GDAL configuration: **a task that has been through a
refresh has no GDAL HTTP retries (`GDAL_HTTP_MAX_RETRY=10`) for the rest of its life**, and
re-lists a directory on every open. Losing ten GDAL retries is directly relevant to surviving a
source that is briefly refusing.

**Corrected in place on integration.** This section said "NOT fixed here", because the fix was a
separate change on a separate branch. Both are in this branch now, so read the paragraph above as
the measurement and this as the resolution: the drift branch in
`auth._patch_odc_thread_session_for_env_drift` clears the cached session directly
(`self._session = None`, `self._aws = None`) instead of calling `ThreadSession.reset()`, so
`rasterio.env.delenv()` is never reached from inside `rio_env()` and the outer environment's
options — the HTTP retry ladder among them — survive the refresh. `reset()` is unchanged at its
other two call sites, neither of which is nested inside `rio_env()`. A task that has been through a
refresh therefore keeps its ten GDAL retries, which is the containment this cause's bounded skip
sits on top of.

## Coupling to the leg retry

`orchestration/prefect/flows/ingest_zone_year.py` retries a failed leg up to
`ingest_settings.max_leg_attempts` (3) unless the failure matches
`_NON_RETRYABLE_LEG_MARKERS`. That list does **not** mention the warp failure, and it cannot
usefully be made to:

- a **403-driven** warp failure is retryable and *should* consume attempts
- a **corrupt-object** warp failure will burn all three attempts and every hour of fleet time they
  cost, and can never succeed

Because the wrapper discards the inner cause, the exception text is identical in both cases, so
**no marker on the message can separate them.** The discriminators that do work are elsewhere:

- **the object URL** — a corrupt object fails on the *same* URL every attempt, while a credential
  expiry fails *many* URLs at once
- **`response_code=403` in the preceding log line** — present for cause 1, absent for cause 2

So the classification has to live where the read fails, not where the leg is retried. Capturing
the inner GDAL cause and the object URL into the failure detail is the prerequisite for ever
letting the retry classifier act on this at all.

**A second instance of this shape, since resolved for the catalogue query rather than the read:**
the section above. The optical leg's STAC search was refused
deterministically (repeated 502s from `earth-search`) and burned the same budget for the same
reason — the wrapper discarded the request, so no marker could separate a busy catalogue from one
that cannot serve a particular query. The fix follows the prescription above: classify where the
failure happens, carry the request identity into the failure detail, and let the layer holding the
budget act on a REPEAT of it. Note that the "Caveat on the sample" 502s above are the first
recorded observation of that same upstream defect.

**A third instance, resolved for the radar read itself:** Cause 5 above. The prescription is the
same and was followed literally — `is_provider_refusal` classifies at the point the read fails,
the verdict is carried into the store's own record as `scope`, and the layer holding the budget
acts on the count rather than on the message. What is new is the direction of the verdict: the
error the ceiling raises is left OUT of `_NON_RETRYABLE_LEG_MARKERS` on purpose, because a
refusal clears and the retry is what recovers the dates.

## Why the object was hard to identify — an attribution gap, now closed

No `READ FAILED roi=` line existed for any of these failures, despite `read_failure_context`
wrapping the optical coverage gate. The gate computes **only SCL**, so a broken *reflectance* band
passes the gate and fails later, in the write's compute — and the optical write was the one compute
in either sensor path with no attribution context around it. The radar write had one; the gate had
one; this did not.

So the object's identity existed only in odc's own `Aborting load` line on some worker's stream,
and finding it meant correlating by timestamp across the whole fleet. The optical write is now
wrapped and carries the day's items, so the failure names the granule directly.

The general shape, which is the part worth keeping: **an attribution context has to wrap every
compute that reads source pixels, not just the first one.** A lazy graph moves the read to wherever
it is finally computed, and a gate that computes a subset of bands only protects that subset.

## Method notes

- `WarpOperationError`, `ZIPDecode`, and `Chunk and warp failed` each appear **several times per
  actual failure** (the pickled payload, the traceback line, the `Exception:` line, the Prefect
  retry warning). Raw counts of those strings are inflated ~3–5× and are only safe to read as
  zero-versus-nonzero. `Aborting load due to failure while reading` is emitted **once per failure**
  and is the countable line.
- The failing source URL appears only on the `Aborting load` line and the GDAL `ERROR 1` line —
  never inside the exception. Any attribution to a granule has to come from those.
- CloudWatch Logs Insights `like` is case-sensitive; see the `querying-logs-and-apis` playbook.
