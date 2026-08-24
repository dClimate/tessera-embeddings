# Ingest source-read failures: nine causes, and the retry budget they share

Investigation record. Causes 1 and 2 traced 2026-08-04/05 on `global-tessera-dev`; cause 3 absorbed
2026-08-18 from its own document, for the reason given at that section; cause 4 traced 2026-08-20,
cause 5 on 2026-08-21, causes 6 and 7 on 2026-08-22/23, and causes 8 and 9 on 2026-08-24, all from
live campaign legs. **Corrected in place: this document said "three causes" until cause 4 was
added, "four causes" until cause 5 joined it, "five causes" until cause 6 did, "six causes" until
cause 7 did, and "eight causes" until cause 9 did.** Cause 9 is the second half of cause 8 rather
than a new mechanism: the same refusal, classified correctly and then given a budget that could
not act on the classification. Cause 7 is the
only one here that is not a source failure — it is our own bookkeeping, and the only one that left
holes nothing can fill. Traces
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

**And for a while it could not tell them apart from the CAUSE either, which is the deeper
finding.** Every predicate here reads the cause chain, and the chain was being destroyed on its
way out of the Dask worker — so four of the causes below were diagnosed as gaps in a marker list
when three of them were one architectural loss in different clothes. See "Why the reason was hard
to read" at the end; read it before widening any predicate in `ingest/duplicates.py`.

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
| leg | `IngestSettings.max_leg_attempts` | 3 | retryable — no marker matches an exhausted-ladder `APIError`, and the default is TRUE. **Corrected in place:** this row said "`_NON_RETRYABLE_LEG_MARKERS` names no HTTP failure", which stopped being the whole test when the read-failure taxonomy joined the classifier (see "When the words DO carry the verdict"). The verdict is unchanged: the taxonomy reads a status only in GDAL's `HTTP response code: NNN` form, which this text does not carry |
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
asserts the named sets jointly cover the ladder's force-list, which is the set that can actually
reach us as an exhaustion.

**Corrected in place, 2026-08-23.** That test read a module-level `_STAC_RETRY` constant, and
`403` has since become a fourth case: named for one provider and deliberately unnamed for the
rest (cause 6). There is no longer one ladder to read, so the invariant is checked per provider
against `stac._retry_for(provider)` — a status a catalogue is waited out for and the taxonomy
cannot name is the same silent failure whether the retry is global or scoped.

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

A plain `Exception` carrying a repr. **Corrected in place 2026-08-24: this said an `isinstance`
check "is therefore not available here" and that "text is the only evidence that survives". Both
were wrong, and see the evidence-gap section at the end for why** — the chain was reconstructable
all along, and the reason it did not arrive was a single unpicklable class. The fix below is still
three markers, and still correct: the markers are matched against the CHAIN, which now has one.

The three are `ObjectNotFound` (GDAL and rasterio), `NoSuchKey` (the S3 API error code), and
`The specified key does not exist` (the response body) — covering the three layers that can be the
one to surface it.

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

### An asymmetry in `is_unreadable_source` — deferred here, and it cost eleven stores

> **FIXED in Cause 8, 2026-08-24.** Read the deferral below as written, because the diagnosis was
> complete and correct and the decision not to act on it is the most instructive thing in this
> document. Three days later the same asymmetry — a numeric refusal claimed by
> `is_unreadable_source` on a path that never asks about refusals — made **eleven optical stores
> unfillable** and stopped the campaign. All 120 radar stores were sound, exactly as the paragraph
> below predicts, because radar resolves it by order.
>
> The deferral reasoning was not unreasonable: widening a classification during a live campaign is a
> real risk, and the direction genuinely was not obviously safe. What was missing is the other half
> of the trade. The cost of ACTING was estimated ("changes optical behaviour during a live
> campaign"); the cost of NOT acting was never estimated, and it was a stopped campaign, eleven
> deleted stores and a day of remediation.
>
> **The habit to take from it:** when a known defect is deferred, write down what it costs to leave
> it, not only what it costs to fix it. And note this was the THIRD written record of the same
> asymmetry — the marker tuple's own comment and
> `test_a_numeric_refusal_is_claimed_by_both_predicates`'s docstring were the other two. A defect
> documented three times and fixed zero times is not a known risk being managed; it is a decision
> nobody re-opened.



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

## Cause 6 — the catalogue refuses the RATE (optical), 2026-08-22/23

The catalogue analogue of cause 5. Same status, opposite end of the pipeline: cause 5 is a
provider refusing OBJECT READS on the radar path, this is a catalogue refusing SEARCH REQUESTS on
the optical one.

| | source | preceded by | retryable? |
|---|---|---|---|
| **Catalogue refusing the rate** | Earth Search (Element 84), `sentinel-2-l2a` | `HTTP 403 {"message":"Forbidden"}` on `POST /v1/search` | **yes** — by waiting, not by asking differently |

### The failure

16 flow runs failed in a 2.5-minute burst about fifteen minutes after the campaign started from a
freshly wiped store, every one a Sentinel-2 catalogue query refused with 403. Sentinel-1 was
entirely spared: 13,764 CMR query lines, zero refusals — so this is one catalogue's constraint,
not a network fault. Nine leg failures in total, every one re-dispatched, no cell lost. Survivable
but degrading.

**The request is not the problem; the aggregate instantaneous rate is.** One of the exact failing
requests — same bbox, same date window, same page size of 100 — replayed from a single sequential
client walked **95 pages, every one HTTP 200, in 68 seconds** at 2.1 MB per page. Page depth is
not the mechanism either: the fleet's 403s span pages 1 through 94.

### Corrected in place: it is not a cold-start phenomenon

This was first read as a single burst caused by phase alignment — 60 cells started at one instant,
each spending about as long provisioning, all arriving at the catalogue together, where previous
campaigns at the same 60-cell width had been resumed runs whose cells sat at random phases. **That
was measured on the wrong needle.** Counting the `CATALOGUE REFUSED` log line found one episode;
counting `CATALOGUE_REFUSAL=` — the `REFUSAL_TOKEN` the code says to match by name — finds three:

| bin (UTC) | events | pages |
|---|---|---|
| 23:20 | 96 | 40–93 |
| 23:35 | 6 | 43 |
| 23:40 | 6 | 43 |

And zone 47N failed **24 minutes into a healthy leg**: it had committed 2017-01-25 through
2017-01-31 normally, then died querying the next window (`2017-01-31/2017-03-01`, page 43). Phase
alignment explains the opening peak and nothing after it.

**The two needles disagree in both directions, and that is a measurement hazard, not a code
defect.** The ERROR line yielded 446 events where the token yielded 96, and zero for the 23:35 and
23:40 episodes the token found. They read different populations: the line is emitted once per
refusal reaching `raise_catalogue_query_error` and lands in the application log, while the token
travels inside the exception MESSAGE and is re-logged by the leg loop at up to five sites per
attempt (`_run_leg`'s re-dispatch, load-refusal, doomed, wall-clock and terminal branches all
interpolate the failure detail). Neither count is the number of refusals. **Use the token for
existence and the line for per-request forensics, and do not compare their totals.** Not fixed
here — it needs the log lines restructured, which is not a mid-incident change.

### Why no lever engaged

`403` was in none of the three status sets in `ingest/catalogue_refusal.py`, so `_kind_for(403)`
returned `UNKNOWN`, `is_oversized_response` was False, and neither recovery lever ran. It was also
absent from the urllib3 force-list, so it arrived on the FIRST refusal with no backoff at all and
raised straight out of the leg. That is the literal content of "without being retried" in the
failure text: a leg representing hours of work and a live Dask fleet, ended by a refusal that
clears on its own in seconds.

### The change, and why it is provider-scoped

`THROTTLE_STATUSES = frozenset({403})`, deliberately NOT added to `LOAD_REFUSAL_STATUSES`.
`STACProvider` gains `throttles_with_forbidden`, set only on `earth-search`; `stac._retry_statuses`
builds each provider's force-list from its flags, and `stac._throttle_statuses` feeds the same flag
into the classification so the ladder and the taxonomy cannot fall out of step.

The scoping is the whole point. Earth Search is public and unauthenticated — we send it no
credential — so a 403 from it cannot be a statement about who is asking. Everywhere else it is
exactly that, and an authorization verdict on a backoff ladder is pure cost: **364 s of measured
backoff** per refused request (`total=8`, `backoff_factor=2`, urllib3's `DEFAULT_BACKOFF_MAX` of
120 capping the tail at 2, 4, 8, 16, 32, 64, 120, 120), spent to learn something already known.
This mirrors `refuses_oversized_pages`, which scopes the 502 exclusion the same way.

**Bounding the pair rather than each lever.** The two existing recovery levers — window re-cut and
page-size halving — both gate on `is_oversized_response`, which is `status in {502}`. A 403
therefore cannot reach either, so it cannot hand a ladder to a recursion of child windows the way
a force-listed 500 would. Its worst case is the ladder's own 364 s per page request, and
`max_leg_wall_clock_s` (6 h) bounds the leg above that. A unit test pins the non-interaction, since
adding 403 to `OVERSIZED_RESPONSE_STATUSES` later would be a silent multiplication.

### Corrected in place, again: the mechanism is the WINDOW-WALK CONCURRENCY

The section above blamed phase alignment, and phase alignment is real but it is not what
made this campaign different. `stac._QUERY_WINDOW_WORKERS` walks each leg's catalogue date
windows **six at a time**. At 60 cells that is roughly **360 concurrent search streams**
against one provider for the same total request volume, and **221 were measured in flight**
at the moment Earth Search began refusing. The campaign of 19 August, which ran at the same
fleet width before this concurrency existed, refused nothing of any kind.

The setting's own comment predicted exactly this — it calls itself "a MULTIPLIER on the
concurrent search streams Element 84 sees, and what they answer an overload with ... is a load
refusal the whole leg then waits out". The number was the error, not the reasoning.

**Lowered to 2.** Not to 1: a walk is idle for almost all of its wall clock (the latency split
is in the setting's comment), so overlap is the only lever on a leg's query time and
serialising it would be a large throughput regression for a problem a small overlap solves.
It stays a module constant rather than a setting because nothing in `ingest/` reads
`IngestSettings` and the query is reached from three call sites that would each have to carry
one; tuning it in an incident means a release, and that is the trade accepted.

### The stagger, and what it does not fix

`leg_stagger_window_s` (default 600 s) delays each leg's first dispatch by an offset derived from
`sha256(zone/year/label)` — deterministic so a test can assert it, and not `random` or `hash`,
neither of which is reproducible inside a worker. At the campaign's width that puts legs roughly
5–10 s apart, which is the separation the fleet needed.

**It addresses the opening peak only.** It cannot help a leg that starts a new window mid-run,
which is what zone 47N did, and it should not be described as the fix for this cause — the ladder
is. It is also paid as latency on every leg's first dispatch, not only on a cold start.

## Cause 7 — a skipped date whose record died with the process, 2026-08-23

Not a source failure at all, and the only cause here that produced **unfillable holes in
committed stores**. Independent of the 403s; found while investigating them.

| | source | preceded by | retryable? |
|---|---|---|---|
| **A lost date offered twice** | our own bookkeeping | `DATA LOSS roi=... date=...: every catalogue copy failed to read` | **no** — the store can no longer be completed |

### The mechanism

When every catalogue copy of a date fails to read, `s2_roi._record_unreadable` skips the date
deliberately — losing one date beats losing every later date — and later dates commit normally.
The loss went into an in-memory list that reached durability only at the single
`record_assessed_window` call **after the whole drive loop**, at the very end of the leg.

If the leg died before that line, for any reason, **the record was gone.** The next attempt
rebuilds its outstanding work from what was WRITTEN, finds the date absent, and offers it
again. If a copy reads that time, the append is refused by the monotonic guard — later dates
are already on the axis, and the axis is sampled positionally so it cannot be inserted into.
The cell can then never progress and the date can never be filled in place.

`_record_skip`'s own docstring anticipated it: *"a hole nobody recorded is a hole no later run
revisits."* The record it protects simply did not survive a mid-leg death. The coverage path
was weaker still — it incremented an in-memory counter and left no trace anywhere.

### Evidence

Four `DATA LOSS` events. **Five stores hold an unfillable hole**: 34N/2017 and 32N/2017
(2017-01-18), 35N/2017 (2017-02-01), 43N/2017 (2017-02-11), 33N/2018 (2018-01-24). Only two of
the five had actually raised the error — the campaign was stopped before the rest were retried
— so the error count understates the damage, and that silence is what makes it dangerous.

### The change — and the durable record was BUILT, then REMOVED

**The append refusal stays FATAL.** Softening it was considered and rejected: a leg that reported
the refusal and carried on would complete the cell with a hole, and the parent reads a completed
cell as success, so a silently degraded store would ship. A log line is not a guard, because
nothing downstream reads one. There is also a cost argument — a date that cannot be appended in
place means the store must be wiped and re-ingested anyway, so continuing spends compute on a
store already destined for deletion. It now names the store as well as the date, because the
remedy is per store.

**A crash-durable skip record was built for this, and then deleted.** It wrote the loss onto the
store at the moment of the skip so a resume would not re-offer the date. Two independent reviewers
found five root causes in it across three review rounds, and three were new defects it had
introduced — including two fresh ways to wedge a store permanently: an attribute-only root created
in a rootless repository, and coverage rejections carried into the unreadable-dates attribute,
which the inference layer reads as imagery loss.

**It kept failing because it was compensating for Cause 8.** Its purpose was to remember a skip
decision *because that decision might not recompute* — and that was only possible because transient
failures were being classified as skips. Once a refusal can no longer be read as unreadable data,
every remaining skip cause is deterministic: unreadable bytes, a coverage rejection, no live
window, a corroborated absent object. A resume that re-offers such a date simply skips it again.
**There is nothing to remember.**

The original design said so before any of this was added: *"re-offering it costs one re-evaluation
and cannot wedge anything."* That was right, and became wrong only because transients were leaking
into the skip path. So the record is gone, and `record_assessed_window` at end-of-leg is unchanged —
it serves the other purpose, telling the coverage gate a window was examined, and a leg that dies
leaving no assessment correctly reads as never having got there.

## Cause 8 — a provider's refusal read as unreadable DATA, 2026-08-24

**The root cause of the eleven unfillable stores.** Cause 7 above describes how a hole forms once a
date is skipped; this is why the date was skipped at all, and it is the defect that was actually
fixed.

### The mechanism

Two predicates in `ingest/duplicates.py` decided a read failure and they OVERLAPPED.
`is_provider_refusal` recognised statements about the SERVICE — `AccessDenied`, `SlowDown`,
`ServiceUnavailable`, `InternalError`, HTTP 403/500/502/503/504. `is_unreadable_source` recognised
statements about the BYTES. A refusal reaches the reader through a block-read wrapper, so its chain
carries **both kinds of signature**, and `is_unreadable_source` excluded only credentials plus
`AccessDenied` and `SlowDown` — not the numeric statuses.

So the verdict came down to which predicate a caller asked first. `s1_roi.py` asks about refusals
first. **`s2_roi.py` never called `is_provider_refusal` at all** — it does not import it; all three
of its sites are `if not is_unreadable_source(exc): raise`. The optical path therefore had no
concept of "the provider refused, wait", only "the data is broken, give up this date".

That is precisely the damage pattern: **all eleven unfillable stores are `reflectance.zarr`, and all
120 radar stores were sound.**

### Scale, and the patience

The campaign window logged **5,687 occurrences of `503`, 785 `Connection reset`, 80 `Broken pipe`** —
the provider was throttling object reads as well as searches. Against that, `SOURCE_READ_ATTEMPTS`
was 3 with backoff of 2 s then 4 s: **about six seconds of patience** before a failure at that layer
decided a date was permanently lost.

### It was known, and pinned as the contract

`test_a_numeric_refusal_is_claimed_by_both_predicates` asserted the overlap, with the docstring:
*"Not fixed here: widening that exclusion changes the optical copy ladder mid-campaign. The radar
caller resolves it by ORDER, asking about the refusal first."* The defect was identified, not fixed
because fixing it would change the optical path mid-campaign, and locked in by a passing test.

### The change, and the shape that mattered more than the fix

The immediate fix was to make the predicates disjoint by construction rather than by call order.
What followed was three more findings of the SAME shape, and they are the reason this cause is worth
reading rather than skimming:

1. **429 was missing** from the refusal markers — the most explicit "slow down" a provider can send.
2. **Every non-enumerated status** was unmatched: 400, 401, 507, 509. A malformed request or a
   rejected credential was recorded as corrupt imagery.
3. **Statusless transport failures** — DNS failure, refused connection, empty reply, TLS error —
   carried no status and were in no list.

Patching each would have been four patches of one shape. **The defect was the POLARITY.**
`is_unreadable_source` returning True means "give up this date", and it was the FALLBACK for
anything the transient lists did not recognise — so every gap in those lists became a silent
data-loss verdict, by construction.

So the resolution was two deletions and a re-shape:

- **Statuses by RANGE, not enumeration.** Any 5xx is transient; 408 and 429 are the transient 4xx;
  404 and 410 are absence; every other 4xx is neither and re-raises. 403 stays named — the one
  judgement about a provider rather than about HTTP.
- **The generic wrappers came out of the unreadable set.** `Chunk and warp failed`,
  `Read failed. See previous exception` and `IReadBlock failed` are what GDAL raises when a block
  read fails FOR ANY REASON. What remains names a codec: `ZIPDecode`, `TIFFReadEncodedTile`. An
  unrecognised failure now re-raises and the leg retries in order.
- `SOURCE_READ_ATTEMPTS` 3 to 8 — about 61 s of backoff, so a bad minute is outlasted.
- **Radar aligned to optical.** It accepted a refusal as grounds to give up a date under
  `scope="provider-refused"`. Same hole, reached by accepting the refusal instead of misclassifying
  it. `is_provider_refusal` is deleted; it had no caller left.

The invariant this restores, and the one to hold the design to: **any missing data is
deterministically missing.** A skip is legitimate only when it recomputes to the same verdict on
every attempt. That is what makes an absence explainable rather than a function of when the leg
happened to run — and it is why Cause 7's durable record turned out to be unnecessary.

### Verified

Reproduced deliberately on `global-tessera-dev`, cell 43N/2017: a leg killed mid-run with three
interior gaps below its axis maximum, resumed, re-offered all three, skipped them again, and
advanced past the old maximum with zero `NonMonotonicDateError`. The polarisation filter ran 134 CMR
queries with zero skip warnings, against 115,276 in 45 minutes of the production run.

## Cause 9 — the refusal was recognised, and then not waited out, 2026-08-24

Cause 8 stopped a refusal being recorded as unreadable data. It left the refusal with nowhere to
go: correctly classified, correctly declined as a give-up, and then handed to a retry an order of
magnitude too short to be the answer.

### The failure

**85 radar legs failed across 53 distinct cells inside two five-minute bins**, 16:20–16:26 UTC, after
seventy minutes of clean running. Sentinel-2 was untouched throughout — S1 reads ASF, S2 reads
Element 84, and only the ASF-backed sensor moved, which is the signature of a provider event
rather than of anything of ours. Alongside them: **1,304 `AccessDenied` and 253 HTTP 403 log
lines, all from the 16:20 bin onward and zero before it.**

Three explanations were ruled out rather than argued away:

- **Not corrupt data.** The refusal is an authorization verdict on ASF's own bucket, across 53
  unrelated cells and many years at once.
- **Not credential expiry.** An expired credential answers `ExpiredToken` or
  `InvalidAccessKeyId`; **zero of each** were logged. The credential ticker recorded **zero**
  refresh failures and **zero** unparseable expiries, with **766 acquisitions and 762
  broadcasts** in the window.
- **Not the documented own-bucket mix-up.** OPERA-scoped env credentials leaking into icechunk
  operations do produce `AccessDenied`, but on OUR bucket. The resource named is
  `asf-cumulus-prod-opera-products`.

### The defect: every budget was an order of magnitude short

| layer | budget |
|---|---|
| ASF refused reads for | **~360 s** (16:20 → 16:26) |
| `store_write_retrying`, 3 attempts, `wait_exponential(min=2, max=8)` | **~6 s of backoff** |
| the leg retry, 3 attempts at 30 s then 60 s | **90 s**, and each attempt walks back into the refusal |

**Credentials were ruled out a second way, independently of the log counts.** Radar legs dispatched
at 10:29 local were still Running at 1 h 18 m elapsed — past the one-hour credential TTL — so the
ticker was refreshing and reads were working on refreshed credentials. The credential path is
sound and is not where any of this belongs.

Radar's source read happens *inside* the write's compute, so `store_write_retrying` is the ladder
that covers it — `source_read_retrying` is optical-only. A refusal therefore walked through all
three layers and killed the cell, with every date the leg had already committed still in the store
and no way to add to it.

The 2026-08-21 event in Cause 5 sets the target: **about six minutes at full rate and roughly
seven more decaying.** Six seconds against six minutes is not a budget that needs raising, it is
the wrong instrument.

### The change: two timescales, because WHERE you wait decides what it costs

The obvious fix — one long in-leg wait — is the expensive one. A write waits with its leg's whole
Dask fleet held idle behind it, and a campaign cell is roughly 356 vCPU: on the order of a quarter
of a dollar per minute per cell, about fourteen dollars a minute across sixty. **Failing the leg
releases the fleet**, and the cell's own retry then waits for nothing. So the patience is split:

| | where | fleet held? | budget |
|---|---|---|---|
| **in-leg** | `storage/zarr_store.store_write_retrying`, via a new `wait_out` predicate | **yes** | `WAIT_OUT_BACKOFF_S` = 300 s |
| **between legs** | `ingest_zone_year._leg_backoff_s`, via a new `leg_refusal_backoff_s` | no | 600 s then 1,200 s |

`is_provider_refusal` is restored in `ingest/duplicates.py` — deleted in Cause 8 as having no
caller, and this is the caller. Radar's per-date write is the one site that passes it.

Three properties of the in-leg budget are load-bearing:

- **It is accumulated BACKOFF, not wall clock.** The bound is then a property of the policy alone,
  since the work each attempt does belongs to the caller and is unmeasurable from the policy. It
  also makes the tolerance testable at zero wall clock: a no-op sleep still spends the budget,
  where a wall-clock budget would spin until the clock caught up.
- **The stop refuses the sleep that would CROSS the budget**, not the one after it, so 300 s is the
  real ceiling rather than the ceiling plus one full sleep.
- **Jitter is unconditional.** 5,532 tasks were in flight. Thousands of writes fail in the same
  second when a shared source falters, and an undithered ladder re-issues all of them in the same
  second too.

Measured over 500 runs of a permanently-refusing write: **10 attempts, 255–280 s of backoff**,
never above 300. Recovery is cheap: a refusal that clears on the fifth attempt costs about 20 s.

### The leg-level delay needed a prerequisite, and it is the same one as always

"Coupling to the leg retry" below states the rule this hit: **the classification has to live where
the read fails, not where the leg is retried**, because the leg sees only a failure DETAIL string
and the wrapper discards the cause. A refused write raised `WarpOperationError('Chunk and warp
failed')`, which is what a crash and a codec error also look like from there.

So the verdict is carried rather than re-derived. A radar write that exhausts the in-leg budget on
a refusal raises `errors.ProviderRefusedReadsError`, which reaches the leg's failure detail as a
type name, and `_leg_backoff_s` keys the long delay on it. Nothing else about that failure changed:
it fails the leg exactly as before and skips exactly as much, which is nothing.

The retry backoff is now RACED against the `doomed` gate rather than slept through, for the reason
the first-dispatch stagger already races its own wait — at ten minutes, a leg that sleeps through a
sibling's terminal failure holds its cell's slot, and the campaign's ingest slot behind it, to
learn something already decided.

### Only a refusal that arrives AFTER a successful read

`AccessDenied: ... is not authorized to perform: s3:GetObject` on a valid credential is either the
provider misbehaving or our permissions being genuinely wrong, and it is the same sentence either
way. Authentication faults are already separable by code — `ExpiredToken`, `InvalidAccessKeyId`,
`SignatureDoesNotMatch`, all in `_OWN_CREDENTIAL_MARKERS` and all excluded — but an authorization
verdict is not.

What separates them is WHEN it arrives. **A permissions fault is total and deterministic, so it
refuses the leg's FIRST date; a provider wobble arrives after the leg has already been served.** So
`s1_roi` carries one per-leg flag, set on the first committed date, and withholds the expensive
wait until it is set. Necessary rather than sufficient, which is all it needs to be: it gates only
the costly response, and the ordinary attempt limit applies either way.

Deliberately not `written_dates`, which a resume pre-seeds from the store: those dates were read by
an earlier leg, under a credential and a permission set that need not still apply.

The consequence worth knowing: **a re-dispatched leg whose first date is the refused one withholds
patience too**, so it fails in seconds rather than minutes. That is the intended direction — it
shifts the waiting onto the cheap lever — but it does mean each escalation step pays a fleet
provisioning to learn the source is still refusing.

### Bounding the PAIR, not each lever

The multiplication is bounded by a decision that already existed rather than a new one: **a refusal
is not a give-up.** When the budget is spent the write fails, `_give_up_date` declines it, and the
exception leaves the date loop and the leg. So **at most ONE date per leg attempt pays the in-leg
budget** — the loop does not carry on to spend it again on the next date.

| | per date | per leg attempt | per cell (3 attempts) |
|---|---|---|---|
| write attempts on a permanent refusal | 10 | 10 | **≤ 30** |
| in-leg backoff, fleet HELD | ≤ 300 s | ≤ 300 s | **≤ 900 s** |
| between-leg delay, fleet RELEASED | — | — | 1,800 s |

So a cell tolerates **about 45 minutes** of provider refusal before failing, of which **at most 15
minutes holds a fleet**, and it spends at most **30 write attempts** doing it. In practice both
figures are lower, because the first-date gate makes the second and third attempts fail in seconds.
Against that, the pre-change behaviour was 9 write attempts inside about two minutes and then a
lost cell — so the change roughly doubles the requests aimed at a struggling provider while
spreading them over twenty times the wall clock, which is the direction that matters.

A refusal outlasting 45 minutes fails the cell back to the campaign work list, which is the outer
loop and re-dispatches it later. Nothing is lost at any point: the time axis never moves.

An INTERMITTENT refusal is the case that does not collapse to one date per leg, and it is the case
worth paying for: each affected date can spend up to the in-leg budget, but every date that spends
it and then succeeds is a date saved rather than a cell lost. The dangerous shape — a long wait per
rung of a fallback ladder — is why radar alone passes `wait_out`. The optical path answers a refusal
with the copy ladder, and a 300 s wait per copy would multiply with it.

### Corrected in place

`leg_retry_backoff_s`'s comment said the difference between a momentary source refusal and a
structural one "does not take minutes to establish". For a catalogue query it does not. **For
object reads it does** — six minutes on 2026-08-24 and roughly thirteen on 2026-08-21 — and that
claim is why re-dispatches walked straight back into a refusing source and spent a cell's whole
attempt budget in two minutes. The comment now says so, and the short backoff it defends is still
the default for every other class.

### Dependence on the cause surviving the worker boundary

A refusal that cannot be RECOGNISED cannot be waited out. About **70 of the 85** failures arrived
on the driver with their cause stripped — `Exception: RasterioIOError('Read failed. See previous
exception for details.')`, `__cause__` of `None` — for the reason the section below documents.
`is_provider_refusal` declines that shape: it names a source reader but no refusal, so the write
takes the ordinary attempt limit and the leg fails recoverably. That is the correct fail-closed
direction, and it means this change realises its full value only once the cause survives the hop.

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

**Since Cause 9, the write retry is also a lever the leg retry multiplies with**, and the product
is bounded by the fact that a refusal is never given up: the failure leaves the date loop, so one
date per leg attempt pays the budget rather than every date in the year. The table in Cause 9 is
the arithmetic.

**A third instance, resolved for the radar read itself:** Cause 5 above. The prescription is the
same and was followed literally — `is_provider_refusal` classifies at the point the read fails,
the verdict is carried into the store's own record as `scope`, and the layer holding the budget
acts on the count rather than on the message. What is new is the direction of the verdict: the
error the ceiling raises is left OUT of `_NON_RETRYABLE_LEG_MARKERS` on purpose, because a
refusal clears and the retry is what recovers the dates.

### When the words DO carry the verdict, the parent acts on them

The prescription above is about failures whose cause the wrapper destroyed, and it was read more
broadly than it should have been — as "the leg retry can never classify". That reading is wrong
for the failures whose OUTERMOST exception is itself the whole verdict, and it cost fleets.

`NonMonotonicDateError` is the clearest of them. A store's time axis is append-only, so a date
below its maximum is refused identically on every attempt and no dispatch moves the axis; the
remedy is a human deleting the store. Re-dispatching it provisions and tears down a whole Dask
fleet to reach the same date and die there, once per remaining attempt. Its class name reaches the
parent intact, so it joins `_NON_RETRYABLE_LEG_MARKERS`.

Alongside it, `_is_retryable_leg_failure` now asks the read-failure taxonomy directly, through
`duplicates.unreadable_source_in` — the text form of `is_unreadable_source`, reading the same
markers in the same order. One taxonomy, so the parent deciding whether to re-dispatch and the
worker deciding whether to step down cannot reach two verdicts from the same words. Only
`UNREADABLE` and `ABSENT` skip the ladder. `PROVIDER_REFUSED`, `OUR_CREDENTIAL`,
`REFUSAL_UNATTRIBUTED`, `CLIENT_ERROR` and `UNDECIDABLE` all keep their attempts, which is exactly
what the ladder and its long refusal backoff exist for.

**The polarity is unchanged, and it is the load-bearing part.** Default TRUE: only a positively
identified permanent verdict may skip anything, and a failure nobody has classified enters the
ladder untouched. Retrying a permanent failure costs one resumable leg's fleet; declining to retry
a transient one strands the cell until a human notices.

### What crosses the deployment boundary

A leg is a separate deployment run, and `arun_deployment` returns it by reading it back over the
API. The parent therefore holds an API object, never the live exception:

- Prefect attaches the exception to the state for in-process use only. It is not serialisable to
  the API at all, so `isinstance` is unavailable here at any price.
- What the API stores is `f"{type(exc).__name__}: {exc}"` behind a fixed prefix. No traceback, and
  no `__cause__` chain — the outermost exception is the whole of it.

Two consequences. The class NAME is the one stable thing to match, which is why both the marker
list and the taxonomy key on names rather than on field order or a formatted number. And a chained
rasterio failure degrades across this boundary to `UNDECIDABLE`, because the GDAL cause is stripped
and the wrapper's own sentence names no bytes — the safe direction, since it earns the retry. This
is the same loss the Dask boundary inflicts by a different mechanism, and the rescue in
`ingest/loader_failures.py` does not reach it.

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

## Why the reason was hard to read — an evidence gap, now closed

The twin of the attribution gap above, and the deeper of the two. That one lost WHICH object
failed. This one lost WHY, and it is why Causes 2, 4 and 8 were each hard in the same way: every
one was a classifier reading a string that no longer contained the answer.

### The mechanism

Every predicate in `ingest/duplicates.py` matches the whole exception CHAIN, because the wrapper
discards the reason and only GDAL's cause states it. The chain did not arrive. Dask serialises a
worker's failure with `tblib`, which rebuilds a class that has a custom `__init__` and no
`__reduce__` by assigning its attributes — including `args`. rasterio's `CPLE_BaseError` publishes
`args` as a **read-only property**, so the assignment throws:

```
AttributeError: property 'args' of 'CPLE_AppDefinedError' object has no setter
distributed.protocol.pickle - INFO - Failed to deserialize b"...tblib...RasterioIOError..."
```

`distributed.core.error_message` unpickles its own output before sending it, so a failure never
becomes an unrecoverable one on the wire — and when that check fails it substitutes
`Exception(repr(e))`. What the orchestrator receives is one line of text with `__cause__` of
`None`. No marker list, however complete, recovers a reason that is not in the message.

The two log streams are the proof: `CPLE_AppDefinedError: ZIPDecode:Decoding error at scanline 0,
unknown compression method` appears **only** in `dask/dask-worker` streams, while the
orchestrator's carries exactly `Exception: RasterioIOError('Read failed. See previous exception for
details.')`.

In roughly one hour of the 2026-08-24 window: **70+ zone/date pairs, 85 radar leg failures across
53 cells, 6 optical, 1 cell failed outright.** 70 of the 85 radar failures arrived as the bare
`Chunk and warp failed` — a decision to be made with no evidence at all, on the path that has no
alternate-copy ladder and so only two answers.

### The change

One reducer, `CPLE_BaseError.__reduce__ = lambda exc: (type(exc), tuple(exc.args))`. Three things
about it are load-bearing:

- **Faithful, not merely sufficient.** GDAL's constructor takes exactly the triple its `args`
  reports, so the rebuilt exception carries the error class, the CPL error number and the codec's
  message rather than a flattened string.
- **`__reduce__`, not a `copyreg` reducer.** tblib consults `__reduce__` and falls back to
  assigning `args` only for classes that define none, so defining one steers tblib onto its own
  good path. A `copyreg` registration would be overwritten — tblib re-registers itself there for
  every class in the chain, on every failure.
- **On the SENDING side.** Dask decides to flatten before it sends, so a process that only
  receives cannot repair what it is handed. The reducer goes where the credential broadcast goes:
  a `WorkerPlugin`, the only thing that reaches a worker which joined after the ingest started.

### Why one class, and why that is a census rather than a guess

tblib's heuristic is broad — **808** loaded exception classes match "custom `__init__`, no
`__reduce__`" — but the assignment only fails where `args` is read-only. Scanning every exception
class in the loaded read stack (rasterio, odc, botocore, aiohttp, urllib3, pystac, s3fs, icechunk,
zarr, xarray), **18** publish `args` read-only and **all 18 derive from `CPLE_BaseError`**,
`ObjectNullError` included. Patching the base covers them all, and covers classes a later GDAL
adds. `test_no_other_exception_in_the_read_stack_needs_an_entry` re-runs the scan, so a nineteenth
fails a test rather than a campaign leg.

### What did NOT change, and why that is the point

The marker lists, the ranged status handling, the Cause 8 polarity — all untouched. They were
never the defect. They were reading a string the architecture had already emptied, and they work
as written the moment it has content.

That is also why the interim mitigation on PR #125 — splitting one predicate into a permissive
`should_try_another_copy` for the reversible ladder and a strict `is_unreadable_source` for the
destructive skip — should stand once it merges, rather than being backed out as superseded. The
undecidable failure is not abolished: the
reducer is installed per process and best effort, so a worker it does not reach reads exactly as
the fleet used to, and produces identical verdicts reached from nothing. `cause_was_flattened`
recognises that substitution and `read_failure_context` logs `READ CAUSE LOST` on it — the only
signal that a re-raise happened for want of evidence rather than because evidence said to.

### The general shape, which is the part worth keeping

**A classifier is only as good as what crosses the boundary to it.** Four causes here were
diagnosed as gaps in a marker list, and three were one architectural loss in different clothes.
Before widening a predicate, check that its input still contains what it is meant to read — and
prefer a test that PRODUCES the real input over one that imitates its shape, because a stand-in
built to the shape you already believe in cannot falsify you.

## Method notes

- `WarpOperationError`, `ZIPDecode`, and `Chunk and warp failed` each appear **several times per
  actual failure** (the pickled payload, the traceback line, the `Exception:` line, the Prefect
  retry warning). Raw counts of those strings are inflated ~3–5× and are only safe to read as
  zero-versus-nonzero. `Aborting load due to failure while reading` is emitted **once per failure**
  and is the countable line.
- The failing source URL appears only on the `Aborting load` line and the GDAL `ERROR 1` line —
  never inside the exception. Any attribution to a granule has to come from those.
- CloudWatch Logs Insights `like` is case-sensitive; see the `querying-logs-and-apis` playbook.
