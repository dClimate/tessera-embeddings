# Ingest source-read failures: twelve causes, and the retry budget they share

**Investigation record.** Twelve ways a source read fails during ingest, how each was identified,
what guard each earned, and — in §13 — the patience budget the imagery read path actually has,
which is what every one of those guards sits on top of.

Investigation record. Causes 1 and 2 traced 2026-08-04/05 on `global-tessera-dev`; cause 3 absorbed
2026-08-18 from its own document, for the reason given at that section; cause 4 traced 2026-08-20,
cause 5 on 2026-08-21, causes 6 and 7 on 2026-08-22/23, causes 8, 9 and 10 on 2026-08-24, and causes
11 and 12 on 2026-08-25, all from live campaign legs. **Corrected in place: this document said
"three causes" until cause 4 was added, "four causes" until cause 5 joined it, "five causes" until
cause 6 did, "six causes" until cause 7 did, "eight causes" until cause 9 did, "nine causes" until
cause 10 did, "ten causes" until cause 11 did, and "eleven causes" until cause 12 did.** Causes 9,
10, 11 and 12 are all later halves of cause 8 rather than new mechanisms: cause 9 is the same
refusal classified correctly and then given a budget that could not act on the classification,
cause 10 is the same refusal never classified at all because the evidence for it was never in the
exception, cause 11 is the sensor built for cause 10 listening to a channel the evidence was not on,
and cause 12 is cause 9's budget correctly armed and then withdrawn one attempt before it would have
paid off. Cause 7 is the
only one here that is not a source failure — it is our own bookkeeping, and the only one that left
holes nothing can fill. Traces
`WarpOperationError('Chunk and warp failed')` and `PermissionError: The provided token has
expired` to their exact causes, so the guards can be aimed at the right terms.

**Corrected in place 2026-08-25: the store attribute `assessed_unreadable_dates` no longer
exists.** Passages below that describe a lost date being written to it are a record of what the
code did at the time, not of what it does now. The attribute was removed once the resume rule
landed: a leg begins the day after the newest date its store holds, so a date it gave up on is
closed to that store for good, and the coverage gate subtracting such months refused a month
nothing could ever fill. A lost date is still logged per date and again in an end-of-leg summary;
what survives a fill is the embeddings store's per-pixel observation counts and per-month coverage
masks. `assessed_window` and `assessed_empty_dates` are unaffected.

Companion to [`ingest-performance.md`](ingest-performance.md), which is the authoritative record of
what ingest *costs* and of the fleet-width contention work. This one is about what makes it *fail*.

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

**Correction, in place.** `yield-embeddings/docs/runbooks/incident-response.md` previously attributed this error to
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
| leg | `IngestSettings.max_leg_attempts` | 3 | retried, because nothing marks it permanent and retrying is the default. **Corrected in place:** this row used to say the permanent-failure list "names no HTTP failure", which stopped being the whole test once the parent also began asking whether the bytes are gone (see "Some failures say enough to be judged"). The answer does not change: that check recognises a status code only in the specific form GDAL writes it, and this text is not in that form |
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

### The default is 6 h — and the 36 h it replaced rested on a misreading of what this bounds

**`IngestSettings.max_leg_wall_clock_s` is `6 * 3600`.** An earlier version of this section derived
36 h and is **withdrawn**. The arithmetic in it was fine; its first premise was wrong, and the wrong
premise is the part worth keeping, because it is the natural thing to assume about any deadline.

**The withdrawn premise: "it must comfortably exceed the longest legitimate single leg."** From that,
the densest measured zone-year (35N, 2,415 live chunks) at **175.6 s/date on ~60 workers** across
**365 dates** gives an optical leg of ~17.8 h, a legitimate band edge of roughly a day once ±35%
per-zone residuals and in-leg catalogue patience are allowed for, and therefore a bound of 36 h.

**Three things make that premise false, and each alone is enough.**

**The deadline is checked only when deciding to START another attempt.** A running leg is never
judged against it, so **a slow-but-succeeding leg cannot trip it however long it takes**. What the
bound actually limits is *patience* — wall clock a leg spends not getting anywhere — and the loop's
true worst case is the deadline plus one final attempt. The longest legitimate leg is simply not the
quantity it has to exceed.

**Firing costs latency, not work.** Icechunk commits each date's time slot atomically with its
pixels, so a leg that gives up returns the cell to the campaign work list and the next dispatch
**resumes from the dates already committed** (`ingest-performance.md` §4.15). A shorter bound does
not discard a long slow leg's output; it releases the campaign slot sooner and picks the work up
where it stopped.

**A leg that is being productive earns more time anyway.** `leg_progress_extension_s` (1 h) grants
extra wall clock each time the deadline would otherwise refuse the next attempt — but only if the
store has GAINED DATES since the last grant, so a leg that commits nothing never leaves the 6 h.
That is precisely the case the 36 h premise was worried about, handled by a mechanism that
distinguishes a leg making progress from one that is stuck, which a single flat deadline cannot. The
ceiling is `max_leg_wall_clock_s + (max_leg_attempts − 1) × 3600` = **8 h**.

Two smaller errors in the same arithmetic, noted because they compound rather than cancel: it used
**365 dates where a zone-year keeps ~250** after the coverage gate, and it used a January per-date
figure, which §11.4 of the ingest record shows understates a full seasonally-weighted year. Those
move the number in opposite directions and neither matters once the premise is gone.

**What survives, and it is the half that still pins the number: a stuck cell must release its
campaign slot promptly.** 6 h clears three legs of a slow dense cell plus their expansive backoff,
while still releasing a pathological cell's slot inside a working day.

**Re-derive against measured leg durations if the retry stack changes** — in particular if the
progress extension is ever turned off (`leg_progress_extension_s = 0` restores the plain deadline
exactly), because then the flat bound is the only thing standing between a productive-but-slow leg
and a refusal, and the withdrawn premise becomes relevant again.

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
  `../storage/staging-identity-and-resume.md` section 5.

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
  the first is in `source-read-failures.md` ("Caveat on the sample"), where it
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
   legitimate absence." `tests/unit/config/test_time_window.py::test_a_month_lost_to_unreadable_imagery_is_not_excused`
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
already logged at error level, so a leg that
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
it on the store as `assessed_unreadable_dates` (since REMOVED — see the head of this document) with `scope=provider-refused` or
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

**Corrected in place 2026-08-25: neither column of that table was reached.** The in-leg budget was
armed and then withdrawn one attempt in (Cause 12, defect 1), and the between-leg delay's second
rung was refused rather than taken whenever it overran the leg's remaining wall clock (Cause 12,
defect 2). The arithmetic below is unchanged and is now attainable rather than theoretical.

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

**And that is only the first of two dependencies.** A cause that survives intact is still only what
the reader chose to raise, and for a refused object that is a codec failure. See Cause 10.

## Cause 10 — the refusal was never classified, because GDAL logged it instead of raising it, 2026-08-24

**The third and last half of cause 8.** Cause 8 made the taxonomy correct. Cause 9 gave the correct
verdict a budget worth having. This is why neither fired: the verdict was never reached, because
the words it is reached from were never in the exception.

### The failure

An ASF credential expiry caused an eight-minute authorization outage. Every affected read returned
HTTP 403 `AccessDenied`, and S3 delivered an XML error document in the place of the imagery. GDAL
handed that document to the TIFF decompressor — that is what the reader had asked for — and the
decompressor failed on it: `ZIPDecode: Decoding error at scanline 0`, of which **25 said "unknown
compression method" outright**, which is a codec reporting that the bytes are not compressed data.

The ingest gave up **158 radar dates across 55 stores** as permanently unreadable. **The imagery is
fine**: 60 of 60 sampled granules read successfully afterwards, against a 30-of-30 control. Sixty of
those dates are now permanently unfillable, because later dates were committed above them and the
time axis is append-only.

### The mechanism, and the one leg that got it right

**Not one of the 158 exception chains contained an HTTP status or the string `AccessDenied`.** GDAL
reported the 403 to its own log as a warning and raised only the decode failure. So
`is_provider_refusal` saw `ZIPDecode`, correctly said no, and `wait_out` stayed `None` — the
patience of Cause 9 never armed, and `_give_up_date` accepted the date as unreadable.

At **00:00:32** one leg happened to surface `AccessDenied` as its final exception instead of a
decode error. It classified as a provider refusal and skipped no date.

**The same outage produced the permanent-loss verdict 158 times and the correct transient verdict
once, decided purely by which GDAL error landed last in the retry ladder.**

### Why the classifier was not at fault

Read from the chain alone, `UNREADABLE` is the right answer to `ZIPDecode`. Nothing in the marker
lists, the ranged statuses or the ordering was wrong. This is the same shape as the evidence gap at
the end of this document — a classifier reading a string that does not contain the answer — with one
difference that matters: there, the reason existed and was destroyed in transit; **here it never
entered the exception at all.** No reducer recovers it, because there is nothing to serialise.

### The change

`ingest/loader_failures.py` already installed a log handler on the loader's own logger, to name the
object a load aborted on. It gains a second, on `rasterio._env` — the logger rasterio's CPL error
handler forwards GDAL's messages to, and the only place the refusal is stated.
`carry_logged_refusal` attaches what it collected to the failing exception as a note,
`_exception_chain_text` reads notes with the rest of the chain, and `classify_read_failure`
therefore decides from all of it.

Both sensors reach that through `roi_processing.read_failure_context`, which every per-date read on
both paths already passes through. One classifier, one set of evidence — the optical path is
corrected by the same change, and a refusal that only GDAL logged now fails its leg with the axis
unmoved instead of walking the copy ladder and recording DATA LOSS.

Radar asks a second time, per retry attempt, through `refusal_wait_out(client)`. That is what arms
Cause 9's budget: the retry policy asks its `wait_out` on every failure, and a predicate reading only
the exception declines the outage the budget exists to outlast.

**Corrected in place 2026-08-25: the predicate is asked per attempt, but the evidence it is asked
over is the whole WRITE.** It re-armed its window on every ask, so patience was withdrawn as soon as
an outage stopped restating itself — which is as it recovers. See Cause 12.

### The polarity, which is the load-bearing part

Adding evidence to a classifier can make a verdict worse in both directions, and only one of them is
recoverable. Reading a refusal as unreadable cost 60 dates permanently. Reading unreadable bytes as
a refusal makes a leg wait minutes and re-dispatch on data that will never read.

**A line is kept only if the classifier reads that line ALONE as `PROVIDER_REFUSED` or
`OUR_CREDENTIAL`** — the classifier is its own filter, so there is no second vocabulary to drift.
Because refusal is tested before any statement about the bytes, an attached line can only move a
verdict into those two members, never into `UNREADABLE` or `ABSENT`. **The capture cannot cost a
date.** A wrong attribution costs patience: the write spends its refusal budget, the leg fails with
its axis unmoved, and the date is judged alone on the re-dispatch.

The rejected cases are what the filter is really for. GDAL probes for sidecars that were never
published, and a kept `HTTP response code: 404` is exactly the marker that says a source object is
absent — a verdict that gives a date up. A capture that took every GDAL warning would have been a
new way to lose dates rather than a fix for the old one.

### Two races, kept apart

The href capture is destructive and cluster-wide, and `s2_roi.py` documents why that race fails
safely: a lost attribution costs precision, a borrowed one would step down a tile that read.

The refusals live in a **separate buffer**. One buffer would have meant the caller that classifies
destroys the evidence the optical copy ladder attributes from — silently, and the ladder would
simply stop attributing. The new buffer is drained on **every** read failure rather than only on an
undecided one: a buffer left unread is a buffer whose lines are still there to be borrowed by a date
that fails minutes later.

### The general shape

**The evidence a classifier needs may never have been in its input, rather than lost from it.** The
evidence gap at the end of this document was about a reason destroyed in transit, and the answer was
to stop destroying it. This one looks identical from the classifier's seat and has no such answer:
the reader has to go and fetch what the raiser never said. Before widening a predicate, ask not only
whether the input still contains what it should, but whether it ever did — and where else the same
process wrote it down.

## Cause 11 — the log the sensor listens to was not where GDAL said it, 2026-08-25

**Cause 10's capture, shipped and working, and three more dates gone.** Cause 10 went in the morning
of 2026-08-25. That afternoon ASF refused reads again and the capture largely held: in one
half-hour window, **663 `ProviderRefusedReadsError`** — the good outcome, no date skipped — and
**102 log lines quoting captured evidence**, against **zero `TooManyGivenUpDatesError`**.

Three radar dates were skipped anyway, all with the pre-fix signature:

| time (UTC) | task | orbit | zone | date |
|---|---|---|---|---|
| 15:20:44.488 | `process-roi-sar-d7b` | ascending | 37N | 2020-12-31 |
| 15:21:23.456 | `process-roi-sar-119` | descending | 36N | 2017-11-19 |
| 15:21:36.917 | `process-roi-sar-135` | ascending | 32N | 2019-11-03 |

Each logged `scope=unreadable error=Chunk and warp failed`.

### The mechanism

GDAL keeps its error handlers **per thread**. `CPLPushErrorHandler`, which is what
`rasterio.Env.__enter__` calls to install rasterio's logging handler, pushes onto the calling
thread's stack; a message reported on a thread whose stack is empty falls through to GDAL's
process-wide handler, `CPLDefaultErrorHandler`, which writes to the process's stderr and to no
logger. GDAL's ranged reader fetches on threads of its own, and those threads never enter a
`rasterio.Env`.

So the refusal for these three reads was stated, in the same process, 43 ms before the loader gave
up on the object — and it was never a Python log record at all. `_LoggedRefusalHandler` is a
`logging.Handler`. There was nothing for it to see.

### Evidence

Everything below is from log group `/ecs/global-tessera-prod`, 15:15–15:25 on 2026-08-25.

The lost read, on worker `9fd26b71`, in order:

```
15:20:11.570  ERROR 1: Request for https://asf-cumulus-prod-opera-products.s3.us-west-2.amazonaws.com/
              OPERA_L2_RTC-S1/...T072-152798-IW1_20201231T150410Z..._VV.tif
              range 1358074-3075915 failed with response_code=403
15:20:11.613  ERROR | odc.loader._rio - Aborting load due to failure while reading: <the same object>
15:20:12.083  INFO  | distributed.worker - Run out-of-band function 'read_local_refusals'
15:20:12.482  WARNING | Retrying <unknown> in 3.22 seconds as it raised WarpOperationError: Chunk and warp failed.
...
15:20:44.465  INFO  | distributed.worker - Run out-of-band function 'read_local_refusals'
15:20:44.480  INFO  | distributed.worker - Run out-of-band function 'read_local_refusals'
15:20:44.488  ERROR | DATA LOSS roi=zone_37N date=2020-12-31 ... scope=unreadable error=Chunk and warp failed
```

**The `ERROR 1:` prefix is the whole diagnosis.** `ERROR %d: %s` is the format string of
`CPLDefaultErrorHandler` and appears nowhere in rasterio's Python or Cython source; rasterio's own
handler formats `%s in %s` from its `code_map`. A line reading `ERROR 1: …` was therefore written by
GDAL's process-wide handler, which means the reporting thread had no handler pushed.

Counts over the same window, which is what makes this a class rather than an incident:

| where GDAL's 403 statements went | count |
|---|---|
| stderr, via `CPLDefaultErrorHandler` (`ERROR 1: Request for … response_code=403`) | **1,462** |
| `rasterio._env` at WARNING (`CPLE_AppDefined in HTTP response code on <url>: 403`) | **13** |

**The capture could see 0.9% of what GDAL said.** The 13 visible lines were on 12 workers; the 51
reads saved by captured evidence all borrowed from those 13, cluster-wide. The three lost reads were
the ones whose own leg held none of them — each leg's collection fanned out to about ten workers, and
none of `d7b`'s ten appears in the list of twelve.

**This document had already written the wording down.** The headline table at the top of this file
records `ERROR 1: Request for <url> range X-Y failed with response_code=403` as the line that
precedes the expired-credential case, and has since cause 1. The `ERROR 1:` prefix that identifies
the channel, and the `response_code=` spelling the classifier could not read, were both sitting in
this file's own summary while cause 10's capture was built against a different wording entirely.

### What was NOT the cause, each checked

The collection worked. Every one of these was eliminated against the logs:

- **A worker died holding the evidence.** `_collect` logs `"… did not answer …"` when a worker fails
  to answer, and that line appears **zero times in the hour**. All four collections for the lost
  read are visible on the worker's own stream.
- **The age bound dropped it.** The read entered `read_failure_context` at about 15:19:52 and the
  first collection ran at 15:20:12.083, so the cutoff was ~20 s; the newest cluster-wide line was
  0.9 s old.
- **`_REFUSAL_RETENTION_S` evicted it.** 3,600 s against a read that lasted 53 s.
- **`is_source_read_failure` returned early.** The chain carried both `WarpOperationError` and
  `CPLE_`, so it did not.
- **The cause was destroyed in transit.** `READ CAUSE LOST` never fired, and the chain arrived whole:
  `WarpOperationError: Chunk and warp failed` ← `CPLE_AppDefinedError: ZIPDecode:Decoding error at
  scanline 0`.
- **The imagery was genuinely bad.** Three objects, in the same minute as 1,462 refusals of the same
  bucket, each preceded by a 403 on the exact object the loader then gave up on.

**And these three are the whole of it.** Eight dates were skipped between 14:30 and 16:00. The other
five are optical, `scope=attributed`, on `sentinel-cogs` and `sentinel-s2-l2a` rather than ASF, and
their last errors are `ObjectNotFound` (cause 4) or a `Chunk and warp failed` on one tile that two
separate legs reproduced 23 minutes apart — a corrupt object (cause 2), which is what the copy
ladder exists for. No optical date was lost to this mechanism.

### A second defect, in the same path and independent of the first

Even reaching the logger, that line would have been dropped. `_LoggedRefusalHandler` keeps a line
only if the classifier reads it ALONE as a refusal, and `_HTTP_STATUS_RE` was anchored on GDAL's two
`code:` wordings. The ranged reader says `response_code=` instead, so the pattern read **no status**
out of the wording that accounts for 99% of production's refusal statements, and a line with no
status is not a refusal. Confirmed directly: `classify_read_failure_in` returned `undecidable` for
the exact line above.

So the fix is two halves, and each was verified to be load-bearing by reverting it alone.

### The change

`ingest/loader_failures.py` gains `hear_gdal_from_every_thread`, installed alongside the two log
handlers by `install_capture` — the same sensor, since the handlers hear what GDAL says to a logger
and this is what makes GDAL say the rest of it to one. It sets GDAL's process-wide handler, through
`CPLSetErrorHandler`, to a callback that forwards to `rasterio._env`.

Four properties carry the safety of it:

- **Chained, not replaced.** The previously installed handler is called first, so GDAL's stderr line
  still appears where an operator greps for it and `CE_Fatal` still aborts through it. All this adds
  is a second copy, on a logger.
- **Thread-local still wins.** GDAL consults the reporting thread's stack first, so nothing rasterio
  already forwards is forwarded twice. Tested, because without it every note would quote every line
  twice.
- **rasterio's own wording, from rasterio's own `code_map`.** The classifier's corroboration that a
  line is the source READER's is GDAL's vocabulary, and `CPLE_AppDefined in <message>` carries the
  `CPLE_` a bare GDAL sentence does not. Borrowing the format is what lets such a line be judged on
  its own, and means no new marker was added to `_SOURCE_READER_MARKERS`.
- **Not levelled the way rasterio levels it.** rasterio downgrades `CE_Failure` to INFO because one
  of its calls may emit several and still succeed. A message reaching the process-wide handler has no
  such call to judge it, and INFO is below both the logger's default level and the capture's, so
  honouring the downgrade would record it nowhere.

`ingest/duplicates.py` widens `_HTTP_STATUS_RE` to read a status out of `response_code=<n>` as well.
`\d{3}` and not `\d+`, because GDAL prints `response_code=0` when the request never completed: that
is the absence of a status, and admitting it would make it a 4xx meaning neither wait nor absence —
the verdict that re-raises and fails a leg on its first date.

The polarity argument of cause 10 is unchanged and still the reason this is safe: a kept line can
only move a verdict into `PROVIDER_REFUSED` or `OUR_CREDENTIAL`, so the capture cannot cost a date,
and a wrong one costs patience.

### The general shape

**A sensor on a log is a sensor on a CHANNEL, and the channel may not be where the thing is said.**
Cause 10 asked whether the evidence was ever in the exception. This asks the same question one layer
out: the evidence existed, in the right process, at the right moment, and the sensor was pointed at a
Python logger while the library was writing to a file descriptor. Before trusting a capture, count
what it sees against what the source emits — 13 against 1,475 was visible in the log group from the
first query, and no amount of reasoning about levels, ages or retention would have found it.

## Cause 12 — the patience was armed, and then withdrawn as the outage cleared, 2026-08-25

**Cause 9's budget, reachable at last, and spent 9% of the way.** Cause 8 made the taxonomy
correct, cause 9 gave the correct verdict a budget worth having, cause 10 got the verdict reached
at all, and cause 11 got the evidence onto the channel the sensor listens to. This is what the
budget then did with all of that: 26.9 s of a 300 s allowance, and a lost cell.

Three separate defects, all in the retry path, all with the same shape — **a decision reached from
a window narrower than the thing it decides about.**

### The failure

One cell's radar leg ran five and a half hours across two attempts and gave up, failing its cell.
The time was not wasted: it committed **250 dates**, and **91.9% of the elapsed time was productive
writing** at a steady **75 s per date**. Slowness is not what killed it.

**Both attempts died on a transient provider refusal, and both gave up as the provider was
recovering.** ASF's S3 answered `AccessDenied` for its own download role in two bursts of one to two
minutes each. The exact date one attempt died on **committed successfully nineteen minutes later**,
and there were **zero refusals in the minute the first write quit**.

### Defect 1 — the in-leg budget is never spent

`WAIT_OUT_BACKOFF_S` is 300 s of accumulated backoff, and cause 9 records it measuring **255–280 s
over ten attempts** against a permanently refusing write. Here it spent **26.9 s** on the first
death and **9.78 s** on the second. The retry trails: sleeps of 2.98, 5.07, 7.91, 10.9 s then
give-up; and 5.78, 4 s then give-up.

**The mechanism.** `refusal_wait_out` re-armed its evidence window on every call — `since` was
reassigned to `time.monotonic()` each time the retry policy asked — so each attempt was judged on
the refusals logged during THAT ATTEMPT alone. An outage states its refusal while it is refusing.
It stops the moment it clears, which is one attempt before the write would have succeeded, so the
question the predicate was really asking — "was anything refused in the last few seconds" — gets a
correct NO at exactly the moment patience was about to pay off.

The signature is **two readings of one failure disagreeing**: the enclosing `read_failure_context`,
whose window is the whole write, concluded `ProviderRefusedReadsError`, while the per-attempt
predicate concluded "not a refusal" about the same words.

**Indicated, then established.** The predicate's return value was not logged, so the above was an
inference from the sleep trail. It was confirmed by driving the real `store_write_retrying` against
the real `refusal_wait_out`, with the refusal stated during the first attempt and never again:

| refusal restated | exception per attempt | attempts | backoff | verdict trail |
|---|---|---|---|---|
| every attempt | fresh | 10 | 266.1 s | `TTTTTTTTTT` |
| first attempt only | **fresh** | **3** | **12.0 s** | **`TFF`** |
| first attempt only | one object re-raised | 10 | 264.6 s | `TTTTTTTTTT` |

The third row is a trap worth not re-hitting. `carry_logged_refusal` attaches its evidence to the
exception as a NOTE, so a harness that re-raises one exception object carries the first attempt's
evidence onto every later attempt and latches by accident — the defect vanishes and the test reports
patience the real path does not have. A real write builds a new graph per attempt and raises a new
exception, so **a test of this predicate has to raise a fresh failure per attempt or it proves
nothing.**

**Cause 11's forwarder is not a substitute, and the question was asked.** The forwarder that put
GDAL's stderr refusals onto a logger shipped seven minutes after this leg died, so a fair question
is whether a sensor that sees everything makes the window irrelevant. It does not, and the reason is
not about sensitivity: a perfect sensor asked "was anything refused in the last few seconds" still
answers NO, correctly, at the moment an outage clears. The two fixes are orthogonal — cause 11 is
about whether the evidence exists, this is about how far back the question looks.

**The fix is not a latch.** A latch — remembering that the predicate once answered yes — was
considered and rejected: it introduces a stored state to reason about, and it freezes one of the two
disagreeing answers rather than removing the disagreement. Fixing the WINDOW does both jobs at once.
`since` is now fixed at the write's start, which is the same window `read_failure_context` already
judges by, so the two readings agree by construction. Nothing is remembered: the evidence is
re-collected and re-classified on every attempt, and a write whose window holds no refusal never
waits. What makes it behave like a latch is only that the buffer is read non-destructively and
retained far longer than any write lasts.

The bound is unchanged, and that is the part that had to be checked. The stop condition is still
"refuse the sleep that would cross `WAIT_OUT_BACKOFF_S`", the budget is still accumulated backoff
rather than wall clock, and the measured spend after the fix is **268.7 s over 10 attempts** — the
same shape cause 9 recorded, now reached by the failure it was written for. The exposure the wider
window adds is the one cause 10 already priced: a write refused early and then broken differently
spends its refusal budget before failing, which costs patience, never a date.

**And it is now observable.** Every ask logs its verdict and the number of refusal lines it read.
An attempt count alone cannot separate "no refusal was logged" from "a refusal was logged and not
read", and those two want opposite repairs — which is why this defect took a sleep trail to find.

### Defect 2 — the second backoff rung had never once fired

The leg retry ladder's rungs are `leg_refusal_backoff_s` then twice it: 600 s then 1,200 s. Over
seven days, **56 legs took the 600 s rung and not one ever took the 1,200**.

That reads as dead tuning until you look at what was refusing it. Three cells lost their third
attempt within the same 43 seconds, all descending radar, all because the next backoff did not fit
the remaining wall-clock budget:

| cell | elapsed | budget left | backoff wanted | short by |
|---|---|---|---|---|
| 43N/2017 | 20,458 s | 1,142 s | 1,200 s | **58 s** |
| 37N/2018 | 20,430 s | 1,170 s | 1,200 s | 30 s |
| 34N/2017 | 20,714 s | 886 s | 1,200 s | 314 s |

Each had fifteen to twenty minutes of budget left and was denied because the next wait was twenty.
**All three would have fitted on the 600 s rung.**

**The rungs are NOT changed, and that is the finding.** Flattening the ladder was the obvious
reading of "a rung that never fires", and it is wrong twice over: the rung was unreachable rather
than mistuned, so the evidence for deleting it was produced by the defect; and deleting it removes
an escalation that has never actually been tried. What changes is the gate. A rung longer than what
is left now DESCENDS the ladder to the longest rung that fits, and only a leg with no room for even
the base rung is refused.

Two things it deliberately is not. It does not cap the wait to the REMAINDER — that was tried
earlier and is recorded in the code as wrong in an instructive way: waiting exactly what is left
makes the next dispatch land on the deadline every time, turning a race into a guarantee of the
thing the budget forbids. A rung is taken only if it is strictly shorter than what remains. And it
cannot change behaviour for any leg that is not within one rung of its deadline, so the escalation
every other leg sees is untouched.

The counter-argument worth answering is whether a shortened wait is a wasted attempt — whether the
rung length encodes how long an outage lasts. It does not: the BASE rung is the policy's statement
of that, derived in `leg_refusal_backoff_s`'s own comment from outages of about six and thirteen
minutes, and the doubling above it is escalation rather than the minimum viable wait. Falling back
to the rung 56 legs have already used is not waiting less than the policy asks for. The alternative
is waiting zero and taking no attempt at all.

**Effect on cause 9's arithmetic:** none to the ceiling, which still assumes both rungs are taken.
What changes is that the ceiling becomes attainable — before this, a cell near its deadline gave up
instead of spending its last attempt.

### Defect 3 — the budget charges time spent succeeding

`max_leg_wall_clock_s` is measured from the leg's first dispatch, so it includes all the productive
work of every prior attempt. Its own comment says the deadline "only ever binds on a cell already
behaving pathologically", and as written it could not keep that promise: the cell above committed
250 dates and was refused its third attempt on exactly the same terms as a cell that had achieved
nothing.

**Re-anchoring the clock was rejected.** Measuring from the last attempt that committed a date means
a leg committing one date an hour runs forever, so it immediately needs a second bound to contain
the first — two mechanisms where there was none. What went in instead is a bounded CLAUSE: a leg
whose store has gained dates earns a fixed extension, `leg_progress_extension_s`, and the absolute
ceiling stays computable.

Two things bound it, and both are load-bearing:

- a grant is a **fixed** size, so a leg cannot earn a deadline proportional to how far it overran —
  the caller re-reads the deadline after a grant rather than assuming one was enough, and an attempt
  that overran by more than a whole extension is past what progress buys;
- each grant has to be **paid for** by dates committed since the previous grant, so a store that
  stops growing stops earning them.

**Payment is also what bounds the RATE, which is why the extension can be asked for at every refusal
rather than at one chosen gate.** An earlier draft limited grants by counting call sites, and that
was the wrong bound: it made the credit depend on which of the deadline's two expressions happened
to fire — "no time left" or "no time left to wait first" — so a leg that failed 480 s into a 500 s
budget, with 20 s left and a 30 s base rung, was refused without its progress ever being asked
about. Both reviewers of PR #145 found exactly that case. The real limit was always payment: only a
RUNNING leg commits dates, and every ask sits after an attempt has failed, so the asks within one
attempt compete for the same growth and at most one of them can be paid for. Grants therefore stay
bounded by the re-dispatch decisions a leg has, one fewer than `max_leg_attempts`, with no counter
anywhere. `test_progress_is_credited_at_the_rung_refusal_too_not_only_the_elapsed_one` pins it.

Ceiling: `max_leg_wall_clock_s + (max_leg_attempts - 1) * leg_progress_extension_s`. A leg that
commits nothing never leaves `max_leg_wall_clock_s`, and 0 restores the plain deadline exactly,
including its store reads.

**Progress is read from the STORE, not from the failure.** The parent holds only a failure detail
string, and a leg dies naming the date that failed rather than one it committed — so the message
cannot answer this, and reading a date out of a message would be the field-position matching the
rest of this document forbids. The reader is `get_existing_dates`, the same one the ingest resumes
from, so the parent and the leg cannot disagree about what the store holds. It is read per LEG,
against the leg's own child store: a radar orbit still committing is no evidence that the optical
leg is, and reading it per cell would hand the deadline to whichever leg was healthy — precisely
the leg not asking for it. An unreadable store earns nothing, which is the same answer as no
progress and leaves the deadline where it was.

### What each defect cost, which is not the same thing

Worth separating, because it decides how much machinery each deserves. Defect 1 loses a cell's
whole attempt budget in seconds to an outage that clears in minutes, and it is the one that reaches
`_give_up_date` territory if the classification ever slips. Defects 2 and 3 cost **fleet-hours and
latency, never work**: the cell returns to the campaign work list and resumes from its committed
dates. That is why defect 1 is a straight bug fix with no lever, defect 2 is a straight bug fix
touching one comparison, and defect 3 is the only one of the three that gets a setting — it widens
a latency bound that the config comments describe as a policy choice, so an operator has to be able
to put it back.

### The general shape

**A decision reached from a window narrower than the thing it decides about.** All three are the
same error at different layers: a refusal judged on one attempt rather than the write, a retry
judged on the rung it escalated to rather than the rungs available, and a leg judged on elapsed time
rather than on what it did with it. In each case the narrow window gives an answer that is correct
about itself and wrong about the question. Before trusting a predicate, ask what interval its
evidence covers and whether that is the interval the decision is about — and if two readers of one
failure can disagree, the narrower one is the one to widen, not the one to freeze.

## 13. What patience the imagery read path actually has — and the three options odc shadows

Record, 2026-08-27, from a live S3 us-west-2 degradation. **This shipped no code.** The scope is far
smaller than the investigation first claimed, and every closure of the remaining gap was worse than
leaving it open — both of which are the point of the record. It belongs here because every guard above
is layered on top of a GDAL retry ladder nobody in this repository controls.

### The mechanism

`odc.loader.capture_rio_env()` composes its readers' GDAL environment from odc's own config object and
the active rasterio `Env` — never from `os.environ` — and returns its three-entry
`GDAL_CLOUD_DEFAULTS` when both are empty. odc then applies that as an **explicit** `rasterio.Env`, and
explicit Env options beat process environment variables.

```python
GDAL_CLOUD_DEFAULTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "0.5",
}
```

### Correction 1: three options are shadowed, not five

**The first version of this record said five settings were "absent from the read path entirely". That
was wrong.** GDAL falls back to `os.environ` for any option the explicit `Env` does not name, and
`configure_gdal_environment()` puts all of them there — including on remote workers, verified by
launching one from a deliberately emptied environment and finding a value that could only have come
from our own setup code running there.

`configure_gdal_environment()` sets **ten** GDAL options. odc names three of them, so **seven remain
effective** and three are shadowed: `GDAL_HTTP_MAX_RETRY`, `GDAL_HTTP_RETRY_DELAY` and
`GDAL_DISABLE_READDIR_ON_OPEN`.

### Correction 2: the ladder is exponential — and odc's config is MORE patient than ours

`GDAL_HTTP_RETRY_DELAY` is the BASE of a doubling ladder with no cap. Measured against a server
refusing every request, arrival times recorded server-side:

```
0.5, 1.01, 2.07, 4.92, 10.96, 24.82, 52.36, 105.94 s
```

Total scales linearly in the base (confirmed at a ratio of 9.99 for a tenfold change). **But the
multiplier is RANDOM between 2.0 and 2.5, not a clean doubling** — GDAL draws it per retry
(`port/cpl_http.cpp`). The measured ratios above are 2.02, 2.05, 2.38, 2.23, 2.26, 2.11, 2.02: one
sample of a random process, which I first read as "roughly doubles" and used as though it were the
rule. **A single ladder is therefore a lower bound, not a typical case.**

Two budgets, and they ADD rather than overlapping — sleep between attempts, and the attempts
themselves. `GDAL_HTTP_TIMEOUT=120` is not shadowed, so a permanently hanging request costs up to
120 s per attempt, and `n` retries means `n+1` attempts:

| config | backoff sleep (×2.0 → ×2.5) | request time, worst | worst total |
|---|---:|---:|---:|
| **ours** — 5 retries, base 5 s | 2.6 → **5.4 min** | 12 min (6 × 120 s) | **~17 min** |
| **odc** — 10 retries, base 0.5 s | 8.5 → **53 min** | 22 min (11 × 120 s) | **~75 min** |
| 10 retries at base 5 s | 85 min → **8.8 h** | 22 min | **~9.2 h** |

**odc's read path is still the more patient of the two** — roughly 3–4× ours depending where in the
random range each lands — which is the point that matters, and the opposite of what I assumed before
review. Only the base is 10× smaller; the retry COUNT is twice as large and the ladder's total is
dominated by its last rungs. **Forcing our values onto the read path would have REDUCED patience
there.**

**And note the last row.** Raising our base to 5 s while leaving odc's count of 10 would give a worst
case near **nine hours for one unreadable object**, not the 2.5 h an average-multiplier model
suggests. The S2 coverage gate then wraps the read in `source_read_retrying()` —
`SOURCE_READ_ATTEMPTS = 8` passed to `stop_after_attempt`, so eight attempts in total, seven after the
first — while `max_leg_wall_clock_s` cannot interrupt a running leg. That is the budget every cause
above is nested inside, and it is why cause 5's credential-refresh defect (which silently drops those
ten GDAL retries for the rest of a task's life) mattered as much as it did.

### What shipped: nothing but this record

**No code change.** The gap is real — an operator's override of the three shadowed options never
reaches the imagery path — but every way of closing it is worse than leaving it open:

* **Forward options whose value differs from our default.** Breaks for the most likely override there
  is: an operator setting `GDAL_HTTP_RETRY_DELAY=5`, our own documented default, is indistinguishable
  from no override at all.
* **Record provenance before `setdefault()`.** Breaks across process boundaries. A worker inherits our
  defaults from its parent's environment, so at the moment it runs, every option is already present
  and would read as operator-supplied — pushing our values into odc on every worker and changing the
  read path by accident.
* **A dedicated override interface.** Works, and is a new public knob plus its plumbing for a
  capability nobody has needed: no incident so far, including the one that prompted this, has been
  handled by tuning GDAL at runtime.

**So there is currently NO knob that changes those three options on the imagery path.** Editing
`os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", ...)` in `config/environment.py` changes the
environment, which odc shadows — it affects direct rasterio use only. Anyone who needs to tune
imagery-read patience has to build the dedicated interface first. **Reopen this if an incident is ever
actually blocked on that**; that is the evidence it would need, and it does not exist yet.

### Validation

17 ingest runs against real Sentinel-2 imagery — 6 baseline, 6 branch, 5 with multiplexing off.

**Every run produced identical embedding arrays and time indexes, and an identical set of committed
dates.** Not byte-identical *stores*: each fresh store stamps `created_at` and `last_appended` via
`utcnow_iso()`, and the runs used different builds whose manifest identities differ. Those metadata
fields were excluded from the comparison by design; the pixel data and the date coverage were not.

Branch 1.1% faster than baseline. **Slowest single chunk read on the multiplex-on arms: 6.92 s.**
Across all 17 runs the slowest was **93.66 s**, on a multiplex-off run.

`GDAL_HTTP_MULTIPLEX` was kept. The caution in our source refers to a macOS development problem, and
the option has been active in production all along through the environment, so removing it would be a
behaviour change rather than a rollback — and the multiplex-off arm produced that 93.66 s outlier.

**That 93.66 s is a CHUNK duration, not a single request**, and one chunk read issues several COG range
requests. `GDAL_HTTP_TIMEOUT=120` caps each individual request, so a sequence of short requests can
produce a 93.66 s chunk with nothing approaching the cap. The validation therefore says nothing about
timeout headroom in either direction.

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
acts on the count rather than on the message.

> **Corrected in place: the ceiling's error is IN `_NON_RETRYABLE_LEG_MARKERS`, not out of it.**
> This paragraph said `TooManyGivenUpDatesError` was left out on purpose, "because a refusal clears
> and the retry is what recovers the dates". It is listed, and the code's own comment gives the
> opposite and correct reasoning: **a provider refusal is not a reason to give up a date**, so every
> date the ceiling counts is one whose bytes will not read. A re-dispatch re-reads the same objects,
> spends the per-read retry ladder on each, and holds a fleet to reach the identical answer. The
> classification advice above is unaffected — only the disposition of this one marker was wrong.

### Some failures say enough to be judged, and those skip the retries

The advice above concerns failures whose real cause was thrown away in transit. It got read more
broadly than intended — as "the parent can never tell why a leg failed" — and that reading is wrong
for failures where the last exception raised is itself the complete answer. Reading it too broadly
cost fleets.

**The clearest example is a date offered out of order.** Dates can only be added to a store newest
last, so a date older than the newest one already stored is refused. It will be refused identically
every time, and no amount of re-running moves anything — the only way forward is a person deleting
the store. Yet a failed leg was re-dispatched up to three times, and each attempt builds a
sixty-worker cluster, walks to the same date, and dies there. Three clusters to learn what the first
failure already established.

So `NonMonotonicDateError` is now in the list of failures the parent will not re-dispatch. Its class
name survives the trip to the parent intact, which is what makes this possible at all.

**Alongside it, the parent now asks the same question the worker asks: are these bytes gone?** It
calls `duplicates.unreadable_source_in`, a text-reading form of the check the worker already uses,
looking at the same markers in the same order. Deliberately the same one, so the parent deciding
whether to re-dispatch and the worker deciding whether to try another copy cannot reach opposite
conclusions from identical words.

Only two answers skip the retries: the data could not be read, and the data is not there. Every
other answer — the provider refused us, our credentials failed, a refusal we could not attribute, a
client error, or no idea — keeps all its attempts. Those are the situations retrying exists for.

**The default is still to retry, and that is the part to protect.** Only a failure positively
identified as permanent may skip anything; a failure nobody recognises goes through the retries
untouched. The asymmetry is what settles it. Retrying something permanent wastes one cluster.
Failing to retry something temporary strands a cell until a person notices.

### What actually reaches the parent when a leg fails

A leg runs as its own separate deployment, and the parent finds out how it went by reading the
record back from the Prefect server over HTTP. So the parent never holds the failure itself, only a
description of it:

- The exception object exists only inside the process that raised it. It cannot be sent to the
  server at all, so the parent cannot ask "what type of error was this?" in the normal way, at any
  price.
- What the server keeps is one line: the exception's class name, a colon, and its message. No stack
  trace, and none of the chain of underlying causes — the last exception raised is all that is left.

That leads to two things worth knowing. **The class name is the only reliable thing to match on**,
which is why both the list of permanent failures and the bytes-are-gone check key on names rather
than on the position of a word or a number in a message.

And **a failure that began as a file that would not open arrives here looking unrecognisable**. The
underlying cause has been stripped, and the surviving wrapper mentions no filenames, so the check
answers "no idea" — which earns a retry. That is the safe direction. It is the same loss of
information the Dask worker boundary causes, arriving by a different route, and the repair in
`ingest/loader_failures.py` does not reach this one.

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
