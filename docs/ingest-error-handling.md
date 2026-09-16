# Error handling in satellite ingestion

Reading imagery from public archives fails often, and the failures do not look different from
each other. A corrupt file, an expired credential and a throttled provider all arrive as the
same exception with the reason stripped out. This document covers how the pipeline recovers the
reason, what it does with each answer, and why guessing costs more than waiting.

## The five classes of failure

Twelve distinct causes were diagnosed over the campaign. They fall into five classes, and the
class is what decides the response:

| class | what it means | the right answer |
|---|---|---|
| **provider system failure** | the service is unwell: throttling, gateway errors, a refusal lasting an excessive number of minutes | wait, and keep waiting |
| **authentication failure** | our credential is expired or wrong | refresh it, or stop the job outright |
| **data corruption** | the object is published but broken — will not decompress, truncated, missing a band | fetch a different copy; never retry the same one |
| **missing data** | the object was never published at all | a different copy, then give the date up |
| **bad response body** | the service answers "success" and sends something else: an error page, or a truncated document | ask for the same page again |

Two pairs are dangerous. A provider system failure and data corruption need opposite responses:
waiting fixes the first and wastes time on the second, switching copies fixes the second and does
nothing for the first. And an expired token wants a refresh and a retry where wrong permissions
want the job stopped, which the provider words identically.

## Why telling them apart needs its own machinery

The class is not in the exception. Three layers each destroy part of the evidence:

- **rasterio** wraps whatever GDAL failed at in a `WarpOperationError` and discards the cause, so
  a corrupt file and an expired credential arrive as the same sentence.
- **GDAL** reports some refusals only to its own log, and raises something unrelated. A refused
  range request comes back as an XML error document, which GDAL feeds to the TIFF decompressor;
  what surfaces is `ZIPDecode: Decoding error at scanline 0` — a corruption message for a
  provider refusal, and those two verdicts are opposites.
- **Dask** cannot serialise rasterio's GDAL error classes out of a worker, so it substitutes a
  plain exception holding one line of text and the chain is gone.

Hence the machinery, which is most of this document. A second log handler captures what GDAL only
logged, including from GDAL's own fetch threads, which write past the handler rasterio installs.
Captured lines are attached to the failing exception as evidence rather than acted on. A worker
plugin keeps causes serialisable, and a leg refuses to start unless every worker confirms it can.
One classifier in `duplicates.py` reads all of it together, and both sensors reach it through a
single context manager, so two callers cannot reach different verdicts about one failure.

## Why a wrong answer is expensive

A store's dates are append-only: a date can only be added after the newest one already there.
Give up on a date and it is gone. Give up too late and the cost is wall clock on a job that gets
dispatched again anyway. The pipeline therefore spends time rather than dates, and abandons a
date only on positive evidence that the imagery itself is unusable.

## One-area runs and global campaign runs

Everything above is shared. Both paths use the same classifier, the same copy ladder, the same
per-date retry, the same GDAL log capture, and the same rule about where a resumed run starts.

The global campaign adds a layer *above* a run. `ingest_zone_year` dispatches a cell's ingest as
a **leg** and retries the leg; a single-area run has no such loop, so it runs once and either
finishes or fails. These are campaign-only:

- the attempt budget, and the wall-clock deadline that bounds it
  (`max_leg_wall_clock_s`, extended while a leg is still committing dates)
- the long between-attempt wait for a provider refusal (`leg_refusal_backoff_s`)
- the leg-retry classifier, which decides whether a named failure is worth another dispatch
- the REPEAT test on a catalogue refusal, which needs two attempts to see and so cannot live
  inside one

The rest is organised by what failed:

- **The catalogue would not answer.** Nothing has been read, so nothing is at risk but time.
- **A read would not produce pixels.** The expensive decision, and most of the machinery.
- **The next run has to handle what was lost**, starting from what the store holds rather than
  what the last run intended.

Every section below opens with a line saying which of the five classes it answers, whether it is
part of the evidence machinery, and whether it exists only on the campaign path.

## How a failure is decided

*Answers all five [classes](#the-five-classes-of-failure). This is the decision the rest of the
document feeds.*

Reading one satellite image can fail for very different reasons, and the right answer to each is
different — sometimes opposite. A failed read is asked one question, once, and gets exactly one
answer.

```
                            a read fails
                                 |
                                 v
                  +-------------------------------+
                  |  What does the error say the  |
                  |  problem actually IS?         |
                  +-------------------------------+
                                 |
   our own login is wrong  <-----+-----> the provider said "no"
   (bad key, expired token)      |       (busy, throttling, 500s, "access denied")
        |                        |            |
        v                        |            v
   STOP the job.                 |       WAIT, then ask again. Their bad day,
   No retry and no other         |       not our data. Minutes, not seconds.
   copy fixes our own key.       |
                                 |
   the request is wrong  <-------+-------> the file is damaged
   (400, 401, a bad URL)         |         (won't decompress, truncated,
        |                        |          missing a band we need)
        v                        |            |
   STOP the job.                 |            v
   Every copy is fetched         |       Try ANOTHER COPY of the same
   the same way, so no           |       image. Providers often publish
   copy will read either.        |       the same scene twice. Only if no
                                 |       copy is left, skip this one date.
                                 |
                                 +-----> the file was never published
                                 |       (404, "no such key")
                                 |            |
                                 |            v
                                 |       Same as damaged: another copy first,
                                 |       then skip the date if there is none.
                                 |
                                 +-----> WE CANNOT TELL
                                              |
                                              v
                                         Fail the job so it runs again later.
                                         Never skip a date on a guess.
```

The last branch is the important one. **A date is only ever abandoned on positive evidence that the
image itself is unusable.** Anything we cannot explain fails the job instead, which costs time and
is recoverable, rather than costing a date, which is not.

### Waiting: where it happens changes what it costs

*[Provider system failure](#the-five-classes-of-failure), and the guard that keeps [authentication
failure](#the-five-classes-of-failure) from buying the same patience.*

Two different budgets, for one reason:

```
  waiting INSIDE a running job     ~ minutes    the machines it rented sit idle, so this is expensive
  waiting BETWEEN attempts          ~ tens of   the machines are already released, so this is nearly
                                      minutes   free — patience goes here
```

A provider having a bad minute is ridden out inside the job. A provider having a bad half-hour is
better handled by letting the job fail, releasing the machines, and trying again later — a restart
begins the day after the newest date the store holds, so nothing is redone (see *Where a resumed
run starts*).

One extra guard: the long wait is only granted after a job has already read something successfully.
"Access denied" looks identical whether the provider is misbehaving or our permissions are simply
wrong, but wrong permissions fail the very first image, while a provider wobble arrives after the
job has already been served. The first successful read is what earns the patience.

## The catalogue would not answer

Both of these happen before a single pixel is read, so the remedy is always to ask again, ask
differently, or stop the leg, never to abandon a date.

### When the catalogue refuses: naming the request, and telling the two refusals apart

*[Provider system failure](#the-five-classes-of-failure), split into rate and request. The REPEAT
test is [campaign only](#one-area-runs-and-global-campaign-runs).*

`catalogue_refusal.py` is where a refused query stops being anonymous. Two things about the
client stack make that necessary:

- **The request is discarded on the way up.** `StacApiIO.request` catches every transport failure
  and re-raises `APIError(str(err))`, which names only the host and endpoint path. A STAC search
  is a request **body**, so the collection, window, bbox and page are gone, and without them a
  refusal cannot be narrowed to a month or a page, reproduced, or reported upstream.
- **Our layer sits ABOVE a retry ladder, and only partly behind it.** For a force-listed status
  what escapes is the ladder reporting its own exhaustion, a much stronger statement than one
  error response; for a status kept out of that list (502) the first refusal arrives directly.
  `CatalogueRefusal.exhausted` records which.

So `_query_stac_items` pages explicitly (`pages_as_dicts`) and wraps **only the page fetch** in a
`CatalogueQueryError` carrying a `CatalogueRequest` — wrapping the page body too would classify
our own validation failures as someone else's outage. Opening the catalogue is page 0, named
separately so a root outage is not attributed to a window never asked for.

```text
CATALOGUE REFUSED collection=sentinel-2-l2a window=2021-09-01/2021-10-02
                  bbox=-3.0000,50.0000,-2.0000,51.0000 page 3
                  with HTTP 502 without being retried after 500 item(s)
                  — classified upstream-error:502
```

The **classification** separates refusals that arrive as one exception type from one endpoint
and need opposite responses:

| refusal | statuses | what it claims | response |
|---|---|---|---|
| `LOAD` | 429, 503 | the upstream names ITSELF as the constraint | wait, however often it recurs — an upstream naming its own load is the one refusal patience actually fixes |
| `UPSTREAM_ERROR` | 500, 502, 504 | the upstream failed to PRODUCE an answer | retry once; a repeat settles it as deterministic |
| `UNKNOWN` | anything else | no readable status | behave as the default does: retry |

A `LOAD` verdict draws the **expansive retry**: the leg-retry ladder's long, doubling delays,
granted without counting against the attempt budget for as long as the upstream keeps naming
itself. Everything else gets the ordinary attempt limit. The two named sets must jointly cover
the ladder's `status_forcelist`, or a status the ladder retries but the taxonomy does not name
falls to `UNKNOWN` and keeps that expansive retry forever; a unit test asserts the containment.
The converse is deliberate: the taxonomy names 502, which the ladder does **not** retry, and a
second test pins that exclusion.

**The ladder it must cover.** `_query_stac_items` configures retries at the HTTP layer via a
custom `urllib3.Retry` built by `make_logging_retry()` (`_http.py`, shared with the CMR Granule
query) and passed into `StacApiIO` (`total=8, backoff_factor=2, status_forcelist=(429, 500, 503,
504)`). The subclass logs each attempt at WARNING — urllib3 otherwise retries silently inside the
`HTTPAdapter`, making a slow query indistinguishable from a hang. Because `search.items()`
paginates lazily each page fetch is a separate HTTP call, so retrying at the adapter recovers a
transient 5xx on page N in place instead of throwing away prior pages and restarting the whole
query. `StacApiIO`'s own default `max_retries=5` passes a bare int to urllib3, whose empty
`status_forcelist` means 5xx is **not** retried, so the explicit `Retry` object is required.

**Why 502 sits outside it.** An Earth Search page refusal then arrives unretried, and the
date-window re-cut described in
[Appendix A](../src/tessera_embeddings/ingest/README.md#appendix-a--the-earth-search-response-cap-in-detail)
starts immediately rather than after the ladder's backoff; a transient 502 is absorbed by the
attempt budget owning the leg. The CMR Granule query keeps its own ladder
(`opera_query._CMR_RETRY`), 502 included, because nothing has measured a response-size cap there.

The status is read from the exception **chain**, not the message: `pystac_client` re-raises
without `from`, so the evidence sits under `__context__` on urllib3's exception. The message is a
documented fallback for a refusal that crossed a boundary carrying no chain.

**A status is necessary and not sufficient.** A gateway can fail for minutes and recover, so one
exhaustion is not proof of a defect. What settles it is a REPEAT — the identical request refused
the identical way on a later attempt, and that belongs to whoever holds the attempt budget,
`ingest_zone_year`'s leg loop: this module classifies, the budget holder supplies the repeat. The
two live in separate deployment runs, so the only thing crossing between them is failure text —
hence one whitespace-free token under a stable name (`CATALOGUE_REFUSAL=`), matched by name and
never by position, covering exactly the fields that decide the answer (collection, window, area,
page) and nothing that varies between attempts. A counter or timestamp inside it would make every
refusal unique and the repeat check dead code.

**Attempts are the only thing those budgets count; elapsed time has exactly one bound.**
*(Campaign only — a single-area run has no leg loop.)* Each
page fetch gets 9 HTTP attempts across 364 s of backoff before anything above sees a failure, and
every budget above it — leg, cell, zone round — treats the layer below as one try. None reads a
clock, and expansive backoff makes the clock the axis that grows without limit.
`IngestSettings.max_leg_wall_clock_s` bounds it in the leg loop: once the deadline passes, the
loop refuses to START another attempt. A running leg is never measured against it, so the worst
case is the deadline plus one final attempt. Failing the cell this way costs latency, not work —
the cell returns to the work list and a later dispatch resumes from the dates already committed.
The derivation is in `context_docs/ingest/source-read-failures.md` (cause 3).

**Two things stop that bound refusing an attempt a leg had the budget for.**

The retry ladder DESCENDS rather than ending the retry. Backoff doubles per attempt, so the rung
an attempt has escalated to can be longer than the deadline has left even while a shorter rung
fits easily. The rungs beneath are the same policy applied one escalation earlier, so the loop
takes the longest rung that FITS and only a leg with no room for even the base rung is refused.
It does not cap the wait to the REMAINDER: waiting exactly what is left makes the next dispatch
land on the deadline every time, turning a race into a guarantee of the thing the deadline
forbids.

A leg that is still COMMITTING DATES earns more deadline, by
`IngestSettings.leg_progress_extension_s`, because a deadline counted from the first dispatch
charges a leg for every prior attempt's productive work and cannot tell a pathological cell from
one working steadily. Progress is read from the leg's own child store through the same
`get_existing_dates` the ingest resumes from, so parent and leg cannot disagree, and a store that
cannot be read earns nothing. A grant is a FIXED size and each must be PAID FOR by dates
committed since the previous grant, which also limits the rate: every ask sits after a failed
attempt and the asks within one attempt compete for the same growth, so at most one is paid. The
ceiling is `max_leg_wall_clock_s + (max_leg_attempts - 1) * leg_progress_extension_s`; a leg that
commits nothing never leaves `max_leg_wall_clock_s`, and an extension of 0 restores the plain
deadline.

`source_coverage.py`'s preflight probe deliberately does **not** use any of this. Every failure
of that probe is already INCONCLUSIVE by design, which is the right answer for both refusals at
once, so telling them apart would buy nothing.

### When the archive says "success" but sends something that is not JSON

*[Bad response body](#the-five-classes-of-failure).*

Asking the archive for a page of radar granules normally returns a success code and a JSON
document. Occasionally it returns a success code and a body that is not JSON at all — an error
page, or a document cut off partway through.

**This slips past every defence we have.** Everything that decides whether to retry a request
looks at the response's status code, and here the status code is fine: it says success, and by the
only measure those checks apply it *was* a success. Only the body is wrong, and nothing inspects
the body. The request sails through the retry logic untouched and fails later, when something
tries to read it as JSON, with a message that says only:

```
Expecting value: line 1 column 1 (char 0)
```

That line names no address, no status, nothing about what arrived, and not even which of the
several services we query was the one that broke. In production it ended a radar run that had been
working for half an hour.

**So the page is simply asked for again**, a small fixed number of times. This is deliberately
narrow: every other kind of failure is left exactly as it was, and a server error is not re-asked
here, because the ordinary retry logic has already waited and tried for that one. The re-ask uses
the position marker the archive itself gave us, so it asks for the same page rather than the next
one and cannot step over granules.

If the retries are used up, the failure now describes itself: which address answered, what status
it gave, what kind of document it claimed to be sending, and how big it was. A document claiming
to be JSON alongside a body that will not parse means it was cut off; one claiming to be a web
page means an error page was substituted.

**The body that arrived is written to the log, and deliberately kept out of the error message.**
Whether to run a failed leg again is decided by searching the failure's text for certain words,
and the body is text the provider chose rather than us — so an error page containing one of those
words could flip a leg that should have been retried into one treated as permanently dead, costing
a whole zone-year. In the log it is just as readable and steers nothing.

**The credential requests never log their body at all.** They use the same helper, because they can
fail the same way, but with the body capture switched off: a credential document cut off partway
through is precisely the one that fails to parse, and its opening characters are the credential.

## A read would not produce pixels

This is where the cost asymmetry bites, and where most of the machinery is. The sections below
follow one failure outwards: where the retry sits, how a corrupt object is told apart from a
provider having a bad hour, what to do when GDAL declines to say which it was, and what the radar
path does differently because it has no second copy to fall back on.

### GDAL network tuning

*[Provider system failure](#the-five-classes-of-failure): absorbs the transient ones before they
reach a classifier.*

`configure_gdal_environment()` (in [`config/environment.py`](../src/tessera_embeddings/config/environment.py)) must be
called before importing `rasterio` or `odc.stac`. It sets GDAL config options for network
resilience (retry counts, timeouts, connection pooling) that affect all subsequent COG reads.

### Where the retry sits, and how a failed date is attributed

*[Machinery](#why-telling-them-apart-needs-its-own-machinery): where evidence is gathered, and how
a failure is tied to a date and an ROI.*

`roi_processing.source_read_retrying` wraps the point where a date's graph is first *computed*.
S1's read happens inside its write's `compute()` and is already covered by the write retry; S2's
fires earlier, in its coverage gate. Scoped per date deliberately: a task-level retry would re-run
the whole multi-day loop, so `tasks/ingest.py` refuses `@task(retries=...)`. Unlike the write
policy it is **not** narrowed by exception type — reads fail through rasterio, GDAL/CPL, botocore
and bare socket timeouts, a read is idempotent, and enumerating those surfaces risks a new
transient class becoming fatal.

**A failed date must say which date, and on which ROI.** Per-date telemetry is emitted *after* a
date commits, so the furthest date in a log is the last one that WORKED and a failure otherwise
leaves no trace. `roi_processing.read_failure_context` emits `READ FAILED roi=… date=… items=…
first=…` with the traceback on both sensors' per-date paths. `roi=` is what makes it attributable:
the exception is raised on a Dask worker whose log stream is an ECS task id, so without it the
same text appears for every zone and belongs to none. The traceback recovers rasterio's cause,
which reports only `Read failed. See previous exception for details.` — GDAL's actual reason is
discarded unless the chain is logged. It is also where the reason GDAL never raised is attached;
see *When GDAL logs the reason instead of raising it*.

### When a source object will not read

*[Data corruption](#the-five-classes-of-failure) and [missing data](#the-five-classes-of-failure),
told apart from a refusal by [reading the
chain](#why-telling-them-apart-needs-its-own-machinery).*

Some published objects are corrupt: a tile of the COG will not inflate, and no retry of any
length recovers it. That is a different condition from a throttle or an expired credential,
which look similar coming out of the loader — `rasterio` wraps both in a
`WarpOperationError` that discards the cause — so `is_unreadable_source` inspects the whole
exception chain and matches only the codec-level signatures, excluding the credential and
throttle markers explicitly. It fails CLOSED: anything unrecognised propagates rather than
being treated as bad data, because responding to a bad minute by reading worse imagery is
the one outcome the recovery must never produce.

**An intact chain is still only what the reader chose to RAISE.** GDAL states some refusals in
its own log and raises something else entirely, and the section *When GDAL logs the reason instead
of raising it* below is what closes that.

**The chain only exists if something kept it**, and a leg **refuses to start** unless every
worker confirms it can: a job that cannot explain its own failures can quietly ruin a dataset. The
read fails on a Dask worker, and rasterio's GDAL error classes cannot be serialised out of one by
default — Dask substitutes a plain `Exception` holding the wrapper's repr, so what arrives is one
line with no cause and every predicate here has nothing to read.
`loader_failures.keep_causes_picklable`, installed on every worker by the same plugin as the
object capture, is what makes the cause arrive. It is best effort, so `cause_was_flattened`
recognises a failure that arrived without one and `read_failure_context` logs `READ CAUSE LOST`.

**An object that was never published counts too, and needs its own markers.** Every
codec-level signature comes from a BLOCK READ, and a missing object fails at open before any
block is requested. `ObjectNotFound`, `NoSuchKey` and `The specified key does not exist` cover
the three layers that surface it, matched only alongside the source reader's own vocabulary
(`RasterioIOError`, `WarpOperationError`, `CPLE_`, `HTTP response code:`). That pairing is what
makes them mean SOURCE: those strings belong to the S3 layer that every S3 client in the process
shares — `icechunk`'s error enum carries two verbatim — so unpaired they would record a hole in
the destination store as provider data loss. `NoSuchBucket` is deliberately excluded: a vanished
bucket is systemic and must fail the leg on its first date.

Nothing counts or caps these skips on the OPTICAL path: they are rare enough per granule that a
ceiling would only fire on a fault of another kind, and every date given up is logged, restated
in the end-of-run summary, and written to the store. The radar skip below does carry a ceiling,
because it answers a provider refusal, which arrives fleet-wide and all at once.

Past that point the response is a ladder, in `s2_roi.py`'s consume path:

1. **Attribute.** Ask the cluster which objects the loader gave up on
   (`loader_failures.collect_aborted_hrefs`) and map them back to tile-dates.
2. **Step down** those tile-dates to their next catalogue copy and re-prepare the date. The
   copy is older reprocessing, so this trades processing baseline for a date that reads.
3. **Give up, loudly,** when the implicated tile-dates have no copies left: the date is
   skipped rather than the leg failed, and a `DATA LOSS` line names the date, the objects and
   the scope. Nothing is written to the store — see *Why nothing records what was missed*.

Two properties are what the attribution step buys, and tests hold them rather than comments.
**Blast radius**: with attribution one bad object steps one tile-date, where without it every
duplicated tile-date steps together, downgrading hundreds of tiles on a wide ROI that read
perfectly well. **Termination**: a bad object whose tile-date has no alternate is given up
immediately, where without attribution the ladder first walks every *other* tile's alternates, at
a full re-read of the date per rung, to reach the same answer.

Attribution can fail — a worker that died with the read, a cluster already gone, a loader that
words its message differently, and the unattributed behaviour above is then the fallback. The
record says which happened: `scope=attributed` means the named objects are the ones that failed,
`scope=whole-date` means the failing object was not identified and the tiles listed are every tile
in the date.

The batched write path cannot reach the ladder — a batch is one graph and one commit — so it
isolates first: an unreadable source anywhere in a batch re-runs the batch's dates one at a
time, each then getting the per-date recovery. That isolation is what stops one corrupt
object from failing a zone-year identically on every retry.

### When the provider refuses the read

*[Provider system failure](#the-five-classes-of-failure) and [authentication
failure](#the-five-classes-of-failure), which share a message.*

An authorization refusal, a throttle and a server error are a different finding again. They say
nothing about the imagery — the same object read minutes earlier and reads again once the service
recovers — so no fallback copy helps and no date should be given up for one. That verdict is
reached by the same `is_unreadable_source` the section above describes: it answers one question
for both cases, returning true for codec-level damage and **false** for a refusal, which it tests
for first. There is no second predicate to ask.

**What a positive verdict buys is TIME, and nothing else.** It is passed to the shared write retry
as `wait_out`, and the policy re-attempts that one failure until it has spent `WAIT_OUT_BACKOFF_S`
of backoff. It is never spent on giving up a date, for the append-only reason above: a date
abandoned now cannot be written later. If the wait is not enough the write fails, the leg fails
with its time axis unmoved, and the leg's own retry re-offers the date in order.

The in-leg budget is `WAIT_OUT_BACKOFF_S`, which applies to every run. The between-attempt one,
`leg_refusal_backoff_s`, is campaign only: it is the delay before `ingest_zone_year` dispatches
the leg again.

Carrying the verdict between the two takes a type: the leg-retry layer sees only a failure DETAIL
string, and no marker on it can separate a refused read from a crash, since the wrapper discarded
the cause. A radar write that exhausts its in-leg budget on a refusal raises
`errors.ProviderRefusedReadsError`, whose name reaches the detail and is what `_leg_backoff_s`
keys the long delay on. Nothing else about the failure changes.

### When GDAL logs the reason instead of raising it

*The core of the [machinery](#why-telling-them-apart-needs-its-own-machinery): recovers a
[provider system failure](#the-five-classes-of-failure) that GDAL reported as [data
corruption](#the-five-classes-of-failure).*

Everything above reads the exception chain. Some of a read failure's reason never reaches it.

A refused object is not empty: S3 answers the range request with an XML error document, and GDAL
hands it to the TIFF decompressor, which fails on it — `ZIPDecode: Decoding error at scanline 0`,
sometimes `unknown compression method`. That is what gets raised. GDAL states the refusal as a
warning in its own log and raises nothing about it. The chain says the bytes are bad and the
log says the service refused, and those verdicts are opposites: bad bytes gives the date up,
refused waits and gives up nothing.

`loader_failures` closes it with a second handler on `rasterio._env`, the logger rasterio's CPL
error handler forwards to. `carry_logged_refusal` attaches what it collected to the failing
exception as a note, `_exception_chain_text` reads notes with the rest of the chain, and
`classify_read_failure` decides from all of it. Both sensors reach this through
`roi_processing.read_failure_context`, so it is one classifier over one set of evidence.

That handler only hears what reaches a logger, and most of it does not: rasterio installs its CPL
handler with `CPLPushErrorHandler`, which GDAL keeps **per thread**, so a message from one of
GDAL's own fetch threads goes to the process-wide handler and out to stderr, where no
`logging.Handler` can reach it. `hear_gdal_from_every_thread` gives that process-wide handler
somewhere to forward to, and `install_capture` installs it alongside the two log handlers. It
chains to the existing handler rather than replacing it, so GDAL's stderr line still appears and a
fatal error still aborts through it, and GDAL consults the reporting thread's own handler first so
nothing rasterio already forwards is duplicated.

Four properties are what make it safe to add evidence at all:

- **Only refusals are recorded.** A line is kept only if the classifier reads that line alone as
  `PROVIDER_REFUSED` or `OUR_CREDENTIAL`, so there is no second vocabulary to drift. Everything
  else GDAL says is dropped, which matters because GDAL probes for sidecars that were never
  published — a kept `HTTP response code: 404` is the marker for an ABSENT source object, and
  gives a date up.
- **The direction is bounded.** Refusal is tested before any statement about the bytes, so an
  attached line can only move a verdict into those two, never into `UNREADABLE` or `ABSENT`. A
  wrong attribution therefore costs patience rather than a date: the write spends its refusal
  budget, the leg fails with its time axis unmoved, and the date is judged alone on re-dispatch.
- **Only onto a source read failure**, gated by `is_source_read_failure` on the same
  `_SOURCE_READER_MARKERS` every other corroboration uses. A store conflict raised while some
  other read is being refused is still a store conflict, and a wait fixes nothing.
- **Reading the refusals does not consume them**, and they sit in a separate buffer from the
  aborted hrefs, which is still drained destructively. Two reads are in flight whenever the
  optical path pipelines a date, each inside its own `read_failure_context`, and a destructive
  collection let whichever failed first take the other's evidence. What keeps a stale line out is
  therefore its AGE: a read that logs a refusal and then succeeds drains nothing, so without an
  age bound its line would be inherited by whatever failed next and a genuinely corrupt object
  would read as a refusal and never step down the copy ladder. Each worker reports its lines' age
  by its own clock and the caller applies the cutoff **after** the round trip against its own, so
  nothing depends on the fleet's clocks agreeing or on the collection being quick.

The evidence is ATTACHED rather than answered: a caller that classified and discarded would hand
the next reader of the same exception the opposite verdict, and the radar path has two readers —
the retry policy that spends patience, and the handler that decides whether the date is lost.

Radar asks for it twice, and the second ask is what arms `wait_out`. `refusal_wait_out(client)` is
`is_provider_refusal` over evidence gathered at the moment the policy asks; a predicate reading
only the exception declines the outage the budget exists to outlast and spends the ordinary three
attempts on it.

**The evidence window is the WRITE, not the attempt.** The policy asks once per failed attempt,
but what it asks is whether this WRITE is being refused, and an outage states its refusal while it
is refusing rather than on the ladder's schedule. Re-armed per attempt, the question becomes "was
anything refused in the last few seconds" — which a recovering provider answers NO one attempt
before the write would have succeeded, withdrawing patience exactly when it was about to pay off.
One window per write also makes the two readings of one failure agree, since
`read_failure_context` judges the same failure over the whole write when the retry gives up.
Nothing is remembered: the window is derived, so evidence is re-read on every attempt, a write
whose window holds no refusal never waits, and `WAIT_OUT_BACKOFF_S` bounds one that keeps seeing
one. Each ask logs its verdict and how many refusal lines it read, because an attempt count cannot
separate "no refusal was logged" from "a refusal was logged and not read".

Optical does not pass `wait_out` at all: its answer to a refusal is a leg failure with the axis
unmoved, because its per-date remedy is the copy ladder and a long wait per rung would multiply
with it. The evidence still reaches optical through the same context manager, so a refusal there
is declined by `is_unreadable_source` and fails the leg rather than stepping copies.

The prior-success guard lives here too: `s1_roi` keeps one per-leg flag, set on the first
committed date, and the long wait is withheld until it is set.

Three shapes are excluded, each costing only the ordinary attempt limit. A credential fault on
THIS side, being repairable here. A refusal nothing attributes to the source reader —
`AccessDenied`, `SlowDown` and `InternalError` are S3's words, so they count only alongside
GDAL's own vocabulary, by the same pairing rule as the not-found markers above. And a refusal
carrying neither a name nor a status: a transport failure with no code, or a cause destroyed
crossing the worker boundary. `cause_was_flattened` says which of those last two happened, so a
leg reading without a decidable cause is visible rather than silent.

The two predicates are disjoint by construction, and stay so by sharing one classification rather
than keeping two lists in step: both read the same markers and the same HTTP status RANGES, so a
status nobody enumerated cannot be a refusal to one predicate and bad data to the other, and a
caller that knows only one of them cannot misclassify.

### The radar bounded skip (`s1_roi.py`)

*[Provider system failure](#the-five-classes-of-failure) and [data
corruption](#the-five-classes-of-failure) on a path with no second copy. The terminal ceiling is
[campaign only](#one-area-runs-and-global-campaign-runs).*

Every OPERA read on the radar path happens inside a date's write, so a failed read raises out of
the per-date loop. Until this skip, one refused read cost every LATER date in the window too: a
source that refused reads for thirteen minutes emptied 178 zone-years that had already committed
months of sound data.

The radar response is the tail of the optical one without the copy ladder, which radar has no use
for: OPERA publishes one copy of a granule, so there is nothing to step down to.

1. **Retry**, through the shared `store_write_retrying` policy, and for a provider refusal that
   arrived after a successful read, retry past the attempt limit, because waiting is the only
   response a refusal has. Radar is the one caller that asks for this.
2. **Fail the leg under a name the cell can act on** if that wait was not enough
   (`ProviderRefusedReadsError`), so the re-dispatch waits on the long schedule. No date is
   skipped and the time axis does not move.
3. **Give up the date** once that retry is exhausted, if and only if the failure is one the source
   is answerable for AND recomputes. One scope, `unreadable`, one remedy: a reprocessed copy at
   the provider. A refusal is deliberately not a second recoverable scope — giving up a date and
   then committing a later one puts the earlier one permanently below the append-only maximum, so
   the re-run meant to recover it is refused instead.
4. **Name it in the log**, per date and again in an end-of-leg summary. Nothing durable: the day
   is below the store's newest date by the time the next date commits.
5. **Stop past `MAX_GIVEN_UP_DATES`**, and stopping is TERMINAL. On the campaign path
   `TooManyGivenUpDatesError` is in the leg-retry classifier's non-retryable set, because nothing
   counted toward the ceiling can clear: every date reaching that counter failed for a cause that
   recomputes, so a re-dispatch would re-read the same objects to reach the identical answer.

A date offered by two consecutive batches is given up ONCE. Batch queries are padded a day either
side, so a boundary solar day comes back from two queries and would otherwise be listed twice and
cost twice.

## What a resumed run does about it

A leg that failed is dispatched again, and what it does next is decided by the store rather than
by anything the failed run recorded. This is the constraint the rest of the section is written
around.

### Where a resumed run starts

*Not a class — the consequence every class is judged against.*

A store's dates can only be added in order, newest last. Slotting one into the middle would mean
shifting every chunk after it, and a Zarr store's chunks sit at fixed positions — there is nowhere
to shift them to. So **every day at or before the newest date a store already holds is closed to
that store for good**, whatever the imagery for that day later turns out to be.

Most runs are resumes: a leg dispatched for a calendar year fails part way through and is
dispatched again, so everything below the line it reached is settled and only the days above it
are open. **A run therefore starts the day after the newest date its own store holds**
(`resume_window_start` in `solar_days.py`). Three questions, asked before the catalogue is
queried and before any date is prepared:

1. **Does the window end before it begins?** That is a caller mistake and always was, so the run
   refuses it. Asked first, so a misconfigured leg can never be reported as a successful skip.
2. **Is the store's newest date already at or past the window's last day?** Then nothing in this
   window is open to it. The run reports a skip and stops without querying anything. The
   comparison is against the raw date, not against anything derived from it: a value floored to
   its month first can precede the window's end while the date it came from does not.
3. **Otherwise, begin the SOLAR day after that newest date** — not the first of its month,
   which would re-offer days already below the line.

### Why nothing records what was missed

*Not a class — why no class is recorded once its date is closed.*

Once a day is closed, what happened on it stops mattering: an image that would not read this
morning and reads this afternoon still cannot be written. A ledger of missed days would unlock no
action, and would be deleted along with the mosaic it described.

Coverage questions are answered by the published product instead, which carries per-pixel
observation counts and month masks (`config/store_layout.py`). Downstream every absence is the
same absence — no pass, too cloudy, unreadable file — so nothing consuming a mosaic tells them
apart anyway.

What the store does carry is `assessed_window`, the date range a leg examined in full, so an
empty month inside it reads as "we looked and there was nothing" rather than "no run got here".
See [Recording the window an ingest examined](../src/tessera_embeddings/ingest/README.md#recording-the-window-an-ingest-examined).

A lost day still produces a `DATA LOSS` line and an end-of-leg summary. What it does not produce
is a record anything later reads.

---

