# Ingest source-read failures: the two causes behind `WarpOperationError`

Investigation record, 2026-08-04/05, `global-tessera-dev`. Traces
`WarpOperationError('Chunk and warp failed')` and `PermissionError: The provided token has
expired` to their exact causes, so the guards can be aimed at the right terms.

Companion to `ingest_optimization_campaign_2026_07.md` (the authoritative ingest record) and
`ingest_concurrency_investigation_2026_08.md` (the fleet-width contention work).

## Headline

`WarpOperationError('Chunk and warp failed')` is **not a cause**. It is rasterio's wrapper around
whatever GDAL failed at, and it **discards the reason**. Behind it are **two unrelated causes**
that need opposite responses:

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

### What the logs show

Split by source bucket, the 226 `Aborting load due to failure while reading` lines are:

| hour (UTC) | radar | optical |
|---|---:|---:|
| 08-04 16:00 | 10 | 0 |
| 08-04 17:00 | 0 | 15 |
| 08-04 18:00 | 0 | 1 |
| 08-04 19:00 | 4 | 0 |
| 08-04 20:00 | 3 | 0 |
| 08-04 21:00 | 1 | 0 |
| 08-04 22:00 | 1 | 0 |
| **08-04 23:00** | **169** | 0 |
| 08-05 00:00 | 0 | 1 |
| 08-05 01:00 | 0 | 2 |
| 08-05 02:00 | 0 | 19 |

Radar is **188** and concentrated in one mass event; optical is **38** and scattered. The 23:00
hour carries 224 `response_code=403` lines against 400 warp-failure lines. No optical failure has
a 403 anywhere near it — `sentinel-cogs` is public and the reads are unsigned.

### The credential's own record

`set_s3_credentials` logs each broadcast with the credential's advertised expiry, which makes the
lifecycle directly measurable. Grouping those lines by flow-runner log stream over 21:00–01:00
(each stream is one leg, each with its own Dask cluster and its own credential):

| broadcasts | last broadcast | credential dies |
|---:|---|---|
| 1 (× 6 legs) | ~21:14 | **22:14–22:15** |
| 2 (× 3 legs) | 22:08–22:12 | **23:08–23:12** |
| 3 (× 5 legs) | 22:44–22:59 | 23:44–23:59 |

The two expired-token spikes are these cohorts dying:

- the six single-broadcast legs expire 22:14–22:15 → the **22:30 spike (95 errors)**
- the three two-broadcast legs expire 23:08–23:12 → the **23:15 spike (83 errors)**, with the
  403 burst leading it at 23:00

The five legs that kept refreshing on cadence do not appear in either spike. **The legs that died
are exactly the legs that stopped refreshing.**

### Why they stopped refreshing

ASF mints a 1-hour credential. `s1_roi.refresh_credentials_if_stale` renews at
`CRED_EXPIRY_MARGIN_SEC` (15 min) before expiry, giving a ~45-minute cadence — which the
three-broadcast legs show exactly. The renewal call itself never fails: counted in the same
15-minute buckets, renewals continue right through both spikes.

**The defect is that the check is only reached when the work loop advances.** It is called at a
batch boundary (`s1_roi.py:596`) and before each date's write (`s1_roi.py:654`) — so the credential
can only be renewed *between* units of work. Any single unit that outlives the remaining margin
has no opportunity to renew inside itself:

- a date's `write_day_windows` compute, which is where the source reads actually happen, and which
  under fleet-width contention can far exceed 15 minutes
- a batch prepare (STAC query plus graph build) ahead of the first write
- anything that stalls

This is a **positive feedback loop, and that is what makes it dangerous at campaign width**: the
slower a leg runs, the longer it goes without renewing; the longer it goes, the more likely its
credential dies; a dead credential fails every read, which stops progress, which guarantees no
further renewal. A leg that is merely slow converts into a leg that is dead.

### The second limb: the plugin distributes a snapshot

`_S3CredentialPlugin.__init__` freezes `_build_aws_env(creds)` into `self.env` at construction, and
`client.register_plugin` stores that pickled object on the scheduler. `setup()` runs on every
worker that joins later — the docstring is right that this is what makes it work under adaptive
scaling — but each late joiner receives **the credential as it was when the plugin was built**.

A worker joining N minutes after the last broadcast starts life with (60 − N) minutes of
credential, and past 60 minutes it starts with none. This is directly visible: one worker
registered its `s3-creds` plugin at 16:58:18 and took a 403 at 16:59:27 — **69 seconds into its
life**, on a credential it had just been handed.

Adaptive scaling means new workers join throughout a leg, so this limb fires continuously once a
leg's broadcast cadence lapses.

### What is *not* the cause

Each of these was excluded, and each would have implied a different and wrong fix:

- **Not a missing refresh.** The loop renews before every date's write, driven by the credential's
  own advertised expiry rather than a fixed cadence.
- **Not too small a margin.** 15 minutes comfortably exceeds a single radar date's write under
  normal conditions. The margin is not what breaks; being unable to *check* it is.
- **Not a batch outrunning the token.** The original suspicion. The per-date call already closed
  it — renewal does not wait for a batch boundary.
- **Not the known per-thread session cache.** `_patch_odc_thread_session_for_env_drift` already
  makes odc.loader's thread-local `AWSSession` self-invalidate on env drift, and it is installed at
  module import so it reaches Dask worker processes.
- **Not a failing renewal call.** Renewals continue throughout both spikes. This excludes the
  whole family of fixes aimed at the renewer: more retries around it, a longer margin, a different
  cadence. None would change anything.

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

### What the logs show

The 38 optical failures name a single object over and over:

```
sentinel-cogs/sentinel-s2-l2a-cogs/34/W/FA/2021/9/S2B_34WFA_20210908_1_L2A/B02.tif
```

It fails at **many distinct tile offsets** (X 1–10, Y 3–9; scanlines 1024 through 10240, i.e. Y
offset × the 1024-pixel tile height), across **at least seven independent Dask workers**, and again
five minutes later on retry with the same result. No 403, no HTTP error of any kind.

### Reproduced independently

Read directly from a laptop with **no AWS credentials at all**
(`AWS_NO_SIGN_REQUEST=YES`), walking every block window of band 1:

```
opened: 10980 x 10980  blocks [(1024, 1024)]  compress deflate
tiles ok=41 bad=80
```

**80 of 121 tiles are unreadable.** Two-thirds of the object is broken, permanently, at the
provider. No retry, credential, worker configuration, or ulimit has any bearing on it.

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
`catalogue_refusal_classification_2026_08.md`. The optical leg's STAC search was refused
deterministically (repeated 502s from `earth-search`) and burned the same budget for the same
reason — the wrapper discarded the request, so no marker could separate a busy catalogue from one
that cannot serve a particular query. The fix follows the prescription above: classify where the
failure happens, carry the request identity into the failure detail, and let the layer holding the
budget act on a REPEAT of it. Note that the "Caveat on the sample" 502s above are the first
recorded observation of that same upstream defect.

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
