# The optical read path ran a GDAL configuration nobody had written down

Investigation record, 2026-08-27, found while answering "are these HTTP codes adequately caught?"
during a live S3 us-west-2 degradation. **Answer: the codes were classified correctly, and the
retry budget that absorbs them was 10x shorter than the repo says.**

**Corrected 2026-08-27, after the live validation recorded at the end of this document.** The
first version of this note also claimed that five of the eight read-relevant GDAL options were
"absent from the read path entirely". That was wrong, and the sections below say where. GDAL
consults the process environment for any option the active rasterio `Env` does not name, so the
five odc never named were in force, at our values, the whole time. **One value changes on the
optical read path: `GDAL_HTTP_RETRY_DELAY`, 0.5 s to 5 s.** The second correction is larger: the
retry ladder is exponential, not fixed, so that single value moves the give-up time from about
15 minutes per object to about 2.5 hours, not from 5 seconds to 50.

## The mechanism

`config/environment.py` sets eleven GDAL options with `os.environ.setdefault`, and both the module
docstring and `ingest/stac.py` state the contract as "must be set BEFORE importing rasterio /
odc.stac to take effect". That is true for direct rasterio use. **It is not true for the odc read
path, which is where essentially every source byte is read.**

From `odc/loader/_rio.py`:

```python
GDAL_CLOUD_DEFAULTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "0.5",
}

def capture_rio_env() -> Dict[str, Any]:
    if _CFG._configured:
        env = {**_CFG._gdal_opts, "_aws": _CFG._aws}
    else:
        env = {}
    env.update(get_rio_env(sanitize=False, no_session_keys=True))
    env.pop("GDAL_DATA", None)
    if len(env) == 0:
        # not customized, supply defaults
        return {**GDAL_CLOUD_DEFAULTS}
    return env
```

Two facts make this bite:

* `_CFG` is only populated by `configure_rio` / `configure_s3_access`, and **we never called
  either.**
* `get_rio_env()` returns the options of the *active rasterio `Env`*, not the process environment.

So both inputs were empty and `capture_rio_env()` returned `GDAL_CLOUD_DEFAULTS` — three options —
which odc then ships to its readers as an explicit `rasterio.Env`.

**The first version of this note continued: "Explicit Env options beat process environment
variables, so ours could not have applied even where they were also set." That is the error the
whole overstatement rests on.** An explicit `Env` beats the environment only for the options it
*names*. `CPLGetConfigOption` resolves thread-local config, then global config, then `getenv()`,
so an option the `Env` leaves out still reaches GDAL from the process environment. Measured
directly on GDAL 3.12.1, inside `rio_env(**GDAL_CLOUD_DEFAULTS)` with our environment set:

| | inside odc's Env, our env vars set | inside odc's Env, environment scrubbed |
|---|---|---|
| `GDAL_HTTP_RETRY_DELAY` (odc names it) | `0.5` — odc's wins | `0.5` |
| `GDAL_HTTP_TIMEOUT` (odc omits it) | `120` — **ours, from the environment** | `None` |
| `GDAL_HTTP_LOW_SPEED_LIMIT` | `1` — ours | `None` |
| `GDAL_HTTP_LOW_SPEED_TIME` | `60` — ours | `None` |
| `GDAL_HTTP_MULTIPLEX` | `YES` — ours | `None` |
| `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES` | `YES` — ours | `None` |

The right-hand column is what the original note measured, and it is real — but a scrubbed
environment is not the deployed condition. `tests/unit/test_environment.py` now pins the
left-hand column, since it is the fact that decides how much of this is a live defect.

**The environment is populated on the worker too**, which is the other half of the argument and
was never checked before. The read task carries a `_BoaCorrectingDriver` defined in
`ingest/stac.py`; unpickling it on the worker executes that module, and the module body calls
`configure_gdal_environment()`. Confirmed by fingerprint: on a pre-fix worker launched from a
process with no `GDAL_*` variables at all, `os.environ["GDAL_HTTP_MAX_RETRY"]` read back `5` —
a value that exists nowhere but the pre-fix function body.

## What was actually in force, versus what the repo said

Read off a real Dask worker on both branches, `GDAL_*` scrubbed from the launching process, one
dump per worker from inside odc's `restore_env` — the context the read itself runs in. Four
workers agreed on each side.

| option | pre-fix, in force | post-fix, in force | changed? | how it reached GDAL pre-fix |
|---|---|---|---|---|
| `GDAL_HTTP_RETRY_DELAY` | **`0.5`** | **`5`** | **YES — the only one** | odc named it; ours displaced |
| `GDAL_HTTP_MAX_RETRY` | `10` | `10` | no | odc named it; our `5` displaced |
| `GDAL_HTTP_TIMEOUT` | `120` | `120` | no | process environment |
| `GDAL_HTTP_LOW_SPEED_LIMIT` | `1` | `1` | no | process environment |
| `GDAL_HTTP_LOW_SPEED_TIME` | `60` | `60` | no | process environment |
| `GDAL_HTTP_MULTIPLEX` | `YES` | `YES` | no | process environment |
| `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES` | `YES` | `YES` | no | process environment |
| `GDAL_DISABLE_READDIR_ON_OPEN` | `EMPTY_DIR` | `EMPTY_DIR` | no | both agreed |

What did change is *who names them*. Pre-fix odc shipped three option names to its readers and
GDAL filled the rest from the environment; post-fix odc ships all eight. The first version of this
note read the "absent" column off a clean-environment probe and reported those five as absent from
the read path. They were not.

That leaves a smaller but real reason to keep this change: an option carried only by the process
environment is one that any future odc release, or any caller that opens its own `Env`, can
displace silently — which is exactly what `GDAL_HTTP_RETRY_DELAY` shows happening.

`GDAL_NUM_THREADS` and `GDAL_CACHEMAX` are process-wide rather than per-Env and are unaffected.

## Why it mattered on 2026-08-27

S3 us-west-2 returned **2,120 × HTTP 500 and 174 × HTTP 503 in 90 minutes** across three unrelated
buckets (`sentinel-cogs`, `asf-cumulus-prod-opera-products`, and our own staging bucket answering
`SlowDown`). GDAL retried every one — **2,341 of them at 0.5 s**, which is how the defect was
noticed at all: the log disagreed with the configured value.

**The first version of this note said: "A 0.5 s delay over 10 attempts is about five seconds of
patience... Our intended 5 s delay would have given roughly fifty seconds." Both figures are
wrong, by about two orders of magnitude, because the ladder is exponential rather than fixed.**

`GDAL_HTTP_RETRY_DELAY` is the *base* of the ladder; GDAL roughly doubles it per attempt. Measured
against a local server answering 500 to everything, recording arrival times server-side so the
gaps are GDAL's own sleeps and not an inference from its logs — base 0.5 s, `MAX_RETRY=8`:

```
gaps between successive requests, seconds:
0.5, 1.01, 2.07, 4.92, 10.96, 24.82, 52.36, 105.94      total 202.6 s
```

The total is dominated by the last rung and scales linearly in the base, confirmed at
`MAX_RETRY=6`: 44.3 s at base 0.5, 442.5 s at base 5 — a ratio of 9.99 for a 10x base.

Extending the measured ladder to the `MAX_RETRY=10` that is actually in force:

| base delay | give-up per object | share of a 6 h `max_leg_wall_clock_s` |
|---|---|---|
| `0.5` (pre-fix, odc's) | ~15 min | ~4% |
| `5` (post-fix, ours) | ~2.5 h | ~41% |

So a burst had to outlast about **fifteen minutes**, not five seconds, to exhaust the pre-fix
budget — and when it did, GDAL proceeded into the codec with an unusable buffer and the failure
surfaced as `ZIPDecode:Decoding error at scanline 0`. `GDAL_HTTP_TIMEOUT=120` does not bound any
of this: it caps a single request, not the ladder.

**This is the open question the change leaves behind, and it is a budget question rather than a
defect.** Reads within a date are concurrent, so a date's failure time is about the ladder's
length rather than the sum over objects; a leg that could absorb roughly two dozen unreadable
dates inside its 6 h can now absorb two. Set against that, 2.5 h of patience may ride out a
degradation that 15 min would not — the recorded outage ran 90 minutes. Nothing measured here
settles which way that trade goes.

**The classification itself was never broken**, and that is worth stating because it was the
original question. `_HTTP_STATUS_RE` already matches GDAL's real wording
(`HTTP error code for <url> range X-Y: NNN` — note *error*, not *response*), 500 and 503 fall in
`_names_a_transient_refusal`'s `>= 500` branch, and **zero dates were lost in the two hours of the
storm.** The gap was patience, not verdicts.

## A withdrawal that is itself withdrawn

While ruling out a self-inflicted cause I checked for `CURL error`, `Operation too slow` and
`Timeout was reached`, found **zero**, and concluded that CPU-starved workers tripping the
low-speed watchdog were not the mechanism. The first version of this note then called that
inference void, on the grounds that `GDAL_HTTP_LOW_SPEED_LIMIT` was not active on the read path.
**The watchdog WAS active** — through the process environment — so the original inference stands.

Measured three ways against a server that sends headers and then trickles at 0.1 B/s:

| configuration | outcome |
|---|---|
| post-fix (all eight named) | aborts at **60.0 s** |
| actual pre-fix (odc's three named, our env vars set) | aborts at **60.0 s** |
| clean environment, odc's three only | **still hanging at 240 s** — no watchdog |

A separate reason the log search was weak evidence either way: the low-speed abort does not
surface as `Operation too slow`. The raised exception was
`RasterioIOError: probe.tif: ... Cannot read TIFF header` — GDAL logs the refusal and raises the
symptom. Grepping for the CURL wording will keep finding zero however the watchdog is configured.

## The fix

`GDAL_READ_OPTIONS` in `config/environment.py` is now the single definition, consumed twice:
`configure_gdal_environment()` puts it in the environment (direct rasterio use), and
`configure_odc_rio()` hands the same dict to `odc.stac.configure_rio(cloud_defaults=True, ...)`.
Merge order there is `{**GDAL_CLOUD_DEFAULTS, **ours}`, so our values win and odc's remain the base
for anything we do not name. `ingest/stac.py` calls it immediately after importing `odc.stac`.

`MAX_RETRY` is deliberately kept at odc's **10** rather than reverted to our 5: ten attempts at a
5 s delay is the more patient policy, and patience is what this outage wanted.

Four tests pin it, in `tests/unit/test_environment.py`:

* the odc constant itself, so a future odc bundling these settings fails loudly rather than making
  our override a silent no-op;
* every read option present in `capture_rio_env()` — the env odc actually ships to its readers,
  which is the only place the assertion means anything;
* our retry delay beating odc's;
* GDAL's environment fallback: inside odc's `Env`, the option odc names comes back as odc's and the
  ones it omits come back as ours. This pins GDAL's behaviour rather than ours, and it is what
  keeps the "before" state of this change honest — including the consequence that **removing an
  option from `GDAL_READ_OPTIONS` is a behaviour change, not a revert**, since the same constant
  fills the process environment.

## Live validation, 2026-08-27

Run against real `sentinel-cogs` imagery on a local Dask cluster of four worker **processes** —
the point being that odc's `capture_env()` runs in the client and `restore_env()` on the worker,
so a process boundary is what the primary claim needs. One ~15 km Iowa ROI (2011 x 1507 px at
10 m, EPSG:32615), a 2024-07-01..2024-07-16 window yielding 6 solar days of which 4 pass the
coverage gate. **17 legs**: 6 pre-fix (`origin/main` via `PYTHONPATH`), 6 post-fix, 5 post-fix with
`GDAL_HTTP_MULTIPLEX=NO`.

Not run on the dev ECS fleet. Nothing in the mechanism under test is platform- or
fleet-dependent — `capture_rio_env()` is pure Python and the GDAL fallback was measured directly —
and the client/worker boundary a local cluster provides is the boundary the claim is about. What a
laptop cannot represent is absolute read latency: it is *slower* than an in-region ECS read, which
makes the read-duration figures below conservative and the throughput comparison more sensitive to
a round-trip regression, not less.

### Options in force on a worker

Dumped from inside odc's `restore_env`, once per worker, with `GDAL_*` scrubbed from the launching
process so the worker's own environment is unambiguous. Four workers agreed on each side. The
result is the corrected table in "What was actually in force": **3 options named pre-fix and 8 named
post-fix, but 8 in force on both sides**, differing only in `GDAL_HTTP_RETRY_DELAY`.

### Pixels and committed dates

All **17** stores are **byte-identical** — every band (`blue`, `green`, `red`, `rededge1..3`, `nir`,
`nir08`, `swir16`, `swir22`), `scl`, the `easting`/`northing`/`time` coordinates, and the committed
date set (`2024-07-04`, `-07-07`, `-07-12`, `-07-14`) — across all three configurations, compared by
SHA-256 per array and per time index. Store attributes agree too, including `baselines_applied` and
`doy`; only `created_at`, `last_appended` and `ingest_code_identity` differ, as they must.

That includes the one leg containing a 93.66 s read (below): a slow read did not become a wrong one.

### Throughput

Nine consecutive legs, three per configuration, no retries and no `ERROR 1` in any of them, so
these are clean-path numbers. Median per-date `total` from the `Stage timings` line:

| configuration | legs | dates | median | mean | min | max | vs pre |
|---|---|---|---|---|---|---|---|
| pre-fix | 3 | 12 | 4.40 s | 4.81 s | 3.70 | 8.30 | — |
| post-fix | 3 | 12 | 4.35 s | 4.38 s | 3.80 | 5.40 | **-1.1%** |
| post-fix, `MULTIPLEX=NO` | 3 | 12 | 4.70 s | 12.62 s | 3.90 | 95.00 | +6.8% |

Expected, and observed: the retry settings are inert when nothing fails.

### Read durations against the 120 s cap

883 chunk reads. A chunk read may contain several range requests, so it upper-bounds the single
range request `GDAL_HTTP_TIMEOUT` actually caps.

| configuration | n | median | p95 | p99 | max | headroom under 120 s |
|---|---|---|---|---|---|---|
| pre-fix | 300 | 2.04 s | 3.01 s | 4.12 s | 9.28 s | 12.9x |
| post-fix | 322 | 2.06 s | 2.90 s | 3.34 s | 6.92 s | **17.3x** |
| post-fix, `MULTIPLEX=NO` | 261 | 2.03 s | 3.30 s | 6.80 s | **93.66 s** | 1.28x |

No read in any configuration timed out, so no single range request exceeded 120 s anywhere in the
sample. **The one read that came near the cap is on the `MULTIPLEX=NO` arm**, and it is a single
observation, so it is suggestive rather than conclusive — but it points away from removing that
option rather than toward it.

Worth separating from this change: `GDAL_HTTP_TIMEOUT=120` is **not new**. It was in force pre-fix
through the process environment. Whether 120 s is the right cap is a live question — a 93.66 s read
did happen — but it is a pre-existing one that this change neither creates nor worsens.

### `MULTIPLEX=YES` versus `NO`

The A/B the tree's own commented-out caution asked for. Errors: zero on both arms across five and
six legs. Pixels: identical. Throughput: `NO` is 6.8% slower at the median and carries the only
near-cap read in the whole sample. **`GDAL_HTTP_MULTIPLEX` stays.**

The caution it answers is retired by the correction above, not by this A/B: `MULTIPLEX=YES` has
been in force on the optical read path all along, via the process environment. The comment records
a macOS local-dev problem, and production has been running `YES` regardless. Removing the option
from `GDAL_READ_OPTIONS` would also remove it from the environment, making it a change of
production behaviour rather than a revert.

**Not measured: the A/B on Linux.** It was planned as the gate on turning a new option on. Since the
option was never off in production, that gate does not exist, and the evidence that retires the
caution is the fallback measurement rather than a platform comparison.

## What is not addressed

**The `GDAL_HTTP_RETRY_DELAY=5` value is not itself derived from a measurement.** It is the value
the repo always intended; this change makes it real. Its *cost* is now measured — about 2.5 h of
give-up per unreadable object against the pre-fix 15 min, on a 6 h leg budget — and that cost is
the one thing in this change that could make a degradation worse rather than better. Whether to
accept it, lower `GDAL_HTTP_MAX_RETRY` to bound the ladder, or leave `GDAL_HTTP_RETRY_DELAY` at
odc's 0.5 s is an open decision; the numbers to make it are in the section above. The honest
reopen criterion remains a repeat of this outage with whatever value ships.

**Not applied to the running campaign.** The live fills run the installed package; this takes effect
on the next image build.
