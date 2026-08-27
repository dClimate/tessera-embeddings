# The optical read path ran a GDAL configuration nobody had written down

Investigation record, 2026-08-27, found while answering "are these HTTP codes adequately caught?"
during a live S3 us-west-2 degradation. **Answer: the codes were classified correctly, but the
retry budget that absorbs them was 10x shorter than the repo says, and three of the eight
read-relevant GDAL options were absent from the read path entirely.**

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
which odc then ships to its readers as an explicit `rasterio.Env`. **Explicit Env options beat
process environment variables**, so ours could not have applied even where they were also set.

## What was actually in force, versus what the repo said

| option | repo said | actually in force | consequence |
|---|---|---|---|
| `GDAL_HTTP_RETRY_DELAY` | 5 s | **0.5 s** | ~5 s of total patience instead of ~50 s |
| `GDAL_HTTP_MAX_RETRY` | 5 | 10 | more attempts than intended, harmless |
| `GDAL_HTTP_TIMEOUT` | 120 s | **absent** | GDAL default |
| `GDAL_HTTP_LOW_SPEED_LIMIT` | 1 B/s | **absent** | **no hung-connection watchdog** |
| `GDAL_HTTP_LOW_SPEED_TIME` | 60 s | **absent** | as above |
| `GDAL_HTTP_MULTIPLEX` | YES | **absent** | more connection overhead |
| `GDAL_HTTP_MERGE_CONSECUTIVE_RANGES` | YES | **absent** | more round-trips |
| `GDAL_DISABLE_READDIR_ON_OPEN` | EMPTY_DIR | EMPTY_DIR | matched by luck |

`GDAL_NUM_THREADS` and `GDAL_CACHEMAX` are process-wide rather than per-Env and are unaffected.

## Why it mattered on 2026-08-27

S3 us-west-2 returned **2,120 × HTTP 500 and 174 × HTTP 503 in 90 minutes** across three unrelated
buckets (`sentinel-cogs`, `asf-cumulus-prod-opera-products`, and our own staging bucket answering
`SlowDown`). GDAL retried every one — **2,341 of them at 0.5 s**, which is how the defect was
noticed at all: the log disagreed with the configured value.

A 0.5 s delay over 10 attempts is about **five seconds** of patience. Any burst outlasting that
exhausts the budget, GDAL proceeds into the codec with an unusable buffer, and the failure surfaces
as `ZIPDecode:Decoding error at scanline 0` with the status left behind in GDAL's log. Our intended
5 s delay would have given roughly **fifty seconds**.

**The classification itself was never broken**, and that is worth stating because it was the
original question. `_HTTP_STATUS_RE` already matches GDAL's real wording
(`HTTP error code for <url> range X-Y: NNN` — note *error*, not *response*), 500 and 503 fall in
`_names_a_transient_refusal`'s `>= 500` branch, and **zero dates were lost in the two hours of the
storm.** The gap was patience, not verdicts.

## A withdrawn inference this also corrects

While ruling out a self-inflicted cause I checked for `CURL error`, `Operation too slow` and
`Timeout was reached`, found **zero**, and concluded that CPU-starved workers tripping the low-speed
watchdog were not the mechanism. **That inference was void**: `GDAL_HTTP_LOW_SPEED_LIMIT` was not
active on the read path, so it could not have fired and its silence proved nothing. With the
watchdog now applied, those messages become meaningful evidence for the first time.

## The fix

`GDAL_READ_OPTIONS` in `config/environment.py` is now the single definition, consumed twice:
`configure_gdal_environment()` puts it in the environment (direct rasterio use), and
`configure_odc_rio()` hands the same dict to `odc.stac.configure_rio(cloud_defaults=True, ...)`.
Merge order there is `{**GDAL_CLOUD_DEFAULTS, **ours}`, so our values win and odc's remain the base
for anything we do not name. `ingest/stac.py` calls it immediately after importing `odc.stac`.

`MAX_RETRY` is deliberately kept at odc's **10** rather than reverted to our 5: ten attempts at a
5 s delay is the more patient policy, and patience is what this outage wanted.

Three tests pin it, in `tests/unit/test_environment.py`:

* the odc constant itself, so a future odc bundling these settings fails loudly rather than making
  our override a silent no-op;
* every read option present in `capture_rio_env()` — the env odc actually ships to its readers,
  which is the only place the assertion means anything;
* our retry delay beating odc's.

## What is not addressed

**The `GDAL_HTTP_RETRY_DELAY=5` value is not itself derived from a measurement.** It is the value
the repo always intended; this change makes it real. Whether 5 s x 10 is the right budget against a
regional S3 degradation is a separate tuning question, and the honest reopen criterion is a repeat
of this outage with the fix in place.

**Not applied to the running campaign.** The live fills run the installed package; this takes effect
on the next image build.
