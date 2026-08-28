# odc shadows three GDAL options on the imagery read path

Record, 2026-08-27, from a live S3 us-west-2 degradation. **This shipped no code.** The scope is far
smaller than the investigation first claimed, and every closure of the remaining gap was worse than
leaving it open — both of which are the point of the record.

## The mechanism

`odc.loader.capture_rio_env()` composes its readers' GDAL environment from odc's own config object
and the active rasterio `Env` — never from `os.environ` — and returns its three-entry
`GDAL_CLOUD_DEFAULTS` when both are empty. odc then applies that as an **explicit** `rasterio.Env`,
and explicit Env options beat process environment variables.

```python
GDAL_CLOUD_DEFAULTS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_MAX_RETRY": "10",
    "GDAL_HTTP_RETRY_DELAY": "0.5",
}
```

## Correction 1: three options are shadowed, not five

**The first version of this document said five settings were "absent from the read path entirely".
That was wrong.** GDAL falls back to `os.environ` for any option the explicit `Env` does not name,
and `configure_gdal_environment()` puts all of them there — including on remote workers, verified by
launching one from a deliberately emptied environment and finding a value that could only have come
from our own setup code running there.

`configure_gdal_environment()` sets **ten** GDAL options. odc names three of them, so **seven remain
effective** and three are shadowed: `GDAL_HTTP_MAX_RETRY`, `GDAL_HTTP_RETRY_DELAY` and
`GDAL_DISABLE_READDIR_ON_OPEN`.

## Correction 2: the ladder is exponential — and odc's config is MORE patient than ours

`GDAL_HTTP_RETRY_DELAY` is the BASE of a doubling ladder with no cap. Measured against a server
refusing every request, arrival times recorded server-side:

```
0.5, 1.01, 2.07, 4.92, 10.96, 24.82, 52.36, 105.94 s
```

Total scales linearly in the base (confirmed at a ratio of 9.99 for a tenfold change). **The two
shadowed retry settings must be read together, and doing so reverses the obvious conclusion:**

| config | retries | base | backoff sleep before giving up |
|---|---:|---:|---:|
| **odc** (what the read path runs) | 10 | 0.5 s | **~14 min** |
| **ours** (what the environment carries) | 5 | 5 s | **~3 min** |

**odc is ~5x MORE patient than our own defaults.** Only the base is 10x smaller; the retry COUNT is
twice as large, and the ladder's total is dominated by its last rungs. Forcing our values onto the
read path would have REDUCED patience there — the opposite of what the outage wanted. That is the
strongest argument for the no-code outcome, and I had it backwards until review.

**Those figures are BACKOFF SLEEP ONLY.** `GDAL_HTTP_TIMEOUT=120` is not shadowed and still applies,
so a request that hangs rather than failing fast adds up to 120 s per attempt: roughly **15 min** for
five retries and **36 min** for ten, in the worst case. The real budget is the sleep plus the
requests.

For scale, ten retries at a base of **5 s** would be ~2.5 h of sleep alone. The S2 coverage gate then
wraps the read in `source_read_retrying()` — `SOURCE_READ_ATTEMPTS = 8` is passed to
`stop_after_attempt`, so **eight attempts in total, seven of them after the first** — while
`max_leg_wall_clock_s` cannot interrupt a running leg.

## What shipped: nothing but this record

**No code change.** The gap is real — an operator's override of the three shadowed options never
reaches the imagery path — but every way of closing it is worse than leaving it open:

* **Forward options whose value differs from our default.** Breaks for the most likely override
  there is: an operator setting `GDAL_HTTP_RETRY_DELAY=5`, our own documented default, is
  indistinguishable from no override at all.
* **Record provenance before `setdefault()`.** Breaks across process boundaries. A worker inherits
  our defaults from its parent's environment, so at the moment it runs, every option is already
  present and would read as operator-supplied — pushing our values into odc on every worker and
  changing the read path by accident.
* **A dedicated override interface.** Works, and is a new public knob plus its plumbing for a
  capability nobody has needed: no incident so far, including the one that prompted this, has been
  handled by tuning GDAL at runtime.

**So there is currently NO knob that changes those three options on the imagery path.** Editing
`os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", ...)` in `config/environment.py` changes the
environment, which odc shadows — it affects direct rasterio use only. Anyone who needs to tune
imagery-read patience has to build the dedicated interface first. **Reopen this if an incident is
ever actually blocked on that**; that is the evidence it would need, and it does not exist yet.

## Validation

17 ingest runs against real Sentinel-2 imagery — 6 baseline, 6 branch, 5 with multiplexing off.

**Every run produced identical embedding arrays and time indexes, and an identical set of committed
dates.** Not byte-identical *stores*: each fresh store stamps `created_at` and `last_appended` via
`utcnow_iso()`, and the runs used different builds whose manifest identities differ. Those metadata
fields were excluded from the comparison by design; the pixel data and the date coverage were not.

Branch 1.1% faster than baseline. **Slowest single chunk read on the multiplex-on arms: 6.92 s.**
Across all 17 runs the slowest was **93.66 s**, on a multiplex-off run.

`GDAL_HTTP_MULTIPLEX` was kept. The caution in our source refers to a macOS development problem, and
the option has been active in production all along through the environment, so removing it would be
a behaviour change rather than a rollback — and the multiplex-off arm produced that 93.66 s outlier.

**That 93.66 s is a CHUNK duration, not a single request**, and one chunk read issues several COG
range requests. `GDAL_HTTP_TIMEOUT=120` caps each individual request, so a sequence of short
requests can produce a 93.66 s chunk with nothing approaching the cap. The validation therefore says
nothing about timeout headroom in either direction.
