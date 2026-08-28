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
dominated by its last rungs. Forcing our values onto the read path would have REDUCED patience there.

**And note the last row.** Raising our base to 5 s while leaving odc's count of 10 would give a
worst case near **nine hours for one unreadable object**, not the 2.5 h an average-multiplier model
suggests. The S2 coverage gate then wraps the read in `source_read_retrying()` — `SOURCE_READ_ATTEMPTS = 8`
passed to `stop_after_attempt`, so eight attempts in total, seven after the first — while
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
