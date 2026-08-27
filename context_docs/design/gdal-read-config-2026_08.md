# An operator's GDAL override does not reach the odc read path

Record, 2026-08-27, from a live S3 us-west-2 degradation. **Scope is much smaller than this
investigation first claimed, and both corrections are the point of the record.**

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

## Correction 1: only three options are affected, not eight

**The first version of this document said five settings were "absent from the read path entirely".
That was wrong.** GDAL falls back to `os.environ` for any option the caller has not named
explicitly, and `configure_gdal_environment()` puts all eight there — including on remote workers,
verified by launching one from a deliberately emptied environment and finding a value that could
only have come from our own setup code running there.

So the affected set is exactly the three odc names. Everything else already reached the reader.

## Correction 2: the ladder is exponential — and odc's config is MORE patient than ours

`GDAL_HTTP_RETRY_DELAY` is the BASE of a doubling ladder with no cap, not a fixed wait. Measured
against a server refusing every request, arrival times recorded server-side:

```
0.5, 1.01, 2.07, 4.92, 10.96, 24.82, 52.36, 105.94 s
```

Total scales linearly in the base (confirmed at a ratio of 9.99 for a tenfold change). **The two
shadowed settings must be read together, and doing so reverses the obvious conclusion:**

| config | retries | base | give-up per unreadable object |
|---|---:|---:|---:|
| **odc** (what the read path runs) | 10 | 0.5 s | **~14 min** |
| **ours** (what the environment carries) | 5 | 5 s | **~3 min** |

**odc is ~5x MORE patient than our own defaults.** Forcing our values onto the read path would have
REDUCED patience there — the opposite of what the outage that prompted this wanted. That is the
strongest argument for the no-code outcome below, and I had it backwards until review.

For scale: ten retries at a base of **5 s** would be ~2.5 h per object, and the S2 coverage gate
wraps the read in `source_read_retrying()` (`SOURCE_READ_ATTEMPTS = 8`, so **seven** further
attempts after the first) while `max_leg_wall_clock_s` cannot interrupt a running leg. **The pair is
a wall-clock budget, and the two settings move it in opposite directions.**

## What shipped: nothing but this record

**No code change.** The gap is real — an operator's override of the three odc-shadowed options
never reaches the imagery path — but every way of closing it is worse than leaving it open:

* **Forward options whose value differs from our default.** Breaks for the most likely override
  there is: an operator setting `GDAL_HTTP_RETRY_DELAY=5`, our own documented default, is
  indistinguishable from no override at all.
* **Record provenance before `setdefault()`.** Breaks across process boundaries. A worker inherits
  our defaults from its parent's environment, so at the moment it runs, every option is already
  present and would read as operator-supplied — pushing our values into odc on every worker and
  changing the read path by accident.
* **A dedicated override interface.** Works, and is a new public knob plus its plumbing, for a
  capability nobody has yet needed: no incident so far has been handled by tuning GDAL at runtime.

So the honest outcome is the knowledge, not the code. `config/environment.py` carries a comment
where someone would go to change these values, saying which three odc shadows and what the ladder
costs. **Reopen this if an incident is ever actually blocked on tuning one of the three** — that
is the evidence the dedicated interface would need, and it does not exist yet.

## Validation

17 ingest runs against real Sentinel-2 imagery (6 baseline, 6 branch, 5 with multiplexing off): all
**byte-identical output stores and identical committed date sets**. Branch 1.1% faster. Slowest
single read 6.92 s against the 120 s per-request cap — 17x headroom.

`GDAL_HTTP_MULTIPLEX` was kept. The caution in our source refers to a macOS development problem, and
the option has been active in production all along through the environment, so removing it would be
a behaviour change rather than a rollback. With it off, one read stretched to 93.66 s — the only
measurement anywhere near the cap.

## Open question, deliberately not decided here

Whether the 5 s base is right at all. It buys patience through an outage — the one that prompted
this lasted 90 minutes — and costs a run that could absorb roughly two dozen unreadable dates the
ability to absorb more than about two. Nothing measured settles it. Changing it is a one-line edit to the
`os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", ...)` call in `config/environment.py` — there is no
`GDAL_READ_OPTIONS` collection, this change having shipped no code — and this document is the
context for whoever makes that call.
