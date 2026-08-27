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

## Correction 2: the retry delay is exponential, and the budget is large

`GDAL_HTTP_RETRY_DELAY` is the BASE of a doubling ladder with no cap, not a fixed wait. Measured
against a server refusing every request, arrival times recorded server-side:

```
0.5, 1.01, 2.07, 4.92, 10.96, 24.82, 52.36, 105.94 s
```

Total scales linearly in the base (confirmed at a ratio of 9.99 for a tenfold change). At odc's
`MAX_RETRY=10`, a base of 5 s gives roughly **85 minutes for ONE unreadable object** — and the S2
coverage gate wraps that read in `source_read_retrying()` for 8 more attempts. **These two values
are a wall-clock budget, not a politeness setting.**

## What shipped

**No default changed.** `configure_odc_rio()` forwards to odc only an option whose environment
value DIFFERS from our own default — the signal that a human set it deliberately. With no override
present it does nothing, and odc behaves exactly as before.

The difference test is what makes this survive process inheritance: a worker's environment carries
our defaults down from its parent, so "was it in the environment already" cannot tell an operator
from an ancestor, while "does it differ from our default" can.

`GDAL_HTTP_MAX_RETRY` stays at **5**, its pre-existing value, rather than adopting odc's 10.

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
ability to absorb more than about two. Nothing measured settles it. Changing it is a one-line edit
to `GDAL_READ_OPTIONS`, and this document is the context for whoever makes that call.
