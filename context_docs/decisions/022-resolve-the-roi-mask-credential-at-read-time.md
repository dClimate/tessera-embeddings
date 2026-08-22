# 022 — Resolve the ROI mask's S3 credential inside the read, not when the graph is built

**Status:** Accepted (2026-08-22). Built.

## Context

A Sentinel-1 leg on dev wrote 88 dates over six hours and then died:

```
ingest/s1_roi.py            ingest_s1_roi_sar
storage/zarr_store.py:1651  write_day_windows
storage/zarr_store.py:1830  write_days_windows
storage/zarr_store.py:1607  _write_windows_overlapped
s3fs/core.py:1311           _cat_file
s3fs/core.py:460            _call_s3
PermissionError: The provided token has expired.
```

Line 1607 is the single `dask` compute the overlapped window write runs. The only fsspec read in
that graph is the ROI mask: `ingest.roi.read_roi_mask` builds it lazily, and the pixels are fetched
when the write computes, not when the reader was called.

Two readings of the error were on the table. The first was that the read declared no identity at
all and picked up the environment, which on the radar path holds ASF's OPERA token
(`ingest.auth.set_s3_credentials` puts it there for GDAL and never removes it). That reading does
not survive the evidence: the campaign already threads `iam_s3_storage_options` into the leg, and
88 dates of mask reads had succeeded — an OPERA-scoped token would have been refused on our own
bucket from the first date, not the eighty-ninth.

The second reading is what the code does. `read_roi_mask` called `da.from_zarr(..., storage_options=
resolve_storage_options(storage_options))`, which resolves the provider **once** and hands the
resulting `FsspecStore` — credential strings and all — into the task graph. On the radar path that
graph is built once per 30-day batch, in `_prepare_batch`, and then computed date by date for as
long as the batch's writes take. Botocore hands out a frozen role credential with as little as ten
to fifteen minutes of life remaining (its own advisory/mandatory refresh thresholds), so a batch
whose writes run longer than that presents an expired credential and the read fails.

## Reproduction

The failure reproduces without AWS, in seconds. A moto S3 server sits behind a front door that
records the access key that signed each request and can refuse a nominated key with S3's own
`ExpiredToken` response. The role identity is a `credential_process` that issues a new,
two-second-lived key per invocation, reachable only with the `env` credential provider removed;
the environment holds source-shaped credentials, as it does on a live leg.

Build the mask graph, mark every key that was valid at build time as expired, then write. Before
the change the write presented the build-time key and raised
`PermissionError: The provided token has expired.` from `s3fs/core.py:215` — the reported error,
from the reported frame. After it the write re-resolves, signs with a later key, and succeeds. The
source-shaped environment key is never presented in either case.

## Decision

**`read_roi_mask` opens the store inside each block read.** The array is assembled with
`da.map_blocks` over a closure that calls `resolve_storage_options` and `zarr.open_array` itself, so
the credential is no older than the read that presents it. Nothing else changes: the same provider,
the same call sites, the same returned dask array.

This completes a contract the code already documented rather than adding a new one.
`resolve_storage_options`'s docstring says a provider "is re-invoked at each read, which is where
the credential is actually consumed" — true for the eager readers beside it, and false for this one,
which invoked it once per *call* and read much later.

**Rejected: a session instead of a snapshot.** Handing s3fs an env-stripped `aiobotocore` session
(`storage_options={"session": ...}`) would resolve per request and refresh indefinitely, and it
pickles cleanly. It was dropped because s3fs replaces a caller-supplied session on some paths and
silently fell back to the environment in testing — measured, not theorised, and disqualifying for a
credential guard.

**Rejected: a longer guaranteed credential life.** Forcing botocore to refresh whenever the
remaining life is under some margin is four lines and no new mechanism, but it trades one constant
against another — the exact shape of mistake `_ASSUMED_ROLE_CRED_TTL` already records in
`providers/aws/credentials.py`. A batch long enough defeats any margin; resolving at read time
cannot be defeated that way.

## Cost

**MEASURED 2026-08-22, and it corrects the figure this section first carried.** The original claim
was "one or two metadata GETs" per block. It is **four**, and the request count for a whole mask
read **triples**. Measured against a real S3 (moto) by counting every call at `s3fs`' own
`_call_s3` choke point, for a 12-block mask:

| | old (`da.from_zarr`) | new (`da.map_blocks`) |
|---|---|---|
| store opens, graph build | 1 | 1 |
| S3 requests, graph build | 4 (metadata only) | 4 (metadata only) |
| store opens, full read | 0 | 12 — one per block |
| S3 requests, full read | 24 | 72 |
| ...of which chunk bytes | 24 | 24 — unchanged |
| ...of which metadata | 0 | 48 |

Six requests per block against the old two, so a flat **3x** whatever the block count. The reason it
is four rather than one is that zarr 3 probes both layouts on every open — `zarr.json` twice, plus
the v2 `.zarray` and `.zattrs`. Passing `zarr_format=3` to the closure's `open_array` would cut it,
and is deliberately NOT done here: it would refuse a v2 store that the old path read fine, and the
absolute cost does not justify narrowing what the reader accepts.

Absolutely, this stays small. Metadata GETs are a few hundred bytes; fsspec caches filesystem
instances by their options, so blocks sharing a credential — every block of a date, and every date
between botocore's refreshes — share one client and one connection pool; and every production
consumer slices the mask to live windows, so dask culls the opens along with the blocks (measured: a
one-block read opens the store once, not 3,876 times). It is negligible against the imagery reads
beside it. The figures are pinned by
`tests/unit/test_roi_mask_construction.py::test_the_store_is_opened_once_per_block_where_it_used_to_be_once`,
which asserts the table exactly so that making it worse requires editing the table.

**A second measured consequence: the graph is no longer plain-picklable.** `pickle.dumps` on the
returned array raises `Can't get local object 'read_roi_mask.<locals>._read_block'`, where the old
`da.from_zarr` graph pickled fine. It is safe because distributed falls back to cloudpickle and
nothing in `src` plain-pickles a graph, but it is a real narrowing of where this array may be sent —
the same constraint `AssumedRoleIcechunkCredentials` exists to document for icechunk's callback.

## Left standing

`ingest.roi_processing.apply_roi_mask` falls back to `read_roi_mask(roi_zarr_path, spatial_chunks)`
with no `storage_options` when its caller passes no `roi_mask`. That is the one read in the ingest
write path that declares no identity, and reaching it would resolve the environment — but the only
production caller always passes `roi_mask`, so it is unreachable today. Noted rather than guarded,
because closing it means adding a parameter no caller needs.
