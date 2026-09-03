# 022 — Resolve the ROI mask's S3 credential inside the read, not when the graph is built

**Status:** Accepted (2026-08-22). Built.

## Context

A Sentinel-1 leg on dev wrote 88 dates over six hours and then died inside the single `dask`
compute that the overlapped window write runs:

```
storage/zarr_store.py:1607  _write_windows_overlapped
s3fs/core.py:460            _call_s3
PermissionError: The provided token has expired.
```

The only fsspec read in that graph is the ROI mask. `ingest.roi.read_roi_mask` called
`da.from_zarr(..., storage_options=resolve_storage_options(storage_options))`, which resolves the
credential provider **once** and hands the resulting `FsspecStore` — credential strings and all —
into the task graph. On the radar path that graph is built once per 30-day batch, in
`_prepare_batch`, then computed date by date for as long as the batch's writes take. Botocore hands
out a frozen role credential with as little as ten to fifteen minutes of life left, so a batch
whose writes run longer presents an expired one and the read fails on a bucket our role can always
read. The rival diagnosis — that the read declared no identity and picked up the source's
OPERA-scoped token — does not fit 88 successful dates on our own bucket.

## Reproduction

It reproduces without AWS, in seconds, in `tests/unit/ingest/test_roi_mask_credential_expiry.py`: a moto
S3 server behind a front door that records which key signed each request and can refuse a nominated
one with S3's own `ExpiredToken`, and a role identity issuing a fresh two-second credential per
resolution. Build the mask graph, mark every key valid at build time as expired, then write. Before
the change the write presented the build-time key and raised `PermissionError: The provided token
has expired.` — the reported error, from the reported frame. After it the write re-resolves, signs
with a later key, and succeeds. The source-shaped key in the environment is never presented either
way.

## Decision

**`read_roi_mask` opens the store inside each block read.** The array is assembled with
`da.map_blocks` over a closure that calls `resolve_storage_options` and `zarr.open_array` itself, so
the credential is no older than the read that presents it. Nothing else changes: same provider,
same call sites, same returned dask array.

Two alternatives were rejected. An env-stripped `aiobotocore` session handed to s3fs would refresh
indefinitely, but s3fs replaces a caller-supplied session on some paths and fell back to the
environment in testing. Forcing botocore to refresh below some margin of remaining life is four
lines, but a long enough batch defeats any margin, where resolving at read time cannot be.

## Cost

**MEASURED 2026-08-22, correcting the figure this section first carried.** The original claim of
"one or two metadata GETs" per block was wrong: it is **four**, because zarr 3 probes both layouts
on every open (`zarr.json` twice, plus the v2 `.zarray` and `.zattrs`). Counted at s3fs' own
`_call_s3` for a 12-block mask, reading the whole mask went from 24 requests to 72 — six per block
against two, a flat **3x** — with the 24 chunk-byte reads unchanged, so all the growth is metadata.
It stays small in absolute terms: metadata GETs are a few hundred bytes, blocks sharing a credential
share one cached fsspec client, and every consumer slices the mask to live windows, so dask culls
the opens with the blocks.

The graph is also **no longer plain-picklable**: `pickle.dumps` raises `Can't get local object
'read_roi_mask.<locals>._read_block'` where `da.from_zarr` pickled fine. Safe today, since
distributed falls back to cloudpickle and nothing in `src` plain-pickles a graph, but it narrows
where this array may be sent. Recorded here rather than asserted in a test, because a test
demanding the graph *cannot* be plain-pickled would fail the day someone made it picklable.

Left standing: `ingest.roi_processing.apply_roi_mask` reads the mask with no `storage_options`, which
resolves the environment, if a caller passes no `roi_mask`. Every production caller passes one, and
`tests/unit/ingest/test_roi_mask_construction.py` keeps it that way.
