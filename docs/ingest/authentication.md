# Ingest authentication: Earthdata Login and the OPERA credentials

OPERA RTC-S1 radar hosted by ASF requires NASA Earthdata Login (EDL) credentials, because ASF
uses NASA's OAuth2/URS system for access control. Unlike Sentinel-2, OPERA data is not publicly
readable from S3. The credentials the pipeline obtains last about an hour, are scoped to ASF's
bucket alone, and arrive through the same environment variables every other AWS client reads,
and most of this document is about the consequences of those three facts.

It is for anyone setting up a radar ingest, or diagnosing an `AccessDenied`, `ExpiredToken` or
instance-metadata (IMDS) error during one. It covers setup, the S3 direct access path the pipeline
uses and the credential-lifetime rules around it, the identity used for the project's own bucket,
how asset URLs are rewritten to S3, and the legacy signed-URL fallback. The rest of the radar
path is in [radar.md](radar.md).

## Setup

```bash
export EARTHDATA_USERNAME=your-username
export EARTHDATA_PASSWORD=your-password
```

You must also approve the **ASF Cumulus** application at
[urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) → Authorized Apps.

## S3 direct access (preferred)

S3 direct access is preferred because it removes a 5-hop OAuth redirect chain per OPERA COG
tile: one round trip buys an hour of STS credentials and GDAL then reads `s3://asf-cumulus-prod-opera-products`
without per-file HTTPS redirects, which at batch scale is the dominant latency reduction.

`auth.get_s3_credentials` exchanges EDL credentials for temporary AWS STS credentials:

1. `GET https://urs.earthdata.nasa.gov/api/users/tokens` — reuse an existing EDL bearer
   token (EDL accounts have a maximum token limit; creating a new one unnecessarily can hit
   that limit).
2. If no token exists, `POST .../api/users/token` to create one.
3. `GET https://cumulus.asf.alaska.edu/s3credentials` with `Authorization: Bearer <token>` —
   returns `accessKeyId`, `secretAccessKey`, `sessionToken` (valid 1 hour) for the
   `asf-cumulus-prod-opera-products` bucket in `us-west-2`.

`set_s3_credentials` then injects these onto both the orchestrator process and all current and
future Dask workers via a `WorkerPlugin`. It sets `AWS_*` environment variables (consumed by
boto3 when `odc.loader` builds an `AWSSession`) and resets the cached per-thread session so the
next `/vsis3/` open picks up the new credentials.

### Renewal runs on a timer, not on the work loop

`s1_roi.credential_ticker` re-checks the
credential's remaining life every `CRED_TICK_INTERVAL_SEC` while batches are being consumed; the
loop's per-batch and per-date checks remain as a fallback, on `ingest_s1_roi_sar`'s
`cred_refresh_interval_sec`. Loop-driven renewal can only fire
*between* units of work, so a unit outliving its margin cannot renew from inside itself — and that
coupling is self-reinforcing, since failing reads stop the progress that would trigger renewal.
What a worker receives is also a **snapshot**: the plugin freezes the credential at construction,
so a worker joining N minutes after the last broadcast starts with only the remaining TTL and past
it with none. Under adaptive scaling workers join throughout a leg, so the broadcast cadence is a
correctness condition and the ticker is what bounds N. Every broadcast logs the advertised expiry
(`S3 credentials broadcast to workers`).

### The per-thread session cache

`odc.loader` caches a boto3 `AWSSession` per thread in
`threading.local` on first use and ignores subsequent env var updates for that thread's lifetime.
Dask task pool threads are long-lived, so the initial 1 h STS token was being pinned across
refreshes and expiring mid-read. `auth.py` patches `odc.loader._rio.ThreadSession` at module
import time so each thread self-detects `AWS_ACCESS_KEY_ID` drift and rebuilds its cached session
from current env vars; `rasterio.env.Env`, entered by `odc.loader.rio_env()` on every `/vsis3/`
open, then hands GDAL the refreshed session's frozen credentials, so no `gdal.SetConfigOption` or
`VSICurlClearCache` is needed. This reaches into private `odc.loader` internals
(`_OdcThreadSession`, `_local`): if odc renames those symbols the import fails loudly, and
`tests/unit/ingest/test_auth.py` catches it in CI.

### The project's own bucket is read and written with the IAM role

OPERA asset STS credentials are intentionally **never cleaned up** from env vars, which avoids
one Dask task's cleanup removing credentials another still needs. The consequence: once
`set_s3_credentials` runs, the `AWS_*` env vars hold OPERA-scoped tokens granting access **only**
to `asf-cumulus-prod-opera-products`, so any S3 access to the project's own bucket resolving from
those env vars — every icechunk `Repository.open`/`create` in the S1 write path — fails with
`AccessDenied`.

Icechunk/Zarr operations on the project's own bucket therefore must resolve **IAM-role**
credentials, bypassing the env vars. The mechanism:

- `providers/aws/credentials.py::iam_icechunk_credentials` resolves credentials from the
  botocore chain with the `env` provider **removed**, so it always lands on the deployment's
  IAM role (instance-metadata / ECS task role / local SSO) regardless of what STS tokens the
  env vars hold. It returns `icechunk.S3StaticCredentials`.
- `storage.zarr_store` exposes a `credentials_provider(provider)` **context manager**.
  `_create_storage` uses the registered provider as the `get_credentials` callback for any S3
  open lacking an explicit one, for the duration of the block — scoped rather than permanent, so a
  reused Dask worker is not left pinned to it, and the previous provider is restored even if the
  body raises. The storage layer ships this as `None` and never imports botocore, per the
  `no-botocore-outside-aws-provider` architecture rule; only the AWS provider supplies it.
- The `process_roi_sar` Prefect task registers `iam_icechunk_credentials` through that hook when
  `use_s3_direct=True`. **In the task shell, not the flow body** — under the Dask task runner the
  domain function and its store writes execute in a *worker* process, so a provider registered in
  the flow-runner process would never reach them.

### The ROI mask resolves its credential at read time

The plain-Zarr side needs the same identity and one property beyond it. An ROI mask is read
through fsspec rather than icechunk, and
`providers/aws/credentials.py::iam_s3_storage_options` is the fsspec counterpart — the same
env-stripped chain in the shape fsspec takes as `storage_options`. The ingest is handed the
**callable**, not its result, and `read_roi_mask` resolves it inside each block read rather than
once at graph build.

That matters because the mask array is LAZY: its block reads happen inside a later
`write_day_windows` compute, which on the radar path spans a whole 30-day batch. One credential
resolved at graph-build time would be presented by every one of those reads and, once expired,
fail with `ExpiredToken` on a bucket the role can always read — a lifetime problem wearing a
permissions problem's error message. Two consequences: each block read pays its own store open
and metadata round trip where a credential resolved once at graph build paid one for the whole
array, and the returned array is cloudpickle-only because the closure is nested. Both are
measured in [ADR 022](../../context_docs/decisions/022-resolve-the-roi-mask-credential-at-read-time.md),
and both argue against handing this array to a plain-pickle boundary or reading a zone grid you
do not need.

### IMDS throttling: why `_resolve_iam_credentials` is cached

The credential machinery has two distinct TTLs, and conflating them overwhelms the EC2 Instance Metadata
Service (IMDS):

- `iam_icechunk_credentials` sets `expires_after=15min` on the returned `S3StaticCredentials`,
  which is how often **icechunk** re-invokes the callback per repo client — not how often we
  should touch IMDS.
- `_resolve_iam_credentials` is `@lru_cache(maxsize=1)`, so the botocore session and the live
  `RefreshableCredentials` it returns are built **once per process**. Those refresh themselves in
  the background, and `get_frozen_credentials()` is a pure expiry check that only re-hits IMDS
  inside botocore's refresh window (advisory ~15 min before the ~6 h token expiry), lock-guarded
  against stampedes.

Without the cache every callback builds a fresh session and does a **cold IMDS resolve**, which
under many concurrent workers bursts IMDS past its per-instance rate limit and surfaces as
`failed to load IMDS session token / invalid token` or `no providers in chain provided
credentials` — transient, but enough to fail a run of chunks. Caching decouples how often
icechunk asks from how often we hit IMDS: the former stays at 15 min, the latter drops to roughly
once per token lifetime. `lru_cache` does not cache exceptions, so a failed cold resolve retries.
The same provider is injected into inference workers, where long-lived Ray actors made this the
dominant failure mode.

## URL rewriting

CMR-STAC returns HTTPS asset URLs in two formats depending on satellite vintage:

| Format | Example |
|---|---|
| **datapool** (older S1A) | `https://datapool.asf.alaska.edu/RTC/OPERA-S1/<filename>` |
| **earthdatacloud** (newer S1C) | `https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/<dir>/<file>` |

`auth.rewrite_assets_to_s3` converts both to `s3://asf-cumulus-prod-opera-products/...` by pure
string manipulation, no HTTP calls. For the datapool format the granule directory name is
reconstructed by stripping the band suffix (`_VV.tif`, `_VH.tif`, `_mask.tif`) from the flat
filename.

## Legacy CloudFront signed URLs (fallback)

`_EDLSession` is a `requests.Session` subclass that preserves the `Authorization` header
across cross-domain redirects. Python `requests` strips this header when following a redirect
to a different domain. The ASF download chain goes: `datapool.asf.alaska.edu` →
`urs.earthdata.nasa.gov` (OAuth exchange) → CloudFront CDN. Because the header is stripped
at the first hop, it is missing by the time URS sees the request. `_EDLSession.rebuild_auth`
re-injects credentials whenever the redirect target URL contains `urs.earthdata.nasa.gov`.

`resolve_item_assets` follows the full redirect chain per asset and mutates the STAC item's
asset HREFs to CloudFront signed URLs before `odc.stac.load` reads them. This path is kept
for out-of-region access where S3 direct is not available, but is significantly slower.
