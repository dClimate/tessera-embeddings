"""NASA Earthdata Login (EDL) authentication for ASF-hosted OPERA data.

Provides two access modes for OPERA RTC-S1 data on ASF:

1. **S3 direct access** (preferred): Fetches temporary STS credentials from
   ASF's cumulus endpoint, enabling in-region S3 reads from us-west-2.
   One HTTP call per task (~0.5s) replaces hundreds of redirect chains.

2. **CloudFront signed URLs** (legacy): Pre-resolves asset URLs through a
   5-hop OAuth redirect chain. Kept for backwards compatibility.

Prerequisites:
  - EDL account with EARTHDATA_USERNAME + EARTHDATA_PASSWORD env vars
  - ASF Cumulus app approved at urs.earthdata.nasa.gov (Authorized Apps)
"""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING

import requests
from dask.distributed import WorkerPlugin, get_client
from odc.loader._rio import ThreadSession as _OdcThreadSession  # type: ignore[attr-defined]
from odc.loader._rio import _local as _odc_thread_session  # type: ignore[attr-defined]
from osgeo import gdal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

if TYPE_CHECKING:
    import pystac

logger = logging.getLogger(__name__)

# The URS host that issues OAuth tokens during the redirect chain.
_AUTH_HOST = "urs.earthdata.nasa.gov"
_EDL_TOKEN_URL = f"https://{_AUTH_HOST}/api/users/token"
_S3_CREDENTIALS_URL = "https://cumulus.asf.alaska.edu/s3credentials"
_OPERA_S3_BUCKET = "asf-cumulus-prod-opera-products"

# Two HTTPS URL patterns for OPERA RTC-S1 assets:
#   1. datapool (flat):      https://datapool.asf.alaska.edu/RTC/OPERA-S1/{filename}
#   2. earthdatacloud (nested): https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/{dir}/{file}
# Both map to: s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/{granule_dir}/{filename}
_DATAPOOL_RE = re.compile(r"https://datapool\.asf\.alaska\.edu/RTC/OPERA-S1/(.+)")
_EARTHDATACLOUD_RE = re.compile(r"https://cumulus\.asf\.earthdatacloud\.nasa\.gov/OPERA/(OPERA_L2_RTC-S1/.+)")
_BAND_SUFFIX_RE = re.compile(r"_(VV|VH|mask)\.tif$")


class _EDLSession(requests.Session):
    """Session that preserves auth headers across NASA's cross-domain redirects.

    Python ``requests`` strips the Authorization header when following a
    redirect to a different domain.  ASF's download chain redirects from
    datapool.asf.alaska.edu → urs.earthdata.nasa.gov → CloudFront, so the
    credentials are lost and URS returns 401.

    This subclass overrides ``rebuild_auth`` to keep the header whenever the
    redirect target is (or came from) ``urs.earthdata.nasa.gov``, following
    NASA's documented pattern.
    """

    def __init__(self) -> None:
        super().__init__()
        retry = Retry(
            total=6,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.mount("https://", adapter)
        self.mount("http://", adapter)

    def rebuild_auth(self, prepared_request: requests.PreparedRequest, response: requests.Response) -> None:
        """Re-inject auth header for redirects targeting the URS host.

        The default ``requests`` behavior strips Authorization on cross-domain
        redirects.  By the time a redirect reaches URS, the header is already
        gone from an earlier hop (e.g. datapool → cumulus stripped it).  So we
        must actively re-apply ``self.auth`` rather than just skipping the
        strip — a plain ``return`` would preserve "already missing".
        """
        redirect_url = prepared_request.url or ""

        if _AUTH_HOST in redirect_url and self.auth:
            # Heading to URS — re-inject credentials that earlier hops stripped
            prepared_request.prepare_auth(self.auth, redirect_url)
            return

        # All other redirects: default behaviour (strip cross-domain auth)
        super().rebuild_auth(prepared_request, response)


def get_edl_session() -> _EDLSession:
    """Create an authenticated requests session for ASF datapool.

    Returns a session that preserves auth headers across NASA's
    cross-domain OAuth redirect chain (ASF → URS → CloudFront).

    Returns:
        Authenticated session with redirect-safe auth handling

    Raises:
        RuntimeError: If credentials are not set in environment
    """
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    if not (username and password):
        raise RuntimeError("EDL authentication requires EARTHDATA_USERNAME + EARTHDATA_PASSWORD")

    session = _EDLSession()
    session.auth = (username, password)
    return session


def resolve_signed_url(session: requests.Session, url: str) -> str:
    """Follow ASF's redirect chain to get a CloudFront signed URL.

    Args:
        session: Authenticated requests session from get_edl_session()
        url: HTTPS asset URL from datapool.asf.alaska.edu

    Returns:
        CloudFront signed URL (no further auth needed for GDAL)

    Raises:
        RuntimeError: If the auth flow fails (e.g., app not approved)
    """
    resp = session.get(url, allow_redirects=True, stream=True)
    resp.close()
    if resp.status_code != 200:
        raise RuntimeError(f"EDL auth failed for {url}: HTTP {resp.status_code}")
    return resp.url


def resolve_item_assets(
    session: requests.Session,
    item: pystac.Item,
    band_keys: list[str],
) -> None:
    """Replace STAC item asset HREFs with signed CloudFront URLs in-place.

    Args:
        session: Authenticated requests session from get_edl_session()
        item: pystac.Item with assets to resolve
        band_keys: Asset keys to resolve (e.g., ["0_VV", "0_VH"])
    """
    for key in band_keys:
        if key in item.assets:
            item.assets[key].href = resolve_signed_url(session, item.assets[key].href)


# =============================================================================
# S3 Direct Access
# =============================================================================


def _get_edl_token(session: _EDLSession) -> str:
    """Get an EDL bearer token, reusing an existing one if available.

    EDL accounts have a maximum token limit. This function first checks
    for existing tokens (GET) before creating a new one (POST).

    Args:
        session: Authenticated _EDLSession (with .auth set)

    Returns:
        EDL bearer token string

    Raises:
        RuntimeError: If authentication fails or no token can be obtained.
    """
    # Try to reuse an existing token
    list_resp = session.get(
        f"https://{_AUTH_HOST}/api/users/tokens",
        timeout=30,
    )
    if list_resp.status_code == 401:
        raise RuntimeError("EDL authentication failed — check username/password")
    list_resp.raise_for_status()

    tokens = list_resp.json()
    if tokens:
        return tokens[0]["access_token"]

    # No existing tokens — create one
    create_resp = session.post(_EDL_TOKEN_URL, timeout=30)
    create_resp.raise_for_status()
    token = create_resp.json().get("access_token")
    if not token:
        raise RuntimeError(f"Unexpected EDL token response: {create_resp.text[:200]}")
    return token


def get_s3_credentials(
    username: str | None = None,
    password: str | None = None,
) -> dict[str, str]:
    """Fetch temporary AWS STS credentials for S3 direct access to OPERA data.

    Two HTTP calls: (1) get an EDL bearer token, (2) exchange it for STS
    credentials at ASF's cumulus endpoint. Credentials are valid for 1 hour
    and grant s3:GetObject + s3:ListBucket on the OPERA bucket in us-west-2.

    Args:
        username: EDL username. Falls back to EARTHDATA_USERNAME env var.
        password: EDL password. Falls back to EARTHDATA_PASSWORD env var.

    Returns:
        Dict with accessKeyId, secretAccessKey, sessionToken, expiration.

    Raises:
        RuntimeError: If credentials cannot be obtained.
    """
    username = username or os.environ.get("EARTHDATA_USERNAME")
    password = password or os.environ.get("EARTHDATA_PASSWORD")
    if not (username and password):
        raise RuntimeError("EDL authentication requires EARTHDATA_USERNAME + EARTHDATA_PASSWORD")

    session = _EDLSession()
    session.auth = (username, password)

    # Step 1: Get EDL bearer token (reuse existing if available)
    token = _get_edl_token(session)

    # Step 2: Exchange bearer token for S3 credentials
    # Clear basic auth so it doesn't clobber the Bearer header
    session.auth = None
    s3_resp = session.get(
        _S3_CREDENTIALS_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    s3_resp.raise_for_status()
    creds = s3_resp.json()

    required_keys = {"accessKeyId", "secretAccessKey", "sessionToken"}
    if not required_keys.issubset(creds):
        raise RuntimeError(f"S3 credentials response missing keys: {required_keys - set(creds)}")

    logger.info("Obtained S3 direct access credentials (1hr TTL)")
    return creds


def _datapool_url_to_s3(url: str) -> str:
    """Convert an ASF OPERA HTTPS URL to an S3 URI.

    CMR-STAC returns two URL patterns depending on satellite vintage.
    Older S1A products use datapool (flat filename); newer S1C products
    use earthdatacloud (nested path matching S3 layout). Both map to the
    same S3 bucket but require different transformations:

    1. **datapool** (flat) — strip band suffix to derive granule dir::

        HTTPS: .../RTC/OPERA-S1/OPERA_L2_RTC-S1_T168-..._v1.0_VH.tif
        S3:    .../OPERA_L2_RTC-S1/OPERA_L2_RTC-S1_T168-..._v1.0/..._VH.tif

    2. **earthdatacloud** (nested) — simple prefix swap::

        HTTPS: .../OPERA/OPERA_L2_RTC-S1/{granule_dir}/{filename}
        S3:    .../OPERA_L2_RTC-S1/{granule_dir}/{filename}

    Args:
        url: HTTPS URL from ASF (datapool or earthdatacloud)

    Returns:
        S3 URI in the asf-cumulus-prod-opera-products bucket

    Raises:
        ValueError: If the URL doesn't match either expected pattern.
    """
    # Pattern 2: earthdatacloud — already has the nested S3 path
    match = _EARTHDATACLOUD_RE.match(url)
    if match:
        return f"s3://{_OPERA_S3_BUCKET}/{match.group(1)}"

    # Pattern 1: datapool — flat filename, needs granule dir
    match = _DATAPOOL_RE.match(url)
    if match:
        filename = match.group(1)
        granule_dir = _BAND_SUFFIX_RE.sub("", filename)
        return f"s3://{_OPERA_S3_BUCKET}/OPERA_L2_RTC-S1/{granule_dir}/{filename}"

    raise ValueError(f"URL does not match ASF datapool pattern: {url}")


def _build_aws_env(creds: dict[str, str]) -> dict[str, str]:
    """Map ASF STS credential keys to the AWS env vars expected by boto3/GDAL."""
    return {
        "AWS_ACCESS_KEY_ID": creds["accessKeyId"],
        "AWS_SECRET_ACCESS_KEY": creds["secretAccessKey"],
        "AWS_SESSION_TOKEN": creds["sessionToken"],
        "AWS_NO_SIGN_REQUEST": "NO",
        "AWS_DEFAULT_REGION": "us-west-2",
    }


def _apply_gdal_config(env: dict[str, str]) -> None:
    """Set GDAL config options so /vsis3/ bypasses its credential cache."""
    for key, val in env.items():
        gdal.SetConfigOption(key, val)


def _patch_odc_thread_session_for_env_drift() -> None:
    """Make odc.loader's per-thread AWSSession cache self-invalidate on env drift.

    odc.loader's ThreadSession caches a boto3 AWSSession in threading.local on
    first use — subsequent os.environ updates are ignored for that thread's
    lifetime. Since Dask task pool threads are long-lived, an initial STS token
    (1hr TTL) gets pinned across refreshes and expires mid-read.

    Resetting _local from the plugin setup() only clears the worker's main
    thread, not the task pool threads. Patching session() itself pushes the
    check into every thread: on each call, if the cached session's access key
    differs from AWS_ACCESS_KEY_ID in env, reset and rebuild from current env.

    Idempotent — safe to call multiple times.
    """
    if getattr(_OdcThreadSession.session, "_env_drift_patched", False):
        return

    original_session = _OdcThreadSession.session

    def session_with_env_check(self: object, session: object = None) -> object:
        cached = getattr(self, "_session", None)
        if cached is not None and session is None:
            current_key = os.environ.get("AWS_ACCESS_KEY_ID")
            try:
                cached_key = cached._session.get_credentials().access_key  # type: ignore[attr-defined]
            except Exception:
                cached_key = None
            if current_key and cached_key and current_key != cached_key:
                self.reset()  # type: ignore[attr-defined]
        return original_session(self, session)  # type: ignore[arg-type]

    session_with_env_check._env_drift_patched = True  # type: ignore[attr-defined]
    _OdcThreadSession.session = session_with_env_check  # type: ignore[method-assign]


# Install at module import time, before any thread has a chance to call
# ThreadSession.session() and cache a session without the drift check.
# Runs on both the orchestrator and on workers (this module is imported
# when the pickled plugin is unpickled).
_patch_odc_thread_session_for_env_drift()


class _S3CredentialPlugin(WorkerPlugin):
    """Dask WorkerPlugin that sets ASF S3 credentials on every worker.

    Unlike ``client.run``, a plugin's ``setup`` is called on workers
    that join *after* registration — critical for adaptive scaling.
    """

    name = "s3-creds"

    def __init__(self, creds: dict[str, str]) -> None:
        self.env = _build_aws_env(creds)

    def setup(self, worker: object) -> None:  # noqa: ARG002
        os.environ.update(self.env)
        # GDAL's /vsis3/ handler caches credentials resolved from env vars
        # in a static member (VSIS3HandleHelper).  VSICurlClearCache only
        # clears curl handles and file-property caches — NOT the credential
        # cache.  SetConfigOption bypasses that cache entirely: GDAL checks
        # config options before env vars and re-reads them on every file open.
        _apply_gdal_config(self.env)
        gdal.VSICurlClearCache()
        # Reset the main-thread AWSSession cache. Task pool threads handle
        # their own reset via the module-level env-drift patch.
        _odc_thread_session.reset()


def set_s3_credentials(creds: dict[str, str]) -> None:
    """Inject ASF STS credentials for GDAL/rasterio S3 reads.

    Sets AWS env vars on the orchestrator and (if a Dask cluster is
    active) registers a ``WorkerPlugin`` that sets env vars on all
    current and future workers.  Rasterio resolves credentials through
    boto3, which reads env vars.

    Credentials are never restored — this is intentional:

    - Avoids a thread-safety race where one thread's cleanup
      removes credentials that another thread still needs.
    - Safe because all non-GDAL S3 callers (icechunk writes,
      store cleanup) explicitly strip the ``env`` provider from
      botocore's credential chain, so they always resolve IAM
      credentials regardless of what env vars are set.

    Args:
        creds: Dict from get_s3_credentials() with accessKeyId,
            secretAccessKey, sessionToken.
    """
    # Register plugin on all current + future Dask workers
    try:
        client = get_client()
        client.register_plugin(_S3CredentialPlugin(creds))
        logger.info(
            "S3 credentials broadcast to workers: key=...%s expiration=%s",
            creds["accessKeyId"][-8:],
            creds.get("expiration", "<unknown>"),
        )
    except ValueError as e:
        raise RuntimeError("No Dask client found — S3 credentials cannot be broadcast to workers") from e

    # Set on orchestrator too — both env vars (for boto3/rasterio session
    # creation) and GDAL config options (bypasses GDAL's credential cache).
    env = _build_aws_env(creds)
    os.environ.update(env)
    _apply_gdal_config(env)


def rewrite_assets_to_s3(item: pystac.Item, band_keys: list[str]) -> None:
    """Rewrite STAC item asset HREFs from HTTPS datapool URLs to S3 URIs in-place.

    Pure string manipulation — no HTTP calls. Used with S3 direct access
    credentials to bypass the CloudFront redirect chain entirely.

    Args:
        item: pystac.Item with assets to rewrite
        band_keys: Asset keys to rewrite (e.g., ["0_VV", "0_VH"])
    """
    for key in band_keys:
        if key in item.assets:
            item.assets[key].href = _datapool_url_to_s3(item.assets[key].href)
