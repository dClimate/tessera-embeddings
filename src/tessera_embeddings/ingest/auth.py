"""Credentials for imagery that needs them: NASA Earthdata Login for OPERA, and Planetary Computer
read tokens for Sentinel-2 on Azure.

Planetary Computer refuses an unsigned read. Its token is free, anonymous and short-lived (about 45
minutes), so it is appended to each URL at the moment of reading (:func:`sign_azure_href`), never
when the item is queried: an ingest outlives many tokens.

NASA Earthdata Login (EDL) authentication for ASF-hosted OPERA data has two access modes for OPERA RTC-S1 data on ASF:

1. **S3 direct access** (preferred): temporary STS credentials from ASF's cumulus endpoint enable
   in-region S3 reads from us-west-2. One HTTP call per task (~0.5 s) replaces hundreds of
   redirect chains.
2. **CloudFront signed URLs** (legacy): pre-resolves asset URLs through a 5-hop OAuth redirect
   chain. Kept for backwards compatibility.

Prerequisites: an EDL account exported as EARTHDATA_USERNAME + EARTHDATA_PASSWORD, and the ASF
Cumulus app approved under Authorized Apps at urs.earthdata.nasa.gov.
"""

from __future__ import annotations

import datetime
import logging
import os
import random
import re
import threading
from typing import TYPE_CHECKING
from urllib.parse import urlparse, urlsplit

import requests
from dask.distributed import WorkerPlugin, get_client  # noqa: TID251 — registers a worker plugin cluster-wide
from odc.loader._rio import ThreadSession as _OdcThreadSession
from odc.loader._rio import _local as _odc_thread_session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tessera_embeddings.ingest._http import NonJsonResponseError, json_or_raise

if TYPE_CHECKING:
    import pystac

logger = logging.getLogger(__name__)

# The URS host that issues OAuth tokens during the redirect chain.
_AUTH_HOST = "urs.earthdata.nasa.gov"
_EDL_TOKEN_URL = f"https://{_AUTH_HOST}/api/users/token"
_S3_CREDENTIALS_URL = "https://cumulus.asf.alaska.edu/s3credentials"
_OPERA_S3_BUCKET = "asf-cumulus-prod-opera-products"

# Hosts trusted to receive EDL credentials across redirects. The datapool → URS → cumulus → CloudFront chain stays
# within these three; any redirect to a host outside the set must drop the Authorization header.
_TRUSTED_AUTH_SUFFIXES = (
    "urs.earthdata.nasa.gov",
    "asf.alaska.edu",
    "earthdatacloud.nasa.gov",
)

# Two HTTPS URL patterns for OPERA RTC-S1 assets:
#   1. datapool (flat):      https://datapool.asf.alaska.edu/RTC/OPERA-S1/{filename}
#   2. earthdatacloud (nested): https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/{dir}/{file}
# Both map to: s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/{granule_dir}/{filename}
_DATAPOOL_RE = re.compile(r"https://datapool\.asf\.alaska\.edu/RTC/OPERA-S1/(.+)")
_EARTHDATACLOUD_RE = re.compile(r"https://cumulus\.asf\.earthdatacloud\.nasa\.gov/OPERA/(OPERA_L2_RTC-S1/.+)")
_BAND_SUFFIX_RE = re.compile(r"_(VV|VH|mask)\.tif$")


_AZURE_BLOB_HOST = ".blob.core.windows.net"
_SAS_TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{account}/{container}"
#: A token is replaced once less than this is left, so a read started on it finishes inside it.
_SAS_REFRESH_MARGIN = datetime.timedelta(minutes=10)
_sas_tokens: dict[tuple[str, str], tuple[str, datetime.datetime]] = {}
#: Held by the one thread fetching a token, so a process asks the service once at a time.
_sas_lock = threading.Lock()
_NO_TOKEN = ("", datetime.datetime.min.replace(tzinfo=datetime.UTC))
#: After a failed refresh, when this process may try again: a jittered pause, so a throttled
#: fleet does not retry on every read, in step.
_sas_next_try: dict[tuple[str, str], datetime.datetime] = {}


def sign_azure_href(href: str) -> str:
    """``href`` with a current Planetary Computer read token appended, if it is on Azure Blob Storage.

    Anything else is returned unchanged. A token already on the href is replaced, so signing again
    never reads on a stale one. One token signs a whole storage container and is cached per
    process. Two cases, so that waiting on the token service never stalls reads that need not wait:

    - **A valid token nearing expiry** is refreshed by one thread with a single attempt, while
      every other read carries on with the cached token; a failed attempt is retried by a later
      read. Nothing waits for it.
    - **No valid token** (a starting worker, or one whose refreshes all failed): reads wait on
      one patient, jittered fetch (:func:`_sas_session`), and its error fails the read.
    """
    parts = urlsplit(href)
    if not parts.netloc.endswith(_AZURE_BLOB_HOST):
        return href
    key = (parts.netloc.removesuffix(_AZURE_BLOB_HOST), parts.path.split("/")[1])
    token, expiry = _sas_tokens.get(key, _NO_TOKEN)
    now = datetime.datetime.now(datetime.UTC)
    due = now >= _sas_next_try.get(key, _NO_TOKEN[1])
    if now < expiry < now + _SAS_REFRESH_MARGIN and due and _sas_lock.acquire(blocking=False):
        try:
            token, expiry = _fetch_sas_token(key, patient=False)
        except _SAS_ERRORS:
            _sas_next_try[key] = now + datetime.timedelta(seconds=random.uniform(30, 60))
            logger.warning("Planetary Computer token refresh failed; reading on the current one", exc_info=True)
        finally:
            _sas_lock.release()
    if expiry <= datetime.datetime.now(datetime.UTC):
        with _sas_lock:
            token, expiry = _sas_tokens.get(key, _NO_TOKEN)  # another thread may have fetched it
            if expiry <= datetime.datetime.now(datetime.UTC):
                token, expiry = _fetch_sas_token(key, patient=True)
    return f"{parts._replace(query='').geturl()}?{token}"


_SAS_ERRORS = (requests.RequestException, NonJsonResponseError, ValueError, KeyError)


def _fetch_sas_token(key: tuple[str, str], *, patient: bool) -> tuple[str, datetime.datetime]:
    """Fetch and cache a token for ``(account, container)``."""
    response = _sas_session(patient).get(_SAS_TOKEN_URL.format(account=key[0], container=key[1]), timeout=30)
    response.raise_for_status()
    body = json_or_raise(response, log_body=False)
    _sas_tokens[key] = (body["token"], datetime.datetime.fromisoformat(body["msft:expiry"]))
    return _sas_tokens[key]


def _sas_session(patient: bool) -> requests.Session:
    """A token-service session: one attempt, or a patient one that waits out throttling.

    The service refuses with 429 when a starting fleet's workers all ask at once from one NAT
    address, so the patient waits are jittered, and ``Retry-After`` is ignored because obeying
    it would put the refused workers back in step. Ten attempts capped at two minutes each ride
    out a burst of several fleets.
    """
    session = requests.Session()
    retry = Retry(
        total=10 if patient else 0,
        backoff_factor=2,
        backoff_max=120,
        backoff_jitter=30,
        respect_retry_after_header=False,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


class _EDLSession(requests.Session):
    """Session that preserves auth headers across NASA's cross-domain redirects.

    ``requests`` strips the Authorization header on a redirect to a different domain, and ASF's
    download chain redirects datapool.asf.alaska.edu -> urs.earthdata.nasa.gov -> CloudFront, so
    the credentials are lost and URS returns 401. ``rebuild_auth`` is overridden to keep the
    header whenever the redirect target is (or came from) ``urs.earthdata.nasa.gov``, following
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
        """Re-inject auth on cross-domain redirects within the EDL/ASF chain.

        ``requests`` strips Authorization on every cross-domain redirect, and by the time a
        redirect reaches URS the header is already gone from an earlier hop (datapool -> cumulus
        stripped it). Credentials must be actively RE-APPLIED, not merely spared the strip — a
        plain ``return`` would preserve "already missing".

        The two auth modes are scoped differently because their blast radii differ:

        * **Basic auth** via ``self.auth`` is the user's permanent EDL password, re-injected only
          on redirects to URS (``urs.earthdata.nasa.gov``), the OAuth handshake target. Other ASF
          / Earthdata Cloud hosts do not need it — URS issues a token, sets cookies and redirects
          onward — and sending the password to a non-URS host would leak the user's credentials.

        * **Bearer token** via ``self.headers['Authorization']`` (the local SAML fallback) is
          scoped and time-limited. Session headers merge into the prepared request BEFORE
          ``rebuild_auth`` runs, so requests strips it on every cross-domain hop; it has to be
          restored across the whole trusted chain (datapool -> URS -> cumulus -> CloudFront) for
          the bearer path to work at all.
        """
        redirect_host = (urlparse(prepared_request.url or "").hostname or "").lower()

        if self.auth:
            # Basic-auth path: only URS may receive the user's password.
            if redirect_host == _AUTH_HOST or redirect_host.endswith("." + _AUTH_HOST):
                prepared_request.prepare_auth(self.auth, prepared_request.url or "")
                return
        else:
            session_auth = self.headers.get("Authorization")
            if session_auth and any(
                redirect_host == h or redirect_host.endswith("." + h) for h in _TRUSTED_AUTH_SUFFIXES
            ):
                prepared_request.headers["Authorization"] = session_auth
                return

        # Untrusted target (or no creds to re-inject): default strip behaviour.
        super().rebuild_auth(prepared_request, response)


def get_edl_session() -> _EDLSession:
    """Create an authenticated requests session for ASF datapool.

    The session preserves auth headers across NASA's cross-domain OAuth redirect chain
    (ASF -> URS -> CloudFront). Two modes:

    * **Basic auth** via ``EARTHDATA_USERNAME`` + ``EARTHDATA_PASSWORD`` is the production path,
      and the contract deployed pipelines rely on.
    * **Bearer token** via ``EARTHDATA_TOKEN`` is a LOCAL DEVELOPMENT FALLBACK ONLY. NASA EDL
      accounts that authenticate through SAML / Launchpad cannot complete a basic-auth handshake
      against datapool — the redirect chain returns 401 even when the user has approved the ASF
      Data Access app. A user-generated token (urs.earthdata.nasa.gov/profile) survives the chain
      when set as a SESSION-LEVEL header. Production MUST NOT set it: tokens have a finite TTL
      and are unsuitable for unattended workloads. Bearer wins when both are set, since a
      developer who exported a token did so to override basic auth their account cannot use.

    Returns:
        Authenticated session with redirect-safe auth handling

    Raises:
        RuntimeError: If neither credential set is available.
    """
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    token = os.environ.get("EARTHDATA_TOKEN")

    if token:
        # LOCAL-ONLY FALLBACK: bearer at SESSION level so _EDLSession's rebuild_auth can re-inject it on each hop of
        # the chain (datapool → URS → cumulus → CloudFront).
        session = _EDLSession()
        session.headers["Authorization"] = f"Bearer {token}"
        return session

    if username and password:
        session = _EDLSession()
        session.auth = (username, password)
        return session

    raise RuntimeError(
        "EDL authentication requires EARTHDATA_USERNAME + EARTHDATA_PASSWORD "
        "(production) or EARTHDATA_TOKEN (local-only fallback for SAML accounts)."
    )


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

    EDL accounts have a maximum token limit, so existing tokens are listed (GET) before a new one
    is created (POST).

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

    # log_body=False on all three EDL calls: the body that fails to parse is a truncated credential document, and its
    # leading bytes are the credential.
    tokens = json_or_raise(list_resp, log_body=False)
    if tokens:
        return tokens[0]["access_token"]

    # No existing tokens — create one
    create_resp = session.post(_EDL_TOKEN_URL, timeout=30)
    create_resp.raise_for_status()
    token = json_or_raise(create_resp, log_body=False).get("access_token")
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

    # Step 2: Exchange bearer token for S3 credentials Clear basic auth so it doesn't clobber the Bearer header
    session.auth = None
    s3_resp = session.get(
        _S3_CREDENTIALS_URL,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    s3_resp.raise_for_status()
    creds = json_or_raise(s3_resp, log_body=False)

    required_keys = {"accessKeyId", "secretAccessKey", "sessionToken"}
    if not required_keys.issubset(creds):
        raise RuntimeError(f"S3 credentials response missing keys: {required_keys - set(creds)}")

    logger.info("Obtained S3 direct access credentials (1hr TTL)")
    return creds


def _datapool_url_to_s3(url: str) -> str:
    """Convert an ASF OPERA HTTPS URL to an S3 URI.

    CMR-STAC returns two URL patterns depending on satellite vintage: older S1A products use
    datapool (flat filename), newer S1C products use earthdatacloud (nested path matching the S3
    layout). Both map to the same bucket but need different transformations:

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


def _patch_odc_thread_session_for_env_drift() -> None:
    """Make odc.loader's per-thread AWSSession cache self-invalidate on env drift.

    odc.loader's ThreadSession caches a boto3 AWSSession in threading.local on first use, and
    ignores later os.environ updates for that thread's lifetime. Dask task pool threads are
    long-lived, so an initial STS token (1 hr TTL) gets pinned across refreshes and expires
    mid-read. Resetting _local from the plugin setup() clears only the worker's main thread;
    patching session() pushes the check into every thread — on each call, if the cached session's
    access key differs from AWS_ACCESS_KEY_ID in env, drop it and rebuild from current env.

    Dropping it must NOT go through ``ThreadSession.reset()``: that also calls
    ``rasterio.env.delenv()``, and ``session()`` is called from inside ``rio_env()``, nested in
    the environment odc wraps around a whole Dask chunk task. Tearing that down loses the task's
    GDAL options — the HTTP retry ladder among them — for every read after the refresh.

    Idempotent.
    """
    if getattr(_OdcThreadSession.session, "_env_drift_patched", False):
        return

    original_session = _OdcThreadSession.session

    def session_with_env_check(self: object, session: object = None) -> object:
        cached = getattr(self, "_session", None)
        if cached is not None and session is None:
            current_key = os.environ.get("AWS_ACCESS_KEY_ID")
            try:
                cached_key = cached._session.get_credentials().access_key
            except Exception:
                cached_key = None
            if current_key and cached_key and current_key != cached_key:
                # Drop the session only; leave rasterio's env stack alone.
                self._session = None  # ty: ignore[unresolved-attribute]
                self._aws = None  # ty: ignore[unresolved-attribute]
        return original_session(self, session)  # ty: ignore[invalid-argument-type]

    session_with_env_check._env_drift_patched = True  # ty: ignore[unresolved-attribute]
    _OdcThreadSession.session = session_with_env_check  # ty: ignore[invalid-assignment]


# Installed at module import, before any thread can call ThreadSession.session() and cache a session without the drift
# check. Runs on the orchestrator and on workers alike, since this module is imported when the pickled plugin is
# unpickled.
_patch_odc_thread_session_for_env_drift()


class _S3CredentialPlugin(WorkerPlugin):
    """Dask WorkerPlugin that sets ASF S3 credentials on every worker.

    Unlike ``client.run``, a plugin's ``setup`` runs on workers that join *after* registration —
    critical under adaptive scaling.

    What it distributes is a SNAPSHOT: the credential is frozen at construction and stored on the
    scheduler, so a worker joining N minutes after the last broadcast starts with the remaining
    (TTL - N), and past the TTL with none. The broadcast CADENCE, not just the credential's own
    margin, is therefore a correctness condition — the caller must re-broadcast on a timer for
    late joiners to be usable.
    """

    name = "s3-creds"

    def __init__(self, creds: dict[str, str]) -> None:
        self.env = _build_aws_env(creds)

    def setup(self, worker: object) -> None:  # noqa: ARG002
        os.environ.update(self.env)
        # Reset the main-thread AWSSession cache; task pool threads refresh themselves via the module-level env-drift
        # patch on their next session() call. Together they ensure rasterio.env.Env (used by odc.loader on every
        # /vsis3/ open) signs with the new key, which makes gdal.SetConfigOption and gdal.VSICurlClearCache redundant
        # here.
        _odc_thread_session.reset()


def set_s3_credentials(creds: dict[str, str]) -> None:
    """Inject ASF STS credentials for GDAL/rasterio S3 reads.

    Sets AWS env vars on the orchestrator and, if a Dask cluster is active, registers a
    ``WorkerPlugin`` that sets them on all current and future workers. Rasterio resolves
    credentials through boto3, which reads env vars.

    Credentials are never restored, deliberately: cleanup on one thread would race with another
    thread still using them, and every non-GDAL S3 caller (icechunk writes, store cleanup) strips
    the ``env`` provider from botocore's credential chain, so those always resolve IAM
    credentials whatever the env vars say.

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

    # On the orchestrator too. odc.loader's rio_env() wraps rasterio.env.Env around _local.session() on every /vsis3/
    # open, and rasterio.env.Env passes the AWSSession's frozen credentials into GDAL on entry, so updating os.environ
    # suffices to sign with the new key.
    env = _build_aws_env(creds)
    os.environ.update(env)
    _odc_thread_session.reset()


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
