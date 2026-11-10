"""Unit tests for ingest/auth.py — EDL session and URL helpers.

These tests exercise the session factory, redirect auth, URL conversion and the odc
credential-drift patch. No HTTP call is made; no network access is required.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable

import pytest
import rasterio.env
import requests
from odc.loader._rio import _local, rio_env

from tessera_embeddings.ingest._http import NonJsonResponseError, json_or_raise
from tessera_embeddings.ingest.auth import _datapool_url_to_s3, _EDLSession, get_edl_session

# ---------------------------------------------------------------------------
# get_edl_session
# ---------------------------------------------------------------------------


def test_get_edl_session_bearer_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bearer token overrides basic-auth creds when both are set."""
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok123")
    monkeypatch.setenv("EARTHDATA_USERNAME", "user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "pass")

    session = get_edl_session()
    assert session.headers.get("Authorization") == "Bearer tok123"
    assert session.auth is None


def test_get_edl_session_basic_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Username + password produces basic-auth session when no token set."""
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.setenv("EARTHDATA_USERNAME", "user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "pass")

    session = get_edl_session()
    assert session.auth == ("user", "pass")
    assert "Authorization" not in session.headers or not session.headers["Authorization"].startswith("Bearer")


@pytest.mark.parametrize("username", [None, "user"], ids=["nothing", "username-without-password"])
def test_get_edl_session_no_creds_raises(monkeypatch: pytest.MonkeyPatch, username: str | None) -> None:
    """RuntimeError when neither credential set is complete."""
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)
    if username is None:
        monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    else:
        monkeypatch.setenv("EARTHDATA_USERNAME", username)

    with pytest.raises(RuntimeError, match="EARTHDATA_USERNAME"):
        get_edl_session()


# ---------------------------------------------------------------------------
# _EDLSession.rebuild_auth — redirect handling for both auth modes
#
# Regression tests for the bearer-token redirect bug: requests strips the
# Authorization header (whether session-level or per-request) on every
# cross-domain hop, so rebuild_auth must actively re-inject creds for both
# basic-auth (self.auth) and bearer (self.headers['Authorization']) modes
# whenever the redirect target is in the trusted EDL/ASF host set.
# ---------------------------------------------------------------------------


def _prepared(url: str) -> requests.PreparedRequest:
    return requests.Request("GET", url).prepare()


def _fake_response(original_url: str = "https://datapool.asf.alaska.edu/RTC/OPERA-S1/x.tif") -> requests.Response:
    """Build a Response carrying an original request — requests' default
    rebuild_auth dereferences response.request.url, so this must be set.
    """
    resp = requests.Response()
    resp.request = requests.Request("GET", original_url).prepare()
    return resp


def test_rebuild_auth_basic_auth_reinjected_on_urs_redirect() -> None:
    """Basic-auth creds are re-applied when redirected to URS."""
    session = _EDLSession()
    session.auth = ("user", "pass")
    req = _prepared("https://urs.earthdata.nasa.gov/oauth/authorize")

    session.rebuild_auth(req, _fake_response())

    assert req.headers.get("Authorization", "").startswith("Basic ")


def test_rebuild_auth_bearer_reinjected_on_cross_domain_redirect() -> None:
    """Session-level bearer header survives datapool → cumulus redirect.

    This is the bug the test user hit: the previous implementation only
    re-applied self.auth, so SAML accounts using EARTHDATA_TOKEN got 401
    when the bearer header was stripped on the datapool → cumulus hop.
    """
    session = _EDLSession()
    session.headers["Authorization"] = "Bearer tok123"
    req = _prepared("https://cumulus.asf.alaska.edu/some/path")
    # Simulate requests' default strip before rebuild_auth runs.
    req.headers.pop("Authorization", None)

    session.rebuild_auth(req, _fake_response())

    assert req.headers.get("Authorization") == "Bearer tok123"


def test_rebuild_auth_bearer_reinjected_on_earthdatacloud_redirect() -> None:
    """Bearer is also restored on redirects to earthdatacloud.nasa.gov."""
    session = _EDLSession()
    session.headers["Authorization"] = "Bearer tok123"
    req = _prepared("https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/x.tif")
    req.headers.pop("Authorization", None)

    session.rebuild_auth(req, _fake_response())

    assert req.headers.get("Authorization") == "Bearer tok123"


def test_rebuild_auth_basic_auth_dropped_on_non_urs_asf_redirect() -> None:
    """Basic-auth credentials must NOT be sent to non-URS ASF hosts.

    self.auth carries the user's permanent EDL password — only URS, the
    OAuth handshake target, should ever receive it. A datapool → cumulus
    redirect must drop the password (URS itself will set a cookie / issue
    a token before redirecting onward).
    """
    session = _EDLSession()
    session.auth = ("user", "pass")
    req = _prepared("https://cumulus.asf.alaska.edu/some/path")
    req.headers.pop("Authorization", None)

    session.rebuild_auth(req, _fake_response())

    assert "Authorization" not in req.headers


def test_rebuild_auth_drops_auth_on_untrusted_redirect() -> None:
    """Untrusted redirect targets must not receive EDL credentials."""
    session = _EDLSession()
    session.headers["Authorization"] = "Bearer tok123"
    req = _prepared("https://evil.example.com/steal")
    req.headers.pop("Authorization", None)

    session.rebuild_auth(req, _fake_response())

    assert "Authorization" not in req.headers


# ---------------------------------------------------------------------------
# _datapool_url_to_s3
# ---------------------------------------------------------------------------


def test_datapool_url_to_s3_earthdatacloud() -> None:
    """Earthdatacloud URLs get a simple prefix swap."""
    url = "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/gran_dir/file_VV.tif"
    result = _datapool_url_to_s3(url)
    assert result == "s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/gran_dir/file_VV.tif"


@pytest.mark.parametrize("band", ["VV", "VH"])
def test_datapool_url_to_s3_flat_datapool(band: str) -> None:
    """Flat datapool URLs derive the granule directory from the filename, band suffix stripped."""
    name = f"OPERA_L2_RTC-S1_T168-357760-IW1_20240701T001322Z_20240703T004929Z_S1A_30_v1.0_{band}.tif"
    result = _datapool_url_to_s3(f"https://datapool.asf.alaska.edu/RTC/OPERA-S1/{name}")
    assert result.startswith("s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/")
    assert result.endswith(f"/{name}")
    assert not result.split("/")[-2].endswith(f"_{band}"), "the granule dir must not carry the band suffix"


def test_datapool_url_to_s3_unrecognised_raises() -> None:
    """ValueError for a URL that matches neither ASF pattern."""
    with pytest.raises(ValueError, match="does not match"):
        _datapool_url_to_s3("https://example.com/some/other/file.tif")


# ---------------------------------------------------------------------------
# odc ThreadSession env-drift patch
#
# Regression test for the simplification that drops gdal.SetConfigOption and
# gdal.VSICurlClearCache from set_s3_credentials. The simplification is safe
# only if the patched ThreadSession.session() returns an AWSSession whose
# frozen access key tracks os.environ, because rio_env() wraps
# rasterio.env.Env around that session on every /vsis3/ open and passes its
# frozen credentials into GDAL — so GDAL signs with whatever key the cached
# session holds at that moment.
# ---------------------------------------------------------------------------


def _frozen_access_key(session: object) -> str:
    """Return the access-key id baked into a boto3 AWSSession's frozen creds."""
    boto = session._session  # type: ignore[attr-defined]
    return boto.get_credentials().get_frozen_credentials().access_key


def _set_aws_env(tag: str) -> None:
    os.environ["AWS_ACCESS_KEY_ID"] = f"{tag}_KEY"
    os.environ["AWS_SECRET_ACCESS_KEY"] = f"{tag}_SECRET"
    os.environ["AWS_SESSION_TOKEN"] = f"{tag}_TOKEN"


@pytest.fixture
def in_fresh_thread(monkeypatch: pytest.MonkeyPatch) -> Callable[[Callable[[], None]], None]:
    """Run a body on a new thread (isolated ``threading.local``), restoring the AWS env after.

    Importing ``ingest.auth`` (done at the top of this module) installs the env-drift patch.
    """
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(k, os.environ.get(k, ""))

    def run(body: Callable[[], None]) -> None:
        t = threading.Thread(target=body)
        t.start()
        t.join()

    return run


def test_odc_thread_session_tracks_env_after_drift(in_fresh_thread) -> None:
    """Cached AWSSession self-invalidates when AWS_ACCESS_KEY_ID changes.

    This is the load-bearing claim that lets set_s3_credentials skip
    gdal.SetConfigOption / gdal.VSICurlClearCache: ``rio_env()`` wraps rasterio.env.Env around
    the cached session on every /vsis3/ open, so GDAL signs with whatever key it holds.
    """
    out: dict[str, str] = {}

    def body() -> None:
        _set_aws_env("INITIAL")
        out["initial"] = _frozen_access_key(_local.session())
        _set_aws_env("REFRESHED")
        out["after_env_update"] = _frozen_access_key(_local.session())

    in_fresh_thread(body)
    assert out["initial"] == "INITIAL_KEY"
    # Without the patch this would still be the cached INITIAL key.
    assert out["after_env_update"] == "REFRESHED_KEY"


def test_credential_drift_keeps_the_tasks_gdal_options(in_fresh_thread) -> None:
    """A refresh noticed mid-task leaves the task's GDAL options in force.

    ``rio_env`` is entered twice over: once by ``RioDriver.restore_env`` around a
    whole Dask chunk task, and again inside every read. The drift check fires on
    the inner entry, so whatever it does to rasterio's env stack is done while
    the task's own options are the ones installed.
    """
    out: dict[str, str | None] = {}

    def body() -> None:
        _set_aws_env("INITIAL")
        with rio_env(GDAL_HTTP_MAX_RETRY="10", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            out["task"] = rasterio.env.getenv().get("GDAL_HTTP_MAX_RETRY")
            _set_aws_env("REFRESHED")
            with rio_env(VSI_CACHE=False):  # the read that notices the refresh
                out["refresh_read"] = rasterio.env.getenv().get("GDAL_HTTP_MAX_RETRY")
            with rio_env(VSI_CACHE=False):  # every read after it
                opts = rasterio.env.getenv()
                out["later_read"] = opts.get("GDAL_HTTP_MAX_RETRY")
                out["later_readdir"] = opts.get("GDAL_DISABLE_READDIR_ON_OPEN")
            out["key"] = _frozen_access_key(_local.session())

    in_fresh_thread(body)
    assert out["task"] == out["refresh_read"] == out["later_read"] == "10"
    assert out["later_readdir"] == "EMPTY_DIR"
    # The refresh still has to land, or the options were kept by not refreshing.
    assert out["key"] == "REFRESHED_KEY"


def test_a_credential_body_is_never_logged_or_raised(caplog: pytest.LogCaptureFixture) -> None:
    """A credential document truncated mid-write is exactly the body that fails to parse.

    Its leading bytes are the credential, so the three EDL calls pass ``log_body=False``.
    """
    resp = requests.Response()
    resp.status_code = 200
    resp._content = b'{"accessKeyId": "ASIAEXAMPLE", "secretAccessKey": "SUPERSECRET'
    resp.url = "https://cumulus.asf.alaska.edu/s3credentials"

    with caplog.at_level(logging.DEBUG), pytest.raises(NonJsonResponseError) as exc_info:
        json_or_raise(resp, log_body=False)

    assert "SUPERSECRET" not in str(exc_info.value)
    assert "SUPERSECRET" not in caplog.text
    # Suppressing the body must not cost the fields that identify the failure.
    assert "cumulus.asf.alaska.edu" in str(exc_info.value)
