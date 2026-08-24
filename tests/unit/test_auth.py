"""Unit tests for ingest/auth.py — EDL session and URL helpers.

These tests exercise only the pure-Python logic (session factory, URL
conversion, credential building).  All HTTP calls and odc/GDAL imports
are either not invoked or patched; no network access is required.
"""

from __future__ import annotations

import logging
import os

import pytest
import requests

from tessera_embeddings.ingest._http import NonJsonResponseError, json_or_raise
from tessera_embeddings.ingest.auth import _EDLSession

# ---------------------------------------------------------------------------
# get_edl_session
# ---------------------------------------------------------------------------


def test_get_edl_session_bearer_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bearer token overrides basic-auth creds when both are set."""
    monkeypatch.setenv("EARTHDATA_TOKEN", "tok123")
    monkeypatch.setenv("EARTHDATA_USERNAME", "user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "pass")

    from tessera_embeddings.ingest.auth import get_edl_session

    session = get_edl_session()
    assert session.headers.get("Authorization") == "Bearer tok123"
    assert session.auth is None


def test_get_edl_session_basic_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    """Username + password produces basic-auth session when no token set."""
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.setenv("EARTHDATA_USERNAME", "user")
    monkeypatch.setenv("EARTHDATA_PASSWORD", "pass")

    from tessera_embeddings.ingest.auth import get_edl_session

    session = get_edl_session()
    assert session.auth == ("user", "pass")
    assert "Authorization" not in session.headers or not session.headers["Authorization"].startswith("Bearer")


def test_get_edl_session_no_creds_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError when neither credential set is available."""
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.delenv("EARTHDATA_USERNAME", raising=False)
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)

    from tessera_embeddings.ingest.auth import get_edl_session

    with pytest.raises(RuntimeError, match="EARTHDATA_USERNAME"):
        get_edl_session()


def test_get_edl_session_partial_basic_auth_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Username without password is not enough for basic auth."""
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    monkeypatch.setenv("EARTHDATA_USERNAME", "user")
    monkeypatch.delenv("EARTHDATA_PASSWORD", raising=False)

    from tessera_embeddings.ingest.auth import get_edl_session

    with pytest.raises(RuntimeError):
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
    from tessera_embeddings.ingest.auth import _datapool_url_to_s3

    url = "https://cumulus.asf.earthdatacloud.nasa.gov/OPERA/OPERA_L2_RTC-S1/gran_dir/file_VV.tif"
    result = _datapool_url_to_s3(url)
    assert result == "s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/gran_dir/file_VV.tif"


def test_datapool_url_to_s3_flat_datapool_vv() -> None:
    """Flat datapool URLs derive the granule directory from the filename."""
    from tessera_embeddings.ingest.auth import _datapool_url_to_s3

    url = "https://datapool.asf.alaska.edu/RTC/OPERA-S1/OPERA_L2_RTC-S1_T168-357760-IW1_20240701T001322Z_20240703T004929Z_S1A_30_v1.0_VV.tif"
    result = _datapool_url_to_s3(url)
    assert result.startswith("s3://asf-cumulus-prod-opera-products/OPERA_L2_RTC-S1/")
    assert "OPERA_L2_RTC-S1_T168-357760-IW1_20240701T001322Z_20240703T004929Z_S1A_30_v1.0_VV.tif" in result
    # Granule dir must not include the band suffix
    parts = result.split("/")
    granule_dir = parts[-2]
    assert not granule_dir.endswith("_VV")


def test_datapool_url_to_s3_flat_datapool_vh() -> None:
    """VH band suffix is also stripped when deriving the granule directory."""
    from tessera_embeddings.ingest.auth import _datapool_url_to_s3

    url = "https://datapool.asf.alaska.edu/RTC/OPERA-S1/OPERA_L2_RTC-S1_T168-xyz_VH.tif"
    result = _datapool_url_to_s3(url)
    parts = result.split("/")
    granule_dir = parts[-2]
    assert "_VH" not in granule_dir


def test_datapool_url_to_s3_unrecognised_raises() -> None:
    """ValueError for a URL that matches neither ASF pattern."""
    from tessera_embeddings.ingest.auth import _datapool_url_to_s3

    with pytest.raises(ValueError, match="does not match"):
        _datapool_url_to_s3("https://example.com/some/other/file.tif")


# ---------------------------------------------------------------------------
# _build_aws_env
# ---------------------------------------------------------------------------


def test_build_aws_env_maps_keys() -> None:
    """STS response keys are mapped to the expected AWS env var names."""
    from tessera_embeddings.ingest.auth import _build_aws_env

    creds = {
        "accessKeyId": "AKID",
        "secretAccessKey": "SECRET",
        "sessionToken": "TOKEN",
    }
    env = _build_aws_env(creds)
    assert env["AWS_ACCESS_KEY_ID"] == "AKID"
    assert env["AWS_SECRET_ACCESS_KEY"] == "SECRET"
    assert env["AWS_SESSION_TOKEN"] == "TOKEN"
    assert env["AWS_NO_SIGN_REQUEST"] == "NO"
    assert env["AWS_DEFAULT_REGION"] == "us-west-2"


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


def test_odc_thread_session_tracks_env_after_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cached AWSSession self-invalidates when AWS_ACCESS_KEY_ID changes.

    This is the load-bearing claim that lets set_s3_credentials skip
    gdal.SetConfigOption / gdal.VSICurlClearCache: as long as
    _local.session().get_frozen_credentials().access_key tracks
    os.environ['AWS_ACCESS_KEY_ID'], rasterio.env.Env will hand GDAL the
    refreshed key on every rio_env() entry.
    """
    pytest.importorskip("odc.loader._rio")
    from odc.loader._rio import _local

    # Importing auth.py installs the env-drift patch at module import.
    from tessera_embeddings.ingest import auth  # noqa: F401

    def run_in_fresh_thread(env_initial: dict[str, str], env_refreshed: dict[str, str]) -> dict[str, str]:
        """Prime _local with one set of creds, mutate env, return what session() yields."""
        import threading

        out: dict[str, str] = {}

        def body() -> None:
            for k, v in env_initial.items():
                os.environ[k] = v
            out["initial"] = _frozen_access_key(_local.session())
            for k, v in env_refreshed.items():
                os.environ[k] = v
            # Patched session() must observe the env mismatch and rebuild.
            out["after_env_update"] = _frozen_access_key(_local.session())

        # Run on a fresh thread so threading.local state is isolated.
        t = threading.Thread(target=body)
        t.start()
        t.join()
        return out

    # Track keys we've stomped on so we can restore them.
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(k, os.environ.get(k, ""))

    initial = {
        "AWS_ACCESS_KEY_ID": "INITIAL_KEY_AAAAAA",
        "AWS_SECRET_ACCESS_KEY": "INITIAL_SECRET",
        "AWS_SESSION_TOKEN": "INITIAL_TOKEN",
    }
    refreshed = {
        "AWS_ACCESS_KEY_ID": "REFRESHED_KEY_BBBBBB",
        "AWS_SECRET_ACCESS_KEY": "REFRESHED_SECRET",
        "AWS_SESSION_TOKEN": "REFRESHED_TOKEN",
    }
    keys = run_in_fresh_thread(initial, refreshed)

    assert keys["initial"] == "INITIAL_KEY_AAAAAA"
    # Without the patch this would be "INITIAL_KEY_AAAAAA" (cached) — the
    # patch detects the env mismatch and rebuilds from current env vars.
    assert keys["after_env_update"] == "REFRESHED_KEY_BBBBBB"


def test_odc_thread_session_explicit_reset_picks_up_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """_local.reset() rebuilds the cached AWSSession from current os.environ.

    Mirrors what _S3CredentialPlugin.setup() does on each worker: after
    updating os.environ it calls _local.reset(), and the next session() call
    must return an AWSSession bound to the new access key.
    """
    pytest.importorskip("odc.loader._rio")
    from odc.loader._rio import _local

    from tessera_embeddings.ingest import auth  # noqa: F401

    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(k, os.environ.get(k, ""))

    import threading

    out: dict[str, str] = {}

    def body() -> None:
        os.environ["AWS_ACCESS_KEY_ID"] = "INITIAL_KEY_AAAAAA"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "INITIAL_SECRET"
        os.environ["AWS_SESSION_TOKEN"] = "INITIAL_TOKEN"
        out["initial"] = _frozen_access_key(_local.session())

        os.environ["AWS_ACCESS_KEY_ID"] = "REFRESHED_KEY_BBBBBB"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "REFRESHED_SECRET"
        os.environ["AWS_SESSION_TOKEN"] = "REFRESHED_TOKEN"
        _local.reset()
        out["after_reset"] = _frozen_access_key(_local.session())

    t = threading.Thread(target=body)
    t.start()
    t.join()

    assert out["initial"] == "INITIAL_KEY_AAAAAA"
    assert out["after_reset"] == "REFRESHED_KEY_BBBBBB"


def test_credential_drift_keeps_the_tasks_gdal_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refresh noticed mid-task leaves the task's GDAL options in force.

    ``rio_env`` is entered twice over: once by ``RioDriver.restore_env`` around a
    whole Dask chunk task, and again inside every read. The drift check fires on
    the inner entry, so whatever it does to rasterio's env stack is done while
    the task's own options are the ones installed.
    """
    pytest.importorskip("odc.loader._rio")
    import threading

    import rasterio.env
    from odc.loader._rio import _local, rio_env

    from tessera_embeddings.ingest import auth  # noqa: F401

    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(k, os.environ.get(k, ""))

    out: dict[str, str | None] = {}

    def body() -> None:
        os.environ["AWS_ACCESS_KEY_ID"] = "INITIAL_KEY_AAAAAA"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "INITIAL_SECRET"
        os.environ["AWS_SESSION_TOKEN"] = "INITIAL_TOKEN"

        with rio_env(GDAL_HTTP_MAX_RETRY="10", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            out["task"] = rasterio.env.getenv().get("GDAL_HTTP_MAX_RETRY")

            os.environ["AWS_ACCESS_KEY_ID"] = "REFRESHED_KEY_BBBBBB"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "REFRESHED_SECRET"
            os.environ["AWS_SESSION_TOKEN"] = "REFRESHED_TOKEN"

            with rio_env(VSI_CACHE=False):  # the read that notices the refresh
                out["refresh_read"] = rasterio.env.getenv().get("GDAL_HTTP_MAX_RETRY")
            with rio_env(VSI_CACHE=False):  # every read after it
                opts = rasterio.env.getenv()
                out["later_read"] = opts.get("GDAL_HTTP_MAX_RETRY")
                out["later_readdir"] = opts.get("GDAL_DISABLE_READDIR_ON_OPEN")
            out["key"] = _frozen_access_key(_local.session())

    t = threading.Thread(target=body)
    t.start()
    t.join()

    assert out["task"] == "10"
    assert out["refresh_read"] == "10"
    assert out["later_read"] == "10"
    assert out["later_readdir"] == "EMPTY_DIR"
    # The refresh still has to land, or the options were kept by not refreshing.
    assert out["key"] == "REFRESHED_KEY_BBBBBB"


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
