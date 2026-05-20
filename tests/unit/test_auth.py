"""Unit tests for ingest/auth.py — EDL session and URL helpers.

These tests exercise only the pure-Python logic (session factory, URL
conversion, credential building).  All HTTP calls and odc/GDAL imports
are either not invoked or patched; no network access is required.
"""

from __future__ import annotations

import os

import pytest

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
