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
    """earthdatacloud URLs get a simple prefix swap."""
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
