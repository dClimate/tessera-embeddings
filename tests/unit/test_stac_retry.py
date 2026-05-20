"""Unit tests for the STAC retry/timeout configuration.

Verifies that _query_stac_items wires urllib3 Retry into the pystac-client
transport layer (replacing the old tenacity decorator).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from urllib3.util.retry import Retry

from tessera_embeddings.ingest.stac import _STAC_RETRY, _STAC_TIMEOUT, _query_stac_items


def test_stac_retry_constants():
    """The retry policy matches the yield-modeling reference values."""
    assert isinstance(_STAC_RETRY, Retry)
    assert _STAC_RETRY.total == 8
    assert _STAC_RETRY.backoff_factor == 2
    assert 429 in _STAC_RETRY.status_forcelist
    assert 503 in _STAC_RETRY.status_forcelist
    assert "GET" in _STAC_RETRY.allowed_methods
    assert "POST" in _STAC_RETRY.allowed_methods
    assert _STAC_RETRY.respect_retry_after_header is True


def test_stac_timeout_is_connect_read_tuple():
    """Timeout is a (connect, read) pair, not a single float."""
    assert _STAC_TIMEOUT == (10, 60)


@patch("tessera_embeddings.ingest.stac.Client")
@patch("tessera_embeddings.ingest.stac.StacApiIO")
@patch("tessera_embeddings.ingest.stac._build_stac_query", return_value={"collections": ["s2"]})
def test_query_stac_items_wires_retry_and_timeout(mock_build, mock_stac_io_cls, mock_client_cls):
    """_query_stac_items passes _STAC_RETRY and _STAC_TIMEOUT to StacApiIO."""
    mock_stac_io = MagicMock()
    mock_stac_io_cls.return_value = mock_stac_io

    mock_client = MagicMock()
    mock_client_cls.open.return_value = mock_client
    mock_client.search.return_value.items.return_value = iter([])

    provider = MagicMock()
    provider.catalog_url = "https://example.com/stac"
    collection_config = MagicMock()

    _query_stac_items(provider, collection_config, "T15TYH", "2024-01-01", "2024-01-31")

    mock_stac_io_cls.assert_called_once_with(max_retries=_STAC_RETRY, timeout=_STAC_TIMEOUT)
    mock_client_cls.open.assert_called_once_with(provider.catalog_url, stac_io=mock_stac_io)


@patch("tessera_embeddings.ingest.stac.Client")
@patch("tessera_embeddings.ingest.stac.StacApiIO")
@patch("tessera_embeddings.ingest.stac._build_stac_query", return_value={"collections": ["s2"]})
def test_query_stac_items_no_tenacity_decorator(mock_build, mock_stac_io_cls, mock_client_cls):
    """_query_stac_items is no longer wrapped in a tenacity retry decorator."""
    import tenacity

    assert not hasattr(_query_stac_items, "retry")
    assert not isinstance(getattr(_query_stac_items, "__wrapped__", None), type(None)) or True
    # The definitive check: tenacity-decorated functions have a .retry attribute
    assert not hasattr(_query_stac_items, "retry")
