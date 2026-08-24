"""Unit tests for the non-JSON provider response guard in ``ingest._http``.

A provider that answers 2xx with a body that is not JSON defeats every status-based
defence, so these tests pin the two things that make the next occurrence diagnosable:
the error names what arrived, and it survives the worker boundary intact.
"""

import pickle

import pytest
import requests

from tessera_embeddings.ingest._http import (
    _EXCERPT_CHARS,
    NonJsonResponseError,
    json_or_raise,
)

_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"


def _response(body: bytes, *, status: int = 200, content_type: str = "application/json") -> requests.Response:
    """A REAL response, so the parse fails the way the provider makes it fail."""
    resp = requests.Response()
    resp.status_code = status
    resp._content = body
    resp.url = _URL
    resp.headers["Content-Type"] = content_type
    return resp


class TestJsonOrRaise:
    """A non-JSON body must raise something that identifies the endpoint that sent it."""

    def test_valid_json_is_returned_unchanged(self):
        assert json_or_raise(_response(b'{"feed": {"entry": []}}'), quote_body=True) == {"feed": {"entry": []}}

    def test_empty_body_raises_naming_url_status_type_and_length(self):
        with pytest.raises(NonJsonResponseError) as exc_info:
            json_or_raise(_response(b"", content_type="text/html"), quote_body=True)
        exc = exc_info.value
        assert exc.url == _URL
        assert exc.status == 200
        assert exc.content_type == "text/html"
        assert exc.body_length == 0
        # The whole point: the message identifies the endpoint, not just the parse.
        assert _URL in str(exc)
        assert "200" in str(exc)
        assert "text/html" in str(exc)

    def test_html_error_page_is_captured_on_the_attribute(self):
        with pytest.raises(NonJsonResponseError) as exc_info:
            json_or_raise(_response(b"<html><title>502 Bad Gateway</title>", content_type="text/html"), quote_body=True)
        assert "502 Bad Gateway" in (exc_info.value.excerpt or "")

    def test_the_captured_body_never_reaches_the_message(self):
        """The message is substring-matched to decide whether a leg may be retried.

        ``ingest_zone_year._is_retryable_leg_failure`` tests the failure text against
        ``_NON_RETRYABLE_LEG_MARKERS``. A provider-chosen body carrying one of those words
        would turn a retryable leg permanently dead, costing a zone-year of coverage — so
        the body must stay off the message however useful it is on the attribute.
        """
        body = b"<html>ValidationError: ObjectNotFound CorruptedStoreError</html>"
        with pytest.raises(NonJsonResponseError) as exc_info:
            json_or_raise(_response(body, content_type="text/html"), quote_body=True)
        message = str(exc_info.value)
        for marker in ("ValidationError", "ObjectNotFound", "CorruptedStoreError"):
            assert marker not in message
            assert marker in (exc_info.value.excerpt or "")

    def test_credential_body_is_never_quoted_when_not_asked(self):
        """A truncated credential document is exactly the body that fails to parse."""
        secret = b'{"accessKeyId": "ASIAEXAMPLE", "secretAccessKey": "SUPERSECRETVALUE'
        with pytest.raises(NonJsonResponseError) as exc_info:
            json_or_raise(_response(secret), quote_body=False)
        exc = exc_info.value
        assert exc.excerpt is None
        assert "SUPERSECRETVALUE" not in str(exc)
        assert "ASIAEXAMPLE" not in str(exc)
        # Suppressing the body must not cost the fields that identify the failure.
        assert exc.body_length == len(secret)
        assert exc.url == _URL

    def test_excerpt_is_capped(self):
        with pytest.raises(NonJsonResponseError) as exc_info:
            json_or_raise(_response(b"x" * 5000, content_type="text/plain"), quote_body=True)
        exc = exc_info.value
        assert exc.excerpt is not None
        assert len(exc.excerpt) == _EXCERPT_CHARS
        # A truncated 5 kB body is still reported at its true size.
        assert exc.body_length == 5000

    def test_original_parse_error_is_kept_as_the_cause(self):
        with pytest.raises(NonJsonResponseError) as exc_info:
            json_or_raise(_response(b"not json"), quote_body=True)
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestSurvivesTheWorkerBoundary:
    """Dask serialises a worker's failure before sending it; the fields must survive."""

    def test_pickle_round_trip_keeps_every_field(self):
        with pytest.raises(NonJsonResponseError) as exc_info:
            json_or_raise(_response(b"<html>boom", content_type="text/html"), quote_body=True)
        original = exc_info.value

        restored = pickle.loads(pickle.dumps(original))

        assert isinstance(restored, NonJsonResponseError)
        assert restored.url == original.url
        assert restored.status == original.status
        assert restored.content_type == original.content_type
        assert restored.body_length == original.body_length
        assert restored.excerpt == original.excerpt
        assert str(restored) == str(original)

    def test_pickle_round_trip_keeps_a_suppressed_excerpt_suppressed(self):
        with pytest.raises(NonJsonResponseError) as exc_info:
            json_or_raise(_response(b'{"secretAccessKey": "SUPERSECRETVALUE'), quote_body=False)
        restored = pickle.loads(pickle.dumps(exc_info.value))
        assert restored.excerpt is None
        assert "SUPERSECRETVALUE" not in str(restored)
