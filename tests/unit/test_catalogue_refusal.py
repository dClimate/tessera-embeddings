"""Telling a busy catalogue apart from one that cannot serve a request.

Two refusals arrive as one exception type from one endpoint and need opposite responses:
waiting is the entire remedy for the first and pure waste for the second. So these tests
pin the divergence itself, not two labels — a classifier that returned the right names
while both paths did the same thing would be worthless.

The refusals here are built by driving urllib3's OWN retry ladder to exhaustion and
wrapping it exactly as the client libraries wrap it, rather than by writing the message
out. That matters: a test that asserts a classifier's output against a string the test
also wrote can agree with itself while disagreeing with the library, and a test whose two
sides are disjoint by construction can only ever pass.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pystac_client.exceptions import APIError
from requests.exceptions import RetryError
from urllib3.exceptions import MaxRetryError
from urllib3.util.retry import Retry

from tessera_embeddings.ingest import stac
from tessera_embeddings.ingest.catalogue_refusal import (
    LOAD_REFUSAL_STATUSES,
    REFUSAL_TOKEN,
    UPSTREAM_ERROR_STATUSES,
    CatalogueQueryError,
    CatalogueRefusal,
    CatalogueRequest,
    RefusalKind,
    classify_refusal,
    refusal_in,
    repeat_is_deterministic,
)

_ENDPOINT = "/v1/search"


def _exhaust_the_ladder(status: int) -> None:
    """Drive a real ``urllib3`` Retry to exhaustion on ``status``.

    urllib3 formats the exhaustion cause itself, so nothing about the resulting message
    is authored here — which is what lets these tests notice a wording change upstream
    instead of agreeing with a copy of it.
    """
    retry = Retry(total=1, backoff_factor=0, status_forcelist=(status,), allowed_methods=frozenset(["GET", "POST"]))
    response = SimpleNamespace(status=status, get_redirect_location=lambda: False)
    while True:  # increment() raises MaxRetryError once the budget is spent
        retry = retry.increment(method="POST", url=_ENDPOINT, response=response)


def _refused(status: int) -> APIError:
    """The exception the catalogue client actually raises, chain and all.

    Mirrors the two wrappings between urllib3 and us: ``requests`` converts a
    status-exhausted ``MaxRetryError`` into ``RetryError``, and ``pystac_client`` catches
    that and re-raises ``APIError(str(err))`` — without ``from``, which is why the
    evidence is reachable through ``__context__`` and not ``__cause__``.
    """
    try:
        try:
            try:
                _exhaust_the_ladder(status)
            except MaxRetryError as err:
                raise RetryError(err)  # noqa: B904 — reproducing requests' own wrapping
        except Exception as err:
            raise APIError(str(err))  # noqa: B904 — reproducing pystac_client's own wrapping
    except APIError as err:
        return err
    raise AssertionError("the ladder did not refuse")  # pragma: no cover


def _request(page: int = 3) -> CatalogueRequest:
    return CatalogueRequest("sentinel-2-l2a", "2021-09-01/2021-10-02", "bbox=-3.0000,50.0000,-2.0000,51.0000", page)


class TestClassification:
    """Which refusal is which, and that the two do not collapse into one answer."""

    @pytest.mark.parametrize("status", sorted(LOAD_REFUSAL_STATUSES))
    def test_a_stated_overload_is_a_load_refusal(self, status: int) -> None:
        """The upstream naming itself as the constraint. Waiting is the remedy."""
        refusal = classify_refusal(_refused(status))
        assert refusal.kind is RefusalKind.LOAD
        assert refusal.status == status
        assert refusal.exhausted is True
        assert repeat_is_deterministic(refusal.kind) is False

    @pytest.mark.parametrize("status", sorted(UPSTREAM_ERROR_STATUSES))
    def test_a_backend_failure_is_an_upstream_error(self, status: int) -> None:
        """The upstream failing to produce an answer. A repeat settles it."""
        refusal = classify_refusal(_refused(status))
        assert refusal.kind is RefusalKind.UPSTREAM_ERROR
        assert refusal.status == status
        assert refusal.exhausted is True
        assert repeat_is_deterministic(refusal.kind) is True

    def test_the_two_refusals_take_different_paths(self) -> None:
        """The load-bearing assertion: same exception type, same endpoint, opposite verdicts.

        Asserted as a divergence rather than as two independent expectations, because a
        classifier that mapped everything to one answer would satisfy either expectation
        on its own if the other were ever relaxed.
        """
        load = classify_refusal(_refused(503))
        upstream = classify_refusal(_refused(502))
        assert load.kind is not upstream.kind
        assert repeat_is_deterministic(load.kind) != repeat_is_deterministic(upstream.kind)
        assert load.signature != upstream.signature

    def test_the_exception_chain_answers_without_the_message(self) -> None:
        """Classification reads structure, so a redacted or reworded message still classifies.

        The status is a field on urllib3's own exception two links down the chain; the
        top-level text is a stringification of it. Reading the chain is what keeps this
        from being string matching on a message we do not own.
        """
        refused = _refused(502)
        refused.args = ("redacted",)
        assert classify_refusal(refused).status == 502

    def test_the_message_answers_when_the_chain_is_gone(self) -> None:
        """A refusal re-raised across a boundary that carries no chain still classifies.

        urllib3's exhaustion cause is quoted inside the message it stringified, so losing
        the chain degrades the read to weaker evidence rather than to no evidence.
        """
        detached = APIError(str(_refused(502)))
        assert detached.__context__ is None and detached.__cause__ is None
        assert classify_refusal(detached).kind is RefusalKind.UPSTREAM_ERROR

    def test_a_refusal_with_no_readable_status_is_not_a_finding(self) -> None:
        """UNKNOWN must behave as the default does today: retryable, never terminal.

        A refusal we cannot ground in a status is an absence of evidence, and treating it
        as deterministic would refuse cells on the strength of an unparsed message.
        """
        refusal = classify_refusal(APIError("connection reset by peer"))
        assert refusal.kind is RefusalKind.UNKNOWN
        assert refusal.status is None
        assert repeat_is_deterministic(refusal.kind) is False

    def test_an_unretried_response_status_is_read_but_not_called_exhausted(self) -> None:
        """A single non-200 carries its status on the exception and spent no patience."""
        refusal = classify_refusal(APIError.from_response(SimpleNamespace(status_code=503, text="busy")))
        assert refusal.kind is RefusalKind.LOAD
        assert refusal.exhausted is False

    def test_every_status_the_ladder_retries_is_classified(self) -> None:
        """A status the ladder retries but the taxonomy does not name would fall to UNKNOWN.

        That failure is silent and permanent: the refusal keeps its expansive retry
        forever and no repeat can ever end it. The two sets must therefore cover the
        ladder's own force-list, which is the list that decides what reaches us as an
        exhaustion in the first place.
        """
        forcelist = set(stac._STAC_RETRY.status_forcelist or ())
        assert forcelist, "the STAC ladder must force-list statuses for this invariant to mean anything"
        assert forcelist <= (LOAD_REFUSAL_STATUSES | UPSTREAM_ERROR_STATUSES)
        assert not (LOAD_REFUSAL_STATUSES & UPSTREAM_ERROR_STATUSES), "a status cannot be both refusals"


class TestSignature:
    """What makes two refusals "the same refusal", across attempts that share no objects."""

    def test_the_signature_ignores_everything_that_varies_between_attempts(self) -> None:
        """Two attempts asking the same thing must produce the same token.

        If a counter, a timestamp or a progress figure reached the signature, every
        refusal would be unique and the repeat check would be dead code that always
        reports "not repeated".
        """
        first = CatalogueQueryError(_request(), classify_refusal(_refused(502)), _refused(502))
        second = CatalogueQueryError(_request(), classify_refusal(_refused(502)), APIError("a different wording"))
        assert _token(first) == _token(second)

    @pytest.mark.parametrize(
        "other",
        [
            CatalogueRequest("sentinel-1-grd", "2021-09-01/2021-10-02", "bbox=-3.0000,50.0000,-2.0000,51.0000", 3),
            CatalogueRequest("sentinel-2-l2a", "2021-10-01/2021-11-02", "bbox=-3.0000,50.0000,-2.0000,51.0000", 3),
            CatalogueRequest("sentinel-2-l2a", "2021-09-01/2021-10-02", "bbox=-9.0000,50.0000,-2.0000,51.0000", 3),
            CatalogueRequest("sentinel-2-l2a", "2021-09-01/2021-10-02", "bbox=-3.0000,50.0000,-2.0000,51.0000", 4),
        ],
        ids=["collection", "window", "area", "page"],
    )
    def test_a_different_request_is_a_different_signature(self, other: CatalogueRequest) -> None:
        """Each field that decides the answer must move the token.

        A field left out would let one month's refusal be read as a repeat of another's,
        and abandon a cell on evidence about a different request.
        """
        assert other.signature != _request().signature

    def test_a_different_refusal_of_one_request_is_a_different_signature(self) -> None:
        """The token carries the refusal too: a 502 then a 503 is not a repeat."""
        one = CatalogueQueryError(_request(), classify_refusal(_refused(502)), _refused(502))
        two = CatalogueQueryError(_request(), classify_refusal(_refused(503)), _refused(503))
        assert _token(one) != _token(two)

    def test_the_whole_signature_survives_the_match(self) -> None:
        """Emitted and recovered must be the same string, character for character.

        The recovery matches a run of non-whitespace, so a space anywhere inside the
        signature TRUNCATES the token at that space — and a truncated token is still
        whitespace-free, still parses, and still compares equal to another truncated copy
        of itself. Asserting "no whitespace in the recovered token" is therefore a
        tautology that no defect can violate; only round-tripping catches it.
        """
        request, refusal = _request(), classify_refusal(_refused(502))
        error = CatalogueQueryError(request, refusal, _refused(502))
        recovered = refusal_in(str(error))
        assert recovered is not None
        assert recovered.token == f"{refusal.signature}|{request.signature}"

    def test_the_token_survives_a_boundary_that_carries_only_text(self) -> None:
        """The querying leg and the layer holding the attempt budget are separate runs.

        Only the failure text crosses, wrapped in whatever the orchestrator prefixes, so
        recovery is by token NAME and must tolerate text on both sides of it.
        """
        error = CatalogueQueryError(_request(), classify_refusal(_refused(502)), _refused(502))
        detail = f"ingest_s2_roi_reflectance: Flow run encountered an exception: {error} (state=Failed)"
        recovered = refusal_in(detail)
        assert recovered is not None
        assert recovered.kind is RefusalKind.UPSTREAM_ERROR
        assert recovered.token == _token(error)

    def test_text_carrying_no_token_is_not_classified_as_a_refusal(self) -> None:
        """A failure that is not a catalogue refusal must not be read as one."""
        assert refusal_in("PermissionError: The provided token has expired.") is None
        assert refusal_in("") is None

    def test_an_unreadable_kind_in_the_token_is_refused(self) -> None:
        """Text that merely contains the token name is not evidence of a classification."""
        assert refusal_in(f"{REFUSAL_TOKEN}=nonsense:502|whatever") is None


def _token(error: CatalogueQueryError) -> str:
    recovered = refusal_in(str(error))
    assert recovered is not None, f"no token in {error}"
    return recovered.token


def _pages_then_refusal(pages: list[dict], refusal: BaseException):
    """An iterator that yields ``pages`` and then refuses, as a live cursor does.

    A generator FUNCTION whose body only raises is not a generator at all — it raises at
    the call site, before the code under test is entered, and the test then passes or
    fails for a reason that has nothing to do with the code. Built here so no test can
    make that mistake silently.
    """

    def cursor():
        yield from pages
        raise refusal

    return cursor()


def _item(item_id: str) -> dict:
    return {
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": item_id,
        "geometry": None,
        "bbox": None,
        "properties": {"datetime": "2021-09-05T10:00:00Z"},
        "links": [],
        "assets": {},
        "collection": "sentinel-2-l2a",
    }


class TestTheFailingRequestIsNamed:
    """A refusal must say WHICH request it refused, in the log and on the exception.

    The client library discards the request when it wraps a transport failure: the
    surviving message names the host and the endpoint path, and a catalogue search is a
    request BODY. Without this a refusal cannot be narrowed to a window or a page,
    reproduced, or handed to whoever runs the archive.
    """

    @staticmethod
    def _query(pages, caplog, **kwargs):
        provider = stac._get_provider_config("earth-search")
        collection = stac._get_collection_config("earth-search", "sentinel-2-l2a")
        with (
            patch.object(stac, "StacApiIO"),
            patch.object(stac, "Client") as client,
            caplog.at_level(logging.ERROR, logger=stac.__name__),
        ):
            client.open.return_value.search.return_value.pages_as_dicts.return_value = pages
            return stac._query_stac_items(
                provider,
                collection,
                None,
                "2021-09-01",
                "2021-10-02",
                bbox=(-3.0, 50.0, -2.0, 51.0),
                **kwargs,
            )

    def test_the_page_that_failed_is_named_not_just_the_query(self, caplog) -> None:
        """Which PAGE the refusal died on, not merely that the query failed.

        "The catalogue cannot answer at all" and "one deep cursor breaks" are different
        defects with different reproductions, and an item count cannot tell them apart.
        """
        pages = _pages_then_refusal([{"features": [_item("a"), _item("b")]}], _refused(502))

        with pytest.raises(CatalogueQueryError) as caught:
            self._query(pages, caplog)

        error = caught.value
        assert error.request.page == 2, "the ordinal must count pages, not report a constant"
        assert error.request.collection == "sentinel-2-l2a"
        assert error.request.window == "2021-09-01/2021-10-02"
        assert error.request.area == "bbox=-3.0000,50.0000,-2.0000,51.0000"
        assert error.refusal.kind is RefusalKind.UPSTREAM_ERROR

        logged = caplog.text
        assert "sentinel-2-l2a" in logged
        assert "2021-09-01/2021-10-02" in logged
        assert "bbox=-3.0000,50.0000,-2.0000,51.0000" in logged
        assert "page 2" in logged
        # The volatile context belongs in the log and nowhere else, because a signature
        # that moved with progress could never repeat.
        assert "2 item(s)" in logged

    def test_the_transport_cause_is_preserved_under_the_wrapper(self, caplog) -> None:
        """The wrapper adds the request; it must not become a second wrapper that
        discards a cause.
        """
        with pytest.raises(CatalogueQueryError) as caught:
            self._query(_pages_then_refusal([], _refused(502)), caplog)
        assert isinstance(caught.value.__cause__, APIError)
        assert "too many 502 error responses" in str(caught.value.__cause__)

    def test_a_refusal_before_any_search_is_named_as_the_catalogue_root(self, caplog) -> None:
        """Opening the catalogue is its own request.

        Reading its refusal as a refusal of the search would attribute a root outage to a
        window that was never asked for.
        """
        provider = stac._get_provider_config("earth-search")
        collection = stac._get_collection_config("earth-search", "sentinel-2-l2a")
        with (
            patch.object(stac, "StacApiIO"),
            patch.object(stac, "Client") as client,
            caplog.at_level(logging.ERROR, logger=stac.__name__),
        ):
            client.open.side_effect = _refused(502)
            with pytest.raises(CatalogueQueryError) as caught:
                stac._query_stac_items(
                    provider, collection, None, "2021-09-01", "2021-10-02", bbox=(-3.0, 50.0, -2.0, 51.0)
                )
        assert caught.value.request.page == 0
        assert caught.value.request.area == "catalogue-root"

    def test_a_failure_in_the_page_body_is_not_a_catalogue_refusal(self, caplog) -> None:
        """Our own validation must stay outside the wrapper.

        The retry policy acts on a catalogue refusal, so classifying a defect of ours as
        one would hand it a policy meant for someone else's outage — and the classifier
        would report a refusal that never happened.
        """

        def pages():
            yield {"features": [{"no": "id"}]}

        with pytest.raises(ValueError, match="without an 'id'"):
            self._query(pages(), caplog)
        assert REFUSAL_TOKEN not in caplog.text

    def test_a_search_that_returns_an_iterable_of_pages_is_accepted(self, caplog) -> None:
        """The page walk asks for an ITERATOR; it must not assume it was handed one.

        Advancing by hand is what lets a failure name its page, and it needs an iterator —
        so an iterable that is not one is either a ``TypeError`` (a list of pages) or, for a
        stub that answers every call rather than stopping, a loop that never ends and
        consumes memory until something dies. Both are silent in production and violent in
        a test suite, which is where such a stub actually appears.
        """
        items = self._query([{"features": [_item("a")]}], caplog)  # a LIST, not a generator
        assert [i.id for i in items] == ["a"]

        empty = self._query(MagicMock(), caplog)  # answers every call, never stops
        assert empty == []

    def test_a_query_that_paginates_cleanly_logs_no_refusal(self, caplog) -> None:
        """The success path must be untouched: same items, no refusal line."""

        def pages():
            yield {"features": [_item("a")]}
            yield {"features": [_item("b"), _item("a")]}  # the antimeridian dedupe still applies

        items = self._query(pages(), caplog)
        assert [i.id for i in items] == ["a", "b"]
        assert REFUSAL_TOKEN not in caplog.text


class TestRefusalLabels:
    """The human-readable halves, which are what an operator actually reads."""

    def test_the_label_says_whether_patience_was_already_spent(self) -> None:
        exhausted = CatalogueRefusal(RefusalKind.UPSTREAM_ERROR, 502, exhausted=True)
        single = CatalogueRefusal(RefusalKind.UPSTREAM_ERROR, 502, exhausted=False)
        assert "exhausting" in exhausted.label
        assert "without being retried" in single.label

    def test_the_root_request_label_says_it_precedes_the_search(self) -> None:
        assert "before any search" in CatalogueRequest("c", "w", "catalogue-root", 0).label
