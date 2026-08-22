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

import dataclasses
import logging
import pickle
import re
import threading
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


def _unretried(status: int) -> APIError:
    """The exception the client raises for a status the ladder does NOT force-list.

    ``StacApiIO.request`` calls ``APIError.from_response`` on any non-200 it was handed, so
    the status arrives as a structured attribute with no ladder in the chain. This is the
    shape a 502 has in production, since 502 is excluded from the force-list precisely so
    the window re-cut is reached on the first refusal.
    """
    return APIError.from_response(SimpleNamespace(status_code=status, text=f"HTTP {status}"))


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

    def test_only_the_provider_that_refuses_oversized_pages_skips_the_502_retry(self) -> None:
        """502 must reach the remedy on the FIRST refusal — but only where a remedy exists.

        Force-listing it costs the ladder's whole backoff, measured at 364 s per refused
        request, before a shorter window or a smaller page can be tried, and buys nothing:
        the refusal is a function of the request, so every attempt re-asks a question whose
        answer is already known.

        Scoped, though. A provider with no such remedy wants its retries, and nothing was
        measured about anyone else's 502 — so this pins BOTH directions, since a change in
        either would be silent.
        """
        reduced = set(stac._STAC_RETRY_NO_502.status_forcelist or ())
        full = set(stac._STAC_RETRY.status_forcelist or ())
        assert 502 not in reduced
        assert 502 in full, "a provider with no re-cut to fall back on keeps its 502 retries"
        assert reduced >= LOAD_REFUSAL_STATUSES, "a stated overload must still be waited out"
        assert full >= LOAD_REFUSAL_STATUSES

    def test_the_ladder_is_chosen_by_the_providers_own_flag(self) -> None:
        """Which ladder a request sits behind must follow the provider, not the call site."""
        earth_search = stac._get_provider_config("earth-search")
        assert earth_search.refuses_oversized_pages is True
        assert stac._retry_for(earth_search) is stac._STAC_RETRY_NO_502

        cmr = stac._get_provider_config("cmr-asf")
        assert cmr.refuses_oversized_pages is False
        assert stac._retry_for(cmr) is stac._STAC_RETRY


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

    def test_the_token_survives_a_boundary_that_carries_pickled_objects(self) -> None:
        """The querying leg runs on a Dask worker, which returns the exception itself.

        An exception pickle cannot rebuild is replaced in transit by the TypeError raised
        while rebuilding it, so the token never reaches the attempt-budget layer and the
        repeat check sees no refusals at all.
        """
        error = CatalogueQueryError(_request(), classify_refusal(_refused(502)), _refused(502))
        restored = pickle.loads(pickle.dumps(error))
        assert str(restored) == str(error)
        assert _token(restored) == _token(error)

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
    def _query(pages, caplog, start="2021-09-01", end="2021-10-02", **kwargs):
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
                start,
                end,
                bbox=(-3.0, 50.0, -2.0, 51.0),
                **kwargs,
            )

    def test_the_page_that_failed_is_named_not_just_the_query(self, caplog, monkeypatch) -> None:
        """Which PAGE the refusal died on, not merely that the query failed.

        "The catalogue cannot answer at all" and "one deep cursor breaks" are different
        defects with different reproductions, and an item count cannot tell them apart.

        A one-day window at the page-size FLOOR, which is the state where no lever remains:
        the window cannot be shortened further and the page cannot be halved further. A
        one-day window alone is no longer enough — it is answered by asking for a smaller
        page (see ``TestARefusalNoShorterWindowCanReachAsksForASmallerPage``) — so the floor
        is raised here to reach exhaustion in a single search rather than by driving the
        ladder down through four of them.
        """
        monkeypatch.setattr(stac, "_MIN_PAGE_SIZE", 10**9)
        pages = _pages_then_refusal([{"features": [_item("a"), _item("b")]}], _refused(502))

        with pytest.raises(CatalogueQueryError) as caught:
            self._query(pages, caplog, start="2021-09-01", end="2021-09-01")

        error = caught.value
        assert error.request.page == 2, "the ordinal must count pages, not report a constant"
        assert error.request.collection == "sentinel-2-l2a"
        assert error.request.window == "2021-09-01/2021-09-01"
        assert error.request.area == "bbox=-3.0000,50.0000,-2.0000,51.0000"
        assert error.refusal.kind is RefusalKind.UPSTREAM_ERROR

        logged = caplog.text
        assert "sentinel-2-l2a" in logged
        assert "2021-09-01/2021-09-01" in logged
        assert "bbox=-3.0000,50.0000,-2.0000,51.0000" in logged
        assert "page 2" in logged
        # The volatile context belongs in the log and nowhere else, because a signature
        # that moved with progress could never repeat.
        assert "2 item(s)" in logged

    def test_the_transport_cause_is_preserved_under_the_wrapper(self, caplog, monkeypatch) -> None:
        """The wrapper adds the request; it must not become a second wrapper that
        discards a cause.

        At the page-size floor, so the refusal propagates rather than being answered by a
        smaller page.
        """
        monkeypatch.setattr(stac, "_MIN_PAGE_SIZE", 10**9)
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


class TestADeepRefusalIsRePartitioned:
    """A 502 at a deep cursor becomes more, smaller searches — not a failed leg.

    The counterpart to ``TestTheFailingRequestIsNamed``: that pins what still propagates,
    this pins what no longer does. Waiting cannot clear this refusal, so the only remedy is
    to ask for the window in shorter pieces.

    Both arrival shapes are exercised. 502 is kept out of the ladder's force-list, so what
    production sees is a single unretried response — but a 502 can still reach us wrapped in
    an exhausted ladder (a redirect chain, a force-list someone widens later), and the
    re-partition must not depend on which one it is.
    """

    @staticmethod
    def _query(pages, caplog, start, end):
        provider = stac._get_provider_config("earth-search")
        collection = stac._get_collection_config("earth-search", "sentinel-2-l2a")
        with (
            patch.object(stac, "StacApiIO"),
            patch.object(stac, "Client") as client,
            caplog.at_level(logging.WARNING, logger=stac.__name__),
        ):
            search = client.open.return_value.search
            search.return_value.pages_as_dicts.return_value = pages
            stac._query_stac_items(provider, collection, None, start, end, bbox=(-3.0, 50.0, -2.0, 51.0))
            return search.call_count

    @pytest.mark.parametrize(
        "refusal",
        [
            pytest.param(_unretried(502), id="unretried-response"),
            pytest.param(_refused(502), id="exhausted-ladder"),
        ],
    )
    def test_a_month_refused_at_depth_is_re_queried_in_pieces(self, refusal, caplog) -> None:
        pages = _pages_then_refusal([{"features": [_item("a")]}], refusal)

        searches = self._query(pages, caplog, "2021-09-01", "2021-10-02")

        assert searches > 1, "the refusal must have produced further, shorter searches"
        assert "shorter window" in caplog.text

    def test_a_load_refusal_is_left_to_the_retry_ladder(self, caplog) -> None:
        """A 503 says the upstream is BUSY. Slicing it multiplies the load that caused it."""
        pages = _pages_then_refusal([{"features": [_item("a")]}], _refused(503))

        with pytest.raises(CatalogueQueryError) as caught:
            self._query(pages, caplog, "2021-09-01", "2021-10-02")

        assert caught.value.refusal.kind is RefusalKind.LOAD


class TestARefusalNoShorterWindowCanReachAsksForASmallerPage:
    """The two refusals the window re-cut cannot answer, and the lever that can.

    Earth Search refuses a request whose RESPONSE would exceed about 6 MB, so asking for
    fewer items is the direct remedy. The re-cut reaches it indirectly, by regrouping items
    into smaller responses -- but it cannot reach a FIRST page, which a shorter window asks
    identically, nor a single day, which is the re-cut's own floor. Both of those failed the
    leg outright before this, so every test here covers a path that previously raised.

    Five of these eight FAIL when the step-down is removed and are therefore proofs. The
    remaining three pass either way and say so in their own docstrings: they guard the
    preference order, the termination, and the refusal class the step-down is gated on.
    """

    @staticmethod
    def _query(page_sets, caplog, start, end):
        """Drive one query, giving each successive search its own page iterator.

        Returns the ``limit`` each search was asked with, in call order, and the items. A
        per-search iterator is the point: one shared iterator is exhausted by the first
        search, so a retry would appear to succeed by returning nothing at all.
        """
        provider = stac._get_provider_config("earth-search")
        collection = stac._get_collection_config("earth-search", "sentinel-2-l2a")
        limits: list[int] = []
        guard = threading.Lock()

        def search(**kwargs):
            with guard:
                limits.append(kwargs["limit"])
                index = len(limits) - 1
            chosen = page_sets[index] if index < len(page_sets) else page_sets[-1]
            result = MagicMock()
            result.pages_as_dicts.return_value = chosen()
            return result

        with (
            patch.object(stac, "StacApiIO"),
            patch.object(stac, "Client") as client,
            caplog.at_level(logging.WARNING, logger=stac.__name__),
        ):
            client.open.return_value.search.side_effect = search
            items = stac._query_stac_items(provider, collection, None, start, end, bbox=(-3.0, 50.0, -2.0, 51.0))
        return limits, items

    @staticmethod
    def _refuses_at(page_no: int, refusal=None):
        """A page iterator serving ``page_no - 1`` pages and then refusing."""
        refusal = refusal or _unretried(502)
        served = [{"features": [_item(f"p{i}")]} for i in range(1, page_no)]
        return lambda: _pages_then_refusal(served, refusal)

    @staticmethod
    def _serves(ids):
        return lambda: iter([{"features": [_item(i) for i in ids]}])

    def test_a_first_page_refusal_is_retried_at_half_the_page(self, caplog) -> None:
        """The case with no shorter window to fall back on: a shorter one asks page 1 too."""
        limits, items = self._query([self._refuses_at(1), self._serves(["a", "b"])], caplog, "2021-09-01", "2021-10-02")

        assert limits == [100, 50], "the refused window must be re-asked at half the page"
        assert [i.id for i in items] == ["a", "b"]
        assert "items per page" in caplog.text
        # And NOT by shortening the window, which would ask the first page identically.
        assert "shorter window" not in caplog.text

    def test_a_single_day_refused_past_page_one_is_retried_at_half_the_page(self, caplog) -> None:
        """The re-cut's floor. A single day cannot be shortened, so the page must shrink."""
        limits, items = self._query([self._refuses_at(4), self._serves(["a"])], caplog, "2021-09-01", "2021-09-01")

        assert limits == [100, 50]
        assert "items per page" in caplog.text
        assert [i.id for i in items] == ["p1", "p2", "p3", "a"], (
            "the pages walked before the refusal are kept, and the re-walk's are appended"
        )

    def test_shortening_the_window_is_preferred_over_shrinking_the_page(self, caplog) -> None:
        """Two halves walk about as many pages as the parent; a smaller page walks twice as many.

        A GUARD, not a proof: this passes with the page step-down removed, because without it
        no search asks for a smaller page anyway. It exists to catch the preference being
        inverted later, which would make every deep refusal twice as expensive as it needs
        to be without failing anything else.
        """
        limits, _ = self._query(
            [self._refuses_at(4), self._serves(["a"]), self._serves(["b"])], caplog, "2021-09-01", "2021-10-02"
        )

        assert set(limits) == {100}, "no search should have asked for a smaller page"
        assert "shorter window" in caplog.text
        assert "items per page" not in caplog.text

    def test_the_page_halves_to_a_floor_and_then_the_refusal_is_reported(self, caplog) -> None:
        """It must give up. A window nothing can serve is a refusal, not an infinite retry.

        A GUARD, not a proof: the refusal propagates with or without the step-down, so this
        does not distinguish them. What it does catch is the step-down failing to terminate,
        which would hang a leg rather than fail it -- the worse of the two outcomes, and the
        one no other test here would notice.
        """
        with pytest.raises(CatalogueQueryError) as caught:
            self._query([self._refuses_at(1)], caplog, "2021-09-01", "2021-09-01")

        assert caught.value.refusal.kind is RefusalKind.UPSTREAM_ERROR
        assert caught.value.request.page == 1

    def test_the_floor_is_reached_by_halving_and_nothing_below_it_is_asked(self, caplog) -> None:
        """The ladder is pinned, so a change to the step or the floor is visible here."""
        with pytest.raises(CatalogueQueryError):
            self._query([self._refuses_at(1)], caplog, "2021-09-01", "2021-09-01")

        asked = [int(m) for m in re.findall(r"at (\d+) items per page", caplog.text)]
        assert asked == [50, 25, 12], f"expected halving to the floor, got {asked}"
        assert all(size >= stac._MIN_PAGE_SIZE for size in asked)

    @pytest.mark.parametrize("status", [500, 504])
    def test_a_backend_failure_is_not_re_asked_as_a_smaller_request(self, status: int, caplog) -> None:
        """Only a SIZE refusal is worth asking again for less. 500 and 504 mean "I broke".

        Both are still force-listed, so one reaches the handler having already spent the
        ladder's whole backoff. Re-cutting it would hand that same ladder to every child and
        turn a persistent outage into minutes of backoff multiplied across a recursion, so it
        propagates once and lets the attempt budget decide instead.
        """
        with pytest.raises(CatalogueQueryError) as caught:
            self._query([self._refuses_at(4, _refused(status))], caplog, "2021-09-01", "2021-10-02")

        assert caught.value.refusal.kind is RefusalKind.UPSTREAM_ERROR
        assert caught.value.refusal.status == status
        assert "shorter window" not in caplog.text
        assert "items per page" not in caplog.text

    def test_a_stated_overload_on_the_first_page_is_not_answered_with_a_smaller_page(self, caplog) -> None:
        """A 503 means BUSY. A smaller page is MORE requests, which is the wrong direction.

        A GUARD, not a proof: a 503 propagates either way. It pins the classification the
        step-down is gated on, so widening that gate to every refusal -- which would answer
        an overload by multiplying the requests causing it -- fails here.
        """
        with pytest.raises(CatalogueQueryError) as caught:
            self._query([self._refuses_at(1, _refused(503))], caplog, "2021-09-01", "2021-10-02")

        assert caught.value.refusal.kind is RefusalKind.LOAD
        assert "items per page" not in caplog.text

    def test_a_reduced_page_is_inherited_by_the_windows_cut_from_it(self, caplog) -> None:
        """A window must never return to a size that was already refused."""
        limits, _ = self._query(
            [
                self._refuses_at(1),  # page 1 refused at 100 -> retry the whole window at 50
                self._refuses_at(4),  # at 50 it refuses at depth -> cut into shorter windows
                self._serves(["a"]),
                self._serves(["b"]),
            ],
            caplog,
            "2021-09-01",
            "2021-10-02",
        )

        assert limits[:2] == [100, 50]
        assert set(limits[2:]) == {50}, "the shorter windows must keep the reduced page size"

    def test_the_re_walk_does_not_double_count_the_pages_it_repeats(self, caplog) -> None:
        """The whole window is asked again, so every item before the refusal comes back."""
        limits, items = self._query(
            [self._refuses_at(3), self._serves(["p1", "p2", "later"])], caplog, "2021-09-01", "2021-09-01"
        )

        assert limits == [100, 50]
        ids = [i.id for i in items]
        assert ids == ["p1", "p2", "later"], f"expected each item once, in walk order, got {ids}"
        assert len(ids) == len(set(ids))


class TestARefusalThatKeepsArrivingCannotAmplify:
    """Both levers terminate on their own; together, across a window, they multiplied.

    Each is individually bounded — windows bottom out at a single day, pages at
    `_MIN_PAGE_SIZE` — so nothing hangs. But a refusal that keeps arriving, a real gateway
    outage rather than the size cap, makes both recurse across the whole window. Measured
    against a stub refusing every walk past page 1 before this bound existed: a one-month
    query fired 328 requests, a year 3,648, and nine years 32,868 — all aimed at a service
    already answering with 5xx.
    """

    @staticmethod
    def _always_refuses_past_page_one(start: str, end: str) -> tuple[int, BaseException | None]:
        """Walk a window whose every search serves one page and then 502s. Returns the count."""
        provider = stac._get_provider_config("earth-search")
        collection = stac._get_collection_config("earth-search", "sentinel-2-l2a")
        searches = 0
        guard = threading.Lock()

        def search(**_kwargs):
            nonlocal searches
            with guard:
                searches += 1
            result = MagicMock()

            def pages():
                yield {"features": [_item("a")]}
                raise _unretried(502)

            result.pages_as_dicts.return_value = pages()
            return result

        with patch.object(stac, "StacApiIO"), patch.object(stac, "Client") as client:
            client.open.return_value.search.side_effect = search
            try:
                stac._query_stac_items(provider, collection, None, start, end, bbox=(-3.0, 50.0, -2.0, 51.0))
            except BaseException as exc:
                return searches, exc
        return searches, None

    def test_it_gives_up_after_a_bounded_number_of_replacements(self) -> None:
        searches, raised = self._always_refuses_past_page_one("2019-02-28", "2019-04-01")

        assert isinstance(raised, CatalogueQueryError), "it must still report the refusal"
        assert searches <= 4 * stac._MAX_REFUSAL_RE_PARTITIONS, (
            f"{searches} searches for one month is amplification, not recovery"
        )

    def test_the_bound_does_not_grow_with_the_window(self) -> None:
        """The property that matters: a year must not cost ten times a month.

        Before the budget the cost scaled with the days in the window, because every branch
        of the recursion had its own independent stopping condition.
        """
        month, _ = self._always_refuses_past_page_one("2019-02-28", "2019-04-01")
        year, _ = self._always_refuses_past_page_one("2019-01-01", "2019-12-31")
        nine_years, _ = self._always_refuses_past_page_one("2017-01-01", "2025-12-31")

        # Not exactly equal: how a window spends the budget depends a little on how many pieces
        # it can be cut into. What must not happen is the cost SCALING with the days, which is
        # what it did before — 328 for a month against 32,868 for nine years.
        assert nine_years <= month + 4, f"{month}, {year}, {nine_years} — the cost must not grow"
        assert year <= month + 4


class TestTheCatalogueRootKeepsTheRetryTheLadderStoppedGiving:
    """The root is a small static document, so no remedy in this module applies to it.

    A shorter window and a smaller page both work by making the ANSWER smaller. The catalogue
    root cannot be made smaller, so taking 502 out of this provider's ladder left the root with
    no retry at all — and that is worse than one wasted attempt, because the leg-retry layer
    declares a refusal deterministic once it sees the identical signature twice. A gateway blip
    on the root could therefore end a leg rather than delay it.
    """

    @staticmethod
    def _open(monkeypatch, failures: list[BaseException]):
        """Open the root, failing the first ``len(failures)`` attempts. Returns the count."""
        monkeypatch.setattr(stac, "_ROOT_OPEN_BACKOFF_S", 0.0)
        provider = stac._get_provider_config("earth-search")
        attempts = 0

        def opener(*_a, **_k):
            nonlocal attempts
            attempts += 1
            if attempts <= len(failures):
                raise failures[attempts - 1]
            return MagicMock()

        with patch.object(stac, "StacApiIO"), patch.object(stac, "Client") as client:
            client.open.side_effect = opener
            clients = stac._PerThreadClient(provider, "sentinel-2-l2a", "2021-09-01/2021-10-02")
            try:
                clients.get()
            except CatalogueQueryError as exc:
                return attempts, exc
        return attempts, None

    def test_a_transient_root_502_is_retried_rather_than_ending_the_leg(self, monkeypatch) -> None:
        attempts, raised = self._open(monkeypatch, [_unretried(502), _unretried(502)])

        assert raised is None, "a root 502 that clears must not surface at all"
        assert attempts == 3

    def test_a_root_502_that_never_clears_is_reported_as_page_zero(self, monkeypatch) -> None:
        """It must still give up, and still be named as the root rather than as a search."""
        attempts, raised = self._open(monkeypatch, [_unretried(502)] * 10)

        assert attempts == stac._ROOT_OPEN_ATTEMPTS
        assert isinstance(raised, CatalogueQueryError)
        assert raised.request.page == 0
        assert raised.request.area == "catalogue-root"

    def test_only_the_status_the_ladder_dropped_is_retried_here(self, monkeypatch) -> None:
        """503 still has its full ladder underneath, so retrying it again here would double it."""
        attempts, raised = self._open(monkeypatch, [_refused(503)])

        assert attempts == 1, "a status the ladder still holds must not be retried a second time"
        assert isinstance(raised, CatalogueQueryError)


class TestTheRemedyFollowsTheProviderNotJustTheStatus:
    """A 502 from a provider we never measured must not be answered with this workaround.

    Re-cutting is measured for one catalogue's response-size cap. Elsewhere a 502 keeps its full
    ladder, so it reaches us having already spent the backoff — and re-cutting it would hand
    that same ladder to every child window, multiplying an outage instead of recovering from it.
    """

    def test_a_provider_without_the_flag_is_not_re_cut(self, caplog) -> None:
        provider = dataclasses.replace(stac._get_provider_config("earth-search"), refuses_oversized_pages=False)
        collection = stac._get_collection_config("earth-search", "sentinel-2-l2a")
        pages = _pages_then_refusal([{"features": [_item("a")]}], _refused(502))

        with (
            patch.object(stac, "StacApiIO"),
            patch.object(stac, "Client") as client,
            caplog.at_level(logging.WARNING, logger=stac.__name__),
            pytest.raises(CatalogueQueryError),
        ):
            search = client.open.return_value.search
            search.return_value.pages_as_dicts.return_value = pages
            stac._query_stac_items(
                provider, collection, None, "2021-09-01", "2021-10-02", bbox=(-3.0, 50.0, -2.0, 51.0)
            )

        assert "shorter window" not in caplog.text
        assert "items per page" not in caplog.text
