"""The verdict every read failure gets, pinned case by case.

This is a CHARACTERISATION test: it records what the classifier does rather than arguing for it,
and its purpose is to make a restructuring of that classifier provable instead of merely reasoned
about. The table below was generated from the code as it stood before `classify_read_failure`
existed, so a change in any row is a change in behaviour and has to be defended as one.

Why this file exists at all: correctness here used to rest on nine marker tuples being jointly
complete and mutually exclusive, with nothing checking either. Gaps recurred one status at a time
and an overlap let call order decide whether a date was skipped or a leg retried. A table is the
cheapest thing that notices.
"""

from __future__ import annotations

import pytest
from rasterio._err import CPLE_AppDefinedError, CPLE_OpenFailedError
from rasterio.errors import RasterioIOError

from tessera_embeddings.ingest.duplicates import (
    ReadFailure,
    cause_was_flattened,
    classify_read_failure,
    classify_read_failure_in,
    is_provider_refusal,
    is_unreadable_source,
    unreadable_source_in,
)


def _chained(cause: str, *, opening: bool = False) -> BaseException:
    """A read failure shaped as a worker raises one: a rasterio wrapper over a real GDAL cause.

    Raised rather than constructed, because ``raise ... from`` is the only thing that puts a chain
    on an exception and the chain is what is classified. ``opening`` selects the GDAL class that
    reports a failure at open, which is how a missing object arrives.
    """
    cls = CPLE_OpenFailedError if opening else CPLE_AppDefinedError
    try:
        raise cls(1, 4, cause)
    except cls as inner:  # type: ignore[misc]
        try:
            raise RasterioIOError("Read failed. See previous exception for details.") from inner
        except RasterioIOError as failure:
            return failure


#: (cause, raised at open, gives up the date, earns the refusal budget)
TABLE: list[tuple[str, bool, bool, bool]] = [
    ("ZIPDecode:Decoding error at scanline 0", False, True, False),
    ("TIFFReadEncodedTile() failed", False, True, False),
    ("No such band/alias: scl", False, False, False),
    ("ObjectNotFound: The specified key does not exist.", True, True, False),
    ("An error occurred (NoSuchKey) when calling the GetObject operation", True, True, False),
    ("HTTP response code: 404", False, True, False),
    ("HTTP response code: 410", False, True, False),
    ("HTTP response code: 401", False, False, False),
    ("HTTP response code: 400", False, False, False),
    ("HTTP response code: 403", False, False, True),
    ("HTTP response code: 408", False, False, True),
    ("HTTP response code: 429", False, False, True),
    ("HTTP response code: 500", False, False, True),
    ("HTTP response code: 502", False, False, True),
    ("HTTP response code: 503", False, False, True),
    ("HTTP response code: 504", False, False, True),
    ("HTTP response code: 599", False, False, True),
    ("AccessDenied: not authorized to perform: s3:GetObject", False, False, True),
    ("SlowDown", False, False, True),
    ("InternalError", False, False, True),
    ("ServiceUnavailable", False, False, True),
    ("TooManyRequests", False, False, True),
    ("InvalidAccessKeyId", False, False, False),
    ("ExpiredToken", False, False, False),
    ("SignatureDoesNotMatch", False, False, False),
    ("Connection reset by peer", False, False, True),
    ("Broken pipe", False, False, True),
    ("Could not resolve host: sentinel-cogs.s3.us-west-2.amazonaws.com", False, False, True),
    ("Connection refused", False, False, True),
    ("Empty reply from server", False, False, True),
    ("SSL peer handshake failed", False, False, True),
    ("RequestTimeout", False, False, True),
    ("Connection aborted", False, False, True),
    ("operation timed out", False, False, True),
    # Two causes in one chain, which is the case an ORDER of asking used to decide. The refusal
    # outranks the codec name: a service that said no has not told us anything about the bytes.
    ("Connection refused: ZIPDecode:Decoding error at scanline 0", False, False, True),
    ("HTTP response code: 503, and ZIPDecode:Decoding error at scanline 0", False, False, True),
]


@pytest.mark.parametrize(("cause", "opening", "gives_up", "refusal"), TABLE, ids=[c[:38] for c, *_ in TABLE])
def test_the_verdict_for_this_cause_has_not_moved(cause: str, opening: bool, gives_up: bool, refusal: bool) -> None:
    """Both predicates, on a real chain, against the table."""
    failure = _chained(cause, opening=opening)
    assert is_unreadable_source(failure) is gives_up
    assert is_provider_refusal(failure) is refusal


@pytest.mark.parametrize(("cause", "opening"), [(c, o) for c, o, *_ in TABLE], ids=[c[:38] for c, *_ in TABLE])
def test_no_cause_earns_both_verdicts(cause: str, opening: bool) -> None:
    """The property the old design could only assert. One classifier returns one member, so this
    cannot fail by construction — which is the point, and why it is cheap to keep asserting.
    """
    failure = _chained(cause, opening=opening)
    assert not (is_unreadable_source(failure) and is_provider_refusal(failure))


@pytest.mark.parametrize(("cause", "gives_up"), [(c, g) for c, _, g, _ in TABLE], ids=[c[:38] for c, *_ in TABLE])
def test_the_words_alone_reach_the_same_verdict(cause: str, gives_up: bool) -> None:
    """One taxonomy, whichever side of a boundary the caller stands on.

    Not every caller holds an exception. A leg that runs as its own deployment reports through a
    state message, and its parent decides from that whether to re-dispatch — so it asks
    :func:`unreadable_source_in` where a worker asks :func:`is_unreadable_source`. Two answers to
    "will these bytes ever read" is how a leg comes to give up a date the parent then keeps
    retrying, so the same words must produce the same verdict on both paths.
    """
    assert unreadable_source_in(f"RasterioIOError: Read failed. | CPLE_AppDefinedError: {cause}") is gives_up


def test_words_that_lost_their_cause_claim_nothing() -> None:
    """The boundary degrades in the SAFE direction, and this is what that looks like.

    A Prefect state message carries the outermost exception only, so a rasterio wrapper arrives
    with the GDAL cause stripped — and the wrapper's own sentence names no bytes. That is an
    absence of evidence, not evidence of transience, and it earns the retry rather than the
    permanent verdict: retrying a permanent failure costs one leg's fleet, refusing to retry a
    transient one strands the cell.
    """
    assert unreadable_source_in("RasterioIOError: Read failed. See previous exception for details.") is False


def test_the_flattened_shape_earns_nothing() -> None:
    """A cause destroyed crossing the worker boundary: no wait on suspicion, no date given up."""
    flattened = Exception("RasterioIOError('Read failed. See previous exception for details.')")
    assert classify_read_failure(flattened) is ReadFailure.UNDECIDABLE
    assert is_unreadable_source(flattened) is False
    assert is_provider_refusal(flattened) is False


def test_every_verdict_in_the_closed_set_is_reachable() -> None:
    """A member nothing can produce is either dead or a gap in this table. Both want finding.

    ``REFUSAL_UNATTRIBUTED`` needs refusal words with no source-reader corroboration, which the
    table cannot express because every row is a rasterio chain — so it is built here.
    """
    reached = {classify_read_failure(_chained(c, opening=o)) for c, o, *_ in TABLE}
    reached.add(classify_read_failure(Exception("RasterioIOError('Read failed.')")))
    reached.add(classify_read_failure(RuntimeError("AccessDenied")))
    assert reached == set(ReadFailure), f"never produced: {set(ReadFailure) - reached}"


def test_a_note_is_part_of_the_chain_the_verdict_is_read_from() -> None:
    """Evidence attached to a failure counts, because not all of it ever reaches the exception.

    GDAL states some refusals only in its own log and raises the codec failure that follows, so
    the reader holding that log writes what it found onto the exception — see
    ``loader_failures.carry_logged_refusal``, the only writer of notes here. If this text were
    not read, that evidence would be gathered and then ignored, and the verdict would be reached
    from half of what is known.
    """
    failure = _chained("ZIPDecode:Decoding error at scanline 0")
    assert classify_read_failure(failure) is ReadFailure.UNREADABLE

    failure.add_note("CPLE_AWSAccessDenied in HTTP response code: 403")
    assert classify_read_failure(failure) is ReadFailure.PROVIDER_REFUSED


def test_a_note_cannot_shorten_how_far_down_the_chain_is_read() -> None:
    """The chain's bound counts LINKS, not entries. Counting entries would let notes push a
    cause out of view — and the cause is the whole of what a verdict is reached from, so
    attaching evidence would cost evidence.

    Built at the bound rather than near it: a short chain passes either way, which is how a
    bound that counts the wrong thing survives a test written for it.
    """
    deepest: BaseException = ValueError("CPLE_AppDefinedError: ZIPDecode:Decoding error at scanline 0")
    current = deepest
    for i in range(19):
        wrapper: BaseException = RuntimeError(f"RasterioIOError: wrapper {i}")
        wrapper.add_note("a note that says nothing")
        wrapper.__cause__ = current
        current = wrapper

    assert classify_read_failure(current) is ReadFailure.UNREADABLE


class TestLosingTheEvidenceVersusNotRecognisingIt:
    """Two different things that both end in no verdict, and the relationship between them.

    ``cause_was_flattened`` is about SHAPE — did the reason survive the hop from the worker that
    read to the worker that decides. ``UNDECIDABLE`` is about OUTCOME — nothing in what arrived
    matched anything we know. One implies the other and not the reverse, and both halves matter:

    * a flattened failure is always undecidable, because a repr carries no cause to read;
    * an undecidable failure is NOT always flattened — an intact chain naming something nobody has
      enumerated is undecidable too.

    Keeping them separate is what lets the two be told apart in a log, and they need different
    responses: the first says a worker is reading without the rescue installed, which is an
    infrastructure fault; the second says the taxonomy has a gap, which is a code change. Defining
    either in terms of the other would collapse that, so the containment is asserted rather than
    assumed — and so is its one-directionality, to stop a later simplification.
    """

    def test_a_flattened_failure_is_always_undecidable(self) -> None:
        for wrapper in (
            "RasterioIOError('Read failed. See previous exception for details.')",
            "WarpOperationError('Chunk and warp failed')",
            "RasterioIOError('B02.tif, band 1: IReadBlock failed at X offset 6')",
        ):
            flattened = Exception(wrapper)
            assert cause_was_flattened(flattened) is True, wrapper
            assert classify_read_failure(flattened) is ReadFailure.UNDECIDABLE, wrapper

    def test_undecidable_does_not_imply_flattened(self) -> None:
        """The half that stops one being written in terms of the other."""
        intact = _chained("a failure mode nobody has enumerated")
        assert classify_read_failure(intact) is ReadFailure.UNDECIDABLE
        assert cause_was_flattened(intact) is False

    @pytest.mark.parametrize(("cause", "opening"), [(c, o) for c, o, *_ in TABLE], ids=[c[:38] for c, *_ in TABLE])
    def test_no_case_with_a_real_chain_is_ever_reported_as_flattened(self, cause: str, opening: bool) -> None:
        """Every row of the table is an intact chain, so a true here would be a false alarm — and a
        false alarm on this detector reads as "the fleet cannot classify", which stops a leg.
        """
        assert cause_was_flattened(_chained(cause, opening=opening)) is False


#: The two wordings GDAL actually used during the 2026-08-24 ASF outage, copied verbatim from
#: CloudWatch (log group ``/ecs/global-tessera-prod``, 23:45 to 00:10 UTC) with only the granule
#: name shortened. Both arrived on ``rasterio._env`` at WARNING; 24 of the 25 records in that
#: window were the first shape and one was the second.
#:
#: They are checked in as CAPTURED text rather than composed here on purpose. The classifier read
#: neither, so every one of the 158 dates lost that night was judged ``undecidable`` and skipped
#: as unreadable data — and the tests that were meant to cover this asserted against a wording
#: rasterio never emits, so they passed throughout.
REAL_GDAL_REFUSALS = {
    # A chunk READ refused mid-transfer. The object's URL sits BETWEEN the phrase and the colon,
    # 249 characters of it, so a pattern anchored on `HTTP response code:` matches nothing.
    "mid-read 403": (
        "CPLE_AppDefined in HTTP response code on "
        "https://asf-cumulus-prod-opera-products.s3.us-west-2.amazonaws.com/OPERA_L2_RTC-S1/"
        "OPERA_L2_RTC-S1_T072-152803-IW2_20211108T150433Z_S1B_30_v1.0_VV.tif: 403"
    ),
    # The provider overloaded rather than refusing. "error code" where the other says "response".
    "mid-read 503": (
        "CPLE_AppDefined in HTTP error code: 503 - "
        "https://asf-cumulus-prod-opera-products.s3.us-west-2.amazonaws.com/OPERA_L2_RTC-S1/"
        "OPERA_L2_RTC-S1_T072-152803-IW2_20211108T150433Z_S1B_30_v1.0_VV.tif. "
        "Retrying again in 0.5 secs"
    ),
}


@pytest.mark.parametrize("shape", sorted(REAL_GDAL_REFUSALS))
def test_the_wordings_gdal_really_used_are_read_as_refusals(shape: str) -> None:
    """The verdict on text captured from the incident, not on text written to pass.

    Both say a status the service chose to send. Read as anything else, the response is to give
    up the date — which is what happened, 158 times.
    """
    assert classify_read_failure_in(REAL_GDAL_REFUSALS[shape]) is ReadFailure.PROVIDER_REFUSED


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        # Absence keeps its own verdict in the same wording: widening how a status is FOUND must
        # not widen what a status MEANS. This is the line that decides a date is abandoned.
        (
            "CPLE_OpenFailed in HTTP response code on https://asf/x_VV.tif: 404 NoSuchKey",
            ReadFailure.ABSENT,
        ),
        # A 4xx that is neither absence nor a reason to wait still fails the leg.
        ("CPLE_AppDefined in HTTP response code on https://asf/x_VV.tif: 401", ReadFailure.CLIENT_ERROR),
        # And the exception the refusal hides behind, alone, still claims nothing.
        ("WarpOperationError: Chunk and warp failed", ReadFailure.UNDECIDABLE),
    ],
)
def test_the_wider_status_match_does_not_widen_what_a_status_means(text: str, verdict: ReadFailure) -> None:
    """Negative controls for the pattern change, one per verdict it could have leaked into."""
    assert classify_read_failure_in(text) is verdict


def test_a_status_is_not_bound_to_a_phrase_far_away_from_it() -> None:
    """The gap between the phrase and the status is bounded, so one line cannot bind across it.

    A traceback flattened onto a single line can carry an unrelated number hundreds of characters
    later. Unbounded, the pattern would reach it and read a status the message never stated.
    """
    far = "HTTP response code" + " x" * 400 + ": 503"
    assert classify_read_failure_in(far) is ReadFailure.UNDECIDABLE
