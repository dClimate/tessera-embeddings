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
    is_provider_refusal,
    is_unreadable_source,
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
