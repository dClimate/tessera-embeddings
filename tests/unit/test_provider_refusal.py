"""A refusal from the source PROVIDER is a different finding from unreadable data.

The two failures arrive through the same wrapper and read almost alike, but they call for
opposite responses. An object that will not decode is not going to decode on the next attempt
either, so the useful move is to try a different copy of the imagery or give the date up. A
provider refusing reads — an authorization refusal, a throttle, a server error — says nothing
about the imagery at all: the same object read minutes earlier and reads again once the service
recovers, so substituting a different copy would swap in worse data to work around a bad
minute, and giving up on the data permanently would be worse still.

``is_unreadable_source`` therefore declines every refusal, and ``is_provider_refusal`` claims
exactly those. Both fail closed, so a cause neither recognises stops the caller rather than
quietly costing it data.

The messages below are the shapes a refused OPERA read actually surfaces as on the driver: the
read runs on a Dask worker and is re-raised through tblib, which cannot rebuild rasterio's
GDAL-backed classes, so what arrives is a plain exception carrying the original's text.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.ingest.duplicates import is_provider_refusal, is_unreadable_source

#: The four failure shapes a fleet-wide refusal of OPERA reads presents as, in the proportions
#: they arrive: the warp wrapper dominates because the refusal lands inside a chunk read, and
#: the bare forms surface when it lands at open instead.
_REFUSAL_SHAPES = [
    pytest.param("RasterioIOError: HTTP response code: 403", id="403-at-open"),
    pytest.param(
        "AccessDenied: An error occurred (AccessDenied) when calling the GetObject operation",
        id="access-denied",
    ),
    pytest.param("SlowDown: Please reduce your request rate", id="throttled"),
    pytest.param("RasterioIOError: HTTP response code: 503", id="service-unavailable"),
]


@pytest.mark.parametrize("message", _REFUSAL_SHAPES)
def test_a_refusal_is_the_providers(message: str) -> None:
    assert is_provider_refusal(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            "AccessDenied: An error occurred (AccessDenied) when calling the GetObject operation",
            id="access-denied",
        ),
        pytest.param("SlowDown: Please reduce your request rate", id="throttled"),
    ],
)
def test_a_named_refusal_is_not_unreadable_data(message: str) -> None:
    """The predicate that gates the copy ladder declines the refusals it can name.

    Stepping a tile-date down to an older copy because the service was briefly refusing reads
    would permanently substitute worse imagery for a transient fault, which is what that
    predicate documents itself as avoiding.

    It recognises the NAMED forms only. The numeric ones are covered by
    :func:`test_both_predicates_claim_a_numeric_refusal`, which pins what that leaves.
    """
    assert not is_unreadable_source(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("RasterioIOError: HTTP response code: 403", id="403"),
        pytest.param("RasterioIOError: HTTP response code: 503", id="503"),
    ],
)
def test_both_predicates_claim_a_numeric_refusal(message: str) -> None:
    """A refusal reported as a bare status code is claimed by BOTH predicates.

    ``is_unreadable_source`` excludes refusals by matching their names, so a status code
    carries none of the words it looks for and the wrapper's own decode marker decides. The
    consequence is a real one and not this predicate's to fix here: a caller that asks only
    that question treats a numeric refusal as data that will never read.

    What resolves it is ORDER at the call site. A caller that can act on both asks about the
    refusal first, because that is the answer that is specific — see the radar loop's
    give-up path, whose scoping is pinned in ``test_s1_given_up_dates``.
    """
    exc = RuntimeError("WarpOperationError: Chunk and warp failed")
    exc.__cause__ = RuntimeError(message)
    assert is_provider_refusal(exc)
    assert is_unreadable_source(exc)


def test_the_warp_wrapper_is_looked_through_to_find_the_refusal() -> None:
    """The exception that propagates DISCARDS the reason; the refusal is its cause.

    ``WarpOperationError('Chunk and warp failed')`` is what a refused chunk read raises, and it
    says nothing about why. A predicate reading only the top message would see a decode failure
    and nothing else — and a decode failure is the one verdict a refusal must not receive.
    """
    wrapper = RuntimeError("WarpOperationError: Chunk and warp failed")
    wrapper.__cause__ = RuntimeError("AccessDenied: when calling the GetObject operation")

    assert is_provider_refusal(wrapper)
    # The cause outranks the wrapper for the copy ladder too, so a named refusal inside a warp
    # failure does not step a tile-date down.
    assert not is_unreadable_source(wrapper)


def test_a_bare_warp_failure_is_not_attributed_to_the_provider() -> None:
    """With no refusal anywhere in the chain, the provider is not implicated.

    The complement of the test above, and what stops the refusal predicate from claiming every
    chunk read that fails for any reason.
    """
    assert not is_provider_refusal(RuntimeError("WarpOperationError: Chunk and warp failed"))


def test_a_decode_failure_is_not_a_provider_refusal() -> None:
    """The complement, so neither predicate is quietly answering for the other."""
    exc = RuntimeError("ZIPDecode:Decoding error at scanline 0")
    assert is_unreadable_source(exc)
    assert not is_provider_refusal(exc)


@pytest.mark.parametrize(
    "message",
    [
        pytest.param("ExpiredToken: The provided token has expired", id="expired"),
        pytest.param("InvalidAccessKeyId: The AWS Access Key Id does not exist", id="unknown-key"),
        pytest.param("SignatureDoesNotMatch: The request signature we calculated does not match", id="bad-signature"),
    ],
)
def test_our_own_credential_fault_is_never_the_providers(message: str) -> None:
    """A credential fault on this side must not be absorbed as somebody else's outage.

    It is repairable here and no amount of waiting fixes it, so spending a bounded budget of
    given-up dates on one would hide the fault and lose data for it.
    """
    assert not is_provider_refusal(RuntimeError(message))
    assert not is_unreadable_source(RuntimeError(message))


def test_a_credential_fault_wins_over_a_refusal_in_the_same_chain() -> None:
    """An expired token often surfaces AS a 403, so the order of the two checks decides.

    The credential marker has to be matched first, or every expiry on this side would read as
    an upstream refusal and be waited out instead of repaired.
    """
    exc = RuntimeError("RasterioIOError: HTTP response code: 403")
    exc.__cause__ = RuntimeError("ExpiredToken: The provided token has expired")
    assert not is_provider_refusal(exc)


def test_an_unrecognised_failure_belongs_to_neither() -> None:
    """Fails closed: the caller re-raises rather than giving up a date for an unexamined cause."""
    exc = ValueError("time axis is not monotonic")
    assert not is_provider_refusal(exc)
    assert not is_unreadable_source(exc)
