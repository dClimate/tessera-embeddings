"""The OPERA read credential must be renewed off its OWN expiry, not a fixed cadence.

Regression cover for a failure seen under load: the credential ASF mints lives about an
hour, and the S1 ingest renewed it only between time batches. A batch longer than the
credential — which is what a dense region on a narrow fleet produces — could therefore
never renew, and every read after the hour failed with an expired token. Sparse regions
finished inside the hour and were unaffected, which is why it presented as "some runs
fail and some don't" rather than as a configuration error.

These tests pin the parsing contract and the degrade-safely path. The call-site contract
(renew before every date, not only per batch) is pinned by the parity/integration cover
that drives the real loop.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tessera_embeddings.ingest.s1_roi import (
    CRED_EXPIRY_MARGIN_SEC,
    DEFAULT_CRED_REFRESH_INTERVAL_SEC,
    _parse_credential_expiry,
)


@pytest.mark.parametrize(
    "raw",
    [
        "2026-07-28T03:05:01Z",
        "2026-07-28T03:05:01+00:00",
        "2026-07-28 03:05:01+00:00",
    ],
)
def test_parses_the_spellings_asf_actually_returns(raw: str) -> None:
    """All three differ only in punctuation and must give the same instant."""
    assert _parse_credential_expiry({"expiration": raw}) == datetime(
        2026, 7, 28, 3, 5, 1, tzinfo=UTC
    ).timestamp()


def test_naive_timestamp_is_read_as_utc() -> None:
    """A missing zone must not be read as local time — that would mis-date the expiry
    by the machine's offset and could make a stale credential look fresh."""
    assert _parse_credential_expiry({"expiration": "2026-07-28T03:05:01"}) == datetime(
        2026, 7, 28, 3, 5, 1, tzinfo=UTC
    ).timestamp()


@pytest.mark.parametrize("creds", [{}, {"expiration": ""}, {"expiration": "not a date"}])
def test_unusable_expiry_degrades_instead_of_raising(creds: dict) -> None:
    """No expiry, or an unparseable one, falls back to the age-based cadence.

    Deliberately not an error: an unreadable expiry must not sink an ingest that would
    otherwise run, because the fallback cadence is still safe.
    """
    assert _parse_credential_expiry(creds) is None


def test_margin_exceeds_the_fallback_cadence_is_not_required_but_margin_is_generous() -> None:
    """The margin must be large enough to cover a single date's write.

    Stated as a floor rather than an exact value: renewing is two cheap HTTP calls, so
    the margin should be generous, and the failure it prevents costs the rest of the run.
    """
    assert CRED_EXPIRY_MARGIN_SEC >= 10 * 60
    assert DEFAULT_CRED_REFRESH_INTERVAL_SEC > 0
