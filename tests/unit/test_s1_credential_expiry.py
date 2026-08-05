"""The OPERA read credential must be renewed off its OWN expiry, and on a TIMER.

Two contracts, both load-bearing under fleet width:

* The renewal instant comes from the credential's advertised expiry, with an unreadable
  expiry degrading to an age-based cadence rather than sinking the ingest.
* Renewal runs on a timer independent of the work loop. A loop-driven renewal can only
  fire between units of work, so any unit that outlives the remaining margin cannot renew
  from inside itself — and since the credential's roughly one-hour life is unrelated to
  how long a unit takes, slow work silently loses its credentials.

The ticker tests assert LIVENESS — that a refresh arrives while the caller is blocked.
Counting calls after the block would pass for a ticker that never ticked. The call-site
contract is pinned by the parity cover that drives the real loop.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from tessera_embeddings.ingest.s1_roi import (
    CRED_EXPIRY_MARGIN_SEC,
    CRED_TICK_INTERVAL_SEC,
    DEFAULT_CRED_REFRESH_INTERVAL_SEC,
    _parse_credential_expiry,
    credential_ticker,
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
    assert _parse_credential_expiry({"expiration": raw}) == datetime(2026, 7, 28, 3, 5, 1, tzinfo=UTC).timestamp()


def test_naive_timestamp_is_read_as_utc() -> None:
    """A missing zone must not be read as local time — that would mis-date the expiry
    by the machine's offset and could make a stale credential look fresh.
    """
    assert (
        _parse_credential_expiry({"expiration": "2026-07-28T03:05:01"})
        == datetime(2026, 7, 28, 3, 5, 1, tzinfo=UTC).timestamp()
    )


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


def test_tick_interval_is_well_inside_the_margin() -> None:
    """The tick bounds how far past the renewal point a credential can drift.

    A tick as long as the margin would let the credential expire between checks, which is
    the failure the ticker exists to prevent.
    """
    assert CRED_TICK_INTERVAL_SEC < CRED_EXPIRY_MARGIN_SEC / 2


def test_ticker_refreshes_while_the_caller_is_blocked() -> None:
    """The point of the ticker: renewal must not wait for the work to finish.

    Asserted as liveness, not shape — a ticker that only refreshed on entry or exit would
    satisfy any test that merely counted calls after the block.
    """
    calls = threading.Event()

    with credential_ticker(calls.set, logging.getLogger(__name__), "ascending", "zone_test", interval_sec=0.01):
        # Still inside the block: nothing here advances the "work", so a refresh that
        # arrives can only have come from the timer.
        assert calls.wait(timeout=5.0), "ticker never refreshed while the caller was blocked"


def test_ticker_stops_when_the_block_exits() -> None:
    """A ticker that outlived its block would keep broadcasting for a finished leg."""
    log = logging.getLogger(__name__)
    counter = itertools.count()
    seen: list[int] = []

    with credential_ticker(lambda: seen.append(next(counter)), log, "ascending", "zone_test", interval_sec=0.01):
        assert _wait_until(lambda: len(seen) >= 1, timeout=5.0), "ticker never started"

    after_exit = len(seen)
    time.sleep(0.2)  # many tick intervals
    assert len(seen) == after_exit


def test_ticker_survives_a_failing_refresh() -> None:
    """One failed renewal must not end the ticker — the next tick is still inside the margin."""
    attempts: list[int] = []

    def flaky() -> None:
        attempts.append(len(attempts))
        if len(attempts) <= 2:
            raise RuntimeError("ASF said no")

    with credential_ticker(flaky, logging.getLogger(__name__), "ascending", "zone_test", interval_sec=0.01):
        assert _wait_until(lambda: len(attempts) >= 4, timeout=5.0), (
            f"ticker stopped after a failure; attempts={len(attempts)}"
        )


def _wait_until(predicate: Callable[[], bool], timeout: float) -> bool:
    """Poll ``predicate`` until true or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()
