"""An absent month must be excused only when the ingest provably examined it.

A mosaic's time axis carries no record of what was LOOKED AT, so a missing month means either
"the ingest examined it and there was nothing reachable" or "the ingest never got there". The
first is a finding about the world — a Sentinel-1 pass covers a swath, not a whole UTM zone,
and some zones have an orbit reaching their land on no date of the year. The second is a
partial mosaic that must never be published.

`assessed_window` is what separates them: the ingest records the range it processed in full,
so a month wholly inside it was examined. Everything here is about the boundary between those
two readings, because getting it wrong in the permissive direction publishes a mosaic with a
hole and getting it wrong in the strict direction only costs a re-ingest.
"""

from __future__ import annotations

from tessera_embeddings.inference.data_loading import _months_within_assessed

JAN, FEB, MAR, DEC = (2024, 1), (2024, 2), (2024, 3), (2024, 12)


def test_a_month_wholly_inside_the_assessed_range_is_excused() -> None:
    """The golden path: a calendar-year ingest examined every month it is asked about."""
    assert _months_within_assessed([FEB], ["2024-01-01", "2024-12-31"]) == {FEB}


def test_every_month_of_a_calendar_year_window_is_excusable() -> None:
    """Campaign windows are calendar years, so strictness must cost nothing there."""
    months = [(2024, m) for m in range(1, 13)]
    assert _months_within_assessed(months, ["2024-01-01", "2024-12-31"]) == set(months)


def test_a_month_outside_the_assessed_range_is_not_excused() -> None:
    """The failure this exists to preserve: an ingest that never covered the month."""
    assert _months_within_assessed([DEC], ["2024-01-01", "2024-06-30"]) == set()


def test_a_partially_assessed_month_is_not_excused() -> None:
    """Half a month examined could hide unexamined days, so it stays an error.

    March is only covered to the 15th here. Excusing it would let an ingest that stopped
    mid-month look complete.
    """
    assert _months_within_assessed([MAR], ["2024-01-01", "2024-03-15"]) == set()
    # …and the months it DID cover in full are still excused, so the rule is per-month.
    assert _months_within_assessed([JAN, FEB, MAR], ["2024-01-01", "2024-03-15"]) == {JAN, FEB}


def test_a_month_starting_before_the_assessed_range_is_not_excused() -> None:
    """Guards the lower boundary as well as the upper — an ingest starting mid-January."""
    assert _months_within_assessed([JAN], ["2024-01-02", "2024-12-31"]) == set()


def test_december_boundary_arithmetic_is_right() -> None:
    """December's last day needs a year rollover to compute; an off-by-one here would
    silently excuse a December the ingest stopped short of.
    """
    assert _months_within_assessed([DEC], ["2024-01-01", "2024-12-31"]) == {DEC}
    assert _months_within_assessed([DEC], ["2024-01-01", "2024-12-30"]) == set()


def test_a_leap_february_is_measured_to_the_29th() -> None:
    assert _months_within_assessed([FEB], ["2024-02-01", "2024-02-29"]) == {FEB}
    assert _months_within_assessed([FEB], ["2024-02-01", "2024-02-28"]) == set()


def test_an_unusable_attribute_excuses_nothing() -> None:
    """Every degraded path must make the gate STRICTER, never more permissive.

    A damaged or absent record means we cannot show the month was examined, and the
    conservative answer to that is to fail. The asymmetry is the whole safety argument.
    """
    for bad in (None, [], ["2024-01-01"], "2024-01-01", ["not-a-date", "2024-12-31"], 42, {}):
        assert _months_within_assessed([FEB], bad) == set(), f"excused something for {bad!r}"
