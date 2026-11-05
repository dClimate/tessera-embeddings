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

from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference import data_loading
from tessera_embeddings.inference.data_loading import _months_within_assessed, check_time_window_coverage

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


# --- the gate that applies it: check_time_window_coverage -------------------------------------

WINDOW = parse_time_window("July 2025")  # August 2024 through July 2025


def _time_group(start: str | None = None, end: str | None = None, **attrs) -> zarr.Group:
    """An in-memory group with a MONTHLY ``time`` array over [start, end], or none at all.

    Monthly (first of each month) rather than just the two endpoints, so a store that "spans"
    the window actually contains every month the per-month coverage check requires.
    """
    root = zarr.group(zarr.storage.MemoryStore())
    months = np.arange(np.datetime64(start, "M"), np.datetime64(end, "M") + 1) if start and end else np.array([])
    times = months.astype("datetime64[ns]").astype("int64")
    array = root.create_array("time", shape=times.shape, dtype=np.int64, chunks=(max(times.size, 1),))
    if times.size:
        array[:] = times
    root.attrs.update(attrs)
    return root


@pytest.fixture
def stores(monkeypatch) -> SimpleNamespace:
    """Every store opens as ``stores.group()`` (default: spans the window); ``stores.paths`` records each open."""
    served = SimpleNamespace(paths=[], group=lambda: _time_group("2024-01-01", "2025-12-31"))

    def _open(path, **kwargs):  # accept get_credentials/region
        served.paths.append(path)
        return served.group()

    monkeypatch.setattr(data_loading, "open_store_as_zarr_group", _open)
    return served


def _check(*, s1_orbit: str = "ascending", **kwargs):
    return check_time_window_coverage("s3://fake/mosaic", WINDOW, s1_orbit=s1_orbit, **kwargs)


def test_stores_spanning_the_window_pass(stores) -> None:
    _check()


@pytest.mark.parametrize(
    ("store_start", "store_end"),
    [
        ("2024-10-01", "2025-12-31"),  # missing the window's Aug/Sep 2024 months
        ("2024-01-01", "2025-06-15"),  # missing the window's July 2025 month
    ],
    ids=["too-late-start", "too-early-end"],
)
def test_a_store_missing_window_months_fails(stores, store_start, store_end) -> None:
    stores.group = lambda: _time_group(store_start, store_end)
    with pytest.raises(InsufficientCoverageError, match="missing"):
        _check()


@pytest.mark.parametrize(
    ("s1_orbit", "expected"),
    [
        ("ascending", {"reflectance", "sar_ascending"}),
        ("both", {"reflectance", "sar_ascending", "sar_descending"}),
    ],
)
def test_each_orbit_checks_its_own_stores(stores, s1_orbit, expected) -> None:
    _check(s1_orbit=s1_orbit)
    names = ("reflectance", "sar_ascending", "sar_descending")
    assert {n for n in names if any(n in p for p in stores.paths)} == expected
    assert len(stores.paths) == len(expected)


@pytest.mark.parametrize("skip_coverage_check", [False, True])
def test_an_empty_store_raises(stores, skip_coverage_check) -> None:
    """``skip_coverage_check`` does not bypass the empty-store guard (it runs first)."""
    stores.group = _time_group
    with pytest.raises(InsufficientCoverageError, match="no time entries"):
        _check(skip_coverage_check=skip_coverage_check)


def test_skip_coverage_check_requires_in_window_data(stores) -> None:
    """Partial-window mode still rejects a store with only out-of-window dates."""
    stores.group = lambda: _time_group("2023-01-01", "2023-12-31")  # entirely before the window
    with pytest.raises(InsufficientCoverageError, match="no timestamps within the window"):
        _check(skip_coverage_check=True)


def test_skip_coverage_check_bypasses_range_check(stores) -> None:
    """A store ending well before the window would normally raise; the skip bypasses that."""
    stores.group = lambda: _time_group("2024-10-01", "2024-12-31")
    _check(skip_coverage_check=True)


def test_an_assessed_window_cannot_excuse_a_store_with_no_in_window_dates(stores) -> None:
    """STRICT mode needs the in-window guard too, despite its every-month rule.

    An `assessed_window` says a month was examined and held nothing reachable, which
    is a finding rather than a gap — but it can explain away EVERY month of the
    window, emptying the missing list. A store the ingest looked at and wrote nothing
    into would then pass the one gate that exists to fail before a GPU fleet is
    provisioned, and the run would die at the first read instead, because the loaders
    raise on an empty filtered index.
    """
    # Dates from a PRIOR year only: non-empty, but nothing the window can use.
    stores.group = lambda: _time_group("2023-01-01", "2023-12-31", assessed_window=["2024-08-01", "2025-07-31"])
    with pytest.raises(InsufficientCoverageError, match="no timestamps within the window"):
        _check()


def test_a_month_whose_dates_were_all_lost_is_excused_like_any_other(stores) -> None:
    """The behaviour this replaces, stated as a decision rather than left as a silence.

    Absence inside the assessed window used to be split in two: a month with nothing to find
    was excused, and a month whose acquisitions were skipped as unreadable was not. The store
    recorded the second in ``assessed_unreadable_dates``, and the gate subtracted it.

    That distinction cannot survive the resume rule. A day only ever fails to be written once
    a later day has committed above it, and a store's time axis only grows — so the day is
    closed for good, whatever the imagery turns out to be. Refusing the month therefore does
    not protect anything; it deadlocks the cell, because no run can ever fill it and the only
    exit is deleting the store. That is a judgement a human makes from an audit.

    What the reader gets instead is the published product: per-pixel observation counts and a
    per-month coverage mask per sensor, which say what a pixel actually has rather than why.

    This is also the legitimate sparse-zone case — part of the window held, the rest examined
    and empty — which the in-window guard above must not undo.
    """
    stores.group = lambda: _time_group(
        "2024-08-01",
        "2024-10-31",  # first 3 months only
        assessed_window=["2024-08-01", "2025-07-31"],
        # What an old store may still carry. It must be inert, not merely unread: a store
        # written before the removal must not behave differently from one written after.
        assessed_unreadable_dates=[{"date": "2025-03-14", "objects": 4, "scope": "tile"}],
    )
    assert _check()["stores"]["reflectance"]["months_absent_unexplained"] == 0
