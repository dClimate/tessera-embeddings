"""Solar-day ownership: the shared mechanism both ingests query through.

The bug this module exists to prevent is invisible in the output — a day lands looking
complete and is missing acquisitions — so the tests pin the two properties that make it
impossible rather than checking any one symptom:

* **Owned ranges tile the window exactly.** Nothing is processed twice, and nothing
  outside the requested window is processed at all. This is what makes the unclamped
  query padding safe.
* **A queried range always covers its owned days in full.** Whatever the solar offset,
  every UTC instant a chunk's owned days can draw on lies inside the range it asks for.

The second property is the one the S1 batch loop violated at every boundary. It is
checked here against the real offset arithmetic, at the longitudes where it actually
bites, rather than against a hand-picked example.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from tessera_embeddings.ingest.solar_days import (
    SolarDayRange,
    bbox_mid_longitude,
    fixed_day_ranges,
    month_ranges,
    normalize_to_solar_day,
    owned_items,
    resolve_grouping_longitude,
    resume_window_start,
    solar_day_of,
    solar_day_offset_seconds,
    whole_window_range,
)

#: Longitudes whose whole-hour offset puts UTC midnight on top of a Sentinel-1 pass
#: (~06:00 descending, ~18:00 local solar). Zone 47's central meridian is +99 and it is
#: the third-densest UTM zone on Earth, so this is where a boundary split costs most.
#: +/-180.0 are here because they are VALID longitudes that the previous offset formula
#: truncated to +/-12 h, and +12 h is the one value that breaks
#: `normalize_to_solar_day`'s idempotence (noon + 12 h is midnight of the next day). Every
#: property below now runs at the dateline as well as at the 15-degree boundaries.
_HOSTILE_LONGITUDES = [99.0, 93.0, -93.0, -99.0, 0.0, 179.0, -179.0, 180.0, -180.0]


def _dates(start: str, end: str) -> list[str]:
    d0 = datetime.date.fromisoformat(start)
    d1 = datetime.date.fromisoformat(end)
    return [(d0 + datetime.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


class _Item:
    """Minimum surface both ingests' items expose to the ownership filter."""

    def __init__(self, when: datetime.datetime) -> None:
        self.datetime = when


# --- ownership tiles the window -------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2024-01-01", "2024-12-31"),  # the campaign's calendar year
        ("2024-01-15", "2024-03-04"),  # partial months at both ends
        ("2024-02-01", "2024-02-29"),  # a leap February, alone
        ("2024-06-10", "2024-06-10"),  # a single day
    ],
)
@pytest.mark.parametrize("producer", ["month", "batch7", "batch30", "whole"])
def test_owned_ranges_tile_the_window_exactly(start: str, end: str, producer: str) -> None:
    """Every date in the window is owned once; no date outside it is owned at all.

    This is what lets the query padding go unclamped. The pads reach a day beyond the
    window at both ends deliberately — an edge solar day draws on the UTC day outside —
    and nothing out-of-window escapes because writing is gated on ownership, never on
    the query bound.
    """
    ranges = {
        "month": lambda: month_ranges(start, end),
        "batch7": lambda: fixed_day_ranges(start, end, 7),
        "batch30": lambda: fixed_day_ranges(start, end, 30),
        "whole": lambda: [whole_window_range(start, end)],
    }[producer]()

    owned: list[str] = []
    for rng in ranges:
        owned.extend(_dates(rng.own_start, rng.own_end))

    assert owned == _dates(start, end), "owned days must tile the window exactly once"


@pytest.mark.parametrize("producer", ["month", "batch30", "whole"])
def test_the_query_pad_is_not_clamped_to_the_window(producer: str) -> None:
    """The window's own edges are padded like every other boundary.

    Clamping them is the S2 bug: the first owned solar day includes acquisitions dated
    the preceding UTC day at eastern longitudes, and the last owned day acquisitions
    dated the following UTC day at western ones. Clamped, no chunk could fetch them and
    the first and last day of the run were written short — silently, since the recorded
    assessed window still covered them.
    """
    start, end = "2024-01-01", "2024-12-31"
    ranges = {
        "month": lambda: month_ranges(start, end),
        "batch30": lambda: fixed_day_ranges(start, end, 30),
        "whole": lambda: [whole_window_range(start, end)],
    }[producer]()

    assert ranges[0].query_start == "2023-12-31", "the first chunk must reach the preceding UTC day"
    assert ranges[-1].query_end == "2025-01-01", "the last chunk must reach the following UTC day"


# --- a queried range always covers its owned days in full -----------------------------


def test_the_dateline_keeps_its_true_twelve_hour_offset() -> None:
    """+/-180 really is +/-12 h, and the offset must say so.

    This was briefly clamped to +/-11 h to make normalising idempotent, which bought that
    property with a wrong answer: at +180 it left 12:00-12:59 UTC on the current date
    instead of advancing it, one hour of every day, and `bbox_mid_longitude` maps every
    antimeridian-crossing box to exactly 180. Idempotence is handled in
    `normalize_to_solar_day` instead, where it holds for any offset.
    """
    assert solar_day_offset_seconds(180.0) == 12 * 3600
    assert solar_day_offset_seconds(-180.0) == -12 * 3600
    assert solar_day_offset_seconds(179.9) == 11 * 3600


@pytest.mark.parametrize("longitude", [*_HOSTILE_LONGITUDES, 179.9, -179.9])
def test_normalising_twice_is_normalising_once(longitude: float) -> None:
    """Idempotence end to end, not just via the offset — this is what callers rely on."""

    class _Item:
        def __init__(self) -> None:
            # An acquisition close enough to UTC midnight that any offset moves its date.
            self.datetime = datetime.datetime(2025, 6, 1, 23, 30, tzinfo=datetime.UTC)

    (once,) = normalize_to_solar_day([_Item()], mid_longitude=longitude)
    first = once.datetime

    (twice,) = normalize_to_solar_day([once], mid_longitude=longitude)
    assert twice.datetime == first, f"re-normalising moved the stamp at longitude {longitude}"

    (thrice,) = normalize_to_solar_day([twice], mid_longitude=longitude)
    assert thrice.datetime == first, "the streamed path normalises three times over"


def test_grouping_longitude_is_refused_rather_than_defaulted_to_utc() -> None:
    """No longitude means no solar day — and silence was the bug.

    `normalize_to_solar_day(mid_longitude=None)` groups by UTC date, which was a sound
    default while a caller could legitimately load by UTC. `_load_from_stac` now rejects
    every grouping but `solar_day`, so no such caller remains, and what the default actually
    produced was every item stamped to noon of its UTC date: right in the middle of a day,
    a full day wrong near midnight in the far east or far west, and reported nowhere.
    """
    with pytest.raises(ValueError, match="solar-day grouping needs a longitude"):
        resolve_grouping_longitude(None, None)

    assert resolve_grouping_longitude(42.0, None) == 42.0, "an explicit longitude wins"
    assert resolve_grouping_longitude(None, (10.0, 0.0, 20.0, 5.0)) == 15.0, "else derive from the bbox"


def test_an_antimeridian_bbox_midpoint_is_not_averaged() -> None:
    """A box crossing 180 is written west > east, and averaging it lands half a world away.

    (179, -179) averages to 0.0 — the middle of the wrong hemisphere, and a solar offset
    about twelve hours out. The box is walked eastward across the dateline instead. The
    result is exactly 180.0 — the meridian the box actually straddles — and its offset is the
    true +12 h, which is safe to produce now that idempotence no longer rests on the offset.
    """
    assert bbox_mid_longitude((179.0, 0.0, -179.0, 5.0)) == 180.0
    assert bbox_mid_longitude((10.0, 0.0, 20.0, 5.0)) == 15.0
    assert solar_day_offset_seconds(bbox_mid_longitude((179.0, 0.0, -179.0, 5.0))) == 12 * 3600


def test_a_real_noon_acquisition_still_gets_its_offset() -> None:
    """Noon UTC is a real acquisition time as well as our canonical stamp.

    Idempotence was briefly obtained by treating a noon stamp as "already normalised", which
    is a heuristic over a value the catalogue can legitimately produce: an item acquired at
    exactly 12:00:00.000000 skipped its offset and landed on the wrong solar day — at the
    dateline, where the offset is a whole day, which is precisely where it matters most.

    The day is now computed from the ACQUISITION instant in `properties["datetime"]`, which
    normalisation never modifies, so re-normalising recomputes the same day from the same
    input. Idempotence by construction rather than by self-detection.
    """
    raw = "2025-06-01T12:00:00Z"
    item = SimpleNamespace(
        datetime=datetime.datetime(2025, 6, 1, 12, 0, tzinfo=datetime.UTC), properties={"datetime": raw}
    )

    # +180 puts a 12:00Z acquisition on the FOLLOWING solar day.
    (out,) = normalize_to_solar_day([item], mid_longitude=180.0)
    assert out.datetime.date() == datetime.date(2025, 6, 2), "the offset must be applied to a real noon item"

    # ...and stays there, however many times the streamed path re-normalises it.
    for _ in range(3):
        (out,) = normalize_to_solar_day([out], mid_longitude=180.0)
        assert out.datetime.date() == datetime.date(2025, 6, 2)
    assert out.properties["datetime"] == raw, "the acquisition instant must survive untouched"


def test_an_already_canonical_stamp_is_left_alone() -> None:
    """The mechanism idempotence rests on, asserted directly rather than through an offset.

    Noon UTC is this package's canonical stamp — `solar_day_of` refuses anything else — so
    an item carrying it has been normalised already and its date IS its solar day. Adding
    the offset a second time is what used to move it, and at +/-12 h that crossed midnight.
    """

    class _Item:
        def __init__(self) -> None:
            self.datetime = datetime.datetime(2025, 6, 1, 12, 0, tzinfo=datetime.UTC)

    for longitude in (180.0, -180.0, 150.0, 0.0):
        item = _Item()
        (out,) = normalize_to_solar_day([item], mid_longitude=longitude)
        assert out.datetime == datetime.datetime(2025, 6, 1, 12, 0, tzinfo=datetime.UTC), longitude


@pytest.mark.parametrize("longitude", _HOSTILE_LONGITUDES)
@pytest.mark.parametrize("batch_days", [1, 7, 30])
def test_every_owned_solar_day_is_wholly_inside_the_queried_range(longitude: float, batch_days: int) -> None:
    """The property whose absence truncated a solar day at every batch boundary.

    A solar day is a 24-hour window in UTC offset by the region's longitude. For the
    owning chunk to write it complete, every instant of it must fall inside the UTC
    range that chunk queried. Checked at the longitudes where the offset is largest and
    where UTC midnight lands on a satellite pass — the two ways this goes wrong.
    """
    offset = datetime.timedelta(seconds=solar_day_offset_seconds(longitude))

    for rng in fixed_day_ranges("2024-01-01", "2024-03-31", batch_days):
        # The queried range is inclusive of whole UTC days.
        q0 = datetime.datetime.fromisoformat(rng.query_start)
        q1 = datetime.datetime.fromisoformat(rng.query_end) + datetime.timedelta(days=1)
        for day in _dates(rng.own_start, rng.own_end):
            # Solar day D spans the UTC interval [D - offset, D + 1day - offset).
            d0 = datetime.datetime.fromisoformat(day) - offset
            d1 = d0 + datetime.timedelta(days=1)
            assert q0 <= d0 and d1 <= q1, (
                f"solar day {day} at longitude {longitude} spans UTC {d0}..{d1}, "
                f"outside its owner's query {rng.query_start}..{rng.query_end}"
            )


@pytest.mark.parametrize("longitude", _HOSTILE_LONGITUDES)
def test_a_straddling_day_is_owned_by_exactly_one_batch_and_owned_whole(longitude: float) -> None:
    """The end-to-end statement, in the terms the failure was reported in.

    Every acquisition of a solar day goes to ONE chunk, and that chunk gets all of them.
    Before ownership existed, the acquisitions on either side of a UTC cut went to
    different batches and the second batch's were discarded as an already-written date.
    """
    ranges = fixed_day_ranges("2024-01-01", "2024-02-29", 30)
    # An acquisition every three hours across the whole window: whatever the offset,
    # some of them land either side of every cut.
    items = [
        _Item(datetime.datetime.fromisoformat(d) + datetime.timedelta(hours=h))
        for d in _dates("2023-12-31", "2024-03-01")
        for h in range(0, 24, 3)
    ]

    normalize_to_solar_day(items, mid_longitude=longitude)
    claimed: dict[str, set[str]] = {}
    for rng in ranges:
        for it in owned_items(items, rng):
            claimed.setdefault(solar_day_of(it), set()).add(rng.own_start)

    multi = {day: owners for day, owners in claimed.items() if len(owners) > 1}
    assert not multi, f"a solar day was claimed by more than one batch: {multi}"

    # ...and every day the window owns got all eight of its acquisitions.
    for day in _dates("2024-01-01", "2024-02-29"):
        got = [it for it in items if solar_day_of(it) == day]
        assert len(got) == 8, f"{day} at longitude {longitude} has {len(got)} acquisitions in the fixture"
        assert day in claimed, f"{day} was owned by no batch at all"


def test_items_outside_the_window_are_owned_by_nobody() -> None:
    """The pad reaches beyond the window; ownership must not follow it there."""
    ranges = fixed_day_ranges("2024-02-01", "2024-02-29", 7)
    items = [_Item(datetime.datetime.fromisoformat(f"{d}T12:00:00")) for d in _dates("2024-01-20", "2024-03-10")]

    normalize_to_solar_day(items, mid_longitude=0.0)
    claimed = {solar_day_of(it) for rng in ranges for it in owned_items(items, rng)}
    assert claimed == set(_dates("2024-02-01", "2024-02-29"))


# --- small surface --------------------------------------------------------------------


def test_owns_is_inclusive_at_both_ends() -> None:
    rng = SolarDayRange(own_start="2024-03-01", own_end="2024-03-31", query_start="2024-02-29", query_end="2024-04-01")
    assert rng.owns("2024-03-01")
    assert rng.owns("2024-03-31")
    assert not rng.owns("2024-02-29")
    assert not rng.owns("2024-04-01")


def test_a_reversed_window_raises_rather_than_yielding_nothing() -> None:
    """Silently producing no ranges would ingest nothing and report success."""
    for call in (
        lambda: month_ranges("2024-03-31", "2024-01-01"),
        lambda: fixed_day_ranges("2024-03-31", "2024-01-01", 30),
        lambda: whole_window_range("2024-03-31", "2024-01-01"),
    ):
        with pytest.raises(ValueError, match="precedes"):
            call()


def test_a_non_positive_batch_length_raises() -> None:
    with pytest.raises(ValueError, match="days must be >= 1"):
        fixed_day_ranges("2024-01-01", "2024-01-31", 0)


def test_no_longitude_falls_back_to_the_utc_date() -> None:
    """A caller not grouping by solar day must be left alone, not silently shifted."""
    raw = "2024-01-01T23:00:00"
    assert (
        solar_day_of(normalize_to_solar_day([_Item(datetime.datetime.fromisoformat(raw))], mid_longitude=None)[0])
        == "2024-01-01"
    )
    assert (
        solar_day_of(normalize_to_solar_day([_Item(datetime.datetime.fromisoformat(raw))], mid_longitude=99.0)[0])
        == "2024-01-02"
    )


def test_normalisation_is_idempotent() -> None:
    """The property the defensive re-normalisation at each consumption point relies on.

    `stream_stac_months` and `has_new_stac_dates` both normalise again rather than trust
    whoever supplied their items. That is only safe because a second pass recomputes the
    same solar day — noon plus any offset the grid produces stays inside the same day.
    """
    for lon in (-179.0, -99.0, 0.0, 99.0, 179.0):
        for hour in (0, 1, 11, 12, 13, 23):
            item = _Item(datetime.datetime(2024, 1, 15, hour, 30))
            once = normalize_to_solar_day([item], mid_longitude=lon)[0].datetime
            twice = normalize_to_solar_day([item], mid_longitude=lon)[0].datetime
            assert once == twice, f"lon {lon}, hour {hour}: {once} != {twice}"


def test_deriving_a_date_from_an_unnormalised_item_raises() -> None:
    """The guard that stops the whole convention degrading into a convention.

    A raw item's UTC date is a *plausible-looking wrong answer*: identical to the solar day
    at central longitudes, wrong only where the offset crosses midnight. So a path that
    skips the chokepoint produces correct output everywhere anyone would notice and
    mislabelled mosaics in the far east and far west. Failing loudly is the only way that
    class of bug surfaces before the data does.
    """
    raw = _Item(datetime.datetime(2024, 1, 15, 23, 30))
    with pytest.raises(ValueError, match="not the canonical noon-UTC solar-day timestamp"):
        solar_day_of(raw)

    # ...and it is happy the moment the item has been through the chokepoint.
    assert solar_day_of(normalize_to_solar_day([raw], mid_longitude=150.0)[0]) == "2024-01-16"


def test_owned_items_inherits_the_guard() -> None:
    """Ownership is the busiest consumer, so it must not be the one that degrades quietly."""
    rng = whole_window_range("2024-01-01", "2024-01-31")
    with pytest.raises(ValueError, match="normalize_to_solar_day"):
        owned_items([_Item(datetime.datetime(2024, 1, 15, 23, 30))], rng)


def test_a_resume_starts_on_the_frontiers_month_boundary() -> None:
    """The floor is the frontier's MONTH, and the reason is the query padding.

    A solar day straddles the UTC boundary, so a day's query is padded either side. Starting a
    resume at ``frontier + 1 day`` would query the first owned day on an unpadded bound and
    write it short — the same defect the batch slicing exists to prevent, reintroduced by the
    resume. A month boundary keeps every owned day's padding inside the query.
    """
    assert resume_window_start("2018-01-01", "2018-05-27") == "2018-05-01"
    assert resume_window_start("2018-01-01", "2018-05-01") == "2018-05-01"
    assert resume_window_start("2018-01-01", "2018-12-31") == "2018-12-01"


def test_a_floor_never_moves_a_window_start_backwards() -> None:
    """A frontier below the window says nothing about months the window never asked for.

    An empty store has no frontier at all, and a store whose frontier predates the window is
    being extended forwards — in both cases the leg starts where it was told to.
    """
    assert resume_window_start("2018-06-01", None) == "2018-06-01"
    assert resume_window_start("2018-06-01", "2018-03-14") == "2018-06-01"
