"""Solar-day ownership for catalogue queries that are bounded in UTC.

Every ingest in this package has the same shape: slice a date window into chunks, ask
a catalogue for each chunk, then let the loader group the returned images into **solar
days**. The catch is that the window and the chunks are expressed in solar days while
the query is bounded in **UTC**, and the two disagree wherever a region's solar offset
crosses UTC midnight — the far-eastern and far-western zones.

Getting that wrong is silent, which is why it belongs in one module instead of being
re-derived per provider. Both spellings of the mistake have been in this codebase:

* **Query the chunk's own range and write everything you get.** A solar day straddling
  the cut is split. The earlier chunk writes it from its half; the later chunk's half
  is then dropped as an already-written date, so the day lands looking complete and is
  missing acquisitions. This is what the S1 batch loop did at every batch boundary.
* **Pad the query but clamp the pad to the window.** The padding then vanishes at the
  window's own two edges, so the first and last solar day of the run lose the
  acquisitions dated the adjacent UTC day. This is what the S2 month slicing did.

The fix is one idea, and it is the whole content of this module: a chunk **owns** a
range of solar days and **queries** a wider range of UTC dates. Owned ranges tile the
window exactly, so nothing is processed twice and nothing outside the window is written
— the ownership filter is what guarantees that, never the query bound, which cannot see
solar days at all. The query is padded a day either side and is NOT clamped, because a
solar day owned at the very edge of the window still draws on the UTC day beyond it.

One day of padding is always enough: the offset is a whole number of hours in
``[-12, +12]``, so solar day ``D`` lies entirely within UTC ``[D-1, D+1]``.

Adding a provider means adding a span producer. It does not mean re-deriving any of
the above.

    ranges = fixed_day_ranges("2024-01-01", "2024-12-31", 30)
    for rng in ranges:
        items = normalize_to_solar_day(catalogue_query(rng.query_start, rng.query_end), mid_longitude=lon)
        mine = owned_items(items, rng)                      # hand only these to the loader

Filtering the ITEMS, before the loader sees them, is what makes this work for both
paths at once. The loader then builds a group only for a day the chunk owns, and every
day it owns arrives whole because the query reached a day either side.
"""

from __future__ import annotations

import calendar
import datetime
from dataclasses import dataclass
from typing import Any, final

__all__ = [
    "SolarDayRange",
    "bbox_mid_longitude",
    "fixed_day_ranges",
    "month_ranges",
    "normalize_to_solar_day",
    "owned_items",
    "resolve_grouping_longitude",
    "resume_window_start",
    "solar_day_of",
    "solar_day_offset_seconds",
    "solar_grouping_longitude",
    "whole_window_range",
]

#: Padding applied to each side of a chunk's query range, in days. One is sufficient
#: and two would only widen the catalogue call: the solar offset is a whole number of
#: hours no larger than 12, so a solar day never reaches further than the adjacent UTC
#: day in either direction.
_QUERY_PAD = datetime.timedelta(days=1)


def solar_day_offset_seconds(mid_longitude: float) -> int:
    """UTC-to-solar-day offset for a longitude, in whole hours.

    Mirrors ``odc.stac``'s own conversion exactly, including the truncation to whole
    hours. Matching it matters more than being astronomically precise: the two must
    agree on which day an acquisition belongs to, and any divergence reappears as a
    date group that loads with more time slices than the caller grouped for.

    This agreement is load-bearing for :func:`owned_items` in particular. That filter
    decides which images the loader is allowed to see, so if our offset and the
    loader's ever disagreed at a 15-degree boundary, an image could be filtered out as
    another chunk's while the loader would have grouped it into this one — dropping it
    from the run entirely rather than merely mislabelling it.

    **The full +/-12 h at the dateline, deliberately.** This was briefly clamped to +/-11 h
    to keep :func:`normalize_to_solar_day` idempotent — noon plus twelve hours is midnight of
    the NEXT day, so a canonical stamp advanced a day on every re-normalisation, and the
    streamed S2 path normalises defensively three times over. That bought idempotence with a
    wrong answer: at exactly +180 the true offset IS +12 h, and clamping misplaces one hour
    of every day (12:00-12:59 UTC stays on the current date instead of advancing), which
    `bbox_mid_longitude` then feeds every antimeridian-crossing box into.

    Idempotence is handled where it belongs instead — `normalize_to_solar_day` recognises an
    already-canonical stamp and leaves it alone, which holds for ANY offset rather than only
    for offsets small enough not to cross midnight.
    """
    return int(mid_longitude / 15) * 3600


def bbox_mid_longitude(bbox: object) -> float | None:
    """The mid-longitude of a WGS84 ``(west, south, east, north)`` box, or ``None``.

    Handles the antimeridian case, which a plain average gets badly wrong: the GeoJSON /
    STAC convention for a box crossing 180 is ``west > east`` (e.g. 179 to -179), and
    averaging those gives 0 — the middle of the wrong hemisphere, half a planet from the
    box, and a solar-day offset ~12 hours out. Such a box is walked EASTWARD across the
    dateline instead, then wrapped back into range.
    """
    if bbox is None or not isinstance(bbox, (list, tuple)) or len(bbox) < 3:
        return None
    west, east = float(bbox[0]), float(bbox[2])
    if west <= east:
        return (west + east) / 2.0
    span = (180.0 - west) + (east + 180.0)
    mid = west + span / 2.0
    return mid - 360.0 if mid > 180.0 else mid


def resolve_grouping_longitude(mid_longitude: float | None, bbox: object, geobox: object = None) -> float:
    """The longitude to stamp solar days with. RAISES rather than degrading to UTC.

    Refusing is the point. ``normalize_to_solar_day`` treats ``mid_longitude=None`` as
    "group by UTC date", which was a sound default while a caller could legitimately load
    by UTC — but ``_load_from_stac`` now rejects every grouping mode except ``solar_day``,
    so there is no such caller left. What the default produced instead was every item
    stamped to noon of its UTC date, silently: correct in the middle of a UTC day, and
    wrong by a whole day for an acquisition near midnight in the far east or far west,
    where the two calendars differ. That reaches the store as a mislabelled or split
    mosaic, with nothing anywhere reporting a problem.

    So the longitude is taken from the caller, else derived from the query's own ``bbox``,
    and if neither exists the call is refused. A path that cannot say where on Earth it is
    reading cannot say which day it read.
    """
    if mid_longitude is not None:
        return float(mid_longitude)
    # Geobox before bbox, for the reason `solar_grouping_longitude` gives: the loader shifts
    # by its geobox extent's centroid, so taking the same source makes the two agree by
    # construction, and a bbox midpoint can differ from an extent centroid by enough to
    # cross a 15-degree boundary.
    if geobox is not None:
        try:
            ((lon, _),) = geobox.extent.centroid.to_crs("epsg:4326").points  # type: ignore[attr-defined]
            return float(lon)
        except (AttributeError, TypeError, ValueError):
            pass
    derived = bbox_mid_longitude(bbox)
    if derived is not None:
        return derived
    msg = (
        "solar-day grouping needs a longitude: pass mid_longitude (the ROI geobox centroid's "
        "longitude in WGS84, via solar_grouping_longitude), or a geobox, or a WGS84 bbox to "
        "derive it from. "
        "Neither was given, and defaulting to UTC dates would stamp every item to noon of its "
        "UTC day — right in the middle of a day and a full day wrong near midnight in the far "
        "east or west, with nothing to show it happened."
    )
    raise ValueError(msg)


def solar_grouping_longitude(roi: object) -> float | None:
    """The longitude to group solar days by, matching the loader's own choice.

    The loader shifts every item by ONE longitude — its geobox extent's centroid in
    WGS84 — so preferring the geobox here makes the two agree by construction rather
    than by approximation. Both must land on the same whole-hour offset, and a bbox
    midpoint can differ from an extent centroid by enough to cross a 15-degree
    boundary and so disagree.

    Falls back to the WGS84 bbox midpoint, then to ``None``, which restores UTC-date
    grouping. Degrading rather than raising is deliberate: a missing geobox is a
    caller that is not loading by solar day, and no longitude is recoverable from
    nothing.
    """
    geobox = getattr(roi, "geobox", None)
    if geobox is not None:
        try:
            ((lon, _),) = geobox.extent.centroid.to_crs("epsg:4326").points
            return float(lon)
        except (AttributeError, TypeError, ValueError):
            pass
    return bbox_mid_longitude(getattr(roi, "bbox_wgs84", None))


def _acquisition_instant(item: Any) -> datetime.datetime | None:  # noqa: ANN401 — any STAC-like item
    """The raw acquisition instant from ``properties["datetime"]``, or ``None``.

    A local copy of :func:`~tessera_embeddings.ingest.duplicates.acquisition_instant`, which
    cannot be imported here: that module imports :func:`solar_day_of` from this one, so the
    dependency only runs one way. Both read the same field for the same reason — it is the only
    record of the acquisition that survives the canonical stamp.
    """
    properties = getattr(item, "properties", None)
    raw = properties.get("datetime") if isinstance(properties, dict) else None
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_to_solar_day(items: list[Any], *, mid_longitude: float | None) -> list[Any]:
    """Stamp every item with **noon UTC of the solar day it belongs to**. In place.

    **This is the only place in the package that applies the solar offset.** Call it once,
    as early as possible — immediately after the catalogue query — and from then on an
    item's ``datetime`` *is* its solar day: every downstream date derivation is a plain
    ``strftime("%Y-%m-%d")`` with no offset. That invariant is the point. When the offset
    was applied independently at six sites, two of them disagreed with the rest (the
    cloud pre-sort and the baseline map both keyed on the UTC date while the
    loader grouped by solar day), and adding a seventh application would have been one more
    chance to disagree. Applying it once and then never again cannot drift.

    **Noon, specifically.** The canonical timestamp has to survive two different readings:
    ``item.datetime.strftime`` must give the solar day, and so must the loaded slice's
    coordinate after ``odc.stac.load`` groups on it. Noon is the only choice with half a
    day of margin on both sides, so neither reading crosses a midnight for any offset the
    grid produces (the extreme is ±11 h, at the zones nearest the antimeridian).

    Grouping identical timestamps is also what makes ``odc.stac.load`` mosaic same-day
    granules into ONE time slice instead of one slice per granule — the original reason
    this existed for OPERA's burst products, now serving both sensors.

    **Idempotent**, which is what lets the consumption points below call it defensively
    rather than trusting whoever supplied the items: re-normalising an already-normalised
    item recomputes the same solar day, because noon plus any offset the grid produces
    stays inside the same day.

    ``mid_longitude`` of ``None`` groups by UTC date instead, for callers that genuinely
    are not working in solar days.
    """
    offset = datetime.timedelta(seconds=solar_day_offset_seconds(mid_longitude) if mid_longitude is not None else 0)
    groups: dict[datetime.date, list[Any]] = {}
    for item in items:
        # THE SOLAR DAY IS COMPUTED FROM THE ACQUISITION INSTANT, which survives normalisation
        # in `properties["datetime"]` — assigning `item.datetime` does not write through to it
        # (pinned against real pystac in `TestAgainstRealPystacItems`). Deriving the day from an
        # input this function never modifies is what makes it idempotent, by construction rather
        # than by detecting its own past work: re-normalising recomputes the same day from the
        # same instant, whatever the offset.
        #
        # This replaced a heuristic that treated a noon-UTC stamp as "already normalised". That
        # is a real acquisition time as well as our canonical one, so an item genuinely acquired
        # at 12:00:00.000000 was mistaken for normalised and skipped its offset — landing on the
        # wrong solar day exactly where the offset matters most, at the dateline.
        acquired = _acquisition_instant(item)
        if acquired is not None:
            day = (acquired + offset).date()
        else:
            # No usable property: an object that never came from a catalogue. Fall back to the
            # canonical-stamp reading, which is still right for anything this package produced.
            when = item.datetime
            is_canonical = (when.hour, when.minute, when.second, when.microsecond) == (12, 0, 0, 0)
            day = when.date() if is_canonical else (when + offset).date()
        groups.setdefault(day, []).append(item)
    for day, group in groups.items():
        canonical = datetime.datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=datetime.UTC)
        for item in group:
            item.datetime = canonical
    return items


def solar_day_of(item: Any) -> str:  # noqa: ANN401 — any STAC-like item
    """The ``YYYY-MM-DD`` solar day of an item ALREADY normalised by
    :func:`normalize_to_solar_day`.

    No offset is applied here, deliberately — see that function.

    **Raises rather than guessing** if the item is not normalised. A raw item's UTC date
    is a plausible-looking wrong answer: it is right at central longitudes and wrong only
    where the offset crosses midnight, so the failure hides in exactly the zones nobody
    tests interactively. The canonical stamp is noon UTC, so anything else means the item
    reached here without passing the chokepoint — an ordering bug, and one worth failing
    loudly on because the alternative is a silently mislabelled mosaic.
    """
    when = item.datetime
    if (when.hour, when.minute, when.second, when.microsecond) != (12, 0, 0, 0):
        raise ValueError(
            f"solar_day_of() received an item stamped {when!r}, which is not the canonical "
            "noon-UTC solar-day timestamp. Items must pass through normalize_to_solar_day() "
            "at the catalogue chokepoint before any date is derived from them — see this "
            "module's docstring for why the offset is applied exactly once."
        )
    return when.strftime("%Y-%m-%d")


@final
@dataclass(frozen=True)
class SolarDayRange:
    """One chunk of an ingest window: the solar days it owns, the UTC dates it asks for.

    ``own_start``/``own_end`` are inclusive solar days and are what decides whether an
    image is processed here. ``query_start``/``query_end`` are inclusive UTC dates and
    are only ever handed to a catalogue. Keeping them apart is the entire point of this
    type — see the module docstring for what happens each time they are conflated.
    """

    own_start: str
    own_end: str
    query_start: str
    query_end: str

    def owns(self, solar_day: str) -> bool:
        """Whether ``solar_day`` (``YYYY-MM-DD``) is this chunk's to process."""
        return self.own_start <= solar_day <= self.own_end


def _padded(spans: list[tuple[datetime.date, datetime.date]]) -> list[SolarDayRange]:
    """Turn owned spans into ranges, padding the query a day either side.

    The pad is deliberately NOT clamped to the spans' collective extent. Clamping it
    was the S2 bug: at the first and last chunk the padding disappeared, and the edge
    solar days lost whatever imagery was dated the adjacent UTC day. Nothing
    out-of-window escapes as a result, because the owned spans are what gate writing
    and they are unchanged.
    """
    return [
        SolarDayRange(
            own_start=own_start.isoformat(),
            own_end=own_end.isoformat(),
            query_start=(own_start - _QUERY_PAD).isoformat(),
            query_end=(own_end + _QUERY_PAD).isoformat(),
        )
        for own_start, own_end in spans
    ]


def _window(start_date: str, end_date: str) -> tuple[datetime.date, datetime.date]:
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} precedes start_date {start_date}")
    return start, end


def month_ranges(start_date: str, end_date: str) -> list[SolarDayRange]:
    """One range per calendar month intersecting ``[start_date, end_date]``.

    Used by the S2 reflectance ingest, which streams a month at a time so that item
    retention stays bounded: a zone-year is hundreds of thousands of items and holding
    them all exhausts the worker long before the first date is written.

    Owned ranges PARTITION the window — every calendar date belongs to exactly one
    month — which is what makes the slices independent. No cross-month state is needed
    to deduplicate, and that matters because the worker can be restarted at any point.
    """
    start, end = _window(start_date, end_date)
    spans: list[tuple[datetime.date, datetime.date]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last_day = calendar.monthrange(year, month)[1]
        spans.append((max(start, datetime.date(year, month, 1)), min(end, datetime.date(year, month, last_day))))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return _padded(spans)


def fixed_day_ranges(start_date: str, end_date: str, days: int) -> list[SolarDayRange]:
    """One range per ``days``-long run of solar days across ``[start_date, end_date]``.

    Used by the S1 SAR ingest, which walks the window in batches to keep each Dask
    graph manageable rather than to bound item retention.

    The spans are runs of SOLAR days, not UTC dates. That is the correction: the batch
    loop used to cut on UTC dates and write every group the loader produced, which
    truncated any solar day landing on a cut.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    start, end = _window(start_date, end_date)
    spans: list[tuple[datetime.date, datetime.date]] = []
    span_start = start
    while span_start <= end:
        span_end = min(span_start + datetime.timedelta(days=days - 1), end)
        spans.append((span_start, span_end))
        span_start = span_end + datetime.timedelta(days=1)
    return _padded(spans)


def resume_window_start(start_date: str, frontier: str | None) -> str:
    """Where a leg resuming over a store must begin: the first day of ``frontier``'s MONTH.

    ``frontier`` is the latest date already on that store's time axis, and the axis is
    append-only in order — so every date at or below it is unreachable and re-walking the
    months below it is pure cost. On a year-long window a resume near the end was re-querying
    and re-evaluating the whole year to write a handful of dates.

    The floor is the frontier's MONTH, not the frontier itself and not the day after it. A
    solar day straddles the UTC boundary, so the query for a day is padded either side
    (:func:`whole_window_range`, :func:`owned_items`) — start at ``frontier + 1 day`` and the
    first owned solar day is queried on a bound that excludes part of its imagery, and it is
    written short. Starting on a month boundary keeps the padding intact and costs at most
    the re-evaluation of the frontier's own month.

    Dropping the earlier months does NOT by itself make the resume safe: dates below the
    frontier inside its own month are still offered. The caller must drop those per date —
    the floor bounds the cost, the per-date check is the correctness.

    Each store has its OWN frontier. A cell's three child stores (reflectance and both radar
    orbits) advance independently, so a floor computed once and shared would silently skip
    months a lagging store never reached.
    """
    if frontier is None:
        return start_date
    return max(start_date, f"{frontier[:7]}-01")


def whole_window_range(start_date: str, end_date: str) -> SolarDayRange:
    """A single range owning the entire window.

    For callers that query the window in one shot — the S2 path with monthly streaming
    turned off. It needs the padding for exactly the same reason the sliced paths do:
    without it the first and last solar day of the run are queried on a UTC bound that
    excludes part of their imagery.
    """
    start, end = _window(start_date, end_date)
    return _padded([(start, end)])[0]


def owned_items(items: list[Any], rng: SolarDayRange) -> list[Any]:
    """The subset of ``items`` whose solar day ``rng`` owns.

    Apply this between the catalogue query and the loader, never after loading. Filtered
    here, the loader builds a group only for an owned day and that group holds every
    image of it, because the query reached a day either side. Filtered after loading, a
    straddling day has already been split into two partial groups and the information
    needed to rejoin them is gone.
    """
    return [it for it in items if rng.owns(solar_day_of(it))]
