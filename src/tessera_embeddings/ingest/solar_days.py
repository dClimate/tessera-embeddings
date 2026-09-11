"""Solar-day ownership for catalogue queries that are bounded in UTC.

Every ingest in this package has the same shape: slice a date window into chunks, ask a
catalogue for each chunk, then let the loader group the returned images into **solar days**.
The window and the chunks are expressed in solar days while the query is bounded in **UTC**,
and the two disagree wherever a region's solar offset crosses UTC midnight — the far-eastern
and far-western zones.

Getting that wrong is silent, which is why it belongs in one module rather than being
re-derived per provider. Both spellings of the mistake have been in this codebase:

* **Query the chunk's own range and write everything you get.** A solar day straddling the cut
  is split. The earlier chunk writes it from its half; the later chunk's half is then dropped
  as an already-written date, so the day lands looking complete and is missing acquisitions.
  This is what the S1 batch loop did at every batch boundary.
* **Pad the query but clamp the pad to the window.** The padding then vanishes at the window's
  own two edges, so the first and last solar day of the run lose the acquisitions dated the
  adjacent UTC day. This is what the S2 month slicing did.

The fix is one idea and it is the whole content of this module: a chunk **owns** a range of
solar days and **queries** a wider range of UTC dates. Owned ranges tile the window exactly, so
nothing is processed twice and nothing outside the window is written — the ownership filter is
what guarantees that, never the query bound, which cannot see solar days at all. The query is
padded a day either side and is NOT clamped, because a solar day owned at the very edge of the
window still draws on the UTC day beyond it. One day of padding is always enough: the offset is
a whole number of hours in ``[-12, +12]``, so solar day ``D`` lies entirely within UTC
``[D-1, D+1]``.

Adding a provider means adding a span producer, not re-deriving any of the above.

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

    Mirrors ``odc.stac``'s own conversion exactly, including the truncation to whole hours.
    Matching it matters more than astronomical precision: the two must agree on which day an
    acquisition belongs to, and any divergence reappears as a date group that loads with more
    time slices than the caller grouped for.

    That agreement is load-bearing for :func:`owned_items`, which decides which images the
    loader is allowed to see. If our offset and the loader's disagreed at a 15-degree boundary,
    an image could be filtered out as another chunk's while the loader would have grouped it
    into this one — dropped from the run entirely rather than merely mislabelled.

    **The full +/-12 h at the dateline, deliberately, and NOT clamped to +/-11 h.** At exactly
    +180 the true offset IS +12 h, and clamping misplaces one hour of every day (12:00-12:59 UTC
    stays on the current date instead of advancing) — which `bbox_mid_longitude` then feeds every
    antimeridian-crossing box into. :func:`normalize_to_solar_day` stays idempotent without any
    clamp because it derives the day from the unmodified acquisition instant; see there.
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

    Refusing is the point. ``normalize_to_solar_day`` treats ``mid_longitude=None`` as "group by
    UTC date", and ``_load_from_stac`` rejects every grouping mode except ``solar_day``, so no
    caller legitimately wants that. What the default produces instead is every item stamped
    silently to noon of its UTC date: correct in the middle of a UTC day, wrong by a whole day
    for an acquisition near midnight in the far east or far west. That reaches the store as a
    mislabelled or split mosaic with nothing reporting a problem.

    So the longitude comes from the caller, else from the query's own ``bbox``, and if neither
    exists the call is refused: a path that cannot say where on Earth it is reading cannot say
    which day it read.
    """
    if mid_longitude is not None:
        return float(mid_longitude)
    # Geobox before bbox, for the reason `solar_grouping_longitude` gives: the loader shifts by
    # its geobox extent's centroid, so taking the same source makes the two agree by
    # construction.
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

    The loader shifts every item by ONE longitude — its geobox extent's centroid in WGS84 — so
    preferring the geobox makes the two agree by construction. Both must land on the same
    whole-hour offset, and a bbox midpoint can differ from an extent centroid by enough to cross
    a 15-degree boundary.

    Falls back to the WGS84 bbox midpoint, then to ``None``, which restores UTC-date grouping.
    Degrading rather than raising is deliberate: a missing geobox is a caller that is not loading
    by solar day, and no longitude is recoverable from nothing.
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
    dependency runs only one way. Both read the same field because it is the only record of the
    acquisition that survives the canonical stamp.
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

    **THIS IS THE ONLY PLACE IN THE PACKAGE THAT APPLIES THE SOLAR OFFSET.** Call it once, as
    early as possible — immediately after the catalogue query — and from then on an item's
    ``datetime`` *is* its solar day: every downstream date derivation is a plain
    ``strftime("%Y-%m-%d")`` with no offset. When the offset was applied independently at six
    sites, two of them disagreed with the rest — the cloud pre-sort and the baseline map both
    keyed on the UTC date while the loader grouped by solar day. Applying it once and never
    again cannot drift.

    **Noon, specifically.** The canonical timestamp has to survive two different readings:
    ``item.datetime.strftime`` must give the solar day, and so must the loaded slice's coordinate
    after ``odc.stac.load`` groups on it. Noon leaves the most margin — half a day either side of
    a midnight — against offsets that reach ±12 h at the dateline.

    Grouping identical timestamps is also what makes ``odc.stac.load`` mosaic same-day granules
    into ONE time slice instead of one per granule — the original reason this existed for OPERA's
    burst products, now serving both sensors.

    **Idempotent**, which is what lets the consumption points call it defensively rather than
    trusting whoever supplied the items: the solar day is recomputed from the acquisition instant
    in ``properties["datetime"]``, which normalisation never writes, so the answer does not
    depend on how many times this has already run.

    ``mid_longitude`` of ``None`` groups by UTC date instead, for callers that genuinely
    are not working in solar days.
    """
    offset = datetime.timedelta(seconds=solar_day_offset_seconds(mid_longitude) if mid_longitude is not None else 0)
    groups: dict[datetime.date, list[Any]] = {}
    for item in items:
        # THE SOLAR DAY IS COMPUTED FROM THE ACQUISITION INSTANT, which survives normalisation in
        # `properties["datetime"]` — assigning `item.datetime` does not write through to it
        # (pinned against real pystac in `TestAgainstRealPystacItems`). Deriving the day from an
        # input this function never modifies is what makes it idempotent for ANY offset, by
        # construction rather than by detecting its own past work.
        #
        # NOT by treating a noon-UTC stamp as "already normalised": that is a real acquisition
        # time as well as our canonical one, so an item genuinely acquired at 12:00:00.000000 is
        # mistaken for normalised and skips its offset — landing on the wrong solar day exactly
        # where the offset matters most, at the dateline. The fallback below wears that risk only
        # because it has nothing else to read.
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

    No offset is applied here, deliberately — the offset is applied exactly once, at the
    catalogue chokepoint; see :func:`normalize_to_solar_day`.

    **Raises rather than guessing** if the item is not normalised. A raw item's UTC date is a
    plausible-looking wrong answer: right at central longitudes and wrong only where the offset
    crosses midnight, so the failure hides in exactly the zones nobody tests interactively. The
    canonical stamp is noon UTC, so anything else means the item reached here without passing the
    chokepoint — an ordering bug, worth failing loudly on because the alternative is a silently
    mislabelled mosaic.
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

    ``own_start``/``own_end`` are inclusive solar days and decide whether an image is processed
    here. ``query_start``/``query_end`` are inclusive UTC dates and are only ever handed to a
    catalogue. Keeping them apart is the entire point of this type — see the module docstring for
    what happens each time they are conflated.
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

    The pad is deliberately NOT clamped to the spans' collective extent. Clamping it was the S2
    bug: at the first and last chunk the padding disappeared and the edge solar days lost
    whatever imagery was dated the adjacent UTC day. Nothing out-of-window escapes as a result,
    because the owned spans gate writing and they are unchanged.
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


def validated_window(start_date: str, end_date: str) -> tuple[datetime.date, datetime.date]:
    """The two bounds as dates, refusing anything that is not a real window.

    Public because the range builders below are not the only callers that need it. A resumed
    ingest decides whether it has anything to do BEFORE it builds any range, and it decides by
    comparing ``YYYY-MM-DD`` strings — exact for well-formed dates and silently wrong otherwise,
    since ``"2024-02-30"`` sorts like a real date and is not one. Without this call first, a leg
    handed one is reported as a successful skip rather than refused.

    **A caller that goes on to compare strings must take them from the returned dates**, not from
    what it was handed. ``date.fromisoformat`` accepts every ISO spelling, including the compact
    ``"20180101"``, and those do not sort with the canonical form: ``"20180101"`` orders ABOVE
    ``"2018-12-31"``, because ``"0"`` exceeds ``"-"``, so a window spelled that way reads as
    entirely behind its own end. Returning dates rather than strings is deliberate for the same
    reason.
    """
    start = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    if end < start:
        raise ValueError(f"end_date {end_date} precedes start_date {start_date}")
    return start, end


def month_ranges(start_date: str, end_date: str) -> list[SolarDayRange]:
    """One range per calendar month intersecting ``[start_date, end_date]``.

    Used by the S2 reflectance ingest, which streams a month at a time to bound item retention:
    a zone-year is hundreds of thousands of items and holding them all exhausts the worker long
    before the first date is written.

    Owned ranges PARTITION the window — every calendar date belongs to exactly one month — which
    is what makes the slices independent. No cross-month state is needed to deduplicate, and that
    matters because the worker can be restarted at any point.
    """
    start, end = validated_window(start_date, end_date)
    spans: list[tuple[datetime.date, datetime.date]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        last_day = calendar.monthrange(year, month)[1]
        spans.append((max(start, datetime.date(year, month, 1)), min(end, datetime.date(year, month, last_day))))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return _padded(spans)


def validated_batch_days(days: int) -> int:
    """The batch width, refusing one that cannot partition anything.

    Public for the same reason :func:`validated_window` is: a resumed ingest can find its whole
    window already closed and build no range at all, so a rule living only in
    :func:`fixed_day_ranges` would let the same invalid configuration raise over a partial store
    and report a successful skip over a complete one. The caller asks first; the rule and its
    message stay here so there is only one of each.
    """
    if days < 1:
        raise ValueError(f"days must be >= 1, got {days}")
    return days


def fixed_day_ranges(start_date: str, end_date: str, days: int) -> list[SolarDayRange]:
    """One range per ``days``-long run of solar days across ``[start_date, end_date]``.

    Used by the S1 SAR ingest, which walks the window in batches to keep each Dask graph
    manageable rather than to bound item retention.

    The spans are runs of SOLAR days, not UTC dates: cutting on UTC dates truncates any solar day
    landing on a cut (the first failure in the module docstring).
    """
    validated_batch_days(days)
    start, end = validated_window(start_date, end_date)
    spans: list[tuple[datetime.date, datetime.date]] = []
    span_start = start
    while span_start <= end:
        span_end = min(span_start + datetime.timedelta(days=days - 1), end)
        spans.append((span_start, span_end))
        span_start = span_end + datetime.timedelta(days=1)
    return _padded(spans)


def whole_window_range(start_date: str, end_date: str) -> SolarDayRange:
    """A single range owning the entire window.

    For callers that query the window in one shot — the S2 path with monthly streaming off. It
    needs the padding for the same reason the sliced paths do: without it the first and last
    solar day of the run are queried on a UTC bound that excludes part of their imagery.
    """
    start, end = validated_window(start_date, end_date)
    return _padded([(start, end)])[0]


def owned_items(items: list[Any], rng: SolarDayRange) -> list[Any]:
    """The subset of ``items`` whose solar day ``rng`` owns.

    Apply this between the catalogue query and the loader, NEVER after loading. Filtered here,
    the loader builds a group only for an owned day and that group holds every image of it,
    because the query reached a day either side. Filtered after loading, a straddling day has
    already been split into two partial groups and the information needed to rejoin them is gone.
    """
    return [it for it in items if rng.owns(solar_day_of(it))]


def resume_window_start(start_date: str, last_written_date: str | None) -> str:
    """Where a run should begin, given what the store already holds.

    **A store's dates can only be added in order, newest last.** Slotting one into the middle
    would mean shifting every chunk after it, and a Zarr store's chunks sit at fixed positions.
    So every day at or before the newest date a store holds is closed to it for good.

    Two things follow. Searching the catalogue back there cannot write anything, and searching is
    most of what a resumed run does, so a run over a mostly-full store spends nearly all its time
    on months it cannot write to. And a day down there must never reach the writer: the append is
    refused and the refusal is fatal, which is what wedged real stores.

    Starting the day AFTER the newest held date, rather than at the start of its month, makes both
    true at once — a month start still offers the earlier days of that month, which is exactly
    where an old gap sits.

    Nothing is lost to the tighter bound: the query is padded a day either side of what a run OWNS
    (see :func:`_padded`), which catches a solar day's imagery carried on the adjacent UTC date,
    and the owned span that gates writing starts the day after.

    Each store works this out for itself: a cell's three child stores advance independently, so a
    start computed once and shared would skip days a lagging store never reached.
    """
    if last_written_date is None:
        return start_date
    day_after = datetime.date.fromisoformat(last_written_date[:10]) + datetime.timedelta(days=1)
    return max(start_date, day_after.isoformat())
