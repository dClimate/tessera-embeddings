"""A mosaic slice must be labelled with the SOLAR DAY it represents, not an item's timestamp.

The store's time axis is day-granular — it normalises to midnight — so the label decides only
WHICH day. Taking it from a member item gets that wrong wherever the solar offset crosses UTC
midnight: odc stamps each group with `group[0].nominal_datetime`, and because
`preserve_original_order=True` (needed so the clearest tile paints last) `group[0]` is the
CLOUDIEST item, whose acquisition time is arbitrary within the day. Its calendar date can be
the day BEFORE the solar day, so two consecutive solar days normalise onto one date.

That is not hypothetical. On zone 56N (+10 h) it blocked S2 ingestion outright, and in two
different ways depending on the path: the batched write rejected the dates as not strictly
increasing, and the unbatched write rejected the second day as a duplicate time slot. Both
failed CLOSED, which is why no corrupt data exists — but neither zone could be ingested.

Measured there: SIX of twenty-two solar days landed on the previous date, and which six
depended on cloud cover rather than on geography, so the error was inconsistent within one
zone-year rather than a uniform offset.
"""

from __future__ import annotations

import datetime as dt

from tessera_embeddings.ingest.solar_days import solar_day_offset_seconds
from tessera_embeddings.ingest.stac import group_items_by_date

#: 56N's centroid: 153 E, a +10 h offset, the zone where this actually fired.
FAR_EAST_LON = 153.0


class _Item:
    def __init__(self, iso: str, cloud: float = 50.0) -> None:
        self.datetime = dt.datetime.fromisoformat(iso).replace(tzinfo=dt.UTC)
        self.properties = {"eo:cloud_cover": cloud}


def _solar_day_of(item: _Item, lon: float) -> str:
    """The label the fix assigns: the item's own solar day."""
    off = dt.timedelta(seconds=solar_day_offset_seconds(lon))
    return (item.datetime + off).strftime("%Y-%m-%d")


def test_two_solar_days_can_share_a_utc_date_at_a_large_offset() -> None:
    """The precondition for the collision, stated as a fact about the data.

    At +10 h a solar day spans UTC 14:00 the previous day to 13:59. So an acquisition at
    23:45 UTC belongs to the NEXT solar day while carrying the current UTC date.
    """
    early, late = _Item("2024-01-20T00:52:00"), _Item("2024-01-20T23:45:00")
    assert _solar_day_of(early, FAR_EAST_LON) == "2024-01-20"
    assert _solar_day_of(late, FAR_EAST_LON) == "2024-01-21"
    # …yet both carry the same UTC calendar date, which is what collides.
    assert early.datetime.strftime("%Y-%m-%d") == late.datetime.strftime("%Y-%m-%d")


def test_the_label_must_not_come_from_the_cloudiest_item() -> None:
    """Reproduces the 56N failure exactly: two solar days, one written date.

    Both groups' cloudiest item sits on 2024-01-20 UTC, so labelling from it normalises both
    onto that date — the batched path then sees non-increasing dates and the unbatched path
    a duplicate slot. Labelling from the solar day keeps them distinct.
    """
    # Solar 01-20: cloudiest early in the UTC day. Solar 01-21: cloudiest late on 01-20 UTC.
    day_a = [_Item("2024-01-20T00:52:00", cloud=90.0), _Item("2024-01-20T01:10:00", cloud=5.0)]
    day_b = [_Item("2024-01-20T23:45:00", cloud=90.0), _Item("2024-01-21T01:10:00", cloud=5.0)]

    from_item = {items[0].datetime.strftime("%Y-%m-%d") for items in (day_a, day_b)}
    from_solar = {_solar_day_of(items[0], FAR_EAST_LON) for items in (day_a, day_b)}

    assert len(from_item) == 1, "the bug: two solar days collapse onto one date"
    assert from_item == {"2024-01-20"}
    assert from_solar == {"2024-01-20", "2024-01-21"}, "the fix: distinct days stay distinct"


def test_every_item_in_a_group_yields_the_same_solar_day() -> None:
    """Why taking it from ``day_items[0]`` is sound even though that item is arbitrary.

    The solar day is the grouping KEY, so every member shares it. That is what lets the label
    be derived from any one item instead of threading the key through the pipeline.
    """
    items = [_Item("2024-01-20T23:45:00"), _Item("2024-01-21T01:10:00"), _Item("2024-01-21T13:00:00")]
    groups = group_items_by_date(items, mid_longitude=FAR_EAST_LON)
    assert len(groups) == 1, groups
    assert {_solar_day_of(i, FAR_EAST_LON) for i in items} == {"2024-01-21"}


def test_labels_are_monotonic_across_consecutive_solar_days() -> None:
    """What makes the batched write's strictly-increasing requirement hold by construction.

    Consecutive solar days differ by exactly one day, so nothing needs sorting. Labels taken
    from items give no such guarantee.
    """
    days = ["2024-01-20T23:45:00", "2024-01-21T23:50:00", "2024-01-22T23:40:00"]
    labels = [_solar_day_of(_Item(d), FAR_EAST_LON) for d in days]
    assert labels == ["2024-01-21", "2024-01-22", "2024-01-23"]
    assert labels == sorted(labels) and len(set(labels)) == len(labels)


def test_mid_longitude_zones_are_unaffected() -> None:
    """Why this changes nothing for the great majority of zones.

    At a zero offset the solar day IS the UTC date, so the label is identical either way —
    which is why the bug hid until a far-eastern zone was attempted.
    """
    for lon in (0.0, 10.0, -10.0):
        item = _Item("2024-06-15T10:30:00")
        assert _solar_day_of(item, lon) == item.datetime.strftime("%Y-%m-%d")
