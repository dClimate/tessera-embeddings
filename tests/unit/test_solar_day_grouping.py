"""Item grouping must agree with the loader on which day an acquisition belongs to.

Regression cover for a failure seen only on far-eastern zones. The loader groups by LOCAL
solar day, shifting timestamps by the ROI's centroid longitude; this module used to group
by UTC calendar date. Where the solar offset is large enough to cross UTC midnight the two
disagree, so a group the caller believed was one day arrived as TWO time slices — against a
cloud mask reduced to a single slice, which surfaces as an xarray dimension conflict that
names nothing about the cause.

Zone 56N is the worked example: it spans 150-156 degrees east, an offset of +10 hours, and
Sentinel-2 images there at roughly 00:30 UTC — right on the boundary. Central-longitude
zones never hit it, which is why it stayed hidden.
"""

from __future__ import annotations

import datetime
from types import SimpleNamespace

import pytest

from tessera_embeddings.ingest.solar_days import solar_day_offset_seconds
from tessera_embeddings.ingest.stac import group_items_by_date

# +10 h: the offset for UTM zone 56 (150-156 E), where the failure was observed.
FAR_EAST_LON = 153.0
# Greenwich: no shift, so UTC and solar day coincide.
PRIME_LON = 0.0


def _item(iso: str, cloud: float = 0.0):
    return SimpleNamespace(
        datetime=datetime.datetime.fromisoformat(iso),
        properties={"eo:cloud_cover": cloud},
    )


def test_offset_matches_the_loaders_whole_hour_truncation() -> None:
    """The loader truncates to whole hours; being astronomically nicer would DISAGREE."""
    assert solar_day_offset_seconds(153.0) == 10 * 3600
    assert solar_day_offset_seconds(0.0) == 0
    assert solar_day_offset_seconds(-75.0) == -5 * 3600
    # 14.9 degrees is under one hour and must truncate to zero, not round up.
    assert solar_day_offset_seconds(14.9) == 0


def test_far_east_acquisitions_across_utc_midnight_are_one_solar_day() -> None:
    """23:30 and 00:30 UTC are the same local morning at +10 h — one group, not two.

    This is the case that broke: grouped by UTC date these are two days; the loader
    grouped them as one.
    """
    items = [_item("2026-01-05T23:30:00"), _item("2026-01-06T00:30:00")]
    groups = group_items_by_date(items, mid_longitude=FAR_EAST_LON)
    assert len(groups) == 1, groups
    assert sum(len(v) for v in groups.values()) == 2


def test_far_east_one_utc_date_can_hold_two_solar_days() -> None:
    """The converse, and the shape that actually produced the dimension conflict.

    00:30 and 23:30 on the same UTC date fall on different local days at +10 h, so a
    UTC-date group would hand the loader two solar days at once.
    """
    items = [_item("2026-01-06T00:30:00"), _item("2026-01-06T23:30:00")]
    assert len(group_items_by_date(items, mid_longitude=FAR_EAST_LON)) == 2
    # Grouped the OLD way they collapse to one — which is precisely the bug.
    assert len(group_items_by_date(items)) == 1


@pytest.mark.parametrize("lon", [PRIME_LON, 10.0, -10.0])
def test_central_longitudes_are_unaffected(lon: float) -> None:
    """Mid-longitude zones image far from UTC midnight, so nothing changes for them.

    Explains why this survived every earlier test: the zones exercised were central.
    """
    items = [_item("2026-01-06T10:30:00"), _item("2026-01-06T11:30:00")]
    assert len(group_items_by_date(items, mid_longitude=lon)) == 1


def test_omitting_longitude_keeps_utc_behaviour() -> None:
    """The parameter is opt-in, so callers that do not group by solar day are unchanged."""
    items = [_item("2026-01-06T00:30:00"), _item("2026-01-06T23:30:00")]
    assert list(group_items_by_date(items)) == ["2026-01-06"]


def test_within_group_order_is_preserved() -> None:
    """Order carries the painter's-algorithm contract: the clearest tile must land LAST.

    Grouping must not reorder, or the mosaic silently takes the cloudier pixel.
    """
    items = [_item("2026-01-06T00:30:00", cloud=90.0), _item("2026-01-06T00:40:00", cloud=5.0)]
    (group,) = group_items_by_date(items, mid_longitude=FAR_EAST_LON).values()
    assert [i.properties["eo:cloud_cover"] for i in group] == [90.0, 5.0]
