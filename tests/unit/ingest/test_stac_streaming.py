"""Month-by-month STAC query streaming (pure, offline).

Querying a whole window up front retains every item for the run's duration, which a
zone-year cannot fit on the worker the ingest runs on. Streaming bounds retention to two
months. These tests pin the two properties the correctness of that rests on — the owned
ranges PARTITION the window, and the prefetch is depth-1 — plus the failure behaviour,
because a silently-dropped month would be an incomplete mosaic that still reported success.
"""

from __future__ import annotations

import datetime
import gc
import threading
import weakref
from types import SimpleNamespace

import pytest

from tessera_embeddings.ingest.solar_days import month_ranges, normalize_to_solar_day
from tessera_embeddings.ingest.stac import (
    _filter_existing_dates,
    _prefetch,
    stream_stac_months,
)


def _item(date: str, cloud: float = 0.0):
    return SimpleNamespace(
        datetime=datetime.datetime.fromisoformat(f"{date}T10:00:00"), properties={"eo:cloud_cover": cloud}
    )


def _dates(start: str, end: str) -> list[str]:
    d, e = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    out = []
    while d <= e:
        out.append(d.isoformat())
        d += datetime.timedelta(days=1)
    return out


class TestMonthPartition:
    """The owned ranges must partition the window: no date twice, none missed."""

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            ("2024-01-01", "2024-12-31"),  # a full leap year
            ("2023-01-01", "2023-12-31"),  # a full non-leap year
            ("2024-01-20", "2024-02-05"),  # mid-month both ends
            ("2024-05-10", "2024-05-10"),  # one day
            ("2024-02-01", "2024-02-29"),  # exactly February, leap
            ("2024-11-15", "2025-02-03"),  # crossing a year boundary
        ],
    )
    def test_owned_ranges_partition_the_window(self, start, end):
        covered: list[str] = []
        for mr in month_ranges(start, end):
            covered.extend(_dates(mr.own_start, mr.own_end))
        assert covered == _dates(start, end), "owned dates must tile the window exactly once"

    def test_query_end_is_padded_by_one_day_including_at_the_window_end(self):
        """A date-only interval end covers that day, so without a pad an item in the
        final seconds of a month's last day could fall outside every slice's query.

        The LAST month is padded too. It used to be clamped to the window, on the
        reasoning that a query should not reach outside what was asked for — but the
        window is in solar days and the query is bounded in UTC, so at western
        longitudes the final owned solar day includes acquisitions dated the following
        UTC day. Clamping made those unfetchable by any slice and the last day of the
        year was written incomplete, with nothing to signal it.
        """
        months = month_ranges("2024-01-01", "2024-03-31")
        assert months[0].query_end == "2024-02-01"  # padded past its own end
        assert months[1].query_end == "2024-03-01"
        assert months[-1].query_end == "2024-04-01", "the last month must still reach the following UTC day"

    def test_query_start_is_padded_by_one_day_including_at_the_window_start(self):
        """Solar-day ownership can hand a month an item whose UTC date is the day
        before it. At eastern longitudes an acquisition late on a month's last UTC day
        has its SOLAR day in the next month — so the owning month must ask for that
        day, or no slice ever fetches the item.

        That applies to the FIRST month against the window's own start exactly as it
        applies between months; clamping it lost the first solar day's evening imagery.
        """
        months = month_ranges("2024-01-01", "2024-03-31")
        assert months[0].query_start == "2023-12-31", "the first month must still reach the preceding UTC day"
        assert months[1].query_start == "2024-01-31"
        assert months[2].query_start == "2024-02-29"

    def test_out_of_window_dates_are_excluded_by_ownership_not_by_the_query_bound(self):
        """What the unclamped pad relies on: owned ranges never leave the window.

        The query now reaches a day either side of the whole window, so the only thing
        stopping an out-of-window solar day from being written is the ownership filter.
        Pin that the owned ranges stay inside the requested window exactly.
        """
        months = month_ranges("2024-01-01", "2024-12-31")
        assert months[0].own_start == "2024-01-01"
        assert months[-1].own_end == "2024-12-31"
        assert months[0].query_start < months[0].own_start
        assert months[-1].query_end > months[-1].own_end

    def test_padding_cannot_double_process(self):
        """The padded day is OWNED by the next month, so overlap in the query does not
        become overlap in the work.
        """
        a, b = month_ranges("2024-01-01", "2024-02-29")[:2]
        assert a.query_end == b.own_start
        assert b.query_start <= a.own_end  # the pads overlap...
        assert a.own_end < b.own_start  # ...but the owned ranges never do

    def test_leap_day_is_owned(self):
        feb = next(mr for mr in month_ranges("2024-01-01", "2024-12-31") if mr.own_start == "2024-02-01")
        assert feb.own_end == "2024-02-29"

    def test_rejects_a_backwards_window(self):
        with pytest.raises(ValueError, match="precedes"):
            month_ranges("2024-03-01", "2024-01-01")


class TestStreaming:
    """What the generator yields, and what it does when a query fails."""

    def _fake_query(self, per_date: dict[str, int], calls: list | None = None):
        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox, **_):
            if calls is not None:
                calls.append((start_date, end_date, frozenset(existing_dates or ())))
            items = [
                _item(d)
                for d in _dates(start_date, end_date)
                for _ in range(per_date.get(d, 0))
                if d not in (existing_dates or set())
            ]
            return items, dict.fromkeys(per_date, 1)

        return query

    def _stream(self, per_date, start, end, existing=frozenset(), calls=None):
        return list(
            stream_stac_months(
                provider="p",
                collection="c",
                tile_id=None,
                start_date=start,
                end_date=end,
                bbox=(0, 0, 1, 1),
                existing_dates_fn=lambda: set(existing),
                query_fn=self._fake_query(per_date, calls),
            )
        )

    def test_items_outside_the_owned_month_are_dropped(self):
        """The query is padded, so it returns a day the next month owns. Yielding it here
        would process that date twice.
        """
        got = self._stream({"2024-01-31": 1, "2024-02-01": 1}, "2024-01-01", "2024-02-29")
        jan = next(m for m in got if m[0].own_start == "2024-01-01")
        assert [str(it.datetime)[:10] for it in jan[1]] == ["2024-01-31"]

    def test_every_date_is_yielded_exactly_once(self):
        per_date = dict.fromkeys(_dates("2024-01-01", "2024-03-31"), 1)
        seen = [
            str(it.datetime)[:10]
            for _mr, items, _b in self._stream(per_date, "2024-01-01", "2024-03-31")
            for it in items
        ]
        assert sorted(seen) == sorted(per_date), "streaming must neither drop nor duplicate a date"

    def test_empty_months_are_skipped_not_yielded(self):
        got = self._stream({"2024-03-05": 1}, "2024-01-01", "2024-03-31")
        assert [mr.own_start for mr, _i, _b in got] == ["2024-03-01"]

    def test_existing_dates_are_probed_per_month(self):
        """A later month's query must exclude what earlier months committed, so the probe
        cannot be hoisted out of the loop.
        """
        calls: list = []
        self._stream(dict.fromkeys(_dates("2024-01-01", "2024-03-31"), 1), "2024-01-01", "2024-03-31", calls=calls)
        assert len(calls) == 3

    def test_the_next_month_is_prefetched_before_the_current_is_consumed(self):
        """Depth-1 prefetch is the whole point: without it the fleet idles through every
        query. Observed by blocking inside the consumer and checking the next query ran.
        """
        second_started = threading.Event()
        queried: list[str] = []

        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox, **_):
            queried.append(start_date)
            if start_date == "2024-01-31":  # February's slice, padded a day early
                second_started.set()
            # Every date in the PADDED range, so each month still owns something after
            # the ownership filter — returning only the query's first date would hand
            # each month nothing but the pad day its neighbour owns.
            return [_item(d) for d in _dates(start_date, end_date)], {}

        gen = stream_stac_months(
            provider="p",
            collection="c",
            tile_id=None,
            start_date="2024-01-01",
            end_date="2024-02-29",
            bbox=None,
            existing_dates_fn=set,
            query_fn=query,
        )
        next(gen)  # consume January; February's query should already have been submitted
        assert second_started.wait(timeout=5), "February's query was never submitted"
        assert "2024-01-31" in queried  # February's query starts one padded day early
        list(gen)

    def test_a_failing_month_query_raises_rather_than_truncating(self):
        """A dropped month would be an incomplete mosaic that still reported success —
        the failure must reach the caller.
        """

        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox, **_):
            if start_date.startswith("2024-02"):
                raise RuntimeError("STAC exhausted its retries")
            return [_item(start_date)], {}

        gen = stream_stac_months(
            provider="p",
            collection="c",
            tile_id=None,
            start_date="2024-01-01",
            end_date="2024-03-31",
            bbox=None,
            existing_dates_fn=set,
            query_fn=query,
        )
        with pytest.raises(RuntimeError, match="exhausted"):
            list(gen)

    def test_a_single_day_window_streams(self):
        got = self._stream({"2024-05-10": 2}, "2024-05-10", "2024-05-10")
        assert len(got) == 1
        assert len(got[0][1]) == 2

    def test_no_items_at_all_yields_nothing(self):
        assert self._stream({}, "2024-01-01", "2024-02-29") == []


class TestRetentionIsBounded:
    """The whole point of streaming: a month's items must become collectable once the
    caller moves on, or a year accumulates and kills the worker exactly as before.

    Asserted by weak reference rather than by measuring RSS — a memory series can be
    misread from early samples, whereas reachability is exact.
    """

    class _Item:
        """A weak-referenceable stand-in; SimpleNamespace cannot be weakly referenced."""

        def __init__(self, date: str) -> None:
            self.datetime = datetime.datetime.fromisoformat(f"{date}T10:00:00")
            self.properties: dict = {}

    def _gen(self, months: int):
        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox, **_):
            # Dated at the MIDPOINT of the queried span, which is always inside the
            # slice's own range — both ends are padded into a neighbour's, so an item
            # dated at either one would be owned by that neighbour and this month would
            # yield nothing. Distinct objects per month, deliberately NOT retained by
            # the fixture: a fixture that held them would make this test vacuous.
            lo = datetime.date.fromisoformat(start_date)
            hi = datetime.date.fromisoformat(end_date)
            mid = (lo + (hi - lo) / 2).isoformat()
            return [TestRetentionIsBounded._Item(mid) for _ in range(3)], {}

        gen = stream_stac_months(
            provider="p",
            collection="c",
            tile_id=None,
            start_date="2024-01-01",
            end_date=f"2024-0{months}-28",
            bbox=None,
            existing_dates_fn=set,
            query_fn=query,
        )
        return gen

    def test_an_earlier_months_items_are_released_once_the_caller_advances(self):
        gen = self._gen(3)
        refs: list[weakref.ref] = []
        held = None
        for _mr, items, _b in gen:
            refs.append(weakref.ref(items[0]))
            held = items  # only the CURRENT month is kept, as the driver does
        del held
        gc.collect()
        # The last month may still be referenced by the loop machinery; earlier ones
        # must not be — that is the property that makes a year survivable.
        assert refs[0]() is None, "January's items were still reachable after streaming past them"
        assert refs[1]() is None, "February's items were still reachable after streaming past them"

    def test_accumulating_across_months_is_what_streaming_prevents(self):
        """The negative control: a caller that hoards every month defeats the bound, so
        this test would pass for the wrong reason without it.
        """
        gen = self._gen(3)
        hoarded = [items for _mr, items, _b in gen]
        gc.collect()
        assert all(h is not None for h in hoarded)
        assert len(hoarded) == 3, "the generator should have yielded three months"


def test_prefetch_runs_on_a_daemon_thread_so_it_can_be_abandoned():
    """An in-flight query nobody wants must not keep the interpreter alive.

    `cancel_futures=True` only drops futures that have not STARTED, and a
    ThreadPoolExecutor's workers are non-daemon and joined by an atexit hook — so
    abandoning a running catalog walk pinned the process (and, on the Prefect path,
    its per-run ECS task) for the query's full timeout-and-retry budget. There is no
    way to interrupt an in-flight request from outside, so abandonment is the
    mechanism, and a daemon thread is what makes abandonment free.
    """
    release = threading.Event()
    started = threading.Event()
    captured: list[threading.Thread] = []

    def slow(_arg):
        captured.append(threading.current_thread())
        started.set()
        release.wait(30)  # stands in for an HTTP walk with retries
        return "late"

    _prefetch(slow, None)  # started, then dropped on the floor
    assert started.wait(5)
    assert captured[0].daemon and captured[0].is_alive()  # nothing for atexit to join
    release.set()


def test_prefetch_reraises_in_the_caller_thread():
    """A failing prefetch must surface where the caller waits, not vanish."""

    def boom(_arg):
        raise RuntimeError("catalog exploded")

    pending = _prefetch(boom, None)
    with pytest.raises(RuntimeError, match="catalog exploded"):
        pending()


class TestSolarDayOwnership:
    """The month partition must agree with how the loader groups, or a day splits.

    S2 loads with ``groupby="solar_day"``, which shifts each timestamp by the ROI's
    longitude. Partitioning months on the UTC date instead puts the two halves of one
    solar day in different months, and each half is then driven as its own write:
    either a duplicate timestamp, or half the day's imagery. Only far-eastern and
    far-western ROIs are affected, which is why it survives ordinary testing.
    """

    @staticmethod
    def _months(mid_longitude):
        seen: dict[str, list[str]] = {}

        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox, **_):
            # One acquisition at 22:00 UTC on 31 January. At +150° the solar offset is
            # +10 h, so its solar day is 1 February.
            item = SimpleNamespace(datetime=datetime.datetime(2024, 1, 31, 22, 0), properties={})
            return ([item] if start_date <= "2024-01-31" <= end_date else []), {}

        for mr, items, _b in stream_stac_months(
            provider="p",
            collection="c",
            tile_id=None,
            start_date="2024-01-01",
            end_date="2024-02-29",
            bbox=None,
            existing_dates_fn=set,
            query_fn=query,
            mid_longitude=mid_longitude,
        ):
            seen.setdefault(mr.own_start, []).extend(str(i.datetime)[:10] for i in items)
        return seen

    def test_an_eastern_acquisition_is_owned_by_its_solar_month(self):
        """And carries its solar day afterwards — the two are now the same statement.

        A 31 January 22:00 UTC acquisition at +150° is 1 February locally. It is owned by
        February, and because ownership works by stamping the item with noon of its solar
        day (see solar_days.normalize_to_solar_day), the item's own timestamp reads
        2024-02-01 from here on. Everything downstream therefore gets the solar day from a
        plain strftime, with no offset to reapply and no chance to disagree.
        """
        assert self._months(150.0) == {"2024-02-01": ["2024-02-01"]}

    def test_the_same_acquisition_stays_in_january_without_a_longitude(self):
        """Omitting the longitude keeps UTC ownership — correct for callers that do
        not group by solar day, and the reason this cannot be applied unconditionally.
        """
        assert self._months(None) == {"2024-01-01": ["2024-01-31"]}  # unshifted

    def test_it_is_owned_exactly_once_either_way(self):
        for lon in (150.0, -150.0, 0.0, None):
            owners = [m for m, dates in self._months(lon).items() if dates]
            assert len(owners) == 1, f"longitude {lon} produced {owners}"

    def test_the_existing_date_filter_is_keyed_the_same_way(self):
        """Month ownership and the resume filter must agree, or a rerun half-filters.

        The store's dates are SOLAR days. Matching an item's UTC date against them keeps
        every item on the far side of midnight — the committed group's other half — which
        then loads, regroups onto the day already present, and is written a second time.
        """

        def at_150(hour: int = 22):
            it = SimpleNamespace(datetime=datetime.datetime(2024, 1, 31, hour, 0), properties={})
            return normalize_to_solar_day([it], mid_longitude=150.0)

        # Committed as 1 February (its solar day at +150°), so a resume must drop it.
        assert _filter_existing_dates(at_150(), {"2024-02-01"}) == []
        # ...and its UTC date being absent from the store is not a reason to keep it.
        assert len(_filter_existing_dates(at_150(), {"2024-01-31"})) == 1

    def test_the_filter_keeps_utc_keying_without_a_longitude(self):
        """The same reason ownership cannot shift unconditionally: a caller grouping by
        UTC date stores UTC dates, and shifting the filter would strand its resume.
        """
        item = SimpleNamespace(datetime=datetime.datetime(2024, 1, 31, 22, 0), properties={})
        assert _filter_existing_dates([item], {"2024-01-31"}) == []

    def test_the_longitude_reaches_the_query(self):
        """The filter runs inside the query, so ownership alone is not enough — the
        stream has to hand the same longitude down for the two to agree.
        """
        seen: list = []

        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox, **kw):
            seen.append(kw.get("mid_longitude"))
            return [], {}

        list(
            stream_stac_months(
                provider="p",
                collection="c",
                tile_id=None,
                start_date="2024-01-01",
                end_date="2024-01-31",
                bbox=None,
                existing_dates_fn=set,
                query_fn=query,
                mid_longitude=150.0,
            )
        )
        assert seen == [150.0]
