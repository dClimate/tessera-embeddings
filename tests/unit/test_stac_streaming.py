"""Month-by-month STAC query streaming (pure, offline).

Querying a whole window up front retains every item for the run's duration, which a
zone-year cannot fit on the worker the ingest runs on. Streaming bounds retention to two
months. These tests pin the two properties the correctness of that rests on — the owned
ranges PARTITION the window, and the prefetch is depth-1 — plus the failure behaviour,
because a silently-dropped month would be an incomplete mosaic that still reported success.
"""

from __future__ import annotations

import datetime
import threading
from types import SimpleNamespace

import pytest

from tessera_embeddings.ingest.stac import iter_month_ranges, stream_stac_months


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
        for mr in iter_month_ranges(start, end):
            covered.extend(_dates(mr.own_start, mr.own_end))
        assert covered == _dates(start, end), "owned dates must tile the window exactly once"

    def test_query_end_is_padded_by_one_day_except_at_the_window_end(self):
        """A date-only interval end covers that day, so without a pad an item in the
        final seconds of a month's last day could fall outside every slice's query.
        """
        months = iter_month_ranges("2024-01-01", "2024-03-31")
        assert months[0].query_end == "2024-02-01"  # padded past its own end
        assert months[1].query_end == "2024-03-01"
        assert months[-1].query_end == "2024-03-31", "the last month must not query past the window"

    def test_padding_cannot_double_process(self):
        """The padded day is OWNED by the next month, so overlap in the query does not
        become overlap in the work.
        """
        a, b = iter_month_ranges("2024-01-01", "2024-02-29")[:2]
        assert a.query_end == b.own_start
        assert a.own_end < b.own_start

    def test_leap_day_is_owned(self):
        feb = next(mr for mr in iter_month_ranges("2024-01-01", "2024-12-31") if mr.own_start == "2024-02-01")
        assert feb.own_end == "2024-02-29"

    def test_rejects_a_backwards_window(self):
        with pytest.raises(ValueError, match="precedes"):
            iter_month_ranges("2024-03-01", "2024-01-01")


class TestStreaming:
    """What the generator yields, and what it does when a query fails."""

    def _fake_query(self, per_date: dict[str, int], calls: list | None = None):
        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox):
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

        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox):
            queried.append(start_date)
            if start_date.startswith("2024-02"):
                second_started.set()
            return [_item(start_date)], {}

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
        assert "2024-02-01" in queried
        list(gen)

    def test_a_failing_month_query_raises_rather_than_truncating(self):
        """A dropped month would be an incomplete mosaic that still reported success —
        the failure must reach the caller.
        """

        def query(*, provider, collection, tile_id, start_date, end_date, existing_dates, bbox):
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
