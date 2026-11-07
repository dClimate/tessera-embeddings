"""Tests for temporal window parsing.

The gate that checks a store against a window is tested with its assessed-window rule in
``tests/unit/inference/test_assessed_window_coverage.py``.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.config.time_windows import parse_time_window

# ---------------------------------------------------------------------------
# parse_time_window — rolling 12-month window
# ---------------------------------------------------------------------------


class TestParseTimeWindow:
    """Tests for parse_time_window()."""

    def test_basic_rolling_window(self):
        """12-month window ending at June 2025."""
        tw = parse_time_window("June 2025")
        assert tw.window_start == (2024, 7)
        assert tw.window_end == (2025, 6)
        assert len(tw.months) == 12
        assert tw.months[0] == (2024, 7)
        assert tw.months[-1] == (2025, 6)
        assert tw.window_end_label == "2025-06-01"

    def test_january_window(self):
        """Window ending in January spans Feb prior year through January."""
        tw = parse_time_window("January 2025")
        assert tw.window_start == (2024, 2)
        assert tw.window_end == (2025, 1)
        assert tw.months[0] == (2024, 2)
        assert tw.months[-1] == (2025, 1)
        assert len(tw.months) == 12

    @pytest.mark.parametrize(
        ("date", "expected_start", "expected_end"),
        [
            ("December 2025", (2025, 1), (2025, 12)),
            ("September 2025", (2024, 10), (2025, 9)),
        ],
        ids=["calendar-year", "fiscal-year"],
    )
    def test_various_end_months(self, date, expected_start, expected_end):
        tw = parse_time_window(date)
        assert len(tw.months) == 12
        assert tw.months[0] == expected_start
        assert tw.months[-1] == expected_end


class TestToDateRange:
    """Tests for TimeWindow.to_date_range()."""

    def test_basic_range(self):
        """First-of-start-month through last-of-end-month."""
        tw = parse_time_window("June 2025")
        start, end = tw.to_date_range()
        assert start == "2024-07-01"
        assert end == "2025-06-30"

    def test_end_month_last_day_uses_calendar(self):
        """The end date uses the actual last day of the month (calendar-aware)."""
        # February 2024 is a leap year → 29 days.
        tw = parse_time_window("February 2024")
        _, end = tw.to_date_range()
        assert end == "2024-02-29"


class TestParseTimeWindowErrors:
    """Tests for parse_time_window() validation errors."""

    @pytest.mark.parametrize(
        ("date", "match"),
        [
            ("2025-06", "Cannot parse"),
            ("Smarch 2025", "Cannot parse"),
        ],
        ids=["iso-format", "bad-month-name"],
    )
    def test_invalid_input(self, date, match):
        with pytest.raises(ValueError, match=match):
            parse_time_window(date)
