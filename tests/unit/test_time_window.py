"""Tests for temporal window parsing, filtering, and coverage checking."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference.data_loading import check_time_window_coverage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_chronological(months: tuple[tuple[int, int], ...]) -> None:
    """Assert that a months tuple is in strict chronological order."""
    for i in range(len(months) - 1):
        assert months[i] < months[i + 1], f"Not chronological at index {i}: {months[i]} >= {months[i + 1]}"


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
        _assert_chronological(tw.months)

    def test_january_window(self):
        """Window ending in January spans Feb prior year through January."""
        tw = parse_time_window("January 2025")
        assert tw.window_start == (2024, 2)
        assert tw.window_end == (2025, 1)
        assert tw.months[0] == (2024, 2)
        assert tw.months[-1] == (2025, 1)
        assert len(tw.months) == 12
        _assert_chronological(tw.months)

    def test_december_window(self):
        """Window ending in December spans full calendar year."""
        tw = parse_time_window("December 2025")
        assert tw.window_start == (2025, 1)
        assert tw.window_end == (2025, 12)
        assert tw.months[0] == (2025, 1)
        assert tw.months[-1] == (2025, 12)
        assert len(tw.months) == 12
        assert tw.window_end_label == "2025-12-01"
        _assert_chronological(tw.months)

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
        _assert_chronological(tw.months)


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

    def test_february_non_leap_year(self):
        tw = parse_time_window("February 2025")
        _, end = tw.to_date_range()
        assert end == "2025-02-28"

    def test_31_day_end_month(self):
        tw = parse_time_window("December 2025")
        start, end = tw.to_date_range()
        assert start == "2025-01-01"
        assert end == "2025-12-31"


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


# ---------------------------------------------------------------------------
# check_time_window_coverage
# ---------------------------------------------------------------------------


def _make_time_group(start: str, end: str):
    """Build an in-memory zarr group with a ``time`` array spanning [start, end]."""
    store = zarr.storage.MemoryStore()
    root = zarr.group(store)

    times = np.array([np.datetime64(start), np.datetime64(end)], dtype="datetime64[ns]").astype("int64")

    t_arr = root.create_array("time", shape=times.shape, dtype=np.int64, chunks=times.shape)
    t_arr[:] = times

    return root


@pytest.fixture()
def _mock_open_store(monkeypatch):
    """Factory fixture that patches open_store_as_zarr_group with configurable time bounds.

    Returns a callable: ``configure(start, end)`` sets the mock store's time range.
    """
    import tessera_embeddings.inference.data_loading as dl

    store_times: dict[str, tuple[str, str]] = {}

    def _open_store(path):
        start, end = store_times.get("default", ("2024-01-01", "2025-12-31"))
        return _make_time_group(start, end)

    monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)

    def configure(start: str, end: str) -> None:
        store_times["default"] = (start, end)

    return configure


class TestCheckTimeWindowCoverage:
    """Tests for check_time_window_coverage()."""

    def test_coverage_ok(self, _mock_open_store):
        """No error when stores span the window."""
        _mock_open_store("2024-01-01", "2025-12-31")
        tw = parse_time_window("July 2025")
        check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending")

    @pytest.mark.parametrize(
        ("store_start", "store_end", "match"),
        [
            ("2024-10-01", "2025-12-31", "store starts at"),
            ("2024-01-01", "2025-06-15", "store ends at"),
        ],
        ids=["too-late-start", "too-early-end"],
    )
    def test_coverage_fails(self, _mock_open_store, store_start, store_end, match):
        _mock_open_store(store_start, store_end)
        tw = parse_time_window("July 2025")
        with pytest.raises(InsufficientCoverageError, match=match):
            check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending")

    def test_ascending_orbit_checks_two_stores(self, monkeypatch):
        """With s1_orbit='ascending', reflectance and sar_ascending are checked."""
        import tessera_embeddings.inference.data_loading as dl

        checked_paths: list[str] = []

        def _open_store(path):
            checked_paths.append(path)
            return _make_time_group("2024-01-01", "2025-12-31")

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        tw = parse_time_window("July 2025")
        check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending")
        assert len(checked_paths) == 2
        assert any("reflectance" in p for p in checked_paths)
        assert any("sar_ascending" in p for p in checked_paths)
        assert not any("sar_descending" in p for p in checked_paths)

    def test_both_orbits_checks_all_three_stores(self, monkeypatch):
        """With s1_orbit='both', reflectance, sar_ascending, and sar_descending are checked."""
        import tessera_embeddings.inference.data_loading as dl

        checked_paths: list[str] = []

        def _open_store(path):
            checked_paths.append(path)
            return _make_time_group("2024-01-01", "2025-12-31")

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        tw = parse_time_window("July 2025")
        check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="both")
        assert len(checked_paths) == 3
        assert any("reflectance" in p for p in checked_paths)
        assert any("sar_ascending" in p for p in checked_paths)
        assert any("sar_descending" in p for p in checked_paths)

    def test_empty_store_raises(self, monkeypatch):
        """A store with zero time entries raises InsufficientCoverageError."""
        import tessera_embeddings.inference.data_loading as dl

        def _open_store(path):
            store = zarr.storage.MemoryStore()
            root = zarr.group(store)
            root.create_array("time", shape=(0,), dtype=np.int64, chunks=(1,))
            return root

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        tw = parse_time_window("July 2025")
        with pytest.raises(InsufficientCoverageError, match="no time entries"):
            check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending")

    def test_skip_coverage_check_bypasses_range_check(self, _mock_open_store):
        """skip_coverage_check=True bypasses the range check even when out of range."""
        # Store ends well before the window — would normally raise.
        _mock_open_store("2024-10-01", "2024-12-31")
        tw = parse_time_window("July 2025")
        # Should NOT raise because the range check is skipped.
        check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending", skip_coverage_check=True)

    def test_skip_coverage_check_still_rejects_empty_store(self, monkeypatch):
        """skip_coverage_check does not bypass the empty-store guard (it runs first)."""
        import tessera_embeddings.inference.data_loading as dl

        def _open_store(path):
            store = zarr.storage.MemoryStore()
            root = zarr.group(store)
            root.create_array("time", shape=(0,), dtype=np.int64, chunks=(1,))
            return root

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        tw = parse_time_window("July 2025")
        with pytest.raises(InsufficientCoverageError, match="no time entries"):
            check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending", skip_coverage_check=True)
