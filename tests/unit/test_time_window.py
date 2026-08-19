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
    """Build an in-memory zarr group with a MONTHLY ``time`` array over [start, end].

    Monthly (first of each month) rather than just the two endpoints, so a store
    that "spans" the window actually contains every month the per-month coverage
    check now requires.
    """
    store = zarr.storage.MemoryStore()
    root = zarr.group(store)

    months = np.arange(np.datetime64(start, "M"), np.datetime64(end, "M") + 1)
    times = months.astype("datetime64[ns]").astype("int64")

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

    def _open_store(path, **kwargs):  # accept get_credentials/region
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
        ("store_start", "store_end"),
        [
            ("2024-10-01", "2025-12-31"),  # missing the window's Aug/Sep 2024 months
            ("2024-01-01", "2025-06-15"),  # missing the window's July 2025 month
        ],
        ids=["too-late-start", "too-early-end"],
    )
    def test_coverage_fails(self, _mock_open_store, store_start, store_end):
        _mock_open_store(store_start, store_end)
        tw = parse_time_window("July 2025")
        with pytest.raises(InsufficientCoverageError, match="missing"):
            check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending")

    def test_ascending_orbit_checks_two_stores(self, monkeypatch):
        """With s1_orbit='ascending', reflectance and sar_ascending are checked."""
        import tessera_embeddings.inference.data_loading as dl

        checked_paths: list[str] = []

        def _open_store(path, **kwargs):  # accept get_credentials/region
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

        def _open_store(path, **kwargs):  # accept get_credentials/region
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

        def _open_store(path, **kwargs):  # accept get_credentials/region
            store = zarr.storage.MemoryStore()
            root = zarr.group(store)
            root.create_array("time", shape=(0,), dtype=np.int64, chunks=(1,))
            return root

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        tw = parse_time_window("July 2025")
        with pytest.raises(InsufficientCoverageError, match="no time entries"):
            check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending")

    def test_skip_coverage_check_requires_in_window_data(self, _mock_open_store):
        """Partial-window mode still rejects a store with only out-of-window dates."""
        _mock_open_store("2023-01-01", "2023-12-31")  # entirely before the July 2025 window
        tw = parse_time_window("July 2025")
        with pytest.raises(InsufficientCoverageError, match="no timestamps within the window"):
            check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending", skip_coverage_check=True)

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

        def _open_store(path, **kwargs):  # accept get_credentials/region
            store = zarr.storage.MemoryStore()
            root = zarr.group(store)
            root.create_array("time", shape=(0,), dtype=np.int64, chunks=(1,))
            return root

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        tw = parse_time_window("July 2025")
        with pytest.raises(InsufficientCoverageError, match="no time entries"):
            check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending", skip_coverage_check=True)

    def test_an_assessed_window_cannot_excuse_a_store_with_no_in_window_dates(self, monkeypatch):
        """STRICT mode needs the in-window guard too, despite its every-month rule.

        An `assessed_window` says a month was examined and held nothing reachable, which
        is a finding rather than a gap — but it can explain away EVERY month of the
        window, emptying the missing list. A store the ingest looked at and wrote nothing
        into would then pass the one gate that exists to fail before a GPU fleet is
        provisioned, and the run would die at the first read instead, because the loaders
        raise on an empty filtered index.
        """
        import tessera_embeddings.inference.data_loading as dl

        def _open_store(path, **kwargs):
            # Dates from a PRIOR year only: non-empty, but nothing the window can use.
            root = _make_time_group("2023-01-01", "2023-12-31")
            root.attrs["assessed_window"] = ["2024-08-01", "2025-07-31"]  # the whole window
            return root

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        tw = parse_time_window("July 2025")
        with pytest.raises(InsufficientCoverageError, match="no timestamps within the window"):
            check_time_window_coverage("s3://fake/mosaic", tw, s1_orbit="ascending")

    def test_a_month_lost_to_unreadable_imagery_is_not_excused(self, monkeypatch):
        """A whole-month DATA-LOSS hole must not publish as a legitimate absence.

        The assessed window says "we looked here". It cannot say "and there was nothing to
        find" for a month whose every acquisition was skipped as unreadable — there the
        imagery existed and was lost. Both look identical to a present-month count, and the
        year is write-once, so excusing this one makes the hole permanent and mislabelled.
        """
        import tessera_embeddings.inference.data_loading as dl

        def _open_store(path, **kwargs):
            root = _make_time_group("2024-08-01", "2024-10-31")  # first 3 months only
            root.attrs["assessed_window"] = ["2024-08-01", "2025-07-31"]  # would excuse them all
            root.attrs["assessed_unreadable_dates"] = [{"date": "2025-03-14", "objects": 4, "scope": "tile"}]
            return root

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        with pytest.raises(InsufficientCoverageError, match="2025-03"):
            check_time_window_coverage("s3://fake/mosaic", parse_time_window("July 2025"), s1_orbit="ascending")

    def test_an_unparseable_unreadable_record_excuses_nothing(self, monkeypatch):
        """Same asymmetry the assessed-window parser uses: a damaged record makes it STRICTER.

        If the list cannot be read, which month lost imagery is unknown — so no month may be
        excused. Over-excusing publishes a hole; under-excusing costs a re-ingest.
        """
        import tessera_embeddings.inference.data_loading as dl

        def _open_store(path, **kwargs):
            root = _make_time_group("2024-08-01", "2024-10-31")
            root.attrs["assessed_window"] = ["2024-08-01", "2025-07-31"]
            root.attrs["assessed_unreadable_dates"] = [{"date": "not-a-date"}]
            return root

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        with pytest.raises(InsufficientCoverageError):
            check_time_window_coverage("s3://fake/mosaic", parse_time_window("July 2025"), s1_orbit="ascending")

    def test_an_assessed_window_still_excuses_absent_months_when_data_exists(self, monkeypatch):
        """The guard must not undo what the assessed window is FOR.

        A store holding part of the window, with the rest examined and empty, is the
        legitimate sparse-zone case and has to keep passing.
        """
        import tessera_embeddings.inference.data_loading as dl

        def _open_store(path, **kwargs):
            root = _make_time_group("2024-08-01", "2024-10-31")  # first 3 months only
            root.attrs["assessed_window"] = ["2024-08-01", "2025-07-31"]
            return root

        monkeypatch.setattr(dl, "open_store_as_zarr_group", _open_store)
        check_time_window_coverage("s3://fake/mosaic", parse_time_window("July 2025"), s1_orbit="ascending")
