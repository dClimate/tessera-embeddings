"""Recording how much of its window a published year's INPUT actually held.

The store guarantees calendar-year slots, so the *requested* window is never in doubt and is
deliberately not recorded. What was missing is the different question: how much of that
window the source mosaics contained. A cell can be filled from a partial input through the
coverage check's documented relaxation and still be marked complete — and the mosaics are
deleted once it lands, so the evidence for the question outlives neither the run nor the
mosaic. These tests pin the field that makes it answerable from the store alone.

The relaxed path is what the field exists for, so it is tested on both sides: the summary
must be produced when the strict rule passes AND when it is bypassed, and it must say which.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from tessera_embeddings.config.time_windows import parse_time_window
from tessera_embeddings.errors import InsufficientCoverageError
from tessera_embeddings.inference import data_loading
from tessera_embeddings.inference.data_loading import check_time_window_coverage
from tessera_embeddings.storage.shard_writer import run_provenance


def _serve(monkeypatch, dates: list[str]) -> None:
    """Serve one in-memory mosaic for every store the check opens.

    Only the store ACCESS is stood in for; the month arithmetic under test is the real
    function's. Building a local icechunk repository would test the storage layer instead,
    which is covered elsewhere and is what these assertions are not about.
    """
    root = zarr.open_group(store=zarr.storage.MemoryStore(), mode="w")
    values = np.array(dates, dtype="datetime64[ns]").astype("int64")
    arr = root.create_array("time", shape=(len(values),), dtype="int64")
    arr[:] = values
    arr.attrs["units"] = "nanoseconds since 1970-01-01"
    monkeypatch.setattr(data_loading, "open_store_as_zarr_group", lambda *a, **k: root)


def _year_dates(year: int, months: range) -> list[str]:
    return [f"{year}-{m:02d}-15" for m in months]


def test_record_carries_input_coverage_when_supplied() -> None:
    """The field lands on the year's record, beside the other per-run summaries."""
    coverage = {"window_months": 12, "relaxed": False, "stores": {"reflectance": {"months_present": 12}}}
    runs = run_provenance(None, 2024, "run-1", input_coverage=coverage)
    assert runs["2024"]["input_coverage"] == coverage


def test_absent_coverage_is_recorded_as_nothing() -> None:
    """A caller that measured nothing claims nothing — the field is omitted, not empty."""
    runs = run_provenance(None, 2024, "run-1")
    assert "input_coverage" not in runs["2024"]


def test_full_window_is_measured_and_reported(monkeypatch) -> None:
    """A store spanning the window passes AND returns what it holds."""
    _serve(monkeypatch, _year_dates(2024, range(1, 13)))
    summary = check_time_window_coverage("mem://m", parse_time_window("December 2024"), s1_orbit="none")
    assert summary["window_months"] == 12
    assert summary["relaxed"] is False
    assert summary["stores"]["reflectance"]["months_present"] == 12
    assert summary["stores"]["reflectance"]["first"] == "2024-01-15"
    assert summary["stores"]["reflectance"]["last"] == "2024-12-15"


def test_a_half_year_is_refused_under_the_strict_rule(monkeypatch) -> None:
    """The control for the test below: without the relaxation this input never publishes."""
    _serve(monkeypatch, _year_dates(2024, range(1, 7)))
    with pytest.raises(InsufficientCoverageError, match="missing 6"):
        check_time_window_coverage("mem://m", parse_time_window("December 2024"), s1_orbit="none")


def test_a_relaxed_half_year_is_measured_and_says_it_was_relaxed(monkeypatch) -> None:
    """The case the field exists for: published from half a year, and the record shows it.

    Without this, a cell filled this way is indistinguishable afterwards from one built on a
    full year — the mosaic that would prove otherwise is deleted when the cell lands.
    """
    _serve(monkeypatch, _year_dates(2024, range(1, 7)))
    summary = check_time_window_coverage(
        "mem://m", parse_time_window("December 2024"), s1_orbit="none", skip_coverage_check=True
    )
    assert summary["relaxed"] is True
    assert summary["stores"]["reflectance"]["months_present"] == 6
    assert summary["stores"]["reflectance"]["last"] == "2024-06-15"

    runs = run_provenance(None, 2024, "run-1", input_coverage=summary)
    recorded = runs["2024"]["input_coverage"]
    assert recorded["stores"]["reflectance"]["months_present"] < recorded["window_months"]


def test_out_of_window_dates_do_not_count_toward_coverage(monkeypatch) -> None:
    """Coverage is of the year ASKED for, not of the store's own extent.

    A store padded with neighbouring-year dates would otherwise report a fuller window than
    the published year actually drew on.
    """
    dates = _year_dates(2024, range(1, 7)) + _year_dates(2023, range(7, 13))
    _serve(monkeypatch, dates)
    summary = check_time_window_coverage(
        "mem://m", parse_time_window("December 2024"), s1_orbit="none", skip_coverage_check=True
    )
    store = summary["stores"]["reflectance"]
    assert store["months_present"] == 6
    assert store["dates_in_window"] == 6
    assert store["first"].startswith("2024-") and store["last"].startswith("2024-")
