"""The S2 date loop under both drive modes (pure, offline).

``pipeline_dates`` moves WHEN a date is prepared, and nothing else: the same
dates must be written, in the same order, with the same counters, and a failure
must still reach the caller. Everything outside the loop is stubbed here — the
STAC query, the band load, the ROI store, the write — so what these tests
actually exercise is the loop, the gate's dask arithmetic, and the two log lines
the A/B is read from.

Every behavioural test runs under BOTH modes against the same literal
expectation rather than comparing two runs to each other, so a bug that moved
both modes the same way still fails.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from itertools import pairwise
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from tessera_embeddings.config.satellites import S2_SCL_INVALID_CLASSES
from tessera_embeddings.ingest import s2_roi
from tessera_embeddings.storage.zarr_store import store_write_retrying

BOTH_MODES = pytest.mark.parametrize("pipeline_dates", [False, True], ids=["serial", "pipelined"])

SIZE = 8
VALID_CLASS = next(c for c in range(12) if c not in S2_SCL_INVALID_CLASSES)
INVALID_CLASS = sorted(S2_SCL_INVALID_CLASSES)[0]
COVERAGE_THRESHOLD = 50.0

#: ``Pipeline date=<d>: prepare=<s>s hidden=<s>s stall=<s>s`` — the stable format the
#: A/B greps for, parsed here so a change to it fails a test rather than an analysis.
PIPELINE_LINE = re.compile(
    r"Pipeline date=(?P<date>\S+): prepare=(?P<prepare>[\d.]+)s hidden=(?P<hidden>[\d.]+)s stall=(?P<stall>[\d.]+)s"
)
STAGE_LINE = re.compile(
    r"Stage timings date=(?P<date>\S+): build=(?P<build>[\d.]+)s gate=(?P<gate>[\d.]+)s "
    r"write=(?P<write>[\d.]+)s total=(?P<total>[\d.]+)s"
)

_ROI = SimpleNamespace(
    bbox_wgs84=(-105.0, 39.0, -104.9, 39.1),
    native_crs="EPSG:32613",
    geobox=None,
    height=SIZE,
    width=SIZE,
)


class _FakeClient:
    """Identity ``persist``, so the gate's reduce runs on dask's local scheduler.

    This does not stub the gate itself — its validity arithmetic really executes.
    Only the hand-off to a distributed scheduler is replaced.
    """

    @staticmethod
    def persist(obj):
        return obj

    @staticmethod
    def compute(obj):
        """The cropped gate submits its reduce explicitly; hand back a local future."""
        return SimpleNamespace(result=obj.compute)


def _item(date: str):
    return SimpleNamespace(datetime=datetime.fromisoformat(f"{date}T10:00:00"), properties={"eo:cloud_cover": 0.0})


def _day_ds(date: str, *, valid: bool) -> xr.Dataset:
    """One date's toy mosaic: one reflectance band plus the SCL the gate reads."""
    scl = np.full((1, SIZE, SIZE), VALID_CLASS if valid else INVALID_CLASS, dtype="uint8")
    blue = np.arange(SIZE * SIZE, dtype="uint16").reshape(1, SIZE, SIZE)
    return xr.Dataset(
        {
            "blue": (("time", "northing", "easting"), da.from_array(blue, chunks=(1, SIZE, SIZE))),
            "scl": (("time", "northing", "easting"), da.from_array(scl, chunks=(1, SIZE, SIZE))),
        },
        coords={"time": [np.datetime64(f"{date}T10:00:00", "ns")]},
    )


class _Run:
    """What one offline ingest did: its result, and the order things happened in."""

    def __init__(self) -> None:
        self.result = None
        self.loaded: list[str] = []
        self.written: list[str] = []
        self.load_started: dict[str, float] = {}
        self.write_ended: dict[str, float] = {}
        self.attempts: list[str] = []


@pytest.fixture
def run_ingest(monkeypatch):
    """Return a factory driving :func:`ingest_s2_roi_reflectance` fully offline.

    ``dates`` maps each date to whether its SCL is valid, so a False entry is a
    date the coverage gate drops. ``fail_on`` makes that date's write raise, and
    ``write_s`` slows the write so overlap is observable.
    """

    # Retries keep their COUNT (three attempts, then reraise) and their concurrent-writer
    # exclusion, and lose only the SLEEP, which would otherwise cost the suite six seconds
    # per failure case. Stubbing the policy's sleep rather than its wait strategy keeps this
    # independent of how the backoff is expressed — the strategy used to be built inline
    # here and now comes from storage.zarr_store.store_write_retrying.
    def _no_sleep(log):
        retrying = store_write_retrying(log)
        retrying.sleep = lambda _seconds: None
        return retrying

    monkeypatch.setattr(s2_roi, "store_write_retrying", _no_sleep)

    # Cropping is unconditional, so window derivation runs on every path and needs a mask
    # these tests do not have. One window covering the whole ROI keeps the date pipeline —
    # what this module tests — writing exactly what it used to; real window geometry is
    # covered in test_live_windows.
    #
    # Installed at FIXTURE SETUP, not inside the helper: a test that stubs these itself to
    # assert on the arguments they receive does so in its own body, which runs after the
    # fixture and before the helper, so setting them here lets that override win.
    whole_roi = SimpleNamespace(y0=0, y1=SIZE, x0=0, x1=SIZE)
    monkeypatch.setattr(s2_roi, "live_windows_for_mask", lambda *_a, **_k: [whole_roi])
    monkeypatch.setattr(s2_roi, "windows_for_date", lambda run_windows, *_a, **_k: run_windows)

    runs: list[_Run] = []

    def _run(
        dates: dict[str, bool],
        *,
        pipeline_dates: bool,
        fail_on: str | None = None,
        write_s: float = 0.0,
        log: logging.Logger | None = None,
        existing_dates: set[str] | None = None,
        **ingest_kwargs,
    ) -> _Run:
        run = _Run()
        runs.append(run)  # so a test can inspect a run whose write raised

        def load_stac_items(day_items, **_kwargs):
            date = day_items[0].datetime.strftime("%Y-%m-%d")
            run.load_started[date] = time.monotonic()
            run.loaded.append(date)
            return _day_ds(date, valid=dates[date])

        def _record_write(day_ds) -> None:
            """The write stub's whole behaviour, shared by every writer it stands in for.

            Kept in one place because the pipeline tests measure WHEN a write starts and
            ends; a second copy that forgot ``write_s`` would silently stop testing the
            overlap this module exists to check.
            """
            date = str(day_ds.time.dt.date.values[0])
            run.attempts.append(date)
            if date == fail_on:
                raise RuntimeError(f"write of {date} failed")
            time.sleep(write_s)
            run.written.append(date)
            run.write_ended[date] = time.monotonic()

        def write_day_windows(_store, day_ds, _windows, **_kwargs):
            _record_write(day_ds)

        def write_days_windows(_store, days, **_kwargs):
            # The batched writer takes (day_ds, windows) pairs; each still counts as its
            # own write, which is what the one-commit-per-date accounting assumes.
            for day_ds, _windows in days:
                _record_write(day_ds)

        monkeypatch.setattr(s2_roi, "query_stac_items", lambda **_kwargs: ([_item(d) for d in dates], {}))
        monkeypatch.setattr(s2_roi, "load_stac_items", load_stac_items)
        # What the store already holds. Default empty — a fresh ingest — but a test can
        # populate it to stand a run up over an existing store, which is what a resume is.
        monkeypatch.setattr(s2_roi, "get_existing_dates", lambda *_a, **_k: set(existing_dates or ()))
        monkeypatch.setattr(s2_roi, "read_roi_metadata", lambda *_a, **_k: _ROI)
        monkeypatch.setattr(s2_roi, "read_roi_mask", lambda *_a, **_k: da.ones((SIZE, SIZE), dtype=bool))
        monkeypatch.setattr(s2_roi, "IngestManifest", SimpleNamespace(from_roi_store=lambda _p: None))
        monkeypatch.setattr(s2_roi, "write_day_windows", write_day_windows)
        monkeypatch.setattr(s2_roi, "write_days_windows", write_days_windows)

        run.result = s2_roi.ingest_s2_roi_reflectance(
            roi_zarr_path="memory://roi.zarr",
            start_date=min(dates),
            end_date=max(dates),
            store_path="memory://store",
            client=_FakeClient(),
            min_valid_coverage=COVERAGE_THRESHOLD,
            log=log,
            stream_stac_monthly=False,
            pipeline_dates=pipeline_dates,
            # ONE COMMIT PER DATE. This module measures the per-date pipeline — when a
            # date's preparation overlaps the previous date's write — and the batched
            # writer reports per BATCH instead, so auto-sizing would silently stop
            # exercising what these tests assert on. Batching has its own tests.
            **{"batch_dates": 1, **ingest_kwargs},
        )
        return run

    _run.runs = runs  # type: ignore[attr-defined]
    return _run


# A run of gate failures between two kept dates: the case where the pipeline is
# draining skips while a write is in flight, and the one most likely to miscount.
SKIP_CHAIN = {
    "2024-01-01": True,
    "2024-01-02": False,
    "2024-01-03": False,
    "2024-01-04": False,
    "2024-01-05": True,
}


@BOTH_MODES
def test_a_skip_chain_counts_and_writes_identically(run_ingest, pipeline_dates):
    """Three consecutive gate failures between two kept dates, counted the same way.

    The counters are the ingest's report of what a (zone, year) contains, and the
    marker makes them permanent, so a mode that dropped or double-counted a skip
    would corrupt the record rather than merely run differently.
    """
    run = run_ingest(SKIP_CHAIN, pipeline_dates=pipeline_dates)
    assert run.written == ["2024-01-01", "2024-01-05"], "the kept dates, in date order"
    assert run.loaded == sorted(SKIP_CHAIN), "every date is prepared exactly once, in order"
    assert run.result.dates_processed == 2
    assert run.result.dates_filtered_coverage == 3
    assert run.result.status == "success"


@BOTH_MODES
def test_every_date_failing_the_gate_reports_skipped(run_ingest, pipeline_dates):
    run = run_ingest(dict.fromkeys(SKIP_CHAIN, False), pipeline_dates=pipeline_dates)
    assert run.written == []
    assert run.result.status == "skipped"
    assert run.result.dates_filtered_coverage == 5


@BOTH_MODES
def test_a_write_failure_surfaces_after_its_retries(run_ingest, pipeline_dates):
    """A failed write must reach the caller in both modes, on the date that failed.

    Under pipelining the NEXT date is already being prepared when the write
    raises, so the risk is a failure that is swallowed, deferred to the wrong
    date, or reported after later dates have been committed.
    """
    with pytest.raises(RuntimeError, match="write of 2024-01-03 failed"):
        run_ingest(SKIP_CHAIN | {"2024-01-03": True}, pipeline_dates=pipeline_dates, fail_on="2024-01-03")


@BOTH_MODES
def test_a_write_failure_retries_three_times_then_stops(run_ingest, pipeline_dates):
    """Retry scope is unchanged by the split: three attempts at the WRITE, no more.

    A retry that re-ran preparation would rebuild the graph and re-run the gate,
    turning one transient GDAL error into a second cluster round trip per attempt.

    The FOLLOWING date may already have been prepared when the write fails — that
    is what depth-1 pipelining does — and its preparation is simply discarded. What
    must not happen is the failing date being prepared again per attempt.
    """
    dates = {"2024-01-01": True, "2024-01-02": True}
    with pytest.raises(RuntimeError):
        run_ingest(dates, pipeline_dates=pipeline_dates, fail_on="2024-01-01")
    run = run_ingest.runs[-1]
    assert run.attempts == ["2024-01-01"] * 3
    assert run.loaded.count("2024-01-01") == 1, "a write retry must not re-prepare its date"


@BOTH_MODES
def test_the_pipeline_line_is_emitted_once_per_written_date(run_ingest, pipeline_dates, caplog):
    """One line per WRITTEN date — skipped dates never reach the write, so never log.

    The A/B reads per-date medians off this line, so a duplicated or missing line
    silently rescales the result it is used to judge.
    """
    with caplog.at_level(logging.INFO, logger="s2-pipeline-test"):
        run = run_ingest(SKIP_CHAIN, pipeline_dates=pipeline_dates, log=logging.getLogger("s2-pipeline-test"))
    lines = [m.groupdict() for m in map(PIPELINE_LINE.search, caplog.messages) if m]
    assert [line["date"] for line in lines] == run.written


@BOTH_MODES
def test_the_pipeline_lines_fields_are_non_negative(run_ingest, pipeline_dates, caplog):
    with caplog.at_level(logging.INFO, logger="s2-pipeline-test"):
        run_ingest(SKIP_CHAIN, pipeline_dates=pipeline_dates, log=logging.getLogger("s2-pipeline-test"))
    lines = [m.groupdict() for m in map(PIPELINE_LINE.search, caplog.messages) if m]
    assert lines, "no Pipeline line was emitted at all"
    for line in lines:
        prepare, hidden, stall = (float(line[k]) for k in ("prepare", "hidden", "stall"))
        assert prepare >= 0.0 and hidden >= 0.0 and stall >= 0.0
        assert hidden + stall >= prepare, f"hidden and stall must account for prepare: {line}"


def test_serial_mode_reports_nothing_hidden(run_ingest, caplog):
    """The measurement contract: serially the driver waits out every preparation.

    Reporting ``hidden`` above zero without a pipeline would make the serial
    baseline look like a working pipeline and the whole A/B unreadable — so this
    pins the direction, not just the arithmetic.
    """
    with caplog.at_level(logging.INFO, logger="s2-pipeline-test"):
        run_ingest(SKIP_CHAIN, pipeline_dates=False, log=logging.getLogger("s2-pipeline-test"))
    lines = [m.groupdict() for m in map(PIPELINE_LINE.search, caplog.messages) if m]
    assert lines
    for line in lines:
        assert float(line["hidden"]) == 0.0, f"serial mode hid nothing, yet reported it: {line}"
        assert line["stall"] == line["prepare"], f"serial mode stalls for its whole preparation: {line}"


@BOTH_MODES
def test_stage_timings_total_stays_the_serial_equivalent_cost(run_ingest, pipeline_dates, caplog):
    """``total`` is build+gate+write in both modes, so the field stays comparable.

    Under pipelining part of that total ran concurrently with the previous write,
    which is what the Pipeline line reports; making `total` a wall-clock figure
    instead would break every comparison against a pre-pipeline run.
    """
    with caplog.at_level(logging.INFO, logger="s2-pipeline-test"):
        run_ingest(SKIP_CHAIN, pipeline_dates=pipeline_dates, log=logging.getLogger("s2-pipeline-test"))
    lines = [m.groupdict() for m in map(STAGE_LINE.search, caplog.messages) if m]
    assert lines
    for line in lines:
        build, gate, write, total = (float(line[k]) for k in ("build", "gate", "write", "total"))
        assert total == pytest.approx(build + gate + write, abs=0.15)


def test_the_next_date_is_prepared_during_the_current_write(run_ingest):
    """The property the flag exists for, observed from a recorded timeline.

    Asserted against the serial run as its control: without one, a flag that did
    nothing at all would still pass every other test in this module.
    """
    dates = dict.fromkeys(["2024-01-01", "2024-01-02", "2024-01-03"], True)
    overlaps = {}
    for pipeline_dates in (False, True):
        run = run_ingest(dates, pipeline_dates=pipeline_dates, write_s=0.1)
        overlaps[pipeline_dates] = [
            run.load_started[later] < run.write_ended[earlier] for earlier, later in pairwise(run.written)
        ]
    assert all(overlaps[True]), "pipelined mode prepared no date during a write"
    assert not any(overlaps[False]), "serial mode overlapped a preparation with a write"


@pytest.mark.parametrize("overlapped", [False, True], ids=["sequential-writes", "overlapped-writes"])
def test_per_date_narrowing_is_priced_like_the_run(run_ingest, monkeypatch, overlapped: bool):
    """The run's window price must reach the per-date re-merge, not stop at the run.

    The two merges answer the same question — is this window boundary worth the dead
    area it saves? — so answering it differently on either side of narrowing undoes the
    calibration for every date the footprint narrows. See the same rule pinned on
    ``windows_for_date`` itself in test_live_windows.
    """
    seen: dict[str, int | None] = {}
    window = SimpleNamespace(y0=0, y1=SIZE, x0=0, x1=SIZE)

    def fake_live_windows_for_mask(*_a, window_cost_in_chunks=None, **_k):
        seen["run"] = window_cost_in_chunks
        return [window]

    def fake_windows_for_date(run_windows, *_a, window_cost_in_chunks=None, **_k):
        seen["date"] = window_cost_in_chunks
        return run_windows

    monkeypatch.setattr(s2_roi, "live_windows_for_mask", fake_live_windows_for_mask)
    monkeypatch.setattr(s2_roi, "windows_for_date", fake_windows_for_date)
    monkeypatch.setattr(s2_roi, "write_day_windows", lambda *_a, **_k: None)
    monkeypatch.setattr(s2_roi, "write_days_windows", lambda *_a, **_k: None)

    run_ingest(
        dict.fromkeys(["2024-01-01"], True),
        pipeline_dates=False,
        overlap_window_writes=overlapped,
    )
    expected = s2_roi.WINDOW_COST_IN_CHUNKS_OVERLAPPED if overlapped else s2_roi.WINDOW_COST_IN_CHUNKS
    assert seen == {"run": expected, "date": expected}


def test_the_assessed_window_lands_on_the_reflectance_repo(run_ingest, monkeypatch):
    """It must be written to the repo the coverage gate opens, not the parent directory.

    ``store_path`` holds all three child repos; the S2 mosaic is the ``reflectance.zarr``
    inside it. ``record_assessed_window`` only logs when its open fails, so aiming at the
    parent left the attribute unwritten in silence — and a month that was examined and
    found genuinely empty then reads as an unexplained gap, failing the fill.
    """
    seen: list[str] = []
    monkeypatch.setattr(s2_roi, "record_assessed_window", lambda path, *_a, **_k: seen.append(path))
    run_ingest(dict.fromkeys(["2024-01-01"], True), pipeline_dates=False)
    assert seen == ["memory://store/reflectance.zarr"]


def test_the_assessed_window_is_recorded_when_a_populated_store_gains_no_dates(run_ingest, monkeypatch):
    """The permanent-failure case: a run that wrote nothing over a store that is already full.

    Two ways in, one shape. A resume whose query filtered every date away as already
    committed writes nothing; so does a pass whose only dates all fail coverage — the case
    below, and the very case the attribute exists to explain. Keyed on what THIS invocation
    wrote, the record was skipped, and skipped again on every retry, because each one takes
    the same zero-write path. The coverage gate then reads the empty month as an
    unexplained gap and the zone-year can never complete. Keyed on the STORE, it is written.
    """
    seen: list[str] = []
    monkeypatch.setattr(s2_roi, "record_assessed_window", lambda path, *_a, **_k: seen.append(path))
    run = run_ingest(dict.fromkeys(["2024-01-01"], False), pipeline_dates=False, existing_dates={"2023-12-31"})

    assert run.written == [], "the only date fails coverage; nothing should be written"
    assert seen == ["memory://store/reflectance.zarr"]


def test_no_assessed_window_is_recorded_when_the_store_does_not_exist(run_ingest, monkeypatch):
    """The complement: nothing written and nothing present means there is no repo to annotate.

    Guards the probe added for the case above from becoming an unconditional write against
    a store that was never created — ``record_assessed_window`` opens and never creates, so
    that would be a warning on every genuinely empty run.
    """
    seen: list[str] = []
    monkeypatch.setattr(s2_roi, "record_assessed_window", lambda path, *_a, **_k: seen.append(path))

    run_ingest(dict.fromkeys(["2024-01-01"], False), pipeline_dates=False)

    assert seen == []
