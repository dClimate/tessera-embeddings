"""The S1 batch look-ahead must overlap the catalogue query without duplicating a date.

Two independent properties, and the second is the one with teeth.

**It overlaps.** A batch's catalogue query is ~14% of its wall clock and is otherwise pure
serial time, since the query holds no credential and touches no store. Preparing the next
batch during the current batch's writes removes all of it but the first batch's.

**It cannot write a date twice.** The serial loop re-read ``get_existing_dates`` from the
store on every batch so that a date written by an earlier batch was skipped by a later one.
A look-ahead makes that read unsafe — the next query is built before the current batch has
committed — so the write loop tracks written dates in-process and is the authority. This
matters because batches are cut on UTC dates while the loader groups by SOLAR day, so an
acquisition late on a batch's last UTC day can belong to the next batch's solar day, and
two consecutive batches can then contain the same day. Writing it twice would trip the
store's duplicate-date guard part-way through a run.

The overlap tests drive ``pipelined`` directly, since ordering is all they assert. The
duplicate-date tests drive the real ``ingest_s1_roi_sar`` loop against a faked catalogue and
store, because the rule they pin lives in that loop rather than in the pipeline — and both
were checked to fail when the skip is removed, so they are not passing vacuously.
"""

from __future__ import annotations

import threading
import time

from tessera_embeddings.ingest._pipeline import pipelined


def test_next_batch_is_prepared_during_the_current_batch_write() -> None:
    """The whole point: batch N+1's query runs while batch N is still writing."""
    events: list[str] = []
    lock = threading.Lock()

    def prepare(batch: int) -> int:
        with lock:
            events.append(f"query{batch}")
        return batch

    for prepared, _stall in pipelined([0, 1, 2], prepare, depth=1):
        # A real write is long relative to a query; sleep so the background thread has
        # time to start the next one, which is exactly the condition being asserted.
        time.sleep(0.05)
        with lock:
            events.append(f"write{prepared}")

    # query0 must precede write0 (nothing to hide behind), but query1 must land BEFORE
    # write0 finishes — that is the overlap. Asserting on the interleaving rather than on
    # elapsed time keeps this deterministic.
    assert events[0] == "query0"
    assert events.index("query1") < events.index("write0"), events
    assert events.index("query2") < events.index("write1"), events


def test_first_batch_stalls_for_its_whole_query_and_later_ones_do_not() -> None:
    """``stall`` is the health metric the batch timing line reports as hidden/stall.

    The first batch has nothing preceding it, so it waits out its own query; later
    batches should wait for nothing because their query ran during the previous write.
    """
    query_s = 0.05

    def prepare(batch: int) -> int:
        time.sleep(query_s)
        return batch

    stalls = []
    for _prepared, stall in pipelined([0, 1, 2], prepare, depth=1):
        stalls.append(stall)
        time.sleep(query_s * 4)  # a write comfortably longer than a query

    assert stalls[0] >= query_s * 0.5, f"first batch should wait for its own query: {stalls}"
    assert all(s < query_s * 0.5 for s in stalls[1:]), f"later batches should not stall: {stalls}"


def test_prepare_exceptions_surface_when_the_batch_is_consumed() -> None:
    """A failing query must raise rather than silently yielding no dates.

    A dropped batch is an incomplete mosaic that reports success, which is worse than a
    failure — and the failure must arrive in the same ORDER the serial loop would have hit
    it, so earlier batches are still written.
    """
    consumed: list[int] = []

    def prepare(batch: int) -> int:
        if batch == 1:
            raise RuntimeError("catalogue unavailable")
        return batch

    try:
        for prepared, _stall in pipelined([0, 1, 2], prepare, depth=1):
            consumed.append(prepared)
    except RuntimeError as exc:
        assert "catalogue unavailable" in str(exc)
    else:  # pragma: no cover - the raise above is the expected path
        raise AssertionError("the failing batch did not raise")

    assert consumed == [0], f"batch 0 should have been consumed before the failure: {consumed}"


def _run_ingest(monkeypatch, batch_dates: dict[str, list[str]], existing: set[str], **kwargs) -> list[str]:
    """Drive ``ingest_s1_roi_sar`` against a faked catalogue and store.

    ``batch_dates`` maps a batch's start date to the solar dates its query returns, which
    is how a shared boundary day is expressed: name it in two consecutive batches.
    Returns the dates actually handed to ``write_day_windows``, in order.
    """
    import numpy as np
    import xarray as xr

    from tessera_embeddings.ingest import s1_roi

    written: list[str] = []

    def fake_ingest_tile(*, start_date: str, existing_dates=None, **_):
        dates = batch_dates.get(start_date, [])
        if not dates:
            return None, {}
        ds = xr.Dataset(
            {"VV": (("time", "y", "x"), np.zeros((len(dates), 1, 1), dtype="float32"))},
            coords={"time": np.array(dates, dtype="datetime64[ns]")},
        )
        return ds, {}

    def fake_write_day_windows(_store, data, *_args, **_kwargs):
        written.append(str(data["time"].values[0])[:10])

    monkeypatch.setattr(s1_roi, "ingest_tile", fake_ingest_tile)
    monkeypatch.setattr(s1_roi, "write_day_windows", fake_write_day_windows)
    monkeypatch.setattr(s1_roi, "get_existing_dates", lambda _store: set(existing))
    monkeypatch.setattr(s1_roi, "apply_roi_mask", lambda data, *a, **k: data)
    monkeypatch.setattr(s1_roi, "read_roi_mask", lambda *a, **k: object())
    monkeypatch.setattr(
        s1_roi,
        "read_roi_metadata",
        lambda _p: type("M", (), {"geobox": object(), "native_crs": "EPSG:32635", "bbox_wgs84": (0, 0, 1, 1)})(),
    )
    monkeypatch.setattr(
        s1_roi, "live_windows_for_mask", lambda *a, **k: [type("W", (), {"y0": 0, "y1": 1, "x0": 0, "x1": 1})()]
    )
    monkeypatch.setattr(s1_roi, "make_s1_item_provider", lambda *a, **k: lambda: [])
    # Reads the ROI store for provenance; irrelevant here and the only other S3 caller.
    monkeypatch.setattr(s1_roi.IngestManifest, "from_roi_store", classmethod(lambda _cls, _p: object()))

    s1_roi.ingest_s1_roi_sar(
        roi_zarr_path="s3://bucket/roi.zarr",
        start_date="2024-01-01",
        end_date="2024-01-04",
        store_path="s3://bucket/mosaics",
        client=type("C", (), {"persist": staticmethod(lambda x: x)})(),
        orbit="ascending",
        batch_days=2,
        crop_to_live_windows=True,
        **kwargs,
    )
    return written


def test_a_solar_day_in_two_batches_is_written_once(monkeypatch) -> None:
    """The rule the in-process written-date set exists to enforce, through the real loop.

    Two consecutive batches both report 2024-01-02, which is what a UTC-cut boundary
    falling inside a solar day produces. Committing it twice would trip the store's
    duplicate-date guard part-way through a run.
    """
    written = _run_ingest(
        monkeypatch,
        batch_dates={"2024-01-01": ["2024-01-01", "2024-01-02"], "2024-01-03": ["2024-01-02", "2024-01-04"]},
        existing=set(),
    )
    assert written.count("2024-01-02") == 1, f"the shared solar day was written twice: {written}"
    assert written == ["2024-01-01", "2024-01-02", "2024-01-04"]


def test_the_guard_holds_with_the_look_ahead_disabled_too(monkeypatch) -> None:
    """The serial path must not be the only correct one — both share the same authority."""
    written = _run_ingest(
        monkeypatch,
        batch_dates={"2024-01-01": ["2024-01-01", "2024-01-02"], "2024-01-03": ["2024-01-02", "2024-01-04"]},
        existing=set(),
        pipeline_batches=False,
    )
    assert written == ["2024-01-01", "2024-01-02", "2024-01-04"]


def test_dates_already_in_the_store_are_skipped_on_resume(monkeypatch) -> None:
    """Seeding from the store is what makes a resumed run skip finished work."""
    written = _run_ingest(
        monkeypatch,
        batch_dates={"2024-01-01": ["2024-01-01", "2024-01-02"], "2024-01-03": ["2024-01-03"]},
        existing={"2024-01-01"},
    )
    assert written == ["2024-01-02", "2024-01-03"]
