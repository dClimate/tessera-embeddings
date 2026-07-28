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
    monkeypatch.setattr(s1_roi, "get_existing_dates", lambda _store, **_kw: set(existing))
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


def _run_with_footprints(monkeypatch, dates, bboxes_by_date, *, narrow: bool, windows: int = 4):
    """Drive the ingest with a controllable per-date footprint.

    ``bboxes_by_date`` maps a date to the item bboxes reported for it, which is what decides
    the footprint. A date whose items all carry ``None`` has imagery reaching NO window; a
    date with no items at all is UNKNOWN, which is a different case and must fall back.
    """
    import numpy as np
    import xarray as xr

    from tessera_embeddings.ingest import s1_roi

    written: list[tuple[str, int]] = []
    all_windows = [type("W", (), {"y0": i, "y1": i + 1, "x0": 0, "x1": 1})() for i in range(windows)]

    def fake_ingest_tile(*, start_date: str, item_provider_fn=None, **_):
        if item_provider_fn is not None:
            item_provider_fn()  # let the capture wrapper see the items
        ds = xr.Dataset(
            {"VV": (("time", "y", "x"), np.zeros((len(dates), 1, 1), dtype="float32"))},
            coords={"time": np.array(dates, dtype="datetime64[ns]")},
        )
        return ds, {}

    def fake_items():
        import datetime as dt

        out = []
        for d, boxes in bboxes_by_date.items():
            for b in boxes:
                out.append(
                    type("I", (), {"datetime": dt.datetime.fromisoformat(d).replace(tzinfo=dt.UTC), "bbox": b})()
                )
        return out

    # windows_for_date is real geometry; stub it to express "reaches these many windows"
    # so the test pins the SKIP and NARROW rules rather than re-testing that function.
    def fake_windows_for_date(run_windows, bboxes, _geobox, **_kw):
        reach = len([b for b in bboxes if b is not None])
        return run_windows[:reach]

    monkeypatch.setattr(s1_roi, "ingest_tile", fake_ingest_tile)
    monkeypatch.setattr(s1_roi, "windows_for_date", fake_windows_for_date)
    monkeypatch.setattr(s1_roi, "make_s1_item_provider", lambda *a, **k: fake_items)
    monkeypatch.setattr(
        s1_roi,
        "write_day_windows",
        lambda _s, data, w, **_k: written.append((str(data["time"].values[0])[:10], len(w))),
    )
    monkeypatch.setattr(s1_roi, "get_existing_dates", lambda _s, **_kw: set())
    monkeypatch.setattr(s1_roi, "apply_roi_mask", lambda data, *a, **k: data)
    monkeypatch.setattr(s1_roi, "read_roi_mask", lambda *a, **k: object())
    monkeypatch.setattr(s1_roi, "live_windows_for_mask", lambda *a, **k: all_windows)
    monkeypatch.setattr(s1_roi, "solar_grouping_longitude", lambda _roi: 0.0)
    monkeypatch.setattr(
        s1_roi,
        "read_roi_metadata",
        lambda _p: type("M", (), {"geobox": object(), "native_crs": "EPSG:32635", "bbox_wgs84": (0, 0, 1, 1)})(),
    )
    monkeypatch.setattr(s1_roi.IngestManifest, "from_roi_store", classmethod(lambda _c, _p: object()))

    s1_roi.ingest_s1_roi_sar(
        roi_zarr_path="s3://bucket/roi.zarr",
        start_date="2024-01-01",
        end_date="2024-01-02",
        store_path="s3://bucket/mosaics",
        client=type("C", (), {"persist": staticmethod(lambda x: x)})(),
        orbit="ascending",
        batch_days=5,
        crop_to_live_windows=True,
        narrow_windows_per_date=narrow,
    )
    return written


def test_a_date_reaching_no_live_window_is_skipped(monkeypatch) -> None:
    """Writing it would build a full graph to store nothing — all-fill chunks never persist.

    Skipped regardless of the narrowing flag, since it costs nothing either way.
    """
    for narrow in (False, True):
        written = _run_with_footprints(
            monkeypatch,
            dates=["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
            # The second date HAS imagery (so its group exists and keys to the slice) but
            # none of it reaches a live window.
            bboxes_by_date={"2024-01-01T00:00:00": [(0, 0, 1, 1)], "2024-01-02T00:00:00": [None]},
            narrow=narrow,
        )
        assert [d for d, _ in written] == ["2024-01-01"], f"narrow={narrow}: {written}"


def test_a_date_with_no_items_at_all_is_written_in_full_not_skipped(monkeypatch) -> None:
    """Reaches-nothing and we-do-not-know must not collapse into the same branch.

    No items for a slice means the footprint is unknown, and the conservative answer is to
    write every window. Treating it as "reaches nothing" would silently drop the date.
    """
    written = _run_with_footprints(
        monkeypatch,
        dates=["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        bboxes_by_date={"2024-01-01T00:00:00": [(0, 0, 1, 1)]},  # nothing for the 2nd date
        narrow=True,
    )
    assert ("2024-01-02", 4) in written, f"unknown footprint must write everything: {written}"


def test_narrowing_writes_only_the_windows_a_date_reaches(monkeypatch) -> None:
    """With the flag on, a date writes its own footprint rather than the run's full set."""
    written = _run_with_footprints(
        monkeypatch,
        dates=["2024-01-01T00:00:00"],
        bboxes_by_date={"2024-01-01T00:00:00": [(0, 0, 1, 1), (0, 0, 1, 1)]},  # reaches 2 of 4
        narrow=True,
    )
    assert written == [("2024-01-01", 2)]


def test_narrowing_off_writes_every_window(monkeypatch) -> None:
    """The flag is the only thing that changes what is written; the skip rule is separate."""
    written = _run_with_footprints(
        monkeypatch,
        dates=["2024-01-01T00:00:00"],
        bboxes_by_date={"2024-01-01T00:00:00": [(0, 0, 1, 1), (0, 0, 1, 1)]},
        narrow=True,
    )
    assert written == [("2024-01-01", 2)]
    written = _run_with_footprints(
        monkeypatch,
        dates=["2024-01-01T00:00:00"],
        bboxes_by_date={"2024-01-01T00:00:00": [(0, 0, 1, 1), (0, 0, 1, 1)]},
        narrow=False,
    )
    assert written == [("2024-01-01", 4)], "narrowing off must write the full window set"


def test_an_unmatched_timestamp_falls_back_to_every_window(monkeypatch) -> None:
    """THE safety property. A slice whose timestamp matches no captured group must write
    EVERYTHING, never nothing — narrowing to a footprint we cannot verify would silently
    drop imagery, which is the one failure nothing downstream would catch.
    """
    written = _run_with_footprints(
        monkeypatch,
        dates=["2024-01-01T00:00:00"],
        # Items sit at a different instant, so no group keys to the loaded slice.
        bboxes_by_date={"2024-01-03T09:15:00": [(0, 0, 1, 1)]},
        narrow=True,
    )
    assert written == [("2024-01-01", 4)], f"unmatched slice must not be narrowed or skipped: {written}"
