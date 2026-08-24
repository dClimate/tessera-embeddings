"""A radar date the source will not hand over must cost that date, not the zone-year.

Every OPERA read on the radar path happens inside a date's write, so an exception from a read
leaves the per-date loop — and before this change it left the whole leg. One refused read
therefore cost every LATER date in the window too, which is how a source that refused reads for
thirteen minutes emptied 178 zone-years that had committed months of sound data.

The tests drive the real ``ingest_s1_roi_sar`` loop, since the rule lives in that loop. The write
retry is replaced with one that keeps the real attempt COUNT and drops only the sleeps, so a
give-up still happens where production has it — after the retry is exhausted — without the test
paying the backoff.
"""

from __future__ import annotations

import logging

import pytest
from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt

from tessera_embeddings.ingest import s1_roi
from tessera_embeddings.storage.zarr_store import CONCURRENT_WRITER_ERRORS, STORE_WRITE_ATTEMPTS

_CATALOGUE = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]


def _raise(message: str, cause: str | None = None) -> Exception:
    """A failure shaped the way the driver receives one.

    tblib cannot rebuild rasterio's GDAL-backed classes across the worker boundary, so what
    reaches the loop is a plain exception carrying the original's text.
    """
    exc = RuntimeError(message)
    if cause is not None:
        exc.__cause__ = RuntimeError(cause)
    return exc


def _run(monkeypatch, *, failures: dict[str, Exception], catalogue: list[str] = _CATALOGUE, **kwargs):
    """Drive the real loop, failing the write for the dates named in ``failures``.

    Returns ``(result, written, recorded, attempts)``. ``recorded`` collects the keyword arguments
    each ``record_assessed_window`` call was given — the durable half of the skip, and the only
    place a reader of the store can learn a date was lost.
    """
    import collections

    import numpy as np
    import xarray as xr

    from tessera_embeddings.ingest import s1_roi

    written: list[str] = []
    recorded: list[dict] = []
    attempts: collections.Counter[str] = collections.Counter()

    def fake_ingest_tile(*, start_date: str, end_date: str, **_):
        dates = [d for d in catalogue if start_date <= d <= end_date]
        if not dates:
            return None, {}
        ds = xr.Dataset(
            {"VV": (("time", "y", "x"), np.zeros((len(dates), 1, 1), dtype="float32"))},
            coords={"time": np.array(dates, dtype="datetime64[ns]")},
        )
        return ds, {}

    def fake_write_day_windows(_store, data, *_a, **_k):
        date = str(data["time"].values[0])[:10]
        attempts[date] += 1
        if date in failures:
            raise failures[date]
        written.append(date)

    def no_wait_retrying(_log):
        """The production attempt count and exclusion, without the production backoff."""
        return Retrying(
            stop=stop_after_attempt(STORE_WRITE_ATTEMPTS),
            retry=retry_if_not_exception_type(CONCURRENT_WRITER_ERRORS),
            reraise=True,
        )

    monkeypatch.setattr(s1_roi, "ingest_tile", fake_ingest_tile)
    monkeypatch.setattr(s1_roi, "write_day_windows", fake_write_day_windows)
    monkeypatch.setattr(s1_roi, "store_write_retrying", no_wait_retrying)
    monkeypatch.setattr(s1_roi, "record_assessed_window", lambda *a, **kw: recorded.append(kw))
    monkeypatch.setattr(s1_roi, "get_existing_dates", lambda _s, **_k: set())
    monkeypatch.setattr(s1_roi, "skipped_dates", lambda _s, **_k: set())
    monkeypatch.setattr(s1_roi, "record_skipped_date", lambda *_a, **_k: True)
    monkeypatch.setattr(s1_roi, "apply_roi_mask", lambda data, *a, **k: data)
    monkeypatch.setattr(s1_roi, "read_roi_mask", lambda *a, **k: object())
    monkeypatch.setattr(
        s1_roi,
        "read_roi_metadata",
        lambda _p, **_k: type("M", (), {"geobox": object(), "native_crs": "EPSG:32635", "bbox_wgs84": (0, 0, 1, 1)})(),
    )
    monkeypatch.setattr(
        s1_roi, "live_windows_for_mask", lambda *a, **k: [type("W", (), {"y0": 0, "y1": 1, "x0": 0, "x1": 1})()]
    )
    monkeypatch.setattr(s1_roi, "make_s1_item_provider", lambda *a, **k: lambda: [])
    monkeypatch.setattr(s1_roi.IngestManifest, "from_roi_store", classmethod(lambda _c, _p, **_k: object()))

    result = s1_roi.ingest_s1_roi_sar(
        roi_zarr_path="s3://bucket/zone_35N.zarr",
        start_date="2024-01-01",
        end_date="2024-01-04",
        store_path="s3://bucket/mosaics",
        client=type("C", (), {"persist": staticmethod(lambda x: x)})(),
        orbit="ascending",
        batch_days=2,
        **kwargs,
    )
    return result, written, recorded, attempts


@pytest.mark.parametrize(
    ("message", "cause"),
    [
        pytest.param(
            "WarpOperationError: Chunk and warp failed", "RasterioIOError: HTTP response code: 403", id="warp"
        ),
        pytest.param("RasterioIOError: HTTP response code: 403", None, id="403-at-open"),
        pytest.param("WarpOperationError: Chunk and warp failed", "ZIPDecode: error at scanline 0", id="undecodable"),
    ],
)
def test_a_failed_date_costs_only_itself(monkeypatch, message: str, cause: str | None) -> None:
    """THE containment rule. Every date after the failed one must still be written.

    Without the skip the loop stopped at 2024-01-02 and the two dates after it were never
    attempted — a whole window lost to one refused read.
    """
    result, written, _recorded, _attempts = _run(monkeypatch, failures={"2024-01-02": _raise(message, cause)})
    assert written == ["2024-01-01", "2024-01-03", "2024-01-04"]
    assert result.status == "success"
    assert result.dates_processed == {"ascending": 3}


def test_the_loss_is_recorded_on_the_store_and_scoped_by_cause(monkeypatch) -> None:
    """A skip nobody can read from the store is a gap, not a finding.

    ``scope`` decides what to do next: a refused date is recoverable by re-running the window,
    an unreadable one needs a reprocessed copy at the provider. Both causes here, because the
    record has to tell them apart.
    """
    _r, _w, recorded, _a = _run(
        monkeypatch,
        failures={
            "2024-01-02": _raise("RasterioIOError: HTTP response code: 403"),
            "2024-01-03": _raise("WarpOperationError: Chunk and warp failed", "ZIPDecode: error at scanline 0"),
        },
    )
    given_up = recorded[0]["unreadable"]
    assert [(g["date"], g["scope"]) for g in given_up] == [
        ("2024-01-02", "provider-refused"),
        ("2024-01-03", "unreadable"),
    ]
    assert "403" in given_up[0]["error"]


def test_a_refusal_inside_a_warp_failure_is_scoped_as_a_refusal(monkeypatch) -> None:
    """A numeric refusal wrapped in a warp failure is scoped as a refusal, not as bad data.

    This asked about the refusal FIRST because the two predicates used to overlap and order was
    the only thing separating them. It no longer is: ``is_unreadable_source`` now declines every
    refusal, so the scope is correct however the caller asks. The order here is kept because it
    still reads as the intent, not because the answer depends on it.
    """
    _r, _w, recorded, _a = _run(
        monkeypatch,
        failures={"2024-01-02": _raise("WarpOperationError: Chunk and warp failed", "HTTP response code: 403")},
    )
    assert recorded[0]["unreadable"][0]["scope"] == "provider-refused"


def test_the_retry_is_exhausted_before_a_date_is_given_up(monkeypatch) -> None:
    """Giving up is the LAST resort. A refusal that clears inside the write's own retry costs
    nothing, so the skip sits outside that retry — and the record holds one entry for the date
    rather than one per attempt.

    2024-01-02 also ends the first batch and is offered again by the second batch's padded
    query, so this pins the boundary case too: attempted once, listed once.
    """
    _r, _w, recorded, attempts = _run(
        monkeypatch, failures={"2024-01-02": _raise("RasterioIOError: HTTP response code: 503")}
    )
    assert attempts["2024-01-02"] == STORE_WRITE_ATTEMPTS
    assert [g["date"] for g in recorded[0]["unreadable"]] == ["2024-01-02"]


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError("RasterioIOError: ExpiredToken: the token has expired"), id="our-own-credential"),
        pytest.param(RuntimeError("IcechunkError: AccessDenied"), id="the-destination-store"),
        pytest.param(ValueError("time axis is not monotonic"), id="a-bug"),
    ],
)
def test_a_cause_the_source_does_not_own_still_fails_the_leg(monkeypatch, exc: Exception) -> None:
    """Fails closed. These repeat on every date and are repairable here, so absorbing them would
    spend the budget hiding a fault — and record the loss against the imagery provider.
    """
    with pytest.raises(type(exc)):
        _run(monkeypatch, failures={"2024-01-02": exc})


def test_past_the_ceiling_the_leg_stops_and_asks_to_be_re_dispatched(monkeypatch) -> None:
    """A leg losing dates at this rate is in an outage, and a refusal clears.

    So the dates named in the message are written by a re-dispatch rather than lost, which is why
    stopping beats grinding on. No assessed-window record: that attribute says the range was
    examined in full, and this leg never reached most of it.
    """
    from tessera_embeddings.ingest import s1_roi

    monkeypatch.setattr(s1_roi, "MAX_GIVEN_UP_DATES", 1)
    with pytest.raises(s1_roi.TooManyGivenUpDatesError, match=r"2 date\(s\).*ceiling of 1") as caught:
        _run(monkeypatch, failures={d: _raise("RasterioIOError: HTTP response code: 403") for d in _CATALOGUE})
    message = str(caught.value)
    assert "RE-DISPATCH" in message
    assert "2024-01-01(provider-refused)" in message
    assert "roi=zone_35N" in message


def test_a_leg_that_gave_up_everything_fails_instead_of_reporting_skipped(monkeypatch, caplog) -> None:
    """Giving up every date is data LOSS, and must not be returned as absence.

    ``status="skipped"`` reads to the parent as "the source does not cover this orbit". For
    ``s1_orbit="both"`` that lets the cell finish with an orbit missing and inference run on
    optical alone, which is the outcome this whole path exists to prevent. A warning in the log
    is not a guard: nothing downstream reads it.

    The bar is deliberately narrow. Giving up SOME dates while committing others still
    succeeds -- that is the bounded skip working, and the store carries the record. Only a leg
    that committed nothing at all, so that no store exists to record the loss on, fails here.
    """
    with caplog.at_level(logging.WARNING), pytest.raises(s1_roi.TooManyGivenUpDatesError) as caught:
        _run(monkeypatch, failures={d: _raise("RasterioIOError: HTTP response code: 403") for d in _CATALOGUE})

    assert "gave up every one" in str(caught.value)
    assert "date(s) and committed none" in str(caught.value)
    # The diagnostic still has to be emitted before it raises, or the reason is lost.
    lines = [r.getMessage() for r in caplog.records if "WROTE NO DATES" in str(r.msg)]
    assert lines, "a leg that wrote nothing must still say so"
    assert "given_up=4" in lines[0]


def test_a_leg_that_gave_up_some_dates_and_committed_others_still_succeeds(monkeypatch) -> None:
    """The complement, so the fix above cannot quietly widen into failing the bounded skip."""
    result, written, recorded, _attempts = _run(
        monkeypatch, failures={_CATALOGUE[0]: _raise("RasterioIOError: HTTP response code: 403")}
    )

    assert result.status == "success"
    assert written == _CATALOGUE[1:]
    assert recorded, "the store must carry the record of what was given up"


def test_the_record_of_a_given_up_date_is_marked_load_bearing(monkeypatch) -> None:
    """`record_assessed_window` warns and continues on failure, which is wrong when it is
    the only durable trace of a lost date.

    A swallowed failure there lets the leg succeed, collect a completion marker, and be
    short-circuited by every later run, leaving a hole nothing names. The caller therefore
    has to say the write is required, and only when there is something to lose.
    """
    _result, _written, recorded, _attempts = _run(
        monkeypatch, failures={_CATALOGUE[0]: _raise("RasterioIOError: HTTP response code: 403")}
    )

    assert recorded, "the window must be recorded"
    assert recorded[-1]["required"] is True


def test_a_clean_leg_does_not_make_the_record_load_bearing(monkeypatch) -> None:
    """Nothing was lost, so a failed metadata write must stay a warning rather than fail a leg."""
    _result, _written, recorded, _attempts = _run(monkeypatch, failures={})

    assert recorded
    assert recorded[-1]["required"] is False
