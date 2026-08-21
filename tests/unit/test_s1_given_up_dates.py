"""A radar date the source will not hand over must cost that date, not the zone-year.

The rule this pins is a containment rule. Every OPERA read on the radar path happens inside a
date's write, so an exception from a read leaves the per-date loop — and before the bounded skip
it left the whole leg. One refused read therefore cost every LATER date in the window too,
which is how a source that was refusing reads for a few minutes emptied entire zone-years that
had otherwise committed months of sound data.

Three properties, and the third is what keeps the skip honest:

* A date whose source read fails is given up and the leg continues.
* The loss is recorded ON THE STORE, not only logged. An assessed window makes absence inside
  it a finding rather than a gap, so an unrecorded skip is indistinguishable from a day with no
  imagery and nothing downstream ever revisits it.
* Only a failure the SOURCE is answerable for is absorbed. A credential fault on this side, a
  store conflict or a bug still fails the leg, because giving up dates one at a time is the
  wrong response to a cause that repeats on every date.

The tests drive the real ``ingest_s1_roi_sar`` loop against a faked catalogue and store, since
the rule lives in that loop. The write retry is replaced with one that keeps the real attempt
COUNT and drops only the sleeps, so a give-up still happens where production has it — after the
retry is exhausted — without the test paying the backoff.
"""

from __future__ import annotations

import logging

import pytest
from tenacity import Retrying, retry_if_not_exception_type, stop_after_attempt

from tessera_embeddings.storage.zarr_store import CONCURRENT_WRITER_ERRORS, STORE_WRITE_ATTEMPTS

#: The failure shapes a refused OPERA read presents as on the driver, in the proportions a
#: fleet-wide refusal produces them. The warp wrapper dominates because the refusal lands inside
#: a chunk read; the bare forms surface when it lands at open instead.
_REFUSED = [
    pytest.param("WarpOperationError: Chunk and warp failed", "RasterioIOError: HTTP response code: 403", id="warp"),
    pytest.param("RasterioIOError: HTTP response code: 403", None, id="403"),
    pytest.param("AccessDenied: when calling the GetObject operation", None, id="access-denied"),
]


def _raise(message: str, cause: str | None = None) -> Exception:
    """Build a failure the way the driver receives one: a wrapper whose cause carries the reason.

    tblib cannot rebuild rasterio's GDAL-backed exception classes across the worker boundary, so
    what reaches the loop is a plain exception carrying the original's text. Modelling it as
    anything richer would test a shape production never produces.
    """
    exc = RuntimeError(message)
    if cause is not None:
        exc.__cause__ = RuntimeError(cause)
    return exc


def _run(monkeypatch, *, failures: dict[str, Exception], catalogue: list[str], **kwargs):
    """Drive the real loop, failing the write for the dates named in ``failures``.

    Returns ``(result, written, recorded, attempts)``. ``recorded`` collects the keyword
    arguments each ``record_assessed_window`` call was given — the durable half of the skip, and
    the only place a reader of the store can learn a date was lost. ``attempts`` counts write
    calls per date, which is how a test can tell a give-up that waited out the retry from one
    that fired on the first failure.
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


_CATALOGUE = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]


@pytest.mark.parametrize(("message", "cause"), _REFUSED)
def test_a_refused_date_costs_only_itself(monkeypatch, message: str, cause: str | None) -> None:
    """THE containment rule. Every date after the refused one must still be written.

    Without the skip the loop stopped at 2024-01-02 and the two dates after it were never
    attempted — a whole window lost to one refused read.
    """
    result, written, _recorded, _attempts = _run(
        monkeypatch,
        catalogue=_CATALOGUE,
        failures={"2024-01-02": _raise(message, cause)},
    )
    assert written == ["2024-01-01", "2024-01-03", "2024-01-04"]
    assert result.status == "success"
    assert result.dates_processed == {"ascending": 3}


def test_the_loss_is_recorded_on_the_store_with_its_cause(monkeypatch) -> None:
    """A skip nobody can read from the store is a gap, not a finding.

    ``scope`` is the field that decides what to do next: a refused date is recoverable by
    re-running the window, an unreadable one needs a reprocessed copy at the provider.
    """
    _result, _written, recorded, _attempts = _run(
        monkeypatch,
        catalogue=_CATALOGUE,
        failures={"2024-01-02": _raise("AccessDenied: when calling the GetObject operation")},
    )
    assert len(recorded) == 1
    given_up = recorded[0]["unreadable"]
    assert [g["date"] for g in given_up] == ["2024-01-02"]
    assert given_up[0]["scope"] == "provider-refused"
    assert "AccessDenied" in given_up[0]["error"]


def test_an_undecodable_object_is_scoped_apart_from_a_refusal(monkeypatch) -> None:
    """The two causes need different responses, so the record has to tell them apart."""
    _result, _written, recorded, _attempts = _run(
        monkeypatch,
        catalogue=_CATALOGUE,
        failures={"2024-01-02": _raise("WarpOperationError: Chunk and warp failed", "ZIPDecode: error at scanline 0")},
    )
    assert recorded[0]["unreadable"][0]["scope"] == "unreadable"


def test_a_refusal_inside_a_warp_failure_is_scoped_as_a_refusal(monkeypatch) -> None:
    """Both predicates claim a numeric refusal wrapped in a warp failure, so ORDER decides.

    ``is_unreadable_source`` recognises refusals by name, and a bare status code carries none of
    those words, so the wrapper's decode marker is all it sees. Asking about the refusal first
    is what stops a transient 403 from being recorded as imagery that will never read — the
    field an operator would act on, and the wrong action.
    """
    _result, _written, recorded, _attempts = _run(
        monkeypatch,
        catalogue=_CATALOGUE,
        failures={"2024-01-02": _raise("WarpOperationError: Chunk and warp failed", "HTTP response code: 403")},
    )
    assert recorded[0]["unreadable"][0]["scope"] == "provider-refused"


def test_the_retry_is_exhausted_before_a_date_is_given_up(monkeypatch) -> None:
    """Giving up must be the LAST resort, not the first response to a bad minute.

    A refusal that clears inside the write's own retry should cost nothing at all, so the skip
    has to sit OUTSIDE that retry — and the record has to hold one entry for the date rather
    than one per attempt.
    """
    _result, _written, recorded, attempts = _run(
        monkeypatch,
        catalogue=_CATALOGUE,
        failures={"2024-01-02": _raise("RasterioIOError: HTTP response code: 503")},
    )
    assert attempts["2024-01-02"] == STORE_WRITE_ATTEMPTS
    assert len(recorded[0]["unreadable"]) == 1


def test_a_date_offered_by_two_batches_is_given_up_once(monkeypatch) -> None:
    """Batch queries are padded a day either side, so a boundary date arrives twice.

    Attempting it again would list it twice on the store and spend two of the leg's bounded
    budget on one date — which would halve how long an outage a leg can absorb.
    """
    # 2024-01-02 ends the first batch and is returned by the second batch's padded query too.
    _result, _written, recorded, attempts = _run(
        monkeypatch,
        catalogue=_CATALOGUE,
        failures={"2024-01-02": _raise("AccessDenied: refused")},
    )
    assert attempts["2024-01-02"] == STORE_WRITE_ATTEMPTS
    assert [g["date"] for g in recorded[0]["unreadable"]] == ["2024-01-02"]


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError("ExpiredToken: The provided token has expired"), id="our-own-credential"),
        pytest.param(ValueError("time axis is not monotonic"), id="a-bug"),
    ],
)
def test_a_cause_the_source_does_not_own_still_fails_the_leg(monkeypatch, exc: Exception) -> None:
    """Fails closed. These repeat on every date and are repairable here, so absorbing them
    would spend a bounded budget hiding a fault and lose data for it.
    """
    with pytest.raises(type(exc)):
        _run(monkeypatch, catalogue=_CATALOGUE, failures={"2024-01-02": exc})


def test_past_the_ceiling_the_leg_stops_and_records_no_assessed_window(monkeypatch) -> None:
    """A leg losing dates at this rate is in an outage, and stopping is the cheaper response.

    No assessed-window record, deliberately: that attribute says the range was examined in
    full, and writing it from a leg that stopped part-way would excuse the months it never
    reached.
    """
    from tessera_embeddings.ingest import s1_roi

    monkeypatch.setattr(s1_roi, "MAX_GIVEN_UP_DATES", 1)
    with pytest.raises(s1_roi.TooManyGivenUpDatesError, match=r"2 date\(s\).*ceiling of 1"):
        _run(
            monkeypatch,
            catalogue=_CATALOGUE,
            failures={d: _raise("AccessDenied: refused") for d in _CATALOGUE},
        )


def test_the_stop_names_the_dates_and_asks_to_be_re_dispatched(monkeypatch) -> None:
    """The message is the whole interface to an operator, so it has to say which verdict it is.

    A refusal clears, so the dates named here are written by a retry rather than lost — the
    opposite of what an unreadable object deserves, and the difference between re-dispatching
    the leg and investigating the catalogue.
    """
    from tessera_embeddings.ingest import s1_roi

    monkeypatch.setattr(s1_roi, "MAX_GIVEN_UP_DATES", 1)
    with pytest.raises(s1_roi.TooManyGivenUpDatesError) as caught:
        _run(
            monkeypatch,
            catalogue=_CATALOGUE,
            failures={d: _raise("AccessDenied: refused") for d in _CATALOGUE},
        )
    message = str(caught.value)
    assert "RE-DISPATCH" in message
    assert "2024-01-01(provider-refused)" in message
    assert "roi=zone_35N" in message


def test_a_leg_that_gave_up_everything_says_so_at_warning(monkeypatch, caplog) -> None:
    """A leg that writes nothing returns a status the parent reads as success.

    So the count of given-up dates has to reach the log, or a cell finishes green with an orbit
    absent from the store and no line saying the source refused every read.
    """
    with caplog.at_level(logging.WARNING):
        result, written, _recorded, _attempts = _run(
            monkeypatch,
            catalogue=_CATALOGUE,
            failures={d: _raise("AccessDenied: refused") for d in _CATALOGUE},
        )
    assert written == []
    assert result.status == "skipped"
    lines = [r.getMessage() for r in caplog.records if "WROTE NO DATES" in str(r.msg)]
    assert lines, "a leg that wrote nothing must say so"
    assert "given_up=4" in lines[0]
