"""The S1 batch loop: how batches tile the window, and why a look-ahead cannot write a date twice.

Preparing the next batch's catalogue query during the current batch's writes is
``_pipeline.pipelined``, whose overlap and stall behaviour ``test_pipeline.py`` covers.

**It cannot write a date twice.** The serial loop re-read ``get_existing_dates`` from the
store on every batch so that a date written by an earlier batch was skipped by a later one.
A look-ahead makes that read unsafe — the next query is built before the current batch has
committed — so the write loop tracks written dates in-process and is the authority. This
matters because batches are cut on UTC dates while the loader groups by SOLAR day, so an
acquisition late on a batch's last UTC day can belong to the next batch's solar day, and
two consecutive batches can then contain the same day. Writing it twice would trip the
store's duplicate-date guard part-way through a run.

The tests drive the real ``ingest_s1_roi_sar`` loop against a faked catalogue and store,
because the rule they pin lives in that loop rather than in the pipeline.
"""

from __future__ import annotations

import datetime as dt
import json
from datetime import date, datetime, timedelta
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from tessera_embeddings.ingest import live_windows as live_windows_mod
from tessera_embeddings.ingest import s1_roi
from tessera_embeddings.ingest.live_windows import WINDOW_COST_IN_CHUNKS, WINDOW_COST_IN_CHUNKS_OVERLAPPED
from tessera_embeddings.ingest.opera_query import BURST_CENSUS_ATTR, BURST_ID
from tessera_embeddings.ingest.provenance import stamp_fusion_positions

# Every S1 leg these tests run is kept off disk and network the same way.
pytestmark = pytest.mark.usefixtures("s1_leg_offline")


def _fake_client():
    """A stand-in cluster that answers the read-failure rescue check.

    The leg now refuses to start unless every worker confirms the rescues are installed, so a
    client that ignored ``run`` would assert a fleet state nothing had established. ``run`` really
    calls the function here, which reports the in-process install.
    """
    return type(
        "C",
        (),
        {
            "persist": staticmethod(lambda x: x),
            "register_plugin": staticmethod(lambda _plugin: None),
            "run": staticmethod(lambda fn, *a, **k: {"inproc://fake": fn(*a, **k)}),
        },
    )()


def _burst(day: str) -> SimpleNamespace:
    """One radar granule acquired mid-morning UTC on ``day``, as the provider would hand it over."""
    return SimpleNamespace(
        id=f"OPERA_L2_RTC-S1_T001-000001-IW1_{day.replace('-', '')}T100000Z_X_S1A_30_v1.0",
        datetime=datetime.fromisoformat(f"{day}T10:00:00+00:00"),
        properties={"datetime": f"{day}T10:00:00Z", BURST_ID: "T001-000001-IW1"},
        assets={},
        geometry=None,
        bbox=None,
    )


def _run_ingest(
    monkeypatch,
    catalogue: list[str],
    existing: set[str],
    providers: list | None = None,
    writes: list | None = None,
    recorded_census: dict | None = None,
    **kwargs,
) -> list[str]:
    """Drive ``ingest_s1_roi_sar`` against a faked catalogue and store.

    ``catalogue`` is every solar date the catalogue holds; the fake returns the ones
    inside whatever range each batch asks for, which is what a real catalogue does.
    Returns the dates actually written, in order. ``providers``, when given, collects the
    keyword arguments each batch built its item provider with; ``writes`` the keyword
    arguments of each date's write. ``recorded_census`` is what the store's root attr holds.

    That range is deliberately WIDER than the batch owns — ``fixed_day_ranges`` pads it a
    day either side so a solar day straddling a boundary is complete for whichever batch
    owns it. A boundary day therefore comes back from two consecutive queries, and this
    fake reproduces that faithfully: it stands in for ``ingest_tile``, which is upstream
    of the ownership filter, so what these tests exercise is the write loop's in-process
    dedup — the backstop behind ownership. Ownership itself is covered in
    test_solar_days.

    Writes go a date at a time through ``write_day_windows`` — the only writer this
    module has now that cropping is unconditional and the whole-batch ``write_dataset``
    branch is gone.
    """
    written: list[str] = []

    def fake_ingest_tile(
        *, start_date: str, end_date: str, existing_dates=None, item_provider_fn=None, source_index=False, **_
    ):
        if source_index:
            # What the real loader does with the record on: fuse the provider's items, stamped.
            fused = item_provider_fn()
            stamp_fusion_positions(fused, fused)
        dates = [d for d in catalogue if start_date <= d <= end_date]
        if not dates:
            return None, {}
        ds = xr.Dataset(
            {"VV": (("time", "y", "x"), np.zeros((len(dates), 1, 1), dtype="float32"))},
            coords={"time": np.array(dates, dtype="datetime64[ns]")},
        )
        return ds, {}

    def fake_write_day_windows(_store, data, *_args, **kwargs):
        written.append(str(data["time"].values[0])[:10])
        if writes is not None:
            writes.append(kwargs)

    monkeypatch.setattr(s1_roi, "ingest_tile", fake_ingest_tile)
    monkeypatch.setattr(s1_roi, "write_day_windows", fake_write_day_windows)
    monkeypatch.setattr(s1_roi, "get_existing_dates", lambda _store, **_kw: set(existing))
    monkeypatch.setattr(s1_roi, "get_root_attr", lambda _store, _key, **_kw: recorded_census)

    def fake_provider(_orbit, _bbox, query_start, query_end, **kwargs):
        if providers is not None:
            providers.append(kwargs)
        return lambda: [_burst(d) for d in catalogue if query_start <= d <= query_end]

    monkeypatch.setattr(s1_roi, "make_s1_item_provider", fake_provider)

    s1_roi.ingest_s1_roi_sar(
        roi_zarr_path="s3://bucket/roi.zarr",
        # Defaults unless a test names them — which is how the resume cases set a window that
        # straddles what the store already holds.
        start_date=kwargs.pop("start_date", "2024-01-01"),
        end_date=kwargs.pop("end_date", "2024-01-04"),
        store_path="s3://bucket/mosaics",
        client=_fake_client(),
        orbit="ascending",
        batch_days=kwargs.pop("batch_days", 2),
        **kwargs,
    )
    return written


#: Both supply modes. The dedupe guard belongs to the LOOP, not to one branch of it, so
#: it must hold whether the next batch is prefetched or fetched serially.
#:
#: This was four cases until cropping became unconditional: the other two exercised the
#: full-extent branch, where `write_dataset` appended a whole batch and a duplicate solar
#: day landed on the time axis silently. That branch no longer exists, so those cases
#: cannot be written — not merely redundant, unreachable.
_WRITE_PATHS = [pytest.param(pipe, id="lookahead" if pipe else "serial") for pipe in (True, False)]


@pytest.mark.parametrize("pipe", _WRITE_PATHS)
def test_a_solar_day_in_two_batches_is_written_once(monkeypatch, pipe: bool) -> None:
    """The rule the in-process written-date set exists to enforce, through the real loop.

    Batches own two days each, so 2024-01-02 is owned by the first and 2024-01-03 begins
    the second — but the queries are padded a day either side, so both of them RETURN
    2024-01-02. That overlap is deliberate and permanent: it is what makes a straddling
    solar day complete for its owner. Committing it twice would put a duplicate timestamp
    on the time axis, so the write loop dedups against what it has already written.
    """
    written = _run_ingest(
        monkeypatch,
        catalogue=["2024-01-01", "2024-01-02", "2024-01-04"],
        existing=set(),
        pipeline_batches=pipe,
    )
    assert written.count("2024-01-02") == 1, f"the shared solar day was written twice: {written}"
    assert written == ["2024-01-01", "2024-01-02", "2024-01-04"]


@pytest.mark.parametrize("pipe", _WRITE_PATHS)
def test_dates_already_in_the_store_are_skipped_on_resume(monkeypatch, pipe: bool) -> None:
    """Seeding from the store is what makes a resumed run skip finished work."""
    written = _run_ingest(
        monkeypatch,
        pipeline_batches=pipe,
        catalogue=["2024-01-01", "2024-01-02", "2024-01-03"],
        existing={"2024-01-01"},
    )
    assert written == ["2024-01-02", "2024-01-03"]


def _run_with_footprints(
    monkeypatch, dates, bboxes_by_date, *, narrow: bool, windows: int = 4, costs: dict | None = None, **ingest_kwargs
):
    """Drive the ingest with a controllable per-date footprint.

    ``bboxes_by_date`` maps a date to the item bboxes reported for it, which is what decides
    the footprint. A date whose items all carry ``None`` has imagery reaching NO window; a
    date with no items at all is UNKNOWN, which is a different case and must fall back.

    Pass ``costs`` to collect the window price each of the two merges was given, under
    the keys ``"run"`` and ``"date"``. Recorded here rather than by re-patching from the
    caller, whose patches this function would otherwise overwrite.
    """
    written: list[tuple[str, int]] = []
    all_windows = [type("W", (), {"y0": i, "y1": i + 1, "x0": 0, "x1": 1})() for i in range(windows)]

    def fake_ingest_tile(*, start_date: str, item_provider_fn=None, **_):
        if item_provider_fn is not None:
            item_provider_fn()  # let the capture wrapper see the items
        # Coordinates at NOON of each solar day, which is what odc returns once items have
        # passed normalize_to_solar_day. Building them from the raw strings instead would
        # model a state production cannot reach — the footprint join keys on the exact
        # timestamp, so the two sides have to agree.
        coords = np.array([f"{d[:10]}T12:00:00" for d in dates], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {"VV": (("time", "y", "x"), np.zeros((len(dates), 1, 1), dtype="float32"))},
            coords={"time": coords},
        )
        return ds, {}

    def fake_items():

        out = []
        for d, boxes in bboxes_by_date.items():
            for b in boxes:
                out.append(
                    type("I", (), {"datetime": dt.datetime.fromisoformat(d).replace(tzinfo=dt.UTC), "bbox": b})()
                )
        return out

    # windows_for_date is real geometry; stub it to express "reaches these many windows"
    # so the test pins the SKIP and NARROW rules rather than re-testing that function.
    def fake_windows_for_date(run_windows, bboxes, _geobox, *, window_cost_in_chunks=None, **_kw):
        if costs is not None:
            costs["date"] = window_cost_in_chunks
        reach = len([b for b in bboxes if b is not None])
        return run_windows[:reach]

    def fake_live_windows_for_mask(*_a, window_cost_in_chunks=None, **_k):
        if costs is not None:
            costs["run"] = window_cost_in_chunks
        return all_windows

    monkeypatch.setattr(s1_roi, "ingest_tile", fake_ingest_tile)
    monkeypatch.setattr(s1_roi, "windows_for_date", fake_windows_for_date)
    monkeypatch.setattr(s1_roi, "make_s1_item_provider", lambda *a, **k: fake_items)
    monkeypatch.setattr(
        s1_roi,
        "write_day_windows",
        lambda _s, data, w, **_k: written.append((str(data["time"].values[0])[:10], len(w))),
    )
    monkeypatch.setattr(s1_roi, "get_existing_dates", lambda _s, **_kw: set())
    monkeypatch.setattr(live_windows_mod, "live_windows_for_mask", fake_live_windows_for_mask)
    monkeypatch.setattr(s1_roi, "solar_grouping_longitude", lambda _roi: 0.0)

    s1_roi.ingest_s1_roi_sar(
        roi_zarr_path="s3://bucket/roi.zarr",
        start_date="2024-01-01",
        end_date="2024-01-02",
        store_path="s3://bucket/mosaics",
        client=_fake_client(),
        orbit="ascending",
        batch_days=5,
        narrow_windows_per_date=narrow,
        **ingest_kwargs,
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


@pytest.mark.parametrize(
    ("overlap", "price"), [(True, WINDOW_COST_IN_CHUNKS_OVERLAPPED), (False, WINDOW_COST_IN_CHUNKS)]
)
def test_per_date_narrowing_is_priced_like_the_run(monkeypatch, overlap, price) -> None:
    """The run's window price must reach the per-date re-merge, not stop at the run.

    Both merges answer the same question — is a window boundary worth the dead area it
    saves? — so pricing them differently across narrowing undoes the calibration for
    every date the footprint narrows. Pinned for S2 in test_s2_date_pipeline and on
    ``windows_for_date`` itself in test_live_windows.
    """
    costs: dict = {}
    _run_with_footprints(
        monkeypatch,
        dates=["2024-01-01T00:00:00"],
        bboxes_by_date={"2024-01-01T00:00:00": [(0, 0, 1, 1)]},
        narrow=True,
        costs=costs,
        overlap_window_writes=overlap,
    )
    assert costs == {"run": price, "date": price}


def test_the_assessed_window_is_recorded_when_a_resume_writes_no_new_date(monkeypatch) -> None:
    """The SAR half of the zero-write repair, and the reason it cannot be left to a retry.

    A run interrupted after its last date commit but before the assessed-window write
    leaves the orbit store complete and unannotated. Every retry then dedupes all of its
    dates away, writes nothing, and — keyed on what this invocation wrote — skips the
    record again. A month the orbit genuinely never saw stays indistinguishable from a
    gap, and ``check_time_window_coverage`` fails the zone-year forever.
    """
    recorded: list[str] = []
    monkeypatch.setattr(s1_roi, "record_assessed_window", lambda path, *_a, **_k: recorded.append(path))

    written = _run_ingest(
        monkeypatch,
        catalogue=["2024-01-01", "2024-01-02"],
        existing={"2024-01-01", "2024-01-02"},
    )

    assert written == [], "every date was already committed; nothing should be rewritten"
    assert recorded == ["s3://bucket/mosaics/sar_ascending.zarr"]


def test_no_assessed_window_is_recorded_when_the_orbit_store_does_not_exist(monkeypatch) -> None:
    """The complement: an orbit with no store has nothing to annotate, and that is unambiguous.

    ``record_assessed_window`` opens and never creates, so an unconditional call here would
    be a warning on every orbit that this zone-window simply has no granules for.
    """
    recorded: list[str] = []
    monkeypatch.setattr(s1_roi, "record_assessed_window", lambda path, *_a, **_k: recorded.append(path))

    written = _run_ingest(monkeypatch, catalogue=[], existing=set())

    assert written == []
    assert recorded == []


def test_an_all_ocean_roi_banks_no_empty_dates(monkeypatch) -> None:
    """An ROI with no live window must ingest nothing rather than commit empty dates.

    With no window there is nowhere to put a pixel, and each date would still be
    COMMITTED — a time slot holding nothing, which `get_existing_dates` then reports as
    ingested, so no later run revisits it. The per-date footprint fallback cannot help:
    the set it falls back to is the empty one. The campaign screens these cells out
    up front; the public ROI path has no such preflight.
    """
    written: list[str] = []
    queried: list[str] = []

    def fake_ingest_tile(**_kwargs):
        queried.append("query")
        return None, {}

    monkeypatch.setattr(s1_roi, "ingest_tile", fake_ingest_tile)
    monkeypatch.setattr(s1_roi, "write_day_windows", lambda *a, **k: written.append("write"))
    monkeypatch.setattr(s1_roi, "get_existing_dates", lambda _store, **_kw: set())
    monkeypatch.setattr(
        s1_roi,
        "read_roi_metadata",
        lambda *a, **k: SimpleNamespace(bbox_wgs84=(0.0, 0.0, 1.0, 1.0), geobox=object(), crs="EPSG:32615"),
    )
    monkeypatch.setattr(live_windows_mod, "live_windows_for_mask", lambda *a, **k: [])  # all ocean
    monkeypatch.setattr(s1_roi, "solar_grouping_longitude", lambda *a, **k: 0.0)

    result = s1_roi.ingest_s1_roi_sar(
        roi_zarr_path="s3://bucket/ocean.zarr",
        start_date="2024-01-01",
        end_date="2024-01-04",
        store_path="s3://bucket/mosaics",
        client=_fake_client(),
        orbit="ascending",
    )

    assert result.status == "skipped"
    assert result.dates_processed == {"ascending": 0}
    assert written == [], "no date may be committed when nothing can be stored"
    assert queried == [], "and the catalogue need not be queried at all"


def test_a_resume_offers_nothing_at_or_before_the_newest_held_date(monkeypatch) -> None:
    """The radar half of the resume rule.

    2018-05-10 is absent from the axis and below the newest held date. Offering it would build a
    dataset the store then refuses, and that refusal is fatal and unrepairable. Starting the day
    after 2018-05-27 means it is never queried for and never offered.
    """
    written = _run_ingest(
        monkeypatch,
        catalogue=["2018-01-24", "2018-05-10", "2018-06-05"],
        existing={"2018-05-27"},
        start_date="2018-01-01",
        end_date="2018-06-30",
        batch_days=31,
    )

    assert written == ["2018-06-05"], "only days after the newest held one"


def test_a_no_op_resume_still_repairs_the_assessed_window(monkeypatch) -> None:
    """Writing that record is what a resume over a complete store exists to do.

    A leg interrupted between its last date commit and the assessed-window write leaves the orbit
    store complete and unannotated. Returning as soon as there is nothing left to ingest would
    skip the repair on that retry and on every retry after it, and the coverage gate would read a
    month the orbit genuinely never saw as an unexplained gap forever.
    """
    recorded: list[tuple] = []
    monkeypatch.setattr(s1_roi, "record_assessed_window", lambda *a, **_k: recorded.append(a))

    written = _run_ingest(
        monkeypatch,
        catalogue=["2018-05-10"],
        existing={"2018-06-30"},
        start_date="2018-01-01",
        end_date="2018-06-15",
        batch_days=31,
    )

    assert written == []
    assert [a[1:] for a in recorded] == [("2018-01-01", "2018-06-15")]


def test_the_assessed_window_records_the_requested_start_not_the_resumed_one(monkeypatch) -> None:
    """Which days this run queried is not what the store was examined over.

    ``record_assessed_window`` overwrites the attribute with the bounds it is handed, and the
    coverage gate excuses a month holding no dates only while the attribute covers it. Handing it
    the resumed start would retract the earlier leg's assessment of every month beneath the line —
    and an absent month down there would become an unexplained gap that no later run can clear,
    because no later run can write below the line either.
    """
    recorded: list[tuple] = []
    monkeypatch.setattr(s1_roi, "record_assessed_window", lambda *a, **_k: recorded.append(a))

    written = _run_ingest(
        monkeypatch,
        catalogue=["2018-01-24", "2018-06-05"],
        existing={"2018-05-27"},
        start_date="2018-01-01",
        end_date="2018-12-31",
        batch_days=31,
    )

    assert written == ["2018-06-05"], "the resumed start still governs what is queried"
    assert [a[1:] for a in recorded] == [("2018-01-01", "2018-12-31")], (
        "but the record names the window that was asked for, not 2018-05-28"
    )


def test_an_unusable_batch_width_is_refused_whatever_the_store_holds(monkeypatch) -> None:
    """One invalid configuration, one answer — the store's contents must not change the verdict.

    ``fixed_day_ranges`` was the only place enforcing a width of at least one, and a resume over a
    complete store now builds no range and never calls it. Left there, ``batch_days=0`` would have
    raised over the first two stores and reported ``status="skipped"`` over the third, so the same
    misconfigured leg could be recorded as complete depending on how far an earlier attempt got.
    """
    with pytest.raises(ValueError, match="days must be >= 1"):
        _run_ingest(
            monkeypatch,
            catalogue=["2018-05-10"],
            existing={"2018-06-30"},  # past the window, so no range is built
            start_date="2018-01-01",
            end_date="2018-06-15",
            batch_days=0,
        )


def test_a_compact_iso_window_is_not_read_as_closed(monkeypatch) -> None:
    """A window spelled `20180101` must behave exactly like one spelled `2018-01-01`.

    ``date.fromisoformat`` accepts both, but the resume compares strings, and `"20180101"` sorts
    ABOVE `"2018-12-31"` because `"0"` exceeds `"-"`. Taken as handed, the leg reads its own
    window as entirely behind its end and silently writes nothing.
    """
    written = _run_ingest(
        monkeypatch,
        catalogue=["2018-01-24", "2018-06-05"],
        existing={"2018-05-27"},
        start_date="20180101",
        end_date="2018-12-31",
        batch_days=31,
    )

    assert written == ["2018-06-05"], "the open date is written, not skipped"


def test_a_bound_that_is_not_a_date_is_refused(monkeypatch) -> None:
    """Every comparison a resume makes is between strings, and a string can sort without meaning.

    ``"2018-02-30"`` orders before ``"2018-05-27"``, so a store past it would take the no-op path
    and report a misconfigured leg as complete. Before this, ``fixed_day_ranges`` was the only
    thing that parsed the bounds — and a leg with nothing to search for never reaches it.
    """
    with pytest.raises(ValueError):
        _run_ingest(
            monkeypatch,
            catalogue=["2018-06-05"],
            existing={"2018-05-27"},
            start_date="2018-01-01",
            end_date="2018-02-30",
            batch_days=31,
        )


@pytest.mark.parametrize("pipe", _WRITE_PATHS)
def test_one_burst_census_ranks_every_batch_over_the_requested_window(monkeypatch, pipe: bool) -> None:
    """The radar fusion order ranks a burst by its acquisitions over the WHOLE window (ADR 033).

    Built per batch, it would rank a burst by a month. Built over the resumed window — this store
    already holds the first two days, so the search starts on the third — a resumed leg would rank
    bursts differently from the run it resumes, and write different pixels for the same days.
    """
    built: list[tuple] = []

    class _Census:
        def __init__(self, *args, **kwargs) -> None:
            built.append((args, kwargs))

        def record(self) -> dict:
            return {}

    monkeypatch.setattr(s1_roi, "BurstCensus", _Census)
    providers: list[dict] = []
    _run_ingest(
        monkeypatch,
        catalogue=["2024-01-03", "2024-01-05"],
        existing={"2024-01-01", "2024-01-02"},
        providers=providers,
        start_date="2024-01-01",
        end_date="2024-01-06",
        pipeline_batches=pipe,
    )

    assert built == [
        (((0, 0, 1, 1), ("2024-01-01", "2024-01-06"), "ascending"), {"mid_longitude": 0.5, "recorded": None})
    ]
    assert len(providers) > 1, "needs several batches to show they share one census"
    assert all(k["burst_census"] is providers[0]["burst_census"] for k in providers)


@pytest.mark.parametrize(
    ("existing", "recorded", "expected"),
    [
        pytest.param(set(), None, {"T001-000001-IW1": 2}, id="first-leg-counts-and-records"),
        pytest.param(
            {"2024-01-01"},
            {"window": ["2024-01-01", "2024-01-04"], "mid_longitude": 0.5, "counts": {"T001-000001-IW1": 9}},
            {"T001-000001-IW1": 9},
            id="resume-reuses-the-record",
        ),
    ],
)
def test_a_store_holds_the_census_that_ranked_its_dates(monkeypatch, existing, recorded, expected) -> None:
    """Every date's write carries the census as a root attr, and a resumed leg ranks by the one
    its store already holds rather than counting the catalogue again (ADR 033): a backfilled
    acquisition could otherwise flip a tie and fuse later days by a different burst.
    """
    writes: list[dict] = []
    _run_ingest(
        monkeypatch,
        catalogue=["2024-01-01", "2024-01-03"],
        existing=existing,
        writes=writes,
        recorded_census=recorded,
        start_date="2024-01-01",
        end_date="2024-01-04",
    )

    assert writes, "the open date is written"
    record = {"window": ["2024-01-01", "2024-01-04"], "mid_longitude": 0.5, "counts": expected}
    assert all(w["store_attrs"] == {BURST_CENSUS_ATTR: record} for w in writes)


def test_s1_batch_windows_tile_inclusive_range_without_overlap(monkeypatch) -> None:
    """Batches OWN a tiling of the inclusive [start, end] range and QUERY a day past it.

    The overlap between consecutive queries is the point: a solar day straddling a batch cut
    has acquisitions on both sides, so the owning batch asks for the day beyond its own range.
    Ownership, applied before the loader, stops that becoming duplicate work. ``end_date`` is
    inclusive, and the item provider is built for the same window the query uses.
    """
    queried: list[tuple[str, str]] = []
    provided: list[tuple[str, str]] = []

    def fake_ingest_tile(**kwargs):
        queried.append((kwargs["start_date"], kwargs["end_date"]))
        return None, None

    def fake_provider(*args, **_kwargs):
        provided.append((args[2], args[3]))
        return lambda: []

    monkeypatch.setattr(s1_roi, "ingest_tile", fake_ingest_tile)
    monkeypatch.setattr(s1_roi, "make_s1_item_provider", fake_provider)
    monkeypatch.setattr(s1_roi, "get_existing_dates", lambda *_a, **_k: set())

    result = s1_roi.ingest_s1_roi_sar(
        roi_zarr_path="/tmp/roi.zarr",
        start_date="2024-01-01",
        end_date="2024-03-01",
        store_path="/tmp/store",
        client=_fake_client(),
        orbit="ascending",
        batch_days=30,
    )

    assert result.status == "skipped"
    assert queried == [
        ("2023-12-31", "2024-01-31"),
        ("2024-01-30", "2024-03-01"),
        ("2024-02-29", "2024-03-02"),
    ]
    owned = [("2024-01-01", "2024-01-30"), ("2024-01-31", "2024-02-29"), ("2024-03-01", "2024-03-01")]
    covered: list[str] = []
    for own_start, own_end in owned:
        d0, d1 = date.fromisoformat(own_start), date.fromisoformat(own_end)
        covered.extend((d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1))
    start, end = date(2024, 1, 1), date(2024, 3, 1)
    assert covered == [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]
    for (_, prev_end), (next_start, _) in pairwise(queried):
        assert next_start < prev_end, f"the query pad was lost at {prev_end}/{next_start}"
    assert provided == queried


@pytest.mark.parametrize("pipe", _WRITE_PATHS)
def test_each_written_date_carries_its_own_burst_list(monkeypatch, pipe: bool) -> None:
    """With the record on, a date's write carries the list of the bursts its load fused."""
    writes: list = []
    written = _run_ingest(
        monkeypatch,
        catalogue=["2024-01-01", "2024-01-02", "2024-01-04"],
        existing=set(),
        writes=writes,
        pipeline_batches=pipe,
        provenance=True,
    )
    assert written == ["2024-01-01", "2024-01-02", "2024-01-04"]
    for day, kwargs in zip(written, writes, strict=True):
        (granule,) = json.loads(kwargs["granules"])
        assert granule["granule_id"] == _burst(day).id
        assert (granule["image_key"], granule["platform"]) == ("T001-000001-IW1", "S1A")


def test_with_the_record_off_no_radar_write_carries_a_list(monkeypatch) -> None:
    writes: list = []
    _run_ingest(monkeypatch, catalogue=["2024-01-01"], existing=set(), writes=writes)
    assert [kwargs["granules"] for kwargs in writes] == [None]
