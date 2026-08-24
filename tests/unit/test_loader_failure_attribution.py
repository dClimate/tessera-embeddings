"""Keeping a failed load's evidence: the object it gave up on, and the reason it gave.

Attributing an unreadable read to the object the loader actually gave up on.

The loader names that object in its own log record and raises an exception that does not
carry it. Without the name, the duplicate ladder has to step every duplicated tile-date in
the date, and these tests hold the two properties that makes the difference between:

* **blast radius** — one bad object steps ONE tile-date down, not every tile that happens to
  have an alternate; and
* **termination** — a bad object with no alternate is given up immediately, instead of
  walking every other tile's ladder first at a full re-read of the date per rung.

Both properties are stated against a date shaped like a real wide-ROI one: many duplicated
tiles, one of them broken.

The reason is held in the exception's CAUSE, and the tests for it run the exception through
Dask's own error path rather than a stand-in, because the loss happens inside that path. A
hand-made substitute is what let the loss ship: it looked like the flattened exception the
orchestrator receives without being produced the way the real one is.
"""

from __future__ import annotations

import copyreg
import inspect
import logging
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from distributed.core import clean_exception, error_message
from rasterio._err import CPLE_AppDefinedError, CPLE_BaseError
from rasterio.errors import RasterioIOError, WarpOperationError

from tessera_embeddings.ingest import loader_failures
from tessera_embeddings.ingest.duplicates import (
    alternates_for,
    cause_was_flattened,
    copies_label,
    is_unreadable_source,
    select_preferred_duplicates,
    step_down_copies,
)
from tessera_embeddings.ingest.loader_failures import (
    AbortedReadCapture,
    ReadRescuesNotInstalledError,
    collect_aborted_hrefs,
    drain_local,
    href_key,
    implicated_tile_dates,
    install_capture,
    install_capture_everywhere,
    keep_causes_picklable,
    label_objects,
)

#: The message odc's reader emits for the object it gave up on, verbatim from a fleet log.
ABORT_MESSAGE = (
    "Aborting load due to failure while reading: "
    "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/34/W/FA/2021/9/"
    "S2B_34WFA_20210908_1_L2A/B02.tif:1"
)


class _Asset:
    def __init__(self, href: str) -> None:
        self.href = href


class _Item:
    """Stands in for a pystac Item, carrying only what attribution reads."""

    def __init__(self, ident: str, tile: str, sequence: int, assets: dict[str, str] | None = None) -> None:
        self.id = ident
        # normalize_to_solar_day stamps noon UTC and solar_day_of raises on anything else.
        self.datetime = datetime(2021, 9, 8, 12, 0, 0, tzinfo=UTC)
        self.properties: dict[str, object] = {"grid:code": tile, "s2:sequence": str(sequence)}
        self.assets = {k: _Asset(v) for k, v in (assets or {}).items()}


def _copy(tile: str, sequence: int) -> _Item:
    """One catalogue copy of one tile-date, with the asset href the loader would open."""
    bare = tile.removeprefix("MGRS-")
    zone, band, square = bare[:-3], bare[-3], bare[-2:]
    ident = f"S2B_{bare}_20210908_{sequence}_L2A"
    return _Item(
        ident,
        tile,
        sequence,
        assets={"B02": f"s3://sentinel-cogs/sentinel-s2-l2a-cogs/{zone}/{band}/{square}/2021/9/{ident}/B02.tif"},
    )


def _wide_date(n_duplicated: int = 40) -> list[_Item]:
    """A date shaped like a wide ROI's: many tiles duplicated, 34WFA among them."""
    items: list[_Item] = [_copy("MGRS-34WFA", 0), _copy("MGRS-34WFA", 1)]
    for i in range(n_duplicated):
        tile = f"MGRS-33T{chr(ord('A') + i % 26)}{chr(ord('A') + i // 26)}"
        items.extend([_copy(tile, 0), _copy(tile, 1)])
    return items


class TestNamingTheObject:
    """Turning the loader's log line into an object identity a caller can match."""

    def test_the_href_survives_the_band_suffix(self) -> None:
        """The loader appends ``:1`` for the band; the object's identity does not include it."""
        assert href_key(ABORT_MESSAGE.split(": ", 1)[1]) == "S2B_34WFA_20210908_1_L2A/B02.tif"

    def test_the_three_url_spellings_agree(self) -> None:
        """A catalogue asset, a virtual-hosted URL and a path-style URL name one object.

        If they did not agree, every comparison between what the loader opened and what the
        catalogue offered would silently find nothing.
        """
        key = "S2B_34WFA_20210908_1_L2A/B02.tif"
        prefix = "sentinel-s2-l2a-cogs/34/W/FA/2021/9"
        assert href_key(f"s3://sentinel-cogs/{prefix}/{key}") == key
        assert href_key(f"https://sentinel-cogs.s3.us-west-2.amazonaws.com/{prefix}/{key}") == key
        assert href_key(f"https://s3.us-west-2.amazonaws.com/sentinel-cogs/{prefix}/{key}") == key

    def test_the_label_names_nothing_when_nothing_was_captured(self) -> None:
        """``unattributed`` is a distinct answer from a list of objects, and must read as one."""
        assert label_objects([]) == "unattributed"

    def test_the_label_caps_and_says_it_capped(self) -> None:
        hrefs = [f"s3://b/p/S2B_33TAA_2021090{i}_0_L2A/B02.tif" for i in range(20)]
        label = label_objects(hrefs)
        assert label.endswith("+12 more")
        assert label.count(",") == 8


class TestCapturingWhatTheLoaderReports:
    """The handler that records the loader's own abort messages."""

    def test_the_capture_records_the_href_the_loader_names(self) -> None:
        install_capture()
        drain_local()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        assert [href_key(h) for h in drain_local()] == ["S2B_34WFA_20210908_1_L2A/B02.tif"]

    def test_installing_twice_does_not_double_the_record(self) -> None:
        """The worker plugin's setup runs again on reconnect, and a second handler would
        report every object twice — which reads as two failing objects.
        """
        install_capture()
        install_capture()
        drain_local()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        assert len(drain_local()) == 1

    def test_draining_clears(self) -> None:
        """One failure's attribution must not be inherited by the next."""
        install_capture()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        drain_local()
        assert drain_local() == []

    def test_an_unrelated_error_is_not_captured(self) -> None:
        install_capture()
        drain_local()
        logging.getLogger("odc.loader._rio").error("Something else entirely went wrong")
        assert drain_local() == []

    def test_collection_without_a_cluster_returns_the_local_record(self) -> None:
        """Serial runs have no client, and attribution must still work there."""
        install_capture()
        drain_local()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        assert len(collect_aborted_hrefs(None)) == 1


class TestMappingObjectsBackToCopies:
    """From an aborted href to the ``(tile, solar day)`` whose copy produced it."""

    def test_the_granule_id_in_the_path_is_enough(self) -> None:
        """Covers a catalogue whose asset hrefs are signed, aliased, or simply absent."""
        item = _Item("S2B_34WFA_20210908_1_L2A", "MGRS-34WFA", 1)
        keys = implicated_tile_dates([item], [ABORT_MESSAGE.split(": ", 1)[1]])
        assert keys == {("MGRS-34WFA", "2021-09-08")}

    def test_the_asset_href_matches_across_spellings(self) -> None:
        item = _copy("MGRS-34WFA", 1)
        item.id = "an-id-that-appears-nowhere-in-the-url"
        keys = implicated_tile_dates([item], [ABORT_MESSAGE.split(": ", 1)[1]])
        assert keys == {("MGRS-34WFA", "2021-09-08")}

    def test_an_href_from_another_date_attributes_nothing(self) -> None:
        """A fleet's workers read many dates at once, so a foreign href is expected. Blaming
        a tile for it would step down a copy that read perfectly well.
        """
        assert implicated_tile_dates([_copy("MGRS-34WFA", 1)], ["s3://b/p/S2B_33TAA_20210101_0_L2A/B02.tif"]) == set()

    def test_no_hrefs_attributes_nothing(self) -> None:
        assert implicated_tile_dates([_copy("MGRS-34WFA", 1)], []) == set()


class TestBlastRadius:
    """What a single bad object costs the rest of the date."""

    def test_attribution_steps_down_exactly_the_blamed_tile(self) -> None:
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)
        blamed = implicated_tile_dates(kept, [ABORT_MESSAGE.split(": ", 1)[1]])

        result = step_down_copies(alternates, kept, only=blamed)
        assert result is not None
        stepped_items, swapped = result
        assert swapped == {("MGRS-34WFA", "2021-09-08")}
        # Every other tile keeps the newer copy it was chosen for.
        downgraded = {i.id for a, i in zip(kept, stepped_items, strict=True) if a is not i}
        assert downgraded == {"S2B_34WFA_20210908_0_L2A"}

    def test_without_attribution_the_whole_date_steps_down(self) -> None:
        """The behaviour to be avoided, asserted so the improvement is measurable rather than
        asserted: 41 duplicated tiles all lose their newer copy for one bad object.
        """
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)

        result = step_down_copies(alternates, kept, only=None)
        assert result is not None
        _, swapped = result
        assert len(swapped) == 41

    def test_an_empty_attribution_is_not_a_licence_to_step_everything(self) -> None:
        """Attributed to nothing in this date is a DIFFERENT answer from unattributed, and
        must not fall through to the unattributed behaviour.
        """
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)
        assert step_down_copies(alternates, kept, only=set()) is None

    def test_a_failure_this_date_cannot_claim_is_unattributed_not_attributed_to_nothing(self) -> None:
        """The other half of the pair above, and the one the caller has to get right.

        ``collect_aborted_hrefs`` drains a CLUSTER-WIDE record, so the hrefs it returns can
        belong to another date entirely, or be a stale line from a failure already handled.
        Matching none of them against this date means "could not attribute" — not
        "attributed to nothing". Handing the empty set straight to ``step_down_copies``
        conflates the two, steps down no copy, and records a date with a perfectly good
        older copy as permanently lost. The consume path coerces it to ``None`` so the
        whole-date ladder still runs, which is what it did before attribution existed.
        """
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)
        foreign = ["s3://b/p/S2B_33TAA_20210101_0_L2A/B02.tif"]

        assert implicated_tile_dates(kept, foreign) == set()  # nothing in this date matches
        # What s2_roi._consume passes on, having read that as "unknown":
        blamed = implicated_tile_dates(kept, foreign) or None
        assert step_down_copies(alternates, kept, only=blamed) is not None


class TestTermination:
    """When the date is given up, and how many rungs that takes."""

    def test_a_blamed_tile_with_no_alternate_gives_up_at_once(self) -> None:
        """The expensive case: the broken object is a single-copy tile. Every other tile still
        has alternates, so an unattributed ladder walks them all — each rung a full re-read of
        the date — before reaching the same answer.
        """
        items = [_copy("MGRS-34WFA", 1)]  # one copy only
        for i in range(40):
            tile = f"MGRS-33T{chr(ord('A') + i)}A"
            items.extend([_copy(tile, 0), _copy(tile, 1)])
        kept, alternates = select_preferred_duplicates(items)
        blamed = implicated_tile_dates(kept, [ABORT_MESSAGE.split(": ", 1)[1]])

        assert blamed == {("MGRS-34WFA", "2021-09-08")}
        assert step_down_copies(alternates, kept, only=blamed) is None
        assert step_down_copies(alternates, kept, only=None) is not None

    def test_a_copy_that_failed_is_never_offered_again(self) -> None:
        """The ladder consumes what it hands out, or a retry loop cannot terminate."""
        items = [_copy("MGRS-34WFA", 0), _copy("MGRS-34WFA", 1), _copy("MGRS-34WFA", 2)]
        kept, alternates = select_preferred_duplicates(items)
        blamed = {("MGRS-34WFA", "2021-09-08")}

        first = step_down_copies(alternates, kept, only=blamed)
        assert first is not None
        second = step_down_copies(alternates, first[0], only=blamed)
        assert second is not None
        assert [i.id for i in first[0]] != [i.id for i in second[0]]
        assert step_down_copies(alternates, second[0], only=blamed) is None


class TestWhatTheLogSays:
    """The labels a reader uses to tell what changed."""

    def test_the_label_names_only_what_was_stepped(self) -> None:
        """A whole-date label on a wide ROI runs to tens of thousands of characters and leaves
        the reader to diff two of them to find the one tile that moved.
        """
        items = _wide_date()
        kept, alternates = select_preferred_duplicates(items)
        blamed = {("MGRS-34WFA", "2021-09-08")}
        result = step_down_copies(alternates, kept, only=blamed)
        assert result is not None
        stepped_items, swapped = result

        assert copies_label(kept, only=swapped) == "MGRS-34WFA#1"
        assert copies_label(stepped_items, only=swapped) == "MGRS-34WFA#0"

    def test_a_whole_date_label_is_capped(self) -> None:
        items = _wide_date()
        kept, _ = select_preferred_duplicates(items)
        assert copies_label(kept).endswith("more")

    def test_narrowing_alternates_is_an_intersection_not_a_replacement(self) -> None:
        """``only`` must not be able to reach a tile-date absent from ``items`` — a date's
        ladder reaching into another date's copies would swap imagery for a day that read.
        """
        items = _wide_date(n_duplicated=1)
        kept, alternates = select_preferred_duplicates(items)
        foreign = ("MGRS-99XXX", "2021-09-08")
        assert foreign not in alternates_for(alternates, kept, only={foreign})


#: What GDAL says when libtiff cannot inflate a tile, verbatim from a fleet log. The message a
#: verdict about a failed read is supposed to be reached from.
CODEC_FAILURE = "ZIPDecode:Decoding error at scanline 0, unknown compression method"

#: What rasterio wraps it in, verbatim. It reports that a read failed and nothing about why.
READ_FAILED = "Read failed. See previous exception for details."


def _as_rasterio_ships() -> None:
    """Drop any reducer already installed, leaving the class as the installed wheel defines it."""
    if "__reduce__" in CPLE_BaseError.__dict__:
        del CPLE_BaseError.__reduce__


@pytest.fixture
def gdal_errors_travel_as_shipped() -> Iterator[None]:
    """Start every test from rasterio as shipped, and put back whatever was installed.

    The reducer is an attribute on a third-party class and tblib registers itself in
    ``copyreg``, so both are process-global for the rest of the session. Any earlier test that
    starts an ingest installs the reducer, and a test asserting the loss would then observe a
    rescue it never asked for — which is order-dependent, and which is what happened: the same
    three tests passed alone and failed after a radar leg test in the same worker.

    So establishing the precondition matters as much as restoring afterwards. Restoring alone
    leaves the assertion at the mercy of what ran before it.
    """
    installed = CPLE_BaseError.__dict__.get("__reduce__")
    dispatch = dict(copyreg.dispatch_table)
    _as_rasterio_ships()
    try:
        yield
    finally:
        if installed is None:
            _as_rasterio_ships()
        else:
            CPLE_BaseError.__reduce__ = installed  # type: ignore[method-assign]
        copyreg.dispatch_table.clear()
        copyreg.dispatch_table.update(dispatch)


def _codec_failure_reading_a_tile() -> RasterioIOError:
    """The optical failure, raised the way a reading worker raises it.

    Raised rather than constructed: the cause is attached by ``raise ... from``, which is the
    only thing that puts a chain on an exception, and the chain is the subject here.
    """
    try:
        raise CPLE_AppDefinedError(1, 4, CODEC_FAILURE)
    except CPLE_AppDefinedError as cause:
        try:
            raise RasterioIOError(READ_FAILED) from cause
        except RasterioIOError as failure:
            return failure


def _refusal_behind_the_warp_wrapper() -> WarpOperationError:
    """The radar failure: a provider refusal, three links down under a bare wrapper."""
    try:
        raise CPLE_AppDefinedError(1, 11, "HTTP response code: 503")
    except CPLE_AppDefinedError as cause:
        try:
            raise RasterioIOError(READ_FAILED) from cause
        except RasterioIOError as inner:
            try:
                raise WarpOperationError("Chunk and warp failed") from inner
            except WarpOperationError as failure:
                return failure


def _across_the_worker_boundary(exc: BaseException) -> BaseException:
    """What the orchestrating worker receives for ``exc`` raised on a reading worker.

    Dask's own two halves, unmodified. ``error_message`` runs where the read failed and
    pickles the exception, and — this is the half that matters — it unpickles its own output
    first and substitutes a flattened exception when that fails. So the loss is decided on the
    SENDING side, which is why a reducer installed only where the failure lands is too late.
    """
    message = error_message(exc)
    payload = message["exception"]
    _, arrived, _ = clean_exception(exception=getattr(payload, "data", payload))
    assert arrived is not None
    return arrived


@pytest.mark.usefixtures("gdal_errors_travel_as_shipped")
class TestTheReasonDoesNotSurviveTheWorkerBoundary:
    """What arrives without the reducer — the shape every read verdict was reached from."""

    def test_the_cause_is_replaced_by_the_wrappers_repr(self) -> None:
        """The exact production shape, produced rather than imitated.

        A plain ``Exception`` whose entire message is the outer exception's repr, and no chain
        under it. tblib rebuilds an exception with a custom ``__init__`` by assigning to
        ``args``, GDAL publishes ``args`` read-only, the assignment throws, and Dask falls back
        to this. The GDAL class, its error number and the codec's message are all gone.
        """
        arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())

        assert type(arrived) is Exception
        assert str(arrived) == repr(RasterioIOError(READ_FAILED))
        assert arrived.__cause__ is None
        assert CODEC_FAILURE not in str(arrived)

    def test_the_verdict_is_undecidable(self) -> None:
        """Corrupt bytes, and the classifier cannot say so — nothing in what arrived says it.

        This is the whole defect. The verdict is not wrong, it is absent: the predicate fails
        closed, the caller re-raises, and the leg dies on a date another copy could have served.
        """
        arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())
        assert is_unreadable_source(arrived) is False
        assert cause_was_flattened(arrived) is True

    def test_the_bare_warp_wrapper_arrives_saying_nothing_at_all(self) -> None:
        """The radar shape. Its own message names neither a codec nor a status."""
        arrived = _across_the_worker_boundary(_refusal_behind_the_warp_wrapper())
        assert str(arrived) == repr(WarpOperationError("Chunk and warp failed"))
        assert cause_was_flattened(arrived) is True


@pytest.mark.usefixtures("gdal_errors_travel_as_shipped")
class TestTheReasonSurvivesOnceTheReducerIsInstalled:
    """The same failures, through the same Dask path, with the reducer on the sending side."""

    def test_the_whole_chain_arrives(self) -> None:
        keep_causes_picklable()
        arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())

        assert type(arrived) is RasterioIOError
        assert str(arrived) == READ_FAILED
        assert type(arrived.__cause__) is CPLE_AppDefinedError
        assert str(arrived.__cause__) == CODEC_FAILURE

    def test_the_gdal_error_number_survives_too(self) -> None:
        """Faithful, not merely sufficient: the class is rebuilt through its own constructor
        from its own ``args``, so it carries what it carried rather than a flattened string.
        """
        keep_causes_picklable()
        arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())
        assert arrived.__cause__ is not None
        assert arrived.__cause__.args == (1, 4, CODEC_FAILURE)

    def test_corrupt_bytes_are_now_classified_as_unreadable(self) -> None:
        """The payoff. The same failure that arrived undecidable now decides a step-down."""
        keep_causes_picklable()
        arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())
        assert is_unreadable_source(arrived) is True
        assert cause_was_flattened(arrived) is False

    def test_a_refusal_three_links_down_is_still_not_the_datas_fault(self) -> None:
        """Decidable must not mean permissive. The status is now visible and it says transient,
        so the verdict declines — for a stated reason rather than for want of evidence.
        """
        keep_causes_picklable()
        arrived = _across_the_worker_boundary(_refusal_behind_the_warp_wrapper())
        assert is_unreadable_source(arrived) is False
        assert cause_was_flattened(arrived) is False
        assert "503" in str(arrived.__cause__.__cause__)

    def test_installing_twice_changes_nothing(self) -> None:
        """A worker's plugin setup runs again on reconnect."""
        keep_causes_picklable()
        keep_causes_picklable()
        arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())
        assert type(arrived.__cause__) is CPLE_AppDefinedError

    def test_every_gdal_error_class_is_covered_by_the_one_entry(self) -> None:
        """One entry names the BASE, so no subclass has to be enumerated — including ones a
        later GDAL adds. Asserted over the installed wheel's classes so an upgrade that breaks
        the assumption fails here rather than on a campaign leg.
        """
        keep_causes_picklable()
        classes = [CPLE_BaseError, *_all_subclasses(CPLE_BaseError)]
        assert len(classes) > 1

        for cls in classes:
            arrived = _across_the_worker_boundary(cls(1, 4, CODEC_FAILURE))
            assert type(arrived) is cls, f"{cls.__name__} did not survive"
            assert str(arrived) == CODEC_FAILURE, f"{cls.__name__} lost its message"

    def test_no_other_exception_in_the_read_stack_needs_an_entry(self) -> None:
        """Why the entry is one class and not a list to keep in step with rasterio.

        tblib's assignment only fails where ``args`` is a read-only property, and every class
        in the loaded read stack with one is a ``CPLE_BaseError``. That is what makes the entry
        complete rather than merely current, so it is checked rather than asserted in prose.
        """
        readonly = []
        for cls in _all_subclasses(BaseException):
            descriptor = inspect.getattr_static(cls, "args", None)
            if isinstance(descriptor, property) and descriptor.fset is None:
                readonly.append(cls)
        assert readonly
        assert all(issubclass(cls, CPLE_BaseError) for cls in readonly), [
            f"{c.__module__}.{c.__qualname__}" for c in readonly if not issubclass(c, CPLE_BaseError)
        ]


def _all_subclasses(cls: type) -> list[type]:
    """Every direct and indirect subclass of ``cls`` currently loaded."""
    found: dict[int, type] = {}
    todo = list(cls.__subclasses__())
    while todo:
        current = todo.pop()
        if id(current) in found:
            continue
        found[id(current)] = current
        todo += current.__subclasses__()
    return list(found.values())


@pytest.mark.usefixtures("gdal_errors_travel_as_shipped")
class TestBothRescuesReachTheWorker:
    """The wiring. Each rescue needs code running on the reader BEFORE the read fails, and
    the plugin is the only thing that reaches a worker that joined after the ingest started.
    """

    def test_the_plugin_installs_the_reducer_as_well_as_the_capture(self) -> None:
        AbortedReadCapture().setup(worker=None)

        arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())
        assert type(arrived.__cause__) is CPLE_AppDefinedError

        drain_local()
        logging.getLogger("odc.loader._rio").error(ABORT_MESSAGE)
        assert len(drain_local()) == 1

    def test_a_serial_run_with_no_cluster_still_gets_both(self) -> None:
        """Reads happen in-process when there is no client, and must be as decidable there."""
        install_capture_everywhere(None)
        arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())
        assert type(arrived.__cause__) is CPLE_AppDefinedError

    @pytest.mark.parametrize(
        ("registers", "answers", "expected"),
        [
            pytest.param(
                OSError("scheduler unreachable"), None, "cannot classify read failures", id="registration-refused"
            ),
            pytest.param(
                None, {"tcp://a": True, "tcp://b": False}, "reported it missing", id="a-worker-says-it-is-missing"
            ),
            pytest.param(None, {"tcp://a": True, "tcp://b": True}, None, id="every-worker-confirms"),
        ],
    )
    def test_a_fleet_that_cannot_classify_refuses_to_start(
        self,
        monkeypatch: pytest.MonkeyPatch,
        registers: Exception | None,
        answers: dict[str, bool] | None,
        expected: str | None,
    ) -> None:
        """USED TO BE TOLERATED, and that was right while these rescues only sharpened attribution.

        The reducer now decides whether a read failure is decidable at all, and an undecidable
        failure strands the whole zone-year on one bad object rather than costing a date. Verified
        rather than assumed, because registration returning is not the worker having run setup.
        """
        # The attempts are what this asserts; their spacing is not, and paying it would put the
        # backoff's wall clock on every run of the failing path.
        monkeypatch.setattr(loader_failures, "_REGISTRATION_BACKOFF_S", 0.0)

        class _Client:
            def register_plugin(self, *_a: object, **_k: object) -> None:
                if registers is not None:
                    raise registers

            def run(self, *_a: object, **_k: object) -> dict[str, bool]:
                assert answers is not None, "must not be reached when registration itself failed"
                return answers

        if expected is None:
            install_capture_everywhere(_Client())  # type: ignore[arg-type]
            arrived = _across_the_worker_boundary(_codec_failure_reading_a_tile())
            assert type(arrived.__cause__) is CPLE_AppDefinedError
            return
        with pytest.raises(ReadRescuesNotInstalledError, match=expected):
            install_capture_everywhere(_Client())  # type: ignore[arg-type]
