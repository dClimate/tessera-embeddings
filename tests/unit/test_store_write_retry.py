"""The one retry policy every ingest store write uses.

The policy has to make two opposite calls correctly. A transient failure — throttling,
a credential rolling over, a flaky COG read — must be retried, because a failed write
commits nothing and the next attempt starts from clean committed state. A **second
writer** must NOT be, because the ingest path commits without ``rebase_with`` precisely
so that a concurrent commit is refused, and a retry re-opens the session from the tip
that writer moved — which turns the refusal back into a success and lets two writers
interleave dates onto one axis.

That distinction was missing from all three write sites at once (S1 per-date, S2
per-date, S2 per-batch), each of which had built its own ``Retrying``. Hence the shared
factory, and hence the last test here: an inlined fourth copy would silently reopen the
hole, so the absence of one is asserted rather than assumed.

A third call sits on top of the two: a **source provider that stops serving reads** is
resolved by waiting and by nothing else, and the attempt limit is far too short to be that
wait. A caller that can recognise such a failure passes ``wait_out`` and buys a bounded
backoff budget for it alone — so these tests have to pin both directions, since a budget
that leaked onto a failure which recomputes would spend minutes per date to reach the same
verdict, and one that never engaged would be the defect it exists to fix.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable

import icechunk
import pytest

from tessera_embeddings.errors import DuplicateDateError, InconclusiveStoreProbeError
from tessera_embeddings.ingest import s1_roi, s2_roi
from tessera_embeddings.ingest.duplicates import is_provider_refusal
from tessera_embeddings.ingest.loader_failures import drain_local_refusals, install_capture, refusal_wait_out
from tessera_embeddings.storage import zarr_store
from tessera_embeddings.storage.zarr_store import (
    CONCURRENT_WRITER_ERRORS,
    STORE_WRITE_ATTEMPTS,
    WAIT_OUT_BACKOFF_S,
    store_write_retrying,
)

log = logging.getLogger(__name__)


def _run(
    exc: BaseException | None,
    *,
    fail_times: int = 99,
    wait_out: Callable[[BaseException], bool] | None = None,
    while_attempting: Callable[[], None] | None = None,
) -> tuple[int, float, BaseException | None]:
    """Run the policy against a callable that raises ``exc``.

    Returns the attempts made, the backoff slept, and whatever the policy finally re-raised.

    Sleeps are counted rather than taken — the policy's backoff is deliberately seconds long and
    the wait-out budget is minutes long, so taking either would make this file the slowest in the
    suite for no added coverage. Counting works because the budget is accumulated BACKOFF and not
    wall clock: a no-op sleep still spends it, where a wall-clock budget would spin until the
    clock caught up.
    """
    calls = 0
    slept = 0.0
    retrying = store_write_retrying(log, wait_out=wait_out)

    def _count(seconds: float) -> None:
        nonlocal slept
        slept += seconds

    retrying.sleep = _count  # type: ignore[method-assign]
    try:
        for attempt in retrying:
            with attempt:
                calls += 1
                if while_attempting is not None:
                    while_attempting()
                if exc is not None and calls <= fail_times:
                    raise exc
    except BaseException as raised:  # the policy's verdict is exactly what is under test
        return calls, slept, raised
    return calls, slept, None


def _attempts(exc: BaseException | None, *, fail_times: int = 99) -> int:
    """Attempts the default policy makes, re-raising whatever it gave up on."""
    calls, _slept, raised = _run(exc, fail_times=fail_times)
    if raised is not None:
        raise raised
    return calls


def _exhausted(
    exc: BaseException,
    *,
    wait_out: Callable[[BaseException], bool],
    while_attempting: Callable[[], None] | None = None,
) -> tuple[int, float]:
    """Attempts and backoff spent on a failure that never clears.

    Asserts the re-raise rather than suppressing it: a spent budget must fail the write. A policy
    that swallowed the failure would satisfy every attempt count here while losing the date.

    ``while_attempting`` runs inside each attempt, which is where GDAL writes its log line. An
    outage states its refusal on every attempt it refuses, not once before the write begins.
    """
    calls, slept, raised = _run(exc, wait_out=wait_out, while_attempting=while_attempting)
    assert type(raised) is type(exc), "an exhausted policy must re-raise the failure, not absorb it"
    return calls, slept


def _conflict() -> icechunk.ConflictError:
    """The error icechunk raises when another writer has moved the branch tip."""
    return icechunk.ConflictError("SNAPSHOTEXPECTED0000", "SNAPSHOTACTUAL000000")


def test_transient_failure_is_retried_to_the_attempt_limit():
    with pytest.raises(RuntimeError, match="throttled"):
        _attempts(RuntimeError("throttled"))
    assert _attempts(RuntimeError("throttled"), fail_times=STORE_WRITE_ATTEMPTS - 1) == STORE_WRITE_ATTEMPTS


def test_transient_failure_recovers_on_a_later_attempt():
    """Not merely "no exception": the attempt count proves the retry actually re-ran."""
    assert _attempts(RuntimeError("flaky COG"), fail_times=1) == 2


def test_a_moved_branch_tip_is_never_retried():
    """Retrying re-reads the tip the other writer moved, so a second attempt could
    SUCCEED — writing into a store another process owns. Exactly one attempt.
    """
    with pytest.raises(icechunk.ConflictError):
        _attempts(_conflict())
    calls = 0
    retrying = store_write_retrying(log)
    retrying.sleep = lambda _seconds: None  # type: ignore[method-assign]
    with pytest.raises(icechunk.ConflictError):
        for attempt in retrying:
            with attempt:
                calls += 1
                raise _conflict()
    assert calls == 1


def test_a_duplicate_date_is_never_retried():
    """The same collision seen a batch later: the other writer committed the date this
    one was about to write. Retrying cannot help and buries the cause a level deeper.
    """
    calls = 0
    retrying = store_write_retrying(log)
    retrying.sleep = lambda _seconds: None  # type: ignore[method-assign]
    with pytest.raises(DuplicateDateError):
        for attempt in retrying:
            with attempt:
                calls += 1
                raise DuplicateDateError("date 2021-04-12 is already on the time axis")
    assert calls == 1


def test_every_failure_surfaces_as_itself_not_as_a_retryerror():
    """Both paths out of the policy must preserve the exception type: the excluded ones
    because tenacity re-raises what the predicate declined, the retried ones because of
    ``reraise=True``. A ``RetryError`` escaping here would break every
    ``except icechunk.ConflictError`` and every ``except ValueError`` upstream.
    """
    with pytest.raises(icechunk.ConflictError):
        _attempts(_conflict())
    with pytest.raises(DuplicateDateError):
        _attempts(DuplicateDateError("dup"))
    with pytest.raises(RuntimeError, match="throttled"):
        _attempts(RuntimeError("throttled"))


def _refusal() -> Exception:
    """A refused source read, shaped the way the driver receives one.

    tblib cannot rebuild rasterio's GDAL-backed classes across the worker boundary, so what
    arrives is a plain exception carrying the original's text.
    """
    exc = RuntimeError("WarpOperationError: Chunk and warp failed")
    exc.__cause__ = RuntimeError("RasterioIOError: AccessDenied on asf-cumulus-prod-opera-products")
    return exc


def test_a_failure_the_caller_waits_out_outlasts_the_attempt_limit():
    """The whole point. A provider that stops serving reads does not come back inside three
    attempts of exponential backoff, so the attempt limit is not a short budget for this failure —
    it is the wrong instrument, and the write has to be allowed to keep asking.
    """
    calls, _slept = _exhausted(_refusal(), wait_out=is_provider_refusal)
    assert calls > STORE_WRITE_ATTEMPTS


def test_the_wait_out_budget_bounds_the_backoff_it_authorises():
    """A longer wait that is not BOUNDED is an outage of its own — a leg holds a Dask fleet while
    it waits, and thousands of them wait at once. The ladder is exponential and capped, so the
    budget converts into a small attempt count rather than a tight loop against a struggling
    provider.
    """
    calls, slept = _exhausted(_refusal(), wait_out=is_provider_refusal)
    assert slept <= WAIT_OUT_BACKOFF_S
    assert slept > WAIT_OUT_BACKOFF_S / 2, "the budget must be nearly spent, or the ladder wastes it"
    assert calls < 2 * STORE_WRITE_ATTEMPTS * 5, f"{calls} attempts is a tight loop, not a backoff"


def test_a_failure_the_predicate_declines_gets_only_the_attempt_limit():
    """The negative control, and the reason the long wait is gated on a predicate at all.

    A codec failure is a statement about the bytes: it recomputes to the same verdict on every
    attempt, so waiting cannot change it and the entire cost falls on a path that was always going
    to fail. Passing ``wait_out`` must not widen what gets waited out — only the failures the
    predicate positively recognises.
    """
    exc = RuntimeError("RasterioIOError: ZIPDecode: Decoding error at scanline 0")
    assert _exhausted(exc, wait_out=is_provider_refusal)[0] == STORE_WRITE_ATTEMPTS


def test_a_cause_that_was_stripped_in_transit_gets_no_long_wait():
    """The blind case, which must fail SHORT rather than expensively.

    A read failure whose cause did not survive the Dask hop arrives as the bare wrapper, and a
    refusal is unrecognisable in it. That must not draw the long wait on suspicion — the leg fails
    on its attempt limit and retries, which costs a leg attempt instead of ten idle minutes per
    date, and it must not be given up either.
    """
    assert (
        _run(RuntimeError("WarpOperationError: Chunk and warp failed"), wait_out=is_provider_refusal)[0]
        == STORE_WRITE_ATTEMPTS
    )


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(RuntimeError("RasterioIOError: ExpiredToken: the token has expired"), id="our-own-credential"),
        pytest.param(RuntimeError("IcechunkError: AccessDenied writing the mosaic"), id="the-destination-store"),
    ],
)
def test_a_refusal_that_is_not_the_sources_gets_no_long_wait(exc):
    """Both exclusions the predicate exists to make, priced.

    A credential fault on this side is repairable here and no waiting fixes it. A refusal from the
    DESTINATION store is not the source provider's at all — and it says ``AccessDenied`` in the
    same words the source does, so only the source reader's own vocabulary separates them. Either
    one must fail the leg on its first date instead of holding a fleet idle first.
    """
    assert _exhausted(exc, wait_out=is_provider_refusal)[0] == STORE_WRITE_ATTEMPTS


class TestArmingThePatienceOnEvidenceTheChainDoesNotCarry:
    """Whether the budget engages for the failure it was written for.

    A refused object comes back as an error document where the imagery should be, so the codec
    raises a decode failure and the refusal is stated only in GDAL's own log. A predicate reading
    the exception alone therefore declines the very outage the budget exists to outlast, and the
    write spends three attempts on it — which is the ordinary limit, reached in seconds.

    Both directions are pinned, because widening what gets waited out is expensive in its own
    right: a codec failure recomputes to the same verdict, and every second spent on it is a
    fleet held idle to reach the answer it already had.
    """

    @pytest.fixture(autouse=True)
    def _capture(self):
        install_capture()
        gdal = logging.getLogger("rasterio._env")
        previous = gdal.level
        gdal.setLevel(logging.WARNING)
        drain_local_refusals()
        yield
        gdal.setLevel(previous)
        drain_local_refusals()

    @staticmethod
    def _decode_failure() -> BaseException:
        exc = RuntimeError("WarpOperationError: Chunk and warp failed")
        exc.__cause__ = RuntimeError("RasterioIOError: ZIPDecode: Decoding error at scanline 0")
        return exc

    #: The wording GDAL really used, from CloudWatch during the 2026-08-24 outage. The composed
    #: "CPLE_AWSAccessDenied in HTTP response code: 403" that stood here is the rasterio
    #: EXCEPTION class's phrasing and is never written to a log.
    REFUSAL = (
        "CPLE_AppDefined in HTTP response code on "
        "https://asf-cumulus-prod-opera-products.s3.us-west-2.amazonaws.com/OPERA_L2_RTC-S1/"
        "OPERA_L2_RTC-S1_T072-152803-IW2_20211108T150433Z_S1B_30_v1.0_VV.tif: 403"
    )

    def test_a_refusal_only_gdal_logged_outlasts_the_attempt_limit(self):
        """The refusal is stated on every attempt it refuses, which is when GDAL logs it.

        The predicate considers the evidence of the attempt that just failed, so a line from
        before the write began is not this write's evidence — see the control below.
        """
        calls, slept = _exhausted(
            self._decode_failure(),
            wait_out=refusal_wait_out(None),
            while_attempting=lambda: logging.getLogger("rasterio._env").warning(self.REFUSAL),
        )
        assert calls > STORE_WRITE_ATTEMPTS
        assert slept <= WAIT_OUT_BACKOFF_S

    def test_a_line_left_over_from_an_earlier_write_buys_nothing(self):
        """A refusal that has since recovered is not evidence about this failure.

        Unbounded, it would hand an unrelated codec failure the whole refusal budget: a fleet
        held idle for minutes to reach the verdict it already had.
        """
        logging.getLogger("rasterio._env").warning(self.REFUSAL)
        time.sleep(0.05)
        assert _exhausted(self._decode_failure(), wait_out=refusal_wait_out(None))[0] == STORE_WRITE_ATTEMPTS

    def test_the_same_failure_with_nothing_logged_gets_only_the_attempt_limit(self):
        """The control. The predicate must complete a reason, never invent one."""
        assert _exhausted(self._decode_failure(), wait_out=refusal_wait_out(None))[0] == STORE_WRITE_ATTEMPTS


def test_a_moved_branch_tip_is_never_retried_even_when_waiting_something_out():
    """The exclusion outranks the budget. Otherwise the one error that must never be retried
    would be retried for as long as the budget allows, on the caller that opted into it.
    """
    calls = 0
    retrying = store_write_retrying(log, wait_out=is_provider_refusal)
    retrying.sleep = lambda _seconds: None  # type: ignore[method-assign]
    with pytest.raises(icechunk.ConflictError):
        for attempt in retrying:
            with attempt:
                calls += 1
                raise _conflict()
    assert calls == 1


def test_concurrent_writer_errors_names_both_collision_shapes():
    assert icechunk.ConflictError in CONCURRENT_WRITER_ERRORS
    assert DuplicateDateError in CONCURRENT_WRITER_ERRORS


@pytest.mark.parametrize("module", [s1_roi, s2_roi], ids=["s1_roi", "s2_roi"])
def test_no_ingest_module_builds_its_own_retry_policy(module):
    """The regression that mattered: three hand-built policies, none of them excluding a
    second writer. A new ``Retrying(`` in either module is that hole reopening.
    """
    source = inspect.getsource(module)
    assert "Retrying(" not in source, (
        f"{module.__name__} constructs its own Retrying — use store_write_retrying() so the "
        "concurrent-writer exclusion cannot be omitted"
    )
    assert "store_write_retrying" in source


def test_a_losing_racer_never_deletes_the_winners_store(tmp_path, monkeypatch):
    """The gap between the empty-store CHECK and the COMMIT.

    Two first-date writers can both pass the empty-store probe. The loser then fails at
    ``session.commit()`` with ``ConflictError`` — which is not ``StoreHoldsCommittedDataError``,
    so it used to fall through to ``cleanup_on_failure``'s delete and destroy the winner's
    committed data. A conflict is positive evidence the prefix is not ours.
    """
    deleted: list[str] = []
    monkeypatch.setattr(zarr_store, "_delete_store", lambda p, **k: deleted.append(p))

    @zarr_store.cleanup_on_failure
    def _loser(store_path: str) -> None:
        raise icechunk.ConflictError("expected-parent", "actual-parent")

    with pytest.raises(icechunk.ConflictError):
        _loser(str(tmp_path / "mosaic.icechunk"))
    assert deleted == [], "a conflicted writer must not delete the store it lost the race for"


def test_an_ordinary_failure_still_cleans_up(tmp_path, monkeypatch):
    """The complementary half — otherwise the guard above would strand every partial store."""
    deleted: list[str] = []
    monkeypatch.setattr(zarr_store, "_delete_store", lambda p, **k: deleted.append(p))

    @zarr_store.cleanup_on_failure
    def _broken(store_path: str) -> None:
        raise RuntimeError("half-written")

    with pytest.raises(RuntimeError):
        _broken(str(tmp_path / "mosaic.icechunk"))
    assert deleted == [str(tmp_path / "mosaic.icechunk")]


def test_an_unanswered_emptiness_probe_never_deletes(tmp_path, monkeypatch):
    """A probe that could not tell is not a probe that said "safe to delete".

    `_write_new`'s probe reads the network, so a transient failure — or a decode error
    while inspecting a repo another writer is creating — is ordinary. Reaching
    `cleanup_on_failure`'s generic handler on that basis would erase whatever is at the
    prefix, possibly another writer's committed store. Deletion needs POSITIVE evidence.
    """
    deleted: list[str] = []
    monkeypatch.setattr(zarr_store, "_delete_store", lambda p, **k: deleted.append(p))

    @zarr_store.cleanup_on_failure
    def _probe_failed(store_path: str) -> None:
        raise InconclusiveStoreProbeError("network said no")

    with pytest.raises(InconclusiveStoreProbeError):
        _probe_failed(str(tmp_path / "mosaic.icechunk"))
    assert deleted == [], "an unanswered probe must not authorise a delete"
