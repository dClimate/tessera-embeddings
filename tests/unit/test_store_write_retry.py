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
"""

from __future__ import annotations

import inspect
import logging

import icechunk
import pytest

from tessera_embeddings.errors import DuplicateDateError, InconclusiveStoreProbeError
from tessera_embeddings.ingest import s1_roi, s2_roi
from tessera_embeddings.storage import zarr_store
from tessera_embeddings.storage.zarr_store import (
    CONCURRENT_WRITER_ERRORS,
    STORE_WRITE_ATTEMPTS,
    store_write_retrying,
)

log = logging.getLogger(__name__)


def _attempts(exc: BaseException | None, *, fail_times: int = 99) -> int:
    """Run the policy against a callable that raises ``exc``; return attempts made.

    Sleeps are stubbed out — the policy's backoff is deliberately seconds long, and
    waiting it out would make this file the slowest in the suite for no added coverage.
    """
    calls = 0
    retrying = store_write_retrying(log)
    retrying.sleep = lambda _seconds: None  # type: ignore[method-assign]
    for attempt in retrying:
        with attempt:
            calls += 1
            if exc is not None and calls <= fail_times:
                raise exc
    return calls


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
    """ "Could not tell" is not "safe to delete".

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
