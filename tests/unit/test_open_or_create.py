"""``open_or_create_repo`` must create only on PROVEN absence.

Icechunk reports everything through one exception class — in 2.1.1 the exported types are
``IcechunkError``, ``ConflictError`` and ``RebaseFailedError``, nothing narrower — so absence,
an expired credential, a throttle, an IO error and real corruption are indistinguishable by
type. Catching the class wholesale on the open leg therefore reads every failure as "not
there" and creates.

Creating in response to a transient failure is not merely imprecise, it is destructive, and
the reason is worth stating once here because it is the whole point of this module:

1. The create trips Icechunk's clean-prefix rule and surfaces as ``CorruptedStoreError``
   naming a store that is perfectly healthy.
2. Its advice is to delete that store.
3. It is **deterministic** from then on. Callers wrap these writes in a retry that retries
   every exception, so a transient open failure was already survivable — but once the repo
   exists, every retry's create fails identically and the retry cannot escape. A momentary
   blip becomes a hard failure, reported as corruption, with the real error destroyed.

So the tests come in pairs: absence must still create (or the fix has over-tightened into
never creating), and every non-absence failure must propagate untouched with no create
attempted. The messages asserted on are the real ones observed in production, not invented.
"""

from __future__ import annotations

from pathlib import Path

import icechunk
import pytest
import zarr

from tessera_embeddings.storage import zarr_store
from tessera_embeddings.storage.zarr_store import CorruptedStoreError, open_or_create_repo

#: Real messages, from Icechunk and from a real failure. Named so a failure report says
#: which class of transient slipped through rather than just quoting a string.
TRANSIENTS = [
    pytest.param("Error contacting storage: throttled, slow down", id="throttle"),
    pytest.param("failed to get credentials: expired token", id="expired-credential"),
    pytest.param("io error: No space left on device (os error 28)", id="disk-full"),
    pytest.param("error deserializing snapshot: unexpected end of input", id="real-corruption"),
]


@pytest.fixture
def no_create(monkeypatch) -> list[str]:
    """Record any create attempt instead of performing one.

    Asserting the create was never ATTEMPTED is stronger than asserting the raised type:
    a future refactor could preserve the exception while still touching the prefix, and
    touching the prefix is the irreversible half.
    """
    attempted: list[str] = []

    def _spy(store_path: str, *args, **kwargs):
        attempted.append(store_path)
        raise AssertionError(f"create attempted for {store_path!r} after a non-absence failure")

    monkeypatch.setattr(zarr_store, "_create_repo", _spy)
    return attempted


def _open_raises(monkeypatch, exc: BaseException) -> None:
    def _boom(*args, **kwargs):
        raise exc

    monkeypatch.setattr(zarr_store, "open_repo", _boom)


# --- absence still creates -------------------------------------------------------------


def test_a_missing_path_creates(tmp_path: Path) -> None:
    """The golden path, and the guard against the fix over-tightening."""
    repo, is_new = open_or_create_repo(str(tmp_path / "fresh"))
    assert is_new is True
    assert repo.list_branches() == {"main"}


def test_an_existing_repo_opens(tmp_path: Path) -> None:
    path = str(tmp_path / "twice")
    open_or_create_repo(path)
    repo, is_new = open_or_create_repo(path)
    assert is_new is False
    assert repo.list_branches() == {"main"}


@pytest.mark.parametrize("message", ["the repository doesn't exist", "Repository does not exist"])
def test_an_absence_message_creates(tmp_path: Path, monkeypatch, message: str) -> None:
    """Pins the strings ``is_missing_repo`` keys on, at the level that consumes them.

    Both spellings are asserted because the predicate accepts both and an upstream reword
    to either one must keep working.
    """
    _open_raises(monkeypatch, icechunk.IcechunkError(message))
    _, is_new = open_or_create_repo(str(tmp_path / "reported-absent"))
    assert is_new is True


def test_a_file_not_found_creates(tmp_path: Path, monkeypatch) -> None:
    """Zarr's ``GroupNotFoundError`` subclasses ``FileNotFoundError``; both mean absence."""
    _open_raises(monkeypatch, zarr.errors.GroupNotFoundError("nope"))
    _, is_new = open_or_create_repo(str(tmp_path / "no-group"))
    assert is_new is True


# --- everything else propagates, untouched ---------------------------------------------


@pytest.mark.parametrize("message", TRANSIENTS)
def test_a_non_absence_failure_is_reraised_and_nothing_is_created(
    tmp_path: Path, monkeypatch, no_create: list[str], message: str
) -> None:
    """THE regression test. Before the fix each of these returned a created repo, and
    against a non-empty prefix a ``CorruptedStoreError`` blaming a healthy store.
    """
    _open_raises(monkeypatch, icechunk.IcechunkError(message))
    with pytest.raises(icechunk.IcechunkError):
        open_or_create_repo(str(tmp_path / "transient"))
    assert no_create == []


def test_the_original_exception_object_propagates(tmp_path: Path, monkeypatch) -> None:
    """Re-raised, not re-wrapped: the caller's retry and any log scraping see the real error.

    Identity, not type — a helpful-looking wrap would still lose the icechunk detail that
    says which transient it was.
    """
    original = icechunk.IcechunkError("Error contacting storage: throttled, slow down")
    _open_raises(monkeypatch, original)
    with pytest.raises(icechunk.IcechunkError) as caught:
        open_or_create_repo(str(tmp_path / "identity"))
    assert caught.value is original


def test_a_create_failure_is_not_caught_by_this_function(tmp_path: Path, monkeypatch) -> None:
    """Pins the control-flow shape: the create runs OUTSIDE the try.

    If it were inside, an error from the create leg could be caught by this function's own
    handlers and re-attributed — the same class of confusion the fix removes.
    """
    sentinel = icechunk.IcechunkError("the repository doesn't exist")
    _open_raises(monkeypatch, sentinel)

    def _create_boom(*args, **kwargs):
        raise RuntimeError("create failed for its own reasons")

    monkeypatch.setattr(zarr_store, "_create_repo", _create_boom)
    with pytest.raises(RuntimeError, match="its own reasons"):
        open_or_create_repo(str(tmp_path / "create-fails"))


# --- the rootless store, and the dirty prefix that remains legitimate -------------------


def test_a_rootless_repo_opens_rather_than_creates(tmp_path: Path) -> None:
    """A repo created but never given a root group must OPEN.

    This is the case ``write_days_windows``' seeding probe depends on: the probe fires for a
    missing repo and for a rootless one alike, and it relies on this function's open leg
    winning for the rootless one. Creating there would hit the clean-prefix rule and wedge
    the very crash window the recovery exists for.
    """
    path = str(tmp_path / "rootless")
    open_or_create_repo(path)  # creates the repo; no schema is ever committed
    repo, is_new = open_or_create_repo(path)
    assert is_new is False
    assert repo.list_branches() == {"main"}


def test_a_dirty_prefix_still_raises_corrupted_with_an_actionable_message(tmp_path: Path) -> None:
    """A prefix holding objects but no repo is genuinely unusable, and must still raise.

    What the fix changes is the wording, not the outcome: the message names the state, keeps
    the icechunk error as ``__cause__``, and makes deleting a judgement rather than an
    instruction — because the prefix may hold chunks somebody wants.
    """
    path = tmp_path / "dirty"
    path.mkdir()
    (path / "stray.bin").write_bytes(b"not a repository")

    with pytest.raises(CorruptedStoreError) as caught:
        open_or_create_repo(str(path))

    message = str(caught.value)
    assert "no readable repository" in message
    assert "interrupted" in message
    assert "inspect it before deleting" in message
    # The old text ordered a delete outright; that must not come back.
    assert "Delete it or use a different path" not in message
    assert isinstance(caught.value.__cause__, icechunk.IcechunkError)
