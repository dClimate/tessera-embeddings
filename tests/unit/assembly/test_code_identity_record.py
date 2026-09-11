"""Which build produced a published cell.

It was inferred once and only worked by luck: a fill was found to have run an image
predating a provenance field, and the only evidence was that the field it should have written
was absent. That proxy exists exactly once, for exactly that field, so the identity is now
recorded outright.

The property these tests exist for is that it can never break a fill. It is a diagnostic aid
on a record, so unreadable metadata must yield no field rather than an exception, and nothing
anywhere may compare two values and act on the difference — a mid-campaign change is a normal
event, and refusing a dispatch over one is an operator's decision, not the architecture's.
"""

from __future__ import annotations

import json

from tessera_embeddings.config import environment
from tessera_embeddings.config.environment import code_identity
from tessera_embeddings.storage.shard_writer import run_provenance


class _Dist:
    """A distribution standing in for the installed package's metadata."""

    def __init__(self, version: str, direct_url: dict | None) -> None:
        self.version = version
        self._raw = json.dumps(direct_url) if direct_url is not None else None

    def read_text(self, name: str) -> str | None:
        return self._raw if name == "direct_url.json" else None


def _serve(monkeypatch, dist: object) -> None:
    import importlib.metadata as md

    monkeypatch.setattr(md, "distribution", lambda _name: dist)


def test_a_git_install_records_the_commit_and_the_revision(monkeypatch) -> None:
    """The production shape, pinned against the metadata a git install actually writes.

    Both terms go in and they answer different questions: the revision says which line of
    development was asked for, the commit says which build arrived.
    """
    _serve(
        monkeypatch,
        _Dist(
            "0.1.0",
            {
                "url": "https://github.com/dClimate/tessera-embeddings.git",
                "vcs_info": {
                    "vcs": "git",
                    "commit_id": "68e4bbb3bbdf5eadb0829682aaca3aac2cd3bb77",
                    "requested_revision": "global-tessera-scoping",
                },
            },
        ),
    )
    identity = code_identity()
    assert identity == {
        "version": "0.1.0",
        "commit": "68e4bbb3bbdf5eadb0829682aaca3aac2cd3bb77",
        "revision": "global-tessera-scoping",
    }


def test_an_editable_install_says_so_rather_than_going_quiet(monkeypatch) -> None:
    """A working copy has no commit, and that is the single most useful thing to know about a
    result nobody can reproduce — so it is stated, not omitted.
    """
    _serve(monkeypatch, _Dist("0.1.0", {"url": "file:///home/x/repo", "dir_info": {"editable": True}}))
    identity = code_identity()
    assert identity == {"version": "0.1.0", "editable": True}


def test_a_wheel_install_records_the_version_alone(monkeypatch) -> None:
    """No direct-url metadata at all is legitimate, not an error."""
    _serve(monkeypatch, _Dist("0.1.0", None))
    assert code_identity() == {"version": "0.1.0"}


def test_unreadable_metadata_returns_nothing_and_never_raises(monkeypatch) -> None:
    """The property that matters most: no fill may fail because its own metadata was
    unreadable. A raising lookup and malformed JSON both have to yield None.
    """
    import importlib.metadata as md

    def _boom(_name: str) -> object:
        raise md.PackageNotFoundError("tessera-embeddings")

    monkeypatch.setattr(md, "distribution", _boom)
    assert code_identity() is None

    class _Malformed(_Dist):
        def read_text(self, name: str) -> str:
            return "{not json"

    _serve(monkeypatch, _Malformed("0.1.0", None))
    assert code_identity() is None


def test_the_record_carries_the_identity_without_the_caller_supplying_one(monkeypatch) -> None:
    """Resolved at the record rather than threaded: it is an ambient fact about the process,
    so five signatures that have no opinion about it do not grow an argument for it.
    """
    monkeypatch.setattr(environment, "code_identity", lambda: {"version": "9.9", "commit": "deadbeef"})
    import tessera_embeddings.storage.shard_writer as sw

    monkeypatch.setattr(sw, "code_identity", lambda: {"version": "9.9", "commit": "deadbeef"})
    runs = run_provenance(None, 2024, "run-1")
    assert runs["2024"]["code"] == {"version": "9.9", "commit": "deadbeef"}


def test_a_caller_may_pin_the_identity(monkeypatch) -> None:
    """So a test — or a replay — can state the build instead of reading the live process."""
    runs = run_provenance(None, 2024, "run-1", code={"commit": "abc123"})
    assert runs["2024"]["code"] == {"commit": "abc123"}


def test_no_identity_means_no_field(monkeypatch) -> None:
    """An absent identity is recorded as nothing, so a reader never sees an empty claim."""
    import tessera_embeddings.storage.shard_writer as sw

    monkeypatch.setattr(sw, "code_identity", lambda: None)
    runs = run_provenance(None, 2024, "run-1")
    assert "code" not in runs["2024"]
