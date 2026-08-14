"""delete_prefix: s5cmd --all-versions on S3, fsspec fallback elsewhere."""

from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace

import pytest

from tessera_embeddings.storage import object_store


def test_delete_prefix_local_removes_dir(tmp_path):
    d = tmp_path / "staging" / "run1"
    d.mkdir(parents=True)
    (d / "a.txt").write_text("x")
    object_store.delete_prefix(str(d))
    assert not d.exists()


def test_delete_prefix_missing_local_is_noop(tmp_path):
    object_store.delete_prefix(str(tmp_path / "does-not-exist"))  # must not raise


def test_s5cmd_rm_passes_all_versions(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    object_store._s5cmd_rm("s3://b/p", logging.getLogger("t"), all_versions=True)
    assert "--all-versions" in captured["cmd"]
    assert captured["cmd"][-1] == "s3://b/p/*"


def _cleared(monkeypatch):
    """Every delete pass leaves the prefix empty — the ordinary case."""
    monkeypatch.setattr(object_store, "_survivors", lambda uri, log: [])


def _never_called():
    raise AssertionError("the fsspec fallback must not run for a verified-but-incomplete delete")


def test_delete_prefix_s3_defaults_to_all_versions(monkeypatch):
    seen: dict = {}

    def fake_s5(uri, log, *, all_versions):
        seen.update(uri=uri, all_versions=all_versions)

    monkeypatch.setattr(object_store, "_s5cmd_rm", fake_s5)
    _cleared(monkeypatch)
    object_store.delete_prefix("s3://bucket/mosaics/33N/2025")
    assert seen == {"uri": "s3://bucket/mosaics/33N/2025", "all_versions": True}


def test_delete_prefix_verifies_and_stops_after_one_clean_pass(monkeypatch):
    """The verification must not cost a second delete when the first one worked."""
    passes = []
    monkeypatch.setattr(object_store, "_s5cmd_rm", lambda uri, log, **k: passes.append(uri))
    _cleared(monkeypatch)
    object_store.delete_prefix("s3://b/p")
    assert len(passes) == 1


def test_delete_prefix_retries_while_objects_survive(monkeypatch):
    """s5cmd reported deleting 145,195 objects and left 807 behind, exiting zero throughout.

    A count of what a tool removed is not a statement about what is left, so the prefix is read
    back; a delete that keeps leaving survivors is reported rather than returned as success.
    """
    passes = []
    monkeypatch.setattr(object_store, "_s5cmd_rm", lambda uri, log, **k: passes.append(uri))
    monkeypatch.setattr(object_store, "_survivors", lambda uri, log: ["s3://b/p/chunks/ABC"])
    monkeypatch.setattr(object_store.fsspec, "filesystem", lambda proto: _never_called())

    object_store.delete_prefix("s3://b/p")  # best-effort: reports, does not raise
    assert len(passes) == object_store._DELETE_PASSES

    with pytest.raises(object_store.PrefixNotEmptyError):
        object_store.delete_prefix("s3://b/p", strict=True)


def test_delete_prefix_treats_an_unlistable_prefix_as_unknown_not_as_survivors(monkeypatch):
    """A listing that failed proves nothing — it must not manufacture an endless retry."""
    passes = []
    monkeypatch.setattr(object_store, "_s5cmd_rm", lambda uri, log, **k: passes.append(uri))
    monkeypatch.setattr(object_store, "_survivors", lambda uri, log: None)
    object_store.delete_prefix("s3://b/p", strict=True)  # must not raise
    assert len(passes) == 1


def test_delete_prefix_strict_raises_when_delete_fails(monkeypatch):
    """strict=True propagates a failed delete (else the caller ingests onto stale data)."""

    def _s5_fail(*a, **k):
        raise RuntimeError("s5cmd rc=1")

    class _FS:
        def exists(self, p):
            return True

        def rm(self, p, recursive):
            raise OSError("access denied")

    monkeypatch.setattr(object_store, "_s5cmd_rm", _s5_fail)
    monkeypatch.setattr(object_store.fsspec, "filesystem", lambda proto: _FS())

    object_store.delete_prefix("s3://b/p")  # best-effort: swallows
    with pytest.raises(OSError, match="access denied"):
        object_store.delete_prefix("s3://b/p", strict=True)
