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


def test_delete_prefix_s3_defaults_to_all_versions(monkeypatch):
    seen: dict = {}

    def fake_s5(uri, log, *, all_versions):
        seen.update(uri=uri, all_versions=all_versions)

    monkeypatch.setattr(object_store, "_s5cmd_rm", fake_s5)
    object_store.delete_prefix("s3://bucket/mosaics/33N/2025")
    assert seen == {"uri": "s3://bucket/mosaics/33N/2025", "all_versions": True}


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
