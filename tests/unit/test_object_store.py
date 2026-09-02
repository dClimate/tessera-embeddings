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


def test_delete_prefix_s3_does_not_pass_all_versions_by_default(monkeypatch):
    """OFF by default, because on an unversioned bucket the flag is not a free precaution.

    It raises the required permission to `s3:DeleteObjectVersion` and enumerates through
    `ListObjectVersions` instead of `ListObjectsV2`. The campaign's buckets are unversioned,
    so every production delete was paying both costs for nothing.
    """
    seen: dict = {}

    def fake_s5(uri, log, *, all_versions):
        seen.update(uri=uri, all_versions=all_versions)

    monkeypatch.setattr(object_store, "_s5cmd_rm", fake_s5)
    _cleared(monkeypatch)
    object_store.delete_prefix("s3://bucket/mosaics/33N/2025")
    assert seen == {"uri": "s3://bucket/mosaics/33N/2025", "all_versions": False}


def test_delete_prefix_forwards_all_versions_when_asked(monkeypatch):
    """The flag still WORKS — it is opt-in, not removed. A versioned bucket needs it."""
    seen: dict = {}

    def fake_s5(uri, log, *, all_versions):
        seen.update(all_versions=all_versions)

    monkeypatch.setattr(object_store, "_s5cmd_rm", fake_s5)
    _cleared(monkeypatch)
    object_store.delete_prefix("s3://bucket/p", all_versions=True)
    assert seen == {"all_versions": True}


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
    monkeypatch.setattr(object_store.time, "sleep", lambda _s: None)
    monkeypatch.setattr(object_store.fsspec, "filesystem", lambda proto: _never_called())

    object_store.delete_prefix("s3://b/p")  # best-effort: reports, does not raise
    assert len(passes) == object_store._DELETE_PASSES

    with pytest.raises(object_store.PrefixNotEmptyError):
        object_store.delete_prefix("s3://b/p", strict=True)


def test_delete_prefix_treats_an_unlistable_prefix_as_unknown_not_as_survivors(monkeypatch):
    """A listing that failed proves nothing, and the two halves of that pull apart.

    It must not manufacture an endless retry — the delete is not what failed, so running it
    again cannot help. And it must not read as success under ``strict``, whose entire purpose
    is callers that cannot proceed onto un-cleared data; ``_survivors`` documents reporting an
    unlistable prefix as clean as the way an unverified delete comes to look verified.

    So: one pass either way, best-effort returns, strict raises — and the error says UNKNOWN
    rather than claiming survivors nobody has seen.
    """
    passes = []
    monkeypatch.setattr(object_store, "_s5cmd_rm", lambda uri, log, **k: passes.append(uri))
    monkeypatch.setattr(object_store, "_survivors", lambda uri, log: None)

    object_store.delete_prefix("s3://b/p")  # best-effort: reports and returns
    assert len(passes) == 1

    with pytest.raises(object_store.DeleteUnverifiedError):
        object_store.delete_prefix("s3://b/p", strict=True)
    assert len(passes) == 2, "one delete pass per call — an unlistable prefix is not retried"


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
    monkeypatch.setattr(object_store.time, "sleep", lambda _s: None)
    monkeypatch.setattr(object_store.fsspec, "filesystem", lambda proto: _FS())

    object_store.delete_prefix("s3://b/p")  # best-effort: swallows

    # strict + all_versions is refused BEFORE the fallback runs: fsspec cannot remove
    # non-current versions, so the promise cannot be kept whatever the fallback does. Now
    # that all_versions defaults OFF this arm has to ask for it explicitly.
    with pytest.raises(object_store.DeleteUnverifiedError):
        object_store.delete_prefix("s3://b/p", strict=True, all_versions=True)

    # At the DEFAULT (all_versions=False) the fallback CAN keep the promise, so it runs — and
    # its own failure propagates under strict, which is what this test was written to pin.
    with pytest.raises(OSError, match="access denied"):
        object_store.delete_prefix("s3://b/p", strict=True)


def test_a_throttled_delete_is_retried_before_the_fallback(monkeypatch):
    """S3 answered `SlowDown: Please reduce your request rate` on a quarter-million-object
    prefix (48S/2022, live). A throttle is transient, so it earns the same retry budget as a
    pass that left survivors — falling straight through to the serial fsspec sweep meets the
    same rate limit with none of s5cmd's parallelism.
    """
    calls = []

    def throttled_then_ok(uri, log, **k):
        calls.append(uri)
        if len(calls) < 3:
            raise RuntimeError("s5cmd failed (rc=1): SlowDown: Please reduce your request rate")

    monkeypatch.setattr(object_store, "_s5cmd_rm", throttled_then_ok)
    monkeypatch.setattr(object_store.time, "sleep", lambda _s: None)
    _cleared(monkeypatch)
    monkeypatch.setattr(object_store.fsspec, "filesystem", lambda proto: _never_called())

    object_store.delete_prefix("s3://b/p")
    assert len(calls) == 3


def test_a_missing_binary_is_not_retried(monkeypatch):
    """The adverse direction: s5cmd absent will still be absent next time, and the fsspec
    fallback is the whole answer for it — so it must reach the fallback immediately.
    """
    calls = []

    def absent(uri, log, **k):
        calls.append(uri)
        raise FileNotFoundError("s5cmd binary not found")

    fell_back = []

    class _FS:
        def exists(self, p):
            return True

        def rm(self, p, recursive):
            fell_back.append(p)

    monkeypatch.setattr(object_store, "_s5cmd_rm", absent)
    monkeypatch.setattr(object_store.time, "sleep", lambda _s: None)
    monkeypatch.setattr(object_store.fsspec, "filesystem", lambda proto: _FS())

    object_store.delete_prefix("s3://b/p")
    assert len(calls) == 1, "a missing binary must not be retried"
    assert fell_back == ["s3://b/p"]


def test_strict_all_versions_refuses_the_fsspec_fallback(monkeypatch):
    """Fsspec's rm removes CURRENT objects only; on a versioned bucket the old versions stay.

    So when s5cmd is missing or fails, the fallback cannot honour `all_versions` — and returning
    success from it tells a strict caller a prefix is reclaimed while terabytes of non-current
    versions are still billed. `_InputRetention.cleanup` then releases the cell's storage-budget
    slot and admits another ingest on the strength of it.

    Best-effort callers still get the fallback; only a caller that said it cannot proceed on an
    unclean prefix is refused.
    """

    def _s5_missing(*a, **k):
        raise FileNotFoundError("s5cmd not on PATH")

    monkeypatch.setattr(object_store, "_s5cmd_rm", _s5_missing)
    monkeypatch.setattr(object_store.time, "sleep", lambda _s: None)

    class _FS:
        def exists(self, p):
            return True

        def rm(self, p, recursive):
            return None

        def invalidate_cache(self, p=None):
            return None

        def find(self, p):
            return []  # the delete worked; see the strict-verification test for when it did not

    monkeypatch.setattr(object_store.fsspec, "filesystem", lambda proto: _FS())

    # Best-effort: the fallback runs and the caller is warned, not stopped.
    object_store.delete_prefix("s3://b/p", all_versions=True)

    with pytest.raises(object_store.DeleteUnverifiedError, match="non-current versions"):
        object_store.delete_prefix("s3://b/p", all_versions=True, strict=True)

    # all_versions=False is a different promise and the fallback CAN keep it — but only once it
    # READS BACK. This assertion used to pass against a fallback that verified nothing, which
    # pinned the defect as the contract: it proved the version promise and was silent about the
    # verification one. It now also requires the prefix to come back empty.
    object_store.delete_prefix("s3://b/p", all_versions=False, strict=True)


def _fallback_fs(monkeypatch, *, find):
    """s5cmd absent, so the fsspec fallback runs; `find` decides what it reads back."""

    def _s5_missing(*a, **k):
        raise FileNotFoundError("s5cmd not on PATH")

    monkeypatch.setattr(object_store, "_s5cmd_rm", _s5_missing)
    monkeypatch.setattr(object_store.time, "sleep", lambda _s: None)

    class _FS:
        def exists(self, p):
            return True

        def rm(self, p, recursive):
            return None

        def invalidate_cache(self, p=None):
            return None

        def find(self, p):
            return find()

    monkeypatch.setattr(object_store.fsspec, "filesystem", lambda proto: _FS())


def test_a_strict_fsspec_fallback_is_verified_not_assumed(monkeypatch):
    """`strict` promises to raise when the delete ran and left objects behind. `fs.rm` returning
    without error is not that: it reports per call, not per prefix.

    This path used to be unreachable for a strict caller — `all_versions` defaulted ON, so the
    `strict and all_versions` refusal fired before fsspec was tried. Making the flag opt-in made
    the fallback reachable, and an unverified success there answers the one caller whose next move
    is to release a retention slot for the prefix.
    """
    _fallback_fs(monkeypatch, find=lambda: ["s3://b/p/left-behind.zarr/c/0"])
    with pytest.raises(object_store.PrefixNotEmptyError, match="still holds 1 object"):
        object_store.delete_prefix("s3://b/p", strict=True)


def test_a_strict_fsspec_fallback_treats_an_unlistable_prefix_as_unknown(monkeypatch):
    """Symmetrical with the s5cmd path: a listing that failed proves nothing, and reporting it as
    clean is how an unverified delete comes to look verified.
    """

    def _boom():
        raise OSError("AccessDenied on ListBucket")

    _fallback_fs(monkeypatch, find=_boom)
    with pytest.raises(object_store.DeleteUnverifiedError, match="could not list it to confirm"):
        object_store.delete_prefix("s3://b/p", strict=True)


def test_a_best_effort_fsspec_fallback_still_returns_when_objects_survive(monkeypatch):
    """The verification is for `strict` only. Best-effort cleanup after a success must not start
    failing cells over residue it was never asked to guarantee.
    """
    _fallback_fs(monkeypatch, find=lambda: ["s3://b/p/left-behind.zarr/c/0"])
    object_store.delete_prefix("s3://b/p")  # must not raise
