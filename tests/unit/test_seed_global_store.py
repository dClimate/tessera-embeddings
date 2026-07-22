"""seed_global_store flow: the recorded checkpoint identity default.

Exercised via ``.fn`` with the repo ops mocked, so no Icechunk store is created —
the point is only which ``model_version`` (→ ``checkpoint_id``) is stamped.
"""

from __future__ import annotations

import logging

import icechunk
import pytest

import tessera_embeddings.orchestration.prefect.flows.seed_global_store as mod
from tessera_embeddings.config.inference import checkpoint_filename
from tessera_embeddings.config.paths import BucketPaths

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


def _capture_seed(monkeypatch) -> dict:
    """Force the create path and capture the model_version passed to seed_zone_groups."""
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-seed"))
    monkeypatch.setattr(mod, "create_global_repo", lambda *a, **k: object())

    def _raise_open(*a, **k):
        raise FileNotFoundError("no store yet")

    monkeypatch.setattr(mod, "open_global_repo", _raise_open)  # → create path, seeded=set()
    captured: dict = {}

    def _fake_seed(repo, todo, *, years, model_version, commit_msg):
        captured["model_version"] = model_version
        return "snapshot"

    monkeypatch.setattr(mod, "seed_zone_groups", _fake_seed)
    return captured


def test_seed_defaults_checkpoint_id_to_checkpoint_filename(monkeypatch):
    """model_version defaults to checkpoint_filename() so the fill's checkpoint gate
    is effective by default — geoemb:model alone can't distinguish aws vs mpc.
    """
    captured = _capture_seed(monkeypatch)
    mod.seed_global_store.fn(paths=_PATHS)
    assert captured["model_version"] == checkpoint_filename()


def test_seed_preserves_explicit_model_version(monkeypatch):
    """An explicit model_version overrides the checkpoint-filename default."""
    captured = _capture_seed(monkeypatch)
    mod.seed_global_store.fn(paths=_PATHS, model_version="custom-encoder-v2")
    assert captured["model_version"] == "custom-encoder-v2"


def test_seed_creates_on_missing_repo_icechunk_error(monkeypatch):
    """A genuinely-missing repo (IcechunkError "doesn't exist") takes the create path."""
    captured = _capture_seed(monkeypatch)

    def _raise_missing(*a, **k):
        raise icechunk.IcechunkError("the repository doesn't exist")

    monkeypatch.setattr(mod, "open_global_repo", _raise_missing)
    mod.seed_global_store.fn(paths=_PATHS)
    assert captured["model_version"] == checkpoint_filename()  # seed_zone_groups ran → create path taken


def test_seed_reraises_transient_icechunk_error(monkeypatch):
    """A transient IcechunkError (timeout/auth) must NOT be treated as a missing repo —
    it re-raises instead of creating a fresh repo against a store that may already exist.
    """
    _capture_seed(monkeypatch)

    def _raise_transient(*a, **k):
        raise icechunk.IcechunkError("connection timed out")

    monkeypatch.setattr(mod, "open_global_repo", _raise_transient)
    created: list = []
    monkeypatch.setattr(mod, "create_global_repo", lambda *a, **k: created.append(1) or object())
    with pytest.raises(icechunk.IcechunkError, match="timed out"):
        mod.seed_global_store.fn(paths=_PATHS)
    assert created == []  # never entered the create path
