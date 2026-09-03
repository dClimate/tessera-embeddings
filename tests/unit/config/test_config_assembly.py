"""Unit tests for ``config.assembly.AssemblyConfig`` (process-pool sizing)."""

from __future__ import annotations

import pytest

from tessera_embeddings.config.assembly import AssemblyConfig
from tessera_embeddings.inference.assembly import TARGET_AGGREGATE_S3_CONCURRENCY


def test_compute_n_workers_scales_linearly() -> None:
    """``ceil(n_chunks / chunks_per_worker)`` up to ``max_workers``."""
    cfg = AssemblyConfig(chunks_per_worker=10, max_workers=8)
    assert cfg.compute_n_workers(0) == 1  # floor of 1
    assert cfg.compute_n_workers(1) == 1
    assert cfg.compute_n_workers(10) == 1
    assert cfg.compute_n_workers(11) == 2
    assert cfg.compute_n_workers(75) == 8  # caps at max_workers
    assert cfg.compute_n_workers(10_000) == 8


def test_invalid_chunks_per_worker_raises() -> None:
    """``chunks_per_worker`` must be a positive integer."""
    with pytest.raises(ValueError, match="chunks_per_worker must be > 0"):
        AssemblyConfig(chunks_per_worker=0)


def test_invalid_max_workers_raises() -> None:
    """``max_workers`` must be a positive integer."""
    with pytest.raises(ValueError, match="max_workers must be > 0"):
        AssemblyConfig(max_workers=0)


def test_default_max_workers_honors_s3_concurrency_target() -> None:
    """Default process cap must not exceed the aggregate S3 PUT target.

    ``per_worker_cap`` floors at 1, so aggregate PUT concurrency is
    ``>= n_workers``. If the default cap exceeded
    ``TARGET_AGGREGATE_S3_CONCURRENCY`` the pool would burst over S3's
    per-prefix rate and draw ``503 SlowDown``. Locks the two constants
    in sync (see assembly.TARGET_AGGREGATE_S3_CONCURRENCY).
    """
    assert AssemblyConfig().max_workers <= TARGET_AGGREGATE_S3_CONCURRENCY


def test_aggregate_put_concurrency_stays_under_target_at_cap() -> None:
    """At any live-chunk count, n_workers * per_worker_cap <= target.

    Mirrors the ``per_worker_cap = max(1, target // n_workers)`` math in
    ``ZarrWriter.assemble`` and checks the product never blows the target,
    including chunk counts that pin n_workers at the cap.
    """
    cfg = AssemblyConfig()
    target = TARGET_AGGREGATE_S3_CONCURRENCY
    for n_live in (1, 50, 250, 850, 2761, 10_000):
        n_workers = cfg.compute_n_workers(n_live)
        per_worker_cap = max(1, target // n_workers)
        assert n_workers * per_worker_cap <= target, f"{n_live=} blew target"
