"""Unit tests for ``config.assembly.AssemblyConfig`` (process-pool sizing)."""

from __future__ import annotations

import pytest

from tessera_embeddings.config.assembly import AssemblyConfig


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


def test_the_default_pool_is_thirty_two_on_the_large_runner() -> None:
    """32 assembly workers, and the pair of facts that makes it safe and worth doing.

    RAM: each worker holds at most one staged-tile slice (~1-1.5 GB), so 32 peaks around
    ~48 GB — inside the 244 GiB ``assembly_large`` flow runner and NOT inside the 64 GiB
    inference family, which is why the number and the runner family move together.

    Concurrency: 32 was refused while assembly divided a fleet S3-request budget into a
    per-fork cap, because more forks pushed that cap down to 1 — the value at which icechunk
    deadlocks. The cap is gone, so the worker count is bounded by the box alone.
    """
    assert AssemblyConfig().max_workers == 32
    assert AssemblyConfig().compute_n_workers(10_000) == 32
