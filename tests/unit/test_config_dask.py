"""Unit tests for ``config.dask.AssemblyConfig``."""

from __future__ import annotations

import pytest

from tessera_embeddings.config.dask import AssemblyConfig


def test_compute_n_workers_scales_linearly() -> None:
    """``ceil(n_chunks / chunks_per_worker)`` up to ``max_workers``."""
    cfg = AssemblyConfig(chunks_per_worker=40, max_workers=200)
    assert cfg.compute_n_workers(0) == 1  # floor of 1
    assert cfg.compute_n_workers(1) == 1
    assert cfg.compute_n_workers(40) == 1
    assert cfg.compute_n_workers(41) == 2
    assert cfg.compute_n_workers(850) == 22  # documented calibration: ~850 → ~20+
    assert cfg.compute_n_workers(8000) == 200  # caps at max_workers


def test_compute_n_workers_caps_at_max() -> None:
    """Worker count never exceeds ``max_workers``."""
    cfg = AssemblyConfig(max_workers=10)
    assert cfg.compute_n_workers(10_000) == 10


def test_invalid_chunks_per_worker_raises() -> None:
    """``chunks_per_worker`` must be a positive integer."""
    with pytest.raises(ValueError, match="chunks_per_worker must be > 0"):
        AssemblyConfig(chunks_per_worker=0)


def test_invalid_max_workers_raises() -> None:
    """``max_workers`` must be a positive integer."""
    with pytest.raises(ValueError, match="max_workers must be > 0"):
        AssemblyConfig(max_workers=0)
