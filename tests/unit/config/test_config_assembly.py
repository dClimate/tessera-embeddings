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


@pytest.mark.parametrize("field", ["chunks_per_worker", "max_workers"])
def test_a_non_positive_size_is_refused(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be > 0"):
        AssemblyConfig(**{field: 0})
