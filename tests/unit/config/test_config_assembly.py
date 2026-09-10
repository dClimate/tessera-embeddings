"""Unit tests for ``config.assembly.AssemblyConfig`` (process-pool sizing)."""

from __future__ import annotations

import pytest

from tessera_embeddings.config.assembly import AssemblyConfig
from tessera_embeddings.orchestration.prefect.flows import run_global_campaign as campaign


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


def test_the_default_pool_is_sixteen_and_the_campaign_asks_for_thirty_two() -> None:
    """The worker count belongs to the runner that can hold it. Measured at ~2.9 GiB per worker,
    16 peaks near 47 GiB and fits the 64 GiB host every default-taking caller runs on; 32 peaks
    near 94 GiB, which does NOT fit it, and only the chained campaign fill is on the 244 GiB
    family, so only it asks explicitly.
    """
    assert AssemblyConfig().max_workers == 16
    assert AssemblyConfig().compute_n_workers(10_000) == 16
    assert campaign.ASSEMBLY_WORKERS_ON_THE_LARGE_RUNNER == 32
