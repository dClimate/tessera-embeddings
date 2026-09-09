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
    """The worker count belongs to the runner that can hold it, not to the library default.

    Each worker holds at most one staged-tile slice (~1-1.5 GB), so 16 peaks around ~24 GB and
    32 around ~48 GB. Only the chained campaign fill runs on the 32 vCPU / 244 GiB
    ``assembly_large`` family; the standalone flow, the plain runner and ``fill-zone-year`` all
    take this default on a 64 GiB host, where 32 would oversubscribe them. So the default is the
    number that is safe everywhere and the campaign overrides it where it is not.

    32 was refused outright until 2026-09-09, for a different reason: assembly divided a fleet
    S3-request budget into a per-fork cap, so more forks pushed that cap toward 1 — the value at
    which icechunk deadlocks. That cap is gone.
    """
    assert AssemblyConfig().max_workers == 16
    assert AssemblyConfig().compute_n_workers(10_000) == 16
    assert campaign.ASSEMBLY_WORKERS_ON_THE_LARGE_RUNNER == 32
