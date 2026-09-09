"""Worker-process sizing for the embedding-assembly phase.

Assembly runs as a pool of local worker processes driving raw-zarr fork/merge writes
(see :mod:`tessera_embeddings.inference.assembly`) — there is no Dask cluster to
provision. :class:`AssemblyConfig` scales the process count from the number of *live*
(ROI-intersecting) spatial chunks and caps it at a RAM-budgeted ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class AssemblyConfig:
    """Process-pool sizing for assembly.

    Worker count is derived from *live* (ROI-intersecting) chunks, not the full grid:
    only live chunks have staged data to read and write.

    ``max_workers`` defaults to 32: each worker holds at most one staged-tile slice in
    memory (~1-1.5 GB at a 2048-px full-band tile), so the pool peaks around ~48 GB. That
    needs the 32 vCPU / 244 GiB ``assembly_large`` flow-runner family the campaign fill now
    runs on (yield-embeddings ``FAMILY_ASSEMBLY_LARGE``); on the 16 vCPU / 64 GiB inference
    family it would leave no headroom for the coordinator and the Ray head. Measured 20 GB
    peak when the pool was 8.

    Each fork opens the store at icechunk's default request concurrency; nothing here caps
    it. Measurements: ``context_docs/storage/writing-to-the-global-store.md``.
    """

    chunks_per_worker: int = 10
    #: 32, on the 32-vCPU `assembly_large` runner (2026-09-08). The history: 8 left the 16-vCPU box
    #: half idle on the largest assembly attempted; 16 was what `consumer_stack.py` sized that box
    #: for and saturated it (assembly is roughly half blocked on S3, so 16 workers kept ~16 cores
    #: busy). Assembly is the campaign's longest stage and, once inference is done, its ONLY
    #: remaining one — a cluster's whole backlog drains through one trailing thread at one assembly
    #: at a time — so its speed is the campaign's tail. Doubling the workers on a box with twice
    #: the cores is the cheapest lever there is: the runner is one Fargate task per cluster against
    #: a GPU fleet of hundreds. Safe at 32 only because the per-fork request cap is gone: it was
    #: the cap of 1, not the worker count, that deadlocked icechunk, and 32 uncapped workers per
    #: fill were measured at fleet width with zero throttling. Measurements:
    #: context_docs/storage/writing-to-the-global-store.md
    max_workers: int = 32

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.chunks_per_worker <= 0:
            raise ValueError(f"chunks_per_worker must be > 0, got {self.chunks_per_worker}")
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {self.max_workers}")

    def compute_n_workers(self, n_chunks: int) -> int:
        """Return the worker-process count for ``n_chunks`` live spatial chunks.

        Scales linearly at ``ceil(n_chunks / chunks_per_worker)`` up to
        ``max_workers``, with a floor of 1, so tiny ROIs don't spawn idle
        processes.
        """
        return max(1, min(-(-n_chunks // self.chunks_per_worker), self.max_workers))
