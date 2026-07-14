"""Worker-process sizing for the embedding-assembly phase.

Assembly runs as a pool of local worker processes driving raw-zarr fork/merge
writes (see :mod:`tessera_embeddings.inference.assembly`) — there is no Dask
cluster to provision. :class:`AssemblyConfig` scales the process count from the
number of *live* (ROI-intersecting) spatial chunks and caps it at a RAM- and
S3-budgeted ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final


@final
@dataclass(frozen=True)
class AssemblyConfig:
    """Process-pool sizing for assembly.

    Worker count is derived from *live* (ROI-intersecting) chunks, not the
    full grid: only live chunks have staged data to read and write, so they
    account for essentially all the work.

    ``max_workers`` defaults to 8: each worker holds at most one staged-tile
    slice in memory (~1-1.5 GB at a 2048-px full-band tile), so the pool peaks
    around ~12 GB — comfortable on the flow runner. It also keeps aggregate S3
    PUT concurrency trivially under
    ``assembly.TARGET_AGGREGATE_S3_CONCURRENCY`` (the per-fork request cap is
    ``target // n_workers``, so aggregate <= target whenever
    ``max_workers <= target``).
    """

    chunks_per_worker: int = 10
    max_workers: int = 8

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
