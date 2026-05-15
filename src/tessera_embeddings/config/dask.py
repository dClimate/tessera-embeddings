"""Dask cluster configuration for the embedding-assembly phase.

The :class:`AssemblyConfig` config object scales worker count from the
number of *live* (ROI-intersecting) spatial chunks. It is consumed by
the AWS Dask provider (and any equivalent) as a substrate-agnostic
recipe — the substrate decides what a "worker" actually is.

The reference repo also defines a ``CoarsenConfig`` for an embedding
coarsening flow; that flow is out of scope for the open-source release
and is not ported here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _ChunkScaledClusterConfig:
    """Base config for Dask clusters that scale worker count from chunk count.

    Subclasses set defaults for ``chunks_per_worker`` and ``max_workers``
    to match their workload profile.
    """

    chunks_per_worker: int
    worker_cpu: int = 4096
    worker_mem: int = 8192
    # Dask scheduler has trouble past ~200 workers; cap conservatively.
    max_workers: int = 200

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.chunks_per_worker <= 0:
            raise ValueError(f"chunks_per_worker must be > 0, got {self.chunks_per_worker}")
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {self.max_workers}")

    def compute_n_workers(self, n_chunks: int) -> int:
        """Return the number of workers for ``n_chunks`` spatial chunks.

        Scales linearly at ``ceil(n_chunks / chunks_per_worker)`` up to
        ``max_workers``, with a floor of 1.
        """
        return max(1, min(-(-n_chunks // self.chunks_per_worker), self.max_workers))


@dataclass(frozen=True)
class AssemblyConfig(_ChunkScaledClusterConfig):
    """Configuration for the Dask assembly phase of inference.

    Controls per-worker resource allocation and dynamic scaling of the
    cluster that assembles staged chunk Zarrs into the final output.

    Worker count is derived from *live* (ROI-intersecting) chunks, not
    the full grid, so very sparse ROIs don't oversaturate S3 with
    writes. Calibrated so ~850 live chunks → 20 workers, scaling up to
    200 workers for dense ROIs.
    """

    chunks_per_worker: int = 40
