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
    the full grid: only live chunks have staged data to read and write,
    so they account for essentially all the work — non-intersecting
    chunks are constant fill in the Dask graph and never touch S3.
    Calibrated so ~850 live chunks → 85 workers, scaling up to the
    ``max_workers`` cap once an ROI exceeds that.

    ``max_workers`` is capped at 100 to match
    ``assembly.TARGET_AGGREGATE_S3_CONCURRENCY``. Each Dask worker forks its
    own icechunk Repository that issues at least 1 concurrent S3 PUT, so
    aggregate PUT concurrency is >= n_workers. Capping workers at the target
    keeps the fleet-wide PUT rate under S3's ~3500 req/s/prefix ceiling; a
    higher cap would burst over it and draw ``503 SlowDown`` on append.
    """

    chunks_per_worker: int = 10
    max_workers: int = 100


# Auto-sizing caps for the master pipeline's ingest cluster + Ray pool.
INGEST_MIN_WORKERS_CAP = 150
"""Dask scheduler connection limit for ingest min_workers."""

INGEST_MAX_WORKERS_CAP = 225
"""Dask scheduler connection limit for ingest max_workers."""

INGEST_MAX_WORKERS_RATIO = 1.5
"""Workers-per-chunk ratio for the ingest cluster ceiling."""

INGEST_MIN_WORKERS_FLOOR = 3
"""Minimum viable ingest cluster size (smaller fights startup overhead)."""

NUM_ACTORS_CAP = 80
"""GPU actor count ceiling — typical AWS A10G quota for a single account."""


def compute_pipeline_cluster_sizing(
    n_chunks: int,
    *,
    ingest_min_workers: int | None = None,
    ingest_max_workers: int | None = None,
    num_actors: int | None = None,
) -> tuple[int, int, int]:
    """Derive worker / actor counts from ROI chunk count when not explicitly set.

    Defaults used when the corresponding argument is ``None``:

    * ``ingest_min_workers``: 1 per chunk, capped at
      :data:`INGEST_MIN_WORKERS_CAP`, floored at
      :data:`INGEST_MIN_WORKERS_FLOOR`.
    * ``ingest_max_workers``: ``ratio * n_chunks`` (floored), capped at
      :data:`INGEST_MAX_WORKERS_CAP`.
    * ``num_actors``: 1 per chunk, capped at :data:`NUM_ACTORS_CAP`.

    When ``n_chunks < INGEST_MIN_WORKERS_FLOOR`` the floor kicks in and
    the ratio between min and max workers does not hold (max is clamped
    up to min instead).
    """
    import math

    if ingest_min_workers is None:
        ingest_min_workers = max(min(n_chunks, INGEST_MIN_WORKERS_CAP), INGEST_MIN_WORKERS_FLOOR)
    if ingest_max_workers is None:
        ingest_max_workers = min(math.floor(n_chunks * INGEST_MAX_WORKERS_RATIO), INGEST_MAX_WORKERS_CAP)
    ingest_max_workers = max(ingest_max_workers, ingest_min_workers)
    if num_actors is None:
        num_actors = min(n_chunks, NUM_ACTORS_CAP)
    return ingest_min_workers, ingest_max_workers, num_actors
