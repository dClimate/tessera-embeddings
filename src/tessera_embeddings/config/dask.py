"""Dask cluster sizing for the ingest phase.

Ingest keeps Dask for compute (STAC reads, mosaicking); these caps size its cluster and
the master pipeline's Ray pool. Assembly does not run on Dask — its process-pool sizing
lives in :mod:`tessera_embeddings.config.assembly`. The reference repo's ``CoarsenConfig``
belongs to an embedding-coarsening flow that is out of scope here and is not ported.
"""

from __future__ import annotations

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
"""GPU actor count ceiling — typical AWS G-family GPU quota for a single account."""


def compute_pipeline_cluster_sizing(
    n_chunks: int,
    *,
    ingest_min_workers: int | None = None,
    ingest_max_workers: int | None = None,
    num_actors: int | None = None,
) -> tuple[int, int, int]:
    """Derive worker / actor counts from ROI chunk count when not explicitly set.

    Defaults for a ``None`` argument: ``ingest_min_workers`` 1 per chunk, capped at
    :data:`INGEST_MIN_WORKERS_CAP` and floored at :data:`INGEST_MIN_WORKERS_FLOOR`;
    ``ingest_max_workers`` ``ratio * n_chunks`` floored, capped at
    :data:`INGEST_MAX_WORKERS_CAP`; ``num_actors`` 1 per chunk, capped at
    :data:`NUM_ACTORS_CAP`.

    Below :data:`INGEST_MIN_WORKERS_FLOOR` chunks the floor wins and the min/max ratio no
    longer holds — max is clamped up to min instead.
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
