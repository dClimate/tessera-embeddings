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

    ``max_workers`` defaults to 16: each worker holds at most one staged-tile
    slice in memory (~1-1.5 GB at a 2048-px full-band tile), so the pool peaks
    around ~24 GB — inside the flow runner's 64 GiB, and measured at 20 GB peak
    when the pool was 8.

    It also bounds this fill's S3 PUT concurrency. For a LONE fill the per-fork cap
    is ``TARGET_AGGREGATE_S3_CONCURRENCY // n_workers``, so aggregate stays at or
    under the target whenever ``max_workers <= target``. A campaign fill is passed a
    DIVIDED budget instead, which can fall below the worker count; the per-fork cap
    then floors at 1 and aggregate is the worker count itself. That is deliberate --
    the fork pool is not sacrificed to the request ceiling -- so ``max_workers`` is
    also what bounds a fill's contribution to the fleet's PUT rate. See
    ``assembly._s3_budget_split``.
    """

    chunks_per_worker: int = 10
    #: Raised 8 -> 16 on 2026-08-06 from a measurement, not a guess. At 8 the flow runner's box was
    #: HALF IDLE on the largest assembly attempted: 9,050 tiles wrote a dead-flat 248 objects/min for
    #: 160 minutes while CPU peaked at 7,443 of 16,384 allocated units and memory at 20 of 64 GiB.
    #: A flat rate at half the CPU is the signature of a fixed worker count rather than a resource
    #: limit — and the count did not scale with the job, since a 9,050-tile zone got the same 8
    #: processes as a 267-tile one.
    #:
    #: 16 is what the flow runner was sized for: `consumer_stack.py` says "16 vCPU / 64 GiB leaves
    #: headroom for n_workers=16 (~19 GiB)". Beyond 16 the box saturates and the next step is the
    #: already-registered 32-vCPU `assembly_large` family — with evidence, not before.
    max_workers: int = 16

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
