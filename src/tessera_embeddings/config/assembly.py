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

    ``max_workers`` defaults to 16: each worker holds one staged-tile slice, and the pool peaks at
    roughly 40 GiB — inside the 64 GiB inference flow-runner family, the smallest host any caller
    of this default runs on.

    **A bigger pool belongs to the runner that can hold it, not to this default.** 32 workers peak
    near 75 GiB, which does NOT fit that 64 GiB family, so the campaign passes
    ``n_assembly_workers=32`` in its chained-fill dispatch
    (``run_global_campaign.ASSEMBLY_WORKERS_ON_THE_LARGE_RUNNER``), which is the only deployment on
    the 244 GiB ``assembly_large`` family. Sizing this default up would break every other caller.

    Nothing here caps icechunk's request concurrency any more. The write path is
    ``context_docs/storage/writing-to-the-global-store.md``; the measured footprint and what the
    pool is actually bound by are ``context_docs/assembly/what-bounds-assembly-2026-09-09.md``.
    """

    chunks_per_worker: int = 10
    #: 16, not 8: at 8 the 16-vCPU box ran half idle, and 16 is what `consumer_stack.py` sized it
    #: for. Safe on every host that takes this default; see the class docstring for the campaign's
    #: 32, and `assembly/what-bounds-assembly-2026-09-09.md` for the measured footprint.
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
