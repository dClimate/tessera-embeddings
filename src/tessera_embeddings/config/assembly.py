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

    ``max_workers`` defaults to 16: each worker holds at most one staged-tile slice in
    memory (~1-1.5 GB at a 2048-px full-band tile), so the pool peaks around ~24 GB — inside
    the 16 vCPU / 64 GiB inference flow-runner family, which is the smallest host any caller
    of this default runs on. Measured 20 GB peak when the pool was 8.

    **A bigger pool belongs to the runner that can hold it, not to this default.** 32 workers
    peak around ~48 GB and need the 32 vCPU / 244 GiB ``assembly_large`` family
    (yield-embeddings ``FAMILY_ASSEMBLY_LARGE``), which only the chained campaign fill runs on;
    on the inference family they would leave no headroom for the coordinator and the Ray head.
    So the campaign passes ``n_assembly_workers=32`` in its chained-fill dispatch
    (``run_global_campaign.ASSEMBLY_WORKERS_ON_THE_LARGE_RUNNER``) and every other caller —
    the plain runner, the standalone flow, a hand-run fill — gets a number that is safe on the
    host it is actually on.

    Each fork opens the store at icechunk's default request concurrency; nothing here caps
    it. Measurements: ``context_docs/storage/writing-to-the-global-store.md``.
    """

    chunks_per_worker: int = 10
    #: 16, not 8: at 8 the flow runner's box ran half idle on the largest assembly attempted and
    #: the worker count did not scale with the job. 16 is what `consumer_stack.py` sized the 16-vCPU
    #: box for ("leaves headroom for n_workers=16 (~19 GiB)") and it saturates it, assembly being
    #: roughly half blocked on S3. It stays the DEFAULT because it is safe on every host that takes
    #: this default; the campaign asks for 32 explicitly on the runner sized for it (see the class
    #: docstring). Measurements: context_docs/storage/writing-to-the-global-store.md
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
