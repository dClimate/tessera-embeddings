"""tessera_embeddings — distributed satellite ingestion + Tessera inference.

:data:`__all__` is the entire public API; everything else may change without warning. CI
verifies it stays in sync with ``docs/public-api.md``.

Conventions:

* Underscore-prefixed module names, functions and attributes are private.
* ``@typing.final`` classes must not be subclassed; mypy enforces this.
* Domain code (``ingest``, ``inference``, ``storage``, ``providers``,
  ``orchestration.concurrency``) is orchestrator-agnostic — the same function works under
  Prefect flows, the plain runner, and any future orchestrator.
"""

from tessera_embeddings.config.assembly import AssemblyConfig
from tessera_embeddings.config.inference import (
    EMBEDDING_DIM,
    INFERENCE_CHUNK_SIZE,
    InferenceConfig,
    checkpoint_filename,
)
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.config.time_windows import TimeWindow, parse_time_window
from tessera_embeddings.errors import (
    ConfigMismatchError,
    CorruptedStoreError,
    InsufficientCoverageError,
)
from tessera_embeddings.ingest.s1_roi import (
    S1Orbit,
    SarIngestResult,
    ingest_s1_roi_sar,
)
from tessera_embeddings.ingest.s2_roi import (
    IngestResult,
    ingest_s2_roi_reflectance,
)


def __getattr__(name: str) -> object:
    if name == "run_inference":
        from tessera_embeddings.inference.runner import run_inference

        return run_inference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EMBEDDING_DIM",
    "INFERENCE_CHUNK_SIZE",
    "AssemblyConfig",
    "BucketPaths",
    "ConfigMismatchError",
    "CorruptedStoreError",
    "InferenceConfig",
    "IngestResult",
    "InsufficientCoverageError",
    "S1Orbit",
    "SarIngestResult",
    "TimeWindow",
    "checkpoint_filename",
    "ingest_s1_roi_sar",
    "ingest_s2_roi_reflectance",
    "parse_time_window",
    "run_inference",
]
