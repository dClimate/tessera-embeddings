"""Configuration for the satellite data pipeline.

Re-exports the configuration constants, dataclasses and provider registries, so either of these
works::

    from tessera_embeddings.config import S2_L2A_BANDS, CollectionConfig, PROVIDERS
    from tessera_embeddings.config.satellites import S2_BASELINE_THRESHOLD
"""

from .providers import (
    PROVIDERS,
    CollectionConfig,
    STACProvider,
)
from .satellites import (
    LANDSAT_BASELINE_OFFSET,
    LANDSAT_BASELINE_THRESHOLD,
    LANDSAT_C2_BANDS,
    S1_BASELINE_OFFSET,
    S1_BASELINE_THRESHOLD,
    S1_DB_CLIP_MAX,
    S1_DB_SCALE,
    S1_DB_SHIFT,
    S1_GRD_BANDS,
    S1_OPERA_BANDS,
    S2_BAND_MAPPING,
    S2_BASELINE_OFFSET,
    S2_BASELINE_THRESHOLD,
    S2_L2A_BANDS,
    S2_SCL_BAND,
    S2_SCL_INVALID_CLASSES,
    S2_STORED_BANDS,
)

__all__ = [
    "LANDSAT_BASELINE_OFFSET",
    "LANDSAT_BASELINE_THRESHOLD",
    "LANDSAT_C2_BANDS",
    "PROVIDERS",
    "S1_BASELINE_OFFSET",
    "S1_BASELINE_THRESHOLD",
    "S1_DB_CLIP_MAX",
    "S1_DB_SCALE",
    "S1_DB_SHIFT",
    "S1_GRD_BANDS",
    "S1_OPERA_BANDS",
    "S2_BAND_MAPPING",
    "S2_BASELINE_OFFSET",
    "S2_BASELINE_THRESHOLD",
    "S2_L2A_BANDS",
    "S2_SCL_BAND",
    "S2_SCL_INVALID_CLASSES",
    "S2_STORED_BANDS",
    "CollectionConfig",
    "STACProvider",
]
