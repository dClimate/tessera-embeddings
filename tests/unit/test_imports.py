"""Smoke test: every domain module imports without side effects.

Catches circular imports and missing dependencies at the cheapest possible
test cost. Update this file whenever a new public module is added.
"""

from __future__ import annotations


def test_top_level_packages_import() -> None:
    """Every top-level domain package imports."""
    import tessera_embeddings  # noqa: F401
    from tessera_embeddings import config, errors, inference, ingest, storage, utils  # noqa: F401


def test_config_modules_import() -> None:
    """Every config submodule imports."""
    from tessera_embeddings.config import (  # noqa: F401
        dask,
        environment,
        inference,
        providers,
        satellites,
        time_windows,
    )


def test_ingest_modules_import() -> None:
    """Every ingest submodule imports."""
    from tessera_embeddings.ingest import (  # noqa: F401
        auth,
        opera_query,
        roi,
        roi_processing,
        s1_roi,
        s2_roi,
        stac,
        transforms,
    )


def test_inference_modules_import() -> None:
    """Every inference submodule imports."""
    from tessera_embeddings.inference import (  # noqa: F401
        actors,
        assembly,
        chunk_spec,
        conventions,
        data_loading,
        dataset,
        diagnostics,
        inference,
        lifecycle,
        profiling,
        progress,
        quantization,
        reference_grid,
        resource_monitor,
        runner,
        sampling,
        scheduling,
    )


def test_inference_models_import() -> None:
    """Inference model builder + module sources import."""
    from tessera_embeddings.inference.models import builder, modules, ssl_model  # noqa: F401


def test_storage_modules_import() -> None:
    """Storage submodules import."""
    from tessera_embeddings.storage import manifest, zarr_store  # noqa: F401


def test_provider_modules_import() -> None:
    """Provider submodules import."""
    from tessera_embeddings.providers.aws import dask as aws_dask  # noqa: F401
    from tessera_embeddings.providers.aws import diagnostics  # noqa: F401
    from tessera_embeddings.providers.aws import ray as aws_ray  # noqa: F401
    from tessera_embeddings.providers.local import dask as local_dask  # noqa: F401
    from tessera_embeddings.providers.local import ray as local_ray  # noqa: F401


def test_orchestration_modules_import() -> None:
    """Orchestration helpers import (no Prefect required)."""
    from tessera_embeddings.orchestration import concurrency  # noqa: F401
