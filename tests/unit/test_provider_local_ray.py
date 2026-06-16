"""Smoke test for the local Ray provider.

Verifies that ``providers.local.ray.ray_cluster`` is a working context
manager: it initialises Ray on enter, yields the local sentinel address,
and shuts down on exit. No AWS, no SSM, no cluster YAML.
"""

from __future__ import annotations

import pytest
import ray

from tessera_embeddings.providers.local.ray import LOCAL_ADDRESS_SENTINEL, ray_cluster


@pytest.mark.slow
def test_local_ray_cluster_enters_and_exits() -> None:
    """``ray_cluster`` initialises Ray on enter and shuts it down on exit."""
    assert not ray.is_initialized()
    with ray_cluster(num_cpus=1) as address:
        assert address == LOCAL_ADDRESS_SENTINEL
        assert ray.is_initialized()
    assert not ray.is_initialized()
