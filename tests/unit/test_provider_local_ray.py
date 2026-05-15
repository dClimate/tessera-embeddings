"""Smoke test for the local Ray provider.

Verifies that ``providers.local.ray.ray_cluster`` is a working context
manager: it initialises Ray on enter, yields the local sentinel address,
and shuts down on exit. No AWS, no SSM, no cluster YAML.
"""

from __future__ import annotations

import ray

from tessera_embeddings.providers.local.ray import LOCAL_ADDRESS_SENTINEL, ray_cluster


def test_local_ray_cluster_enters_and_exits() -> None:
    """``ray_cluster`` initialises Ray on enter and shuts it down on exit."""
    assert not ray.is_initialized()
    with ray_cluster(num_cpus=1) as address:
        assert address == LOCAL_ADDRESS_SENTINEL
        assert ray.is_initialized()
    assert not ray.is_initialized()


def test_local_ray_cluster_handles_reentry() -> None:
    """Re-entering an already-initialized runtime is a no-op, not an error.

    ``ignore_reinit_error=True`` in the provider keeps nested context
    managers safe (e.g. when a test sets up a Ray runtime and the code
    under test enters another ``ray_cluster``).
    """
    with ray_cluster(num_cpus=1), ray_cluster(num_cpus=1):
        assert ray.is_initialized()
        # Inner shutdown tears Ray down even though the outer ctx is still active;
        # this is a known consequence of ray's global runtime model.
    assert not ray.is_initialized()
