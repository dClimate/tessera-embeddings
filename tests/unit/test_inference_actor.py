"""Smoke tests for the Ray InferenceActor's CPU/GPU resource flexibility.

Avoids spinning up a Ray cluster by checking the static class-level
contract: the decorator no longer pins ``num_gpus=1``, and ``.options()``
accepts both GPU and CPU resource reservations. Full end-to-end actor
construction is covered in integration tests where Ray is available.
"""

from __future__ import annotations

import ray

from tessera_embeddings.inference.actors import InferenceActor


def test_actor_class_has_no_static_gpu_reservation() -> None:
    """The ``@ray.remote`` decorator must not pin ``num_gpus`` on the class.

    Phase 4 moves ``num_gpus`` to runtime ``.options()`` so a single class
    supports both GPU and CPU deployments. If a static reservation creeps
    back in, this test catches it before the CPU runner breaks.
    """
    # Ray's RemoteClass exposes default options via _default_options
    default_options = getattr(InferenceActor, "_default_options", None) or {}
    assert default_options.get("num_gpus") in (None, 0), (
        f"InferenceActor must not statically reserve GPUs; got {default_options!r}"
    )


def test_actor_options_accepts_cpu_and_gpu() -> None:
    """``.options(num_gpus=0)`` and ``.options(num_gpus=1)`` both bind cleanly.

    We do NOT call ``.remote()`` — that would require a running Ray
    runtime and a real checkpoint. We only verify the option-binding API
    accepts both reservations.
    """
    cpu_cls = InferenceActor.options(num_gpus=0)  # type: ignore[attr-defined]
    gpu_cls = InferenceActor.options(num_gpus=1)  # type: ignore[attr-defined]
    assert cpu_cls is not None
    assert gpu_cls is not None
    # Sanity check: both still expose .remote
    assert callable(cpu_cls.remote)
    assert callable(gpu_cls.remote)


def test_actor_is_a_ray_remote_class() -> None:
    """InferenceActor is a Ray remote class (not a plain Python class)."""
    assert isinstance(InferenceActor, ray.actor.ActorClass)
