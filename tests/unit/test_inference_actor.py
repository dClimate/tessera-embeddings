"""Smoke tests for the Ray InferenceActor's CPU/GPU resource flexibility.

Avoids spinning up a Ray cluster by checking the static class-level
contract: the decorator no longer pins ``num_gpus=1``, and ``.options()``
accepts both GPU and CPU resource reservations. Full end-to-end actor
construction is covered in integration tests where Ray is available.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ray

from tessera_embeddings.inference.actors import InferenceActor, download_checkpoint


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


# --- download_checkpoint: real filesystem, no mocking ---
#
# fsspec.open() handles bare local paths, so the "remote" source is just a
# real file on disk and the cache dir is a tmp_path. This exercises the actual
# download → stage-to-temp → atomic-rename path; the only injected seam is the
# existing ``local_dir`` argument.


def _write_source(tmp_path: Path, payload: bytes = b"model-weights") -> Path:
    src = tmp_path / "remote" / "tessera_v1_1_aws_encoder.pt"
    src.parent.mkdir(parents=True)
    src.write_bytes(payload)
    return src


def test_download_checkpoint_copies_to_cache(tmp_path: Path) -> None:
    """A cold cache stages the file under local_dir and returns its path."""
    src = _write_source(tmp_path)
    cache = tmp_path / "cache"

    local = download_checkpoint(str(src), local_dir=str(cache))

    assert Path(local) == cache / src.name
    assert Path(local).read_bytes() == b"model-weights"


def test_download_checkpoint_reuses_existing_file(tmp_path: Path) -> None:
    """A warm cache returns the cached file without re-reading the source."""
    src = _write_source(tmp_path)
    cache = tmp_path / "cache"

    first = download_checkpoint(str(src), local_dir=str(cache))
    # Delete the source: a second call must hit the cache, not the source.
    src.unlink()
    second = download_checkpoint(str(src), local_dir=str(cache))

    assert first == second
    assert Path(second).read_bytes() == b"model-weights"


def test_download_checkpoint_leaves_no_partial_files(tmp_path: Path) -> None:
    """Only the published file remains — no leftover .part staging files."""
    src = _write_source(tmp_path)
    cache = tmp_path / "cache"

    download_checkpoint(str(src), local_dir=str(cache))

    assert sorted(p.name for p in cache.iterdir()) == [src.name]


def test_download_checkpoint_concurrent_callers_get_intact_file(tmp_path: Path) -> None:
    """Many actors racing on a cold cache never publish a half-written file.

    Each thread runs the full download/stage/rename. Atomic rename means
    last-writer-wins, so every returned path holds the complete payload and
    no truncated ``.part`` file is left behind.
    """
    payload = b"x" * (4 * 1024 * 1024)  # 4 MB, big enough to interleave writes
    src = _write_source(tmp_path, payload)
    cache = tmp_path / "cache"

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: download_checkpoint(str(src), local_dir=str(cache)), range(32)))

    assert all(Path(r).read_bytes() == payload for r in results)
    assert sorted(p.name for p in cache.iterdir()) == [src.name]


def test_configure_actor_logging_enables_debug_on_real_module_loggers() -> None:
    """The DEBUG allowlist must name the loggers the modules actually use.

    These names were once hardcoded with a stale ``src.inference.`` prefix, so
    every DEBUG diagnostic (TIMING breakdowns, EFFECTIVE TFLOPS, autocast
    probe) was silently suppressed in production. Resolving through the
    imported modules' ``__name__`` keeps this test rename-proof.
    """
    import logging

    from tessera_embeddings.inference import inference as inference_mod
    from tessera_embeddings.inference import profiling as profiling_mod
    from tessera_embeddings.inference.actors import _configure_actor_logging
    from tessera_embeddings.inference.models import modules as modules_mod
    from tessera_embeddings.inference.models import ssl_model as ssl_model_mod

    mods = (inference_mod, profiling_mod, modules_mod, ssl_model_mod)
    root = logging.getLogger()
    saved_root = (root.level, root.handlers[:])
    saved_levels = {m.__name__: logging.getLogger(m.__name__).level for m in mods}
    try:
        _configure_actor_logging()
        for m in mods:
            assert logging.getLogger(m.__name__).getEffectiveLevel() == logging.DEBUG, (
                f"{m.__name__} not at DEBUG after _configure_actor_logging()"
            )
    finally:
        # basicConfig(force=True) replaced root handlers; restore for other tests.
        root.setLevel(saved_root[0])
        root.handlers[:] = saved_root[1]
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)
