"""Smoke tests for the Ray InferenceActor's CPU/GPU resource flexibility.

Avoids spinning up a Ray cluster by checking the static class-level
contract: the decorator no longer pins ``num_gpus=1``, and ``.options()``
accepts both GPU and CPU resource reservations. Full end-to-end actor
construction is covered in integration tests where Ray is available.
"""

from __future__ import annotations

import ast
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import ray
import torch

from tessera_embeddings.inference import actors as actors_mod
from tessera_embeddings.inference.actors import InferenceActor, _gpu_total_gib, download_checkpoint


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


def test_the_actor_hands_its_own_gpu_index_to_the_monitor() -> None:
    """Review of #154 found the capability unwired: the actor built `ResourceMonitor(30)` with
    no index, so on a packed host the monitor queried every GPU, rejected the multi-row answer,
    and emitted neither statistics nor attribution — the whole point of the index.
    """
    import inspect

    src = inspect.getsource(actors_mod)
    assert "ResourceMonitor(interval_sec=30, gpu_index=self.gpu_index)" in src, (
        "the monitor must be given THIS actor's host GPU index, or it queries every GPU on a "
        "packed host, rejects the multi-row answer, and emits neither stats nor attribution"
    )
    assert "self.gpu_index = _accelerator_index()" in src


def test_the_ram_peak_is_reset_at_each_chunk_not_once_per_actor() -> None:
    """Also unwired: `reset_peak_host_ram()` existed but nothing called it, so `RAMpeak` was the
    maximum since actor startup and a deep chunk's peak was reported against every later chunk.
    """
    import inspect

    src = inspect.getsource(actors_mod)
    assert "self._resource_monitor.reset_peak_host_ram()" in src, (
        "nothing called it, so RAMpeak was the maximum since actor startup"
    )
    reset = src.index("self._resource_monitor.reset_peak_host_ram()")
    prologue = src.index('set_context("work", f"{chunk.label}:prologue")')
    assert reset < prologue, "the reset must precede the chunk's own context, not follow it"


# ---------------------------------------------------------------------------
# The batch size an actor runs is the one its own card can hold
# ---------------------------------------------------------------------------


class _FakeProps:
    def __init__(self, total_memory: int) -> None:
        self.total_memory = total_memory


class _FakeCuda:
    def __init__(self, total_memory: int) -> None:
        self._props = _FakeProps(total_memory)

    def get_device_properties(self, index: int) -> _FakeProps:
        assert index == 0
        return self._props


class _FakeTorch:
    def __init__(self, total_memory: int) -> None:
        self.cuda = _FakeCuda(total_memory)


def test_gpu_total_gib_reports_the_cards_size() -> None:
    """The A10G in g5.2xlarge reports 22.06 GiB, which is what the batch policy keys on."""
    torch_mod = _FakeTorch(int(22.06 * 1024**3))
    got = _gpu_total_gib(torch_mod, torch.device("cuda"))
    assert got is not None
    assert got == pytest.approx(22.06, abs=0.01)


def test_gpu_total_gib_is_none_without_a_card() -> None:
    """A CPU run has nothing to measure, and the policy must leave the batch alone."""
    assert _gpu_total_gib(_FakeTorch(0), torch.device("cpu")) is None


def test_the_actor_reads_no_un_narrowed_config_after_fitting_the_batch() -> None:
    """``__init__`` must not touch the raw ``config`` parameter once it has narrowed it.

    ``batch_size`` is consumed in two different functions downstream — the sub-batch split
    and the pinned host buffers — so the actor narrows it ONCE and stores the result on
    ``self.config``. Any later read of the bare parameter puts the tuned value back into
    circulation beside the fitted one, and the two disagree only on the card where it
    matters. Structural, because the failure is silent: it costs a GPU, not a test.
    """
    tree = ast.parse(inspect.getsource(actors_mod))
    init = next(
        node
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "InferenceActor"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    fit_line = next(
        node.lineno
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "replace"
    )
    late = [
        node.lineno
        for node in ast.walk(init)
        if isinstance(node, ast.Name) and node.id == "config" and node.lineno > fit_line
    ]
    assert not late, f"__init__ reads the un-narrowed `config` at line(s) {late}; use self.config"
