"""Smoke tests for the Ray InferenceActor's CPU/GPU resource flexibility.

Avoids spinning up a Ray cluster by checking the static class-level
contract: the decorator no longer pins ``num_gpus=1``, and ``.options()``
accepts both GPU and CPU resource reservations. Full end-to-end actor
construction is covered in integration tests where Ray is available.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ray

from tessera_embeddings.inference import actors as actors_mod
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


# ---------------------------------------------------------------------------
# Peak-VRAM instrumentation and per-host attribution on the CHUNK_SUMMARY line.
# ---------------------------------------------------------------------------


class _FakeCuda:
    """The four ``torch.cuda`` calls the VRAM helpers make, and nothing else."""

    def __init__(self, allocated: int, reserved: int, total: int) -> None:
        self._allocated, self._reserved, self._total = allocated, reserved, total
        self.resets = 0

    def reset_peak_memory_stats(self) -> None:
        self.resets += 1

    def max_memory_allocated(self) -> int:
        return self._allocated

    def max_memory_reserved(self) -> int:
        return self._reserved

    def get_device_properties(self, index: int):
        assert index == 0, "torch sees only this actor's GPU, and calls it 0"
        return SimpleNamespace(total_memory=self._total)


def _fake_torch(allocated_gib: float, reserved_gib: float, total_gib: float = 44.7):
    gib = 1024**3
    return SimpleNamespace(cuda=_FakeCuda(int(allocated_gib * gib), int(reserved_gib * gib), int(total_gib * gib)))


class TestVramPeakFields:
    """The peak-VRAM figure the campaign has never had."""

    def test_reports_allocated_and_reserved_separately(self) -> None:
        """Both, because only one of them is a requirement.

        ``max_memory_allocated`` is what live tensors needed and therefore what a
        smaller card would have to fit. ``max_memory_reserved`` is the caching
        allocator's pool, which is what nvidia-smi sees — so the gap between the two
        IS the amount by which our only previous reading ("97% of the card")
        overstates the requirement.
        """
        fields = actors_mod._vram_peak_fields(_fake_torch(18.5, 31.0))
        assert fields == {
            "vram_peak_gib": 18.5,
            "vram_reserved_peak_gib": 31.0,
            "vram_total_gib": 44.7,
        }

    def test_carries_the_cards_own_capacity(self) -> None:
        """So a percentage is computed against the card that ran, not a remembered spec.

        The remembered spec has been wrong: the same L40S is recorded as "46 GB" in
        one design doc and "48 GB" in a README, because 45,776 MiB is 44.7 GiB and
        48.0 GB and the unit was dropped.
        """
        assert actors_mod._vram_peak_fields(_fake_torch(1, 2, total_gib=44.7))["vram_total_gib"] == 44.7

    def test_off_cuda_the_keys_are_absent_not_zero(self) -> None:
        """A zero would read as "measured, and it was nothing"."""
        assert actors_mod._vram_peak_fields(None) == {}

    def test_reset_is_a_noop_off_cuda(self) -> None:
        actors_mod._reset_vram_peak(None)  # must not raise

    def test_reset_zeroes_the_peak_counters(self) -> None:
        """Per-chunk, not per-process.

        Without the reset, every chunk after the deepest one reports the deepest
        chunk's peak — and VRAM scales with optical depth.
        """
        torch_mod = _fake_torch(1, 2)
        actors_mod._reset_vram_peak(torch_mod)
        assert torch_mod.cuda.resets == 1


class TestHostFieldsOnTheChunkSummary:
    """Per-host attribution, which the log stream cannot provide."""

    @staticmethod
    def _actor(torch_mod, instance_id="i-abc", gpu_index="2"):
        """An actor instance with only the three attributes ``_host_fields`` reads.

        ``@ray.remote`` replaces the name with an ``ActorClass`` wrapper, so the real
        class is reached through Ray's metadata. Constructing it bare — no
        ``__init__`` — is deliberate: a real one downloads a checkpoint and loads a
        model onto a GPU, and this method's contract is three attributes wide.
        """
        cls = actors_mod.InferenceActor.__ray_metadata__.modified_class  # type: ignore[attr-defined]
        actor = cls.__new__(cls)
        actor.instance_id = instance_id
        actor.gpu_index = gpu_index
        actor._torch = torch_mod
        return actor

    def test_carries_instance_id_and_gpu_index(self) -> None:
        """Neither was on the line before, and the stream cannot substitute.

        One CloudWatch log stream is one ECS flow runner, so the whole fleet's chunks
        share it. Ray's log prefix carries ``ip=``, which separates hosts but not the
        four actors on one — so the pair here is the minimum that decomposes a packed
        fleet, and without it a packed host running 15% slow is invisible.
        """
        fields = self._actor(_fake_torch(18.5, 31.0))._host_fields()
        assert fields["instance_id"] == "i-abc"
        assert fields["gpu"] == "2"
        assert fields["vram_peak_gib"] == 18.5

    def test_serialises_into_the_chunk_summary_line(self) -> None:
        """The line is JSON, so a new field must survive a round trip."""
        line = actors_mod._chunk_summary_line(
            label="chunk_0_0", status="success", **self._actor(_fake_torch(18.5, 31.0))._host_fields()
        )
        assert line.startswith("CHUNK_SUMMARY: ")
        payload = json.loads(line[len("CHUNK_SUMMARY: ") :])
        assert payload["instance_id"] == "i-abc"
        assert payload["gpu"] == "2"
        assert payload["vram_reserved_peak_gib"] == 31.0

    def test_a_cpu_actor_still_produces_a_valid_line(self) -> None:
        fields = self._actor(None, gpu_index=None)._host_fields()
        assert fields == {"instance_id": "i-abc", "gpu": None}
        json.loads(actors_mod._chunk_summary_line(label="c", **fields)[len("CHUNK_SUMMARY: ") :])


class TestAcceleratorIndex:
    """The actor's own GPU index, for anything that does not honour CUDA_VISIBLE_DEVICES."""

    def test_returns_the_first_assigned_id(self) -> None:
        ctx = SimpleNamespace(get_accelerator_ids=lambda: {"GPU": ["3"], "TPU": []})
        with patch.object(ray, "get_runtime_context", return_value=ctx):
            assert actors_mod._accelerator_index() == "3"

    def test_none_when_no_gpu_is_assigned(self) -> None:
        ctx = SimpleNamespace(get_accelerator_ids=lambda: {"GPU": []})
        with patch.object(ray, "get_runtime_context", return_value=ctx):
            assert actors_mod._accelerator_index() is None

    def test_none_without_a_ray_runtime(self) -> None:
        """The local CPU runner and unit tests have no runtime; that is not an error."""
        with patch.object(ray, "get_runtime_context", side_effect=RuntimeError("no runtime")):
            assert actors_mod._accelerator_index() is None


def test_the_actor_class_survives_cloudpickle() -> None:
    """Ray serialises the actor CLASS at first submission, class attributes included.

    So an attribute default that is a live object rather than a sentinel fails the
    whole run before a single node launches, in the driver, with a
    ``TypeError: cannot pickle '_thread.lock' object`` that names neither the class
    nor the attribute. A `ResourceMonitor()` default did exactly that — it holds a
    `threading.Event` and two `threading.Lock`s — and no unit test noticed, because
    every other test either constructs the actor or drives its methods, and neither
    of those serialises anything.

    This is the cheapest possible guard on the whole class of defect: one dump of
    the underlying class, which is what Ray's client pickler does to it.
    """
    from ray.cloudpickle import dumps

    dumps(InferenceActor.__ray_metadata__.modified_class)
