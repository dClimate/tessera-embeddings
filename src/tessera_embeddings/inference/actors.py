"""Ray actor for distributed GPU inference.

Each actor loads the model once, then processes multiple spatial chunks sequentially.
This amortizes model loading and GPU warm-up across chunks.

NOTE: torch and modules that import torch (models.builder, dataset, inference) are
deferred to __init__ / method bodies so the module can be imported on the Fargate
flow runner (which has ray but not torch). Ray serializes the actor class reference;
torch is only needed on GPU workers at runtime.
"""

from __future__ import annotations

import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
import numpy as np
import ray
import requests

from tessera_embeddings.config.inference import EMBEDDING_DIM, S2_BAND_ORDER, InferenceConfig
from tessera_embeddings.inference.assembly import OBS_COUNT_VARS, ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.data_loading import count_s2_window_timesteps, load_chunk
from tessera_embeddings.inference.resource_monitor import ResourceMonitor

if TYPE_CHECKING:
    import types

    import torch

    from tessera_embeddings.inference.data_loading import ChunkData
    from tessera_embeddings.inference.models.ssl_model import MultimodalBTInferenceModel

logger = logging.getLogger(__name__)

# Floor on derived strip height. A strip narrower than this reads so few
# northing rows that the per-strip fixed overhead (window-time filter, zarr
# group open, dataset bucketing) dominates and read amplification climbs without
# meaningfully lowering peak RAM. Dense chunks bottom out here rather than
# degenerating into hundreds of one-row reads.
_MIN_STRIP_H = 256

# Bytes per northing row of resident S2 *input* — the dominant term in the
# per-chunk working set (masks at 1 byte/elem and SAR at ~10x fewer timesteps
# are second order). One row is W * 10 bands * 2 bytes (uint16) * T_estimate;
# this constant is the per-(row, timestep) part, multiplied by W and T at call
# time.
_S2_BYTES_PER_ROW_TIMESTEP = len(S2_BAND_ORDER) * np.dtype(np.uint16).itemsize


def _compute_strip_height(*, height: int, width: int, t_estimate: int, strip_budget_bytes: int) -> int:
    """Derive the northing strip height that keeps one resident strip under budget.

    The resident input is dominated by the S2 band array, ``T * W * 10 * 2``
    bytes, which scales linearly in the northing extent. We solve for the
    largest strip height whose S2 band footprint fits ``strip_budget_bytes``,
    clamp it to ``[_MIN_STRIP_H, height]``, and return it. When the result is
    ``>= height`` the chunk loads as a single strip — byte-for-byte the
    unstriped path. ``t_estimate`` of 0 (no timesteps) returns the full height.
    """
    if t_estimate <= 0:
        return height
    bytes_per_row = width * t_estimate * _S2_BYTES_PER_ROW_TIMESTEP
    if bytes_per_row <= 0:
        return height
    strip_h = strip_budget_bytes // bytes_per_row
    strip_h = max(_MIN_STRIP_H, min(int(strip_h), height))
    return strip_h


def _strip_slices(height: int, strip_h: int) -> list[slice]:
    """Tile ``[0, height)`` into chunk-relative northing strips of ``strip_h`` rows.

    The final strip is shorter when ``height`` is not a multiple of ``strip_h``.
    ``strip_h >= height`` yields a single full-height strip.
    """
    return [slice(start, min(start + strip_h, height)) for start in range(0, height, strip_h)]


def _fetch_ec2_instance_id() -> str:
    """Fetch EC2 instance ID from instance metadata (IMDSv2).

    Returns "unknown" if metadata is unavailable (e.g., local development).
    """
    try:
        token = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
            timeout=2,
        ).text

        return requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=2,
        ).text
    except Exception:
        return "unknown"


def _configure_actor_logging() -> None:
    """Configure logging for inference actors.

    Sets root logger to INFO with a standard format, silences noisy third-party
    loggers, and enables DEBUG for inference profiling modules.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    for noisy in ("zarr", "zarr.group", "icechunk", "botocore", "s3transfer", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for name in (
        "src.inference.inference",
        "src.inference.profiling",
        "src.inference.models.modules",
        "src.inference.models.ssl_model",
    ):
        logging.getLogger(name).setLevel(logging.DEBUG)


def _select_device(torch_mod: types.ModuleType, instance_id: str) -> torch.device:
    """Detect CUDA availability and return the appropriate torch device.

    Args:
        torch_mod: The torch module (passed to avoid top-level import).
        instance_id: EC2 instance ID for log messages.
    """
    if torch_mod.cuda.is_available():
        device = torch_mod.device("cuda")
        gpu_name = torch_mod.cuda.get_device_name(0)
        logger.info("InferenceActor on instance %s, GPU: %s", instance_id, gpu_name)
    else:
        device = torch_mod.device("cpu")
        logger.warning("InferenceActor on instance %s: no GPU available, using CPU", instance_id)
    return device


def _log_vram_breakdown(model: MultimodalBTInferenceModel, torch_mod: types.ModuleType) -> None:
    """Log VRAM usage breakdown after model loading."""
    allocated = torch_mod.cuda.memory_allocated() / 1024**3
    reserved = torch_mod.cuda.memory_reserved() / 1024**3
    total = torch_mod.cuda.get_device_properties(0).total_memory / 1024**3
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())
    logger.info(
        "VRAM after model load: allocated=%.2f GB, reserved=%.2f GB, total=%.2f GB, "
        "model params=%.2f GB, model buffers=%.2f GB",
        allocated,
        reserved,
        total,
        param_bytes / 1024**3,
        buffer_bytes / 1024**3,
    )


# Host-RAM env vars, all set via runtime_env (not the decorator process) so Ray
# exports them into the worker environment BEFORE this module imports torch /
# instantiates the C allocator — each is read once at init and ignored if set
# later. On the decorator so they cover every creation site: the initial pool,
# the ActorPool replacement path (scheduling.py), and the local CPU runner.
#
# CUBLAS_WORKSPACE_CONFIG caps the cuBLAS host-side workspace. The model runs
# two CUDA streams (one per backbone), so the default workspace is reserved
# twice and inflates the per-chunk host-RAM plateau. Pinning it small claws that
# back. cuBLAS reads the var only when the first CUDA handle is created.
#
# MALLOC_ARENA_MAX=2 / MALLOC_TRIM_THRESHOLD_=0 attack glibc heap retention. The
# per-sub-batch prefetch thread churns numpy/torch CPU buffers every ~250ms;
# across multiple glibc arenas the freed regions are held on per-arena free
# lists at the high-water mark instead of returned to the OS, which an smaps
# rollup shows as dirty-anon RSS (much of it THP-backed). Capping arenas to 2
# (main + prefetch) and forcing trim-on-free returns that churn to the OS.
# These are A/B probes: if peak RSS drops, the plateau was retained churn, not
# live working set; if it barely moves, the working set is genuinely resident
# and chunking is the only remaining lever.
@ray.remote(
    runtime_env={
        "env_vars": {
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
            "MALLOC_ARENA_MAX": "2",
            "MALLOC_TRIM_THRESHOLD_": "0",
        }
    }
)
class InferenceActor:
    """Ray actor that runs embedding inference on a single GPU or CPU.

    Loads the model checkpoint once at initialization, then processes
    chunks on demand via :meth:`process_chunk`.

    Resource reservations (``num_gpus``, ``num_cpus``, ...) are NOT set on
    the decorator. Callers pass them via ``InferenceActor.options(...)``
    at ``.remote()`` call time so the same class supports GPU and CPU
    deployments without duplication. Typical patterns::

        # GPU worker (production / CUDA host)
        InferenceActor.options(num_gpus=1).remote(config, ckpt)

        # CPU-only worker (local runner, smoke tests)
        InferenceActor.options(num_gpus=0).remote(config, ckpt)
    """

    def __init__(self, config: InferenceConfig, checkpoint_path: str) -> None:
        """Initialize actor: download checkpoint (if S3) and load model onto GPU.

        Args:
            config: Inference configuration.
            checkpoint_path: S3 URI or local path to the model checkpoint.
        """
        import torch as _torch

        from tessera_embeddings.inference.models.builder import build_inference_model

        _configure_actor_logging()

        self.config = config
        self.instance_id = _fetch_ec2_instance_id()
        self.device = _torch.device("cpu") if self.config.num_gpus == 0 else _select_device(_torch, self.instance_id)

        local_ckpt = download_checkpoint(checkpoint_path) if _is_remote_uri(checkpoint_path) else checkpoint_path
        self.model: MultimodalBTInferenceModel = build_inference_model(
            config,
            self.device,
            checkpoint_path=local_ckpt,
        )

        if self.device.type == "cuda":  # No-op on CPU
            _log_vram_breakdown(self.model, _torch)

        self._resource_monitor = ResourceMonitor(interval_sec=30)
        self._resource_monitor.start()
        logger.info("InferenceActor ready on instance %s", self.instance_id)

    def ping(self) -> bool:
        """No-op health check used to wait for actor initialization.

        Called after actor creation to block until __init__ completes
        (model loaded, GPU ready). Returns True when ready.
        """
        return True

    def get_instance_id(self) -> str:
        """Return the EC2 instance ID this actor is running on."""
        return self.instance_id

    def process_chunk(
        self,
        chunk: ChunkSpec,
        mosaic_base: str,
        staging_base: str,
        run_id: str,
        tracker: ray.actor.ActorHandle | None = None,
    ) -> dict[str, str | int | float]:
        """Process a single spatial chunk: load data, run inference, write output.

        Args:
            chunk: Spatial chunk specification.
            mosaic_base: Base path for the mosaic stores
                (e.g., "s3://bucket/mosaics/small_minnesota").
            staging_base: Base path for staging output.
            run_id: Unique run identifier.
            tracker: Optional ProgressTracker actor handle for batch-level progress.

        Returns:
            Result dict with chunk label, status, pixel count, and timing.
        """
        from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
        from tessera_embeddings.inference.inference import run_inference

        t0 = time.monotonic()

        # Report loading phase so stall detection has visibility before batch 50
        if tracker:
            tracker.report.remote(chunk.label, 0, 0, "loading")  # type: ignore[union-attr]

        try:
            # Size northing strips from the byte budget so a dense chunk's
            # resident *input* stays bounded. Probe the window timestep count
            # from the 1-D time coord (cheap, no spatial read) to estimate T
            # before the first full load. A normal-density chunk resolves to a
            # single full-height strip — byte-for-byte the unstriped path.
            t_estimate = count_s2_window_timesteps(mosaic_base, self.config.time_window)
            strip_h = _compute_strip_height(
                height=chunk.height,
                width=chunk.width,
                t_estimate=t_estimate,
                strip_budget_bytes=self.config.strip_budget_bytes,
            )
            strips = _strip_slices(chunk.height, strip_h)
            logger.info(
                "Chunk %s: T_estimate=%d, strip_h=%d -> %d strip(s)",
                chunk.label,
                t_estimate,
                strip_h,
                len(strips),
            )

            # Whole-chunk output buffers, allocated once and held for the chunk.
            # We sub-tile INPUTS only; the output and write path are untouched.
            # save_dim mirrors run_inference: the canonical 128-D slice, or the
            # full representation width for smaller (test) models.
            save_dim = min(EMBEDDING_DIM, self.config.representation_dim)
            embeddings = np.zeros((chunk.height, chunk.width, save_dim), dtype=np.int8)
            scales = np.full((chunk.height, chunk.width), 1e-8, dtype=np.float32)

            writer = ZarrWriter(staging_base, embedding_dim=save_dim)

            def _load_strip(y_sub: slice) -> ChunkData:
                return load_chunk(
                    chunk,
                    mosaic_base,
                    time_window=self.config.time_window,
                    s1_orbit=self.config.s1_orbit,
                    y_sub=y_sub,
                )

            on_batch = (
                (lambda b, t: tracker.report.remote(chunk.label, b, t, "inference"))  # type: ignore[union-attr]
                if tracker
                else None
            )

            # accumulate obs counts per strip into whole-chunk buffers so the
            # single write_chunk carries the full-chunk obs maps.
            obs_buffers: dict[str, np.ndarray] = {
                var: np.zeros((chunk.height, chunk.width), dtype=np.uint16) for var in OBS_COUNT_VARS
            }

            total_valid = 0
            # 1-deep prefetch pipeline: strip i+1 loads while strip i runs
            # inference (same shape as inference.run_inference's prefetcher).
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="strip-prefetch") as pool:
                next_future = pool.submit(_load_strip, strips[0]) if strips else None

                for i, strip in enumerate(strips):
                    assert next_future is not None
                    chunk_data = next_future.result()

                    # Kick off the next strip's load before running inference on this one.
                    next_future = pool.submit(_load_strip, strips[i + 1]) if i + 1 < len(strips) else None

                    for var in OBS_COUNT_VARS:
                        arr = getattr(chunk_data, var)
                        if arr is not None:
                            obs_buffers[var][strip] = arr

                    dataset = MosaicChunkInferenceDataset(
                        chunk_data,
                        num_obs_checkpoints=self.config.num_obs_checkpoints,
                        s1_orbit=self.config.s1_orbit,
                    )

                    if len(dataset) == 0:
                        # Empty strip: leave its output rows at the zero/1e-8
                        # initialised value, mirroring run_inference's handling
                        # of fully-invalid chunks. The strip still contributes
                        # its (zero) obs counts, already written above.
                        logger.info("Chunk %s strip %s: no valid pixels, leaving zero-filled", chunk.label, strip)
                        continue

                    result = run_inference(self.model, dataset, self.config, self.device, on_batch=on_batch)
                    embeddings[strip] = result.embeddings
                    scales[strip] = result.scales
                    total_valid += len(dataset)

            if total_valid == 0:
                # Every strip was empty. ROI pre-filter means chunks get here
                # only if they intersect the ROI, so this fires only when every
                # pixel in a live chunk fails the SCL/S1 validity thresholds.
                # Assembly will fill the chunk's footprint with zeros/NaN from
                # the Dask graph — no placeholder zarr needed. We still drop a
                # zero-byte skip marker so verify_staged_completeness can
                # distinguish a legitimate skip from a silently-failed chunk
                # (Ray worker crash, etc.).
                logger.info("Chunk %s has no valid pixels, skipping (assembly will fill)", chunk.label)
                writer.write_skip_marker(chunk, run_id)
                return {
                    "chunk": chunk.label,
                    "status": "skipped",
                    "valid_pixels": 0,
                    "elapsed_sec": time.monotonic() - t0,
                    "instance_id": self.instance_id,
                }

            # Report writing phase before S3 write
            if tracker:
                tracker.report.remote(chunk.label, 0, 0, "writing")  # type: ignore[union-attr]

            # Single whole-chunk write — assembly / skip-marker logic untouched.
            writer.write_chunk(
                chunk,
                embeddings,
                run_id,
                embeddings_std=None,
                scales=scales,
                obs_counts=obs_buffers,
            )

            elapsed = time.monotonic() - t0
            logger.info(
                "Chunk %s complete: %d valid pixels, %.1fs",
                chunk.label,
                total_valid,
                elapsed,
            )

            return {
                "chunk": chunk.label,
                "status": "success",
                "valid_pixels": total_valid,
                "elapsed_sec": elapsed,
                "instance_id": self.instance_id,
            }

        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.exception("Chunk %s failed after %.1fs on instance %s", chunk.label, elapsed, self.instance_id)
            return {
                "chunk": chunk.label,
                "status": "failed",
                "error": str(e),
                "elapsed_sec": elapsed,
                "instance_id": self.instance_id,
            }


# Schemes that mean "fetch this from somewhere else first". A local filesystem
# path (or an explicit file:// URI) is loaded in place by torch.
_REMOTE_CKPT_SCHEMES = ("s3://", "http://", "https://", "gs://", "az://", "abfs://")


def _is_remote_uri(path: str) -> bool:
    """True if ``path`` must be downloaded before torch.load can open it."""
    return path.startswith(_REMOTE_CKPT_SCHEMES)


def _default_checkpoint_cache() -> str:
    """Pick a download cache dir that exists on the running host.

    On AWS DLAMI GPU boxes the NVMe instance store (~1.5 GB/s) is the right
    target — the root EBS volume (~42 MB/s) is too slow and torch.load with
    mmap hangs on it. Off that path (laptops, CI, non-AWS GPUs) the NVMe mount
    doesn't exist, so fall back to a temp dir under the system tmp.
    """
    nvme = Path("/opt/dlami/nvme")
    if nvme.is_dir():
        return str(nvme / "tessera-checkpoints")
    return str(Path(tempfile.gettempdir()) / "tessera-checkpoints")


def download_checkpoint(remote_path: str, local_dir: str | None = None) -> str:
    """Download a model checkpoint from a remote URI to local storage.

    Handles any fsspec-supported remote scheme — ``s3://``, ``https://``
    (e.g. a HuggingFace ``resolve/main`` URL), ``gs://``, etc. The file is
    staged locally because torch.load wants a real path and reads it twice.

    Args:
        remote_path: Remote URI (e.g. ``"s3://bucket/path/model.pt"`` or
            ``"https://huggingface.co/.../tessera_v1_1_aws_encoder.pt"``).
        local_dir: Local directory for downloads. Defaults to the NVMe
            instance store on AWS DLAMI hosts, else a system temp dir.

    Returns:
        Local file path.

    Concurrency: many actors on the same host may call this with the same
    ``remote_path`` and shared cache dir at once (cold cache, 100s of actors).
    The download writes to a unique temp file and is published to the final
    path with an atomic rename, so a concurrent reader never observes a
    partially-written checkpoint and concurrent writers can't corrupt each
    other's output — the last rename wins, and every byte is identical.
    """
    filename = remote_path.rsplit("/", 1)[-1]

    local = Path(local_dir or _default_checkpoint_cache())
    local.mkdir(parents=True, exist_ok=True)
    local_path = local / filename

    if local_path.exists():
        logger.info("Checkpoint already cached: %s", local_path)
        return str(local_path)

    logger.info("Downloading checkpoint: %s → %s", remote_path, local_path)
    # Checkpoints are ~200 MB, so reading the whole file into memory is fine.
    with fsspec.open(remote_path, "rb") as remote:
        data = remote.read()
    # Stage into a unique temp file in the same dir (so the rename stays on one
    # filesystem and is atomic), then atomically publish — concurrent actors
    # publishing the same checkpoint can't observe a half-written file.
    with tempfile.NamedTemporaryFile(dir=local, prefix=f"{filename}.", suffix=".part", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(local_path)

    downloaded_size = local_path.stat().st_size
    logger.info("Download complete: %s (%.1f MB)", local_path, downloaded_size / 1024 / 1024)

    return str(local_path)
