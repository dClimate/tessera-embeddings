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
import time
from pathlib import Path
from typing import TYPE_CHECKING

import boto3
import ray
import requests

from tessera_embeddings.config.inference import InferenceConfig
from tessera_embeddings.inference.assembly import OBS_COUNT_VARS, ZarrWriter
from tessera_embeddings.inference.chunk_spec import ChunkSpec
from tessera_embeddings.inference.data_loading import load_chunk
from tessera_embeddings.inference.resource_monitor import ResourceMonitor

if TYPE_CHECKING:
    import types

    import torch

    from tessera_embeddings.inference.models.ssl_model import MultimodalBTInferenceModel

logger = logging.getLogger(__name__)


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


@ray.remote(num_gpus=1)
class InferenceActor:
    """Ray actor that runs embedding inference on a single GPU.

    Loads the model checkpoint once at initialization, then processes
    chunks on demand via process_chunk().
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
        self.device = _select_device(_torch, self.instance_id)

        local_ckpt = download_checkpoint(checkpoint_path) if checkpoint_path.startswith("s3://") else checkpoint_path
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
            # Load data (optimized: only fetch the S2 timesteps the model will sample)
            chunk_data = load_chunk(
                chunk,
                mosaic_base,
                sample_size_s2=self.config.sample_size_s2,
                repeat_times=self.config.repeat_times,
                s1_orbit=self.config.s1_orbit,
                time_window=self.config.time_window,
            )

            # Build dataset
            dataset = MosaicChunkInferenceDataset(
                chunk_data, self.config.min_valid_timesteps, s1_orbit=self.config.s1_orbit
            )

            writer = ZarrWriter(staging_base)

            if len(dataset) == 0:
                # ROI pre-filter means chunks get here only if they intersect the
                # ROI, so this branch fires only when every pixel in a live chunk
                # fails the SCL/S1 validity thresholds. Assembly will fill the
                # chunk's footprint with zeros/NaN from the Dask graph — no
                # placeholder zarr needed. We still drop a zero-byte skip marker
                # so verify_staged_completeness can distinguish a legitimate
                # skip from a silently-failed chunk (Ray worker crash, etc.).
                logger.info("Chunk %s has no valid pixels, skipping (assembly will fill)", chunk.label)
                writer.write_skip_marker(chunk, run_id)
                return {
                    "chunk": chunk.label,
                    "status": "skipped",
                    "valid_pixels": 0,
                    "elapsed_sec": time.monotonic() - t0,
                    "instance_id": self.instance_id,
                }

            # Build progress callback (Ray-agnostic: inference.py just calls a function)
            on_batch = (
                (lambda b, t: tracker.report.remote(chunk.label, b, t, "inference"))  # type: ignore[union-attr]
                if tracker
                else None
            )

            # Run inference
            result = run_inference(self.model, dataset, self.config, self.device, on_batch=on_batch)

            # Report writing phase before S3 write
            if tracker:
                tracker.report.remote(chunk.label, 0, 0, "writing")  # type: ignore[union-attr]

            # Write to staging
            writer.write_chunk(
                chunk,
                result.embeddings,
                run_id,
                embeddings_std=result.embeddings_std,
                scales=result.scales,
                obs_counts={var: getattr(chunk_data, var) for var in OBS_COUNT_VARS},
            )

            elapsed = time.monotonic() - t0
            logger.info(
                "Chunk %s complete: %d valid pixels, %.1fs",
                chunk.label,
                len(dataset),
                elapsed,
            )

            return {
                "chunk": chunk.label,
                "status": "success",
                "valid_pixels": len(dataset),
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


def download_checkpoint(s3_path: str, local_dir: str = "/opt/dlami/nvme/tessera-checkpoints") -> str:
    """Download a model checkpoint from S3 to local NVMe storage.

    Downloads to the NVMe instance store for fast torch.load. The root
    EBS volume (~42 MB/s) is too slow for large checkpoints — torch.load
    with mmap hangs indefinitely. The NVMe SSD (~1.5 GB/s) avoids this.

    Args:
        s3_path: S3 URI (e.g., "s3://bucket/path/model.pt").
        local_dir: Local directory for downloads. Defaults to NVMe instance store.

    Returns:
        Local file path.
    """
    filename = s3_path.rsplit("/", 1)[-1]
    bucket, key = s3_path[5:].split("/", 1)

    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    local_path = local / filename

    if local_path.exists():
        logger.info("Checkpoint already cached: %s", local_path)
        return str(local_path)

    logger.info("Downloading checkpoint: %s → %s", s3_path, local_path)
    boto3.client("s3").download_file(bucket, key, str(local_path))
    s3_downloaded_size = local_path.stat().st_size
    logger.info("Download complete: %s (%.1f MB)", local_path, s3_downloaded_size / 1024 / 1024)

    return str(local_path)
