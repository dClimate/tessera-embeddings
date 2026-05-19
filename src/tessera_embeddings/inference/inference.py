"""Core inference loop for Tessera v1.1 embeddings.

Iterates buckets (pixels sharing a ``(s2_target, s1_target)`` key), then
sub-batches each bucket by ``config.batch_size``. Each sub-batch goes through
a single forward pass; the 192-D output is sliced to the canonical 128-D
downstream representation and written into a flat per-pixel output array.

Sampling is deterministic under v1.1 (no random repeats), so ``compute_std``
from v1.0 is now a no-op and the ``embeddings_std`` field is always ``None``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from tessera_embeddings.config.inference import EMBEDDING_DIM, InferenceConfig
from tessera_embeddings.inference.dataset import MosaicChunkInferenceDataset
from tessera_embeddings.inference.models.ssl_model import MultimodalBTInferenceModel
from tessera_embeddings.inference.profiling import (
    disable_model_profiling,
    enable_model_profiling,
    log_autocast_dtype_probe,
    log_cuda_diagnostics,
    log_effective_tflops,
    log_profiled_batch_summary,
)
from tessera_embeddings.inference.quantization import quantize_embeddings

logger = logging.getLogger(__name__)

PROFILE_BATCH_IDX = 10
"""Batch index for the one-time deep profile (per-layer timing + dtype checks)."""


@dataclass
class _BatchTimings:
    """Per-phase wall-clock times (seconds) for one sub-batch."""

    get_batch: float
    transfer: float
    forward: float
    postprocess: float

    @property
    def total(self) -> float:
        return self.get_batch + self.transfer + self.forward + self.postprocess

    def avg_ms(self, n: int) -> _BatchTimings:
        scale = 1000.0 / n
        return _BatchTimings(
            get_batch=self.get_batch * scale,
            transfer=self.transfer * scale,
            forward=self.forward * scale,
            postprocess=self.postprocess * scale,
        )

    def __iadd__(self, other: _BatchTimings) -> _BatchTimings:
        self.get_batch += other.get_batch
        self.transfer += other.transfer
        self.forward += other.forward
        self.postprocess += other.postprocess
        return self


@dataclass
class InferenceResult:
    """Output of :func:`run_inference`.

    Attributes:
        embeddings: ``(H, W, 128)`` int8 quantized array.
        embeddings_std: Always ``None`` under v1.1 (deterministic sampling).
        scales: ``(H, W)`` float32 per-pixel scale factors for dequantization.
    """

    embeddings: np.ndarray
    embeddings_std: np.ndarray | None
    scales: np.ndarray


def _prepare_gpu(model: MultimodalBTInferenceModel, device: torch.device) -> bool:
    """Configure model and GPU for inference.

    On CUDA: converts to FP16, enables cuDNN benchmark mode, probes autocast dtypes.

    Returns:
        Whether to use FP16 (True when device is CUDA).
    """
    log_cuda_diagnostics(device)

    if device.type == "cuda":
        model.half()
        logger.info("Model converted to FP16")
        torch.backends.cudnn.benchmark = True
        logger.debug("cuDNN benchmark mode enabled")

    log_autocast_dtype_probe(device)
    return device.type == "cuda"


def _prepare_batch(
    dataset: MosaicChunkInferenceDataset,
    bucket_key: tuple[int, int],
    pixel_start: int,
    pixel_end: int,
) -> tuple[dict[str, np.ndarray], float]:
    """Run CPU-side batch preparation. Returns (batch, elapsed_seconds)."""
    t0 = time.monotonic()
    batch = dataset.get_bucket_batch(bucket_key, pixel_start, pixel_end)
    return batch, time.monotonic() - t0


def _run_gpu_sub_batch(
    model: MultimodalBTInferenceModel,
    batch: dict[str, np.ndarray],
    bucket_key: tuple[int, int],
    device: torch.device,
    use_fp16: bool,
    flat_out: np.ndarray,
    get_batch_secs: float,
    *,
    profile: bool = False,
    config: InferenceConfig | None = None,
    save_dim: int = EMBEDDING_DIM,
) -> _BatchTimings:
    """Transfer → forward → slice → write for a pre-prepared batch.

    Must be called inside a ``torch.no_grad()`` context.
    Writes into ``flat_out[global_idxs]`` in place.
    """
    tb1 = time.monotonic()

    global_idxs = batch["global_idxs"]
    s2_np = batch["s2"]
    s1_np = batch["s1"]

    s2_input = torch.from_numpy(s2_np).to(device, non_blocking=True)
    s1_input = torch.from_numpy(s1_np).to(device, non_blocking=True)
    if use_fp16:
        s2_input = s2_input.half()
        s1_input = s1_input.half()
    if device.type == "cuda":
        torch.cuda.synchronize()
    tb2 = time.monotonic()

    if profile:
        enable_model_profiling(model)

    z = model(s2_input, s1_input)  # (B, representation_dim) — 192 for v1.1
    if device.type == "cuda":
        torch.cuda.synchronize()
    tb3 = time.monotonic()

    if profile:
        disable_model_profiling(model)
        fwd_ms = (tb3 - tb2) * 1000
        log_profiled_batch_summary(
            PROFILE_BATCH_IDX,
            s2_input,
            s1_input,
            z,
            model,
            forward_ms=fwd_ms,
            device=device,
        )
        if config is not None:
            s2_target, _s1_target = bucket_key
            log_effective_tflops(
                forward_ms=fwd_ms,
                batch_size=len(global_idxs),
                seq_len=s2_target,
                d_model=config.latent_dim * 4,
                dim_feedforward=config.dim_feedforward,
                num_layers=config.num_encoder_layers,
            )

    # Slice 192-D rep to canonical save_dim, then to CPU.
    z = z[:, :save_dim].float()
    flat_out[global_idxs] = z.cpu().numpy()
    tb4 = time.monotonic()

    return _BatchTimings(
        get_batch=get_batch_secs,
        transfer=tb2 - tb1,
        forward=tb3 - tb2,
        postprocess=tb4 - tb3,
    )


def run_inference(
    model: MultimodalBTInferenceModel,
    dataset: MosaicChunkInferenceDataset,
    config: InferenceConfig,
    device: torch.device,
    on_batch: Callable[[int, int], None] | None = None,
) -> InferenceResult:
    """Run v1.1 inference on all valid pixels in a dataset.

    Args:
        model: Frozen v1.1 inference model in eval mode.
        dataset: Bucketed pixel dataset for one spatial chunk.
        config: Inference configuration.
        device: Torch device (cpu or cuda).
        on_batch: Optional callback invoked periodically with
            (sub_batch_idx, total_sub_batches). Used for stall detection in
            distributed actors.

    Returns:
        InferenceResult with:
          - embeddings: (H, W, 128) int8 quantized. Zeros for invalid pixels.
          - embeddings_std: Always None under v1.1.
          - scales: (H, W) float32 per-pixel scale factors.
    """
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")

    n_valid = len(dataset)
    batch_size = config.batch_size

    bucket_sizes = dataset.bucket_sizes()
    total_sub_batches = sum((n + batch_size - 1) // batch_size for n in bucket_sizes.values())

    # Save the canonical 128-D slice of the model's 192-D representation.
    # If the model is configured smaller (e.g. tiny test model), save the full width.
    save_dim = min(EMBEDDING_DIM, config.representation_dim)

    h, w = dataset.H, dataset.W
    flat_out = np.zeros((h * w, save_dim), dtype=np.float32)

    logger.info(
        "Starting v1.1 inference: %d buckets, %d sub-batches, %d valid pixels, batch_size=%d",
        len(bucket_sizes),
        total_sub_batches,
        n_valid,
        batch_size,
    )
    use_fp16 = _prepare_gpu(model, device)

    t_total = _BatchTimings(0.0, 0.0, 0.0, 0.0)
    pixels_processed = 0
    sub_batch_idx = 0
    t0 = time.monotonic()

    # Flatten (bucket_key, start, end) triples so the prefetch thread can
    # pipeline CPU batch prep across bucket boundaries.
    sub_batches: list[tuple[tuple[int, int], int, int]] = []
    for bucket_key, pixel_indices in dataset.iter_buckets(largest_first=True):
        n_in_bucket = int(pixel_indices.size)
        logger.debug(
            "Bucket %s: %d pixels in %d sub-batches",
            bucket_key,
            n_in_bucket,
            (n_in_bucket + batch_size - 1) // batch_size,
        )
        for start in range(0, n_in_bucket, batch_size):
            end = min(start + batch_size, n_in_bucket)
            sub_batches.append((bucket_key, start, end))

    # One prefetch worker pipelines CPU batch prep while GPU runs forward pass.
    # CPU prep (~165ms) < forward (~250ms), so the GPU consumes faster than the
    # prefetcher produces and a deeper queue would just sit full.
    with torch.no_grad(), ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as pool:
        next_future = pool.submit(_prepare_batch, dataset, *sub_batches[0]) if sub_batches else None

        for i, (bucket_key, start, end) in enumerate(sub_batches):
            assert next_future is not None
            batch, get_batch_secs = next_future.result()

            # Kick off prefetch of i+1 before the GPU work on i starts.
            next_future = (
                pool.submit(_prepare_batch, dataset, *sub_batches[i + 1]) if i + 1 < len(sub_batches) else None
            )

            bt = _run_gpu_sub_batch(
                model,
                batch,
                bucket_key,
                device,
                use_fp16,
                flat_out,
                get_batch_secs,
                profile=(sub_batch_idx == PROFILE_BATCH_IDX),
                config=config,
                save_dim=save_dim,
            )
            t_total += bt
            pixels_processed += end - start
            sub_batch_idx += 1

            if on_batch and (sub_batch_idx % 50 == 0 or sub_batch_idx == total_sub_batches):
                on_batch(sub_batch_idx, total_sub_batches)

            if sub_batch_idx % 200 == 0 or sub_batch_idx == total_sub_batches:
                elapsed = time.monotonic() - t0
                rate = pixels_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Sub-batch %d/%d — %d px — %.0f px/sec — %.1fs elapsed",
                    sub_batch_idx,
                    total_sub_batches,
                    pixels_processed,
                    rate,
                    elapsed,
                )
                avg = t_total.avg_ms(sub_batch_idx)
                logger.debug(
                    "  TIMING avg ms/sub-batch (n=%d): "
                    "get_batch=%.1f  transfer=%.1f  forward=%.1f  postprocess=%.1f  total=%.1f",
                    sub_batch_idx,
                    avg.get_batch,
                    avg.transfer,
                    avg.forward,
                    avg.postprocess,
                    avg.total,
                )

    out = flat_out.reshape(h, w, save_dim)
    out, scales = quantize_embeddings(out)
    logger.info("Quantized embeddings to int8 (scales shape %s)", scales.shape)

    elapsed = time.monotonic() - t0
    logger.info(
        "Inference complete: %d valid pixels, output shape %s, dtype %s, %.1fs total, %.0f px/sec avg",
        pixels_processed,
        out.shape,
        out.dtype,
        elapsed,
        pixels_processed / elapsed if elapsed > 0 else 0,
    )

    return InferenceResult(embeddings=out, embeddings_std=None, scales=scales)
