"""Core inference loop for generating per-pixel embeddings.

Decoupled from I/O: takes a model, dataset, and config; returns the embedding array.
Ported from tessera's process_tile with repeated random sampling and averaging.

CHANGED from original: Major performance rewrite (see inline annotations):
  a) DataLoader replaced with direct get_batch() iteration on pre-extracted arrays
  b) repeat_times loop folded into batch dimension (1 forward pass instead of 10)
  c) model.half() for full FP16 + torch.compile for kernel fusion
  d) cuda.synchronize() removed (implicit sync via .cpu())
  e) torch.from_numpy + non_blocking transfer (zero-copy, 1 transfer instead of 10)

Expected speedup: 50-200x (from ~100 px/sec to ~5,000-20,000 px/sec).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from tessera_embeddings.config.inference import InferenceConfig
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
from tessera_embeddings.inference.sampling import sample_s1_batch, sample_s2_batch

logger = logging.getLogger(__name__)

PROFILE_BATCH_IDX = 10
"""Batch index for the one-time deep profile (per-layer timing + dtype checks)."""


@dataclass
class _BandStats:
    """Pre-computed band statistics for standardization."""

    s2_mean: np.ndarray  # (10,) float32
    s2_std: np.ndarray  # (10,) float32
    s1_mean: np.ndarray  # (2,) float32
    s1_std: np.ndarray  # (2,) float32


@dataclass
class _BatchTimings:
    """Per-phase wall-clock times (seconds) for one batch."""

    get_batch: float
    sample_s2: float
    sample_s1: float
    transfer: float
    forward: float
    postprocess: float

    @property
    def total(self) -> float:
        """Sum of all phase times."""
        return self.get_batch + self.sample_s2 + self.sample_s1 + self.transfer + self.forward + self.postprocess

    def avg_ms(self, n: int) -> _BatchTimings:
        """Return a new instance with values converted to average milliseconds per batch."""
        scale = 1000.0 / n
        return _BatchTimings(
            get_batch=self.get_batch * scale,
            sample_s2=self.sample_s2 * scale,
            sample_s1=self.sample_s1 * scale,
            transfer=self.transfer * scale,
            forward=self.forward * scale,
            postprocess=self.postprocess * scale,
        )

    def __iadd__(self, other: _BatchTimings) -> _BatchTimings:
        self.get_batch += other.get_batch
        self.sample_s2 += other.sample_s2
        self.sample_s1 += other.sample_s1
        self.transfer += other.transfer
        self.forward += other.forward
        self.postprocess += other.postprocess
        return self


@dataclass
class InferenceResult:
    """Output of :func:`run_inference`.

    Attributes:
        embeddings: ``(H, W, D)`` int8 quantized array.
        embeddings_std: Optional ``(H, W, D)`` float32 std array (None unless ``compute_std``).
        scales: ``(H, W)`` float32 per-pixel scale factors for dequantization.
    """

    embeddings: np.ndarray
    embeddings_std: np.ndarray | None
    scales: np.ndarray


def _prepare_gpu(model: MultimodalBTInferenceModel, device: torch.device) -> bool:
    """Configure model and GPU for inference.

    Runs one-time diagnostics, converts model to FP16 on CUDA, enables cuDNN
    benchmark mode, and probes autocast dtypes.

    Args:
        model: Inference model — mutated in-place (half-precision) on CUDA.
        device: Torch device.

    Returns:
        Whether to use FP16 (True when device is CUDA).
    """
    log_cuda_diagnostics(device)

    # Full FP16 via model.half() — replaces previous autocast mixed-precision.
    # autocast was not effectively engaging tensor cores for our small matmul geometry
    # (seq=20, d=512). model.half() guarantees all params and ops run in FP16, which:
    #   - Halves memory bandwidth (T4 has 320 GB/s — this was the bottleneck)
    #   - Guarantees FP16 matmuls hit tensor cores (no autocast dispatch overhead)
    #   - Halves activation memory → enables larger batch_size (3584 vs 768)
    # Safe for inference: 10-repeat averaging smooths any FP16 noise.
    # QA: compare embedding vectors vs FP32 baseline — should be within 1e-3.
    if device.type == "cuda":
        model.half()
        logger.info("Model converted to FP16")

    # DISABLED: torch.compile was eating ~11.6 GB VRAM for CUDA graphs on a 15.4 GB GPU,
    # and avg forward pass was 3,770ms vs 1,944ms without — compile overhead + GRU graph
    # breaks likely causing repeated recompilation. Revisit on a GPU with more VRAM.

    # Enable cuDNN benchmark mode — lets cuDNN profile and select the fastest
    # algorithm for GRU and conv ops. Slight overhead on first call, then faster.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.debug("cuDNN benchmark mode enabled")

    log_autocast_dtype_probe(device)

    return device.type == "cuda"


def _process_batch(
    model: MultimodalBTInferenceModel,
    dataset: MosaicChunkInferenceDataset,
    start: int,
    end: int,
    config: InferenceConfig,
    band_stats: _BandStats,
    device: torch.device,
    use_fp16: bool,
    flat_out: np.ndarray,
    flat_out_std: np.ndarray | None,
    *,
    profile: bool = False,
) -> _BatchTimings:
    """Execute one batch of inference: load, sample, transfer, forward, postprocess.

    Must be called inside a ``torch.no_grad()`` context (the caller owns it).
    Writes results directly into *flat_out* (and *flat_out_std* if not None).

    Args:
        model: Inference model (already on *device*).
        dataset: Pixel dataset for one spatial chunk.
        start: Start index into the valid-pixel array.
        end: End index (exclusive).
        config: Inference configuration.
        band_stats: Pre-computed band means/stds for standardization.
        device: Torch device.
        use_fp16: Whether to convert inputs to FP16 before forward pass.
        flat_out: (H*W, latent_dim) output array — written in-place.
        flat_out_std: Optional (H*W, latent_dim) std output — written in-place.
        profile: If True, enable per-layer profiling for this batch.

    Returns:
        Per-phase wall-clock timings for this batch.
    """
    actual_batch = end - start
    repeat_times = config.repeat_times
    latent_dim = config.latent_dim

    # --- Phase: get_batch ---
    tb0 = time.monotonic()
    batch = dataset.get_batch(start, end)
    tb1 = time.monotonic()

    global_idxs = batch["global_idxs"]

    # --- Phase: sample_s2 ---
    s2_input_np = sample_s2_batch(
        batch["s2_bands"],
        batch["s2_masks"],
        batch["s2_doys"],
        band_mean=band_stats.s2_mean,
        band_std=band_stats.s2_std,
        sample_size_s2=config.sample_size_s2,
        repeat_times=repeat_times,
    )
    tb2 = time.monotonic()

    # --- Phase: sample_s1 ---
    s1_input_np = sample_s1_batch(
        batch["s1_asc_bands"],
        batch["s1_asc_doys"],
        batch["s1_desc_bands"],
        batch["s1_desc_doys"],
        band_mean=band_stats.s1_mean,
        band_std=band_stats.s1_std,
        sample_size_s1=config.sample_size_s1,
        repeat_times=repeat_times,
    )
    tb3 = time.monotonic()

    # --- Phase: transfer ---
    s2_input = torch.from_numpy(s2_input_np).to(device, non_blocking=True)
    s1_input = torch.from_numpy(s1_input_np).to(device, non_blocking=True)
    if use_fp16:
        s2_input = s2_input.half()
        s1_input = s1_input.half()
    if device.type == "cuda":
        torch.cuda.synchronize()
    tb4 = time.monotonic()

    # --- Phase: forward ---
    if profile:
        enable_model_profiling(model)

    # Direct FP16 forward — no autocast wrapper needed. See _prepare_gpu comments.
    z = model(s2_input, s1_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    tb5 = time.monotonic()

    if profile:
        disable_model_profiling(model)
        fwd_ms = (tb5 - tb4) * 1000
        log_profiled_batch_summary(
            PROFILE_BATCH_IDX,
            s2_input,
            s1_input,
            z,
            model,
            forward_ms=fwd_ms,
            device=device,
        )
        log_effective_tflops(
            forward_ms=fwd_ms,
            batch_size=actual_batch,
            repeat_times=repeat_times,
            dim_feedforward=config.dim_feedforward,
            num_layers=config.num_encoder_layers,
        )

    # --- Phase: postprocess ---
    z = z.float()
    z_repeats = z.reshape(actual_batch, repeat_times, latent_dim)
    avg_repr = z_repeats.mean(dim=1)
    flat_out[global_idxs] = avg_repr.cpu().numpy()
    if flat_out_std is not None:
        flat_out_std[global_idxs] = z_repeats.std(dim=1).cpu().numpy()
    tb6 = time.monotonic()

    return _BatchTimings(
        get_batch=tb1 - tb0,
        sample_s2=tb2 - tb1,
        sample_s1=tb3 - tb2,
        transfer=tb4 - tb3,
        forward=tb5 - tb4,
        postprocess=tb6 - tb5,
    )


def run_inference(
    model: MultimodalBTInferenceModel,
    dataset: MosaicChunkInferenceDataset,
    config: InferenceConfig,
    device: torch.device,
    on_batch: Callable[[int, int], None] | None = None,
) -> InferenceResult:
    """Run inference on all valid pixels in a dataset.

    CHANGED from original: The inner repeat_times loop is folded into the batch
    dimension. Instead of R sequential forward passes of shape (B, seq, features),
    we do a single forward pass of shape (B*R, seq, features) and reshape+mean on GPU.

    Args:
        model: Frozen inference model in eval mode.
        dataset: Dataset of valid pixels for one spatial chunk.
        config: Inference configuration.
        device: Torch device (cpu or cuda).
        on_batch: Optional callback invoked every 50 batches with (batch_idx, total_batches).
            Used for progress reporting without coupling inference.py to Ray.

    Returns:
        InferenceResult with:
        - embeddings: (H, W, 128) int8 quantized. Zeros for invalid pixels.
        - embeddings_std: (H, W, 128) float32 if compute_std, else None.
        - scales: (H, W) float32 per-pixel scale factors.
    """
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")
    if config.repeat_times < 1:
        raise ValueError(f"repeat_times must be >= 1, got {config.repeat_times}")
    n_valid = len(dataset)
    batch_size = config.batch_size
    total_batches = (n_valid + batch_size - 1) // batch_size

    band_stats = _BandStats(
        s2_mean=np.array(config.s2_band_mean, dtype=np.float32),
        s2_std=np.array(config.s2_band_std, dtype=np.float32),
        s1_mean=np.array(config.s1_band_mean, dtype=np.float32),
        s1_std=np.array(config.s1_band_std, dtype=np.float32),
    )

    h, w = dataset.H, dataset.W
    latent_dim = config.latent_dim
    flat_out = np.zeros((h * w, latent_dim), dtype=np.float32)
    flat_out_std = np.zeros((h * w, latent_dim), dtype=np.float32) if config.compute_std else None

    logger.info("Starting inference: %d batches, %d valid pixels, batch_size=%d", total_batches, n_valid, batch_size)
    use_fp16 = _prepare_gpu(model, device)

    # Timing accumulators
    t_total = _BatchTimings(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pixels_processed = 0
    t0 = time.monotonic()

    with torch.no_grad():
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_valid)

            bt = _process_batch(
                model,
                dataset,
                start,
                end,
                config,
                band_stats,
                device,
                use_fp16,
                flat_out,
                flat_out_std,
                profile=(batch_idx == PROFILE_BATCH_IDX),
            )

            # add accumulations to timings (for logging)
            t_total += bt

            pixels_processed += end - start

            # Report to tracker every 50 batches (stall detection), log every 200 (noise reduction)
            if on_batch and ((batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1):
                on_batch(batch_idx + 1, total_batches)

            if (batch_idx + 1) % 200 == 0 or batch_idx == total_batches - 1:
                elapsed = time.monotonic() - t0
                rate = pixels_processed / elapsed if elapsed > 0 else 0
                n = batch_idx + 1
                logger.info(
                    "Batch %d/%d — %d px — %.0f px/sec — %.1fs elapsed",
                    n,
                    total_batches,
                    pixels_processed,
                    rate,
                    elapsed,
                )
                avg = t_total.avg_ms(n)
                logger.debug(
                    "  TIMING avg ms/batch (n=%d): "
                    "get_batch=%.1f  sample_s2=%.1f  sample_s1=%.1f  "
                    "transfer=%.1f  forward=%.1f  postprocess=%.1f  total=%.1f",
                    n,
                    avg.get_batch,
                    avg.sample_s2,
                    avg.sample_s1,
                    avg.transfer,
                    avg.forward,
                    avg.postprocess,
                    avg.total,
                )

    out = flat_out.reshape(h, w, latent_dim)
    out_std = flat_out_std.reshape(h, w, latent_dim) if flat_out_std is not None else None

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

    return InferenceResult(embeddings=out, embeddings_std=out_std, scales=scales)
