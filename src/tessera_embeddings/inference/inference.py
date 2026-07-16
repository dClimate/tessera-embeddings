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
    transformer_flops,
)
from tessera_embeddings.inference.quantization import quantize_rows_torch

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


def _prepare_gpu(model: MultimodalBTInferenceModel, device: torch.device) -> torch.dtype | None:
    """Configure model and GPU for inference.

    On CUDA: converts to BF16 (preferred) and enables cuDNN benchmark mode.
    Falls back to FP16 on GPUs that don't support BF16 (e.g. T4, compute
    capability < 8.0). BF16 is strongly preferred — it has the same exponent
    range as FP32 and avoids overflow on large intermediate activations. The
    FP16 fallback is functional but carries a risk of inf/NaN from saturation
    at 65504; treat it as a best-effort path, not a validated production config.

    The dtype conversion is idempotent: actors reuse one persistent model
    across every strip and chunk, so once converted the model already carries
    the target dtype and we skip the (non-trivial) re-cast. This hoists the
    one-time conversion cost to the first strip instead of paying it per strip.

    Returns:
        The reduced-precision dtype to use for inputs (bfloat16 or float16), or None
        on CPU (inputs stay float32).
    """
    log_cuda_diagnostics(device)

    if device.type != "cuda":
        log_autocast_dtype_probe(device, dtype=None)
        return None

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    current_dtype = next(model.parameters()).dtype
    if current_dtype == dtype:
        logger.debug("Model already in %s; skipping conversion", dtype)
    elif dtype == torch.bfloat16:
        model.bfloat16()
        logger.info("Model converted to BF16")
    else:
        model.half()
        logger.warning("BF16 not supported on this GPU; falling back to FP16")

    # benchmark=False: with variable bucket shapes (per-bucket seq lengths +
    # partial final sub-batches) the autotuner re-searches constantly for little
    # gain, and inflates the host-side cuDNN footprint by trial-loading multiple
    # algorithms and holding the largest workspace it tried. Disabling it lowers
    # the first-batch host-RAM plateau. See inference RAM investigation.
    torch.backends.cudnn.benchmark = False
    logger.debug("cuDNN benchmark mode disabled (variable input shapes)")
    log_autocast_dtype_probe(device, dtype=dtype)
    return dtype


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


def _transfer_and_forward(
    model: MultimodalBTInferenceModel,
    batch: dict[str, np.ndarray],
    bucket_key: tuple[int, int],
    device: torch.device,
    reduced_precision_dtype: torch.dtype | None,
    save_dim: int,
    *,
    profile: bool = False,
    config: InferenceConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Transfer → forward → slice → quantize on-device → copy int8 to host.

    Must be called inside a ``torch.no_grad()`` context, on the thread that owns
    the GPU. Returns ``(q_host, scales_host, global_idxs, transfer_secs,
    forward_secs)`` where ``q_host`` is the ``(B, save_dim)`` int8 array and
    ``scales_host`` the ``(B,)`` float32 scales, both already on the host and
    ready for :func:`_write_quantized_rows`.

    Quantization runs on-device (:func:`quantize_rows_torch`) in the forward's
    compute stream, so its seq-length-independent cost folds into GPU work rather
    than gating a separate CPU stage — the case that hurt low-observation chunks,
    whose forwards are tiny. It also shrinks the D2H copy ~4x (int8 + scales vs.
    float32 embeddings).

    On CUDA the transfer, forward, and quantize run on the default stream with no
    intervening ``synchronize`` — the trailing ``.cpu()`` is the sync point, so
    ordering is preserved. Per-phase times are coarse wall clock (for rough
    logging, not precise attribution); ``profile`` adds one ``synchronize`` so
    the one-shot profile batch reports an accurate forward time.
    """
    tb1 = time.monotonic()

    global_idxs = batch["global_idxs"]
    s2_input = torch.from_numpy(batch["s2"]).to(device, non_blocking=True)
    s1_input = torch.from_numpy(batch["s1"]).to(device, non_blocking=True)
    if reduced_precision_dtype is not None:
        s2_input = s2_input.to(reduced_precision_dtype)
        s1_input = s1_input.to(reduced_precision_dtype)
    tb2 = time.monotonic()

    if profile:
        enable_model_profiling(model)

    z = model(s2_input, s1_input)  # (B, representation_dim) — 192 for v1.1
    if profile and device.type == "cuda":
        # Only the profile batch pays a sync, to time the forward accurately.
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
            log_effective_tflops(
                forward_ms=fwd_ms,
                batch_size=len(global_idxs),
                seq_len=bucket_key[0],
                d_model=config.latent_dim * 4,
                dim_feedforward=config.dim_feedforward,
                num_layers=config.num_encoder_layers,
                s1_seq_len=bucket_key[1],
            )

    # Slice 192-D rep to canonical save_dim, quantize on-device, then pull the
    # compact int8 + scales to host in one D2H. The .cpu() blocks until the
    # forward+quantize complete, which is what bounds the GPU work for this batch.
    q, scales = quantize_rows_torch(z[:, :save_dim])
    q_host = q.cpu().numpy()
    scales_host = scales.cpu().numpy()
    return q_host, scales_host, global_idxs, tb2 - tb1, tb3 - tb2


def _write_quantized_rows(
    q_host: np.ndarray,
    scales_host: np.ndarray,
    global_idxs: np.ndarray,
    flat_q: np.ndarray,
    flat_scales: np.ndarray,
) -> None:
    """Scatter already-quantized rows into the flat output buffers in place.

    A cheap memcpy — the arithmetic now happens on-device in
    :func:`_transfer_and_forward`. Each call writes only its own bucket's
    ``global_idxs`` (disjoint across sub-batches), so writes never contend.
    """
    flat_q[global_idxs] = q_host
    flat_scales[global_idxs] = scales_host


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
          - embeddings: (H, W, 128) int8 quantized. Zeros for pixels whose
            embeddings can't be generated (failed validity, or outside the ROI
            but inside the bbox).
          - embeddings_std: Always None under v1.1.
          - scales: (H, W) float32 per-pixel scale factors. NaN for those same
            can't-generate pixels.
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
    # Quantize per bucket into skinny int8 + scale buffers instead of accumulating
    # the whole chunk in float32 (4x smaller resident accumulator). Pixels whose
    # embeddings can't be generated — failed validity, or outside the ROI but
    # inside the bbox (zeroed during ingest, so they fail the nonzero check and
    # never enter a bucket) — keep their initial values: embeddings 0 in every
    # band, scale NaN. Only pixels run through the model overwrite their slot via
    # flat_scales[global_idxs] = s. This matches the NaN-scale / 0-embedding fill
    # assembly applies to skipped and non-intersecting chunks.
    flat_q = np.zeros((h * w, save_dim), dtype=np.int8)
    flat_scales = np.full(h * w, np.nan, dtype=np.float32)

    logger.info(
        "Starting v1.1 inference: %d buckets, %d sub-batches, %d valid pixels, batch_size=%d",
        len(bucket_sizes),
        total_sub_batches,
        n_valid,
        batch_size,
    )
    reduced_precision_dtype = _prepare_gpu(model, device)

    t_total = _BatchTimings(0.0, 0.0, 0.0, 0.0)
    pixels_processed = 0
    # px/s conflates sequence length across chunks (a sparse chunk's pixel is
    # ~10x cheaper than a dense one's), so also track tokens (= pixels x
    # (T_s2 + T_s1)) and transformer FLOPs for density-neutral throughput.
    tokens_processed = 0
    flops_processed = 0
    d_model = config.latent_dim * 4
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

    # One prefetch worker pipelines CPU batch prep for i+1 while the GPU runs i.
    # CPU prep (~165ms) < forward, so the GPU consumes faster than the prefetcher
    # produces and a deeper queue would just sit full. Quantization is no longer a
    # CPU stage — it runs on-device inside _transfer_and_forward — so the write is
    # a cheap inline scatter with no second pool (that pool gated low-observation
    # chunks, whose tiny forwards couldn't hide the thread handoff).
    with torch.no_grad(), ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") as pool:
        next_future = pool.submit(_prepare_batch, dataset, *sub_batches[0]) if sub_batches else None

        for i, (bucket_key, start, end) in enumerate(sub_batches):
            assert next_future is not None
            batch, get_batch_secs = next_future.result()

            # Kick off prefetch of i+1 before the GPU work on i starts.
            next_future = (
                pool.submit(_prepare_batch, dataset, *sub_batches[i + 1]) if i + 1 < len(sub_batches) else None
            )

            tb0 = time.monotonic()
            q_host, scales_host, global_idxs, transfer_secs, forward_secs = _transfer_and_forward(
                model,
                batch,
                bucket_key,
                device,
                reduced_precision_dtype,
                save_dim,
                profile=(sub_batch_idx == PROFILE_BATCH_IDX),
                config=config,
            )
            _write_quantized_rows(q_host, scales_host, global_idxs, flat_q, flat_scales)

            t_total += _BatchTimings(
                get_batch=get_batch_secs,
                transfer=transfer_secs,
                forward=forward_secs,
                postprocess=time.monotonic() - tb0 - transfer_secs - forward_secs,
            )
            n_px = end - start
            pixels_processed += n_px
            tokens_processed += n_px * (bucket_key[0] + bucket_key[1])
            flops_processed += transformer_flops(
                n_px,
                bucket_key[0],
                bucket_key[1],
                d_model=d_model,
                dim_feedforward=config.dim_feedforward,
                num_layers=config.num_encoder_layers,
            )
            sub_batch_idx += 1

            if on_batch and (sub_batch_idx % 50 == 0 or sub_batch_idx == total_sub_batches):
                on_batch(sub_batch_idx, total_sub_batches)

            if sub_batch_idx % 200 == 0 or sub_batch_idx == total_sub_batches:
                elapsed = time.monotonic() - t0
                rate = pixels_processed / elapsed if elapsed > 0 else 0
                logger.info(
                    "Sub-batch %d/%d — %d px — %.0f px/sec — %.0f tok/sec — %.2f eff TFLOPS — %.1fs elapsed",
                    sub_batch_idx,
                    total_sub_batches,
                    pixels_processed,
                    rate,
                    tokens_processed / elapsed if elapsed > 0 else 0,
                    flops_processed / elapsed / 1e12 if elapsed > 0 else 0,
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

    # Rows were already quantized per bucket on arrival; just reshape to (H,W,*).
    out = flat_q.reshape(h, w, save_dim)
    scales = flat_scales.reshape(h, w)
    logger.info("Quantized embeddings to int8 (scales shape %s)", scales.shape)

    elapsed = time.monotonic() - t0
    logger.info(
        "Inference complete: %d valid pixels, output shape %s, dtype %s, %.1fs total, "
        "%.0f px/sec avg, %.0f tok/sec avg, %.2f eff TFLOPS avg (transformer-only)",
        pixels_processed,
        out.shape,
        out.dtype,
        elapsed,
        pixels_processed / elapsed if elapsed > 0 else 0,
        tokens_processed / elapsed if elapsed > 0 else 0,
        flops_processed / elapsed / 1e12 if elapsed > 0 else 0,
    )

    return InferenceResult(embeddings=out, embeddings_std=None, scales=scales)
