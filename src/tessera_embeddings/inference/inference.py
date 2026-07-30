"""Core inference loop for Tessera embeddings (v1.1 and v2 Large).

Iterates buckets (pixels sharing a ``(s2_target, s1_target)`` key), then
sub-batches each bucket by ``config.batch_size``. Each sub-batch goes through
a single forward pass; the model's representation is sliced to the canonical
128-D downstream width (a real slice for v1.1's 192-D output, a no-op for v2
Large's native 128-D) and written into a flat per-pixel output array.

"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from tessera_embeddings.config.inference import EMBEDDING_DIM, PREFETCH_DEPTH, InferenceConfig
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
from tessera_embeddings.inference.quantization import quantize_rows_torch, raise_on_nonfinite_scales

logger = logging.getLogger(__name__)

PROFILE_BATCH_IDX = 10
"""Batch index for the one-time deep profile (per-layer timing + dtype checks)."""

_SERIAL_LOOP_ENV = "TESSERA_SERIAL_GPU_LOOP"
"""Set to any non-empty value to force the synchronous per-batch GPU loop
(pageable transfers, host sync per sub-batch) instead of the pipelined one.
Escape hatch for debugging the async pipeline in production."""


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
        scales: ``(H, W)`` float32 per-pixel scale factors for dequantization.
    """

    embeddings: np.ndarray
    scales: np.ndarray


def _prepare_gpu(model: MultimodalBTInferenceModel, device: torch.device) -> None:
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

    The model's own weights are the single source of the compute dtype from here
    on: inputs stay FP32 and each backbone casts its band channels itself, which
    is what keeps the DOY channel out of BF16 (see
    :class:`~tessera_embeddings.inference.models.modules.TemporalPositionalEncoder`).
    """
    log_cuda_diagnostics(device)

    if device.type != "cuda":
        log_autocast_dtype_probe(device, dtype=None)
        return

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
    # Transferred and handed to the model as FP32. The backbones cast their band
    # channels to the weights' dtype internally, which leaves the DOY channel in
    # FP32 — a whole-tensor BF16 cast here would round DOY 257 to 256 and lose
    # one-day resolution for the last third of the year.
    s2_input = torch.from_numpy(batch["s2"]).to(device, non_blocking=True)
    s1_input = torch.from_numpy(batch["s1"]).to(device, non_blocking=True)
    tb2 = time.monotonic()

    if profile:
        enable_model_profiling(model)

    z = model(s2_input, s1_input)  # (B, representation_dim) — 192 for v1.1, 128 for v2 Large
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

    # Slice the representation to the canonical save_dim, quantize on-device, then pull the
    # compact int8 + scales to host in one D2H. The .cpu() blocks until the
    # forward+quantize complete, which is what bounds the GPU work for this batch.
    q, scales = quantize_rows_torch(z[:, :save_dim])
    q_host = q.cpu().numpy()
    scales_host = scales.cpu().numpy()
    # Finiteness is validated on the host-side scales (equivalent to checking
    # the embeddings — see raise_on_nonfinite_scales) so the GPU pipeline never
    # pays a device-wide sync for the check.
    raise_on_nonfinite_scales(scales_host)
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


_BatchIter = Iterator[tuple[int, tuple[int, int], int, int, dict[str, np.ndarray], float]]
"""Yields (i, bucket_key, start, end, prepared_batch, get_batch_secs)."""


@dataclass
class _LoopProgress:
    """Per-sub-batch accounting + periodic logging shared by both GPU loops."""

    config: InferenceConfig
    d_model: int
    total_sub_batches: int
    t0: float
    on_batch: Callable[[int, int], None] | None
    t_total: _BatchTimings = field(default_factory=lambda: _BatchTimings(0.0, 0.0, 0.0, 0.0))
    pixels: int = 0
    tokens: int = 0
    flops: int = 0
    sub_batch_idx: int = 0
    # Worst single-batch CPU prep this call — averages hide the spikes that
    # actually starve the GPU (feed-fix telemetry; see actors' reserve_cpus).
    get_batch_max: float = 0.0
    # Per-bucket (pixels, sub-batches) — the workload-side input to the
    # adaptive token-budget gate (temp/token-budget-batching-findings.md §Step 0):
    # the expected gain is bucket occupancy weighted by the per-shape speedup
    # curve, and occupancy varies by region, so it's logged on every run.
    # Counts only — per-bucket forward times are NOT recorded because the
    # pipelined loop's timings are enqueue-side and would mislead.
    bucket_px: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)

    def record(self, bucket_key: tuple[int, int], n_px: int, timings: _BatchTimings) -> None:
        self.t_total += timings
        self.get_batch_max = max(self.get_batch_max, timings.get_batch)
        self.pixels += n_px
        self.tokens += n_px * (bucket_key[0] + bucket_key[1])
        px, nb = self.bucket_px.get(bucket_key, (0, 0))
        self.bucket_px[bucket_key] = (px + n_px, nb + 1)
        self.flops += transformer_flops(
            n_px,
            bucket_key[0],
            bucket_key[1],
            d_model=self.d_model,
            dim_feedforward=self.config.dim_feedforward,
            num_layers=self.config.num_encoder_layers,
        )
        self.sub_batch_idx += 1

        if self.on_batch and (self.sub_batch_idx % 50 == 0 or self.sub_batch_idx == self.total_sub_batches):
            self.on_batch(self.sub_batch_idx, self.total_sub_batches)

        if self.sub_batch_idx % 200 == 0 or self.sub_batch_idx == self.total_sub_batches:
            elapsed = time.monotonic() - self.t0
            logger.info(
                "Sub-batch %d/%d — %d px — %.0f px/sec — %.0f tok/sec — %.2f eff TFLOPS — %.1fs elapsed",
                self.sub_batch_idx,
                self.total_sub_batches,
                self.pixels,
                self.pixels / elapsed if elapsed > 0 else 0,
                self.tokens / elapsed if elapsed > 0 else 0,
                self.flops / elapsed / 1e12 if elapsed > 0 else 0,
                elapsed,
            )
            avg = self.t_total.avg_ms(self.sub_batch_idx)
            logger.debug(
                "  TIMING avg ms/sub-batch (n=%d): "
                "get_batch=%.1f (max %.0f)  transfer=%.1f  forward=%.1f  postprocess=%.1f  total=%.1f",
                self.sub_batch_idx,
                avg.get_batch,
                self.get_batch_max * 1000,
                avg.transfer,
                avg.forward,
                avg.postprocess,
                avg.total,
            )


def _serial_loop(
    model: MultimodalBTInferenceModel,
    batches: _BatchIter,
    config: InferenceConfig,
    device: torch.device,
    save_dim: int,
    flat_q: np.ndarray,
    flat_scales: np.ndarray,
    progress: _LoopProgress,
) -> None:
    """Synchronous per-batch loop: transfer → forward → sync D2H → scatter.

    The CPU path, and the CUDA escape hatch (``TESSERA_SERIAL_GPU_LOOP``).
    """
    for i, bucket_key, start, end, batch, get_batch_secs in batches:
        _run_sync_batch(
            model,
            batch,
            bucket_key,
            end - start,
            get_batch_secs,
            config,
            device,
            save_dim,
            flat_q,
            flat_scales,
            progress,
            profile=(i == PROFILE_BATCH_IDX),
        )


def _run_sync_batch(
    model: MultimodalBTInferenceModel,
    batch: dict,
    bucket_key: tuple[int, int],
    n_px: int,
    get_batch_secs: float,
    config: InferenceConfig,
    device: torch.device,
    save_dim: int,
    flat_q: np.ndarray,
    flat_scales: np.ndarray,
    progress: _LoopProgress,
    *,
    profile: bool,
) -> None:
    """One synchronous transfer → forward → scatter batch, with timing record.

    Shared by ``_serial_loop`` and the pipelined loop's profile batch so the
    postprocess-timing arithmetic lives once.
    """
    tb0 = time.monotonic()
    q_host, scales_host, global_idxs, transfer_secs, forward_secs = _transfer_and_forward(
        model, batch, bucket_key, device, save_dim, profile=profile, config=config
    )
    _write_quantized_rows(q_host, scales_host, global_idxs, flat_q, flat_scales)
    progress.record(
        bucket_key,
        n_px,
        _BatchTimings(
            get_batch=get_batch_secs,
            transfer=transfer_secs,
            forward=forward_secs,
            postprocess=time.monotonic() - tb0 - transfer_secs - forward_secs,
        ),
    )


class _PinnedSlot:
    """Reusable pinned staging buffers for one in-flight sub-batch.

    Inputs are staged as flat float32 buffers sized to the chunk's largest
    (batch, seq-len) so any sub-batch shape carves a contiguous view — a
    contiguous pinned view is what makes ``.to(device, non_blocking=True)`` a
    single async DMA instead of a hidden staging copy.
    """

    def __init__(self, batch_size: int, t_s2_max: int, t_s1_max: int, save_dim: int) -> None:
        self.s2 = torch.empty(batch_size * t_s2_max * 11, dtype=torch.float32, pin_memory=True)
        self.s1 = torch.empty(batch_size * t_s1_max * 3, dtype=torch.float32, pin_memory=True)
        self.q = torch.empty((batch_size, save_dim), dtype=torch.int8, pin_memory=True)
        self.scales = torch.empty(batch_size, dtype=torch.float32, pin_memory=True)
        self.done = torch.cuda.Event()


@dataclass
class _InFlight:
    """Bookkeeping for a sub-batch whose GPU work is enqueued but not drained.

    Holds the device-side ``q``/``scales`` refs until the D2H completes so the
    caching allocator cannot hand their blocks to a later batch while the
    async copy still reads them (belt-and-braces; the copy is stream-ordered).
    """

    slot: _PinnedSlot
    global_idxs: np.ndarray
    bucket_key: tuple[int, int]
    q_dev: torch.Tensor
    scales_dev: torch.Tensor
    get_batch: float
    transfer: float
    forward: float


def _pipelined_gpu_loop(
    model: MultimodalBTInferenceModel,
    batches: _BatchIter,
    config: InferenceConfig,
    device: torch.device,
    save_dim: int,
    flat_q: np.ndarray,
    flat_scales: np.ndarray,
    sub_batches: list[tuple[tuple[int, int], int, int]],
    progress: _LoopProgress,
) -> None:
    """Two-deep asynchronous GPU loop: batch i+1 is enqueued while i executes.

    Everything runs on the current CUDA stream in the same op order as the
    serial loop (bit-identical results); the difference is purely *when* the
    host waits. Inputs are staged in pinned buffers (async H2D), outputs copy
    back asynchronously with a CUDA event per batch, and the host only blocks
    on the event of the batch one behind — so the GPU always has the next
    sub-batch's work queued and never idles on Python bookkeeping, H2D, or the
    scatter-back. Per-phase TIMING numbers here are *enqueue-side* costs
    (postprocess = drain wait); wall-clock px/s and tok/s are unaffected.

    The one-shot deep profile still runs through the synchronous path: the
    pipeline is drained first, so its per-layer timings stay accurate.
    """
    if not sub_batches:
        return
    t_s2_max = max(key[0] for key, _, _ in sub_batches)
    t_s1_max = max(key[1] for key, _, _ in sub_batches)
    slots = [_PinnedSlot(config.batch_size, t_s2_max, t_s1_max, save_dim) for _ in range(2)]
    inflight: deque[_InFlight] = deque()

    def _drain_oldest() -> None:
        rec = inflight.popleft()
        n = len(rec.global_idxs)
        t_wait = time.monotonic()
        rec.slot.done.synchronize()
        scales_host = rec.slot.scales[:n].numpy()
        raise_on_nonfinite_scales(scales_host)
        flat_q[rec.global_idxs] = rec.slot.q[:n].numpy()
        flat_scales[rec.global_idxs] = scales_host
        progress.record(
            rec.bucket_key,
            n,
            _BatchTimings(rec.get_batch, rec.transfer, rec.forward, time.monotonic() - t_wait),
        )

    for i, bucket_key, start, end, batch, get_batch_secs in batches:
        if i == PROFILE_BATCH_IDX:
            # The deep profile needs a quiet device: drain, then run serially.
            while inflight:
                _drain_oldest()
            _run_sync_batch(
                model,
                batch,
                bucket_key,
                end - start,
                get_batch_secs,
                config,
                device,
                save_dim,
                flat_q,
                flat_scales,
                progress,
                profile=True,
            )
            continue

        slot = slots[i % 2]  # slot of batch i-2 was drained during iteration i-1
        global_idxs = batch["global_idxs"]
        n = len(global_idxs)

        tb1 = time.monotonic()
        s2_np, s1_np = batch["s2"], batch["s1"]
        s2_pin = slot.s2[: s2_np.size].view(s2_np.shape)
        s1_pin = slot.s1[: s1_np.size].view(s1_np.shape)
        s2_pin.copy_(torch.from_numpy(s2_np))
        s1_pin.copy_(torch.from_numpy(s1_np))
        # FP32 all the way into the model — see _transfer_and_forward on why the
        # band/DOY split has to happen inside the backbones, not here.
        s2_dev = s2_pin.to(device, non_blocking=True)
        s1_dev = s1_pin.to(device, non_blocking=True)
        tb2 = time.monotonic()

        z = model(s2_dev, s1_dev)
        q, scales = quantize_rows_torch(z[:, :save_dim])
        slot.q[:n].copy_(q, non_blocking=True)
        slot.scales[:n].copy_(scales, non_blocking=True)
        slot.done.record()
        tb3 = time.monotonic()

        inflight.append(
            _InFlight(
                slot=slot,
                global_idxs=global_idxs,
                bucket_key=bucket_key,
                q_dev=q,
                scales_dev=scales,
                get_batch=get_batch_secs,
                transfer=tb2 - tb1,
                forward=tb3 - tb2,
            )
        )
        if len(inflight) == 2:
            _drain_oldest()

    while inflight:
        _drain_oldest()


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
          - scales: (H, W) float32 per-pixel scale factors. NaN for those same
            can't-generate pixels.
    """
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")

    n_valid = len(dataset)
    batch_size = config.batch_size

    bucket_sizes = dataset.bucket_sizes()
    total_sub_batches = sum((n + batch_size - 1) // batch_size for n in bucket_sizes.values())

    # Save the canonical 128-D slice of the model's representation — a real slice
    # under v1.1 (192-D), the identity under v2 Large (native 128-D). If the model
    # is configured smaller (e.g. tiny test model), save the full width.
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
        "Starting %s inference: %d buckets, %d sub-batches, %d valid pixels, batch_size=%d",
        config.model_version,
        len(bucket_sizes),
        total_sub_batches,
        n_valid,
        batch_size,
    )
    _prepare_gpu(model, device)

    # px/s conflates sequence length across chunks (a sparse chunk's pixel is
    # ~10x cheaper than a dense one's), so _LoopProgress also tracks tokens
    # (= pixels x (T_s2 + T_s1)) and transformer FLOPs for density-neutral
    # throughput reporting.
    d_model = config.latent_dim * 4
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

    progress = _LoopProgress(
        config=config,
        d_model=d_model,
        total_sub_batches=total_sub_batches,
        t0=t0,
        on_batch=on_batch,
    )

    # Two prep workers keep PREFETCH_DEPTH batches staged: depth 1 starved the
    # GPU whenever a forward ran shorter than one CPU prep (partial sub-batches,
    # short sequences). Quantization is not a CPU stage — it runs on-device —
    # so the scatter-back is a cheap memcpy either way.
    use_pipelined = device.type == "cuda" and not os.environ.get(_SERIAL_LOOP_ENV)
    with torch.no_grad(), ThreadPoolExecutor(max_workers=PREFETCH_DEPTH, thread_name_prefix="prefetch") as pool:
        prefetched: deque[Future[tuple[dict[str, np.ndarray], float]]] = deque()
        for j in range(min(PREFETCH_DEPTH, len(sub_batches))):
            prefetched.append(pool.submit(_prepare_batch, dataset, *sub_batches[j]))

        def _batches() -> _BatchIter:
            for i, (bucket_key, start, end) in enumerate(sub_batches):
                batch, get_batch_secs = prefetched.popleft().result()
                nxt = i + PREFETCH_DEPTH
                if nxt < len(sub_batches):
                    prefetched.append(pool.submit(_prepare_batch, dataset, *sub_batches[nxt]))
                yield i, bucket_key, start, end, batch, get_batch_secs

        if use_pipelined:
            _pipelined_gpu_loop(
                model,
                _batches(),
                config,
                device,
                save_dim,
                flat_q,
                flat_scales,
                sub_batches,
                progress,
            )
        else:
            _serial_loop(
                model,
                _batches(),
                config,
                device,
                save_dim,
                flat_q,
                flat_scales,
                progress,
            )
    pixels_processed = progress.pixels
    tokens_processed = progress.tokens
    flops_processed = progress.flops

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
    if progress.bucket_px:
        logger.info(
            "Bucket occupancy: %s",
            " ".join(f"({k[0]},{k[1]}):{px}px/{nb}sb" for k, (px, nb) in sorted(progress.bucket_px.items())),
        )

    return InferenceResult(embeddings=out, scales=scales)
