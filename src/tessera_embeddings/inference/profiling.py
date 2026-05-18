"""Performance profiling utilities for the inference pipeline.

Provides GPU diagnostics, autocast dtype probing, and per-layer timing
via a profiling flag on model modules. Designed to stay in production —
profiling runs once per inference call on a single designated batch,
adding negligible overhead to the overall run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tessera_embeddings.inference.models.ssl_model import MultimodalBTInferenceModel

logger = logging.getLogger(__name__)


def log_cuda_diagnostics(device: torch.device) -> None:
    """Log GPU hardware, driver, and configuration details.

    Reports compute capability, SM count, VRAM, framework versions, and
    whether tensor cores / TF32 are enabled. Run once per inference call.
    """
    if device.type != "cuda":
        return

    props = torch.cuda.get_device_properties(0)
    has_tensor_cores = props.major >= 7

    logger.debug(
        "GPU: %s | compute=%d.%d | SMs=%d | VRAM=%.1f GB | torch=%s | cuda=%s | cuDNN=%s",
        props.name,
        props.major,
        props.minor,
        props.multi_processor_count,
        props.total_memory / 1024**3,
        torch.__version__,
        torch.version.cuda,
        torch.backends.cudnn.version(),
    )
    logger.debug(
        "TENSOR CORES: available=%s | cudnn_benchmark=%s | allow_tf32_cudnn=%s | allow_tf32_matmul=%s",
        has_tensor_cores,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )


def log_autocast_dtype_probe(device: torch.device) -> None:
    """Probe which dtypes autocast actually uses for key operations.

    Creates small test tensors and runs them through matmul, Linear, GRU,
    LayerNorm, and TransformerEncoderLayer under autocast. Logs the output
    dtype of each — this reveals whether tensor cores can engage (FP16) or
    whether autocast is silently keeping ops in FP32.

    Critical for understanding GRU behavior: PyTorch autocast may force RNNs
    to FP32 for numerical stability, which would explain FP32-like throughput
    despite FP16 model weights.
    """
    if device.type != "cuda":
        return

    # 2D tensor for matmul/linear/layernorm probes
    a2d = torch.randn(32, 512, device=device, dtype=torch.float16)
    w2d = torch.randn(512, 512, device=device, dtype=torch.float16)
    # 3D tensor for GRU and transformer probes: (batch, seq_len, d_model)
    a3d = torch.randn(4, 20, 512, device=device, dtype=torch.float16)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        mm_out = torch.mm(a2d, w2d)

        lin = torch.nn.Linear(512, 512, device=device).half()
        lin_out = lin(a2d)

        gru = torch.nn.GRU(512, 512, batch_first=True, device=device).half()
        gru_out, _ = gru(a3d)

        ln = torch.nn.LayerNorm(512, device=device).half()
        ln_out = ln(a2d)

        tel = torch.nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=4096,
            batch_first=True,
            device=device,
            dropout=0.0,
        ).half()
        tel_out = tel(a3d)

    logger.debug(
        "AUTOCAST DTYPE PROBE: matmul=%s | linear=%s | GRU=%s | layernorm=%s | transformer_layer=%s",
        mm_out.dtype,
        lin_out.dtype,
        gru_out.dtype,
        ln_out.dtype,
        tel_out.dtype,
    )


def enable_model_profiling(model: MultimodalBTInferenceModel) -> None:
    """Set ``_profile`` flag on model and all sub-modules.

    When ``_profile=True``, the forward methods in ``modules.py`` and
    ``ssl_model.py`` log per-layer timing and intermediate dtypes.
    """
    object.__setattr__(model, "_profile", True)
    for name, mod in model.named_modules():
        if hasattr(mod, "forward") and name:
            object.__setattr__(mod, "_profile", True)


def disable_model_profiling(model: MultimodalBTInferenceModel) -> None:
    """Clear ``_profile`` flag from model and all sub-modules."""
    object.__setattr__(model, "_profile", False)
    for name, mod in model.named_modules():
        if hasattr(mod, "forward") and name:
            object.__setattr__(mod, "_profile", False)


def log_profiled_batch_summary(
    batch_idx: int,
    s2_input: torch.Tensor,
    s1_input: torch.Tensor,
    z: torch.Tensor,
    model: MultimodalBTInferenceModel,
    forward_ms: float,
    device: torch.device,
) -> None:
    """Log detailed diagnostics for the profiled batch.

    Reports forward-pass time, input/output dtypes and shapes, model parameter
    dtype, and VRAM usage (allocated vs reserved).
    """
    vram_alloc = torch.cuda.memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0
    vram_reserved = torch.cuda.memory_reserved() / 1024 / 1024 if device.type == "cuda" else 0

    logger.debug(
        "PROFILED BATCH %d: forward=%.1fms | s2=%s %s | s1=%s %s | z=%s | params=%s | "
        "VRAM alloc=%.0f MiB reserved=%.0f MiB",
        batch_idx,
        forward_ms,
        s2_input.dtype,
        list(s2_input.shape),
        s1_input.dtype,
        list(s1_input.shape),
        z.dtype,
        next(model.parameters()).dtype,
        vram_alloc,
        vram_reserved,
    )


def log_effective_tflops(
    forward_ms: float,
    batch_size: int,
    seq_len: int = 20,
    d_model: int = 512,
    dim_feedforward: int = 4096,
    num_layers: int = 8,
    num_backbones: int = 2,
) -> None:
    """Estimate effective TFLOPS from forward-pass timing and known model structure.

    Compares against GPU theoretical peaks to determine hardware utilization.
    Reference hardware: A10G (current production GPU).
      - A10G FP16: 125 TFLOPS (peak), ~40-60 TFLOPS (realistic with overhead)
      - A10G memory bandwidth: 600 GB/s
    Historical: T4 FP16=65 peak, L4 FP16=121 peak.

    Only counts transformer layer FLOPs (attention + FFN), which dominate the model.
    GRU, embedding MLP, and positional encoding are excluded (< 5% of total FLOPs).
    """
    if forward_ms <= 0:
        return

    effective_batch = batch_size

    # Per-layer FLOPs (standard transformer accounting -- each matmul (M,K)x(K,N) = 2MKN FLOPs):
    # Self-attention: 8*B*S*D^2 (4 projections, 2 FLOPs each) + 4*B*S^2*D (scores + context)
    attn_flops = 8 * effective_batch * seq_len * d_model**2 + 4 * effective_batch * seq_len**2 * d_model
    # FFN: 4*B*S*D*FF (2 linear layers, 2 FLOPs each)
    ffn_flops = 4 * effective_batch * seq_len * d_model * dim_feedforward

    per_layer_flops = attn_flops + ffn_flops
    total_flops = per_layer_flops * num_layers * num_backbones

    effective_tflops = total_flops / (forward_ms / 1000) / 1e12

    logger.debug(
        "EFFECTIVE TFLOPS: %.2f TFLOPS (transformer layers only) | "
        "forward=%.1fms | effective_batch=%d | "
        "A10G FP16=125 TFLOPS (peak) ~40-60 (real) | mem_bw=600 GB/s | "
        "verdict=%s",
        effective_tflops,
        forward_ms,
        effective_batch,
        "FP16 tensor cores ACTIVE"
        if effective_tflops > 20
        else "FP32 range or poor utilization"
        if effective_tflops < 12
        else "PARTIAL — tensor cores engaged but below expected range",
    )
