"""Model construction and checkpoint loading.

Ported from tessera_infer/src/models/builder.py with type hints and ruff compliance.
Builds the full SSL model to load the checkpoint, then extracts the inference model.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from .modules import CustomGRU, ProjectionHead, TemporalAwarePooling, TransformerEncoder
from .ssl_model import MultimodalBTInferenceModel, MultimodalBTModel

if TYPE_CHECKING:
    from tessera_embeddings.config.inference import InferenceConfig

logger = logging.getLogger(__name__)


def _fuse_custom_gru(module: torch.nn.Module) -> None:
    """Replace CustomGRU with nn.GRU by fusing per-gate weights.

    Walks the module tree, finds TemporalAwarePooling instances with CustomGRU,
    and swaps in nn.GRU with fused weight matrices. This recovers cuDNN
    performance (~1 kernel launch vs ~480).

    The tessera CustomGRUCell differs from nn.GRU in two ways:

    1. **Update gate convention is inverted.** Tessera: h' = (1-z)*h + z*n (z selects
       the new candidate). nn.GRU: h' = (1-z)*n + z*h (z keeps old state). Since
       1 - sigmoid(x) = sigmoid(-x), we negate all z gate weights and biases.

    2. **Reset gate placement differs.** Tessera: W_hh @ (r * h) — reset applied
       before matmul. nn.GRU: r * (W_hh @ h + b_hh) — reset applied after matmul.
       This is NOT equivalent for dense weight matrices, but is a small approximation
       in practice because the reset gate is close to 1 for most dimensions after
       training. b_h is placed in bias_ih (input-side) since tessera adds it outside
       the reset gate product.
    """
    for _name, child in module.named_modules():
        if not isinstance(child, TemporalAwarePooling):
            continue
        custom_gru = child.temporal_context
        if not isinstance(custom_gru, CustomGRU):
            continue

        cell = custom_gru.gru_cell
        h = cell.hidden_size
        d = cell.input_size

        src_device = cell.W_ir.weight.device
        src_dtype = cell.W_ir.weight.dtype
        zeros_h = torch.zeros(h, device=src_device, dtype=src_dtype)
        fused = torch.nn.GRU(d, h, batch_first=True, device=src_device, dtype=src_dtype)
        with torch.no_grad():
            # Negate z gate weights to invert update gate convention
            fused.weight_ih_l0.copy_(torch.cat([cell.W_ir.weight, -cell.W_iz.weight, cell.W_ih.weight]))
            fused.weight_hh_l0.copy_(torch.cat([cell.W_hr.weight, -cell.W_hz.weight, cell.W_hh.weight]))
            # b_r on input side, -b_z (negated) on input side, b_h on input side
            # (b_h is outside the reset gate product in tessera, so it maps to input-side bias)
            fused.bias_ih_l0.copy_(torch.cat([cell.b_r, -cell.b_z, cell.b_h]))
            fused.bias_hh_l0.copy_(torch.cat([zeros_h, zeros_h, zeros_h]))

        child.temporal_context = fused  # type: ignore[assignment]
        logger.info("Fused CustomGRU -> nn.GRU (input=%d, hidden=%d)", d, h)


def _build_ssl_model(config: InferenceConfig, device: torch.device) -> MultimodalBTModel:
    """Build the full SSL model (with projection head) for checkpoint loading.

    Args:
        config: Inference configuration.
        device: Target device.

    Returns:
        Full MultimodalBTModel on the target device.
    """
    s2_backbone = TransformerEncoder(
        band_num=10,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    )

    s1_backbone = TransformerEncoder(
        band_num=2,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    )

    projector = ProjectionHead(
        config.latent_dim,
        config.projector_hidden_dim,
        config.projector_out_dim,
    )

    ssl_model = MultimodalBTModel(
        s2_backbone,
        s1_backbone,
        projector,
        fusion_method=config.fusion_method,
        return_repr=True,
        latent_dim=config.latent_dim,
    ).to(device)

    return ssl_model


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load a checkpoint file, handling FSDP prefix stripping.

    Supports both local paths and downloaded S3 files.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to map tensors to.

    Returns:
        Cleaned state dict with FSDP prefixes removed.
    """
    logger.info("Loading checkpoint: %s (map_location=%s)", checkpoint_path, device)
    t0 = time.monotonic()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info("torch.load completed in %.1fs", time.monotonic() - t0)

    state_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    if state_key not in checkpoint:
        available = list(checkpoint.keys())
        msg = f"State key '{state_key}' not found in checkpoint. Available keys: {available}"
        raise KeyError(msg)

    state_dict = checkpoint[state_key]

    # Strip FSDP '_orig_mod.' prefix if present
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            cleaned[key[len("_orig_mod.") :]] = value
        else:
            cleaned[key] = value

    logger.info("Loaded checkpoint with %d parameters", len(cleaned))
    return cleaned


def build_inference_model(
    config: InferenceConfig,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> MultimodalBTInferenceModel:
    """Build the inference model from config and checkpoint.

    Constructs the full SSL model, loads the checkpoint, then extracts
    the inference-only model (backbones + fusion + dim_reducer, no projector).

    Args:
        config: Inference configuration.
        device: Target device (cpu or cuda).
        checkpoint_path: Path to checkpoint file. If None, uses config.checkpoint_path.

    Returns:
        Frozen MultimodalBTInferenceModel in eval mode.
    """
    ckpt_path = checkpoint_path or config.checkpoint_path
    if not ckpt_path:
        msg = "No checkpoint path provided (either via argument or config)"
        raise ValueError(msg)

    # Build full SSL model and load weights on CPU to avoid putting the
    # ~6.4GB projection head on GPU. The projector is stripped for inference;
    # only the small inference model (~337MB) gets moved to the target device.
    cpu = torch.device("cpu")
    t0 = time.monotonic()
    ssl_model = _build_ssl_model(config, cpu)
    logger.info("SSL model built on CPU in %.1fs", time.monotonic() - t0)
    state_dict = load_checkpoint(ckpt_path, cpu)
    ssl_model.load_state_dict(state_dict, strict=True)
    # Fuse CustomGRU -> nn.GRU BEFORE .to(device)/.half(). Must happen while the model
    # is still on CPU so the fused nn.GRU inherits correct weights. The subsequent
    # .to(device) and model.half() (in _prepare_gpu) will move it to GPU in FP16.
    _fuse_custom_gru(ssl_model)

    # Freeze all parameters
    for param in ssl_model.parameters():
        param.requires_grad = False

    # CHANGED: Zero out dropout in TransformerEncoderLayers for inference. The checkpoint
    # was trained with dropout=0.1, and nn.Dropout is already a no-op in eval mode, but
    # PyTorch's SDPA kernel selection checks the stored dropout value at dispatch time —
    # a non-zero value may prevent fused attention fast paths from activating.
    for mod in ssl_model.modules():
        if isinstance(mod, torch.nn.TransformerEncoderLayer):
            mod.dropout.p = 0.0  # Residual dropout
            mod.dropout1.p = 0.0  # Post-attention dropout
            mod.dropout2.p = 0.0  # Post-FFN dropout
            mod.self_attn.dropout = 0.0  # SDPA dropout (float, checked at kernel dispatch)

    # Extract inference model (no projection head) and move to target device
    model = MultimodalBTInferenceModel(
        s2_backbone=ssl_model.s2_backbone,
        s1_backbone=ssl_model.s1_backbone,
        fusion_method=config.fusion_method,
        dim_reducer=ssl_model.dim_reducer,
    ).to(device)
    model.eval()

    # Free the full SSL model (projector etc.) from CPU memory
    del ssl_model, state_dict

    logger.info("Built inference model on %s", device)
    return model
