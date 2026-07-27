"""Model construction and checkpoint loading for Tessera v1.1 and v2 Large.

``build_inference_model`` dispatches on ``config.model_version``:

* **v1.1** — two ``modules.TransformerEncoder`` backbones (GRU-based
  ``TemporalAwarePooling``) + the MLP ``build_dim_reducer``. Loads the encoder
  checkpoint (``model_state`` / ``model_state_dict``), strips FSDP/compile
  prefixes and training-only keys (projector, segmented-matryoshka-projector) so
  it loads ``strict=False``, then fuses CustomGRU to ``nn.GRU`` for cuDNN
  performance.
* **v2-large** — two ``student_v2.StudentTransformerEncoder`` backbones (plain
  attention pooling, no recurrence) + ``build_v2_dim_reducer``. The published
  student payload is ``{"model": state_dict, "args": {...}}`` with no prefixes
  and no training-only heads, so it loads ``strict=True``; the stored ``args``
  are cross-checked against the config's architecture. There is no GRU, so the
  fusion step is skipped.

Both paths then freeze the model, zero the TransformerEncoderLayer dropouts for
the SDPA fast path, and move to the target device in eval mode.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import torch

from tessera_embeddings.config.inference import MODEL_ARCHS

from .modules import CustomGRU, TemporalAwarePooling, TransformerEncoder
from .ssl_model import MultimodalBTInferenceModel, build_dim_reducer
from .student_v2 import StudentTransformerEncoder, build_v2_dim_reducer

if TYPE_CHECKING:
    from tessera_embeddings.config.inference import InferenceConfig

logger = logging.getLogger(__name__)

# State-dict prefixes emitted by v1.1 training that are not part of the
# inference graph. ``projector`` is the BarlowTwins head; the segmented
# matryoshka projector is used for the variable-width training objective.
_TRAINING_ONLY_PREFIXES: tuple[str, ...] = ("projector.", "segmented_matryoshka_projector.")


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
            fused.weight_ih_l0.copy_(torch.cat([cell.W_ir.weight, -cell.W_iz.weight, cell.W_ih.weight]))
            fused.weight_hh_l0.copy_(torch.cat([cell.W_hr.weight, -cell.W_hz.weight, cell.W_hh.weight]))
            fused.bias_ih_l0.copy_(torch.cat([cell.b_r, -cell.b_z, cell.b_h]))
            fused.bias_hh_l0.copy_(torch.cat([zeros_h, zeros_h, zeros_h]))

        child.temporal_context = fused  # type: ignore[assignment]
        logger.info("Fused CustomGRU -> nn.GRU (input=%d, hidden=%d)", d, h)


def _build_inference_model(config: InferenceConfig, device: torch.device) -> MultimodalBTInferenceModel:
    """Construct the v1.1 inference model (pre-checkpoint-load) on *device*.

    The two backbones share architecture hyperparameters. Both consume
    ``band_num + 1`` input features (the ``+1`` is DOY appended by the sampler).
    S2 is 10 bands; the merged S1 stream (asc + desc concatenated) is 2 bands.
    """
    max_seq_len = max(config.num_obs_checkpoints)

    s2_backbone = TransformerEncoder(
        band_num=10,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=max_seq_len,
    )

    s1_backbone = TransformerEncoder(
        band_num=2,
        latent_dim=config.latent_dim,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=max_seq_len,
    )

    active_backbones = 2 if config.fusion_method == "concat" else 1
    dim_reducer = build_dim_reducer(
        latent_dim=config.latent_dim,
        active_backbones=active_backbones,
        repr_dim=config.representation_dim,
    )

    return MultimodalBTInferenceModel(
        s2_backbone=s2_backbone,
        s1_backbone=s1_backbone,
        dim_reducer=dim_reducer,
        fusion_method=config.fusion_method,
    ).to(device)


def _build_v2_inference_model(config: InferenceConfig, device: torch.device) -> MultimodalBTInferenceModel:
    """Construct the v2 student inference model (pre-checkpoint-load) on *device*.

    Same two-backbone/concat-fusion topology as v1.1 — the differences are the
    v2 pooling head and the reducer's trailing non-affine LayerNorm (see
    ``student_v2``). Both backbones consume ``band_num + 1`` input features (the
    ``+1`` is DOY appended by the sampler): S2 is 10 bands, the merged S1 stream
    (asc + desc concatenated) is 2.
    """
    max_seq_len = max(config.num_obs_checkpoints)
    enable_qk_norm = MODEL_ARCHS[config.model_version].enable_qk_norm

    def make_encoder(band_num: int) -> StudentTransformerEncoder:
        return StudentTransformerEncoder(
            band_num=band_num,
            latent_dim=config.latent_dim,
            nhead=config.nhead,
            num_encoder_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            max_seq_len=max_seq_len,
            enable_qk_norm=enable_qk_norm,
        )

    if config.fusion_method != "concat":
        msg = f"v2 students are trained with concat fusion; got fusion_method={config.fusion_method!r}"
        raise ValueError(msg)

    return MultimodalBTInferenceModel(
        s2_backbone=make_encoder(10),
        s1_backbone=make_encoder(2),
        dim_reducer=build_v2_dim_reducer(
            latent_dim=config.latent_dim,
            repr_dim=config.representation_dim,
        ),
        fusion_method=config.fusion_method,
    ).to(device)


def load_v2_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a v2 student checkpoint payload.

    The published students ship ``{"model": state_dict, "args": {...}}``: no FSDP
    or ``torch.compile`` prefixes, and no training-only heads, so the state dict
    is used verbatim (``strict=True``). ``args`` records the architecture the
    checkpoint was trained with (``latent_dim``, ``repr_dim``, ``num_layers``,
    ``nhead``, ``dim_feedforward``, ``max_seq_len``, ``matryoshka_dims``,
    ``enable_qk_norm``).

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to map tensors to.

    Returns:
        ``(state_dict, args)``. ``args`` is a plain dict (an ``argparse.Namespace``
        payload is converted).
    """
    logger.info("Loading v2 checkpoint: %s (map_location=%s)", checkpoint_path, device)
    t0 = time.monotonic()
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info("torch.load completed in %.1fs", time.monotonic() - t0)

    if not isinstance(payload, dict) or "model" not in payload:
        available = list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__
        msg = f"v2 checkpoint must be a dict with a 'model' state dict. Got: {available}"
        raise KeyError(msg)

    raw_args = payload.get("args") or {}
    args = dict(raw_args) if isinstance(raw_args, dict) else dict(vars(raw_args))
    logger.info("v2 checkpoint: %d params, stored args=%s", len(payload["model"]), args)
    return payload["model"], args


def _verify_v2_args(config: InferenceConfig, args: dict[str, Any]) -> None:
    """Reject a v2 checkpoint whose stored architecture disagrees with the config.

    The config drives model construction (and the FLOPs accounting in
    ``profiling``), so a mismatch would otherwise surface as an opaque
    ``load_state_dict`` shape error — or, for ``max_seq_len``, not at all.
    Keys absent from ``args`` are not checked.
    """
    expected = {
        "latent_dim": config.latent_dim,
        "repr_dim": config.representation_dim,
        "nhead": config.nhead,
        "num_layers": config.num_encoder_layers,
        "dim_feedforward": config.dim_feedforward,
        "enable_qk_norm": MODEL_ARCHS[config.model_version].enable_qk_norm,
    }
    mismatched = {key: (args[key], value) for key, value in expected.items() if key in args and args[key] != value}
    if mismatched:
        detail = ", ".join(f"{k}: checkpoint={ckpt!r} config={cfg!r}" for k, (ckpt, cfg) in mismatched.items())
        msg = f"v2 checkpoint architecture does not match config ({detail})"
        raise ValueError(msg)

    max_seq_len = args.get("max_seq_len")
    if max_seq_len is not None and max(config.num_obs_checkpoints) > int(max_seq_len):
        msg = (
            f"num_obs_checkpoints reaches {max(config.num_obs_checkpoints)} but the checkpoint was "
            f"trained with max_seq_len={max_seq_len}"
        )
        raise ValueError(msg)


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Load a v1.1 checkpoint, strip FSDP prefixes and training-only heads.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to map tensors to.

    Returns:
        Cleaned state dict suitable for ``load_state_dict(..., strict=False)``.
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

    raw = checkpoint[state_key]

    cleaned: dict[str, torch.Tensor] = {}
    dropped = 0
    for key, value in raw.items():
        k = key.removeprefix("_orig_mod.") if key.startswith("_orig_mod.") else key
        if any(k.startswith(prefix) for prefix in _TRAINING_ONLY_PREFIXES):
            dropped += 1
            continue
        cleaned[k] = value

    logger.info("Loaded checkpoint: kept %d params, dropped %d training-only params", len(cleaned), dropped)
    return cleaned


def build_inference_model(
    config: InferenceConfig,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> MultimodalBTInferenceModel:
    """Build the inference model for ``config.model_version`` from its checkpoint.

    Constructs the model on CPU, loads the checkpoint (v1.1: ``strict=False``,
    since projector/matryoshka keys are filtered in ``load_checkpoint``; v2:
    ``strict=True``), fuses CustomGRU to nn.GRU where one exists (v1.1 only),
    freezes, zeros TransformerEncoderLayer dropouts for the fused-attention fast
    path, and moves to *device* in eval mode.

    Args:
        config: Inference configuration.
        device: Target device (cpu or cuda).
        checkpoint_path: Path to checkpoint file. If None, uses ``config.checkpoint_path``.

    Returns:
        Frozen ``MultimodalBTInferenceModel`` in eval mode.
    """
    ckpt_path = checkpoint_path or config.checkpoint_path
    if not ckpt_path:
        msg = "No checkpoint path provided (either via argument or config)"
        raise ValueError(msg)

    cpu = torch.device("cpu")
    t0 = time.monotonic()
    is_v2 = config.model_version != "v1.1"
    model = _build_v2_inference_model(config, cpu) if is_v2 else _build_inference_model(config, cpu)
    logger.info("%s inference model built on CPU in %.1fs", config.model_version, time.monotonic() - t0)

    if is_v2:
        state_dict, args = load_v2_checkpoint(ckpt_path, cpu)
        _verify_v2_args(config, args)
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = load_checkpoint(ckpt_path, cpu)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Checkpoint missing %d expected params (first few): %s", len(missing), list(missing)[:5])
        if unexpected:
            logger.warning("Checkpoint has %d unexpected params (first few): %s", len(unexpected), list(unexpected)[:5])

        # Must run on CPU before .to(device)/.bfloat16() so the fused nn.GRU inherits
        # correct weights. v2 has no recurrence in its pooling head — nothing to fuse.
        _fuse_custom_gru(model)

    for param in model.parameters():
        param.requires_grad = False

    # Zero dropout in TransformerEncoderLayers so SDPA fused-attention fast paths engage.
    for mod in model.modules():
        if isinstance(mod, torch.nn.TransformerEncoderLayer):
            mod.dropout.p = 0.0
            mod.dropout1.p = 0.0
            mod.dropout2.p = 0.0
            mod.self_attn.dropout = 0.0

    model = model.to(device)
    model.eval()
    del state_dict

    logger.info("Built %s inference model on %s", config.model_version, device)
    return model
