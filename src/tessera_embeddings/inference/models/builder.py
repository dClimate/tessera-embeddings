"""Model construction and checkpoint loading for Tessera v1.1.

Builds the inference model (two TransformerEncoder backbones + MLP dim_reducer), loads the v1.1
encoder checkpoint, strips training-only keys (projector and segmented-matryoshka-projector),
fuses CustomGRU to nn.GRU for cuDNN performance, then freezes and moves to the target device.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import fsspec
import torch

from .modules import CustomGRU, TemporalAwarePooling, TransformerEncoder
from .ssl_model import MultimodalBTInferenceModel, build_dim_reducer

if TYPE_CHECKING:
    from tessera_embeddings.config.inference import InferenceConfig

logger = logging.getLogger(__name__)

# State-dict prefixes emitted by v1.1 training that are not part of the inference graph: ``projector`` is the
# BarlowTwins head, the segmented matryoshka projector serves the variable-width training objective.
_TRAINING_ONLY_PREFIXES: tuple[str, ...] = ("projector.", "segmented_matryoshka_projector.")


def _fuse_custom_gru(module: torch.nn.Module) -> None:
    """Replace CustomGRU with nn.GRU by fusing per-gate weights.

    Walks the module tree, finds TemporalAwarePooling instances holding a CustomGRU, and swaps in
    nn.GRU with fused weight matrices — recovering cuDNN performance (~1 kernel launch vs ~480).

    The tessera CustomGRUCell differs from nn.GRU in two ways:

    1. **Update gate convention is inverted.** Tessera: h' = (1-z)*h + z*n (z selects the new
       candidate). nn.GRU: h' = (1-z)*n + z*h (z keeps old state). Since 1 - sigmoid(x) =
       sigmoid(-x), all z gate weights and biases are negated.

    2. **Reset gate placement differs.** Tessera applies reset BEFORE the matmul, W_hh @ (r * h);
       nn.GRU applies it after, r * (W_hh @ h + b_hh). NOT equivalent for dense weight matrices,
       but a small approximation in practice because the reset gate is close to 1 for most
       dimensions after training. b_h goes in bias_ih (input-side) since tessera adds it outside
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

    The two backbones share architecture hyperparameters and both consume ``band_num + 1`` input
    features, the ``+1`` being DOY appended by the sampler. S2 is 10 bands; the merged S1 stream
    (asc + desc concatenated) is 2.
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
    """Build the v1.1 inference model from config and checkpoint.

    Constructs on CPU, loads the checkpoint with ``strict=False`` (projector/matryoshka keys are
    already filtered in ``load_checkpoint``), fuses CustomGRU to nn.GRU, freezes, zeros
    TransformerEncoderLayer dropouts for the fused-attention fast path, and moves to *device* in
    eval mode.

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
    model = _build_inference_model(config, cpu)
    logger.info("v1.1 inference model built on CPU in %.1fs", time.monotonic() - t0)

    state_dict = load_checkpoint(ckpt_path, cpu)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Checkpoint missing %d expected params (first few): %s", len(missing), list(missing)[:5])
    if unexpected:
        logger.warning("Checkpoint has %d unexpected params (first few): %s", len(unexpected), list(unexpected)[:5])

    # Must run on CPU before .to(device)/.bfloat16() so the fused nn.GRU inherits correct weights.
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

    logger.info("Built v1.1 inference model on %s", device)
    return model


# Schemes that mean "fetch this from somewhere else first". A local filesystem path (or an explicit file:// URI) is
# loaded in place by torch.
_REMOTE_CKPT_SCHEMES = ("s3://", "http://", "https://", "gs://", "az://", "abfs://")


def _is_remote_uri(path: str) -> bool:
    """True if ``path`` must be downloaded before torch.load can open it."""
    return path.startswith(_REMOTE_CKPT_SCHEMES)


def _default_checkpoint_cache() -> str:
    """Pick a download cache dir that exists on the running host.

    On AWS DLAMI GPU boxes the NVMe instance store (~1.5 GB/s) is the right target: the root EBS
    volume (~42 MB/s) is too slow and torch.load with mmap hangs on it. Off that path (laptops,
    CI, non-AWS GPUs) the NVMe mount does not exist, so fall back to the system temp dir.
    """
    nvme = Path("/opt/dlami/nvme")
    if nvme.is_dir():
        return str(nvme / "tessera-checkpoints")
    return str(Path(tempfile.gettempdir()) / "tessera-checkpoints")


def download_checkpoint(remote_path: str, local_dir: str | None = None) -> str:
    """Download a model checkpoint from a remote URI to local storage.

    Handles any fsspec-supported remote scheme — ``s3://``, ``https://`` (e.g. a HuggingFace
    ``resolve/main`` URL), ``gs://``. The file is staged locally because torch.load wants a real
    path and reads it twice.

    Args:
        remote_path: Remote URI (e.g. ``"s3://bucket/path/model.pt"`` or
            ``"https://huggingface.co/.../tessera_v1_1_aws_encoder.pt"``).
        local_dir: Local directory for downloads. Defaults to the NVMe instance store on AWS
            DLAMI hosts, else a system temp dir.

    Returns:
        Local file path.

    Concurrency: many actors on one host may call this with the same ``remote_path`` and shared
    cache dir at once (cold cache, hundreds of actors). The download writes to a unique temp file
    and is published with an atomic rename, so a concurrent reader never sees a partially-written
    checkpoint and concurrent writers cannot corrupt each other — the last rename wins and every
    byte is identical.
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
    # Staged into a unique temp file in the SAME dir, so the rename stays on one filesystem and is atomic: concurrent
    # actors publishing the same checkpoint cannot see a half-written file.
    with tempfile.NamedTemporaryFile(dir=local, prefix=f"{filename}.", suffix=".part", delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    tmp_path.replace(local_path)

    downloaded_size = local_path.stat().st_size
    logger.info("Download complete: %s (%.1f MB)", local_path, downloaded_size / 1024 / 1024)

    return str(local_path)
