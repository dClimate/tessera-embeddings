"""Configuration for the Tessera embedding inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tessera_embeddings.config.time_windows import TimeWindow

# Model checkpoint filenames. Full URIs are assembled by callers from a base
# path supplied via ``InferenceConfig.checkpoint_path`` (or directly from
# ``BucketPaths`` once Phase 3 lands).
_CHECKPOINT_FULL_NAME = "best_model_fsdp_20250427_084307.pt"
_CHECKPOINT_QAT_NAME = "best_model_fsdp_20250608_220648_QAT.pt"


def checkpoint_filename(quantized: bool = True) -> str:
    """Return the canonical filename for a model checkpoint.

    Args:
        quantized: Use the QAT checkpoint when True, full when False.
    """
    return _CHECKPOINT_QAT_NAME if quantized else _CHECKPOINT_FULL_NAME


# Band statistics from tessera training data (used for standardization)
S2_BAND_MEAN = [
    1711.0938,
    1308.8511,
    1546.4543,
    3010.1293,
    3106.5083,
    2068.3044,
    2685.0845,
    2931.5889,
    2514.6928,
    1899.4922,
]
S2_BAND_STD = [
    1926.1026,
    1862.9751,
    1803.1792,
    1741.7837,
    1677.4543,
    1888.7862,
    1736.3090,
    1715.8104,
    1514.5199,
    1398.4779,
]
S1_BAND_MEAN = [5484.0407, 3003.7812]
S1_BAND_STD = [1871.2334, 1726.0670]

# Band order for S2 — must match training-time order (tessera s2_stack output order)
S2_BAND_ORDER = ["red", "blue", "green", "nir", "nir08", "rededge1", "rededge2", "rededge3", "swir16", "swir22"]

# SCL classes considered valid for masking (complement of S2_SCL_INVALID_CLASSES)
SCL_VALID_CLASSES = frozenset({4, 5, 6, 7, 10, 11})

# Embedding output dimension
EMBEDDING_DIM = 128

# Spatial chunk size for storage and inference. 2000x2000 keeps peak RAM
# ~9.8 GB on g5.2xlarge (31 GB), providing comfortable headroom even
# with high-timestep windows (126 S2 + 84 SAR dates observed for 33UWP).
DEFAULT_CHUNK_SIZE = 2000

# Tessera: 2000x2000 spatial chunks, time=1 so each date is its own chunk.
# time=1 supports random date sampling during embedding generation.
# Callers pass this dict to write_dataset(chunks=TESSERA_CHUNKS).
TESSERA_CHUNKS = {"time": 1, "northing": DEFAULT_CHUNK_SIZE, "easting": DEFAULT_CHUNK_SIZE}


@dataclass
class InferenceConfig:
    """Configuration for the full inference pipeline.

    Attributes:
        Model architecture (must match checkpoint):
            latent_dim: Latent dimension of the transformer encoders.
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            dim_feedforward: Feedforward dimension in transformer layers.
            dropout: Dropout rate (0 at inference, kept for architecture compat).
            max_seq_len: Maximum sequence length (kept for architecture compat).
            fusion_method: How S2 and S1 representations are fused.
            projector_hidden_dim: Hidden dim of projection head (for loading checkpoint).
            projector_out_dim: Output dim of projection head (for loading checkpoint).

        Inference parameters:
            batch_size: Batch size for the GPU.
            repeat_times: Number of random samplings to average over.
            sample_size_s2: Number of S2 timesteps sampled per pixel per repeat.
            sample_size_s1: Number of S1 timesteps sampled per pixel per repeat.
            min_valid_timesteps: Minimum valid timesteps to include a pixel.
            num_workers: GPU workers.
            s1_orbit: Which S1 orbit direction(s) to use ("ascending", "descending").

        I/O:
            checkpoint_path: Path to model checkpoint (.pt file).
            inputs_bucket: Base S3 path for input Icechunk stores.
            output_bucket: Base S3 path for output embeddings.
            chunk_size: Spatial chunk size in pixels.

        Ray cluster:
            ray_address: Ray cluster address (None for local mode).
            gpu_instance_type: EC2 instance type for GPU workers.
            use_spot: Whether to use spot instances for GPU workers.
            max_gpu_workers: Maximum number of GPU workers.
    """

    # Time window (required — no default)
    time_window: TimeWindow

    # Model architecture
    latent_dim: int = 128
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 4096
    dropout: float = 0.1
    max_seq_len: int = 40
    fusion_method: str = "concat"
    projector_hidden_dim: int = 16384
    projector_out_dim: int = 16384

    # Inference
    # Batch size empirically validated to fit within GPU/VRAM headroom w/ these repeats.
    # Do not exceed without re-profiling.
    batch_size: int = 3584
    # CHANGED (alpha_1.0): repeat_times 10 → 3 (tessera alpha_1.0 uses 1; we use 3 for
    # noise smoothing with our with-replacement sampling approach).
    repeat_times: int = 3
    # CHANGED (alpha_1.0): sample sizes 20 → 40, matching tessera alpha_1.0.
    sample_size_s2: int = 40
    sample_size_s1: int = 40
    # CHANGED (alpha_1.0): min_valid_timesteps 10 → 0, matching tessera alpha_1.0.
    # All pixels with non-zero S2 bands are now included regardless of valid SCL count.
    min_valid_timesteps: int = 0
    num_workers: int = 4
    s1_orbit: Literal["ascending", "descending"] = "ascending"
    compute_std: bool = False

    # Ray actor resource reservation. ``num_gpus=1`` is the production
    # default (one A10G per actor); set to ``0`` for CPU-only runs (local
    # smoke tests, the plain runner on a non-GPU host). Passed at
    # ``InferenceActor.options(...).remote()`` time so a single actor
    # class supports both.
    num_gpus: float = 1.0

    # I/O. Callers must supply absolute URIs; storage paths come from
    # ``BucketPaths`` (config/paths.py, Phase 3) or equivalent caller-side
    # configuration. No environment-derived defaults.
    checkpoint_path: str = ""
    inputs_bucket: str = ""
    output_bucket: str = ""
    chunk_size: int = DEFAULT_CHUNK_SIZE

    # Ray cluster
    ray_address: str | None = None
    # CHANGED: g4dn.4xlarge (T4) → g6.4xlarge (L4) → g5.4xlarge (A10G) → g5.2xlarge (A10G).
    # Same A10G GPU (600 GB/s bandwidth, 125 TFLOPS FP16, 24 GB VRAM) but 32 GB RAM / 8 vCPUs
    # instead of 64 GB / 16. RAM and CPU are underutilized (<40% RAM, <6% CPU batch time),
    # so the smaller instance saves ~25% cost ($1.212/hr vs $1.624/hr) with no throughput impact.
    # Requires the incremental source-array deletion in dataset.py _pre_extract() to avoid
    # OOM during extraction (peak was ~28 GB without it, ~14 GB with it).
    gpu_instance_type: str = "g5.2xlarge"
    use_spot: bool = False
    max_gpu_workers: int = 500

    def __post_init__(self) -> None:
        """Validate the orbit literal."""
        if self.s1_orbit not in {"ascending", "descending"}:
            raise ValueError(f"Invalid s1_orbit: {self.s1_orbit!r}. Must be 'ascending' or 'descending'.")

    # Band statistics (frozen from training data)
    s2_band_mean: list[float] = field(default_factory=lambda: list(S2_BAND_MEAN))
    s2_band_std: list[float] = field(default_factory=lambda: list(S2_BAND_STD))
    s1_band_mean: list[float] = field(default_factory=lambda: list(S1_BAND_MEAN))
    s1_band_std: list[float] = field(default_factory=lambda: list(S1_BAND_STD))
