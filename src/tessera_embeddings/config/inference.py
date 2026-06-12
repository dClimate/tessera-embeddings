"""Configuration for the Tessera v1.1 embedding inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, final

from tessera_embeddings.config.time_windows import TimeWindow

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------

# v1.1 ships two checkpoints keyed by norm_source. We default to "aws" because
# the AWS norm stats match the pixel distributions produced by Earth Search
# ingestion (which does not re-apply the S2 baseline correction).
_CHECKPOINT_NAMES: dict[str, str] = {
    "mpc": "tessera_v1_1_mpc_encoder.pt",
    "aws": "tessera_v1_1_aws_encoder.pt",
}


def checkpoint_filename(norm_source: str = "aws") -> str:
    """Return the canonical filename for the v1.1 encoder checkpoint.

    Args:
        norm_source: Which checkpoint to use — ``"aws"`` (default) or ``"mpc"``.
    """
    if norm_source not in _CHECKPOINT_NAMES:
        valid = ", ".join(repr(k) for k in _CHECKPOINT_NAMES)
        raise ValueError(f"Unknown norm_source: {norm_source!r}. Must be one of {valid}.")
    return _CHECKPOINT_NAMES[norm_source]


def _normalize_obs_checkpoints(checkpoints: tuple[int, ...]) -> tuple[int, ...]:
    """Coerce and validate a num_obs_checkpoints value.

    Deduplicates, sorts, and filters out non-positive values. Safe to call on
    values arriving as lists from YAML deserialization.
    """
    result = tuple(sorted({int(v) for v in checkpoints if int(v) > 0}))
    if not result:
        raise ValueError("num_obs_checkpoints must contain at least one positive integer")
    return result


# ---------------------------------------------------------------------------
# Per-modality normalisation stats (v1.1)
# ---------------------------------------------------------------------------
# Sourced from ucam-eo/tessera tag v1.1, src/datasets/v1_1_norm_stats.py.
# S1 ascending and descending have DIFFERENT stats — each orbit is normalised
# with its own mean/std BEFORE concatenation. This matches v1.1 training preprocessing.

_NORM_STATS: dict[str, dict[str, list[float]]] = {
    "mpc": {
        "s2_mean": [
            2683.4553,
            2223.3630,
            2432.0950,
            3633.1970,
            3602.1755,
            3006.4324,
            3400.2710,
            3515.6392,
            2456.9163,
            1983.8783,
        ],
        "s2_std": [
            2739.5217,
            2846.2993,
            2690.8250,
            2290.0439,
            2088.8970,
            2673.1106,
            2381.4521,
            2229.5225,
            1601.0942,
            1495.3545,
        ],
        "s1_asc_mean": [5588.3291, 3025.6270],
        "s1_asc_std": [1713.4646, 1693.0471],
        "s1_desc_mean": [5552.9683, 2955.0520],
        "s1_desc_std": [1685.5857, 1677.6414],
    },
    "aws": {
        "s2_mean": [
            2793.6589,
            2356.7776,
            2551.0496,
            3741.9229,
            3713.7844,
            3120.1997,
            3516.3342,
            3637.0342,
            2501.0283,
            2038.1504,
        ],
        "s2_std": [
            2810.0093,
            2933.8835,
            2755.6360,
            2344.5027,
            2145.7986,
            2743.9019,
            2438.8601,
            2286.5977,
            1680.7367,
            1585.5529,
        ],
        "s1_asc_mean": [5664.5439, 2802.9736],
        "s1_asc_std": [1678.7821, 1786.0414],
        "s1_desc_mean": [5710.6992, 2830.1045],
        "s1_desc_std": [1616.1969, 1761.8499],
    },
}

# Module-level aliases for the default (AWS) stats — kept for direct importers.
S2_BAND_MEAN: list[float] = _NORM_STATS["aws"]["s2_mean"]
S2_BAND_STD: list[float] = _NORM_STATS["aws"]["s2_std"]
S1_ASC_BAND_MEAN: list[float] = _NORM_STATS["aws"]["s1_asc_mean"]
S1_ASC_BAND_STD: list[float] = _NORM_STATS["aws"]["s1_asc_std"]
S1_DESC_BAND_MEAN: list[float] = _NORM_STATS["aws"]["s1_desc_mean"]
S1_DESC_BAND_STD: list[float] = _NORM_STATS["aws"]["s1_desc_std"]

# Band order for S2 — must match training-time order (tessera s2_stack output order)
S2_BAND_ORDER = ["red", "blue", "green", "nir", "nir08", "rededge1", "rededge2", "rededge3", "swir16", "swir22"]

# SCL classes considered valid for masking (complement of S2_SCL_INVALID_CLASSES)
SCL_VALID_CLASSES = frozenset({4, 5, 6, 7, 10, 11})

# Embedding output dimension — v1.1 produces 192-D reps; we save the first 128.
EMBEDDING_DIM = 128

# Internal model representation dimension (before the 128-D slice).
REPRESENTATION_DIM = 192

# v1.1 observation-count buckets. Every pixel with k valid observations is resampled
# to the next bucket size; pixels sharing a bucket form rectangular batches for the
# transformer. Multiples of 8 from 8 to 256 match tessera v1.1 defaults.
DEFAULT_NUM_OBS_CHECKPOINTS: tuple[int, ...] = tuple(range(8, 257, 8))

# Spatial read-tile size for inference. The read/ChunkSpec grid stays 2000x2000;
# the *resident input working set* is bounded separately via northing strips (see
# InferenceConfig.strip_budget_bytes), so a 2000x2000 chunk's peak host RAM is
# capped by the strip budget rather than fixed by T x H x W. Independent of the
# storage chunk size written at ingest (config.ingest.INGEST_CHUNK_SIZE).
INFERENCE_CHUNK_SIZE = 2000

# Default byte budget for one resident input strip (see
# InferenceConfig.strip_budget_bytes). Sized so a normal-density chunk (T~=40
# valid S2 timesteps) resolves to a SINGLE strip — strip_h >= 2000, identical to
# the unstriped path — while a dense T~=120 chunk splits into multiple narrower
# strips. The actor sizes the strip from the dominant resident input cost: the
# S2 band array at T * W * 10 bands * 2 bytes/elem per northing row. At T=40,
# W=2000 that is ~1.6 MB/row, so all 2000 rows fit under 4 GiB (one strip); at
# T=120 it is ~4.8 MB/row, so 4 GiB / 4.8 MB ~= 890 rows -> ceil(2000/890) = 3
# strips of ~670 rows. Tune down to force more, narrower strips (lower peak RAM,
# higher read amplification) or up to disable striping entirely.
DEFAULT_STRIP_BUDGET_BYTES = 4 * 1024**3


@final
@dataclass
class InferenceConfig:
    """Configuration for the v1.1 inference pipeline.

    Attributes:
        Model architecture (must match checkpoint):
            latent_dim: Encoder base dim. Transformer d_model = latent_dim * 4.
            representation_dim: Output dim of the MLP dim_reducer (192 in v1.1).
            nhead: Transformer attention heads.
            num_encoder_layers: Transformer encoder layer count.
            dim_feedforward: Transformer FFN hidden dim.
            dropout: Dropout rate (zeroed at inference, kept for arch compat).
            fusion_method: "concat" or "sum" for combining S2/S1 representations.
            num_obs_checkpoints: Sorted bucket sizes for the all-obs sampler.

        Inference:
            batch_size: Per-GPU sub-batch size within a bucket.
            norm_source: Which v1.1 checkpoint/stats — "aws" (default) or "mpc".
            num_workers: GPU workers.
            s1_orbit: Which S1 orbit(s) — "ascending", "descending", or "both".
            compute_std: No-op under v1.1 (deterministic sampling); always False.

        I/O:
            checkpoint_path: Path to model checkpoint (.pt file).
            inputs_bucket: Base path for input Icechunk stores.
            output_bucket: Base path for output embeddings.
            chunk_size: Spatial chunk size in pixels.
            strip_budget_bytes: Byte budget bounding the resident *input*
                working set per chunk. ``process_chunk`` derives a northing
                strip height from this budget and the chunk's valid-timestep
                count, then loads/infers one strip at a time (full easting
                width) while the output buffers stay whole-chunk. A
                normal-density chunk resolves to a single strip (identical to
                the unstriped path); dense chunks split into several narrower
                strips so peak host RAM stays bounded. See
                ``DEFAULT_STRIP_BUDGET_BYTES``.

        Ray cluster:
            ray_address: Ray cluster address (None for local mode).
            use_spot: Whether to use spot instances.
            max_gpu_workers: Maximum number of GPU workers.
    """

    # Time window (required — no default)
    time_window: TimeWindow

    # Model architecture (v1.1 defaults)
    latent_dim: int = 192
    representation_dim: int = REPRESENTATION_DIM
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    fusion_method: str = "concat"
    num_obs_checkpoints: tuple[int, ...] = field(default_factory=lambda: DEFAULT_NUM_OBS_CHECKPOINTS)

    # Inference
    batch_size: int = 3584
    num_workers: int = 4
    norm_source: Literal["mpc", "aws"] = "aws"
    s1_orbit: Literal["ascending", "descending", "both"] = "ascending"
    # Deterministic sampling under v1.1 — no repeat variance; forced False in __post_init__.
    compute_std: bool = False

    # Ray actor resource reservation. num_gpus=1 is production default (one A10G per actor);
    # set to 0 for CPU-only runs (local smoke tests, plain runner on a non-GPU host).
    num_gpus: float = 1.0

    # I/O. Callers must supply absolute URIs; no environment-derived defaults.
    checkpoint_path: str = ""
    inputs_bucket: str = ""
    output_bucket: str = ""
    chunk_size: int = INFERENCE_CHUNK_SIZE
    strip_budget_bytes: int = DEFAULT_STRIP_BUDGET_BYTES

    # Ray cluster
    ray_address: str | None = None
    use_spot: bool = False
    max_gpu_workers: int = 500

    def __post_init__(self) -> None:
        """Validate and normalise config fields."""
        if self.norm_source not in _NORM_STATS:
            valid = ", ".join(repr(k) for k in _NORM_STATS)
            raise ValueError(f"Invalid norm_source: {self.norm_source!r}. Must be one of {valid}.")
        if self.s1_orbit not in {"ascending", "descending", "both"}:
            raise ValueError(f"Invalid s1_orbit: {self.s1_orbit!r}. Must be 'ascending', 'descending', or 'both'.")
        self.num_obs_checkpoints = _normalize_obs_checkpoints(self.num_obs_checkpoints)

        # v1.1 sampling is deterministic — no repeat variance to measure.
        self.compute_std = False
