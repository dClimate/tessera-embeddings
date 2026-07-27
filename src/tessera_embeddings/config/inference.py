"""Configuration for the Tessera embedding inference pipeline (v1.1 and v2 Large)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, final

from tessera_embeddings.config.time_windows import TimeWindow

# ---------------------------------------------------------------------------
# Model versions
# ---------------------------------------------------------------------------
# Which upstream Tessera model the pipeline runs. This selects the architecture,
# the checkpoint payload format, the band normalisation stats, and the model's
# native output width — everything downstream of the data plane. The input
# contract (band order, raw integer DOY, all-observation bucketing) is identical
# across versions, so loading/sampling/bucketing is version-agnostic.
#
# NOTE: distinct from the ``model_version`` string threaded into store
# provenance attrs (``conventions.build_convention_attrs``), which is a
# checkpoint *filename stem*, not a member of this literal.
ModelVersion = Literal["v1.1", "v2-large"]

DEFAULT_MODEL_VERSION: ModelVersion = "v1.1"

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

# v2 has a single checkpoint per student size (no norm_source split). The Large
# student's artifact is published on Hugging Face as
# ``geotessera/TESSERA-V-2.0-2B-L``, file ``ckpt/student_large.pt`` (175 MB); the
# filename below is the name it is expected to carry in our own model directory
# (``{inputs}/models/`` by default), mirroring how v1.1 checkpoints are staged.
# Nothing in the inference path fetches from Hugging Face — a full URI can be
# supplied instead via ``InferenceConfig.checkpoint_path``.
V2_LARGE_CHECKPOINT_NAME = "student_large.pt"


def checkpoint_filename(
    norm_source: str = "aws",
    *,
    model_version: ModelVersion = DEFAULT_MODEL_VERSION,
) -> str:
    """Return the canonical checkpoint filename for *model_version*.

    Args:
        norm_source: Which v1.1 checkpoint to use — ``"aws"`` (default) or
            ``"mpc"``. Ignored for ``"v2-large"``, which ships one checkpoint.
        model_version: Which model family — ``"v1.1"`` (default) or
            ``"v2-large"``.
    """
    if model_version == "v2-large":
        return V2_LARGE_CHECKPOINT_NAME
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
# Per-modality normalisation stats (v1.1 — keyed by norm_source)
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
        "s1_asc_mean": [5697.0859, 2838.6687],
        "s1_asc_std": [1671.3737, 1789.4116],
        "s1_desc_mean": [5759.1367, 2873.2854],
        "s1_desc_std": [1583.2858, 1747.8390],
    },
}

# ---------------------------------------------------------------------------
# Per-modality normalisation stats (v2 students — one fixed set)
# ---------------------------------------------------------------------------
# Sourced from the v2 student bundle's ``model.py`` (S2_BAND_MEAN/STD,
# S1A_BAND_MEAN/STD, S1D_BAND_MEAN/STD) as published with
# ``geotessera/TESSERA-V-2.0-2B-L``. v2 dropped the MPC/AWS split: every student
# hard-codes this single set, so ``norm_source`` does not apply. Ascending and
# descending S1 still carry their OWN stats and are normalised per orbit before
# being concatenated into the merged S1 stream.
_V2_NORM_STATS: dict[str, list[float]] = {
    "s2_mean": [
        1633.0042,
        1341.1090,
        1539.5536,
        3054.8269,
        3117.4658,
        2004.1648,
        2694.7275,
        2945.1504,
        2266.6079,
        1657.3094,
    ],
    "s2_std": [
        1999.4603,
        2014.7549,
        1929.2201,
        1754.2493,
        1649.9807,
        1936.8988,
        1748.6041,
        1708.6991,
        1207.5250,
        1108.6046,
    ],
    "s1_asc_mean": [5909.3921, 3405.0322],
    "s1_asc_std": [1507.1750, 1531.2615],
    "s1_desc_mean": [5816.1382, 3277.7576],
    "s1_desc_std": [1554.6475, 1546.4733],
}


def band_stats(
    model_version: ModelVersion = DEFAULT_MODEL_VERSION,
    norm_source: str | None = None,
) -> dict[str, list[float]]:
    """Return the ``{s2,s1_asc,s1_desc}_{mean,std}`` stats for a model version.

    v1.1 selects between the MPC- and AWS-normalised stat sets; v2 students
    hard-code one set and ignore *norm_source*.

    Args:
        model_version: Which model family the stats are for.
        norm_source: v1.1 stat set — ``"aws"`` (the default when ``None``) or
            ``"mpc"``.
    """
    if model_version == "v2-large":
        return _V2_NORM_STATS
    source = norm_source or "aws"
    if source not in _NORM_STATS:
        valid = ", ".join(repr(k) for k in _NORM_STATS)
        raise ValueError(f"Invalid norm_source: {source!r}. Must be one of {valid}.")
    return _NORM_STATS[source]


# Module-level aliases for the v1.1 default (AWS) stats — kept for direct importers.
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

# Embedding output dimension saved to the store. v1.1 produces 192-D reps and we
# save the first 128; v2 Large produces 128-D natively (Matryoshka-ordered), so
# the slice is a no-op there.
EMBEDDING_DIM = 128

# Internal model representation dimension (before the 128-D slice) — v1.1.
REPRESENTATION_DIM = 192


# ---------------------------------------------------------------------------
# Per-version model architecture
# ---------------------------------------------------------------------------


@final
@dataclass(frozen=True)
class ModelArch:
    """Architecture hyperparameters that must match a version's checkpoint.

    ``latent_dim`` is the encoder base dim; the transformer's ``d_model`` is
    ``latent_dim * 4`` in both versions.
    """

    latent_dim: int
    representation_dim: int
    nhead: int
    num_encoder_layers: int
    dim_feedforward: int
    enable_qk_norm: bool = False

    def as_dict(self) -> dict[str, int | bool]:
        """Field name → value, for comparing against an ``InferenceConfig``."""
        return {
            "latent_dim": self.latent_dim,
            "representation_dim": self.representation_dim,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
        }


# v1.1: latent_dim 192 → d_model 768, 192-D representation (first 128 saved).
_V11_ARCH = ModelArch(
    latent_dim=192,
    representation_dim=REPRESENTATION_DIM,
    nhead=4,
    num_encoder_layers=4,
    dim_feedforward=2048,
)

# v2 Large (43.8M params): read off the checkpoint's stored ``args`` —
# latent_dim 160 → d_model 640, 128-D Matryoshka representation, QK-norm off.
_V2_LARGE_ARCH = ModelArch(
    latent_dim=160,
    representation_dim=EMBEDDING_DIM,
    nhead=4,
    num_encoder_layers=4,
    dim_feedforward=2560,
    enable_qk_norm=False,
)

MODEL_ARCHS: dict[str, ModelArch] = {
    "v1.1": _V11_ARCH,
    "v2-large": _V2_LARGE_ARCH,
}

# v1.1 observation-count buckets. Every pixel with k valid observations is resampled
# to the next bucket size; pixels sharing a bucket form rectangular batches for the
# transformer. Multiples of 8 from 8 to 256 match tessera v1.1 defaults.
DEFAULT_NUM_OBS_CHECKPOINTS: tuple[int, ...] = tuple(range(8, 257, 8))

# CPU batch-prep pipeline depth for the inference loop (also the number of prep
# workers). Depth 1 starved the GPU whenever a forward ran shorter than one prep;
# depth 2 keeps a batch ready across consecutive short forwards. Lives here
# (torch-free) because actors.py sizes its background-load CPU reservation to
# match — one reserved core per prep worker — and cannot import inference.py at
# module scope (the Fargate flow runner has no torch).
PREFETCH_DEPTH = 2

# Spatial read-tile size for inference. The read/ChunkSpec grid stays 2000x2000;
# the *resident input working set* is bounded separately via density-sized
# northing strips (see actors._strip_height_for_density), so a 2000x2000 chunk's
# peak host RAM is capped by a per-strip byte budget rather than fixed by
# T x H x W. Sparse chunks load in one full-height strip; only dense chunks
# split. Independent of the storage chunk size written at ingest
# (config.ingest.INGEST_CHUNK_SIZE).
INFERENCE_CHUNK_SIZE = 2000


@final
@dataclass
class InferenceConfig:
    """Configuration for the inference pipeline (v1.1 or v2 Large).

    Attributes:
        Model selection:
            model_version: Which upstream model to run — ``"v1.1"`` (default) or
                ``"v2-large"``. Selects the architecture, the checkpoint payload
                format, the band normalisation stats, and the native output
                width. Architecture fields left at their v1.1 defaults are
                replaced with the selected version's spec (``MODEL_ARCHS``); a
                conflicting explicit value is rejected.

        Model architecture (must match checkpoint):
            latent_dim: Encoder base dim. Transformer d_model = latent_dim * 4.
            representation_dim: Model output dim (192 in v1.1, 128 in v2 Large).
            nhead: Transformer attention heads.
            num_encoder_layers: Transformer encoder layer count.
            dim_feedforward: Transformer FFN hidden dim.
            dropout: Dropout rate (zeroed at inference, kept for arch compat).
            fusion_method: "concat" or "sum" for combining S2/S1 representations.
            num_obs_checkpoints: Sorted bucket sizes for the all-obs sampler.

        Inference:
            batch_size: Per-GPU sub-batch size within a bucket.
            norm_source: Which v1.1 checkpoint/stats — ``"aws"`` (the resolved
                default) or ``"mpc"``. v2 hard-codes one stat set, so passing a
                value with ``model_version="v2-large"`` is rejected and the
                field resolves to ``None`` there.
            num_workers: GPU workers.
            s1_orbit: Which S1 orbit(s) — "ascending", "descending", or "both".
            compute_std: No-op (deterministic sampling); always False.

        I/O:
            checkpoint_path: Path to model checkpoint (.pt file).
            inputs_bucket: Base path for input Icechunk stores.
            output_bucket: Base path for output embeddings.
            chunk_size: Spatial chunk size in pixels.

        Ray cluster:
            ray_address: Ray cluster address (None for local mode).
            use_spot: Whether to use spot instances.
            max_gpu_workers: Maximum number of GPU workers.
            actor_request_batch_size: Request actors this many at a time (0 =
                all at once). Paces the EC2 demand the autoscaler forwards to
                AWS, which fulfils a large simultaneous ask slowly. Inference
                still starts on the first ready actor.
            actor_batch_placement_timeout_sec: Max seconds to wait for a batch's
                instances to join the cluster before requesting the next batch
                regardless (capacity-shortfall escape hatch).
    """

    # Time window (required — no default)
    time_window: TimeWindow

    # Which upstream model family to run (see ModelVersion).
    model_version: ModelVersion = DEFAULT_MODEL_VERSION

    # Model architecture (v1.1 defaults; overridden per model_version in
    # __post_init__ for any field still at its v1.1 default)
    latent_dim: int = 192
    representation_dim: int = REPRESENTATION_DIM
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    fusion_method: str = "concat"
    num_obs_checkpoints: tuple[int, ...] = field(default_factory=lambda: DEFAULT_NUM_OBS_CHECKPOINTS)

    # Inference
    batch_size: int = 7168
    num_workers: int = 4
    # None means "unset": resolved to "aws" for v1.1, kept None for v2 (which
    # has no norm_source split — an explicit value is rejected in __post_init__).
    norm_source: Literal["mpc", "aws"] | None = None
    s1_orbit: Literal["ascending", "descending", "both"] = "both"
    # Deterministic sampling — no repeat variance; forced False in __post_init__.
    compute_std: bool = False

    # Ray actor resource reservation. num_gpus=1 is production default (one GPU per
    # actor — L40S on g6e.xlarge workers);
    # set to 0 for CPU-only runs (local smoke tests, plain runner on a non-GPU host).
    num_gpus: float = 1.0

    # I/O. Callers must supply absolute URIs; no environment-derived defaults.
    checkpoint_path: str = ""
    inputs_bucket: str = ""
    output_bucket: str = ""
    chunk_size: int = INFERENCE_CHUNK_SIZE

    # Ray cluster
    ray_address: str | None = None
    use_spot: bool = False
    max_gpu_workers: int = 500

    # Actor request batching. AWS fulfils a large simultaneous EC2 ask slowly,
    # so we can request actors in batches and let the autoscaler see demand for
    # only one batch at a time. 0 disables batching (request all actors up
    # front — the historical behaviour). When enabled, inference still starts on
    # the first ready actor; subsequent batches are requested by the
    # work-stealing loop once the prior batch's instances have joined the
    # cluster (placement), so a slow model load never gates the next AWS ask.
    actor_request_batch_size: int = 50
    # Max seconds to wait for a batch's instances to be placed before requesting
    # the next batch anyway. Escape hatch so a capacity shortfall (e.g. AWS only
    # provisions 48/50) can't gate every remaining batch forever.
    actor_batch_placement_timeout_sec: float = 300.0

    def __post_init__(self) -> None:
        """Validate and normalise config fields."""
        if self.model_version not in MODEL_ARCHS:
            valid = ", ".join(repr(k) for k in MODEL_ARCHS)
            raise ValueError(f"Invalid model_version: {self.model_version!r}. Must be one of {valid}.")

        if self.model_version == "v1.1":
            self.norm_source = self.norm_source or "aws"
            if self.norm_source not in _NORM_STATS:
                valid = ", ".join(repr(k) for k in _NORM_STATS)
                raise ValueError(f"Invalid norm_source: {self.norm_source!r}. Must be one of {valid}.")
        elif self.norm_source is not None:
            raise ValueError(
                f"norm_source={self.norm_source!r} does not apply to model_version={self.model_version!r}: "
                "v2 students hard-code a single set of band statistics. Leave norm_source unset."
            )

        self._apply_arch_defaults()

        if self.s1_orbit not in {"ascending", "descending", "both"}:
            raise ValueError(f"Invalid s1_orbit: {self.s1_orbit!r}. Must be 'ascending', 'descending', or 'both'.")
        self.num_obs_checkpoints = _normalize_obs_checkpoints(self.num_obs_checkpoints)

        # Sampling is deterministic in both versions — no repeat variance to measure.
        self.compute_std = False

    def _apply_arch_defaults(self) -> None:
        """Adopt the selected version's architecture spec where fields are unset.

        The dataclass defaults describe v1.1, so for any other version a field
        still carrying its v1.1 default is treated as "unset" and replaced by the
        version's value (``MODEL_ARCHS``). A value that matches neither the v1.1
        default nor the version's spec was set deliberately and wrongly — the
        checkpoint would fail to load — so it is rejected here instead.
        """
        if self.model_version == "v1.1":
            return
        v11 = _V11_ARCH.as_dict()
        for name, value in MODEL_ARCHS[self.model_version].as_dict().items():
            current = getattr(self, name)
            if current not in (value, v11[name]):
                raise ValueError(
                    f"{name}={current!r} conflicts with model_version={self.model_version!r} "
                    f"(expected {value!r}); the checkpoint would not load."
                )
            setattr(self, name, value)
