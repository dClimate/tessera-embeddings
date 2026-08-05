"""Configuration for the Tessera v1.1 embedding inference pipeline."""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, final

from tessera_embeddings.config.time_windows import TimeWindow

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Staged-output code identity
# ---------------------------------------------------------------------------

#: SEED for the source whose behaviour determines what a staged tile CONTAINS.
#:
#: The fingerprint covers this plus everything it imports, transitively
#: (:func:`_first_party_import_closure`) — a hand-maintained list cannot keep up with
#: the imports, and it fails in the silent direction when it falls behind.
#:
#: A change anywhere in that closure means already-staged tiles were produced by
#: different logic and must not be mixed with new ones. Code outside it — orchestration,
#: provider and tooling code, and the ingest and storage modules inference never reaches
#: — does not invalidate staging, which is the point of narrowing away from the old
#: whole-build identity.
#:
#: The whole ``inference`` package seeds it rather than a hand-picked module list, and
#: the closure then pulls in what those modules import (about a third of the package).
#: Both OVER-include: ``assembly.py`` reads staged tiles rather than producing them, and
#: ``storage.zarr_store`` arrives for ``compute_doy`` but brings its own dependencies
#: with it. Deliberate, because the two errors are not symmetric — over-including costs
#: a spurious re-inference, and ``force_staging_reuse`` exists to wave one through, while
#: under-including silently assembles tiles from two code versions into one write-once
#: zone-year and has no escape hatch at all. Err toward the expensive failure, never the
#: silent one.
_STAGED_OUTPUT_SOURCES: tuple[str, ...] = ("inference", "config/inference.py")


def _first_party_import_closure(seed: list[Path], root: Path) -> set[Path]:
    """Every in-package module reachable from *seed*, following imports transitively.

    The seed names the code that OBVIOUSLY produces a staged tile; this finds the code
    it delegates to. Without it the identity misses exactly the dependencies that matter
    most — ``data_loading`` calls ``compute_doy`` from ``storage.zarr_store`` to build a
    model input, and reads ``TimeWindow`` from ``config.time_windows`` to choose which
    observations are used. A change to either alters a staged tile's CONTENT while
    leaving a hand-listed identity unmoved, so a retry mixes tiles from two code
    versions into one write-once zone-year.

    A closure rather than a longer list, because a list only covers the dependencies
    someone remembered: it goes stale the first time an import is added, and it goes
    stale silently, in the direction that loses data.

    Parses rather than imports — this runs on a flow runner that has no torch, and
    importing to inspect would execute module bodies for a hash. ``from x import y``
    contributes both ``x`` and ``x.y`` as candidates, since either may name the module.
    Non-package imports and names that resolve to no file are skipped.
    """
    package = __name__.split(".")[0]

    def module_file(dotted: str) -> Path | None:
        rel = dotted.removeprefix(package + ".").replace(".", "/")
        for candidate in (root / f"{rel}.py", root / rel / "__init__.py"):
            if candidate.exists():
                return candidate
        return None

    def imported_modules(path: Path) -> set[str]:
        found: set[str] = set()
        for node in ast.walk(ast.parse(path.read_bytes())):
            if isinstance(node, ast.Import):
                found.update(a.name for a in node.names if a.name.startswith(package))
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(package):
                found.add(node.module)
                found.update(f"{node.module}.{a.name}" for a in node.names)
        return found

    reached, stack = set(seed), list(seed)
    while stack:
        for dotted in imported_modules(stack.pop()):
            path = module_file(dotted)
            if path is not None and path not in reached:
                reached.add(path)
                stack.append(path)
    return reached


def inference_code_identity() -> str:
    """Fingerprint of the code that determines staged tile CONTENT, not of the whole build.

    Replaces the AMI-ID-plus-tarball-ETag identity that used to feed the campaign's staging
    ``run_id``. That identity was correct but far too wide: re-baking the worker AMI, or a
    hotfix anywhere in the repo, changed it and so abandoned every staged tile and re-ran
    inference for no semantic reason. At campaign scale that is the difference between a
    hotfix costing minutes and costing a re-run.

    Hashes :data:`_STAGED_OUTPUT_SOURCES` **and everything it imports**, transitively —
    the seed alone misses the code it delegates the tile's contents to. Contents are
    hashed with their package-relative paths, sorted, so the digest is stable across
    machines and checkouts.

    **What this deliberately no longer covers: dependency drift.** The old AMI-based identity
    changed when the image changed, so a re-bake onto a new torch was caught; a source hash is
    blind to it. Two things bound that gap — the AMI is resolved once and PINNED into every
    fill of a campaign (so one run cannot straddle two images), and a model change ships under
    a new checkpoint filename, which is still in the fingerprint. A deliberate library upgrade
    mid-campaign is the residual case, and it wants the force-new escape hatch rather than a
    silent reuse.
    """
    root = Path(__file__).resolve().parent.parent
    seed: list[Path] = []
    for entry in _STAGED_OUTPUT_SOURCES:
        target = root / entry
        if target.is_dir():
            seed.extend(p for p in target.rglob("*.py") if "__pycache__" not in p.parts)
        elif target.is_file():
            seed.append(target)
        else:  # pragma: no cover - a rename must fail loudly, not fingerprint nothing
            raise FileNotFoundError(
                f"{target} is listed in _STAGED_OUTPUT_SOURCES but does not exist. A moved or "
                f"renamed module must update that list — silently fingerprinting fewer files "
                f"would let two code versions share one staging prefix."
            )
    files = _first_party_import_closure(seed, root)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    return f"infcode-{digest.hexdigest()[:16]}"


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
        "s1_asc_mean": [5697.0859, 2838.6687],
        "s1_asc_std": [1671.3737, 1789.4116],
        "s1_desc_mean": [5759.1367, 2873.2854],
        "s1_desc_std": [1583.2858, 1747.8390],
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

#: Resolved value meaning "this ROI has no usable radar at all, and that is a finding".
#:
#: Distinct from an empty request: nobody ASKS for ``"none"``, it is what ``"both"`` resolves
#: to once probing shows neither orbit wrote a store. Some land has no dual-pol VV+VH radar in
#: principle — over ice Sentinel-1 runs Extra Wide swath with HH/HV, which the OPERA query
#: correctly discards — so a zone can be permanently radar-free while the catalogue holds a
#: hundred thousand granules for it. Requiring a SAR store there fails the cell forever.
#:
#: Defined HERE, in the layer the config lives in, because ``InferenceConfig`` has to validate
#: it and the loader that resolves it already depends on this module.
S1_ORBIT_NONE = "none"

# Embedding output dimension — v1.1 produces 192-D reps; we save the first 128.
EMBEDDING_DIM = 128

# Internal model representation dimension (before the 128-D slice).
REPRESENTATION_DIM = 192

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

# Spatial read-tile size for inference, on both paths: one tile is exactly one
# 2048-px output shard (ADR-008 D3), so assembly writes whole shards instead of
# read-modify-writing a partial output chunk at each tile edge. A literal rather
# than an import of ``store_layout.SHARD_PX`` — that module imports EMBEDDING_DIM
# from this one — and ``test_store_layout`` pins the two together.
#
# A tile's peak host RAM is not T x H x W: the resident input working set is
# bounded by density-sized northing strips (actors._strip_height_for_density),
# so sparse tiles load in one full-height strip and only dense ones split.
INFERENCE_CHUNK_SIZE = 2048


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
            s1_orbit: Which S1 orbit(s) — "ascending", "descending", "both", or
                "none" for radar-free land (which requires allow_s2_only).
            compute_std: No-op under v1.1 (deterministic sampling); always False.

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
    batch_size: int = 7168
    num_workers: int = 4
    norm_source: Literal["mpc", "aws"] = "aws"
    s1_orbit: Literal["ascending", "descending", "both", "none"] = "both"
    """Which S1 orbit direction(s) to read.

    ``"none"`` is a RESOLVED value, not a request: it is what ``"both"`` becomes once probing
    finds that neither orbit wrote a store. Parts of the globe are radar-free in principle —
    over ice Sentinel-1 flies Extra Wide swath with HH/HV, which the dual-pol query correctly
    discards — so this is a permanent property of the terrain rather than an ingest failure,
    and a global product cannot refuse it. It requires ``allow_s2_only``: with no radar at all
    every pixel has zero S1 observations, so the default gate would skip every one of them.
    """
    # Deterministic sampling under v1.1 — no repeat variance; forced False in __post_init__.
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

    # Appended last on purpose: InferenceConfig is public API (docs/public-api.md)
    # and this is a positional dataclass, so a new field in the middle would
    # silently rebind later positional args in downstream construction. Keep new
    # fields at the tail.
    #
    # Embed S2-valid pixels that have ZERO S1 observations (sub-zone SAR coverage
    # gaps — swath edges/holes; worst at high latitudes). Such a pixel gets the
    # upstream v1.1 missing-S1 convention: an all-zeros (normalized-space) S1 slice
    # at the smallest bucket — exactly ucam-eo/tessera's `_sample_s1_merged` zero
    # return — so this restores upstream parity rather than inventing an input.
    # Default False: pixels without S1 are skipped (this pipeline's historical
    # gate). Per-pixel provenance is free either way: an embedded pixel with
    # s1_asc_obs_count + s1_desc_obs_count == 0 is an S2-only embedding. NOTE:
    # S2-only embedding QUALITY is unvalidated for this S1-trained checkpoint —
    # see the optional-S1 ADR before enabling in production.
    allow_s2_only: bool = False

    def __post_init__(self) -> None:
        """Validate and normalise config fields."""
        if self.norm_source not in _NORM_STATS:
            valid = ", ".join(repr(k) for k in _NORM_STATS)
            raise ValueError(f"Invalid norm_source: {self.norm_source!r}. Must be one of {valid}.")
        if self.s1_orbit not in {"ascending", "descending", "both", S1_ORBIT_NONE}:
            raise ValueError(
                f"Invalid s1_orbit: {self.s1_orbit!r}. Must be 'ascending', 'descending', 'both', or {S1_ORBIT_NONE!r}."
            )
        if self.s1_orbit == S1_ORBIT_NONE and not self.allow_s2_only:
            # FORCED, not refused, and not left alone. Refusing would defeat the decision that
            # radar-free land is acceptable — a global product cannot reject terrain that has no
            # dual-pol radar in principle. Leaving the flag alone would be worse than either:
            # with no radar every pixel has zero S1 observations, the default gate skips every
            # one, and the fill would COMPLETE having written nothing while tagging the year
            # done. An empty result that reads as success is the one outcome no later run
            # revisits.
            #
            # Safe to derive rather than demand from the caller because it is a function of
            # s1_orbit alone, so a resume computes the same value from the same inputs and the
            # staged-chunk consistency check in the embeddings flow still holds.
            self.allow_s2_only = True
            logger.warning(
                "s1_orbit=%r: forcing allow_s2_only=True. This ROI has no radar at all, so "
                "EVERY pixel gets the missing-S1 input (all-zeros normalised S1) and every "
                "embedding here is S2-only — whose quality is unvalidated for this S1-trained "
                "checkpoint (see the optional-S1 ADR). Without this the fill would write "
                "nothing and still report success.",
                self.s1_orbit,
            )
        self.num_obs_checkpoints = _normalize_obs_checkpoints(self.num_obs_checkpoints)

        # v1.1 sampling is deterministic — no repeat variance to measure.
        self.compute_std = False
