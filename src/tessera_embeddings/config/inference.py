"""Configuration for the Tessera embedding inference pipeline (v1.1 and v2 Large)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, final

from tessera_embeddings.config.code_identity import source_identity
from tessera_embeddings.config.time_windows import TimeWindow

logger = logging.getLogger(__name__)

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

#: Prefix marking a staging run_id whose chunks were produced by a v2 student.
#:
#: Lives HERE rather than beside the flow that mints it, because the check it enables has to
#: run in two layers. The Prefect flow composes the run_id; ``inference.runner`` is a
#: documented public entry point that a caller can drive directly, with its own run_id and
#: config, and it is the boundary staged chunks are actually reused at. The flow imports
#: prefect, so the runner cannot reach the constant there without dragging the orchestration
#: layer into the inference one — which is why the vocabulary moved down instead.
V2_RUN_PREFIX = "v2-"


def run_id_prefix(model_version: ModelVersion) -> str:
    """The run_id prefix that marks *model_version*'s staging, empty for v1.1.

    **Minting and checking have to share this, or the check refuses correct runs.** The
    encoder guard reads an unprefixed id as v1.1 — which is right for a resume, and fatal for
    a fresh run if the minting site did not add the prefix: the plain runner minted a bare
    uuid, so selecting v2 there raised before an actor started. Relaxing the guard for fresh
    runs would have hidden that and kept the real defect, because the bare id it left behind
    is then misread as v1.1 by every LATER resume of the same run.

    v1.1 gets no prefix, so every existing id, staging path and resume is unchanged.
    """
    return V2_RUN_PREFIX if model_version != "v1.1" else ""


def staged_by_v2(run_id: str | None) -> bool:
    """Whether *run_id* names staging produced by a v2 student."""
    return bool(run_id) and (run_id or "").startswith(V2_RUN_PREFIX)


def assert_run_id_matches_model(run_id: str | None, model_version: ModelVersion) -> None:
    """Raise if *run_id*'s encoder prefix contradicts *model_version*.

    **A staged chunk does not record which student wrote it.** It is (H, W, 128) int8 either
    way — v1.1 saves the first 128 of its 192-d representation, v2 emits 128 natively — so a
    resume that reuses staging by run_id alone cannot tell the two apart by inspection, and a
    mixed store is published with a single encoder stamped on all of it.

    Enforced at every boundary that reuses staging rather than only where the run_id is minted:
    the flow's guard protects the flow's callers, and this protects everyone else.
    """
    if run_id and staged_by_v2(run_id) != (model_version != "v1.1"):
        staged = "v2" if staged_by_v2(run_id) else "v1.1"
        raise ValueError(
            f"Cannot resume run {run_id!r} (staged by {staged}) with model_version={model_version!r}: "
            "its staged chunks came from the other encoder, and continuing would publish a mix of "
            "both and stamp the store with one of them. Resume with the "
            f"{staged} model, or start a fresh run."
        )


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


# ---------------------------------------------------------------------------
# Staged-output code identity
# ---------------------------------------------------------------------------

#: SEED for the source whose behaviour determines what a staged tile CONTAINS.
#:
#: The fingerprint covers this plus everything it imports, transitively (see
#: :mod:`tessera_embeddings.config.code_identity`) — a hand-maintained list cannot keep
#: up with the imports, and it fails in the silent direction when it falls behind.
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


def inference_code_identity() -> str:
    """Fingerprint of the code that determines staged tile CONTENT, not of the whole build.

    Replaces the AMI-ID-plus-tarball-ETag identity that used to feed the campaign's staging
    ``run_id``. That identity was correct but far too wide: re-baking the worker AMI, or a
    hotfix anywhere in the repo, changed it and so abandoned every staged tile and re-ran
    inference for no semantic reason. At campaign scale that is the difference between a
    hotfix costing minutes and costing a re-run.

    Hashes :data:`_STAGED_OUTPUT_SOURCES` **and everything it imports**, transitively —
    see :func:`~tessera_embeddings.config.code_identity.source_identity` for the closure
    and for what a source hash cannot see.

    The residual dependency-drift case is bounded twice here: the AMI is resolved once
    and PINNED into every fill of a campaign, so one run cannot straddle two images, and
    a model change ships under a new checkpoint filename, which is in the fingerprint
    separately. A deliberate library upgrade mid-campaign wants the force-new escape
    hatch rather than a silent reuse.
    """
    return source_identity(_STAGED_OUTPUT_SOURCES, "infcode")


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

#: A pixel with fewer than this many radar observations in the year is "radar-thin".
#:
#: Reported per year alongside the radar-free count so a downstream user can tell a pixel
#: the radar barely saw from one it saw normally — a distinction the embedding itself does
#: not expose, since both produce an embedding. The exact per-pixel counts are in the store;
#: this only sets where the summary draws its line, and it is deliberately generous:
#: twelve is roughly one observation a month, below which a year's radar signal is thin
#: however it is sampled.
RADAR_THIN_MAX_OBS = 12

#: Minimum valid Sentinel-2 observations for a pixel to be EMBEDDED AT ALL, in the calendar
#: year being filled. A pixel below it is written as fill, exactly as an out-of-ROI pixel is.
#:
#: **Not the counterpart of :data:`RADAR_THIN_MAX_OBS`, and the asymmetry is deliberate.** The
#: radar line labels; this one refuses. They are also not comparable as numbers: radar sees
#: through cloud, so its count is set by orbit geometry and one observation a month is a
#: thin-but-usable year, while optical loses most of its passes to cloud and what survives
#: masking is a small fraction of the overpasses.
#:
#: **This is a refusal, not a label, and the difference is that it is not reversible.** Under
#: its old name (``OPTICAL_THIN_MAX_OBS``, 40 until 2026-08-12, then 15) nothing was refused for
#: being under it and the per-pixel counts in ``s2_obs_count`` told the whole story either way —
#: so raising or lowering it changed what summaries said and never what was published. That is no
#: longer true: a refused pixel has no embedding, so recovering one is a re-run of its whole
#: shard. **A reader of commits or documents from before 2026-08-13 will find the opposite claim
#: under the old name.**
#:
#: Two things bound how freely this value moves:
#:
#: * it is stamped into the global store's root attrs and is part of their write-once identity,
#:   so **a store cannot be re-stamped with a different value** — changing the line means a new
#:   store, not a migration;
#: * the seeder takes it explicitly and does not default to this constant, so nothing can stamp
#:   a store by inheriting whatever happens to be here.
#:
#: **What the thin counters mean now that this is the only line.** ``s2_thin_px`` per chunk and
#: ``s2_thin_below_obs`` in run provenance count EMBEDDED pixels below this value. While nothing
#: refuses, that is a preview of what a refusal would remove. Once the gate is enforced it is an
#: invariant: **the count must be zero, and a non-zero one means the gate leaked.** Each cell's
#: provenance records the line its own numbers were produced under, so cells filled before and
#: after a change are comparable rather than silently restated.
#:
#: **15 is a DECISION (Robert and colleague, 2026-08-17), not a placeholder** — it replaces the 25
#: that stood here from 2026-08-13 purely so the machinery could be built. Coverage was chosen over
#: reproducibility: the line keeps **94% of pixels rather than 79%**, and the cost, accepted
#: knowingly, is that two independent embeddings of the same ground agree less well. The trade is
#: recorded in full, including what 15 admits that 20 would not, in
#: ``context_docs/design/minimum-optical-depth-plan.md``.
#:
#: **A store already seeded at another value keeps it.** The root attr decides, and it is write-once,
#: so this constant changes what a NEW store is seeded with and what the thin counters report —
#: never what an existing store enforces.
#: The four orbit selections, as one name rather than a Literal repeated at every boundary.
#: ``"none"`` is a real member: the S2-only path (ADR-013) carries it end to end, and a signature
#: that omits it forces a cast at each hop instead of saying what the value is.
S1Orbit = Literal["ascending", "descending", "both", "none"]

OPTICAL_MIN_OBS = 15

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

# --- Per-model deployment facts (same keys as MODEL_ARCHS) -----------------
# Non-architecture properties of the model. Centralised because the failure
# mode is a constant that silently keeps a PREVIOUS version's value.

#: PUBLIC encoder URL a store advertises as ``geoemb:model`` — the externally
#: resolvable identity of the model FAMILY, deliberately not the internal
#: checkpoint filename (that is ``checkpoint_id``).
MODEL_ENCODER_URLS: dict[str, str] = {
    "v1.1": "https://geotessera.org/model/1.1",
    # v2 Large is published as a Hugging Face repo, not under the
    # geotessera.org/model/<version> scheme. Replace if a canonical path is
    # minted; seed and fill need only agree on whatever is registered here.
    "v2-large": "https://huggingface.co/geotessera/TESSERA-V-2.0-2B-L",
}

#: Planning-only inference throughput (px/s/worker). Strip and prefetch
#: planning divides by this to ask "will the GPU stay busy long enough to hide
#: this read?", so a faster model must not inherit a slower one's figure.
#: Strategy only — never correctness, never a RAM bound. Calibration in
#: context_docs.
MODEL_EST_PX_PER_SEC: dict[str, float] = {
    "v1.1": 16_000.0,
    "v2-large": 22_000.0,
}


def encoder_url(model_version: ModelVersion = DEFAULT_MODEL_VERSION) -> str:
    """Public ``geoemb:model`` URL for *model_version*.

    Raises on unknown versions rather than defaulting: a store stamped with
    another encoder's URL misidentifies itself to every downstream reader, and
    does so silently.

    MERGE NOTE (global-tessera-scoping): that branch's
    ``conventions.expected_model_url()`` re-derives this URL at fill time and
    compares it to the seed's, from a single module constant. Route it through
    here and pass the fill's ``model_version``, or the gate checks a v2 fill
    against v1.1's URL — rejecting a correct run, or waving through a real
    mismatch when the seed carried the same stale constant.
    """
    try:
        return MODEL_ENCODER_URLS[model_version]
    except KeyError:
        valid = ", ".join(repr(k) for k in MODEL_ENCODER_URLS)
        msg = f"No public encoder URL registered for model_version={model_version!r}. Known: {valid}."
        raise ValueError(msg) from None


def est_px_per_sec(model_version: str = DEFAULT_MODEL_VERSION) -> float:
    """Planning-only inference-rate estimate; unknown versions fall back.

    Falls back rather than raising because this is a speed hint — planning with
    a stale number beats refusing to plan.
    """
    return MODEL_EST_PX_PER_SEC.get(model_version, MODEL_EST_PX_PER_SEC[DEFAULT_MODEL_VERSION])


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
    s1_orbit: S1Orbit = "both"
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

    # Minimum valid optical observations for a pixel to be embedded at all, or None for "embed
    # everything with any optical input" — the historical behaviour, and what every non-campaign
    # caller wants. See OPTICAL_MIN_OBS for what a refusal costs. None rather than 0 because the
    # two are different statements and only one of them is recoverable from a config dump: a
    # campaign whose value silently resolved to 0 would publish under no rule while believing it
    # had one, which is the shape of two failures already in this repo's register.
    optical_min_obs: int | None = None

    # Which upstream model family to run (see ModelVersion).
    model_version: ModelVersion = DEFAULT_MODEL_VERSION

    def __post_init__(self) -> None:
        """Validate and normalise config fields."""
        if self.optical_min_obs is not None and self.optical_min_obs <= 0:
            raise ValueError(
                f"optical_min_obs={self.optical_min_obs} refuses nothing — pass None for no "
                "minimum optical depth, or a positive number of observations."
            )
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
