# Prefect Leakage Audit

Diagnostic pass on the active codebase to quantify how much Prefect has leaked into domain code. This tells us how much refactoring Option A (thin Prefect wrapping) actually requires in the new repo.

**Methodology.** Grep for `import prefect`, `from prefect`, `get_run_logger`, `prefect.runtime`, `PrefectFuture`, `@task`, `@flow`, `.submit(`, `.map(` across `src/`. Archive directories (`src/*/_archive/`) are dead code and excluded from conclusions.

## Headline finding

**The domain layer is already orchestrator-clean.** `import prefect` appears in exactly *zero* files under `src/ingest/`, `src/inference/`, `src/config/`, and `src/storage/` (active code). Every Prefect import lives under `src/flows/` or `src/flows/flow_utils/`.

The Option-A refactor isn't "extract Prefect from domain code." It's "lift the existing `flows/` layer out as the reference orchestration wrapper and ship the rest as-is."

## Surface area by layer

| Layer | Files | Lines | Prefect imports |
|---|---|---|---|
| Domain (`src/ingest/`, `src/inference/`, `src/config/`, `src/storage/`) | ~40 | ~6,900+ | 0 |
| Flow orchestration (`src/flows/` + `src/flows/flow_utils/`) | 12 | ~2,770 | All of them |
| Archive (`src/*/_archive/`) | ~20 | — | Heavy, irrelevant |

Ratio of flows to domain lines: roughly 30/70. Target under Option A is ~10/90. The path to that target is mostly moving generic Python helpers that currently live in `flow_utils/` out into the domain layer, and slimming flows to sequencing only.

## What's in the flow layer today

All 12 active flow/flow_utils files:

| File | Lines | Role |
|---|---|---|
| `flows/tessera_full_pipeline.py` | 306 | Master orchestrator: chains ingest → embeddings → coarsen deployments |
| `flows/ingest_s2_roi_reflectance.py` | 335 | S2 reflectance ingest |
| `flows/ingest_s1_roi_sar.py` | 280 | S1 SAR ingest |
| `flows/tessera_embeddings.py` | 277 | GPU inference orchestration |
| `flows/coarsen_embeddings.py` | 109 | Coarsen 10m→500m embeddings |
| `flows/generate_roi.py` | 122 | Rasterize ROI GeoJSON to Zarr mask |
| `flow_utils/inference.py` | 437 | Actor lifecycle, Ray scheduling glue, two `@task`-decorated entry points |
| `flow_utils/config.py` | 412 | Dask cluster factory, `sliding_window_submit`, Fargate config |
| `flow_utils/availability.py` | 74 | Tile availability check (uses stdlib `ThreadPoolExecutor`, `@task` wrapper) |
| `flow_utils/results.py` | 44 | Collect results from futures |

`flow_utils/inference.py` and `flow_utils/config.py` are the thickest — these are where the meaningful Prefect-adjacent logic lives.

## Leakage categories observed

| Pattern | Count (active) | Location | Severity |
|---|---|---|---|
| `import prefect` / `from prefect` | 16 lines | Only `flows/` | None — correctly contained |
| `get_run_logger()` | ~10 call sites | Only `flows/` | None — correctly contained |
| `PrefectFuture` in type signature | 6 references | `flow_utils/config.py`, `flow_utils/results.py` | Low — confined to utilities |
| `prefect.runtime.*` | 0 | — | None |
| `@task` / `@flow` | Only in `flows/` | Only `flows/` | None |
| `.submit()` on tasks | 5 | Only in `flows/` | None |
| Prefect's `.map()` | 0 | — | None — team already doesn't use it |
| `prefect_dask.DaskTaskRunner` | 1 (flow_utils) | `flow_utils/config.py` | Medium — hard dependency |
| `prefect_aws.*` | 0 (active) | Only in archive | None |

Noteworthy pattern: `ingest_s2_roi_reflectance.py:49` and `ingest_s1_roi_sar.py:48` already have a dual-mode logger helper that falls back to stdlib when not inside a flow (catching `MissingContextError`). The team has been thinking about orchestrator-independent logging.

## AWS coupling (separate from Prefect)

| File | Coupling |
|---|---|
| `src/config/utils.py` | Hardcoded `s3://cl-tessera-*` bucket names in `resolve_buckets()` |
| `src/ingest/roi.py` | `_iam_storage_options()` for IAM-backed S3 reads |
| `src/inference/diagnostics.py` | Direct `boto3` import for CloudWatch log reading |
| `src/inference/actors.py` | Direct `boto3.client("s3")` for checkpoint download |
| `src/inference/assembly.py` | `fsspec.filesystem("s3", anon=False)` branches |

AWS coupling is far more scattered than Prefect coupling but remains in concentrated, boundary-level call sites (config resolution, checkpoint download, diagnostics). Every single one is a narrow boundary — not scattered throughout domain logic. Replacing with fsspec-only code plus an injected config is a straightforward refactor.

## Substrate coupling (Dask / Ray)

Worth tracking as a separate category because Dask and Ray are cloud-agnostic execution substrates and don't need removal — only provisioning abstraction:

- **Dask**: `Client(cluster)`, `FargateCluster`, `ECSCluster` construction is localized to `flow_utils/config.py`. Domain code accepts a Dask client as a context from the enclosing flow (e.g., `assemble_embeddings` enters its own cluster context). The AWS-specific part is `FargateCluster`/`ECSCluster`; the non-AWS part (standard `Client`) lives in `flow_utils/inference.py:392`.
- **Ray**: `ray.remote`, `ray.wait`, `ray.get`, actor handles are used throughout `src/inference/` (scheduling.py, progress.py, actors.py). This is *execution substrate* coupling, not orchestrator coupling. Ray runs on any cloud. What's AWS-specific is cluster provisioning, which lives in `infra/ray/cluster.py` (outside `src/`).

The key insight: Ray is part of the domain layer here — it's the data-parallel engine the inference code is written against. That's fine under the three-layer model. The port is about abstracting *provisioning*, not substituting the substrate.

## `infra/` coupling

Six active files in `src/flows/` and `src/flows/flow_utils/` import from `infra/`:

- `infra.prefect.utils` — SSM dashboard logging helpers (AWS-specific diagnostics)
- `infra.ray.cluster` — Ray cluster context manager, instance termination, AMI detection

Zero files under `src/ingest/`, `src/inference/`, etc. touch `infra/`. The coupling is entirely at the flow boundary — good.

## Per-file port verdict

A rough migration grade for each active flow file, based on how much rework is needed to reduce it to a thin orchestration wrapper:

| File | Grade | Notes |
|---|---|---|
| `flows/coarsen_embeddings.py` | A | Already thin; mostly config + one Dask cluster context |
| `flows/generate_roi.py` | A | Thin wrapper over domain ROI functions |
| `flows/ingest_s2_roi_reflectance.py` | **C** | Reclassified after deep inspection (see Prefect flow handling appendix §2.3). Outer flow is thin, but `@task process_roi_reflectance` contains ~130 lines of real work (two-phase SCL filtering, painter's-algorithm sort, per-day loop, retry policy). Domain extraction required. |
| `flows/ingest_s1_roi_sar.py` | **C** | Reclassified (appendix §2.4). Same shape as S2 ROI plus credential-refresh pattern inside the batched time loop. |
| `flows/tessera_embeddings.py` | B | Orchestration logic clean, Ray cluster entry via infra/ |
| `flows/tessera_full_pipeline.py` | B | Uses Prefect deployments — ties to Prefect server |
| `flow_utils/availability.py` | A | Uses stdlib concurrent.futures — a good template |
| `flow_utils/results.py` | A | ~40 LOC utility, trivial port |
| `flow_utils/config.py` | C | Heavily AWS-coupled (FargateCluster, ECSCluster, SSM envs). Needs split into substrate (generic Dask client factory) + provisioning (AWS-specific) |
| `flow_utils/inference.py` | C | Holds actor lifecycle + Ray scheduling + AWS tempfile cleanup. Most of this is domain, not flow. Needs extraction. |

The two C files are where most of the work is. `flow_utils/inference.py` in particular has several hundred lines of Ray actor orchestration and work-stealing glue that don't belong in the flow layer at all — they belong in `src/inference/`. Moving them up is a net simplification independent of Prefect.

## Bottom line

Option A is substantially scaffolded but requires real engineering work, not a file move. Corrected decomposition:

1. **Lift and publish domain code** (~6,900 LOC) from `src/ingest/`, `src/inference/`, `src/config/`, `src/storage/` — most moves with minor touch-up (replace hardcoded buckets with config-supplied paths, consolidate fsspec usage).
2. **Relocate misplaced logic** — move Ray actor lifecycle + work-stealing from `flow_utils/inference.py` into `src/inference/`. Simultaneously thins the flow layer and fattens the domain layer.
3. **Split `flow_utils/config.py`** into AWS-specific provisioning helper and the unified `sliding_window_submit`. The generic `Client(cluster)` wrap stays an inlined one-liner.
4. **Extract ~130 LOC each from S2 ROI and S1 ROI ingest `@task` bodies** into new domain modules (`ingest/s2_roi.py`, `ingest/s1_roi.py`) per Shape C pattern. This is real refactor work — two-phase SCL filtering, painter's-algorithm sort, credential refresh, retry policy — not a mechanical lift. See Prefect flow handling appendix for effort estimate (~5 days combined).
5. **Port the flows as the reference Prefect wrapping.** Outer flows become genuinely thin; `@task` shells under `orchestration/prefect/tasks/` are ~20 LOC each, calling domain functions.
6. **Write `runners/plain.py`** — the orchestrator-free sequencer that proves the split.
7. **Ship a generic Ray cluster YAML template + documented gotchas** instead of `infra/ray/cluster.py`.
8. **Replace `infra.prefect.utils` dashboard helpers** with stdlib logging or a pluggable diagnostics shim.

Steps 1, 5, 6, 7 are largely mechanical. Steps 2, 3, 4 are genuine refactors. Total engineering budget for the orchestration + provider surface: ~14 days single-engineer, per the Prefect flow handling appendix §8.
