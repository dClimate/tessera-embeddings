# tessera_embeddings

[![Lint](https://github.com/dClimate/tessera-embeddings/actions/workflows/lint.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/lint.yml)
[![Unit tests](https://github.com/dClimate/tessera-embeddings/actions/workflows/unit.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/unit.yml)
[![Architecture](https://github.com/dClimate/tessera-embeddings/actions/workflows/architecture.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/architecture.yml)
[![Nightly](https://github.com/dClimate/tessera-embeddings/actions/workflows/nightly.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/nightly.yml)

Generate per-pixel satellite embeddings at scale. Ports the HPC-based
[Tessera](https://github.com/ucam-eo/tessera) embedding pipeline to a
cloud-native, distributed architecture that runs on any major cloud —
or on a laptop (slowly).

## What this is

A Python library for:

- **Ingesting** Sentinel-1, Sentinel-2, and Landsat data from open
  STAC catalogs into chunked Zarr stores.
- **Cloud-masking and transforming** the data with scientifically
  validated pipelines.
- **Generating 128-dimensional Tessera embeddings** via distributed
  GPU inference with Ray.
- **Coarsening and assembling** the output into analysis-ready stores
  at configurable resolution.

The domain code — the scientific transformations, the inference
engine, the Zarr I/O — is cloud-agnostic and orchestrator-agnostic.
It's plain Python over `xarray`, `dask`, `zarr`, `ray`, and `fsspec`.
Runs on one laptop or a thousand GPUs.

Alongside the library we ship **reference orchestration**: opinionated
Prefect flows and AWS provisioning helpers that demonstrate how we
run this at production scale. They are examples, not requirements.

## What this isn't

- **Not a universal orchestration framework.** Prefect is the
  recommended and only core-maintained orchestrator. If you use
  Airflow, Dagster, Flyte, or Argo, the domain layer is a drop-in
  library — you'll rewrite the thin flow layer in your orchestrator's
  idiom. Community-maintained adapters for other orchestrators are
  welcome (see Contributing); we review them for fit and correctness
  but don't commit to maintaining them. See
  [`docs/orchestrator-swap.md`](docs/orchestrator-swap.md) for a
  worked example.
- **Not a multi-cloud abstraction.** AWS is the fully maintained
  reference cloud. Other clouds (GCP, Azure, Kubernetes) are
  supported by forking the provider templates —
  [`src/tessera_embeddings/providers/aws/ray.py`](src/tessera_embeddings/providers/aws/ray.py)
  and [`providers/aws/dask.py`](src/tessera_embeddings/providers/aws/dask.py)
  are explicit AWS glue you can use as a reference implementation,
  not an abstraction. See
  [`src/tessera_embeddings/providers/README.md`](src/tessera_embeddings/providers/README.md).
- **Not infrastructure-as-code.** We ship Ray cluster YAML templates
  and Python provisioning helpers, not Terraform or CDK. You bring
  your own IaC to create VPCs, security groups, and IAM.
- **Not a plugin system.** Providers aren't discovered via
  `entry_points`; you import the one you want.
- **Not a framework.** No base classes to inherit, no interfaces to
  implement. Flows are reference compositions; the domain layer is
  functions you call.

## Installation

`tessera_embeddings` is an inference library. The base install is the
ingestion pipeline (Sentinel-2/S1 data preparation, Zarr store management —
no torch, no Ray). Add `[inference]` for the Tessera embedding model and
distributed execution — that is what this library is for. The split is
practical: torch is large and CUDA variants are platform-specific.

```bash
# Typical install — ingestion pipeline + Tessera inference
pip install tessera_embeddings[inference]

# With uv (recommended for reproducible environments):
uv pip install tessera_embeddings[inference]

# Full production stack — inference + Prefect orchestration + AWS:
pip install tessera_embeddings[inference,prefect,aws]
```

For contributors, fresh-clone convenience: `uv sync` does the right thing.

```bash
git clone https://github.com/dClimate/tessera-embeddings
cd tessera-embeddings
uv sync                                    # reads uv.lock (universal CPU, all extras + dev)
```

Pre-solved lock files for specific environments live in `lock/`:

```
lock/inference-cpu.lock    cross-platform CPU torch + extras (laptop, CI, plain runner)
lock/inference-cu121.lock  Linux x86_64, CUDA 12.1 (production GPU AMI)
lock/dev.lock              inference-cpu + pytest/ruff/mypy (CI)
```

`lock/inference-cpu.lock` is `--universal` (Linux x86_64, macOS arm64/x86_64,
Linux arm64). `lock/inference-cu121.lock` is Linux x86_64 only. See
[`docs/environment-setup.md`](docs/environment-setup.md) for the full
matrix, MPS/Apple Silicon status, Windows guidance (use WSL2), GDAL setup,
and the rationale for the multi-lock-file setup.

## Quickstart

```bash
git clone https://github.com/dClimate/tessera-embeddings
cd tessera-embeddings
uv sync                                    # uv.lock covers all extras

# End-to-end pipeline on the bundled Story-County, IA quickstart ROI.
# Ingest → cloud mask → CPU inference → assemble. Expect ~30+ minutes;
# a warning banner confirms CPU inference is slow before kicking off.
python -m tessera_embeddings.orchestration.runners.plain examples/quickstart/config.yaml

# Skip inference for fast ingest-only sanity checks (~5 min).
python -m tessera_embeddings.orchestration.runners.plain \
    examples/quickstart/config.yaml --skip-inference
```

The default mode runs the full chain — inference and assembly are
coupled, so end-to-end is the primary demo. `--skip-inference` is the
fast path for contributors iterating on ingest changes without waiting
for CPU torch. Production inference always runs on GPU. See
[`docs/quickstart.md`](docs/quickstart.md) for prerequisites
(Earthdata Login credentials for OPERA, model checkpoint download).

## Running at scale

Two supported paths:

1. **Prefect + AWS (reference):** Flows in
   [`src/tessera_embeddings/orchestration/prefect/flows/`](src/tessera_embeddings/orchestration/prefect/flows/)
   run against `providers/aws/ray.py` + `providers/aws/dask.py`. See
   [`docs/providers/aws.md`](docs/providers/aws.md) for AWS
   provisioning.
2. **Your orchestrator + your cloud:** Reuse the domain layer; port
   the flow layer to your orchestrator; fork the provider templates
   for your cloud. See
   [`docs/orchestrator-swap.md`](docs/orchestrator-swap.md) and
   [`docs/providers/adding-your-own.md`](docs/providers/adding-your-own.md).

## Architecture

Three strict layers:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Prefect flows                                      │
│   orchestration/prefect/flows/                              │
│   Reference orchestration. Swap this directory for yours.   │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ Layer 2: Thin @task wrappers                                │
│   orchestration/prefect/tasks/                              │
│   Prefect-specific retry, caching, logger bridge.           │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ Layer 1: Domain (ingest/, inference/, storage/, config/)    │
│   Plain Python. No Prefect. No AWS-specific code.           │
│   Uses Ray for GPU parallelism, Dask for CPU scale-out.     │
└─────────────────────────────────────────────────────────────┘

Prefect is 100% quarantined under orchestration/prefect/.
orchestration/runners/ is the Prefect-free peer.

Per-cloud provisioning lives separately:
┌─────────────────────────────────────────────────────────────┐
│ Providers (providers/aws/, providers/local/, …)             │
│   providers/aws/ contains ray.py and dask.py.               │
│   AWS is fully maintained; local is for demo/tests.         │
└─────────────────────────────────────────────────────────────┘
```

Six hard rules enforced in CI:

1. No `import prefect` outside the flow layer.
2. Stdlib `logging` in the domain layer, not `get_run_logger()`.
3. Config is pydantic, not a Prefect Block (Blocks load into pydantic
   at flow entry).
4. Storage is fsspec, not orchestrator-specific filesystem
   abstractions.
5. Secrets enter at flow entry and travel as plain values.
6. Dask/Ray clients are passed in, never summoned below the flow
   layer.

If those rules hold, you can rewrite the flow layer for any
orchestrator without touching the domain.

### Why chunk size dominates everything

A subtle reality of distributed array workloads: **the task graph
your scheduler has to plan grows quadratically with how finely you
chunk the data.** Chunks too small means the scheduler spends more
time managing tasks than tasks spend doing work. Chunks too large
means workers can't fit a chunk in memory.

```
ROI: 20 km × 20 km, S2 reflectance, 10 m resolution, 12 dates:

chunks=200×200 (10× too small)        chunks=2000×2000 (the right size)
─────────────────────────────         ───────────────────────────────
□□□□□□□□□□  □□□□□□□□□□  □□□□□           ┌────────┐
□□□□□□□□□□  □□□□□□□□□□  □□□□□           │        │
□□□□□□□□□□  □□□□□□□□□□  □□□□□           │  ████  │  ← 1 chunk
□□□□□□□□□□  □□□□□□□□□□  □□□□□           │  ████  │
…  10 000 graph nodes  …                └────────┘
                                          12 nodes
graph build:    ~30 s                   graph build:   <1 s
scheduler RAM:  ~1 GB                   scheduler RAM: <50 MB
overhead:       95% of wall-clock       overhead:      <5%
```

The defaults (`DEFAULT_CHUNK_SIZE = 2000`) are calibrated for the
inference pipeline; if you go smaller you'll watch the Dask scheduler
hang on graph construction before any worker does any work, and
larger you'll OOM on a g5.2xlarge. If you change them, profile.

### Using these architecture checks in your own repo

The hard-rule checks ship as a reusable module so downstream consumers
(closed-source forks, community adapter contributors) can apply the
same contract to their own code:

```bash
# Run against any source tree
uv run python -m tessera_embeddings.architecture_tests \
    --source path/to/your_package/ \
    --allowlist your-arch-allowlist.toml
```

The allowlist file (TOML) documents intentional deviations (e.g.
"Prefect imports in my own `orchestration/prefect/` are expected").
See
[`src/tessera_embeddings/architecture_tests/`](src/tessera_embeddings/architecture_tests/)
for the rule definitions, allowlist schema, and worked examples.

### Public API surface

This library follows semver for the documented public API surface.
Anything outside it — underscore-prefixed names, modules whose names
start with `_`, anything under `tessera_embeddings.orchestration.prefect.*` —
is implementation detail and may change between minor releases. The
full public-API surface is listed in
[`docs/public-api.md`](docs/public-api.md). External code should
depend only on items listed there.

## The test that proves decoupling

[`src/tessera_embeddings/orchestration/runners/plain.py`](src/tessera_embeddings/orchestration/runners/plain.py)
is an orchestrator-free sequencer that calls the same domain
functions as the Prefect flows, without Prefect. By default it runs
the full end-to-end pipeline (ingest → cloud mask → inference →
assembly) on a laptop with torch on CPU via Ray's local mode. Slow on
real workloads, practical on the Story-County quickstart ROI we ship
for exactly this purpose.

A `--skip-inference` flag runs only ingest for fast sanity checks;
assembly is skipped because it has nothing to assemble without
embeddings.

Why end-to-end on CPU is the credibility bar we chose:

- Assembly depends on inference outputs — "ingest-only" is a
  convenience path for contributors, not a meaningful full-stack demo.
- If CPU torch works without modification, no GPU-specific coupling
  has leaked into the domain layer. That's the strongest
  architectural separation check we can make without deploying to
  multiple cloud targets.
- `plain.py` is the reference for users porting to
  Airflow/Dagster/Flyte: everything it does is the non-Prefect wiring
  they'll need to reproduce.

For CI: `plain.py --skip-inference` is the fast PR check (minutes).
The end-to-end run on the quickstart ROI runs as a nightly or
opt-in job (too slow for every PR). Fast PR checks also use
AST-based architecture rules (§Architecture) to catch Prefect leaks
at the import level without running the pipeline.

## What's in here

```
src/tessera_embeddings/
  config/                pydantic config models
  ingest/                STAC ingestion, ROI rasterization, auth
  inference/             GPU inference (Ray actors, work-stealing scheduler)
  storage/               Zarr stores, manifests
  orchestration/
    concurrency.py       sliding_window_submit — shared by flows + runners
    prefect/             Prefect — 100% quarantined here
      flows/             @flow-decorated orchestration (Layer 3)
      tasks/             thin @task wrappers (Layer 2)
    runners/             non-Prefect entry points (plain.py)
  providers/             concrete cloud-provisioning glue
    aws/                 ray.py, dask.py, gotchas.md
    local/               ray.py, dask.py
  architecture_tests/    reusable layer-rule checker (CLI + Python API)
```

## Documentation

- [`docs/quickstart.md`](docs/quickstart.md) — laptop demo
  end-to-end, including GPU inference.
- [`docs/environment-setup.md`](docs/environment-setup.md) — lock
  files, CUDA variants, uv setup.
- [`docs/configuration.md`](docs/configuration.md) — the pydantic
  config tree.
- [`docs/prefect-setup.md`](docs/prefect-setup.md) — standing up your
  own Prefect server: work pool shape, Blocks used, deployment
  examples, common gotchas. We don't ship IaC for the server itself;
  this doc tells you what to build.
- [`docs/providers/aws.md`](docs/providers/aws.md) — running on AWS
  with Prefect.
- [`docs/providers/adding-your-own.md`](docs/providers/adding-your-own.md) —
  porting to GCP, Azure, k8s.
- [`docs/orchestrator-swap.md`](docs/orchestrator-swap.md) — running
  without Prefect.
- [`docs/public-api.md`](docs/public-api.md) — the documented public
  API surface covered by semver.
- [`src/tessera_embeddings/providers/aws/gotchas.md`](src/tessera_embeddings/providers/aws/gotchas.md) —
  operational knowledge for Ray clusters (head sizing, autoscaler,
  spot, AMI bake, teardown safety nets).
- [`context_docs/`](context_docs/) — design decisions, framing,
  rationale.

## Downstream consumers

This library has a known production downstream consumer:
`yield_modeling`, a private repo that imports this library, supplies
AWS infrastructure, and runs production workloads. We've wired the
OSS CI to run a fast smoke test against `yield_modeling` on every
PR — catches accidental breaking changes at the point of change
instead of in production.

The smoke-test workflow lives at
`.github/workflows/downstream-smoke.yml`. It is **initially disabled**
(only `workflow_dispatch` enabled, no `pull_request` trigger).
Activation criteria:

1. `yield_modeling` has its first internal release.
2. A read-only GitHub token (`YIELD_MODELING_READ_TOKEN`) is
   configured as a repo secret.
3. `yield_modeling/main` reliably has a green test suite.

Once active, the smoke test runs `yield_modeling`'s
`pytest tests/unit tests/architecture` against the OSS PR's SHA.
**Failure is informational, not blocking** — it gives the OSS PR
author a heads-up about downstream impact. We never make this a
required status check; that would give a private repo veto power
over public releases.

Other downstreams (community adapters, external production users)
can wire up the same pattern against their own forks. See the
smoke-test workflow file for the template.

## Contributing

We accept:

- Bug fixes and improvements to the domain layer.
- Documentation and examples.
- Additional reference provider implementations (new clouds, new
  substrates). Ship them as concrete code under
  `providers/<your-target>/`, not as abstractions. See
  [`docs/providers/adding-your-own.md`](docs/providers/adding-your-own.md).
- **Community-maintained orchestrator adapters** (Airflow, Dagster,
  Flyte, Argo, …). These are welcome but are not core-maintained.
  Requirements for acceptance:
    1. **Explicit maintenance commitment** from the contributor,
       named in the adapter's own README. If the named maintainer
       goes silent and the adapter falls into disrepair, it will be
       moved to an `archived/` directory with a deprecation notice —
       not deleted, but clearly labeled as unmaintained.
    2. **Parity test against `runners/plain.py`** on the bundled
       quickstart ROI, in CI. Your adapter's flow must produce
       identical output to the plain runner for the same inputs. See
       [`tests/parity/adapter_template/`](tests/parity/adapter_template/)
       for the starter template.
    3. **Parity doc** — a short markdown file listing which features
       map cleanly from our Prefect reference, which have idiomatic
       equivalents in the new stack, and which have no analog.
    4. **Clear labeling** — the adapter's README and module docstring
       both state "community-maintained, not core-supported."
  Core maintainers will review for correctness and fit, but won't
  debug adapter-specific issues or unblock adapter-only breakages.

We don't accept:

- Abstract `Runner` / `Orchestrator` / `Provider` interfaces. The
  architecture deliberately avoids them. See
  [`context_docs/decisions/`](context_docs/decisions/) for the
  reasoning.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Acknowledgments

Ports the [Tessera](https://github.com/ucam-eo/tessera) pipeline to a
cloud-native architecture. Built at
[Cyclops](https://cyclops-mrv.com).
