# TESSERA Inference

[![Lint](https://github.com/dClimate/tessera-embeddings/actions/workflows/lint.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/lint.yml)
[![Unit tests](https://github.com/dClimate/tessera-embeddings/actions/workflows/unit.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/unit.yml)
[![Architecture](https://github.com/dClimate/tessera-embeddings/actions/workflows/architecture.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/architecture.yml)

Generate per-pixel (10m^2) TESSERA satellite embeddings at any scale. Ports the HPC-based
[Tessera](https://github.com/ucam-eo/tessera) embedding pipeline to a
cloud-native, distributed architecture that runs on any major cloud —
or on a laptop (slowly).

**This repository is currently set up for TESSERA v1.1.** The model architecture, the
band statistics, the temporal sampling and the published global store are all v1.1, and a
run you start today is a v1.1 run. Support for a **v2 Large student model is in flight**
([PR #98](https://github.com/dClimate/tessera-embeddings/pull/98)) — not merged, and with
no release date to quote.

## Contents

- [What this is](#what-this-is)
- [One area, or the whole world](#one-area-or-the-whole-world)
- [What this isn't](#what-this-isnt)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Running at scale](#running-at-scale)
- [Architecture](#architecture)
- [The global embeddings store](#the-global-embeddings-store)
- [Repo structure](#repo-structure)
- [Documentation](#documentation)
- [Using this for your own project](#using-this-for-your-own-project)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---
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

Output stores are self-describing via GeoZarr conventions: every embedding
store carries the [`proj:`](https://github.com/zarr-conventions/geo-proj) and
[`spatial:`](https://github.com/zarr-conventions/spatial) conventions for
CRS/affine metadata, plus the
[`geoemb:` geoembeddings convention](https://github.com/geo-embeddings/embeddings-zarr-convention)
for encoder-model provenance and quantization (built in
[`storage/conventions.py`](src/tessera_embeddings/storage/conventions.py)).

The domain code — the scientific transformations, the inference
engine, the Zarr I/O — is cloud-agnostic and orchestrator-agnostic.
It's plain Python over `xarray`, `dask`, `zarr`, `ray`, and `fsspec`.
Runs on one laptop or a thousand GPUs.

Alongside the library we ship **reference orchestration**: opinionated
Prefect flows and AWS provisioning helpers that demonstrate how we
run this at production scale. They are examples, not requirements.

## One area, or the whole world

There are two ways to run this, and they are more alike than they look.

**One area.** You supply an area of interest — a polygon, a set of Sentinel-2 tiles, or any mask
you can draw — and get embeddings for it over any twelve-month window you choose. This runs on one
machine; the [quickstart](docs/quickstart.md) does it on a laptop in a few minutes.

**The whole world.** The same pipeline, run as a campaign over the world's land between
**59.45°S and 83.65°N** (Antarctica is excluded by decision — see below), one UTM zone and one
calendar year at a time. The result is published as **global TESSERA v1.1** at
`s3://tessera-embeddings/v1.1/dclimate.icechunk/`, so in most cases you can read it rather than
compute anything — checking each zone's `years_complete` first, because unfilled cells read back
as fill values rather than as an error.

**The model and the ingest are the same code in both**, and assembly is shared up to the point of
writing. What differs is scale, how you say which ground you want, the store's conventions — most
importantly that the global store holds calendar years only, while a single area can use any
twelve-month window — and the campaign's different pixel-selection settings — two of them looser than the
library defaults and one stricter — which mean the same
area and year can give different results on the two paths.

**→ [`docs/single-vs-global.md`](docs/single-vs-global.md)** explains the differences that are
real, including how to supply your own mask (it does not have to be land) and why the global store
insists on calendar years.

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

# Full production stack — inference + Prefect orchestration + AWS:
pip install tessera_embeddings[inference,prefect,aws]

# GPU (CUDA 12.4, Python 3.12-3.13) — install torch first so pip keeps the CUDA wheel:
pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
pip install "tessera_embeddings[inference]"

# 3.12-3.13 is the supported range: it is what CI tests, and cu124 tops out at torch 2.6.0,
# which publishes cp39-cp313 and no cp314. Python 3.14 is untested rather than blocked --
# see docs/environment-setup.md if you intend to run it anyway.
```

For contributors:

```bash
git clone https://github.com/dClimate/tessera-embeddings
cd tessera-embeddings
uv sync --all-extras   # resolves uv.lock; all extras + dev tools
```

`uv.lock` at repo root is the single lock file. See
[`docs/environment-setup.md`](docs/environment-setup.md) for CUDA GPU
installs and platform guidance.

## Quickstart

```bash
git clone https://github.com/dClimate/tessera-embeddings
cd tessera-embeddings
uv sync --all-extras   # resolves uv.lock; all extras + dev tools
source .venv/bin/activate   # REQUIRED — and use plain `python`, never `uv run python`:
                            # uv run spawns a subprocess that kills Ray's GCS on macOS.
                            # docs/quickstart.md has the detail.

# End-to-end pipeline on the bundled Denver, CO quickstart ROI.
# Ingest → cloud mask → CPU inference → assemble. About three and a half minutes on a
# laptop, most of it ingest; CPU inference of the single chunk takes about a minute.
python -m tessera_embeddings.orchestration.runners.plain examples/quickstart/config.yaml

# Ingest only — shorter, because it stops before inference. For contributors
# iterating on ingest changes without waiting for CPU torch.
python -m tessera_embeddings.orchestration.runners.plain \
    examples/quickstart/config.yaml --skip-inference
```

The default mode runs the full chain, and that is the primary demo: inference and
assembly are coupled, so ingest-only is a convenience rather than a full-stack run.
Production inference always runs on GPU. See
[`docs/quickstart.md`](docs/quickstart.md) for prerequisites
(Earthdata Login credentials for OPERA; the model checkpoint is
pulled from HuggingFace automatically).

**The quickstart is also this project's decoupling test, which is why it runs end to end
on a CPU.** The runner behind it,
[`src/tessera_embeddings/orchestration/runners/plain.py`](src/tessera_embeddings/orchestration/runners/plain.py),
is an orchestrator-free sequencer: it calls the same domain functions as the Prefect
flows, without Prefect, with torch on CPU through Ray's local mode. Three things follow,
and together they are why this is the bar we hold ourselves to:

- If CPU torch works without modification, no GPU-specific coupling has leaked into the
  domain layer. That is the strongest architectural separation check available without
  deploying to several cloud targets.
- Assembly has nothing to assemble without embeddings, so the ingest-only path cannot
  stand in for it.
- `plain.py` is the worked reference for anyone porting to Airflow, Dagster or Flyte:
  everything it does is the non-Prefect wiring they would have to reproduce.

For CI, `plain.py --skip-inference` is the fast pull-request check. **The end-to-end run
on the quickstart ROI is not automated at all** — it is verified by running it by hand
([ADR 023](context_docs/decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md)).
Pull-request checks also apply the AST-based architecture rules described under
[Architecture](#architecture), which catch Prefect leaks at the import level without
running the pipeline.

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

Ingest cost scales with the area you actually keep, not your ROI's
bounding extent: mosaic loads and writes are restricted to the
chunk-aligned windows that intersect the ROI mask (measured
campaign-wide: ~4.3× less compute; a sparse island zone drops from
3,706 chunks per band-date to 4). This is unconditional and has no
flag — it serves a single sparse ROI and a global campaign zone alike.
See [the ingest README](src/tessera_embeddings/ingest/README.md#cropping-to-live-windows-unconditional).

### Profiling a run

Both compute stages ship a profiling harness — `te-watch-scheduler` and friends
for the Dask ingest scheduler, `te-observe-cluster` for the Ray GPU fleet. They
install as console scripts with the AWS extra:

```
pip install "tessera_embeddings[aws]"   # or: uv sync --extra aws
```

They are **AWS-specific** (CloudWatch, ECS, EC2, SSM) but are written to be a
template for other clouds, and PRs generalizing them are welcome. See
[`src/tessera_embeddings/profiling/README.md`](src/tessera_embeddings/profiling/README.md).

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

### Using these architecture checks in your own repo

The hard-rule checks ship as a reusable module, so a fork or an adapter can hold its own
code to the same contract:

```bash
# Run against any source tree
uv run python -m tessera_embeddings.architecture_tests \
    --source path/to/your_package/ \
    --allowlist your-arch-allowlist.toml
```

The allowlist file (TOML) documents intentional deviations — for example, "Prefect
imports in my own `orchestration/prefect/` are expected". See
[`src/tessera_embeddings/architecture_tests/`](src/tessera_embeddings/architecture_tests/)
for the rule definitions, the allowlist schema, and worked examples.

## The global embeddings store

Alongside per-area stores, the library ships the storage layout and the write path for a
**global 10 m store**: one Icechunk repository holding 120 Zarr groups — one per UTM zone,
named by its common name (`01N`–`60N`, `01S`–`60S`) — each pre-allocated with a 2017–2025
annual time axis and filled one (zone, year) at a time. The published result is **global
TESSERA v1.1** at `s3://tessera-embeddings/v1.1/dclimate.icechunk/`, and it is readable
without an AWS account.

**What "global" means here: land between 59.45°S and 83.65°N**, the extent of the coverage
registry the campaign is built from. **Antarctica is excluded by decision**, not omitted by
accident — the registry offers no Antarctic land cell, and the UTM grid could not place one
if it did (UTM's usable range stops at 80°S). See
[ADR-017](context_docs/decisions/017-no-antarctic-coverage.md).

```
one Icechunk repo (BucketPaths.global_store())
├── 01N/    embeddings (time, northing, easting, band)  int8
│           (1, 256, 256, 128) inner chunks in (1, 2048, 2048, 128) shards
│           scales / obs counts sharded on the same 2048² spatial grid
├── 02N/    … one group per UTM zone, seeded metadata-only up front …
⋮
└── 60S/    attrs: crs, zone_scheme, years_complete, runs, conventions
```

`run_global_campaign` drives the fill year-serial with bounded zone parallelism and
triggers its own ingestion (ADR-011); `orchestration/runners/zone_fill.py` is the
end-to-end (zone, year) callable, and `storage/campaign.py` holds the per-cell tags,
snapshot expiry and the zone×year progress reader. The
[Prefect flow README](src/tessera_embeddings/orchestration/prefect/README.md) documents
the dispatch chain, the cancellation sweep and the per-branch deployment routing.

**→ [`docs/single-vs-global.md`](docs/single-vs-global.md#the-published-global-dataset)**
is the reader's and writer's guide to this store: how to open a zone group, what its time
axis promises, what the per-pixel observation counts record, how shards and manifests are
laid out, why the chunk sizes are what they are — get that wrong and a scheduler drowns in
tasks or a worker runs out of memory — and, before you trust any of it, how to check that
the cell you want was actually filled. The architecture is settled in
[ADR-008](context_docs/decisions/008-global-store-architecture.md); the operational plan
for running a campaign is
[`context_docs/campaign/campaign-plan.md`](context_docs/campaign/campaign-plan.md).

## Repo structure

```
src/tessera_embeddings/
  config/                pydantic config models — ingest, inference, assembly,
                         store layout, paths, time windows
  ingest/                STAC search and ingestion (Sentinel-2, Sentinel-1/OPERA),
                         ROI rasterization, the campaign land mask, auth
  inference/             GPU inference: Ray actors, work-stealing scheduler,
                         per-tile loading, quantization, staging and assembly
  storage/               Zarr and Icechunk stores, manifests, zone grids,
                         empty-store seeding, shard writer, campaign tags
  orchestration/
    concurrency.py       sliding_window_submit — shared by flows and runners
    prefect/             Prefect — 100% quarantined here
      flows/             @flow-decorated orchestration (Layer 3)
      tasks/             thin @task wrappers (Layer 2)
    runners/             non-Prefect entry points: plain.py (one area),
                         zone_fill.py and sequential_fill.py (one campaign cell)
  providers/             concrete cloud-provisioning glue
    aws/                 ray.py, dask.py, credentials.py, fleet_mix.py,
                         cluster.yaml.template, gotchas.md
    local/               ray.py, dask.py — demo and tests
  profiling/             AWS-specific harnesses for watching a live run
  architecture_tests/    reusable layer-rule checker (CLI + Python API)

docs/                    how to run, configure and port it  (docs/README.md)
context_docs/            why it is shaped this way, and what was measured
examples/quickstart/     the bundled Denver, CO area of interest and its config
scripts/                 scoping and analysis scripts, not part of the library
tests/                   unit, architecture, integration, parity and GPU tiers
```

## Documentation

- [`docs/README.md`](docs/README.md) — what is in `docs/` and how it
  differs from `context_docs/`. Start here if you are not sure which
  you want.
- [`docs/single-vs-global.md`](docs/single-vs-global.md) — running for
  one area versus the global campaign: what is shared, what differs,
  how to supply your own mask, why the global store takes calendar
  years only, and how to read the published store — its layout, its
  chunk sizes, and how to check a cell was filled.
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

## Using this for your own project

The domain layer is a library, not a framework: there is nothing to inherit and nothing
to register. You import the functions you want and call them. Four things are worth
knowing before you build on it.

**Depend only on the documented public API.** This library follows semver for the surface
listed in [`docs/public-api.md`](docs/public-api.md). Anything outside it —
underscore-prefixed names, and everything under
`tessera_embeddings.orchestration.prefect.*` — is implementation detail and may change
between minor releases.

**Keep your own orchestration and cloud glue separate.** The Prefect flows and the AWS
provider are reference implementations you are meant to replace, not extend.
[`docs/orchestrator-swap.md`](docs/orchestrator-swap.md) walks through running without
Prefect, and [`docs/providers/adding-your-own.md`](docs/providers/adding-your-own.md)
through targeting another cloud.

**Apply the same layering rules to your own code.** The hard rules in
[Architecture](#architecture) ship as a reusable checker you can point at any source
tree, with a TOML allowlist for the deviations you intend — see
[Using these architecture checks in your own repo](#using-these-architecture-checks-in-your-own-repo).

**Wire a smoke test against your own fork.** `.github/workflows/downstream-smoke.yml` is
a template for running a downstream project's test suite against a pull request in this
repository, so a breaking change is caught at the point of change rather than in
production. It is shipped disabled (`workflow_dispatch` only), and it takes a read-only
token for the downstream repository plus the test command you want run. Keep it
**informational rather than blocking**: a private downstream should not hold a veto over
a public release.

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
