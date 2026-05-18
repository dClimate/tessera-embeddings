# Open-Sourcing Conceptual Background

Reference doc for the effort to extract this repo into (1) an open-source embeddings generation library and (2) a separate closed-source library for our preferred infrastructure + orchestration. The work itself happens in a different repo — this doc is purely conceptual framing and vocabulary to reason about the split.

---

## 1. The core framing: ports and adapters

The canonical architecture for this kind of decomposition is **Hexagonal Architecture** (Alistair Cockburn, 2005), also called **Ports and Adapters**. Domain logic sits in the middle, ignorant of the outside world. It exposes *ports* — interfaces describing what it needs ("read bytes from somewhere", "run this task on some compute"). *Adapters* implement those ports against specific technologies (S3, GCS, local disk, Dask, Ray, ECS, k8s).

The related principle is **Dependency Inversion**: the domain defines the abstractions it depends on; infrastructure code implements them. Embeddings code should say "give me a `StorageBackend` that implements `read_zarr` / `write_zarr`" rather than calling `boto3` directly.

Related slogan: **"separation of mechanism and policy"** (Unix philosophy). Your code decides *what* to compute; the substrate decides *how and where*.

## 2. Vocabulary for this specific problem

We are juggling four concerns that get conflated in typical AWS/Prefect codebases:

| Concern | What it answers | Current answer in this repo |
|---|---|---|
| **Domain logic** | What scientific transformation happens | STAC load → transforms → cloud mask → Zarr write |
| **Execution substrate** | What actually runs a Python callable | Dask / Ray / local Python |
| **Orchestration layer** | What sequences, retries, schedules work | Prefect flows and tasks |
| **Infrastructure provisioning** | What conjures machines into existence | CDK + Fargate + SSM + Packer AMI |

These are four separable axes. External users will want to mix-and-match: on-prem Slurm cluster with Dask-on-Slurm + Airflow, no cloud. GCP user wants Dataflow or Vertex. The four-layer model lets us reason about which layers to abstract and which to hard-depend on.

## 3. The spectrum of approaches

A spectrum from pure library to full framework:

**Pole A — pure library** ("transformers" model). Export composable functions and classes. No assumptions about how they're run. Users write their own orchestration. Maximum flexibility, minimum batteries. Risks: users reinvent the wheel; our parallelization-specific innovations become non-obvious; no "docker compose up" demo experience.

**Pole B — framework with pluggable adapters** ("Kedro/Lightning" model). Export domain logic + a thin runtime that takes pluggable implementations of infrastructure ports. Ship reference adapters (local filesystem, local Dask, maybe a vanilla k8s runner). Closed-source ships the AWS/Prefect-optimized adapters. Good batteries, still swappable.

**Pole C — opinionated framework** ("Dagster/Prefect/Airflow" model). Impose a workflow DSL and lifecycle. Pluggable at well-defined extension points but the framework owns the execution model. Most batteries-included but highest lock-in for users.

Classic **framework vs library** distinction: a library is called by user code; a framework calls user code (Hollywood principle, "inversion of control"). More framework = more opinions forced on users, more power for our UX.

Working hypothesis: **Pole B**, with Prefect as a *recommended but not required* orchestration adapter rather than ripped out entirely.

## 4. Where the real seams are

A useful exercise per file in `src/`: *does this file know it's running on AWS? Does it know about Prefect?* Most domain code (STAC ingestion, transforms, Zarr writing) likely doesn't — we're already using cloud-agnostic tools (`odc-stac`, `xarray`, `dask`, `zarr`, `icechunk`, fsspec-friendly paths). AWS/Prefect coupling is concentrated at specific boundaries:

- **Storage URIs**: `s3://...` hardcoded. Seam: a storage protocol abstraction (fsspec already provides this for free).
- **Compute provisioning**: CDK stack + Fargate work pool. Seam: "bring your own cluster" — the library assumes a Dask/Ray client is handed to it, not created by it.
- **Config resolution**: SSM parameters, IAM. Seam: plain config objects (pydantic-settings, Hydra) sourced from anywhere.
- **Prefect task boundaries**: mixed with domain logic. Seam: the painful one — see §6.
- **AMI/image build**: Packer + ECR. Seam: ship a `Dockerfile` in OSS; leave the registry/baking pipeline closed-source.
- **Ray cluster topology**: hard-coded to AWS. Seam: declarative spec (YAML) + documented "gotchas", no IaC — exactly the user's instinct.

Good news: most domain code is probably already cloud-agnostic in practice. Lock-in is concentrated in a thin shell.

## 5. The Prefect question specifically

Prefect is the hardest decoupling question because flow-decorated functions often *contain* domain logic rather than *wrapping* it. The canonical refactor pattern:

> Domain logic lives in plain functions and classes that know nothing about Prefect. Prefect flows are thin orchestration wrappers that call those functions and handle retries/parallelism/state.

If flows are mostly `@task`-decorated helpers with heavy logic inside, decoupling requires extracting that logic into plain modules. If flows are already thin sequencers of pure functions, we're most of the way there — anyone can swap Prefect for Airflow/Dagster/plain Python by rewriting the thin layer.

**Pragmatic middle road**: keep Prefect in the OSS library as the recommended orchestrator, but structure the code so Prefect is a thin decorator layer over substrate-free domain functions. Gives users who want to swap orchestrators a clear path (rewrite the flow files, everything else is stock Python) without forcing us to build a fully abstract orchestration interface — a known tarpit. Airflow, Dagster, Flyte, and Argo all tried to build that and ended up defining their own DSLs anyway.

## 6. Projects worth studying

Ordered by directness of mapping to our problem:

- **Kedro** (QuantumBlack/McKinsey). Data-pipeline framework that nails the decomposition. Pipelines are pure Python functions; I/O is abstracted through a `DataCatalog` with pluggable datasets (`ParquetDataSet`, `SparkDataSet`, etc.); execution uses swappable runners (`SequentialRunner`, `ParallelRunner`, `ThreadRunner`). Almost exactly the pattern we want.
- **PyTorch Lightning + Fabric**. Clean separation of model code (user) from training engine (framework). `Strategy` and `Accelerator` abstractions are how to decouple distributed execution cleanly.
- **Apache Beam**. "Write once, run anywhere" via pluggable Runners (Dataflow, Spark, Flink, Direct). Solved the exact "same pipeline, many execution substrates" problem. Programming model is heavyweight but the Runner abstraction is the reference.
- **Dagster**. Distinction between **assets** (what) and **resources** (how/where). Resources are pluggable dependency-injected objects — S3 client, DB connection, Spark session. Exemplary IoC for data pipelines. Good reference for declarative asset modeling.
- **Ray's cluster launcher**. The "Ray as a spec sheet, not IaC" instinct maps well to Ray's `cluster.yaml` format, which abstracts AWS/GCP/Azure/local behind a declarative config. `ray up` as a UX precedent.
- **fsspec**. Canonical Python storage abstraction. `xarray`, `zarr`, `pandas`, `dask` all speak it. Likely most of our storage decoupling is "replace hardcoded `s3://` with fsspec protocol URLs and be done with it."
- **Flyte**. Typed task interfaces, pluggable execution backends, strong separation of workflow definition from runtime. A more modern take than Airflow, closer in spirit to our goals.
- **OpenTelemetry**. Not a data pipeline tool, but the canonical example of "standard API, pluggable backend" done well — worth a brief look for how the API/SDK split is structured.

**Cautionary tale: LangChain.** Tried to abstract everything; ended up with abstraction sprawl, leaky interfaces, and constant churn. Lesson: abstract only what users actually swap, not every possible axis of variation.

## 7. One strategic question worth answering early

**Library or platform?** A library gives pieces; a platform gives an opinionated end-to-end path. Hugging Face started as the former and became the latter. Prefect/Dagster are platforms. Kedro is in between.

This repo is a platform today — an opinionated, batteries-included pipeline. The cleanest OSS split:

- **OSS: library + reference implementation.** Domain logic, reference Prefect flows, local-filesystem adapters, Dockerfile, Ray cluster YAML templates, comprehensive docs of "gotchas" per cloud. Someone can clone it, run it on one laptop or self-hosted Dask, get embeddings.
- **Closed-source: production platform.** Our CDK stack, Fargate work pools, SSM/IAM config, AMI baking pipeline, monitoring. Imports the OSS library and composes it into our production stack.

This framing makes every decision cleaner: "would a third party on a different cloud need this?" If yes, OSS. If no, closed.
