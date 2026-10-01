# Graviton (ARM64) for the ingest fleet: scoping and recommendation

**Status:** proposed, 2026-10-01. Nothing here is implemented. The infrastructure it changes lives in
yield-embeddings; this repo needs no code change.

## Recommendation

- **Move the Dask workers to ARM64 Fargate before the next campaign or annual update.** Leave the
  scheduler and the flow runners on x86.
- **It saves about 19% of ingest's Fargate cost.** That is ARM's 20% lower price, not extra speed,
  so it holds provided ARM workers keep pace. On a campaign the size of the last one (a $187,441
  Fargate line) that is about **$32,000–35,000**, or about **$3,600–3,900 per global year**. The yield account's own Fargate use is dev-scale, so the saving
  there is negligible.
- **The change is small:** two Dockerfile lines, a `platforms` input on the image build, and one
  property on one task definition. Then one paired dev test costing about $15–25.
- **The risk is bounded.** All dependencies, both base images and the ARM64 image build itself are
  verified, and a real Sentinel-2 warp and the Sentinel-1 dB conversion come out bit-identical
  across architectures. Two unknowns remain, and the dev test measures both: how fast an ARM worker
  is when the fleet is saturated, and whether production x86 hosts round differently (AVX-512).
- **Run one architecture per campaign.** The ingest code identity hashes source only, so it cannot
  see an architecture change. Switch only when no mosaic is half-built.
- **Fargate Spot is the bigger lever** (up to 70% off, against Graviton's 20%), and it combines
  with ARM. It has its own failure mode, so it needs separate scoping (see the last section).

## What moves

One campaign cell runs three ingest legs (S2 and both S1 orbits). Each leg is a flow runner, a Dask
scheduler and a worker fleet, all on Fargate, all from the ingestion image:

```
  one cell, S2 width 60                          today    proposed
  ├── parent flow runner        4 vCPU / 16 GB   x86      x86
  ├── S2 leg    flow runner     4 vCPU / 16 GB   x86      x86
  │             scheduler       4 vCPU /  8 GB   x86      x86
  │             workers × 60    4 vCPU / 24 GB   x86      ARM64   <- 94% of the
  ├── S1 asc    runner + scheduler               x86      x86        cell's cost
  │             workers × 13                     x86      ARM64   <-
  └── S1 desc   runner + scheduler               x86      x86
                workers × 13                     x86      ARM64   <-
```

Workers are 94% of a cell's hourly cost at 60 S2 workers and 95% at 80. That is the case for moving
only them: the runners and schedulers are the other 6%, and not worth extra risk.

**Why the scheduler stays on x86.** It is single-threaded and has been the scaling wall
([campaign-ingest-measurements.md §6 and §12.3](campaign-ingest-measurements.md)). Per thread,
Graviton2 generally trails current x86. Moving the schedulers would save about 0.5% of the cell,
and a slower scheduler slows the whole fleet.

## What it saves

AWS Pricing API, us-west-2, on-demand:

| | x86 | ARM64 | |
|---|---|---|---|
| per vCPU-hour | $0.04048 | $0.03238 | −20.0% |
| per GB-hour | $0.004445 | $0.00356 | −19.9% |
| one worker (4 vCPU, 24 GB) | $0.2686/h | $0.2150/h | |

The last campaign's container line was 2,875,455 vCPU-hours and 15,982,667 GB-hours, $187,441
([campaign-cost-model.md §12](../campaign/campaign-cost-model.md)). Billing does not separate
ingest from the inference flow runners and assembly that share that line. Assembly is about $1,300
and the runners are a small share, so ingest is most of it. At equal worker-hours, ARM workers save 18.8% of
ingest spend, which is **$32,000–35,000** depending on ingest's share. That campaign covered nine
years, so it is **$3,600–3,900 per global year**.

Workers are billed for their lifetime, not for CPU used, so the saving depends on whether ARM
lengthens a run. With `r` = ARM worker-hours ÷ x86 worker-hours for the same work:

| r | 0.85 | 1.00 | 1.10 | 1.25 |
|---|---|---|---|---|
| saving on worker cost | 32% | 20% | 12% | 0, break-even |

One second-order cost bounds `r` more tightly than break-even does. Ingest feeds the GPU fleet,
which was $537,000 of the campaign, and a slower ingest can leave cards idle. The campaign's Fargate
peak used 73% of the 25,000 vCPU quota (one quota covers both architectures), so a slowdown of
about 10% can be absorbed by running wider. Much more than that cannot.

## Will ARM workers be faster or slower?

The saving above comes from the price alone. Speed moves it up or down, and only where it changes
how long a worker runs. Probably not by much either way, but only the dev test can say.

- **AWS's headline is mostly the price cut.** "Up to 40% improved price/performance at 20% lower
  cost" means at best about 12% more work per vCPU (1.4 × 0.8 = 1.12). A Graviton vCPU is a whole
  core, where an x86 vCPU is one hyperthread, which helps multi-threaded work like the workers'
  reads and warps. Per thread, x86 still leads, which is what matters to the single-threaded
  scheduler. Fargate offers no choice of generation: it launched on Graviton2 (Neoverse N1), and the
  faster Graviton3 and 4 are EC2 instance choices.
- **Our workers are not CPU-bound at dev scale.** Across the Frisky dev runs on 2026-10-01 (a tiny
  ROI, 15SWC and Iowa, up to 60 S2 workers, both engines), Container Insights shows workers using
  **20–27% of reserved CPU** over the runs, and about 60% in the busiest minute. A worker spends
  most of its time waiting on S3 reads and dispatch. That cuts both ways: a slower core costs little,
  and a faster one saves little. No figure from the last campaign was read: those metrics are in
  the global-tessera account.
- **The fleet-saturated regime is the real question.** Of a dense zone's task work, 72% is reading
  and resampling source imagery, and one date's work oversubscribes the fleet
  ([§1](campaign-ingest-measurements.md)). In that regime, worker speed sets the pace. The dev test
  builds that regime on purpose, by running a narrow fleet.
- **One serial step does land on a worker.** The ingest body (catalogue query, graph build, date
  loop) runs inside one Prefect task on one Dask worker ([§12.5](campaign-ingest-measurements.md)).
  Its per-date graph build is about 3.7 s, single-threaded, and the look-ahead in
  [`ingest/_pipeline.py`](../../src/tessera_embeddings/ingest/_pipeline.py) runs it while the
  previous date writes. A slower core lengthens it, but mostly out of sight.

## Will the mosaics change?

**The native libraries are the same on both architectures.** rasterio 1.5.0's x86_64 and aarch64
wheels both bundle GDAL 3.12.1, PROJ 9.7.1, GEOS 3.14.1, libtiff 6.2.0 and zstd 1.5.7.

**Measured: identical output on a real scene.** Both architectures' builds of the ingestion image
warped the same windows of `S2B_15SWC_20240724_0_L2A` to EPSG:5070 at 10 m. The windows were B04 at
10 m and B11 at 20 m, about 9.4 M valid output pixels each, using `rasterio.warp.reproject` with its
default approximate transformer. That is the GDAL warp the loader makes. Both architectures then
ran the S1 conversion, `amplitude_to_db`'s exact formula and constants, on 20 M seeded float32
amplitudes. **Zero pixels differed** in the bilinear warps, the nearest warps, the uint16 dB values,
or the raw float32 `log10` bits.

**One gap: the x86 side ran without AVX-512.** It ran under Docker's emulation on Apple Silicon,
which offers AVX2 but not AVX-512. That matters because numpy's x86 wheel ships Intel SVML AVX-512
kernels for `log10` (the ARM wheel has none), and numpy uses them only on AVX-512 CPUs. AWS does
not say which x86 CPUs run Fargate. If they have AVX-512, production S1 values may differ from ARM
by a rounding step. The dev test's store comparison settles it.

**The code identity cannot see any of this.**
[`config/code_identity.py`](../../src/tessera_embeddings/config/code_identity.py) hashes source,
and its docstring says it "cannot see dependency drift". An ARM run would therefore append to a
mosaic an x86 run began, and nothing would record it. Mosaics are deleted after publication, so the
only exposure is at the switch. **The rule: one architecture per campaign, switched when no mosaic
is mid-build.** This parallels the fill pinning one AMI for a whole campaign.

**If the test does find differences,** the precedent is the 2048-vs-4096 chunk comparison on the
Frisky experiment branch. There, differences within GDAL's warp approximation, with no gate decision
changed, were accepted. [ADR 012](../decisions/012-validated-equivalence-for-inference-outputs.md)
is the same policy for embeddings.

## Feasibility, item by item

| Item | Status | Evidence |
|---|---|---|
| Python dependencies | **ready.** Of the image's 167 locked packages, 137 are pure Python and all 29 compiled ones ship manylinux aarch64 wheels. The remaining one, `pywin32`, is Windows-only. | yield-embeddings `uv.lock` |
| Base images | **ready.** `ghcr.io/osgeo/gdal:ubuntu-small-3.12.1` and `prefecthq/prefect-aws:0.7.7-python3.12` both publish `linux/arm64` | registry manifests |
| Image build | **ready.** Builds natively for `linux/arm64` with only the two `--platform=linux/amd64` pins removed; every ingest module imports; `dask`, `prefect`, `aws`, `gdalinfo` and the `te-*` CLIs run | local build, 2026-10-01 |
| Fargate | **ready.** ARM64 needs platform 1.4.0 or later and supports every size we use. Fargate's vCPU quota covers both architectures | AWS ECS docs |
| Fargate Spot on ARM64 | **available** since 2024-09-06 | AWS announcement |
| CI runners | **available.** `ubuntu-24.04-arm` standard runners in private repositories since 2026-01-29, 2 vCPU, inside plan minutes. QEMU on `ubuntu-latest` also works | GitHub changelog |
| ECR lifecycle | **safe.** The repos expire untagged images after 7 days, and a multi-arch image's per-architecture children are untagged. But ECR never expires an image a manifest list references | ECR docs |
| Branch task definitions | **safe.** `register_branch_task_defs.py` copies every field but `DESCRIBE_ONLY_FIELDS`, which does not include `runtimePlatform`, so branch clones keep the architecture | yield-embeddings `scripts/_branch_infra.py` |
| Dask provider | **no change.** With pinned task definitions, as every yield deployment has, dask-cloudprovider registers nothing, so the CDK definition alone sets the architecture. Its `cpu_architecture` argument applies only when it registers its own | dask-cloudprovider 2025.9.0 `aws/ecs.py` |
| Frisky, if adopted | **ready.** 0.7.2 ships a manylinux aarch64 wheel | experiment branch `uv.lock` |

## The change

All in yield-embeddings.

1. **`infra/docker/ingestion.Dockerfile`:** delete `--platform=linux/amd64` from both `FROM`
   lines, so buildx builds each stage for the target platform.
2. **`.github/workflows/_build-image.yml`:** add a `platforms` input defaulting to `linux/amd64`,
   and pass `linux/amd64,linux/arm64` from the two ingestion callers only. The smallest form adds
   `docker/setup-qemu-action` and passes `platforms` to `build-push-action`; the dependency layer
   is cached, so emulation costs time only when the lock changes. If that slows `dev-deploy` too
   much, use Docker's documented pattern instead: one native job per architecture
   (`ubuntu-24.04-arm` for arm64), then `docker buildx imagetools create` to join them. The
   inference image stays amd64-only.
3. **`infra/aws/stacks/consumer_stack.py`, `_dask_task_def`:** give the worker an ARM64 runtime
   platform:

   ```python
   runtime_platform=ecs.RuntimePlatform(
       cpu_architecture=ecs.CpuArchitecture.ARM64 if kind == "worker" else ecs.CpuArchitecture.X86_64,
       operating_system_family=ecs.OperatingSystemFamily.LINUX,
   ),
   ```

   That docstring's three-level re-registration applies: a branch run picks this up only after its
   clone and its deployment are re-registered.

**Not changed:** the runner families, coarsen, `merge_kind`, the EC2 `merge_kind_ec2` family on
c8in, the inference image and the Ray AMI. A multi-arch manifest lets each task definition pull its
own architecture, so everything else keeps pulling amd64. Rolling back is a one-property revert.

**Effort:** about half a day for the image build, then about a day for the CDK change and the dev
test.

## The dev test

**Where and when:** the yield dev account, after the Frisky plan's runs finish, so the two
experiments do not share catalogue and S3 conditions. Use two yield-embeddings `dev/<slug>`
branches (say `dev/graviton-x86` and `dev/graviton-arm`) that differ only in the worker
architecture. Run each rung's two arms at the same time, S2 and S1 together, with a fresh store for
every run.

| Rung | ROI and window | Width | What it measures |
|---|---|---|---|
| 1. production width | `iowa_epsg5070`, July 2024 | S2 60, S1 13 | cost and time per date as the campaign runs; at least 14 S2 dates per arm, because per-date variance is 19% |
| 2. fleet-bound | the same | S2 15 | worker throughput. A date's ~1,180-task read width oversubscribes 60 task slots about 20×, so worker speed sets the pace, as on a dense zone |
| 3. optional | the same as rung 1 | S2 60 | scheduler on ARM as well, to decide whether the last 6% can move |

**Record per arm:** per-date build, gate and write times (`te-ingest-log-queries --query
date_stage_timings`); worker-hours and cost per date, from billed task lifetimes; Container
Insights `CpuUtilized` / `CpuReserved` per task family; worker exits and restarts.

**Correctness:** read every chunk of every array from both stores and compare exactly, with NaN
positions and time coordinates included. Date lists must match.

**Adopt when:**

- the stores are identical, or any difference is a one-count rounding with no pixel switching
  between valid and nodata; anything larger stops the switch until it is explained;
- ARM cost per date is at most 0.9× x86 on both rungs;
- on rung 2, ARM per-date time is no more than about 10% slower.

**Cost:** about $15–25 for both rungs and both arms, scaled from the Frisky plan's measured Iowa
runs at about $0.27 per worker-hour.

**Cheaper evidence first, if permitted:** one read-only Container Insights query of worker CPU
utilisation over the last campaign, in the global-tessera production account. It would show
whether workers at campaign scale are as far from CPU-bound as they are at dev scale.

## Out of scope

- **Fargate Spot.** It is a bigger saving than ARM and stacks with it. dask-cloudprovider's
  `fargate_spot=True` puts workers on `FARGATE_SPOT` and keeps the scheduler on `FARGATE`. The
  hazard is particular to us: the whole ingest body runs on one Dask worker, so reclaiming that
  worker restarts the run's driver, and how the run resumes needs testing.
- **The other ingestion-image families.** The ingestion runner, coarsen and Fargate `merge_kind`
  become a one-line change each once the image is multi-arch. `merge_kind` is CPU-bound and worth
  its own measurement; the EC2 `merge_kind_ec2` family would need a Graviton instance type instead.
- **Inference.** The GPU workers are x86 CUDA instances, and the inference image follows them.

## Sources

- [Amazon ECS: task definitions for 64-bit ARM workloads](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-arm64.html)
- [Amazon ECS: Fargate task sizes and architectures](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-tasks-services.html)
- [Amazon ECS now supports Graviton-based Spot compute with Fargate (2024-09-06)](https://aws.amazon.com/about-aws/whats-new/2024/09/amazon-ecs-graviton-based-spot-compute-fargate/)
- [AWS Fargate FAQs](https://aws.amazon.com/fargate/faqs/), for the price/performance claim
- [Announcing Graviton2 support for Fargate (2021)](https://aws.amazon.com/about-aws/whats-new/2021/11/aws-fargate-amazon-ecs-aws-graviton2-processors)
- [Amazon ECR lifecycle policy evaluation rules](https://docs.aws.amazon.com/AmazonECR/latest/userguide/LifecyclePolicies.html)
- [arm64 standard runners in private repositories (GitHub, 2026-01-29)](https://github.blog/changelog/2026-01-29-arm64-standard-runners-are-now-available-in-private-repositories/)
- [Comparing Intel, AMD and Graviton2 (Cribl)](https://cribl.io/blog/comparing-intel-amd-and-graviton2/), for per-thread performance
