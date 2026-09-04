# TESSERA Inference

[![Lint](https://github.com/dClimate/tessera-embeddings/actions/workflows/lint.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/lint.yml)
[![Unit tests](https://github.com/dClimate/tessera-embeddings/actions/workflows/unit.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/unit.yml)
[![Architecture](https://github.com/dClimate/tessera-embeddings/actions/workflows/architecture.yml/badge.svg)](https://github.com/dClimate/tessera-embeddings/actions/workflows/architecture.yml)

Generate per-pixel (10m^2) TESSERA satellite embeddings at any scale. Ports the HPC-based
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

Output stores are self-describing via GeoZarr conventions: every embedding
store carries the [`proj:`](https://github.com/zarr-conventions/geo-proj) and
[`spatial:`](https://github.com/zarr-conventions/spatial) conventions for
CRS/affine metadata, plus the
[`geoemb:` geoembeddings convention](https://github.com/geo-embeddings/embeddings-zarr-convention)
for encoder-model provenance and quantization (built in
[`inference/conventions.py`](src/tessera_embeddings/inference/conventions.py)).

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

# Full production stack — inference + Prefect orchestration + AWS:
pip install tessera_embeddings[inference,prefect,aws]

# GPU (CUDA 12.1) — install torch first so pip keeps the CUDA wheel:
pip install "torch==2.6.0+cu121" --index-url https://download.pytorch.org/whl/cu121
pip install "tessera_embeddings[inference]"
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

# End-to-end pipeline on the bundled Denver, CO quickstart ROI.
# Ingest → cloud mask → CPU inference → assemble. ~3-4 minutes on a laptop,
# most of it ingest; CPU inference of the single chunk takes about a minute.
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
(Earthdata Login credentials for OPERA; the model checkpoint is
pulled from HuggingFace automatically).

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

### Why chunk size dominates everything

A subtle reality of distributed array workloads: **the task graph
your scheduler has to plan grows quadratically with how finely you
chunk the data.** Chunks too small means the scheduler spends more
time managing tasks than tasks spend doing work. Chunks too large
means workers can't fit a chunk in memory.

```
ROI: 20 km × 20 km, S2 reflectance, 10 m resolution, 12 dates:

chunks=200×200 (10× too small)        chunks=2048×2048 (the right size)
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

Storage and read granularity are tuned separately. Ingest writes
`INGEST_CHUNK_SIZE = 4096` storage chunks to keep the satellite-ingest
Dask graph small (¼ the spatial tasks), while inference reads a smaller
sub-tile out of them — small enough to keep peak GPU-node RAM in check.
Zarr's `oindex` reads a sub-tile out of a 4096 chunk with no alignment
requirement, so the two sizes are independent.

The read tile divides the OUTPUT chunking, and both paths use the same
one: `INFERENCE_CHUNK_SIZE = 2048`, so one inference tile is exactly one
2048-px shard (ADR-008 D3). The global campaign also passes it explicitly,
since that path requires the identity rather than merely matching it. Go
smaller on the ingest chunk and the satellite-ingest Dask scheduler drowns
in tasks; go larger on the read tile and you OOM on a g6e.xlarge. If you
change either, profile.

The powers of two are not cosmetic — they align every stage of the
pipeline on one grid, so no stage rechunks its input:

```
ingest chunk    4096 px  = 2×2 inference tiles
inference tile  2048 px  = 1 output shard
shard           2048 px  = 8×8 inner chunks
inner chunk      256 px  = the unit downstream readers decode

one ingest store chunk (4096²) — what one satellite read/write touches
┌─ inference tile (2048²) ─┬─ inference tile (2048²) ─┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░ │                          │
│ ░ 8×8 grid of 256²     ░ │   each tile is read out  │
│ ░ inner chunks — the   ░ │   of the ingest chunk by │
│ ░ same grid the output ░ │   one GPU actor, staged  │
│ ░ shard will store     ░ │   as one file, and lands │
│ ░░░░░░░░░░░░░░░░░░░░░░░░ │   as ONE shard object    │
├──────────────────────────┼──────────────────────────┤
│                          │                          │
│    inference tile        │    inference tile        │
│                          │                          │
└──────────────────────────┴──────────────────────────┘
```

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

## The global embeddings store

Beyond single-ROI stores, the library ships the storage layout and write
path for a **global 10 m campaign**: one Icechunk repo holding 120 Zarr
groups — one per UTM zone, named by its **common name** (`01N`–`60N`,
`01S`–`60S`; the EPSG:326xx/327xx code is retained only as the CRS) — each
pre-allocated with a 2017–2025 annual time axis and filled one
(zone, year) at a time. The architecture is settled in
[ADR-008](context_docs/decisions/008-global-store-architecture.md), and the
operational plan for running the campaign is
[`context_docs/campaign/campaign-plan.md`](context_docs/campaign/campaign-plan.md).

**What "global" means here: land between 59.45°S and 83.65°N**, which is the
extent of the coverage registry the campaign is built from.
**Antarctica is excluded by decision**, not omitted by accident — the registry
offers no Antarctic land cell, and the UTM grid could not place one if it did
(UTM's usable range stops at 80°S). Reaching further south would need a
different projection, a different coverage source and a different zone scheme;
see [ADR-017](context_docs/decisions/017-no-antarctic-coverage.md).

```
one Icechunk repo (BucketPaths.global_store())
├── 01N/    embeddings (time, northing, easting, band)  int8
│           (1, 256, 256, 128) inner chunks in (1, 2048, 2048, 128) shards
│           scales / obs counts sharded on the same 2048² spatial grid
├── 02N/    … one group per UTM zone, seeded metadata-only up front …
⋮
└── 60S/    attrs: crs, zone_scheme, years_complete, runs, conventions
```

Zone groups, mosaic paths, and tags all use the UTM **common name**
(`canonicalize_zone` parses `"33n"`/`" 7s "` → `"33N"`/`"07S"`) — a
deliberate deviation from the geoembeddings `utm_zones` spec, whose
`utm{NN}` group name can't express the hemisphere.

Four write paths, all committing atomically (`storage/zarr_store.py` has
the first three; `inference/assembly.py` + `storage/shard_writer.py` the
fourth):

1. **create** — `write_dataset` on a fresh store; it adopts a repo an
   interrupted attempt left behind rather than failing forever on a dirty
   prefix;
2. **append** — extend the time axis of an existing store;
3. **region overwrite** — rewrite a temporal/spatial slice in place;
4. **shard-assemble** — staged inference tiles written as whole, lean
   2048-px shards into a pre-allocated zone group, one fork/merge commit
   per (zone, year). Commits are ungated: they contend on the repo's single
   branch tip, but that costs seconds and never a conflict — see
   `context_docs/storage/writing-to-the-global-store.md`.

### Anatomy of a shard: what a write emits, what a read fetches

A shard is one S3 object wrapping an 8×8 grid of independently
compressed **inner chunks**, plus a tiny index mapping each inner chunk
to its byte range inside the object:

```
zone group "32601" ▸ embeddings ▸ year 2025 ▸ one shard
┌─ shard object (2048² px × 128 bands ≈ 0.5 GB max on S3) ────────────┐
│   8×8 inner chunks, 256² px × 128 bands (~8.4 MB int8+zstd each)    │
│   ┌────┬────┬────┬────┬────┬────┬────┬────┐                         │
│   │▓▓▓▓│▓▓▓▓│▓▓▓▓│    │    │▓▓▓▓│▓▓▓▓│▓▓▓▓│   ▓ = data: encoded    │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤       bytes + an index  │
│   │▓▓▓▓│▓▓▓▓│    │    │    │    │▓▓▓▓│▓▓▓▓│       entry             │
│   ├────┼────┼────┼────┼────┼────┼────┼────┤   blank = all-fill (no  │
│   │▓▓▓▓│    │    │    │    │    │    │▓▓▓▓│     valid observations):│
│   └────┴────┴────┴────┴────┴────┴────┴────┘       zero bytes stored │
│   + shard index: inner chunk → (offset, length)    — a "lean" shard │
└──────────────────────────────────────────────────────────────────────┘

WRITE  1 staged inference tile (2048²) == 1 shard: the assembly worker
       emits the whole object exactly once — no read-modify-write, and
       an all-ocean tile costs nothing (never staged, never written).
READ   a point/window read GETs the shard index, then ranged-GETs only
       the inner chunks it overlaps — ~8 MB per point, not 0.5 GB.
       (Single-ROI stores use this same geometry — one preset, two names.)
```

### Manifest splitting: why a commit costs one year, not the store

An Icechunk **manifest** is the index mapping every chunk to its object.
By default there is one per array — so every commit rewrites the whole
index, O(store), regardless of how little changed. The global store
splits manifests at `time@1`:

```
    unsplit (default)                     split time@1 (global store)
    one manifest per array                one manifest per (array, year)

    MANIFEST: all 9 years                 M2017 M2018 ⋯ M2024 M2025
    ┌────────────────────────┐            ┌────┐┌────┐  ┌────┐┌────┐
    │ every (year, y, x)     │            │ ρρ ││ ρρ │  │ ρρ ││ ρρ │
    │ chunk → object ref     │            └────┘└────┘  └────┘└─▲──┘
    └───────────▲────────────┘                                  │
                │                         WRITE  filling 2025 rewrites
    WRITE  ANY commit rewrites                   only M2025 — commit
           the whole thing:                      cost stays O(one year)
           O(entire store)                       for all nine years
                                          READ   opening a group loads
                                                 only the manifests of
                                                 the arrays/years read
```

Single-ROI stores use the same idea spatially: a 32-chunk-per-axis 2D
split so a region overwrite rewrites only the manifest tiles it touches
(see `zarr_store.manifest_split`).

The per-zone pixel grids are derived from the EPSG registry
(`storage/zone_grid.py`), snapped to the 20,480 m shard pitch.
**Zone-boundary policy:** zones are pure nominal 6° longitude bands —
disjoint, every pixel-center in exactly one zone. The Norway/Svalbard
MGRS width exceptions (32V, 31X–37X) are deliberately **not** honored:
they exist for navigation, not data grids. Consumers must not assume
MGRS behavior near those zones; the dataset advertises this via the
`zone_scheme: "utm_6deg_nominal"` group attribute.

Campaign operations — per-cell tags, snapshot expiry + GC, and a
zone×year progress reader — live in `storage/campaign.py`; the
end-to-end (zone, year) fill callable is
`orchestration/runners/zone_fill.py`.

`run_global_campaign` drives the whole thing year-serial with bounded
zone parallelism, and **triggers its own ingestion** (ADR-011): per
pending cell it dispatches `ingest-zone-year` (synthesize a zone-shaped
ROI from the coverage bitmap → run the S1/S2 ROI ingest flows onto
`mosaics/{zone}/{year}`) → `fill-zone-year` (a pre-Ray coverage gate,
then inference → shard-assemble → tag) → delete the transient mosaic
(`s5cmd --all-versions`). A `zones=["33N", "15S"]` filter restricts the
run; the default (all 120) skips already-finished cells, and `ingest=False`
bypasses ingestion when mosaics already exist upstream. A `branch` slug routes
every dispatched deployment — the fill, the ingest, and the S1/S2 grandchildren
`ingest-zone-year` dispatches — to its `-<branch>` variant, so a downstream that
registers dev-branch deployments can exercise the whole chain (including
ingestion) off prod; `branch=None` (default) is the unsuffixed production path.

### Reading a zone group (xarray)

Open a zone group through an Icechunk readonly session and ask xarray to
decode CF-linked variables as coordinates:

```python
import xarray as xr
from tessera_embeddings.storage.global_store import open_global_repo

repo = open_global_repo("s3://<bucket>/global/tessera.icechunk")
session = repo.readonly_session(branch="main")
ds = xr.open_zarr(session.store, group="33N", consolidated=False, decode_coords="all")
```

```
<xarray.Dataset>
Coordinates:
  * time         (time) datetime64[ns] 2017-01-01 2018-01-01 ... 2025-01-01
  * northing     (northing) float64 ...
  * easting      (easting) float64 ...
  * band         (band) int64 0 1 ... 127
    time_bnds    (time, bnds) datetime64[ns] ...     ← [Jan 1, Dec 31] per slot
Data variables:
    embeddings   (time, northing, easting, band) int8 ...
    scales       (time, northing, easting) float32 ...
    s2_obs_count (time, northing, easting) uint16 ...
```

**Time semantics (guaranteed).** Each `time` point is **January 1 of its
calendar year — the start of the exact Jan–Dec window that slot holds**
(`time_convention="calendar_year"`; the fill runner rejects any other
window, so the label always matches the data). The companion `time_bnds`
variable (shape `(time, 2)`, linked via `time.attrs["bounds"]` per CF)
states each slot's covered interval explicitly: `[YYYY-01-01, YYYY-12-31]`.

`decode_coords="all"` is what promotes `time_bnds` from *Data variables*
to *Coordinates* — xarray sets variables referenced by `bounds` (and
`grid_mapping`) attributes as coordinates. Without it the dataset is
identical; `time_bnds` just lists under data variables. Non-calendar
12-month windows are **not** written to this store — the single-ROI
output stores (`time_convention="12mo_window_end"`, one time entry per
window-end label) are the home for rolling windows.

**Per-pixel input provenance.** The obs-count layers record how many
observations fed each pixel's embedding: `s2_obs_count`,
`s1_asc_obs_count`, `s1_desc_obs_count` (always written; `0` = none).
By default every embedded pixel has ≥1 S1 observation; when a fill ran
with `allow_s2_only=True` (opt-in — embeds S2-valid pixels inside S1
coverage gaps using the upstream v1.1 missing-S1 convention), the
**S2-only pixels are exactly those with a finite `scales` value and
`s1_asc_obs_count + s1_desc_obs_count == 0`**. S2-only embedding quality
is unvalidated against S1-informed embeddings — see
[ADR-013](context_docs/decisions/013-optional-s1-s2-only-pixels.md).

Reference docs:
[`xarray.open_zarr` / `decode_coords`](https://docs.xarray.dev/en/stable/generated/xarray.open_zarr.html) ·
[xarray weather & climate (CF) guide](https://docs.xarray.dev/en/stable/user-guide/weather-climate.html) ·
[CF conventions §7.1 Cell Boundaries](https://cfconventions.org/cf-conventions/cf-conventions.html#cell-boundaries) ·
[cf-xarray bounds handling](https://cf-xarray.readthedocs.io/en/latest/bounds.html)

## The test that proves decoupling

[`src/tessera_embeddings/orchestration/runners/plain.py`](src/tessera_embeddings/orchestration/runners/plain.py)
is an orchestrator-free sequencer that calls the same domain
functions as the Prefect flows, without Prefect. By default it runs
the full end-to-end pipeline (ingest → cloud mask → inference →
assembly) on a laptop with torch on CPU via Ray's local mode. Slow on
real workloads, practical on the Denver quickstart ROI we ship for
exactly this purpose.

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
**The end-to-end run on the quickstart ROI is not automated at all** — it
is verified by running it by hand, which takes about three and a half
minutes on a laptop
([ADR 024](context_docs/decisions/024-the-single-path-end-to-end-is-the-quickstart-run.md)). Fast PR checks also use
AST-based architecture rules (§Architecture) to catch Prefect leaks
at the import level without running the pipeline.

## What's in here

```
src/tessera_embeddings/
  config/                pydantic config models
  ingest/                STAC ingestion, ROI rasterization, auth
  inference/             GPU inference (Ray actors, work-stealing scheduler)
  storage/               Zarr stores, manifests, empty-store seeding
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
