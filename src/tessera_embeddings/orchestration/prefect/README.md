# Prefect orchestration layer

The reference orchestration for `tessera_embeddings`. Every file
under this subtree is allowed to `import prefect`; nothing outside
this subtree is. The architecture-tests in `tests/architecture/`
enforce that rule.

```
orchestration/prefect/
├── flows/                  Layer 3: @flow definitions
│   ├── generate_roi.py
│   ├── ingest_s2_roi_reflectance.py
│   ├── ingest_s1_roi_sar.py
│   ├── tessera_embeddings.py        # single-ROI: ROI → mosaic → inference → assembly
│   ├── tessera_full_pipeline.py     # single-ROI master: chains the four above via run_deployment
│   ├── build_land_mask.py           # global campaign: registry → per-zone coverage bitmaps (no cluster)
│   ├── seed_global_store.py         # global campaign: seed the 120 UTM-zone groups (no cluster)
│   ├── fill_zone_year.py            # global campaign: one (zone, year) via Ray → assembly → tag
│   └── run_global_campaign.py       # global campaign driver: dispatch every pending (zone, year)
├── tasks/                  Layer 2: thin @task wrappers (~20 LOC each)
│   ├── ingest.py                    # process_roi_reflectance, process_roi_sar
│   ├── inference.py                 # run_inference_task, assemble_embeddings_task
│   └── land_mask.py                 # build / verify / validate coverage (no-cluster steps)
└── _dask_runner.py         internal helper: prefect_dask DaskTaskRunner factory
```

## Global campaign (120 UTM zones)

The global 10 m embeddings campaign (ADR-008) has its own four flows, distinct
from the single-ROI path above:

```text
build_land_mask   →   seed_global_store   →   run_global_campaign
(coverage bitmaps)    (metadata-only)          ├─ cluster-per-zone: per pending (zone, year): fill_zone_year (Ray cluster each)
                                               └─ chained-clusters: per year: K fill_zones_sequential runs (K long-lived Ray clusters)
```

`run_global_campaign` reads live progress via `storage.campaign.campaign_status`
and dispatches fills for every pending cell, **year by year** (outer serial
loop), under one of two strategies:

- **`fill_strategy="chained-clusters"`** (default): up to `max_parallel_clusters`
  `fill-zones-sequential` runs per year, each owning ONE long-lived Ray
  cluster whose actors are created once and **stream** through a
  size-balanced share of the year's zones — strictly ordered, with the next
  zone's tiles interleaving only once the current zone's queue is exhausted,
  so zone tails never idle the fleet and there is no per-zone teardown, actor
  churn, or model reload. Amortizes `ray up` (~5-10 min), per-worker EC2
  bringup (minutes of billed GPU idle each), the per-worker model-load cold
  start, and the EC2 capacity roll across the whole cluster instead of per zone
  (`max_parallel_clusters=1` = a single cluster for the whole year). Zones whose
  mosaics resolve a different S1 orbit than the session run per-cell after
  the stream. The shared fleet is kept busy at the seams by **ingest look-ahead**
  (the next zones' mosaics ingest while the current one infers) and **trailing
  assembly** (a zone's shard write runs on a background thread — assembly is
  ~10-15% of a zone's inference wall — while the next zone's inference keeps the
  GPUs busy). Retag-only and all-ocean cells settle before the cluster exists,
  and all-ocean cells are never ingested at all.
- **`fill_strategy="cluster-per-zone"`**: a `fill-zone-year` run per cell, with
  **bounded parallelism** within each year (`max_parallel_clusters`
  simultaneous Ray clusters). Simpler, and every zone pays a full cluster
  bringup and model load.

> **Naming.** A **zone** is always a UTM zone. A **cluster** is one Ray cluster
> and the UTM zones assigned to it. A **shard** is always a storage shard — never
> a group of zones.

### Ingest runs wide, under one fleet-wide cap

Nothing throttles ingest against fill throughput. Ingest is the cheap half of the
campaign and measures far better across many narrow fleets than a few wide ones,
so the semaphore that used to make a cell hold a slot from ingest through to
cleanup is gone (see ADR-011, where the older "peak input storage is bounded by
in-flight cells" claim is marked superseded).

What remains is a single number: **`max_parallel_ingest` (40) is how many UTM
zones may ingest simultaneously across the whole campaign**, however many clusters
are running. With `max_parallel_clusters` at 8 that is 5 zones per cluster.

Because the clusters are separate Prefect flow runs on separate machines, no
in-process semaphore can see across them, so the cap is a **Prefect global
concurrency limit** — the same mechanism as the commit gate. Each zone's ingest
holds one slot for its whole duration. The campaign upserts the limit to
`max_parallel_ingest` at start, so the parameter is the only place the number is
written and it cannot drift from the server's. Each cluster also takes an even
share of the cap as its own window, so the clusters divide it by construction
rather than racing for slots.

Mosaics can still pile up ahead of the fills that consume them — hundreds of
terabytes, transient. `cleanup_mosaics` deletes each one as its fill lands and
`sweep_orphan_mosaics` collects what a crash left behind. What was removed is the
*backpressure*, not the cleanup.

### GPUs are never booted speculatively — and density ordering keeps them fed

A cluster starts its ingest window, waits for its **first** zone alone, then calls
`ray up`. Waiting for one zone is safe only because of how zones are ordered, and
that ordering is load-bearing in two places:

1. **Across clusters.** Zones are dealt out densest-first to the currently-lightest
   cluster, so the N densest zones of the year go one to each of the N clusters.
   Every cluster therefore *opens* on a big zone.
2. **Within a cluster.** Zones are sorted by their true live-tile count, descending,
   so a cluster works from dense to sparse.

The opening zone being dense is what makes the single-zone wait safe: it takes long
enough to infer that the rest of the window lands behind it, and inference is slower
than ingest in almost every case, so the stream does not run dry and the fleet does
not idle.

```text
start window (1 + look_ahead zones) ──►│
        wait for the densest zone ─────►│
                                         ray up ──► stream dense ──────► sparse
                                                    (rest of the window ingests behind)
```

Two subtleties worth knowing:

- Sort on the **unclamped** tile count. The per-cell actor request is
  `min(num_actors, n_tiles)`, so every zone bigger than the fleet collapses to the
  same value — sorting on that leaves the whole dense end of the list in arbitrary
  order, which is exactly the part that decides what the fleet opens on.
- Waiting for a cluster's *entire* ingest was tried and is worse: unoverlapped
  ingest time paid up front, for a risk the ordering already removes.

> **Size the fleets to match.** These caps count *clusters* and *zones*, not
> machines. Eight inference clusters at the default `num_actors` plus forty
> concurrent ingests at the default `IngestSettings.max_workers` is a large amount
> of EC2; lower `num_actors` and `max_workers` alongside, since running many narrow
> fleets rather than few wide ones is the point of these defaults.

Zone-parallelism (either flavor) is safe because inference is independent
across zones and only *same-zone* fills conflict (shared group attrs →
`RebaseFailedError`) — the year-serial loop guarantees a zone never fills two
years at once, and within a sequential run the depth-1 trailing assembly can
never overlap a commit for the same zone group. The fleet-wide **committer
bound is a Prefect global concurrency limit** (`commit_limit_name`, ADR-008 D6),
passed to every fill so commits stay under the storm threshold while GPU
inference runs unbounded. `build_land_mask` and `seed_global_store` are
cluster-less (they run on the flow runner like `generate_roi`); only
`fill_zone_year` / `fill_zones_sequential` provision Ray.

## Master pipeline

`tessera_full_pipeline.py` is the recommended entry point for end-to-end
runs. It dispatches the four child flows via `arun_deployment` so
cancelling the master cancels all running children:

```
generate_roi  →  ingest_s1_roi_sar       \
              →  ingest_s2_roi_reflectance ─→  tessera_embeddings
```

The two ingestion stages run concurrently. Cluster sizes are
auto-derived from ROI chunk count via
`config.dask.compute_pipeline_cluster_sizing`; pass explicit
`ingest_min_workers` / `ingest_max_workers` / `num_actors` to
override.

## Individual flows

| File | What it does |
|---|---|
| `generate_roi.py` | Rasterise a GeoJSON or S2-tile-footprint AOI into a chunked Zarr ROI mask. No cluster — runs entirely on the flow runner. Idempotent. |
| `ingest_s2_roi_reflectance.py` | Two-phase S2 L2A ingest: load only SCL for cloud screening, then load reflectance bands for days that pass coverage. Writes mosaicked `reflectance.zarr`. |
| `ingest_s1_roi_sar.py` | OPERA RTC-S1 ingest with EDL auth, batched windows, amplitude-to-dB conversion, orbit filtering. Writes `sar_<orbit>.zarr`. |
| `tessera_embeddings.py` | Distributed GPU inference: spin up Ray cluster, work-stealing dispatch across actors, Dask-based assembly. On-cancellation hook tears down EC2 instances. |
| `tessera_full_pipeline.py` | Async master flow chaining the four above via `arun_deployment`. |
| `build_land_mask.py` | Global campaign: build per-zone coverage bitmaps from the partner delivery registry (ADR-010). Optional pre-build delivery verification + post-build validation. No cluster. |
| `seed_global_store.py` | Global campaign: create the global-store repo and seed every unseeded UTM-zone group (metadata-only, ADR-008 D1). Idempotent. No cluster. |
| `fill_zone_year.py` | Global campaign: fill one `(zone, year)` on a Ray cluster (coverage mask → inference → shard assembly → tag). Commit gate = a Prefect global concurrency limit. |
| `fill_zones_sequential.py` | Global campaign: fill one cluster's zones sequentially on a SINGLE shared Ray cluster (densest-first, ingest look-ahead, trailing assembly, idle-retirement gated until the final zone). Waits for its densest zone's mosaic before requesting GPUs. Pre-cluster triage settles retag/all-ocean cells. |
| `ingest_zone_year.py` | Global campaign: build one cell's S1/S2 mosaics on the fixed zone grid by dispatching the ROI ingest deployments onto a synthesised zone-shaped ROI. Marker-gated and crash-safe: a stale or half-written mosaic is cleared and rebuilt, never appended onto. |
| `run_global_campaign.py` | Global campaign driver: dispatch fills per pending `(zone, year)`, year-serial — per-cell `fill-zone-year` runs with bounded zone parallelism (`fill_strategy="cluster-per-zone"`), or size-balanced `fill-zones-sequential` runs on long-lived clusters (`"chained-clusters"`). |

### Cancelling reaches every level

`arun_deployment` creates an **independent** run: killing the flow that started
one does not touch it. A cancelled campaign would otherwise leave Dask and Ray
fleets billing, still writing into prefixes a retry is about to clear and
rebuild — the one race the clear-and-rebuild recovery cannot survive.

Each dispatching flow therefore stamps a tag derived from its own flow-run id on
every child, and cancels anything still live under that tag from **both** its
cancellation and its crashed hook (a crashed parent orphans children exactly like
a cancelled one). The tag is re-derived rather than remembered, because Prefect
runs terminal hooks in a fresh import after the flow process is gone. The shared
machinery is `flows/_child_runs.py`; the chain is three deep:

```text
run_global_campaign   --"campaign:<id>"-->        ingest-zone-year, fill-zone-year,
                                                  fill-zones-sequential
ingest_zone_year      --"ingest-zone-year:<id>"-->  ingest_s1_roi_sar, ingest_s2_roi_reflectance
fill_zones_sequential --"chained-ingest:<id>"-->    ingest-zone-year (look-ahead)
```

Each flow keeps its own teardown hook as well; the sweep stops the runs those
hooks then clean up after.

> **Why so many flow files?** The two-flow pattern below explains the inner/outer
> split per file. The flows themselves are kept thin — task-graph discipline
> (per-date iteration for S2, batched windows for S1, ChunkSpec-granularity
> assembly) lives in the domain modules. See
> [`ingest/README.md`](../../ingest/README.md#background-how-dask-task-graphs-consume-scheduler-ram)
> for the scheduler-RAM cost model that drives those choices, and
> [`inference/README.md`](../../inference/README.md#three-layer-chunk-anatomy)
> for the assembly-side equivalent.

## The two-flow pattern

You'll notice each flow file has an outer `@flow` and an inner
`@flow` invoked via `.with_options(task_runner=...)`. This is a
deliberate Prefect idiom — `task_runner=` must be set at flow
definition time or via `.with_options()` on a callable, which
requires a separate inner `@flow`. Don't try to collapse it. The
docstring at the top of each flow file states this explicitly.

```
ingest_s2_roi_reflectance(...)            ← outer @flow: provisions the cluster
  with ecs_cluster(...) as cluster:
    runner = DaskTaskRunner(cluster.scheduler_address)
    _ingest_s2_roi_impl.with_options(     ← inner @flow: runs against that runner
        task_runner=runner,
    )(...)
      └── process_roi_reflectance.submit(...)   ← @task: pulls client + log from context
```

## Task shells

Each `@task` shell is ~20 LOC. The pattern:

```python
@task(name="...")
def some_task(*, ...):
    """Pull client + log from context, delegate to a domain function,
    convert the dataclass result to a dict at the boundary."""
    result = some_domain_function(
        client=get_client(),
        log=get_run_logger(),
        ...
    )
    return asdict(result)
```

Domain functions never reach for context — they take `client` and
`log` as explicit parameters. That keeps them testable without a
running orchestrator (see `tests/parity/`).

## Provider selection

Each flow accepts `use_local: bool = False`. When True, the flow
imports `providers.local.dask` / `providers.local.ray` instead of
`providers.aws.*` and runs against single-machine clusters. This is
how `runners.plain` and the parity tests exercise the same code path
without AWS.

To run on a non-AWS cloud, fork the relevant flow file and replace
the `from tessera_embeddings.providers.aws.dask import ecs_cluster`
import with your provider. The domain functions don't change. See
[`docs/providers/adding-your-own.md`](../../../../docs/providers/adding-your-own.md).

## Running

```bash
# Unit-test parity (no Prefect server needed — uses prefect_test_harness):
uv run pytest tests/parity -m parity

# Local end-to-end (Local Dask + Local Ray):
uv run python -m tessera_embeddings.orchestration.runners.plain \
    examples/quickstart/config.yaml

# Production (Prefect deployment):
prefect deploy --name tessera-embeddings flows/tessera_embeddings.py
prefect deployment run 'tessera-embeddings/tessera-embeddings'
```

For Prefect-server provisioning (work pool, Blocks, deployment
templates), see [`docs/prefect-setup.md`](../../../../docs/prefect-setup.md).
