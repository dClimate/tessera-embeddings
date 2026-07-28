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
                                               └─ chained-clusters: per year: K fill_zones_sequential shards (K long-lived Ray clusters)
```

`run_global_campaign` reads live progress via `storage.campaign.campaign_status`
and dispatches fills for every pending cell, **year by year** (outer serial
loop), under one of two strategies:

- **`fill_strategy="chained-clusters"`** (default): up to `max_parallel_zones`
  `fill-zones-sequential` runs per year, each owning ONE long-lived Ray
  cluster whose actors are created once and **stream** through a
  size-balanced shard of the year's zones — strictly ordered, with the next
  zone's tiles interleaving only once the current zone's queue is exhausted,
  so zone tails never idle the fleet and there is no per-zone teardown, actor
  churn, or model reload. Amortizes `ray up` (~5-10 min), per-worker EC2
  bringup (minutes of billed GPU idle each), the per-worker model-load cold
  start, and the EC2 capacity roll across the whole shard instead of per zone
  (`max_parallel_zones=1` = a single cluster for the whole year). Zones whose
  mosaics resolve a different S1 orbit than the session run per-cell after
  the stream. Each shard **waits for its first mosaic before requesting GPUs**,
  so a fleet is never provisioned against an ingest that has not finished. The
  shared fleet is then kept busy at the seams by **ingest look-ahead** (the next
  cells' mosaics ingest while the current cell infers) and **trailing assembly**
  (a cell's shard write runs on a background thread — assembly is ~10-15% of a
  cell's inference wall — while the next cell's inference keeps the GPUs busy).
  Retag-only and all-ocean cells settle before the cluster exists, and
  all-ocean cells are never ingested at all.
- **`fill_strategy="cluster-per-zone"`**: a `fill-zone-year` run per cell, with
  **bounded zone parallelism** within each year (`max_parallel_zones`
  simultaneous Ray clusters). Simpler, and every zone pays a full cluster
  bringup and model load.

### Ingest runs wide, and ahead

Nothing throttles ingest against fill throughput. Ingest is the cheap half of the
campaign and measures far better across many narrow fleets than a few wide ones,
so `max_parallel_ingest` (60) is deliberately **larger** than `max_parallel_zones`
(40), and the semaphore that used to make a cell hold a slot from ingest through
to cleanup is gone.

The consequence is real and accepted: a year's mosaics can pile up ahead of the
fills that consume them — of order a hundred zone-mosaics, hundreds of terabytes.
They are transient. `cleanup_mosaics` deletes each one as its fill lands, and
`sweep_orphan_mosaics` collects what a crash left behind. What was removed is the
*backpressure*, not the cleanup. See ADR-011's consequences, where the older
"peak input storage is bounded by in-flight cells" claim is marked superseded.

The asymmetry is intentional: ingest waiting is cheap, and a provisioned GPU fleet
waiting is not.

Under `chained-clusters` the ingest cap is divided across the shards as their
per-shard look-aheads, **rounded up** — a shard cannot hold less than one, and
flooring would silently deliver fewer concurrent ingests than asked for. The real
ceiling is `ceil(max_parallel_ingest / shards) × shards`; any mismatch with the
requested number is logged at dispatch. Set `max_parallel_ingest` to a multiple of
`max_parallel_zones` for an exact fit.

> **Size the fleets down to match.** These caps count *clusters*, not machines.
> Forty inference clusters at the default `num_actors` and sixty ingest clusters at
> the default `IngestSettings.max_workers` is far more EC2 than a stock account
> quota allows. Lower `num_actors` and `max_workers` alongside — running many
> narrow fleets rather than few wide ones is the point of these defaults.

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
| `fill_zones_sequential.py` | Global campaign: fill one year's zones sequentially on a SINGLE shared Ray cluster (largest-first, ingest look-ahead, trailing assembly, idle-retirement gated until the final zone). Pre-cluster triage settles retag/all-ocean cells. |
| `ingest_zone_year.py` | Global campaign: build one cell's S1/S2 mosaics on the fixed zone grid by dispatching the ROI ingest deployments onto a synthesised zone-shaped ROI. Marker-gated and crash-safe: a stale or half-written mosaic is cleared and rebuilt, never appended onto. |
| `run_global_campaign.py` | Global campaign driver: dispatch fills per pending `(zone, year)`, year-serial — per-cell `fill-zone-year` runs with bounded zone parallelism (`fill_strategy="cluster-per-zone"`), or size-balanced `fill-zones-sequential` shards on long-lived clusters (`"chained-clusters"`). |

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
