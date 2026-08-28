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
│   ├── _cell_validation.py          # internal helper: hand a tagged cell to its validator, don't wait
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

### A failed zone does not cost the campaign

Interruptions are expected — the orphan sweeper cancels child runs by design, and
an interrupted mosaic is **resumed rather than rebuilt**, so no ingest work is lost.
What the campaign adds on top is a **bounded re-dispatch**: `max_dispatch_rounds`
(default 2) rounds, each re-reading the store for what is genuinely still missing, so a
retry never repeats a landed zone. It is the only recovery that survives a child run
DYING, since a killed or cancelled run takes its own `attempts_per_cell_in_cluster`
counter with it.

Failures are recorded, never raised mid-flight. A cluster that dies having landed
most of its zones keeps that work — every zone-year is committed and tagged
independently — and the next round picks up only the remainder. Later years still
run.

Retries stop early when a round makes **no progress at all**. That is what a
deterministic failure looks like from the driver: a coverage gate, a fingerprint
mismatch, an unseeded group. Those want a human, not another GPU fleet, so the
campaign logs them at ERROR and moves on rather than burning a cluster per attempt.

#### Not waiting for the round: `immediate_refill`

A dispatch round is a **barrier**: it collects every cluster's outcome before re-reading
the store. Since a cluster owns a roster of zones, one dying early costs that whole
roster the wait for the round's slowest sibling. `immediate_refill` (default **off**,
and off is byte-for-byte the behaviour above) re-dispatches a settled cluster's
still-missing cells into the slot it just vacated — so the replacement inherits its
ingest share, its committer share and its place under `max_parallel_clusters`, and the
fleet's width and cost do not change.

**Nothing locks a zone; the zone partition IS the guarantee that a zone's years never
land on two clusters.** So a replacement is admitted only when both of these hold, and
declining is always safe because the cells then wait for the round exactly as they do
today:

1. **The predecessor reached a state that proves it stopped writing.** A fill that
   returned or raised has, by then, joined the trailing assembly thread that does its
   committing, cancelled its child ingests and waited for them to confirm terminal, and
   torn down its fleet — all inside `finally` blocks that complete before the state is
   set. A **crash** carries none of that (the process that would run the `finally` is the
   process that died, and the verdict can be reached from missed heartbeats while the run
   is still writing), and a **cancellation** is a request rather than a fact. Both wait.
2. **Enough time has passed for its descendants to have stopped.** Condition 1 covers the
   fill's own writers, and — because its teardown cancels its children and waits on them
   before the state is set — most of its direct children too. **That wait is best effort:**
   it gives up after its budget and logs whatever it could not confirm. And *their*
   grandchildren are only ever asked, by a hook that does not block. So condition 1 is not a
   guarantee about descendants, and the delay is not a belt on a working brace — it is the
   time those two unconfirmed levels actually need. The delay is that same confirmation budget again, for that level, derived from
   it rather than chosen. Counted from the cancellation request, the two together come to
   the interval the crash-recovery record already recommends between a run's death and
   re-dispatching its cells.

**Why a wait and not a check of who is writing.** A census of live runs was built and then
removed. It can only report what was true a moment ago, and it cannot make a lingering child
stop — the wrong instrument for settling an asynchronous cancellation. Two mechanisms already
*act* rather than observe, and the delay is simply what lets them finish: the fill's own
teardown, which waits for its children; and the orphan sweep, which independently finds and
stops whatever outlived a teardown, on its own schedule.

**It reserves nothing**, and should not be read as doing so. A dispatcher outside the campaign
could still start a writer. That is not introduced here — the round's own re-dispatch has
always worked this way — and closing it means fencing at the write, the same prerequisite the
crashed and cancelled cases are waiting on. The store keeps the residual affordable rather
than silent: mosaic commits do not rebase, so a second writer fails loudly.

Bounded per **cell**, for the life of the campaign rather than of a round: a cell that has
had its replacement is not eligible for another on a later round, a replacement is not
itself eligible for one, and none is issued unless a sibling is still running. A cell
therefore gains at most one attempt beyond `max_dispatch_rounds` in total.

Every read the decision makes declines on failure rather than propagating — the store's tip
and tags, and the replacement's own land-mask and SSM probes. Declining
costs a round's wait; an exception escaping mid-round would fail the campaign while sibling
fills were still writing, and an ordinary `FAILED` state does not fire the child-cancel hook
that would sweep them.

What it deliberately does not address: crashed and cancelled fills, which need fencing at
the write rather than an inference about who has stopped; the barrier itself, which still
closes the round for everything the immediate path declined; and the lifetime coupling
between a fill and its ingest children, which is load-bearing — see below.

#### A dying fill's ingest bytes are not lost

Worth stating plainly, because the opposite is the natural assumption. A failing fill
cancels its in-flight ingests and waits for them to stop, and that wait is deliberate: a
retry is a NEW parent run, deriving its child tag from its own run id, so it can neither
find an orphaned ingest nor be told about it — and mosaic commits do not rebase, so two
writers on one prefix is a failure nothing downstream detects. What is cancelled is the
*process*, not the work: a failed cell's mosaic is retained (cleanup runs only for a cell
that landed, and the orphan-mosaic sweep only touches cells that are complete AND tagged),
and an interrupted mosaic is resumed rather than rebuilt. So a fill's death costs the wall
clock of the wait, not the bytes — which is why shortening the wait is the whole fix, and
why loosening any teardown rule would be the wrong one.

At the very end — after every year has had its attempts — the campaign raises with
the complete list of unfilled cells. It fails loudly, but only once it has done
everything it could, and a re-run resumes from exactly there.

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
concurrency limit**. Each zone's ingest
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

A cluster starts its ingest window, waits for the **first mosaic to land — any of
them** — then calls `ray up`, and thereafter its feeder always takes a landed cell
over its head cell. Waiting on a *named* zone would mean waiting on the densest
one, which is also the slowest to ingest: on real coverage counts a cluster's
opening window spans about 4 to 10 hours, so blocking on the head idles the fleet
for roughly six hours with finished mosaics already on disk.

A *failed* ingest is not a landed mosaic. Bad credentials or a bad parameter fail a
child within seconds, and treating that as the signal to start would boot the paid
fleet immediately for a mosaic that does not exist. The wait therefore passes over a
failed cell while any sibling is still running. If every cell in the window finishes
and every one has failed, it **raises** — no mosaic is coming, so requesting a fleet
would buy five to ten minutes of billed GPU bringup and then tear it straight back
down when the feeder hit the same failure. The underlying ingest error is chained as
the cause. The priming and the wait both sit inside the flow's shutdown guard, so a
failure anywhere in them still cancels the child ingests already dispatched rather
than leaving them writing to mosaic prefixes a retry would race.

The density ordering is still load-bearing in two places — it is just not a
barrier any more:

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
start window (1 + look_ahead zones), all ingesting in parallel
   smallest lands ──►│                       (densest lands much later)
                      ray up ──► stream, taking whichever zone has landed
                                 (density order preserved, minus the barrier)
```

Two subtleties worth knowing:

- Sort on the **unclamped** tile count. The per-cell actor request is
  `min(num_actors, n_tiles)`, so every zone bigger than the fleet collapses to the
  same value — sorting on that leaves the whole dense end of the list in arbitrary
  order, which is exactly the part that decides what the fleet opens on.
- Waiting for a cluster's *entire* ingest was tried and is worse: unoverlapped
  ingest time paid up front, for a risk the ordering already removes.

How this plays out on the real world — 112 live UTM zones, 360,953 land tiles —
and what happens if you move off 8 clusters is measured in
[`context_docs/design/campaign-cluster-sizing.md`](../../../../context_docs/design/campaign-cluster-sizing.md).
Short version: 8 splits the year to within 0.0%, 16 costs 0.6% and roughly halves
wall clock, and past ~20 the largest zones start to dominate.

> **Size the fleets to match.** These caps count *clusters* and *zones*, not
> machines. Eight inference clusters at the default `num_actors` plus forty
> concurrent ingests at the default `IngestSettings.max_workers` is a large amount
> of EC2; lower `num_actors` and `max_workers` alongside, since running many narrow
> fleets rather than few wide ones is the point of these defaults.

Zone-parallelism (either flavor) is safe because inference is independent
across zones and only *same-zone* fills conflict (shared group attrs →
`RebaseFailedError`) — the year-serial loop guarantees a zone never fills two
years at once, and within a sequential run the depth-1 trailing assembly can
never overlap a commit for the same zone group. **Commits are otherwise ungated.**
They contend on the branch-tip CAS, since all 120 zone groups share one repo, but
run 1 measured that as 2.2 s at 16 committers and 15 s at 120 with zero
unresolvable conflicts. See `context_docs/design/commit-gate-removal-2026_08.md`. `build_land_mask` and `seed_global_store` are
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

### A landed cell is handed to a validator, and nothing waits for it

Both fill flows take a `validation_deployment`. As each cell is tagged they create one run
of it and return immediately (`flows/_cell_validation.py`), so cell N is checked while cell
N+1 is still being filled — on the chained path the dispatch happens on the trailing
assembly thread, which already has that pipelining.

Three properties are deliberate. The default is `None`, because the validator is a
**consumer's** flow and this library names none — so unlike every other child ref here it
is not derived from `branch`. A cell that fails validation **is already tagged**: the tag
records that the cell landed and the verdict records that it is sound, which are different
questions and get different records. And every failure in the dispatch is **swallowed** — a
cell that has landed must not be undone by an unreachable API. What keeps that honest is
not the log line but the verdict: a dispatch that never happened leaves none, and the
consumer's monitoring reads published cells against the verdicts on file.

The trace tag (`validates-cell-of:<id>`) is deliberately *not* the tag the cancellation
hooks below sweep. A validation describes a cell that has already landed, holds no fleet,
and is worth finishing even when its parent fill is cancelled.

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

## Deliberate fault injection (supervised drills only)

Two of the campaign's failure modes cannot be produced from outside a run. One
occupies the gap between a zone-year's two commits — bounded by two lines of one
function, so there is nothing external to aim a kill at. The other needs a GPU fleet
that is alive and holding nothing, a condition every healthy mechanism removes as soon
as it appears. A drill that cannot reach the failure it names produces a false pass, so
both are injected from inside the run.

`config/fault_injection.py` owns the mechanism, and its module docstring is the
authority on the guarantee. What matters at this layer:

| flow | parameter | fault it hosts | where it fires |
|---|---|---|---|
| `fill_zone_year.py` | `fault_injection` | `die_between_commits` | between the shard commit and the commit that marks the year complete (`storage.shard_writer.write_year_shards`) |
| `fill_zones_sequential.py` | `fault_injection` | `withhold_work` | where prepared work crosses from the feeder to the scheduler (`orchestration.runners.sequential_fill`) |

Both parameters default to nothing, and both flows *arm* before doing any work.
Arming refuses every deployment outside the drill allowlist — including a run whose
deployment identity does not resolve — and refuses a fault the flow does not host, so
an armed drill can never quietly inject nothing and be recorded as a pass. Identity is
read off the run's own injected Ray control-plane prefix rather than asked for. A run
carrying an armed fault says so at error level in its own logs under a fixed prefix, so
nobody reading those logs later mistakes a drill's artifacts for an incident.

Between the flow and the firing site the request travels as an explicit argument on
every hop (`fill_zone_year` → `assemble_zone_year` → `assemble_global` →
`write_year_shards`). Explicit rather than ambient, deliberately: one of these sits on
the commit path of the store that is the campaign's only output, and a reviewer of any
one of those functions should be able to see that it can be asked to fail.

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
