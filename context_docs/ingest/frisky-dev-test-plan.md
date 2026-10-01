# Frisky on dev: test plan and run record

What we run on the dev stack to decide whether Frisky, and the 2048-px ingest chunk it may allow,
are stable, correct, faster and better instrumented than Dask. The run log at the end records
every run with its configuration, so any figure here can be rerun. Why Frisky is wired the way it
is: [`frisky-experiment.md`](frisky-experiment.md). How to operate it:
[`docs/frisky.md`](../../docs/frisky.md).

## Ground rules

- **Yield dev only.** Standard yield-embeddings `dev/<slug>` branches, which deploy to the yield
  account (profile `yield`). The global-tessera accounts are not used. Every ROI the plan needs is
  already in `s3://arbol-tessera-inputs-dev`, so nothing is read across accounts.
- **Ingest only, until Phase C.** Frisky changes only the ingest.
- **One change at a time.** Two code versions, deployed separately:

  | Version | tessera-embeddings commit | Chunk | `ingest_code_identity` | YE branch |
  |---|---|---|---|---|
  | A | `b3e902b0` | 4096 | `ingcode-0869839717820a9c` | `dev/frisky-4096` |
  | B | `aef6c55b` | 2048 | `ingcode-5f707a14bb6d5062` | `dev/frisky-2048` |

  Version B also moves `inference_code_identity`, from `infcode-c9905e5aa8f93d3e` to
  `infcode-2613e97f0ad55679`, because `config/ingest.py` is inside the inference import closure.

- **Paired arms.** Every Frisky run has a Dask run from the same deployment with the same
  parameters; only `use_frisky` differs. Dask runs with `min_workers == max_workers`, so both arms
  have a fixed fleet and the comparison measures the engine, not autoscaling.
- **Arms run one after the other, never at the same time.** The log queries select by time window
  on the shared `/ecs/yield-embeddings` log group, so concurrent arms would mix.
- **Alternate which engine goes first** on any rung that is repeated.
- **A fresh store for every run**, under `s3://arbol-tessera-inputs-dev/mosaics/_frisky/<run>/`.
  Never an existing mosaic: version B could not append to one anyway.
- **Telemetry on every run.** Set `perf_report_uri` to `s3://arbol-tessera-inputs-dev/perf/_frisky/<run>`.
  A Dask arm writes a performance report there; a Frisky arm writes its live and final bundle
  under it.
- **Width:** 60 workers, the production width, for every performance rung. The tiny ROI uses 4,
  because its single chunk cannot occupy more.

## Phase 0: deploy (no runs)

1. In yield-embeddings, branch `dev/frisky-4096` from `main`. Point both tessera-embeddings pins in
   `pyproject.toml` at `@b3e902b0`, then relock:
   `source scripts/get_uv_index_url.sh --profile arbol-packages && scripts/lock.sh`.
   Frisky arrives in the ingestion image through the lock. Push: `dev-deploy.yml` builds the
   `dev-frisky-4096` image, registers the branch's task definitions, and bakes a Ray AMI, because
   the dependency closure changed.
2. Run `uv sync --group prefect-mgmt` locally, so the registration below sees the new
   `use_frisky` parameter. Then register the two ingest deployments:
   `uv run python scripts/deploy_flow.py --deployment yield --env dev --branch frisky-4096 --flow ingest-s2-roi-reflectance`,
   and the same with `--flow ingest-s1-roi-sar`.
3. Dispatch each run with
   `uv run python scripts/run_campaign_cell.py --deployment yield --flow <flow> --branch frisky-4096 --params-json <run>.json --watch`.
   Keep the JSON with the run log entry.

Already checked, read-only: the yield Dask security group (`sg-03b679cab3d7d672b`) admits all
traffic from itself, so Frisky's random scheduler port is reachable. Dask and the flow runner both
log to `/ecs/yield-embeddings`.

Version B repeats steps 1 to 3 on `dev/frisky-2048`, pinned at `@aef6c55b`.

## What every run records

| Measure | Source |
|---|---|
| Wall clock per leg | the flow run's start and end (`watch_run.py`) |
| Per-date build, gate and write | `te-ingest-log-queries --log-group /ecs/yield-embeddings --query date_stage_timings` (`batch_timings` for batched runs) |
| Worker restarts and exits | `worker_lifecycle_counts` and `worker_exit_reasons` from the same tool. On Frisky also the bundle's `events.json`, where any `worker_removed` is a death mid-run |
| Frisky's live state | the `frisky state:` lines (`--query frisky_state`), and `live/overview.json` in the bundle |
| Scheduler process load | the `scheduler health` lines (`te-watch-scheduler`). On Frisky only `cpu`, `rss` and `lag` mean anything |
| Correctness | the paired stores compared with `tests/parity/helpers.assert_zarr_equivalent` after `aws s3 sync` to local disk, plus identical date lists |
| Cost | workers × fleet lifetime × the Fargate rate (about $0.27 a worker-hour at 4 vCPU and 24 GiB) |
| The dossier | `te-ingest-report`, with `--frisky <bundle>` on Frisky runs |

## Phase A: Frisky at 4096 (version A), a basic check

Does Frisky work on dev, with nothing else changed?

| Rung | ROI | Window | Sensors | Workers | Arms |
|---|---|---|---|---|---|
| A1 | `tiny_epsg5070` (622 × 454 px, one chunk) | 2024-07-01 to 2024-07-31 | S2, S1 ascending | 4 | Frisky, Dask |
| A2 | `15SWC_epsg5070` (11,425 × 11,223 px, one MGRS tile) | 2024-07-01 to 2024-07-31 | S2, S1 ascending | 60 | Frisky, Dask |

**Pass, on both rungs:**

- Every Frisky leg completes.
- The bundle shows no worker leaving mid-run.
- CloudWatch shows no "Restarting worker".
- `frisky state:` shows the full fleet registered (late joiners included).
- The paired stores are equivalent, with identical date lists.

**On a failure,** stop. Diagnose from the bundle (`events.json`, `logs.json`) and the workers'
log streams before going further.

## Phase B: the 2048-px chunk (version B)

**B1: the same two rungs at 2048.** The same pass criteria as Phase A, and one more: each 2048 store
must hold the same pixels and dates as the 4096 store of the same rung and engine. Chunking differs;
content must not.

**B2: Iowa, the performance matrix.** `iowa_epsg5070` (the whole state), 2024-07-01 to
2024-07-14, S2, then S1 ascending (Iowa is single-orbit), 60 workers. Four arms, two of them on
version A's deployment:

| | Dask | Frisky |
|---|---|---|
| **4096** | the baseline | the engine alone |
| **2048** | does Dask cope with four times the tasks? | the target configuration |

Repeat the two Frisky arms. Report per-date cycle time, its build, gate and write parts, and
worker-hours for each cell, with the spread across repeats.

**B3: breadth, on ROIs ingested before.** Each for one month, Frisky against Dask at 2048, for
correctness and stability:

| ROI | Character | Orbits |
|---|---|---|
| `choco_sparse_epsg32618` | sparse, cloudy tropics | both |
| `rondonia_humidtropical_epsg32720` | humid tropics | descending |
| `alps_highrelief_epsg32632` | high relief | both |

**B4: one bounded dense-zone run, for telemetry at scale.** `zone_35N` for 2024-01-01 to
2024-01-07 only, S2, at 60 workers. That is the record's reference rung; yield-embeddings'
`conc_ref` measured 118 to 179 s a date on Dask at 4096. It is not a zone-year.

- **Arms:** Frisky at 2048, then Dask at 2048.
- **Watch it live:** use the SSM port-forward and `frisky observe overview`, `stragglers` and
  `transfers`.
- **Check the capture:** its size and how long it takes.
- **Stop rule:** cancel an arm that exceeds twice the other's per-date time, or 45 minutes.

**B5: re-measure the provisional constants.** Version B's chunk-counted thresholds are the 4096
measurements converted by area:

| Constant | Value at 2048 |
|---|---|
| `WINDOW_COST_IN_CHUNKS` | 800 |
| `WINDOW_COST_IN_CHUNKS_OVERLAPPED` | 80 |
| `AUTO_BATCH_DATES_MAX_COVERED_CHUNKS` | 2,000 |
| `_WORKERS_PER_LIVE_CHUNK` | 0.125 |

- **The window costs** re-derive offline from the zone masks, the way they were first measured.
- **The batching threshold and fleet rate** need B2 and B4's runs, and are revisited once those
  are in.
- **`MAX_TASKS_PER_WINDOW`** (24,000) is task-denominated on purpose, because tasks are what
  saturate Dask's scheduler. Raising it on Frisky is a separate rung, after B4.

## Phase C: end to end, up to Iowa

Only after Phase B passes. Run the single-ROI pipeline on version B's deployment on
`tiny_epsg5070`, then `15SWC_epsg5070`, then `iowa_epsg5070`, and no larger. Pass explicit
worker and actor counts rather than the auto-sizing, which is still calibrated in 4096-px chunks.

**Compare** each against the same ROI and window on `main`'s pipeline. Check two things: the
embeddings with `te-compare-outputs`, and the total wall clock and cost, split into ingest,
inference and assembly.

## Acceptance

| Goal | Passes when |
|---|---|
| Stability | No worker death, hang or failed leg attributable to Frisky in any phase |
| Correctness | Every paired store equivalent, date lists identical; 2048 stores equal to 4096 stores in content; Phase C embeddings equivalent to `main`'s |
| Performance | Reported per rung with its spread; whether a gain is worth adopting is the maintainers' call, not this plan's |
| Telemetry | For each rung's slowest date, the Frisky bundle answers where the time went (compute, transfer, scheduler, idle) and which worker straggled, and the Dask report from the paired arm is compared on the same questions |

## Rough cost

At about $0.27 a worker-hour, ingest only:

| Phase | Estimate |
|---|---|
| A | about $35 |
| B1 | about $35 |
| B2 | about $100 |
| B3 | about $50 |
| B4 | about $30 |

That is about $250 for all of the ingest phases. Phase C adds GPU time, estimated once the actor counts are fixed. These are estimates; the run
log records the actual figures.

## Run log

| Date | Run | Version | Engine | ROI and window | Sensor | Workers | Flow run | Wall | s/date (median) | Restarts | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
