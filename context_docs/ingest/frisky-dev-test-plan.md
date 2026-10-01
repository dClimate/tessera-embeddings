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
- **One change at a time.** Each chunk size is its own deployment:

  | Version | tessera-embeddings commit | Chunk | `ingest_code_identity` | YE branch |
  |---|---|---|---|---|
  | A | `b3e902b0` | 4096 | `ingcode-0869839717820a9c` | `dev/frisky-4096` |
  | A′ | `4c12a5d1` | 4096 | `ingcode-0869839717820a9c` | `dev/frisky-4096` |
  | B | `9d994f84` | 2048 | `ingcode-5f707a14bb6d5062` | `dev/frisky-2048` |

  Version B also moves `inference_code_identity`, from `infcode-c9905e5aa8f93d3e` to
  `infcode-2613e97f0ad55679`, because `config/ingest.py` is inside the inference import closure.
  B1 ran on `aef6c55b`, which differs from B only in how `ecs_cluster` connects its setup clients
  (by the cluster object, so the hijack waited for the fleet). That is outside both code
  identities, so B1's stores hold what B writes. A′ is A with that same fix, on the branch
  `experiment/frisky-4096-startup-fix`; B2's 4096 arms ran on it.

- **Paired arms.** Every Frisky run has a Dask run from the same deployment with the same
  parameters; only `use_frisky` differs. Dask runs with `min_workers == max_workers`, so both arms
  have the same fleet size.
- **A rung's arms run at the same time:** S2 and S1 together, as in production, each on both
  engines. Paired arms then see the same catalog and S3 conditions. Their log lines share
  `/ecs/yield-embeddings`, so every timing row is attributed by its `@logStream`: the stream name
  ends in the ECS task id, and each task carries `tessera-flow-run-id=<flow run id>`.
- **A fresh store for every run**, under `s3://arbol-tessera-inputs-dev/mosaics/_frisky/<run>/`.
  Never an existing mosaic: version B could not append to one anyway.
- **Telemetry on every run.** Set `perf_report_uri` to `s3://arbol-tessera-inputs-dev/perf/_frisky/<run>`.
  A Dask arm writes a performance report there; a Frisky arm writes its live and final bundle
  under it.
- **Width:** the production widths. S2 uses 60 workers; S1 uses 13, which is 0.22 of S2's width
  (`ingest_settings.s1_worker_fraction`). The tiny ROI uses 4, because its single chunk cannot
  occupy more. Phase A's S1 arms ran at 60.

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

Version B repeats steps 1 to 3 on `dev/frisky-2048`, pinned at `@9d994f84`.

## What every run records

| Measure | Source |
|---|---|
| Wall clock per leg | the flow run's start and end (`watch_run.py`) |
| Per-date build, gate and write | `te-ingest-log-queries --log-group /ecs/yield-embeddings --query date_stage_timings` (`batch_timings` for batched runs, `s1_batch_timings` for S1), keeping the rows from the run's own task ids |
| Worker restarts and exits | `worker_lifecycle_counts` and `worker_exit_reasons` from the same tool. On Frisky also the bundle's `events.json`, where any `worker_removed` is a death mid-run |
| Frisky's live state | the `frisky state:` lines (`--query frisky_state`), and `live/overview.json` in the bundle |
| Scheduler process load | the `scheduler health` lines (`te-watch-scheduler`). On Frisky only `cpu`, `rss` and `lag` mean anything |
| Correctness | once per version: every chunk of every array read from both stores and compared exactly, NaN positions and time coordinates included. Every run: its date list |
| Cost | each tagged ECS task's billed lifetime (image pull to stop) × its vCPU and memory × the Fargate rate (about $0.27 a worker-hour at 4 vCPU and 24 GiB) |
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

**Result: passed.** Every leg completed; all 240 workers of the four A2 arms registered and left
cleanly, with no exits, kills or restarts. The A2 stores are bit-identical between engines: S2's
11 bands over 10 dates and S1's VV and VH over 2 dates, every chunk equal, NaN positions too.

| A2 arm | Engine time | Cluster created to work done | Cost |
|---|---|---|---|
| S2 Dask | 50.2 s (gate 15.5, write 33.3, build 1.4) | 127 s | $0.57 |
| S2 Frisky | 58.7 s (gate 21.9, write 35.6, build 1.2) | 170 s | $0.75 |
| S1 Dask | 10.6 s (stall 0.7, write 9.9) | 78 s | $0.38 |
| S1 Frisky | 11.3 s (stall 1.0, write 10.3) | 102 s | $0.49 |

Engine time is close. Frisky's S2 gate is slower by 6 s over three batches, which Iowa should
explain from its bundle. The cost gap is setup: on a Frisky arm the first `Client(cluster)` after
`cluster.scale(60)` waits until all 60 Fargate tasks have started (about 75 s), because `distributed`'s
`Client` awaits the cluster's pending workers. Dask's `adapt` has requested no workers at that
point, so its client connects at once and its work overlaps worker boot. Version B connects its
setup clients by address, which does not wait.

## Phase B: the 2048-px chunk (version B)

**B1: the same two rungs at 2048.** The same pass criteria as Phase A, and one more: each 2048 store
must hold the same dates as the 4096 store of the same rung, and the same pixels to within GDAL's
warp approximation.

**Result: passed.** Every leg completed with no worker exits or restarts, and every store holds its
rung's dates. Frisky and Dask are bit-identical at 2048, as at 4096. The 2048 stores differ slightly
from the 4096 ones, and the cause is the warp, not the ingest. Each chunk is a separate GDAL warp,
and GDAL's approximate transformer places each sample within 0.125 source pixels of its exact
position by interpolating across the destination window, so a different window size moves the
samples. Comparing the 2048 Frisky stores with Phase A's 4096 Dask stores on 15SWC:

| | Valid pixels that differ | Typical difference | 1st to 99th percentile |
|---|---|---|---|
| S2 red | about 35% | 1 to 2 counts | −27 to +28 |
| S1 VV | 82% | 0.1% of the value | −50 to +51 |

- **No gate decision changed.** About 0.05% of S2 pixels switch between nodata and valid, at data
  and mask edges and equally in both directions; no window does.
- **S1 moves most** because its tolerance is in 30 m source pixels, which is 0.375 output pixels.
- **The mechanism is reproduced.** Warping one S2 scene over the 15SWC grid both ways gives the
  same spread (−38 to +39), and with the exact transformer (`tolerance=0`) both window sizes give
  identical output.
- **2048 is the closer to exact.** On a 20 m band, the mean difference from the exact warp is 1.5
  counts for 2048 windows and 6.7 for 4096. The exact transformer would cost eight times the warp
  time; 2048 windows cost 3% more than 4096 ones.

**B2: Iowa, the performance matrix.** `iowa_epsg5070` (the whole state), 2024-07-01 to
2024-07-14, S2 and S1 ascending (Iowa is single-orbit). Four arms per sensor, two of them on
version A's deployment:

| | Dask | Frisky |
|---|---|---|
| **4096** | the baseline | the engine alone |
| **2048** | does Dask cope with four times the tasks? | the target configuration |

Repeat the two Frisky arms. Report per-date cycle time, its build, gate and write parts, and
worker-hours for each cell, with the spread across repeats.

**Result.** Every arm completed, and each sensor's six stores hold the same dates. The repeats
agree within 6%.

| Iowa, s/date | Dask | Frisky, two runs |
|---|---|---|
| S2 at 4096 | 17.2 | 22.8, 24.2 |
| S2 at 2048 | 18.6 | 23.2, 23.3 |
| S1 at 4096 | 13.7 | 16.4, 16.6 |
| S1 at 2048 | 13.0 | 14.6, 14.8 |

An S2 arm cost $0.94 on Dask at 4096, $1.11 on Dask at 2048, $1.21 to $1.25 on Frisky at 4096 and
$1.36 to $1.38 on Frisky at 2048. An S1 arm cost $0.14 to $0.16.

- **At Iowa scale the scheduler is not the bound.** Dask runs four times the tasks at 2048 for 8%
  more S2 time and slightly less S1 time.
- **Frisky is slower: 25 to 40% on S2 and 12 to 21% on S1, almost all of it in the write.** Per
  task, from Frisky's spans and the Dask report's task stream of the 2048 S2 arms, Frisky reads and
  warps a band chunk about 10% faster (a median 880 ms against 1,030 ms). But its store write
  (`getitem-where-ice-changeset`, which hands one chunk to icechunk) takes 54% longer, 203 ms
  against 132 ms, over 23,000 such tasks. GIL waits and deserialisation are negligible, so the time
  is inside the write call itself. Why is open.
- **The bundle covers only the run's tail.** The end-of-run capture keeps the most recent 500,000
  spans, and at Iowa scale Frisky emits about 15,000 a second, so it holds the last 33 s. The live
  views fetch 200,000, about 13 s. The per-task comparison above came from that tail.

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

- **Arms:** Frisky at 2048 with `frisky_drain_spans`, then Dask at 2048.
- **Watch it live:** use the SSM port-forward and `frisky observe overview`, `stragglers` and
  `transfers`.
- **Check the capture:** the drained parts' size, whether they hold every date, and the
  scheduler's CPU while they are taken.
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
| Correctness | Every paired store equivalent, date lists identical; 2048 stores the same as 4096 stores to within GDAL's warp approximation; Phase C embeddings equivalent to `main`'s |
| Performance | Reported per rung with its spread; whether a gain is worth adopting is the maintainers' call, not this plan's |
| Telemetry | For each rung's slowest date, the Frisky bundle answers where the time went (compute, transfer, scheduler, idle) and which worker straggled, and the Dask report from the paired arm is compared on the same questions |

## Rough cost

Ingest only, at about $0.27 a worker-hour. Phases A to B2 are measured; the rest are estimates:

| Phase | Cost |
|---|---|
| A | $2.30, measured |
| B1 | $1.72, measured, with the startup fix's two smoke runs |
| B2 | $8.12, measured |
| B3 | about $10 |
| B4 | about $15 (two arms of about 20 worker-hours) |

That is about $37 for the ingest phases. Phase C adds GPU time, estimated once the actor counts
are fixed. The run log records the actual figures.

## Run log

Wall is the flow run, setup included. s/date is engine time per date: the batches' build, gate and
write for S2, or stall and write for S1, divided by the dates written.

| Date | Run | Version | Engine | ROI and window | Sensor | Workers | Flow run | Wall | s/date | Restarts | Result |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-01 | a1-s2-frisky | A | Frisky | tiny_epsg5070, 2024-07 | S2 | 4 | `ea817d60` | 3m05s | smoke only | 0 | pass: 11 dates |
| 2026-10-01 | a1-s1-frisky | A | Frisky | tiny_epsg5070, 2024-07 | S1 asc | 4 | `3de4ae35` | 2m33s | smoke only | 0 | pass: 2 dates |
| 2026-10-01 | a2-s2-frisky | A | Frisky | 15SWC_epsg5070, 2024-07 | S2 | 60 | `60a14518` | 4m25s | 5.9 | 0 | pass: 10 dates, $0.75 |
| 2026-10-01 | a2-s2-dask | A | Dask | 15SWC_epsg5070, 2024-07 | S2 | 60 | `a34bdea5` | 3m34s | 5.0 | 0 | pass: 10 dates, $0.57 |
| 2026-10-01 | a2-s1-dask | A | Dask | 15SWC_epsg5070, 2024-07 | S1 asc | 60 | `8a0ff606` | 2m43s | 5.3 | 0 | pass: 2 dates, $0.38 |
| 2026-10-01 | a2-s1-frisky | A | Frisky | 15SWC_epsg5070, 2024-07 | S1 asc | 60 | `ec58103a` | 3m12s | 5.6 | 0 | pass: 2 dates, $0.49 |
| 2026-10-01 | b1-s2-frisky-tiny | B (`aef6c55b`) | Frisky | tiny_epsg5070, 2024-07 | S2 | 4 | `46fe6e45` | 3m17s | smoke only | 0 | pass |
| 2026-10-01 | b1-s1-frisky-tiny | B (`aef6c55b`) | Frisky | tiny_epsg5070, 2024-07 | S1 asc | 4 | `fe49fc1b` | 2m32s | smoke only | 0 | pass |
| 2026-10-01 | b1-s2-frisky | B (`aef6c55b`) | Frisky | 15SWC_epsg5070, 2024-07 | S2 | 60 | `5cf9e61f` | 4m27s | 4.9 | 0 | pass: 10 dates, $0.79 |
| 2026-10-01 | b1-s2-dask | B (`aef6c55b`) | Dask | 15SWC_epsg5070, 2024-07 | S2 | 60 | `868cf1f3` | 3m31s | 4.3 | 0 | pass: 10 dates, $0.55 |
| 2026-10-01 | b1-s1-frisky | B (`aef6c55b`) | Frisky | 15SWC_epsg5070, 2024-07 | S1 asc | 13 | `9eafc4f3` | 3m39s | 4.7 | 0 | pass: 2 dates, $0.15 |
| 2026-10-01 | b1-s1-dask | B (`aef6c55b`) | Dask | 15SWC_epsg5070, 2024-07 | S1 asc | 13 | `cfa56cf5` | 2m39s | 4.3 | 0 | pass: 2 dates, $0.09 |
| 2026-10-01 | bfix-s2-frisky-tiny | B | Frisky | tiny_epsg5070, 2024-07 | S2 | 4 | `9d0e434d` | 3m11s | smoke only | 0 | pass: workers joined after the hijack |
| 2026-10-01 | bfix-s1-frisky-tiny | B | Frisky | tiny_epsg5070, 2024-07 | S1 asc | 4 | `0b497120` | 2m25s | smoke only | 0 | pass: workers joined after the hijack |
| 2026-10-01 | b2-s2-frisky-2048 | B | Frisky | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `99b04091` | 6m54s | 23.2 | 0 | pass: 8 dates, $1.38 |
| 2026-10-01 | b2-s2-dask-2048 | B | Dask | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `c0a4cfd8` | 5m48s | 18.6 | 0 | pass: 8 dates, $1.11 |
| 2026-10-01 | b2-s1-frisky-2048 | B | Frisky | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `dfdec4ff` | 4m08s | 14.6 | 0 | pass: 4 dates, $0.16 |
| 2026-10-01 | b2-s1-dask-2048 | B | Dask | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `e4bcd4f0` | 3m49s | 13.0 | 0 | pass: 4 dates, $0.14 |
| 2026-10-01 | b2-s2-frisky-4096 | A′ | Frisky | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `fe79b0a0` | 6m20s | 22.8 | 0 | pass: 8 dates, $1.21 |
| 2026-10-01 | b2-s2-dask-4096 | A′ | Dask | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `4f0dda2d` | 5m13s | 17.2 | 0 | pass: 8 dates, $0.94 |
| 2026-10-01 | b2-s1-frisky-4096 | A′ | Frisky | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `8abf9d49` | 3m37s | 16.4 | 0 | pass: 4 dates, $0.14 |
| 2026-10-01 | b2-s1-dask-4096 | A′ | Dask | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `1c11f591` | 3m31s | 13.7 | 0 | pass: 4 dates, $0.14 |
| 2026-10-01 | b2-s2-frisky-2048-r2 | B | Frisky | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `c85d4321` | 6m50s | 23.3 | 0 | pass: 8 dates, $1.36 |
| 2026-10-01 | b2-s1-frisky-2048-r2 | B | Frisky | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `0310ec3a` | 3m51s | 14.8 | 0 | pass: 4 dates, $0.15 |
| 2026-10-01 | b2-s2-frisky-4096-r2 | A′ | Frisky | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `297322b4` | 6m18s | 24.2 | 0 | pass: 8 dates, $1.25 |
| 2026-10-01 | b2-s1-frisky-4096-r2 | A′ | Frisky | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `7358d35d` | 3m34s | 16.6 | 0 | pass: 4 dates, $0.14 |
