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
| C | `cb2307d6` | 2048 | `ingcode-dfef8f7ea6482ad6` | `dev/frisky-2048c` |

  Version B also moves `inference_code_identity`, from `infcode-c9905e5aa8f93d3e` to
  `infcode-2613e97f0ad55679`, because `config/ingest.py` is inside the inference import closure.
  B1 ran on `aef6c55b`, which differs from B only in how `ecs_cluster` connects its setup clients
  (by the cluster object, so the hijack waited for the fleet). That is outside both code
  identities, so B1's stores hold what B writes. A′ is A with that same fix, on the branch
  `experiment/frisky-4096-startup-fix`; B2's 4096 arms ran on it. B's later runs ran on later
  commits of its branch that change neither identity (`8d461c61`, the span drain; `31ea6ac0`, the
  pickling fix); the run log names each.

  Version C adds the two fixes B's runs found inside the ingest identity: the ROI mask's block
  reader made module-level, and a date's write submitted in groups (the record's item 5). It also
  carries the span-telemetry bounds, which sit outside both identities. It moves
  `inference_code_identity` to `infcode-678da02d8d87d6aa`.

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

Version B repeats steps 1 to 3 on `dev/frisky-2048`, pinned at `@9d994f84`, and version C on
`dev/frisky-2048c` at `@cb2307d6`, registering `tessera-embeddings` too for Phase C.

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
- **Frisky is slower: 25 to 40% on S2 and 12 to 21% on S1, mostly in the write phase, but not in
  its tasks.** Over the whole drained run below, against the Dask arm's task stream at 2048:
  Frisky's median store write (`getitem-where-ice-changeset`, one chunk handed to icechunk) is
  128 ms against 132 ms, over 23,187 and 23,188 writes, and it reads and warps a band chunk 11%
  faster (907 ms against 1,017 ms). It spends less task time on the same graph, 25,712 s against
  27,753 s. The time is lost between computes: the whole fleet sat idle 80 s of the drained
  208 s run, against 20 s on Dask, longest before the two write batches (32 s and 25 s, against
  8 s and 7 s) while Frisky's client pickled each graph before submitting it. The record's item 5
  has the mechanism and the fix.
- **The pickling fix closes the S2 gap** (`31ea6ac0`, the `-fix` rows in the
  run log, Frisky and Dask side by side). Per date, S2 took 17.9 s on both engines and S1 13.1 s
  on Frisky against 13.8 s on Dask. The idle stretches before the two write batches fell from
  31.7 s and 25.1 s to 10.3 s and 8.4 s, against Dask's 9.1 s and 8.0 s, and Frisky's write
  phases ran 5% shorter than Dask's. Both engines logged their last batch at the same second;
  Frisky's run then spent 45 s on its end-of-run telemetry against Dask's 7 s, which is the
  remaining cost gap ($1.21 against $1.07) and only arises on runs that set `perf_report_uri`.
  The rest of Frisky's idle time is its client pickling each coverage gate's graph, mostly the
  ROI mask's reader (the record's item 5).
- **At 4096, each batch also ends on a long tail.** The final batch held the same work on every
  arm, about 10,350 task-seconds, but took 66 s and 86 s on Frisky against 49 s on Dask: one
  worker was handed up to 1.8 times the mean work and kept it while the rest went idle.
- **A tail sample misleads here.** Write times rise through each batch, from about 90 ms to
  210 ms, so the end-of-run capture's last 33 s alone put Frisky's writes 54% above Dask's
  whole-run median. Compare engines on whole runs.
- **The bundle covers only the run's tail.** The end-of-run capture keeps the most recent 500,000
  spans, and at Iowa scale Frisky emits about 15,000 a second, so it holds the last 33 s. The live
  views fetch 200,000, about 13 s.

**The span drain, on the same S2 and S1 Frisky arms at 2048** (`frisky_drain_spans`, the
`-drain` rows in the run log):

- **It keeps the run.** The S2 run's 4 parts hold 419,652 spans in 12.1 MB gzipped, from all 60
  workers over 208 s. They count 23,187 store writes and 2,113 reads of each band, against the
  23,188 and 2,113 in the Dask arm's task stream.
- **It loses nothing Frisky keeps.** Over the last 33 s, every drained-name span in the
  end-of-run capture is in the parts. The few tasks short of a span in the parts (two band reads,
  and about 0.05% of call, GIL and deserialise spans) are short in Frisky's own buffers too: its
  tracing misses about one span in 2,000.
- **It costs little at this scale.** 24.1 s a date on S2 against 23.2 and 23.3 without it, and
  15.1 on S1 against 14.6 and 14.8: within the repeats' spread, though higher on both. The
  scheduler's CPU peaked at 79% of a core against 54% and 59%, its median unchanged at about 10%.

**B3: breadth, on ROIs ingested before.** Each for one month, Frisky against Dask at 2048, for
correctness and stability:

| ROI | Character | Orbits |
|---|---|---|
| `choco_sparse_epsg32618` | sparse, cloudy tropics | both |
| `rondonia_humidtropical_epsg32720` | humid tropics | descending |
| `alps_highrelief_epsg32632` | high relief | both |

**Result: passed** (version B at `31ea6ac0`, 2024-07, all 16 arms at once). Every arm completed,
each pair of stores holds the same dates, and none of the 490 workers died or restarted. S2 took
2% to 19% longer a date on Frisky; S1 went both ways, from 47% faster to 49% slower, on two to
eight dates an arm with 16 arms competing for the catalog. It cost $6.15.

**B4: one bounded dense-zone run, for telemetry at scale.** `zone_35N` for 2024-01-01 to
2024-01-07 only, S2, at 60 workers. That is the record's reference rung; yield-embeddings'
`conc_ref` measured 118 to 179 s a date on Dask at 4096. It is not a zone-year.

- **Arms:** Frisky at 2048 with `frisky_drain_spans`, then Dask at 2048.
- **Watch it live:** use the SSM port-forward and `frisky observe overview`, `stragglers` and
  `transfers`.
- **Check the capture:** the drained parts' size, whether they hold every date, and the
  scheduler's CPU while they are taken.
- **Stop rule:** cancel an arm that exceeds twice the other's per-date time, or 45 minutes.

**Result: Frisky is 18% to 30% faster a date, because Dask's scheduler saturates.** Three
attempts, all version B at `31ea6ac0` with the campaign's 0.1% coverage threshold after the
first:

| 35N, 1 to 7 January 2024 | Dask | Frisky |
|---|---|---|
| 4096 mask, no pipelining, s/date | 275 | 193 |
| 2048 mask, `pipeline_dates`, wall between dates, 2 to 6 January | 170 to 246, mean 211 | 136 to 204, mean 173 |
| 2048 mask, `pipeline_dates`, run wall | 31 min | 27 min, failed at close |

- **The first attempt sat in the mask scan.** The zone masks were exported at 4096. At 2048 the
  live-window code cannot list the mask's chunk keys, and reads it block by block on one
  thread, about 10 minutes for 35N, before any task exists. It was cancelled at 9 minutes for
  that. The second attempt paid the scan on both arms; the third ran on masks exported at 2048
  (`s3://arbol-tessera-inputs-dev/_frisky_2048/rois/zarrs/`, all 112 zones with land), which
  take the listing.
- **Dask's scheduler is the bound at zone scale.** It sat at 95% to 101% of its core through every
  date, and froze its event loop for up to 10 s taking a write graph of 108,000 to 168,000 tasks.
  It ran the gate's 42,000 small tasks with the fleet 24% to 32% busy and the write at about
  80%. Frisky's scheduler ran at a median 8% of a core and kept the fleet 97% busy through every
  write. Median task times are the same on both engines.
- **Frisky's remaining loss is between computes.** Over the second attempt no task ran anywhere
  for 31% of the run, against Dask's 35%: 18 to 35 s before each write and 24 to 26 s before each
  gate, while its client converted each graph. `pipeline_dates` hid the gate's share on every
  date of the third attempt, where Dask still stalled 23 to 26 s a date; version C's grouped
  writes target the write's share.
- **The pipelined Frisky arm wrote every date and then failed.** Its scheduler's span buffer, at
  Frisky's default of 1,000,000, grew it to 5.7 GiB of 8, and the end-of-run capture's three span
  queries took it to 6.8 GiB before it died; `cluster.close()` then timed out reaching it. The
  record's span-buffer finding has the fix, which version C carries.

**On version C** (the 2048 mask, `pipeline_dates`, both arms at once, wall between date commits):

| 35N, 2 to 7 January | Dask | Frisky |
|---|---|---|
| Version B, s/date | 211 (to 6 January) | 165 |
| Version C, s/date | 269 | 152 |
| Version C, run wall | 41 min | 23 min |
| Version C, cost | $9.26 | $5.18 |

- **Grouped writes help Frisky and hurt Dask.** Frisky's writes ran 3% to 16% shorter from the
  second date; Dask's ran 31% to 35% longer on every date (194 to 296 s, against 147 to 223 s on
  both earlier runs). The record's item 5 has the numbers.
- **Both completed, and the stores agree.** The same 7 dates, and a random 400 chunks of each of
  the 11 arrays identical, NaN positions included (45 to 53 of them held data; the rest of the
  zone is sea), read with `temp/frisky-dev/compare_stores.py --sample 400`. Frisky's span drain
  kept 4,822,258 spans in 19 parts with no failed request.
- **Scheduler memory.** Frisky's levelled at 3.4 to 3.5 GiB after stepping up at the first two
  live snapshots, then reached 6.6 GiB under the end-of-run `spans.json` query, which timed out
  at the proxy; both are now kept off the scheduler for drained runs. Dask's reached 6.8 GiB
  writing its performance report, which took 4 min 14 s after the last date.

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

One run: `iowa_epsg5070` for a year, 2024-11-01 to 2025-10-31, the window of the Iowa inference
baseline in [`inference-on-gpus.md`](../inference/inference-on-gpus.md). The baseline's
`a60550ae` inferred 404 chunks in about 73 minutes and 34 GPU-hours on 30 actors. Ingest S2 at 60
workers and S1 ascending at 13 on Frisky with the drain, then run `tessera-embeddings` with the
baseline's parameters (30 actors, `s1_orbit` both, `time_window_end` October 2025). Everything
writes under its own prefix in each bucket, `_frisky_e2e_c/` for version C, with the Iowa ROI and
the model checkpoint (`models/tessera_v1_1_aws_encoder.pt`, which inference reads from under the
inputs prefix) copied there unchanged, so no baseline store is touched; the mosaics are kept for reruns.

**Compare** against the baseline: the embeddings with `te-compare-outputs`, and the wall clock
and cost, split into ingest, inference and assembly. Inference code has changed since the
baseline, so a difference is only attributed to the chunk size once that is ruled out.

**On version B, cancelled.** The S2 ingest slowed about 63 minutes in, as the growing cost of
Frisky's scheduler heartbeats levelled its loop at 17–18% busy (the record's heartbeat finding):
gates rose from 10 to 93 s and writes from 52 to 228 s, with gaps of 11 to 61 s between small
tasks while the workers idled. The run was cancelled at 1 h 36 min, with S1's year complete, and
the year rerun on version C.

**On version C, ingest.** S1's year took 33 min at 10.8 s a date. S2 ran at about 18 s a date
between commits through August, then slowed at the same hour mark, gates reaching 88 s; it was
cancelled at 1 h 05 min with 192 dates written and resumed on a fresh cluster, which starts the
day after the newest held date and wrote the last 29 dates in 11 min at 20.6 s a date. Ingest cost
$19.58 in all.

**On version C, inference** (`d7e25533`; the first attempt, `b65429de`, found no checkpoint under
the prefix and started no actor). The 2048 store did not change inference at Iowa scale, because
the starter prefetch already hides the chunk reads it was meant to shorten:

| Iowa inference | Baseline `a60550ae` (4000² mosaics) | 2048² mosaics |
|---|---|---|
| Chunks | 404 | 394, all written |
| Inference span | about 73 min, 30 actors | 72 min, 28 at peak (GPU capacity) |
| GPU-hours, peak actors × span | about 34 | 33.5 ($62) |
| GPU-hours busy | not recorded | 25.2 ($47) |
| Per-chunk GPU overhead, median | about 6 s | 5.7 s |

The flow took 1 h 37 min: 9 min to start the Ray cluster and actors, 72 min inferring, 15 min
assembling. The baseline's own logs are past CloudWatch's retention, so its column is the
inference record's figures.

The embeddings agree broadly but are not equivalent. Over 40 random 64-px windows (163,840 pixels,
`temp/frisky-dev/compare_embeddings.py`), the median cosine similarity is 0.993 and the 5th
percentile 0.947. The difference is in the mosaics, not the inference: the two S2 stores hold the
same dates (all 217 of the baseline's, plus 4), but the per-pixel S2 observation counts agree on
only 28% of pixels, far beyond the warp's 0.05% (S1 ascending: 95%). The baseline's mosaics predate
later ingest changes, the clearest-scene fix of PR #121 among them, so isolating the chunk size
needs a 4096 ingest of the same year on current code.

## Acceptance

| Goal | Passes when |
|---|---|
| Stability | No worker death, hang or failed leg attributable to Frisky in any phase |
| Correctness | Every paired store equivalent, date lists identical; 2048 stores the same as 4096 stores to within GDAL's warp approximation; Phase C embeddings equivalent to `main`'s |
| Performance | Reported per rung with its spread; whether a gain is worth adopting is the maintainers' call, not this plan's |
| Telemetry | For each rung's slowest date, the Frisky bundle answers where the time went (compute, transfer, scheduler, idle) and which worker straggled, and the Dask report from the paired arm is compared on the same questions |

## Rough cost

Ingest only, at about $0.27 a worker-hour:

| Phase | Cost |
|---|---|
| A | $2.30, measured |
| B1 | $1.72, measured, with the startup fix's two smoke runs |
| B2 | $12.27, measured, with the drain's and the fix's runs |
| B3 | $6.15, measured |
| B4 | $36.92, measured, over three attempts |
| C on version B | $24.27, measured, ingest only, cancelled |
| C on version C | $19.58 of ingest and $47 to $62 of GPU time, measured |

That is $59.36 for the ingest phases. The run log records each run's figures.

## Run log

Wall is the flow run, setup included. s/date is engine time per date: the batches' build, gate and
write for S2, or stall and write for S1, divided by the dates written. With `pipeline_dates` the
stages overlap, so those runs give the wall clock between one date's commit and the next.

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
| 2026-10-01 | b2-s2-frisky-2048-drain | B (`8d461c61`) | Frisky | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `6f7f663a` | 7m05s | 24.1 | 0 | pass: 8 dates, $1.42, 419,652 spans drained |
| 2026-10-01 | b2-s1-frisky-2048-drain | B (`8d461c61`) | Frisky | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `e99aebd7` | 3m58s | 15.1 | 0 | pass: 4 dates, $0.16, 26,767 spans drained |
| 2026-10-01 | b2-s2-frisky-2048-fix | B (`31ea6ac0`) | Frisky | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `3f2efe64` | 6m20s | 17.9 | 0 | pass: 8 dates, $1.21, drained |
| 2026-10-01 | b2-s2-dask-2048-fix | B (`31ea6ac0`) | Dask | iowa_epsg5070, 2024-07-01..14 | S2 | 60 | `2289dd5c` | 5m43s | 17.9 | 0 | pass: 8 dates, $1.07 |
| 2026-10-01 | b2-s1-frisky-2048-fix | B (`31ea6ac0`) | Frisky | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `b60a9706` | 3m34s | 13.1 | 0 | pass: 4 dates, $0.14, drained |
| 2026-10-01 | b2-s1-dask-2048-fix | B (`31ea6ac0`) | Dask | iowa_epsg5070, 2024-07-01..14 | S1 asc | 13 | `4600ef05` | 3m54s | 13.8 | 0 | pass: 4 dates, $0.15 |
| 2026-10-01 | b3-choco-s2-frisky | B (`31ea6ac0`) | Frisky | choco_sparse_epsg32618, 2024-07 | S2 | 60 | `0d7815a1` | 6m04s | 8.1 | 0 | pass: 4 dates, $0.95, drained |
| 2026-10-01 | b3-choco-s2-dask | B (`31ea6ac0`) | Dask | choco_sparse_epsg32618, 2024-07 | S2 | 60 | `2fa77a40` | 6m07s | 7.2 | 0 | pass: 4 dates, $0.59 |
| 2026-10-01 | b3-choco-s1-asc-frisky | B (`31ea6ac0`) | Frisky | choco_sparse_epsg32618, 2024-07 | S1 asc | 13 | `6af69cbf` | 5m04s | 7.0 | 0 | pass: 2 dates, $0.14, drained |
| 2026-10-01 | b3-choco-s1-asc-dask | B (`31ea6ac0`) | Dask | choco_sparse_epsg32618, 2024-07 | S1 asc | 13 | `63d7efa9` | 5m38s | 6.4 | 0 | pass: 2 dates, $0.18 |
| 2026-10-01 | b3-choco-s1-des-frisky | B (`31ea6ac0`) | Frisky | choco_sparse_epsg32618, 2024-07 | S1 des | 13 | `6d6513f2` | 5m22s | 10.8 | 0 | pass: 3 dates, $0.12, drained |
| 2026-10-01 | b3-choco-s1-des-dask | B (`31ea6ac0`) | Dask | choco_sparse_epsg32618, 2024-07 | S1 des | 13 | `2353f446` | 5m08s | 9.1 | 0 | pass: 3 dates, $0.14 |
| 2026-10-01 | b3-rondonia-s2-frisky | B (`31ea6ac0`) | Frisky | rondonia_humidtropical_epsg32720, 2024-07 | S2 | 60 | `67f94ad3` | 5m39s | 6.4 | 0 | pass: 12 dates, $0.77, drained |
| 2026-10-01 | b3-rondonia-s2-dask | B (`31ea6ac0`) | Dask | rondonia_humidtropical_epsg32720, 2024-07 | S2 | 60 | `667de6e8` | 5m50s | 6.3 | 0 | pass: 12 dates, $0.83 |
| 2026-10-01 | b3-rondonia-s1-des-frisky | B (`31ea6ac0`) | Frisky | rondonia_humidtropical_epsg32720, 2024-07 | S1 des | 13 | `85a843d5` | 5m47s | 10.3 | 0 | pass: 4 dates, $0.20, drained |
| 2026-10-01 | b3-rondonia-s1-des-dask | B (`31ea6ac0`) | Dask | rondonia_humidtropical_epsg32720, 2024-07 | S1 des | 13 | `4441e872` | 3m40s | 6.9 | 0 | pass: 4 dates, $0.12 |
| 2026-10-01 | b3-alps-s2-frisky | B (`31ea6ac0`) | Frisky | alps_highrelief_epsg32632, 2024-07 | S2 | 60 | `b3094666` | 5m37s | 6.9 | 0 | pass: 12 dates, $0.79, drained |
| 2026-10-01 | b3-alps-s2-dask | B (`31ea6ac0`) | Dask | alps_highrelief_epsg32632, 2024-07 | S2 | 60 | `6d1e2ec8` | 5m38s | 5.8 | 0 | pass: 12 dates, $0.69 |
| 2026-10-01 | b3-alps-s1-asc-frisky | B (`31ea6ac0`) | Frisky | alps_highrelief_epsg32632, 2024-07 | S1 asc | 13 | `639ebe69` | 5m27s | 5.2 | 0 | pass: 8 dates, $0.13, drained |
| 2026-10-01 | b3-alps-s1-asc-dask | B (`31ea6ac0`) | Dask | alps_highrelief_epsg32632, 2024-07 | S1 asc | 13 | `ceb41466` | 5m05s | 9.8 | 0 | pass: 8 dates, $0.16 |
| 2026-10-01 | b3-alps-s1-des-frisky | B (`31ea6ac0`) | Frisky | alps_highrelief_epsg32632, 2024-07 | S1 des | 13 | `0eb80dbb` | 5m51s | 6.2 | 0 | pass: 8 dates, $0.19, drained |
| 2026-10-01 | b3-alps-s1-des-dask | B (`31ea6ac0`) | Dask | alps_highrelief_epsg32632, 2024-07 | S1 des | 13 | `f1ef3f31` | 5m47s | 9.2 | 0 | pass: 8 dates, $0.15 |
| 2026-10-01 | b4-s2-frisky | B (`31ea6ac0`) | Frisky | zone_35N (4096 mask), 2024-01-01..07, 5% coverage | S2 | 60 | `6c4b5125` | 9m40s | – | 0 | cancelled in the mask scan, $2.10 |
| 2026-10-01 | b4-s2-dask | B (`31ea6ac0`) | Dask | zone_35N (4096 mask), 2024-01-01..07, 5% coverage | S2 | 60 | `43caa8f3` | 9m30s | – | 0 | cancelled in the mask scan, $2.07 |
| 2026-10-01 | b4b-s2-frisky | B (`31ea6ac0`) | Frisky | zone_35N (4096 mask), 2024-01-01..07 | S2 | 60 | `844c18eb` | 37m00s | 193.2 | 0 | pass: 7 dates, $8.50, drained |
| 2026-10-01 | b4b-s2-dask | B (`31ea6ac0`) | Dask | zone_35N (4096 mask), 2024-01-01..07 | S2 | 60 | `9e1718dc` | 51m02s | 275.4 | 0 | pass: 7 dates, $11.70 |
| 2026-10-01 | b4c-s2-frisky | B (`31ea6ac0`) | Frisky | zone_35N (2048 mask), 2024-01-01..07, pipelined | S2 | 60 | `e4581666` | 26m51s | 165 (wall, dates 2 to 7) | 0 | fail at close: 7 dates written, scheduler died after the capture, $5.55, drained |
| 2026-10-01 | b4c-s2-dask | B (`31ea6ac0`) | Dask | zone_35N (2048 mask), 2024-01-01..07, pipelined | S2 | 60 | `fec46c8d` | 30m58s | 211 (wall, dates 2 to 6) | 0 | pass: 7 dates, $7.00 |
| 2026-10-01 | c-iowa-s1-frisky | B (`31ea6ac0`) | Frisky | iowa_epsg5070, 2024-11-01..2025-10-31 | S1 asc | 13 | `88928e92` | 35m10s | 11.2 | 0 | pass: 169 dates, $1.84, drained |
| 2026-10-01 | c-iowa-s2-frisky | B (`31ea6ac0`) | Frisky | iowa_epsg5070, 2024-11-01..2025-10-31 | S2 | 60 | `eeff91a8` | 1h35m33s | 23.0 | 0 | cancelled: slowed after an hour, 180 dates written, $22.43, drained |
| 2026-10-02 | vc-35n-s2-frisky | C | Frisky | zone_35N (2048 mask), 2024-01-01..07, pipelined | S2 | 60 | `11a8f199` | 23m03s | 152 (wall, dates 2 to 7) | 0 | pass: 7 dates, $5.18, 4,822,258 spans drained |
| 2026-10-02 | vc-35n-s2-dask | C | Dask | zone_35N (2048 mask), 2024-01-01..07, pipelined | S2 | 60 | `2f72bf4b` | 40m39s | 269 (wall, dates 2 to 7) | 0 | pass: 7 dates, $9.26 |
| 2026-10-02 | vc-iowa-s1-frisky | C | Frisky | iowa_epsg5070, 2024-11-01..2025-10-31 | S1 asc | 13 | `26a5b4ac` | 33m19s | 10.8 | 0 | pass: 169 dates, $1.74, drained |
| 2026-10-02 | vc-iowa-s2-frisky | C | Frisky | iowa_epsg5070, 2024-11-01..2025-10-31, pipelined | S2 | 60 | `edb1c173` | 1h05m27s | 23.6 to August, 48.2 from September | 0 | cancelled at the slowdown: 192 dates written, $15.46, drained |
| 2026-10-02 | vc-iowa-s2-frisky-r2 | C | Frisky | iowa_epsg5070, resumed from 2025-09-20, pipelined | S2 | 60 | `be279b05` | 11m06s | 20.6 | 0 | pass: 29 dates, $2.38, drained |
| 2026-10-02 | vc-iowa-embed | C | Ray | iowa_epsg5070, to October 2025 | inference | 30 actors | `b65429de` | 13m01s | – | – | fail: no checkpoint under the experiment prefix, no actor started |
| 2026-10-02 | vc-iowa-embed-r2 | C | Ray | iowa_epsg5070, to October 2025 | inference | 30 actors, 28 at peak | `d7e25533` | 1h37m32s | 230 s/chunk median | – | pass: 394 chunks, 25.2 GPU-hours busy, $47 to $62 |
