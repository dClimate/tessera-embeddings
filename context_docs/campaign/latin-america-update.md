# The Latin America update of v1.1, 2024–2025

The first targeted update of the published store ([ADR 027](../decisions/027-targeted-updates.md)), and
the first with its own optical line ([ADR 028](../decisions/028-embed-every-pixel-with-an-optical-observation.md)).
It embedded the pixels v1.1's line of 15 had refused in core Latin America, at a line of one, for 2024 and
2025. **It landed on 2026-10-20 at 09:53 UTC** (tag `update-latam-infill-2024-2025-complete`), every planned
tile written and every check passing, at about half the estimated cost and about twice the estimated time.
How to run one is [`docs/updates.md`](../../docs/updates.md); this page is what this one measured.

## In short

| | planned | actual |
|---|---|---|
| zone-years / tiles | 40 / 13,029 | 40 / 13,029, none failed |
| end to end | about 4.1 h | 8.6 h for the restart, after run 1's 2 h of ingest |
| GPU work / billed | about 160 / 186 GPU-h | 201 / 205–214 GPU-h |
| cost, list prices | $2,500–3,700 | about $1,350 |
| blocking validation findings | none expected | none |

## What ran

- **Scope.** Every tile touching the World Bank's "Latin America & Caribbean" region in Natural Earth, plus
  French Guiana and the French and Dutch Caribbean, widened by 0.3°; remote oceanic islands excluded. In each
  year, the fewest tiles holding 99% of that area's thin pixels (1–14 valid optical observations), pooled
  across the region: 13,029 tiles in 40 zone-years of 21 zones (11N–22N, 17S–25S), 8.11 B thin pixels.
- **Plan.** `holes` mode, `optical_min_obs` 1, the campaign's `global` mask, `on_skip: keep`,
  `min_valid_coverage` 1e-9, partial windows allowed.
- **Run.** One `run-global-campaign` dispatch: `allow_s2_only=true` (a `holes` update refuses without it),
  4 clusters, 20 concurrent ingests, 30 GPUs per cluster, both years in one batch, and a 90-minute GPU hold
  after each cluster's first mosaic (`inference_start_delay_minutes`). Run 1's mosaics were resumed with
  `allow_ingest_code_mismatch=true`. Code: TE `26136f5b`, YE `c3e561b`.

## How it went

**Run 1 was stopped after two hours** (2026-10-19, 17:57–20:00 UTC) because its optical ingest ran 15–65%
slower than planned: each date paid about 17 s of fixed cost, none of it overlapped with the write, and the
largest zone-years wrote one date at a time. PR #235 fixed both; the restart was the fix's first production
run, and it ingested 1.20× faster per date, on later dates of the same zone-years rather than the same
dates ([`campaign-ingest-measurements.md`](../ingest/campaign-ingest-measurements.md) §3.21). Run 1 wrote nothing to the store and left 12 complete mosaics and 28 part-built ones. The restart
resumed them under `allow_ingest_code_mismatch`, and each resumed mosaic records both ingest identities in
`mixed_ingest_code_identities`.

**The restart** (2026-10-20, UTC):

| time | |
|---|---|
| 01:19 | dispatched |
| 01:22–01:24 | the 12 complete zone-years matched their markers at once; all four clusters began the hold |
| 02:22, 02:44 | clusters B and A ended the hold early, every ingest finished; C and D at the 90-minute limit, 02:53 |
| 04:14 | the last ingest (19S-2024) finished |
| 06:29, 07:50, 08:51, 09:52 | clusters B, C, A and D completed |
| 09:53 | campaign completed and tagged |

**GPU supply set the pace.** From the first request, EC2 refused `g6e.xlarge` in every zone
(`InsufficientInstanceCapacity`; Ray tries each subnet and logs only the last, so every refusal reads as
us-west-2b). Fleets grew by about one machine every two minutes across the region and peaked at 42 against
the 120 requested. Prod's `gpu-worker-ladder` names `g6e.xlarge` only, so there was nothing to fall back to.
The same account ran up to about 8,600 on-demand G-class vCPUs in this region from 24 August to 9 September,
against a quota of 10,000 that has not changed; the shortage was AWS's.

**2025 cost more GPU than planned.** The plan priced a tile at 30 s plus 100 s × its thin share, from dev
timings mostly on 2024. At the same hole pixels per tile, 2025 carried 20% more optical observations and 59%
more radar, so its tiles took 22% longer to read and 47% longer to infer. 2025 zone-years ran 1.17–1.8× the
model and 2024 ones 0.86–1.25×, 1.23× overall ([`inference-on-gpus.md`](../inference/inference-on-gpus.md) §6).

**The hold worked as intended.** 205–214 GPU-hours were billed for 201 of tile work, 2–6% overhead against the
16% the plan expected; the slow arrival of machines also left little idle time.

## Cost

Usage measured, priced at us-west-2 on-demand list prices. Cost Explorer's cost metrics read as zero in this
account (`scripts/scoping/campaign_cost_actuals.py`), so usage is the measurement: Container Insights for
Fargate (it agreed with Cost Explorer's usage for 19 October to within 1%), each GPU machine's first and last
log line, and Cost Explorer for S3 requests.

| | run 1 | restart | |
|---|---|---|---|
| Fargate, ARM: Dask workers, fills, ingest runners | $571 | $250 | 10,690 and 4,617 vCPU-h |
| Fargate, x86: schedulers, optical flow runners | $35 | $14 | above the account's daily baseline |
| GPU, `g6e.xlarge` at $1.861/h | — | $382–398 | 205–214 h |
| Ray head nodes, EBS | — | $11 | |
| S3 requests | $37 | $42+ | 20 October was still arriving |
| **total** | **$643** | **$700–715** | **about $1,350** |

Ingest was about 70% of it and GPU about 30%; the estimate, built from the campaign's averages, had ingest
at 80%. Run 1's two hours cost almost as much as the whole restart because twenty zone-years ingested at once
on full-width fleets; the restart resumed its mosaics, so that work was not lost.

## Checks after landing

All read-only, against the published store and registry.

- **Validation:** 40 completed `validate-zone-year` runs, none blocking. 38 verdicts have exactly the
  campaign's statuses; 17N-2024 and 17S-2024 add one expected disagreement, because the radar record covers
  the tiles the campaign wrote and the year now also holds never-written Ecuador tiles the update filled.
- **The store:** the diff from `update-latam-infill-2024-2025-base` to `-complete` touches exactly the 13,029
  planned tiles in each of the eight arrays, and no group or array was created or deleted. The root keeps
  `optical_min_obs` 15; each zone-year holds one record for the update at line 1, and `cell_optical_min_obs`
  reads 1.
- **One tile, 18N-2024 `chunk_412_10`:** its 2,947,924 existing pixels are bit-identical in `scales`,
  `embeddings` and `s2_obs_count`; 1,246,336 holes were filled, exactly its earlier `refused_thin_px`; the 44
  left are its pixels with no optical observation.
- **The registry:** all 13,029 planned `master/` rows rewritten at line 1 with `refused_thin_px` 0; the other
  182,825 rows in these zones untouched. `rebuild_registry_measurements.py verify` over the 21 zones: 881,343
  rows against 881,343 tile-years in `parts/`.

## What it taught

- **Price an update's GPU per year.** The thin-share model is right for 2024 and low for 2025, whose deeper
  sequences cost more per hole pixel. Calibrate it per year before the next estimate.
- **Sparse tiles are read-bound.** Tiles with under 250,000 hole pixels took 44% of the GPU-hours, at about
  10% inference; the card waits on a year of imagery that a holes mask does not shrink, and hole-aware reads
  skipped no strip here because every tile had holes in every strip.
- **One instance type is a single point of failure for the fleet.** The ladder has no fallback.
- **Three monitoring gaps.** The "appending under `allow_ingest_code_mismatch`" warning is logged on a module
  logger Prefect does not capture, so the append is verified from the mosaic's `mixed_ingest_code_identities`
  instead. `te-observe-cluster --report` stopped early on fleets without pollers (fixed in #236).
  `campaign_cost_actuals.py` has no price for ARM Fargate, which the Dask workers have run on since the
  Graviton change, so it reports their usage as excluded.

## Open

- Expiring the base tag and collecting garbage both change the published store, so each needs its own decision.
- Add ARM Fargate prices to `campaign_cost_actuals.py`.
- Decide whether prod's ladder gets a fallback rung.
- The batch budget: a paired dev test at 16 and 20 chunks per worker (§3.21).
