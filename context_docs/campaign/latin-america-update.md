# Filling Latin America's gaps in the v1.1 store, 2024–2025

In October 2026 we filled most of the empty pixels that the published v1.1 embeddings store had left in Latin
America for 2024 and 2025. Every one of the 13,029 tiles we planned to update was written, and every check
afterwards passed. It finished on 20 October 2026 at 09:53 UTC, cost about $1,350 against an estimate of
$2,500–3,700, and took about twice as long as planned. This page records why we did it, what it covered, how it
ran, what it cost, how we checked the result, and what we learned. How to run an update like this one is in
[`docs/updates.md`](../../docs/updates.md).

## Why it was needed

The v1.1 store holds one embedding per 10 m pixel of land per year: 128 numbers summarising what the
Sentinel-2 (optical) and Sentinel-1 (radar) satellites saw at that spot over the year. When the store was built,
a pixel was left empty unless it had at least 15 usable optical observations that year, meaning 15 dates on
which the pixel was not hidden by cloud, shadow or missing data.

That rule left large gaps wherever the sky is often cloudy, and much of Latin America is. A gap breaks a map
and shortens a time series, which matters more to users than the slightly lower quality of an embedding built
from fewer observations. So we decided that one usable observation is enough
([ADR 028](../decisions/028-embed-every-pixel-with-an-optical-observation.md)).

Rebuilding the whole store to apply the new rule would have been very expensive. Instead we used a **targeted
update** ([ADR 027](../decisions/027-targeted-updates.md)): it re-processes only chosen tiles, computes only
their empty pixels, and leaves every pixel that already had an embedding exactly as it was. This was the first
targeted update of the published store, and the first to apply the lower rule.

## Terms used here

| term | meaning |
|---|---|
| tile | a square of 2,048 × 2,048 pixels, about 20 km on a side, the unit the model processes |
| zone-year | one UTM zone (a 6° band of longitude) in one calendar year; the store is filled one zone-year at a time |
| observation | one date on which a pixel was clearly seen by Sentinel-2 |
| thin pixel | a pixel with 1 to 14 observations in its year, left empty by the old rule of 15 |
| gap | a pixel with no embedding in the store |
| ingest | downloading a year of satellite imagery for an area and stacking it into a mosaic, on ordinary computers |
| inference | running the model on that mosaic to produce embeddings, on GPU machines |
| GPU-hour | one GPU machine running for one hour, the unit GPU cost is billed in |

## What it covered

The update covered core Latin America: every tile touching the World Bank's "Latin America and Caribbean" region,
plus French Guiana and the French and Dutch Caribbean, widened by about 30 km so that coastal tiles counted.
Remote oceanic islands such as the Galápagos, Easter Island and the Falklands were left out. The years were 2024
and 2025.

In each year we chose the fewest tiles that together held 99% of the region's thin pixels. That came to 13,029
tiles in 40 zone-years, across 21 UTM zones (11N to 22N and 17S to 25S), holding 8.11 billion thin pixels. About
82 million thin pixels, the 1% scattered across tiles with very few of them, were left as gaps by design.
Pixels with no observation at all cannot be filled under any rule, so they remain gaps.

## How it ran

The update ran on AWS in two attempts, with the same plan both times and both years processed together. The
tiles were split between four independent groups of GPU machines, with up to 30 machines per group, and up to 20
ingests ran at once across all four.

### The first attempt

The first attempt ran for two hours on 19 October (17:57 to 20:00 UTC) and was stopped on purpose, because
ingest was running 15–65% slower than planned. Each date of imagery carried about 17 seconds of fixed overhead,
preparing the next date never overlapped with writing the current one, and the largest areas wrote one date at
a time. We changed the ingest so that preparation overlaps writing and small dates share a single write, which
spreads that overhead (PR #235), and started again.

The first attempt had written nothing to the published store, and its downloads were kept: 12 zone-years were
complete and 28 partly done. The second attempt picked them up where they had stopped. With the change, ingest
ran about 1.2 times faster per date, though that compares later dates of the same areas rather than the same
dates (details in [`campaign-ingest-measurements.md`](../ingest/campaign-ingest-measurements.md) §3.21).

### The second attempt

The second attempt ran on 20 October (times UTC):

| time | event |
|---|---|
| 01:19 | started |
| 01:22–01:24 | the 12 complete zone-years were recognised as done and went straight to waiting for GPUs |
| 02:22–02:53 | each group requested its GPU machines, either when all its ingests had finished or after a 90-minute wait |
| 04:14 | the last ingest finished |
| 06:29–09:52 | the four groups finished inference, one after another |
| 09:53 | the update was complete and the store was marked with its completion tag |

### Why it took longer than planned

AWS was short of GPU machines. From the first request, AWS refused to supply the GPU machine type we use in
every part of the region. Machines arrived at about one every two minutes across all four groups, and the groups
never had more than 42 between them, against the 120 requested. The same account had run over a thousand GPU
machines in this region during August and September, and its limit had not changed, so the shortage was on AWS's
side. Our setup names only that one machine type, so there was nothing else to fall back to. This slowed the run
but barely changed its cost, because machines that never start are not paid for.

2025 also took more GPU time per tile than 2024. The plan estimated each tile's GPU time from its share of thin
pixels, using test timings mostly from 2024. At the same number of gaps per tile, 2025 pixels carried 20% more
optical observations and 59% more radar observations, so their tiles took 22% longer to load and 47% longer to
compute. 2025 zone-years took 1.2 to 1.8 times the estimate, 2024 ones 0.9 to 1.25 times, and the whole update
1.23 times ([`inference-on-gpus.md`](../inference/inference-on-gpus.md) §6).

### Waiting before requesting GPUs

The 90-minute wait before a group requested its GPUs was deliberate. Without it, a group starts its machines on
the first small mosaic, runs out of work, shuts most of them down, and starts them again for the next one,
paying each start-up again. With it, 205–214 GPU-hours were paid for against 201 GPU-hours of actual work, an
overhead of 2–6% against the 16% the plan expected.

## Results against the plan

| | planned | actual |
|---|---|---|
| zone-years and tiles | 40 and 13,029 | 40 and 13,029, none failed |
| time from start to finish | about 4.1 hours | 8.6 hours for the second attempt, after 2 hours of ingest in the first |
| GPU-hours of work, and GPU-hours paid for | about 160 and 186 | 201 and 205–214 |
| cost at AWS list prices | $2,500–3,700 | about $1,350 |
| problems found by validation | none expected | none |

## Cost

| | first attempt | second attempt | notes |
|---|---|---|---|
| ordinary machines: ingest workers, and the tasks that coordinate the run and write its results | $606 | $264 | about 16,200 processor-hours in all |
| GPU machines at $1.861 an hour | — | $382–398 | 205–214 machine-hours |
| GPU group coordinators and disks | — | $11 | |
| storage requests | $37 | $42 or more | the second day's count was still arriving |
| **total** | **$643** | **$700–715** | **about $1,350** |

These are AWS's published on-demand prices applied to measured usage. AWS's billing tool does not show dollar
amounts for this account, so the figures were built from usage instead:
- container usage from AWS's monitoring, which agreed with AWS's own usage records to within 1% for the day both
  covered;
- each GPU machine's lifetime from its first and last log line;
- storage request counts from AWS's usage records.

The ordinary machines and storage requests came to about 70% of the cost and the GPUs about 30%. The estimate,
built from the original campaign's averages, had put ingest alone at 80%. The first attempt's two hours cost
almost as much as the whole second attempt, because it ran twenty full-size ingests at once. That work was not
wasted: the second attempt reused its downloads.

## How the result was checked

All of these were read-only checks against the published store and its registry, the public table summarising
every tile and year.

- **Automated validation.** Each updated zone-year was checked by the same validator the original campaign
  used. None of the 40 found a blocking problem. 38 gave exactly the same results as the original campaign. The
  other 2, Ecuador in 2024 (zones 17N and 17S), each showed one expected difference: they now hold tiles the
  original campaign never wrote, which the stored radar summary does not cover.
- **Only the planned tiles changed.** The store keeps a tagged version from before the update
  (`update-latam-infill-2024-2025-base`) and one from after it (`-complete`). Between the two, exactly the 13,029
  planned tiles changed, in each of the 8 data layers, and nothing was created or deleted.
- **Existing pixels were left untouched.** One tile (zone 18N, 2024, `chunk_412_10`) was compared pixel by pixel:
  - its 2,947,924 pixels that already had embeddings are identical to the bit;
  - 1,246,336 gaps were filled, exactly the number of thin pixels it had;
  - 44 gaps remain, exactly its pixels with no observation at all.
- **The rule is recorded where it applies.** The store-wide setting still says 15, because the rest of the store
  was built under that rule. Each updated zone-year carries a record of the update stating that it used a rule of
  one.
- **The registry matches.** All 13,029 updated tiles have new registry rows showing the rule of one and no thin
  pixels left. The other 182,825 rows for these zones are unchanged. The registry's own consistency audit passed
  for all 21 zones.

## What we learned

- **Estimate GPU time separately for each year.** The per-tile estimate was right for 2024 and too low for 2025,
  which has more observations per pixel, probably because Sentinel-2C and Sentinel-1C joined that year.
  Calibrate it per year before the next update.
- **Tiles with few gaps mostly wait on imagery.** Tiles with fewer than 250,000 gaps took 44% of the GPU time
  but spent only about a tenth of it computing. The rest went to loading the tile's full year of imagery, which
  costs the same however few pixels need filling.
  - The obvious remedy, loading only the dates the gaps can use, was measured afterwards and saves only 2–3% of
    GPU time, because the gaps in a tile are spread out and together use most dates.
  - Loading time depends on how many separate pieces of imagery are read rather than how many bytes. The larger
    gain would come from reading fewer, larger pieces.
- **One GPU machine type is a single point of failure.** When AWS runs short of it, every group stalls.
- **Three shortcomings in the monitoring tools.**
  - The log message confirming that the second attempt correctly resumed the first attempt's downloads never
    reached the orchestration logs, so we confirmed it from the downloaded data instead.
  - The GPU profiling tool's live report stopped early unless its samplers had been started (fixed in PR #236).
  - The cost script had no prices for the ARM-based machines ingest now runs on, so it left their usage out
    (fixed in PR #239).

## Still open

- **Cleaning up the "before" version of the store.** Removing the pre-update version frees storage but makes
  rolling back impossible, so it needs its own decision.
- **Whether to give the GPU setup a second machine type to fall back to.**
- **Testing a larger batch size for ingest writes**, as set out in §3.21 of the ingest measurements.
- **Imagery dropped by how the ingest chooses between catalogue copies.** While measuring a follow-up, we found
  that when a satellite pass is split into two pieces over one tile, the ingest can keep only one piece and lose
  part of that day's imagery. This affects this update as well as the original campaign, and is being
  investigated separately.

## For reproducing or auditing

| item | value |
|---|---|
| update name | `latam-infill-2024-2025` (permanent: it names the tags, records and registry rows) |
| plan | `s3://global-tessera-inputs/updates/latam-infill-2024-2025/plan.json` |
| plan settings | mode `holes`, `optical_min_obs` 1, land mask `global`, `on_skip: keep`, `min_valid_coverage` 1e-9, partial windows allowed |
| run settings | `run-global-campaign` with `allow_s2_only=true`, `max_parallel_clusters` 4, `max_parallel_ingest` 20, `num_actors` 30, `overlap_years` true, `inference_start_delay_minutes` 90; the second attempt added `allow_ingest_code_mismatch=true` to resume the first attempt's downloads |
| code | tessera-embeddings `26136f5b`, deployed through yield-embeddings `c3e561b` |
| store tags | `update-latam-infill-2024-2025-base` (before) and `update-latam-infill-2024-2025-complete` (after) |
| registry check | `scripts/maintenance/rebuild_registry_measurements.py verify` over the 21 zones: 881,343 rows against 881,343 tile-years |
