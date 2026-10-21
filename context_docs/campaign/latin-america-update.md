# Filling Latin America's gaps in the v1.1 store, 2024–2025

In October 2026 two targeted updates filled most of the empty pixels the published v1.1 embeddings store had left
in Latin America and the Caribbean for 2024 and 2025:
- **The main update** (`latam-infill-2024-2025`) wrote 13,029 tiles across the region. It landed on 20 October
  2026 at 09:53 UTC and cost about $1,430, against an estimate of $2,500–3,700. Counting both of its attempts it
  was active for 10.6 hours, about 2.6 times the plan.
- **The islands follow-up** (`infill-latam-islands-2024-2025`) filled small islands the main update had missed or
  that the store had never covered: 743 tile-years. It landed on 21 October 2026 at 20:01 UTC, took 3 hours 46
  minutes and cost about $225.

Every planned tile was written, and every check afterwards passed. This page records why, what each update
covered, how they ran, what they cost, how the results were checked, and what we learned. How to run an update
is in [`docs/updates.md`](../../docs/updates.md).

## Why it was needed

The v1.1 store holds one embedding per 10 m pixel of land per year: 128 numbers summarising what the
Sentinel-2 (optical) and Sentinel-1 (radar) satellites saw at that spot over the year. When the store was built,
a pixel was left empty unless it had at least 15 usable optical observations that year, meaning 15 dates on
which the pixel was not hidden by cloud, shadow or missing data.

That rule left large gaps wherever the sky is often cloudy, and much of Latin America is. A gap breaks a map
and shortens a time series, which matters more to users than the slightly lower quality of an embedding built
from fewer observations. So we decided that one usable observation is enough
([ADR 028](../decisions/028-embed-every-pixel-with-an-optical-observation.md)).

Rebuilding the whole store to apply the new rule would have been very expensive. Instead we used **targeted
updates** ([ADR 027](../decisions/027-targeted-updates.md)): each re-processes only chosen tiles, computes only
their empty pixels, and leaves every pixel that already had an embedding exactly as it was.

Small islands needed the second update for two reasons:
- the main update left out remote oceanic islands by choice, and missed a few others because its outline of
  the region omitted them;
- many small islands were never in the store at all. The campaign's **land mask**, the map of which tiles
  hold land, comes from the partner's registry of land cells, and that registry leaves many cays and islets
  out. A tile the land mask does not hold is never processed, under any rule.

After the main update, the Cambridge team surveyed 5,465 islands in the region (geotessera issue #424) and
listed 346 that were still under half embedded or missing parts.

## Terms used here

| term | meaning |
|---|---|
| tile | a square of 2,048 × 2,048 pixels, about 20 km on a side, the unit the model processes |
| zone-year | one UTM zone (a 6° band of longitude) in one calendar year; the store is filled one zone-year at a time |
| land mask | the map of which tiles hold land, built from a registry of 0.1° land cells (about 11 km); the campaign's is `global` |
| observation | one date on which a pixel was clearly seen by Sentinel-2 |
| thin pixel | a pixel with 1 to 14 observations in its year, left empty by the old rule of 15 |
| gap | a pixel with no embedding in the store |
| ingest | downloading a year of satellite imagery for an area and stacking it into a mosaic, on ordinary computers |
| inference | running the model on that mosaic to produce embeddings, on GPU machines |
| GPU-hour | one GPU machine running for one hour, the unit GPU cost is billed in |

## What each update covered

### The main update

It covered core Latin America: every tile touching the World Bank's "Latin America and Caribbean" region, plus
French Guiana and the French and Dutch Caribbean, widened by about 30 km so that coastal tiles counted. Remote
oceanic islands such as the Galápagos, Easter Island and the Falklands were left out. The years were 2024 and
2025.

Across both years together, we chose the fewest tiles that held 99% of the region's thin pixels. That meant
updating every tile with at least about 16,200 thin pixels: 13,029 tiles in 40 zone-years, across 21 UTM zones
(11N to 22N and 17S to 25S), holding 8.11 billion thin pixels. About 82 million thin pixels, on tiles with fewer
than that, were left as gaps by design. Because the cutoff is one number for the whole region, dry areas were
barely touched: in north-west Mexico (zones 11N and 12N) one tile in each zone reached it in 2024 and none in 2025,
so those two zone-years of 2025 were not run. Pixels with no observation at all cannot be filled under any rule,
so they remain gaps.

The region's outline came from Natural Earth's medium-scale country boundaries, which leave out some small
islands. Four island tiles above the cutoff fell outside the region for that reason alone: La Orchila
(Venezuela, 19N `chunk_392_31`), Gorgona (Colombia, 17N `chunk_439_31`), Providencia (Colombia, 17N
`chunk_383_14`) and Grand Turk (19N `chunk_340_5`). The islands follow-up filled all four.

### The islands follow-up

It took the Cambridge list for 2024 and 2025, less the three entries below. The list falls into two groups:
- **308 islands outside the land mask**, about 314 km² of missing land: the Bahamas banks, Cuba's cays, the
  Nicaraguan banks and San Andrés cays, Belize's atolls, Chile's fjord islands and others. They needed land
  added to the mask.
- **38 islands inside the store but holding thin pixels** in tiles the main update did not choose, about
  1,950 km²: above all the Galápagos (10.8 million thin pixels in 2024, 9.8 million in 2025), with Gorgona,
  La Orchila, Providencia, Cocos, Juan Fernández, Malpelo, Trindade, South Georgia and Bermuda. They needed an
  ordinary `holes` update.

Three entries of the first group were left out, so 305 of its islands and all 38 of the second were planned.
Rocas Alijos (Mexico) has no Sentinel-2 imagery in 2024. Salas y Gómez (Chile) and the islet beside it are the
only land in zone 13S, whose years the store publishes as empty, a path no update had exercised.

The update ran under its own land mask, `latam-islands`: the partner's registry plus 273 cells covering the
islands. A cell is about 11 km across, so it can make up to four 20 km tiles live, and the new cells made 245
tiles a year live where the islands themselves touch about 156. Each new tile is processed whole, sea included.
In all: 743 tile-years in 38 zone-years across 19 zones, of which 490 were new tiles and 253 were existing tiles
holding 47.1 million thin pixels.

## How they ran

Both ran on AWS through the campaign flow, with both years processed together.

### The main update

The tiles were split between four independent groups of GPU machines, with up to 30 machines per group, and up
to 20 ingests ran at once across all four. It took two attempts with the same plan.

**The first attempt** ran for two hours on 19 October (17:57 to 20:00 UTC) and was stopped on purpose, because
ingest was running 15–65% slower than planned. Each date of imagery carried about 17 seconds of fixed overhead,
preparing the next date never overlapped with writing the current one, and the largest areas wrote one date at a
time. We changed the ingest so that preparation overlaps writing and small dates share a single write (PR #235),
and started again. The first attempt had written nothing to the published store, and its downloads were kept: 12
zone-years were complete and 28 partly done.

**The second attempt** picked them up where they had stopped. With the change, ingest ran about 1.2 times faster
per date, though that compares later dates of the same areas rather than the same dates
([`campaign-ingest-measurements.md`](../ingest/campaign-ingest-measurements.md) §3.21). It ran on 20 October
(times UTC):

| time | event |
|---|---|
| 01:19 | started |
| 01:22–01:24 | the 12 complete zone-years were recognised as done and went straight to waiting for GPUs |
| 02:22–02:53 | each group requested its GPU machines, either when all its ingests had finished or after a 90-minute wait |
| 04:14 | the last ingest finished |
| 06:29–09:52 | the four groups finished inference, one after another |
| 09:53 | the update was complete and the store was marked with its completion tag |

The 90-minute wait before a group requested its GPUs was deliberate. Without it, a group starts its machines on
the first small mosaic, runs out of work, shuts most of them down, and starts them again for the next one,
paying each start-up again. With it, 205–214 GPU-hours were paid for against 201 GPU-hours of actual work.

### The islands follow-up

It ran in one attempt on 21 October, in two groups of GPU machines with up to 40 machines each, and each group
ingested its 18 or 20 zone-years all at once. It used only the L40S GPU machine type, by choice.

| time (UTC) | event |
|---|---|
| 16:11 | the `latam-islands` land mask was built and checked against the plan |
| 16:15 | started |
| 16:19–16:20 | all 38 ingests started |
| 17:05 and 17:10 | each group requested its GPUs after a 30-minute wait, with 11 of 20 and 13 of 18 mosaics in |
| 18:07 | the last ingest finished |
| 19:45 and 20:01 | the two groups finished; the update was complete and the store marked with its completion tag |

Half the zone-years finished ingesting within 42 minutes and the fastest in 15, but the four largest (17N and 18N,
both years) took 93–108 minutes, well beyond the plan's hour.

### Why both took longer than planned

**AWS was short of L40S GPU machines both times.** During the main update, machines arrived at about one every
two minutes across four groups, and the groups never had more than 42 between them against the 120 requested.
During the islands follow-up, AWS refused 29,465 launch requests in all three of the region's data centres
between 17:00 and 20:00 UTC and granted 32; the two groups held 18 machines between them by 18:49, against the
80 requested. The same account had run over a thousand GPU machines in this region during August and September,
and its limit had not changed, so the shortage was on AWS's side. Both runs named only that one machine type. A
shortage slows a run but barely changes its cost, because machines that never start are not paid for.

**2025 took more GPU time per tile than 2024.** The main update's plan estimated each tile's GPU time from its
share of thin pixels, using test timings mostly from 2024. At the same number of gaps per tile, 2025 pixels
carried 20% more optical observations and 59% more radar observations, so their tiles took 22% longer to load and
47% longer to compute. 2025 zone-years took 1.2 to 1.8 times the estimate, 2024 ones 0.9 to 1.25 times, and the
whole update 1.23 times ([`inference-on-gpus.md`](../inference/inference-on-gpus.md) §6).

## Results against the plan

| | main update, planned | main update, actual | islands, planned | islands, actual |
|---|---|---|---|---|
| zone-years and tiles | 40 and 13,029 | 40 and 13,029, none failed | 38 and 743 | 38 and 743, none failed |
| time from start to finish | about 4.1 hours | 10.6 hours: 2 in the first attempt, 8.6 in the second | 1.2–2 hours, or 2–3 short of GPUs | 3 hours 46 minutes |
| GPU-hours of work, and GPU-hours paid for | about 160 and 186 | 201 and 205–214 | 25–46 and 39–59 | 35.4 and 37.0 |
| cost at AWS list prices | $2,500–3,700 | about $1,430 | $210–570 | about $225 |
| problems found by validation | none expected | none | none expected | none |

The islands' plan was the main update's cost method, scaled by how far that method had overestimated the main
update (1.9–2.8 times).

## Cost

| | main, first attempt | main, second attempt | islands | notes |
|---|---|---|---|---|
| ordinary machines: ingest workers, and the tasks that coordinate the run and write its results | $606 | $264 | $144 | about 19,000 processor-hours in all |
| GPU machines at $1.861 an hour | — | $382–398 | $69 | 205–214 and 37.0 machine-hours |
| GPU group coordinators and disks | — | $11 | $2.50 | |
| storage requests | $37 | $102 | about $5–10 | the islands' figure is estimated from the main update's rate |
| temporary storage of imagery and staged tiles | under $1 | $19 | under $5 | |
| **total** | **$643** | **$778–794** | **about $225** | **about $1,650 for both updates** |

These are AWS's published on-demand prices applied to measured usage. AWS's billing tool does not show dollar
amounts for this account, so the figures were built from usage:
- container usage from AWS's monitoring, which agreed with AWS's usage records to within 2% on the days both
  covered;
- each GPU machine's lifetime: for the main update from its first and last log line, for the islands from its
  launch and termination records;
- storage requests and storage from AWS's usage records, read a day after each day closed. AWS still marks
  October's figures as estimated until the month closes, and the islands' day had not yet arrived when this was
  written.

In the main update the ordinary machines, storage and requests came to about 70% of the cost and the GPUs about
30%, where the estimate had put ingest alone at 80%. The first attempt's two hours cost almost as much as the
whole second attempt, because it ran twenty full-size ingests at once. That work was not wasted: the second
attempt reused its downloads. In the islands follow-up the GPUs were busy for 96% of the hours paid for.

## How the results were checked

All of these were read-only checks against the published store and its registry, the public table summarising
every tile and year.

**The main update:**
- **Automated validation.** Each updated zone-year was checked by the same validator the original campaign
  used. None of the 40 found a blocking problem. 38 gave the same verdict on every check as the original
  campaign. The other 2, Ecuador in 2024 (zones 17N and 17S), each showed one expected difference: they now hold
  tiles the original campaign never wrote, which the stored radar summary does not cover.
- **Only the planned tiles changed.** The store keeps a tagged version from before the update
  (`update-latam-infill-2024-2025-base`) and one from after it (`-complete`). Between the two, exactly the 13,029
  planned tiles changed, in each of the 8 data layers, and nothing was created or deleted.
- **Existing pixels were left untouched.** One tile (zone 18N, 2024, `chunk_412_10`) was compared pixel by pixel:
  - its 2,947,924 pixels that already had embeddings are identical to the bit in `embeddings`, `scales` and
    `s2_obs_count`, the three layers compared;
  - 1,246,336 gaps were filled, exactly the number of thin pixels it had;
  - 44 gaps remain, exactly its pixels with no observation at all.
- **The rule is recorded where it applies.** The store-wide setting still says 15, because the rest of the store
  was built under that rule. Each updated zone-year carries a record of the update stating that it used a rule of
  one.
- **The registry matches.** All 13,029 updated tiles have new registry rows showing the rule of one and no thin
  pixels left. The other 182,825 rows for these zones in 2024 and 2025 are unchanged. The registry's own
  consistency audit passed for all 21 zones.

**The islands follow-up:**
- **The land mask before the run.** `latam-islands` was rehearsed in a test account and then built in production.
  Both times it kept every land tile of `global` and added exactly the 245 planned tiles a year, and every planned
  tile was live in it.
- **Only the planned tiles changed.** Between `update-infill-latam-islands-2024-2025-base` and `-complete`,
  exactly the 743 planned tiles changed, in each of the 8 data layers, and nothing else. Each of the 38
  zone-years carries one record of the update, at a rule of one.
- **Automated validation.** None of the 38 found a blocking problem. Every new tile reconciles: the tiles written
  plus those skipped equal the land tiles of `latam-islands`. Two non-blocking differences from the original
  campaign's verdicts were expected, because the original fill's records describe fewer tiles than the year now
  holds: its skip record's count of land tiles (36 zone-years) and its radar summary (25).
- **New and existing pixels.**
  - A new tile (zone 17N, 2024, `chunk_308_28`) went from no embeddings to 4,194,304, exactly its pixels with an
    optical observation.
  - In a Galápagos tile (zone 15S, 2024, `chunk_1_24`), the 4,194,288 pixels that already had embeddings are
    identical to the bit in `embeddings`, `scales` and the three observation counts. Its 16 gaps were filled.
- **The registry matches.** 743 new rows carry the update's name, a rule of one, and no thin pixels left. The
  registry's consistency audit passed for all 19 zones.
- **The islands themselves.** Reading the store at nine points on each island, every island of the second group
  and 99% of the points on the first group are now embedded.

## What we learned

- **An update can add land the mask leaves out, under three conditions.**
  - Its mask must be the campaign's registry plus the new cells, and nothing removed.
  - Its plan must hold every tile the new cells make live. The validator reconciles a zone-year against the
    whole mask, so a live tile left out of the plan reads as lost.
  - Whatever validates the store later must judge each year against every mask its updates ran under.
    yield-embeddings #109 made the validator and the campaign monitor read that from each update's record.
    The record holds the mask's name, not a snapshot, so `latam-islands` must never be rebuilt under that
    name; a changed mask takes a new one.
- **Adding land must also cover the existing tiles the new land touches.** The islands plan took the thin
  pixels of the second group's islands and the new tiles of the first group, but not thin pixels in existing
  tiles that first-group islands reach into. About 70,000 such pixels remain, in 54 tile-years across 13
  zone-years, on the edges of islands such as Umbrella Cay, Bonds Cay, Isla Mulchey and Isla de la Plata.
- **Plan every update for a GPU shortage.** One machine type failed both runs in three days. A shortage costs
  time rather than money, so planning both durations is enough when no deadline binds.
- **Estimate GPU time separately for each year.** The per-tile estimate was right for 2024 and too low for 2025,
  which has more observations per pixel, probably because Sentinel-2C and Sentinel-1C joined that year.
- **Large scattered ingests take longer than the per-window average says.** The islands' 17N and 18N zone-years
  took 93–108 minutes against a modelled hour or less.
- **Tiles with few gaps mostly wait on imagery.** In the main update, tiles with fewer than 250,000 gaps took 44%
  of the GPU time but spent only about a tenth of it computing. The rest went to loading the tile's full year of
  imagery, which costs the same however few pixels need filling.
  - The obvious remedy, loading only the dates the gaps can use, was measured afterwards and saves only 2–3% of
    GPU time, because the gaps in a tile are spread out and together use most dates.
  - Loading time depends on how many separate pieces of imagery are read rather than how many bytes. The larger
    gain would come from reading fewer, larger pieces.
- **Three shortcomings in the monitoring tools surfaced during the main update.**
  - The log message confirming that the second attempt correctly resumed the first attempt's downloads never
    reached the orchestration logs, so we confirmed it from the downloaded data instead.
  - The GPU profiling tool's live report stopped early unless its samplers had been started (fixed in PR #236).
  - The cost script had no prices for the ARM-based machines ingest now runs on, so it left their usage out
    (fixed in PR #239).

## Still open

- **The edge pixels the islands follow-up missed**, above. Deferred: filling them needs a third update.
- **The "before" versions of the store.** Each update's base tag pins the store as it was before the update.
  Deleting a tag only makes that version unreachable by name; it frees nothing. Freeing storage needs snapshot
  expiry and garbage collection, which can remove every older untagged version of the store, not only these. The
  two are separate decisions.
- **Whether to give the GPU setup a second machine type to fall back to.** The islands follow-up stayed on the
  L40S by choice.
- **Small islands outside Latin America.** The Cambridge survey covered this region only; the same registry gap
  almost certainly exists elsewhere.
- **Testing a larger batch size for ingest writes**, as set out in §3.21 of the ingest measurements.
- **Imagery dropped by how the ingest chooses between catalogue copies.** When a satellite pass is split into two
  pieces over one tile, the ingest can keep only one piece and lose part of that day's imagery. This affects both
  updates and the original campaign; the fix is PR #240.

## For reproducing or auditing

| item | main update | islands follow-up |
|---|---|---|
| update name (permanent: it names the tags, records and registry rows) | `latam-infill-2024-2025` | `infill-latam-islands-2024-2025` |
| plan | `s3://global-tessera-inputs/updates/latam-infill-2024-2025/plan.json` | `s3://global-tessera-inputs/updates/infill-latam-islands-2024-2025/plan.json` |
| plan settings | mode `holes`, `optical_min_obs` 1, land mask `global`, `on_skip: keep`, `min_valid_coverage` 1e-9, partial windows allowed | the same, with land mask `latam-islands` |
| land mask | `global` | `s3://global-tessera-inputs/masks/latam-islands.icechunk`, snapshot `EF2N8WWZPZED8C8FNCF0`, built from `masks/registries/latam-islands.txt` (sha256 `27739da0…89adba96`) with `verify=false` |
| run settings | `run-global-campaign` with `allow_s2_only=true`, `max_parallel_clusters` 4, `max_parallel_ingest` 20, `num_actors` 30, `overlap_years` true, `inference_start_delay_minutes` 90; the second attempt added `allow_ingest_code_mismatch=true` to resume the first attempt's downloads | `allow_s2_only=true`, `max_parallel_clusters` 2, `max_parallel_ingest` 40, `num_actors` 40, `overlap_years` true, `inference_start_delay_minutes` 30, `allow_ingest_code_mismatch=true` |
| code | first attempt: tessera-embeddings `43bbf1e2` (ingest identity `ingcode-67740a8483141eda`); second: `26136f5b`, deployed through yield-embeddings `c3e561b` | tessera-embeddings `26136f5b`, deployed through yield-embeddings `49f04599` |
| store tags | `update-latam-infill-2024-2025-base` and `-complete` | `update-infill-latam-islands-2024-2025-base` and `-complete` |
| registry check | `scripts/maintenance/rebuild_registry_measurements.py verify` over the 21 zones: 881,343 rows against 881,343 tile-years | the same over the 19 zones: 690,952 rows against 690,952 |
