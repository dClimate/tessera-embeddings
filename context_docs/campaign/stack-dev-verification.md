# The registry, update and optical-line stack: dev verification (2026-10-14)

The end-to-end check of #219 (an always-current registry `master/`), #221 (targeted updates) and
#223 (the optical line of one) together, on global-tessera-dev, before any of them merged. **Every
check passed, and the run found one defect**: the update planner could not read a signed registry
outside us-east-1, fixed in #221. It also measured #221's hole-aware reads, added during the run.
Earlier dev runs of #221 and #223 alone are recorded in
[`update-dev-verification.md`](update-dev-verification.md) and ADR 028 §4. The 2026-10-14 runs below
lowered the line by loosening a zone's rule in place (`loosen_optical_min_obs`), which was withdrawn
before merging (ADR 028 §2). The last section, from 2026-10-15, verifies what replaced it: an update
that carries its own line, on a tile the store never wrote.

## Setup

- **Code:** TE #223 at `d4ff0048`, which holds #221 at `c4414f48` and #219 at `3a861af9`, deployed
  from the yield-embeddings branch `dev/global-tessera-line-one`. The hole-aware reads comparison
  ran that branch as its control against `dev/global-tessera-hole-reads`, which holds #223 at
  `f0814c7d` (#221 at `74ee8ea6`). The procedure is yield-embeddings' `dev-test` skill.
- **Isolation:** every global-path run had its own `paths`, under `_stack-test/`, `_mock-global/`
  or `_hole-reads-{a,b}/` in both dev buckets, so masks, zone ROIs, mosaics, staging, stores and
  registries never touched shared data.

## Results

### The single-area path is unchanged

- The quickstart (Denver, one tile), run locally on `main` at `2ea97c22` and on the stack, produced
  **byte-identical output in all eight arrays**, from identical mosaics.
- On dev, the single-area flow ran `m10_parity_15S_epsg32715` (one 1024-px chunk, December 2024
  window) end to end: 1,048,576 valid pixels, the count of the August reference run.
- A single-area update (#221) was run locally through the quickstart: a 20 × 20 block of embedded
  pixels was blanked, and a `holes` update refilled exactly those 400 pixels with int8-identical
  embeddings, changing no other pixel in any array. A `shards` update rewrote the tile whole, a
  rerun was a no-op, and an update that changed the radar policy was refused by the store's manifest.

### Dense ingest: zone 35N, 1–7 January 2024

The whole-zone ROI through the ingest flows, against the October measurements in
[`campaign-ingest-measurements.md`](../ingest/campaign-ingest-measurements.md) §15:

| leg | workers | running time, 2026-10-14 | 2026-10-02 | write per date |
|---|---|---|---|---|
| S2 | 60 | 19:08 | 20:23 | 110–139 s (then 134 s) |
| S1 ascending | 13 | 9:04 | 11:16 | 62 s per date over the batch (then 56 s) |
| S1 descending | 13 | 12:05 | 13:17 | 77 s (then 77 s) |

Level. Both runs used x86 Dask workers; the account moved to ARM64 later the same day.

### The three PRs on two cloudy tiles

18N `chunk_448_1` (Colombia) and 18S `chunk_28_1` (Peru), 2023, the tiles of ADR 028 §4, in a store
seeded at 15:

1. **Baseline**, `run-global-campaign` on two clusters (35 min). 18N embedded 1,684,989 pixels and
   18S 1,679,709, and the registry rows refused 2,505,700 and 2,514,335 as thin: the published
   store's figures exactly. Each fill merged its own row into `master/`, and `verify` passed against
   the store's commits.
2. **Loosen** 18N to 1 (`loosen_optical_min_obs.py`, dry run then `--write`). The root and 18S stayed
   at 15, and 18N's history recorded the line it gave up.
3. **`holes` update** planned from the test registry (`refused_thin_px > 0`) and run on two clusters:

   | | 18N (rule 1) | 18S (rule 15) |
   |---|---|---|
   | embedded before → after | 1,684,989 → 4,190,689 | 1,679,709 → 1,679,709 (tile kept) |
   | newly embedded | 2,505,700, every thin refusal | 0 |
   | new pixels below the zone's rule | 0 | 0 |
   | holes left with an optical observation | 0 | 2,514,335 |
   | pixels that held an embedding and changed, any of the eight arrays | 0 | 0 |
   | `master/` row | the update's: `update_name = stack-holes`, rule 1, 0 thin | the fill's, unchanged |

   Both update tags were pinned, and `verify` passed: #219's store check accepts update commits and
   their parts.
4. **Catch-up.** 18N's `master/` file was deleted, as if a run had died between recording the update
   and merging the master. Re-dispatching the update through `fill-zones-sequential` found it
   recorded, merged the master back, and finished in 2 s without a commit: the store's tip stayed at
   `QDXP00XVG619V1CREHYG`.

**Cost.** On 18N, the update inferred 2.51 million hole pixels in 218 s, against the fill's 205 s
for 1.68 million: about 180 s of the tile was reading a year of imagery, which no pixel mask saves
(the hole-aware reads below address it). Assembly took 14.0 s against the fill's 9.8 s, 3.2 s of it
the merge into the existing shard.

### Mock-global: two zones, eight tiles

A 2 × 2 tile block near Minsk (35N `chunk_164_18`–`chunk_165_19`) and one on the coast of Bohol
(51N `chunk_403_20`–`chunk_404_21`), 2024, in a store seeded at 1 as the campaign now seeds them:

- **Baseline**, `run-global-campaign` on two clusters (about 50 minutes, a year of imagery):
  - every pixel of all eight tiles embedded, 16,777,216 per zone, the sea off Bohol included (water
    is a clear observation);
  - eight `master/` rows at rule 1 with no thin refusals;
  - `verify` passed.
- **`shards` update** of one tile per cell (35N `chunk_164_18`, 51N `chunk_403_21`) through
  `fill-zones-sequential` on one cluster:
  - each cell's update commit wrote only its target tile;
  - the rewritten tiles are byte-identical to the baseline in all eight arrays, re-inferred from
    freshly ingested mosaics;
  - each cell's `master/` holds the update's row for its target and the fill's rows for the other
    three tiles, and `verify` passed.

**Assembly memory**, from Container Insights on the 32-vCPU, 244 GiB assembly runner (per-minute
samples of 10–18 s assemblies, so lower bounds):

| run | workers | peak |
|---|---|---|
| fills, one tile | 1 | 0.8–0.9 GiB |
| `holes` update, one tile (merge) | 1 | 0.9 GiB |
| `shards` updates, one tile per cell | 1 | 1.0 GiB |
| mock baselines, four tiles | 4 | 1.5 and 5.6 GiB |

Memory scales with workers, not tiles, and these runs used one to four. The campaign's 32-worker
fills peaked at 93.5 GiB in production
([`what-bounds-assembly-2026-09-09.md`](../assembly/what-bounds-assembly-2026-09-09.md)); a `holes`
merge also holds the shard it merges into, about 0.9 GiB more a worker
([`update-dev-verification.md`](update-dev-verification.md)), so about 122 GiB at 32 workers, half
the runner.

### Hole-aware reads: eight tiles, paired

#221's read plan now skips the strips of a tile that hold no hole. Two stores were filled with the
same eight 18S/2024 tiles, spanning the published registry's spread of thin refusals (0.003% to 56%
of a tile), loosened to 1 and updated in `holes` mode concurrently: one on the control branch and
one on `dev/global-tessera-hole-reads`. The per-tile table is in
[`update-dev-verification.md`](update-dev-verification.md#hole-aware-reads-2026-10-14).

- **Both arms filled the same 4.0 million hole pixels**, and no pixel that already held an
  embedding changed in either.
- **Three tiles read one strip instead of two**, three of the four with under 0.5% holes; every
  denser tile has holes in all its strips. A skipped strip saves 10–15 s, so a sparse tile takes
  about 20 s instead of 30–36 s. Narrowing a strip's rows and columns saves nothing measurable.
- **Outputs:** byte-identical where the read plan did not change; where it did, inside ADR 012's
  cross-config envelope (worst-pixel cosine 0.99993), with observation counts exact.

### The update's own line, on a tile the store never wrote (2026-10-15)

The plan-carried line, and the case the Latin America update meets 18 times: a target the store
never wrote, because the line of 15 refused every one of its pixels. TE #223 at `3b5253a3`, deployed
from the yield-embeddings branch `dev/global-tessera-unwritten-tile` at #96's `b0b7548`. Zone 17S,
2024, a store seeded at 15 and a land mask of two tiles near Santo Domingo, Ecuador:

- `chunk_3_25`, the target. No pixel reaches 15 observations, so the fill skipped it and wrote no shard.
- `chunk_3_26`, its eastern neighbour, written by the fill and left out of the update's plan.

A one-tile mask does not reproduce production: with its only live tile skipped, the fill marks the
whole year empty, a different path from a published zone-year.

1. **Baseline**, `fill-zones-sequential`, 38 min: ingest 20 min, the Ray cluster and its actor 14 min,
   inference and assembly 4 min. It reported `1 staged + 1 skipped = 2/2`.
2. **Plan**, `plan_update.py from-cells --mode holes --optical-min-obs 1`, the target only.
3. **Update**, `fill-zones-sequential` with `update=`, 33 min: ingest 16 min, cluster and actor 11 min,
   inference and assembly 6 min. It logged `Optical depth rule for this fill: 1 (the store root's: 15)`.

| | `chunk_3_25`, the target | `chunk_3_26`, not in the plan |
|---|---|---|
| embedded before → after | 0 → 4,134,860 | 1,069,780 → 1,069,780 |
| embedded below the zone-year's line (1) | 0 | 0 |
| holes left with an optical observation | 0 | 3,124,404 |
| holes left with none | 59,444 | 120 |
| pixels changed, any of the eight arrays | `embeddings` and `scales` at the 4,134,860 new pixels | 0 |
| `master/` row | the update's: `optical_min_obs` 1, `refused_thin_px` 0, embedded | the fill's, unchanged (15) |

- **The store kept its rule.** The root still says 15 and the zone group has no rule attr; the
  zone-year's update record states 1, and `cell_optical_min_obs` reads 1 for 2024 and 15 for 2023.
- **The update's ingest reproduced the baseline's observation counts exactly**, under its own date
  filter (1e-9), so only `embeddings` and `scales` changed.
- **The baseline does not match the published counts exactly.** `chunk_3_25` is close (59,444 pixels
  with no optical observation, against 60,220 published); `chunk_3_26` is not (1,069,780 embedded with
  up to 31 observations, against about 62,000 with up to 20). The likely cause is the date filter, a
  share of a two-tile mask here and of the whole zone in the campaign. It does not bear on the update.
- **yield-embeddings' validator** (`audit_cell`, run from a laptop against this store) passed the
  baseline, and on the updated zone-year:

  | check | before #96's fix (`b0b7548`) | after it (`4af3cc9`) |
  |---|---|---|
  | written + skipped == live (blocking) | disagrees: 2 written + 1 skipped = 3, mask live 2 | passes: 2 + 0 = 2 |
  | skipped tiles hold no shard | disagrees: `chunk_3_25` holds one | passes |
  | published pixels honour the minimum optical depth (blocking) | passes, at the line of 1 | passes |

  The fix applies the year's update records to the fill's skip record. The one disagreement left
  does not block: the fill's `radar_coverage` record summarises one tile and the year now holds two.

## Found and fixed

- **The planner could not read a signed registry outside us-east-1.** `plan_update.py from-registry`
  passed the region to Polars only for an anonymous read, so a signed scan asked us-east-1 and S3
  redirected. Fixed in #221 (`ff5bfe59`); the anonymous path, the published registry's, was fine.

## Teardown

The four test folders were deleted in-region from both dev buckets, apart from the 35N mosaics,
kept for reuse at `s3://global-tessera-inputs-dev/_stack-test/mosaics/35n-7d-{s2,s1-asc,s1-desc}/`
(about 660 GB). The fills' and updates' mosaics had already gone with the pipeline's own cleanup.
`dev/global-tessera-hole-reads` and its deployments were removed.
