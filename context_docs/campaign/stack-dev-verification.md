# The registry, update and optical-line stack: dev verification (2026-10-14)

The end-to-end check of #219 (an always-current registry `master/`), #221 (targeted updates) and
#223 (the optical line of one) together, on global-tessera-dev, before any of them merged. **Every
check passed, and the run found one defect**: the update planner could not read a signed registry
outside us-east-1, fixed in #221. It also measured #221's hole-aware reads, added during the run.
Earlier dev runs of #221 and #223 alone are recorded in
[`update-dev-verification.md`](update-dev-verification.md) and ADR 028 §4.

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

Memory scales with workers, not tiles. At about 1.4 GiB a worker, the runner's 32 workers need about
45 GiB for a fill; a `holes` merge also holds the shard it merges into, about 0.7 GB more a worker,
so at most about 67 GiB. Both are well inside the runner.

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

## Found and fixed

- **The planner could not read a signed registry outside us-east-1.** `plan_update.py from-registry`
  passed the region to Polars only for an anonymous read, so a signed scan asked us-east-1 and S3
  redirected. Fixed in #221 (`ff5bfe59`); the anonymous path, the published registry's, was fine.
