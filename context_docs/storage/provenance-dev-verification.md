# The provenance record: dev verification

The end-to-end checks of the provenance record ([ADR 035](../decisions/035-a-provenance-record-for-the-published-store.md);
PRs #253, #255 and #256) on global-tessera-dev, over four days: the first layout on 2026-10-27,
its ingest cost on 2026-10-28, on 2026-11-10 the signed planes, the copies set aside with a
status, and the days skipped whole, on twelve shards, and on 2026-11-12 `source_runs` derived from
the store's shard commits.
**Every correctness check passed.** The first run's one radar pair suggested the record costs
ingest time at zone scale; the second day's paired runs put that inside run-to-run variation.
The plan's size for the `granules` table was low by about half. The third day measured the plane
layouts: chunking all dates together would make one-pixel reads ten times faster (finding 8).

## Setup

- **Code.** The three PRs merged with `main` (TE `dev-test/provenance-on-main` at `3e89d0a3`),
  deployed from the yield-embeddings branch `dev/global-tessera-provenance`. The stack's base,
  #250, gained two commits of its own afterwards (the radar census records its window; a trim);
  the tested code predates them.
- **Image.** The dev account's PyTorch 2.5.1 image had been deregistered (finding 6), so v1.1 ran
  on the standing image from the new bake, PyTorch 2.14. The record does not depend on the
  model's numerics: the planes come from the mosaics and the validity test.
- **Area.** A mask of four 0.1° cells, `prov-test`, lighting two shards of zone 33N, `chunk_209_14`
  and `chunk_209_15`, on the Slovenia–Croatia border near 45.5°N 14.4°E. Year 2024.
- **Stores.** `prov-test`, seeded at `optical_min_obs` 40 so the fill would leave holes for an
  update to fill, and two more, `prov-test-b` and `prov-test-c`, each under its own outputs
  prefix: staging is keyed by zone, year and inputs rather than by store, so two stores under one
  prefix would share it. All three read the same mosaics.
- **Isolation.** Everything under `_prov-test*/` in the two dev buckets. Neither campaign watcher
  was scheduled, so the deliberate failures below paged no one.
- **Third day.** The stack with `main` merged, tested at #259's head (`5630851e`), deployed from
  the yield-embeddings branch `dev/global-tessera-prov-signed`, on the standing v1.1 image for the
  same reason as above. A mask of 20 cells, `prov-signed`, lighting twelve shards of zone 33N,
  `chunk_208`–`211` by `13`–`15`, around the first area. Year 2024, store `prov-signed` with no
  optical line, under `_prov-signed/`.
- **Fourth day.** #256 at `867e50fd`, where `source_runs` is derived from the store's shard commits
  in the order they landed and lives on the master only, deployed from the yield-embeddings branch
  `dev/global-tessera-prov-lineage` (yield-embeddings main plus the pin), v1.1 on the standing image.
  A mask of four cells, `prov-lineage`, lighting `chunk_209_14` and `chunk_209_15` of zone 33N, the
  first day's two shards. Year 2024, store `prov-lineage` seeded at `optical_min_obs` 40, under
  `_prov-lineage/`.

## Results

### The ingest returns the same mosaics, plus the record

Iowa, July 2024, both arms at once, one with the switch on and one off (`ingest-s2-roi-reflectance`
and `ingest-s1-roi-sar`, ascending):

| check | result |
|---|---|
| every band, on against off, 18 optical and 8 radar slots, six 512² windows each | bit-identical, all 11 optical bands, SCL, VV and VH |
| manifest | `provenance_version` 1 on the on-arm, absent on the off-arm |
| `granules` array | one non-empty list per slot: 15–25 optical granules, 31–62 radar bursts |
| `source_index` | `uint16`; every value names a granule in its slot's list |
| index zero where the band is nodata | optical: exactly. Radar: the index is never zero where VV has data, but names a granule at about one pixel in ten thousand whose VV is stored as zero (finding 3) |

### The published record

The record tested in this section and the next four had the first layout: one `granules` file per
run, holding every list that run read, and an `update_pixels` mask of the pixels a holes update was
asked to fill. The plan has since keyed `granules` by mosaic snapshot, with a `snapshot_id` on every
`shard_dates` row, and made the mask the pixels the update embedded. The third day's run, the
section after them, tested today's layout; the update mask's new meaning has unit tests only.

`fill-zones-sequential` filled 33N/2024 with the switch on (run `33N-2024-9a1fc94c`, 13 min), and
publication wrote 575 `granules` rows and 386 `shard_dates` rows over the two shards. The cell
validation passed (14 checks, 4 seam checks undecidable on two shards). 109 checks, none failed:

| check | result |
|---|---|
| each mosaic | manifest version 1; 118 optical, 61 ascending and 58 descending slots, every one with a granule list; `source_index` `uint16` |
| `shard_dates` | every row's day is a slot; every entry of `sources` is a granule of that day |
| planes, 36 sampled shard-dates over three sensors | each plane equals the mosaic's `source_index` under the model's validity test, ranked into `sources`; its non-zero count equals `valid_px` |
| imagery, 48 sampled optical pixels | the mosaic's red value equals the named granule's own file at that point, every one |
| asset URLs, 10 sampled optical | HTTP 200. Radar URLs are `s3://asf-cumulus-prod-…` and need Earthdata credentials, so were not fetched |
| registry | `source_runs` = the run, on the run file and the master (the first derivation; see the fourth day below) |

### The switch

Iowa radar stores, the window extended by ten days so a resume has work:

| run | result |
|---|---|
| switch on, into the store made with it off | refused: `provenance_version: store has '<not recorded, i.e. off>', current config has 1` |
| switch off, into the store made with it on | refused: `store has 1, current config has None`, now worded `'<not set, i.e. off>'` (finding 4) |
| switch off, into the store made with it off | resumed and appended three dates, once `allow_ingest_code_mismatch` waived the code fingerprint, which had moved since the store was made |

A store made with the switch off carries the same manifest as one made before the field existed,
so the last row stands for that case too.

### Interrupted and retried

Store `prov-test-b`, filled through `fill-zone-year` with `die_between_commits`, which ends the
process after the shard commit and before the commit that marks the year complete, and so before
publication:

| check | result |
|---|---|
| after the crash | year not complete, no tag, staging intact (1,349 objects), no provenance files, mosaics present |
| the retry, same run id | resumed both shards from staging with no inference, published, tagged `zone-33N-2024`, deleted staging |
| its record against the uninterrupted fill's | `granules` and `shard_dates` equal row for row but for `run_id`; every plane bit-identical over the two shards; the same three mosaic snapshots |
| the run record | `input_coverage.provenance_version` = 1 |

### A commit to a mosaic mid-fill

Store `prov-test-c`, filled through `fill-zone-year` (run `db2f17abe3e8`). 26 s after the fill
planned, and before any actor was ready, a commit to the shared optical mosaic left its time axis
unchanged but renamed every granule id of 9 May and zeroed one `source_index` chunk over the two
shards on that date. The mosaic was reset to its earlier snapshot afterwards.

| check | result |
|---|---|
| snapshots the record cites | the pre-commit ones (`TW7AJPY7…` for optical), not the new tip |
| renamed ids in the record | none |
| its record against fill A's | `granules` and `shard_dates` equal but for `run_id`; every plane bit-identical |
| could the test have failed? | yes: the zeroed block holds 1,489,388 non-zero plane values on 9 May, which a read of the new tip would have lost |

### Updates and `source_runs`

A holes update, `prov-holes`, at a line of one (run `33N-2024-e892e33b`, 48 min, most of it
re-ingesting the year), then a registry rebuild, then a `shards` update of `chunk_209_14`:

| check | result |
|---|---|
| store | every zone-year `done`, base and complete tags, the diff touches only the two planned shards; the 15,668 and 653,412 pixels embedded before the update unchanged in every array; 4,178,636 and 3,540,892 holes filled; `rebuild_registry_measurements verify` passes |
| the update's record | its own `granules` (577 rows: it re-ingested with a lower coverage floor and found 120 optical dates to the fill's 118) and `shard_dates` (339 rows) |
| `update_pixels` | one mask per shard; its count equals the update's `embedded_px`, and the update's planes are zero outside it |
| `source_runs` after the update's merge | `[fill, update]` on the master and on the update's run file rows; the fill's run file keeps `[fill]` |
| after `repair --write` | unchanged, `[fill, update]` |
| the `shards` update of `chunk_209_14` (run `33N-2024-0d0f2647`) | only that shard in the diff, rewritten whole (4,194,304 pixels); its own record, 404 `granules` and 150 `shard_dates` rows for the one shard, and no `update_pixels`; `source_runs` restarted at `[shards]`, while `chunk_209_15` kept `[fill, update]`; `verify` passes |

That `source_runs` came from the zone group's run records. Review of #256 found two orders those
records cannot express: a refill keeps the updates before it in `updates[year]`, so they would be
stacked on the new fill, and of two concurrent refills the `runs` record names whichever recorded
itself last, not whichever's shards landed last. #256 now derives the list from the year's tagged
shard commits, oldest first, and keeps it on the master only. The fourth day repeated the sequence
on that code (fill `33N-2024-4769823f`, 35 min; holes update `prov-lineage-holes` at a line of one,
`33N-2024-a6d4b9af`, 43 min; `shards` update `prov-lineage-shards` of `chunk_209_14`,
`33N-2024-df5eaa09`, 40 min):

| step | `chunk_209_14` | `chunk_209_15` | `verify` |
|---|---|---|---|
| fill | `[fill]` | `[fill]` | |
| holes update (4,178,636 and 3,540,984 holes filled) | `[fill, holes]` | `[fill, holes]` | passes |
| `repair --write` | unchanged | unchanged | passes |
| `shards` update | `[shards]` | `[fill, holes]` | passes |

No run file carries a `source_runs` column at any step. `ProvenanceRecord.pixel_granules` at each
step named the run the rule picks: after the holes update, the update for a pixel inside its
`update_pixels` mask and the fill for one outside it, in each shard; after the `shards` update, the
`shards` run in `chunk_209_14` and the holes update in `chunk_209_15`. A refill after updates and
concurrent refills, the two orders review found, have unit tests only.

### Signed planes, set-aside copies and skipped days

`fill-zones-sequential` filled 33N/2024 on twelve shards with the switch on (run
`33N-2024-4d08ceff`, 47 min: 22 of ingest, 11 waiting for GPUs, 38 of inference on up to two
GPUs, then 70 s of assembly).
Publication wrote 1,579 `granules` rows and 2,148 `shard_dates` rows. The cell validation passed
18 checks; its two findings, seams in the optical observation count at shard boundaries, are
non-blocking and come from where the swaths fall, not from the record. A check script read the
published record against the mosaics; nothing failed:

| check | result |
|---|---|
| `granules` statuses | optical: 255 `fused`, 50 `low_coverage`; ascending 667 and descending 607, all `fused` |
| `fusion_order` | set on every `fused` row and on no other |
| days skipped whole | 24 optical days skipped by the coverage gate, held in the mosaic; the days the record publishes off the mosaic's time axis are exactly those 24, and none lists a `fused` copy |
| slots | 119 optical, 61 ascending and 88 descending, every one with a granule list |
| fully rejected shard-dates | 43, all optical: rows with `valid_px` 0 whose planes are wholly negative |
| planes, 33 sampled shard-dates over two shards and three sensors, the rejected ones included | each equals the mosaic's `source_index` signed by the model's validity test and ranked into `sources`; its positive count equals `valid_px` |
| the reader | `ProvenanceRecord.pixel_granules_at` on a pixel of `chunk_209_14`: 121 used dates and 32 rejected ones, each with its granule |

No copy was set aside as `duplicate`, `unreadable` or `missing_bands`, and no day was skipped for a
producer conflict or a failed read, in this area and year; those paths have unit tests only.

## What it costs

### Ingest time

Each pair ran both arms at once, on separate Dask clusters, so contention falls on both alike.
Per-date time from the ingest's `Stage timings` and `S1 stage timings` lines, on against off.
The first day (2026-10-27, x86 workers of the previous image):

| per-date time, on ÷ off | Iowa, July 2024 | zone 35N, 1–7 January 2024, first pair | second pair |
|---|---|---|---|
| optical (write; gate in brackets) | 0.89 | 1.05 (1.07) | 1.17 (1.19) |
| radar, ascending | 0.95 | 1.31 | 1.30 |
| radar, descending | | 1.10 | 1.29 |

The second day (2026-10-28, ARM64 workers, the standing image), zone 35N, 1–7 January 2024, run as
nine ingests under `_prov-cost/`, with worker CPU from Container Insights:

| pair | off | on | on ÷ off | worker CPU off / on |
|---|---|---|---|---|
| optical 1, write s over six batched dates | 472 | 528 | 1.12 | 34% / 46% |
| optical 2 | 461 | 467 | 1.01 | 35% / 48% |
| optical 3, alongside three radar arms | 766 | 669 | 0.87 | 35% / 30% |
| radar ascending, batch write s | 230 | 201 | 0.87 | 42% / 41% |
| radar ascending, graph fused before the compute | 165 | 226 | 1.37 | 31% / 34% |

The plain radar ingest took 230 s in one run and 165 s in the next, same code; the dense optical
ingest's three pairs span 0.87–1.12. Every arm's workers sat at 30–48% CPU. The record's ingest
time cost is inside that spread, and the first day's +30% radar pair was one draw from it.

What the record adds for certain is tasks. Dask performance reports of the radar arms (task-stream
rows over the week; `_prov-cost/perf/`): off 38,293 and 38,500; on 49,525, and 46,012 with the
graph fused before the compute, which folds the index's `astype` into its write. The index is one
more chunk write per date: +50% writes on a two-band radar graph, +8% on a twelve-band optical one.

Writing the index inside its band's task (`pair_index_write`, branch `provenance/ingest-merged-write`)
was built and verified correct on these stores, bands and index bit-identical to the unpaired arm's,
but gave 52,250 and 52,193 rows: distributed Dask never fuses a two-input task with its parents, so
the pairing only moved the task boundary. Not shipped. Attempt 2 of it, pairing after the window
cut, wrote an all-zero index on the cluster while passing locally; its cause was not established and
its code is gone.

### Storage

| piece | measured | per zone-year |
|---|---|---|
| `pixel_sources`, optical | 1.14 and 1.22 MB a shard-year | as the plan's Appendix A (1.17) |
| `pixel_sources`, radar | 0.07–0.20 MB a shard-year per orbit | a sixth of optical or less |
| `granules`, zone 35N, one week | about 970 optical granules a day (151 B a row), 900–2,600 ascending and 1,000–2,500 descending bursts a day (93–110 B a row) | about 160 MB for a land-dense zone; the plan said under 100 MB |
| the `granules` lists inside the mosaic | 0.10–0.17 MB a day per sensor, compressed | about 150 MB, deleted with the mosaic |

A zone-day's radar list runs past 255 bursts, which is why the mosaic's index is `uint16`.

The third day measured the published signed planes of two shard-years, `chunk_209_14` and
`chunk_210_13`, in MB, compressed with the published codec (Blosc zstd 3, bit shuffle):

| shard, sensor | dates | signed, one date × one shard (published) | unsigned, as before | all dates × 256² | all dates × 128² | `int16` fusion order |
|---|---|---|---|---|---|---|
| 209_14 optical | 64 | 1.188 | 1.142 | 1.097 | 1.113 | 1.412 |
| 209_14 ascending | 61 | 0.164 | 0.164 | 0.152 | 0.160 | 0.279 |
| 209_14 descending | 28 | 0.073 | 0.073 | 0.066 | 0.072 | 0.105 |
| 210_13 optical | 64 | 0.900 | 0.867 | 0.817 | 0.837 | 1.130 |
| 210_13 ascending | 61 | 0.490 | 0.476 | 0.430 | 0.465 | 0.858 |
| 210_13 descending | 28 | 0.195 | 0.185 | 0.171 | 0.169 | 0.356 |

The sign adds about 4% on optical and 0–5% on radar. All dates × 256² is 7–12% smaller than the
published layout. The `int16` fusion order is 19–83% larger than the rank. Written to S3 both ways,
the optical planes of `chunk_209_14` gave one pixel's year in a median 1.92 s from the published
layout (64 objects) and 0.20 s from all dates × 256² (one object), over three pixels, read from a
laptop outside the region.

### Publication

The publication step took about 16 s for two shards and 31 s for twelve, including the registry
part. It copies planes eight shards at a time, so twelve shards are two batches of about 15 s; at
that rate a zone-year of about 3,000 shards takes about an hour and a half.

## Findings

1. **The record's ingest time cost is inside run-to-run variation** (above); its certain cost is
   one more chunk write per date. The first day's one radar pair read as +30%, and the plan and
   the ingest docs state the seven-pair result.
2. **The `granules` table is larger than planned.** A land-dense zone-year is about 160 MB, not
   under 100 MB; the campaign total stays in the order of 100 GB.
3. **A radar index can name a granule where VV is stored as zero.** The dB transform stores an
   amplitude too faint for its scale as zero, the store's nodata, while the index follows the
   granule's own data. The model's validity test drops such a pixel unless VH has data, so the
   published record is unaffected. Documented in `docs/ingest/provenance.md`.
4. **A refusal said `None` for a switch that was off.** Now `'<not set, i.e. off>'`, as the other
   direction already read.
5. **A fill's planes describe every pixel it was asked to embed, refused ones included.** At a line
   of 40 the fill embedded 15,668 pixels of `chunk_209_14`, but its planes name dates for all 4.19
   million. This is the plan's own rule (a refused shard still gets rows, because imagery was
   seen), but one sentence of the plan said a run's files describe only the pixels it wrote. The
   plan now states the rule once: the record says what each pixel had; the store says whether it
   was embedded.
6. **Outside this stack: the dev account's v1.1 image is gone.** `/global-tessera-dev/ray/ami-id-torch-2-5-1`
   names `ami-02fb8eda258aed9ea`, which a prod-image bake deregistered on 2026-10-26 by reusing its
   name. Any v1.1 run on dev fails at `ray up` until the parameter is repointed or the image
   rebaked.
7. **Outside this stack: ADR 034 has two claimants**, #240 (an acquisition is one datastrip) and
   #254 (UTM zone layout). This record's ADR moved to 035.
8. **Chunking all dates together reads a pixel ten times faster and is smaller.** All dates ×
   256² beat the published one date × one shard on every shard-year measured, by 7–12% in size,
   and fetched a pixel's year in one object instead of 64. Adopted, at the v2 store's chunk edge
   of 512 × 512 in 2048 × 2048 shards, so a plane chunk covers the same pixels as a store chunk.
   512 × 512 itself was not measured; 128 × 128 against 256 × 256 suggests a larger chunk is
   no bigger, and a one-pixel read decompresses four times the bytes of 256 × 256 in one fetch.
9. **The signed fusion order is 19–83% larger than the rank.** Storing it as `int16` takes
   `sources` off the pixel path, one join fewer. On this measurement the rank was kept at first;
   after review the fusion order was adopted with its size accepted (ADR 035, decision 3). These
   sizes are for one date × one shard; the published layout's is not yet measured.

## Teardown

Removed in-region with the dev-test skill's `teardown.py`, which confirmed each prefix empty:
`_prov-test/` in both dev buckets, `_prov-test-b/` and `_prov-test-c/`, including the 35N one-week
mosaics and the performance reports, and on the second day `_prov-cost/` with its fourteen mosaic
stores and reports. The optical mosaic changed in the mid-fill test was reset to
its earlier snapshot first. The shared ROI stores were only read. The branch's 11 deployments were
unregistered, and the yield-embeddings branch `dev/global-tessera-provenance` and the TE branch
`dev-test/provenance-on-main` were deleted. On the third day, `_prov-signed/` in both dev buckets,
including the layout test's planes, the 11 deployments of `dev/global-tessera-prov-signed`, and
that branch. On the fourth day, `_prov-lineage/` in both dev buckets, the 11 deployments of
`dev/global-tessera-prov-lineage`, and that branch.
