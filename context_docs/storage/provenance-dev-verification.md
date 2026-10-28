# The provenance record: dev verification (2026-10-27)

The end-to-end check of the provenance record ([ADR 035](../decisions/035-a-provenance-record-for-the-published-store.md);
PRs #253, #255 and #256) on global-tessera-dev, against the dev test in the plan's §11.
**Every correctness check passed.** The run also found that the record costs ingest time at zone
scale, about 6–20% of ingest worker time, where the plan expected none, and that the plan's size
for the `granules` table was low by about half. Both are now in the plan and the ingest docs.

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
| registry | `source_runs` = the run, on the part and the master |

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
| `source_runs` after the update's merge | `[fill, update]` on the master and on the update's part rows; the fill's parts keep `[fill]` |
| after `repair --write` | unchanged, `[fill, update]` |
| the `shards` update of `chunk_209_14` (run `33N-2024-0d0f2647`) | only that shard in the diff, rewritten whole (4,194,304 pixels); its own record, 404 `granules` and 150 `shard_dates` rows for the one shard, and no `update_pixels`; `source_runs` restarted at `[shards]`, while `chunk_209_15` kept `[fill, update]`; `verify` passes |

## What it costs

### Ingest time

Each pair ran both arms at once, on separate Dask clusters, so contention falls on both alike.
Per-date time from the ingest's `Stage timings` and `S1 stage timings` lines, on against off:

| per-date time, on ÷ off | Iowa, July 2024 | zone 35N, 1–7 January 2024, first pair | second pair |
|---|---|---|---|
| optical (write; gate in brackets) | 0.89 | 1.05 (1.07) | 1.17 (1.19) |
| radar, ascending | 0.95 | 1.31 | 1.30 |
| radar, descending | | 1.10 | 1.29 |

Every radar date of the second pair was slower, by 6–40%. Iowa's workers had time to spare; a
whole zone saturates them, as a campaign does, so the zone figures are the ones to plan on.
Weighting optical at about four-fifths of ingest worker time (60 workers for 12–13 min against two
radar orbits of 13 workers for 7–8 min), the record adds about 6–20% to the ingest.

A third pair, radar ascending only, ran with Dask performance reports
(`_prov-test/perf/z35c-s1a-{on,off}.html`):

| task-seconds over the week | off | on |
|---|---|---|
| all tasks | 10,895 | 14,212 (+30%) |
| VV load (packed when on) | 3,392 | 4,309 |
| VH load (never packed) | 3,427 | 4,034 |
| chunk writes, with their count | 2,856 (4,888) | 4,293 (7,332) |

The index is a third array in every window, so there are half as many writes again; each of the
index's costs about a third of a band's (median 0.28 s against 0.80 s). The rest is the loads and
band writes running slower on workers with more to do: VH, which is never packed, slowed 18%. The
stored index is not the cost: it is 0.1% of an optical mosaic's stored bytes and 0.3% of a radar
one's.

### Storage

| piece | measured | per zone-year |
|---|---|---|
| `pixel_sources`, optical | 1.14 and 1.22 MB a shard-year | as the plan's §9 (1.17) |
| `pixel_sources`, radar | 0.07–0.20 MB a shard-year per orbit | a sixth of optical or less |
| `granules`, zone 35N, one week | about 970 optical granules a day (151 B a row), 900–2,600 ascending and 1,000–2,500 descending bursts a day (93–110 B a row) | about 160 MB for a land-dense zone; the plan said under 100 MB |
| the `granules` lists inside the mosaic | 0.10–0.17 MB a day per sensor, compressed | about 150 MB, deleted with the mosaic |

A zone-day's radar list runs past 255 bursts, which is why the mosaic's index is `uint16`.

### Publication

The publication step took about 16 s for two shards, including the registry part. It copies
planes eight shards at a time, so a zone-year of about 3,000 shards is unmeasured.

## Findings

1. **The record costs ingest time** (above). The plan and the ingest docs said it would be level;
   both now state the measured cost. Nothing was changed in the code.
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

## Teardown

Removed in-region with the dev-test skill's `teardown.py`, which confirmed each prefix empty:
`_prov-test/` in both dev buckets, `_prov-test-b/` and `_prov-test-c/`, including the 35N one-week
mosaics and the performance reports. The optical mosaic changed in the mid-fill test was reset to
its earlier snapshot first. The shared ROI stores were only read. The branch's 11 deployments were
unregistered, and the yield-embeddings branch `dev/global-tessera-provenance` and the TE branch
`dev-test/provenance-on-main` were deleted.
