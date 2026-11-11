# Re-deriving the registry measurements the v1.1 campaign never recorded

The published registry is the Parquet dataset beside the store that answers "is my area covered,
and how well" without opening a petabyte. The global campaign published **1,519,045 of its
3,247,410 rows (47%) with null in every measurement column**: they named a tile and said nothing
about it. Those columns are what an infill of refused pixels ranks shards by: how many pixels the
depth rule refused, and how close the thin ones came to the line.

**All 1,519,045 were re-derived from the store and published in the registry's `master/` on
2026-10-12**: 995 objects, `master/zone=*/year=*/part-0.parquet` plus `_common_metadata`, with
measurements on every row and `measured_by` saying whether the fill or the re-derivation produced
them. The re-derived rows live in `master/` alone; they are not published as a prefix of their own.
This is why the rows were empty, why re-deriving them is exact, the run, and what stops it
recurring. How `master/` stays current after that is
[`registry-master.md`](registry-master.md). The folder this record calls `parts/`, and a file in it
a part, is now `runs/` and a run file ([ADR 036](../decisions/036-the-registry-folder-is-runs.md)).

## Why the rows were empty

A shard's coverage record — three refusal counts, optical depth statistics, radar presence — is
built once, in `actors._coverage_record`. In the v1.1 campaign it reached the registry by one of
two routes:

```
  refused shard   ── record ──▶ <label>.skipped      on S3, survives every resume
  embedded shard  ── record ──▶ Ray result ──▶ RAM   ✗ gone the moment a leg ends
```

A resumed run's finishing leg reports earlier legs' tiles as synthetic successes carrying no
record, so `registry_rows` wrote their measurements as null — correctly, since null is how the
registry says "nothing measured this" and a zero would assert a measurement nobody took.

The campaign's parts match the mechanism exactly: **all 17,865 wholly-refused tiles were measured,
and only 53% of embedded ones were.** The gap was at cell granularity: 299 zone-years had no
measurements at all, 533 were complete, and 162 were partial.

## The re-derivation is exact, and that is provable

`registry.py` promises every column is derivable from the store. For the refusal columns that rests
on one thing being true, and it is checkable from the registry alone without reading a pixel.

The fill's optical test is `has_optical = s2_nonzero & (s2_valid_count > 0)`
(`inference/dataset.py`), where `s2_nonzero` tests reflectance bands the store does not keep. If
that term had ever removed a pixel, no store-only rebuild could reproduce the split. It never did:

```
refused_no_optical_px == eligible_px - px_with_any_optical
  → 1,728,365 of 1,728,365 measured rows, exact, zero pixels of discrepancy
```

So on the delivered product `has_optical` **is** `s2_obs_count > 0`, and because the campaign ran
`allow_s2_only = True` — `radar_rule_enforced` is `False` and `refused_no_radar_px` is `0` on every
measured row — a pixel is embedded exactly when `s2_obs_count >= optical_min_obs`. Every refusal
column reduces to counting one array.

That identity is the whole basis, so it is checked rather than remembered: `gate` evaluates it over
every measured row of the registry it is given (`registry_rebuild.basis_violations`) before it
reads a pixel, and fails on a single violation. A row with no `eligible_px` was itself re-derived
from the store, as a targeted update's merged tile is, so it is not counted.

| column | rebuild rule over the tile's `s2_obs_count` |
|---|---|
| `refused_no_optical_px` | `count(obs == 0)` |
| `refused_thin_px` | `count(0 < obs < optical_min_obs)` |
| `refused_px` | their sum |
| `px_with_any_optical` | `count(obs > 0)` |
| `obs_max` | `obs.max()` |
| `median_obs_where_any` | `median(obs[obs > 0])` |
| `median_obs_where_thin` | `median(obs[0 < obs < optical_min_obs])` |
| `px_with_any_radar` | `count(s1_asc > 0 or s1_desc > 0)` — needs both S1 arrays |
| `chunk_px` | the tile's own size |

`registry_rebuild.measurements_from_obs` is that table, mirroring `_coverage_record` field for
field including its rounding; a unit test drives both over the same array, because two
implementations of one record is how two registry rows stop being comparable.

### `eligible_px` is left null, on purpose

It is the footprint the fill's reasons were counted over, and it shrinks when the read plan cropped
a chunk in x: to the columns holding any S2 observation, when that box is at most 90% of the width
(`read_plan._chunk_read_plan`). The obs arrays are written at full width regardless, but the box is
reproducible from the stored `s2_obs_count` — the rule reproduces `eligible_px` exactly on all 237
cropped measured rows, and predicts no crop on 200 sampled uncropped rows with partial optical
coverage. A rebuilt row nonetheless leaves `eligible_px` null, the registry's own word for "not
measured", and counts over the whole tile. Applying the crop instead is a small change to
`measurements_from_obs` and an open decision; until it is taken, a rebuilt row for a cropped tile
counts its never-imaged columns as `refused_no_optical_px`.

The gap is small and bounded: **237 of 1,728,365 measured rows (0.014%)** have
`eligible_px < chunk_px`, 157 of them in 2017. `compare_row` reports such a tile as `cropped`
rather than as a mismatch, because a whole-tile rebuild of one is expected to differ.

## Running a re-derivation

`scripts/maintenance/rebuild_registry_measurements.py`. Every writing subcommand is a **dry run**
without `--write`.

```
gate      re-derive rows that ALREADY carry measurements and compare. Writes nothing, ever.
repair    rebuild master/ cells, re-deriving every row their fills left null
verify    check master/ itself, cell by cell, against parts/, and parts/ against the store
```

**The gate is the precondition, and it is checkable at scale rather than sampled on faith**: 1.73
million rows are available as ground truth, and the default `--sample 50000` spreads 39,578 tile
reads over all 695 cells that hold a measured row. `repair` also gates each cell against up to 64 of
its own measured tiles before accepting a re-derived row for it — but only tiles that saw some
optical, because in 122 of the 162 partly measured cells every measured tile is a wholly refused
one, usually never imaged, whose shard is absent and matches trivially. So 87 partly measured cells
check themselves; the other 75, and the 299 with no measured tile at all, rely on the global gate.

**The radar column can be a second pass.** `px_with_any_radar` needs both Sentinel-1
observation-count arrays. `repair --skip-radar` lands the optical answer; a later full `repair`
replaces it with rows that measured radar too, and a later optical-only one cannot undo that. Radar
shards are smaller than optical ones and often absent, so the full pass moves 1.8× the optical
pass's bytes, not 3×.

**Rehearse without credentials.** `--anonymous` reads the public store and registry unsigned, and a
non-`s3://` `--registry` is the local filesystem, so `repair --write` runs end to end against a
downloaded copy of the registry without touching the bucket. From the repository root, with the
registry copied into `./reg`:

```
S=scripts/maintenance/rebuild_registry_measurements.py
uv run python $S gate   --anonymous --registry ./reg --zones 09N --sample 100000
uv run python $S repair --anonymous --registry ./reg --zones 49S --skip-radar --write
uv run python $S verify --anonymous --registry ./reg
```

**Split a full run across processes by zone.** The per-tile decode and counting hold the GIL for
part of their time, so one process keeps only a few cores busy: about 3.6 on a laptop, and about
2.2 in-region, where it also waits on reads. The v1.1 run used eight processes on 32 vCPU, each over
its own `--zones` list. `--zones` is applied in Arrow before rows become Python objects; the whole
registry as dicts is ~6.7 GB, so each process holds only its share. Each cell is written whole, so
a failed process is re-run for its zones alone.

**Failures fail.** A missing or empty `parts/` (a mistyped `--registry`) stops every subcommand
rather than reading as a registry with nothing to check. So does a `--zones` entry the registry does
not hold, which in a zone-split run would otherwise leave that share unmeasured while every later
step passed. A cell fails `gate`, and is left untouched by `repair`, unless the store holds a time
slot for it and the store's record of the year accounts for the cell's parts
([`registry-master.md`](registry-master.md)): a refill that committed to the store but never
published its part would otherwise be re-derived from pixels its rows do not describe. In v1.1 that
record is the zone group's `runs` attribute naming the cell's one run, and every one of the 994
published cells matches.

## The v1.1 run

Commit `dff02e9e` ran on 2026-10-10 on one `c7i.8xlarge` (32 vCPU) in us-west-2, using the
subcommands of that commit. It read the public store and a copy of `parts/` anonymously, wrote to
local disk, and re-derived rows as 8 processes over zone lists balanced by null rows. The merged
master was uploaded to a staging prefix and verified there. On 2026-10-12, `master/` was copied into
`s3://tessera-embeddings/v1.1/dclimate.registry/`, and `verify` passed against the published
registry.

| step | wall time | read | result |
|---|---|---|---|
| `gate --sample 50000` | 28 min | 41 GB | 39,578 rows across 695 cells: 39,539 identical, 39 `cropped`, **0 mismatches**; the identity held on 1,728,365 of 1,728,365 rows |
| re-derivation, full pass with radar | 2 h 52 min | 1,553 GB | 1,519,045 rows across 461 cells; 0 cells blocked |
| merge into `master/`, then `verify` | under 1 min each | — | 3,247,410 rows: 1,728,365 measured by the fill, 1,519,045 by the re-derivation, 0 unmeasured |

An independent read of the published `master/` confirms 3,247,410 distinct tile-years, no null
`refused_px` or `px_with_any_radar`, and `refused_no_optical_px == chunk_px - px_with_any_optical`
on every rebuilt row. `master/` is 995 objects and 151 MB: one file per cell plus
`_common_metadata`. The run cost about $7: $5 of instance time and $1.50 of GETs.

## The proof, rehearsed first

The run was rehearsed read-only on 2026-10-08 from a laptop: anonymous reads of the public store, a
local copy of the registry, and every write to local disk. Four zone-years were chosen for what they
hold:

| zone-year | what it holds | what ran | result |
|---|---|---|---|
| 09N/2021 | fully measured, 1,743 embedded tiles | `gate` over every row, radar included | **1,743 of 1,743 identical** in every compared column |
| 06S/2019 | 75 measured tiles with optical, 37 null | re-derivation, full pass | in-cell gate 64 of 64 identical; 37 rebuilt |
| 23N/2017 | 1,063 wholly refused tiles never imaged, 339 null | re-derivation, both passes | 339 rebuilt; optical columns identical across the two passes |
| 49S/2021 | wholly null, 943 tiles | re-derivation, both passes | 943 rebuilt |

The 09N/2021 gate is the proof. A cell the fill measured completely was re-derived from the store
without reference to the fill's numbers, and it agrees on every refusal count, depth statistic and
radar count of every tile.

## What a run costs

Per embedded tile, the full pass reads about 1.0 MB in 2.5 GETs: 1,553 GB over 1,519,045 rows in
the v1.1 run. The optical-only pass reads about 0.7 MB in one GET, since `s2_obs_count` is a single
~0.5–0.75 MB shard; radar shards are smaller and often absent. An absent shard costs no request,
because Icechunk answers it from the manifest.

**Run it in us-west-2.** In-region, transfer is free and the whole repair costs a few dollars.
From a laptop the run is bound by the link: about 7 days for the optical pass and 13 for the full
one on a 1.8 MB/s link. It also costs about $290 of internet egress, billed to the **bucket owner**
because anonymous reads cannot be requester-pays. On a fresh host, set `UV_PYTHON=3.13`. Left to
itself, `uv` picked a Python release that `msgpack` publishes no wheel for, and the environment
failed to build.

## Stopping it happening again

**The embedded path persists its record.** `write_chunk` writes the coverage record as a tile
attribute, with `staged_complete` and therefore before `.done`, and `assemble_global` reads it back
from the staged tiles for any label that arrived without one, so the embedded path persists its
record exactly as the refused path always has. Because `inference_code_identity` hashes the whole
inference package, that change moved the identity: a fill on this code re-stages rather than
resuming into a prefix staged by earlier code. `ingest_code_identity` did not move, so no mosaic's
append identity changed. See [`staging-identity-and-resume.md`](staging-identity-and-resume.md).

**The gap is visible.** `scripts/diagnostic/published_registry_census.py` counts rows with no
measurements, per cell and in total, and fails above `--max-unmeasured` (zero by default). It counts
over `master/` when one exists and audits it against `parts/` cell by cell; over a registry without
a master it counts `parts/`. On the published v1.1 registry it finds no unmeasured row. The
campaign finished green with half its coverage record absent because every other check asked
whether the rows were shaped right, and none asked whether they held numbers.
