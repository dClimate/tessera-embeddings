# Re-deriving the registry measurements a resumed fill never recorded

The published registry is the Parquet dataset beside the store that answers "is my area covered,
and how well" without opening a petabyte. The global campaign published **1,519,045 of its
3,247,410 rows (47%) with null in every measurement column**: they named a tile and said nothing
about it. Those columns are what an infill of refused pixels ranks shards by: how many pixels the
depth rule refused, and how close the thin ones came to the line.

**All 1,519,045 were re-derived from the store and published on 2026-10-12.** The registry's
`master/` now carries measurements on every row, and `measured_by` says whether the fill or the
rebuild produced them. This is the code path that did it, the run, and what stops it happening
again.

## Why the rows were empty

A shard's coverage record — three refusal counts, optical depth statistics, radar presence — is
built once, in `actors._coverage_record`, and reaches the registry by one of two routes:

```
  refused shard   ── record ──▶ <label>.skipped      on S3, survives every resume
  embedded shard  ── record ──▶ Ray result ──▶ RAM   ✗ gone the moment a leg ends
```

A resumed run's finishing leg reports earlier legs' tiles as synthetic successes carrying no
record, so `registry_rows` writes their measurements as null — correctly, since null is how the
registry says "nothing measured this" and a zero would assert a measurement nobody took.

The campaign's parts match the mechanism exactly: **all 17,865 wholly-refused tiles were measured,
and only 53% of embedded ones were.** The gap was at cell granularity: 299 zone-years had no
measurements at all, 533 were complete, and 162 were partial.

## The rebuild is a re-derivation, and that is provable

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
reads a pixel, and fails on a single violation.

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

## Three prefixes, and only one of them is for reading

```
  parts/     the campaign's own output, append-only, NEVER rewritten. A consumer reading it
             today sees exactly what the campaign published: same rows, same count, same schema.
  rebuild/   measurements re-derived from the store, written per cell and keyed by run.
             Additive; rollback is deleting this prefix.
  master/    the compacted view — one row per (zone, year, tile), one schema, a dataset-level
             _common_metadata, and two columns saying where each row's numbers came from.
```

The compaction is not new work invented here. `registry.py` already defers deduplication of a
twice-filled cell to "the compaction step's decision". Without a compacted master and its
`_common_metadata`, every consumer had to state the schema themselves or risk `pyarrow` dropping a
column. The master closes both; [`reading-the-published-store.md`](reading-the-published-store.md)
§5 has the reader's recipe.

**A refill supersedes a cell wholesale.** A part is one complete run of one cell, so only each
cell's most recently assembled fill run is a candidate, and only rebuild rows re-derived against
that run — their `filled_at` is its stamp. Choosing per tile across runs would build a union no
run produced and keep an older run's numbers for a tile the store now holds from a newer one. No
published cell has two runs today; every future refill will.

**Within that run, precedence is not latest-wins.** A fill measured pixels as it wrote them; a
rebuild re-derived them afterwards. Where a fill recorded a measurement, that measurement stands:

1. a fill row carrying measurements beats everything;
2. otherwise a rebuild row carrying measurements — one that measured radar ahead of an optical-only
   one, then the most recent — so the radar pass supersedes the optical-only pass and a later
   optical-only retry cannot undo it;
3. otherwise the most recent row of any kind, so a tile nothing has measured stays in the master as
   a null row rather than disappearing.

**`measured_by` is load-bearing.** It is `"fill"`, `"rebuild"` or null, and it names who measured
the numbers rather than which pass wrote the row — an unmeasured fill row is null, not `"fill"`.
Without it a re-derived value is indistinguishable from a measured one and the registry loses the
property that makes it worth trusting. `filled_at` is when the cell was filled, on every master
row: a fill row's own `assembled_at`, and on a rebuilt row the stamp it would otherwise lose to its
own `assembled_at`.

## Running it

`scripts/maintenance/rebuild_registry_measurements.py`, four subcommands run in this order. It is
how v1.1 was repaired (next section), and how a future store is repaired after a resumed fill
until the durable fix below lands. Every writing subcommand is a **dry run** without `--write`.

```
gate      re-derive rows that ALREADY carry measurements and compare. Writes nothing, ever.
rebuild   re-derive the null rows into rebuild/, one part per cell
compact   merge parts/ + rebuild/ into master/
verify    check master/ against its invariants, against parts/, and against a fresh merge
```

**The gate is the precondition, and it is checkable at scale rather than sampled on faith**: 1.73
million rows are available as ground truth, and the default `--sample 50000` spreads 39,578 tile
reads over all 695 cells that hold a measured row. `rebuild` also gates each cell against up to 64
of its own measured tiles before writing a re-derived row for it — but only tiles that saw some
optical, because in 122 of the 162 partly measured cells every measured tile is a wholly refused
one, usually never imaged, whose shard is absent and matches trivially. So 87 partly measured cells
check themselves; the other 75, and the 299 with no measured tile at all, rely on the global gate.

**The radar column is a second pass by design.** `px_with_any_radar` needs both Sentinel-1
observation-count arrays. Run `rebuild --skip-radar` to land the optical answer, then `rebuild` in
full; the later pass's rows supersede the earlier ones by `assembled_at`. Radar shards are smaller
than optical ones and often absent, so the full pass moves 1.8× the optical pass's bytes, not 3×.

**Rehearse without credentials.** `--anonymous` reads the public store and registry unsigned, and a
non-`s3://` `--registry` is the local filesystem, so `rebuild --write` and `compact --write` run end
to end against a downloaded copy of `parts/` without touching the bucket. From the repository root,
with `parts/` copied into `./reg/parts`:

```
S=scripts/maintenance/rebuild_registry_measurements.py
uv run python $S gate    --anonymous --registry ./reg --zones 09N --sample 100000
uv run python $S rebuild --anonymous --registry ./reg --zones 49S --skip-radar --write
uv run python $S compact --registry ./reg --write
uv run python $S verify  --registry ./reg
```

**Split a full run across processes by zone.** The per-tile decode and counting hold the GIL for
part of their time, so one process keeps only a few cores busy: about 3.6 on a laptop, and about
2.2 in-region, where it also waits on reads. The v1.1 run used eight processes on 32 vCPU, each
over its own `--zones` list. `--zones` is applied in Arrow before rows become Python
objects; the whole registry as dicts is ~6.7 GB, so each process holds only its share. Re-running a
pass with the same `--run-id` overwrites the same parts, so a failed process is re-run for its zones
alone. `compact` peaks at 8.7 GB and `verify` at 12.1 GB of memory, measured over the full registry.

**Failures fail.** A missing or empty `parts/` (a mistyped `--registry`) stops every subcommand
rather than reading as a registry with nothing to check; only `rebuild/` may be absent. So does a
`--zones` entry the registry does not hold, which in a zone-split run would otherwise leave that
share unmeasured while every later step passed. A cell fails `gate` and is blocked in `rebuild`
unless the store holds a time slot for it and the zone group's `runs` attribute names the same run
as the registry: a refill that committed to the store but never published its part would otherwise
be re-derived from pixels its rows do not describe. Every one of the 994 published cells matches.
Fill runs are ordered by parsed, timezone-aware `assembled_at`, and a stamp that will not parse
stops the compaction rather than letting text order pick a run. `compact` rewrites `master/` in
place, because S3 has no rename and a pointer to a staged prefix is a protocol every reader would
have to learn: a reader during those seconds can see a partial master, `verify` fails on one, and
re-running `compact` repairs it.

## The v1.1 run

Commit `dff02e9e` ran on 2026-10-10 on one `c7i.8xlarge` (32 vCPU) in us-west-2. It read the
public store and a copy of `parts/` anonymously, wrote to local disk, and ran `rebuild` as 8
processes over zone lists balanced by null rows. The output was uploaded to a staging prefix and
verified there. On 2026-10-12, `rebuild/` and `master/` were copied into
`s3://tessera-embeddings/v1.1/dclimate.registry/`, and `verify` passed against the published
registry.

| step | wall time | read | result |
|---|---|---|---|
| `gate --sample 50000` | 28 min | 41 GB | 39,578 rows across 695 cells: 39,539 identical, 39 `cropped`, **0 mismatches**; the identity held on 1,728,365 of 1,728,365 rows |
| `rebuild`, full pass with radar | 2 h 52 min | 1,553 GB | 1,519,045 rows in 461 parts, one per cell; 0 cells blocked |
| `compact`, then `verify` | under 1 min each | — | 3,247,410 rows: 1,728,365 measured by the fill, 1,519,045 by the rebuild, 0 unmeasured |

An independent read of the published `master/` confirms 3,247,410 distinct tile-years, no null
`refused_px` or `px_with_any_radar`, and `refused_no_optical_px == chunk_px - px_with_any_optical`
on every rebuilt row. `master/` is 995 objects and 151 MB (one file per cell plus
`_common_metadata`); `rebuild/` is 461 objects and 70 MB. The run cost about $7: $5 of instance
time and $1.50 of GETs.

## The proof, rehearsed first

The run was rehearsed read-only on 2026-10-08 from a laptop: anonymous reads of the public store, a
local copy of the registry, and every write to local disk. Four zone-years were chosen for what they
hold:

| zone-year | what it holds | what ran | result |
|---|---|---|---|
| 09N/2021 | fully measured, 1,743 embedded tiles | `gate` over every row, radar included | **1,743 of 1,743 identical** in every compared column |
| 06S/2019 | 75 measured tiles with optical, 37 null | `rebuild`, full pass | in-cell gate 64 of 64 identical; 37 rebuilt |
| 23N/2017 | 1,063 wholly refused tiles never imaged, 339 null | `rebuild`, both passes | 339 rebuilt; optical columns identical across the two passes |
| 49S/2021 | wholly null, 943 tiles | `rebuild`, both passes | 943 rebuilt |

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

## Why the new modules sit outside `inference/`

`storage/registry.py` is inside the inference code-identity closure — `inference_code_identity`
hashes the whole `inference` package and everything it imports, and `assembly.py` imports the
registry. Editing it would move the identity and invalidate every staged tile a future fill might
have resumed into. Nothing about a rebuild should cost that, so `registry_rebuild.py` and
`registry_master.py` are new modules that nothing in the closure imports, and they declare their
own schemas rather than extending the registry's writer.

## Stopping it happening again

The rebuild repaired what was published; it does not stop the next resumed fill from leaving the
same gap. Two pieces close that, and only the second is part of the backfill.

**The durable fix is a separate decision.** Branch `registry/persist-coverage-record` makes
`write_chunk` persist the coverage record as a tile attribute, written with `staged_complete` and
therefore before `.done`, and has `assemble_global` read it back from the staged tiles for any label
that arrived without one — the embedded path then persists its record exactly as the refused path
always has. Because `inference_code_identity` hashes the whole inference package, that change
**moves the identity**, so a fill on the new code re-stages rather than resuming into a prefix
staged by the old. `ingest_code_identity` does not move, so no mosaic's append identity changes.
Whether that cost is worth paying now, or at the next store boundary, depends on what is staged at
the time; see [`staging-identity-and-resume.md`](staging-identity-and-resume.md). Nothing in the
backfill depends on it.

**The gap is now visible.** `scripts/diagnostic/published_registry_census.py` counts rows with no
measurements, per cell and in total, and fails above `--max-unmeasured` (zero by default). It counts
over `master/` when one exists, and fails a master that does not hold exactly the newest runs'
tile-years. Over a registry without a master it counts `parts/`. On the published v1.1 registry it
counts `master/` and finds no unmeasured row. The campaign finished green with half its coverage
record absent because every other check asked whether the rows were shaped right, and none asked
whether they held numbers.
