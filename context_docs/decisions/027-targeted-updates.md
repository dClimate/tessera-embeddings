# 027 — Targeted updates re-run chosen shards of filled cells, beside the original fill

**Status:** Accepted 2026-10-12 (repo owner). Built on `feat/targeted-updates`, stacked on the
registry PR that makes `master/` the one always-current registry (#219).

The global store is filled one UTM zone-year at a time, and a filled cell was built never to be
touched again. That is right for a campaign and wrong for what comes after it: the published v1.1
store has refused pixels in 36.8% of its shards, and more imagery, a fixed ingest, a model fix or a
different land mask will each call for re-running *part* of a filled cell. This records how that is
done without disturbing anything else.

---

## 1. Decision

**An update is a named, immutable plan.** It lists cells (`"14N/2019"`) and, per cell, the shards to
re-run, or every live shard; it names the land mask its shards were chosen under, its mode, and the
ingest settings it needs. It is written once to `{inputs}/updates/<name>/plan.json`
(`storage/update.py`) and every flow that runs it reads it from there, so a run is identified by
its name alone. A plan comes from the registry's `master/` (a SQL `where` and/or an AOI) or from an
explicit list (`scripts/update/plan_update.py`).

**Two modes, because there are two reasons to re-run a shard.**

| mode | what is inferred | what is written | for |
|---|---|---|---|
| `holes` | only the pixels that are holes in the store now (`scales` NaN) | those pixels merged into the shard, all eight arrays | more imagery for refused pixels |
| `shards` | every valid pixel | the whole shard, every array | a model or defect fix |

`holes` is the cheap one. The model embeds each pixel independently and the dataset batches only
valid pixels, so a pixel excluded by the hole mask costs the GPU nothing; what remains per tile is
reading its imagery. A hole pixel takes every array the new run produced, its observation counts and
month coverage included, even where it is still refused, so the store shows the newest imagery for
every pixel the update touched. A pixel that already held an embedding keeps the counts that
embedding was built from: the per-pixel provenance stays exact.

**`holes` runs under the campaign's radar policy** (`allow_s2_only`). A merged shard's registry rows
are re-derived from the store, which records whether a pixel had radar but not which pixels a radar
rule refused, so they could not describe a run that enforced one. Both drivers refuse the
combination before any ingest.

**A target shard where nothing new embeds is kept, not blanked.** The campaign clears a skipped tile
to fill, so that a retry cannot leave an earlier attempt's data under its completion mark; an update
has no such attempt to fear, and blanking would lose good data to a bad run. A `shards` plan may set
`on_skip: "clear"` for the case where the old values are the defect; it refuses to run if it lists
a tile its mask has no land in, since clearing never reaches such a tile. In `holes` mode a kept
tile is not written at all, so where none of a tile's holes embeds, its still-refused pixels keep
their old observation counts rather than taking the new run's. Under a line of one optical
observation (#223) such a pixel has no optical observation before or after, so nothing stale remains
that drives refusal or planning.

**The update records itself beside the fill, never over it.** `years_complete`, `runs[year]` and the
zone-year tags are untouched. Each cell gains one record in its zone group's `updates[year]` list:
the update, its mode and run, its code, its mask, its shard `commit`, and which shards it
`written`, `kept` and `cleared`. A cell is therefore `runs[year]` with `updates[year]` applied on top, in order, and an
update is done on a cell exactly when that cell holds its record — which is what makes a rerun free.

**Two tags per update, none per cell.** `update-<name>-base` pins the store before the update's first
write, for comparison and rollback; `update-<name>-complete` pins it once every planned cell holds
the record. A cell the campaign never filled is refused: an update inserts into existing data.
`run-global-campaign` returns such cells as `never_filled`, and the update gets no complete tag
until they are filled and updated.

**Everything an update writes outside the store is its own.** Its ROI, mosaics and hole masks live
under `{inputs}/updates/<name>/`, its staging under `{outputs}/staging/updates/<name>/<store>/`. The
campaign rewrites a zone's shared ROI whenever its coverage changes and sweeps the mosaics of complete
cells, and every cell an update touches is complete. In an update run, `sweep_orphan_mosaics` sweeps
the leftover mosaics of the update's own recorded cells, under its root, and never the campaign's.

**Hole masks and staging are keyed by destination store** (`<store>`: the first 12 hex digits of the
SHA-256 of its URI). A `holes` update infers only the pixels that are holes in its own store, so if
one plan run against two stores shared either, a retry could resume tiles inferred for the other
store's holes, write nothing into its own, and record the cell as done.

**Partial-zone ingests must not be blocked by settings sized for whole zones.** The date threshold
`min_valid_coverage` is a percentage of the ROI's pixels, so on an update's small ROI the campaign's
0.1 drops cloudy dates a zone-wide ingest kept. Plans default to `1e-9` — one valid pixel passes —
and to `allow_partial_window`, since a small cloudy ROI can lack a valid date in a month a whole zone
never did. Both are plan fields, overridable.

**The registry describes each tile as it now stands.** Update parts land in `parts/` like any fill's
and #219's per-tile merge keeps `master/` current, a tile's newest row winning. Rows carry a new
`update_name` column (null for the campaign). A merged shard holds two runs' pixels, so its rows are
re-derived from the committed shard rather than taken from either run's record. An update's shard
commit carries #219's run tag like a fill's, and its part is stamped with that commit, so #219's store
check covers updates unchanged; an update that wrote nothing made no shard commit and has no part.
`master/_common_metadata` is
rewritten when its schema falls behind, so the new column is visible to readers that take the schema
from it, and the planner reads `master/` with the declared schema, so cell files from before and
after the column read together.

**Both drivers run updates.** `fill-zones-sequential` (one cluster) and `run-global-campaign`
(chained clusters) take `update=<name>`; the second balances its clusters on the plan's shards.

**A single-area store takes updates too** (`orchestration/runners/plain.py`). An `update` block in
the plain runner's config (`name`, `mode`, optional `tiles`) rewrites part of a window the store
already holds, through the same shard writer at the store's root: `holes` merges only the pixels
with no embedding, `shards` rewrites the listed tiles whole, and every other pixel is left as it was.
The record goes in the root's `updates[year]`, the year being the window's last, and
`update-<name>-base` pins the store first. There is no plan file, registry or complete tag, since one
run is the whole update. A single-area store's last row and column of tiles can be narrower than a
shard, so the shard writer takes each tile's size from the store's edge (`tile_extent`). The update
infers from the area's mosaics as they stand; ingest resumes them rather than redoing them, so taking
in new imagery means deleting the mosaics first.

## 2. Rejected alternatives

- **`runs[year]` as an append-only list** (sketched in
  [`minimum-optical-depth.md`](../inference/minimum-optical-depth.md) §10). Right in a store nobody
  reads yet; the published v1.1 store has `runs[year]` as a dict, so changing its type would break
  every reader. `updates[year]` adds the history without changing anything that exists.
- **Merging pixel by pixel in every mode.** For a model fix the old values are the problem, and a
  whole-shard rewrite is cheaper than a merge. The merge is reserved for `holes`, where the point is
  that existing pixels are not touched.
- **Whole-shard replacement only.** Re-embedding a shard to fill its holes pays for every embedded
  pixel again. Across the 1.19 million v1.1 shards with any refused pixel, holes average 7% of the
  pixels, so nearly all of that cost buys nothing.
- **Reopening the cell** (`yield-embeddings/scripts/reopen_zone_year.py`). Clears the completion mark,
  rewrites every live shard, fails by design at the write-once zone-year tag and loses the original
  run's provenance. A plan with `cells: {"<zone>/<year>": null}` in `shards` mode does the same work
  without any of that.
- **Campaign paths for update artifacts.** Saves a parameter and lets the campaign's ROI rewrite and
  mosaic sweep collide with an update in flight.

## 3. Consequences

- **`inference_code_identity` moves** with this change (the dataset, actors, assembly, shard writer
  and registry schema). `ingest_code_identity` does not: the plan, the paths and the shard narrowing
  sit outside both closures, and a test keeps them there.
- **A `holes` merge reads the shard it writes.** Assembly reads each merged shard back before
  replacing its holes, about 700 MiB logical across the eight arrays, one variable at a time:
  measured at about 3 s and 0.9 GiB per merging worker, well inside the assembly runner
  ([`update-dev-verification.md`](../campaign/update-dev-verification.md)).
- **A `holes` tile costs at least its reads.** The GPU skips every pixel outside the mask, but the
  tile's imagery is still read and resampled: about 30–40 s a tile on dev, against 52–169 s for a
  full re-run of the same tiles.
- **The base tag retains the replaced shards** against any future garbage collection. Deleting it
  once an update is verified releases them.
- **One update per cell at a time.** Two updates of the same cell conflict on its chunks; the
  `updates` record itself retries like any attr commit.
- **`runs[year]` no longer describes every pixel of an updated cell.** Its `optical_skips` and
  coverage summaries are the original fill's; the update's own are on its record.
