# Re-deriving the registry measurements a resumed fill never recorded

The published registry is the Parquet dataset beside the store that answers "is my area covered,
and how well" without opening a petabyte. On the global campaign, **1,519,045 of its 3,247,410 rows
(47%) carry null in every measurement column** — they name a tile and say nothing about it.

This is the code path that repairs them, and the one that stops it happening again.

## Why the rows are empty

A shard's coverage record — three refusal counts, optical depth statistics, radar presence — is
built once, in `actors._coverage_record`, and reaches the registry by one of two routes:

```
  refused shard   ── record ──▶ <label>.skipped      on S3, survives every resume
  embedded shard  ── record ──▶ Ray result ──▶ RAM   ✗ gone the moment a leg ends
```

A resumed run's finishing leg reports earlier legs' tiles as synthetic successes carrying no
record, so `registry_rows` writes their measurements as null — correctly, since null is how the
registry says "nothing measured this" and a zero would assert a measurement nobody took.

The evidence matches the mechanism exactly: **all 17,865 wholly-refused tiles are measured, and
only 53% of embedded ones are.** The gap is at cell granularity — 299 zone-years have no
measurements at all, 533 are complete, 162 partial.

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
a chunk in x. The crop lives in the read plan, not in the store, and the obs arrays are written at
full chunk width regardless — so it is not derivable and a rebuilt row leaves it null, which is the
registry's own word for "not measured". Rebuilt counts are therefore over the whole tile.

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

The compaction is not new work invented here: `registry.py` already defers deduplication of a
twice-filled cell to "the compaction step's decision", and
[`reading-the-published-store.md`](reading-the-published-store.md) §6 records the absent compacted
master and `_common_metadata` as what forces every consumer to state the schema themselves. This
closes both.

**Precedence is not latest-wins.** A fill measured pixels as it wrote them; a rebuild re-derived
them afterwards. Where a fill recorded a measurement, that measurement stands:

1. a fill row carrying measurements beats everything;
2. otherwise the most recent rebuild row carrying measurements — which is how a later pass adding
   `px_with_any_radar` supersedes an earlier optical-only one;
3. otherwise the most recent row of any kind, so a tile nothing has measured stays in the master as
   a null row rather than disappearing.

**`measured_by` is load-bearing.** It is `"fill"`, `"rebuild"` or null, and it names who measured
the numbers rather than which pass wrote the row — an unmeasured fill row is null, not `"fill"`.
Without it a re-derived value is indistinguishable from a measured one and the registry loses the
property that makes it worth trusting. `filled_at` carries the cell's original fill stamp, which a
rebuilt row would otherwise lose to its own `assembled_at`.

## Running it

`scripts/maintenance/rebuild_registry_measurements.py`. Every writing subcommand is a **dry run**
without `--write`, and a non-`s3://` registry path reads and writes locally, so the whole sequence
can be rehearsed against a downloaded copy of `parts/` before anything is published.

```
gate      re-derive rows that ALREADY carry measurements and compare. Writes nothing, ever.
rebuild   re-derive the null rows into rebuild/, one part per cell
compact   merge parts/ + rebuild/ into master/
verify    check master/ against its invariants and against parts/
```

**The gate is the precondition, and it is checkable at scale rather than sampled on faith**: 1.73
million rows are available as ground truth. `rebuild` also gates each cell against its own measured
tiles before writing a single re-derived row for it, so every cell that can prove the rebuild
reproduces its geometry and rule does so; the 299 cells with no measured tile at all rely on the
global gate.

**The radar column is a second pass by design.** `px_with_any_radar` needs both Sentinel-1
observation-count arrays, which triples the bytes moved for one informational column. Run
`rebuild --skip-radar` to land the optical answer, then `rebuild` in full; the later pass's rows
supersede the earlier ones by `assembled_at`.

### Cost

Only `s2_obs_count` is needed for every column that matters. 1,519,045 tiles at 2048² × uint16 —
one shard, one object each — is 12.7 TB decompressed, read in-region where transfer is free, with
about 1.5 M GET requests costing under a dollar. The work is integer counting: no model, no GPU, no
mosaic reads. Hours, not days.

## Why the new modules sit outside `inference/`

`storage/registry.py` is inside the inference code-identity closure — `inference_code_identity`
hashes the whole `inference` package and everything it imports, and `assembly.py` imports the
registry. Editing it would move the identity and invalidate every staged tile a future fill might
have resumed into. Nothing about a rebuild should cost that, so `registry_rebuild.py` and
`registry_master.py` are new modules that nothing in the closure imports, and they declare their
own schemas rather than extending the registry's writer.

## Stopping it happening again

The rebuild repairs what is published. The fix is in `inference/`: `write_chunk` persists the
coverage record as a tile attribute, written with `staged_complete` and therefore before `.done`,
and `assemble_global` tops up the records it was handed by reading the staged tiles for any label
missing one. The embedded path now persists its record exactly as the refused path always did.

That change **moves `inference_code_identity`**, since it hashes the inference package — a future
fill will re-stage rather than resume into a prefix staged by the previous code. `ingest_code_identity`
is untouched, so no mosaic's append identity moves. See
[`staging-identity-and-resume.md`](staging-identity-and-resume.md).

The second half is making the gap visible. `scripts/diagnostic/published_registry_census.py` gains a
completeness check that counts rows with no measurements and fails above a threshold — zero by
default. The campaign finished green with half its coverage record absent, and every other check
passed, because they all ask whether the rows are shaped right and none asked whether they hold
numbers.
