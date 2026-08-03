# Aligning the single-ROI pipeline with the global campaign

**Dated 2026-08-03.** Why the single-ROI and campaign paths were allowed to diverge,
what the divergence cost, and what remains unaligned. The current behaviour is
documented in the code and READMEs; this file is the reasoning behind the change and
the record of what was deliberately left alone.

---

## What diverged, and why

The global campaign was built alongside an existing single-ROI pipeline under an
explicit compatibility constraint (ADR-008 D8): vanilla single-ROI users were not to
be affected. So the campaign got a new store geometry and the single-ROI path kept
reproducing its historical output exactly.

That was the right call while the campaign was unproven. It stopped being the right
call once the campaign's geometry became the one with evidence behind it (ADR-008 D2,
D3, and the d3/d3v2 S3 benchmark runs), because the compatibility being preserved was
compatibility with a shape nobody had chosen on purpose.

### The old geometry

| | single-ROI (before) | campaign |
|---|---|---|
| inference tile | 2000 px | 2048 px (`SHARD_PX`) |
| embeddings chunk | `(1, 500, 500, 4)` | `(1, 256, 256, 128)` |
| shards | none | `(1, 2048, 2048, 128)` |
| obs-count codec | raw | zstd |

Three consequences:

1. **The band axis was split into 32 pieces.** Reading one pixel's 128-dimensional
   embedding cost 32 object reads. ADR-008 D2 forbids this for the campaign; the
   single-ROI path was doing it because the pre-ADR engine had.
2. **The tile did not divide the shard grid.** 2000 was chosen to divide 500, which
   was internally consistent, but it meant the two pipelines could not share a tile
   size — and the ingest chunk (4096) divided one and not the other.
3. **Assembly re-read tiles.** Northing bands are aligned to the output write
   granularity. At 500-px granularity with 2000-px tiles, a band boundary could fall
   anywhere inside a tile, so a tile straddling one was read once per band. With the
   pitch equal to the tile size, boundaries land on tile edges.

### After

One geometry, two names (`SINGLE` and `GLOBAL`), built from one definition and pinned
equal by a test. `INFERENCE_CHUNK_SIZE` is 2048 and pinned to `SHARD_PX` the same way —
as a literal, because `store_layout` imports `EMBEDDING_DIM` from `config.inference`
and the dependency can only run one way.

```
4096 ingest chunk -> 2048 inference tile -> 2048 output shard -> 256 inner chunk
```

## Compatibility

Existing stores are unaffected. A layout is read only when a store, or a variable
missing from one, is created — never to reshape an array that already exists — and the
append-safety manifest (`EmbeddingManifest`) records model and upstream identity, not
geometry. An old 500-px store keeps its geometry and still accepts appends.

A variable **added** to such a store takes the store's geometry, not the current
preset's (`_layout_matching_store`, copying chunks and shards from an existing array of
the same rank while keeping the layout's dtype, fill and codec).

That is not cosmetic, and the first draft of this change got it wrong by assuming it
was. Two reviewers caught it independently on PR #100. `_write_granularity` requires
every data variable to agree on a write granularity — two disagreeing arrays would let
separate forks share an output object — so an obs-count array created at 2048 inside a
500-px store makes the store **unassemblable**, not merely mixed:

```
ValueError: Data variables disagree on northing write granularity:
            {'embeddings': 500, 'scales': 500, 's2_obs_count': 2048}
```

and it surfaces after inference has already run. Reproduced before fixing, and pinned by
`TestVariablesAddedToAnExistingStore`.

**Still true of legacy stores:** a 2048-px tile cuts across their 500-px output chunks,
so assembly read-modify-writes boundary chunks where the old 2000-px tile (exactly
4 x 500) did not. That is a cost, not a failure — the merge is sequential within one
fork and correct. Selecting the tile size from the existing store's geometry was
considered and rejected: it would make the read tile a property of the output store
rather than of the pipeline, reintroducing exactly the per-path divergence this change
removes, to optimise appends to stores that should be re-created.

## Also aligned in the same pass

- **Manifest split.** `assemble` used `{"northing": 32, "easting": 32, "time": 1}`,
  inherited from the Dask engine and sized in 500-px chunks. Now `{"time": 1}`, matching
  the global store and the rule in `storage.zarr_store`: split the axis a single commit
  is *narrow* along. An assemble writes one timestep across the full spatial extent.
- **Preflight gates.** `num_actors < 1` and the embedding-manifest compatibility check
  both used to fail after a GPU cluster was provisioned — the manifest check after the
  entire inference had run. Both now run at flow entry, on parameters and one metadata
  read. The manifest builder and the output-store path each have one definition, shared
  by the gate and the assembly task.
- **Retired-actor instance termination.** Both campaign flows pass a
  `make_instance_terminator` into `on_actor_retire`; the single-ROI flow accepted the
  parameter through its task shell and never passed one, so idle GPU nodes billed to the
  end of the run.

## Test coverage was the reason none of this was caught

Moving the tile size and rechunking the entire single-ROI output geometry broke **zero**
tests. The campaign path has `test_fill_zone_year_flow.py`,
`test_fill_zones_sequential_flow.py` and an end-to-end in `test_zone_fill.py`; the
single-ROI path had no flow-level or pipeline-level test at all. The assembly unit tests
build their own small grids, so they assert whatever geometry they are handed.

`TestSingleRoiChainIsAligned` (in `tests/unit/test_assembly.py`) now drives enumerate →
stage → assemble at the real tile size over a 3000-px mosaic — whole shards plus a ragged
952-px edge — and asserts output chunks, shards, full band width, value placement and
fill outside the ROI. `tests/slow/` still has no occupant; the full plain-runner
end-to-end described in its README remains unwritten, and
`tests/parity/test_full_pipeline_parity.py` is still an `xfail` skeleton.

## Left unaligned, deliberately

**The single-ROI staging `run_id` is a random UUID.** The campaign derives its `run_id`
from a fingerprint of everything that determines the embeddings — window, orbit,
threshold, checkpoint filename, S2-only flag, inference code identity, and the
per-(zone,year) mosaic identity — so a retry with identical inputs lands on the same
staging prefix and resumes, while any input change starts a fresh one. The single-ROI
path generates `uuid4().hex[:12]` per run, so:

- a crashed run's staged tiles are never resumed automatically; an operator has to read
  the `run_id` out of the logs and pass `previous_run_id`;
- `cleanup_staging` only runs after success, so every failed run leaks a full staging
  prefix that nothing will ever find again.

This was not fixed here because the fingerprint needs a **mosaic build identity** the
single-ROI path does not have. The campaign hit this exact trap once already: its
`_mosaic_identity` was keyed on the ingest marker, which records *how* a mosaic was
built rather than *which build it is*, so re-ingesting under identical settings
reproduced the marker byte-for-byte and a resume could mix two mosaic revisions. It was
fixed by folding in `ingest_completed_at`. A single-ROI mosaic's `IngestManifest` carries
`roi_manifest_hash` and (for zone ROIs only) `coverage_sha256` — both policy, neither a
build identity — so the same trap is waiting.

The most promising input is the mosaic repo's current snapshot ID, which is a genuine
build identity, cheap to read, and monotonic. That is a design decision with its own
review, and it changes resume semantics on a public entry point, so it wants to be its
own change.
