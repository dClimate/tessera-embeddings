# Global store implementation plan

Builds what ADR-008 decided (`context_docs/decisions/008-global-store-architecture.md`
— see its **Conclusion** for the settled architecture; every design choice below
traces to a FIRM decision D1–D9 or a measured run-1/d3/d3v2 number). Scoping is
done; this is the build.

Two additions beyond the ADR, requested at planning time:

1. **Assembly is rewritten on vanilla Zarr** — raw chunk writes into icechunk
   sessions (the region-write/store-builder pattern), replacing the Dask
   mosaic + `to_icechunk` engine. Assembly also gains sharded output.
2. **Ingest and inference chunk sizes move to powers of two** to align with the
   shard grid: `INGEST_CHUNK_SIZE` 4000 → **4096**, `INFERENCE_CHUNK_SIZE`
   2000 → **2048**.

Work is managed on `global-tessera-scoping` (same branch as the decision doc).

---

## 1. The alignment (why 4096/2048/256/2048 fit together)

```
inner chunk      256 px   (1, 256, 256, 128) int8 ≈ 8.4 MB          ← D2
shard           2048 px   8×8 = 64 inner chunks, ~0.5 GB object      ← D3
inference tile  2048 px   1 ChunkSpec == exactly 1 shard
ingest chunk    4096 px   = 2×2 inference tiles = 2×2 shards = 16×16 inner chunks

ingest store chunk (4096²)
┌───────────────┬───────────────┐
│ inference tile│ inference tile│   each tile = one staged file
│  = one shard  │  = one shard  │   = one shard object on write
├───────────────┼───────────────┤
│ inference tile│ inference tile│   GPU actors stage per tile;
│  = one shard  │  = one shard  │   assembly writes per shard, 1:1
└───────────────┴───────────────┘
```

The 1:1 ChunkSpec↔shard mapping is what makes the vanilla-Zarr assembly simple:
one worker reads one staged file and emits exactly one whole, lean shard —
shard-aligned by construction, no read-modify-write (the d3v2-verified 0.46×
write path). Inference keeps reading quarter-chunks out of ingest stores
(2048 from 4096), the same 2:1 ratio as today's 2000/4000.

## 2. Workstreams

### W1 — Constants, layout presets, zone grid

*Files: `config/ingest.py`, `config/inference.py`, new `config/store_layout.py`,
new `storage/zone_grid.py`.*

- `INGEST_CHUNK_SIZE = 4096` (`config/ingest.py:10`). Ripple: none in code
  (everything reads the constant — `s2_roi`/`s1_roi`/`feature_grid`/
  `empty_store` defaults); **behavioral**: new ROI/mosaic stores get 4096
  chunks. Existing 4000-px stores stay readable; appends to them under the new
  config are **blocked by the `RoiManifest.chunk_size` check** — intended
  (finish old campaigns on the old config or re-ingest).
- `INFERENCE_CHUNK_SIZE = 2048` (`config/inference.py:149`). Ripple:
  `ChunkSpec` grid, `_strip_height_for_density` (parameterized), staged-file
  extents. Lands together with W4 (staged layout and assembly must agree).
- **`StoreLayout` presets** (new, small): a frozen dataclass naming the output
  geometry per array — inner chunks, shards (or None), serializer/compressor,
  fill. Two presets:
  - `LEGACY` — today's `(1, 500, 500, 4)` unsharded output; remains the default
    for existing single-ROI entry points (D8: vanilla users unaffected).
  - `GLOBAL_V1` — `(1, 256, 256, 128)` int8+zstd inner chunks in
    `(1, 2048, 2048, 128)` shards for `embeddings`; `scales` float32+PCodec,
    same spatial shards (**sharded — D3**: unsharded `scales` caps the
    object-count win; PCodec-inside-shards verified locally: roundtrip + NaN
    fill OK); obs-count vars uint16, same spatial shards; `embedding_std`
    (when computed) mirrors `scales`.
- **`storage/zone_grid.py`**: the 120-zone registry — group name, EPSG
  (326xx N / 327xx S), pixel grid (10 m), northing/easting extents snapped to
  the 20,480 m shard pitch, coord-array builders. Proposed defaults (OPEN
  QUESTION #1): easting 166,000–834,000 m; northing N 0–9,331,200 m (84°N),
  S 1,105,920–10,000,000 m (80°S); group names = EPSG code strings (`"32601"`).

### W2 — Group-aware storage primitives

*Files: `storage/empty_store.py`, `storage/zarr_store.py`, `config/paths.py`.*
*Productionizes what `scripts/scale_tests/seeding.py` proved.*

- `VarSpec` gains `shards: tuple | None` (forwarded to `create_array`).
- `create_empty_store_from_coords(group=None, …)`: `group=None` keeps today's
  root-clobber semantics exactly; `group="32601"` opens root with `mode="a"`
  and `require_group`s the zone — seeding sibling groups without clobbering
  (the primitive the scale tests had to hand-roll). New
  `seed_zone_groups(repo, zones, years, layout)` on top: full 2017–2025 time
  axis (D1 — metadata-only, verified `nchunks_initialized == 0`), per-group
  coords + attrs (`crs`, `_manifest`, `years_complete: []`,
  conventions attrs from `build_convention_attrs` per zone CRS).
- Group-aware opens: `open_store`, `open_store_as_zarr_group`,
  `resolve_region`, `get_existing_dates` gain `group: str | None = None`.
  Readers target one group; never `open_datatree` the repo (run 1: 31 s vs
  0.15 s).
- Repo config for the global store: manifest split **time@1** (D4 — the
  existing `manifest_split()` contextmanager with `{"time": 1}`), preload
  tuned (`ManifestPreloadConfig(max_arrays_to_scan≈1000)` — the default 50
  never reaches later groups' coords, icechunk #1464), `repo.save_config()`
  on create.
- `BucketPaths` gains the global layout: `global_store()` → one repo URI
  (e.g. `{outputs}/global/tessera.icechunk`), groups addressed by zone name.

### W3 — Shard-aligned writer + commit discipline

*Files: new `storage/shard_writer.py`, `storage/zarr_store.py` (helpers).*
*Productionizes `store_builder.fill_year_shard_aligned` + `_workers.write_fork_shards`.*

- `write_year_shards(repo, group, year_index, shard_sources, *, n_workers, gate)`:
  cooperative fork/merge — coordinator forks the session, spawn-context workers
  each write whole shards from their assigned sources (raw zarr assignment,
  land-masked: staged inner chunks get data, the rest stays fill and the codec
  elides it), coordinator merges and makes **one commit per (zone, year)** (D6),
  updating `years_complete` in the same commit (D1). `spawn`/`forkserver` only.
- `commit_with_rebase(session, msg, *, tries)`: bounded
  `ConflictError → rebase(ConflictDetector) → retry` helper (cross-group
  commits auto-rebase cleanly — run 1 T0/T5, zero unresolvable at every N).
- **Commit gate**: `CommitGate` protocol + in-process semaphore impl, default
  cap **6** (middle of the run-1-mandated 4–8). Cross-machine gating for the
  real campaign belongs to the orchestrator (Prefect global concurrency limit
  on the commit step — OPEN QUESTION #5); the library primitive enforces it
  within a process and documents the contract.

### W4 — Assembly rewritten on vanilla Zarr (the big one)

*Files: `inference/assembly.py` (major), `inference/runner.py` (staged-layout
constant), `config/dask.py` (assembly sizing).*

**What dies:** the Dask mosaic (`_build_var_grid`, `_assemble_var_block`,
per-variant dask templates), `to_icechunk` + `align_chunks` + `split_every`,
`_commit_preserving_attrs`'s clobber dance (raw zarr writes never touch root
attrs), the Dask-cluster dependency of assembly (`AssemblyConfig` becomes
process-pool sizing; ingest keeps Dask for compute). This is the same class of
change as the removed `write_regions` batch path and the yield-embeddings
O(total-store-chunks) graph fixes — we stop building task graphs over stores
entirely.

**What replaces it:**

1. GPU actors stage per 2048² ChunkSpec as today, but staged layout becomes
   `(256, 256, 128)` full-band int8 + f32 scales (raw, uncompressed).
   `BAND_CHUNK_DIVISOR` is deleted (D2: never split the band axis).
2. `assemble()` — coordinator: open repo (layout-appropriate config), writable
   session, `fork()`; enumerate live staged ChunkSpecs → shard positions
   (1:1); partition across a `ProcessPoolExecutor` (spawn). Worker: read its
   staged file(s) into memory (~0.5 GB int8 + scales per shard), one raw zarr
   assignment per array per shard — whole, lean shards. Coordinator:
   `merge(*forks)` → attrs (conventions, run metadata, `years_complete`) →
   **one commit** via `commit_with_rebase`, behind the gate.
3. Two targets, one engine: **single-ROI mode** (create-or-append to a
   standalone store — append = `resize` + write at the new index, replacing
   `mode="a"`; `LEGACY` layout default) and **global mode** (region-write into
   a pre-allocated zone group at a year index — no resize ever, D1).
4. S3 discipline carries over: `TARGET_AGGREGATE_S3_CONCURRENCY` maps to
   `per_worker_cap = target // n_processes` passed as
   `max_concurrent_requests` per worker repo open (no `save_config` gymnastics
   — workers get it explicitly).
5. **Parity gate before the old engine is deleted:** run old vs new assembly on
   the same staged inputs (plain runner, small ROI); assert value-identical
   arrays and equivalent attrs. Recommend wholesale deletion after parity
   (OPEN QUESTION #3).

Memory envelope: ~1–1.5 GB per worker (staged read + shard block); 8 workers
≈ 12 GB — fine on current runners, far below the Dask cluster footprint.

### W5 — Campaign operations

*Files: new `storage/campaign.py` (or `orchestration/` helpers), `storage/zarr_store.py`.*

- Tags: `tag_zone_year(repo, zone, year)` after each landed commit;
  `tag_year_complete(year)` when all 120 zones land (D7 — tags protect
  snapshots from expiry).
- GC/expiry helpers: `expire_and_gc(repo, *, older_than, dry_run)` wrapping
  `expire_snapshots` + `garbage_collect` with the campaign guard (cutoff must
  predate the oldest live session; never during active fills). **Deferred
  measurement** (non-blocking): GC wall-time at 10⁸ objects before fixing the
  production cadence — reuse `t6_gc_bench` against a large repo.
- Resume/progress: `years_complete` per group + staged-file markers are the
  resume state; a `campaign_status(repo)` reader summarizing zone×year fill.
- Zone-fill runner: orchestration-facing composition — for (zone, year):
  enumerate zone ChunkSpecs → inference (existing Ray path) → `assemble()` in
  global mode → tag. The Prefect/AWS wiring lives downstream (yield-embeddings);
  the OSS side ships the callable.

### W6 — Verification & docs

- Unit tests per workstream (moto + local icechunk, as today): group seeding
  (sibling groups don't clobber; `nchunks_initialized == 0`), VarSpec shards,
  group-aware opens, shard writer round-trip + ocean-elision + edge shards,
  commit-gate/rebase behavior, assembly parity (old vs new), append-as-resize
  parity, chunk-constant ripple (`test_chunk_spec`, ingest tests at 4096).
- `scripts/scale_tests/` stays green at tiny/local as the regression harness;
  optional: point `t8`'s masked mode at the *production* writer once it exists
  (same metrics, real code path) and spot-check on S3.
- Docs: storage module docstrings + README (three write paths → four: create /
  append / region / **shard-assemble**), `context_docs` updates, ADR-008 stays
  the source of truth for *why*.

## 3. PR sequencing (each lands green on `global-tessera-scoping`)

| PR | Contents | Size | Depends on |
|---|---|---|---|
| 1 | W1: constants (`4096`), `StoreLayout` presets, `zone_grid` | M | — |
| 2 | W2: group seeding + shards in `VarSpec`, group-aware opens, paths | M | 1 |
| 3 | W3: shard writer, `commit_with_rebase`, `CommitGate` | M | 2 |
| 4 | W4: assembly rewrite + `INFERENCE_CHUNK_SIZE=2048` + staged layout + parity test | L | 3 |
| 5 | W4 cleanup: delete Dask engine + `BAND_CHUNK_DIVISOR` (post-parity) | S | 4 |
| 6 | W5: tags, GC helpers, campaign status, zone-fill runner | M | 3 |
| 7 | W6: docs sweep + scale-test wiring to production writer | S | 4,6 |

Ingest 4096 (PR 1) and inference 2048 (PR 4) land separately on purpose: the
former is standalone; the latter must move in lockstep with staged layout and
the assembly engine.

## 4. Risks

- **Attrs/conventions regression** in W4 — `to_icechunk` used to (destructively)
  own root attrs; the new engine writes attrs explicitly. Parity test asserts
  attr equivalence, not just pixel values.
- **Worker RAM** — one shard in flight per worker is the invariant (~1.5 GB);
  never batch multiple shards per worker without re-budgeting.
- **Old-store append blocking** (4096 flip) — intended, but must be in release
  notes; `RoiManifest` gives a loud, actionable error.
- **Upstream watch** — zarr PR #3004 (partial-shard reads) only helps reads;
  icechunk #1600/#1558 don't bite at our commit sizes (verified flat, run 1).
- **Cross-machine commit gating** is orchestration-level; the library cap alone
  doesn't protect a 120-job fleet (documented contract, OPEN QUESTION #5).

## 5. Open questions (tracked; answers fold into W1–W5)

1. **Zone grid spec** — exact per-zone northing/easting extents and group
   naming; partner conventions? (Proposed defaults in W1.)
2. **Annual time coordinate** — `YYYY-01-01` timestamps, or the existing
   `12mo_window_end` convention? (Affects the seeded axis + `time_convention`
   attr.)
3. **Delete the Dask assembly engine after parity, or keep behind a flag for
   one release?** (Recommend delete.)
4. **Global-store variable set** — embeddings + scales + obs counts
   (+ `embedding_std` when computed), all sharded? (Assumed yes.)
5. **Campaign orchestration home** — Prefect (yield-embeddings) global
   concurrency limit as the cross-machine commit gate? (Assumed yes.)
6. **Single-ROI default layout** — keep `LEGACY` as the default for existing
   entry points (strict D8), with `GLOBAL_V1` opt-in? (Assumed yes.)
