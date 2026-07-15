# Global store implementation plan

> **Revision note:** some W2/W3 API sketches below were superseded during the
> build's cleanup pass — `VarSpec.shards` and `create_empty_store_from_coords(group=)`
> were dropped as unused (the global store seeds via `ArrayLayout`/`StoreLayout`),
> and the `CommitGate` protocol + semaphore class became a plain
> `threading.Semaphore` behind a context-manager type alias. The code is the map.

> **Status (2026-07-14): built.** Stage A landed as commits 9367360 (W1),
> a66ec12+0513edd (W2), 7528051 (W3), aa50d27 (W5); Stage B as ef33f8e (W4
> new engine + parity gate), bd7abfe (W4 finalize — Dask engine deleted,
> assembly on worker processes), 4541c02 (zone-fill runner + empty-cell
> marking), plus the W6 docs sweep. The parity gate (legacy Dask engine vs
> raw-zarr engine, value/coord/attr-identical on create and append) passed
> before the deletion. All on `global-tessera-scoping`, one PR to main.

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
  the 20,480 m shard pitch, coord-array builders. **Extents are derived from an
  authoritative source, not hand-typed (Q1, resolved):** each EPSG:326xx/327xx
  CRS carries its official area-of-use in the EPSG registry, which we already
  ship via `pyproj` (`CRS.area_of_use` → project → snap outward to shard
  pitch). A small generator script emits the registry table into
  `zone_grid.py`, and a pinning test asserts the values against pyproj so an
  EPSG-database bump can't silently move the grid. (This is the "reference UTM
  grid from an authoritative source" — the EPSG dataset itself; it needs no
  vendored GeoJSON. If the partner later supplies a specific grid file, the
  generator regenerates from that instead.) Expected values ≈ easting
  166,000–834,000 m; northing N 0–9,331,200 m (84°N), S 1,105,920–10,000,000 m
  (80°S). Group names = EPSG code strings (`"32601"`).
  **Boundary policy (decided):** pure nominal 6° longitude bands — disjoint
  coverage, every pixel-center belongs to exactly one zone; the Norway/Svalbard
  MGRS width exceptions (32V, 31X–37X) are **not** honored (they exist for
  navigation, not data grids, and would make zone extents irregular for no
  storage benefit). This deviation from MGRS expectations MUST be documented
  user-facing: in the `zone_grid` module docstring, the storage README section
  on the global layout, and the dataset-level attrs
  (`zone_scheme: "utm_6deg_nominal"`) so downstream consumers can't misassume
  MGRS behavior (W6 carries the docs task).
- **Annual time axis (Q2, resolved):** one timestep per calendar year,
  timestamped `YYYY-01-01T00:00:00` (int64 ns, `TIME_ENCODING`), with a
  group-level `time_convention: "calendar_year"` attr replacing the per-run
  `12mo_window_end` convention for the global store.

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
  on the commit step — confirmed home, Q5); the library primitive enforces it
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
   arrays and equivalent attrs. The old engine is then **deleted outright**
   (Q3, resolved) — no compatibility flag.

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
  enumerate zone ChunkSpecs against the **partner-supplied campaign land mask**
  (an input to the runner, same contract as today's ROI-mask zarr; assumed
  delivered — no fallback mask is built) → inference (existing Ray path) →
  `assemble()` in global mode → tag. The Prefect/AWS wiring lives downstream
  (yield-embeddings); the OSS side ships the callable.

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
  the source of truth for *why*. Includes the **zone-boundary-policy note**
  (pure 6° bands, no Norway/Svalbard MGRS exceptions) in the zone_grid
  docstring, the README global-layout section, and the `zone_scheme` attr —
  a user-facing deviation from MGRS expectations that must not be discoverable
  only by reading code.

## 3. Staging (two stages of commit-milestones on `global-tessera-scoping`; one eventual PR to main)

The workstreams are *linearly* dependent (later builds on earlier, never the
reverse), so they don't need bundling — but seven separate PRs was ceremony.
Development follows the pattern that worked for `scripts/scale_tests/` (M0–M5):
verified commit-milestones on this branch, each landing green with its tests,
merged to `main` as a single PR at the end. Two stages, because exactly two
boundaries carry real risk-attribution value:

| Stage | Contents (workstreams) | Why it's a boundary |
|---|---|---|
| **A — foundations** | W1 constants (`INGEST_CHUNK_SIZE=4096`) + `StoreLayout` presets + zone grid; W2 group seeding, `VarSpec.shards`, group-aware opens, paths; W3 shard writer, `commit_with_rebase`, `CommitGate`; W5 tags, GC helpers, campaign status, zone-fill runner | Purely **additive** — no existing behavior touched except the 4096 flip (one line, called out in its commit; affects new ingests independently of assembly, so it must be attributable on its own) |
| **B — assembly rewrite** | W4: vanilla-Zarr engine + `INFERENCE_CHUNK_SIZE=2048` + staged layout `(256,256,128)` + parity test vs. the old engine, then **delete** the Dask engine + `BAND_CHUNK_DIVISOR` in the same stage once parity passes; W6 docs sweep + scale-test wiring | The one change that **replaces** verified behavior — isolated on top of already-verified foundations so a parity regression's suspect surface is the rewrite diff alone, and a revert is clean |

Inference 2048 rides with Stage B (it must move in lockstep with staged layout
and the engine); ingest 4096 rides with Stage A (standalone). Deleting the old
engine inside Stage B is safe on a feature branch — main sees introduce-and-
delete as one merge, and the parity test is the gate, not elapsed time.

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
  doesn't protect a 120-job fleet (documented contract; Prefect global
  concurrency limit owns fleet-level gating — Q5, resolved).

## 5. Questions — resolved 2026-07-14

1. **Zone grid spec** → derive from an authoritative source: the **EPSG
   registry** (each zone CRS's official area-of-use, via `pyproj`), projected
   and snapped to shard pitch; generator + pinning test; no vendored grid file
   unless the partner supplies one. Group names = EPSG code strings.
2. **Annual time coordinate** → `YYYY-01-01` calendar-year timestamps
   (`time_convention: "calendar_year"`).
3. **Dask assembly engine** → **deleted outright** after the parity gate.
4. **Global-store variable set** → confirmed: embeddings + scales + obs counts
   (+ `embedding_std` when computed), all sharded on the same 2048² grid.
5. **Cross-machine commit gate** → Prefect global concurrency limit
   (yield-embeddings side); library ships the in-process primitive + contract.
6. **Single-ROI default layout** → `LEGACY` stays the default for single-ROI
   entry points; `GLOBAL_V1` is opt-in (strict D8).

7. **Zone boundary policy** → confirmed: pure nominal 6° longitude bands,
   disjoint, **no Norway/Svalbard MGRS exceptions**. Must be documented
   user-facing (zone_grid docstring, README, `zone_scheme` attr — W6).
8. **Campaign land mask** → partner-supplied; the zone-fill runner takes it as
   an input (same contract as today's ROI-mask zarr) and we assume it lands —
   no fallback mask is built.

**The plan is final — no open questions remain.** Stage A is unblocked.
