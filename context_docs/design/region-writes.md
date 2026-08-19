# Region Writes for Icechunk Stores

> **Status (updated).** The single-region `write_region` path described here is
> current. The **batch** path (`write_regions` / `_write_regions` /
> `_aligned_region_sources`, built on `icechunk.dask.store_dask`) was **removed**
> as unused: its `O(runs × bands × spatial_chunks)` Dask task graph — built
> single-threaded on the flow runner before any compute — made continental merges
> take days. Its replacement — a process-parallel raw-Zarr region merge, no Dask —
> lands in a stacked follow-up PR. The `store_dask` references in §3.2, §5.4, and
> the §5.4 update note below are retained as history; the code they describe no
> longer exists.

Design + implementation plan for adding **region-scoped writes** (overwrite a
time slice and/or spatial sub-box of an existing store) to
`storage/zarr_store.py`. Today the module supports only whole-array `create`
and time-axis `append`; this adds a third write path.

Verified against the installed stack: **icechunk 2.0.4**, **xarray 2026.4.0**,
Zarr v3. Code references are `file:line` against the tree at time of writing.

---

## What this is

The region-write primitive lets a caller write one spatial window of an existing store without
rewriting the whole array, which is what makes a windowed ingest possible at zone scale at all.

**`storage/zarr_store.py` is the record of what was built** and its tests pin the contract. What
this document carries is the reasoning the code cannot: why the alignment problem is shaped the way
it is, and why the batch path was removed.

## Why `align_chunks=True` is not enough

A region that does not land on chunk boundaries forces a read-modify-write of the boundary chunks,
and `align_chunks` handles that inside one call — but two calls writing into the SAME chunk in one
session are a lost update, because each reads the chunk before the other's write is visible. That is
why the ingest snaps its windows to the chunk grid: chunk-disjoint windows make the problem not
arise rather than handling it, which is also what lets N windows share one session and one commit.

The reconciliation the merge tier has to do — overlapping windows, shared boundary chunks — is
exactly the work this path avoids by construction.
## 5. Implementation — shipped

The staged build plan that stood here (the `resolve_region` resolver, the `_pad_region_to_chunks`
read-modify-write pad, the `write_region` entry point, its dask orchestration, NaN-fill semantics,
the test list and the two-PR sequencing) described work that is now code. **`storage/zarr_store.py`
is the record of what was built**, and its tests pin the contract the plan specified. In git
history if the sequencing is ever wanted.

The batch path from that plan — `write_regions` / `_aligned_region_sources` over
`icechunk.dask.store_dask` — was **removed as unused**, for the reason in the status note at the
top: its `O(runs x bands x spatial_chunks)` graph was built single-threaded on the flow runner
before any compute ran.

## 7. Gotcha checklist (carry into implementation)

1. **`align_chunks=True` ≠ chunk-boundary safety.** It rechunks producer-side
   dask blocks and avoids parallel races; it does **not** pad partial boundary
   chunks and **rejects** unaligned regions in `r+`. The RMW pad (§5.2) is what
   makes arbitrary regions safe.
2. **Drop coords** — region-dim coords *and* non-region-dim coords; store coords
   are authoritative.
3. **Attrs clobbered** — `to_icechunk` overwrites root attrs; snapshot/restore as
   `_write_append` already does.
4. **Read chunk sizes from the store**, not config — the store is authoritative.
5. **Overwriting committed data.** A partial overwrite of an existing region
   exposes *stale real data mixed with new* to concurrent readers. Keep one
   commit per logical region; only split for memory, and flag the inconsistent
   window (§5.5). Also: "unwritten" vs "real NaN" remains indistinguishable —
   don't infer population from contents.
6. **One session per region/slab**, committed per slab — bounds the changeset.
7. **`to_icechunk` is already distributed** on a dask-backed dataset (internal
   fork/merge across the graph). Per-region commit is the chosen strategy.
8. **Conflict detection** — concurrent region writes to *disjoint* chunks don't
   conflict; overlapping chunks raise `ChunkDoubleUpdate`, resolvable via
   `BasicConflictSolver(on_chunk_conflict=UseOurs)` + `rebase`. Only relevant if
   a future flow parallelizes uncoordinated writes to one array.
9. **No true mid-array insert in scope** — deferred (resize + back-to-front
   shift + coord rewrite across commits; icechunk #1873 ordering hazard).
10. **Region slices are normalized and validated** — `_pad_region_to_chunks`
    resolves open bounds (`slice(None)`) via `slice.indices`, rejects non-unit
    steps and empty slices, transposes incoming data to the store's dim order,
    and asserts each variable's shape matches the region (so a broadcastable
    smaller array can't silently fill the whole region).
11. **`resolve_region` requires contiguous hits** — a coordinate range that
    straddles a gap on an unsorted axis is rejected rather than silently widened
    to cover intervening cells. Forward `get_credentials`/`s3_region` so the
    resolve and write open the same store.

---

## 8. Decisions log

| Decision | Choice | Why |
|---|---|---|
| Insert semantics (Phase 1) | Overwrite-in-place only | Native, low-risk; matches the standard intersection-insert. True mid-array insert deferred. |
| Unaligned regions | RMW pad (`_pad_region_to_chunks`) | `align_chunks=True` can't pad partial boundary chunks in `r+`; callers shouldn't bear manual alignment. The genuinely new work — references only pad the time axis. |
| Commit strategy | Per-region commit | Distribution already free via `to_icechunk`; fork/merge only adds cross-region atomicity, not needed yet. |
| Chunk sizes | Read from store array `.chunks` | Store is authoritative; config may drift. |
| Storage hang protection | Finite per-attempt timeouts + backed-off retries in `_default_repo_config` | Icechunk defaults to unbounded timeouts and a single try, so a wedged socket (`sk_wait_data`) blocks a write forever. Applied at every repo open — region writes inherit it for free. Values and full rationale in `zarr_store._default_repo_config`'s docstring. |
