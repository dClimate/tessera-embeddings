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

## 1. Motivation & scope

We want to rewrite a **region** of an existing store — a temporal slice (dates
already present), a spatial sub-box (`northing`/`easting`), or both — without
rewriting the whole array or appending. Use cases: re-running a date that's
already ingested, backfilling/correcting a spatial sub-box at existing
timesteps, patching inference output for a tile range.

**Scope decision (Phase 1): overwrite-in-place only.** The region being written
must already exist in the store; we rewrite the cells it covers. This matches
the common ETL notion of an "insert" for already-present times
(`insert = update_times ∩ existing_times`).

**Explicitly out of scope:** *true mid-array insertion* — physically inserting a
new date between two existing dates and shifting later data to higher indices.
That requires `resize` → back-to-front shift → coord rewrite across separate
commits, is order-sensitive across commits (icechunk #1873), and warrants its
own design. For a regular time grid you append and reindex logically instead of
physically reshuffling chunks. Deferred until a concrete need appears.

---

## 2. What we have today

`storage/zarr_store.py` has two write modes, both whole-array along time:

| Mode | Function | Call |
|---|---|---|
| Create | `_write_new` (zarr_store.py:319) | `to_icechunk(data, session, mode="w", encoding=…, align_chunks=True)` |
| Append | `_write_append` (zarr_store.py:334) | `to_icechunk(data, session, mode="a", append_dim="time", align_chunks=True)` |

Stores are 3D `(time, northing, easting)`, Zarr v3, chunked
`time=1, northing=4000, easting=4000` (`INGEST_CHUNKS`). `_write_append`
snapshots and restores root attrs because `to_icechunk` clobbers them with
`data.attrs` (zarr_store.py:344-354). Reads already do arbitrary spatial slicing
via `oindex`. No `region=` anywhere in the write path.

```text
   CREATE (mode="w")          APPEND (append_dim="time")     REGION (mode="r+")  ← new
   whole array, new store     extend the time axis           overwrite a slice of an
   ┌───────────────┐          ┌───────────────┬──┐           existing array
   │               │          │   existing    │NN│           ┌───────────────┐
   │   all data    │          │     data      │NN│           │      ░░░░░    │  ░ = region
   │               │          │               │NN│           │      ░░░░░    │      overwritten
   └───────────────┘          └───────────────┴──┘           └───────────────┘
                                           new dates              time and/or northing/easting
```

---

## 3. Reference patterns

These come from relevant outside examples (other Icechunk/Zarr pipelines, not in
this repo). They inform the design but aren't dependencies.

### 3.1 The `to_icechunk` region call

A relevant outside example writes a region of an existing Icechunk store like
this:

```python
to_icechunk(
    to_write, session, mode="r+",
    region={"time": slice(time_idx, time_idx + 1), "northing": y_slice, "easting": x_slice},
    align_chunks=True, split_every=8,
)
```

We reuse the **coord-drop** (region-dim coords *and* any coord not sharing a
region dim — the store's coords are authoritative; xarray rejects writing them
into a region slice) and the **per-region session/commit shape**.

That's all such a pattern gives us, though. Those flows typically build a **new**
store and fill **freshly-seeded NaN** timesteps, with output chunks sized so
every write is whole-chunk by construction. They never overwrite committed data
and never pad an unaligned region — the two hard parts of *our* problem. Those
come from §3.2 and §5.2.

### 3.2 Chunk-alignment via read-modify-write

The pattern for handling **unaligned** regions (seen in plain-Zarr ETL stacks):
widen the region to enclosing chunk boundaries and backfill the now-included-but-
unchanged cells from the store before writing:

```python
chunk_start = (region[0] // chunk_size) * chunk_size
chunk_end   = (((region[1] - 1) // chunk_size) + 1) * chunk_size
original_slice = original_dataset.isel({dim: slice(chunk_start, chunk_end)})
full_slice = insert_slice.combine_first(original_slice)          # incoming wins
full_slice_rechunked = full_slice.chunk({dim: chunk_size})
return full_slice_rechunked, (chunk_start, chunk_end)            # widened region
```

That reference only does this on the **time** axis (small dim). We generalize to
all three axes and trim it to read only boundary chunks (see §5).

---

## 4. Why `align_chunks=True` is not enough (the alignment investigation)

A natural assumption is that `to_icechunk(..., align_chunks=True)` handles
unaligned regions for us. It does not, and the naming is a trap — there are two
different "align" knobs:

- The **read-modify-write padding** approach from §3.2 (some ETL stacks expose
  it as an `align_update_chunks`-style flag). This is what actually makes an
  unaligned region safe, and it's what §5.2 reimplements.
- **`align_chunks=True`** (icechunk's `to_icechunk`) is a **different**
  mechanism. Reading the implementation it delegates to
  (`xarray/backends/chunks.py`):
  - `align_nd_chunks` rechunks the **dask blocks you provide** to map cleanly
    onto the zarr grid. The "add artificial data to the borders" step
    (chunks.py:58-62) is *arithmetic on chunk sizes only* — it never reads the
    store. It hard-requires `sum(your_chunks) == sum(backend_chunks)`
    (chunks.py:18): the data you hand it must already contain exactly the
    elements the region spans.
  - In `r+` mode, `allow_partial_chunks = (mode != "r+")` → **False**, and
    `validate_grid_chunks_alignment` then **raises** if the region *start*
    (chunks.py:267) or *end* (chunks.py:272-279) lands mid-chunk.

**Conclusion:** `align_chunks=True` fixes producer-side dask↔zarr chunk mapping
and avoids parallel-write races; it does **not** read neighboring store data to
pad a partial boundary chunk, and will **reject** an unaligned region in `r+`.
Only a read-modify-write pad (§3.2 / §5.2) makes an arbitrary region safe. We
therefore implement the RMW pad and keep `align_chunks=True` on the final write
for its actual job (producer-side chunk mapping).

---

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
