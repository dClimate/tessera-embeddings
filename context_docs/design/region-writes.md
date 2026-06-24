# Region Writes for Icechunk Stores

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

## 5. Phase 1 — the region write primitive (self-contained PR)

All of Phase 1 lives in `storage/zarr_store.py` plus tests/docs. It adds a write
mode without touching ingest/inference call sites, so it ships and merges on its
own.

> **Status: implemented.** Phase 1 landed in `storage/zarr_store.py`
> (`resolve_region`, `_pad_region_to_chunks`, `_store_chunk_sizes`,
> `_drop_region_coords`, `_write_region`, `write_region`) with the test suite in
> `tests/unit/test_zarr_store.py` (`TestResolveRegion`, `TestPadRegionToChunks`,
> `TestWriteRegion`). Two refinements vs. the sketch below are noted inline:
> `resolve_region` takes coordinate **ranges**, and the RMW overlay uses a
> positional dask `setitem` rather than label-based `combine_first`.

### 5.1 `resolve_region` — coordinate → integer-index

As built, it takes inclusive coordinate **ranges** (not single values), which
covers both single-slice and multi-slice selections:

```python
def resolve_region(
    store_path, *, time=None, northing=None, easting=None
) -> dict[str, slice]:
    """Map coordinate-value RANGES to integer slices against the existing store."""
```

- Each arg is an inclusive `(low, high)` range or `None` (= full axis, omitted
  from the result). `time` accepts anything `np.datetime64` parses; spatial
  bounds match the store's coord values regardless of axis direction (northing
  descends).
- Reads the store's existing coord arrays, returns **half-open** integer slices
  (`slice(first_hit, last_hit + 1)`).
- Validates the range selects **at least one existing coordinate**
  (overwrite-in-place contract); raises a clear `ValueError` if it matches
  nothing — appending is `write_dataset`'s job.

### 5.2 `_pad_region_to_chunks` — the read-modify-write pad

```python
def _pad_region_to_chunks(store_array, data, region):
    """Widen each region dim to enclosing whole-chunk bounds, backfilling
    the shell from the store so the r+ write sees only whole chunks.

    Returns (padded_data, widened_region)."""
```

- Reads store chunk sizes from the opened array (`.chunks`), **not** from
  config — the store is authoritative.
- **Validates shape on every path first.** Each written variable is transposed
  to the store's dim order and asserted to cover exactly the region (region
  length on each region dim, full store length elsewhere). This runs before the
  aligned/padded split, so neither path can let dask/NumPy broadcasting silently
  repeat a too-small (or axis-missing) input across the region.
- Reads the enclosing chunks as a lazy dask slab on the store's chunk grid
  (`existing.isel(widened)`). **Note:** the positional overlay below keeps *every*
  overlapped chunk — including interior chunks fully covered by the region —
  dependent on that read, so a large unaligned write fetches the whole widened
  slab, not just its edge chunks. The extra IO is bounded by the padding (at most
  one chunk per region face) and is acceptable at current tile sizes. Reading
  only the boundary chunks would require an edge-concatenation rebuild (interior
  blocks sourced straight from `incoming`, edges read-modify-written) — since
  built, see §9.
- **Overlay via positional dask `setitem`**, not label-based `combine_first`:
  `shell[idx] = data[var].data`, where `idx` is the region offset *within* the
  widened frame. Positional overlay sidesteps coordinate-label alignment
  pitfalls (e.g. descending northing) — the store's coords are authoritative and
  the incoming data is placed by index. Incoming values win; untouched cells
  keep the shell (store) value.
- Aligned regions **skip the store read**: when no edge needs widening, the
  shape-validated data is returned with the (already whole-chunk) region, no
  shell slab fetched or overlaid.
- Result is whole-chunk on every axis → `r+` validation passes.

This means **callers pass any region they want** (aligned or not); the primitive
makes it safe. No caller-facing alignment burden, no "fail loud" assert.

### 5.3 `write_region` — public entry point

```python
def write_region(
    store_path: str,
    data: xr.Dataset,
    *,
    region: dict[str, slice],            # integer slices, e.g. from resolve_region
    update_attrs: dict[str, Any] | None = None,
    get_credentials=None,
) -> None:
```

Body, combining the coord-drop with `_write_append`'s attr discipline and the
RMW pad:

```python
def _write_region(store_path, data, region, message, update_attrs=None):
    repo = _open_repo(store_path)
    session = repo.writable_session("main")

    root = zarr.open_group(session.store, mode="r")
    preserved_attrs = dict(root.attrs)

    padded, widened = _pad_region_to_chunks(root[var], data, region)  # §5.2

    # Drop region-dim coords + any coord not sharing a region dim — the store's
    # coords are authoritative; xarray rejects writing them into a region slice.
    region_dims = set(widened)
    drop = [
        name for name, v in padded.variables.items()
        if name in padded.coords and (name in region_dims
            or not region_dims.intersection(v.dims))
    ]
    to_write = padded.drop_vars(drop)

    to_icechunk(to_write, session, mode="r+", region=widened,
                align_chunks=True, split_every=8)

    root = zarr.open_group(session.store, mode="r+")
    root.attrs.update(preserved_attrs)
    if update_attrs:
        root.attrs.update(update_attrs)
    session.commit(message)
```

### 5.4 Dask orchestration

In the common case the caller's region is a single chunk-sized patch and the
write is one `to_icechunk` call — no striping, no slab loop. From the
consolidated learnings (§2.6, §7):

- **Distribution comes free.** `to_icechunk` on a dask-backed dataset forks the
  session across the dask graph internally (via `icechunk.dask.store_dask` /
  `merge_sessions`, both present in 2.0.4) and fans compute + chunk writes out
  to the cluster; the driver only does the final `commit`. **Every
  `to_icechunk` call is already a distributed write** — so even a single
  `write_region` is distributed without any fork/merge plumbing.
- **Coarse dask blocks, fine on-disk chunks** (only if the region itself is
  large). One dask block per on-disk chunk and let `align_chunks=True` map them;
  don't build a graph at sub-chunk granularity (the 1.5M-task scheduler OOM,
  Case Study 4). For a small region this is a non-issue.
- **Commit strategy: per-region commit** (decision below). Fresh
  `writable_session` → `to_icechunk` → `commit`. Bounds the in-flight changeset.
- **Spatial striping is optional — a memory bound, not correctness.** Only tile
  a region too large to materialize at once. The RMW pad (§5.2) keeps each tile
  whole-chunk; most region writes won't need striping at all.
- `split_every=8` to bound reduction-tree depth (matches assembly.py).

#### Decision: per-region commit, not manual fork/merge

We considered manual `session.fork()` → workers write → `session.merge(*forks)`
→ single atomic commit. Verified the API exists in 2.0.4 (`Session.fork() ->
ForkSession`, `Session.merge(self, *others)`, plus `icechunk.dask.store_dask` /
`merge_sessions`). **Decision: per-region commit.**

Rationale: the distribution we want is already provided by `to_icechunk`
internally — manual fork/merge does not add distribution, it only **coalesces N
independent region writes into one atomic cross-region commit** (and shaves
per-region commit overhead). The per-region-commit shape is a known-good pattern
in production Icechunk pipelines. Manual fork/merge stays available if a future
flow needs all-or-nothing atomicity across a set of regions; it's an
orchestration-layer optimization, not a primitive-level prerequisite.

> **Update: the fork/merge batch path is now implemented** as `write_regions`
> (`_write_regions` / `_aligned_region_sources`). It takes a list of
> `(data, region)` items, pads each (reusing `_pad_region_to_chunks`), fans all
> regions through **one** `icechunk.dask.store_dask` compute on a single
> `session.fork()` — full cluster saturation across every region at once instead
> of one compute wave per region — then `session.merge`es the changesets and
> commits **once**. `store_dask` writes chunk data into the pre-seeded arrays via
> the fork and never rewrites the root group, so root attrs survive without the
> snapshot/restore dance `to_icechunk` needs; `update_attrs` is still applied
> explicitly before the commit. Because the merge does **no** conflict
> resolution, the caller must guarantee the regions are mutually
> **chunk-disjoint**. A store whose `time` chunk size is 1 makes this trivial:
> writes at distinct time indices never share a chunk regardless of spatial
> overlap, so a batch of per-time-index writes is always safe. Sources are
> explicitly rechunked to the store grid because the raw `store_dask` path does
> no `align_chunks` realignment. The single-region `write_region` /
> per-region-commit path is unchanged.

### 5.5 NaN-fill / population semantics

There is no sentinel distinguishing "never written" from "legitimately NaN"
(see commit `ebc17b6`, "Mark ungenerated-embedding pixels with NaN scale"). Each
region write is **one atomic Icechunk commit**, so readers never see a torn
region.

Because we overwrite *committed real data* (not seeded NaN), a region split
across multiple commits exposes a concurrent reader to **stale data mixed with
new** — plausible-looking and silent. Mitigation: one commit per logical region
(the per-region-commit strategy already does this for a one-shot write); only
split for memory, and document that the region is inconsistent until the last
commit lands. Cross-commit atomicity for a split region is exactly the case for
the deferred manual fork/merge (§5.4).

### 5.6 Tests (same PR)

Following `tests/unit/test_zarr_store.py` + moto fixtures (no cassettes):

- Round-trip: create store, overwrite a time slice, read back; assert only that
  slice changed and attrs/CRS/`_manifest` survived.
- Spatial region overwrite: overwrite a `northing`/`easting` sub-box at an
  existing date.
- **Unaligned region**: overwrite a region whose edges fall mid-chunk; assert
  the RMW pad backfills correctly and untouched neighboring cells are unchanged.
- Coord-drop: assert region-dim coords aren't required on input and aren't
  corrupted in the store.
- Atomicity: a failed region write leaves the prior commit intact (extend
  `TestCleanupOnFailure`).
- Contract: reject a region write to a time value not already in the store.

### 5.7 Docs

Per repo convention, update the storage module docstring / README to document
the third write path (`create` / `append` / `region-overwrite`), the
coord-drop rule, the RMW-pad behavior, and the ASCII diagram from §2.

---

## 6. Phase 2 — adapt flows to read/write regions (separate PR)

Phase 1 ships the primitive; Phase 2 wires callers:

- `write_dataset` (zarr_store.py:410) gains a branch: when incoming dates
  **already exist** in the store, route to `write_region` instead of the
  create/append fork. This is the standard update split (intersection → region,
  difference → append) brought into our entry point.
- Ingest (`s2_roi.py`, `s1_roi.py`) and inference assembly (`assembly.py`) opt
  into region overwrite for re-runs / corrections of already-ingested dates or
  spatial sub-boxes.
- Decide per-flow striping (slab tiling vs. the inference northing-striping
  from `9428b38`).

---

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
| Backfill shape | Edge strips (§9), evolved from full-slab RMW | IO `O(perimeter)` not `O(area)`; required at state/CONUS scale. |

---

## 9. Backfill evolution — full slab → zarr-direct edge strips

The RMW pad's backfill went through three shapes as the master grew from
100km-tile to state/CONUS scale. Each fixed a distinct scaling wall; the history
matters because the failure modes look similar (a stalled flow-runner) but have
different causes.

**v1 — full slab via the lazy view (`existing.isel(widened)` + `setitem`).**
Read the whole widened slab as a lazy dask slab off the store *view*, then
overlaid the incoming block with a positional `setitem`. Two problems compounded
as the store grew: (a) the view's graph references **every chunk of the whole
store array**, so `isel` drags an `O(total store chunks)` layer into every
shell — independent of how much the write covers; and (b) `setitem` over a full
lazy slab rewires the graph across every chunk in the window, `O(slab area)`.
Fine at tile scale; at CONUS it builds ~1.7M graph nodes for a single run and
the flow-runner stalls before any compute.

**v2 — full slab via zarr-direct reads (`_slab_array_from_zarr`).** Fixed (a):
enumerate, in closed form, only the chunk blocks the widened slab touches and
emit one read task per block on a fresh `HighLevelGraph` with no parent layer.
Graph drops to `O(slab chunks)`, flat in store size — the scheduler-OOM fix.
But the backfill still reads the **entire widened slab**, interior included,
even though every interior cell is overwritten by the incoming data. IO is still
`O(area)`: a Nebraska-scale box reads ~126 chunks per write/var, ~80 of them
discarded interior. That wasted read is what made state-scale merge batches take
~7 minutes each.

**v3 (current) — zarr-direct edge strips.** Read only the pad margin on each
unaligned face (a thin slab per face), `da.concatenate`-d onto the incoming
block one axis at a time; the interior is never read. Concatenating axis by axis
— each strip spanning the already-extended extent on previously-padded axes —
backfills the corner cells with no separate corner read. IO drops to
`O(perimeter)`: Nebraska ~126→46 reads/var, Texas ~357→76, the ratio widening
with ROI size. Strips are read zarr-direct (keeping v2's `O(perimeter chunks)`
graph); reading them through the lazy view would reintroduce v1's whole-store
layer.

**The tradeoff.** Edge strips add a small, fixed number of graph nodes per face
(strip read + concat op) versus v2's single slab read — `+~400` nodes/var at a
100-chunk box, `+~800` at Texas scale. In absolute terms that's negligible
(~sub-second flow-runner build), and it buys the `O(area)→O(perimeter)` IO cut.
Output is byte-identical to v2 across both paths (verified in
`tests/unit/test_zarr_store.py::TestPadRegionToChunks`). The graph-size test
asserts the strip count is flat in store time length — the invariant that
distinguishes a zarr-direct strip from a lazy-view one — not an absolute node
count, since the perimeter constant is the part that doesn't matter.
