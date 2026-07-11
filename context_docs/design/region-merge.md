# Region Merge — no-cluster merge of grid-aligned stores

`storage/region_merge.py` merges many **feature** stores into one **master**
store by direct, process-parallel raw-Zarr chunk writes — no Dask, no cluster. A
*feature* store is any store whose grid is an exact pixel-subset of a shared
*master* grid: same CRS and resolution, `northing`/`easting` coordinate values a
contiguous subset of the master's, and dates all present in the master's `time`
axis ("master-snapped"). This is the batch-merge path; it replaced the Dask
`write_regions` batch write (see [History](#history)).

The motivating use case is a **sparsely populated large ROI**: the master is a big
rectangular grid but only scattered regions within it are populated and maintained
(specific tiles/counties/footprints), the rest staying all-fill. The all-fill seed
writes no chunks outside the regions (creation cost independent of extent — see the
[empty-store note](#empty-store-seeding-companion)), so a continental master with a
few small regions is cheap; and because each feature merges independently by an
idempotent write, regions are added/refreshed incrementally over the master's life.
A dense fully-rectangular master is the degenerate case.

The intended workflow, end to end:

1. Seed an empty master over the master grid with the **union** of the feature
   dates — `empty_store.create_empty_store` (metadata-only, all-fill; see
   [`empty-store` note below](#empty-store-seeding-companion)) seeded from
   `region_merge.gather_time_union(feature_paths)`. Seed with `time` chunk size 1
   (the default) so every merged date occupies its own time chunk.
2. Merge each feature into the master, sequentially, with
   `merge_feature_into_master`.
3. Delete the temp per-feature stores once the merge succeeds
   (`region_merge.delete_store`).

Each feature's pixels are copied straight from its raw Zarr arrays into the
master's positional slice — **pure byte movement**, no compute.

```text
per-feature stores (each an exact pixel-subset of the master grid)
        │  gather_time_union → seed master with the date union
        ▼
  for each feature, sequentially:
    1. map feature dates → master time indices (exact datetime64)
    2. resolve the feature's spatial slice in the master (constant across dates)
    3. tile the slice to the master chunk grid → (var, date, chunk-block) units
    4. fork the icechunk session N ways; one ProcessPoolExecutor worker per shard
       writes its chunk bytes straight to storage (own session lock + S3 pool)
    5. merge the returned changesets → one commit per feature
        ▼
   master store, then temp per-feature stores deleted
```

## Why processes, not just threads

Within one icechunk session every chunk-write serializes on that session's store
mutex (Rust), so a thread pool plateaus regardless of width (production: 32 ≡ 128
threads ≈ 250 MB/s, threads parked in `futex_wait_queue`, NIC at ~10% and CPU
bursting only during the GIL-released codec windows). The lock is **per session**,
so the only axis that scales is processes: the merge forks the session
`max_workers` ways (`icechunk.Session.fork`), ships one fork per process, and each
fork is an independent session — own lock, own object-store connection pool — that
writes its shard's bytes directly to storage. Only the changeset references travel
back, where `Session.merge` folds them into one commit (icechunk's canonical
`ProcessPoolExecutor` distributed-write pattern). A few threads per process still
overlap S3 latency inside each fork's lock-release windows.

This replaced the old `write_regions` path, whose `O(runs × bands ×
spatial_chunks)` Dask task graph — built single-threaded on the flow runner before
any compute — was itself the bottleneck that made continental merges take days.
Copying at chunk granularity is `O(chunks)` with no graph at all.

Workers are started with the **spawn** context (not fork) so a child never
inherits the parent's icechunk runtime / credential state.

## Correctness — four invariants

* **Spatial.** Each feature grid's `northing`/`easting` coords are an exact,
  contiguous subset of the master's, so the feature maps to one master slice whose
  length equals its pixel count. The slice need not be chunk-aligned: a mid-chunk
  feature edge is handled by zarr's read-modify-write on the boundary chunk. The
  slice-length check raises if the feature is not a pixel-subset (e.g. a different
  resolution); exact coordinate identity within the resolved slice is the caller's
  contract.
* **Temporal.** Each feature date maps (by exact `datetime64` value) to one master
  time index. Distinct dates must occupy distinct **time chunks** — otherwise two
  per-date copy units in the same time chunk would be written by different forks
  with no conflict resolution and silently race. A master `time` chunk size of 1
  guarantees this; `merge_feature_into_master` validates it and raises otherwise.
* **Within-feature disjointness.** Writes are tiled to the master chunk grid and
  fanned over `(var, date, chunk-block)` units. Distinct vars are distinct arrays,
  distinct dates distinct time-chunks, distinct blocks distinct spatial chunks — so
  no two workers (processes or threads) ever write the same chunk, and the fork
  merge is conflict-free.
* **Fill-masked overlay across features.** A feature's rectangular window is the
  master-snapped bounding box of its (possibly irregular) footprint, so it carries
  fill (`_fill_mask`: NaN for float vars, 0 for integer vars) wherever the footprint
  does not cover. Two footprint-disjoint features can still have overlapping
  *windows*, and there one feature owns the pixels while the other is fill. So each
  copy unit overlays only the feature's **real (non-fill) pixels** — `where(feature
  is fill, master, feature)` — never blindly assigning the window. A feature
  therefore never overwrites a neighbour's data with its own fill. (An all-real
  interior block skips the master read and writes `src` directly; only partial-edge
  blocks pay the read-modify-write.) Because features are pixel-disjoint on their
  real data, the merge is order-independent. The read-modify-write also reconciles a
  master boundary chunk shared by two pixel-adjacent features across the
  **sequential per-feature commits**: the second feature reads the chunk with the
  first's pixels already committed and overlays only its own — both survive. This is
  why features need not be chunk-disjoint, where the old distributed path required
  it.

### Fill-masked overlay, illustrated

**Where "fill" comes from.** A feature's real footprint is irregular, but its
master-snapped *window* is the rectangular bounding box the merge copies. Cells
inside the window but outside the footprint are fill (`0` for integer vars, `NaN`
for float):

```text
   feature footprint                 master-snapped window
   (what it actually covers)         (what the merge sees: real ▓ + fill ·)

        ▓ ▓ ▓                         ┌───────────┐
        ▓ ▓                           │ ▓  ▓  ▓  · │
        ▓                             │ ▓  ▓  ·  · │
                                      │ ▓  ·  ·  · │
                                      │ ·  ·  ·  · │
                                      └───────────┘
                                       ▓ = real   · = fill (0 int / NaN float)
```

**The overlay.** Two footprint-disjoint features `A` and `B` can still have
overlapping *windows*. In the overlap, `A` is real where `B` is fill and vice
versa — never both (pixel-disjoint real data). Merge `A`, then `B`; each
contributes only its real pixels, so `B`'s fill never clobbers `A`'s data. Values:
`11` = A real, `22` = B real, `0` = fill/seed.

```text
  step 1 — after A's commit          step 2 — B's window over the overlap
     ┌────────┐                          ┌────────┐
     │ 11  11 │ A's real                 │  0   0 │ B is FILL here (A's turf)
     │  0   0 │ seed (A is fill here)    │ 22  22 │ B's real
     └────────┘                          └────────┘

  step 3 — overlay  where(B is fill, master, B)     vs.  blind copy (assign B)
     ┌────────┐                                          ┌────────┐
     │ 11  11 │ B fill → keep master → A survives ✓       │  0   0 │ A clobbered ✗
     │ 22  22 │ B real → take B                           │ 22  22 │
     └────────┘                                          └────────┘
```

**Interior vs. edge blocks.** The window is tiled to the master chunk grid. A
block entirely inside the footprint is all-real and written straight through (no
master read); only an edge block that straddles the footprint boundary carries
fill and pays the read-modify-write:

```text
   feature window tiled to the master chunk grid:

   ┌──────┬──────┬──────┐
   │ ▓▓▓▓ │ ▓▓▓▓ │ ▓▓·· │   ▓▓▓▓ interior block, all real
   │ ▓▓▓▓ │ ▓▓▓▓ │ ▓▓·· │         → write src directly, skip the master read
   ├──────┼──────┼──────┤
   │ ▓▓▓▓ │ ▓▓▓▓ │ ▓▓·· │   ▓▓·· edge block, real + fill
   │ ▓▓·· │ ▓▓·· │ ▓··· │         → read-modify-write: overlay only the real cells
   └──────┴──────┴──────┘
```

## Hang protection

A multi-hour merge will eventually hit a transient S3 drop; without protection a
worker parks on the dead socket (`sk_wait_data`), the coordinator blocks in
`f.result()` with no timeout, and the run hangs silently forever. Two layers guard
against this:

1. **Per-attempt S3 timeouts + retries.** Every repo the package opens inherits
   finite connect/read/operation-attempt timeouts and a backed-off retry budget
   from `_open_repo`'s defaults (set in `_default_repo_config`), passed down to
   every fork. Icechunk's default is unbounded timeouts and a single try, so a hung
   socket read would block forever; this caps a single attempt and retries it with
   backoff. The values (`storage/zarr_store.py`): connect 30s, read 120s,
   operation-attempt 180s; 10 tries, backoff 200ms → 30s cap. `read_timeout_ms` is
   the one that bites the diagnosed hang (worker stuck in `sk_wait_data`
   mid-response).

   These live in `_default_repo_config` rather than being applied only to the merge
   because the failure mode is **request-level, not merge-level**: `write_dataset`,
   `write_region`, distributed assembly, and every read issue the same object-store
   attempts and were equally exposed — the merge was merely where multi-hour
   runtimes made the hang inevitable. Tradeoff: a *retriable* failure (sustained
   5xx, repeated timeouts) now takes minutes to surface across the retry budget
   instead of failing fast; non-retriable errors (403 etc.) still fail immediately.
   Callers needing a different budget can `reopen()` with their own
   `StorageSettings` (what the merge used to do), so no opt-out knob is pre-built.
2. **CPU-stall watchdog** (`_wait_with_stall_detection`). A merge times out for two
   reasons that want opposite responses: a worker wedged on a dead socket (kill +
   retry fresh) versus a region that is simply huge but copying fine (a fixed
   wall-clock budget would kill it and the retry would re-fail identically). A fixed
   budget can't tell them apart; CPU progress can. A healthy copy keeps every worker
   cycling through GIL-released codec windows, so system-wide `user+system` CPU
   climbs continuously; workers wedged in the kernel accrue ~zero. The watchdog
   resets a stall clock whenever CPU advances and declares a wedge only after CPU has
   been flat for a sustained grace window (default 600s) — which a large-but-
   progressing copy never hits no matter how long it runs. On a detected stall it
   SIGKILLs the workers (a clean join would itself hang on the wedged one) and
   retries the whole feature in a fresh session up to `feature_retries` times (the
   region-write is idempotent, so a retry is safe). When all attempts are exhausted
   it raises `TimeoutError` naming the feature, so the run fails loudly instead of
   hanging. An optional `feature_timeout_sec` adds an absolute hard ceiling
   regardless of CPU progress (disabled by default — an operator escape hatch).

The watchdog samples CPU host-wide (`_busy_cpu_seconds`) rather than per-worker-PID
because the coordinator is blocked waiting during a copy, so essentially all CPU on
the box is the workers'. Run the merge on a dedicated machine/container (or rely on
the hard ceiling); on a shared host another tenant's CPU would mask a stall.

## Empty-store seeding companion

`storage/empty_store.py` builds the all-fill master the merge writes into.
`create_empty_store` pre-allocates a store with the correct grid, chunking, dtype,
and root attrs but **no computed pixels** — the icechunk repo and zarr arrays are
created directly (data vars as schema only, zero chunks written), so creation cost
is independent of spatial extent. It resolves fill per-dtype exactly as the merge
does (0 for integer, NaN for float), so the merge's `_fill_mask` and the seed
agree. Seed a merge target with `time` chunk size 1 (the default `INGEST_CHUNKS`)
to satisfy the temporal invariant above.

## Public entry points

- `gather_time_union(store_paths) -> np.ndarray` — the master's date axis.
- `read_master_axes(master_path) -> (times, vars)` — read once, pass into every
  `merge_feature_into_master` call for a master (avoids a per-feature master open).
- `merge_feature_into_master(master_path, feature_path, *, master_times,
  master_vars, ...) -> int` — copy one feature in, one commit; returns dates
  written.
- `read_store_times(store_path) -> np.ndarray` — a single store's date axis
  (tolerates a missing store).
- `delete_store(store_path) -> bool` — drop a temp store; never raises.

## History

This replaced the Dask `write_regions` batch path (see
[`region-writes.md`](region-writes.md)), which was removed in the same change. The
single-region `write_region` overwrite path in `zarr_store.py` is unchanged and
remains the tool for overwriting one slice of an existing store.
