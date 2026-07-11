# Merging regional mosaics into one master

**Use case: sparsely populating (and maintaining) a large ROI.** You have a large
rectangular grid but only care about *parts* of it — scattered regions (specific
tiles, counties, fields, sensor footprints) you want to populate and keep current,
while the rest of the rectangle stays empty. Ingesting and storing the whole
rectangle would be mostly wasted; instead you ingest each region of interest into
its own small Zarr store and **merge** them into a single **master mosaic** whose
empty areas cost effectively nothing — the all-fill seed writes no chunks outside
the populated regions, so a continental-extent master with a few small regions is
cheap to create and store.

This is the multiple-regional-inserts pattern: fan the regions out into small,
parallel, independently-retryable ingests, then merge the results into one store.
Because each region is merged independently by an idempotent write, regions can be
**added or refreshed incrementally** — the master is *maintained* over time (new
regions merged in, existing ones re-merged to backfill), not built once and frozen.
A dense, fully-rectangular master is the degenerate case (every cell populated); the
merge is designed for the sparse case where populated regions are a small fraction
of the master extent.

The merge is **pure byte movement** — no compute, no Dask/Ray cluster. Each
region's pixels are copied straight from its raw Zarr arrays into the master's
positional slice by a process-parallel chunk-write, one commit per region. The
mechanism and its correctness invariants are in
[`context_docs/design/region-merge.md`](../context_docs/design/region-merge.md);
this page is the how-to.

## The one precondition: a common grid

Every region store must be **master-snapped** — an exact pixel-subset of a single
shared master grid:

- **same CRS and resolution** as the master,
- **same axis order** (e.g. northing descending), and
- coordinates that are a **contiguous subset** of the master's — i.e. each
  region's pixels line up exactly on master pixel boundaries, no half-pixel
  offset.

This is what makes the *positional* copy correct: region row/column `i` is written
to master row/column `i`. It is the **caller's** responsibility, established when
the region stores are produced — not something the merge can retrofit.

The reliable way to guarantee it: derive every region grid from the master grid.
The `ingest.feature_grid` module does exactly this — given a GeoJSON of regions and
the master's `odc.geo` geobox (from `read_roi_metadata(master_roi_path).geobox`), it
snaps each feature to a grid-aligned window of the master and rasterizes it into a
mask the ingest path reads:

```python
from tessera_embeddings.ingest.feature_grid import (
    load_features, assert_features_disjoint, feature_window, rasterize_feature_roi,
)

features = load_features(geojson, target_crs=master.native_crs, id_property="tile_id")
assert_features_disjoint(features)                       # region writes overwrite — disjoint required
for feat in features:
    sub_geobox = feature_window(master.geobox, feat.geometry)   # exact pixel-subset, by construction
    rasterize_feature_roi(f"{roi_base}/{feat.feature_id}.zarr",
                          master_geobox=master.geobox, feature=feat)
    # ...ingest the region onto that ROI, producing a store on sub_geobox's grid...
```

`feature_window` uses `master_geobox.enclosing`, which snaps the geometry's bounds
outward to whole master pixels sharing the master's origin and resolution, so the
window's coordinates are an exact subset of the master's. `assert_features_disjoint`
enforces the no-overlap requirement (region writes overwrite, so two regions
covering the same master cell would last-writer-win).

`merge_stores`/`merge_feature_into_master` *validate* the snap (extent, coordinate
alignment to a quarter-pixel, dtype, chunking) and raise a clear `ValueError` if a
region is not snapped — but they cannot fix it. The `feature_grid` helpers above
guarantee it. Wrapping them into a single fan-out flow (load a GeoJSON → snap +
rasterize each → dispatch one ingest per region → gate → merge) is planned but not
yet in this repo; see "Producing the region stores" below.

## The recipe

The whole sequence is one call:

```python
import numpy as np
from tessera_embeddings.ingest.roi import read_roi_metadata
from tessera_embeddings.storage.region_merge import merge_stores

roi = read_roi_metadata("s3://.../rois/master.zarr")   # the grid authority

summary = merge_stores(
    "s3://.../master_mosaic.zarr",                     # created + seeded here
    ["s3://.../regions/west.zarr", "s3://.../regions/east.zarr", ...],
    roi=roi,
    var_dtypes={"0_VV": np.dtype("uint16"), "0_VH": np.dtype("uint16")},
    tile_id="my_master",
    delete_temp=True,                                  # drop the region stores after
)
# summary: {"master_path", "n_dates", "merged": {path: dates}, "deleted", "skipped", "elapsed_sec"}
```

`merge_stores` does, in order:

1. **`gather_time_union(feature_paths)`** — the master's date axis: the sorted,
   de-duplicated union of the regions' dates (region date sets are heterogeneous —
   satellite revisit differs by location). Missing/empty regions contribute
   nothing.
2. **Seed** the master with `empty_store.create_empty_store` over `roi` and that
   union — metadata-only (no pixels computed; creation cost is independent of
   extent). Skipped when `resume=True`.
3. **`read_master_axes`** once (the axes are constant per master).
4. **`merge_feature_into_master`** for each region, **sequentially** — which is
   also what makes shared boundary chunks correct: successive commits reconcile a
   shared chunk by read-modify-write, so regions need not be chunk-disjoint.
5. **`delete_store`** each region (only if `delete_temp`, only after success).

If you want the steps individually (for a custom loop, or a pre-seeded master), the
primitives are all public in `storage.region_merge` and `storage.empty_store` — see
[public-api.md](public-api.md). `merge_stores` is just the correct sequencing of
them.

### As a Prefect flow

`orchestration.prefect.flows.merge_mosaic` is a thin wrapper over `merge_stores`
for deployment (`var_dtypes` are passed as dtype **strings**, e.g. `"uint16"`, so
they cross the Prefect parameter boundary). It needs no cluster — the merge runs on
the flow runner — so there is no task-runner or provider-cluster setup.

## Seeding: union vs. a fixed calendar axis

There are two ways the master's **time axis** gets decided:

- **Union seeding (the default, what `merge_stores` does):** the regions already
  exist, so the master is seeded with exactly `gather_time_union(feature_paths)` —
  a compact, irregular axis matching the actual observations. Use this when you
  assemble a master from a batch of finished regions.
- **Fixed calendar axis (`daily_times` + `create_empty_store`, done up front):**
  `empty_store.daily_times(start, end)` generates *every* day in a range; seed a
  master with it **before** any region exists (e.g. downstream consumers want a
  predictable daily grid, or regions trickle in over time). Then merge regions into
  it with `resume=True` (the master is already seeded). A daily axis is a superset
  of any region's dates, so absent days stay all-fill.

Both are merge-compatible; pick by whether the axis is discovered from the data or
fixed in advance.

## Scale knobs

Passed through `merge_stores` (and the flow) to each region's copy:

- **`max_workers`** — worker processes per region copy (default: one per core).
  Processes are the axis that scales: each fork is an independent icechunk session
  with its own store lock and S3 connection pool. Threads within one session
  serialize on the store mutex.
- **`threads_per_process`** — small thread pool per worker to overlap S3 latency
  (default: a few).
- **`max_concurrent_requests`** — caps per-repo/per-fork S3 concurrency **and** each
  worker's region read. Many processes hitting one S3 prefix can trip 503 SlowDown;
  lower it (e.g. `64`) to throttle aggregate request rate without dropping worker
  count. `None` leaves icechunk's default (256).
- **`manifest_split_sizes`** (on `merge_stores`) — the icechunk manifest split
  applied across the seed and every open, so a per-region commit rewrites only the
  shards it touched. Defaults to `DEFAULT_MANIFEST_SPLIT_SIZES` (2-D spatial); pass
  `{}` to disable.

## Failure behavior

- **A missing region store** contributes nothing and is skipped (returns 0 dates).
  A store that **exists but can't be read** (corruption, auth, transient S3) fails
  the merge **loudly** — it is never silently dropped.
- **A stalled region copy** (a worker wedged on a dead socket) is caught by the
  CPU-stall watchdog, which SIGKILLs the workers and retries the region in a fresh
  session up to `feature_retries` times (the region write is idempotent). Per-attempt
  S3 timeouts + retries (a package-wide default) mean no single request hangs
  forever. If every attempt stalls, it raises `TimeoutError` naming the region.
- **A non-snapped region** (wrong extent, misaligned/reversed coordinates, dtype
  drift, non-uniform master chunk grid, duplicate dates) raises `ValueError` before
  any pixel is written.

## Producing the region stores

The merge consumes region stores; producing them is upstream of this page.
Anything that writes a `(time, northing, easting)` Zarr on the master grid — i.e.
onto a `feature_window(master_geobox, geom)` (see the precondition above) — is a
valid region. The real path is a per-region satellite ingest: snap + rasterize each
region's ROI with `ingest.feature_grid`, run the S2/S1 ingest (see
[quickstart.md](quickstart.md)) once per region against that ROI, then merge the
results here. `feature_grid.classify_store` / `classify_mask` give the
skip/re-run/manual-curation triage a fan-out needs.

`feature_grid` is the geometry + triage **foundation**; wrapping it into a single
fan-out flow (load a GeoJSON → snap + rasterize each → dispatch one ingest per
region → gate → merge) is planned but not yet in this repo. Until then, drive that
loop yourself with the `feature_grid` helpers and pass the resulting store paths to
`merge_stores`.
