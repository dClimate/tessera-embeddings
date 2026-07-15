# Tessera Inference Pipeline

Distributed GPU inference that generates **128-dimensional per-pixel embeddings** from mosaicked
Sentinel-2 reflectance + Sentinel-1 SAR data. The Prefect flow at
[`orchestration/prefect/flows/tessera_embeddings.py`](../orchestration/prefect/flows/tessera_embeddings.py)
orchestrates the full process: spinning up a Ray cluster on EC2 GPU instances, running inference
in parallel across spatial chunks, and assembling the output into a final Icechunk/Zarr store.

The orchestrator-free equivalent is
[`orchestration/runners/plain.py`](../orchestration/runners/plain.py), which calls the same
domain functions on `ray_cluster(num_gpus=0)` for laptop/CI runs.

---

## Architecture at a Glance

```
Input stores (Icechunk/Zarr on S3):
  reflectance.zarr / sar_ascending.zarr / sar_descending.zarr
            │
            ▼
  enumerate_chunks_from_dataset()     ← 2048×2048 px chunks
            │
            ▼
  filter_chunks_by_roi_mask()         ← drop chunks outside the ROI
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │  Ray Cluster  (EC2, g5.xlarge × N)          │
  │  ┌─────────────────────────────────────┐    │
  │  │ InferenceActor (1 GPU each)         │    │
  │  │  per northing strip (bounds RAM):    │    │
  │  │   1. load_chunk(y_sub=…)  ← selective│    │
  │  │   2. Dataset valid-px filter         │    │
  │  │   3. sample_s2/s1_batch()            │    │
  │  │   4. model forward  (FP16, B=3584)   │    │
  │  │  5. writer.write_chunk() → staging   │    │
  │  └─────────────────────────────────────┘    │
  │  Work-stealing: actors pull from queue      │
  └─────────────────────────────────────────────┘
            │
            ▼
  Staged chunks (S3 staging per run_id) — live chunks only
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │  Worker processes (on the flow runner)      │
  │  writer.assemble(): fork/merge raw-zarr     │
  │  writes into disjoint northing bands,       │
  │  one data commit (no Dask, no task graph;   │
  │  unstaged footprints stay at fill 0/NaN)    │
  └─────────────────────────────────────────────┘
            │
            ▼
  Output:  {roi_name}.zarr  (Icechunk; time axis extends on re-run)
           embedding dims: (time, northing, easting, 128)
           obs count dims:  (time, northing, easting)  — s2/s1_asc/s1_desc

  Global-campaign variant: writer.assemble_global() writes the same staged
  tiles as whole 2048-px shards into a pre-seeded UTM zone group of the
  global store (1 tile == 1 shard), driven by the zone-fill runner
  (orchestration/runners/zone_fill.py). See ADR-008.
```

---

## Pipeline Phases

### 1. Chunk Enumeration and ROI Pre-Filter

The input mosaic is divided into a grid of 2048×2048 pixel `ChunkSpec` objects (edge chunks may
be smaller). ~2k px balances peak RAM during inference (~10 GB vs. ~37 GB at 3000 px) against
scheduling overhead, and 2048 exactly (2¹¹) aligns the tile grid with the global store's shard
grid — one inference tile is one output shard (ADR-008 D3). The read-tile size is independent
of the store's on-disk chunk size: the mosaic is written with larger 4096×4096 chunks at ingest
(`INGEST_CHUNK_SIZE`; 1 ingest chunk = 2×2 inference tiles), and `load_chunk` reads the
2048×2048 sub-tile out of them via `zarr.Array.oindex` with no alignment requirement.

`filter_chunks_by_roi_mask` then drops any chunk whose footprint does not intersect the ROI
zarr mask produced by `generate_roi`. Only the surviving **live chunks** are dispatched to
GPU actors. This matters for sparse ROIs: a polygon inscribed in a large bounding rectangle
may leave the majority of chunks empty, and a GPU actor takes tens of seconds just to open
the zarr store, read SCL, and detect emptiness — pure waste on a GPU-priced node. With
pre-filtering, only intersecting chunks reach the Ray cluster at all, and the cluster is
auto-sized from the live-chunk count.

Non-intersecting chunks have no staged Zarr on S3. Assembly re-runs the same ROI filter
and simply never writes their footprint, so it reads back as the store's fill value —
`0` (for `int8` embeddings / `uint16` obs counts) or `NaN` (for `float32` scales / std).
No placeholder zarrs, no placeholder chunk objects.

### 2. Ray Cluster Startup

`_start_ray_cluster()` resolves the cluster YAML at runtime by reading SSM parameters
(security-group ID, subnet IDs, instance-profile ARN, AMI ID, SSH key), writes a resolved
YAML to a tempfile, then runs `ray up`. The flow connects via Ray Client (`ray://head-ip:10001`).

The cluster is managed in a context manager so it automatically tears down after Step 5 completes.

**Cluster topology:**
- Head: m5.2xlarge — GCS + autoscaler, no inference work
- Workers: g5.xlarge (1× A10G 24 GB VRAM, 16 GB RAM) — on-demand, single AZ
- Workers use a Packer-built AMI with all dependencies pre-installed; boot ready in ~1 minute

### 3. Inference Actors

One `InferenceActor` (a Ray actor pinned to 1 GPU) is created per worker slot. On init each
actor downloads the model checkpoint from S3 to the local NVMe instance store (~1.5 GB/s),
loads `MultimodalBTInferenceModel`, and logs VRAM usage. A `ping()` call confirms readiness
before work is dispatched.

> **Why NVMe?** EBS sequential read saturates at ~42 MB/s. `torch.load` with mmap on EBS
> causes multi-minute hangs. NVMe is ~35× faster.

**Icechunk credentials (injected, not imported).** `InferenceActor` takes a `get_credentials`
callback and wraps each `process_chunk` in `storage.zarr_store.credentials_provider(...)`, so
every store open resolves through it. The AWS-aware orchestration layer injects
`providers.aws.credentials.iam_icechunk_credentials` (threaded
runner → scheduling → actor, including replacement actors); the actor itself imports no AWS
module so `inference/` stays domain-pure (enforced by `tests/architecture`). With no callback
injected (local/dev runs) icechunk uses its default credential chain.

This matters because long-lived actors outlive the credential resolved at startup. The Rust
SDK's default chain resolves the instance-profile credential once and may fail to refresh it;
when it lapses, the failing chunk **and every subsequent chunk on that actor** error with
`no providers in chain provided credentials`. Routing through `iam_icechunk_credentials`
(botocore-backed, self-refreshing) fixes this — but note the IMDS-throttling gotcha documented
in `ingest/README.md`: `_resolve_iam_credentials` must stay `lru_cache`d so the callback's
15-min re-invocation does **not** trigger a cold IMDS resolve each time.

### 4. Per-Chunk Inference (inside each actor)

Each chunk goes through four sub-steps:

#### 4a. Data Loading (`data_loading.py`)

Two-phase loading lowers peak worker RAM by dropping empty dates before the band read. This
caps RAM for typical chunks but does not *bound* it — peak still scales with `T_valid`, which
is what northing striping (4a′) addresses for dense chunks:

```text
All timesteps in the ROI time axis
T ≈ 126 dates  (full year, all S2 acquisitions for this ROI)
      │
      │  Phase 1: Load SCL only (~200 MB)
      │  Drop dates with no SCL-valid pixels in this spatial chunk.
      │  (Most chunks intersect only a fraction of the ROI's date range.)
      ▼
T_valid ≈ 63 dates  (roughly halved for large ROIs)
      │
      │  Phase 2: Load full reflectance bands for all T_valid dates.
      │  v1.1 uses all valid observations (bucketed sampling, not pre-sampled).
      ▼
Peak RAM: ~2 GB   (vs. ~15 GB if loaded without SCL pre-filter)
```

All phases read directly from the raw zarr group via `open_store_as_zarr_group` /
`zarr.Array.oindex`, bypassing xarray and dask entirely. A single icechunk session is opened
per store; time-window filtering and DOY extraction decode the `int64` time coordinate
directly from `root["time"]`. `data_loading.py` imports neither `xarray` nor `dask`.

S1 SAR: VV is read first to identify non-empty timesteps, then VH is loaded for survivors.
Both ascending and descending stores are loaded when `s1_orbit="both"`;
`resolve_s1_orbit(mosaic_base, s1_orbit)` probes for available stores and falls back
gracefully when only one orbit was ingested.

Output: `ChunkData` — numpy arrays for S2 bands/masks/DOYs, S1 ascending/descending
bands/DOYs, and per-pixel observation counts (`s2_obs_count`, `s1_asc_obs_count`,
`s1_desc_obs_count`) as uint16 arrays of shape (H, W).

#### 4a′. Northing Striping (bounds peak input RAM independent of T)

The SCL pre-filter above caps peak RAM for *typical* chunks, but it does not bound it: the
resident S2 band array is `T_valid × H × W × 10 × 2` bytes, which scales with the timestep
count. On dense ROIs `T_valid` can reach ~120, so a 2048×2048 chunk's `s2_bands` alone is
~9.6 GB in a single `np.empty` — and on a 16 GB `g5.xlarge` worker that OOMs the loader
*before* inference runs. `T` is not a free variable (v1.1 uses every valid observation), so
the only lever is the spatial working set.

`process_chunk` therefore loads each chunk as a sequence of **northing strips** (full easting
width) rather than all at once. `load_chunk(..., y_sub=<slice>)` reads only a chunk-relative
horizontal band; each strip is a self-contained `ChunkData` that is bucketed, run through
inference, and written into whole-chunk output buffers by row-slice. Only the *inputs* are
sub-tiled — the int8 output buffer (`H × W × 128`, ~0.5 GB) is held whole, so the write path,
obs-count maps, and `assembly.py` are untouched (a single `write_chunk` at the end). Strips
are loaded through a 1-deep prefetch pipeline (strip *i+1* loads while strip *i* runs the
GPU), the same pattern as the sub-batch prefetcher in `inference.py`.

The strip height is derived per chunk from its **true post-prune timestep count** `T_kept`,
read from a full-chunk SCL mask loaded once up front (see below). The strip loop runs a 1-deep
prefetch (strip *i+1* loads while strip *i* runs the GPU), so once a chunk splits **two strips
are resident at once**; the byte budget is the bound on a single strip, sized so the pair holds
under ~90% of the default 16 GB worker box. A chunk that fits in one strip loads whole (no
prefetched neighbour) — byte-for-byte the unstriped path.

Two reads are decoupled. **SCL** (1 byte/px) is loaded once for the whole chunk
(`load_s2_mask_bundle`) and sliced per strip — it is never re-decompressed, and it doubles as
the `T_kept` source for sizing. **Reflectance bands** (20 bytes/px) are still read per strip on
the chunks that split; that per-strip read is exactly the working set the byte budget bounds.

#### 4b. Valid Pixel Filtering + Bucketing (`dataset.py`)

`MosaicChunkInferenceDataset` identifies pixels eligible for inference and groups them
into `(s2_bin, s1_bin)` buckets for batched processing:

1. **Valid pixel mask** — at least one non-zero S2 observation AND at least one non-zero
   S1 observation. S1 floor of 1 prevents the sampler from crashing on all-zero SAR tiles
   at orbit coverage edges (pixels with zero valid S1 would be OOD for the model anyway).

2. **Bin key assignment** — for each valid pixel, `compute_bin_keys` maps observation
   counts `(s2_obs_count, s1_obs_count)` to the nearest entry in `num_obs_checkpoints`
   (the bucketed sequence-length schedule). Pixels with the same `(s2_bin, s1_bin)` key
   form a bucket — the transformer receives a rectangular `(B, seq_len, bands+1)` tensor
   for each bucket. No padding, no masking, no variable-length overhead.

3. **Deferred extraction** — pixel coordinates for each bucket are stored at init time;
   actual band data is loaded per-batch via fancy indexing. Previously, pre-extracting all
   valid pixels doubled peak RAM (source arrays 14 GB + extracted copy 17 GB = OOM).
   Per-batch indexing adds ~7 ms per batch and avoids the spike entirely.

#### Skip path and skip markers

When every pixel in a live (ROI-intersecting) chunk fails the validity filter, the chunk
takes the `"skipped"` path. The actor writes a zero-byte `{chunk.label}.skipped` marker
and returns. Assembly fills the footprint with constant-zero/NaN fill tasks. The marker
distinguishes a legitimate skip from a silently-failed chunk; `verify_staged_completeness`
requires every live chunk to have either a staged zarr or a skip marker.

#### 4c. Temporal Sampling (`sampling.py`)

For each bucket `(s2_bin, s1_bin)`:

- **`resample_s2_bucket`** — for each pixel, selects `s2_bin` timesteps from the valid
  S2 observations (deterministic, no random repeats). Returns `(B, s2_bin, 12)` including
  normalised bands + DOY feature.
- **`resample_s1_bucket`** — loads ascending + descending observations and concatenates
  per-modality-normalised VV/VH pairs + DOY. Returns `(B, s1_bin, 3)`.

Each modality uses its own `(mean, std)` from `S1_ASC_BAND_MEAN/STD` and
`S1_DESC_BAND_MEAN/STD` in `config/inference.py`. Ascending and descending are normalised
separately before concatenation so that the model sees per-orbit statistics, not blended ones.

> **Why `s1_orbit="both"` is safe for v1.1:** The v1.1 model uses a single merged S1
> backbone (`split_s1_modalities=False`), but Cambridge confirmed that ascending and
> descending observations are each normalised with their **own** mean/std before being
> concatenated — exactly matching v1.1 training-time preprocessing. This makes mixing
> both orbits correct and preferred, in contrast to v1 where the two orbits shared
> normalisation statistics.

`build_resample_indices` handles under-sampled pixels (fewer valid timesteps than the
bucket target) via deterministic repeat-padding — last valid timestep is duplicated until
target is reached. Over-sampled pixels are uniformly sub-sampled.

#### 4d. GPU Forward Pass (`inference.py`)

- **Bucket loop** — `iter_buckets(largest_first=True)` yields buckets from largest to
  smallest; the largest bucket sets the "hot" GPU allocation so smaller buckets reuse it.
- **Prefetch thread** — a `ThreadPoolExecutor(max_workers=1)` pipelines CPU batch
  preparation (data loading + normalization) while the GPU runs the previous forward pass.
- **FP16** — `model.half()` halves memory bandwidth; bucket-level aggregation absorbs FP16
  noise without repeat-averaging.
- **`torch.compile` is disabled** — on g5.xlarge (15.4 GB VRAM), CUDA graph capture
  consumed 11.6 GB and slowed forward passes (3,770 ms vs. 1,944 ms) due to GRU
  recompilation per unique sequence length.
- **cuDNN benchmark mode** — fastest algorithm for GRU/conv ops.
- **Throughput:** ~1,300 px/sec per A10G; 100 actors ≈ 130,000 px/sec.

Output per chunk: `embeddings` array (H, W, 128) int8 with zeros for invalid pixels,
plus a per-pixel float32 `scale` factor for dequantization. The model produces 192-D
representations; only the first 128 dimensions are saved (`save_dim = min(128, repr_dim)`).

### 4e. Quantization (`quantization.py`)

Embeddings are always compressed from float32 to int8 immediately after the forward pass,
before staging. This reduces staged and final store sizes by ~4×.

**How it works:** For each pixel, the absolute maximum across all 128 embedding channels
is computed. Embeddings are scaled so that the max maps to ±127 (the int8 range), then
rounded and clipped. The per-pixel scale factor (float32) is stored alongside the
quantized embeddings so the original values can be reconstructed:

**Per-bucket, not whole-chunk.** Quantization is per-pixel independent (each pixel's scale
comes only from its own 128 channels), so `run_inference` quantizes each bucket's rows via
`quantize_rows` the moment they come off the GPU, accumulating directly into skinny int8 +
scale buffers. It never materializes the full `(H, W, 128)` chunk in float32. This is
numerically identical to quantizing the whole array at the end, but shrinks the resident
accumulator ~4× (~2 GB → ~0.5 GB at `chunk_size=2048`) and removes the end-of-chunk
whole-array quantize and its multi-GB float32 temporaries — lowering the per-chunk host-RAM
plateau. `quantize_embeddings` remains as the `(H, W, D)` entry point and delegates to
`quantize_rows`.

```
reconstructed = quantized.astype(float32) * scale[..., np.newaxis]
```

Round-trip error is bounded by `scale / 127` per channel — typically <1% of the original
magnitude. Non-finite values (NaN, Inf) in embeddings are rejected before quantization
with a `ValueError`.

**Output variables:**
- `embeddings`: int8, shape (H, W, 128)
- `scales`: float32, shape (H, W)
- `embedding_std`: float32, shape (H, W, 128) — unaffected by quantization, present only
  when `compute_std=True`

**Safety checks:** Assembly validates that staged chunks have the expected int8 dtype
for embeddings and float32 for scales. Dtype mismatches are rejected to prevent data
corruption.

### 5. Staged Chunk Writes (`assembly.py`)

Each actor writes its chunk to S3 staging at `{staging_base}/{run_id}/{chunk_label}.zarr`
as **raw, uncompressed** zarr (zero codec CPU on GPU-priced nodes). Staged sub-chunks are
`256×256 × full band` (`INNER_PX`, int8) — exactly the final store's inner-chunk geometry,
so a staged 2048-px tile *is* the 8×8 inner-chunk grid of the output region it becomes,
and the band axis is never split (ADR-008 D2).

Alongside embeddings, each staged chunk includes float32 `scales` and three **observation
count** layers (`s2_obs_count`, `s1_asc_obs_count`, `s1_desc_obs_count`) — uint16 (H, W)
arrays recording how many valid timesteps contributed to each pixel. These are carried
through assembly into the final store as 2D spatial variables (dims: `time, northing,
easting`).

### 6. Raw-Zarr Assembly (fork/merge, no Dask)

After all live chunks complete, `writer.assemble` writes the staged tiles straight into
the final Icechunk store with plain zarr assignments — a pool of worker processes on the
flow runner, each holding a pickled **fork** of one icechunk session, merged and committed
once by the coordinator. No task graph is ever built over the store, so assembly cost
scales with the *live* pixels, not the grid, and the scheduler-RAM ceiling that shaped the
old Dask engine is gone.

The output geometry comes from a `StoreLayout` preset (`config/store_layout.py`) — the
same source of truth the global store's seeder uses — not from the staged files:

```text
Preset       embeddings                          scales / obs / std
──────────────────────────────────────────────────────────────────────────
LEGACY       (1, 500, 500, 4)  int8+zstd         (1, 500, 500)  PCodec / raw
             unsharded — today's single-ROI      unsharded
GLOBAL_V1    (1, 256, 256, 128) int8+zstd        (1, 256, 256)  PCodec/zstd
             in (1, 2048, 2048, 128) shards      in (1, 2048, 2048) shards
```

#### Write-conflict discipline: northing bands

Two forks writing the same output chunk would conflict at merge, and LEGACY's 500-px
chunks don't align with 2048-px tiles. So workers partition the mosaic into **northing
bands aligned to the output's write granularity** (shard height when sharded, chunk height
otherwise) — bands are disjoint and span the full easting extent, so no two forks ever
touch the same output object. Band boundaries are **work-weighted** (live tiles per
granularity unit), not equal-height: an ROI mask clusters live tiles spatially, and
equal-height bands would leave most workers idle while one dragged the assembly:

```text
                    easting ─────────────────►
            ┌tile 0───────┬tile 1───────┬tile 2──┐
  worker 0  │             │             │        │  band [0, 1000)     ── fork 0
            │ ┈┈┈┈┈┈┈┈┈┈┈ band boundary ┈┈┈┈┈┈┈┈ │  (multiple of 500)
  worker 1  │             │             │        │  band [1000, 2048+…)── fork 1
            └─────────────┴─────────────┴────────┘
             a tile straddling a boundary is read
             twice, once per band, as a y-slice —
             each worker streams ONE tile-slice at
             a time (≤ ~0.5 GB int8)

  coordinator:  session.fork() ──► workers write bands ──► merge(*forks)
                └── one data commit via commit_with_rebase ──┘
```

Within a band, tile x-boundaries still cut output chunks mid-chunk; those partial chunks
are read-modify-written *sequentially inside one fork* — icechunk sessions are
read-your-writes, so the merged result is exact.

#### Assembly steps

1. Re-run `filter_chunks_by_roi_mask` to recover the set of live chunk labels (the list
   isn't marshaled through Prefect; the ROI zarr path is the source of truth), minus
   skip-marked chunks.
2. **Phase 1 (metadata commit)** — fresh store: create the layout's array schemas + coord
   arrays (no chunk data; cost independent of extent). Existing store: validate the
   `_manifest`, then either resize every time-dimmed array by one step (an append IS a
   resize + write at the new index) or, when the time value already exists, target its
   index for an in-place overwrite — **idempotent resume**: a crashed assembly re-run
   lands on the same index instead of appending a duplicate timestep.
3. **Phase 2 (data commit)** — fork the session; each worker process writes its band's
   staged-tile y-slices into every output array; the coordinator merges the forks, sets
   root attrs (run metadata, GeoZarr conventions, `_manifest`, merged `time_windows`),
   and commits once via `commit_with_rebase`.
4. Delete the staged chunk zarrs (unless `cleanup_staging=False`).

The **global-campaign variant**, `writer.assemble_global`, skips the banding entirely:
one staged 2048-px tile is exactly one output shard (D3), so whole tiles round-robin
across workers via `storage.shard_writer.write_year_shards` — every shard object is
emitted once, lean (all-fill inner chunks elided: fully-masked tiles cost nothing and
no-valid-observation inner chunks vanish; note water pixels ARE SCL-valid, so coastal
tiles embed their ocean pixels — the mask selects tiles, not pixels), with
`years_complete` and per-year run provenance advanced in the same single commit. The
zone-fill runner (`orchestration/runners/zone_fill.py`) drives it: coverage mask →
inference → `assemble_global` → `campaign.tag_zone_year`.

The **campaign land mask** is not a pixel ROI but a per-zone *coverage bitmap*
(`tile_live_2048`) built from the partner's delivery registry by
`ingest/land_mask.py` and the `build-land-mask-coverage` flow (ADR-010). v1.1 tiles are
all-1s with a ~1-cell sea buffer, so the registry listing *is* the mask: a 2048-px tile
is live iff a land cell's footprint intersects it. The runner reads one small bitmap
(one GET) and selects live tiles by direct index — 1 tile == 1 shard == 1 coverage tile
(D3) — rather than the per-chunk windowed reads `filter_chunks_by_roi_mask` does for the
single-ROI flows.

#### Write units vs read units, per layout

The write path and the read path deliberately touch different granularities —
what a writer emits in one go is much larger than what a reader must fetch:

```text
                  LEGACY (single-ROI)             GLOBAL_V1 (zone group)
─────────────────────────────────────────────────────────────────────────────
S3 object       = one 500×500×4 chunk (~a few   = one 2048² shard (≤ ~0.5 GB:
                  hundred KB)                     8×8 inner chunks + index)

writer emits    band worker streams tile        shard worker emits whole
                y-slices; partial edge chunks    shard objects, exactly once
                read-modify-write in-fork        (never read-modify-write)

reader fetches  whole chunk objects under       shard index (one small GET),
                the window (~KBs per point,      then ranged-GETs of only the
                ×32 band chunks for full         overlapped inner chunks
                depth)                           (~8.4 MB per point, full band)

commit rewrites manifest tiles the write        that year's manifests only
                touched (32-chunk 2D split)      (time@1 split)
```

The commit row is the manifest-splitting story: an Icechunk manifest maps
chunks → objects, one per array by default, so unsplit commits are O(store).
See the README's "Manifest splitting" diagram for the visual.

**S3 concurrency.** The coordinator opens the repo with `max_concurrent_requests =
TARGET_AGGREGATE_S3_CONCURRENCY // n_workers`; forks inherit it through pickling (no
`save_config` round-trip needed), so fleet-wide PUT concurrency stays under S3's
per-prefix ceiling regardless of worker count.

**Manifest splitting.** `assemble` opens the repo under
`manifest_split({"northing": 32, "easting": 32, "time": 1})`. By default icechunk keeps
one manifest object per array, so every commit rewrites the entire chunk index — O(store
size) regardless of how few chunks changed. Splitting tiles the manifest into
32-chunk-per-axis spatial shards (~16k px at 500-px chunks) plus one shard per timestep,
so a one-timestep write rewrites no prior timestep's manifests. The global store bakes
its own split (`time@1`) into the repo config at seeding (`global_store_config`).

The split applies to both new and pre-existing stores. A store created before this change
has a single unsplit manifest; the first write under the `manifest_split` block re-shards
the touched array in that commit (a one-time migration — old chunk data is untouched and
reads back unchanged), and every commit after it rewrites only the shards it touches.

**Codecs in the final store:** quantized int8 embeddings use the zarr default
(bytes + zstd). The float32 `scales` (and `embedding_std`, when computed) use **PCodec**,
a floating-point-aware serializer that models the distribution of float values directly
and beats general-purpose codecs on this kind of data. The tradeoff is that PCodec
decompresses the entire on-disk chunk to read any slice of it (no partial decode), which
is why the inner chunks stay small (500 px legacy / 256 px global). PCodec composes with
the sharding codec (verified: round-trip + fill on partial shards).

**Reading the output store.** On a CONUS-scale store, open with `chunks=None` for
interactive or selective reads:

```python
ds = open_store(store_path, chunks=None)   # or xr.open_zarr(session.store, consolidated=False, chunks=None)
ds.isel(time=0).sel(northing=slice(...), easting=slice(...)).embeddings.values
```

The default chunking builds one dask task per on-disk chunk. With 500×500×4
sub-chunks the band axis alone is 32 chunks (128/4), so the graph is
`n_time × n_y × n_x × 32` tasks — large enough at CONUS extent that even a *lazy*
`isel`/`sel` OOMs while manipulating the graph, before any chunk data is read.
`chunks=None` opens the store zarr-lazy with no dask graph: slicing is pure
metadata and chunks load only when `.values` is pulled. This is unrelated to
manifest splitting — the split bounds commit/manifest cost, not the dask graph.

---

## Fault Tolerance

All fault handling lives in `scheduling.py`. `_process_chunks_work_stealing` owns
the main loop; `ActorPool` encapsulates actor state and lifecycle operations
(replacement, idle retirement, instance-ID resolution).

| Failure mode | Handling |
|---|---|
| Chunk raises exception | Re-queued up to 2 times (`max_chunk_retries`); logged as permanently failed if exhausted |
| Actor dies (OOM, instance loss) | `ActorPool.replace()` spawns a replacement; instance ID of the new node resolved lazily so the main loop isn't stalled |
| >50% of actor slots dead | `ActorPool.replace()` escalates log severity to ERROR / CRITICAL |
| Replacement actor still initialising | `dispatch_idle()` queues work to it anyway — Ray buffers the call until `__init__` completes |
| Idle actor after work drains | `ActorPool.retire_idle()` kills actors idle past `idle_grace_sec` (default 120s), freeing GPU nodes; never drops below remaining work count |
| Chunk stalls (no batch update for 5 min) | `ProgressTracker` detects per-chunk staleness; `_poll_tracker()` aborts if ≥3 chunks stall simultaneously |
| Flow cancelled in Prefect UI | `on_cancellation` hook runs `ray down` |

---

## Key Configuration (`config.py`)

| Parameter | Default | Notes |
|---|---|---|
| `batch_size` | 3584 | GPU pixels per forward pass within a bucket |
| `num_obs_checkpoints` | `range(8, 257, 8)` | Bucketed sequence-length schedule; pixels binned to nearest checkpoint |
| `s1_orbit` | `"both"` | `"ascending"`, `"descending"`, or `"both"` |
| `norm_source` | `"mpc"` | Band stats origin; `"aws"` for the AWS-normalised encoder checkpoint |
| `latent_dim` | 192 | Transformer hidden dim (must match checkpoint) |
| `representation_dim` | 192 | Model output dim; first 128 dims are saved to the store |
| `dim_feedforward` | 2048 | Transformer FFN width |
| `max_gpu_workers` | 500 | Ray autoscaler ceiling |

**Architecture params** (`nhead`, `num_encoder_layers`, `dim_feedforward`, etc.) **must match
the checkpoint**. See `models/README.md` before touching them.

---

## Model Architecture Constraint

`models/` is ported from `tessera_infer` and must stay in sync with the checkpoint. Do **not**
change layer dimensions, activations, or the forward pass unless you are also retraining.
`builder.py` strips FSDP state-dict prefixes on load — if the checkpoint format changes,
that function needs updating.

---

## Provenance

Several modules are ported from the original `tessera_infer` repository:

| File | Ported from | Changes |
|---|---|---|
| `sampling.py` | `tessera_infer/src/multi_tile_infer.py` | v1.1 bucketed deterministic sampling (replaces random repeat-averaging). |
| `inference.py` | `tessera_infer` process_tile logic | v1.1 bucket loop with prefetch thread; removes repeat/averaging path. |
| `models/` | `tessera_infer/src/models/` | See `models/README.md`. MLP `dim_reducer`, encoder-only checkpoint loading. |

New code (not ported):

| File | Purpose |
|---|---|
| `config.py` | `InferenceConfig` dataclass with band statistics and architecture defaults |
| `chunk_spec.py` | Spatial chunk grid enumeration, mosaic-agnostic |
| `data_loading.py` | Icechunk/Zarr loader with 3-phase selective S2 timestep loading |
| `dataset.py` | Valid-pixel filtering and lazy per-batch indexing |
| `quantization.py` | Int8 quantization with per-pixel scale factors and dequantization |
| `assembly.py` | Staged Zarr writes + raw-zarr fork/merge assembly into Icechunk (single-ROI + global modes) |
| `actors.py` | Ray actor wrapping the full per-chunk pipeline |
| `scheduling.py` | Work-stealing scheduler (`_process_chunks_work_stealing`) and `ActorPool` — manages dispatch, actor replacement, idle retirement, and tracker polling |
| `progress.py` | Lightweight Ray actor (`num_cpus=0`) polled by the flow runner every 60s for batch-level progress and 5-minute stall detection |

---

## Accessing the Dask Dashboard

Assembly no longer uses Dask — only the *ingest* Dask cluster remains, in a private subnet
(ECS Fargate). Use SSM port forwarding to reach its Bokeh dashboard:

```bash
# Look up TASK_ID and RUNTIME_ID for the Dask scheduler task in the ECS console
aws ssm start-session \
  --target ecs:yield-cluster_${TASK_ID}_${RUNTIME_ID} \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters '{"host":["localhost"],"portNumber":["8787"],"localPortNumber":["8787"]}'
```

Then open http://localhost:8787 in your browser.

Requires the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (`brew install session-manager-plugin`).
