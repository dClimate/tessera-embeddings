# Tessera Inference Pipeline

Distributed GPU inference that generates **128-dimensional per-pixel embeddings** from mosaicked
Sentinel-2 reflectance + Sentinel-1 SAR data. The Prefect flow at
[`orchestration/prefect/flows/tessera_embeddings.py`](../orchestration/prefect/flows/tessera_embeddings.py)
orchestrates the full process: spinning up a Ray cluster on EC2 GPU instances, running inference
in parallel across spatial chunks, and assembling the output into a final Icechunk/Zarr store.

The orchestrator-free equivalent is
[`orchestration/runners/plain.py`](../orchestration/runners/plain.py), which calls the same
domain functions on `ray_cluster(num_gpus=0)` for laptop/CI runs.

**Performance.** On g6e.xlarge (L40S) workers the pipeline sustains **~89–93% GPU utilization**
and **~21–24K pixels/sec per worker** on mid-density chunks (~10–18K on dense), with **peak host
RAM ~52%** of the 30.9 GB node (budgeted to stay under 60% at UTM-zone scale). Versus the naive
baseline that's **~2–2.8× per-worker throughput**. Outputs match the `main` reference within the
ADR-012 cross-config equivalence envelope (a batch-size difference, not a regression). The mechanisms — vectorised prep, an async GPU
loop, valid-pixel-aware striping, and a RAM-bounded cross-chunk starter prefetch — are detailed
phase-by-phase below, then synthesized into a decision tree with relative impact in
[**How Our Performance Optimizations Fit Together**](#how-our-performance-optimizations-fit-together); full profiling and
gotchas are in
[`context_docs/design/inference_gpu_saturation_profile_2026_07.md`](../../../context_docs/design/inference_gpu_saturation_profile_2026_07.md).

---

## Architecture at a Glance

```
Input stores (Icechunk/Zarr on S3):
  reflectance.zarr / sar_ascending.zarr / sar_descending.zarr
            │
            ▼
  enumerate_chunks_from_dataset()     ← 2000×2000 px chunks
            │
            ▼
  filter_chunks_by_roi_mask()         ← drop chunks outside the ROI
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │  Ray Cluster  (EC2, g6e.xlarge × N)         │
  │  ┌─────────────────────────────────────┐    │
  │  │ InferenceActor (1 GPU each)         │    │
  │  │  per northing strip (bounds RAM):    │    │
  │  │   1. load_chunk(y_sub=…)  ← selective│    │
  │  │   2. Dataset valid-px filter         │    │
  │  │   3. sample_s2/s1_batch()            │    │
  │  │   4. model forward  (BF16, B=7168)   │    │
  │  │  5. writer.write_chunk() → staging   │    │
  │  └─────────────────────────────────────┘    │
  │  Work-stealing: actors pull from queue      │
  └─────────────────────────────────────────────┘
            │
            ▼
  Staged chunks (S3 staging per run_id) — live chunks only
            │
            ▼
  ┌────────────────────────────────────────┐
  │  Dask Cluster (ECS Fargate, 20 workers)│
  │  writer.assemble() → to_icechunk()     │
  │  (non-live chunks filled with 0/NaN    │
  │   directly in the Dask graph)          │
  └────────────────────────────────────────┘
            │
            ▼
  Output:  {roi_name}.zarr  (Icechunk, append on re-run)
           embedding dims: (time, northing, easting, 128)
           obs count dims:  (time, northing, easting)  — s2/s1_asc/s1_desc
```

---

## Pipeline Phases

### 1. Chunk Enumeration and ROI Pre-Filter

The input mosaic is divided into a grid of 2000×2000 pixel `ChunkSpec` objects (edge chunks may
be smaller). 2000 px balances peak RAM during inference (~10 GB vs. ~37 GB at 3000 px) against
scheduling overhead. This read-tile size is independent of the store's on-disk chunk size: the
mosaic is written with larger 4000×4000 chunks at ingest (`INGEST_CHUNK_SIZE`), and `load_chunk`
reads the 2000×2000 sub-tile out of them via `zarr.Array.oindex` with no alignment requirement.

`filter_chunks_by_roi_mask` then drops any chunk whose footprint does not intersect the ROI
zarr mask produced by `generate_roi`. Only the surviving **live chunks** are dispatched to
GPU actors. This matters for sparse ROIs: a polygon inscribed in a large bounding rectangle
may leave the majority of chunks empty, and a GPU actor takes tens of seconds just to open
the zarr store, read SCL, and detect emptiness — pure waste on a GPU-priced node. With
pre-filtering, only intersecting chunks reach the Ray cluster at all, and the cluster is
auto-sized from the live-chunk count.

Non-intersecting chunks have no staged Zarr on S3. Assembly re-runs the same ROI filter
and fills their footprint with `0` (for `int8` embeddings / `uint16` obs counts) or `NaN`
(for `float32` scales / std) as constant Dask tasks directly in the mosaic graph — no
placeholder zarrs are written.

### 2. Ray Cluster Startup

`_start_ray_cluster()` resolves the cluster YAML at runtime by reading SSM parameters
(security-group ID, subnet IDs, instance-profile ARN, AMI ID, SSH key), writes a resolved
YAML to a tempfile, then runs `ray up`. The flow connects via Ray Client (`ray://head-ip:10001`).

The cluster is managed in a context manager so it automatically tears down after Step 5 completes.

**Cluster topology:**
- Head: m5.2xlarge — GCS + autoscaler, no inference work
- Workers: g6e.xlarge (1× L40S 48 GB VRAM, 4 vCPU, 32 GB RAM) — on-demand, single AZ
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

Two-phase loading exploits **temporal sparsity** — it drops empty dates before the band read
(timestep pruning). This caps RAM for typical chunks but does not *bound* it — peak still
scales with the post-prune `T_valid`, which is what northing striping (4a′) bounds for
high-valid-pixel-count chunks:

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
count. On dense ROIs `T_valid` can reach ~120, so a 2000×2000 chunk's `s2_bands` alone is
~9.6 GB in a single `np.empty` — on the earlier 16 GB g5-class workers that OOMed the loader
*before* inference runs, and even on today's 32 GB g6e.xlarge it must share the box with the
SAR stack, output buffers, and model. `T` is not a free variable (v1.1 uses every valid
observation), so the only lever is the spatial working set.

`process_chunk` therefore loads each chunk as a sequence of **northing strips** (full easting
width) rather than all at once. `load_chunk(..., y_sub=<slice>)` reads only a chunk-relative
horizontal band; each strip is a self-contained `ChunkData` that is bucketed, run through
inference, and written into whole-chunk output buffers by row-slice. Only the *inputs* are
sub-tiled — the int8 output buffer (`H × W × 128`, ~0.5 GB) is held whole, so the write path,
obs-count maps, and `assembly.py` are untouched (a single `write_chunk` at the end). On dense
chunks strips are loaded through a 1-deep prefetch pipeline (strip *i+1* loads while strip *i*
runs the GPU), the same pattern as the sub-batch prefetcher in `inference.py`; on chunks with
few valid pixels that pipeline is turned off (see the strip plan below).

The tiling is chosen per chunk by `_strip_plan`, from the full-chunk SCL mask loaded once up
front: its **true post-prune timestep count** `T_kept` sets the byte cost, and its valid-pixel
count estimates how long inference will run. Read bytes scale with `T_kept × H × W` (independent
of how many pixels are valid); inference time scales with valid pixels — so the two can diverge,
and the plan picks the strategy that fits:

```text
 per_set = T_kept·H·W·(20 bands + 1 mask)          budget = _S2_STRIP_BYTE_BUDGET (5.75 GiB)

 per_set ≤ budget ─────────────────────────────▶ SINGLE strip           (most chunks)
 else, inference hides the loads (dense) ───────▶ balanced strips ≤ budget, PREFETCH on
        └─ tall body? ─────────────────────────▶   + small "starter" strip so the GPU starts early
 else (wide bytes, few valid px → not hideable) ▶ strips ≤ PAIR budget, PREFETCH OFF
                                                    (only ONE set resident, so it may use 2× budget)
```

Peak host RAM is bounded the same way in every branch: with prefetch **on**, strip *i+1* loads
while strip *i* infers, so **two** band sets co-reside, each ≤ `_S2_STRIP_BYTE_BUDGET` (5.75 GiB)
→ pair ≤ ~11.5 GiB. With prefetch **off** the prior set is released before the next loads, so only
**one** set is resident and it may use the full pair budget — the same ceiling either way. That
holds peak host RAM under the 60% line of a 30.9 GB g6e.xlarge across UTM-zone-scale density variance
(striping alone measured **45–47% peak** at the 5.75 GiB budget; most chunks fit in a single strip,
so few hit the two-strip co-residency that sets that peak — see the budget constant's comment for the
arithmetic). The bounded cross-chunk starter prefetch (§4a″) adds ≤~2 GiB of stash during the current
chunk's last strip; on the shipped pipeline this lifts measured peak to **~52%** (run `a60550ae`) —
the stash is partly co-resident with the peak — still comfortably under the 60% ceiling.
Turning prefetch off on non-hideable chunks matters because a background load only helps if there
is inference to hide it behind — on a wide but few-valid-pixel chunk the load would otherwise sit naked on
the critical path *and* force a second co-resident set for nothing. A chunk that fits one budget
loads whole — byte-for-byte the unstriped path.

Two reads are decoupled. **SCL** (1 byte/px) is loaded once for the whole chunk
(`load_s2_mask_bundle`) and sliced per strip — it is never re-decompressed, and it doubles as
the `T_kept` source for sizing. **Reflectance bands** (20 bytes/px) are still read per strip on
the chunks that split; that per-strip read is exactly the working set the byte budget bounds.

#### 4a″. Chunk prologue: serial by default, with a bounded cross-chunk starter prefetch

Each chunk pays a serial, GPU-idle "prologue" — SCL mask read, first band read, dataset
build (~24–38 s) — before its first forward pass. A **bounded cross-chunk starter prefetch**
hides most of it: the next chunk's mask + a small starter strip (hard-capped ~2 GiB) are
preloaded during the current chunk's last strip. It deliberately does NOT prefetch the whole
next working set — that co-resides two full chunks and OOMs the node at UTM-zone-scale density
variance (the RAM-safety rationale and history are in the concept doc).

```text
 actor timeline, chunk N → N+1 (starter-prefetch hit)

 [═════════ inference N ═══════════][═ N+1 starter ═][═══ N+1 body ═══]
              [mask N+1][starter N+1]      [body N+1 loads]
  GPU:  busy ─────────────────────── busy ── busy ─────── busy
        └ prefetch runs during N's LAST strip (RAM trough) ┘
```

The scheduler reserves each actor's next chunk (1-deep, `ActorPool.reserved`, passed as
`prefetch_hint`; reservations stop when the queue is shallower than the live pool and are
requeued to the front on actor death). During the CURRENT chunk's last strip — its final
load is done, so the two-strip co-residency has decayed to the RAM trough — the actor
prefetches a **capped** payload for the hinted chunk: its SCL mask bundle + read plan, and,
when the rung allows, its 256-row starter strip. Rungs (`_xchunk_rung`): P2-tiled dense
chunks prefetch their (already small) starter free; single-budget chunks convert to
starter+body only when a net-gain check says the extra fixed read will hide; everything
else (budget-sized first strips, non-hideable few-valid-pixel plans, anything over the
`_XCHUNK_PREFETCH_CAP_BYTES` ~2 GiB cap) gets the mask only. Every miss — cap, steal
(label mismatch evicts the stash), credential-window expiry, load failure, or the
`TESSERA_DISABLE_XCHUNK_PREFETCH=1` hatch — degrades to the serial prologue: slower,
never bigger. Peak host RAM keeps the same pair ceiling (see `_S2_STRIP_BYTE_BUDGET`).

Within a chunk, overlap remains: **bucketing rides the strip-prefetch thread** —
`_load_strip` returns `(ChunkData, dataset)`, so the `MosaicChunkInferenceDataset` build
(valid-pixel filtering + bucketing, ~10 s) overlaps the prior strip's GPU work on split
chunks instead of sitting between load and inference. Background strip loads also
**reserve 2 cores** for the batch-prep workers feeding the GPU (`reserve_cpus` in
`load_chunk`), so band decompression can't starve inference on the 4-vCPU box.

**The staging write is deferred** to a single-slot writer thread, overlapping the next
chunk's prologue load (both are I/O; the GPU is idle either way). Durability: the result
returns `write_deferred=True` and the scheduler holds the chunk out of the completed set
until the write outcome arrives — piggybacked as `prior_write` on the actor's next
result, or drained via `flush_writes()` when the actor idles. Failed writes requeue the
chunk (no actor kill); an actor death with a write in flight requeues too — safe because
staged writes are idempotent. A chunk's "done" can therefore trail its inference by up
to one chunk in the progress logs.

Every in-actor wait on a background I/O future — the prior chunk's deferred write, and
the strip / cross-chunk-starter prefetch loads — is bounded by `_BACKGROUND_IO_TIMEOUT_S`
(600 s, matching the scheduler's `flush_writes()` RPC timeout). A wedged S3/zarr client
with no socket timeout would otherwise hang `process_chunk` itself, where the scheduler's
tail-flush recovery can never reach it (Ray serialises actor calls, and a 1–2-actor run
never hits the ≥3-stall abort). A **timeout** always fails the chunk so the scheduler
replaces the actor (reaping the wedged worker) and requeues — critically because the
writer and prefetch pools are single, *persistent* workers, so a stuck task would poison
every later write/prefetch. Only a background strip's pool is per-chunk and managed
explicitly (not `with`), so its timeout `raise` escapes instead of being re-swallowed by
`ThreadPoolExecutor.__exit__`'s blocking `shutdown(wait=True)`. A prefetch that merely
*errors* (worker free) still degrades gracefully to the serial prologue.

**Spatial sparsity — read less**: strips whose SCL-mask slice has zero valid pixels skip
the S2 band read entirely (empty-strip skip), and chunks whose valid pixels span a narrow
easting window read S2 only for that column bounding box (`x_sub`; applied when it saves
≥10% of the width). SAR is
still read full-width and S2 obs come from the mask bundle, so the saved obs-count
layers keep full-extent fidelity — outputs are bit-identical, only the bytes read
change. A chunk_7_0-class sliver (1.5K valid px, T_kept=82) drops from ~39 s of loading
to roughly bbox-proportional cost.

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
- **Prefetch pool** — two prep workers keep `PREFETCH_DEPTH = 2` batches staged. The
  resamplers (`sampling.py`) are vectorised: per-pixel resample indices depend only on
  `(valid_count, target)` and are memoised, so a sub-batch costs a handful of `np.unique`
  lookups plus large gathers instead of `2 × batch_size` Python iterations (~5× faster;
  bit-identical to the loop, enforced by golden reference tests).
- **Two-deep GPU pipeline** (CUDA; `TESSERA_SERIAL_GPU_LOOP=1` reverts to the sync loop) —
  inputs stage through pinned double-buffers (async H2D), outputs copy back asynchronously
  with a CUDA event per batch, and the host drains one batch behind. Batch *i+1*'s work is
  enqueued while *i* executes, so the GPU never idles on Python bookkeeping, transfers, or
  the scatter-back. Same op order on the same stream as the serial loop → bit-identical.
  Finiteness is validated on host-side scales after D2H (`raise_on_nonfinite_scales`) —
  the old on-device `isfinite` check forced a full sync per sub-batch.

```
 serial loop:     [H2D][═ fwd i ═][D2H][scatter][H2D][═ fwd i+1 ═][D2H][scatter]
                                   GPU idle ↑↑↑ between every sub-batch

 pipelined loop:  [H2D][═ fwd i ═][D2H][H2D][═ fwd i+1 ═][D2H][═ fwd i+2 ═] …
    host thread:        …enqueue i+1… …drain i… …enqueue i+2… …drain i+1…
```
- **BF16** — the model is cast to BF16 on CUDA (`_prepare_gpu`); FP16 is a best-effort
  fallback for pre-Ampere GPUs only (overflow risk above 65504).
- **`torch.compile` is disabled** — in a historical experiment on a g5-class worker
  (15.4 GB VRAM), CUDA graph capture consumed 11.6 GB and slowed forward passes
  (3,770 ms vs. 1,944 ms) due to GRU recompilation per unique sequence length.
- **cuDNN benchmark mode is disabled** — with variable bucket shapes the autotuner
  re-searches constantly and inflates host RAM (see `_prepare_gpu`).
- **GRU is a fused cuDNN `nn.GRU`** — `builder._fuse_custom_gru` replaces the
  checkpoint-faithful `CustomGRU` reference with a single fused `nn.GRU` (~1 kernel launch
  vs ~480) before inference, so the pooling head's recurrence is not launch-bound. (This
  is a small, deliberate reset-gate approximation; see the builder docstring.)
- **Positional encoding without the zeros buffer** — sin/cos are written straight into an
  uninitialised output buffer instead of scatter-writing into a multi-GB FP32 `zeros`
  allocation per forward. Bit-identical values, lower peak memory.
- **Throughput:** ~10–12K px/sec per worker (measured 2026-07, Iowa ROI, L40S). px/sec
  is density-dependent — a sparse pixel costs ~10× less than a dense one — so the
  periodic and end-of-chunk summaries also log **tok/sec** (pixels × (T_s2 + T_s1))
  and **effective TFLOPS** (transformer-layer FLOPs via `profiling.transformer_flops`)
  for density-neutral comparison across chunks and runs.

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
accumulator ~4× (~2 GB → ~0.5 GB at `chunk_size=2000`) and removes the end-of-chunk
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
using Blosc compression (fast for read-writes). Zarr sub-chunks are 500×500 px —
smaller sub-chunks prevent multi-GB decompresses during assembly.

Alongside embeddings, each staged chunk includes three **observation count** layers
(`s2_obs_count`, `s1_asc_obs_count`, `s1_desc_obs_count`) — uint16 (H, W) arrays recording
how many valid timesteps contributed to each pixel. These are carried through assembly into the
final store as 2D spatial variables (dims: `time, northing, easting`).

### 6. Dask Assembly

After all live chunks complete, a Dask cluster (20-500 workers × 4 GB RAM) reads staged
chunks and assembles them into the final Icechunk store.

#### Three-layer chunk anatomy

The assembly design deliberately uses three different chunk granularities for three
independent concerns. Understanding this is key to understanding why the code is shaped
as it is:

```text
Layer                   Size              Controls
──────────────────────────────────────────────────────────────────────
Dask logical block      2000×2000 px      Scheduler task count (RAM)
Staged zarr sub-chunk     500×500 px      Per-task read + decode RAM
Final store sub-chunk     500×500×4       Downstream partial-read cost

                         ┌──────────────────────────┐
  One Dask task ──────►  │  ChunkSpec  2000×2000 px  │
  (one entry in the      │  ┌────┬────┬────┬────┐   │
   scheduler's task      │  │500 │500 │500 │500 │   │
   graph)                │  ├────┼────┼────┼────┤   │
                         │  │    │    │    │    │   │
                         │  ├────┼────┼────┼────┤   │
                         │  │    │    │    │    │   │
                         │  ├────┼────┼────┼────┤   │
                         │  │    │    │    │    │   │
                         │  └────┴────┴────┴────┘   │
                         │  16 on-disk sub-chunks    │
                         └──────────────────────────┘
                           to_icechunk handles the
                           fan-out via align_chunks=True
```

At sub-chunk granularity (millions of tasks), the scheduler's per-task `TaskState` overhead
(~1.5 KB each — see [`ingest/README.md`](../ingest/README.md#background-how-dask-task-graphs-consume-scheduler-ram))
exhausted the scheduler's 8 GB RAM during graph expansion before any worker started. At
ChunkSpec granularity (a few thousand tasks at cornbelt scale), graph planning takes
seconds.

The mechanism that makes this possible is `align_chunks=True` in the `to_icechunk` call.
Without it, on-disk sub-chunk size and Dask block size would be forced to match:

```text
Without align_chunks=True (forced coupling):
  To write 500×500 on-disk chunks, Dask blocks must also be 500×500.
  A 2000×2000 ChunkSpec becomes 16 Dask blocks → 16 TaskStates.

  At cornbelt scale: 5,000 ChunkSpecs × 16 sub-chunks × 2 layers
                   = 160,000 TaskStates — manageable, but grows fast.
  Add more variables, time steps, or larger ROIs and this explodes.

With align_chunks=True:
  Dask block = 2000×2000 ChunkSpec → 1 TaskState per variable per ChunkSpec.
  Worker reads the full 2000×2000 array and writes it in one call.
  zarr/icechunk splits the write into 500×500 on-disk sub-chunks internally.

  Scheduler graph:  1 TaskState per ChunkSpec  (graph stays small)
  On-disk layout:  16 sub-chunks per ChunkSpec (reads stay fast)
  These two numbers are now independent — each can be tuned separately.
```

#### Assembly steps

1. Re-run `filter_chunks_by_roi_mask` to recover the set of live chunk labels (the list
   isn't marshaled through Prefect; the ROI zarr path is the source of truth).
2. Build a lazy Dask mosaic as two unmaterialized `Blockwise` layers, at **ChunkSpec
   granularity** — one dask block per ChunkSpec (full spatial extent × full band axis):
    - a `da.full` template of the right shape filled with the fill value (0 for int8
      embeddings, NaN for floats);
    - a `map_blocks(_assemble_var_block, live_lookup=…)` on top that, per block, consults
      a `(row, col) -> staged_path` dict and either reads the entire staged chunk or
      returns the fill template unchanged.
3. Output zarr sub-chunking (500×500×4) is set via `to_icechunk(..., encoding=...)` and
   read from the staged files' on-disk chunk shape. `align_chunks=True` lets
   `to_icechunk` fan a single dask block out into 512 small on-disk zarr chunks — dask
   graph stays small, final store layout stays suitable for downstream partial reads.
4. Write via `xr.Dataset.to_icechunk()` — float32 embeddings use PCodec compression; int8
   quantized embeddings use default compression. Appends to existing store if present.
5. Delete _all versions_ of the staged chunk zarrs (unless `dev` flag is passed).

**Manifest splitting.** The whole `writer.assemble` call is wrapped in
`manifest_split({"northing": 32, "easting": 32})` (see `tasks/inference.py`). By default
icechunk keeps one manifest object per array, so every commit rewrites the entire chunk
index — O(store size) regardless of how few chunks changed. Splitting tiles the manifest
into 32-chunk-per-axis shards; with 500×500 px on-disk sub-chunks that's ~16k px/shard,
matching `zarr_store.DEFAULT_MANIFEST_SPLIT_SIZES`' target. No `time` split is applied:
assembly only ever writes a single timestep, so one time shard equals the whole array and
splitting time would be a no-op. The split config is persisted via `repo.save_config()` so
it survives the session being shipped to Dask workers.

The split applies to both new and pre-existing stores. A store created before this change
has a single unsplit manifest; the first append under the `manifest_split` block re-shards
the touched array in that commit (a one-time migration — old chunk data is untouched and
reads back unchanged), and every commit after it rewrites only the shards it touches. So
existing stores pick up the same bounded-commit speedup on their next assembly run, with no
manual migration step.

**Why PCodec for the final store:** Embeddings are 128-dim float32 vectors — dense, continuous,
and without the spatial redundancy that makes byte-shuffle + zstd effective for reflectance
data. PCodec is a floating-point-aware codec that models the distribution of float values
directly, achieving significantly better compression ratios on this kind of data than
general-purpose codecs. The tradeoff is that PCodec decompresses the entire on-disk chunk
to read any slice of it (no partial decode), which is why staged chunks use 500×500
sub-chunks to cap decode buffer size.

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

## How Our Performance Optimizations Fit Together

The phases above describe *what the pipeline does*. This section steps back and maps the
performance work layered across them — *what* each optimization targets, *when* it acts,
and *how much* it buys — so the design is legible without reading every phase in depth.

The GPU on each worker is fast enough; a naive pipeline leaves it **idle ~50% of the
time** — not waiting on compute, but on *loading data*, *preparing batches*, and
*writing results*, plus a serial "cold start" at the top of every chunk. Every
optimization here removes one source of that idle time. None changes what the model
computes: all are **bit-identical** — they change scheduling and I/O, not the math —
except the batch-size bump, which shifts int8 values by ≤1–2 levels (inside the ADR-012
cross-config envelope, well within same-code quantization noise).

They come in two families:

- **Always-on core loop** — keep the GPU fed *during* the forward pass. These run on
  every chunk, unconditionally.
- **Per-chunk adaptive** — how a chunk is loaded and pipelined depends on *how much
  valid data it actually holds*. A cloudy coastal sliver and a dense interior tile take
  very different paths. The decision is made per chunk from its Sentinel-2 SCL mask.

### A few concepts first

**Terms.**

- **GEMM** — a general matrix–matrix multiply, the dominant arithmetic inside the
  transformer. GPUs run GEMMs on dedicated **tensor cores**, which are most efficient
  when the matrices are *large*; a bigger **batch** (more pixels multiplied in one go)
  makes each GEMM larger, so the tensor cores idle less.
- **batch / sub-batch** — pixels are inferred in groups, not one at a time. A chunk's
  pixels are split into fixed-size **sub-batches**, each of which is one GPU forward pass.
- **the transfer "bubble" / copy stream / double-buffering** — a batch must be copied
  from CPU to GPU ("host→device") before the GPU can compute it; done naively the copy
  and the compute alternate, so the GPU stalls during every copy — a **bubble**. Staging
  batches in **pinned** (page-locked) host memory and issuing the copy on a separate CUDA
  **copy stream** lets the *next* batch transfer while the current one computes; holding
  two batches in flight at once (**double-buffering**, "two-deep") hides the copy behind
  compute. (**D2H** = device→host, copying results back.)
- **SCL mask** — Sentinel-2's per-pixel Scene Classification Layer; here, the
  cloud/validity mask that records which pixels and which dates hold usable data. Every
  per-chunk decision below starts from it.

**Where the GPU idles — three windows.** A chunk runs through three windows in order;
each optimization reclaims idle from one of them (the summary table tags which):

```
   COLD START (idle) ──▶ FORWARD PASS (busy) ──▶ WRITE (idle) ──▶ next chunk

  1. COLD START — GPU idle: load SCL mask, read the 1st strip, build the dataset.
       reclaimed by:  crop · prune · empty-strip skip · starter strip ·
                      cross-chunk prefetch (next chunk's cold start already done)

  2. FORWARD PASS — GPU busy (the real work): sub-batch → sub-batch, across strips.
       keep it fed, no gaps:  vectorised resampling · async two-deep pipeline ·
                              batch 7168 · intra-chunk strip prefetch

  3. WRITE — GPU idle: staging upload.
       reclaimed by:  background write, overlapping the next chunk's cold start
```

**Two kinds of "sparsity."** The read-reduction optimizations exploit emptiness along
different axes of a chunk's `time × rows × columns` data cube; naming the axis makes
"sparse" unambiguous (both *what* is targeted and *why* the fix works):

- **Temporal sparsity** — whole dates are cloud/nodata → *timestep pruning* drops them
  (shrinks *time*); the resampler never reads an all-empty date.
- **Spatial sparsity** — valid pixels occupy only part of the tile's footprint →
  *easting crop* (shrinks *columns*) + *empty-strip skip* (drops empty *rows*); pixels
  outside the valid footprint produce no embedding, so their bytes are never read.
- Both are distinct from a chunk's **valid-pixel count** — the *volume* of inference
  work left after them — which drives the strip/prefetch plan (Q2–Q4 below), not how
  much gets read. A tile can be spatially compact but temporally deep, or vice versa.

**Impact legend** (qualitative — see the profiling doc for hard numbers):
**● large** · **◐ medium** · **○ small**. "Large" means it removed a dominant chunk of
GPU-idle time in the profiles; "small" means a real but minor trim.

### The always-on core loop (every chunk)

```
Runs on every chunk, regardless of density. The first three keep the GPU busy
MID-FORWARD; the write hides the POST-FORWARD idle (see the three windows above):

  ● vectorised temporal resampling   batch prep 600–650 ms → ~130 ms/sub-batch
                                      (was SLOWER than the GPU forward → prep
                                       gated the GPU; now it doesn't)          [§4c]
  ◐ async two-deep GPU pipeline      pinned double-buffers + copy stream so the
                                      next batch transfers while this one runs
                                      → no per-batch bubble                    [§4d]
  ◐ batch size 7168 (BF16)           bigger GEMMs use the tensor cores more
                                      fully (the one non-bit-identical change) [§4d]
  ○ background staging write         the ~7.5 s S3 upload runs on a writer
                                      thread, overlapping the NEXT chunk's load [§5]
```

### The per-chunk decision tree

Two chunks make the tree concrete before you read it. A **dense interior tile** has
little sparsity of either kind to exploit — it crops nothing and prunes little; its win
comes from the striping + prefetch pipeline (§4a′–4a″), because its high valid-pixel
count keeps the GPU busy enough to hide the strip loads. A **cloudy coastal sliver** is
the opposite: high *temporal* sparsity (prune empty dates) and high *spatial* sparsity
(crop to the bbox, skip empty row bands), so its win comes almost entirely from *reading
less* (§4a) — and since its low valid-pixel count makes it a single serial strip,
reading less is the only lever it has. Same code, opposite paths.

Each chunk takes **one path** through the tree below, chosen from its valid-pixel count
and where the valid data sits. Read top to bottom:

```
A chunk arrives → load its SCL mask → count valid pixels, find their bbox
│
├─ Q1. SPATIAL sparsity: do the valid pixels sit in a narrow easting window?
│      (cropping saves ≥ 10% of the width)
│        ├─ yes → ◐ crop the S2 read to that column bbox — edge/coast   [§4a]
│        │        slivers read a fraction of the bytes
│        └─ no  → read full width (interior tiles stay byte-identical)
│
├─ TEMPORAL sparsity (always): ○ prune S2 timesteps empty everywhere    [§4a]
│           — cloudy dates the resampler would never read
│
├─ Q2. Does bands + full mask fit ONE RAM budget?
│        ├─ yes → single strip — no split, no prefetch (common interior)
│        │
│        └─ no  → SPLIT into northing strips  ● bounds peak host RAM    [§4a′]
│                 │
│                 └─ Q3. Enough valid data for the GPU to hide the strip
│                        loads behind inference?
│                          ├─ yes (dense) →
│                          │     ◐ intra-chunk strip prefetch: strip     [§4a′]
│                          │       i+1 loads while strip i runs the GPU
│                          │     ○ starter strip: small first slice, GPU [§4a″]
│                          │       starts one read sooner
│                          └─ no (wide but few valid px) →
│                                prefetch OFF, strips at the PAIR budget [§4a′]
│                                (one set resident → bigger budget safe;
│                                fewer, larger reads)
│
├─ Q4. On the LAST strip, is this a RAM trough (≤ 1× budget)?
│        ├─ yes, and a next chunk is reserved →
│        │     ● cross-chunk starter prefetch: preload the next chunk's [§4a″]
│        │       mask + 256-row starter NOW, so its GPU work starts
│        │       0–8 s later instead of 24–34 s
│        └─ no (pair budget) → skip it; the next chunk takes the serial
│              prologue (slower, but never over the RAM ceiling)
│
└─ SPATIAL sparsity (per strip): ◐ empty-strip skip — a strip whose    [§4a]
                           mask slice has zero valid pixels skips the S2 band read
```

### Summary

Every output is **bit-identical** with an unoptimized approach except the batch-size change (¹) — the rest alter
scheduling and I/O, not the math. *Window* is which GPU-idle window each reclaims (see
the three-windows diagram above).

| Optimization | Family | Window | Triggers on… | Impact |
|---|---|---|---|---|
| Vectorised temporal resampling (§4c) | core loop | mid-forward | always | ● large |
| Async two-deep GPU pipeline (§4d) | core loop | mid-forward | always (CUDA) | ◐ medium |
| Batch size 3584 → 7168 (§4d) | core loop | mid-forward | always | ◐ medium¹ |
| Background staging write (§5) | core loop | write (post) | always | ○ small |
| Valid-pixel-aware northing striping (§4a′) | adaptive | *enabling* | chunk exceeds one RAM budget | ● large² |
| Intra-chunk strip prefetch (§4a′) | adaptive | mid-forward | dense/hideable split | ◐ medium |
| Starter strip (§4a″) | adaptive | cold-start | dense split with a real body | ○ small |
| Cross-chunk starter prefetch (§4a″) | adaptive | cold-start (next chunk) | last strip is a RAM trough + next chunk reserved | ● large |
| Timestep pruning (§4a) | adaptive | cold-start | **temporal** sparsity (cloudy/empty dates) | ○ small–◐ |
| Empty-strip skip (§4a) | adaptive | cold-start (per strip) | **spatial** sparsity (a row band with no valid px) | ◐ medium |
| Easting bbox crop (§4a) | adaptive | cold-start | **spatial** sparsity (valid px in a narrow column window) | ◐ medium³ |

¹ The only non-bit-identical change. It shifts a small fraction of int8 values by ±1–2
levels (cuBLAS picks different kernels for different batch shapes), so a `main`-vs-branch
diff is judged against the ADR-012 **cross-config** envelope (int8 within ±1 on ≥99.99% of
values, max ≤3; observed max ±2) — not the same-config bit-identity gate the other rows meet.

² Foundational — it bounds peak RAM, which is what makes every other adaptive choice
safe; it also drops a ~13 s fixed read per dense chunk.

³ Large on edge/coast slivers (a 1.5K-valid-pixel chunk dropped from ~39 s of loading to
roughly bbox-proportional), negligible on interior tiles (which skip it).

### What we deliberately did *not* do

Profiling ruled these out, so they're absent by design, not oversight:

- **Greedily prefetching to fill RAM.** We deliberately **leave host RAM on the table.**
  Prefetching the whole next chunk to use the spare RAM co-resides two full working sets
  and spikes peak host RAM to ~92–95% — which OOM-killed a worker. The strip budget plus
  the bounded (~2 GiB) cross-chunk prefetch instead hold peak at ~45–52%, well under the
  60% ceiling, so chunk-density spikes at UTM-zone scale can't OOM the node. The unused
  headroom is intentional insurance, not waste.
- **GRU restructuring** — the model builder already fuses the recurrent stack to cuDNN;
  a hand-restructure was written, measured as no faster, and reverted as dead code.
- **FP16 fast-accumulate** — an L40S GEMM microbench showed BF16 already runs at the
  full dense tensor-core ceiling, so FP16 buys nothing here. BF16 stays.
- **Adaptive token-budget batching** — measured; B=7168 is already throughput-optimal
  across sequence lengths, so a dynamic budget added complexity for no gain.

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
| `batch_size` | 7168 | GPU pixels per forward pass within a bucket |
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
| `assembly.py` | Staged Zarr writes + Dask-based parallel assembly into Icechunk |
| `actors.py` | Ray actor wrapping the full per-chunk pipeline |
| `scheduling.py` | Work-stealing scheduler (`_process_chunks_work_stealing`) and `ActorPool` — manages dispatch, actor replacement, idle retirement, and tracker polling |
| `progress.py` | Lightweight Ray actor (`num_cpus=0`) polled by the flow runner every 60s for batch-level progress and 5-minute stall detection |

---

## Accessing the Dask Dashboard

The Dask assembly cluster runs in a private subnet (ECS Fargate). Use SSM port forwarding to reach the Bokeh dashboard:

```bash
# Look up TASK_ID and RUNTIME_ID for the Dask scheduler task in the ECS console
aws ssm start-session \
  --target ecs:yield-cluster_${TASK_ID}_${RUNTIME_ID} \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters '{"host":["localhost"],"portNumber":["8787"],"localPortNumber":["8787"]}'
```

Then open http://localhost:8787 in your browser.

Requires the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (`brew install session-manager-plugin`).
