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
  enumerate_chunks_from_dataset()     ← 2000×2000 px chunks
            │
            ▼
  filter_chunks_by_roi_mask()         ← drop chunks outside the ROI
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │  Ray Cluster  (EC2, g5.2xlarge × N)         │
  │  ┌─────────────────────────────────────┐    │
  │  │ InferenceActor (1 GPU each)         │    │
  │  │  1. load_chunk()        ← selective  │    │
  │  │  2. Dataset valid-px filter          │    │
  │  │  3. sample_s2/s1_batch()             │    │
  │  │  4. model forward  (FP16, B=3584)    │    │
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
scheduling overhead.

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
- Workers: g5.2xlarge (1× A10G 24 GB VRAM, 32 GB RAM) — on-demand, single AZ
- Workers use a Packer-built AMI with all dependencies pre-installed; boot ready in ~1 minute

### 3. Inference Actors

One `InferenceActor` (a Ray actor pinned to 1 GPU) is created per worker slot. On init each
actor downloads the model checkpoint from S3 to the local NVMe instance store (~1.5 GB/s),
loads `MultimodalBTInferenceModel`, and logs VRAM usage. A `ping()` call confirms readiness
before work is dispatched.

> **Why NVMe?** EBS sequential read saturates at ~42 MB/s. `torch.load` with mmap on EBS
> causes multi-minute hangs. NVMe is ~35× faster.

### 4. Per-Chunk Inference (inside each actor)

Each chunk goes through four sub-steps:

#### 4a. Data Loading (`data_loading.py`)

Three-phase selective loading to keep peak RAM ~2 GB rather than ~15 GB:

1. **Load SCL only** — Read the uint8 Scene Classification Layer for all timesteps (~200 MB)
   and derive a per-pixel binary validity mask. Timesteps where no pixel in the chunk has
   any SCL-valid class are dropped immediately, before pre-sampling or band loading. For
   large ROIs where the per-ROI time axis covers the full year but each chunk only
   intersects a fraction of acquisitions, this typically halves `T` before Phase 2 runs.
2. **Pre-sample timestep indices** — Simulate all `repeat_times` samplings across 10,000 random
   pixels to find the union of timesteps actually needed (~20-40 out of the surviving ~100).
3. **Load only those timesteps** — Fetch reflectance bands only for the needed timestep subset.

All three phases read directly from the raw zarr group via `open_store_as_zarr_group` / `zarr.Array.oindex`,
bypassing xarray and dask entirely. A single icechunk session is opened per store; time-window
filtering and DOY extraction decode the `int64` time coordinate directly from `root["time"]`.
This eliminates the dask task-graph overhead that previously caused Phase 1 (SCL) to take ~3
minutes and Phase 3 (bands) to spike ~3× above output size.

S1 SAR follows the same pattern: VV is read first to identify non-empty timesteps, then VH is
loaded only for the survivors. `data_loading.py` imports neither `xarray` nor `dask`.

Output: `ChunkData` — numpy arrays for S2 bands/masks/DOYs, S1 ascending/descending bands/DOYs,
and per-pixel observation counts (`s2_obs_count`, `s1_asc_obs_count`, `s1_desc_obs_count`) as
uint16 arrays of shape (H, W). S2 obs count is the number of SCL-valid timesteps from the full
(pre-pruning) mask. S1 obs counts are the number of timesteps where either VV or VH is non-zero.

#### 4b. Valid Pixel Filtering (`dataset.py`)

`MosaicChunkInferenceDataset` identifies the subset of pixels eligible for inference:
- At least one non-zero S2 observation
- `≥ min_valid_timesteps` valid S2 frames (from the full, un-pruned SCL mask)
- `≥ max(min_valid_timesteps/2, 1)` valid S1 frames — see below

#### S1 non-zero floor (divergence from upstream)

Cambridge's `tessera_infer` overrides `min_valid_timesteps=0` at inference time
(see `configs/multi_tile_infer_config.py`), and its S1 check is literally
`s1_total_valid >= 0` — always true. Their `.npy` tiles are curated so every
pixel has some S1 observation, so this never bites.

Our Icechunk SAR stores sometimes have chunks where a 2000×2000 spatial extent
is entirely zero-padded for all but a handful of timesteps (e.g. edge of the
ascending-orbit coverage). Pixels in those chunks would pass the original
filter with zero valid S1 observations, then crash the sampler with
`IndexError: index -1 is out of bounds for axis 1 with size 0` when the
latest-timestep force (`indices[:, :, -1] = n_t - 1`) hits an empty time axis.

We diverge from upstream by flooring the S1 threshold at 1 — `s1_total_valid
>= max(s1_threshold, 1)`. This mirrors the implicit `s2_nonzero` floor that
already exists for S2, is a strict superset of upstream's filter (any pixel
accepted by upstream is still accepted), and only excludes pixels whose
embeddings would have been produced from all-zero S1 input anyway — OOD for
the model and not trustworthy.

#### Skip path and skip markers

When every pixel in a live (ROI-intersecting) chunk fails the validity
filter, the chunk takes the `"skipped"` path. This fires in two situations:
(1) the chunk has no usable S1 coverage (the new S1 floor, above) and (2) the
chunk intersects the ROI polygon but its footprint is entirely empty in the
source stores — no non-zero S2 observations anywhere. Case (2) happens for
chunks that clip the ROI along data-coverage edges, where the ROI mask says
"process this" but the underlying imagery is nodata.

In both cases the actor writes a zero-byte `{chunk.label}.skipped` marker to
the staging directory and returns. Assembly fills the chunk's footprint with
constant-zero/NaN tasks in the dask graph — same handling as a chunk rejected
by the ROI pre-filter. The marker distinguishes a legitimate skip from a
silently-failed chunk (Ray actor crash that wasn't re-queued, an exception
path that didn't write the marker, etc.). `verify_staged_completeness`
requires every live chunk to have either a staged zarr or a skip marker and
raises otherwise; it runs after inference in the normal path and at the start
of `assembly_only` runs.

Critically, **pixel extraction is deferred to batch time** (`get_batch(start, end)` uses fancy
indexing). Previously, pre-extracting all valid pixels doubled peak RAM (source arrays 14 GB
+ extracted copy 17 GB = OOM). Per-batch fancy indexing adds ~7 ms per batch and avoids the
spike entirely.

#### 4c. Temporal Sampling (`sampling.py`)

For each pixel and each of `repeat_times` repeats, `sample_size` timesteps are drawn from a
validity-weighted distribution. The implementation is vectorized via cumsum + searchsorted
(~50–100× faster than a per-pixel Python loop).

Repeats are **folded into the batch dimension**: for batch size B and repeat_times R, the
GPU receives a (B×R, sample_size, bands+1) tensor and processes all repeats in one forward pass.
Embeddings are reshaped to (B, R, 128) and averaged on-GPU before CPU transfer.

#### 4d. GPU Forward Pass (`inference.py`)

- **FP16 (`model.half()`)** — halves memory bandwidth and activation memory, enabling
  batch_size=3584; 3-repeat averaging absorbs FP16 noise.
- **`torch.compile` is disabled** — on g5.2xlarge (15.4 GB VRAM), CUDA graph capture consumed
  11.6 GB and made forward passes slower due to GRU recompilation (3,770 ms vs. 1,944 ms).
- **cuDNN benchmark mode** — lets cuDNN select the fastest algorithm for GRU/conv ops.
- **Throughput:** ~1,300 px/sec per A10G; 100 actors ≈ 130,000 px/sec.

Output per chunk: `embeddings` array (H, W, 128) int8 with zeros for invalid pixels,
plus a per-pixel float32 `scale` factor for dequantization.

### 4e. Quantization (`quantization.py`)

Embeddings are always compressed from float32 to int8 immediately after the forward pass,
before staging. This reduces staged and final store sizes by ~4×.

**How it works:** For each pixel, the absolute maximum across all 128 embedding channels
is computed. Embeddings are scaled so that the max maps to ±127 (the int8 range), then
rounded and clipped. The per-pixel scale factor (float32) is stored alongside the
quantized embeddings so the original values can be reconstructed:

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

After all live chunks complete, a Dask cluster (20-500 Fargate workers × 4 GB RAM) reads staged
chunks and assembles them into the final Icechunk store:

1. Re-run `filter_chunks_by_roi_mask` to recover the set of live chunk labels (the list isn't
   marshaled through Prefect; the ROI zarr path is the source of truth).
2. Build a lazy Dask mosaic as two unmaterialized `Blockwise` layers, at **ChunkSpec
   granularity** — one dask block per ChunkSpec (full spatial extent × full band axis):
    - a `da.full` template of the right shape filled with the fill value (0 for int8
      embeddings, NaN for floats);
    - a `map_blocks(_assemble_var_block, live_lookup=…)` on top that, per block, consults
      a `(row, col) -> staged_path` dict and either reads the entire staged chunk or
      returns the fill template unchanged.

   Dask block size is deliberately **decoupled from on-disk sub-chunk size**: task count
   = `n_chunks` (a few thousand at cornbelt scale), not `n_chunks * sub_chunks_per_chunk`
   (millions). The coarser granularity is what keeps the Dask scheduler alive — at the
   sub-chunk granularity, the scheduler's per-task `TaskState` overhead (~1.5 KB) blew
   through 8 GB of RAM during graph expansion before any worker started.
3. Output zarr sub-chunking (500×500×4) is set via `to_icechunk(..., encoding=...)` and
   read from the staged files' on-disk chunk shape. `align_chunks=True` lets
   `to_icechunk` fan a single dask block out into 512 small on-disk zarr chunks — dask
   graph stays small, final store layout stays suitable for downstream partial reads.
4. Write via `xr.Dataset.to_icechunk()` — float32 embeddings use PCodec compression; int8
   quantized embeddings use default compression. Appends to existing store if present.
5. Delete _all versions_ of the staged chunk zarrs (unless `dev` flag is passed).

**Why PCodec for the final store:** Embeddings are 128-dim float32 vectors — dense, continuous,
and without the spatial redundancy that makes byte-shuffle + zstd effective for reflectance
data. PCodec is a floating-point-aware codec that models the distribution of float values
directly, achieving significantly better compression ratios on this kind of data than
general-purpose codecs. The tradeoff is that PCodec decompresses the entire on-disk chunk
to read any slice of it (no partial decode), which is why staged chunks use 500×500
sub-chunks to cap decode buffer size.

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
| `batch_size` | 3584 | GPU pixels per forward pass |
| `repeat_times` | 3 | Independent samplings per pixel; averaged for final embedding |
| `sample_size_s2` | 40 | S2 timesteps per repeat |
| `sample_size_s1` | 40 | S1 timesteps per repeat |
| `min_valid_timesteps` | 0 | Include all pixels with any valid S2 observation |
| `s1_orbit` | `"ascending"` | `"ascending"`, `"descending"` |
| `compute_std` | `False` | Also write per-pixel embedding std across repeats |
| `latent_dim` | 128 | Embedding dimension (must match checkpoint) |
| `gpu_instance_type` | `g5.2xlarge` | Worker EC2 type |
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
| `sampling.py` | `tessera_infer/src/multi_tile_infer.py` | Type hints, ruff formatting. Sampling logic unchanged. |
| `inference.py` | `tessera_infer` process_tile logic | Restructured into `run_inference()`. Same repeated random sampling + averaging. |
| `models/` | `tessera_infer/src/models/` | See `models/README.md`. |

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
