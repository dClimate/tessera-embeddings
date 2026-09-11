# Tessera Inference Pipeline

Distributed GPU inference that turns mosaicked Sentinel-2 reflectance and Sentinel-1 radar
into **128-dimensional embeddings, one per 10 m pixel**.

Two entry points run the same domain code:

- [`orchestration/prefect/flows/tessera_embeddings.py`](../orchestration/prefect/flows/tessera_embeddings.py)
  is the production path. It starts a Ray cluster on EC2 GPU instances, runs inference in
  parallel across the area, and assembles the result into an Icechunk/Zarr store.
- [`orchestration/runners/plain.py`](../orchestration/runners/plain.py) is the
  orchestrator-free equivalent, calling the same functions on `ray_cluster(num_gpus=0)`
  for laptop and CI runs.

This file is the reference for what the code does. What was measured, and what was tried
and abandoned, is in
[`context_docs/inference/inference-on-gpus.md`](../../../context_docs/inference/inference-on-gpus.md).

---

## What all of this is for

**Every design choice below exists to keep the GPU busy — fully busy, as much and as often
as possible.**

That is worth saying once, plainly, because it explains the rest of the document. A
GPU-equipped machine costs many times what a CPU one does, and the model itself is quick.
On a naive pipeline the card sits idle roughly **half the time** — not waiting on
arithmetic, but waiting for imagery to arrive from S3, for the next batch of pixels to be
prepared, and for finished results to be written back. Nothing here makes the model
faster. Everything here removes a reason for the card to wait.

Two consequences run through the whole design. Work is shaped so that reading, preparing
and writing happen *while* the GPU computes rather than in front of it. And the pipeline
deliberately leaves host memory unused, because the cost of running out of it — a killed
worker, and its tile redone — is far higher than the cost of a read that was not
overlapped.

[**How Our Performance Optimizations Fit Together**](#how-our-performance-optimizations-fit-together)
maps every source of that idle time onto the thing that closes it, with relative impact.
If you read one section, read that one; the phase-by-phase description before it is the
reference for what each stage actually does.

## Words used here

The pipeline has its own vocabulary. These are the terms the rest of this file uses
without further explanation.

| term | what it means |
|---|---|
| **mosaic** | The input. Per-date Sentinel-2 reflectance and Sentinel-1 radar, already fetched and written to Zarr stores by the ingest stage. Inference never queries a satellite catalogue. |
| **tile** (`chunk` in the code) | The unit of work: a 2048 × 2048-pixel square of ground, one year deep. One tile becomes exactly one object in the output store. |
| **actor** | A Ray worker process pinned to one GPU. It holds the model in VRAM and processes whole tiles, one after another. |
| **strip** | A horizontal slice of a tile — its full easting (east–west) width, and a range of its northing (north–south) rows. A tile too large to hold in memory is loaded one strip at a time. |
| **SCL** | Sentinel-2's Scene Classification Layer: the per-pixel mask saying which pixels on which dates are usable, and which are cloud, shadow, snow or no data. Most decisions in this pipeline start from it. |
| **optical depth** | How many usable optical observations a pixel has in its year. It varies by an order of magnitude with geography, and it drives both cost and quality. |
| **bucket** | A group of pixels sharing the same `(s2_bin, s1_bin)` target pair. Grouping them lets the model run over one rectangular tensor with no padding and no masking, which is much faster than variable-length input. The optical and radar counts are binned *independently*, each UP to the smallest checkpoint that is greater than or equal to it (`num_obs_checkpoints`) — so pixels in one bucket share a target shape, not an observation count. |
| **prefetch** | Starting a read before the thing that needs it asks for it. While the GPU works on the data it has, a background thread fetches what it will want next — so the read happens *during* compute instead of in front of it. |
| **staging** | Each finished tile is written to a scratch prefix on S3 first, and copied into the real store afterwards. Actors never write to the output store directly. |
| **shard** | One object in the output store: a 2048-pixel square holding an 8 × 8 grid of independently compressed 256-pixel **inner chunks**, plus an index of where each one sits inside it. A reader fetches the index, then only the inner chunks it needs. |

## Contents

- [Architecture at a glance](#architecture-at-a-glance)
- [How a tile becomes embeddings](#how-a-tile-becomes-embeddings)
- [How Our Performance Optimizations Fit Together](#how-our-performance-optimizations-fit-together)
- [Fault tolerance](#fault-tolerance)
- [Key configuration (`config.py`)](#key-configuration-configpy)
- [Model architecture constraint](#model-architecture-constraint)
- [Provenance](#provenance)
- [Accessing the Ray dashboard](#accessing-the-ray-dashboard)

---

## Architecture at a glance

```
Input mosaic stores (Icechunk/Zarr on S3):
  reflectance.zarr / sar_ascending.zarr / sar_descending.zarr
            │
            ▼
  enumerate_chunks_from_dataset()     ← 2048 px tiles (== one output shard)
            │
            ▼
  filter_chunks_by_roi_mask()         ← drop tiles outside the area of interest
            │
            ▼
  ┌─────────────────────────────────────────────┐
  │  Ray cluster  (EC2, g6e.xlarge × N)         │
  │  ┌─────────────────────────────────────┐    │
  │  │ InferenceActor (1 GPU each)         │    │
  │  │  per strip, to bound memory:         │    │
  │  │   1. load_chunk(y_sub=…)             │    │
  │  │   2. choose pixels, group in buckets │    │
  │  │   3. build each pixel's sequence     │    │
  │  │   4. model forward  (BF16, B=7168)   │    │
  │  │  5. writer.write_chunk() → staging   │    │
  │  └─────────────────────────────────────┘    │
  │  Work-stealing: actors pull from a queue    │
  └─────────────────────────────────────────────┘
            │
            ▼
  Staged tiles (S3 staging prefix, per run_id) — live tiles only
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
           embeddings    (time, northing, easting, 128)
           obs counts    (time, northing, easting)  — s2 / s1_asc / s1_desc

  Global-campaign variant: writer.assemble_global() writes the same staged
  tiles as whole 2048-px shards into a pre-seeded UTM zone group of the
  global store (1 tile == 1 shard), driven by the zone-fill runner
  (orchestration/runners/zone_fill.py). See ADR-008.
```

---

## How a tile becomes embeddings

### 1. Deciding which tiles to run

The input mosaic is divided into a grid of square `ChunkSpec` tiles, **2048 px on both
paths** (`INFERENCE_CHUNK_SIZE`, equal to `SHARD_PX`); edge tiles may be smaller. That
size balances peak memory during inference against scheduling overhead, and makes one tile
exactly one output shard (ADR-008 D3), so assembly writes whole objects instead of
read-modify-writing a partial one at every tile boundary. The whole chain of sizes divides
evenly, so no stage rechunks another stage's output —
[`docs/single-vs-global.md`](../../../docs/single-vs-global.md#why-chunk-size-dominates-everything)
has the grid and why it matters.

`filter_chunks_by_roi_mask` then drops every tile whose footprint does not intersect the
area of interest. Only the survivors — the **live tiles** — are dispatched, and the cluster
is sized from that count.

This matters more than it sounds. An area that fills little of its own bounding box — a
polygon inscribed in a large rectangle — can leave most tiles empty, and an actor takes
tens of seconds just to open the store, read the SCL mask and discover that a tile has
nothing in it. On a GPU-priced machine that is pure waste, so the emptiness is established
once, cheaply, before any machine is asked for.

A tile that was never dispatched has nothing staged on S3. Assembly re-runs the same
filter and simply never writes that footprint, so it reads back as the store's fill value
— `0` for the `int8` embeddings and `uint16` counts, `NaN` for the `float32` scales. There
are no placeholder files and no placeholder chunks.

### 2. Starting the GPU cluster

`_start_ray_cluster()` resolves the cluster YAML at runtime from SSM parameters (security
group, subnets, instance profile, AMI, SSH key), writes the resolved file to a tempfile and
runs `ray up`. The flow connects over Ray Client (`ray://head-ip:10001`), and the cluster
lives inside a context manager so it is always torn down when assembly finishes.

- **Head:** m5.2xlarge — Ray's own bookkeeping and the autoscaler, no inference work.
- **Workers:** g6e.xlarge (one L40S, 4 vCPU, 32 GB RAM), on demand, across several
  availability zones; Ray fails over between subnets when one refuses capacity.
- Workers boot from a Packer-built AMI with every dependency already installed, and are
  ready in about a minute.

[`docs/providers/aws.md`](../../../docs/providers/aws.md) covers what to provision, the
capacity-fallback rungs, and how the fleet mix is chosen.

### 3. The inference actors

One `InferenceActor` — a Ray actor holding one GPU — is created per worker slot. On start
it copies the model checkpoint from S3 to the instance's local NVMe disk, loads
`MultimodalBTInferenceModel`, and logs its VRAM use. A `ping()` call confirms it is ready
before any work is sent to it.

> **Why NVMe and not EBS?** Sequential reads from EBS saturate around 42 MB/s, and
> `torch.load` with memory mapping on EBS causes multi-minute hangs. The instance store is
> roughly 35× faster.

**Credentials are injected, never imported.** `InferenceActor` takes a `get_credentials`
callback and wraps each tile in `storage.zarr_store.credentials_provider(...)`, so every
store it opens resolves through that callback. The AWS-aware orchestration layer supplies
`providers.aws.credentials.iam_icechunk_credentials`; the actor imports no AWS module at
all, which is what keeps `inference/` cloud-agnostic (enforced by `tests/architecture`).
With no callback — a local or development run — icechunk falls back to its own default
credential chain.

This is not ceremony. An actor lives for hours, and the default Rust credential chain
resolves the instance-profile credential once and may fail to refresh it. When it lapses,
the tile in flight **and every tile that actor takes afterwards** fail with `no providers
in chain provided credentials`. A botocore-backed, self-refreshing callback fixes that. One
caveat carries over from `ingest/README.md`: `_resolve_iam_credentials` must stay
`lru_cache`d, or the callback's re-invocation every fifteen minutes triggers a cold
instance-metadata lookup each time and runs into throttling.

### 4. Loading a tile's imagery

Everything in this stage reads the mosaic stores **directly as zarr arrays**
(`open_store_as_zarr_group`, `zarr.Array.oindex`), bypassing xarray and Dask entirely —
`data_loading.py` imports neither. One icechunk session is opened per store; the time
coordinate is decoded straight from `root["time"]` to work out which dates fall in the
window and what day of year each one is.

The output is a `ChunkData`: numpy arrays of Sentinel-2 bands, masks and days-of-year,
the same for both Sentinel-1 orbits, and per-pixel observation counts
(`s2_obs_count`, `s1_asc_obs_count`, `s1_desc_obs_count`) as `uint16` arrays.

**Which radar orbits are used.** Sentinel-1 flies both ascending and descending passes,
and `s1_orbit="both"` reads them both. `resolve_s1_orbit` probes for the stores that
actually exist and falls back gracefully to whichever single orbit was ingested, or to
`"none"` when neither is there. `allow_none` defaults to true because **parts of the globe
are radar-free as a matter of geography**, not as a matter of a failed ingest; pass the
flows' `require_s1` to demand radar instead, on a run over terrain you know is imaged. A
resolved `"none"` forces `allow_s2_only` on, because otherwise every pixel would fail the
default pixel filter and the run would complete having written nothing. The probe is
threaded the same credential callback and region as the rest of the run, so it does not
quietly fall back to the default credential chain.

#### 4.1 Read as little as possible

A tile's data is a cube of `time × northing × easting`, and most tiles are empty along at
least one axis. Four reductions exploit that, all unconditional and all bit-identical in
their output — they change which bytes are fetched, not what is computed.

**Drop dates that hold nothing (the *time* axis).** The loader reads the SCL mask first —
one byte per pixel, a few hundred megabytes — and discards every date with no usable pixel
anywhere in this tile. Only then does it read the reflectance bands, which are twenty
bytes per pixel per date. Most tiles intersect only a fraction of the area's date range,
so this typically halves the dates and cuts peak memory by the best part of an order of
magnitude:

```text
All dates in the area's time axis
T ≈ 126 dates  (a full year of Sentinel-2 acquisitions over this area)
      │
      │  Phase 1: read SCL only (~200 MB).
      │  Drop dates with no SCL-valid pixel in this tile.
      ▼
T_kept ≈ 63 dates
      │
      │  Phase 2: read the full reflectance bands, for those dates only.
      │  v1.1 uses every valid observation; nothing is pre-sampled.
      ▼
Peak memory: ~2 GB, against ~15 GB reading the bands without the SCL pass first
```

**Crop to the columns that matter (the *easting* axis).** When a tile's valid pixels sit
in a narrow east–west window — a coastline, the edge of an area — the Sentinel-2 read is
restricted to that column bounding box (`x_sub`), applied when it saves at least 10% of
the width. Radar is still read full width, and the optical counts come from the mask, so
the saved provenance layers keep their full extent.

**Skip row bands that are empty (the *northing* axis).** A strip whose slice of the SCL
mask has no valid pixel skips the reflectance read altogether.

**Probe radar with one polarisation.** VV is read first to find the non-empty dates; VH is
read only for the survivors.

Together these turn a thin sliver of a tile from a fixed cost into a roughly
proportional one: a tile with 1,500 valid pixels and 82 kept dates went from about 39
seconds of loading to something close to the cost of its own bounding box.

#### 4.2 Loading a tile in strips

Dropping empty dates caps memory for a typical tile but does not *bound* it. The resident
Sentinel-2 band array is `T_kept × H × W × 10 bands × 2 bytes`, which still scales with
the date count — and on a dense area `T_kept` can reach 120, which makes a single tile's
bands alone about **10 GB in one allocation**. That has to share a 32 GB machine with the
radar stack, the output buffers and the model.

The date count is not something the pipeline can choose (v1.1 uses every valid
observation), so the only lever is how much ground is resident at once. `process_chunk`
therefore loads a tile as a sequence of **strips**: full easting width, a slice of the
northing rows. Each strip is a self-contained `ChunkData` that is grouped into buckets, run
through the model, and written into the whole-tile output buffers by row range.

```text
 one tile: 2048 px of easting (east–west) × 2048 px of northing (north–south).
 A strip is the FULL easting width and a slice of the northing rows.

         easting  0 ─────────────────────────────────────────────► 2047
  northing   0  ┌──────────────────────────────────────────────────┐
                │ strip 0   rows    0– 511  ░░░░░░░░░░░░░░░░░░░░░░ │ → bucket → GPU
         512    ├──────────────────────────────────────────────────┤
                │ strip 1   rows  512–1023  ░░░░░░░░░░░░░░░░░░░░░░ │ → bucket → GPU
        1024    ├──────────────────────────────────────────────────┤
                │ strip 2   rows 1024–1535  ░░░░░░░░░░░░░░░░░░░░░░ │ → bucket → GPU
        1536    ├──────────────────────────────────────────────────┤
                │ strip 3   rows 1536–2047  ░░░░░░░░░░░░░░░░░░░░░░ │ → bucket → GPU
        2047    └──────────────────────────────────────────────────┘

 Only the INPUT is sliced. The int8 output buffer for the whole tile
 (2048 × 2048 × 128, about half a gigabyte) is held whole and filled in by row
 range, so exactly one staging write leaves the actor — the write path, the
 observation-count maps and assembly.py never learn that striping happened.

 On a dense tile the reads are hidden behind the compute:

   GPU   │ strip 0 ██████████│ strip 1 ██████████│ strip 2 ██████████│
   loads │  strip 1 ░░░░░░   │  strip 2 ░░░░░░   │  strip 3 ░░░░░░   │
         └ each strip's read happens DURING the previous strip's forward passes
```

**How the strips are chosen.** `_strip_plan` decides per tile, from the full-tile SCL mask
it already loaded. Two quantities pull in different directions: the bytes to read scale with
`T_kept × H × W` regardless of how many pixels are valid, while the GPU time scales with the
**valid-pixel count**. A tile can be cheap to read and slow to infer, or the reverse, so the
plan picks the strategy that fits — the same choice appears as Q2 and Q3 of the
[decision tree](#the-per-chunk-decision-tree):

```text
 per_set = T_kept·H·W·(20 bands + 1 mask)      budget = _S2_STRIP_BYTE_BUDGET (5.75 GiB)

 per_set ≤ budget ─────────────────────────────▶ ONE strip, no prefetch   (most tiles)
 else, and enough valid pixels to hide the ───▶ equal strips ≤ budget, prefetch ON
 reads behind inference (a dense tile)
        └─ tall enough to be worth it? ───────▶   + a small "starter" strip, so the
                                                   GPU begins one read sooner
 else (many bytes, few valid pixels, so ──────▶ strips ≤ the PAIR budget, prefetch OFF
 nothing to hide the reads behind)                (only one strip resident, so it may
                                                   safely use twice the budget)
```

**Peak host memory has the same ceiling in every branch.** With prefetching on, two band
sets co-reside — the strip being inferred and the strip being read — each within the budget,
so the pair is bounded at roughly twice it. With prefetching off the previous set is
released before the next is read, so only one is resident and it may use the whole pair
budget. Either way the ceiling is the same, and it holds peak memory under 60% of a 32 GB
worker across the density variation of a whole UTM zone. The arithmetic behind the budget
constant is in its own comment.

**Turning prefetching off is deliberate, not a fallback.** A background read only helps if
there is inference to hide it behind. On a tile with many bytes and few valid pixels the
read would sit exposed on the critical path *and* force a second resident set for nothing.

**Two reads are decoupled.** The SCL mask — one byte per pixel — is read **once** for the
whole tile (`load_s2_mask_bundle`) and sliced per strip, never re-decompressed; it doubles
as the source of `T_kept` for sizing. The reflectance bands — twenty bytes per pixel — are
read per strip on the tiles that split, and that per-strip read is exactly the working set
the byte budget bounds. A tile that fits one budget is loaded whole, byte for byte the same
as if striping did not exist.

#### 4.3 Starting the next tile early, and finishing the last one late

Every tile pays a serial, GPU-idle **prologue** before its first forward pass: read the SCL
mask, read the first band set, build the dataset. That is 24–36 seconds per tile in which
an expensive card does nothing.

The fix is to do it during the *previous* tile's work. The scheduler reserves each actor's
next tile one ahead (`ActorPool.reserved`, passed to the actor as `prefetch_hint`), and
during the current tile's **last strip** — by which point the co-residency has decayed to
its low point — the actor preloads a **capped** payload for the tile it is about to get:
its SCL mask and read plan, and where the budget allows, its first 256 rows.

```text
 actor timeline, tile N → N+1 (prefetch hit)

 [═════════ inference N ═══════════][═ N+1 starter ═][═══ N+1 body ═══]
              [mask N+1][starter N+1]      [body N+1 loads]
  GPU:  busy ─────────────────────── busy ── busy ─────── busy
        └ prefetch runs during N's LAST strip (the memory trough) ┘
```

**It deliberately does not prefetch the whole next tile.** That co-resides two full working
sets, and at UTM-zone density variation it exhausts the machine's memory — measured at
92–95% peak, which killed a worker. The cap is about 2 GiB
(`_XCHUNK_PREFETCH_CAP_BYTES`), and the tiers that decide how much of it to use
(`_xchunk_rung`) are conservative: a dense, already-striped tile gets its small starter
free; a single-budget tile is converted into starter-plus-body only when a net-gain check
says the extra fixed read will actually be hidden; everything else gets the mask alone.

**Every way this can miss degrades to the serial prologue** — slower, never bigger. Hitting
the cap, another actor stealing the reserved tile, a credential window expiring, a failed
read, or the `TESSERA_DISABLE_XCHUNK_PREFETCH=1` escape hatch all simply mean the next tile
loads the way it would have anyway.

**Two smaller overlaps ride on the same idea.** Bucketing rides the strip-prefetch thread:
`_load_strip` returns the built dataset alongside the data, so choosing pixels and grouping
them (about ten seconds) happens during the previous strip's GPU work rather than between
load and inference. And background strip loads **reserve two cores** for the batch-prep
workers feeding the GPU (`reserve_cpus`), so decompressing bands cannot starve inference on
a four-vCPU machine.

**The staging write is deferred too.** It goes to a single-slot writer thread and overlaps
the next tile's prologue — both are I/O, and the GPU is idle for either. The tile's result
comes back marked `write_deferred=True`, and the scheduler holds it out of the completed set
until the write outcome arrives, either piggybacked on that actor's next result or drained
by `flush_writes()` when the actor goes idle. A failed write requeues the tile without
killing the actor; an actor dying with a write in flight requeues too, which is safe because
staged writes are idempotent. One consequence for anyone reading the logs: a tile's "done"
can trail its inference by up to one tile.

**Waits on background I/O are bounded.** Every in-actor wait on a background future — the
previous tile's deferred write, a strip load, a cross-tile prefetch — times out after
`_BACKGROUND_IO_TIMEOUT_S` (600 s, matching the scheduler's own RPC timeout). Without it a
wedged S3 client with no socket timeout would hang `process_chunk` itself, where the
scheduler's recovery cannot reach it. A timeout always fails the tile, so the scheduler
replaces the actor — reaping the wedged worker — and requeues the work. That matters
because the writer and prefetch pools are single, *persistent* threads: one stuck task
would poison every later write and prefetch on that actor. A prefetch that merely errors,
leaving its worker free, degrades to the serial prologue instead.

### 5. Choosing which pixels to run, and grouping them (`dataset.py`)

`MosaicChunkInferenceDataset` decides which pixels are eligible and sorts them into
buckets.

1. **Which pixels are eligible.** A pixel needs at least one non-zero optical observation
   and — by default — at least one non-zero radar observation. The radar requirement is
   optional: with `InferenceConfig.allow_s2_only=True` (the flow parameter
   `allow_s2_only`, off by default), optical-valid pixels inside radar coverage gaps are
   embedded too. They are fed the upstream v1.1 missing-radar convention: an all-zeros
   radar slice *in normalised space*, in the smallest bucket, bit-identical to what
   `ucam-eo/tessera`'s `_sample_s1_merged` returns for the same case. Nothing in the
   encoder requires a radar observation to exist. Radar-informed pixels are unaffected by
   the flag. Downstream, an optical-only pixel is exactly one with a finite `scales` value
   and `s1_asc_obs_count + s1_desc_obs_count == 0`. **Their quality has not been
   validated against radar-informed embeddings** — read
   [ADR-013](../../../context_docs/decisions/013-optional-s1-s2-only-pixels.md) before
   enabling this in production.

2. **Which bucket each pixel goes in.** `compute_bin_keys` maps a pixel's
   `(optical count, radar count)` to the nearest entry in `num_obs_checkpoints`. Pixels
   sharing a `(s2_bin, s1_bin)` key form a bucket, and the model receives one rectangular
   `(B, seq_len, bands+1)` tensor per bucket — no padding, no attention masking, no
   variable-length overhead.

3. **The pixel data is fetched per batch, not up front.** Only the coordinates of each
   bucket's pixels are stored when the dataset is built; the band values are gathered by
   fancy indexing when a batch is actually needed. Pre-extracting every valid pixel used to
   double peak memory — 14 GB of source arrays plus a 17 GB copy, which was an
   out-of-memory kill. Per-batch indexing costs about 7 ms per batch and removes the spike
   entirely.

**When no pixel qualifies.** A tile inside the area of interest whose every pixel fails the
filter takes the **skip path**: the actor writes a zero-byte `{chunk.label}.skipped` marker
and returns, and assembly fills that footprint with constant fill values. The marker is
what distinguishes a legitimate skip from a tile that failed silently —
`verify_staged_completeness` requires every live tile to have either a completed staged
store (§9) or a skip marker. A tile that staged on one attempt and skipped on a retry must
not appear to have done both, so `write_skip_marker` deletes the sibling store and its
completion marker first.

### 6. Building each pixel's observation sequence (`sampling.py`)

Each bucket needs a fixed-length sequence per pixel. For a bucket `(s2_bin, s1_bin)`:

- **`resample_s2_bucket`** selects `s2_bin` dates from each pixel's valid optical
  observations — deterministically, with no random repetition — and returns
  `(B, s2_bin, 12)`: the normalised bands plus a day-of-year feature.
- **`resample_s1_bucket`** loads the ascending and descending observations and returns
  `(B, s1_bin, 3)`: the normalised VV/VH pair plus a day-of-year feature.

`build_resample_indices` handles pixels that have too few observations for their bucket's
target by repeating the last valid date until the target is reached, deterministically;
pixels with too many are sub-sampled uniformly.

**Each radar orbit is normalised on its own statistics.** Ascending and descending
observations are standardised with their own mean and standard deviation —
`S1_ASC_BAND_MEAN/STD` and `S1_DESC_BAND_MEAN/STD` in `config/inference.py` — *before*
they are concatenated, so the model sees per-orbit statistics rather than blended ones.

> **Why `s1_orbit="both"` is safe for v1.1.** The v1.1 model uses a single merged radar
> backbone (`split_s1_modalities=False`), which might suggest the two orbits have to share
> normalisation. They do not: separate per-orbit normalisation before concatenation is what
> v1.1's training-time preprocessing did, and it is what this code does. Mixing both
> orbits is therefore correct, and preferred, because it gives every pixel more
> observations. This is a change from v1, where the two orbits *did* share normalisation
> statistics and mixing them was not safe.

### 7. The forward pass on the GPU (`inference.py`)

This is the part every other section exists to serve. The arrangement below is what keeps
the card from ever waiting.

**One bucket at a time, biggest sequence first.** `iter_buckets(largest_first=True)` sorts by
`s2_target × s1_target` descending, so the first bucket has the largest sequence shape — which
may hold very few pixels. The GPU memory that bucket needs is allocated once, and
every smaller bucket afterwards reuses it, so the run does not grow its memory footprint
after the first bucket.

**Pixels go through in sub-batches of 7,168.** A bucket can hold millions of pixels, so
they are run in fixed-size groups. The size is not arbitrary. The model's arithmetic is
dominated by large matrix multiplications, and GPUs run those on dedicated hardware
(tensor cores) that is only efficient when the matrices are big — a larger batch makes each
multiplication bigger, so less of that hardware sits unused. 7,168 was measured as the
point where the matrices are large enough to saturate it without running out of VRAM;
smaller and larger were both tried.

**Two batches are in flight at once.** A batch has to be copied from host memory into the
GPU before the GPU can compute it, and a card cannot compute during a copy it is waiting
on. So batches are staged in **pinned** host memory — memory the operating system promises
not to move, which is what lets the copy proceed without the CPU shepherding it — and the
copies are issued non-blocking, so the host thread does not wait on them. **There is no
separate copy stream**: every operation goes on the current stream, in the serial loop's order.
What the pipeline buys is host-side queueing — while the GPU works through batch *i*, the host
has already enqueued batch *i+1*'s copy and forward and is draining batch *i−1*'s results, so
the card never waits for the host to catch up. It does **not** overlap a transfer with a
forward pass; same-stream work still runs in order. That distinction matters when you are
deciding whether transfer bandwidth or host-side queueing is the constraint.

```
 serial loop:     [H2D][═ fwd i ═][D2H][scatter][H2D][═ fwd i+1 ═][D2H][scatter]
                                   GPU idle ↑↑↑ between every sub-batch

 pipelined loop:  [H2D][═ fwd i ═][D2H][H2D][═ fwd i+1 ═][D2H][═ fwd i+2 ═] …
    host thread:        …enqueue i+1… …drain i… …enqueue i+2… …drain i+1…

  H2D = host to device, the copy in.   D2H = device to host, the results back.
```

The operations are issued in the same order, on the same stream, as the plain serial loop,
so the arithmetic is unchanged and the outputs are **identical bit for bit**. Set
`TESSERA_SERIAL_GPU_LOOP=1` to use the serial version; the pipelined one is CUDA-only.

**Batch preparation had to be made faster than the GPU.** The resamplers in §6 are
vectorised: a pixel's resample indices depend only on `(valid_count, target)`, so they are
memoised, and a sub-batch costs a handful of `np.unique` lookups plus large gathers rather
than one Python iteration per pixel. Before that, preparing a batch took *longer* than the
forward pass it fed, which meant preparation — not the GPU — set the pipeline's speed. Two
prep workers keep `PREFETCH_DEPTH = 2` batches ready. The vectorised path is bit-identical
to the loop it replaced, and golden-reference tests enforce that.

**The output values are checked on the host, not on the card.** Each pixel's scale factor
is validated for finiteness after it has been copied back
(`raise_on_nonfinite_scales`). Checking it on the GPU instead forces the card to finish and
report before the host can read the answer, which stalls the pipeline once per sub-batch.

**The arithmetic runs in BF16** (`_prepare_gpu`) — a 16-bit float with the same exponent
range as a 32-bit one, so the model's values cannot overflow, at half the bytes. On cards
older than Ampere, which have no BF16, FP16 is a best-effort fallback; that format *does*
overflow, above 65,504.

**Four things are deliberately off or replaced**, each because it was measured and made
things worse:

- **`torch.compile` is disabled.** Capturing the model as a CUDA graph consumed 11.6 GB of
  VRAM and roughly doubled the forward pass, because the recurrent layer recompiled for
  every distinct sequence length it saw.
- **cuDNN's autotuner (`benchmark` mode) is disabled.** It searches for the fastest kernel
  per input shape, and bucketing means the shapes change constantly, so it searches
  constantly — and inflates host memory doing it.
- **The reference GRU is swapped for a fused one.** A GRU is the recurrent layer in the
  model's pooling head. `builder._fuse_custom_gru` replaces the checkpoint-faithful
  `CustomGRU` with PyTorch's fused `nn.GRU` before inference, turning roughly 480 GPU
  kernel launches into one, so the recurrence is no longer bound by launch overhead. This
  is a small, deliberate approximation in the reset gate — see the builder's docstring.
- **Positional encoding writes into an uninitialised buffer.** The sine and cosine values
  are written straight into it, instead of being scattered into a multi-gigabyte block of
  FP32 zeros allocated on every forward pass. Same values, lower peak memory.

**What the logs report, and why it is not pixels per second.** A pixel with few
observations costs about a tenth of a densely observed one, so a pixels-per-second figure
says as much about the geography of the tile as about the machine. The periodic and
end-of-tile summaries therefore report **tokens per second** — pixels multiplied by the
number of observations each carried — and effective TFLOPS, computed from the
transformer's own layer count by `profiling.transformer_flops`. Both compare honestly
across tiles and across runs.

**Output per tile:** an `embeddings` array of `(H, W, 128)` int8, zero where a pixel was
not run, plus a per-pixel float32 `scale` factor for turning it back into real numbers. The
model produces 192 dimensions; the first 128 are saved (`save_dim = min(128, repr_dim)`).

### 8. Compressing embeddings to int8 (`quantization.py`)

Embeddings are always compressed from float32 to int8 immediately after the forward pass,
before staging, which makes the staged files and the final store about four times smaller.

**How it works.** For each pixel, the largest absolute value across its 128 channels is
found, and the pixel's values are scaled so that value lands on ±127, then rounded and
clipped. The per-pixel scale factor is stored as float32 alongside, so the original can be
reconstructed:

```
reconstructed = quantized.astype(float32) * scale[..., np.newaxis]
```

The round-trip error is bounded by `scale / 127` per channel — typically under 1% of the
original magnitude. Non-finite values are rejected with a `ValueError` before quantization
rather than being silently encoded.

**It happens per bucket, not per tile.** Because each pixel's scale comes only from its own
128 channels, quantization is per-pixel independent, so `run_inference` compresses each
bucket's rows with `quantize_rows` the moment they come off the GPU and accumulates
straight into the narrow int8 and scale buffers. The full `(H, W, 128)` tile is never
materialised in float32. That is numerically identical to compressing the whole array at
the end, and it shrinks the resident accumulator about fourfold — from roughly 2 GB to
0.5 GB at a 2048-pixel tile — while removing an end-of-tile whole-array pass and its
multi-gigabyte temporaries. `quantize_embeddings` remains as the `(H, W, D)` entry point
and delegates to `quantize_rows`.

**What comes out:**

- `embeddings` — int8, `(H, W, 128)`
- `scales` — float32, `(H, W)`
- `embedding_std` — float32, `(H, W, 128)`, unquantized, present only with
  `compute_std=True`

Assembly validates that staged tiles carry these dtypes and rejects a mismatch, so a
change of dtype cannot silently corrupt a store.

### 9. Staging a finished tile (`assembly.py`)

Each actor writes its finished tile to a staging prefix on S3, at
`{staging_base}/{run_id}/{chunk_label}.zarr`, as **raw, uncompressed** zarr — compression
costs CPU, and CPU on a GPU-priced machine is the scarcest thing there is. The staged
sub-chunks are `256 × 256 × the full band axis` (`INNER_PX`, int8), which is exactly the
final store's inner-chunk geometry: a staged 2048-pixel tile *is* the 8 × 8 grid of inner
chunks it will become, and the band axis is never split (ADR-008 D2).

**What travels with the embeddings.** Alongside `scales`, each staged tile carries the
per-pixel provenance layers, which come in **pairs — a count and a month mask per
sensor**:

| Sensor | How many (`uint16`, H×W) | Which months (12 booleans) |
|---|---|---|
| Optical | `s2_obs_count` | `s2_month_covered` |
| Radar, ascending | `s1_asc_obs_count` | `s1_asc_month_covered` |
| Radar, descending | `s1_desc_obs_count` | `s1_desc_month_covered` |

Both halves of a pair are derived from **one** validity mask per sensor, in
`data_loading.coverage_from_validity`, so the count and the spread cannot disagree about
what counted:

```
                per-timestep validity            coverage_from_validity
 optical  SCL classes            (T,H,W) ──┐
 asc      any non-zero pol       (T,H,W) ──┼──▶  .sum(axis=0)  ──▶  <sensor>_obs_count   (H,W)
 desc     any non-zero pol       (T,H,W) ──┘     any() per month ─▶  <sensor>_month_covered (12,H,W)
```

**Per sensor rather than merged**, because the sensors fail differently: optical gaps are
weather and repeat seasonally, while radar gaps are orbital — the Sentinel-1B failure left
regions with a single orbit direction for years. Or-ing them into one mask would report a
pixel as covered in a month it was seen only by the sensor a given reader cannot use.

**Which of these assembly copies is derived from the store layout, not written out by
hand.** `store_layout.REQUIRED_VARS` is what every staged tile must carry and every
destination must hold; `store_layout.CARRIED_VARS` is everything else, computed from the
layout's own array set. Assembly keeps whichever of them are present in *both* the staged
tile and the destination, so a store predating an array, or a run that stages nothing for
one, needs no special case. Adding an array to the layout is therefore enough to have it
carried — which was not true while the copy set was a hand-written tuple, and cost one
published zone-year of empty `s2_month_covered` planes. A four-dimensional variable's
trailing extent comes from `store_layout.trailing_extent` per variable, because there are
now two different trailing axes (`band` is as wide as the embedding; `month` is twelve) and
no "four dimensions means bands" rule would be safe.

#### Completion markers: how a resume tells finished from interrupted

A staged `.zarr` is **many objects** — group and array metadata plus one per data chunk —
written with no atomic multi-object commit. So a crash mid-upload leaves a store with valid
metadata and missing data, and zarr reads missing data back as **fill values, not an
error**. The presence of the directory therefore proves nothing, which is why completeness
is recorded as its own signal, *after* the store is fully written:

```text
  write_chunk:  rm <label>.done                   retract the old marker FIRST
                    ↓
                ds.to_zarr(…, mode="w")           many objects, no atomic commit
                    ↓
                staged_complete = True            in-store attribute
                    ↓
                <label>.done  (zero bytes)        sibling object, written LAST

  INVARIANT:  .done  ⟹  attribute  ⟹  to_zarr returned
```

The marker is retracted *before* the rewrite, not after: `to_zarr` replaces a tile in
place, so a marker left from an earlier write would keep vouching for the tile throughout,
and a listing taken in that window would call a half-replaced tile complete.

**The gate is the listing.** A sibling `.done` object is something a prefix LIST can see,
so one listing classifies every tile in a run without opening any of them — which is what
keeps verification and resume cheap at zone scale, where there are hundreds of thousands of
live tiles. `_list_staged` derives the three-way split from that one listing, and it is the
only place the split is made, so verification and resume can never disagree:

| state | listing shows | meaning | what happens |
|---|---|---|---|
| **complete** | `.zarr` + `.done` | the write finished | validate shape and dtype, then skip |
| **interrupted** | one of the pair only | a crash caught it part-way | re-infer (`mode="w"` overwrites; no cleanup needed) |
| **skipped** | `.skipped` | no valid pixels at all | skip, and report it as a skip rather than a success |

Interrupted tiles are excluded from the resume set rather than raising. The `run_id` is
derived from the inputs, so a raise would re-fire on the same artefact on every retry and
wedge the cell until someone deleted it by hand. `verify_staged_completeness` *does* raise
if one survives all the way to assembly, and it names it as **interrupted** rather than
missing — the resume scan re-infers interrupted tiles, so one reaching assembly means
either inference never ran at all, or it crashed the same way twice, and the remedy differs
from a tile that was never attempted.

**The in-store attribute is the backstop, and it covers one case a listing cannot.** The
listing is taken once, in the driver; the tiles are read minutes or hours later, in worker
processes. A tile *rewritten* in that window is reported as the listing last saw it — and
because the `run_id` comes from the inputs, two attempts at one zone-year share a staging
prefix, so that window is genuinely reachable. Only the attribute, which does not exist
until a write finishes, describes the tile as it is now. Every reader goes through one
shared opener (`_open_staged_tile`), so this is a single place in the code rather than a
second gate to maintain.

All of this guards the *staging* layer only. The final store is Icechunk, whose
transactional commit already rolls back cleanly on a crash during assembly.

### 10. Assembling the staged tiles into the store

Once the live tiles are done, `writer.assemble` writes them into the final Icechunk store
with plain zarr assignments: a pool of worker processes on the flow runner, each holding a
pickled **fork** of one icechunk session, merged and committed once by the coordinator. No
task graph is ever built over the store, so assembly cost scales with the *live* pixels
rather than with the grid.

The output geometry comes from a `StoreLayout` preset (`config/store_layout.py`) — the same
source of truth the global store's seeder uses — and never from the staged files. The
single-area and global presets describe the same geometry, built from one definition:
`(1, 256, 256, 128)` int8 inner chunks inside `(1, 2048, 2048, 128)` shards for embeddings,
and the matching 2048-pixel shards for scales, counts and standard deviations. The two
names survive because a caller picks one explicitly and the choice is recorded in the
store's creating commit.

#### Write-conflict discipline: northing bands

Two forks writing the same output object would conflict at merge, so the workers partition
the mosaic into **northing bands aligned to the output's write granularity** — the shard
height when sharded, the chunk height otherwise. Bands are disjoint and span the full
easting extent, so no two forks ever touch the same object, and because the tile size
equals the shard pitch, band boundaries fall on tile edges. Boundaries are **weighted by
work** (live tiles per granularity unit) rather than being equal height: an area's live
tiles cluster spatially, and equal-height bands would leave most workers idle while one
dragged out the whole assembly.

```text
                    easting ─────────────────────────►
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

  atomic publish:  branch _assemble-wip ──► phases 1–3 commit here ──► fast-forward main
                   └── main only advances once the timestep's data is fully written ──┘
```

Within a band, tile boundaries in easting still cut output chunks mid-chunk; those partial
chunks are read-modify-written *sequentially inside one fork*, and icechunk sessions read
their own writes, so the merged result is exact.

#### The steps

1. Re-run `filter_chunks_by_roi_mask` to recover the live tile labels — the list is not
   marshalled through Prefect, and the area mask is the source of truth — minus the
   skip-marked ones.
2. **Phase 1, schema and time-axis placement.** Every phase runs on a private
   `_assemble-wip` branch, never on `main` directly. On a fresh store: create the layout's
   array schemas and coordinate arrays, with no chunk data, at a cost independent of
   extent. On an existing store: validate the `_manifest`, then either resize every
   time-dimensioned array by one step (an append *is* a resize plus a write at the new
   index) or, when the time value already exists, target that index for an in-place
   overwrite. That makes a resume **idempotent**: a crashed assembly re-run lands on the
   same index instead of appending a duplicate timestep. On an overwrite, any
   time-dimensioned array this run does *not* write — `embedding_std` when standard
   deviations are off, say, or the other radar orbit's count — is reset to fill at that
   index, so no stale slice describes the overwritten data.
3. **Phase 2, data.** Fork the session; each worker writes its band's tile slices into every
   output array; the coordinator merges the forks, sets the root attributes (run metadata,
   GeoZarr conventions, `_manifest`, merged `time_windows`) and commits once through
   `commit_with_rebase`.
4. **Atomic publish.** Fast-forward `main` to the fully written work-branch tip, guarded by
   `from_snapshot_id` so a concurrent writer to the same store fails loudly, then drop the
   work branch. `main` never observes the half-written state: committing Phase 1 to `main`
   directly would advertise a resized array and a new time coordinate — or, on a fresh
   store, an empty schema — before any worker had written, so a crash in Phase 2 would
   leave `main` serving an all-fill timestep.
5. Delete the staged tiles, unless `cleanup_staging=False`.

#### The global-campaign variant

`writer.assemble_global` skips the banding entirely. One staged 2048-pixel tile is exactly
one output shard (ADR-008 D3), so whole tiles round-robin across workers through
`storage.shard_writer.write_year_shards`. Every shard object is emitted once and **lean**:
all-fill inner chunks are elided, so a fully masked tile costs nothing and an inner chunk
with no valid observation disappears. (Water is a valid surface class as far as the cloud
mask is concerned, so a coastal tile does embed its ocean pixels — the mask selects tiles,
not pixels.) `years_complete` and the per-year run provenance advance in the same single
commit. The zone-fill runner
([`orchestration/runners/zone_fill.py`](../orchestration/runners/zone_fill.py)) drives the
sequence: coverage mask → inference → `assemble_global` → `campaign.tag_zone_year`.

**The campaign's land mask is not a pixel-resolution area of interest** but a per-zone
*coverage bitmap* (`tile_live_2048`), built from the delivery registry by
`ingest/land_mask.py` and the `build-land-mask-coverage` flow (ADR-010). v1.1 delivery
tiles are entirely land-flagged with roughly a one-cell sea buffer, so the registry listing
*is* the mask: a 2048-pixel tile is live if and only if a land cell's footprint intersects
it. The runner reads one small bitmap in a single GET and selects live tiles by direct
index — one tile is one shard is one coverage tile — rather than doing the per-tile
windowed reads `filter_chunks_by_roi_mask` performs for the single-area flows.

#### Write units vs read units, per layout

The write path and the read path deliberately work at different granularities: what a
writer emits in one go is much larger than what a reader has to fetch.

```text
                  SINGLE (one area)               GLOBAL (zone group)
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

**Manifest splitting** is the last row. `assemble` opens the repo under
`manifest_split({"time": 1})` — one manifest shard per timestep, so a one-timestep write
rewrites no earlier timestep's index. Time only, because the rule is to split the axis a
single commit is *narrow* along, and an assemble writes one timestep across the full spatial
extent. The global store bakes the same split into its repository config at seeding
(`global_store_config`). On a store that predates the change, the first write under the
block re-shards the touched array in that commit — a one-time migration that leaves old
chunk data untouched — and every commit after it rewrites only what it touches. What a
manifest is, and a diagram, are in
[`docs/single-vs-global.md`](../../../docs/single-vs-global.md#how-the-store-is-laid-out).

**S3 concurrency is not capped.** The repository opens at icechunk's default of 256 and the
forks inherit it through pickling. A per-fork cap used to be divided out of a fleet-wide
budget and floored at 1 on every campaign fill — the value at which icechunk deadlocks, per
`context_docs/assembly/icechunk-max-concurrent-requests-1-deadlock.md`.

**Codecs in the final store.** The quantized int8 embeddings use the zarr default of bytes
plus zstd. The float32 `scales` and, when computed, `embedding_std` use **PCodec**, a
serializer that models the distribution of floating-point values directly and beats
general-purpose compressors on this kind of data. The tradeoff is that PCodec has no partial
decode — reading any slice of a chunk decompresses all of it — which is why the inner chunks
stay at 256 pixels. It composes with the sharding codec (verified by round-trip and fill
tests on partial shards).

#### Reading the output store

On a store at CONUS scale, open it with `chunks=None` for interactive or selective reads:

```python
ds = open_store(store_path, chunks=None)   # or xr.open_zarr(session.store, consolidated=False, chunks=None)
ds.isel(time=0).sel(northing=slice(...), easting=slice(...)).embeddings.values
```

The default chunking builds one Dask task per on-disk chunk. With 500×500×4 sub-chunks the
band axis alone is 32 chunks, so the graph is `n_time × n_y × n_x × 32` tasks — large enough
at CONUS extent that even a *lazy* `isel` or `sel` runs out of memory while manipulating the
graph, before any data is read. `chunks=None` opens the store zarr-lazy with no graph at
all: slicing is pure metadata, and chunks load only when `.values` is pulled. This is
unrelated to manifest splitting, which bounds commit cost rather than graph size.

#### Assembly telemetry

Both `assemble` and `assemble_global` emit **one machine-readable log line per assembly** —
`ASSEMBLY_SUMMARY: {json}`, the counterpart of the actors' `CHUNK_SUMMARY` — so a slow
assembly can be attributed without re-running it. Each fork worker times its loop into a
`read` phase (fetching staged tiles) and a `write` phase (the raw-zarr assignments), on two
clocks: wall time says how long the phase held the worker, process CPU time says how much of
that was computation, and the difference is time blocked on the object store.

That CPU-versus-wall split is the only honest boundary between compression and upload,
because the two are **fused inside the zarr-to-icechunk write call** — the codec encodes and
the store uploads within one assignment the worker cannot see into. So `write_cpu_s` bounds
the encode cost, `write_s − write_cpu_s` is store wait, and the record says so in-band via
`fused_compress_put`. One field needs care: `bytes` is **logical, uncompressed write
volume**, so it is neither object-store ingress nor egress. Field-by-field meanings live on
`assembly._assembly_summary_line`; keep the keys stable or update the parsers in the same
change.

A global fill also carries **`catch_ups`**, a tally of what `shard_writer.catch_up_to_branch`
did while the forks were writing. It is reported because a healthy commit looks identical
whether the session was kept up to date or simply got lucky, so the tally is the only
evidence the mechanism ran. Read a run of `blocked` ticks as "one or more writers touched
this zone, from the first blocked tick onward" rather than as a count of collisions: the
range checked runs from the session's base to the branch tip, and the base stops moving once
anything is blocking, so one rival commit early in a fill latches every later tick. A fill in
that state is exposed to the stall
`context_docs/storage/writing-to-the-global-store.md` describes.

**The fork phase is watched, and that is the assembly's one backstop.** A daemon thread in
`shard_writer.run_forked` watches the shard counters the workers update in shared memory; if
they stop for `FORK_STALL_TIMEOUT_S` (thirty minutes) it dumps every thread's stack with
`faulthandler`, terminates the pool, and the fill fails normally with its cell
re-dispatched. Alert on `ASSEMBLY FORK PHASE STALLED`. It cannot unwind a coordinator parked
inside icechunk, but the stack dump still fires — and where `CAP_SYS_PTRACE` is denied, that
dump is the only way to get one
(`context_docs/assembly/assembly-wedges-during-fork-phase-2026-09-04.md`).

---

## How Our Performance Optimizations Fit Together

The phases above describe *what the pipeline does*. This section steps back and maps the
performance work layered across them — *what* each optimization targets, *when* it acts,
and *how much* it buys — so the design is legible without reading every phase in depth.

The GPU on each worker is fast enough; a naive pipeline leaves it **idle ~50% of the
time** — not waiting on compute, but on *loading data*, *preparing batches*, and
*writing results*, plus a serial "cold start" at the top of every chunk. Every
optimization here removes one source of that idle time. None changes what the model
computes: all are **bit-identical to `main`'s outputs** — they change scheduling and I/O,
not the math — except the batch-size bump, which shifts int8 values by ≤1–2 levels (inside
the ADR-012 cross-config envelope, well within same-code quantization noise). (One math
caveat that is *not* part of this work: the model builder's cuDNN-GRU fusion involves a
small reset-gate approximation (§7), but it predates this campaign and runs identically
on `main`, so it cancels out of any before/after comparison here.)

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
- **the transfer "bubble" / double-buffering** — a batch must be copied from CPU to GPU
  ("host→device") before the GPU can compute it; done naively the host waits for each copy
  and each forward pass in turn, so the card stalls whenever the host is the slow one — a
  **bubble**. Staging batches in **pinned** (page-locked) host memory and issuing the copies
  non-blocking lets the host run ahead and keep two batches in flight at once
  (**double-buffering**, "two-deep"), so the queue is never empty. Note this pipeline uses a
  single CUDA stream: it removes host-side stalls, not transfer time. (**D2H** =
  device→host, copying results back.)
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
                                       gated the GPU; now it doesn't)          [§6]
  ◐ async two-deep GPU pipeline      pinned double-buffers, one stream, host runs
                                      a batch ahead → no host-side bubble      [§7]
  ◐ batch size 7168 (BF16)           bigger GEMMs use the tensor cores more
                                      fully (the one non-bit-identical change) [§7]
  ○ background staging write         the ~7.5 s S3 upload runs on a writer
                                      thread, overlapping the NEXT chunk's load [§4.3]
```

### The per-chunk decision tree

Two chunks make the tree concrete before you read it. A **dense interior tile** has
little sparsity of either kind to exploit — it crops nothing and prunes little; its win
comes from the striping + prefetch pipeline (§4.2–4.3), because its high valid-pixel
count keeps the GPU busy enough to hide the strip loads. A **cloudy coastal sliver** is
the opposite: high *temporal* sparsity (prune empty dates) and high *spatial* sparsity
(crop to the bbox, skip empty row bands), so its win comes almost entirely from *reading
less* (§4.1) — and since its low valid-pixel count makes it a single serial strip,
reading less is the only lever it has. Same code, opposite paths.

Each chunk takes **one path** through the tree below, chosen from its valid-pixel count
and where the valid data sits. Read top to bottom:

```
A chunk arrives → load its SCL mask → count valid pixels, find their bbox
│
├─ Q1. SPATIAL sparsity: do the valid pixels sit in a narrow easting window?
│      (cropping saves ≥ 10% of the width)
│        ├─ yes → ◐ crop the S2 read to that column bbox — edge/coast   [§4.1]
│        │        slivers read a fraction of the bytes
│        └─ no  → read full width (interior tiles stay byte-identical)
│
├─ TEMPORAL sparsity (always): ○ prune S2 timesteps empty everywhere    [§4.1]
│           — cloudy dates the resampler would never read
│
├─ Q2. Does bands + full mask fit ONE RAM budget?
│        ├─ yes → single strip — no split, no prefetch (common interior)
│        │
│        └─ no  → SPLIT into northing strips  ● bounds peak host RAM    [§4.2]
│                 │
│                 └─ Q3. Enough valid data for the GPU to hide the strip
│                        loads behind inference?
│                          ├─ yes (dense) →
│                          │     ◐ intra-chunk strip prefetch: strip     [§4.2]
│                          │       i+1 loads while strip i runs the GPU
│                          │     ○ starter strip: small first slice, GPU [§4.3]
│                          │       starts one read sooner
│                          └─ no (wide but few valid px) →
│                                prefetch OFF, strips at the PAIR budget [§4.2]
│                                (one set resident → bigger budget safe;
│                                fewer, larger reads)
│
├─ Q4. On the LAST strip, is this a RAM trough (≤ 1× budget)?
│        ├─ yes, and a next chunk is reserved →
│        │     ● cross-chunk starter prefetch: preload the next chunk's [§4.3]
│        │       mask + 256-row starter NOW (mask-only when the rung says
│        │       the starter wouldn't pay for its extra read), so its GPU
│        │       work starts ~6 s later instead of ~24–36 s
│        └─ no (pair budget) → skip it; the next chunk takes the serial
│              prologue (slower, but never over the RAM ceiling)
│
└─ SPATIAL sparsity (per strip): ◐ empty-strip skip — a strip whose    [§4.1]
                           mask slice has zero valid pixels skips the S2 band read
```

### Summary

Every optimization here leaves outputs **bit-identical to `main`'s** except the batch-size
change (¹) — the rest alter scheduling and I/O, not the math. (The builder's cuDNN-GRU
reset-gate approximation (§7) predates this work and is identical on `main`.) *Window* is
which GPU-idle window each reclaims (see the three-windows diagram above).

| Optimization | Family | Window | Triggers on… | Impact |
|---|---|---|---|---|
| Vectorised temporal resampling (§6) | core loop | mid-forward | always | ● large |
| Async two-deep GPU pipeline (§7) | core loop | mid-forward | always (CUDA) | ◐ medium |
| Batch size 3584 → 7168 (§7) | core loop | mid-forward | always | ◐ medium¹ |
| Background staging write (§4.3) | core loop | write (post) | always | ○ small |
| Valid-pixel-aware northing striping (§4.2) | adaptive | *enabling* | chunk exceeds one RAM budget | ● large² |
| Intra-chunk strip prefetch (§4.2) | adaptive | mid-forward | dense/hideable split | ◐ medium |
| Starter strip (§4.3) | adaptive | cold-start | dense split with a real body | ○ small |
| Cross-chunk starter prefetch (§4.3) | adaptive | cold-start (next chunk) | last strip is a RAM trough + next chunk reserved | ● large |
| Timestep pruning (§4.1) | adaptive | cold-start | **temporal** sparsity (cloudy/empty dates) | ○ small–◐ |
| Empty-strip skip (§4.1) | adaptive | cold-start (per strip) | **spatial** sparsity (a row band with no valid px) | ◐ medium |
| Easting bbox crop (§4.1) | adaptive | cold-start | **spatial** sparsity (valid px in a narrow column window) | ◐ medium³ |

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
  the bounded (~2 GiB) cross-chunk prefetch instead hold peak at ~52%, well under the
  60% ceiling, so chunk-density spikes at UTM-zone scale can't OOM the node. The unused
  headroom is intentional insurance, not waste.
- **GRU restructuring** — the model builder already fuses the recurrent stack to cuDNN;
  a hand-restructure was written, measured as no faster, and reverted as dead code.
- **FP16 fast-accumulate** — an L40S GEMM microbench showed BF16 already runs at the
  full dense tensor-core ceiling, so FP16 buys nothing here. BF16 stays.
- **Adaptive token-budget batching** — measured; B=7168 is already throughput-optimal
  across sequence lengths, so a dynamic budget added complexity for no gain.

---

---

## Fault tolerance

All fault handling lives in `scheduling.py`. `_process_chunks_work_stealing` owns
the main loop; `ActorPool` encapsulates actor state and lifecycle operations
(replacement, idle retirement, instance-ID resolution).

| Failure mode | Handling |
|---|---|
| Chunk raises exception | Re-queued up to 2 times (`max_chunk_retries`); logged as permanently failed if exhausted |
| Actor dies (OOM, instance loss) | `ActorPool.replace()` spawns a replacement; instance ID of the new node resolved lazily so the main loop isn't stalled |
| >50% of actor slots dead | `ActorPool.replace()` escalates log severity to ERROR / CRITICAL |
| Replacement actor still initialising | `dispatch_idle()` queues work to it anyway — Ray buffers the call until `__init__` completes |
| Idle actor after work drains | `ActorPool.retire_idle()` kills actors idle past `idle_grace_sec` (default 120s), freeing GPU nodes; never drops below the remaining work count, nor below the caller's `floor` |
| Idle fleet while a chained source waits | A source answering `[]` has nothing DELIVERABLE right now, so the fleet WINDS DOWN through the wait — all but one ready actor, the liveness floor — and re-grows through the batch machinery when work arrives, the session staying alive throughout. Deliverable, not merely present: a landed cell the single feeder cannot hand over while it is inside any `CellInputs` call does not hold the fleet. Only where the pool can grow back: the mid-run wind-down is skipped when actor batching is off (`actor_request_batch_size = 0`), since nothing would recreate the retired slots. See `context_docs/inference/the-fleet-and-the-work-source.md` |
| Slots still waiting on placement when work runs out | `ActorPool.retire_initializing()` cancels them after the same idle grace, so the run stops asking AWS for machines it has no work for — never below one READY actor, which is the only thing that can run arriving work |
| Several actors packed on one instance (`num_gpus < 1`) | Retirement kills the actor but terminates the INSTANCE only when no live slot is left on it, so retiring one actor cannot kill a busy sibling or the retained floor actor |
| Chunk stalls (no batch update for 5 min) | `ProgressTracker` detects per-chunk staleness; `_poll_tracker()` aborts once `ActorPool.systemic_stall_threshold` chunks stall simultaneously — a tenth of `fleet_size`, floor 3 |
| Flow cancelled in Prefect UI | `on_cancellation` hook runs `ray down` |

---

## Key configuration (`config.py`)

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

## Model architecture constraint

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
| `runner.py` | `run_inference` — the domain-pure end-to-end run: create actors, dispatch tiles, tear down |
| `read_plan.py` | The strip, crop and prefetch arithmetic on its own, with no Ray, torch or actor state |
| `data_loading.py` | Icechunk/Zarr loader with 3-phase selective S2 timestep loading |
| `dataset.py` | Valid-pixel filtering and lazy per-batch indexing |
| `quantization.py` | Int8 quantization with per-pixel scale factors and dequantization |
| `assembly.py` | Staged Zarr writes + raw-zarr fork/merge assembly into Icechunk (single-ROI + global modes) |
| `actors.py` | Ray actor wrapping the full per-chunk pipeline |
| `scheduling.py` | Work-stealing scheduler (`_process_chunks_work_stealing`) and `ActorPool` — manages dispatch, actor replacement, idle retirement, and tracker polling |
| `progress.py` | Lightweight Ray actor (`num_cpus=0`) polled by the flow runner every 60s for batch-level progress and 5-minute stall detection |

---

## Accessing the Ray dashboard

Inference runs a Ray cluster whose head node is an EC2 instance in a private subnet.
`_log_ray_dashboard_ssm_command` (in `providers/aws/ray.py`) logs a ready-to-paste command
once the head is up, with the instance ID and region already filled in:

```bash
# INSTANCE_ID is the head node — tagged ray-node-type=head
aws ssm start-session \
  --target ${INSTANCE_ID} \
  --document-name AWS-StartPortForwardingSession \
  --region <aws-region> \
  --parameters '{"portNumber":["8265"],"localPortNumber":["8265"]}'
```

Then open http://localhost:8265 in your browser.

That tunnel uses `AWS-StartPortForwardingSession`, which forwards a port on the target
itself. The `AWS-StartPortForwardingSessionToRemoteHost` variant is for hosts reachable
*from* the target (RDS, an internal load balancer); current SSM agents refuse loopback
destinations for it and fail with `Forwarding to IP address localhost is forbidden`.

It needs the [Session Manager plugin](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (`brew install session-manager-plugin`).
