# Inference performance: keeping the GPU fed

The GPU is fast enough. A naive pipeline leaves it idle about half the time, and almost none of
that time is spent computing — it is waiting for data. Imagery travels from S3 to host memory,
across PCIe into VRAM, and then into the tensor cores, each hop narrower than the compute it
feeds. That idle falls into **three windows**, the fixes into **two families**, and the adaptive
family turns on
**two kinds of sparsity**. Nothing here changes what the model computes.

## Why the card idles

**A tile does not fit on the card.** Its imagery is far larger than the L40S's 46 GB of VRAM, so
it is swapped through in pieces: read from S3, resampled on the host, copied to the device,
computed, copied back. The card only works on what has arrived, so whichever hop falls behind
becomes idle time. It lands in **three windows** — the *cold start* before a tile's first batch,
gaps *mid-forward* while the card waits on the next sub-batch, and the *write* at the end. Every
optimization below is tagged with the window it reclaims.

```
   COLD START (idle) ──▶ FORWARD PASS (busy) ──▶ WRITE (idle) ──▶ next tile

  1. COLD START — GPU idle: load SCL mask, read the 1st strip, build the dataset.
       reclaimed by:  crop · prune · empty-strip skip ·
                      cross-chunk prefetch (next tile's cold start already done)

  2. FORWARD PASS — GPU busy (the real work): sub-batch → sub-batch, across strips.
       keep it fed, no gaps:  vectorised resampling · async two-deep pipeline ·
                              batch 7168 · intra-chunk strip prefetch

  3. WRITE — GPU idle: staging upload.
       reclaimed by:  background write, overlapping the next tile's cold start
```


## Why a faster card would not fix it

Even with data resident, the card is not short of compute. Fleet telemetry shows the L40S
running its multiprocessors at 99% occupancy while the tensor pipes sit under half engaged, at 85
effective TFLOPS against 362 quoted. An A10G-versus-L4 comparison settled why: the card with
twice the bandwidth and barely half the tensor compute is the faster one. The forward pass is
**memory-bandwidth bound** — the limit is moving operands in and out of VRAM, not multiplying
them. That is what makes batch size a lever, since a larger batch does more computation per byte
moved, and why pinned, double-buffered transfers buy more than raw FLOPS would.

## The two families of fix

- **The always-on core loop** keeps the card fed *during* the forward pass. It runs on every
  tile, unconditionally.
- **The per-tile adaptive path** avoids reading bytes at all, and which route a tile takes
  depends on how much valid data it actually holds. A dense interior tile and a cloudy coastal
  sliver take opposite paths through the same code.

## The two kinds of sparsity

What the adaptive path exploits, both read from the Sentinel-2 SCL mask:

- **Temporal** — whole dates are cloud or nodata, so pruning them shrinks *time*.
- **Spatial** — valid pixels occupy only part of the footprint, so cropping shrinks *columns* and
  skipping empty strips drops *rows*.

Both are distinct from a tile's **valid-pixel count**, which is the volume of inference work left
after them. That count sets how long the GPU works and which strips run first, not how much gets
read or where the strips are cut.

## What none of it changes

**Outputs are bit-identical to `main`'s** except for the two changes that regroup pixels into
different GPU sub-batches, the batch size and the strip budget, and both exceptions are measured
rather than assumed. Comparing `main` at batch 3584 against the shipped branch at 7168
over ~512M values per tile: int8 values land within one level on **≥99.99%** of them, the largest
observed deviation is **2 levels**, per-pixel scale drift stays **≤0.78%**, and the dequantized
embeddings have **cosine similarity ≥0.9999** — which is the number that matters, since these
vectors are used as features rather than read individually. Footprint and observation-count layers
stay exact. The cause is cuBLAS picking different kernels for different batch shapes, not a change
to the model; the envelope and the harness are in
[ADR 012](../context_docs/decisions/012-validated-equivalence-for-inference-outputs.md). The
strip budget, over a full Iowa year, leaves a median 99.95% of values identical, at most one value
4 levels off, and 99.98% of each pixel's nearest neighbours unchanged
([`simplifying-the-strip-loader.md`](../context_docs/inference/simplifying-the-strip-loader.md) §2).

## Terms and impact ratings

*Reference for the sections that follow.*

- **GEMM** — a general matrix–matrix multiply, the dominant computation inside the
  transformer. GPUs run GEMMs on dedicated **tensor cores**, which are most efficient
  when the matrices are *large*; a bigger **batch** (more pixels multiplied in one go)
  makes each GEMM larger, so the tensor cores idle less.
- **batch / sub-batch** — pixels are inferred in groups, not one at a time. A tile's
  pixels are split into fixed-size **sub-batches**, each of which is one GPU forward pass.
- **the transfer "bubble" / double-buffering** — a batch must be copied from CPU to GPU
  ("host→device") before the GPU can compute it; done naively the host waits for each copy
  and each forward pass in turn, so the card stalls whenever the host is the slow one — a
  **bubble**. Staging batches in **pinned** (page-locked) host memory and issuing the copies
  non-blocking lets the host run ahead and keep two batches in flight at once
  (**double-buffering**, "two-deep"), so the queue is never empty. Note this pipeline uses a
  single CUDA stream: it removes host-side stalls, not transfer time. (**D2H** =
  device→host, copying results back.)
- **tile** — the unit of work: a 2048 x 2048-pixel square of ground, one year deep. The code
  calls it a `chunk`, which is why two optimization names below keep that word.
- **SCL mask** — Sentinel-2's per-pixel Scene Classification Layer; here, the
  cloud/validity mask that records which pixels and which dates hold usable data. Every
  per-tile decision below starts from it.

**Impact legend.** **● large** · **◐ medium** · **○ small**. "Large" means it removed a
dominant share of GPU-idle time in the profiles; "small" means a real but minor trim. Hard
numbers are in `context_docs/inference/inference-on-gpus.md`.

## The always-on core loop

*[Core loop family](#the-two-families-of-fix). Reclaims the mid-forward and write [windows](#why-the-card-idles), on every tile,
regardless of [sparsity](#the-two-kinds-of-sparsity).*

```
Runs on every tile, regardless of density. The first three keep the GPU busy
MID-FORWARD; the write hides the POST-FORWARD idle (see the three windows above):

  ● vectorised temporal resampling   batch prep 600–650 ms → ~130 ms/sub-batch
                                      (was SLOWER than the GPU forward → prep
                                       gated the GPU; now it doesn't)          [§6]
  ◐ async two-deep GPU pipeline      pinned double-buffers, one stream, host runs
                                      a batch ahead → no host-side bubble      [§7]
  ◐ batch size 7168 (BF16)           bigger GEMMs use the tensor cores more
                                      fully (the one non-bit-identical change) [§7]
  ○ background staging write         the ~7.5 s S3 upload runs on a writer
                                      thread, overlapping the NEXT tile's load [§4.3]
```

## How a tile's path is chosen

*[Adaptive family](#the-two-families-of-fix). Reclaims the cold-start [window](#why-the-card-idles) by reading less, and the route
a tile takes is decided by its [sparsity](#the-two-kinds-of-sparsity) and how many bytes are left to read.*

A **dense interior tile** has little sparsity of either kind to exploit — it crops nothing and
prunes little; its win comes from the strip and prefetch pipeline
([§4.2](../src/tessera_embeddings/inference/README.md#42-loading-a-tile-in-strips) and
[§4.3](../src/tessera_embeddings/inference/README.md#43-starting-the-next-tile-early-and-finishing-the-last-one-late)),
because its high valid-pixel count keeps the GPU busy long enough to hide each strip load. A **cloudy
coastal sliver** is the opposite: high *temporal* sparsity (prune empty dates) and high *spatial*
sparsity (crop to the bbox, skip empty row bands), so its win comes almost entirely from *reading
less* ([§4.1](../src/tessera_embeddings/inference/README.md#41-read-as-little-as-possible)) — and
since what is left to read usually fits one strip, with nothing to hide it behind, reading less
is the only lever it has. Same code, opposite paths.

Each tile takes **one path** through the tree below, chosen from how many bytes it has to read
and where the valid data sits:

```
A tile arrives → load its SCL mask → count valid pixels, find their bbox
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
│        ├─ yes → single strip — no split (common on sparse and edge tiles)
│        │
│        └─ no  → SPLIT into budget-sized northing strips               [§4.2]
│                 ● bounds peak host RAM
│                 ◐ strip prefetch: strip i+1 loads while strip i
│                   runs the GPU
│                 ○ densest strip first, empty strips last, so no load
│                   waits behind a strip with nothing to infer
│
├─ On the last strip WITH PIXELS, is a next tile reserved?
│        ├─ yes → ● cross-chunk prefetch: the pipeline's next strip is  [§4.3]
│        │        the next tile's first, loaded with its mask, so its
│        │        GPU work starts without a serial prologue
│        └─ no  → the next tile takes the serial prologue
│
└─ SPATIAL sparsity (per strip): ◐ empty-strip skip — a strip whose    [§4.1]
                           mask slice has zero valid pixels skips the S2 band read
```

## Every optimization, and what it buys

*Both [families](#the-two-families-of-fix), each tagged with the [window](#why-the-card-idles) it reclaims.*

Every optimization here leaves outputs **bit-identical to `main`'s** except the two that change
which pixels share a GPU sub-batch, the batch size and the strip budget (¹) — the rest alter
scheduling and I/O, not the math. (The builder's cuDNN-GRU
reset-gate approximation ([§7](../src/tessera_embeddings/inference/README.md#7-the-forward-pass-on-the-gpu-inferencepy)) predates this work and is identical on `main`.) *Window* is
which GPU-idle window each reclaims (see the three-windows diagram above).

| Optimization | Family | Window | Triggers on… | Impact |
|---|---|---|---|---|
| Vectorised temporal resampling ([§6](../src/tessera_embeddings/inference/README.md#6-building-each-pixels-observation-sequence-samplingpy)) | core loop | mid-forward | always | ● large |
| Async two-deep GPU pipeline ([§7](../src/tessera_embeddings/inference/README.md#7-the-forward-pass-on-the-gpu-inferencepy)) | core loop | mid-forward | always (CUDA) | ◐ medium |
| Batch size 3584 → 7168 ([§7](../src/tessera_embeddings/inference/README.md#7-the-forward-pass-on-the-gpu-inferencepy)) | core loop | mid-forward | always | ◐ medium¹ |
| Background staging write ([§4.3](../src/tessera_embeddings/inference/README.md#43-starting-the-next-tile-early-and-finishing-the-last-one-late)) | core loop | write (post) | always | ○ small |
| Budget-sized northing strips ([§4.2](../src/tessera_embeddings/inference/README.md#42-loading-a-tile-in-strips)) | adaptive | *enabling* | tile exceeds one RAM budget | ● large¹ ² |
| Strip prefetch ([§4.2](../src/tessera_embeddings/inference/README.md#42-loading-a-tile-in-strips)) | adaptive | mid-forward | any split tile | ◐ medium |
| Cross-chunk prefetch ([§4.3](../src/tessera_embeddings/inference/README.md#43-starting-the-next-tile-early-and-finishing-the-last-one-late)) | adaptive | cold-start (next tile) | next tile reserved | ● large |
| Timestep pruning ([§4.1](../src/tessera_embeddings/inference/README.md#41-read-as-little-as-possible)) | adaptive | cold-start | **temporal** sparsity (cloudy/empty dates) | ○ small–◐ |
| Empty-strip skip ([§4.1](../src/tessera_embeddings/inference/README.md#41-read-as-little-as-possible)) | adaptive | cold-start (per strip) | **spatial** sparsity (a row band with no valid px) | ◐ medium |
| Easting bbox crop ([§4.1](../src/tessera_embeddings/inference/README.md#41-read-as-little-as-possible)) | adaptive | cold-start | **spatial** sparsity (valid px in a narrow column window) | ◐ medium³ |

¹ Not bit-identical. Both regroup pixels into different GPU sub-batches, cuBLAS picks
different kernels for different batch shapes, and a small fraction of int8 values shift by a
level or two. A before-and-after diff is judged against the ADR-012 **cross-config** envelope
(int8 within ±1 on ≥99.5% of values, scale drift ≤3%, worst-pixel cosine ≥0.999) — not the
same-config bit-identity gate the other rows meet. Observed: ±2 for the batch size, ±4 on one
value for the strip budget over a full Iowa year.

² Foundational — it bounds peak RAM, which is what makes every other adaptive choice
safe; it also drops a ~13 s fixed read per dense tile.

³ Large on edge/coast slivers (a 1.5K-valid-pixel tile dropped from ~39 s of loading to
roughly bbox-proportional), negligible on interior tiles (which skip it).

## The model itself was also changed

Everything above rearranges *when* work happens. Four changes alter *what runs on the card*, and
they are the reason the scheduling work had a fast forward pass to schedule around. All four are
applied at build time in `models/builder.py` and `models/modules.py`, after the checkpoint loads
and before the model is frozen. They were made on v1.1. The v2 Large student shares the positional
encoder, so the two changes to it apply there too; it has no recurrent layer and its checkpoint
carries no training-only heads, so the other two do not.

**The recurrent layer is replaced with a fused one.** The pooling head's `CustomGRU` is
checkpoint-faithful but steps the sequence in Python, one kernel launch per timestep — about 480
of them. `_fuse_custom_gru` swaps in PyTorch's `nn.GRU`, which cuDNN runs as roughly one launch,
so the recurrence stops being bound by launch overhead. The two are not drop-in compatible, and
the swap folds the weights across two differences:

- **The update gate convention is inverted.** Tessera computes `h' = (1-z)h + zn`, where `z`
  selects the new candidate; `nn.GRU` computes `h' = (1-z)n + zh`, where `z` keeps the old state.
  Since `1 - sigmoid(x) = sigmoid(-x)`, every `z` weight and bias is negated.
- **The reset gate sits on the other side of the matmul.** Tessera applies it before,
  `W_hh @ (r * h)`; `nn.GRU` applies it after, `r * (W_hh @ h + b_hh)`. These are **not**
  equivalent for dense weights. It is a real approximation, and it is accepted because the reset
  gate is close to 1 on most dimensions after training. This one predates the performance
  campaign and runs identically on `main`, so it cancels out of any before-and-after comparison
  here — but it is an approximation, not a rearrangement, and it should not be filed with them.

**Positional encoding computes in FP32 and casts its output.** Without the explicit cast,
PyTorch's dtype promotion (BF16 + FP32 → FP32) spreads FP32 through the entire transformer and
the GRU behind it — measured at 7 TFLOPS against 20–30 on tensor cores. One cast on one tensor
keeps the rest of the graph in BF16.

**That encoding is written into uninitialised memory.** `pe` is allocated with `torch.empty`
rather than `zeros`, because the `0::2` and `1::2` strided writes partition an even `d_model`
and leave nothing unwritten. The zero-fill was multi-gigabyte dead work for identical values.
The intermediate `angles` tensor is also freed before the cast rather than after: at the largest
bucket (B=7168, T=256) it is about 2.6 GiB, and holding it through `pe.to()` co-resides it with
both the FP32 encoding and the BF16 output, which is VRAM the two concurrent backbones cannot
spare.

**Training-only parameters never reach the graph.** The checkpoint carries a BarlowTwins
`projector` and a `segmented_matryoshka_projector` that served the variable-width training
objective. Both are stripped by prefix before the model is built, so neither occupies VRAM nor
appears in a forward pass.

Two further options were tried and are off, both measured worse rather than merely unhelpful.
`torch.compile` captured the model as a CUDA graph, consumed 11.6 GB of VRAM and roughly doubled
the forward pass, because the recurrent layer recompiled for every distinct sequence length it
saw. cuDNN's autotuner searches for the fastest kernel per input shape, and bucketing changes the
shape constantly, so it searched constantly and inflated host memory doing it.

## What we ruled out, and why

*Measured and rejected. Absent by design, not oversight.*

Profiling ruled these out, so they're absent by design, not oversight:

- **Greedily prefetching to fill RAM.** We deliberately **leave host RAM on the table.**
  Prefetching the whole next tile to use the spare RAM co-resides two full working sets
  and spikes peak host RAM to ~92–95% — which OOM-killed a worker. The strip budget, and
  a cross-chunk prefetch that loads only the next tile's first strip in place of the
  pipeline's next one, instead hold a full Iowa year at ~50%, under the 60% ceiling, so
  tile-density spikes at UTM-zone scale can't OOM the node. The unused
  headroom is intentional insurance, not waste.
- **GRU restructuring** — the model builder already fuses the recurrent stack to cuDNN;
  a hand-restructure was written, measured as no faster, and reverted as dead code.
- **FP16 fast-accumulate** — an L40S GEMM microbench showed BF16 already runs at the
  full dense tensor-core ceiling, so FP16 buys nothing here. BF16 stays.
- **Adaptive token-budget batching** — measured; B=7168 is already throughput-optimal
  across sequence lengths, so a dynamic budget added complexity for no gain.

