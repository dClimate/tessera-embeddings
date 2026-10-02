# Inference on 2048-px mosaics: what changed, what was measured, what shipped

The mosaics are now stored in 2048-px chunks, the same size as the inference tile. This record
covers what that let inference change, how each change was tested on the finished Iowa mosaic, and
what shipped. The experiment branch is `experiment/frisky-inference`, cut from the frozen ingest
work (`experiment/frisky-ingest`, PR #205). How inference works today:
[`../../src/tessera_embeddings/inference/README.md`](../../src/tessera_embeddings/inference/README.md);
the GPU record it rests on: [`inference-on-gpus.md`](inference-on-gpus.md).

## What 2048 changes

**Where the money is.** Graphics cards were $537K of the last campaign's $828K, 65%
([`../campaign/campaign-cost-model.md`](../campaign/campaign-cost-model.md) §12). The forward pass
is about 90% of a tile's wall clock, so a saving comes from keeping the card busier or from a
cheaper forward pass.

**The read is now exact.** Each tile reads one storage chunk per band and date. At 4000² it
decompressed four times the pixels it used and paid about 13 s of fixed cost per strip read, so
the old strip planner worked hard to avoid strips. At 2048² an extra strip costs a background
decompression and nothing more.

**On its own, 2048 moved neither the GPU bill nor the memory peak.** The year of Iowa on 2048
mosaics (flow run `d7e25533`, against `a60550ae` on 4000²): 33.5 GPU-hours against about 34,
GPU-idle overhead 5.7 s per tile against about 6, utilisation 92%, host RAM peak 16.1 GB of 30.9
either way. The card was already fed, and the peak is set by the strip budget rather than by the
storage chunk. What 2048 changed is the price of a strip, and that is what the work below uses.

**Which card the savings apply to.** Production already runs the smallest L40S box,
`g6e.xlarge` (32 GiB host RAM). No 16 GiB box costs less per unit of work than it (`g5.xlarge`
1.18×, `g6.xlarge` 1.36×, on the throughput measured in `inference-on-gpus.md` §4), and the
measured RAM floor (below) rules them out anyway. The ceiling is peak host RAM under **60%** of
usable; the margin under it is what absorbs the memory spikes a global run does hit.

## Options and outcomes

| Change | Outcome | Evidence |
|---|---|---|
| Collapse the strip planner to budget-sized strips and one uniform prefetch | **shipped** (`6d9d5db3`): −395 lines, ~4 s less GPU-idle overhead per tile (~2.5% of tile time), peak RAM 2–5 points lower | round 2 |
| Fit an actor in 16 GiB for a cheaper A10G fallback (`g5.xlarge`) | **dropped**: even 1 GiB strips leave a ~12 GB peak, ~80% of a 16 GiB box | round 1 |
| Larger strip budgets on 32 GiB boxes, for fewer strips | **moot**: throughput is flat from 1 to 5.75 GiB, so the smaller budget wins on headroom | round 1 |
| Drop the S2 easting crop (`x_sub`) | **held**: it saves only resident memory, and removing it changes `eligible_px`, a published registry field | — |
| Precompute the positional encoding as a day-of-year table | **candidate**: bit-identical, forward 5–7% faster; its peak VRAM is under investigation | benchmark |
| `torch.compile` pieces of the forward pass | **not pursued**: the transformer layers run as one fused op the compiler cannot open (3–5% slower); compiling the encoding instead gains 6–8% but changes outputs, which the table avoids | benchmark |
| PyTorch 2.5.1 → 2.14.1 on the workers | **separate PR off `main`**: eager is no faster (3–5% slower on the deepest bucket) and peaks 15% higher in VRAM there, so the per-card batch fit needs re-measuring | benchmark |
| FP8 matrix multiplies | **denied**: changes the outputs beyond ADR 012, splits the fleet by card, and could not be mixed into the BF16 store | — |

## What shipped: one strip budget, one prefetch

`read_plan._strip_plan` tiles a chunk into the tallest strips whose bands and mask fit
`_S2_STRIP_BYTE_BUDGET`, now **3.5 GiB** (was 5.75). The actor's load pipeline is always one strip
deep, and it runs across tile boundaries: on a tile's last strip, the "next strip" is the next
tile's mask and first strip. Removed: the three-regime plan, the starter strip, the estimator
constants, the cross-tile prefetch tiers and caps, and the `TESSERA_DISABLE_XCHUNK_PREFETCH`
switch. The scheduler's next-chunk reservation (`prefetch_hint`) is unchanged.

**Why it is not bit-identical.** The model's arithmetic is unchanged and two runs of the same code
match bit for bit. What moves is which pixels share a 7,168-pixel sub-batch: new strip boundaries
regroup them and change the size of each strip's last partial sub-batch, cuBLAS picks a different
kernel for a different shape, and its different reduction order shifts the last BF16 bit. ADR 012
names this mechanism (fact 2: "bucket occupancy varying with strip boundaries") and holds it to
its cross-config envelope.

**Why it does not matter for using the embeddings.** Against today's code, 99.84–99.99% of stored
int8 values are identical and ≥ 99.9995% are within one level; the worst value moves 3 levels; the
lowest per-pixel cosine is 0.9999 (an angle under 1°) and the mean 1.000000. Storing in int8
already rounds every value by up to half a level, which by itself puts each stored vector about
cosine 0.99998 from the model's output, and the published store already mixes A10G and L40S tiles,
which differ by the same kind of shimmer. Nearest neighbours, classes and clusters come out the
same except for pixels sitting on a decision boundary, which a different card would flip too. The
inference code identity changes, so old and new tiles cannot mix inside one existing store.
Accepted on these grounds (2026-10-02); two envelope metrics sit near their edges (scale drift
1.56% against 1.6%, cosine 0.999901 against 0.9999).

## Round 1 — yield dev, four arms at once

Four concurrent arms of 6 `g6e.xlarge` workers on the Iowa year, observed for 25 minutes, on code
bundles that changed only worker-side constants. On the 26 tiles every arm finished:

| Arm | Seconds per tile | GPU-idle overhead | Peak RAM |
|---|---|---|---|
| today's code (5.75 GiB strips) | 169 | 23.0 s | 16.4 GB (53%) |
| same code again | 166 | 23.1 s | 16.6 GB (54%) |
| 3.5 GiB strips, 1 GiB cross-tile cap | 160 | 20.4 s | 12.7 GB (41%) |
| 1 GiB strips (8 per tile) | 165 | 16.6 s | 12.4 GB (40%) |

Throughput is flat across budgets, and below ~3.5 GiB the strip budget no longer sets the peak.
The second arm was meant to switch the cross-tile prefetch off, but the switch is read from the
actor's runtime environment, which the driver sets, and a code bundle changes only the workers.
It ran as a second control instead, which gives the noise floor: about 2% between identical arms.
A tile loaded without any prefetch waits about 5 s before its first forward pass, so the
cross-tile prefetch is worth keeping, but not its tiers.

## Round 2 — global-tessera-dev, today's code against the simplified code

Two arms on today's code and two on `6d9d5db3`, 3 `g6e.xlarge` workers each, run side by side on
one worker image while L40S capacity was scarce, observed for 35 minutes. On the 18 tiles all four
finished:

| Arm | Seconds per tile | Inference | GPU-idle overhead | Peak RAM |
|---|---|---|---|---|
| today's code, a | 161 | 140 | 23.8 s | 14.0 GB (45%) |
| today's code, b | 163 | 139 | 23.2 s | 13.9 GB (45%) |
| simplified, a | 164 | 145 | 19.2 s | 12.3 GB (40%) |
| simplified, b | 154 | 134 | 19.2 s | 13.2 GB (43%) |

The inference spread is card-to-card; the overhead drop is the change. The equivalence check
(`te-compare-outputs`, run on a head node in the account) gave 21/21 tiles bit-identical between the
two controls and 0/39 passing the same-config gate between control and simplified, all 39 inside
the cross-config envelope, with the figures quoted above.

## The forward-pass benchmark

One `g6e.xlarge`, the production model and checkpoint, B = 7,168, five (S2, S1) depths from 48/16
to 256/104, median of 8 timed forwards after warm-up (`temp/frisky-dev/bench_fusion.py`):

| Variant (PyTorch 2.5.1) | Forward time vs eager | Outputs vs eager | Peak VRAM, deepest bucket |
|---|---|---|---|
| eager (today) | — (1.98M tokens/s) | — | 18.6 GiB |
| compile the transformer layers | 3–5% slower | identical | 19.3 GiB |
| compile embedding + positional encoding | 6–8% faster | int8 93–96% exact, cosine ≥ 0.99999 | 18.6 GiB |
| positional encoding as a 367-row table | 5–7% faster | bit-identical | 24.0 GiB |

The eager rate matches the 2.06M tokens/s per L40S measured on the fleet. On PyTorch 2.14.1 the
variants rank the same, except that the table took 2.2 s at the deepest bucket, at a 26.7 GiB peak;
eager there is level on shallow buckets and 3–5% slower on the deepest, at 21.3 GiB. The benchmark
script is a local tool, not part of the package.

**Why the table is exact.** The encoding depends only on day of year, an integer 0–366 at the model
input (`storage.time_axis.compute_doy`; 0 for the zero-filled radar rows of `allow_s2_only`
pixels). The table is built once per device by the same code path and indexed by those integers,
so every value is the one today's per-pixel sin/cos produces, without the two full-size FP32
tensors it allocates on every forward.

## Method

**Inputs.** The Iowa year ingested at 2048² on Frisky version C: mosaics, ROI and checkpoint under
`_frisky_e2e_c/`, in `s3://arbol-tessera-inputs-dev/` (round 1) and copied object for object to
`s3://global-tessera-inputs-dev/` (round 2 onward).

**A change that keeps every name can ship as a code bundle; one that removes names needs a
deployment.** Ray workers pull `s3://{code_bucket}/code/src{code_suffix}.tar.gz` at start-up, so a
worker-side change of values runs from a bundle and the `code_suffix` run parameter. But Ray
pickles the actor class on the driver, which runs the deployed code: a bundle that deletes a name
the driver's copy imports fails at actor creation. Round 2 therefore ran each side from its own
yield-embeddings branch (`dev/global-tessera-frisky-inf` pinned to today's code,
`dev/global-tessera-frisky-inf-simple` to `6d9d5db3`). Switches belong in worker-side code, not in
the runtime environment the driver sets.

**Arm shape and reading.** Arms run concurrently with at least one control, each on its own Ray
cluster, and are cancelled once profiled; staging is kept until the equivalence check has run and
deleted after. Per tile, `CHUNK_SUMMARY` gives total, inference, overhead and prologue seconds and
the strip count; per actor, the `RESOURCES:` lines give host RAM and GPU use; tiles are compared by
label (`temp/frisky-dev/inf_arms.py`). In global-tessera-dev the per-cluster GPU cap in SSM
(`/global-tessera-dev/ray/gpu-worker-ladder`) was removed for this work.

**Cost.** About 16 L40S-hours across both rounds plus head nodes, about 1.5 h of the benchmark
box, and about $2 of requests for the 1.6 TB copy (about $36 a month to keep it).

## Open

- **The remaining ~19 s of GPU-idle overhead per tile**, about 12% of tile time and now the largest
  cost outside the forward pass. The prologue is already hidden (0 s median), so it sits inside the
  strip loop or the tile's wind-down.
- **The table's peak VRAM** (+5.4 GiB at the deepest bucket) before it can ship: it removes two
  full-size FP32 tensors, so the rise is likely how the gather allocates, or how the two backbones'
  streams now overlap.
