# v2 in small chunks replaying a CUDA graph: what it buys, per card

Measured on 2026-11-13 in global-tessera-dev, to decide whether ideas from Mark Elvers's C++/CUDA encoder
([`mtelvers/tessera-inference`](https://github.com/mtelvers/tessera-inference)) make v2 Large inference
cheaper on our fleet. Outcome: his encoder is level with our PyTorch forward on an L40S, so it is not
adopted; one idea from it, run inside PyTorch, takes about 9% off inference time on an L40S and is what
PR #273 ships. The idea is his.

## 1. What changed

A v2 sub-batch is 7,168 pixels of one `(T_s2, T_s1)` bucket shape. The forward used to run it as one
batch. It now runs it in chunks of a few dozen pixels. The first time a bucket shape is seen, one chunk's
forward is captured as a `torch.cuda.CUDAGraph`; every later chunk of that shape copies its rows into the
graph's input buffers and replays it. Small chunks keep each layer's activations inside the GPU's cache
instead of streaming them through memory; the graph removes the per-chunk launch cost that would
otherwise eat the gain. All graphs of a model share one memory pool.

The chunk size depends on the card (`GRAPH_CHUNK_PIXELS` in `inference/models/ssl_model.py`): 64 pixels
on an L40S, 32 on an L4, and the whole batch on any other card.

## 2. Method

One `g6e.xlarge` (L40S) per test, on the production worker image (PyTorch 2.14.1+cu130). Two kinds of
test:

- **Synthetic forwards.** A 7,168-pixel batch of one bucket shape, timed with CUDA events. An L40S at
  its 350 W limit lowers its clock as it heats, and a first sweep that ran arms one after another
  differed by 8% between two passes of the same arm. Every figure below comes from a card pre-heated for
  90 seconds, four rounds per shape, every round timing every arm, the order rotating each round.
- **Real tiles.** One long-lived `InferenceActor`, driven as Ray drives it, over six tiles of
  `mosaics/iowa_epsg5070` (calendar 2024, ascending radar), each arm run twice in mirrored order (A, B,
  B, A). Inference time is the actor's own `CHUNK_SUMMARY`. Fresh-process runs of three tiles differ by
  4% between two identical arms, which is why the long-lived actor is used.

Harness and raw results: `s3://global-tessera-inputs-dev/perf/tc/`.

## 3. The C++/CUDA encoder on an L40S

`src/tessera_cuda.cu` at `54fa20a`, built unmodified and called through `ctypes`, weights exported from
our checkpoint. Throughput relative to our whole-batch forward, with the lowest and highest single round:

| shape (S2, S1 steps) | his encoder, 64-pixel chunks | his encoder, 128-pixel chunks | ours, 64-pixel chunks with a graph |
|---|---:|---:|---:|
| 48, 16 | 0.88× [0.88–0.89] | 1.04× [1.04–1.05] | 1.15× [1.15–1.16] |
| 96, 1 | 0.98× [0.96–1.00] | 0.99× [0.97–1.00] | 1.16× [1.13–1.18] |
| 104, 40 | 1.06× [1.02–1.07] | 1.04× [1.00–1.05] | 1.19× [1.17–1.21] |
| 160, 64 | 1.03× [1.01–1.04] | 1.00× [0.99–1.01] | 1.08× [1.07–1.10] |
| 208, 80 | 0.98× [0.97–1.00] | 0.95× [0.94–0.96] | 1.04× [1.00–1.07] |
| 256, 104 | 1.00× [0.98–1.02] | 0.96× [0.93–0.97] | 1.08× [1.06–1.10] |

On three real tiles substituted behind the same call it took 367.6 s of inference against 338.7 s and
353.6 s for two runs of ours. Its output agrees with ours on 85–91% of int8 values, at most 3 levels
apart. The 10% it gains on an L4 in its author's measurements does not carry to an L40S; why was not
diagnosed. A second forward implementation and a compile step in the image build buy nothing here.

## 4. Chunked replay on real tiles, L40S

Six Iowa tiles, long-lived actor:

| | whole batch | 64-pixel chunks | change |
|---|---:|---:|---:|
| inference time | 1,479.0 s | 1,351.6 s | −8.6% |
| tile total | | | −6.9% |
| GPU memory in use | 13.3 GB | 0.13 GB | |

What the card's counters show (DCGM, 1 s samples): memory-bandwidth use falls from 0.65 to 0.40 and
tensor-core use rises from 0.55 to 0.67, which is the cache effect; the share of time a kernel is running
falls from 0.97 to 0.92, the cost of many small launches.

The gain is smaller on real tiles than on synthetic forwards (4–19%) because the forward is only part of
a sub-batch's time; batch preparation and quantisation are untouched.

## 5. Which chunk size

**L40S.** On synthetic shapes the best size moves with sequence length: 64 pixels up to about 150 tokens
per pixel, 16 between 200 and 240, 32 from 288 up, and irregular between. Real buckets sit below 200
tokens per pixel (the published store averages 115), so the deep end does not decide it. On seven bucket
shapes taken from Iowa, sizes 32 to 80 all gain 10–15%, and 80 leads by spikes on two shapes. On six real
tiles, each size run twice:

| chunk size | inference time, two passes | 
|---|---:|
| 64 pixels | 671.5 s, 657.8 s |
| 80 pixels | 651.9 s, 658.8 s |
| 48 pixels | 657.3 s, 653.6 s |

The sizes are 1.4% apart and the two 64-pixel passes are 2.1% apart. 32 against 64 was 0.7% apart in a
separate run. 64 stays: no size beats it on real tiles, and 80 moves outputs further (§6).

**L4.** 32 pixels is best, 7–14% faster than the whole batch.

**A10G.** No size from 32 to 1,792 pixels beats the whole batch, so it keeps the whole batch.

## 6. Outputs

Against the whole-batch forward on the same card, 64-pixel chunks:

| | Iowa tiles, long-lived actor | `m10_parity_15S`, dev pipeline |
|---|---:|---:|
| int8 values exactly equal | 93.8–95.0% | 93.76% |
| within one level | ≥ 99.97% | 99.98% |
| largest difference | 3 levels | 2 levels |
| lowest per-pixel cosine | 0.99987 | 0.99993 |
| largest scale drift | 1.55% | |

That passes ADR 012's cross-config envelope. Usability, with the script and sampling of
[ADR 030](../decisions/030-forward-pass-reorderings-ship-at-a-store-boundary.md), pooled over four Iowa
tiles: top-20 neighbours kept 0.974, same k-means cluster 99.76%, mean per-pixel cosine distance 2.1e-5.
ADR 030's bar, two cards on identical code, is 0.981, 99.85% and 1.1e-5, so the change moves outputs
about twice as far as a card change. It ships on a ruling,
[ADR 042](../decisions/042-chunked-graph-replay-ships-above-the-two-card-yardstick.md).

On one tile, 48-pixel chunks match 64 on every usability measure; 80-pixel chunks are further out
(neighbours 0.968, mean distance 3e-5).

The chunked forward is bit-identical to itself: across passes of one actor, and across two fresh
processes running PR #273's code on two tiles.

## 7. A year of Iowa, both arms side by side

PR #273's code against `main` (`db60e3e8`) through the dev pipeline: `iowa_epsg5070`, calendar 2024,
ascending radar, inference only, eight L40S per arm, both arms started in the same minute. 392 tiles
finished in both arms and are compared as pairs, from each tile's `CHUNK_SUMMARY`:

| | `main`, whole batch | PR #273, 64-pixel chunks | change |
|---|---:|---:|---:|
| inference time, summed over tiles | 50,644 s | 45,281 s | −10.6% |
| tile time, summed over tiles | 58,220 s | 53,988 s | −7.3% |
| time before a tile's first strip | 7,027 s | 7,624 s | +8.5% |
| GPU instance-hours billed | 16.58 h | 15.38 h | −7.2% |
| median per-tile ratio, inference | | 0.893 | |
| median per-tile ratio, tile time | | 0.918 | |

The chunked arm's cards averaged 1,578 MHz while inferring and the control's 1,646 MHz. A chunked
forward draws more power per unit of time, so a card at its 350 W limit settles at a lower clock; the
single-box counters in §4 show the same 5%. The gain above is net of it. No card in either arm was
thermally throttled. The time before a tile's first strip is reads from S3, which the change does not
touch; the two arms read the same mosaics at the same time, and the 8.5% is unexplained.

**Outputs.** Twelve tiles spread across the state, each arm on whichever card ran it, compared with
`te-compare-outputs --cross-config`: all twelve pass, with 93.8–94.8% of int8 values equal, at least
99.956% within one level, a largest difference of 3 levels, a lowest per-pixel cosine of 0.99986 and a
largest scale drift of 1.54%. The remaining tiles were not compared.

**What it is worth.** The v2 campaign estimate
([`../campaign/campaign-cost-model.md`](../campaign/campaign-cost-model.md) §13) has about $363,000 and
244,000 hours of graphics cards. 7.3% of that is about $26,000 and 18,000 card-hours over the nine
years, taking the campaign from about $560,000 to about $534,000. That assumes every hour is an L40S; an A10G
gains nothing (§5), so the saving shrinks by the share of hours that fall back to A10Gs.

## 8. Measured and not adopted

| idea | result |
|---|---|
| glibc allocator settings from the blog post (`MALLOC_MMAP_MAX_=0`, large trim threshold and top pad) | Host RAM rose to 29.4 of 30.9 GB and the box hung. |
| 16 chunks per graph replay, to cut launches | 0.5% apart from one chunk per replay. |
| Preparing the Sentinel-2 batch on the GPU | Inputs bit-identical; inference 1.3% slower. |
| A chunk size per sequence-length band | Real buckets sit below the lengths where the bands differ. |

Not tested, by ruling: a second actor per card and higher precision as uses of the freed GPU memory.
`batch_size_for_gpu` still sizes the whole-batch forward that v1.1 and the A10G run.
