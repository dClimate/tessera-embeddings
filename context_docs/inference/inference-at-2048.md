# Inference on 2048-px mosaics: what we tried, what we measured, what shipped

This is the record of the inference experiments run on the `experiment/frisky-inference` branch in
October 2026, after the mosaics moved to 2048-px storage chunks. It is written for someone arriving
fresh: it explains the few terms it needs, then takes each experiment in turn — the question, the
test, the result and the decision. How inference works in general is in
[`../../src/tessera_embeddings/inference/README.md`](../../src/tessera_embeddings/inference/README.md);
the longer GPU record this builds on is [`inference-on-gpus.md`](inference-on-gpus.md).

## In short

Graphics cards are where the money goes: $537K of the last campaign's $828K
([`../campaign/campaign-cost-model.md`](../campaign/campaign-cost-model.md) §12). So the question
was what the new 2048-px mosaics let us change to spend less on them.

- **Shipped on this branch:** a much simpler way of loading each tile (395 fewer lines). It cuts
  the time the card sits idle on edge tiles and lowers peak memory by 2–5 points, and its outputs
  differ from today's only by the rounding-level shimmer the published store already contains.
  A second change runs each tile's row bands densest first; it is bit-identical and removes one
  kind of stall, but nets only about 0.1% of card time.
- **Shipped separately:** looking the positional encoding up in a 367-row day-of-year table makes
  the model's forward pass 5–8% faster with identical outputs. It lives in its own PR off `main`
  (#208), because the model code is shared with `main` and the v2 model.
- **Ruled out:** a cheaper 16 GiB card, larger memory budgets, compiling the model with
  `torch.compile`, and FP8 arithmetic. A PyTorch upgrade is handled in its own PR; on its own it
  is no faster.
- **What is left:** about 2.3% of card time goes to tiles waiting for their first rows to finish
  loading. A smaller first strip was tried and saved nothing overall, so what remains is making the
  read itself faster, which is a question for how the mosaics are stored (PR #210 and its follow-on).

## The few terms this needs

A **tile** is a 2048 × 2048-pixel square of one UTM zone, the unit of work one GPU worker takes at
a time. A tile's satellite data is too big to hold in memory at once, so it is loaded in
**strips**: horizontal bands of rows, each sized to a fixed memory **budget**. The GPU embeds pixels
in **sub-batches** of 7,168. While the GPU works on one strip, the next one loads in the
background (**prefetching**), and on a tile's last strip the next tile's first strip loads, so the
card can move straight on. Whatever time the card spends waiting rather than computing is
**GPU-idle overhead**.

```
 one worker, tile N then tile N+1

 GPU     [== strip 1 of N ==][== strip 2 of N ==][== strip 1 of N+1 ==][== strip 2 ...
 loading     [strip 2 of N ]     [strip 1 of N+1]      [strip 2 of N+1]
              loads while the GPU works on the previous strip, so the GPU never waits
```

**Output equivalence** is judged by ADR 012
([`../decisions/012-validated-equivalence-for-inference-outputs.md`](../decisions/012-validated-equivalence-for-inference-outputs.md)).
Its strict gate says a change must leave stored values within one int8 level; its looser
"cross-config" envelope covers changes that only alter how pixels are grouped on the GPU.

## What 2048-px storage changed on its own

Nothing in the bill. A full year of Iowa on the 2048 mosaics used the same GPU time as on the old
4000-px ones (33.5 against about 34 GPU-hours) with the same memory peak (16.1 of 30.9 GB). The card
was already kept busy, and the memory peak is set by the strip budget, not by the storage chunk.

What changed is the price of a strip. With 4000-px storage, every strip read re-decompressed a
chunk four times its size, about 13 s each, so the old loader went to great lengths to use few,
large strips. With 2048-px storage, a strip costs one background decompression and nothing more.
That is what made the simplification below possible.

## The experiments

### 1. Could a cheaper card do the work?

Production already uses the smallest L40S machine, `g6e.xlarge` (32 GiB of memory). The only
cheaper machines have 16 GiB, and none of them is cheaper per unit of work: the A10G box costs
1.18× as much and the L4 box 1.36× (from the throughput in `inference-on-gpus.md` §4). The test
in round 1 settled it anyway: even with very small strips, a worker peaks at about 12 GB, about 80%
of a 16 GiB machine against a 60% ceiling. **Dropped.**

### 2. Could the strip loader be simplified now that strips are cheap?

The old loader chose between three strategies per tile, added a small "starter" strip, and ran a
separate, capped prefetch of the next tile with its own tiers and an off switch. Round 1 asked
whether smaller strips cost speed; round 2 tested the simplified loader against today's.

**Round 1** (yield account, four runs side by side, 6 workers each, the first 26 tiles):

| Strip budget | Seconds per tile | GPU-idle overhead | Peak memory |
|---|---|---|---|
| 5.75 GiB (today) | 169 | 23.0 s | 53% |
| 5.75 GiB (today, second run) | 166 | 23.1 s | 54% |
| 3.5 GiB | 160 | 20.4 s | 41% |
| 1 GiB | 165 | 16.6 s | 40% |

Speed is flat across budgets, so the smaller budget is free memory headroom; and the two identical
runs show the noise between runs is about 2%.

**The simplified loader:** one budget (3.5 GiB), strips always loaded one ahead, and the same
mechanism carrying on into the next tile — no strategies, no starter strip, no separate
prefetch machinery.

**Round 2** (global-tessera-dev, today's code and the simplified code twice each, 3 workers each,
the 18 tiles all four finished):

| Run | Seconds per tile | GPU-idle overhead | Peak memory |
|---|---|---|---|
| today's code, a | 161 | 23.8 s | 45% |
| today's code, b | 163 | 23.2 s | 45% |
| simplified, a | 164 | 19.2 s | 40% |
| simplified, b | 154 | 19.2 s | 43% |

About 4 s less idle time per tile, roughly 2.5% of tile time on these tiles, and lower memory.
**Shipped** (`6d9d5db3`).

**Its outputs are not bit-identical, and that was accepted.** The model's arithmetic is unchanged —
two runs of the same code match exactly. What moves is which pixels share a 7,168-pixel
sub-batch: new strip boundaries regroup them, the GPU's matrix library picks a different kernel
for a different batch shape, and that kernel adds numbers in a different order, shifting the last
bit of some results. ADR 012 names exactly this mechanism. Against today's code, 99.84–99.99% of
stored values are identical and at least 99.9995% are within one level; the worst value moves 3
levels; the lowest per-pixel cosine similarity is 0.9999 (an angle under 1°) and the mean is
1.000000. That fails the strict gate and passes the cross-config envelope on all 39 tiles checked,
with two measures near its edges (scale drift 1.56% against 1.6%, cosine 0.999901 against 0.9999).

For using the embeddings it makes no difference. Storing them as int8 already moves every vector
about this much (cosine about 0.99998 from the model's own output), and the published store already
mixes tiles made on A10G and L40S cards, which differ by the same kind of shimmer. Searches,
classifications and clusters come out the same, except for pixels sitting exactly on a decision
boundary, which a different card would flip too. And the inference code identity changes, so old
and new tiles cannot mix inside one store.

### 3. Where did the remaining idle time go?

Round 2's simplified runs still idled about 19 s per tile. Rebuilding each tile's timeline from the
worker logs showed this was not spread across tiles: interior tiles idled about 0.2 s, while tiles
on the edge of the Iowa footprint stalled once each. On such a tile the first strip can be entirely
outside the footprint. The loader skips it immediately and then waits, with the GPU idle, for the
next strip — whose load had only just started, because the empty strip gave it no work to hide
behind.

The fix: run each tile's strips **densest first, empty strips last**, and start the next tile's
prefetch from the last strip that has pixels. Order does not change which pixels share a
sub-batch, so the outputs should be identical. Each tile's summary now records how many of its
strips hold pixels (`live_strips`), which is also what let round 3 sort tiles into classes.

**Round 3** (global-tessera-dev, the simplified code with and without the new order, the whole
Iowa year — 394 tiles each — on up to 8 workers each, every tile paired across the two runs):

| Tile class | Tiles | Share of card time | Change in tile time | Median idle, before → after |
|---|---|---|---|---|
| interior (≥ 99% valid, every strip has pixels) | 305 | 87% | −0.7% | 0.3 → 0.3 s |
| edge | 89 | 13% | −2.7% | 6.2 → 0.3 s |
| all | 394 | | −1.0% | |

Interior tiles run the identical code path, so their −0.7% is run-to-run noise, and the edge figure
has to be read against it. The effect that is attributable to the change is idle time: on the 24
edge tiles that had an empty strip, it fell 43% (284 → 162 s). Across all of Iowa that is about 0.1%
of card time. Outputs are **bit-identical**: 229 of the 394 tiles were compared, every one exactly
equal, including all 24 whose strip order changed. **Kept**: it is bit-identical, small, and removes
the worst stall on footprint edges; its gain is small because that stall is rare.

**What the remaining idle time is.** About 2.3% of card time, over roughly a third of the tiles,
is a tile starting before its first strip has finished loading: the previous tile's last stretch
of work was shorter than the 20–25 s that load takes. (Running densest first makes the last strip
the sparsest, which occasionally shortens that window.) Starting the load one strip earlier would
fix it but hold one more strip in memory, and the full-year runs already peak at 51–53%: another
strip adds about 12 points, past the 60% ceiling. The other remaining cost is each worker's very
first tile, which has nothing before it to prefetch behind: about 0.3%.

### 4. Could a tile's first strip load faster?

On a dense tile the first load takes 20–25 s: about 3 s for the tile's cloud masks, 13 s to read and
decompress the ten bands, 3 s to build the dataset and 4 s for the rest. The band read dominates,
and most of its cost does not depend on how many rows the strip asks for: every read decompresses
each whole storage chunk it touches. On a dense tile, reading a quarter of the rows took 2.4 s
against 3.1 s for the whole height.

**Round 4** (global-tessera-dev, the round 3 code with and without a small first "starter" strip on
tiles dense enough to want one, the whole Iowa year, every tile paired; 362 of the 394 tiles got a
starter):

| | Without starter | With starter |
|---|---|---|
| GPU idle, share of card time | 2.80% | 1.98% |
| Total tile time, 361 tiles | 80,545 s | 80,363 s (−0.2%) |

The idle time fell, but total tile time did not move: the extra strip each tile carried (one more
read, one more partly filled sub-batch) cost about what the idle it removed saved. No gain worth the
added loading logic. **Reverted** (`21bcc8dd`).

One of the starter arm's eight cards overheated for much of the run: its thermal throttle was active
in 43% of its samples, its clock fell as low as 525 MHz, and its tiles ran about 15% slower. Its 33
tiles are left out of the figures above. With them in, the starter arm looks 1.6% slower.

**One setting removed.** The loader limited its own band-reading threads to leave CPU cores for
the GPU feed (`reserve_cpus`). But zarr decompresses on its own thread pool, across every core,
whatever the loader's thread count, so the setting reserved nothing, and the GPU was not short of
CPU anyway. It was removed (`d0336590`); the loader uses two band-reading threads.

**The read can only get faster in storage.** Two changes off `main` target it: Blosc-LZ4 compression,
which decompresses 2.6× faster than today's zstd (PR #210), and sharding the mosaics so that a strip
decompresses only its own rows rather than whole 4096-px chunks (stacked on #210).

### 5. Could the model's forward pass itself be cheaper?

The forward pass is about 90% of a tile's time, so it is the biggest target. Its parts were timed
on one L40S with the production model and checkpoint, batch 7,168, at five observation depths:

| Change | Forward time | Outputs | Decision |
|---|---|---|---|
| positional encoding as a 367-row day-of-year table | **5–8% faster** | identical | **PR #208, off `main`** |
| `torch.compile` of the embedding and encoding | 6–8% faster | int8 93–96% exact | not pursued: the table gets the same gain exactly |
| `torch.compile` of the transformer layers | 3–5% slower | identical | not pursued: the layers run as one fused operation the compiler cannot open |
| PyTorch 2.14.1 instead of 2.5.1 | equal, 3–5% slower at the deepest | not compared | separate PR off `main`; it also uses 15% more GPU memory at the deepest depth |
| FP8 arithmetic | not measured | would change outputs | denied |

The positional encoding turns each observation's day of year into a vector. Day of year is a whole
number from 0 to 366, but the encoding used to be recomputed with sine and cosine for every pixel
and observation on every forward pass. Computing all 367 days once and looking them up gives the
same values and skips two large temporary tensors per pass. Because the model code is shared with
`main` and the v2 model, that change went into its own PR rather than this branch.

### 6. Smaller decisions

- **The easting crop stays.** Edge tiles read only the columns that hold pixels. Removing the crop
  would save code but would change `eligible_px`, a published registry field, so it was held.
- **Larger strip budgets are moot.** Round 1 showed speed does not depend on the budget, so the
  smaller one wins on memory.

## How the tests were run

**Inputs.** The Iowa year ingested at 2048 px under `_frisky_e2e_c/` (mosaics, ROI and checkpoint),
first in `s3://arbol-tessera-inputs-dev/` (round 1, yield account) and then copied object for
object to `s3://global-tessera-inputs-dev/` (rounds 2 to 4, global-tessera-dev).

**Side by side, with a control.** Every round ran its variants at the same time, each on its own Ray
cluster of `g6e.xlarge` workers, so S3 and capacity conditions were shared. Rounds 1 and 2 were
stopped after 25–35 minutes of inference; rounds 3 and 4 ran to completion so that every tile of
Iowa could be paired. Each tile's `CHUNK_SUMMARY` log line gives its total, inference, idle and prologue
seconds and its strip counts; the workers' `RESOURCES` lines give memory. Tiles are compared by
label, and costs as sums of per-tile seconds, which do not depend on how many workers each run got.
Outputs are compared with `te-compare-outputs` on a machine in the same account.

**Cards differ, and one card can sink an arm.** Each run gets its own cards. Under the same 350 W
power cap, a card's clock falls about 11 MHz for every degree it runs hotter, and the occasional card
overheats and throttles itself. Every comparison here was checked for that from the workers'
`RESOURCES` lines (clock, temperature and throttle reasons, every 30 s). Round 4 is the one affected.
Round 3's densest-first arm happened to run 2% faster clocks on average, which matches its −0.7% on
interior tiles whose code did not change.

**A code bundle is not always enough.** Ray workers can load a different version of the code from
a bundle in S3, which suits changes to values. But Ray builds the worker class on the driver, which
runs the deployed code, so a change that removes or renames anything the driver's copy refers to
fails at start-up. Rounds 2 to 4 therefore ran each side from its own yield-embeddings branch
(`dev/global-tessera-frisky-inf`, `-simple`, `-order` and `-starter`). For the same reason, round
1's attempt to switch the old prefetch off through an environment variable did nothing: the driver
sets that environment.

**Capacity.** L40S machines were scarce in us-west-2 for much of the work and arrived one at a time.
In global-tessera-dev a per-cluster cap on GPU workers (`/global-tessera-dev/ray/gpu-worker-ladder`
in SSM) was removed for these tests.

**Cost.** About 120 L40S-hours across the four rounds and the benchmarks (rounds 3 and 4, two full
runs each, about 50 apiece), plus head nodes and small comparison machines; about $2 to copy the 1.6 TB of
inputs and about $36 a month to keep them.

## What is left

- **The first-strip wait** (about 2.3% of card time): a faster read, from the mosaic storage
  changes above, rather than more memory or a smaller first strip.
- **Merging.** Any change to the model or loader code changes the inference code identity, which
  stops staged tiles being reused and puts later appends under a new identity. This branch, the
  table (#208), the PyTorch upgrade and the mosaic storage PRs are best merged together so that
  happens once.
