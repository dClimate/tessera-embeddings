# Simplifying the strip loader: what was tried, measured and shipped

This is the record of the October 2026 experiments on how inference loads each tile for the GPU.
They began on an experimental branch whose mosaics were stored in 2048-px chunks (#205 and #207,
neither merged) and finished on the sharded 4096-px mosaics that replaced that approach (#211),
which is where the simpler loader ships. It is written for someone arriving fresh: it explains the
few terms it needs, then takes each experiment in turn — the question, the test, the result and
the decision. How inference works in general is in
[`../../src/tessera_embeddings/inference/README.md`](../../src/tessera_embeddings/inference/README.md);
the longer GPU record this builds on is [`inference-on-gpus.md`](inference-on-gpus.md).

## In short

Graphics cards are where the money goes: $537K of the last campaign's $828K
([`../campaign/campaign-cost-model.md`](../campaign/campaign-cost-model.md) §12). So the question
was what cheaper reads let us change in how a tile reaches the card.

- **Shipped:** a much simpler loader. Strips are sized to one memory budget, the next strip always
  loads while the card works, that pipeline carries straight on into the next tile, strips run
  densest first, and nothing reserves CPU cores. On sharded mosaics it peaks at 50% of the
  worker's memory against 56% for the loader it replaces, at the same speed. Its outputs differ
  from today's only at the rounding level and keep 99.98% of each pixel's nearest neighbours,
  which was accepted (section 2). They no longer depend on which tile a worker handled before,
  so two runs of the same code on the same kind of card give identical outputs.
- **Shipped separately:** looking the positional encoding up in a day-of-year table makes the
  forward pass 5–8% faster with identical outputs (#208, merged). A PyTorch upgrade has its own PR
  (#209); on its own it is no faster.
- **Ruled out:** a cheaper 16 GiB card, larger memory budgets, compiling the model with
  `torch.compile`, FP8 arithmetic, and a small "starter" first strip.
- **What is left:** about 2.6% of tile time is the card waiting for a tile's first strip.

## The few terms this needs

A **tile** is a 2048 × 2048-pixel square of one UTM zone, the unit of work one GPU worker takes at
a time; the mosaics store each band and date in 4096-px blocks, four tiles to a block. A tile's
satellite data is too big to hold in memory at once, so it is loaded in **strips**: horizontal
bands of rows, each sized to a fixed memory **budget**. The GPU embeds pixels in **sub-batches** of
7,168. While the GPU works on one strip, the next one loads in the background (**prefetching**),
and on a tile's last strip the next tile's first strip loads, so the card can move straight on.
Whatever time the card spends waiting rather than computing is **GPU-idle overhead**.

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

## Why the loader could be simplified

The old loader was built for storage in which reading any rows of a block decompressed the whole
block: about 13 s per strip with 4000-px blocks. So it went to great lengths to use few, large
strips. It chose between three strategies per tile, added a small "starter" strip, priced each
read against an estimate of inference speed, and ran a separate, capped prefetch of the next tile
with its own tiers and an off switch.

Any storage in which a strip costs about its own rows removes the reason for all of that. Two
did. The experimental branch stored mosaics in 2048-px chunks, one tile to a chunk; rounds 1 to 4
below ran on those. Sharding (#211) keeps the 4096-px block as the stored object but splits it
into 512-row × 2048-px inner chunks, and a read decompresses only the inner chunks it touches
([`../ingest/campaign-ingest-measurements.md`](../ingest/campaign-ingest-measurements.md) §3.19).
Round 5 ran on those, and that is the layout the simpler loader ships with.

On its own, the 2048-px storage changed nothing in the bill. A full year of Iowa used the same GPU
time as on 4000-px blocks (33.5 against about 34 GPU-hours) with the same memory peak (16.1 of
30.9 GB): the card was already kept busy, and the memory peak is set by the strip budget, not by
the storage chunk. What changed was the price of a strip.

## The experiments

### 1. Could a cheaper card do the work?

Production already uses the smallest L40S machine, `g6e.xlarge` (32 GiB of memory). The only
cheaper machines have 16 GiB, and none of them is cheaper per unit of work: the A10G box costs
1.18× as much and the L4 box 1.36× (from the throughput in `inference-on-gpus.md` §4). The test
in round 1 settled it anyway: even with very small strips, a worker peaks at about 12 GB, about 80%
of a 16 GiB machine against a 60% ceiling. **Dropped.**

### 2. Could the strip loader be simplified now that strips are cheap?

Round 1 asked whether smaller strips cost speed; round 2 tested the simplified loader against
the old one. Both ran on 2048-px mosaics.

**Round 1** (yield account, four runs side by side, 6 workers each, the first 26 tiles):

| Strip budget | Seconds per tile | GPU-idle overhead | Peak memory |
|---|---|---|---|
| 5.75 GiB (old) | 169 | 23.0 s | 53% |
| 5.75 GiB (old, second run) | 166 | 23.1 s | 54% |
| 3.5 GiB | 160 | 20.4 s | 41% |
| 1 GiB | 165 | 16.6 s | 40% |

Speed is flat across budgets, so the smaller budget is free memory headroom; and the two identical
runs show the noise between runs is about 2%.

**The simplified loader:** one budget (3.5 GiB), strips always loaded one ahead, and the same
mechanism carrying on into the next tile — no strategies, no starter strip, no speed estimate, no
separate prefetch machinery.

**Round 2** (global-tessera-dev, the old code and the simplified code twice each, 3 workers each,
the 18 tiles all four finished):

| Run | Seconds per tile | GPU-idle overhead | Peak memory |
|---|---|---|---|
| old code, a | 161 | 23.8 s | 45% |
| old code, b | 163 | 23.2 s | 45% |
| simplified, a | 164 | 19.2 s | 40% |
| simplified, b | 154 | 19.2 s | 43% |

About 4 s less idle time per tile, roughly 2.5% of tile time on these tiles, and lower memory.
**Kept.**

**Its outputs are not bit-identical.** The model's arithmetic is unchanged — two runs of the same
code match exactly. What moves is which pixels share a 7,168-pixel sub-batch: new strip boundaries
regroup them, the GPU's matrix library picks a different kernel for a different batch shape, and
that kernel adds numbers in a different order, shifting the last bit of some results. ADR 012
names exactly this mechanism. Against the old loader, 99.84–99.99% of stored values are identical
and at least 99.9995% are within one level; the worst value moves 3 levels; the lowest per-pixel
cosine similarity is 0.9999 (an angle under 1°) and the mean is 1.000000. That fails the strict
gate and passes ADR 012's cross-config envelope on every tile. Over the whole Iowa year (round 5)
the worst tiles reach a scale drift of 1.97%, a worst-pixel cosine of 0.999877 and one value moving
4 levels. Those extremes, with the same pattern in the v2 full-stack run, are what ADR 012's envelope
was widened on: its first bounds (1.6%, 0.9999, 3 levels), set on three chunks of a batch-size
change, failed 47 of these 394 tiles on extremes alone. The typical tile is closer than that
batch-size change: a median 99.95% of values identical, against 95–98%.

**The old loader's outputs were not stable either.** It gave a tile a starter strip depending on
which tile its worker had handled before, so the same tile could be cut into different strips on
two runs of the same code, and those tiles differed by up to 3 levels (4 of 394 Iowa tiles in the
sharding test, §3.19). The simpler loader plans a tile from its own mask alone, whether it was
prefetched or loaded serially, so the same code always cuts it the same way.

**What a user consumes barely moves.** Measured on all 394 round-5 tile pairs, by the method of
[`validating-a-model-change.md`](validating-a-model-change.md): per tile, 2,000 query pixels among
20,000 candidates by cosine, and k-means with k = 12.

| measure | old loader against the simpler one | for scale |
|---|---|---|
| pixels whose stored embedding is identical | 99.0% (94.5% on the worst tile) | — |
| mean per-pixel cosine distance | 1.4 × 10⁻⁷ | adjacent pixels: 5.8 × 10⁻³ (median), about 34,000 times larger |
| top-20 neighbours kept | 99.98% (99.90% on the worst tile) | two same-model stores agree on 0.9940; the PyTorch upgrade kept 99.48% (#209) |
| same k-means cluster | 99.999% | — |

The 47 tiles that the envelope's first bounds failed look like the rest: 99.97% of top-20
neighbours kept, 99.999% in the same cluster. The inference code identity changes too, so old and
new tiles cannot mix inside one store. On that evidence the maintainers accepted the change on
2026-10-05.

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

**Round 3** (global-tessera-dev, 2048-px mosaics, the simplified code with and without the new
order, the whole Iowa year — 394 tiles each — on up to 8 workers each, every tile paired across the
two runs):

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
of work was shorter than the 20–25 s that load took. (Running densest first makes the last strip
the sparsest, which occasionally shortens that window.) Starting the load one strip earlier would
fix it but hold one more strip in memory, and the full-year runs already peak at 50–53%: another
strip adds about 12 points, past the 60% ceiling. The other remaining cost is each worker's very
first tile, which has nothing before it to prefetch behind: about 0.3%.

### 4. Could a tile's first strip load faster?

On a dense 2048-px tile the first load took 20–25 s: about 3 s for the tile's cloud masks, 13 s to
read and decompress the ten bands, 3 s to build the dataset and 4 s for the rest. The band read
dominated, and most of its cost did not depend on how many rows the strip asked for: every read
decompressed each whole storage chunk it touched. Reading a quarter of the rows took 2.4 s against
3.1 s for the whole height.

**Round 4** (global-tessera-dev, 2048-px mosaics, the round 3 code with and without a small first
"starter" strip on tiles dense enough to want one, the whole Iowa year, every tile paired; 362 of
the 394 tiles got a starter):

| | Without starter | With starter |
|---|---|---|
| GPU idle, share of card time | 2.80% | 1.98% |
| Total tile time, 361 tiles | 80,545 s | 80,363 s (−0.2%) |

The idle time fell, but total tile time did not move: the extra strip each tile carried (one more
read, one more partly filled sub-batch) cost about what the idle it removed saved. No gain worth the
added loading logic. **Reverted.**

One of the starter arm's eight cards overheated for much of the run: its thermal throttle was active
in 43% of its samples, its clock fell as low as 525 MHz, and its tiles ran about 15% slower. Its 33
tiles are left out of the figures above. With them in, the starter arm looks 1.6% slower.

**One setting removed.** The loader limited its own band-reading threads to leave CPU cores for
the GPU feed (`reserve_cpus`). But zarr decompresses on its own thread pool, across every core,
whatever the loader's thread count, so the setting reserved nothing, and the GPU was not short of
CPU anyway. It was removed; the loader uses two band-reading threads.

**The read could only get faster in storage**, which is what #211 did: Blosc-LZ4 compression,
which decompresses 2.6× faster than the zstd it replaced, and sharding, so that a strip decompresses
little more than its own rows. With the old loader, sharding alone halved GPU idle on Iowa (5.3% to
2.7% of tile time) and cut tile time 3.0% (§3.19).

### 5. The simpler loader on sharded mosaics

**Round 5** ran the simpler loader against the old one, both on sharded mosaics: the whole Iowa
year, 8 L40S each, every tile paired. One of the old loader's cards overheated, so its 42 tiles are
left out:

| | old loader | simpler loader |
|---|---|---|
| Total tile time, 352 tiles | 80,215 s | 78,898 s (−1.6%) |
| GPU idle, share of tile time | 2.5% | 2.6% |
| Peak host memory per worker | 17.2 GB (56%) | 15.4 GB (50%) |

On sharded mosaics both loaders hide the reads equally well. The 1.6% is all in inference time, and
about a third of it is the simpler arm's cards running 2.5% faster clocks; the rest is within the
spread between runs. What the simpler loader brings is memory headroom, reproducible outputs and
far less code. Its outputs against the old loader's are in section 2.

Round 5's mosaics used 256-row inner chunks. #211 then settled on 512 rows, which read a tile at
least as fast with about the same memory and fewer S3 requests (§3.19); the values are identical
either way, so nothing here depends on the choice. The planner does not round strips to 512-row
boundaries: a strip that straddles one decompresses one extra inner chunk in the background, which
costs CPU but not read time.

### 6. Could the model's forward pass itself be cheaper?

The forward pass is about 90% of a tile's time, so it is the biggest target. Its parts were timed
on one L40S with the production model and checkpoint, batch 7,168, at five observation depths:

| Change | Forward time | Outputs | Decision |
|---|---|---|---|
| positional encoding as a 367-row day-of-year table | **5–8% faster** | identical | **#208, merged** |
| `torch.compile` of the embedding and encoding | 6–8% faster | int8 93–96% exact | not pursued: the table gets the same gain exactly |
| `torch.compile` of the transformer layers | 3–5% slower | identical | not pursued: the layers run as one fused operation the compiler cannot open |
| PyTorch 2.14.1 instead of 2.5.1 | equal, 3–5% slower at the deepest | not compared | its own PR (#209); it also uses 15% more GPU memory at the deepest depth |
| FP8 arithmetic | not measured | would change outputs | denied |

The positional encoding turns each observation's day of year into a vector. Day of year is a whole
number from 0 to 366, but the encoding used to be recomputed with sine and cosine for every pixel
and observation on every forward pass. Computing all 367 days once and looking them up gives the
same values and skips two large temporary tensors per pass.

### 7. Smaller decisions

- **The easting crop stays.** Edge tiles read only the columns that hold pixels. Removing the crop
  would save code but would change `eligible_px`, a published registry field, so it was held.
- **Larger strip budgets are moot.** Round 1 showed speed does not depend on the budget, so the
  smaller one wins on memory.
- **The inference-speed estimate is gone.** Only the old planner's read pricing used it
  (`MODEL_EST_PX_PER_SEC`), and varying it had already measured as changing nothing on Iowa
  (`inference-on-gpus.md`, the v2 section).

## How the tests were run

**Inputs.** Rounds 1 to 4: the Iowa year ingested at 2048 px under `_frisky_e2e_c/` (mosaics, ROI
and checkpoint), first in `s3://arbol-tessera-inputs-dev/` (round 1, yield account) and then copied
object for object to `s3://global-tessera-inputs-dev/` (rounds 2 to 4, global-tessera-dev). Round
5: the same year ingested sharded under `s3://global-tessera-inputs-dev/_shard_test/shard/`.

**Side by side, with a control.** Every round ran its variants at the same time, each on its own Ray
cluster of `g6e.xlarge` workers, so S3 and capacity conditions were shared. Rounds 1 and 2 were
stopped after 25–35 minutes of inference; rounds 3 to 5 ran to completion so that every tile of
Iowa could be paired. Each tile's `CHUNK_SUMMARY` log line gives its total, inference, idle and
prologue seconds and its strip counts; the workers' `RESOURCES` lines give memory. Tiles are
compared by label, and costs as sums of per-tile seconds, which do not depend on how many workers
each run got. Outputs are compared with `te-compare-outputs` on a machine in the same account, and
usability with the neighbour and cluster method of [`validating-a-model-change.md`](validating-a-model-change.md).

**Cards differ, and one card can sink an arm.** Each run gets its own cards. Under the same 350 W
power cap, a card's clock falls about 11 MHz for every degree it runs hotter, and the occasional
card overheats and throttles itself. Every comparison here was checked for that from the workers'
`RESOURCES` lines (clock, temperature and throttle reasons, every 30 s). Rounds 4 and 5 were each
affected once. Round 3's densest-first arm happened to run 2% faster clocks on average, which
matches its −0.7% on interior tiles whose code did not change.

**A code bundle is not always enough.** Ray workers can load a different version of the code from
a bundle in S3, which suits changes to values. But Ray builds the worker class on the driver, which
runs the deployed code, so a change that removes or renames anything the driver's copy refers to
fails at start-up. Rounds 2 to 5 therefore ran each side from its own yield-embeddings branch
(`dev/global-tessera-frisky-inf`, `-simple`, `-order`, `-starter` and `-shard-simple`). For the same
reason, round 1's attempt to switch the old prefetch off through an environment variable did
nothing: the driver sets that environment.

**Capacity.** L40S machines were scarce in us-west-2 for much of the work and arrived one at a time.
In global-tessera-dev a per-cluster cap on GPU workers (`/global-tessera-dev/ray/gpu-worker-ladder`
in SSM) was removed for these tests.

**Cost.** About 170 L40S-hours across the five rounds and the benchmarks (rounds 3 to 5, two full
runs each, about 50 apiece), plus head nodes and small comparison machines; about $2 to copy the
1.6 TB of 2048-px inputs.

## What is left

- **The first-strip wait** (about 2.6% of tile time on sharded mosaics). Prefetching one strip
  earlier would close it but breaks the 60% memory ceiling; a smaller first strip saved nothing.
- **Code identity.** Any change to the loader changes the inference code identity, which stops
  staged tiles being reused and puts later appends under a new identity. The loader, sharding and
  the other inference PRs are best merged together, at a store boundary, so that happens once.
