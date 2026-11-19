# Plan: finishing FP8 on the L40S

*Written 2026-11-19 for review. A plan, not a record: when the work is done its results go into
`reduced-precision-inference.md` and ADR 043 (both on PR #275), and this file is deleted.*

FP8 here means what ADR 043 calls Stage 1: on an L40S, the v2 encoder's QKV and first feed-forward
matrix multiplications take their input and weights as 8-bit floats (E4M3), with one scale per token
row and one per weight output channel. Everything else stays bf16. v1.1 is not touched.

## 1. Where things stand

Checked on 2026-11-19 against PR #275's branch, `main` and the run outputs in S3.

| piece | state |
|---|---|
| ADR 043 and the reduced-precision record | written, on PR #275, status Proposed |
| The gate as code (`profiling/inference/precision_gate.py`, `scripts/precision_gate/`) | written and unit-tested, on PR #275 |
| The FP8 forward pass in `src/` | **not written**. FP8 exists only as a stand-in inside `scripts/precision_gate/embed_lattice.py`, which swaps the layer function on a production model |
| FP8 timed at today's chunk sizes | **never measured** |
| FP8 inside a captured CUDA graph | **never run** |
| The gate re-run on today's forward (`rebase-20261118`) | A10G and L4 embedded; **the L40S box never started**, so nothing was scored |
| Evaluation baseline (`evaluation/v2-large.icechunk`) | filled 2026-11-16 on `main` at `21c132b6`, which is still `main` |

Three findings shape the plan.

1. **ADR 043's numbers are out of date.** Its tables came from a classifier that stopped fitting early
   (commit `af33d0a3`). Re-scored with the classifier fitted properly (`rebase-20261118/record/`), the
   same embeddings give a different list of losses:

   | tile | ADR 043 says, Stage 1 | re-scored, Stage 1 |
   |---|---|---|
   | Kansas | fails: winter wheat recall 0.66 | no failure. Unresolved: developed, kNN precision 0.51 ± 0.17. Sorghum still trips |
   | Ecuador | unresolved: wetland recall 0.79 | **fails**: wetland recall 0.84 ± 0.18 |
   | Finland | passes | unresolved: grass recall 0.52 ± 0.09 |
   | UK, Iowa, Punjab | pass | pass |

   All 12 two-card pairs and 239 of 240 noise draws still pass. The committed
   `fp8_exception_bounds.json` was generated from the old reports, so it puts Stage 1 outside its own
   exception on Kansas and Finland. ADR 043's decision 2, its tables, the bounds file and the PR
   description all have to be restated whatever else happens (§5.1).
2. **The speed case is unproven at the shapes production now runs.** The 1.64× per layer was measured
   on batches of 896,000 token rows. Since PR #273 an L40S runs 64-pixel chunks, about 7,400 rows, and
   chunking already took part of the gain FP8 was expected to give.
3. **The evaluation baseline is usable as it is.** It was filled by the code `main` still runs.

## 2. Order of work

Each step ends at a point where the work can stop with a result worth recording.

| step | what | stops the work if |
|---|---|---|
| 0 | **No new production code.** Run the existing harness on one L40S (§4.1): can the FP8 multiplication be captured in a graph, what share of the forward the two multiplications are at 64-pixel chunks, and the L40S lattice embeddings the re-run never got. Score them | FP8 cannot be captured, or the two multiplications are too small a share to reach the bar |
| 1 | Write the forward and the store's precision record (§3), with CPU tests | |
| 2 | On one L40S: GPU checks, timing, and the choices §3.3 leaves open, settled by speed alone. Then embed the lattice | the shipped forward gains less than the bar |
| 3 | Score the lattice through the gate; restate ADR 043 from the result (§5.1) | the losses are ones you do not accept |
| 4 | Dev benchmark: tiny area, six Iowa tiles on one card, a year of Iowa against `main` (§4.2) | the gain does not survive on real tiles |
| 5 | Evaluation areas: fill an FP8 store, score it against the baseline (§5.2) | it fails the runbook's rule |
| 6 | READMEs, the record, the PR description | |

## 3. The code changes

### 3.1 What must stay true, and how each is checked

"CPU test" runs in CI. "GPU check" runs by hand on an L40S, and the lattice bootstrap runs it before
embedding anything, following `test_inference_loop.py`'s convention (no CI runner has a GPU).

| # | what must stay true | check |
|---|---|---|
| 1 | v1.1, the CPU, training, every card other than the L40S, and an L40S writing a bf16 store run exactly the arithmetic they run today | On the Step 2 L40S box: the lattice embedded from a `main` checkout and from the branch with `precision="bf16"` is equal bit for bit. Same on the A10G and L4 boxes. Existing golden tests pass unchanged. CPU test: which models hold FP8 weights |
| 2 | On an L40S the FP8 forward differs from today's in exactly three ways: the two multiplications' inputs are quantised, their weights are quantised, and their bias is applied as §3.3 settles. Attention, both LayerNorms, the output projection and the second feed-forward matrix are the same kernels | The layer is one function serving both paths, differing at two call sites (§3.2). The ReLU moving out of the multiplication changes nothing: rounding to bf16 never changes a sign. The bias is the one item with a choice, and §3.3 measures it |
| 3 | A pixel's embedding does not depend on what else shares its batch or chunk | Scales are per token row and per weight channel, never per batch. CPU test: a pixel embedded alone equals the same pixel inside a batch. GPU check: a short last chunk gives the same kept rows whatever the previous chunk left behind it |
| 4 | The 8-bit weights are still 8-bit at every forward, and precision never changes on a built model | They are plain attributes, built once in `build_inference_model`, as `modules.py` already keeps its fp32 frequency cache out of reach of `model.bfloat16()`. Verified locally on torch 2.14.1: the cast turns a registered E4M3 buffer into bf16 and leaves a plain attribute alone. CPU test pins it. Graphs are keyed by bucket shape only, so a model whose precision changed after a capture would replay the old one; nothing in `src/` mutates it and the harness builds a fresh model per variant |
| 5 | A replayed graph gives the same output as the forward it captured, for every bucket shape, in a second process too | GPU check: replay equals the eager forward bit for bit. The lattice is embedded twice in separate processes and compared |
| 6 | Nothing inside the captured forward waits on the host or allocates by data | Weights are quantised at build. Capture itself fails on a host sync; check 5 covers the rest |
| 7 | A NaN or infinite activation still reaches the host as a non-finite scale, and does not disturb any other pixel | **GPU check, on whichever quantiser ships:** a 64-pixel chunk with one NaN row and one infinite row, among live rows and among stale ones. Those pixels come out non-finite; every other pixel equals the same chunk with those rows zeroed, bit for bit. A CPU test covers the zero row only: the CPU cast saturates (infinity becomes 448), which says nothing about the card |
| 8 | The checkpoint loads `strict=True` and `state_dict` has the same keys | Existing test; the FP8 weights are not in `state_dict` |
| 9 | An FP8 worker cannot write into a run or store that was started in bf16, and the reverse | §3.4, in place before any dev run |

### 3.2 What changes

**`config/inference.py` and the model's `ModelFamily` record.** `InferenceConfig.precision`, `"bf16"` or
`"fp8"`, defaulting from the model's record: v2 `fp8`, v1.1 `bf16`, and v1.1 rejects `fp8` the way v2
rejects `norm_source`. This is the one input model construction reads. The harness and the benchmark
use it to build a bf16 model on an L40S; without it the gate's L40S reference would silently be FP8.

**`models/ssl_model.py`.** `FP8_CARDS = {"NVIDIA L40S"}` beside `GRAPH_CHUNK_PIXELS`, with the same
meaning: the cards FP8 was measured on.

**`models/builder.py`.** After the model moves to its card, where `chunk_pixels` is set today: when
the precision is `fp8` and the card is in `FP8_CARDS`, build the FP8 weights. Any other card builds
today's forward. They are made from the fp32 weights, before `_prepare_gpu` casts the model to bf16,
so each weight is rounded once. The build log line names what was built.

**`models/student_v2.py`.**

- A small record per encoder layer holding the E4M3 weight and its per-channel fp32 scale for QKV
  and for the first feed-forward matrix, stored transposed the way the kernel reads it. Each encoder
  holds a list of them as a plain attribute, `None` unless built for FP8.
- A function that builds that list: scale = largest absolute value in the output channel ÷ 448,
  floored at 1e-12; weight ÷ scale, cast to E4M3.
- A row quantiser for activations: the same rule along the token row, computed in fp32, with a clamp
  to ±448 before the cast. Both matter: a probe during review found that the same division done in
  bf16 lands above 448 in about 3% of rows, and in fp32 it can still land one rounding step above.
- `_encoder_layer_forward` gains one optional argument, the layer's record. Where it is given, the QKV
  projection and the first feed-forward multiplication go through `F.scaled_mm` on quantised rows.
  Every other line is shared. One function, not a copy: a second layer function would repeat ten
  lines that must never drift apart. Its only other caller is one unit test, which passes three
  positional arguments and keeps working.
- `StudentTransformerEncoder.forward` passes each layer its record. The condition for the hand-written
  layer path (CUDA, eval, no gradients) is unchanged, so a model with FP8 weights still runs the
  plain modules in training or with gradients on.

**`scripts/precision_gate/embed_lattice.py`.** A variant that builds the shipped forward
(`precision="fp8"`); the reference variants build with `precision="bf16"`; the stand-in closures take
the new argument; and an assertion that the L40S bf16 and FP8 arrays differ, so a reference that was
quietly built as FP8 cannot be scored.

**Not changed:** `inference.py`, the actor, the graph capture in `ssl_model.py`, quantisation of the
output to int8.

### 3.3 Choices settled on the card, by speed, before anything is scored

Step 2 fixes these five from timing alone and writes them down. Only then is the lattice embedded.
Choosing any of them after seeing gate scores would be fitting the candidate to the bar.

| choice | options | default if equal |
|---|---|---|
| Where activations are quantised | A separate pass on the LayerNorm's bf16 output, in ordinary PyTorch operations. Or one `torch.compile`d kernel that normalises and emits both copies, as `_compiled_add_layer_norm` is built. Or a hand-written Triton kernel (the previous plan's choice: about 250 lines that cannot run on a laptop or in CI) | the separate pass. It is the only option that leaves today's LayerNorm kernel in place, so the only one where row 2 of §3.1 holds without a further measurement. The record found a separate pass too slow at 896,000 rows; at 7,400 the activations fit in the card's cache |
| The bias | Inside the multiplication (`F.scaled_mm`'s `bias`, which the CPU accepts with row scales; the card must be tried). Or added after the product is rounded to bf16, which rounds twice where today rounds once | inside |
| The first layer's QKV | bf16 (its input is the band embedding plus the day-of-year encoding, which no LayerNorm produced; the layer's record simply carries no QKV weight). Or quantised too, as the stand-in did | bf16 |
| Fast accumulation | on, as the record measured, or off, `F.scaled_mm`'s default | on |
| Chunk size | 32, 64 or 128 pixels; 64 was tuned for bf16 | 64 |

The shipped forward will differ from the stand-in ADR 043 measured in up to five ways: weights
quantised from fp32, not from bf16-rounded weights (strictly closer to the trained weights); activations
quantised from the LayerNorm's bf16 output, not an fp32 copy (a probe on 4,096 rows: 2.9% of codes
differ, quantisation error up by 0.6%); the stand-in ran every add and LayerNorm in fp32, not the
compiled bf16 kernel; and the first-QKV and bias choices above. A difference between the shipped
forward's gate result and the stand-in's cannot be traced to one of them, which is one reason §5.1
measures the shipped forward in its own right.

Other things the card must answer in Step 2: whether cuBLASLt needs the row count to be a multiple of
16 (a 64-pixel chunk always is; a whole-batch forward may not be, and would be padded as the stand-in
pads); whether the card's E4M3 cast gives the CPU's codes on fixed inputs including ±448, 449, −0 and
ties; and one-step radar sequences (T = 1) as well as deep buckets.

### 3.4 The store's precision

An FP8 tile and a bf16 tile have the same shape and dtype. ADR 043 decision 5 settles what keeps them
apart: precision belongs to the store. It reuses what ADR 025 built:

- **The store records it when it is created**: at seeding for a global store, at first assembly for a
  single-area one, from the config's precision. It is written **only when it is not bf16**. A store or
  run with no record is therefore bf16 by construction: every existing store and staging run, the
  published v1.1 store included, still compares equal to today's stack, and code that predates FP8
  refuses an FP8 store on its own.
- **A run's `_stack.json` carries the declared precision** on the same rule. Declared, not what one
  card built, so an A10G and an L40S still join the same FP8 run.
- **Fills read it from the store before provisioning**, with the model and checkpoint, and build
  their config from it. The pre-flight check refuses a mismatch before any card is rented.

**I recommend dropping one thing ADR 043 asks for:** refusing an FP8 store on any card that is
neither an L40S nor an A10G. An FP8 store would be exactly as open to an unmeasured card as a bf16
store is today: ADR 025 chose not to record the card, and a card without bf16 already writes FP16
into a bf16 store with a warning. `FP8_CARDS` stays the only place that knows about cards.

What this leaves unrecorded: which tiles of an FP8 store an A10G filled. The store says FP8; the
per-tile answer is in the run's logs only.

### 3.5 Tests

CPU, in `tests/unit/inference/test_student_v2.py`: FP8 weights survive `bfloat16()` and stay out of
`state_dict`; the FP8 forward tracks the bf16 forward (per-pixel cosine above 0.999, which a wrong
scale axis or a transposed weight fails by a wide margin); a pixel alone equals the pixel in a batch;
a zero row; which precision, card and model get FP8 weights (parametrised). In the storage tests: a
record without precision equals a bf16 one; an FP8 run refuses a bf16 store and the reverse.

GPU, in the same file behind `torch.cuda.is_available()` and compute capability 8.9: §3.1 rows 3, 5
and 7, and §3.3's last paragraph.

### 3.6 Not built

- **All four matrices (Stage 2).** It needs fixed activation scales, a different quantisation with
  its own gate round, and the re-scored record shows it losing more than Stage 1.
- **FP8 on the L4 or any other card.** Not in the fleet.
- **A flow parameter to choose precision per run.** The store decides.

### 3.7 Commits

1. This plan. 2. `InferenceConfig.precision`, the forward and its CPU tests. 3. The store and run
record (§3.4) and its tests. 4. Harness: the shipped-forward variant, the reference built as bf16, the
GPU checks in the bootstrap. 5. Results into the record; ADR 043 and the bounds file
restated. 6. READMEs (`inference/README.md` §7, `models/README.md`, `docs/inference-performance.md`)
and the PR description.

## 4. Benchmarking

### 4.1 On one L40S box (Steps 0 and 2)

Method as `v2-chunked-graph-replay.md` §2: card pre-heated 90 seconds, four rounds per shape, every
round timing every arm, order rotating, mean clock reported per arm. Shapes: that record's seven Iowa
bucket shapes, plus one-step radar. The timing scripts are that record's, kept at
`s3://global-tessera-inputs-dev/perf/tc/` (`bench_chunk.py`, `bench_ab.py`, `box.sh`), with an FP8
arm added; no new timing script is written.

**Step 0** uses the harness as it is. Its stand-in already runs under chunked replay, so one hour
answers whether `F.scaled_mm` can be captured and replayed at all, and what share of the forward the
two multiplications take at 64-pixel chunks. If that share is p and FP8 runs them k times faster, the
forward gains p × (1 − 1/k); that bounds what any amount of kernel work can reach. The same box
embeds the lattice with today's bf16 and the stand-in, which is the L40S half of the re-run that
never started, and scoring it gives the stand-in's result against today's forward.

**Step 2** times the shipped forward against a bf16 model built on the same box, with the options of
§3.3, and a ceiling arm that quantises outside the timed region.

**Proposed bar: 5% of tile time**, about $18,000 over the nine-year v2 campaign (one percent of its
card cost is about $3,600, `v2-chunked-graph-replay.md` §7). Step 0 reads the forward's share of
inference time from today's profiled batches and converts that into the forward gain required; the
figure PR #273 gives (10.6% of inference became 7.3% of tile time) predates its own speed-up of the
forward. Below the bar, write the result into the record and stop.

### 4.2 On dev, with the dev-test skill

A yield-embeddings branch `dev/global-tessera-fp8` pinned to the tessera branch; the control arm is a
dev branch locked to `main`. Every output goes to a new store, which records FP8 at creation. Nothing
is ingested: the mosaics for the tiny area and for Iowa 2024 (`mosaics/m10_parity_15S_epsg32715`,
`mosaics/iowa_epsg5070`, reflectance and ascending radar) are in the dev bucket, checked 2026-11-19.

- **Tiny area** (`m10_parity_15S_epsg32715`, December 2024, about 8 minutes): proves an actor builds
  the FP8 model, captures its graphs, stages and assembles. `te-compare-outputs --cross-config`
  against the control must pass ADR 012's envelope **and show a difference of FP8's size**; identical
  outputs would mean the FP8 path did not run, whatever the build log says.
- **Six Iowa tiles on one card**, one long-lived actor holding two models, one built per precision,
  each with its own graphs, run in mirrored order (A, B, B, A). This is the number to trust: both arms
  share a card.
- **A year of Iowa**, inference only, eight L40S per arm, both arms started in the same minute, tiles
  compared as pairs from `CHUNK_SUMMARY`. Per-card clocks and thermal throttling are reported for both
  arms. Figures are given with every pair, and again without pairs where either card was throttled;
  dropping only one arm's hot tiles would hide FP8's own effect on the clock. Twelve tiles are
  compared with `te-compare-outputs --cross-config`.

A week of zone 35N is available as a further step and is not planned: the box already times the deep
bucket shapes a dense zone adds.

## 5. Evaluation data

### 5.1 The lattice, through ADR 043's gate

ADR 043 decision 3 requires the kernels that ship to be measured on its six tiles. One run of the
harness: L40S (bf16, the shipped FP8, the stand-in), A10G and L4 (bf16), then scoring.

**I recommend restating the exception from this run, not from the old one.** ADR 043 defined the
exception on a stand-in because the real forward did not exist, then asked the real one to stay
within those numbers plus an allowance. The stand-in's numbers are now known to be wrong (§1), so the
exception needs a fresh decision either way; the ADR is still Proposed; and the real forward will
exist before it is accepted. Decision 2 should name the losses of the forward that ships, measured
against the forward production runs, and you accept or decline those. The gate, its thresholds and
its controls do not move. Two things keep this honest: §3.3's choices are fixed before scoring, and
Step 0's stand-in result is written down first as the expectation, with the shipped forward reported
against it. `fp8_exception_bounds.json` is then regenerated from the shipped forward's reports and
holds any later change to it.

### 5.2 The six evaluation areas, by the runbook (PR #270)

Version `v2-large-fp8`: seed its store, fill the five zone-years (1,099 tiles) from
`dev/global-tessera-fp8`, score against `v2-large.icechunk` with yield-embeddings' `score_evaluation.py`
(PR #120). The runbook's rule and its coverage check apply as written.

**The fill should run on L40S only.** An A10G's tiles in this store would be bf16 and would dilute
whatever FP8 does. If capacity makes that impossible, the write-up gives the share of labelled points
on FP8 tiles, from the run's logs.

**What the runbook's score can and cannot show.** It fits a classifier on each store's own
embeddings, and the record already found FP8 harmless inside one store. The losses appear when a
classifier fitted on one precision is applied to the other. I recommend two added figures from the
same two stores, at no extra fill: fitted on the baseline and applied to the FP8 store, and the
reverse; and a mixed store put together at scoring time, FP8 embeddings at part of the labelled
points and baseline at the rest, since ADR 043's worst tile-level loss is the mixed store. Both need
a small option in the scoring script.

## 6. Cost

| item | estimate |
|---|---|
| L40S box, Steps 0 and 2 | $10–20 |
| Lattice: GPU boxes, and two scoring runs of about four hours each | about $45 |
| Tiny area and six-tile runs | under $10 |
| A year of Iowa, two arms | about $70 |
| Evaluation fill | about $190 by the runbook's estimate; the baseline's cards cost $88 |
| **Total** | **about $340** |

L40S capacity is the schedule risk, not the cost: the re-run's L40S box never started, and the
baseline's fill waited 78 minutes for its cards.

## 7. What an independent review of this plan changed

- Step 0 now runs before any production code, on the stand-in that already exists.
- `InferenceConfig.precision` and the store record moved ahead of every dev run. Without the first the
  gate's L40S reference would have been FP8; without the second the evaluation store would have been
  stamped as bf16 for good.
- The bias was a third arithmetic difference the first draft missed (§3.1 row 2, §3.3).
- Non-finite handling is checked on the card, not on the CPU (§3.1 row 7).
- The store record is written only when it is not bf16, so existing stores need no migration (§3.4).
- The open choices are fixed by speed before scoring (§3.3), and throttled tiles are dropped as pairs
  (§4.2).

Nothing in this plan has run on a GPU. What the card's kernels do with NaN, row counts and the bias
is unknown until Step 0 and Step 2.

## 8. Decisions for review

1. **Where the code goes.** The forward is measured with the gate harness, which is on PR #275's
   branch and not on `plan/evaluation-suite`. I recommend committing to `docs/adr-043-fp8-gate` so the
   ADR, the gate and the forward are one PR, and touching `plan/evaluation-suite` only if the runbook
   needs a change.
2. **Restate ADR 043's exception from the shipped forward** (§5.1), or re-derive it from the re-scored
   stand-in and hold the shipped forward to that.
3. **Drop ADR 043's refusal for unmeasured cards** (§3.4), or keep it.
4. **The bar** (§4.1): 5% of tile time.
5. **The cross-store and mixed-store figures** in the evaluation scoring (§5.2), which change
   yield-embeddings PR #120.
6. **Provenance.** A standing ruling says the next dev run ingests with `provenance=true`, on its own
   dev prefix. I recommend keeping it off these runs, so each differs from its control in one thing.
