# Plan: the fused-kernel test that gates FP8

*Written 2026-11-18 on `feat/fp8-fused-linears`, stacked on `docs/adr-043-fp8-gate`. A plan, not a
record: when the runs are done, their results go into
[`reduced-precision-inference.md`](reduced-precision-inference.md) and this file is deleted.*

## 1. What the test has to show

[ADR 043](../decisions/043-fp8-linears-ship-on-cross-change-accuracy-and-the-a10g-stays-bf16.md)
accepted FP8's measured losses as a named exception, but it measured the **unfused**
`torch._scaled_mm` path. Decision 3 requires the kernels that ship to be measured on the lattice
themselves, against bf16 on the A10G as well as the L40S. Decision 2 says what they must show:

1. **The gate, run in full** (decision 1) on the six-tile suite: envelope, tile loss, class loss,
   validity lists, tripwire.
2. **Nothing outside the exception.** The fused kernels may fail, or leave unresolved, only the cells
   FP8 already failed or left unresolved, plus Ecuador's tile bound. Each of those may lose no more
   than its recorded loss plus one SE (the cell's jackknife SE; 0.2005 + 0.047 for Ecuador's tile
   bound).
3. **The tripwire trips only in recorded classes.** Each may close no more than 1.1 × its recorded
   gap.
4. **Each size against its own record.** Stage 1 is held to the Stage 1 record and Stage 2 (all four
   linears) to the all-four record.

Anything past those bounds is a new decision, not a failed build. §7 lists every bound.

The test also gives the speed figure that decides whether Stage 2 is built
([reduced-precision record, Next steps 3](reduced-precision-inference.md#next-steps)).

## 2. What changed since the record

**The record ran on code without chunked graph replay.** All the mixed-pool and per-class runs used
`te-28057467`. That code predates PR #273
([ADR 042](../decisions/042-chunked-graph-replay-ships-above-the-two-card-yardstick.md)), so every
card ran whole 7,168-pixel batches (3,593 on the A10G). Today's production forward is different:

| card | today's v2 forward |
|---|---|
| L40S | 64-pixel chunks replaying a captured CUDA graph |
| L4 | 32-pixel chunks replaying a captured CUDA graph |
| A10G | whole batch |

This has four consequences for the plan.

- **The reference moves.** "bf16 on the L40S" now means chunked bf16. ADR 042 puts it about two
  two-card yardsticks from whole-batch bf16 (mean cosine distance 2.1e-5), which is small next to
  FP8's 1e-4. But it shifts every comparison, so the whole calibration (noise draws, controls, z_fam,
  gated cells, tripwire thresholds) is redone against the new reference (§6.3). The recorded losses
  remain the bounds, because ADR 043 states the exception in those numbers. Step 6.2 checks that
  the new reference does not by itself move the old candidate outside them.
- **The speed estimate is for the wrong shapes.** FP8's 1.64× per layer and the ~1.15× Tier A
  projection were measured at M = 896k rows. A 64-pixel chunk at the published store's mean of 115
  tokens per pixel is M ≈ 7,400, and chunking has already taken part of the memory-bandwidth gain
  that fusion was meant to capture. **Step 0 measures FP8 at chunk shapes before any kernel is
  written.**
- **The kernels have to be graph-capturable.** The fused forward runs inside
  `torch.cuda.graph`, so it needs:
  - no host syncs, with scales computed on the device;
  - Triton JIT and autotuning finished in `_capture`'s three warm-up forwards;
  - no allocation that depends on data;
  - weights quantised at build time, not lazily on first call.
- **The record's harness cannot be rerun as written.** `mixed_pool_gpu.py` swaps the layer forward by
  monkeypatching `student_v2._encoder_layer_forward` between variants on one model. Under chunked
  replay the first variant's graphs are cached in `model._graphs` per bucket shape, so later
  variants would replay bf16 graphs and score as bf16. The new harness builds a fresh model for each
  variant through `build_inference_model`, as production does.

The build also casts. `prepare_model_for_device` (`inference/inference.py`) calls
`model.bfloat16()`, which casts every floating-point tensor, float8 included. Quantised weights
must therefore be produced **after** that cast or be shielded from it. A unit test pins this (§5.1).

## 3. Scope

**In scope:**

- The code that runs the FP8 forward:
  - precision as a model-construction argument;
  - the fused add + LayerNorm → bf16 + E4M3 kernel;
  - the Stage 1 encoder-layer forward;
  - weight quantisation at build.
- The gate as in-repo tooling.
- The runs and the verdict.

**Out of scope here** (they follow the verdict, in their own commits):

- recording precision in the store at creation;
- refusing an FP8 store on an unmeasured card;
- the store-boundary plumbing (decisions 4 and 5).

The test does not need any of these: it builds models with an explicit precision. Stage 2 is in
scope only as a second candidate through the same harness, if Step 0 and Stage 1's speed justify
it.

## 4. Step 0: is FP8 still faster at chunk shapes? (half a day, one g6e.xlarge)

This decides whether to build anything at all.

1. On a pre-heated L40S (90 s, as in the chunked-replay record §2), time one encoder layer's four
   GEMMs at M = 64 × T, for T in {48, 96, 115, 160, 208}. Compare:
   - bf16 `F.linear` / `_addmm_activation`, as production runs them;
   - `_scaled_mm` E4M3 with row-wise activation and per-channel weight scales, `use_fast_accum=True`.

   Each runs inside a captured graph, so launch cost matches production.
2. Time a whole chunk forward, bf16 against "FP8 with quantisation free". The free case
   pre-quantises the activations outside the timed region; it is the ceiling for Stage 1 at chunk
   shapes.
3. **Go / no-go.** Stage 1's ceiling at the real-tile shape mix must exceed about 1.08× forward
   time, so that ~1.05× survives on tile time. Below that, stop and write the result into the
   record. FP8 at chunk shapes is then not worth its exception, and the plan ends here.
4. On a g6.xlarge (L4) at 32-pixel chunks, the same four GEMMs. If FP8 is slower there, raise it
   before building. Decision 5 has the L4 build FP8 layers, so a slower L4 is a question for the
   ADR, not something to quietly route around.

Script: `scripts/perf/fp8_chunk_gemm_bench.py`, kept-for-reference tier.

## 5. The code under test

### 5.1 Precision at model construction (commit 1, ~150 lines)

- `build_inference_model(config, device, *, precision: Literal["bf16", "fp8_stage1", "fp8_all4"] = "bf16")`.
  For now the caller passes it; later the store's recorded precision will. v1.1 refuses any value
  but `"bf16"`.
- On CUDA, after `prepare_model_for_device`'s cast, `apply_precision(model, precision, device)`
  quantises the selected linears' weights to E4M3 with per-output-channel fp32 scales. They are held
  as buffers on a small `FP8Linear`-style holder that overrides `_apply`, so a later
  `.bfloat16()` / `.to()` does not upcast them. It also records the precision on the encoder, which
  `StudentTransformerEncoder.forward` reads to pick the layer forward.
- The card check is a constant beside `GRAPH_CHUNK_PIXELS`:
  `FP8_CARDS = {"NVIDIA L40S", "NVIDIA L4"}` and `BF16_IN_FP8_STORE = {"NVIDIA A10G"}`. Any other
  card raises when asked for FP8. That card check is the one part of decision 5 the test needs. On
  the A10G an FP8 precision builds today's bf16 forward unchanged.
- Unit tests (CPU):
  - `"bf16"` is bit-identical to today's build;
  - FP8 weights survive `.bfloat16()`;
  - an unlisted card refuses;
  - the A10G maps FP8 to bf16.

  The card name is monkeypatched.

### 5.2 The fused add + LayerNorm emitting bf16 and E4M3 (commit 2, ~250 lines)

- A Triton kernel `add_layer_norm_fp8(x, residual, w, b, eps) -> (y_bf16, y_e4m3, row_scale)`, with
  one program per row (d_model = 640):
  1. fp32 add and LayerNorm;
  2. write the bf16 result for the residual stream;
  3. row amax / 448 → scale (clamped at 1e-12, as the measured variant);
  4. write E4M3 of y_fp32 / scale.

  Quantising from the fp32 LayerNorm output matches the measured Stage 1 variant.
- **Its bf16 output must equal today's compiled add + LayerNorm on every element** (or differ only
  where the reduction order differs). If they diverge, the residual stream changes too, and that is
  a separate shift on top of FP8. Check it against `_compiled_add_layer_norm` on random and real
  activations.
- CPU unit test: a pure-torch reference of the kernel's contract (fp32 add + LN → bf16; row-wise E4M3
  quantisation), pinned against the measured variant's `add_ln` + `quant_row` on fixed inputs.
- GPU test, behind `torch.cuda.is_available()` and capability ≥ (8, 9), following
  `test_inference_loop.py`'s convention (no CI runner has a GPU, `tests/gpu/README.md`), compares
  the Triton kernel with that reference:
  - bf16 output: exact or within 1 ulp;
  - E4M3 codes: equal except at rounding ties;
  - scales: within fp32 rounding.

  The harness bootstrap runs it first and stops the box on failure (§8).

### 5.3 The Stage 1 layer forward (commit 3, ~150 lines)

- `_encoder_layer_forward_fp8_stage1` beside `_encoder_layer_forward`, with the same structure:
  1. QKV from the incoming E4M3 activation through `_scaled_mm`, then SDPA;
  2. out-proj in bf16;
  3. `add_layer_norm_fp8`;
  4. FFN up through `_scaled_mm`, then bias and ReLU;
  5. FFN down in bf16;
  6. `add_layer_norm_fp8`.
- **Bias and ReLU are not in the FP8 GEMM's epilogue.** `_scaled_mm` takes a bias but no
  activation, so the ReLU is a separate elementwise op. Step 0's timing tells whether that matters.
- **The first layer's QKV** has no LayerNorm in front of it (embedding + temporal encoding). Two
  variants are built, and both go through the gate:
  - `first_qkv="bf16"` leaves that one QKV in bf16;
  - `first_qkv="fp8"` quantises it in a separate row-wise pass, as the measured variant did.

  Stage 1's speed and gate result choose between them (reduced-precision record, Next steps 2).
- Graph capture: the forward is captured by the existing `_capture` with no change to
  `ssl_model.py`. A GPU test checks that a captured replay equals the eager forward bit for bit.
- Out-proj and FFN down stay bf16, so Stage 1 adds no static scales.

### 5.4 Stage 2, only if Stage 1 justifies it (later, its own plan)

Stage 2 adds:

- FFN down in FP8, from an FP8 epilogue or a Triton GEMM;
- static activation scales from a calibration pass;
- out-proj, if it earns its place.

It reuses this harness unchanged as candidate `fp8_all4`, held to the all-four record. Static scales
are a different quantisation (decision 3), and the calibration sample must not overlap the lattice.

## 6. The test harness

The gate becomes supported tooling. ADR 043 makes it the bar for every precision change, so it must
not live only in S3. It goes under `scripts/precision_gate/`, with an entry in `scripts/README.md`
naming the ADR.

| file | from | does |
|---|---|---|
| `embed_lattice.py` | `mixed_pool_gpu.py` | Embeds a tile's lattice through `build_inference_model(..., precision=p)` and `prepare_model_for_device`. That gives one fresh model per variant, at production batch size and chunking for the card. Writes `lattice_<card>_<tile>.npz` holding `<variant>_q` / `<variant>_s`, plus the git SHA, torch version, card name and `chunk_pixels`. |
| `score_envelope.py` | `mixed_pool_score.py` | ADR 012 envelope, neighbours and coherence, for any pair of (card, variant). |
| `perclass_cal.py` | same | Unchanged method. Two generalisations: the reference and candidate are named by `card:variant` on the command line instead of the hard-coded `g6e` / `fp8_*` keys, and matched noise is aimed at a cosine passed in (§6.3), not taken from the candidate. |
| `perclass_rule.py` | same | δ fixed at 0.5 (today it needs `DELTA=fixed`; the refit-widened default is the rejected alternative). Otherwise unchanged. |
| `exception_check.py` | new | Reads a tile's rule JSON and `exception_bounds.json`, and prints, per candidate comparison: **inside the exception**, **outside** (naming the cell or class and its excess), or **clean**. Exit code 1 if any comparison is outside. |
| `exception_bounds.json` | generated once from `perclass/rule_unres/*.json` | §7's table, in machine form. |
| `bootstrap-*.sh`, `watch.sh` | the S3 bootstraps | §8. |

The rule and the exception check are pure numpy over saved predictions, so they get unit tests on
synthetic predictions under `tests/unit/` (a planted class loss fails; a planted loss inside its bound
passes the exception; one past it is outside). This is the part a later precision change will
reuse.

### 6.1 Variants embedded per tile

Six tiles, the existing lattices in `s3://arbol-tessera-inputs-dev/perf/fp8-lattice/`.

| card | variants | forward |
|---|---|---|
| L40S (g6e.xlarge) | `bf16`, `fp8_stage1_fused[first_qkv=bf16]`, `fp8_stage1_fused[first_qkv=fp8]`, `fp8_stage1_unfused`, `fp8_all4_unfused` | 64-pixel chunks, graphs |
| L4 (g6.xlarge) | `bf16`, `fp8_stage1_fused[first_qkv=bf16]`, `fp8_stage1_fused[first_qkv=fp8]` | 32-pixel chunks, graphs |
| A10G (g5.2xlarge) | `bf16` | whole batch |

The unfused variants are the record's code paths running under today's chunked forward. They exist
only in the harness, not in `src/`, and they serve §6.2. The unfused "all four" is there so that a
Stage 2 run later has a replay on the same reference.

### 6.2 Harness checks, before the candidate is scored

**These run first, on `docs/adr-043-fp8-gate`, before ADR 043 is accepted.** They re-base the ADR's
own numbers on the chunked forward, so they belong with the ADR, not with the kernels. The harness
(§6, commits 5–6 in §11) lands on that branch too, and this branch is rebased onto it. That run adds
one more variant: L40S chunked bf16 scored against L40S whole-batch bf16 as a gate candidate. It
answers directly whether chunking moved anything in the gate's terms.

1. **Reproduction of the record.** `embed_lattice.py` runs with `chunk_pixels=None` and the record's
   batch of 7,168. Today's whole-batch forward is the one `te-28057467` ran, because PR #273 only
   added the chunked path around it. On one tile, the output must reproduce
   `mixed_g6e_<tile>.npz`'s `bf16_q` and `fp8_stage1_q`.
   - Bitwise is expected: same card type, same torch, and the record's forward was bit-identical
     across processes.
   - If it is not bitwise, it must at least be within the two-card bf16 figures, with the reason
     found before going on.

   This proves that the lattice, the dtype and the harness plumbing match. About 10 minutes.
2. **Replay inside the exception.** The unfused Stage 1 and all-four variants, scored against the
   **new** chunked reference with the **new** calibration, must sit inside their own exception
   bounds on all six tiles.
   - If they do, the bounds are usable as written.
   - If they don't, the reference change alone has moved FP8 outside the record. Stop: that is an
     ADR question (re-record the exception against the chunked reference), not a kernel problem.
3. **Known-good still passes.** These are the new two-card pairs, on today's forwards:
   - L40S chunked bf16 vs A10G whole-batch bf16;
   - L40S chunked bf16 vs L4 chunked bf16.

   Their verdicts must be pass. Under the record, 12 of 12 pairs and 239 of 240 noise draws passed.

### 6.3 Calibration against the new reference

- **Reference:** L40S chunked bf16.
- **Matched noise:** 20 draws per size, aimed at the mean cosine of the **unfused** variant of that
  size against the new reference, from §6.1's replay. This keeps every threshold set from controls,
  never from the candidate (ADR 043 decision 1). The fused candidate's own cosine is reported beside
  it. If it is more than 1.5× further from the reference than the unfused one, say so in the
  verdict, since the noise draws would then be smaller than the candidate.
- **Controls, refit redraws and folds:** as recorded (same seeds), on the new reference.

### 6.4 Comparisons scored as candidates

| comparison | what it stands for | held to |
|---|---|---|
| L40S fused vs L40S bf16 | an FP8 store against a bf16 store, same card | the size's `fp8:` record |
| A10G bf16 vs L40S fused | a mixed store, g5 fallback | the size's `mixed:` record |
| L4 fused vs L40S fused | two cards on FP8 | the two-card FP8 envelope (≥ 99.99% within ±1, top-20 ≥ 0.994) and a gate pass. The L4 is a new kernel on that card, so decision 5's "measured against the L40S" means this row |
| L4 fused vs L40S bf16 | an FP8 store filled by an L4 | the size's `fp8:` record |

Each comparison must pass the gate, or be inside the exception. Each `first_qkv` variant is scored
separately.

## 7. The bounds the fused kernels are held to

From `perclass/rule_unres/*.json` (δ = 0.5). `fp8` is the L40S FP8 against L40S bf16; `mixed` is
A10G bf16 against L40S FP8. Losses are in points, ± the cell's jackknife SE, and each bound is loss
+ SE. All cells are on the linear probe.

**Stage 1**

| tile | cell | recorded as | fp8 | mixed | bound fp8 / mixed |
|---|---|---|---|---|---|
| Kansas | winter wheat, ref→x, recall | fail | 0.66 ± 0.16 | 0.70 ± 0.15 | 0.82 / 0.85 |
| Ecuador | wetland, x→ref, recall | unresolved (ungated) | 0.79 ± 0.17 | — | 0.96 |
| Ecuador | tile probe, x→ref | tile bound | 0.1992 (under 0.2) | 0.2005 | 0.2475 |

**All four linears** (Stage 2)

| tile | cell | recorded as | fp8 | mixed | bound fp8 / mixed |
|---|---|---|---|---|---|
| Kansas | winter wheat, ref→x, recall | fail | 0.96 ± 0.21 | 1.00 ± 0.21 | 1.17 / 1.21 |
| Iowa | forest, x→ref, precision | fail | 0.64 ± 0.15 | 0.65 ± 0.15 | 0.79 / 0.80 |
| Ecuador | trees, x→ref, recall | fail | 0.55 ± 0.09 | 0.55 ± 0.09 | 0.64 / 0.64 |
| Ecuador | cropland, x→ref, recall | unresolved (gated) | 0.79 ± 0.18 | 0.79 ± 0.18 | 0.97 / 0.97 |
| Kansas | sorghum, x→ref, precision | unresolved (gated) | 0.69 ± 0.25 | 0.78 ± 0.30 | 0.94 / 1.08 |
| Kansas | corn, x→ref, recall | unresolved (gated) | 0.58 ± 0.26 | 0.68 ± 0.32 | 0.84 / 1.00 |
| Kansas | sorghum, ref→x, recall | unresolved (gated) | — | 0.51 ± 0.30 | 0.81 |
| Punjab | trees, ref→x, precision | unresolved (ungated) | 0.53 ± 0.10 | — | 0.63 |

**Tripwire:** the classes that may trip, and the most gap closure each may show (× 1e-3, 1.1 × the
record).

| tile | class | Stage 1 | all four |
|---|---|---:|---:|
| UK | trees | — | 1.28 |
| Iowa | pasture | 2.02 | 3.00 |
| Ecuador | trees | 2.04 | 4.31 |
| Ecuador | grass | 2.87 | 2.17 |
| Punjab | trees | 2.28 | 2.05 |
| Kansas | pasture | 4.84 | 5.76 |
| Kansas | winter wheat | 2.86 | 4.77 |
| Kansas | sorghum | 4.49 | 7.73 |
| Finland | grass | — | 3.61 |

A dash means no trip was recorded, so any trip there is outside the exception. At Stage 1 that
includes every class on UK and Finland. `exception_bounds.json` carries the unrounded values; a unit
test checks that it agrees with this table and with ADR 043's quoted figures to the shown precision.

## 8. Running it

| stage | box | time | cost (on-demand) |
|---|---|---|---|
| Step 0 | g6e.xlarge, plus g6.xlarge for the L4 half | ~2 h each | ~$6 |
| §6.2.1 reproduction | g6e.xlarge | ~30 min with setup | ~$1 |
| embed, §6.1 | one each of g6e.xlarge, g6.xlarge, g5.2xlarge, in parallel | ~2 h (the record's boxes fit in a 150-min cap) | ~$10 |
| calibrate + rule, six tiles | m7i.16xlarge, 60 single-threaded shards | ~1 h (record: about an hour) | ~$4 |
| speed, §9 | g6e.xlarge | ~3 h | ~$6 |

Every bootstrap follows the same pattern as the record's, with the failure handling the record
lacked.

**Ending the box:**

- `set -euo pipefail`, with an `EXIT` trap that uploads the log and a `STATUS` file
  (`ok` / `failed:<step>`) and then shuts down;
- a hard `shutdown -h +N` cap set first.

**The watcher** (`watch.sh`) polls every 60 s for the first 10 minutes, then every 5. On each pass it:

- checks that the instance is running;
- checks for the `STATUS` file;
- greps the uploaded log tail for `Traceback|Error|CUDA error|illegal`;
- checks the progress counter (variants and tiles done), which must advance within 20 minutes.

It exits and reports on the first failure signal. The first poll must see the GPU test (§5.2) pass
and the first variant's first tile start, before the watcher drops to the slow interval.

Lattices, labels and the checkpoint are in `s3://arbol-tessera-inputs-dev/` and are read with
`AWS_PROFILE=ArbolYieldModeling`. Outputs go under `perf/fp8-fused/<run-id>/`, never over the
record's `perf/fp8-mixed/`.

## 9. Speed, measured the way the chunked-replay record was

- **Synthetic:** a pre-heated card, four rounds per shape, rotating arm order, at the seven Iowa
  bucket shapes from `v2-chunked-graph-replay.md` §5. Arms: bf16 chunked, then each Stage 1 variant.
- **Real tiles:** one long-lived `InferenceActor` over six `iowa_epsg5070` tiles, arms in mirrored
  order (A, B, B, A), reading inference time from `CHUNK_SUMMARY`.
- **Report:**
  - forward and tile-time ratios;
  - the card's mean clock while inferring (a chunked forward already runs the L40S at its power
    limit);
  - $/Mpx against $0.030 (bf16 g6e).
- **Stage 2 decision:** build it only if Stage 1's measured tile-time gain leaves at least half of
  Step 0's remaining ceiling unclaimed.

## 10. The verdict, and what it produces

For each `first_qkv` variant, one of three verdicts:

- **clean:** passes the gate outright;
- **inside the exception:** fails or is unresolved only in §7's cells, within bound, and trips only
  in §7's classes, within bound;
- **outside:** a new decision, with the offending cells and classes named.

The validity lists (flag-only, uncovered, unvalidated) go beside the verdict, as decision 1.4
requires.

What it produces:

- results appended to `reduced-precision-inference.md` (a new section "The fused kernels on the
  lattice"), with the Next steps updated;
- an amendment to ADR 043 if a verdict is "outside", or if §6.2.2 finds that the reference change
  moved the record;
- `docs/inference-performance.md` and `inference/README.md` §7 updated when Stage 1 ships;
- then the store-precision commits (decisions 4 and 5), on this branch or one stacked on it.

## 11. Commit sequence

Each commit builds and passes its tests, and each is 100–500 lines.

1. This plan.
2. `precision` at model construction, `FP8_CARDS`, weight quantisation that survives the cast, with
   CPU tests (§5.1).
3. The Triton fused add + LayerNorm → bf16 + E4M3, its pure-torch reference, and the CPU and GPU
   tests (§5.2).
4. The Stage 1 layer forward and both `first_qkv` variants, wired through `precision`, with the
   graph-capture GPU test (§5.3).
5. `scripts/precision_gate/`: the rule, the exception check, `exception_bounds.json` and their unit
   tests.
6. `scripts/precision_gate/`: the embed, calibrate and score scripts, plus the bootstraps and
   watcher.
7. `scripts/perf/fp8_chunk_gemm_bench.py` (Step 0). It runs first, but it can land with or before 2.
8. Results into the record, and the ADR amendment if one is needed.

Step 0 runs before commit 2's code is written. If it says no-go, commits 2–6 never happen, and the
branch carries the plan, the bench and the record update.

## 12. Open questions, with a recommendation each

1. **A cell recorded in only one comparison** (Kansas sorghum recall is mixed-only; Punjab trees
   precision and Ecuador wetland are fp8-only). *Recommend:* allow the cell in either comparison at
   the bound recorded for it. ADR 043's unresolved table already treats the two comparisons as one
   cell ("mixed-store figure where it is larger").
2. **Recalibrate against the chunked reference, or reuse the recorded calibration.** *Recommend:*
   recalibrate (§6.3). The recorded z_fam and gated cells were computed against a reference
   production no longer runs. §6.2.2 catches the case where that alone matters.
3. **Where the gate lives.** *Recommend:* `scripts/precision_gate/` as supported tooling, with the
   pure rule and exception check unit-tested. Moving the rule into the library
   (`profiling/inference/`, beside `compare_outputs.py`) is the alternative if a pipeline step ever
   needs to call it.
4. **The L4 at Stage 1, if Step 0 finds FP8 slower there.** *Recommend:* raise it before building,
   rather than letting the L4 build bf16 layers in an FP8 store. That would be a change to decision
   5, and it needs L4-bf16-vs-L40S-FP8 scored as a mixed store (likely the A10G's numbers, since the
   bf16 cards agree).
