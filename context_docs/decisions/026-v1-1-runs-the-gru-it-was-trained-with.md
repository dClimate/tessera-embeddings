# 026 — v1.1 runs the GRU it was trained with

**Status:** Accepted 2026-10-04 (repo owner). Built on `fix/faithful-gru-pooling`.

v1.1's pooling head ran a GRU with different arithmetic from the one the checkpoint was trained with,
and every v1.1 embedding this repository produced carries the difference. This records the defect,
how large it is on real pixels, what replaces it, and what it does and does not change.

---

## 1. The defect

v1.1 pools each pixel's sequence of observations with a GRU, a LayerNorm and an attention score
(`TemporalAwarePooling`). The checkpoint stores the GRU as `CustomGRU`, whose cell applies the reset
gate to the hidden state **before** its matrix multiply:

```
 trained (CustomGRUCell, and upstream's tessera_infer_QAT):   n = tanh(W_ih x + W_hh (r * h) + b_h)
 nn.GRU:                                                       n = tanh(W_ih x + r * (W_hh h) + b_h)
```

`builder._fuse_custom_gru` swapped every `CustomGRU` for PyTorch's `nn.GRU` before inference, to run
the recurrence as one call. The two agree only when the reset gate `r` is close to 1, and the swap
was justified on exactly that premise, "a small approximation in practice because the reset gate is
close to 1 for most dimensions after training". It was never measured. The swap also never reached
cuDNN, which was its stated reason: cuDNN's RNN kernels accept FP16, FP32 and FP64, and inference
runs BF16, so `nn.GRU` fell back to a step-by-step native loop.

The swap came in with the port of the model code (2026-05-15). Every v1.1 store this repository has
written, the published global store included, was computed with it. v2 has no GRU and is unaffected;
its port is pinned bit-identical to upstream by `tests/unit/inference/test_student_v2_golden.py`.

## 2. How large it is

Measured on 1,572,864 real pixels: the densest 128-row strip of six Iowa tiles chosen to span an
edge tile, sparse and dense optical, light and heavy radar. The pixels went through production's
loaders, 7,168-pixel batches and the BF16 model on one L40S, and were compared with the same model
running the trained arithmetic in FP32.

| Measure | Production (`nn.GRU`) | ADR 012 cross-config envelope |
|---|---|---|
| int8 values identical | 14.9% (11–19% per tile) | — |
| int8 values within ±1 level | 40.6% (31–51%) | ≥ 99.99% |
| Largest deviation | 30 levels | ≤ 3 |
| Per-pixel scale drift, median / max | 7.0% / 24% | ≤ 1.6% |
| Per-pixel cosine, mean / worst | 0.99736 / 0.9899 | worst ≥ 0.9999 |

The difference is the formula alone: running both formulas in FP32 gives the same figures, while BF16
on its own leaves 97.6–97.9% of values within one level. On real pixels the reset gate averages 0.369
in the optical GRU (median 0.22; 3.4% of values above 0.99) and 0.451 in the radar GRU (median 0.35;
13.5% above 0.99), the same within ±0.01 on every tile. The gap grows with radar depth: mean cosine
falls from 0.99856 at a median of 21 radar observations to 0.99615 at 55.

**What it means for a user of the embeddings.** The mean per-pixel shift, 1 − cosine = 2.6 × 10⁻³, is
about 130 times what int8 storage introduces and about three quarters of the typical distance between
adjacent pixels. Within one 128-row strip per tile, so a narrower candidate pool than a real search:

| Comparison with the trained arithmetic | Top-10 neighbours shared | Top-20 shared | Same k-means(12) cluster |
|---|---|---|---|
| production `nn.GRU` | 0.897 | 0.906 | 0.975 |
| BF16 rounding alone | 0.952 | 0.959 | 0.995 |
| int8 storage alone | 0.978 | 0.981 | 0.9985 |

This measures how far production sits from the trained model, not whether either is better at a
downstream task.

## 3. Decision

v1.1 pools with `modules._gru_pool_step`: one timestep of the trained GRU, its LayerNorm and its
attention score, compiled into a single GPU kernel. `nn.GRU` and `_fuse_custom_gru` are gone.
`TemporalAwarePooling` keeps the checkpoint's `CustomGRU` weights and reads them for the step, and
`CustomGRU.forward` stays as the readable reference the step is tested against.

The per-timestep GRU output is never stored: the step returns the new hidden state and one attention
score, and the softmax over the scores weights the inputs at the end. The step's shapes do not depend
on the sequence length, so it compiles once per process, with one recompile when a partial batch
first arrives. CPU runs the same step uncompiled.

## 4. Consequences

- **Every v1.1 output changes,** beyond ADR 012's cross-config envelope and by design: this is a
  correction, not shimmer. The inference code identity changes with it, so staged tiles are not
  reused and existing stores refuse appends from this code. The store boundary is enforced, not
  remembered.
- **Stores written before this are not touched.** They hold the `nn.GRU` arithmetic. Whether any is
  recomputed or relabelled is a separate decision; v1.1 is expected to give way to v2.
- **Faster, not slower.** The fused faithful step ran the forward pass 4.8–6.6% faster than the
  `nn.GRU` path on one L40S (PyTorch 2.5.1, both backbones on their two streams, four batch shapes).
  Running `CustomGRU` as written would have been correct but about 2.7 times slower in the GRU stage.
- **`torch.compile` is now in the inference path,** for this one function. The first compile on a
  fresh machine took up to about a minute; later ones take seconds.

## 5. Not done, and why

- **Keeping `nn.GRU` for continuity with existing stores.** Ruled out by the repo owner: further v1.1
  output should be the most correct possible.
- **CUDA graphs around the step.** No gain, 3.6–8.5 GiB of VRAM held per sequence length, and an
  illegal memory access with both backbones running at the longest shapes.
- **Writing each step's score into a slice of one preallocated buffer.** The prototype did, and at odd
  batch sizes an unaligned slice made the compiled write land wrong (worst cosine 0.974). The step
  returns its score instead, so there is no slice to misalign.
