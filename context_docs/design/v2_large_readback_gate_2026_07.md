# Validating an embedding store when the MODEL changed

The method and calibration behind the readback gate. The script states the
concepts; this file holds the measurements its thresholds were set from, and the
reasoning that picked them.

**Where the gate lives:** `scripts/validate_embedding_model_change.py` in the
**yield-embeddings** repo. It landed there because that is where the runs were
driven from, not by design — its siblings
(`profiling/inference/compare_outputs.py`, `compare_coarsened_stores.py`) are in
this repo, and it arguably belongs beside them.

Run-level figures for the rollout this gate was built for are in
[`v2_large_rollout_2026_07.md`](v2_large_rollout_2026_07.md): that doc records the
verdict, this one explains why the checks are the ones they are.

Context: Tessera v2 Large (128-d distilled student, `student_large.pt`) is being
introduced alongside v1.1 (`tessera_v1_1_aws_encoder`) via tessera-embeddings
PR #98. Everything below concerns proving a v2 run is *correct*, which is a
different question from whether it is *better*.

---

## 1. Why the existing gate does not apply

`te-compare-outputs` implements ADR-012: at least 99.5% of int8 values exactly
equal, at most 1 quantization level of deviation, cosine at least 0.9999. Those
thresholds encode **"same numerics, different code path"** — they were built to
police pipelining and kernel changes that must not move the output.

A different model moves every value by construction. Running that gate across a
v1.1/v2 pair produces a comprehensive failure that carries no information. Its
`--cross-config` mode does not rescue this either: that relaxation exists for
batch-size and library-stack differences, not different weights.

## 2. What replaces it

The invariant that survives a model swap is that **only the model changed**.
Everything upstream of the encoder is model independent, so it must match the
reference exactly, while the values must not match at all. The gate asserts both
halves, then judges the values on structure rather than on agreement.

### The blind spot that motivates the structural checks

v2's `dim_reducer` ends in a **non-affine LayerNorm**. That layer forces every
output vector to mean 0 / unit standard deviation *regardless of what it is fed*.

So "the vectors are well formed" proves nothing about the encoder's input. The
most plausible v2-specific mistake — applying v1.1's band statistics instead of
v2's hard-coded set, or permuting band order — would produce vectors that pass
every numeric check while carrying no information. Checks 4 through 7 in the
script exist solely to close that gap, and they test structure, which survives
the two models occupying unrelated coordinate spaces.

## 3. Calibration — the ceiling

Absolute pass marks cannot be derived from first principles for the structural
checks, so they were calibrated by running two **same-model** stores against each
other. That is the ceiling; the reported chance level is the floor.

Measured 2026-07-28, sampling a 3x3 lattice of 160x160 windows inset 15% from
the grid edges, 2000 pixels for neighbour and spectrum work, k=20.

| store | model |
|---|---|
| `iowa_epsg5070-inference-speedup-phase5.zarr` | `tessera_v1_1_aws_encoder` |
| `iowa_epsg5070-reference.zarr` | `tessera_v1_1_aws_encoder` |

| metric | same-model value |
|---|---|
| spatial coherence — adjacent cosine | +0.9745 |
| spatial coherence — random-pair cosine | +0.7836 |
| spatial coherence — **lift** | **+0.1909** |
| neighbour overlap (top-20) | 0.9940 |
| chance overlap | 0.0100 |
| **neighbour agreement** | **99.4x chance** |
| leading 16 dims vs full | 0.5610 |
| leading 32 dims vs full | 0.6913 |
| leading 64 dims vs full | 0.8183 |
| effective rank (participation ratio) | 4.70 / 128 |
| variance in top 16 dims | 0.892 |
| mean absolute int8 code | 37.2 / 127 |
| saturated codes | 0.0080 |
| per-pixel scale p1 / p50 / p99 | 0.01636 / 0.03962 / 0.06348 |

Two readings worth carrying forward:

* **Variance is genuinely concentrated.** Effective rank is 4.70 out of 128, with
  89% of variance in the leading 16 dimensions. This is a property of the
  embeddings over homogeneous Iowa farmland, not a defect — but it means a rank
  threshold set just under the reference would fail on sampling noise alone.
  `MIN_EFFECTIVE_RANK` is therefore 2.5, which separates *collapsed* from
  *concentrated* rather than policing the exact figure.
* **Prefix retention is weak evidence for v1.1.** v1.1 is not trained with nested
  representations, yet its leading 16 dimensions still recover 0.56 of the full
  neighbourhood — a side effect of the same variance concentration. The check is
  therefore a floor against scrambled dimension order, not a demonstration of the
  Matryoshka property. v2, which *is* trained with nested dims (16/32/64/128 per
  the checkpoint's stored args), should score at least as well; scoring worse
  would be the interesting result.

## 4. Threshold rationale

| constant | value | separates |
|---|---|---|
| `MIN_COHERENCE_LIFT` | 0.05 | structure present vs absent (reference: 0.19) |
| `MIN_KNN_CHANCE_MULTIPLE` | 5.0 | related vs unrelated space (ceiling: 99.4x) |
| `MIN_PREFIX_OVERLAP` | 0.20 | ordered vs scrambled dims (reference: 0.56) |
| `MIN_EFFECTIVE_RANK` | 2.5 | collapsed vs concentrated (reference: 4.70) |

These are loose on purpose. They are set to catch a broken run, not to grade a
working one. A v2 store landing far below the v1.1 ceiling on neighbour agreement
while still clearing 5x is not a pass in any meaningful sense — read the numbers,
not just the verdict.

## 5. Independent evidence already obtained

The port itself is verified separately and more strongly than any readback check
can manage. `tests/unit/test_student_v2_golden.py` in tessera-embeddings loads the
real checkpoint into both our port and a verbatim vendored copy of upstream's
implementation and asserts the forward passes agree. It skips unless
`TESSERA_V2_CKPT` points at the artifact.

Run 2026-07-28 against the staged checkpoint at commit `ad956a3`: 7 tests passed,
covering state-dict key identity, loaded-weight identity, forward agreement at
four batch/shape combinations, and the LayerNorm property of upstream's own
output. The wider v2 unit suite passed alongside it — 54 tests total.

The checkpoint staged at `s3://arbol-tessera-inputs-dev/models/student_large.pt`
was verified to carry the expected payload before the run: keys `args` and
`model`, `latent_dim` 160, `dim_feedforward` 2560, 4 layers, 4 heads, `repr_dim`
128, QK-norm off, and exactly 43,831,170 parameters.

## 6. Not covered here

Whether v2 is *better* than v1.1 is a downstream-task question, not a readback
one. The intended measure is a crop-type probe against the USDA Cropland Data
Layer, which is published natively on EPSG:5070 and so aligns to this grid without
reprojection. Nothing above speaks to it.
