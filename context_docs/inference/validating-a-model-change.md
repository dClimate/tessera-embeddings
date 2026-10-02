# Validating an embedding store when the MODEL changed

**How you prove a run is correct after swapping the encoder — not whether the new encoder is
better.** The method, the thresholds, and the measurements the thresholds were calibrated from.
Written for the Tessera v2 Large rollout, but the argument is about model swaps in general and
applies to the next one.

The rollout's verdict is §5 here; its throughput is §7 of
[`inference-on-gpus.md`](inference-on-gpus.md); what its precision fix did to v1.1 is in
[`v1.1-against-the-published-store.md`](v1.1-against-the-published-store.md).

**Where the gate lives:** `scripts/validate_embedding_model_change.py` in the **yield-embeddings**
repo, beside the runs it was built for.
[ADR-019](../decisions/019-validation-modules-belong-in-the-library.md) moves it here, next to its siblings
(`profiling/inference/compare_outputs.py`, `compare_coarsened_stores.py`), after the campaign.

---

## 1. Why the existing gate does not apply

`te-compare-outputs` implements ADR-012: at least 99.5% of int8 values exactly equal, at most one
quantization level of deviation, cosine at least 0.9999. Those thresholds encode **"same numerics,
different code path"** — they were built to police pipelining and kernel changes that must not move
the output.

A different model moves every value by construction. Running that gate across a v1.1/v2 pair
produces a comprehensive failure that carries no information. Its `--cross-config` mode does not
rescue this either: that relaxation exists for batch-size and library-stack differences, not for
different weights.

## 2. What replaces it

The invariant that survives a model swap is that **only the model changed**. So everything
model-independent must match the reference exactly, while the values must not match at all. The gate
asserts both halves, then judges the values on structure rather than on agreement.

**That boundary falls earlier than "the encoder".** Two stages before it are model-specific, and
treating sampling as shared sends a reader chasing a preprocessing difference through the encoder or
the normalisation, where it is not.

*Shared, and so exact-match testable:* the mosaic reads, the SCL validity mask and per-pixel
validity gate, the observation counts, the band order, the raw integer DOY, and the `{8,16,…,256}`
bucket **schedule** — a pixel with `k` valid observations lands in the same bucket under either
model, because `MODEL_ARCHS["v2-large"]` overrides only the architecture fields and does not touch
`num_obs_checkpoints`.

*Model-specific, and so NOT:*

* **Which observations fill that bucket.** The schedule is shared; the index selection is not.
  `MosaicChunkInferenceDataset` passes `model_version` into the resamplers, and v2 dispatches to
  `build_resample_indices_v2` — a different algorithm from v1.1's, not a refinement of it. Over
  the reachable population (`compute_bin_keys` clips a count to the smallest checkpoint at or
  above it, so the pairs a bucket can actually receive are counts 1..B for each of the 32
  buckets: 4,224 pairs) the two rules select **different indices for 4,128 of them — 97.7%**,
  agreeing on only 96, measured by enumerating both resamplers. So two runs over byte-identical
  pixels hand their encoders different observation sequences, at the same shape and dtype.
* **Band standardisation.** `band_stats(model_version, norm_source)` returns v2's single
  hard-coded set for v2 and the AWS/MPC pair for v1.1, so the normalised tensors differ too.
* **A radar-free pixel's S1 length**, under `allow_s2_only`: one zero step for v2, the smallest
  bucket for v1.1, each model's upstream input.

Requiring either of those to match would assert something false. Hence the exact-match half of the
gate is scoped to the shared list above and stops at the bucket schedule — which is precisely why
**both** of the model-specific stages are *blind spots* the structural checks have to cover. A wrong
padding rule and a wrong statistic set are the same shape of mistake: a correctly-shaped tensor the
encoder accepts, carrying a sequence its training contract never produced.

### The blind spot that motivates the structural checks

v2's `dim_reducer` ends in a **non-affine LayerNorm**. That layer forces every output vector to mean
0 / unit standard deviation *regardless of what it is fed*.

So "the vectors are well formed" proves nothing about the encoder's input. The most plausible
v2-specific mistake — applying v1.1's band statistics instead of v2's hard-coded set, resampling
with v1.1's padding rule, or permuting band order — would produce vectors that pass every numeric
check while carrying no information. Checks 4 through 7 in the script exist solely to close that
gap, and they test structure, which survives the two models occupying unrelated coordinate spaces.

## 3. Calibration — the ceiling

Absolute pass marks cannot be derived from first principles for the structural checks, so they were
calibrated by running two **same-model** stores against each other. That is the ceiling; the
reported chance level is the floor.

Measured 2026-07-28, sampling a 3×3 lattice of 160×160 windows inset 15% from the grid edges, 2000
pixels for neighbour and spectrum work, k=20.

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
| **neighbour agreement** | **99.4× chance** |
| leading 16 dims vs full | 0.5610 |
| leading 32 dims vs full | 0.6913 |
| leading 64 dims vs full | 0.8183 |
| effective rank (participation ratio) | 4.70 / 128 |
| variance in top 16 dims | 0.892 |
| mean absolute int8 code | 37.2 / 127 |
| saturated codes | 0.0080 |
| per-pixel scale p1 / p50 / p99 | 0.01636 / 0.03962 / 0.06348 |

Two readings worth carrying forward:

* **Variance is genuinely concentrated.** Effective rank is 4.70 out of 128, with 89% of variance in
  the leading 16 dimensions. This is a property of the embeddings over homogeneous Iowa farmland,
  not a defect — but it means a rank threshold set just under the reference would fail on sampling
  noise alone. `MIN_EFFECTIVE_RANK` is therefore 2.5, which separates *collapsed* from
  *concentrated* rather than policing the exact figure.
* **Prefix retention is weak evidence for v1.1.** v1.1 is not trained with nested representations,
  yet its leading 16 dimensions still recover 0.56 of the full neighbourhood — a side effect of the
  same variance concentration. The check is therefore a floor against scrambled dimension order, not
  a demonstration of the Matryoshka property. v2, which *is* trained with nested dims (16/32/64/128
  per the checkpoint's stored args), should score at least as well; scoring worse would be the
  interesting result.

## 4. Threshold rationale

| constant | value | separates |
|---|---|---|
| `MIN_COHERENCE_LIFT` | 0.05 | structure present vs absent (reference: 0.19) |
| `MIN_KNN_CHANCE_MULTIPLE` | 5.0 | related vs unrelated space (ceiling: 99.4×) |
| `MIN_PREFIX_OVERLAP` | 0.20 | ordered vs scrambled dims (reference: 0.56) |
| `MIN_EFFECTIVE_RANK` | 2.5 | collapsed vs concentrated (reference: 4.70) |

These are loose on purpose. They are set to catch a broken run, not to grade a working one. A store
landing far below the v1.1 ceiling on neighbour agreement while still clearing 5× is not a pass in
any meaningful sense — **read the numbers, not just the verdict.**

## 5. What the v2 rollout scored

### Readback gate, Iowa run A

921,600 pixels sampled, run A output, against the v1.1 reference store.

| check | result |
|---|---|
| valid-pixel footprint | 0 mismatches |
| S1 asc/desc + S2 obs counts | 0 mismatches |
| per-vector mean / std | **−0.00000 / 1.00001** |
| neighbour agreement vs v1.1 | 0.537 overlap = **80.5× chance** |
| leading 16 dims vs full | 0.676 (v1.1: 0.561) |
| spatial coherence lift | +0.112 (v1.1: +0.173) |
| effective rank | 4.11 / 128 (v1.1: 5.92) |
| mean abs int8 code | 23.1 / 127 (v1.1: 37.2) |

The mean/std result confirms the non-affine LayerNorm head is live — v1.1 scores −0.247 / 1.63 on
the same statistic. Note that this invariant is *forced* by the layer regardless of input, so it
cannot detect wrong band statistics or permuted band order; the structural checks exist for that.
Neighbour agreement at 80.5× chance against a same-model ceiling of 99.4× says v2 describes the same
landscape.

**Quantization note.** v2 uses less of the int8 range than v1.1 (mean |code| 23.1 vs 37.2) with much
tighter scales. This follows from unit-variance output: per-pixel absmax is near-constant at ~3–4σ,
so resolution is spent covering the tail. Not a defect, but v2 gets slightly lower quantization SNR
for the same storage budget.

### End to end on the dev stack, 2026-09-30

The single-area flow on `m10_parity_15S_epsg32715` (1024 × 1025 px, `time_window_end="December
2024"`, `num_actors=2`), once per model from the same code — tessera-embeddings `c550f2bb` through
yield-embeddings `dev/global-tessera-v2-large` — so the model was the only variable.

| check | result |
|---|---|
| the six provenance arrays (three observation counts, three month masks) | identical |
| pixels embedded | the same set, 99.90% of the area |
| embeddings | differ: 0.55% of int8 values coincide, median v1.1-to-v2 cosine −0.05 |
| v2 identity | `geoemb:model` is the Hugging Face URL, `checkpoint_id` is `v2_student_large`, run ID `v2-5f141cb56305`, 128-d |
| v2 output after dequantization | per-pixel mean 0.0000, standard deviation 1.0000 |
| a v2 run pointed at the v1.1 store | refused by the pre-flight before any Ray cluster started |

Flow runs: v1.1 `cf5b18d3`, v2 `f4b57309`, refusal `69a06a5f`.

### The provenance defect the rollout found

The Iowa runs stamped `geoemb:model = https://geotessera.org/model/1.1` while `checkpoint_id` and the
manifest read `student_large`: `ENCODER_VERSION` was a module constant pinned to 1.1. The URL now
comes from `MODEL_ENCODER_URLS` through `encoder_url()`, which raises on an unregistered model rather
than defaulting, because wrong provenance is silent and missing provenance is not. v2's public
reference is its Hugging Face repo, since it has none under `geotessera.org/model/<version>`; replace
it if one is minted. The dev-stack run above is the first to carry the fix.

## 6. Independent evidence, obtained more strongly elsewhere

The port itself is verified separately and far more strongly than any readback check can manage.
`tests/unit/inference/test_student_v2_golden.py` loads the real checkpoint into both our port and a
verbatim vendored copy of upstream's implementation and asserts the forward passes agree. It skips
unless `TESSERA_V2_CKPT` points at the artifact.

Run 2026-07-28 against the staged checkpoint: 7 tests passed, covering state-dict key identity,
loaded-weight identity, forward agreement at four batch/shape combinations, and the LayerNorm
property of upstream's own output. The wider v2 unit suite passed alongside it — 54 tests total.

The checkpoint staged at `s3://arbol-tessera-inputs-dev/models/student_large.pt` was verified to
carry the expected payload before the run: keys `args` and `model`, `latent_dim` 160,
`dim_feedforward` 2560, 4 layers, 4 heads, `repr_dim` 128, QK-norm off, and exactly 43,831,170
parameters.

## 7. Not covered here

Whether a new model is *better* than the old one is a downstream-task question, not a readback one.
For v2 the intended measure is a crop-type probe against the USDA Cropland Data Layer, which is
published natively on EPSG:5070 and so aligns to this grid without reprojection. Nothing above
speaks to it.
