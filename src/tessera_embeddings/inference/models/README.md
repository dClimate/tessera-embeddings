# Tessera Model Architecture (Ported Code)

These files are ported from upstream Tessera model code. The model
architecture and forward-pass logic are **unchanged** from training — do not
modify unless you are also retraining or have verified checkpoint compatibility.

Two model versions live here, selected by `InferenceConfig.model_version`
(`"v1.1"` — the default — or `"v2-large"`). `builder.py` dispatches; everything
upstream of the model (loading, sampling, bucketing) is version-agnostic.

## Source mapping

| File | Ported from | Changes from original |
|---|---|---|
| `modules.py` | `tessera_infer/src/models/modules.py` | Type hints, ruff formatting only. No logic changes. |
| `ssl_model.py` | `tessera_infer/src/models/ssl_model.py` | Type hints, ruff formatting. Backbone annotations widened to `nn.Module` so the wrapper hosts either version's backbones. |
| `student_v2.py` | `geotessera/TESSERA-V-2.0-2B-L` (Hugging Face) `model.py`, = `ucam-eo/tessera` `tessera_infer_v2/student/model.py` | Type hints, ruff formatting; `TransformerEncoder` renamed `StudentTransformerEncoder`; upstream's inline positional encoder replaced by the shared `modules.TemporalPositionalEncoder` (bit-identical at fp32, plus dtype-cast + cached `div_term`); the top-level `PixelStudent` assembly is not duplicated — see below. |
| `builder.py` | `tessera_infer/src/models/builder.py` | Type hints, ruff formatting. Added FSDP prefix stripping in `load_checkpoint()`, plus the v2 build/load path (`_build_v2_inference_model`, `load_v2_checkpoint`, `_verify_v2_args`). |

`tests/fixtures/upstream/v2_student_reference.py` is a **verbatim** copy of
upstream's v2 `model.py`; `tests/unit/test_student_v2_golden.py` runs it beside
our port on the real checkpoint and asserts identical outputs (observed:
bit-identical). Re-fetch instructions are in that file's header.

## v1.1 vs v2 Large

| | v1.1 | v2 Large |
|---|---|---|
| Params (inference graph) | 57.71M | 43.83M |
| `latent_dim` → d_model | 192 → 768 | 160 → 640 |
| Layers / heads / FFN | 4 / 4 / 2048 | 4 / 4 / 2560 |
| Pooling head | `TemporalAwarePooling` (CustomGRU + LayerNorm + attention) | `AttentionPooling` (one `Linear(D,1)` + softmax) — **no recurrence** |
| Output | 192-d; pipeline saves the first 128 | 128-d native, Matryoshka-ordered ({16,32,64,128}) |
| Output normalisation | none | trailing non-affine `LayerNorm` → per-pixel mean 0 / std 1 |
| Checkpoint payload | `{"model_state"/"model_state_dict": …}` + FSDP/compile prefixes + training heads → `strict=False` | `{"model": state_dict, "args": {...}}`, clean → `strict=True` |
| Band stats | MPC/AWS split (`norm_source`) | one fixed set; `norm_source` rejected |
| Checkpoint artifact | `tessera_v1_1_{aws,mpc}_encoder.pt` | `student_large.pt` from `geotessera/TESSERA-V-2.0-2B-L` (`ckpt/student_large.pt`, 175 MB) |

Both versions share the same input contract (S2 = 10 bands in upstream order,
S1 = VV/VH merged with per-orbit normalisation, raw integer DOY 1–365, the
{8,16,…,256} bucket schedule) and the same runtime wrapper,
`MultimodalBTInferenceModel` — its parameter names (`s2_backbone`,
`s1_backbone`, `dim_reducer`) and concat-fusion forward are *identical* to
upstream v2's `PixelStudent.encode`, so a v2 checkpoint loads into it
`strict=True` and it is not duplicated in `student_v2.py`. Reusing it also keeps
the dual-CUDA-stream backbone execution and the profiling hooks for both
versions.

`builder._fuse_custom_gru` is a v1.1-only optimisation and is skipped for v2:
there is no GRU in the v2 graph (so also none of the reset-gate approximation
documented in its docstring).

## What not to touch

- **`modules.py`**: `TransformerEncoder`, `TemporalPositionalEncoder`, `AttentionPooling`,
  `TemporalAwarePooling`, `ProjectionHead` — these define the exact architecture the
  v1.1 checkpoint was trained with. Changing layer dimensions, activation functions, or the
  forward pass will break checkpoint loading.

- **`ssl_model.py`**: `MultimodalBTInferenceModel` (the shared inference wrapper)
  and `build_dim_reducer` (v1.1 reducer). The fusion method (`concat` vs `sum`) and
  `dim_reducer` structure must match the checkpoint.

- **`student_v2.py`**: `StudentTransformerEncoder`, `AttentionPooling`,
  `QKNormEncoderLayer`, `build_v2_dim_reducer` — the v2 Large checkpoint's exact
  architecture. The trailing non-affine `LayerNorm` in the reducer is part of the
  model's output contract, not a cosmetic addition. `QKNormEncoderLayer` is
  inactive for Large (`enable_qk_norm=False`) but is part of the upstream v2
  student family.

## What can be adjusted

- **`builder.py`**: Checkpoint paths can be updated when new checkpoints are
  trained. `load_checkpoint()` handles v1.1 FSDP prefix stripping and
  `load_v2_checkpoint()` the v2 payload — if either format changes, they need
  updating.

- **`InferenceConfig`** (in `../../config/inference.py`): per-version architecture
  defaults live in `MODEL_ARCHS`; fields left at their v1.1 defaults adopt the
  selected version's spec, and a conflicting explicit value is rejected. If you
  train a new model with different hyperparameters, add/update its `ModelArch`.
