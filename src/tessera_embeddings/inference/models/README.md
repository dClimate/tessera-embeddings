# Tessera Model Architecture (Ported Code)

These files are ported from upstream Tessera model code. The model
architecture and forward-pass logic are **unchanged** from training — do not
modify unless you are also retraining or have verified checkpoint compatibility.

Two model versions live here, selected by `InferenceConfig.model_version`
(`"v1.1"` — the default — or `"v2-large"`). `builder.py` dispatches. Loading and
bucketing upstream of the model are shared; the resampling rule and the band statistics
are chosen per version (`../README.md` §6), which is why comparing two models' outputs
has to treat sampling as model-specific
(`context_docs/inference/validating-a-model-change.md` §2).

## Source mapping

| File | Ported from | Changes from original |
|---|---|---|
| `modules.py` | `tessera_infer/src/models/modules.py` | Type hints, ruff formatting. Upstream's `TransformerEncoder` renamed `V11TransformerEncoder` (upstream v2 has a same-named class; see `student_v2.py`). `TemporalPositionalEncoder` caches `div_term` per device in FP32 (computed on the CPU for v1.1 and on the device for v2, as each upstream does) and takes an explicit output dtype, and the encoder casts bands (not DOY) to the weights' dtype — see "The input stays FP32" in `../README.md` §7. `TemporalAwarePooling` runs its GRU, LayerNorm and attention score as one compiled step per timestep. Layer shapes and forward-pass math are unchanged. |
| `ssl_model.py` | `tessera_infer/src/models/ssl_model.py` | Type hints, ruff formatting. Backbone annotations widened to `nn.Module` so the wrapper hosts either version's backbones. |
| `student_v2.py` | `geotessera/TESSERA-V-2.0-2B-L` (Hugging Face) `model.py`, = `ucam-eo/tessera` `tessera_infer_v2/student/model.py` | Type hints, ruff formatting; upstream's `TransformerEncoder` renamed `StudentTransformerEncoder`; upstream's inline positional encoder replaced by the shared `modules.TemporalPositionalEncoder` (bit-identical at fp32, plus an explicit output-dtype cast and a per-device FP32 `div_term` cache); the top-level `PixelStudent` assembly is not duplicated — see below. |
| `builder.py` | `tessera_infer/src/models/builder.py` | Type hints, ruff formatting. Added FSDP prefix stripping in `load_v11_checkpoint()`, plus the v2 build/load path (`_build_v2_inference_model`, `load_v2_checkpoint`, `_verify_v2_args`). |

Both models are pinned to upstream by golden tests that run a **verbatim** copy of
upstream's code beside our port on the real checkpoint, in FP32 on CPU:

| Model | Vendored upstream | Test | Observed |
|---|---|---|---|
| v2 Large | `tests/fixtures/upstream/v2_student_reference.py` | `test_student_v2_golden.py` (`TESSERA_V2_CKPT`) | bit-identical |
| v1.1 | `tests/fixtures/upstream/v1_1_reference/` | `test_v11_golden.py` (`TESSERA_V11_CKPT`) | FP32 rounding, from the fused pooling step |

Each skips unless its environment variable points at the checkpoint, and refuses a file whose digest
differs. Re-fetch instructions are in the vendored files' headers.

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
| Resampling rule | `build_resample_indices` | `build_resample_indices_v2` (upstream `_pad_pattern`) — the bucket SCHEDULE is shared, the index selection is not |
| Radar-free pixel (`allow_s2_only`) | all-zeros S1 slice in the smallest bucket | all-zeros S1 slice, one step |
| Fusion | `concat` or `sum` | `concat` only, refused at config time |
| Checkpoint artifact | `tessera_v1_1_{aws,mpc}_encoder.pt` | `v2_student_large.pt`, staged from `geotessera/TESSERA-V-2.0-2B-L` (`ckpt/student_large.pt`, 175 MB) |

Both versions share the same input *contract* (S2 = 10 bands in upstream order,
S1 = VV/VH merged with per-orbit normalisation, raw integer DOY 1–365, the
{8,16,…,256} bucket schedule) and the same runtime wrapper,
`MultimodalBTInferenceModel` — its parameter names (`s2_backbone`,
`s1_backbone`, `dim_reducer`) and concat-fusion forward are *identical* to
upstream v2's `PixelStudent.encode`, so a v2 checkpoint loads into it
`strict=True` and it is not duplicated in `student_v2.py`. Reusing it also keeps
the dual-CUDA-stream backbone execution and the profiling hooks for both
versions.

v1.1's pooling head runs each timestep of its GRU, LayerNorm and attention score
as one compiled step (`modules._gru_pool_step`), in the arithmetic the checkpoint
was trained with: the reset gate is applied to the hidden state before its matrix
multiply. PyTorch's `nn.GRU` applies it after and cannot stand in for it
([ADR 026](../../../../context_docs/decisions/026-v1-1-runs-the-gru-it-was-trained-with.md)).
v2 has no GRU.

## One record per model

Everything that differs between models is a field of its `ModelFamily` record in
`../../config/inference.py`:

- the architecture, checkpoint file names and band statistics
- the public encoder URL a store advertises as `geoemb:model`
- the fusion its checkpoint loads with
- the student generation, which picks the checkpoint loader and the staging run-id prefix
- the S1 bin given to a pixel with no radar
- whether it runs only on its own worker image
- the names of an output store's Y and X dimensions: `northing`/`easting` for v1.1, `y`/`x` for v2.
  A new store takes them once; an existing store keeps its own
  ([ADR 032](../../../../context_docs/decisions/032-v2-stores-name-their-axes-y-and-x.md))

Code asks `model_family(version)` for the field it needs rather than comparing version strings, so
a new model is a new record, and a record missing a field cannot be built. Two names carry one
meaning each:

| name | means | example |
|---|---|---|
| `model_version` | the model | `"v1.1"`, `"v2-large"` |
| `checkpoint_id` | the stem of the checkpoint file, recorded at a store's root | `tessera_v1_1_aws_encoder` |

**Each run decides its model once:**

- **Single-area runs** take it from `InferenceConfig.model_version`, v1.1 by default.
- **A global store** is seeded for one model (`seed_global_store(model_version=...)`), which stamps
  its `geoemb:model`. Every fill, campaign and targeted update reads it back through
  `store_model_version`, and the checkpoint, worker image, staging run-id prefix and inference config
  all follow from it. The campaign flows take no model parameter. A single zone fill may name
  one, and is refused if it disagrees with the store unless it sets `allow_model_mismatch`.
- **Defaults** exist only at those entry points and on the workers that are handed a config: the
  dataset, the sampler, assembly and zone seeding. The functions that name a checkpoint or write
  provenance (`checkpoint_filename`, `band_stats`, `expected_model_url`) require the model.

## What not to touch

- **`modules.py`**: `V11TransformerEncoder`, `TemporalPositionalEncoder`,
  `TemporalEncoding`, `TemporalAwarePooling`, and the `CustomGRU` / `CustomGRUCell`
  pair — these define the exact architecture the v1.1 checkpoint was trained with.
  Changing layer dimensions, activation functions, or the forward pass will break
  checkpoint loading. `_gru_pool_step` must compute exactly what `CustomGRUCell`,
  the LayerNorm and the attention score compute; `tests/unit/inference/test_models.py`
  holds it to them.

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
  trained. `load_v11_checkpoint()` handles v1.1 FSDP prefix stripping and
  `load_v2_checkpoint()` the v2 payload — if either format changes, they need
  updating. `load_v2_checkpoint()` reads with `weights_only=True`: the documented
  v2 payload is tensors plus a plain dict, and checkpoints are fetched over the
  network onto every worker, so the arbitrary-object unpickler is not an option.
  A payload storing `args` as an `argparse.Namespace` is rejected rather than
  allowlisted — convert it to a dict and re-save.

- **`InferenceConfig`** (in `../../config/inference.py`): per-model architecture
  defaults live in each `ModelFamily` record's `arch`; fields left at their v1.1
  defaults adopt the selected model's spec, and a conflicting explicit value is
  rejected. A new model with different hyperparameters gets its own record
  ([One record per model](#one-record-per-model)).
