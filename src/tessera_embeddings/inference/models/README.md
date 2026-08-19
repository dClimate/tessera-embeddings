# Tessera Model Architecture (Ported Code)

These files are ported from the original `tessera_infer` repository. The model
architecture and forward-pass logic are **unchanged** from training — do not
modify unless you are also retraining or have verified checkpoint compatibility.

## Source mapping

| File | Ported from | Changes from original |
|---|---|---|
| `modules.py` | `tessera_infer/src/models/modules.py` | Type hints, ruff formatting only. No logic changes. |
| `ssl_model.py` | `tessera_infer/src/models/ssl_model.py` | Type hints, ruff formatting only. No logic changes. |
| `builder.py` | `tessera_infer/src/models/builder.py` | Type hints, ruff formatting. Added FSDP prefix stripping in `load_checkpoint()`. |

## What not to touch

- **`modules.py`**: `TransformerEncoder`, `TemporalPositionalEncoder`,
  `TemporalEncoding`, `TemporalAwarePooling`, and the `CustomGRU` / `CustomGRUCell`
  pair — these define the exact architecture the checkpoint was trained with. Changing
  layer dimensions, activation functions, or the forward pass will break checkpoint
  loading.

- **`ssl_model.py`**: `MultimodalBTInferenceModel` (the inference model) and
  `build_dim_reducer`. The fusion method (`concat` vs `sum`) and the `dim_reducer`
  structure must match the checkpoint.

## What can be adjusted

- **`builder.py`**: `load_checkpoint()` handles FSDP prefix stripping — if the
  checkpoint format changes, this may need updating. `build_inference_model()` assembles
  the model from an `InferenceConfig`.

- **Checkpoint names** live in `config/inference.py`, not here:
  `checkpoint_filename(norm_source)` maps a normalization source (`"aws"` / `"mpc"`) to
  the bundled checkpoint's filename. A new checkpoint ships under a new name — which is
  deliberate, because that filename is what the global store's model gate and the
  campaign's staging fingerprint compare against.

- **`InferenceConfig`** (in `config/inference.py`): default architecture parameters must
  match the checkpoint. If you train a new model with different hyperparameters, update
  the defaults there.
