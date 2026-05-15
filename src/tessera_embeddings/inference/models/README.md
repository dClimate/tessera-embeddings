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

- **`modules.py`**: `TransformerEncoder`, `TemporalPositionalEncoder`, `AttentionPooling`,
  `TemporalAwarePooling`, `ProjectionHead` — these define the exact architecture the
  checkpoint was trained with. Changing layer dimensions, activation functions, or the
  forward pass will break checkpoint loading.

- **`ssl_model.py`**: `MultimodalBTModel` (full model for checkpoint loading) and
  `MultimodalBTInferenceModel` (stripped model for inference). The fusion method
  (`concat` vs `sum`) and `dim_reducer` structure must match the checkpoint.

## What can be adjusted

- **`builder.py`**: Checkpoint paths (`CHECKPOINT_FULL`, `CHECKPOINT_QAT`) can be
  updated when new checkpoints are trained. The `load_checkpoint()` function handles
  FSDP prefix stripping — if the checkpoint format changes, this may need updating.

- **`InferenceConfig`** (in `../config.py`): Default architecture parameters must match
  the checkpoint. If you train a new model with different hyperparameters, update the
  defaults there.
