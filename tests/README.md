# Tests

The `tessera_embeddings` test suite is split by cost and intent:

| Directory | Marker | Runs in | Goal |
|---|---|---|---|
| `unit/` | (default) | every PR | Fast, isolated. < 30s each. |
| `architecture/` | (default) | every PR | Layer-rule enforcement (no Prefect outside `orchestration/prefect/`, etc.). |
| `integration/` | `integration` | path-filtered PRs + nightly | Hit moto / VCR cassettes / LocalCluster. Minutes. |
| `parity/` | `parity` | flow-touching PRs + nightly | Prefect flow ↔ plain runner output equivalence. 5–30 min. |
| `slow/` | `slow` | nightly + on-demand | Full plain-runner end-to-end. 30+ min. |
| `gpu/` | `gpu` | inference-touching PRs only | Tiny (< 2 min) checks of GPU code paths. |

Default `pytest` invocation only runs unit + architecture, per
`addopts = "-m 'not integration and not parity and not slow and not gpu'"` in
`pyproject.toml`. CI workflows opt in to the heavier markers explicitly.

## Hypothesis

Property-based tests live alongside example-based tests in `unit/`.
Profiles are configured in `tests/conftest.py`:

- `local` (default): 20 examples, 500 ms deadline.
- `ci` (when `CI` env var is set): 200 examples, 2 s deadline.

## Fixtures

`tests/fixtures/` holds golden inputs:

- `stac_cassettes/` — VCR.py recordings for STAC queries (`pytest-recording`).
  See its `README.md` for the recording workflow.
- `checkpoints/` — small fake model checkpoints for tests that need to
  load a model. Real production checkpoints are too large for git.

## Running subsets

```bash
# Default (unit + architecture):
uv run pytest

# Property-based tests only:
uv run pytest -k property

# Integration suite (hits VCR cassettes / moto):
uv run pytest -m integration

# Parity contract:
uv run pytest -m parity

# Full local end-to-end (very slow):
uv run pytest -m "slow or parity"
```
