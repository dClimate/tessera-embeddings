# Environment setup

`tessera_embeddings` is an inference library. The goal is generating Tessera
per-pixel satellite embeddings. Ingestion is the prerequisite — the base
install covers it. Add `[inference]` to unlock the model and distributed
execution that is the actual point of the library.

## Install tiers

| Tier | Command | Who needs it |
|------|---------|--------------|
| Inference (typical) | `pip install tessera_embeddings[inference]` | most users |
| + Prefect orchestration | `pip install tessera_embeddings[inference,prefect]` | Prefect deployments |
| + AWS | `pip install tessera_embeddings[inference,prefect,aws]` | AWS production |
| Base only | `pip install tessera_embeddings` | contributors, CI, library integrations |

uv is recommended for reproducible installs but not required — all `pip`
commands above work identically with `uv pip`.

## Contributors

```bash
git clone https://github.com/dClimate/tessera-embeddings
cd tessera-embeddings
uv sync --all-extras   # resolves uv.lock; all extras + dev tools
```

`uv.lock` at repo root is the single lock file. It covers every extra
(inference, prefect, aws) plus the dev group (pytest, ruff, mypy). Run
`uv lock` after any change to `pyproject.toml` and commit the updated
`uv.lock`. CI's `lock-check.yml` catches drift.

## CUDA

`[inference]` defaults to CPU torch — the `[tool.uv.sources]` entry in
`pyproject.toml` pins the CPU index so `uv sync` never accidentally pulls
CUDA wheels on a laptop.

For GPU production, install torch with the CUDA wheel explicitly first,
then install the package. `--extra-index-url` alone is not sufficient
because PyPI's CPU wheel stays in the candidate pool and can win:

```bash
# 1. Install CUDA torch from the pytorch index
pip install "torch==2.6.0+cu121" --index-url https://download.pytorch.org/whl/cu121

# 2. Install the package — pip sees torch already satisfied, keeps the CUDA wheel
pip install "tessera_embeddings[inference]"
```

For fully reproducible GPU deploys, generate a pinned constraints file in
your deployment repo:

```bash
uv pip compile pyproject.toml \
    --extra inference --extra prefect --extra aws \
    --python-platform linux --python-version 3.12 \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match \
    --no-sources \
    -o constraints-cu121.txt
```

That file belongs in your deployment repo alongside your Dockerfiles, not
in this OSS library.

### MPS (Apple Silicon GPU)

Untested. CPU is the supported laptop path.

## Platform notes

### Linux x86_64

The blessed deployment platform. For GPU production, follow the explicit
two-step install in the [CUDA section above](#cuda): install
`torch==...+cu121` with `--index-url` first, then install the package.

### macOS arm64 (Apple Silicon)

CPU inference is the supported path.

### Windows

Use WSL2. We don't ship Windows-native wheels.

## Python versions

`pyproject.toml` declares `requires-python = ">=3.12"`. CI tests 3.12 and
3.13. Earlier versions are not supported.

## Verifying the install

```bash
uv run python scripts/check_env.py
```

Prints:

```
Python:        3.13.x
Platform:      darwin-arm64 (or linux-x86_64, etc.)
Torch:         <version> [cuda=False]
Ray:           <version>
Prefect:       <version>
```

If `cuda=True` on a CPU-only host, you installed a CUDA torch wheel by
accident — reinstall without a CUDA index URL.
