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
uv sync --all-extras    # resolves uv.lock; installs all extras + dev tools
```

`uv.lock` at repo root is the single lock file. It covers every extra
(inference, prefect, aws) plus the dev group (pytest, ruff, mypy). Run
`uv lock` after any change to `pyproject.toml` and commit the updated
`uv.lock`. CI's `lock-check.yml` catches drift.

## CUDA

`[inference]` defaults to CPU torch — the `[tool.uv.sources]` entry in
`pyproject.toml` pins the CPU index so `uv sync` never accidentally pulls
CUDA wheels on a laptop.

For GPU production, override the torch index at install time:

```bash
pip install tessera_embeddings[inference] \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

GPU deployment lock files (pinning a specific CUDA version and platform)
are an ops concern for downstream consumers. They belong in your deployment
repo alongside your Dockerfiles and AMI bake scripts, not here.

### MPS (Apple Silicon GPU)

Untested. CPU is the supported laptop path.

## GDAL

`gdal` is declared as an unpinned dependency and is source-only on PyPI —
it builds against the system `libgdal`. You need a matching system library
before installing:

**macOS:**
```bash
brew install gdal
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libgdal-dev
```

The Python binding version must match your installed libgdal. Check with
`gdal-config --version` and install the matching binding:
```bash
pip install gdal==$(gdal-config --version)
```

CI runs inside the `ghcr.io/osgeo/gdal:ubuntu-small-3.13.0` container,
which pins both to 3.13.0.

## Platform notes

### Linux x86_64

The blessed deployment platform. For GPU production, use
`--extra-index-url https://download.pytorch.org/whl/cu121` at install time.

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
