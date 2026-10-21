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
because PyPI's CPU wheel stays in the candidate pool and can win.

**Check the build exists before pinning it.**
`https://download.pytorch.org/whl/<cuXXX>/torch/` lists what each index carries; `torch 2.14.1`
publishes `cu126`, `cu130` and `cu132`. Pick the one your driver supports: the CUDA 13 builds need
an NVIDIA driver of 580 or later, `cu126` runs on older ones. CUDA 12.6 predates Blackwell, so
`cu126` has no kernels for those cards (`g7`, `g7e`): they need a 580+ driver and a CUDA 13 build.
Production runs v2 on `cu130`. v1.1's production runs stay on PyTorch 2.5.1 with CUDA 12.1, the
stack the published v1.1 store was inferred on: on 2.14.1, v1.1's outputs differ from that store's by
about as much as two GPU models' do (ADR 025; measurements in
`context_docs/inference/inference-on-gpus.md` §8).

**Supported Python is 3.12-3.13** — what CI tests, what the classifiers advertise, and what these
instructions assume. `requires-python` is `>=3.12` with no upper bound, so 3.14 installs but is
untested; every `torch 2.14.1` CUDA index publishes `cp314` wheels if you go there.

To support a new version: add it to the `unit.yml` matrix and the `pyproject.toml` classifiers,
which must stay in step. Neither depends on CUDA — that job syncs without `--no-sources`, so
`[tool.uv.sources]` holds torch to the CPU index.

```bash
# 1. Install CUDA torch from the pytorch index (supported: Python 3.12-3.13)
pip install "torch==2.14.1+cu130" --index-url https://download.pytorch.org/whl/cu130

# 2. Install the package — pip sees torch already satisfied, keeps the CUDA wheel
pip install "tessera_embeddings[inference]"
```

For fully reproducible GPU deploys, generate a pinned constraints file in
your deployment repo:

```bash
# Pin uv.lock's torch version; pyproject.toml asks for any torch, which would take the index's newest.
echo "torch==2.14.1" > torch-pin.txt
uv pip compile pyproject.toml \
    --extra inference --extra prefect --extra aws \
    --python-platform linux --python-version 3.12 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --no-sources --constraint torch-pin.txt \
    -o constraints-cu130.txt

# Verify, every time. The filename is not a guarantee; each line fails if its pin is wrong:
grep -x 'torch==2.14.1+cu130' constraints-cu130.txt
grep -E '^nvidia-cuda-runtime==13\.' constraints-cu130.txt
```

**No `--index-strategy unsafe-best-match` here, deliberately.** It searches every index and takes
the best *version* found anywhere, so PyPI's newest torch, built for whatever CUDA PyPI defaults
to, can beat the CUDA index and land a pin the filename does not describe. uv's default
`first-index` keeps torch on the CUDA index. `--torch-backend cu130` (uv 0.11.28+) is the explicit
alternative and needs no `--extra-index-url`.
Either way, grep the output — the filename guarantees nothing.

That file belongs in your deployment repo alongside your Dockerfiles, not
in this OSS library.

### MPS (Apple Silicon GPU)

Untested. CPU is the supported laptop path.

## Platform notes

### Linux x86_64

The blessed deployment platform. For GPU production, follow the explicit
two-step install in the [CUDA section above](#cuda): install
`torch==...+cu130` with `--index-url` first, then install the package.

### macOS arm64 (Apple Silicon)

CPU inference is the supported path.

### Windows

Use WSL2. We don't ship Windows-native wheels.

## Python versions

`pyproject.toml` declares `requires-python = ">=3.12"`. CI tests 3.12 and
3.13. Earlier versions are not supported.

## Verifying the install

```bash
uv run python scripts/diagnostic/check_env.py
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
