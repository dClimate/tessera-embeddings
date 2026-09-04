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

> **Corrected 2026-09-03: this said `cu121`, and `torch==2.6.0+cu121` does not exist.** Verified
> against the indexes: 2.6.0 publishes `cu118`, `cu124` and `cu126`, not `cu121`, so the command
> below failed to resolve. **Check the index before pinning a CUDA build** —
> `https://download.pytorch.org/whl/<cuXXX>/torch/` lists what is actually there — and pick the one
> your driver supports; `cu118` is the other published option for this version.

**The CUDA pin does not cover every Python this package advertises.** Three statements about
Python support currently disagree, and it is worth seeing them together:

| where | says |
|---|---|
| `pyproject.toml` `requires-python` | `>=3.12` — open-ended |
| `pyproject.toml` classifiers | 3.12, 3.13 **and 3.14**, explicitly |
| `unit.yml` CI matrix | 3.12 and 3.13 only |
| this page's GPU install | cu124 publishes `cp39`-`cp313`; **no `cp314`** |

So the package advertises a Python version that CI does not exercise and on which the documented
GPU install cannot resolve — cu124 tops out at `torch 2.6.0`, so there is no newer build on that
index to move to. Verified against the index listings; `cu126` and `cu128` do publish `cp314`
wheels and are the route if you need 3.14.

**Treat Python 3.12-3.13 with cu124 as the supported GPU combination** — the one CI covers and
these instructions are written for. Whether the 3.14 classifier should stay is a support-scope
question for the maintainers, not a documentation one, and it is deliberately left as it is here.

```bash
# 1. Install CUDA torch from the pytorch index (CPython 3.9-3.13)
pip install "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124

# 2. Install the package — pip sees torch already satisfied, keeps the CUDA wheel
pip install "tessera_embeddings[inference]"
```

For fully reproducible GPU deploys, generate a pinned constraints file in
your deployment repo:

```bash
uv pip compile pyproject.toml \
    --extra inference --extra prefect --extra aws \
    --python-platform linux --python-version 3.12 \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    --no-sources \
    -o constraints-cu124.txt

# Verify, every time. The filename is not a guarantee:
grep -E '^(torch|nvidia-cuda-runtime)' constraints-cu124.txt
# torch==2.6.0+cu124
# nvidia-cuda-runtime-cu12==12.4.127
```

> **Corrected 2026-09-03: this command used to carry `--index-strategy unsafe-best-match`, and with
> it the output was not a CUDA 12.4 pin at all.** Run as documented it resolved
> **`torch==2.14.0` with `nvidia-cuda-*==13.0.x`** — the newest release from PyPI, CUDA 13, written
> to a file called `constraints-cu124.txt`. Three things combined: `torch` is unpinned in
> `pyproject.toml`, `--no-sources` drops the `[tool.uv.sources]` mapping that otherwise holds torch
> to one index, and `unsafe-best-match` searches every index and takes the best *version* it finds
> anywhere. Dropping the flag restores uv's default `first-index`, which keeps torch on the CUDA
> index it was found on. Verified both ways by running the compile.
>
> On uv 0.11.28 and later, `--torch-backend cu124` does the same thing more explicitly and needs no
> `--extra-index-url`; either is fine, but **check the output** rather than trusting the filename.

That file belongs in your deployment repo alongside your Dockerfiles, not
in this OSS library.

### MPS (Apple Silicon GPU)

Untested. CPU is the supported laptop path.

## Platform notes

### Linux x86_64

The blessed deployment platform. For GPU production, follow the explicit
two-step install in the [CUDA section above](#cuda): install
`torch==...+cu124` with `--index-url` first, then install the package.

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
