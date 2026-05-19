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
commands above work identically with `uv pip`:

```bash
uv pip install tessera_embeddings[inference]
```

## TL;DR for contributors

```bash
git clone https://github.com/dClimate/tessera-embeddings
cd tessera-embeddings
uv sync --all-extras                      # uv.lock — CPU torch, all extras, dev tools
```

Or with pip-compile lock files for a specific environment:

```bash
# CPU (laptop, CI, plain runner)
pip install -r lock/inference-cpu.lock
pip install --no-deps -e .

# GPU — Linux x86_64, CUDA 12.1
pip install -r lock/inference-cu121.lock
pip install --no-deps -e .

# Development (adds pytest, ruff, mypy)
pip install -r lock/dev.lock
pip install --no-deps -e .
```

## Why multiple lock files?

CUDA torch wheels are platform-specific. A single universal lock
file can't satisfy both "macOS arm64 contributor laptop" and
"Linux x86_64 GPU AMI." Forcing one would either break locally or
ship the CPU wheel to production, silently halving inference throughput.

```
                  ┌────────────────────────┐
                  │   pyproject.toml       │
                  │ (deps + extras)        │
                  └────────────┬───────────┘
                               │
              ┌────────────────┼────────────────┬─────────────────┐
              │                │                │                 │
              ▼                ▼                ▼                 ▼
  lock/inference-cpu.lock  lock/inference-cu121.lock  lock/dev.lock  uv.lock
  (universal)              (Linux x86_64 only)        (superset of   (alias of CPU
                                                       cpu + tools)   for `uv sync`)
```

## The lock files

| File | Platforms | Torch | Purpose |
|------|-----------|-------|---------|
| `uv.lock` | universal | CPU | What `uv sync` reads by default. |
| `lock/inference-cpu.lock` | Linux x86_64, macOS arm64/x86_64, Linux arm64 | CPU | Laptop dev, CI, CPU inference, the plain runner. |
| `lock/inference-cu121.lock` | Linux x86_64 only | CUDA 12.1 | Production GPU AMI (Tessera was trained against this stack). |
| `lock/dev.lock` | universal | CPU | `inference-cpu.lock` + pytest, ruff, mypy, hypothesis. CI uses this. |

## Picking a lock file

```
What are you running?
│
├── CPU inference (laptop, CI, plain runner) ───► uv sync (uv.lock)
│
├── GPU inference (production, Linux x86_64) ───► lock/inference-cu121.lock
│
├── Apple Silicon GPU (MPS) ────────────────────► uv sync (uv.lock); see "MPS status" below
│
└── Local development (tests, lint, types) ─────► uv sync (uv.lock) or lock/dev.lock
```

## CUDA

`[inference]` defaults to CPU torch (the `[tool.uv.sources]` entry in
`pyproject.toml` pins the CPU index so `uv sync` never accidentally
pulls CUDA wheels on a laptop).

For GPU production:

```bash
# Use the pre-solved CUDA lock file (recommended):
grep -v '^gdal==' lock/inference-cu121.lock | pip install -r /dev/stdin

# Or override the torch index at install time:
pip install tessera_embeddings[inference] \
    --extra-index-url https://download.pytorch.org/whl/cu121
```

### MPS (Apple Silicon GPU)

Untested. Anecdotally torch's MPS backend works for forward passes but
specific ops in the Tessera model haven't been verified. CPU is the
supported laptop path.

## GDAL

`gdal` is declared as an unpinned dependency. `pip install tessera_embeddings`
will pull a binary wheel from PyPI that bundles its own libgdal — no system
library required for most platforms. If you are on a platform without a
pre-built wheel (uncommon), you will need system `libgdal` headers and
`gdal-config` available before `pip install`.

## Platform notes

### Linux x86_64

The blessed deployment platform. Both CPU and CUDA 12.1 lock files resolve.
Production AMIs use `lock/inference-cu121.lock`.

### macOS arm64 (Apple Silicon)

`lock/inference-cpu.lock` resolves. CPU inference is the supported path.

### macOS x86_64 (Intel)

`lock/inference-cpu.lock` resolves. Same caveats as Apple Silicon, minus MPS.

### Linux arm64 (Graviton)

`lock/inference-cpu.lock` resolves. Useful for cheap CI runners; not tested
on real workloads.

### Windows

Use WSL2. We don't ship Windows-native wheels.

## Regenerating lock files

Triggered by any change to `pyproject.toml`. CI's `lock-check.yml` catches
drift. To regenerate locally:

```bash
bash lock/lock.sh
```

What it does:

```
lock/lock.sh
├── uv lock                                           → uv.lock (universal)
├── uv pip compile pyproject.toml \
│       --extra inference --extra prefect --extra aws \
│       --universal --no-strip-extras \
│       -o lock/inference-cpu.lock
├── uv pip compile pyproject.toml \
│       --extra inference --extra prefect --extra aws \
│       --python-platform linux \
│       --extra-index-url https://download.pytorch.org/whl/cu121 \
│       --index-strategy unsafe-best-match \
│       --no-sources \
│       -o lock/inference-cu121.lock
└── uv pip compile pyproject.toml \
        --extra inference --extra prefect --extra aws --group dev \
        --universal -o lock/dev.lock
```

Commit all four lock files together — partial regeneration is a common
source of "works on my machine" bugs.

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

If `cuda=True` on a CPU-only host, you installed the wrong lock file.
