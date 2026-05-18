# Environment setup

We ship multiple lock files so different deployments get the right
binary wheels without surprises. Pick the one that matches your
target.

## TL;DR

```bash
uv sync                                    # laptop CPU (universal)
uv sync --frozen -r inference-cu121.lock   # production GPU (Linux x86_64, CUDA 12.1)
uv sync --frozen --group dev --group inference --group prefect --group aws
                                            # full dev environment
```

## Why multiple lock files?

CUDA torch wheels are platform-specific. A single universal lock
file can't satisfy both "macOS arm64 contributor laptop" and
"Linux x86_64 GPU AMI." Forcing one would either break locally or
ship the CPU wheel to production, silently halving inference
throughput.

```
                  ┌────────────────────────┐
                  │   pyproject.toml       │
                  │  (deps + groups)       │
                  └────────────┬───────────┘
                               │
              ┌────────────────┼────────────────┬─────────────────┐
              │                │                │                 │
              ▼                ▼                ▼                 ▼
       inference-cpu.lock  inference-cu121.lock  dev.lock      uv.lock
       (universal)         (Linux x86_64 only)  (superset of   (alias of CPU
                                                 cpu + tools)   for `uv sync`)
```

## The four lock files

| File | Platforms | Torch | Purpose |
|---|---|---|---|
| `uv.lock` | universal | CPU | What `uv sync` reads by default. Aliases `inference-cpu.lock`. |
| `inference-cpu.lock` | Linux x86_64, macOS arm64/x86_64, Linux arm64 | CPU | Laptop dev, CI, CPU inference, the plain runner. |
| `inference-cu121.lock` | Linux x86_64 only | CUDA 12.1 | Production GPU AMI (Tessera was trained against this stack). |
| `dev.lock` | universal | CPU | `inference-cpu.lock` + pytest, ruff, mypy, hypothesis. CI uses this. |

## Picking a lock file

```
What are you running?
│
├── ingestion only (no torch) ──────────────────► uv sync (uv.lock)
│
├── CPU inference (laptop, CI, plain runner) ───► uv sync (uv.lock)
│
├── GPU inference (production, Linux x86_64) ───► uv sync --frozen -r inference-cu121.lock
│
├── Apple Silicon GPU (MPS) ────────────────────► uv sync (uv.lock); see "MPS status" below
│
├── Local development (tests, lint, types) ─────► uv sync --frozen -r dev.lock
│
└── Building docs only ─────────────────────────► uv sync --group dev
```

## Optional dependency groups

`pyproject.toml` declares four groups; pick the ones you need
alongside the lock file:

| Group | Adds |
|---|---|
| `inference` | torch, ray |
| `prefect` | prefect, prefect-dask, prefect-aws |
| `aws` | boto3, botocore, s3fs, dask-cloudprovider, prefect-aws |
| `dev` | pytest, pytest-recording, hypothesis, ruff, mypy |

```bash
uv sync --group inference --group prefect       # only what you need
uv sync --group dev --group inference --group prefect --group aws  # everything
```

## Platform notes

### Linux x86_64

The blessed deployment platform. Both `inference-cpu.lock` and
`inference-cu121.lock` resolve. Production AMIs use the CU121 lock
to match the GPU build of torch.

### macOS arm64 (Apple Silicon)

`inference-cpu.lock` resolves. Inference runs on CPU; **MPS is
untested**. Anecdotally torch's MPS backend works for forward passes
but specific ops in the Tessera model haven't been verified. Treat
MPS as "maybe — file an issue if you try it." For laptop demos, CPU
is the supported path.

### macOS x86_64 (Intel)

`inference-cpu.lock` resolves. Same caveats as Apple Silicon, minus
MPS.

### Linux arm64 (Graviton)

`inference-cpu.lock` resolves. Useful for cheap CI runners. **Not
tested** on real workloads.

### Windows

Use WSL2. We don't ship Windows-native wheels; the surface area of
testing native Windows isn't worth it given WSL2 is one command
away.

## Regenerating lock files

Triggered by any change to `pyproject.toml`. CI's `lock-check.yml`
catches drift. To regenerate locally:

```bash
bash scripts/lock.sh
```

What it does:

```
scripts/lock.sh
├── uv lock                                          → uv.lock (universal)
├── uv pip compile pyproject.toml \
│       --group inference --group prefect --group aws \
│       --universal --no-strip-extras \
│       -o inference-cpu.lock
├── uv pip compile pyproject.toml \
│       --group inference --group prefect --group aws \
│       --python-platform linux \
│       --extra-index-url https://download.pytorch.org/whl/cu121 \
│       --index-strategy unsafe-best-match \
│       -o inference-cu121.lock
└── uv pip compile pyproject.toml \
        --group inference --group prefect --group aws --group dev \
        --universal -o dev.lock
```

Commit all four locks together — partial regeneration is a common
source of "works on my machine" bugs.

## Python versions

`pyproject.toml` declares `requires-python = ">=3.12"`. CI tests
3.12, 3.13, and 3.14. Earlier versions are not supported (we use
`@final`, PEP-695 type-parameter syntax, and `tomllib` from the
stdlib).

If a transitive dep doesn't have a wheel for 3.14 yet, the matrix
leg gracefully skips affected tests via `pytest.importorskip`.
We don't drop Python versions to chase wheel availability.

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
