# GPU tests

Tiny tests that exercise GPU-specific code paths (device placement,
CUDA memory handling). Marked `@pytest.mark.gpu`. Skip cleanly when
no GPU is available — they are not failures.

## Design constraints

GPU CI minutes cost real money, so this directory is held to
deliberately tight bounds:

* **Tiny inputs only.** Single chunk ≤ 256×256, ≤ 4 timesteps,
  ≤ 2 bands.
* **Tiny model.** Use `tests/fixtures/checkpoints/tiny_model.pt`
  (same architecture as production, minimal dims). Loads in < 5 s on
  CPU and < 2 s on GPU.
* **Correctness, not performance.** Compare GPU output to a CPU
  reference within numerics tolerance. No wall-clock assertions —
  shared runners flake.
* **No sustained GPU holding.** Each test acquires GPU, runs < 30 s,
  releases. No multi-minute actor pools.
* **Total CI minutes ≤ 2 per run.** Budget accordingly.

Scientific correctness lives in CPU tests. The job of `tests/gpu/` is
narrow: prove the GPU code path actually exercises a GPU.

## Running

```bash
# Skips on machines without CUDA; runs the full suite on machines with one
uv run pytest -m gpu
```

The `gpu` marker is filtered out of the default invocation.

## What does NOT go here

* Inference correctness checks. CPU is the source of truth.
* Long-running training tests. Out of scope for this package.
* Anything not actually using a GPU.
