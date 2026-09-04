# 023 — The pipelined CUDA path is verified by hand, not by CI

**Status:** Accepted (2026-08-25, repo owner). Recorded as an ADR 2026-09-03.

## Context

`inference/inference.py`'s `_pipelined_gpu_loop` is the hot path of the whole inference stage:
pinned host buffers, CUDA events, a two-deep drain, and backbone stream ordering. It is what took
per-worker throughput from 1× to about 2–2.8× (see
[`../inference/inference-on-gpus.md`](../inference/inference-on-gpus.md) §2), and it is the one part
of the pipeline whose correctness depends on getting concurrency between the host and the device
right.

Exactly one test covers it —
`tests/unit/inference/test_inference_loop.py::TestRunInference::test_pipelined_matches_serial`,
which runs the pipelined loop against the serial one and compares outputs. It is gated on
`torch.cuda.is_available()`.

**No CI runner available to this project has a GPU, and one will not be provisioned.** So that test
skips on every run, in both Python versions of the `unit.yml` matrix. It is the `1 skipped` that
appears in every unit run.

This was found during the 2026-08 test-suite review, which asked whether the `gpu` marker and the
empty `tests/gpu/` tier should simply be deleted.

## Decision

**The gap is accepted. The pipelined CUDA path is verified by hand on a documented trigger, and that
fact is written down in the three places a reader will meet it.**

Four things follow, and each is a consequence rather than a preference:

**The test keeps its `skipif` and must NOT take `@pytest.mark.gpu`.** The default `addopts` in
`pyproject.toml` deselect the `gpu` marker, so a marked test would stop being collected and would
vanish from the run entirely. The `skipif` keeps it collected, so it reports as a skip on every CI
run — **and that skip is the only standing signal that the gap exists.** Losing it in exchange for a
used marker would be a bad trade: an accepted gap that stops being visible becomes an assumed one.

**The gap is stated where a reader hits it**, not only here: in `tests/README.md`'s Roadmap, in the
test class's own docstring, and in
[`../test-suite-streamlining.md`](../test-suite-streamlining.md). A skip that nobody explains reads
as "not applicable here".

**Verification has a named trigger — and the environment is the part to get right.**

> **`uv sync --all-extras --frozen` alone is NOT sufficient, and fails silently.** The lockfile
> pins torch to the **CPU wheel registry** (`download.pytorch.org/whl/cpu`) on every platform, so
> on a CUDA machine that command installs `torch==2.12.0+cpu`, `torch.cuda.is_available()` returns
> False, and the one test this ADR rests on **skips** — reporting exactly as it does in CI, which
> is to say reporting nothing. An operator following a bare `--frozen` would believe they had
> verified the path and would not have.

So the run has two steps, and the first is a check rather than an instruction:

```bash
# 1. Get a CUDA build of torch. The lockfile will not give you one; install it outside the
#    frozen resolution, matching the driver on the box.
uv sync --all-extras --frozen
uv pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124  # or the cuXXX matching the host

# 2. CONFIRM the device is visible before trusting anything the test says.
uv run python -c "import torch; assert torch.cuda.is_available(), 'CPU torch — the test will skip'; print(torch.__version__)"

# 3. Then run it. A PASS is the verification; a SKIP means step 1 did not take.
uv run pytest tests/unit/inference/test_inference_loop.py -k pipelined -v
```

**Read the outcome, not the exit code.** `pytest` exits 0 on a skip, so "the command succeeded" is
not evidence. The only result that verifies anything here is a **passed**.

Run it **after any change to the pipelined loop, the actor pool, or the chunk-staging path, and once
before a campaign starts.** Manual verification with no trigger attached does not happen.

**`tests/gpu/` and its README stay, as a dormant spec.** That README is not junk — it is a written
contract for what a GPU tier would look like (tiny inputs, tiny model, compare GPU output against a
CPU reference, under two CI minutes per run). It describes a design that would be built if a runner
ever appears, so the directory says that rather than reading as work in flight, and the `gpu` marker
stays declared in `pyproject.toml` so those instructions remain valid.

## Rejected alternatives

**Delete the `gpu` marker and `tests/gpu/`.** Tidier, and it destroys a written spec plus the one
place the tier's shape is recorded. The cost of keeping them is two files that do nothing; the cost
of deleting them is re-deriving the design if a runner ever appears.

**Add `@pytest.mark.gpu` to the test so the marker has a user.** This is the tempting one and it is
actively harmful: the marker is deselected by default, so the test would stop appearing in runs at
all and the standing signal would be gone. Tidiness at the cost of the only visible evidence of the
gap.

**Provision a GPU CI runner.** Out of reach for this project. If that changes, this ADR is what to
supersede.

**Write a CPU-only substitute for the pipelined loop.** The behaviours at issue — pinned-buffer
reuse, CUDA event ordering, stream interleaving — do not exist without a device. A CPU stand-in
would pass while proving nothing, which is worse than a visible skip.

## Consequences

**The honest reading of this repository's coverage number is "84% of the codebase verified
automatically, one hot path verified by hand on a documented trigger."** That is a weaker guarantee
than 84% alone suggests, and it is written down so the number is not read as more than it is.

**A change to the pipelined loop that breaks it will reach a campaign unless someone runs the
command.** That is the residual risk this ADR accepts, and the trigger list above is the whole of the
mitigation.

**If a GPU runner ever becomes available**, the work is already specified: `tests/gpu/README.md` is
the contract, and this test moves under the `gpu` marker at that point — not before.

## Related

- [`../test-suite-streamlining.md`](../test-suite-streamlining.md) — the review this came out of
- [`../inference/inference-on-gpus.md`](../inference/inference-on-gpus.md) §2 — what the pipelined
  loop bought, and §7 for the RAM bound it is constrained by
- [ADR 012](012-validated-equivalence-for-inference-outputs.md) — how inference outputs are compared
  when a change is not bit-identical
