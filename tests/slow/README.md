# Slow tests

> **STATUS: empty, and no occupant is planned.** The full plain-runner end-to-end this tier was
> created for is **verified by running the quickstart by hand, not by an automated test** —
> [ADR 024](../../context_docs/decisions/024-the-single-path-end-to-end-is-the-quickstart-run.md),
> a final decision. The `xfail` stub that used to stand for it, and the nightly workflow pointed at
> that stub, are both deleted.

Tests that take more than about two minutes to run. Marked `@pytest.mark.slow`, which the default
`addopts` deselect, so nothing here runs unless a job or a person opts in.

**The tier is a category, not a plan.** It is kept because "this test is genuinely slow" is a real
thing that may need a home again, and the marker stays declared in `pyproject.toml` so it can be
used without ceremony. There is no CI job that runs it today, and adding one is part of the cost of
putting a test here.

## Running locally

```bash
uv run pytest -m slow          # nothing, today
uv run pytest -m "slow or parity"
```

## What does NOT go here

* Anything that requires a GPU. See `tests/gpu/`, and
  [ADR 023](../../context_docs/decisions/023-the-cuda-path-is-verified-by-hand.md).
* Tests that *could* be quick but are slow because they hit live services. Use cassettes; if they
  are still slow, find out why. Two tests reached 61 s and 10 s that way and neither needed to.
