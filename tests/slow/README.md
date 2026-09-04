# Slow tests

> **STATUS: empty, and no occupant is planned.** The full plain-runner end-to-end is **verified by
> running the quickstart by hand, not by an automated test** —
> [ADR 023](../../context_docs/decisions/023-the-single-path-end-to-end-is-the-quickstart-run.md).

Tests that take **more than 30 seconds** to run. Marked `@pytest.mark.slow`, which the default
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

* Anything that requires a GPU. See `tests/gpu/`, and `tests/README.md` Roadmap 2.
* Tests that *could* be quick but are slow because they hit live services. Use cassettes; if they
  are still slow, find out why. Two tests reached 61 s and 10 s that way and neither needed to.
