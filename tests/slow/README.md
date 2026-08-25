# Slow tests

> **STATUS: empty, 2026-08-25.** The canonical occupant described below has not been written
> — it exists as `tests/parity/test_full_pipeline_parity.py`, an `xfail(strict=True)` stub
> raising `NotImplementedError`, filed under `parity/` because it needs both markers.
> `nightly.yml` was pointed at it and its **schedule is suspended** until the test is real.
> See `tests/README.md` → Roadmap 1.

Tests that take more than ~2 minutes to run. Marked
`@pytest.mark.slow`. Run nightly in CI and on demand locally.

The canonical occupant of this directory is the **full plain-runner
end-to-end** — rasterise ROI → S2 + S1 ingest → CPU inference →
assembly. ~30+ minutes against the Story County, IA quickstart AOI.

## Running locally

```bash
# Just the slow tests (full plain runner):
uv run pytest -m slow

# Full plain runner + parity contract:
uv run pytest -m "slow or parity"
```

## What does NOT go here

* Anything that requires GPU. See `tests/gpu/`.
* Tests that *could* be quick but are slow because they hit live
  services. Use cassettes; if they're still slow, find out why.
