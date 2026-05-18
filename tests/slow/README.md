# Slow tests

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
