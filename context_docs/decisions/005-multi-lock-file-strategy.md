# 005 — Lock file strategy

**Status:** Superseded (v0.1.0 → simplified)

## Original decision (multi-lock)

Shipped four lock files (`uv.lock`, `lock/inference-cpu.lock`,
`lock/inference-cu121.lock`, `lock/dev.lock`) to cover laptop, production
GPU, and dev profiles. See git history for the original rationale.

## Current decision (single lock)

Ship only `uv.lock` at repo root. `uv sync --all-extras` covers all
development use. Production GPU deployment is a downstream ops concern.

**Rationale for simplification:**

1. `inference-cpu.lock` was a pip-format mirror of `uv.lock` — same
   packages, two representations. The original justification (the GDAL
   system-library strip workaround) was eliminated when `gdal` was
   unpinned and switched to a binary wheel.

2. `inference-cu121.lock` is a platform-specific production artifact for
   a specific CUDA version on Linux x86_64 only. It belongs in a deployment
   repo alongside Dockerfiles and AMI bake scripts, not in the OSS library.
   OSS canonical practice: document how to generate a CUDA lock; don't ship one.

3. `lock/dev.lock` was identical in content to `uv.lock` — maintaining a
   manual pip-format mirror of the authoritative lock added drift risk with
   no capability gain.

## Consequences

- `uv.lock` is the single source of truth. Run `uv lock` after any change
  to `pyproject.toml` and commit the updated file.
- CI's `lock-check.yml` asserts `uv.lock` is up to date via `uv lock --check`.
- GPU production lock files are generated and owned by downstream consumers.

## Related

- [`docs/environment-setup.md`](../../docs/environment-setup.md)
