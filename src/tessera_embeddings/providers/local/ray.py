"""Local-machine Ray provider.

Single-node ``ray.init``/``ray.shutdown`` as a context manager, so orchestration code
written against :func:`tessera_embeddings.providers.aws.ray.ray_cluster` runs unchanged on
a laptop or in CI.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

import ray

LOCAL_ADDRESS_SENTINEL = "local"
"""Sentinel yielded by :func:`ray_cluster` for single-node mode.

Orchestration code branching on substrate (e.g. skipping AWS-only teardown) matches it.
"""


def _ray_bin() -> str:
    """Return the ray executable path, preferring the current venv's copy."""
    found = shutil.which("ray")
    if found:
        return found
    # pytest invoked as .venv/bin/pytest without activating the venv leaves .venv/bin off
    # PATH, so fall back to the sibling of sys.executable.
    candidate = Path(sys.executable).parent / "ray"
    if candidate.exists():
        return str(candidate)
    return "ray"


@contextlib.contextmanager
def ray_cluster(
    *,
    num_cpus: int | None = None,
    num_gpus: int | None = None,
    include_dashboard: bool = False,
) -> Iterator[str]:
    """Single-node local Ray for demos, tests, and the plain runner.

    ``ignore_reinit_error=True`` makes re-entry into an already-initialized runtime a no-op,
    so the context manager is safe to nest in tests.

    Args:
        num_cpus: CPU count exposed to Ray. ``None`` lets Ray detect.
        num_gpus: GPU count exposed to Ray. ``None`` lets Ray detect; ``0`` forces CPU-only.
        include_dashboard: Start the Ray dashboard on localhost. Off by default — it pulls
            in aiohttp and grpcio, which we don't want as hard requirements.

    Yields:
        :data:`LOCAL_ADDRESS_SENTINEL` (the string ``"local"``).
    """
    # Kill any stale Ray node so the new node gets a clean GCS port file.
    subprocess.run([_ray_bin(), "stop", "--force"], capture_output=True)

    ray_tmpdir = tempfile.mkdtemp(prefix="ray_t_", dir="/tmp")
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            ray.init(
                num_cpus=num_cpus,
                num_gpus=num_gpus,
                include_dashboard=include_dashboard,
                ignore_reinit_error=True,
                _temp_dir=ray_tmpdir,
            )
            last_exc = None
            break
        except RuntimeError as exc:
            last_exc = exc
            ray.shutdown()
            time.sleep(3 * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    try:
        yield LOCAL_ADDRESS_SENTINEL
    finally:
        ray.shutdown()
        shutil.rmtree(ray_tmpdir, ignore_errors=True)
