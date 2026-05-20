"""Local-machine Ray provider.

Single-node ``ray.init`` / ``ray.shutdown`` wrapped as a context
manager so the same orchestration code that uses
:func:`tessera_embeddings.providers.aws.ray.ray_cluster` for AWS works
unchanged on a developer laptop or in CI.
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
"""Sentinel yielded by :func:`ray_cluster` to indicate single-node mode.

Orchestration code that branches on substrate (e.g. to skip cluster
teardown actions that only apply to AWS) can match against this value.
"""


def _ray_bin() -> str:
    """Return the ray executable path, preferring the current venv's copy."""
    found = shutil.which("ray")
    if found:
        return found
    # When pytest is invoked via .venv/bin/pytest without activating the venv,
    # PATH may not include .venv/bin.  Fall back to the sibling of sys.executable.
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

    Calls :func:`ray.init` on enter and :func:`ray.shutdown` on exit.
    Idempotent — ``ignore_reinit_error=True`` means re-entering an
    already-initialized Ray runtime is a no-op rather than an error,
    which keeps the context manager safe to nest in tests.

    Args:
        num_cpus: CPU count exposed to Ray. ``None`` lets Ray detect.
        num_gpus: GPU count exposed to Ray. ``None`` lets Ray detect.
            Set to ``0`` for explicit CPU-only mode.
        include_dashboard: Start the Ray dashboard on localhost. Off by
            default — the dashboard pulls in extra deps (aiohttp,
            grpcio) that we don't want as a hard requirement.

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
