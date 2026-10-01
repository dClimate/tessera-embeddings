"""Local-machine Dask provider.

Single-process :class:`dask.distributed.LocalCluster` wrapped as a context manager so
substrate-agnostic orchestration code can use the same idiom locally as it does on AWS.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from dask.distributed import Client, LocalCluster

from tessera_embeddings.providers import frisky as frisky_engine


@contextlib.contextmanager
def local_cluster(
    *,
    n_workers: int = 2,
    threads_per_worker: int = 2,
    memory_limit: str = "4GB",
    dashboard_address: str | None = None,
    frisky: bool = False,
) -> Iterator[LocalCluster]:
    """Start a single-machine Dask cluster.

    Args:
        n_workers: Number of worker processes.
        threads_per_worker: Threads per worker process.
        memory_limit: Per-worker memory cap (e.g. ``"4GB"``); ``None`` means no limit.
        dashboard_address: Address to bind the dashboard to (e.g. ``":0"`` for an ephemeral
            port). ``None`` disables the dashboard, which avoids pulling in optional
            ``bokeh``/HTTP deps in tests.
        frisky: Load Frisky onto the cluster before yielding it; compute then needs a client
            from :func:`tessera_embeddings.providers.frisky.connect`.

    Yields:
        The :class:`dask.distributed.LocalCluster`. The caller can pass
        ``cluster.scheduler_address`` to a :class:`Client` or use the cluster object directly.
    """
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
    )
    try:
        if frisky:
            with Client(cluster) as client:
                frisky_engine.hijack(client)
        yield cluster
    finally:
        cluster.close()
