"""Prefect orchestration layer.

The **only** place in the package where ``import prefect`` is allowed; the
architecture tests in ``tests/architecture/`` enforce that.

* ``tasks/`` — thin ``@task`` shells (~20 LOC) that pull ``client`` from
  :func:`dask.distributed.get_client` and ``log`` from
  :func:`prefect.get_run_logger`, then delegate to a domain function.
* ``flows/`` — ``@flow`` files that provision substrate (Ray / Dask) via the
  ``providers/`` package and submit one or more task shells.

The two-flow pattern (outer ``@flow`` enters the cluster context manager, inner
``@flow`` invoked via ``.with_options(task_runner=...)``) is load-bearing — see each
flow's docstring.
"""
