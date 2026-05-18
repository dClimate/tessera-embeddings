"""Prefect orchestration layer.

This subtree is the **only** place in the package where ``import prefect``
is allowed. The architecture-tests in ``tests/architecture/`` enforce
that rule.

* ``tasks/`` — thin ``@task`` shells that pull ``client`` from
  :func:`dask.distributed.get_client` and ``log`` from
  :func:`prefect.get_run_logger`, then delegate to a domain function.
  Roughly 20 LOC per task.
* ``flows/`` — ``@flow`` files that provision substrate (Ray / Dask)
  via the ``providers/`` package and submit one or more task shells.

The two-flow pattern (outer ``@flow`` enters the cluster ctx manager,
inner ``@flow`` is invoked via ``.with_options(task_runner=...)``) is
intentional and load-bearing — see each flow's docstring.
"""
