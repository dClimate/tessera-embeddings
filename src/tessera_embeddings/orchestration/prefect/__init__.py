"""Prefect orchestration layer.

The **only** place in the package where ``import prefect`` is allowed; the
architecture tests in ``tests/architecture/`` enforce that.

* ``tasks/`` — thin ``@task`` shells (~20 LOC) that pull ``client`` from
  :func:`dask.distributed.get_client` and ``log`` from
  :func:`prefect.get_run_logger`, then delegate to a domain function.
* ``flows/`` — ``@flow`` files that provision substrate (Ray / Dask) via the
  ``providers/`` package and submit one or more task shells.

An ingest flow enters its cluster context manager and runs its task shell on that cluster
through ``flows._dask_lifecycle.run_on_cluster``, as a task run of the same flow run (see the
"One flow per ingest" section of this package's README).
"""
