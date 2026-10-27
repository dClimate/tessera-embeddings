"""Prefect ``@flow`` definitions.

Each flow file binds a piece of domain logic to a substrate. The shape is consistent across files:

1. Validate / parse parameters.
2. Enter a provider context manager (``providers.aws.dask.ecs_cluster``,
   ``providers.aws.ray.ray_cluster``, or the local equivalents).
3. Inside the context, run a task shell that delegates to a domain function: on the Dask cluster
   via ``_dask_lifecycle.run_on_cluster`` (the ingest flows), or called in the flow body (the Ray
   flows, whose task dispatches work to Ray actors).
4. Return the task's result.
"""
