"""Prefect ``@flow`` definitions.

Each flow file binds a piece of domain logic to a substrate. The shape is consistent across files:

1. Validate / parse parameters.
2. Enter a provider context manager (``providers.aws.dask.ecs_cluster``,
   ``providers.aws.ray.ray_cluster``, or the local equivalents).
3. Inside the context, call an inner ``@flow`` with ``.with_options(task_runner=...)`` so the task
   runner binds late to the cluster's scheduler address. A Prefect idiom — do not try to collapse
   it into a single flow.
4. Submit a task shell that delegates to a domain function.
5. Return the task's result.
"""
