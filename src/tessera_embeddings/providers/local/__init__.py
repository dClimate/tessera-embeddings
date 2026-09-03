"""Local-machine provider.

Single-node fallbacks for tests, demos, and the plain runner. ``ray_cluster`` and
``local_cluster`` wrap ``ray.init``/``ray.shutdown`` and ``LocalCluster`` as context
managers, so orchestration code is identical regardless of substrate.
"""
