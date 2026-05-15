"""Local-machine provider.

Single-node fallbacks for tests, demos, and the plain runner. The
``ray_cluster`` and (Phase 6) ``local_cluster`` context managers wrap
``ray.init`` / ``ray.shutdown`` and ``LocalCluster`` so callers can use
the same orchestration code paths regardless of substrate.
"""
