"""Inference-stage profiling: the Ray GPU cluster, plus output-equivalence gates.

- :mod:`.observe_cluster` — live GPU/DCGM/host-RAM pollers over SSM and post-hoc CloudWatch
  rollups for a Ray inference fleet.
- :mod:`.compare_outputs` — the ADR-012 equivalence gate over staged embeddings (int8 + scales).
- :mod:`.compare_coarsened_stores` — bit-identity / drift comparison between two coarsened
  embedding stores.

See ``README.md`` in this directory for the RAM-spike workflow and baselines.
"""
