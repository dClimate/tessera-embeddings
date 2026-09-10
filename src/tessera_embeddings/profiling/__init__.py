"""Profiling harnesses for the pipeline's two compute-heavy stages.

Ingest (Dask on Fargate) and inference (Ray on EC2 GPUs) saturate on different
resources, so each has its own harness:

- :mod:`tessera_embeddings.profiling.ingest` — the Dask **scheduler** watcher, the
  CloudWatch Logs-Insights query pack, and the run-dossier assembler.
- :mod:`tessera_embeddings.profiling.inference` — the Ray GPU/RAM cluster observer
  plus the output-equivalence gates.

Every tool is a standalone command exposing ``main(argv) -> int``, installed as a
console script (``te-*``; see ``[project.scripts]``) and equally runnable as
``python -m tessera_embeddings.profiling.<stage>.<tool>``. Nothing in the library
imports this subpackage, so a normal import of ``tessera_embeddings`` never pulls in a
cloud SDK. The AWS-facing tools need the ``aws`` extra for boto3. See ``README.md``
here for which harness to reach for.
"""
