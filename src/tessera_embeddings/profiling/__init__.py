"""Profiling harnesses for the pipeline's two compute-heavy stages.

The ingest stage (Dask on Fargate) and the inference stage (Ray on EC2 GPUs)
saturate on completely different resources, so each has its own harness:

- :mod:`tessera_embeddings.profiling.ingest` — the Dask **scheduler** watcher,
  the CloudWatch Logs-Insights query pack, and the run-dossier assembler.
- :mod:`tessera_embeddings.profiling.inference` — the Ray GPU/RAM cluster
  observer plus the output-equivalence gates.

Every tool is a standalone command exposing ``main(argv) -> int``. They are
installed as console scripts (``te-*``; see ``[project.scripts]``) and are
equally runnable as ``python -m tessera_embeddings.profiling.<stage>.<tool>``.
Nothing in the library imports this subpackage — it is operator tooling, loaded
only when a command runs, so a normal import of ``tessera_embeddings`` never
pulls a cloud SDK in on its account.

The AWS-facing tools need the ``aws`` extra (``pip install
tessera_embeddings[aws]``) for boto3. See ``README.md`` here for which harness
to reach for.
"""
