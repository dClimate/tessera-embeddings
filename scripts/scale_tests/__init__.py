"""Scale-test harness for the global TESSERA embeddings store.

Standalone benchmarks (NOT pytest) that answered the PENDING decisions in
``context_docs/decisions/008-global-store-architecture.md``, where every D1-D9 is now FIRM
and annotated with the run that confirmed it. These scripts are the method of record for
what each test measures; ``context_docs/storage/icechunk-api-ledger.md`` carries the
verified icechunk and zarr signatures they were built against. ``README.md`` here is how
to run them.

Run from the ``scripts/`` directory so ``scale_tests`` is importable by the
spawned worker processes::

    cd scripts
    uv run python -m scale_tests.t0_smoke --run-id dev --backend local --scale tiny
"""
