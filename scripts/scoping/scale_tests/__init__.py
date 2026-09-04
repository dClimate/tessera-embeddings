"""Scale-test harness for the global TESSERA embeddings store.

Standalone benchmarks (NOT pytest) that answered the PENDING decisions in
``context_docs/decisions/008-global-store-architecture.md``, where every D1-D9 is now FIRM
and annotated with the run that confirmed it. These scripts are the method of record for
what each test measures; ``context_docs/storage/icechunk-api-ledger.md`` carries the
verified icechunk and zarr signatures they were built against. ``README.md`` here is how
to run them.

Run from the REPOSITORY ROOT — the modules import each other as
``scripts.scoping.scale_tests.x``, so the root must be on ``sys.path``, which ``python -m``
does for the working directory. ``scripts/`` and ``scripts/scoping/`` have no ``__init__.py``
and resolve as namespace packages. The cold-run subprocess inherits this working directory,
so starting anywhere else breaks the cold phases rather than the imports.
spawned worker processes::

    cd scripts
    uv run python -m scripts.scoping.scale_tests.t0_smoke --run-id dev --backend local --scale tiny
"""
