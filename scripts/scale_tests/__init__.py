"""Scale-test harness for the global TESSERA embeddings store.

Standalone benchmarks (NOT pytest) that answer the PENDING decisions in
``context_docs/decisions/008-global-store-architecture.md``. See
``context_docs/design/global-store-test-plan.md`` (what/why) and
``global-store-test-impl-spec.md`` (how), plus ``README.md`` here for how to run.

Run from the ``scripts/`` directory so ``scale_tests`` is importable by the
spawned worker processes::

    cd scripts
    uv run python -m scale_tests.t0_smoke --run-id dev --backend local --scale tiny
"""
