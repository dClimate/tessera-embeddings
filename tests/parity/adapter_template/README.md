# Adapter parity template

Community-contributed orchestrator adapters (Dagster, Airflow, Argo,
Kubeflow, …) must include a parity test before merge. This directory
is the starter template: copy it to a sibling directory named after
your adapter and fill in the body.

## Steps

1. Copy this directory::

       cp -r tests/parity/adapter_template tests/parity/<adapter_name>

2. Implement your adapter under
   ``src/tessera_embeddings/orchestration/<adapter_name>/`` with the
   same domain-function-delegation pattern as
   ``orchestration/prefect/``.

3. Edit ``test_<adapter_name>_parity.py`` (rename from the template):
   replace the placeholder calls with your adapter's flow / pipeline
   invocation. Compare to the same domain function / plain-runner
   path the bundled tests do.

4. Run::

       uv run pytest tests/parity/<adapter_name>/ -m parity

5. PR review checklist:
   - [ ] Architecture rules pass (
     ``python -m tessera_embeddings.architecture_tests --source src/`` returns 0).
   - [ ] Parity test passes locally.
   - [ ] CI workflow updated to include the new test directory.
