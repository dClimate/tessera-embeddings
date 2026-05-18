"""Plain (orchestrator-free) runners.

Drive the full pipeline from a single YAML config without any
Prefect runtime. Built on the same domain functions the Prefect
flows call, so any deviation between the two paths is a bug.

Use cases:

* End-to-end smoke tests in CI.
* Contributor iteration on a developer laptop.
* Adapting the pipeline to a new orchestrator (Airflow, Dagster, …)
  by wrapping the domain functions exactly as :mod:`plain` does.
"""
