"""Plain (orchestrator-free) runners: the full pipeline from one YAML config, no Prefect.

Built on the same domain functions the Prefect flows call, so any deviation between the
two paths is a bug. Used for CI smoke tests, laptop iteration, and as the template for
adapting the pipeline to another orchestrator (wrap the domain functions as :mod:`plain`
does).
"""
