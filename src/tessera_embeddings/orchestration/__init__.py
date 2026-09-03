"""Orchestration glue.

Houses the substrate-agnostic helpers (concurrency primitives, plain runner, Prefect task
shells, Prefect flows) that bind domain code to real workflows. No Prefect imports outside
this package's ``orchestration.prefect`` subtree.
"""
