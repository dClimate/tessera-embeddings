"""Prefect task shells.

Each task shell is ~20 LOC: pull ``client`` and ``log`` from Prefect /
Dask context, delegate to a domain function, convert the dataclass
result to a dict at the boundary so Prefect's UI can display it.

Domain functions never reach for context — they take ``client`` and
``log`` as explicit parameters. That keeps them testable without a
running orchestrator.
"""
