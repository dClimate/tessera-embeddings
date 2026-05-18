"""Repo-wide pytest configuration.

* Hypothesis profiles: ``ci`` runs more examples and tolerates slow
  data generators; ``local`` is fast for developer iteration. Switch
  via the ``CI`` environment variable.
* Shared fixtures live in subdirectory ``conftest.py`` files so that
  test categories can pull in fixtures relevant only to themselves.
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=200,
    deadline=2000,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "local",
    max_examples=20,
    deadline=500,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci" if os.getenv("CI") else "local")
