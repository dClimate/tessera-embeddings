"""The ingest fleet's minimum follows its DERIVED width, so fleets do not churn.

`cluster.adapt(minimum=1, ...)` retired workers in every inter-date gap and relaunched them
cold into the next write. Measured across a 35-cell wave on 2026-08-04: one 60-slot fleet
registered 1,250 distinct workers in five hours, and fleets held only ~85-90% of nominal width.
Adaptivity bought nothing against that — these fleets are busy essentially the whole run, so
there is no idle trough for a minimum of 1 to exploit.

What the resolution must get right is WHICH width it follows. Each leg is sized separately: a
sparse zone's optical fleet is clamped below the configured maximum, and each radar orbit is a
fraction of the optical one. Resolving against the configured maximum would ask for more workers
than a leg was sized for, and the fleet could never reach its own minimum.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.config.ingest import IngestSettings


def test_default_is_a_fixed_size_fleet() -> None:
    assert IngestSettings().min_workers is None, "None is what means 'follow the derived width'"
    assert IngestSettings().floor_for(60) == 60


def test_the_floor_follows_each_leg_not_the_configured_maximum() -> None:
    """A radar orbit sized at 13 must ask for 13, not the optical fleet's 60."""
    settings = IngestSettings(max_workers=60)
    assert settings.floor_for(60) == 60
    assert settings.floor_for(13) == 13


def test_an_explicit_minimum_restores_adaptive_behaviour() -> None:
    """A reference run compared against older measurements needs the churn in both arms."""
    settings = IngestSettings(min_workers=1)
    assert settings.floor_for(60) == 1
    assert settings.floor_for(13) == 1


def test_an_explicit_minimum_cannot_exceed_the_leg_it_is_applied_to() -> None:
    """Otherwise a fleet sized at 13 waits forever for a 40th worker."""
    settings = IngestSettings(min_workers=40, max_workers=60)
    assert settings.floor_for(13) == 13
    assert settings.floor_for(60) == 40


def test_inverted_bounds_are_still_refused() -> None:
    with pytest.raises(ValueError, match="below min_workers"):
        IngestSettings(min_workers=20, max_workers=10)


def test_none_does_not_break_the_bounds_check() -> None:
    """The validator predates the sentinel; None must not compare against an int."""
    assert IngestSettings(min_workers=None, max_workers=1).max_workers == 1
