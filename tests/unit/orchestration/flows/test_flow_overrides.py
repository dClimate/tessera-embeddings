"""An omitted parameter and a falsy one are different instructions.

A flow forwarding optional parameters to a child run has to distinguish "the caller
chose nothing, so the child's registered default stands" from "the caller chose this
value". Prefect offers one channel for both — the key is present or it is not — so the
rule is: omit on ``None``, and on nothing else.

Written by hand as ``if value else {}`` the rule is right only while no legitimate value
is falsy, and it stopped being right when a parameter whose domain includes ``0`` was
routed through it. That guard stood at six call sites across three flows and dropped the
mode at every one. It lives in one place now, and this is that place's test.
"""

from __future__ import annotations

from tessera_embeddings.orchestration.prefect.flows._overrides import set_overrides


def test_an_unset_parameter_is_omitted() -> None:
    """So the child's own registered default decides, rather than being overridden."""
    assert set_overrides(a=None) == {}
    assert set_overrides(a=None, b=None) == {}


def test_zero_is_a_choice_and_survives() -> None:
    """The defect this module exists for: 0 is a documented mode, not an absence.

    ``actor_request_batch_size=0`` selects the all-at-once mode. A truthiness guard drops
    it and leaves the child on its default of 50 — the caller's explicit instruction
    silently replaced by its opposite, with nothing logged.
    """
    assert set_overrides(actor_request_batch_size=0) == {"actor_request_batch_size": 0}


def test_the_other_falsy_values_are_choices_too() -> None:
    """False and the empty string are values a caller can mean; only None is an absence."""
    assert set_overrides(flag=False, name="") == {"flag": False, "name": ""}


def test_set_and_unset_parameters_are_separated_in_one_call() -> None:
    """The mixed case is the real one — a caller sets some parameters and not others."""
    assert set_overrides(kept=0, dropped=None, also_kept=25) == {"kept": 0, "also_kept": 25}
