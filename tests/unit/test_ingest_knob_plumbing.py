"""Every ingest knob must exist, and agree, at all three layers it passes through.

An ingest parameter travels flow -> Prefect task shell -> domain function. Adding one to the
flow and the domain but not the shell in between produces `ParameterBindError: got an
unexpected keyword argument` — and only at RUN time, on the cluster, after the fleet has
launched. That is what happened when ``pipeline_batches`` was added: unit tests passed, the
image built, all four runs of an A/B died about a minute in. Nothing local could have caught
it, because each layer was individually valid.

A default that disagrees between layers is the quieter version of the same bug: the shells
only forward, so a shell defaulting ``overlap_window_writes`` to False while the domain
defaults it to True silently changes behaviour for any caller that does not pass the flag —
and nothing fails, so no test notices. That was live for the whole session in which S1's
overlap became the default.

These assertions are deliberately mechanical: compare signatures, do not restate values.
Restating the expected default would just move the drift into the test.
"""

from __future__ import annotations

import inspect

import pytest

from tessera_embeddings.ingest.s1_roi import ingest_s1_roi_sar as s1_domain
from tessera_embeddings.ingest.s2_roi import ingest_s2_roi_reflectance as s2_domain
from tessera_embeddings.orchestration.prefect.tasks.ingest import process_roi_reflectance, process_roi_sar

#: Knobs that must be plumbed end to end, per sensor. Not every domain parameter belongs
#: here — credential callbacks and loggers are constructed per layer by design — so this
#: lists the ones a CALLER is expected to be able to set.
S1_KNOBS = ("batch_days", "overlap_window_writes", "pipeline_batches")
S2_KNOBS = ("overlap_window_writes", "pipeline_dates", "batch_dates")


def _params(fn) -> dict[str, inspect.Parameter]:
    # Prefect wraps tasks and flows; unwrap to the function whose signature binding fails.
    return dict(inspect.signature(inspect.unwrap(fn)).parameters)


@pytest.mark.parametrize("knob", S1_KNOBS)
def test_s1_knob_reaches_the_task_shell(knob: str) -> None:
    """The shell is the layer that was missed, so assert it by name."""
    assert knob in _params(process_roi_sar), (
        f"{knob!r} is missing from process_roi_sar. A flow passing it would fail at RUN time "
        f"with ParameterBindError, after the fleet has already launched."
    )


@pytest.mark.parametrize("knob", S2_KNOBS)
def test_s2_knob_reaches_the_task_shell(knob: str) -> None:
    assert knob in _params(process_roi_reflectance), f"{knob!r} is missing from process_roi_reflectance"


@pytest.mark.parametrize("knob", S1_KNOBS)
def test_s1_shell_default_matches_the_domain_default(knob: str) -> None:
    """A forwarding shell must not hold an opinion the domain does not share."""
    shell, domain = _params(process_roi_sar)[knob], _params(s1_domain)[knob]
    assert shell.default == domain.default, (
        f"{knob!r} defaults to {shell.default!r} in process_roi_sar but {domain.default!r} in "
        f"ingest_s1_roi_sar. The shell only forwards, so the disagreement silently changes "
        f"behaviour for any caller that omits the flag."
    )


@pytest.mark.parametrize("knob", S2_KNOBS)
def test_s2_shell_default_matches_the_domain_default(knob: str) -> None:
    shell, domain = _params(process_roi_reflectance)[knob], _params(s2_domain)[knob]
    assert shell.default == domain.default, (
        f"{knob!r} defaults to {shell.default!r} in process_roi_reflectance but "
        f"{domain.default!r} in ingest_s2_roi_reflectance"
    )
