"""Per-cell validation dispatch: what it sends, what it declines, and what it swallows.

The dispatch sits between a cell's tag and the campaign's next cell, so it has exactly
two ways to do damage: taking the fill down with it, or looking as if it ran when it did
not. Both are asserted here, and so is the parameter payload — a validation aimed at a
different store than the fill wrote to would pass every other test in this file while
reporting on the wrong data.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import tessera_embeddings.orchestration.prefect.flows._cell_validation as mod
from tessera_embeddings.config.paths import BucketPaths

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")
_LANDED = {"zone": "33N", "year": 2025, "tag": "zone-33N-2025", "empty": False}


@pytest.fixture()
def dispatched(monkeypatch):
    """Capture every ``run_deployment`` call instead of reaching a Prefect server."""
    calls: list[dict] = []

    def _run_deployment(name, *, parameters=None, timeout=None, tags=None):
        calls.append({"name": name, "parameters": parameters, "timeout": timeout, "tags": tags})
        return SimpleNamespace(id="run-1")

    monkeypatch.setattr(mod, "run_deployment", _run_deployment)
    return calls


def _dispatch(summary=None, deployment="validate-zone-year/validate-zone-year", **kwargs):
    return mod.dispatch_cell_validation(
        deployment,
        zone="33N",
        year=2025,
        summary=_LANDED if summary is None else summary,
        parameters=mod.cell_validation_parameters(
            zone="33N",
            year=2025,
            paths=_PATHS,
            store_name="tessera",
            mask_name="global",
            s3_region="us-west-2",
        ),
        log=logging.getLogger("test-validation"),
        **kwargs,
    )


class TestWhatIsDispatched:
    """What reaches the server for a cell that landed."""

    def test_a_landed_cell_is_dispatched_without_waiting(self, dispatched):
        """``timeout=0`` is the whole reason this does not delay the next cell."""
        assert _dispatch() == "run-1"
        assert dispatched[0]["name"] == "validate-zone-year/validate-zone-year"
        assert dispatched[0]["timeout"] == 0

    def test_the_payload_names_the_cell_and_the_store_the_fill_wrote(self, dispatched):
        """The validator must read THIS fill's store, not its own registered default.

        Asserted on the payload rather than on the call happening: a dispatch that omits
        the buckets still succeeds, and the validation run then silently audits whatever
        store its deployment was registered with — a different branch's, in a
        branch-scoped account.
        """
        _dispatch()
        params = dispatched[0]["parameters"]
        assert params["zone"] == "33N"
        assert params["year"] == 2025
        assert params["store_name"] == "tessera"
        assert params["mask_name"] == "global"
        assert params["s3_region"] == "us-west-2"
        # Dumped, not the model: the value crosses the API as JSON.
        assert params["paths"] == _PATHS.model_dump()

    def test_the_trace_tag_is_stamped_when_the_caller_resolved_one(self, dispatched):
        _dispatch(tag=mod.validation_run_tag("parent-9"))
        assert dispatched[0]["tags"] == ["validates-cell-of:parent-9"]

    def test_no_tag_outside_a_flow_run(self):
        """A ``.fn`` call has no flow-run id, and an untagged run is better than a fake one."""
        assert mod.validation_run_tag(None) is None


class TestWhatIsDeclined:
    """Cells with nothing to check, and the case where no validator is configured."""

    def test_no_deployment_configured_dispatches_nothing(self, dispatched):
        assert _dispatch(deployment=None) is None
        assert dispatched == []

    def test_an_empty_cell_is_not_validated(self, dispatched):
        """Nothing was embedded, so every pixel check has no subject.

        This must match the closing sweep's own exclusion. If the two disagree about
        which cells belong to the validated set, monitoring reads one pass's deliberate
        exclusion as the other's missing verdict.
        """
        assert _dispatch({**_LANDED, "empty": True}) is None
        assert dispatched == []

    def test_an_untagged_cell_is_not_validated(self, dispatched):
        """No tag means no committed snapshot — there is nothing to judge yet."""
        assert _dispatch({"zone": "33N", "year": 2025}) is None
        assert dispatched == []


class TestFailureIsSwallowed:
    """A dispatch that cannot reach the API must not undo a landed, tagged cell."""

    def test_a_dispatch_failure_never_reaches_the_fill(self, monkeypatch, caplog):
        """The cell is landed and tagged; losing the CHECK must not undo that."""

        def _boom(*a, **k):
            raise RuntimeError("prefect api unreachable")

        monkeypatch.setattr(mod, "run_deployment", _boom)
        with caplog.at_level(logging.WARNING):
            assert _dispatch() is None
        assert "Cell validation dispatch FAILED" in caplog.text
        # The cell is named, because the recovery is to dispatch that cell by hand.
        assert "33N-2025" in caplog.text
