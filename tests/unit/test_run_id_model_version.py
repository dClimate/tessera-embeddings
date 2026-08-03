"""A staging run must not be resumed under the other model version.

Scoped deliberately narrowly. The hazard exists *because* this PR introduces a
second model version: staged chunks are (H, W, 128) int8 whichever student wrote
them, `model_version` defaults to v1.1, so a resume that omits it would finish a
v2 run with the v1.1 encoder and stamp the store v1.1.

The mechanism mirrors `main`'s `S2_ONLY_RUN_PREFIX`, which encodes the S2-only
mode in the run_id for the same reason — one function, so the two reconcile
trivially when this branch merges. General resume/staging hardening (concurrent
claims, unclaimed legacy staging, ROI and time-window identity) is out of scope
here and belongs with the failure-case work.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.orchestration.prefect.flows.tessera_embeddings import (
    V2_RUN_PREFIX,
    _resolve_run_id,
)


def test_fresh_v11_run_keeps_the_historical_bare_uuid():
    """The single-model path must be untouched by this change."""
    run_id = _resolve_run_id(None, model_version="v1.1")
    assert not run_id.startswith(V2_RUN_PREFIX)
    assert len(run_id) == 12


def test_fresh_v2_run_is_namespaced():
    assert _resolve_run_id(None, model_version="v2-large").startswith(V2_RUN_PREFIX)


def test_fresh_run_ids_are_unique():
    ids = {_resolve_run_id(None, model_version="v2-large") for _ in range(50)}
    assert len(ids) == 50


@pytest.mark.parametrize("model_version", ["v1.1", "v2-large"])
def test_resuming_with_the_matching_model_is_allowed(model_version):
    original = _resolve_run_id(None, model_version=model_version)
    assert _resolve_run_id(original, model_version=model_version) == original


def test_resuming_a_v2_run_as_v11_is_refused():
    """The defect: v2 staging finished by the v1.1 encoder and stamped v1.1."""
    v2_run = _resolve_run_id(None, model_version="v2-large")
    with pytest.raises(ValueError, match="staged by v2"):
        _resolve_run_id(v2_run, model_version="v1.1")


def test_resuming_a_v11_run_as_v2_is_refused():
    v11_run = _resolve_run_id(None, model_version="v1.1")
    with pytest.raises(ValueError, match=r"staged by v1\.1"):
        _resolve_run_id(v11_run, model_version="v2-large")


def test_error_names_both_versions_so_the_fix_is_obvious():
    v2_run = _resolve_run_id(None, model_version="v2-large")
    with pytest.raises(ValueError) as exc:
        _resolve_run_id(v2_run, model_version="v1.1")
    msg = str(exc.value)
    assert v2_run in msg
    assert "v1.1" in msg and "v2" in msg
