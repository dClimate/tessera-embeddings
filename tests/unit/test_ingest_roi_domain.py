"""Smoke + structural tests for the S2/S1 ROI ingest domain functions.

These tests verify the **interface contract** without spinning up a
Dask cluster or hitting STAC. The expensive end-to-end tests
(LocalCluster + VCR-recorded STAC responses) are deferred to the
broader test suite (plan Phase 10) — what matters at Phase 7 is that
the function lives in the domain layer with no Prefect coupling.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from tessera_embeddings.ingest.s1_roi import S1Orbit, SarIngestResult, ingest_s1_roi_sar
from tessera_embeddings.ingest.s2_roi import IngestResult, ingest_s2_roi_reflectance


def test_s2_result_has_expected_fields() -> None:
    """``IngestResult`` is a frozen dataclass with the documented fields."""
    r = IngestResult(roi_path="/tmp/x", status="success", dates_processed=10, dates_filtered_coverage=2)
    assert r.roi_path == "/tmp/x"
    assert r.status == "success"
    assert r.dates_processed == 10
    assert r.dates_filtered_coverage == 2


def test_s1_result_has_expected_fields() -> None:
    """``SarIngestResult`` carries the per-orbit count dict."""
    r = SarIngestResult(roi_path="/tmp/x", status="success", dates_processed={"ascending": 7})
    assert r.dates_processed["ascending"] == 7


def test_s1_orbit_literal_values() -> None:
    """``S1Orbit`` is the documented ascending/descending literal."""
    # The Literal can't be introspected at runtime in a typed way; this is
    # a lightweight check that the alias is wired up rather than a typo.
    assert "ascending" in str(S1Orbit)
    assert "descending" in str(S1Orbit)


def test_s2_signature_takes_client_and_logger() -> None:
    """Required keywords match the orchestrator-injection contract.

    The reference repo's ``@task`` reached for ``get_client()`` and
    ``get_run_logger()`` inside the function body. The OSS domain
    function takes both as keyword parameters so callers are free to
    wire whatever client/logger they already have.
    """
    sig = inspect.signature(ingest_s2_roi_reflectance)
    assert "client" in sig.parameters
    assert "log" in sig.parameters
    assert "storage_options" in sig.parameters


def test_s1_signature_takes_credential_callbacks() -> None:
    """S1 exposes the credential-refresh hook as a parameter.

    Hard rule #5 (secrets at flow entry) means the domain function
    cannot read STS creds from env or a Prefect block; it accepts a
    callback that the substrate sets up.
    """
    sig = inspect.signature(ingest_s1_roi_sar)
    assert "edl_credentials_fn" in sig.parameters
    assert "apply_credentials_fn" in sig.parameters
    assert "use_s3_direct" in sig.parameters


def test_no_prefect_or_get_client_in_domain_modules() -> None:
    """Hard rule: ``ingest/s2_roi.py`` and ``ingest/s1_roi.py`` are domain-pure.

    Walks the module AST so docstrings and comments don't trigger false
    positives. Catches future regressions where someone reaches for
    ``get_client`` or sneaks a ``from prefect import …`` past review.
    """
    import ast

    domain_files = [
        Path("src/tessera_embeddings/ingest/s2_roi.py"),
        Path("src/tessera_embeddings/ingest/s1_roi.py"),
    ]
    forbidden_calls = {"get_run_logger", "get_client"}
    for path in domain_files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # Reject: import prefect, from prefect import …
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("prefect"), (
                        f"{path} imports {alias.name!r}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("prefect"), (
                    f"{path} imports from {node.module!r}"
                )
            # Reject: get_run_logger() / get_client() call sites
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in forbidden_calls:
                    raise AssertionError(f"{path} calls {fn.id}()")
                if isinstance(fn, ast.Attribute) and fn.attr in forbidden_calls:
                    raise AssertionError(f"{path} calls {fn.attr}()")
