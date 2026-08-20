"""Smoke + structural tests for the S2/S1 ROI ingest domain functions.

These tests verify the **interface contract** without spinning up a
Dask cluster or hitting STAC. The expensive end-to-end tests
(LocalCluster + VCR-recorded STAC responses) are deferred to the
broader test suite (plan Phase 10) — what matters at Phase 7 is that
the function lives in the domain layer with no Prefect coupling.
"""

from __future__ import annotations

import inspect
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from unittest.mock import MagicMock, patch

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
                    assert not alias.name.startswith("prefect"), f"{path} imports {alias.name!r}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("prefect"), f"{path} imports from {node.module!r}"
            # Reject: get_run_logger() / get_client() call sites
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in forbidden_calls:
                    raise AssertionError(f"{path} calls {fn.id}()")
                if isinstance(fn, ast.Attribute) and fn.attr in forbidden_calls:
                    raise AssertionError(f"{path} calls {fn.attr}()")


# Window derivation runs on every path now that cropping is unconditional, and this test
# has no ROI mask on disk. One window stands in; the test is about how the DATE RANGE is
# tiled into batches, which the windows do not affect.
@patch("tessera_embeddings.ingest.s1_roi.live_windows_for_mask", return_value=[MagicMock(y0=0, y1=1, x0=0, x1=1)])
@patch("tessera_embeddings.ingest.s1_roi.make_s1_item_provider")
@patch("tessera_embeddings.ingest.s1_roi.get_existing_dates", return_value=set())
@patch("tessera_embeddings.ingest.s1_roi.read_roi_mask")
@patch("tessera_embeddings.ingest.s1_roi.read_roi_metadata")
@patch("tessera_embeddings.ingest.s1_roi.IngestManifest")
@patch("tessera_embeddings.ingest.s1_roi.ingest_tile", return_value=(None, None))
def test_s1_batch_windows_tile_inclusive_range_without_overlap(
    mock_ingest_tile,
    mock_manifest,
    mock_read_meta,
    mock_read_mask,
    mock_existing,
    mock_provider,
    mock_live_windows,
):
    """Batches OWN a tiling of the inclusive [start, end] range and QUERY a day past it.

    The two ranges are different on purpose, and the overlap between consecutive
    queries is the point rather than a defect. A solar day straddling a batch cut has
    acquisitions on both sides of it, so the owning batch has to ask for the day beyond
    its own range or it commits that day missing half its imagery. Ownership — applied
    to items before the loader sees them — is what stops the overlap becoming duplicate
    work. See ingest.solar_days.

    Still guarded here, unchanged in substance:

    * ``end_date`` is inclusive: the final day must be owned, not dropped.
    * The item provider is built for the same window the query uses; they diverged once
      and the result was a batch loading imagery it had not asked for.
    """
    mock_read_meta.return_value = MagicMock(bbox_wgs84=(-105.0, 39.0, -104.0, 40.0))

    result = ingest_s1_roi_sar(
        roi_zarr_path="/tmp/roi.zarr",
        start_date="2024-01-01",
        end_date="2024-03-01",
        store_path="/tmp/store",
        client=MagicMock(),
        orbit="ascending",
        batch_days=30,
    )

    # ingest_tile returned no data each batch, so nothing was written.
    assert result.status == "skipped"

    windows = [(c.kwargs["start_date"], c.kwargs["end_date"]) for c in mock_ingest_tile.call_args_list]

    # 30-day batches OWN a tiling of [2024-01-01, 2024-03-01] — each covering batch_days
    # solar days, the next starting the following day, the last landing exactly on the
    # inclusive end_date — and each QUERIES that span padded a day either side.
    owned = [
        ("2024-01-01", "2024-01-30"),
        ("2024-01-31", "2024-02-29"),
        ("2024-03-01", "2024-03-01"),
    ]
    assert windows == [
        ("2023-12-31", "2024-01-31"),
        ("2024-01-30", "2024-03-01"),
        ("2024-02-29", "2024-03-02"),
    ]

    # The owned spans still tile the window exactly once — that, not the query bound, is
    # what stops a day being written twice or written at all outside the window.
    covered: list[str] = []
    for own_start, own_end in owned:
        d0, d1 = date.fromisoformat(own_start), date.fromisoformat(own_end)
        covered.extend((d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1))
    start, end = date(2024, 1, 1), date(2024, 3, 1)
    assert covered == [(start + timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]

    # Consecutive queries DO overlap, by exactly the pad. Asserting it keeps a future
    # "tidy-up" from clamping the pads away again and silently truncating boundary days.
    for (_, prev_end), (next_start, _) in pairwise(windows):
        assert next_start < prev_end, f"the query pad was lost at {prev_end}/{next_start}"

    # The item provider is built for the same window as the query.
    provider_windows = [(c.args[2], c.args[3]) for c in mock_provider.call_args_list]
    assert provider_windows == windows


@patch("tessera_embeddings.ingest.stac.Client")
@patch("tessera_embeddings.ingest.stac.StacApiIO")
@patch("tessera_embeddings.ingest.stac._build_stac_query", return_value={})
def test_query_stac_items_passes_stac_io_to_client(mock_build, mock_stac_io_cls, mock_client_cls):
    """Retry config reaches Client.open — dropping stac_io= would silently skip retries."""
    from tessera_embeddings.ingest.stac import _query_stac_items

    mock_stac_io = MagicMock()
    mock_stac_io_cls.return_value = mock_stac_io
    # An EMPTY page cursor, named as the method the query actually walks. A bare MagicMock
    # here answers `next()` forever, which is an unbounded loop rather than a failed test.
    mock_client_cls.open.return_value.search.return_value.pages_as_dicts.return_value = iter([])

    provider = MagicMock()
    _query_stac_items(provider, MagicMock(), "T15TYH", "2024-01-01", "2024-01-31")

    _, kwargs = mock_client_cls.open.call_args
    assert kwargs.get("stac_io") is mock_stac_io
