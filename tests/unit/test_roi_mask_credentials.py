"""The ROI mask's S3 credentials must be resolved per read, not frozen per leg.

The mask lives in our own bucket, which our role can always read — but the radar path
injects the SOURCE's short-lived token into the process environment for GDAL, so any
fsspec read that resolves credentials from the environment picks up the wrong one. The
mitigation is to resolve the role's credentials with the env provider removed.

That mitigation was correct and still failed, because its result was handed over as
**frozen strings, once, at leg entry**. A role credential expires in hours and a leg can
run longer, so every mask read past expiry failed on a bucket that was never inaccessible.

Passing a *provider* instead moves resolution to each CALL, which is what these tests
assert: that the reader calls a callable rather than storing it, and that the per-date
consumers re-resolve rather than reusing one graph.

Per call is NOT per read — the mask array is lazy, and its reads happen inside a later
write's compute. ``test_roi_mask_credential_expiry.py`` covers the read itself.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import zarr

from tessera_embeddings.ingest.roi import read_roi_mask, resolve_storage_options

_SRC = Path(__file__).resolve().parents[2] / "src" / "tessera_embeddings"


def test_a_dict_is_passed_through_unchanged() -> None:
    """The snapshot form still works — local and test callers pass plain options."""
    assert resolve_storage_options({"key": "static"}) == {"key": "static"}


def test_none_stays_none() -> None:
    """``None`` means "let fsspec decide", which is the correct local behaviour."""
    assert resolve_storage_options(None) is None


def test_a_provider_is_invoked() -> None:
    assert resolve_storage_options(lambda: {"key": "fresh"}) == {"key": "fresh"}


def test_a_provider_is_invoked_on_every_call_never_memoised() -> None:
    """The whole point: two reads far apart must not share one resolution.

    A provider that were called once and remembered would pass any test that only checked
    the first value, and would reproduce exactly the expiry this change exists to fix.
    """
    seen: list[int] = []

    def provider() -> dict:
        seen.append(len(seen))
        return {"key": f"cred-{len(seen)}"}

    first = resolve_storage_options(provider)
    second = resolve_storage_options(provider)
    assert first == {"key": "cred-1"}
    assert second == {"key": "cred-2"}, "the provider must be re-invoked, not memoised"
    assert len(seen) == 2


def test_the_mask_reader_resolves_a_provider(tmp_path) -> None:
    """``read_roi_mask`` must call the provider itself.

    Resolving inside the reader rather than at its call sites is what stops a new call site
    silently reintroducing the frozen behaviour by forgetting to re-resolve.
    """
    path = tmp_path / "mask.zarr"
    zarr.create_array(store=str(path), shape=(4, 4), chunks=(2, 2), dtype="bool")
    calls: list[int] = []

    def provider() -> dict | None:
        calls.append(1)
        return None  # a local path needs no options; we only assert it was ASKED

    read_roi_mask(str(path), {"northing": 2, "easting": 2}, storage_options=provider)
    assert calls, "read_roi_mask must invoke the provider, not store it"


def test_the_mask_reader_still_reads_correctly_through_a_provider(tmp_path) -> None:
    """Resolving must not change what comes back — a guard that corrupts data is worse."""
    path = tmp_path / "mask.zarr"
    arr = zarr.create_array(store=str(path), shape=(4, 4), chunks=(2, 2), dtype="bool")
    arr[:] = np.array([[1, 0, 1, 0]] * 4, dtype=bool)

    direct = read_roi_mask(str(path), {"northing": 2, "easting": 2})
    via_provider = read_roi_mask(str(path), {"northing": 2, "easting": 2}, storage_options=lambda: None)
    assert np.array_equal(direct.compute(), via_provider.compute())


class TestTheCallersPassAProviderAndReResolve:
    """Source-level checks: the wiring is what regressed, not the helper."""

    def test_the_task_layer_passes_the_callable_not_its_result(self) -> None:
        """``iam_s3_storage_options()`` — with parentheses — is the frozen snapshot."""
        src = (_SRC / "orchestration" / "prefect" / "tasks" / "ingest.py").read_text()
        assert "storage_options = iam_s3_storage_options\n" in src, (
            "the task must pass the provider itself, so each read resolves afresh"
        )
        assert "storage_options = iam_s3_storage_options()" not in src, (
            "calling it here freezes the credential for the whole leg — the original defect"
        )

    def test_the_optical_path_rebuilds_its_mask_per_date(self) -> None:
        """One graph reused across a leg carries one credential across a leg.

        Checked by CONTAINMENT: the read must be inside the per-date function, not merely
        present in the file, since the leg-entry read for the pixel total is legitimate.
        """
        tree = ast.parse((_SRC / "ingest" / "s2_roi.py").read_text())
        prepare = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name.startswith("_prepare_date")
        )
        reads = [
            node
            for node in ast.walk(prepare)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "read_roi_mask"
        ]
        assert reads, "the per-date path must build its own mask graph, not reuse the leg's"
        for call in reads:
            assert any(kw.arg == "storage_options" for kw in call.keywords), (
                f"the per-date mask read at line {call.lineno} must pass storage_options — "
                "without them fsspec resolves the environment, which holds the SOURCE token"
            )

    def test_no_mask_read_omits_storage_options(self) -> None:
        """A read with no options resolves the poisoned environment and eventually fails."""
        offenders = []
        for module in ("s1_roi.py", "s2_roi.py"):
            tree = ast.parse((_SRC / "ingest" / module).read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                    continue
                if node.func.id != "read_roi_mask":
                    continue
                if not any(kw.arg == "storage_options" for kw in node.keywords):
                    offenders.append(f"{module}:{node.lineno}")
        assert not offenders, f"mask read(s) with no storage_options: {offenders}"


@pytest.mark.parametrize("module", ["s1_roi.py", "s2_roi.py", "live_windows.py", "roi.py"])
def test_no_module_narrows_the_annotation_back_to_a_dict(module: str) -> None:
    """A ``dict``-only annotation is how the provider form gets refused at a boundary."""
    src = (_SRC / "ingest" / module).read_text()
    assert "storage_options: dict | None = None" not in src, (
        f"{module} annotates storage_options as dict-only, which rejects the provider form"
    )
