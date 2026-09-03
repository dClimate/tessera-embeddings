"""export_zone_rois flow: per-zone export/validate, and the gate it enforces.

The per-zone task is exercised against a real local coverage repo and a real
written mask — the point of the flow is the validation, so mocking it away would
test nothing. The outer flow is orchestration only, so its inner fan-out is
stubbed and only the aggregation and the failure gate are asserted.
"""

from __future__ import annotations

import logging

import pytest
import zarr

import tessera_embeddings.orchestration.prefect.flows.export_zone_rois as mod
from tessera_embeddings.config.paths import BucketPaths
from tessera_embeddings.ingest import land_mask
from tessera_embeddings.storage import zone_grid
from tests.unit.coverage_repo import make_coverage

_PATHS = BucketPaths(inputs="s3://in", outputs="s3://out")


@pytest.fixture(autouse=True)
def _no_prefect_runtime(monkeypatch):
    """Run the flow/task bodies directly: no Prefect engine, no AWS credentials.

    The credential callback is neutralised so the domain calls take the same
    local-store path the land_mask unit tests do.
    """
    monkeypatch.setattr(mod, "get_run_logger", lambda: logging.getLogger("test-export-zone-rois"))
    monkeypatch.setattr("tessera_embeddings.providers.aws.credentials.iam_icechunk_credentials", None, raising=False)


def _export(zone: str, cov: str, dest: str, *, validate_only: bool = False) -> dict:
    return mod.export_one_zone_roi.fn(
        zone, land_mask_path=cov, roi_path=dest, validate_only=validate_only, s3_region=None
    )


class TestPerZoneTask:
    """The per-zone unit: what it writes, what it skips, and what it reports."""

    def test_exports_and_validates(self, tmp_path) -> None:
        cov = make_coverage(tmp_path, "31N", [(10, 5), (11, 5)])
        dest = str(tmp_path / "roi.zarr")
        row = _export("31N", cov, dest)
        assert row["status"] == "exported"
        assert row["problems"] == []
        assert row["live_chunks"] == land_mask.live_chunk_count("31N", land_mask_path=cov)
        assert land_mask.validate_zone_roi("31N", land_mask_path=cov, roi_path=dest) == []

    def test_all_ocean_zone_writes_nothing(self, tmp_path) -> None:
        """An all-ocean zone has no mask by design, so it must be reported rather
        than validated — validating it would fail on an artifact that should not exist.
        """
        cov = make_coverage(tmp_path, "01N", [])
        dest = tmp_path / "roi.zarr"
        row = _export("01N", cov, str(dest))
        assert row == {"zone": "01N", "status": "all_ocean", "live_chunks": 0, "problems": []}
        assert not dest.exists()

    def test_validate_only_does_not_write(self, tmp_path) -> None:
        cov = make_coverage(tmp_path, "31N", [(10, 5)])
        dest = tmp_path / "roi.zarr"
        row = _export("31N", cov, str(dest), validate_only=True)
        assert row["status"] == "invalid"
        assert not dest.exists()
        assert any("cannot open" in p for p in row["problems"])

    def test_validate_only_passes_an_already_current_mask(self, tmp_path) -> None:
        cov = make_coverage(tmp_path, "31N", [(10, 5)])
        dest = str(tmp_path / "roi.zarr")
        _export("31N", cov, dest)
        assert _export("31N", cov, dest, validate_only=True)["status"] == "validated"

    def test_reports_invalid_without_raising(self, tmp_path) -> None:
        """A bad mask must come back as data, not an exception: one broken zone
        should not abort the other 111, and the flow decides what to do with it.
        """
        cov = make_coverage(tmp_path, "31N", [(10, 5)])
        dest = str(tmp_path / "roi.zarr")
        _export("31N", cov, dest)
        z = zarr.open(dest, mode="a")
        z.attrs["crs"] = "EPSG:4326"  # a mask claiming the wrong projection
        row = _export("31N", cov, dest, validate_only=True)
        assert row["status"] == "invalid"
        assert any("crs" in p for p in row["problems"])


class _FakeImpl:
    """Stand-in for the inner flow: records its call and returns canned rows."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[dict] = []

    def with_options(self, **_):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.rows


class TestOuterFlow:
    """Aggregation and the pass/fail gate; the fan-out itself is stubbed."""

    def _run(self, monkeypatch, rows, **kwargs):
        impl = _FakeImpl(rows)
        monkeypatch.setattr(mod, "_export_zone_rois_impl", impl)
        return impl, mod.export_zone_rois.fn(paths=_PATHS, **kwargs)

    def test_summarises_by_status(self, monkeypatch) -> None:
        rows = [
            {"zone": "31N", "status": "exported", "live_chunks": 12, "problems": []},
            {"zone": "32N", "status": "exported", "live_chunks": 30, "problems": []},
            {"zone": "01N", "status": "all_ocean", "live_chunks": 0, "problems": []},
        ]
        _, out = self._run(monkeypatch, rows, zones=["31N", "32N", "01N"])
        assert out["zones"] == 3
        assert out["by_status"] == {"exported": 2, "all_ocean": 1}
        assert out["live_chunks"] == 42
        assert out["invalid_zones"] == []

    def test_defaults_to_every_zone(self, monkeypatch) -> None:
        impl, _ = self._run(monkeypatch, [])
        assert impl.calls[0]["zones"] == list(zone_grid.ZONES)

    def test_canonicalizes_requested_zones(self, monkeypatch) -> None:
        impl, _ = self._run(monkeypatch, [], zones=["31n"])
        assert impl.calls[0]["zones"] == ["31N"]

    def test_fails_the_run_when_a_zone_is_invalid(self, monkeypatch) -> None:
        """The flow's terminal state IS the campaign gate, so an invalid zone must
        fail the run rather than be reported in a summary nobody reads.
        """
        rows = [
            {"zone": "31N", "status": "exported", "live_chunks": 12, "problems": []},
            {"zone": "32N", "status": "invalid", "live_chunks": 30, "problems": ["transform ..."]},
        ]
        with pytest.raises(ValueError, match="32N"):
            self._run(monkeypatch, rows, zones=["31N", "32N"])

    def test_passes_the_zone_mask_path_the_ingest_reads(self, monkeypatch) -> None:
        impl, _ = self._run(monkeypatch, [], zones=["31N"], mask_name="global")
        assert impl.calls[0]["land_mask_path"] == _PATHS.land_mask_store("global")
        assert impl.calls[0]["paths"].zone_roi_store("31N") == "s3://in/rois/zarrs/zone_31N.zarr"
