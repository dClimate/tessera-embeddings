"""Unit tests for tessera_embeddings/storage/manifest.py — store manifest validation.

Covers to_dict/from_dict round-tripping, hash stability/sensitivity,
validate_against config-mismatch detection, legacy stores without a manifest,
hash chaining (ROI -> ingest -> embedding), and validation edge cases.
"""

from __future__ import annotations

import pytest

from tessera_embeddings.errors import ConfigMismatchError
from tessera_embeddings.storage.manifest import (
    EmbeddingManifest,
    IngestManifest,
    RoiManifest,
    StoreManifest,
)


class TestManifestSerialization:
    """Tests for to_dict / from_dict round-tripping."""

    def test_roi_manifest_to_dict(self):
        m = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        d = m.to_dict()
        assert d["resolution"] == 10.0
        assert d["chunk_size"] == 2000
        assert d["crs"] == "EPSG:32615"
        assert d["manifest_type"] == "RoiManifest"

    def test_to_dict_excludes_none_values(self):
        m = IngestManifest(roi_manifest_hash=None)
        assert "roi_manifest_hash" not in m.to_dict()

    def test_from_dict_round_trip(self):
        original = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        restored = RoiManifest.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_ignores_extra_keys(self):
        d = {"resolution": 10.0, "chunk_size": 2000, "crs": "EPSG:32615", "extra": "ignored"}
        m = RoiManifest.from_dict(d)
        assert m == RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")

    def test_empty_ingest_manifest_to_dict(self):
        m = IngestManifest()
        d = m.to_dict()
        assert d["manifest_type"] == "IngestManifest"
        assert "roi_manifest_hash" not in d

    def test_embedding_manifest_from_dict_coerces_list_to_tuple(self):
        """EmbeddingManifest.from_dict coerces num_obs_checkpoints list -> tuple."""
        original = EmbeddingManifest(
            model_checkpoint="tessera_v1_1_aws_encoder",
            num_obs_checkpoints=tuple(range(8, 257, 8)),
        )
        restored = EmbeddingManifest.from_dict(original.to_dict())
        assert restored == original
        assert isinstance(restored.num_obs_checkpoints, tuple)

    def test_base_class_is_store_manifest(self):
        """Concrete manifests subclass StoreManifest."""
        m = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        assert isinstance(m, StoreManifest)


class TestManifestHash:
    """Tests for manifest hashing."""

    def test_hash_is_deterministic(self):
        m = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        assert m.hash() == m.hash()

    def test_hash_changes_with_different_values(self):
        m1 = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        m2 = RoiManifest(resolution=20.0, chunk_size=2000, crs="EPSG:32615")
        assert m1.hash() != m2.hash()

    def test_hash_is_16_chars(self):
        m = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        assert len(m.hash()) == 16


class TestValidateAgainst:
    """Tests for validate_against (instance method)."""

    def test_matching_config_passes(self):
        m = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        # Should not raise
        m.validate_against(m.to_dict(), "test.zarr")

    def test_mismatched_resolution_raises(self):
        existing = {"resolution": 10.0, "chunk_size": 2000, "crs": "EPSG:32615"}
        current = RoiManifest(resolution=20.0, chunk_size=2000, crs="EPSG:32615")
        with pytest.raises(ConfigMismatchError, match="resolution"):
            current.validate_against(existing, "test.zarr")

    def test_mismatched_crs_raises(self):
        existing = {"resolution": 10.0, "chunk_size": 2000, "crs": "EPSG:32615"}
        current = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32633")
        with pytest.raises(ConfigMismatchError, match="crs"):
            current.validate_against(existing, "test.zarr")

    def test_mismatched_chunk_size_raises(self):
        existing = {"resolution": 10.0, "chunk_size": 2000, "crs": "EPSG:32615"}
        current = RoiManifest(resolution=10.0, chunk_size=1000, crs="EPSG:32615")
        with pytest.raises(ConfigMismatchError, match="chunk_size"):
            current.validate_against(existing, "test.zarr")

    def test_multiple_mismatches_reported(self):
        existing = {"resolution": 10.0, "chunk_size": 2000, "crs": "EPSG:32615"}
        current = RoiManifest(resolution=20.0, chunk_size=1000, crs="EPSG:32615")
        with pytest.raises(ConfigMismatchError, match="resolution") as exc_info:
            current.validate_against(existing, "test.zarr")
        assert "chunk_size" in str(exc_info.value)

    def test_error_message_includes_store_path(self):
        existing = {"resolution": 10.0, "chunk_size": 2000, "crs": "EPSG:32615"}
        current = RoiManifest(resolution=20.0, chunk_size=2000, crs="EPSG:32615")
        with pytest.raises(ConfigMismatchError, match=r"s3://bucket/store\.zarr"):
            current.validate_against(existing, "s3://bucket/store.zarr")

    def test_error_message_includes_old_and_new_values(self):
        existing = {"resolution": 10.0, "chunk_size": 2000, "crs": "EPSG:32615"}
        current = RoiManifest(resolution=20.0, chunk_size=2000, crs="EPSG:32615")
        with pytest.raises(ConfigMismatchError, match=r"store has 10\.0.*current config has 20\.0"):
            current.validate_against(existing, "test.zarr")


class TestValidateAgainstLegacyStores:
    """Tests for legacy stores without manifests."""

    def test_none_manifest_warns_but_passes(self, caplog):
        m = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        # Should not raise
        m.validate_against(None, "legacy.zarr")
        assert "legacy store" in caplog.text.lower()

    def test_none_manifest_logs_store_path(self, caplog):
        m = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        m.validate_against(None, "s3://bucket/old.zarr")
        assert "s3://bucket/old.zarr" in caplog.text


class TestValidateAgainstEdgeCases:
    """Tests for edge cases in validation."""

    def test_extra_keys_in_existing_ignored(self):
        existing = {"resolution": 10.0, "chunk_size": 2000, "crs": "EPSG:32615", "extra": "val"}
        current = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        # Should not raise
        current.validate_against(existing, "test.zarr")

    def test_missing_key_in_existing_skipped(self):
        existing = {"resolution": 10.0}  # chunk_size and crs not in existing
        current = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        # Should not raise — keys absent from existing are skipped
        current.validate_against(existing, "test.zarr")

    def test_store_has_field_current_dropped_raises(self):
        """Store has a concrete roi_manifest_hash but current is None (dropped by
        to_dict) — the `key in existing but not in current` branch must flag it.
        """
        existing = {"roi_manifest_hash": "abc123", "manifest_type": "IngestManifest"}
        current = IngestManifest(roi_manifest_hash=None)
        with pytest.raises(ConfigMismatchError, match="roi_manifest_hash"):
            current.validate_against(existing, "test.zarr")

    def test_current_concrete_vs_existing_none_passes(self):
        """Store was written without the field; current now provides it — no mismatch."""
        existing = {"manifest_type": "IngestManifest"}
        current = IngestManifest(roi_manifest_hash="abc123")
        # existing lacks roi_manifest_hash -> existing_val is None -> skipped
        current.validate_against(existing, "test.zarr")

    def test_unknown_key_in_existing_ignored(self):
        """Keys in existing that aren't in the manifest schema are ignored (forward-compat)."""
        existing = {
            "resolution": 10.0,
            "chunk_size": 2000,
            "crs": "EPSG:32615",
            "manifest_type": "RoiManifest",
            "future_field": "v2",
        }
        current = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        # Should not raise — future_field isn't a known RoiManifest field
        current.validate_against(existing, "test.zarr")


class TestIngestManifestChaining:
    """Tests for ingestion manifest with chained ROI hash."""

    def test_ingest_manifest_validates_roi_hash(self):
        roi_hash = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615").hash()
        m = IngestManifest(roi_manifest_hash=roi_hash)
        # Should not raise
        m.validate_against(m.to_dict(), "test.zarr")

    def test_ingest_manifest_detects_roi_hash_mismatch(self):
        old_hash = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615").hash()
        new_hash = RoiManifest(resolution=20.0, chunk_size=2000, crs="EPSG:32615").hash()
        existing = {"roi_manifest_hash": old_hash}
        current = IngestManifest(roi_manifest_hash=new_hash)
        with pytest.raises(ConfigMismatchError, match="roi_manifest_hash"):
            current.validate_against(existing, "test.zarr")


class TestEmbeddingManifestChaining:
    """Tests for embedding manifest with chained ingestion hash."""

    def test_embedding_manifest_validates_model_checkpoint(self):
        m = EmbeddingManifest(
            model_checkpoint="tessera_v1_1_aws_encoder",
            num_obs_checkpoints=tuple(range(8, 257, 8)),
            reflectance_manifest_hash="abc123",
            sar_manifest_hash="def456",
        )
        m.validate_against(m.to_dict(), "test.zarr")

    def test_embedding_manifest_detects_model_change(self):
        existing = {
            "model_checkpoint": "best_model_fsdp_20250608_220648_QAT",
            "num_obs_checkpoints": list(range(8, 257, 8)),
        }
        current = EmbeddingManifest(
            model_checkpoint="tessera_v1_1_aws_encoder",
            num_obs_checkpoints=tuple(range(8, 257, 8)),
        )
        with pytest.raises(ConfigMismatchError, match="model_checkpoint"):
            current.validate_against(existing, "test.zarr")

    def test_embedding_manifest_detects_num_obs_checkpoints_change(self):
        existing = {
            "model_checkpoint": "tessera_v1_1_aws_encoder",
            "num_obs_checkpoints": [8, 16, 32, 64],
        }
        current = EmbeddingManifest(
            model_checkpoint="tessera_v1_1_aws_encoder",
            num_obs_checkpoints=tuple(range(8, 257, 8)),
        )
        with pytest.raises(ConfigMismatchError, match="num_obs_checkpoints"):
            current.validate_against(existing, "test.zarr")

    def test_embedding_manifest_list_vs_tuple_equivalent(self):
        """Zarr attrs round-trip tuples as lists; equal contents must validate cleanly."""
        checkpoints = tuple(range(8, 257, 8))
        existing = {
            "model_checkpoint": "tessera_v1_1_aws_encoder",
            "num_obs_checkpoints": list(checkpoints),
        }
        current = EmbeddingManifest(
            model_checkpoint="tessera_v1_1_aws_encoder",
            num_obs_checkpoints=checkpoints,
        )
        # Should not raise
        current.validate_against(existing, "test.zarr")


class TestIngestManifestFromRoiStore:
    """Tests for IngestManifest.from_roi_store (reads + chains the ROI manifest)."""

    def _write_roi_store(self, path: str, manifest: RoiManifest | None) -> None:
        """Create a minimal zarr store at *path*, optionally with a _manifest attr."""
        import zarr

        z = zarr.open(path, mode="w")
        if manifest is not None:
            z.attrs["_manifest"] = manifest.to_dict()

    def test_chains_upstream_roi_hash(self, tmp_path):
        roi = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        roi_path = str(tmp_path / "roi.zarr")
        self._write_roi_store(roi_path, roi)

        ingest = IngestManifest.from_roi_store(roi_path)

        # The chained hash must equal the ROI manifest's own hash.
        assert ingest.roi_manifest_hash == roi.hash()

    def test_roi_hash_changes_when_roi_config_changes(self, tmp_path):
        roi_a = RoiManifest(resolution=10.0, chunk_size=2000, crs="EPSG:32615")
        roi_b = RoiManifest(resolution=20.0, chunk_size=2000, crs="EPSG:32615")
        path_a = str(tmp_path / "a.zarr")
        path_b = str(tmp_path / "b.zarr")
        self._write_roi_store(path_a, roi_a)
        self._write_roi_store(path_b, roi_b)

        hash_a = IngestManifest.from_roi_store(path_a).roi_manifest_hash
        hash_b = IngestManifest.from_roi_store(path_b).roi_manifest_hash
        assert hash_a != hash_b

    def test_legacy_roi_store_without_manifest_yields_none(self, tmp_path):
        """An ROI store with no _manifest attr chains to a None hash (no crash)."""
        roi_path = str(tmp_path / "legacy.zarr")
        self._write_roi_store(roi_path, manifest=None)

        ingest = IngestManifest.from_roi_store(roi_path)
        assert ingest.roi_manifest_hash is None


class TestEmbeddingManifestFromUpstreamStores:
    """Tests for EmbeddingManifest.from_upstream_stores (reflectance + SAR chaining)."""

    def _ingest_dict(self, roi_hash: str) -> dict:
        return IngestManifest(roi_manifest_hash=roi_hash).to_dict()

    def test_chains_reflectance_hash(self):
        ref = self._ingest_dict("roi_ref")
        m = EmbeddingManifest.from_upstream_stores(
            model_checkpoint="tessera_v1_1_aws_encoder",
            num_obs_checkpoints=(8, 16, 32),
            upstream_manifests={"reflectance": ref},
        )
        assert m.reflectance_manifest_hash == IngestManifest.from_dict(ref).hash()
        assert m.sar_manifest_hash is None

    def test_combines_both_sar_orbits(self):
        """Both SAR orbits combine into a single deterministic hash."""
        upstream = {
            "reflectance": self._ingest_dict("roi_ref"),
            "sar_ascending": self._ingest_dict("roi_asc"),
            "sar_descending": self._ingest_dict("roi_desc"),
        }
        m = EmbeddingManifest.from_upstream_stores(
            model_checkpoint="tessera_v1_1_aws_encoder",
            num_obs_checkpoints=(8, 16, 32),
            upstream_manifests=upstream,
        )
        assert m.sar_manifest_hash is not None
        assert len(m.sar_manifest_hash) == 16

    def test_sar_hash_order_independent_of_dict_order(self):
        """The combined SAR hash depends on ascending|descending order, not dict order."""
        asc = self._ingest_dict("roi_asc")
        desc = self._ingest_dict("roi_desc")
        m1 = EmbeddingManifest.from_upstream_stores("ckpt", (8,), {"sar_ascending": asc, "sar_descending": desc})
        m2 = EmbeddingManifest.from_upstream_stores("ckpt", (8,), {"sar_descending": desc, "sar_ascending": asc})
        assert m1.sar_manifest_hash == m2.sar_manifest_hash

    def test_single_sar_orbit_differs_from_both(self):
        """One orbit present must not collide with both orbits present."""
        asc = self._ingest_dict("roi_asc")
        desc = self._ingest_dict("roi_desc")
        only_asc = EmbeddingManifest.from_upstream_stores("ckpt", (8,), {"sar_ascending": asc}).sar_manifest_hash
        both = EmbeddingManifest.from_upstream_stores(
            "ckpt", (8,), {"sar_ascending": asc, "sar_descending": desc}
        ).sar_manifest_hash
        assert only_asc != both

    def test_no_upstream_yields_none_hashes(self):
        m = EmbeddingManifest.from_upstream_stores("ckpt", (8, 16), upstream_manifests={})
        assert m.reflectance_manifest_hash is None
        assert m.sar_manifest_hash is None
        assert m.num_obs_checkpoints == (8, 16)
