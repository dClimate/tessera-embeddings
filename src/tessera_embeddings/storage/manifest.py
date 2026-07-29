"""Store manifest: structural parameter validation for Zarr append safety.

Every Zarr store (ROI, satellite ingestion, embeddings) has "structural
parameters" — settings like resolution, CRS, and chunk size that must be
identical across all data in the store.  Changing these parameters and
appending would silently corrupt the data.

This module provides typed dataclass manifests that are written into zarr
root attrs at creation time and validated before every append or reuse.
If the manifest doesn't match the current config, a ``ConfigMismatchError``
is raised with a human-readable diff.

Legacy stores (created before this module existed) have no ``_manifest``
attr and receive a warning instead of an error, allowing a gradual rollout.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Self, cast

import zarr

from tessera_embeddings.errors import ConfigMismatchError

logger = logging.getLogger(__name__)


def _normalize(v: Any) -> Any:  # noqa: ANN401
    """Normalize a value for JSON/zarr-attr round-trip equality.

    Zarr attrs round-trip through JSON, which turns tuples into lists. To keep
    stored and in-memory manifests comparable, coerce tuples (recursively) to
    lists at serialization time.
    """
    if isinstance(v, tuple | list):
        return [_normalize(x) for x in v]
    return v


def extract_manifest(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract the ``_manifest`` dict from zarr/xarray root attrs.

    Centralizes the common pattern of reading ``_manifest`` from store attrs,
    handling both zarr groups and xarray datasets.
    """
    if "_manifest" not in attrs:
        return None
    val = attrs.get("_manifest")
    return dict(val) if val is not None else None


# ── Base class ──


@dataclass(frozen=True)
class StoreManifest:
    """Base class for typed store manifests.

    Subclasses declare their structural fields as dataclass fields.
    Provides serialization, hashing, and validation.

    Each manifest only contains fields that are *set independently* at its
    stage.  Upstream structural params are captured transitively by recording
    the upstream manifest's hash — if any upstream field changes, the hash
    changes, which triggers a mismatch here without duplicating fields.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for writing to zarr attrs.

        Tuples are normalized to lists so that round-tripping through JSON
        (zarr attrs) doesn't cause equality mismatches during validation.
        """
        d = {k: _normalize(v) for k, v in asdict(self).items() if v is not None}
        d["manifest_type"] = type(self).__name__
        return d

    def hash(self) -> str:
        """Compute a stable SHA-256 hash (first 16 hex chars).

        Used to chain manifests: a downstream store records the hash of its
        upstream store's manifest, so changes propagate transitively.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Deserialize from a dict (e.g., from zarr attrs).

        Ignores keys not present in the dataclass definition so legacy
        manifests with extra/missing keys don't break deserialization.
        """
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    def validate_against(self, existing: dict[str, Any] | None, store_path: str) -> None:
        """Validate that this manifest matches an existing store's manifest.

        Args:
            existing: The ``_manifest`` dict from the existing store's root
                attrs, or ``None`` if the store has no manifest (legacy store).
            store_path: Store path (for error messages only).

        Raises:
            ConfigMismatchError: If any structural parameter differs.
        """
        if existing is None:
            logger.warning(
                "No _manifest in %s — legacy store, skipping structural validation. "
                "Re-create the store to enable manifest-based safety checks.",
                store_path,
            )
            return

        # Compare keys present in current against existing (value changed?),
        # and also flag known fields that the store has but current dropped
        # (e.g. an optional upstream hash that's None this run). Keys in
        # existing but not in our schema are ignored (forward-compat: old
        # code won't fail on fields added by a newer version).
        # Cross-type confusion is caught by manifest_type.
        current = self.to_dict()
        known_fields = {f.name for f in fields(self)} | {"manifest_type"}
        mismatches: dict[str, tuple[Any, Any]] = {}
        for key in current:
            existing_val = existing.get(key)
            current_val = current[key]
            if existing_val is not None and existing_val != current_val:
                mismatches[key] = (existing_val, current_val)
        for key in existing:
            if key in known_fields and key not in current and existing[key] is not None:
                mismatches[key] = (existing[key], None)

        if mismatches:
            lines = [f"Cannot append to {store_path}: structural parameters changed."]
            for key, (old, new) in mismatches.items():
                lines.append(f"  {key}: store has {old!r}, current config has {new!r}")
            lines.append("Delete the existing store or use a different name.")
            raise ConfigMismatchError("\n".join(lines))


# ── Concrete manifests ──


@dataclass(frozen=True)
class RoiManifest(StoreManifest):
    """Manifest for ROI stores: pixel size, chunk size, and CRS must match."""

    resolution: float
    chunk_size: int
    # None when force_crs is unset (auto-detect mode) at the flow level.
    crs: str | None = None


@dataclass(frozen=True)
class IngestManifest(StoreManifest):
    """Manifest for satellite ingestion stores: upstream ROI identity.

    Resolution and CRS are captured transitively via ``roi_manifest_hash``.

    ``coverage_sha256`` pins the LAND MASK the ROI was built from, and it is the one
    piece of identity nothing else caught. ``RoiManifest`` records resolution, chunk
    size and CRS, so two ROIs built from different coverage deliveries of the same zone
    hash IDENTICALLY — same grid, different land. Writes only ever appended, so a
    changed mask would have mixed windows from two geometries with nothing objecting.

    Recorded here because the manifest is validated on EVERY write
    (``validate_against``), which makes a mask change a loud failure at the first
    append rather than a comparison somebody has to remember to make. That matters
    more now that an interrupted store is RESUMED rather than rebuilt: resume assumes
    the existing pixels were computed from the same mask, and this is what enforces it.
    """

    roi_manifest_hash: str | None = None
    coverage_sha256: str | None = None

    @classmethod
    def from_roi_store(cls, roi_zarr_path: str) -> IngestManifest:
        """Build from an ROI store, chaining the upstream ROI manifest hash.

        Args:
            roi_zarr_path: Path to the ROI Zarr store.
        """
        coverage_sha: str | None = None
        try:
            z = zarr.open(roi_zarr_path, mode="r")
            roi_manifest_dict = extract_manifest(z.attrs)
            # A zone ROI carries the coverage delivery it was exported from
            # (``land_mask.export_zone_roi``). Absent for a plain single-ROI store,
            # which has no land mask behind it — None is then correct, not a gap.
            coverage_sha = cast("str | None", z.attrs.get("coverage_sha256"))
        except (KeyError, ValueError) as exc:
            logger.warning("Could not read _manifest from ROI store %s: %s", roi_zarr_path, exc)
            roi_manifest_dict = None
        roi_mhash = RoiManifest.from_dict(roi_manifest_dict).hash() if roi_manifest_dict else None
        return cls(roi_manifest_hash=roi_mhash, coverage_sha256=coverage_sha)


@dataclass(frozen=True)
class EmbeddingManifest(StoreManifest):
    """Manifest for embedding stores: model + inference config + upstream ingestion identity.

    Under Tessera v1.1, sampling is deterministic (no random repeats) and every
    valid observation is used with bucketed sequence lengths. ``num_obs_checkpoints``
    replaces v1.0's ``repeat_times``/``sample_size_s2`` as the structural param.
    """

    model_checkpoint: str
    num_obs_checkpoints: tuple[int, ...]
    reflectance_manifest_hash: str | None = None
    sar_manifest_hash: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmbeddingManifest:
        """Deserialize, coercing ``num_obs_checkpoints`` list → tuple."""
        field_names = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in d.items() if k in field_names}
        if "num_obs_checkpoints" in kwargs:
            kwargs["num_obs_checkpoints"] = tuple(int(v) for v in kwargs["num_obs_checkpoints"])
        return cls(**kwargs)

    @classmethod
    def from_upstream_stores(
        cls,
        model_checkpoint: str,
        num_obs_checkpoints: tuple[int, ...],
        upstream_manifests: dict[str, dict[str, Any] | None],
    ) -> EmbeddingManifest:
        """Build from upstream ingest store manifests.

        Args:
            model_checkpoint: Model version string.
            num_obs_checkpoints: Bucketed sequence-length checkpoints used by the sampler.
            upstream_manifests: Map of store name to manifest dict (or None).
                Expected keys: ``"reflectance"``, optionally ``"sar_ascending"``
                and/or ``"sar_descending"``.
        """
        ref_dict = upstream_manifests.get("reflectance")
        ref_hash = IngestManifest.from_dict(ref_dict).hash() if ref_dict else None

        sar_hashes: list[str] = []
        for name in ("sar_ascending", "sar_descending"):
            d = upstream_manifests.get(name)
            if d is not None:
                sar_hashes.append(IngestManifest.from_dict(d).hash())
        sar_hash: str | None = None
        if sar_hashes:
            combined = "|".join(sar_hashes)
            sar_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]

        return cls(
            model_checkpoint=model_checkpoint,
            num_obs_checkpoints=tuple(int(v) for v in num_obs_checkpoints),
            reflectance_manifest_hash=ref_hash,
            sar_manifest_hash=sar_hash,
        )
