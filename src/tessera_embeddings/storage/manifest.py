"""Store manifest: structural parameter validation for Zarr append safety.

Every Zarr store (ROI, satellite ingestion, embeddings) has "structural parameters" —
resolution, CRS, chunk size and the like — that must be identical across all data in the
store. Changing one and appending would silently corrupt the data.

Typed dataclass manifests are written into zarr root attrs at creation time and validated
before every append or reuse; a mismatch raises ``ConfigMismatchError`` with a
human-readable diff. A store with no ``_manifest`` attr at all gets a warning rather than
an error — see :meth:`StoreManifest.validate_against` for why that asymmetry is deliberate.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Self, cast

import zarr

from tessera_embeddings.config.ingest import ingest_code_identity
from tessera_embeddings.errors import ConfigMismatchError

logger = logging.getLogger(__name__)


def _normalize(v: Any) -> Any:  # noqa: ANN401
    """Normalize a value for JSON/zarr-attr round-trip equality.

    Zarr attrs round-trip through JSON, which turns tuples into lists, so tuples are
    coerced (recursively) to lists at serialization time to keep stored and in-memory
    manifests comparable.
    """
    if isinstance(v, tuple | list):
        return [_normalize(x) for x in v]
    return v


def extract_manifest(attrs: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract the ``_manifest`` dict from zarr/xarray root attrs (either kind of node)."""
    if "_manifest" not in attrs:
        return None
    val = attrs.get("_manifest")
    return dict(val) if val is not None else None


_CODE_IDENTITY = "ingest_code_identity"
_OVERRIDE_FIELD = "allow_ingest_code_mismatch"
#: Root attr: the ingest code identities a store's dates were produced under. Not in
#: ``_manifest``, which is written once at create time — the mixture becomes true later.
#:
#: **An audit trail, and deliberately nothing more.** Nothing reads this attr to decide
#: anything: ``read_upstream_manifests`` extracts only ``_manifest``, so an embedding store's
#: upstream identity is the same whether its input was single-code or mixed, and the zone-year
#: completion marker does not carry it either, so a later strict run short-circuits on the
#: marker and accepts a mixed mosaic.
#:
#: Intended, not an omission. The override is never automatic — an operator chooses it per run
#: knowing the store will hold data from two versions of the ingest code, and choosing that and
#: then refusing to embed the result would mean ingesting imagery that can never be inferred on.
#: A mixed mosaic is a first-class input; this records WHICH versions contributed.
MIXED_CODE_IDENTITIES_ATTR = "mixed_ingest_code_identities"


# ── Base class ──


@dataclass(frozen=True)
class StoreManifest:
    """Base class for typed store manifests: serialization, hashing, and validation.

    Subclasses declare their structural fields as dataclass fields, and only those *set
    independently* at that stage. Upstream structural params are captured transitively by
    recording the upstream manifest's hash: any upstream change moves the hash and so
    triggers a mismatch here, without duplicating fields.
    """

    ABSENT_MEANS_OFF: ClassVar[frozenset[str]] = frozenset()
    """Fields whose absence from a store is a VALUE, not a gap — see ``validate_against``.

    A subclass lists a field here when the store having no opinion is itself an opinion: a
    policy that decides what the data IS, where "written before anyone recorded it" and
    "written with it off" are the same state. Everything else keeps the default, under which
    an unrecorded field is simply unknown and cannot disagree with anything.
    """

    REQUIRED_TO_APPEND: ClassVar[frozenset[str]] = frozenset()
    """Fields whose absence from a store means it cannot be SHOWN safe to append to.

    The third answer to "the store does not record this". :data:`ABSENT_MEANS_OFF` reads
    absence as a known value, which suits a boolean policy and nothing else; the default
    reads it as unknown and lets the append through, which is right for a field that merely
    DESCRIBES a store and exactly wrong for one that exists to make a resume safe — the
    guard then protects every store except the ones written before it existed, which are the
    only stores it was ever needed for.

    A field belongs here when this run knows its value, the store records none, and
    appending anyway would mix two policies in one store with nothing recording it. The
    refusal is deliberately blunt (such a store must be rebuilt) and cheap where it fires,
    because a store being appended to is by definition incomplete.

    Only fires when the CURRENT manifest carries the field, so a value legitimately absent
    this run (no land mask behind a plain ROI, no admission threshold on a radar store)
    never reaches the check.
    """

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for writing to zarr attrs.

        Tuples are normalized to lists so the JSON round-trip through zarr attrs does not
        cause equality mismatches during validation. ``allow_ingest_code_mismatch`` is a
        run's decision, not the store's identity, so it is dropped: never stored, never
        hashed.
        """
        d = {k: _normalize(v) for k, v in asdict(self).items() if v is not None and k != _OVERRIDE_FIELD}
        d["manifest_type"] = type(self).__name__
        return d

    def hash(self) -> str:
        """Compute a stable SHA-256 hash (first 16 hex chars).

        Chains manifests: a downstream store records its upstream's hash, so changes
        propagate transitively.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Deserialize from a dict (e.g. zarr attrs), ignoring keys the dataclass lacks.

        Older manifests with extra or missing keys therefore still deserialize.
        """
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in field_names})

    def validate_against(self, existing: dict[str, Any] | None, store_path: str) -> list[str]:
        """Validate that this manifest matches an existing store's manifest.

        Args:
            existing: The ``_manifest`` dict from the existing store's root
                attrs, or ``None`` if the store has no manifest (legacy store).
            store_path: Store path (for error messages only).

        Returns:
            Identities the caller must record under :data:`MIXED_CODE_IDENTITIES_ATTR`;
            empty unless :meth:`IngestManifest.validate_against` excused a mismatch.

        Raises:
            ConfigMismatchError: If any structural parameter differs.
        """
        if existing is None:
            # A store with NO manifest is treated more softly than one with a PARTIAL
            # manifest, which REQUIRED_TO_APPEND refuses — deliberately, not by oversight. No
            # manifest cannot distinguish "old store" from "store built under a policy that
            # has since changed", so refusing would strand every pre-manifest artifact on a
            # suspicion. A partial manifest is evidence rather than silence: the store
            # demonstrably kept a manifest and demonstrably lacks the field.
            logger.warning(
                "No _manifest in %s — legacy store, skipping structural validation. Nothing "
                "here can tell whether it was built under this run's mask, threshold or "
                "ingest code, so an append MAY mix policies. Re-create the store to enable "
                "manifest-based safety checks.",
                store_path,
            )
            return []

        # Compare each key in current against existing (value changed?), and flag known
        # fields the store has but current dropped (e.g. an optional upstream hash that is
        # None this run). Keys in existing but not in our schema are ignored, so old code
        # does not fail on fields a newer version added. Cross-type confusion is caught by
        # manifest_type.
        current = self.to_dict()
        known_fields = {f.name for f in fields(self)} | {"manifest_type"}
        mismatches: dict[str, tuple[Any, Any]] = {}
        for key in current:
            existing_val = existing.get(key)
            current_val = current[key]
            if existing_val is not None and existing_val != current_val:
                mismatches[key] = (existing_val, current_val)
            elif existing_val is None and key in self.REQUIRED_TO_APPEND:
                # The store cannot demonstrate it was built under this run's policy, and a
                # resume's whole premise is that it was. Refused rather than assumed — see
                # REQUIRED_TO_APPEND for why "unknown" must not read as "compatible" here.
                mismatches[key] = ("<not recorded — cannot be shown compatible>", current_val)
            elif existing_val is None and key in self.ABSENT_MEANS_OFF:
                # ABSENT is a VALUE for these fields, not a gap. The general rule skips a key
                # the store lacks so a manifest from newer code can carry fields older stores
                # never had — right for a field that DESCRIBES a store, wrong for one stating
                # a POLICY it was built under. Turning such a policy on is exactly the append
                # to refuse, and the only shape the general rule cannot see: the store says
                # nothing, the current manifest says True, and nothing disagrees.
                mismatches[key] = ("<not recorded, i.e. off>", current_val)
        for key in existing:
            if key in known_fields and key not in current and existing[key] is not None:
                mismatches[key] = (existing[key], None)

        if mismatches:
            lines = [f"Cannot append to {store_path}: structural parameters changed."]
            for key, (old, new) in mismatches.items():
                lines.append(f"  {key}: store has {old!r}, current config has {new!r}")
            lines.append("Delete the existing store or use a different name.")
            raise ConfigMismatchError("\n".join(lines))
        return []


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

    Resolution and CRS are captured transitively via ``roi_manifest_hash``. The other three
    fields each close one version of the same hole — a resume that appends pixels built
    under one policy onto pixels built under another, with nothing objecting. All are
    validated on EVERY write, which makes such a change a loud failure at the first append
    rather than a comparison somebody has to remember to make.

    ``coverage_sha256`` pins the LAND MASK the ROI was built from. ``RoiManifest`` records
    resolution, chunk size and CRS, so two ROIs built from different coverage deliveries of
    the same zone hash IDENTICALLY — same grid, different land. Resume assumes the existing
    pixels came from the same mask; this is what enforces it.

    ``min_valid_coverage`` is the admission threshold: the mask says WHERE a date could
    land, this decides WHICH dates were admitted at all. An interrupted store carries no
    other record of it, so a resume at a different threshold would skip the dates the old
    run admitted, append new ones under the new rule, and finish with a mosaic built to two
    policies — against the campaign's contract that a changed parameter raises. Optical
    only: the SAR stores have no admission threshold, and their orbit is in the store name.

    ``ingest_code_identity`` says by what CODE — the query, the solar-day normalisation, the
    duplicate-copy preference, the OPERA granule filter. A campaign runs for weeks and its
    ingest source does change in that time, so a resume after a deploy would otherwise stamp
    one fingerprint over dates built two ways. Validated on append rather than folded into
    the completion marker deliberately: the marker decides whether a FINISHED mosaic is
    reusable, and a code hash there would declare every one of them stale on any ingest
    change.
    """

    #: ONE field, not three, and the difference is what each of them can actually change.
    #:
    #: ``ingest_code_identity`` is here because the ingest source DOES change mid-campaign:
    #: the duplicate-copy selection changed which granule supplies a date. A mosaic written
    #: before this field carries no record of the code that built it, so resuming it under
    #: today's code mixes exactly what the field prevents — absence is "cannot be shown
    #: compatible", not "nothing to contradict", and the store is incomplete anyway.
    #:
    #: ``coverage_sha256`` and ``min_valid_coverage`` are deliberately NOT here, and their
    #: legacy allowance stays (``test_a_store_predating_the_field_is_not_retro_blocked``). A
    #: campaign holds its mask and admission threshold fixed by contract, and the repo owner
    #: has ruled that no mid-campaign mask change will occur — so refusing a store for not
    #: recording a value that cannot have changed would guard a pattern nobody anticipates,
    #: at the cost of stranding multi-terabyte mosaics.
    REQUIRED_TO_APPEND: ClassVar[frozenset[str]] = frozenset({"ingest_code_identity"})

    roi_manifest_hash: str | None = None
    coverage_sha256: str | None = None
    min_valid_coverage: float | None = None
    ingest_code_identity: str | None = None
    #: Opt-in: resume a store whose recorded ``ingest_code_identity`` is not this run's.
    #: Relaxes only that term; the append is recorded under :data:`MIXED_CODE_IDENTITIES_ATTR`.
    allow_ingest_code_mismatch: bool = False

    def validate_against(self, existing: dict[str, Any] | None, store_path: str) -> list[str]:
        """Validate, excusing an ``ingest_code_identity`` mismatch when this run opted in."""
        stored = (existing or {}).get(_CODE_IDENTITY)
        # `stored is None` keeps the ordinary refusal: the override excuses a DIFFERENT known
        # identity, not a missing one. Substituting this run's identity for an absent field
        # would satisfy `REQUIRED_TO_APPEND` with nothing compared, then record the literal
        # string "None" as though it were a second producer.
        if (
            existing is None
            or stored is None
            or not self.allow_ingest_code_mismatch
            or stored == self.ingest_code_identity
        ):
            return super().validate_against(existing, store_path)
        logger.warning(
            "%s: appending under allow_ingest_code_mismatch — store built by %s, this run is %s; both recorded.",
            store_path,
            stored,
            self.ingest_code_identity,
        )
        # Substituting this ONE term is what keeps it narrow: every other key still compared.
        super().validate_against({**existing, _CODE_IDENTITY: self.ingest_code_identity}, store_path)
        return sorted([str(stored), str(self.ingest_code_identity)])

    @classmethod
    def from_roi_store(
        cls,
        roi_zarr_path: str,
        *,
        min_valid_coverage: float | None = None,
        storage_options: dict | None = None,
        allow_ingest_code_mismatch: bool = False,
    ) -> IngestManifest:
        """Build from an ROI store, chaining the upstream ROI manifest hash.

        Args:
            roi_zarr_path: Path to the ROI Zarr store.
            storage_options: fsspec options for the open. The ROI is a plain zarr, read
                through fsspec rather than Icechunk, so it does not travel on the credential
                callback the rest of an ingest leg threads — without this, an open that
                happens before any date is processed runs on the ambient chain.
            min_valid_coverage: The admission threshold this ingest applies. ``None`` (the
                SAR stores, which have no such gate) is dropped from the manifest by
                ``to_dict``, so it neither appears in the store nor moves the hash — which
                keeps this field from invalidating manifests written before it existed.
            allow_ingest_code_mismatch: See the field of the same name.
        """
        coverage_sha: str | None = None
        try:
            z = zarr.open(roi_zarr_path, mode="r", storage_options=storage_options)
            roi_manifest_dict = extract_manifest(z.attrs)
            # A zone ROI carries the coverage delivery it was exported from
            # (``land_mask.export_zone_roi``). Absent for a plain single-ROI store, which has
            # no land mask behind it, where None is correct rather than a gap.
            coverage_sha = cast("str | None", z.attrs.get("coverage_sha256"))
        except (KeyError, ValueError) as exc:
            logger.warning("Could not read _manifest from ROI store %s: %s", roi_zarr_path, exc)
            roi_manifest_dict = None
        roi_mhash = RoiManifest.from_dict(roi_manifest_dict).hash() if roi_manifest_dict else None
        return cls(
            roi_manifest_hash=roi_mhash,
            coverage_sha256=coverage_sha,
            min_valid_coverage=min_valid_coverage,
            # Resolved here, at the one place both legs build their manifest, so the two
            # cannot disagree about what code they are — a value each computed for itself
            # is how they would drift.
            ingest_code_identity=ingest_code_identity(),
            allow_ingest_code_mismatch=allow_ingest_code_mismatch,
        )


@dataclass(frozen=True)
class EmbeddingManifest(StoreManifest):
    """Manifest for embedding stores: model + inference config + upstream ingestion identity.

    Under Tessera v1.1 sampling is deterministic (no random repeats) and every valid
    observation is used with bucketed sequence lengths, so ``num_obs_checkpoints`` is the
    sampler's structural param.
    """

    ABSENT_MEANS_OFF: ClassVar[frozenset[str]] = frozenset({"allow_s2_only", "optical_min_obs"})

    model_checkpoint: str
    num_obs_checkpoints: tuple[int, ...]
    reflectance_manifest_hash: str | None = None
    sar_manifest_hash: str | None = None
    #: ``True`` when the run embedded S2-valid pixels that had ZERO S1 observations. It
    #: decides which pixels become embeddings and which stay fill, so two appends to one
    #: store under different values leave time slices that are not comparable, and nothing
    #: else in the manifest would notice.
    #:
    #: OFF is recorded as ``None``, not ``False``, and the two mean the same thing here:
    #: ``to_dict`` drops ``None`` before hashing, so a store written before this field
    #: existed keeps its digest and still matches a later flag-off run — correct, since that
    #: store WAS written with the policy off. Only turning the policy ON moves the digest,
    #: which is the append this needs to catch.
    allow_s2_only: bool | None = None
    #: The minimum optical depth a pixel needed to be embedded, or ``None`` for no rule.
    #: Here for the same reason as ``allow_s2_only`` above and by the same argument: it
    #: decides which pixels become embeddings and which stay fill, so two appends under
    #: different lines leave time slices that are not comparable and no other field would
    #: notice. A store's advertised root rule is write-once for the same reason; this is the
    #: per-append half of it.
    #:
    #: In :data:`ABSENT_MEANS_OFF`, so a store written before this field existed still
    #: matches a later no-rule run — it genuinely had no rule — while an append that
    #: INTRODUCES a line is refused, the case the general "skip a key the store lacks" rule
    #: cannot see.
    optical_min_obs: int | None = None

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
        allow_s2_only: bool | None = None,
        optical_min_obs: int | None = None,
    ) -> EmbeddingManifest:
        """Build from upstream ingest store manifests.

        Args:
            model_checkpoint: Model version string.
            num_obs_checkpoints: Bucketed sequence-length checkpoints used by the sampler.
            upstream_manifests: Map of store name to manifest dict (or None). Expected keys:
                ``"reflectance"``, optionally ``"sar_ascending"`` / ``"sar_descending"``.
            allow_s2_only: The run's per-pixel S1 policy. Falsy is stored as ``None`` so the
                digest matches stores written before the field existed — see the field's own
                note for why those are the same state.
            optical_min_obs: The run's minimum optical depth, or ``None`` for no rule.
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
            allow_s2_only=allow_s2_only or None,
            optical_min_obs=optical_min_obs,
        )
