"""Storage path configuration.

Paths come from caller configuration — never a hardcoded bucket name, never a ``dev/prod``
boolean toggle. The ``BucketPaths`` model is what all domain code sees; Prefect flows and
the plain runner populate it differently (env vars, YAML config, Prefect Block) but domain
function signatures are identical.

URIs may use any fsspec-supported protocol: ``s3://``, ``gs://``, ``file://``,
``/absolute/local/path``, etc.
"""

from __future__ import annotations

import posixpath
from typing import final

from pydantic import BaseModel, Field

_INPUT_KINDS: frozenset[str] = frozenset({"reflectance", "sar_ascending", "sar_descending", "roi"})
_OUTPUT_KINDS: frozenset[str] = frozenset({"embeddings"})
_ALL_KINDS: frozenset[str] = _INPUT_KINDS | _OUTPUT_KINDS


#: The derived store's basename, and the only ``name`` an override tolerates. Named here so the
#: guard in :meth:`BucketPaths.global_store` and that method's own default cannot drift apart.
_DEFAULT_GLOBAL_STORE_NAME = "tessera"


@final
class BucketPaths(BaseModel):
    """Base storage URIs for each pipeline stage.

    All fields accept any fsspec-supported URI. Domain functions receive a ``BucketPaths``
    instance and construct store URIs via :meth:`store_for`.
    """

    inputs: str = Field(..., description="Base URI for ROI masks and intermediate ingest stores.")
    outputs: str = Field(..., description="Base URI for final embedding outputs.")
    global_store_uri: str | None = Field(
        default=None,
        description=(
            "Full URI of the global-embeddings repo, overriding the path derived from `outputs`. "
            "For publishing to a location whose shape the derivation cannot produce — a different "
            "bucket, a different prefix, a different suffix. None means derive it."
        ),
    )

    def store_for(self, roi_name: str, kind: str) -> str:
        """Return the canonical store URI for ``(roi_name, kind)``.

        Args:
            roi_name: Identifier for the region of interest (e.g. ``"33UWP"``).
            kind: One of ``"reflectance"``, ``"sar_ascending"``,
                ``"sar_descending"``, ``"embeddings"``, or ``"roi"``.

        Returns:
            Full URI to the Zarr store. Path structure varies by kind:
            ``roi`` → ``{inputs}/rois/zarrs/{roi_name}.zarr``,
            mosaic kinds → ``{inputs}/mosaics/{roi_name}/{kind}.zarr``,
            ``embeddings`` → ``{outputs}/embeddings/{roi_name}.zarr``.

        Raises:
            ValueError: If ``kind`` is not one of the recognised values.
        """
        if kind not in _ALL_KINDS:
            raise ValueError(f"Unknown store kind {kind!r}. Expected one of: {sorted(_ALL_KINDS)}")

        if kind == "roi":
            return posixpath.join(self.inputs, "rois", "zarrs", f"{roi_name}.zarr")
        elif kind in {"reflectance", "sar_ascending", "sar_descending"}:
            return posixpath.join(self.inputs, "mosaics", roi_name, f"{kind}.zarr")
        else:  # embeddings
            return posixpath.join(self.outputs, "embeddings", f"{roi_name}.zarr")

    def zone_roi_store(self, zone: str) -> str:
        """Return the URI of the campaign ingest ROI mask for one UTM zone.

        Zone masks are ordinary ``roi``-kind stores under the reserved ``zone_{zone}`` name,
        so a zone mask and a conventionally-named ROI can never collide. Every producer and
        consumer addresses one through here: a mask written to a path the ingest does not
        read looks like success and is then silently rebuilt per cell.
        """
        return self.store_for(f"zone_{zone}", "roi")

    def global_store(self, name: str = _DEFAULT_GLOBAL_STORE_NAME) -> str:
        """Return the URI of the single global-embeddings Icechunk repo.

        The global campaign writes all 120 UTM-zone groups into one repo (ADR-008 D5),
        addressed by zone group name — unlike :meth:`store_for`, one ``.zarr`` per
        (roi, kind).

        **``global_store_uri`` overrides the derivation entirely**, because a published
        location need not be shaped like one this method could build: a different bucket, no
        ``global/`` segment, a name chosen by whoever owns the bucket. Every producer and
        consumer of the campaign store asks this one method, which is what makes a single
        field enough and makes it impossible for one tool to write the override while
        another reads the derived path.

        Args:
            name: Repo basename, used only when deriving.

        Raises:
            ValueError: If an override is set AND a caller asks for a non-default ``name``.
                The override IS the store, so the name has nowhere to go, and a silently
                ignored argument leaves a caller certain it addressed a store that does not
                exist.
        """
        if self.global_store_uri:
            if name != _DEFAULT_GLOBAL_STORE_NAME:
                raise ValueError(
                    f"global_store(name={name!r}) with global_store_uri={self.global_store_uri!r}: "
                    "the override is the whole URI, so a store name cannot be honoured. Drop the "
                    "name, or drop the override."
                )
            return self.global_store_uri
        return posixpath.join(self.outputs, "global", f"{name}.icechunk")

    def optical_registry(self, store_uri: str) -> str:
        """Return the URI of the per-shard registry that sits BESIDE ``store_uri``.

        A Parquet dataset indexing what the store contains per shard and year — what was
        embedded, what was refused and how close the refusals were to the line — so a
        consumer can answer "is my area covered, and how well" with one read instead of
        opening a petabyte.

        **Takes the store URI, and the argument is required.** Anything derived instead (the
        default repo basename, or ``outputs``) can name a store the run is not writing, and
        the failure is silent: every tool works, the Parquet part is valid, and it describes
        a zone-year that store does not contain. Worse than an overwrite, because two
        stores' parts then merge into one dataset under the same partition keys with nothing
        able to tell them apart. Deriving from ``outputs`` fails the same way in the other
        direction — production publishes through :attr:`global_store_uri` to a bucket that
        is not ours, so the registry would sit in our bucket and the store in the public
        one. One input, and it IS the store. Incident and schema:
        context_docs/design/optical-registry-2026-08-19.md.

        A SIBLING of the store rather than a path inside it, because Icechunk owns every key
        under its own prefix: garbage collection enumerates that prefix and reconciles it
        against its own manifests, so a Parquet file there is at best unrecognised and at
        worst collected.
        """
        # TRAILING SEPARATORS FIRST. An override ending in "/" leaves `name` empty and the
        # sibling comes out as `.../dclimate.icechunk/.registry` — INSIDE the Icechunk-owned
        # prefix, the one place a Parquet file must never live (see above). A configured URI
        # is human-entered, so the trailing slash is a matter of time.
        store = store_uri.rstrip("/")
        base, sep, name = store.rpartition("/")
        stem = name[: -len(".icechunk")] if name.endswith(".icechunk") else name
        # A store URI with no separator — a bare relative path like ``tessera.icechunk``,
        # which local runs and tests use — leaves ``base`` empty, and f"{base}/{stem}" would
        # name ``/tessera.registry`` at the FILESYSTEM ROOT: a permissions failure if you are
        # lucky, a write unrelated to the configured store if you are not.
        return f"{base}{sep}{stem}.registry"

    def land_mask_store(self, name: str = "global") -> str:
        """Return the URI of the campaign land-mask coverage Icechunk repo.

        One repo of 120 UTM-zone groups (ADR-010), each holding registry-derived coverage
        bitmaps (``tile_live_2048`` / ``chunk_live_256``) addressed by zone group name —
        mirroring :meth:`global_store` so the zone-fill runner reads it with the same
        ``open_store_as_zarr_group(path, group=zone)`` helper. Lives under ``inputs``, being
        a campaign input like the ROI masks.
        """
        return posixpath.join(self.inputs, "masks", f"{name}.icechunk")
