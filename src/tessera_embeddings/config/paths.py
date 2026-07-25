"""Storage path configuration.

Replaces the hardcoded ``s3://cl-tessera-*`` bucket names from the reference
repo. Paths are supplied by caller configuration, not derived from a
``dev/prod`` boolean toggle.

The ``BucketPaths`` model is what all domain code sees. Prefect flows and the
plain runner populate it differently (env vars, YAML config, Prefect Block)
but domain function signatures are identical.

URIs may use any fsspec-supported protocol: ``s3://``, ``gs://``,
``file://``, ``/absolute/local/path``, etc.
"""

from __future__ import annotations

import posixpath
from typing import final

from pydantic import BaseModel, Field

_INPUT_KINDS: frozenset[str] = frozenset({"reflectance", "sar_ascending", "sar_descending", "roi"})
_OUTPUT_KINDS: frozenset[str] = frozenset({"embeddings"})
_ALL_KINDS: frozenset[str] = _INPUT_KINDS | _OUTPUT_KINDS


@final
class BucketPaths(BaseModel):
    """Base storage URIs for each pipeline stage.

    All fields accept any fsspec-supported URI. Domain functions receive a
    ``BucketPaths`` instance and construct store URIs via :meth:`store_for`.
    """

    inputs: str = Field(..., description="Base URI for ROI masks and intermediate ingest stores.")
    outputs: str = Field(..., description="Base URI for final embedding outputs.")

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

        Zone masks are ordinary ``roi``-kind stores under the reserved
        ``zone_{zone}`` name, so a zone mask and a conventionally-named ROI can
        never collide. Every producer and consumer of a zone mask addresses it
        through here: a mask written to a path the ingest does not read would
        look like success and then be silently rebuilt per cell.
        """
        return self.store_for(f"zone_{zone}", "roi")

    def global_store(self, name: str = "tessera") -> str:
        """Return the URI of the single global-embeddings Icechunk repo.

        The global campaign writes all 120 UTM-zone groups into one repo
        (ADR-008 D5), addressed by zone group name — unlike :meth:`store_for`,
        which is one ``.zarr`` per (roi, kind).
        """
        return posixpath.join(self.outputs, "global", f"{name}.icechunk")

    def land_mask_store(self, name: str = "global") -> str:
        """Return the URI of the campaign land-mask coverage Icechunk repo.

        One repo of 120 UTM-zone groups (ADR-010), each holding registry-derived
        coverage bitmaps (``tile_live_2048`` / ``chunk_live_256``) addressed by
        zone group name — mirroring :meth:`global_store` so the zone-fill runner
        reads it with the same ``open_store_as_zarr_group(path, group=zone)``
        helper. Lives under ``inputs`` (it is a campaign input, like ROI masks).
        """
        return posixpath.join(self.inputs, "masks", f"{name}.icechunk")
