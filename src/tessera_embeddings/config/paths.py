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

from pydantic import BaseModel, Field

_INPUT_KINDS: frozenset[str] = frozenset({"reflectance", "sar_ascending", "sar_descending", "roi"})
_OUTPUT_KINDS: frozenset[str] = frozenset({"embeddings"})
_PREPROCESSED_KINDS: frozenset[str] = frozenset({"staging"})
_ALL_KINDS: frozenset[str] = _INPUT_KINDS | _OUTPUT_KINDS | _PREPROCESSED_KINDS


class BucketPaths(BaseModel):
    """Base storage URIs for each pipeline stage.

    All fields accept any fsspec-supported URI. Domain functions receive a
    ``BucketPaths`` instance and construct store URIs via :meth:`store_for`.
    """

    inputs: str = Field(..., description="Base URI for ROI masks and intermediate ingest stores.")
    outputs: str = Field(..., description="Base URI for final embedding outputs.")
    preprocessed: str = Field(..., description="Base URI for mosaicked store outputs.")

    def store_for(self, roi_name: str, kind: str) -> str:
        """Return the canonical store URI for ``(roi_name, kind)``.

        Args:
            roi_name: Identifier for the region of interest (e.g. ``"33UWP"``).
            kind: One of ``"reflectance"``, ``"sar_ascending"``,
                ``"sar_descending"``, ``"embeddings"``, ``"staging"``,
                or ``"roi"``.

        Returns:
            Full URI to the Zarr/Icechunk store.

        Raises:
            ValueError: If ``kind`` is not one of the recognised values.
        """
        if kind not in _ALL_KINDS:
            raise ValueError(f"Unknown store kind {kind!r}. Expected one of: {sorted(_ALL_KINDS)}")

        if kind in _INPUT_KINDS:
            base = self.inputs
        elif kind in _OUTPUT_KINDS:
            base = self.outputs
        else:
            base = self.preprocessed

        return posixpath.join(base, roi_name, kind)
