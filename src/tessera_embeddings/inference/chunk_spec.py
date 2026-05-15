"""Spatial chunk grid specification, enumeration, and ROI filtering.

Derives a grid of processing chunks from Zarr store metadata and filters it
against the ROI mask so only chunks with real coverage reach GPU actors.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import xarray as xr
import zarr

from tessera_embeddings.config.inference import DEFAULT_CHUNK_SIZE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkSpec:
    """Specification for one spatial chunk of the mosaic.

    Attributes:
        row: Row index in the chunk grid.
        col: Column index in the chunk grid.
        y_start: Start index along y dimension (inclusive).
        y_stop: Stop index along y dimension (exclusive).
        x_start: Start index along x dimension (inclusive).
        x_stop: Stop index along x dimension (exclusive).
    """

    row: int
    col: int
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int

    @property
    def height(self) -> int:
        """Height of this chunk in pixels."""
        return self.y_stop - self.y_start

    @property
    def width(self) -> int:
        """Width of this chunk in pixels."""
        return self.x_stop - self.x_start

    @property
    def label(self) -> str:
        """Human-readable label for this chunk."""
        return f"chunk_{self.row}_{self.col}"


def enumerate_chunks(
    total_y: int,
    total_x: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[ChunkSpec]:
    """Enumerate all spatial chunks for a mosaic of given dimensions.

    Args:
        total_y: Total height of the mosaic in pixels.
        total_x: Total width of the mosaic in pixels.
        chunk_size: Size of each chunk in pixels (square).

    Returns:
        List of ChunkSpec objects covering the entire mosaic.
        Edge chunks may be smaller than chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if total_y < 0 or total_x < 0:
        raise ValueError(f"total_y and total_x must be >= 0, got total_y={total_y}, total_x={total_x}")
    n_rows = math.ceil(total_y / chunk_size)
    n_cols = math.ceil(total_x / chunk_size)

    chunks = []
    for row in range(n_rows):
        y_start = row * chunk_size
        y_stop = min(y_start + chunk_size, total_y)
        for col in range(n_cols):
            x_start = col * chunk_size
            x_stop = min(x_start + chunk_size, total_x)
            chunks.append(
                ChunkSpec(
                    row=row,
                    col=col,
                    y_start=y_start,
                    y_stop=y_stop,
                    x_start=x_start,
                    x_stop=x_stop,
                )
            )

    return chunks


def enumerate_chunks_from_dataset(
    ds: xr.Dataset,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[ChunkSpec]:
    """Enumerate chunks from an xarray Dataset's spatial dimensions.

    Args:
        ds: Dataset with 'y' and 'x' dimensions.
        chunk_size: Size of each chunk in pixels.

    Returns:
        List of ChunkSpec objects.
    """
    return enumerate_chunks(ds.sizes["northing"], ds.sizes["easting"], chunk_size)


def filter_chunks_by_roi_mask(
    chunks: list[ChunkSpec],
    roi_zarr_path: str,
) -> list[ChunkSpec]:
    """Return only the chunks whose spatial extent intersects the ROI mask.

    The ROI mask is a boolean zarr array of shape (H, W) with pixels set to
    True where inference is wanted. Chunks with no True pixels in their
    (y_start:y_stop, x_start:x_stop) slice are dropped — there is no reason
    to burn a GPU actor on them, and assembly fills missing chunks with
    zero / NaN downstream.

    Args:
        chunks: Full chunk grid from :func:`enumerate_chunks_from_dataset`.
        roi_zarr_path: S3 URI or local path to the ROI boolean zarr.

    Returns:
        Subset of *chunks* that contain at least one ROI pixel.
    """
    mask = zarr.open(roi_zarr_path, mode="r")
    live: list[ChunkSpec] = []
    for chunk in chunks:
        if mask[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop].any():  # type: ignore[index]
            live.append(chunk)
    logger.info("ROI filter: %d/%d chunks intersect the ROI mask", len(live), len(chunks))
    return live
