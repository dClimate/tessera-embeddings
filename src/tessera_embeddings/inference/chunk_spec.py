"""Spatial chunk grid specification, enumeration, and ROI filtering.

Derives a grid of processing chunks from Zarr store metadata and filters it
against the ROI mask so only chunks with real coverage reach GPU actors.
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import fsspec
import numpy as np
import xarray as xr
import zarr

from tessera_embeddings.config.inference import INFERENCE_CHUNK_SIZE

logger = logging.getLogger(__name__)

# Probing the ROI mask is I/O-latency bound (one S3 read + decompress per
# chunk), so oversubscribe relative to CPU count — the reads release the GIL.
_ROI_PROBE_WORKERS = min(32, (os.cpu_count() or 4) * 4)


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
        pixels: URI of a ``(height, width)`` boolean mask restricting inference to the pixels it
            marks (:func:`write_pixel_mask`); ``None`` infers every valid pixel. Set by a targeted
            update in ``holes`` mode, carried on the tile so the serial load, the cross-chunk
            prefetch and the strip loop all see the same mask.
    """

    row: int
    col: int
    y_start: int
    y_stop: int
    x_start: int
    x_stop: int
    pixels: str | None = None

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
        return chunk_label(self.row, self.col)


def chunk_label(row: int, col: int) -> str:
    """The staged-artifact label for a grid position (single owner of the format)."""
    return f"chunk_{row}_{col}"


def parse_chunk_label(label: str) -> tuple[int, int]:
    """Parse a :func:`chunk_label` back into ``(row, col)``; raises on anything else."""
    parts = label.split("_")
    if len(parts) != 3 or parts[0] != "chunk":
        raise ValueError(f"Label {label!r} is not of the form 'chunk_<row>_<col>'")
    return int(parts[1]), int(parts[2])


def write_pixel_mask(uri: str, mask: np.ndarray) -> None:
    """Write a tile's boolean pixel mask where :attr:`ChunkSpec.pixels` can point at it.

    Compressed ``.npz``: a hole mask is spatially coherent, so deflate shrinks the 4 MiB of a
    2048-px tile to kilobytes. Credentials come from fsspec's ambient chain, the one the actors and
    the flow runner both read the inputs bucket with.
    """
    fs, path = fsspec.url_to_fs(uri)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "wb") as f:
        np.savez_compressed(f, mask=np.asarray(mask, dtype=bool))


def read_pixel_mask(uri: str) -> np.ndarray:
    """Read a mask written by :func:`write_pixel_mask`."""
    with fsspec.open(uri, "rb") as f, np.load(f) as npz:
        return np.asarray(npz["mask"], dtype=bool)


def enumerate_chunks(
    total_y: int,
    total_x: int,
    chunk_size: int = INFERENCE_CHUNK_SIZE,
) -> list[ChunkSpec]:
    """Enumerate all spatial chunks for a mosaic of given dimensions.

    Args:
        total_y: Total height of the mosaic in pixels.
        total_x: Total width of the mosaic in pixels.
        chunk_size: Size of each chunk in pixels (square).

    Returns:
        ChunkSpecs covering the whole mosaic; edge chunks may be smaller than *chunk_size*.
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
    chunk_size: int = INFERENCE_CHUNK_SIZE,
) -> list[ChunkSpec]:
    """Enumerate chunks from a Dataset's ``northing``/``easting`` sizes."""
    return enumerate_chunks(ds.sizes["northing"], ds.sizes["easting"], chunk_size)


def filter_chunks_by_roi_mask(
    chunks: list[ChunkSpec],
    roi_zarr_path: str,
    *,
    storage_options: dict | None = None,
) -> list[ChunkSpec]:
    """Return only the chunks whose spatial extent intersects the ROI mask.

    The ROI mask is a boolean zarr of shape (H, W), True where inference is wanted. A chunk
    with no True pixel in its slice is dropped — no reason to burn a GPU actor on it, and
    assembly fills missing chunks with zero / NaN downstream.

    Args:
        chunks: Full chunk grid from :func:`enumerate_chunks_from_dataset`.
        roi_zarr_path: S3 URI or local path to the ROI boolean zarr.
        storage_options: fsspec options for the open — the credential and region a
            deployment needs to read its own ROI. The mask is a PLAIN zarr, not an Icechunk
            store, so it does not travel on the Icechunk callback its callers thread
            everywhere else; without this it opens on the ambient chain, which in a
            callback-only or non-default-region deployment means the wrong credentials or
            none at all, on the one read that decides which chunks exist.

    Returns:
        Subset of *chunks* that contain at least one ROI pixel.
    """
    mask = zarr.open(roi_zarr_path, mode="r", storage_options=storage_options)

    def intersects(chunk: ChunkSpec) -> bool:
        return bool(mask[chunk.y_start : chunk.y_stop, chunk.x_start : chunk.x_stop].any())  # type: ignore[index]

    # Fanned out so wall time scales with the slowest read rather than their sum. `pool.map`
    # preserves order, so the live subset keeps the input's row-major chunk ordering.
    with ThreadPoolExecutor(max_workers=_ROI_PROBE_WORKERS) as pool:
        hits = pool.map(intersects, chunks)
    live = [chunk for chunk, hit in zip(chunks, hits, strict=True) if hit]

    logger.info("ROI filter: %d/%d chunks intersect the ROI mask", len(live), len(chunks))
    return live
