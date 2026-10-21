"""Chunk and shard geometry shared by every store writer.

Its own module so the mosaic writer can use it without importing the output store's layout, which
would put every layout change inside the ingest code identity.
"""

from __future__ import annotations


def clamp_chunks_and_shards(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """Clamp nominal chunk/shard sizes to an array's ``shape``.

    Chunks are clamped to the shape; shards are clamped to the shape and then floored to a whole
    multiple of the clamped chunks (Zarr v3 requires shards to be exact chunk multiples), never
    below one chunk, so a store smaller than one nominal chunk/shard still creates cleanly. The
    single implementation of this geometry math: both :meth:`ArrayLayout.create_kwargs` and the
    empty-store seeder call it.
    """
    # max(..., 1): a zero-extent axis is legal (a mosaic seeded with an EMPTY time axis, appended per date), but a
    # zero chunk edge is not — keep the nominal chunk so the axis grows into it.
    clamped = tuple(max(min(c, s), 1) for c, s in zip(chunks, shape, strict=True))
    if shards is None:
        return clamped, None
    clamped_shards = tuple(max(c, (min(sh, s) // c) * c) for sh, s, c in zip(shards, shape, clamped, strict=True))
    return clamped, clamped_shards
