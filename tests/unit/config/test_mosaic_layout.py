"""The mosaic shard geometry: the ingest write unit becomes the shard when the inner chunk tiles it."""

from __future__ import annotations

import pytest

from tessera_embeddings.config.ingest import INGEST_CHUNKS, mosaic_chunk_layout


def test_a_full_ingest_block_is_one_shard_of_inference_tile_rows() -> None:
    block = (INGEST_CHUNKS["time"], INGEST_CHUNKS["northing"], INGEST_CHUNKS["easting"])
    assert mosaic_chunk_layout(block) == ((1, 256, 2048), block)


@pytest.mark.parametrize("write_unit", [(1, 1000, 4096), (1, 4096, 3000), (1, 256, 2048)])
def test_a_block_the_inner_chunk_does_not_tile_stays_unsharded(write_unit: tuple[int, int, int]) -> None:
    assert mosaic_chunk_layout(write_unit) == (write_unit, None)
