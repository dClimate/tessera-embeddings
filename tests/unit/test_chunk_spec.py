"""Tests for chunk_spec module: grid enumeration and edge handling."""

import pytest

from tessera_embeddings.inference.chunk_spec import ChunkSpec, enumerate_chunks


class TestChunkSpec:
    """Tests for ChunkSpec properties and frozen behavior."""

    def test_properties(self):
        chunk = ChunkSpec(row=2, col=3, y_start=3000, y_stop=4500, x_start=4500, x_stop=6000)
        assert chunk.height == 1500
        assert chunk.width == 1500
        assert chunk.label == "chunk_2_3"

    def test_edge_chunk_smaller(self):
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=700, x_start=0, x_stop=300)
        assert chunk.height == 700
        assert chunk.width == 300

    def test_frozen(self):
        chunk = ChunkSpec(row=0, col=0, y_start=0, y_stop=100, x_start=0, x_stop=100)
        with pytest.raises(AttributeError):
            chunk.row = 5


class TestEnumerateChunks:
    """Tests for chunk grid enumeration with various dimensions."""

    def test_exact_fit(self):
        """Grid that divides evenly into chunk_size."""
        chunks = enumerate_chunks(3000, 3000, chunk_size=1500)
        assert len(chunks) == 4  # 2x2 grid
        assert all(c.height == 1500 for c in chunks)
        assert all(c.width == 1500 for c in chunks)

    def test_edge_chunks(self):
        """Grid that doesn't divide evenly — edge chunks are smaller."""
        chunks = enumerate_chunks(3200, 1700, chunk_size=1500)
        # 3200/1500 = ceil(2.13) = 3 rows, 1700/1500 = ceil(1.13) = 2 cols
        assert len(chunks) == 6

        # Check edge chunk dimensions
        bottom_right = next(c for c in chunks if c.row == 2 and c.col == 1)
        assert bottom_right.height == 200  # 3200 - 3000
        assert bottom_right.width == 200  # 1700 - 1500

    def test_single_chunk(self):
        """Mosaic smaller than chunk_size → single chunk."""
        chunks = enumerate_chunks(500, 800, chunk_size=1500)
        assert len(chunks) == 1
        assert chunks[0].height == 500
        assert chunks[0].width == 800

    def test_real_test_data_dimensions(self):
        """Dimensions matching the actual test data (33174 x 25906)."""
        chunks = enumerate_chunks(33174, 25906, chunk_size=1500)
        # ceil(33174/1500) = 23, ceil(25906/1500) = 18
        assert len(chunks) == 23 * 18  # 414 chunks

        # Verify full coverage (no gaps, no overlaps)
        for row_idx in range(23):
            row_chunks = sorted([c for c in chunks if c.row == row_idx], key=lambda c: c.col)
            assert row_chunks[0].x_start == 0
            for i in range(len(row_chunks) - 1):
                assert row_chunks[i].x_stop == row_chunks[i + 1].x_start
            assert row_chunks[-1].x_stop == 25906

    def test_chunk_ordering(self):
        """Chunks are enumerated row-major."""
        chunks = enumerate_chunks(3000, 3000, chunk_size=1500)
        assert chunks[0].row == 0 and chunks[0].col == 0
        assert chunks[1].row == 0 and chunks[1].col == 1
        assert chunks[2].row == 1 and chunks[2].col == 0
        assert chunks[3].row == 1 and chunks[3].col == 1

    @pytest.mark.parametrize(
        "total_y,total_x,chunk_size,expected_count",
        [
            (1500, 1500, 1500, 1),
            (1501, 1500, 1500, 2),
            (1, 1, 1500, 1),
            (10000, 10000, 1500, 49),  # 7x7
        ],
    )
    def test_parametrized_counts(self, total_y, total_x, chunk_size, expected_count):
        chunks = enumerate_chunks(total_y, total_x, chunk_size)
        assert len(chunks) == expected_count
