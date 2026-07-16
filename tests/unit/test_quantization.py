"""Tests for INT8 embedding quantization.

The core guarantee is that the row-wise :func:`quantize_rows` (used to quantize
each bucket as it comes off the GPU) is *bit-for-bit identical* to quantizing the
whole ``(H, W, D)`` chunk at the end. We pin that against a frozen copy of the
original whole-array implementation rather than the production function, which now
delegates to ``quantize_rows`` and would make the comparison circular.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tessera_embeddings.inference.quantization import (
    dequantize_embeddings,
    quantize_embeddings,
    quantize_rows,
    quantize_rows_torch,
    raise_on_nonfinite_scales,
)


def _quantize_whw_reference(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frozen copy of the original (H, W, D) quantize_embeddings body.

    Verbatim from quantization.py as shipped (per-pixel symmetric int8). This is
    the golden reference quantize_rows must match exactly. Do NOT refactor this to
    call the production function — that would defeat the test.
    """
    abs_max = np.abs(embeddings).max(axis=-1)  # (H, W)
    scales = np.maximum(abs_max / 127.0, 1e-8).astype(np.float32)
    quantized = np.clip(np.round(embeddings / scales[..., np.newaxis]), -127, 127).astype(np.int8)
    return quantized, scales


def test_quantize_rows_matches_whole_array_reference() -> None:
    """Row-wise quantization is bit-for-bit identical to whole-array quantization."""
    rng = np.random.default_rng(0)
    h, w, d = 64, 50, 128
    # Vary per-pixel magnitude so scales differ across rows (exercises broadcasting).
    emb = rng.standard_normal((h, w, d)).astype(np.float32)
    emb *= rng.uniform(0.1, 5.0, size=(h, w, 1)).astype(np.float32)

    ref_q, ref_s = _quantize_whw_reference(emb)
    q, s = quantize_rows(emb.reshape(h * w, d))

    np.testing.assert_array_equal(q, ref_q.reshape(h * w, d))
    np.testing.assert_array_equal(s, ref_s.reshape(h * w))


def test_quantize_embeddings_delegates_consistently() -> None:
    """Production (H,W,D) wrapper matches the frozen reference (regression guard)."""
    rng = np.random.default_rng(1)
    emb = (rng.standard_normal((16, 16, 128)) * 3.0).astype(np.float32)

    ref_q, ref_s = _quantize_whw_reference(emb)
    q, s = quantize_embeddings(emb)

    np.testing.assert_array_equal(q, ref_q)
    np.testing.assert_array_equal(s, ref_s)
    assert q.dtype == np.int8
    assert s.dtype == np.float32
    assert q.shape == emb.shape
    assert s.shape == emb.shape[:2]


def test_zero_row_uses_scale_floor() -> None:
    """All-zero row -> q all zero, scale at the 1e-8 floor.

    This is the value never-written/invalid pixels get; run_inference relies on it
    to initialize flat_scales so untouched pixels match whole-array quantization.
    """
    q, s = quantize_rows(np.zeros((1, 128), dtype=np.float32))
    np.testing.assert_array_equal(q, np.zeros((1, 128), dtype=np.int8))
    assert s[0] == np.float32(1e-8)


def test_constant_row_saturates_to_127() -> None:
    """A row of equal positive values quantizes to all 127 (scale = val / 127)."""
    val = 1000.0
    q, s = quantize_rows(np.full((1, 128), val, dtype=np.float32))
    np.testing.assert_array_equal(q, np.full((1, 128), 127, dtype=np.int8))
    assert s[0] == pytest.approx(val / 127.0, rel=1e-6)


def test_clip_bound_is_symmetric_127() -> None:
    """Values stay within [-127, 127]; the negative extreme clips to -127, not -128."""
    # One channel far larger than the rest sets the scale; the negative of that
    # magnitude must land on -127.
    row = np.zeros((1, 128), dtype=np.float32)
    row[0, 0] = 100.0
    row[0, 1] = -100.0
    q, _ = quantize_rows(row)
    assert q.min() == -127
    assert q.max() == 127


def test_round_half_matches_numpy_banker_rounding() -> None:
    """Pin np.round (banker's rounding) vs floor(x+0.5) at the .5 boundary.

    With scale s, an input of 1.5*s should round to 2 under banker's rounding
    (round-half-to-even), distinguishing it from floor(x+0.5)=2 ... so use 2.5*s
    where the two rules diverge: banker's -> 2, floor(x+0.5) -> 3.
    """
    # Force scale = 1.0 via a channel at 127.0, then probe a channel at 2.5.
    row = np.zeros((1, 128), dtype=np.float32)
    row[0, 0] = 127.0  # abs_max=127 -> scale=1.0
    row[0, 1] = 2.5
    q, s = quantize_rows(row)
    assert s[0] == pytest.approx(1.0, rel=1e-6)
    assert q[0, 1] == 2  # banker's rounding rounds 2.5 -> 2 (even)


def test_nonfinite_raises() -> None:
    bad = np.zeros((2, 128), dtype=np.float32)
    bad[1, 3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        quantize_rows(bad)


def test_torch_quantize_matches_numpy_rows() -> None:
    """quantize_rows_torch (on-device path) is bit-identical to numpy quantize_rows.

    The inference loop quantizes on the GPU and scatter-writes the int8 result,
    so the on-device path must produce exactly the same int8 codes and scales as
    the CPU golden path on the same float32 input. Run on CPU tensors so this
    holds without a GPU; the arithmetic is device-independent.
    """
    rng = np.random.default_rng(7)
    # Mix of scales, a zero row (scale floor), and a saturating row.
    rows = (rng.standard_normal((64, 128)) * 3.0).astype(np.float32)
    rows[0] = 0.0
    rows[1] = 5.0
    rows[2, 0] = 127.0

    q_np, s_np = quantize_rows(rows)
    q_t, s_t = quantize_rows_torch(torch.from_numpy(rows))

    np.testing.assert_array_equal(q_t.numpy(), q_np)
    np.testing.assert_array_equal(s_t.numpy(), s_np)


def test_torch_quantize_matches_numpy_from_reduced_precision() -> None:
    """A bf16 input quantizes identically whether cast to f32 on GPU or CPU first.

    The forward emits bf16/fp16; quantize_rows_torch casts to float32 internally,
    matching the CPU path which quantizes the ``.float()`` output. Feeding the
    same bf16 values through both must agree exactly.
    """
    rng = np.random.default_rng(11)
    rows_bf16 = torch.from_numpy((rng.standard_normal((32, 128)) * 2.0).astype(np.float32)).bfloat16()

    q_t, s_t = quantize_rows_torch(rows_bf16)
    q_np, s_np = quantize_rows(rows_bf16.float().numpy())

    np.testing.assert_array_equal(q_t.numpy(), q_np)
    np.testing.assert_array_equal(s_t.numpy(), s_np)


def test_torch_quantize_nonfinite_surfaces_in_scales() -> None:
    """Non-finite embeddings always yield non-finite scales, and the host-side
    scale check rejects them.

    quantize_rows_torch deliberately skips on-device validation (it would force
    a host sync per sub-batch); the contract is that any NaN/Inf row propagates
    to its scale, where raise_on_nonfinite_scales catches it after the D2H.
    """
    for bad_value in (float("inf"), float("nan"), float("-inf")):
        bad = torch.zeros((2, 128))
        bad[1, 3] = bad_value
        _, scales = quantize_rows_torch(bad)
        assert not np.isfinite(scales.numpy()).all()
        with pytest.raises(ValueError, match="non-finite"):
            raise_on_nonfinite_scales(scales.numpy())


def test_raise_on_nonfinite_scales_accepts_finite() -> None:
    raise_on_nonfinite_scales(np.array([1.0, 2.5, 1e-8], dtype=np.float32))


def test_round_trip_recovers_within_scale() -> None:
    """Dequantized values are within one quantization step (scale) of the original."""
    rng = np.random.default_rng(2)
    emb = (rng.standard_normal((8, 8, 128)) * 2.0).astype(np.float32)
    q, s = quantize_embeddings(emb)
    recovered = dequantize_embeddings(q, s)
    # Max error per pixel is at most its own scale (one int8 step).
    err = np.abs(recovered - emb).max(axis=-1)
    assert np.all(err <= s + 1e-6)
