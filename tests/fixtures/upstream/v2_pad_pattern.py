"""Verbatim copy of upstream v2's padding pattern, as the parity reference.

Source: ``ucam-eo/tessera``, ``tessera_infer_v2/student/infer.py`` (``_pad_pattern``), fetched
2026-08-21. Copied rather than paraphrased ON PURPOSE, and it must stay copied: a
reimplementation "in our style" would drift toward our own v1.1 habits, which is precisely the
divergence this reference exists to detect. If it looks un-idiomatic, that is the point — do
not tidy it. See ``tests/unit/test_sampling_v2_parity.py``.

Two details survive verbatim because they change the answer: ``np.linspace(..., dtype=np.int64)``
TRUNCATES rather than rounds, and the ``remain <= n`` branch takes group medians while the other
cycles with modulo.
"""

from __future__ import annotations

import numpy as np


def pad_pattern(n: int, B: int) -> np.ndarray:
    """(B,) int64 indices into [0, n) reproducing the training pad_to_bin."""
    if n == 0:
        return np.zeros(B, dtype=np.int64)
    if n >= B:
        return np.linspace(0, n - 1, B, dtype=np.int64)
    remain = B - n
    if remain <= n:
        groups = np.array_split(np.arange(n), remain)
        fill = np.array([gp[len(gp) // 2] for gp in groups], dtype=np.int64)
    else:
        fill = (np.arange(remain) % n).astype(np.int64)
    return np.concatenate([np.arange(n, dtype=np.int64), fill])
