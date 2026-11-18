"""Lay ADR 043's recorded embeddings out as ``embed_lattice.py`` writes them, so ``calibrate.py`` can re-score them.

Re-scoring the record's own embeddings in a new environment separates what the environment changes
(library versions, BLAS) from what a new forward pass changes. The record ran whole batches, so its
``bf16`` is also written as ``bf16_whole``: the ``chunking`` comparison is then the identity, a control.

Record files (``s3://arbol-tessera-inputs-dev/perf/fp8-mixed/``): ``mixed_g6e_<tile>.npz`` (L40S: bf16 and both
FP8 sizes), ``mixed_g6_<tile>.npz`` (L4), and the A10G's ``mixed_g5_<tile>.npz`` on the two original tiles or
``mixed_g5emu_<tile>.npz`` on the four added later (its bf16 arrays are the plain bf16 forward).

usage: stage_record_embeddings.py <record_dir> <out_dir> <tile> [<tile> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def a10g_file(record: Path, tile: str) -> Path:
    """The A10G's record file for a tile: plain g5 where it exists, else the emulation run's (same bf16 forward)."""
    plain = record / f"mixed_g5_{tile}.npz"
    return plain if plain.exists() else record / f"mixed_g5emu_{tile}.npz"


def stage(record: Path, out: Path, tile: str) -> None:
    """Write lattice_{l40s,a10g,l4}_<tile>.npz from the record's files."""
    l40s = np.load(record / f"mixed_g6e_{tile}.npz")
    coords = {"Y": l40s["Y"], "X": l40s["X"]}
    arrays = {f"{v}_{k}": l40s[f"{v}_{k}"] for v in ("bf16", "fp8_all4", "fp8_stage1") for k in ("q", "s")}
    arrays |= {"bf16_whole_q": l40s["bf16_q"], "bf16_whole_s": l40s["bf16_s"]}
    np.savez(out / f"lattice_l40s_{tile}.npz", **coords, **arrays)
    for card, path in (("a10g", a10g_file(record, tile)), ("l4", record / f"mixed_g6_{tile}.npz")):
        f = np.load(path)
        assert (f["Y"] == coords["Y"]).all() and (f["X"] == coords["X"]).all(), path
        np.savez(out / f"lattice_{card}_{tile}.npz", **coords, bf16_q=f["bf16_q"], bf16_s=f["bf16_s"])


def main(argv: list[str]) -> int:
    """Entry point."""
    record, out, *tiles = argv
    Path(out).mkdir(parents=True, exist_ok=True)
    for tile in tiles:
        stage(Path(record), Path(out), tile)
        print(f"staged {tile}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
