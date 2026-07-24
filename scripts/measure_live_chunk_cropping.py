"""Measure how much ingest compute cropping to live tiles would save, per zone.

Ingest mosaics a zone's whole grid regardless of land content, so its cost scales
with EXTENT, not land. This sizes the win before the fix is built, and picks
between cropping strategies — the answer depends on how a zone's live tiles are
shaped, which differs by zone and cannot be guessed:

* ``bbox``  — one window enclosing every live chunk. Cheapest to build; buys
  nothing when a zone's land is scattered (a few islands across an ocean zone).
* ``rows``  — one window per chunk-row, spanning that row's live columns. Handles
  north-south-scattered land; still one load per live row.
* ``exact`` — the live chunks themselves. The floor: what a full rectangle
  decomposition would approach.

Reads the frozen coverage repo only (ADR-010 bitmaps, a few KB per zone) — no
cluster, no mosaic, no writes.

Usage::

    python scripts/measure_live_chunk_cropping.py \
        --land-mask-path s3://global-tessera-inputs-dev/masks/global.icechunk
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from tessera_embeddings.config.ingest import INGEST_CHUNK_SIZE
from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage.zarr_store import open_store_as_zarr_group
from tessera_embeddings.storage.zone_grid import ZONES

# One ingest chunk spans this many live-tile cells per axis (4096 / 2048 = 2).
TILES_PER_CHUNK = INGEST_CHUNK_SIZE // SHARD_PX


def chunk_live_grid(tile_live: np.ndarray, height: int, width: int) -> np.ndarray:
    """Coarsen a 2048-px live-tile bitmap onto the 4096-px ingest chunk grid.

    A chunk is live if ANY constituent tile is live. Padded to the chunk grid's
    shape first, so a zone whose tile grid is odd-sized keeps its last chunk.
    """
    rows, cols = math.ceil(height / INGEST_CHUNK_SIZE), math.ceil(width / INGEST_CHUNK_SIZE)
    padded = np.zeros((rows * TILES_PER_CHUNK, cols * TILES_PER_CHUNK), dtype=bool)
    padded[: tile_live.shape[0], : tile_live.shape[1]] = tile_live
    return padded.reshape(rows, TILES_PER_CHUNK, cols, TILES_PER_CHUNK).any(axis=(1, 3))


def strategy_costs(live: np.ndarray) -> dict[str, int]:
    """Chunks computed per band-date under each strategy (``full`` = today)."""
    rows = np.flatnonzero(live.any(axis=1))
    cols = np.flatnonzero(live.any(axis=0))
    if rows.size == 0:
        return {"full": live.size, "bbox": 0, "rows": 0, "exact": 0, "windows_rows": 0}
    bbox = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
    spans = [np.flatnonzero(live[r]) for r in rows]
    return {
        "full": int(live.size),
        "bbox": int(bbox),
        "rows": int(sum(s[-1] - s[0] + 1 for s in spans)),
        "exact": int(live.sum()),
        "windows_rows": int(rows.size),
    }


def measure(land_mask_path: str, zones: list[str], *, s3_region: str | None) -> list[dict]:
    """Per-zone chunk costs. Zones absent from the coverage repo are skipped."""
    out: list[dict] = []
    for name in zones:
        spec = ZONES[name]
        try:
            cov = open_store_as_zarr_group(land_mask_path, group=name, region=s3_region)
        except (FileNotFoundError, KeyError):
            print(f"  {name}: absent from coverage repo — skipped", file=sys.stderr)
            continue
        tile_live = np.asarray(cov["tile_live_2048"], dtype=bool)
        live = chunk_live_grid(tile_live, spec.height, spec.width)
        out.append({"zone": name, "live_tiles": int(tile_live.sum()), **strategy_costs(live)})
    return out


def _report(rows: list[dict]) -> str:
    """Campaign totals + the extremes that decide the strategy."""
    land = [r for r in rows if r["live_tiles"]]
    tot = {k: sum(r[k] for r in land) for k in ("full", "bbox", "rows", "exact")}
    lines = [
        f"{len(land)} land zones of {len(rows)} measured. Chunks computed per band-date, campaign-wide:",
        "",
        f"  today (full extent) {tot['full']:>12,}",
    ]
    for k in ("bbox", "rows", "exact"):
        lines.append(f"  {k:<19} {tot[k]:>12,}   {tot['full'] / max(tot[k], 1):>6.1f}x less work")
    lines += ["", "Worst zones for bbox (scattered land — where a single window buys least):"]
    for r in sorted(land, key=lambda r: -(r["bbox"] / max(r["exact"], 1)))[:5]:
        lines.append(
            f"  {r['zone']}  full {r['full']:>6,}  bbox {r['bbox']:>6,}  rows {r['rows']:>6,}  "
            f"exact {r['exact']:>5,}  ({r['live_tiles']} live tiles)"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Measure cropping strategies against the frozen coverage repo."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--land-mask-path", required=True, help="coverage Icechunk repo (BucketPaths.land_mask_store)")
    parser.add_argument("--zone", action="append", help="limit to this zone (repeatable; default all)")
    parser.add_argument("--s3-region", default=None)
    parser.add_argument("--json-out", help="write the per-zone rows here")
    args = parser.parse_args(argv)

    rows = measure(args.land_mask_path, args.zone or sorted(ZONES), s3_region=args.s3_region)
    if not rows:
        print("No zones measured — is --land-mask-path right?", file=sys.stderr)
        return 1
    print(_report(rows))
    if args.json_out:
        with Path(args.json_out).open("w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nper-zone rows → {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
