"""How deep the delivered embeddings actually are, read from the published store's own counts.

Answers one question: does the **173 combined tokens per pixel** that
``context_docs/campaign/campaign-cost-model.md`` §6c costs the campaign on survive contact with
what was published? Every token figure in that document divides by it, and until this script ran
the only check on it was circular -- the "measured" total was the same 173 multiplied by the
delivered tile count, so it could only ever reproduce the completion rate (§12).

    python scripts/scoping/census_published_token_depth.py --shards 2000 --census-cache /tmp/c.npz

READ THIS BEFORE COMPARING ANY NUMBER HERE TO 173. There are THREE conventions of "observations
per pixel" in this programme, they differ by construction, and §6b is explicit that pairing two of
them is what put the cost line 19% wrong once already:

1. **Census depth** -- distinct catalogue acquisition dates x clear fraction. Withdrawn as a depth
   basis (§6); survives for composition only.
2. **Chunk sequence length** -- ``t_kept + t_s1_asc + t_s1_desc`` from ``CHUNK_SUMMARY``, where
   ``t_kept = mask_bundle.mask.shape[0]``: the number of timesteps the CHUNK retained, counting a
   date if ANY of the chunk's 4.19 M pixels kept it. **This is what 173 is**, and the rate it is
   paired with (2.127 M tok/s) has the same quantity in its numerator, which is why the pairing is
   self-consistent even though the level is an upper bound.
3. **Per-pixel depth** -- each pixel's own count, which is what the store's ``*_obs_count`` arrays
   hold, and what this script measures.

**So this script cannot confirm 173's LEVEL, and does not claim to.** Convention 2 is a union of
date sets over a whole chunk and is not recoverable from per-pixel counts. What it can do, and
what the output is organised around:

* **Size the convention gap** that §6b asserts and never quantified, by measuring convention 3
  exactly alongside the tokens the encoder actually processed (below).
* **Test the GEOGRAPHY**, which is convention-robust because the same convention applies to every
  latitude band. §6c's last revision moved depth 167 -> 173 purely by re-weighting radar from
  "the chunks we happened to measure" onto campaign land, on the strength of radar being deepest
  at 30-35 degrees and an INTERPOLATED 35-50 degree band holding 19.2% of campaign land. That
  interpolation was recorded as the widest single driver of the whole cost interval, and measuring
  it was decided against as too expensive. The store now answers it for the price of this script.
* **Test the OTHER factor in §12's arithmetic**, which nobody has: §6b's identity is
  ``depth x valid_px``, but §12 multiplies 173 by every pixel of every delivered tile. The gap
  between those is the embedded fraction of a live tile, reported here as ``written``.

WHAT THE ENCODER ACTUALLY PROCESSED, which is a fourth number and an exact one. A pixel is not
run at its own depth: ``sampling.compute_bin_keys`` maps it to the next entry of
``DEFAULT_NUM_OBS_CHECKPOINTS`` (8, 16, ... 256) separately for optical and for asc+desc radar,
clipped at both ends -- so a radar-free pixel still costs 8 radar tokens and a 300-observation
pixel is truncated to 256. ``processed`` applies that ladder per pixel. It is the only figure here
that is a count of tokens a graphics card consumed rather than a count of observations.

FOUR THINGS THAT WOULD MAKE THE ANSWER WRONG, each guarded:

* ``*_obs_count`` is ``uint16`` with fill 0, so a zero is "no observations" OR "never written" and
  the array cannot tell you which. ``scales`` is float32 with a NaN fill and is written only where
  a pixel was embedded, so it IS the written mask. Every statistic here is taken over finite
  ``scales`` and never over a non-zero count.
* On these sharded arrays ``Session.chunk_coordinates`` returns SHARD-grid coordinates, not
  inner-chunk ones. That is what makes the coverage census cheap -- manifests only, no pixels.
* Every live shard covers the same pixel count except at a zone's south and east margins, so
  shard-uniform sampling is nearly area-uniform already. Nothing rests on that: each sampled block
  is weighted by its own written-pixel count, which makes the reported means pixel-weighted --
  the same weighting §6c means by "land-weighted" -- by construction rather than by assumption.
* ``embeddings`` is 32x the bytes of everything read here and answers nothing. It is never opened.

The store is public; no credentials are used or needed.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import itertools
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import zarr

STORE_BUCKET = "tessera-embeddings"
STORE_PREFIX = "v1.1/dclimate.icechunk"
REGION = "us-west-2"

#: Shard side in pixels, from the published layout.
SHARD_PX = 2048
#: Inner chunk side. A sampled block must be a whole number of these or the read fetches more than
#: it uses.
CHUNK_PX = 256

#: The sampler's ladder, from ``config.inference.DEFAULT_NUM_OBS_CHECKPOINTS``. Duplicated as a
#: literal rather than imported so this script can be run against the public store from a checkout
#: that does not have the package installed; asserted against the package when it is importable.
CHECKPOINTS = tuple(range(8, 257, 8))

#: Metres per degree of latitude, for banding northings. A spherical constant is ample: bands are
#: 5 degrees wide and the worst-case error is under a tenth of one.
M_PER_DEG = 111_320.0

#: Latitude band edges for the geography test, chosen to straddle §6c's claims: the 30-35 band it
#: calls the deepest radar anywhere, the 35-50 band it INTERPOLATED, and the >70 bands it held flat.
BAND_EDGES = (0, 10, 20, 30, 35, 40, 45, 50, 60, 70, 90)


def open_root() -> tuple[Any, Any]:
    """(session, root group) on the public store's ``main`` tip, with manifest preload disabled.

    The preload settings cut about 2.5 s off every open by refusing to fetch chunk manifests the
    caller has not asked for. This does it by hand because the ``preload_manifests`` keyword
    postdates the icechunk pin here.
    """
    storage = icechunk.s3_storage(bucket=STORE_BUCKET, prefix=STORE_PREFIX, region=REGION, anonymous=True)
    config = icechunk.Repository.fetch_config(storage)
    if config is None:
        raise SystemExit(f"no icechunk config at s3://{STORE_BUCKET}/{STORE_PREFIX}")
    config.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0),
        splitting=config.manifest.splitting,
    )
    repo = icechunk.Repository.open(storage, config=config)
    session = repo.readonly_session(branch="main")
    return session, zarr.open_group(session.store, mode="r")


def live_shards(session: Any, zone: str) -> np.ndarray:  # noqa: ANN401 — icechunk Session, untyped
    """``(n, 3)`` int32 of ``(year_index, northing_shard, easting_shard)`` for one zone.

    Asked of ``scales`` and not of ``embeddings`` or an obs-count array. All three carry the same
    shard grid, but only ``scales`` has a fill value that cannot be confused with written data, and
    an obs-count array answers a different question: a shard can hold observation counts and no
    embeddings at all, which is what a tile that was imaged and then wholly refused looks like.
    """

    async def drain() -> list[tuple[int, ...]]:
        return [c async for c in session.chunk_coordinates(f"/{zone}/scales")]

    coords = asyncio.run(drain())
    if not coords:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(coords, dtype=np.int32)


def census(session: Any, root: Any, cache: Path | None) -> dict[str, np.ndarray]:  # noqa: ANN401
    """Live shards for every zone, from cache if it is there and from the store if not."""
    if cache and cache.exists():
        with np.load(cache) as fh:
            out = {k: fh[k] for k in fh.files}
        print(f"census: {len(out)} zones from {cache}")
        return out
    zones = sorted(root.group_keys())
    out, t0 = {}, time.monotonic()
    for i, zone in enumerate(zones, 1):
        out[zone] = live_shards(session, zone)
        if i % 20 == 0 or i == len(zones):
            print(f"  census {i}/{len(zones)} zones, {time.monotonic() - t0:.0f}s", file=sys.stderr)
    if cache:
        np.savez_compressed(cache, **out)
        print(f"census cached to {cache}")
    return out


def bucketed(counts: np.ndarray) -> np.ndarray:
    """Per-pixel sequence length the sampler would actually run, from its own checkpoint ladder.

    ``compute_bin_keys`` clips to at least 1 before searching, so a zero-observation stream still
    lands in the smallest bucket and still costs its tokens, and clips at the top, so anything
    past the deepest checkpoint is truncated rather than extrapolated. Both clips are load-bearing:
    the first is most of what radar-free pixels cost, the second is the only place the encoder's
    work stops rising with depth.
    """
    ladder = np.asarray(CHECKPOINTS, dtype=np.int32)
    idx = np.searchsorted(ladder, np.clip(counts, 1, None), side="left")
    return ladder[np.clip(idx, 0, len(ladder) - 1)]


def read_block(root: Any, zone: str, coord: tuple[int, int, int], block_px: int, rng: random.Random) -> dict | None:  # noqa: ANN401
    """Statistics for one random ``block_px``-square block inside one live shard.

    Returns ``None`` for a block with no written pixels, which is a real outcome rather than an
    error: a coastal shard is live because part of it is land.
    """
    year, ni, ei = coord
    grp = root[zone]
    scales, s2a = grp["scales"], grp["s2_obs_count"]
    asca, desca = grp["s1_asc_obs_count"], grp["s1_desc_obs_count"]
    # A block offset must land on the inner-chunk grid, or the read pulls chunks it will not use.
    steps = max(1, SHARD_PX // block_px)
    dn = rng.randrange(steps) * block_px
    de = rng.randrange(steps) * block_px
    n0, e0 = ni * SHARD_PX + dn, ei * SHARD_PX + de
    # Clamp at the zone's south and east margins, where a shard is partial.
    n1 = min(n0 + block_px, scales.shape[1])
    e1 = min(e0 + block_px, scales.shape[2])
    if n1 <= n0 or e1 <= e0:
        return None

    sc = scales[year, n0:n1, e0:e1]
    written = np.isfinite(sc)
    n_written = int(written.sum())
    moved = sc.nbytes
    if not n_written:
        return {"zone": zone, "year": year, "px": int(sc.size), "written": 0, "bytes": moved}

    s2 = s2a[year, n0:n1, e0:e1]
    asc = asca[year, n0:n1, e0:e1]
    desc = desca[year, n0:n1, e0:e1]
    moved += s2.nbytes + asc.nbytes + desc.nbytes

    s2w = s2[written].astype(np.int32)
    radar = asc[written].astype(np.int32) + desc[written].astype(np.int32)
    processed = bucketed(s2w).astype(np.int64) + bucketed(radar).astype(np.int64)
    return {
        "zone": zone,
        "year": year,
        "px": int(sc.size),
        "written": n_written,
        "bytes": moved,
        # Sums, not means: the caller weights by written pixels, so only totals may be pooled.
        "s2_sum": int(s2w.sum()),
        "radar_sum": int(radar.sum()),
        "processed_sum": int(processed.sum()),
        "radar_free": int((radar == 0).sum()),
        # Per-block maxima. A lower bound on the chunk sequence lengths of convention 2, because a
        # chunk keeps a date if ANY of its pixels did and the deepest pixel is only one of them.
        "s2_max": int(s2w.max()),
        "radar_max": int(radar.max()),
        # Per-block per-pixel mean, kept per block so a spread can be reported rather than only a
        # pooled mean. Weighted nowhere; for percentiles only.
        "combined_mean": float((s2w + radar).mean()),
        "northing_m": 0.0,  # filled in by the caller, which holds the zone's coordinate endpoints
    }


def zone_latitude(root: Any, zone: str) -> tuple[float, float]:  # noqa: ANN401
    """``(northing of index 0, metres per index)`` for one zone, from its coordinate array.

    Two element reads rather than the whole 7.5 MB axis, which is affordable per zone and would not
    be per shard. The grid is regular by construction -- it is a raster -- so two endpoints
    determine every position on it.
    """
    axis = root[zone]["northing"]
    first = float(axis[0])
    last = float(axis[-1])
    return first, (last - first) / max(1, axis.shape[0] - 1)


def to_latitude(zone: str, northing_m: float) -> float:
    """Absolute latitude in degrees, from a UTM northing and the zone's hemisphere.

    Southern zones carry the 10,000,000 m false northing, so the equator is 10 Mm there and 0 in
    the north. Absolute value, because the bands §6c argues over are about distance from the
    equator and not about which side of it.
    """
    metres = northing_m - 10_000_000.0 if zone.endswith("S") else northing_m
    return abs(metres) / M_PER_DEG


def band_of(lat: float) -> str:
    """Which ``BAND_EDGES`` band a latitude falls in, as a label."""
    for lo, hi in itertools.pairwise(BAND_EDGES):
        if lo <= lat < hi:
            return f"{lo}-{hi}"
    return f"{BAND_EDGES[-1]}+"


def pooled(rows: list[dict], key: str) -> float:
    """Written-pixel-weighted mean of a per-block sum. This is the land weighting."""
    written = sum(r["written"] for r in rows)
    return sum(r[key] for r in rows) / written if written else float("nan")


def report(rows: list[dict], label: str, *, indent: str = "") -> None:
    """One line of pooled depth for a subset of the sample."""
    live = [r for r in rows if r["written"]]
    if not live:
        print(f"{indent}{label:<12} — no written pixels in sample")
        return
    written = sum(r["written"] for r in live)
    px = sum(r["px"] for r in rows)
    s2 = pooled(live, "s2_sum")
    radar = pooled(live, "radar_sum")
    proc = pooled(live, "processed_sum")
    free = sum(r["radar_free"] for r in live) / written * 100
    print(
        f"{indent}{label:<12} {len(rows):>6} {written / px * 100:>7.1f}% "
        f"{s2:>8.1f} {radar:>8.1f} {s2 + radar:>9.1f} {proc:>10.1f} {free:>8.1f}%"
    )


def main(argv: list[str] | None = None) -> int:
    """Measure per-pixel depth over a sample of the published store. Returns an exit code."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=2000, help="live shards to sample (default 2000)")
    ap.add_argument(
        "--block-px",
        type=int,
        default=512,
        help=f"square block read per shard, a multiple of {CHUNK_PX} (default 512)",
    )
    ap.add_argument("--threads", type=int, default=12, help="concurrent block reads (default 12)")
    ap.add_argument("--seed", type=int, default=20260911, help="sampling seed, so a run repeats")
    ap.add_argument("--census-cache", type=Path, help="npz of per-zone live shards, read or written")
    ap.add_argument("--json", dest="json_path", help="write per-block rows and totals here")
    args = ap.parse_args(argv)

    if args.block_px % CHUNK_PX or not 0 < args.block_px <= SHARD_PX:
        return int(bool(print(f"--block-px must be a multiple of {CHUNK_PX}, at most {SHARD_PX}")))

    # The ladder is duplicated above so this script runs without the package; check it when it does.
    try:
        from tessera_embeddings.config.inference import DEFAULT_NUM_OBS_CHECKPOINTS
    except ImportError:
        print("package not importable — CHECKPOINTS not cross-checked", file=sys.stderr)
    else:
        if tuple(DEFAULT_NUM_OBS_CHECKPOINTS) != CHECKPOINTS:
            print("REFUSING: the sampler's checkpoint ladder has moved from this script's copy", file=sys.stderr)
            return 1

    session, root = open_root()
    shards = census(session, root, args.census_cache)

    zones = sorted(k for k, v in shards.items() if len(v))
    total = sum(len(shards[z]) for z in zones)
    per_year: collections.Counter = collections.Counter()
    for z in zones:
        per_year.update(int(y) for y in shards[z][:, 0])
    print(f"\nlive shards in the published store: {total:,} across {len(zones)} zones")
    print("  by year: " + "  ".join(f"{2017 + y}:{n:,}" for y, n in sorted(per_year.items())))

    # Uniform over live shards, then weighted by written pixels at aggregation time. The zone is
    # drawn with probability proportional to its footprint, so no zone needs its own quota.
    rng = random.Random(args.seed)
    flat = [(z, i) for z in zones for i in range(len(shards[z]))]
    picks = rng.sample(flat, min(args.shards, len(flat)))
    geometry = {z: zone_latitude(root, z) for z in sorted({z for z, _ in picks})}

    def one(pick: tuple[str, int]) -> dict | None:
        zone, i = pick
        coord = tuple(int(v) for v in shards[zone][i])
        row = read_block(root, zone, coord, args.block_px, random.Random(rng.random()))
        if row is not None:
            first, step = geometry[zone]
            row["northing_m"] = first + step * (coord[1] * SHARD_PX)
        return row

    print(f"\nreading {len(picks):,} blocks of {args.block_px}x{args.block_px} on {args.threads} threads...")
    t0 = time.monotonic()
    with ThreadPoolExecutor(args.threads) as ex:
        rows = [r for r in ex.map(one, picks) if r is not None]
    elapsed = time.monotonic() - t0
    moved = sum(r["bytes"] for r in rows)
    sampled_px = sum(r["px"] for r in rows)
    written_px = sum(r["written"] for r in rows)
    print(f"read {len(rows):,} blocks in {elapsed:.0f}s — {moved / 1e9:.2f} GB moved, {moved / 1e6 / elapsed:.0f} MB/s")
    print(
        f"sampled {sampled_px:,} pixels, {written_px:,} written "
        f"({written_px / sampled_px * 100:.1f}%) — {sampled_px / 1.3626e13 * 100:.4f}% of the roster"
    )
    if not rows:
        print("no blocks read", file=sys.stderr)
        return 1

    print(
        f"\n{'group':<12} {'blocks':>6} {'written':>8} {'S2/px':>8} {'S1/px':>8}"
        f" {'comb/px':>9} {'processed':>10} {'S1-free':>9}"
    )
    print("  per-pixel observation counts over WRITTEN pixels, pixel-weighted. 'processed' is the")
    print("  sampler's own checkpoint ladder applied per pixel — the tokens a card actually ran.")
    report(rows, "ALL")

    print("\nby year — 2017 is the thin one, and the model carries a single campaign-wide depth:")
    by_year: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_year[r["year"]].append(r)
    for y in sorted(by_year):
        report(by_year[y], str(2017 + y), indent="  ")

    print("\nby absolute latitude band — the test §6c declined to run:")
    by_band: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_band[band_of(to_latitude(r["zone"], r["northing_m"]))].append(r)
    for band in sorted(by_band, key=lambda b: float(b.split("-")[0].rstrip("+"))):
        report(by_band[band], band, indent="  ")

    live = [r for r in rows if r["written"]]
    means = np.array([r["combined_mean"] for r in live])
    weights = np.array([r["written"] for r in live], dtype=float)
    print("\nspread ACROSS blocks of combined observations per pixel (unweighted percentiles):")
    pcts = [5, 25, 50, 75, 95]
    print("  " + "  ".join(f"p{p}={np.percentile(means, p):.0f}" for p in pcts))
    print(f"  min={means.min():.0f}  max={means.max():.0f}  mean={means.mean():.1f}  sd={means.std():.1f}")
    # Standard error of the pixel-weighted mean, from the between-block spread. Blocks are drawn
    # independently, so this is the honest precision of the headline and the reason to quote one.
    eff_n = weights.sum() ** 2 / (weights**2).sum()
    sem = means.std(ddof=1) / math.sqrt(eff_n)
    combined = pooled(live, "s2_sum") + pooled(live, "radar_sum")
    print(f"  headline {combined:.1f} +/- {sem:.1f} (1 s.e., effective n={eff_n:.0f} blocks)")

    print("\nLOWER BOUNDS on the chunk sequence lengths that 173 is made of (convention 2):")
    print("  A chunk keeps a date if ANY of its pixels did, so the deepest pixel in a block is a")
    print("  floor and not an estimate. Reported so the direction of the convention gap is visible.")
    s2_floor = float(np.average([r["s2_max"] for r in live], weights=weights))
    s1_floor = float(np.average([r["radar_max"] for r in live], weights=weights))
    print(f"  block-max S2 {s2_floor:.1f}   block-max S1 {s1_floor:.1f}   sum {s2_floor + s1_floor:.1f}")

    if args.json_path:
        with Path(args.json_path).open("w") as fh:
            json.dump({"rows": rows, "moved_bytes": moved, "seed": args.seed}, fh)
        print(f"\nwrote {args.json_path}")
    print("\nRead the module docstring before comparing any of this to 173. It is a different")
    print("convention, and the gap is arithmetic rather than disagreement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
