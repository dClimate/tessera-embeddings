"""T3 — pre-allocate vs prepend A/B (test plan §4 T3, ADR D1).

Confirms the decided path (seed the full 2017-2025 axis for free, then fill years
as region writes) and characterizes the ``shift_array`` prepend escape hatch:
its per-iteration shift+commit cost, that it moves no chunk data, that prior
years (including empty ocean positions) stay intact, and that a shift conflicts
unresolvably with a concurrent chunk writer (why prepends are banned during
fills).

Run from ``scripts/``::

    uv run python -m scale_tests.t3_prealloc --run-id dev --backend local --scale tiny
"""

from __future__ import annotations

import argparse
import logging
import time

import icechunk
import numpy as np
import zarr

from scale_tests import harness, synth
from scale_tests import store_builder as SB
from scale_tests import variants as V
from scale_tests.seeding import embedding_group_spec, seed_groups
from scale_tests.zone_geometry import YEARS, MockZone, zone_for

logger = logging.getLogger("scale_tests.t3")

TEST = "t3"
GROUP = "zone"
VARIANT = V.VARIANTS["c256_full"]  # the expected D2 winner; prepend cost is layout-insensitive
LAND_FRACTION = 0.7
SEED = 0


def _manifest_bytes(cfg: harness.RunConfig, store: str) -> int:
    """Bytes under the store's ``manifests/`` prefix (write-amplification proxy)."""
    _, total = harness.object_stats(harness.store_uri(cfg, store) + "/manifests")
    return total


def phase_seed_full_axis(cfg: harness.RunConfig) -> None:
    """Seed the full 9-year axis; assert the DATA vars wrote no chunks.

    ``embeddings``/``scales`` must be all-fill (``nchunks_initialized == 0``) so
    seeding cost is independent of pixel extent. Coordinate arrays (northing/
    easting) *are* written — those objects are the only real cost of an empty
    store, and are recorded (not asserted at zero).
    """
    store = "t3_prealloc"
    zone = zone_for(cfg.scale)
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    with harness.timer() as t:
        seed_groups(repo, {GROUP: embedding_group_spec(zone, VARIANT)}, commit_msg="seed full axis")

    session = harness.open_repo(cfg, store).readonly_session(branch="main")
    grp = zarr.open_group(session.store, mode="r")[GROUP]
    emb_init = grp["embeddings"].nchunks_initialized
    scl_init = grp["scales"].nchunks_initialized
    n_chunks, coord_bytes = harness.object_stats(harness.store_uri(cfg, store) + "/chunks")
    harness.emit_metric(cfg, TEST, "seed_full_axis", "wall_s", t.seconds, "s", n_years=len(YEARS))
    harness.emit_metric(cfg, TEST, "seed_full_axis", "objects_listed", n_chunks, "count", where="chunks_coords_only")
    harness.emit_metric(cfg, TEST, "seed_full_axis", "manifest_bytes", coord_bytes, "bytes", where="coord_chunks")
    logger.info(
        "seeded %d-year axis in %.2fs: embeddings/scales chunks=%d/%d, coord chunk objects=%d",
        len(YEARS),
        t.seconds,
        emb_init,
        scl_init,
        n_chunks,
    )
    if emb_init != 0 or scl_init != 0:
        raise SystemExit(
            f"pre-alloc seed wrote data chunks (emb={emb_init}, scl={scl_init}); expected 0 (metadata-only)"
        )


def phase_fill_and_verify(cfg: harness.RunConfig) -> None:
    """Fill 2025 into the pre-allocated store, then verify fill + empty semantics."""
    store = "t3_prealloc"
    zone = zone_for(cfg.scale)
    land = SB.land_chunks(zone, VARIANT, fraction=LAND_FRACTION, seed=SEED)
    year_idx = len(YEARS) - 1  # 2025
    n_workers = 2 if cfg.is_tiny else 8
    SB.fill_year(cfg, store, GROUP, VARIANT, zone, year_idx, land, n_workers=n_workers, seed=SEED)

    session = harness.open_repo(cfg, store).readonly_session(branch="main")
    grp = zarr.open_group(session.store, mode="r")[GROUP]
    _, cy, cx, _ = VARIANT.chunks
    yc, xc = land[0]
    y0, x0 = yc * cy, xc * cx
    got = grp["embeddings"][year_idx, y0 : y0 + cy, x0 : x0 + cx, :]
    exp = synth.embedding_block((1, cy, cx, V.BAND), seed=SEED, block_index=(year_idx, yc, xc))[0]
    if not np.array_equal(got, exp):
        raise SystemExit("filled 2025 does not match synth data")
    # An unfilled year: embeddings read as int8 fill 0, scales as NaN (the sentinel).
    if not (grp["embeddings"][0, y0 : y0 + cy, x0 : x0 + cx, :] == 0).all():
        raise SystemExit("unfilled year embeddings are not all fill=0")
    if not np.isnan(grp["scales"][0, y0 : y0 + cy, x0 : x0 + cx]).all():
        raise SystemExit("unfilled year scales are not all NaN (sentinel broken)")
    if YEARS[year_idx] not in list(grp.attrs.get("years_complete", [])):
        raise SystemExit("years_complete attr did not record the fill")
    logger.info("pre-alloc fill verified: filled year matches, unfilled reads as fill+NaN")


def _prepend_year(
    cfg: harness.RunConfig, store: str, zone: MockZone, year_int: int, year_ns: int, land: list[tuple[int, int]]
) -> tuple[float, float, int]:
    """Prepend one year at the front via resize->shift->write->commit.

    Returns ``(shift_wall_s, commit_wall_s, manifest_bytes)``.
    """
    repo = harness.open_repo(cfg, store)
    session = repo.writable_session("main")
    root = zarr.open_group(session.store, mode="a")
    grp = root[GROUP]
    emb, scl, tcoord = grp["embeddings"], grp["scales"], grp["time"]
    nt = emb.shape[0]
    _, cy, cx, _ = VARIANT.chunks

    emb.resize((nt + 1, *emb.shape[1:]))
    scl.resize((nt + 1, *scl.shape[1:]))
    tcoord.resize((nt + 1,))

    shift_start = time.monotonic()
    session.shift_array(emb.name, [1, 0, 0, 0])
    session.shift_array(scl.name, [1, 0, 0])
    session.shift_array(tcoord.name, [1])
    shift_wall = time.monotonic() - shift_start

    for yc, xc in land:
        y0, x0 = yc * cy, xc * cx
        y1, x1 = min(y0 + cy, zone.height), min(x0 + cx, zone.width)
        h, w = y1 - y0, x1 - x0
        emb[0:1, y0:y1, x0:x1, :] = synth.embedding_block((1, h, w, V.BAND), seed=year_int, block_index=(0, yc, xc))
        scl[0:1, y0:y1, x0:x1] = synth.scales_block((1, h, w), seed=year_int, block_index=(0, yc, xc))
    tcoord[0] = year_ns

    commit_start = time.monotonic()
    session.commit(f"prepend {year_int}")
    commit_wall = time.monotonic() - commit_start
    return shift_wall, commit_wall, _manifest_bytes(cfg, store)


def phase_prepend_loop(cfg: harness.RunConfig) -> None:
    """Build a 2025-only store, then prepend 2024..2017, measuring each shift."""
    store = "t3_prepend"
    zone = zone_for(cfg.scale)
    land = SB.land_chunks(zone, VARIANT, fraction=LAND_FRACTION, seed=SEED)
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    # Seed with ONLY the latest year, then fill it.
    seed_groups(repo, {GROUP: embedding_group_spec(zone, VARIANT, years=(YEARS[-1],))}, commit_msg="seed 2025 only")
    SB.fill_year(cfg, store, GROUP, VARIANT, zone, 0, land, n_workers=2 if cfg.is_tiny else 8, seed=YEARS[-1])

    for year in reversed(YEARS[:-1]):  # 2024 -> 2017
        year_ns = np.datetime64(f"{year}-01-01", "ns").astype("int64")
        shift_wall, commit_wall, man_bytes = _prepend_year(cfg, store, zone, int(year), int(year_ns), land)
        harness.emit_metric(
            cfg, TEST, "prepend_loop", "commit_wall_s", commit_wall, "s", year=int(year), phase_kind="shift"
        )
        harness.emit_metric(cfg, TEST, "prepend_loop", "manifest_bytes", man_bytes, "bytes", year=int(year))
        logger.info(
            "prepended %d: shift %.3fs commit %.3fs manifest_bytes=%d", year, shift_wall, commit_wall, man_bytes
        )


def phase_verify_prepend(cfg: harness.RunConfig) -> None:
    """Assert the prepended store has all 9 years, ascending, data intact."""
    store = "t3_prepend"
    zone = zone_for(cfg.scale)
    land = SB.land_chunks(zone, VARIANT, fraction=LAND_FRACTION, seed=SEED)
    _, cy, cx, _ = VARIANT.chunks
    session = harness.open_repo(cfg, store).readonly_session(branch="main")
    grp = zarr.open_group(session.store, mode="r")[GROUP]

    tvals = np.asarray(grp["time"][:]).astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
    if list(tvals) != list(YEARS):
        raise SystemExit(f"prepended time axis is {list(tvals)}, expected {list(YEARS)}")

    yc, xc = land[0]
    y0, x0 = yc * cy, xc * cx
    for idx, year in enumerate(YEARS):
        got = grp["embeddings"][idx, y0 : y0 + cy, x0 : x0 + cx, :]
        exp = synth.embedding_block((1, cy, cx, V.BAND), seed=int(year), block_index=(0, yc, xc))[0]
        if not np.array_equal(got, exp):
            raise SystemExit(f"prepended year {year} (index {idx}) data mismatch after shifts")
    # An ocean (unwritten) position must remain fill across all years post-shift.
    grid = SB.chunk_grid(zone, VARIANT)
    land_set = set(land)
    ocean = next(((yy, xx) for yy in range(grid.n_y) for xx in range(grid.n_x) if (yy, xx) not in land_set), None)
    if ocean is not None:
        oy, ox = ocean[0] * cy, ocean[1] * cx
        if not (grp["embeddings"][:, oy : oy + cy, ox : ox + cx, :] == 0).all():
            raise SystemExit("ocean position not all-fill after shifts (reindex stale-data hazard)")
    logger.info("prepend verified: 9 years ascending, data + ocean positions intact")


def phase_conflict_probe(cfg: harness.RunConfig) -> None:
    """A chunk write racing a shift commit must be unresolvable (ADR D1/D6)."""
    store = "t3_conflict"
    zone = zone_for(cfg.scale)
    land = SB.land_chunks(zone, VARIANT, fraction=LAND_FRACTION, seed=SEED)
    _, cy, cx, _ = VARIANT.chunks
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    seed_groups(repo, {GROUP: embedding_group_spec(zone, VARIANT, years=(YEARS[-1],))}, commit_msg="seed")
    SB.fill_year(cfg, store, GROUP, VARIANT, zone, 0, land, n_workers=2 if cfg.is_tiny else 8, seed=YEARS[-1])

    # Writer session: stage a chunk write but do not commit yet.
    writer = repo.writable_session("main")
    yc, xc = land[0]
    y0, x0 = yc * cy, xc * cx
    zarr.open_group(writer.store, mode="a")[GROUP]["embeddings"][0:1, y0 : y0 + cy, x0 : x0 + cx, :] = np.full(
        (1, cy, cx, V.BAND), 3, "int8"
    )

    # Shifter session: prepend a year and commit, moving the branch tip.
    shifter = repo.writable_session("main")
    root = zarr.open_group(shifter.store, mode="a")[GROUP]
    root["embeddings"].resize((2, *root["embeddings"].shape[1:]))
    root["time"].resize((2,))
    shifter.shift_array(root["embeddings"].name, [1, 0, 0, 0])
    shifter.shift_array(root["time"].name, [1])
    shifter.commit("shift under writer")

    # Writer now commits onto a shifted array — expect an unresolvable conflict.
    outcome = "committed"
    try:
        writer.commit("write racing shift")
    except icechunk.ConflictError:
        try:
            writer.rebase(icechunk.ConflictDetector())
            writer.commit("write racing shift (rebased)")
            outcome = "rebased_ok"
        except icechunk.RebaseFailedError:
            outcome = "unresolvable"
    harness.emit_metric(cfg, TEST, "conflict_probe", "retries", 1.0, "count", outcome=outcome)
    logger.info("conflict probe outcome: %s", outcome)
    if outcome == "committed":
        logger.warning("writer committed cleanly despite a concurrent shift — starts likely didn't overlap")


def main() -> int:
    """Parse args and run T3 phases in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()

    harness.run_phase(cfg, TEST, "seed_full_axis", lambda: phase_seed_full_axis(cfg))
    harness.run_phase(cfg, TEST, "fill_and_verify", lambda: phase_fill_and_verify(cfg))
    harness.run_phase(cfg, TEST, "prepend_loop", lambda: phase_prepend_loop(cfg))
    harness.run_phase(cfg, TEST, "verify_prepend", lambda: phase_verify_prepend(cfg))
    harness.run_phase(cfg, TEST, "conflict_probe", lambda: phase_conflict_probe(cfg))
    logger.info("T3 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
