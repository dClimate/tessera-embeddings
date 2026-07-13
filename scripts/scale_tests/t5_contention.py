"""T5 — commit contention across concurrent zone writers (test plan §4 T5).

Feeds the contention arm of the one-repo go/no-go (ADR D5) and commit pacing
(D6): N processes each region-write a distinct group and commit at once
(barrier-synchronized), measuring auto-rebase retries and per-commit wall as N
grows. Kill threshold (D5): at N=16, total wall > 2x serial or retry storms.

Contention counts are only meaningful on ``--backend s3`` (real object-store
CAS); local runs validate the mechanism. Run from ``scripts/``::

    uv run python -m scale_tests.t5_contention --run-id dev --backend local --scale tiny
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from scale_tests import harness
from scale_tests import variants as V
from scale_tests._workers import commit_to_group
from scale_tests.seeding import embedding_group_spec, seed_groups
from scale_tests.zone_geometry import YEARS, MockZone

logger = logging.getLogger("scale_tests.t5")

TEST = "t5"
VARIANT = V.VARIANTS["c256_full"]
# Small zone so seeding/writes are cheap — this test is about commits, not bytes.
ZONE = MockZone(512, 512)
N_SWEEP_TINY = (2, 4, 8)
N_SWEEP_BENCH = (2, 8, 16, 32, 64, 120)
_CY, _CX = VARIANT.chunks[1], VARIANT.chunks[2]
YEAR_IDX = len(YEARS) - 1


def _region(year_idx: int = YEAR_IDX) -> list[list[int]]:
    """One-chunk region (all bands) at the given year index."""
    return [[year_idx, year_idx + 1], [0, _CY], [0, _CX], [0, V.BAND]]


def _seed_n_groups(cfg: harness.RunConfig, store: str, n: int) -> None:
    """Seed ``n`` groups (z0..z{n-1}) with a small single-year array each."""
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    groups = {
        f"z{i}": embedding_group_spec(ZONE, VARIANT, years=(YEARS[YEAR_IDX],), extra_attrs={"zone_index": i})
        for i in range(n)
    }
    seed_groups(repo, groups, commit_msg=f"seed {n} groups")


def _run_contended(cfg: harness.RunConfig, store: str, n: int) -> list[dict[str, Any]]:
    """Run ``n`` committers (one per group) released together on a barrier."""
    store_uri = harness.store_uri(cfg, store)
    payloads = [
        {
            "store_uri": store_uri,
            "group": f"z{i}",
            "array": "embeddings",
            "region": _region(0),  # single-year axis -> index 0
            "seed": 1000 + i,
            "solver": "detector",
            "rebase_tries": 500,
            "jitter_s": 0.05,
        }
        for i in range(n)
    ]
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as mgr:
        barrier = mgr.Barrier(n)
        for p in payloads:
            p["barrier"] = barrier
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as ex:
            return list(ex.map(commit_to_group, payloads))


def phase_for_n(cfg: harness.RunConfig, n: int) -> None:
    """Seed ``n`` groups and run ``n`` synchronized committers; emit contention."""
    store = f"t5_n{n}"
    phase = f"contention_n{n}"  # matches the run_phase name so resume clears cleanly
    _seed_n_groups(cfg, store, n)
    t0 = time.monotonic()
    results = _run_contended(cfg, store, n)
    total_wall = time.monotonic() - t0

    retries = [r["retries"] for r in results]
    commit_walls = [r["commit_wall_s"] for r in results]
    unresolvable = sum(r["unresolvable"] for r in results)
    failed = sum(not r["ok"] for r in results)
    for r in results:
        harness.emit_metric(cfg, TEST, phase, "retries", r["retries"], "count", n=n, group=r["group"])
        harness.emit_metric(cfg, TEST, phase, "commit_wall_s", r["commit_wall_s"], "s", n=n, group=r["group"])
    harness.emit_metric(cfg, TEST, phase, "wall_s", total_wall, "s", n=n, kind="total")
    harness.emit_metric(
        cfg, TEST, phase, "commit_wall_s", float(np.percentile(commit_walls, 95)), "s", n=n, stat="p95"
    )
    logger.info(
        "N=%d: total %.2fs, retries max=%d mean=%.1f, unresolvable=%d failed=%d",
        n,
        total_wall,
        max(retries),
        float(np.mean(retries)),
        unresolvable,
        failed,
    )
    if unresolvable or failed:
        raise SystemExit(
            f"N={n}: distinct-group commits did not all resolve "
            f"(unresolvable={unresolvable}, failed={failed})"
        )


def main() -> int:
    """Parse args and run T5 across the N sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()
    if not cfg.is_s3:
        logger.info("NOTE: retry counts are only meaningful on --backend s3; local validates the mechanism only.")

    sweep = N_SWEEP_TINY if cfg.is_tiny else N_SWEEP_BENCH
    for n in sweep:
        harness.run_phase(cfg, TEST, f"contention_n{n}", lambda k=n: phase_for_n(cfg, k))
    logger.info("T5 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
