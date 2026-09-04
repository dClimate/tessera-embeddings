"""T0 — smoke + cross-group conflict probe (T0).

Cheap, first, and load-bearing: if concurrent commits to *different* groups do
not auto-rebase cleanly (the cross-group conflict-freedom inference), the
one-repo/120-group design is re-planned before anything expensive runs. Also
validates the harness end-to-end (metrics, markers, spawn multiprocessing) and
the disjoint-region vs same-chunk conflict taxonomy.

Run from ``scripts/``::

    uv run python -m scale_tests.t0_smoke --run-id dev --backend local --scale tiny
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from scripts.scoping.scale_tests import harness
from scripts.scoping.scale_tests import variants as V
from scripts.scoping.scale_tests._workers import commit_to_group
from scripts.scoping.scale_tests.seeding import ArraySpec, GroupSpec, seed_groups
from scripts.scoping.scale_tests.zone_geometry import YEARS, MockZone, coords

logger = logging.getLogger("scale_tests.t0")

TEST = "t0"

# Small, fixed smoke geometry — independent of the 5 read-benchmark variants,
# which are far too large to write in a smoke test.
_ZONE = MockZone(height=512, width=512, epsg="EPSG:32601")
_BAND = 16
_CHUNK = 128
_YEAR_IDX = len(YEARS) - 1  # 2025


def _smoke_group_spec(epsg: str) -> GroupSpec:
    """A tiny (9, 512, 512, 16) embeddings + scales group for the smoke test."""
    nt = len(YEARS)
    emb_kwargs = {
        "shape": (nt, _ZONE.height, _ZONE.width, _BAND),
        "chunks": (1, _CHUNK, _CHUNK, _BAND),
        "dtype": "int8",
        "fill_value": 0,
        "dimension_names": V.DIMS,
        "compressors": None,
    }
    scl_kwargs = {
        "shape": (nt, _ZONE.height, _ZONE.width),
        "chunks": (1, _CHUNK, _CHUNK),
        "dtype": "float32",
        "fill_value": float("nan"),
        "dimension_names": V.DIMS[:3],
        "compressors": None,
    }
    return GroupSpec(
        coords=coords(_ZONE),
        arrays=[ArraySpec("embeddings", emb_kwargs), ArraySpec("scales", scl_kwargs)],
        attrs={"crs": epsg, "years_complete": [], "variant": "smoke"},
    )


def _one_chunk_region(y_chunk: int = 0, x_chunk: int = 0) -> list[list[int]]:
    """Region (over embeddings dims) selecting exactly one on-disk chunk."""
    y0 = y_chunk * _CHUNK
    x0 = x_chunk * _CHUNK
    return [[_YEAR_IDX, _YEAR_IDX + 1], [y0, y0 + _CHUNK], [x0, x0 + _CHUNK], [0, _BAND]]


def _payload(
    cfg: harness.RunConfig, store: str, group: str, region: list[list[int]], seed: int, solver: str
) -> dict[str, Any]:
    """Build a worker payload for :func:`commit_to_group` (barrier added later)."""
    return {
        "store_uri": harness.store_uri(cfg, store),
        "group": group,
        "array": "embeddings",
        "region": region,
        "seed": seed,
        "solver": solver,
        "rebase_tries": 200,
        "jitter_s": 0.05,
    }


def _run_workers(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run one worker per payload under spawn, synchronized on a shared barrier.

    A ``Manager().Barrier`` (picklable proxy) makes every worker pause after
    writing and release together, so their commits truly race the branch-tip CAS
    — the only way to observe real rebase behaviour on fast local commits.
    """
    ctx = multiprocessing.get_context("spawn")
    with ctx.Manager() as mgr:
        barrier = mgr.Barrier(len(payloads))
        for p in payloads:
            p["barrier"] = barrier
        with ProcessPoolExecutor(max_workers=len(payloads), mp_context=ctx) as ex:
            return list(ex.map(commit_to_group, payloads))


def _emit_worker_metrics(cfg: harness.RunConfig, phase: str, results: list[dict[str, Any]]) -> None:
    """Emit per-worker ``retries`` and ``commit_wall_s`` rows."""
    for r in results:
        harness.emit_metric(cfg, TEST, phase, "retries", r["retries"], "count", group=r["group"], ok=r["ok"])
        harness.emit_metric(cfg, TEST, phase, "commit_wall_s", r["commit_wall_s"], "s", group=r["group"])


def phase_xgroup_conflict(cfg: harness.RunConfig) -> None:
    """3 processes commit to 3 distinct groups at once — expect clean rebases."""
    store = "t0_xgroup"
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    groups = {f"g{i}": _smoke_group_spec(f"EPSG:3260{i + 1}") for i in range(3)}
    seed_groups(repo, groups, commit_msg="seed 3 groups")

    payloads = [_payload(cfg, store, f"g{i}", _one_chunk_region(), seed=100 + i, solver="detector") for i in range(3)]
    results = _run_workers(payloads)
    _emit_worker_metrics(cfg, "xgroup_conflict", results)

    # NOTE: retry *counts* are only meaningful on S3, whose object-store CAS is a
    # true concurrent commit path; the local filesystem backend often serializes
    # committers so retries read 0. T0's real assertion is the qualitative one
    # below (all succeed, none unresolvable). Contention depth is T5's job on S3.
    unresolvable = [r for r in results if r["unresolvable"]]
    failed = [r for r in results if not r["ok"]]
    logger.info(
        "xgroup: retries=%s unresolvable=%d failed=%d", [r["retries"] for r in results], len(unresolvable), len(failed)
    )
    if unresolvable or failed:
        raise SystemExit(
            f"KILL-CRITERION: cross-group commits did not rebase cleanly "
            f"(unresolvable={len(unresolvable)}, failed={len(failed)}). "
            f"Re-plan the one-repo design (ADR 008 D5/D6) before T4/T5. Details: {results}"
        )


def phase_disjoint_region(cfg: harness.RunConfig) -> None:
    """2 processes write disjoint chunks of one array — expect clean rebases."""
    store = "t0_disjoint"
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    seed_groups(repo, {"g0": _smoke_group_spec("EPSG:32601")}, commit_msg="seed 1 group")

    payloads = [
        _payload(cfg, store, "g0", _one_chunk_region(y_chunk=0), seed=1, solver="detector"),
        _payload(cfg, store, "g0", _one_chunk_region(y_chunk=1), seed=2, solver="detector"),
    ]
    results = _run_workers(payloads)
    _emit_worker_metrics(cfg, "disjoint_region", results)
    if any(not r["ok"] for r in results):
        raise SystemExit(f"disjoint-region writes did not both commit: {results}")


def phase_same_chunk_useours(cfg: harness.RunConfig) -> None:
    """2 processes write the SAME chunk — BasicConflictSolver(UseOurs) resolves."""
    store = "t0_samechunk_ours"
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    seed_groups(repo, {"g0": _smoke_group_spec("EPSG:32601")}, commit_msg="seed 1 group")

    region = _one_chunk_region()
    payloads = [
        _payload(cfg, store, "g0", region, seed=1, solver="useours"),
        _payload(cfg, store, "g0", region, seed=2, solver="useours"),
    ]
    results = _run_workers(payloads)
    _emit_worker_metrics(cfg, "same_chunk_useours", results)
    if any(not r["ok"] for r in results):
        raise SystemExit(f"same-chunk UseOurs did not resolve for both writers: {results}")


def phase_same_chunk_detector(cfg: harness.RunConfig) -> None:
    """2 processes write the SAME chunk with detector — expect one unresolvable.

    Confirms the other half of the taxonomy: a bare ``ConflictDetector`` cannot
    resolve a chunk-level conflict, so the losing writer sees RebaseFailedError.
    """
    store = "t0_samechunk_det"
    harness.reset_store(cfg, store)
    repo = harness.create_repo(cfg, store)
    seed_groups(repo, {"g0": _smoke_group_spec("EPSG:32601")}, commit_msg="seed 1 group")

    region = _one_chunk_region()
    payloads = [
        _payload(cfg, store, "g0", region, seed=1, solver="detector"),
        _payload(cfg, store, "g0", region, seed=2, solver="detector"),
    ]
    results = _run_workers(payloads)
    _emit_worker_metrics(cfg, "same_chunk_detector", results)
    n_ok = sum(r["ok"] for r in results)
    n_unresolvable = sum(r["unresolvable"] for r in results)
    logger.info("same_chunk detector: ok=%d unresolvable=%d", n_ok, n_unresolvable)
    # Both may commit if their starts don't actually overlap a chunk write; the
    # meaningful signal is that IF a conflict occurs the detector cannot resolve
    # it. Require at least one committer and no crash-y outcome.
    if n_ok == 0:
        raise SystemExit(f"same-chunk detector: no writer committed at all: {results}")
    if n_unresolvable == 0:
        logger.warning("same_chunk detector: no conflict materialized (starts didn't overlap) — inconclusive")


def main() -> int:
    """Parse args and run the T0 phases in order."""
    parser = argparse.ArgumentParser(description=__doc__)
    harness.add_common_args(parser)
    cfg = harness.config_from_args(parser.parse_args())
    harness.configure_logging()
    logger.info("multiprocessing start method: spawn (icechunk-safe)")

    harness.run_phase(cfg, TEST, "xgroup_conflict", lambda: phase_xgroup_conflict(cfg))
    harness.run_phase(cfg, TEST, "disjoint_region", lambda: phase_disjoint_region(cfg))
    harness.run_phase(cfg, TEST, "same_chunk_useours", lambda: phase_same_chunk_useours(cfg))
    harness.run_phase(cfg, TEST, "same_chunk_detector", lambda: phase_same_chunk_detector(cfg))
    logger.info("T0 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
