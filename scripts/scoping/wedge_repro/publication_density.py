r"""Reproduce the campaign's publication density against a real store, and measure what it does.

The assembly wedges of 2026-08-29, 08-31 and 09-04 all share one precondition: a coordinator's
session falls two publications (four snapshots) behind the branch tip before it rebases. This
harness drives the REAL write path — ``write_year_shards`` with its fork pool, periodic catch-up,
fork-phase watchdog and publication spacing — from several coordinator processes at once, into one
icechunk repository seeded at production geometry, with terminal marks interleaved at the rate the
campaign produces them, and records what each mechanism did:

* the depth (snapshots behind the tip) at every catch-up tick, per coordinator — the quantity the
  hang tracks, measured rather than inferred;
* the gap between consecutive publications, read back from the STORE's own snapshot timestamps,
  which is the only record of spacing that does not trust the code under test;
* whether the fork-phase watchdog fired, whether a wedged catch-up was re-homed, and whether the
  thread stacks it exists to capture actually landed in the log;
* and whether every coordinator's data came back intact, because a recovery that clobbered a
  neighbour would be worse than the hang.

Arms are selected on the command line. ``--spacing off`` is today's main; ``--spacing on`` is
#183. ``--wedge catch_up`` hangs one coordinator's catch-up after two ticks (the 08-31 failure, made
deterministic); ``--wedge worker`` hangs one of its fork workers (the shape #181's watchdog exists
for). The icechunk hang itself is NOT injected and has never reproduced at small scale; what this
reproduces is the CONDITION, and it says so in its report.

Run from the repository root with the dev account's credentials::

    AWS_PROFILE=global-tessera-dev uv run python scripts/scoping/wedge_repro/publication_density.py \\
        --run-id density-01 --coordinators 10 --seed-snapshots 300 --fork-seconds 120 --spacing on

A local smoke run needs no credentials::

    uv run python scripts/scoping/wedge_repro/publication_density.py --run-id smoke \\
        --store-uri /tmp/wedge-smoke.icechunk --coordinators 3 --fork-seconds 5 --seed-snapshots 10

Everything it writes goes under ``--results-dir`` (default ``temp/wedge_repro/<run-id>``): one JSON
line per cell, one log and one stderr file per coordinator (the stack dumps are on stderr), and
``report.json`` with the arm's verdicts. ``--cleanup`` deletes the store afterwards.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import faulthandler
import functools
import itertools
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Callable
from multiprocessing import synchronize
from pathlib import Path
from typing import Any

import icechunk
import numpy as np

from tessera_embeddings.config.store_layout import EMBEDDING_DIM, INNER_PX, SHARD_PX
from tessera_embeddings.storage import campaign, global_store, publication_spacing, shard_writer
from tessera_embeddings.storage.zone_grid import ZONES

log = logging.getLogger("wedge_repro")

#: Years on the seeded axis. Three keeps the seed commit small and leaves room for a mark per year.
YEARS: tuple[int, ...] = (2023, 2024, 2025)
#: How many shards a coordinator's cell writes. Two per cell keeps the data small; the snapshot
#: and manifest objects are production-shaped regardless, because the LAYOUT is production's.
SHARDS_PER_CELL = 2


# --------------------------------------------------------------------------------------------
# credentials / repo
# --------------------------------------------------------------------------------------------


def _credentials(uri: str) -> Callable[[], icechunk.S3StaticCredentials] | None:
    if not uri.startswith("s3://"):
        return None
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

    return iam_icechunk_credentials


def _open(uri: str, region: str | None) -> icechunk.Repository:
    return global_store.open_global_repo(uri, get_credentials=_credentials(uri), region=region)


# --------------------------------------------------------------------------------------------
# the shard source: production geometry, one land inner chunk per shard, values that name the writer
# --------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class IdentitySource:
    """Two shards per cell; each carries one land inner chunk whose value is the coordinator's id.

    The value is the oracle: after the run, every coordinator's chunk is read back and must hold
    that coordinator's id. A recovery that merged the wrong fork, or a neighbour's commit that
    overwrote ours, shows up as the wrong number — an oracle that can move.

    ``load_delay_s`` stretches the fork phase so catch-ups tick for a realistic while and other
    coordinators publish underneath; ``hang_shard`` makes one shard's load never return, which is
    the 09-04 shape a worker-side wedge takes.
    """

    coordinator: int
    load_delay_s: float
    hang_shard: int | None = None
    #: When set, the hang happens only until this marker file exists: the first worker to reach
    #: the shard creates it and hangs, the re-run's worker finds it and writes. Without it the
    #: hang is deterministic and the re-run stalls again — the fail-after-one-retry path.
    hang_once_marker: str | None = None

    def live_shards(self) -> list[tuple[int, int]]:
        """The first ``SHARDS_PER_CELL`` shards of column 0."""
        return [(0, 0), (1, 0)][:SHARDS_PER_CELL]

    def load(self, shard: tuple[int, int]) -> dict[str, np.ndarray]:
        """One land inner chunk carrying this coordinator's id; hangs forever on ``hang_shard``."""
        if self.hang_shard is not None and shard[0] == self.hang_shard:
            marker = Path(self.hang_once_marker) if self.hang_once_marker else None
            if marker is None or not marker.exists():
                if marker is not None:
                    marker.write_text("hung once")
                time.sleep(24 * 3600)  # never returns within any test's horizon
        time.sleep(self.load_delay_s)
        emb = np.zeros((1, SHARD_PX, SHARD_PX, EMBEDDING_DIM), dtype="int8")
        emb[0, :INNER_PX, :INNER_PX, :] = np.int8(self.coordinator + 1)
        return {"embeddings": emb}


# --------------------------------------------------------------------------------------------
# the fleet-wide gate for a one-machine fleet: a spawn-safe lock, held per publication
# --------------------------------------------------------------------------------------------


class _LockGate:
    """A publication gate backed by a multiprocessing lock — the fleet mutex, on one box."""

    def __init__(self, lock: synchronize.Lock) -> None:
        self._lock = lock

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def _wedging_publish_child(conn, repo, group, base, forks, fill_message, attrs, mode, step_timeout_s) -> None:  # noqa: ANN001
    """A publish child that hangs on its first attempt per marker file, then behaves.

    The 2026-09-04 shape at the publish: the coordinator's commit path parks inside icechunk.
    The parent must kill it and retry on a fresh session with the forks it still holds.
    """
    marker = Path(attrs.pop("_marker"))
    if not marker.exists():
        marker.write_text("wedged once")
        # As the real child does: arm the stack dump inside the parent's timeout, so the wedge
        # this arm injects leaves the artefact a production wedge would.
        faulthandler.dump_traceback_later(max(step_timeout_s - 5.0, 1.0), repeat=False)
        time.sleep(24 * 3600)
    shard_writer._publish_child(conn, repo, group, base, forks, fill_message, attrs, mode, step_timeout_s)


def _install_gate(lock: synchronize.Lock | None) -> None:
    if lock is None:
        publication_spacing.install_publication_gate(None)
        return
    gate = _LockGate(lock)
    publication_spacing.install_publication_gate(lambda: gate)


# --------------------------------------------------------------------------------------------
# depth instrumentation: how far behind the tip a coordinator is at every catch-up tick
# --------------------------------------------------------------------------------------------


def _depth(repo: icechunk.Repository, base: str, tip: str, cap: int = 60) -> int:
    if base == tip:
        return 0
    for depth, snap in enumerate(repo.ancestry(snapshot_id=tip)):
        if snap.id == base:
            return depth
        if depth + 1 >= cap:
            return cap
    return cap


def _recording_catch_up(
    real: Callable[..., str], sink: list[dict[str, Any]], hang_after: int | None
) -> Callable[..., str]:
    """Wrap ``catch_up_best_effort`` to record depth and outcome per tick, optionally wedging."""
    calls = {"n": 0}

    def wrapper(
        repo: icechunk.Repository, session: icechunk.Session, group: str, *, log: logging.Logger | None = None
    ) -> str:
        calls["n"] += 1
        tip = repo.lookup_branch("main")
        base = session.snapshot_id
        depth = _depth(repo, base, tip)
        started = time.monotonic()
        if hang_after is not None and calls["n"] > hang_after:
            sink.append({"t": time.time(), "depth": depth, "outcome": "WEDGED (injected)", "s": None})
            time.sleep(24 * 3600)
        outcome = real(repo, session, group, log=log)
        sink.append({"t": time.time(), "depth": depth, "outcome": outcome, "s": round(time.monotonic() - started, 3)})
        return outcome

    return wrapper


# --------------------------------------------------------------------------------------------
# processes
# --------------------------------------------------------------------------------------------


def _redirect_stderr(path: Path) -> None:
    """Send this process's C-level stderr to a file, so faulthandler's stack dumps are kept."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 2)
    sys.stderr = os.fdopen(2, "w", buffering=1)


def _configure_logging(path: Path) -> None:
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def coordinator(k: int, cfg: dict[str, Any], lock: synchronize.Lock | None, start_delay_s: float) -> None:
    """One cluster's trailing-assembly thread, in effect: fill this coordinator's cells in turn."""
    results = Path(cfg["results_dir"])
    _redirect_stderr(results / f"coord-{k}.stderr")
    _configure_logging(results / f"coord-{k}.log")
    _install_gate(lock)
    if cfg["fork_stall_timeout_s"]:
        # `write_year_shards` runs the fork phase through `run_forks`; `run_forked` is the per-ROI
        # wrapper. Patch both, so the shortened timeout reaches whichever path a caller takes.
        shard_writer.run_forks = functools.partial(  # type: ignore[assignment]
            shard_writer.run_forks, fork_stall_timeout_s=cfg["fork_stall_timeout_s"]
        )
        shard_writer.run_forked = functools.partial(  # type: ignore[assignment]
            shard_writer.run_forked, fork_stall_timeout_s=cfg["fork_stall_timeout_s"]
        )
    ticks: list[dict[str, Any]] = []
    hang_after = 2 if (cfg["wedge"] == "catch_up" and k == cfg["wedge_coordinator"]) else None
    shard_writer.catch_up_best_effort = _recording_catch_up(  # type: ignore[assignment]
        shard_writer.catch_up_best_effort, ticks, hang_after
    )
    if cfg["wedge"] == "publish" and k == cfg["wedge_coordinator"]:
        wedging = functools.partial(
            shard_writer.publish_forks_in_child,
            _child=_wedging_publish_child,
            step_timeout_s=cfg["publish_step_timeout_s"],
        )
        marker = str(results / f"coord-{k}.publish-wedged")

        def _with_marker(*a: Any, **kw: Any) -> Any:  # noqa: ANN401 — a pass-through wrapper
            # The marker rides in the attrs the child receives; `_wedging_publish_child` pops it.
            kw["attrs"] = {**kw["attrs"], "_marker": marker}
            return wedging(*a, **kw)

        shard_writer.publish_forks_in_child = _with_marker  # type: ignore[assignment]
    repo = _open(cfg["store_uri"], cfg["region"])
    time.sleep(start_delay_s)
    out = results / f"coord-{k}.jsonl"
    for zone, year in cfg["assignments"][str(k)]:
        wedged_here = k == cfg["wedge_coordinator"] and cfg["wedge"] in ("worker", "worker_always")
        source = IdentitySource(
            k,
            cfg["fork_seconds"] / SHARDS_PER_CELL,
            hang_shard=1 if wedged_here else None,
            hang_once_marker=str(results / f"coord-{k}.worker-hung") if cfg["wedge"] == "worker" else None,
        )
        telemetry: dict[str, Any] = {}
        record: dict[str, Any] = {"coordinator": k, "zone": zone, "year": year, "started": time.time()}
        t0 = time.monotonic()
        try:
            snapshot = shard_writer.write_year_shards(
                repo,
                zone,
                YEARS.index(year),
                source,
                n_workers=cfg["n_workers"],
                run_id=f"{cfg['run_id']}-{k}",
                telemetry=telemetry,
                log=logging.getLogger(f"coord.{k}"),
            )
            record.update(outcome="published", snapshot=snapshot)
        except BaseException as exc:  # the harness must report a failure, not die of it
            record.update(outcome="failed", error=f"{type(exc).__name__}: {exc}"[:400])
        record.update(
            wall_s=round(time.monotonic() - t0, 3),
            telemetry={k_: v for k_, v in telemetry.items() if k_ != "workers"},
            ticks=list(ticks),
        )
        ticks.clear()
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")


def marker(cfg: dict[str, Any], lock: synchronize.Lock | None, stop_after_s: float) -> None:
    """The campaign's terminal cells: a publication every ``marker_rate_s`` for the run's length."""
    results = Path(cfg["results_dir"])
    _redirect_stderr(results / "marker.stderr")
    _configure_logging(results / "marker.log")
    _install_gate(lock)
    repo = _open(cfg["store_uri"], cfg["region"])
    deadline = time.monotonic() + stop_after_s
    out = results / "marker.jsonl"
    for zone, year in cfg["marker_cells"]:
        if time.monotonic() >= deadline:
            break
        t0 = time.monotonic()
        try:
            campaign.mark_zone_year_empty(repo, zone, year, run_id=f"{cfg['run_id']}-mark")
            rec = {"zone": zone, "year": year, "t": time.time(), "s": round(time.monotonic() - t0, 3), "ok": True}
        except Exception as exc:
            rec = {"zone": zone, "year": year, "t": time.time(), "ok": False, "error": str(exc)[:200]}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(max(0.0, cfg["marker_rate_s"] - (time.monotonic() - t0)))


# --------------------------------------------------------------------------------------------
# seeding, verification, report
# --------------------------------------------------------------------------------------------


def _seed(uri: str, region: str | None, n_seed: int, marker_pool: list[tuple[str, int]]) -> dict[str, Any]:
    t0 = time.monotonic()
    repo = global_store.create_global_repo(uri, get_credentials=_credentials(uri), region=region)
    global_store.seed_zone_groups(repo, ZONES.values(), years=YEARS)
    seeded_at = time.monotonic() - t0
    # Volume: the campaign store carries thousands of snapshots and manifests accumulated over
    # hundreds of cells, and the snapshot object the hang downloads grows with them. Pre-commit
    # terminal marks so the run's coordinators rebase over production-like snapshot objects.
    t1 = time.monotonic()
    used: list[tuple[str, int]] = []
    for zone, year in marker_pool[:n_seed]:
        campaign.mark_zone_year_empty(repo, zone, year, run_id="seed")
        used.append((zone, year))
    return {
        "seed_groups_s": round(seeded_at, 1),
        "seed_snapshots": len(used),
        "seed_snapshots_s": round(time.monotonic() - t1, 1),
        "seed_used": used,
    }


def _history(repo: icechunk.Repository) -> list[dict[str, Any]]:
    out = []
    for snap in repo.ancestry(branch="main"):
        out.append({"id": snap.id, "at": snap.written_at.isoformat(), "message": snap.message})
    out.reverse()  # oldest first
    return out


def _publication_gaps(history: list[dict[str, Any]], since: float) -> dict[str, Any]:
    """Gaps between consecutive PUBLICATIONS, from the store's own timestamps.

    A filled cell is two snapshots ("fill X year Y" then "mark X year Y complete"); a terminal mark
    is one. The spacing invariant is about the gap from one publication's FIRST snapshot to the
    next publication's first, so pairs are collapsed to their first snapshot before measuring.
    """
    snaps = [h for h in history if dt.datetime.fromisoformat(h["at"]).timestamp() >= since]
    starts: list[float] = []
    previous_fill: str | None = None
    for h in snaps:
        msg = h["message"]
        at = dt.datetime.fromisoformat(h["at"]).timestamp()
        if msg.startswith("fill "):
            starts.append(at)
            previous_fill = msg.removeprefix("fill ")
        elif (
            msg.startswith("mark ")
            and previous_fill
            and msg.startswith(f"mark {previous_fill.split(' year ')[0]} year")
        ):
            previous_fill = None  # the second snapshot of a pair: not a new publication
        else:
            starts.append(at)
            previous_fill = None
    gaps = [round(b - a, 3) for a, b in itertools.pairwise(starts)]
    return {"publications": len(starts), "min_gap_s": min(gaps) if gaps else None, "gaps_s": gaps}


def _verify(repo: icechunk.Repository, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Every published cell's identity chunk holds its coordinator's id; nothing else was touched."""
    import zarr

    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    problems: list[str] = []
    checked = 0
    for rec in records:
        if rec["outcome"] != "published":
            continue
        arr = root[rec["zone"]]["embeddings"]
        ti = YEARS.index(rec["year"])
        want = rec["coordinator"] + 1
        for sy in range(SHARDS_PER_CELL):
            y0 = sy * SHARD_PX
            block = arr[ti, y0 : y0 + INNER_PX, 0:INNER_PX, 0]
            if not (block == want).all():
                problems.append(
                    f"{rec['zone']}-{rec['year']} shard {sy}: expected {want}, saw {np.unique(block).tolist()[:5]}"
                )
            checked += 1
    return {"chunks_checked": checked, "problems": problems, "ok": not problems}


def _scan_logs(results: Path) -> dict[str, Any]:
    text = ""
    for p in sorted(results.glob("*.log")) + sorted(results.glob("*.stderr")):
        with contextlib.suppress(OSError):
            text += p.read_text()
    return {
        "watchdog_fired": text.count("ASSEMBLY FORK PHASE STALLED"),
        "publish_wedged": text.count("PUBLISH WEDGED"),
        "stacks_dumped": text.count("Thread 0x"),
        "rehomed": text.count("Re-homing"),
        "catch_up_did_not_stop": text.count("CatchUpDidNotStopError") + text.count("still running"),
        "commit_alarm": text.count("ASSEMBLY COMMIT STALLED"),
        "caught_up_lines": text.count("Caught up "),
    }


def _report(cfg: dict[str, Any], seed: dict[str, Any], started: float, wall: float) -> dict[str, Any]:
    results = Path(cfg["results_dir"])
    records: list[dict[str, Any]] = []
    for p in sorted(results.glob("coord-*.jsonl")):
        records += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    marks = []
    if (results / "marker.jsonl").exists():
        marks = [json.loads(line) for line in (results / "marker.jsonl").read_text().splitlines() if line.strip()]
    depths = [t["depth"] for r in records for t in r["ticks"]]
    repo = _open(cfg["store_uri"], cfg["region"])
    history = _history(repo)
    report = {
        "run_id": cfg["run_id"],
        "arm": {
            k: cfg[k]
            for k in (
                "spacing",
                "wedge",
                "wedge_coordinator",
                "coordinators",
                "fork_seconds",
                "marker_rate_s",
                "n_workers",
                "fork_stall_timeout_s",
            )
        },
        "seed": {k: v for k, v in seed.items() if k != "seed_used"},
        "store_snapshots_total": len(history),
        "wall_s": round(wall, 1),
        "cells": {
            "attempted": len(records),
            "published": sum(r["outcome"] == "published" for r in records),
            "failed": sum(r["outcome"] == "failed" for r in records),
            "rehomed": sum(bool(r.get("telemetry", {}).get("rehomed")) for r in records),
            "publish_retries": sum(int(r.get("telemetry", {}).get("publish_retries", 0) or 0) for r in records),
            "partitions_rerun": sum(len(r.get("telemetry", {}).get("partitions_rerun", []) or []) for r in records),
            "failures": [
                {"coordinator": r["coordinator"], "cell": f"{r['zone']}-{r['year']}", "error": r.get("error")}
                for r in records
                if r["outcome"] == "failed"
            ],
            "wall_s": [r["wall_s"] for r in records],
            "commit_s": [r.get("telemetry", {}).get("commit_s") for r in records if r["outcome"] == "published"],
        },
        "terminal_marks": {"published": sum(m["ok"] for m in marks), "failed": sum(not m["ok"] for m in marks)},
        "catch_up_depth": {
            "ticks": len(depths),
            "max": max(depths) if depths else None,
            "histogram": {str(d): depths.count(d) for d in sorted(set(depths))},
            "at_or_above_4": sum(d >= 4 for d in depths),
        },
        "publication_gaps_from_store": _publication_gaps(history, since=started),
        "logs": _scan_logs(results),
        "integrity": _verify(repo, records),
    }
    report["verdicts"] = {
        "condition_reached (depth >= 4 seen)": report["catch_up_depth"]["at_or_above_4"] > 0,
        # The property spacing exists for: no tick ever sees more than one publication (two
        # snapshots) since the last. Measured at every tick, this is the primary verdict.
        # None (JSON null) when no tick fired: a fork phase shorter than the tick interval is not
        # evidence either way, and must not read as a failed verdict.
        "depth_bounded (max catch-up depth <= 2)": (
            None if report["catch_up_depth"]["max"] is None else report["catch_up_depth"]["max"] <= 2
        ),
        # The mechanism, checked against the store's own clock: consecutive publications at least
        # one spacing apart (a hair of tolerance for timestamp rounding).
        "spacing_held (min publication gap >= spacing)": (
            report["publication_gaps_from_store"]["min_gap_s"] is not None
            and report["publication_gaps_from_store"]["min_gap_s"] >= publication_spacing.PUBLICATION_SPACING_S * 0.98
        ),
        "no_data_clobbered": report["integrity"]["ok"],
        "icechunk_hang_observed": report["logs"]["commit_alarm"] > 0
        or any("hung" in (r.get("error") or "") for r in records),
    }
    return report


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------


def _assign(
    coordinators: int, cells_per_coordinator: int
) -> tuple[dict[str, list[tuple[str, int]]], list[tuple[str, int]]]:
    """Coordinators take distinct zones (the campaign's partition); the rest feed seeding and marks."""
    names = sorted(ZONES)
    assignments: dict[str, list[tuple[str, int]]] = {}
    for k in range(coordinators):
        zone = names[k]
        assignments[str(k)] = [(zone, YEARS[i % len(YEARS)]) for i in range(cells_per_coordinator)]
    pool = [(z, y) for z in names[coordinators:] for y in YEARS]
    return assignments, pool


def main(argv: list[str] | None = None) -> int:
    """Seed the store, run the arm, write the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--store-uri",
        default=None,
        help="Default: s3://global-tessera-embeddings-dev/scoping/wedge-repro/<run-id>.icechunk",
    )
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--coordinators", type=int, default=10)
    ap.add_argument("--cells-per-coordinator", type=int, default=1)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument(
        "--fork-seconds", type=float, default=120.0, help="Stretch each cell's fork phase to about this long."
    )
    ap.add_argument(
        "--stagger-seconds", type=float, default=2.0, help="Start delay between coordinators; small = dense finishes."
    )
    ap.add_argument(
        "--marker-rate-s", type=float, default=3.0, help="Seconds between terminal marks; 0 disables the marker."
    )
    ap.add_argument(
        "--seed-snapshots", type=int, default=300, help="Terminal marks committed before the run, for snapshot volume."
    )
    ap.add_argument("--spacing", choices=["on", "off"], default="on")
    ap.add_argument(
        "--wedge",
        choices=["none", "catch_up", "worker", "worker_always", "publish"],
        default="none",
        help=(
            "worker: one partition hangs ONCE (the re-run succeeds); "
            "worker_always: it hangs every time (fails after one re-run)."
        ),
    )
    ap.add_argument(
        "--publish-step-timeout-s", type=float, default=60.0, help="Per-step publish timeout under --wedge publish."
    )
    ap.add_argument("--wedge-coordinator", type=int, default=3)
    ap.add_argument(
        "--fork-stall-timeout-s",
        type=float,
        default=0.0,
        help="Override the watchdog's timeout (0 = production value).",
    )
    ap.add_argument("--cleanup", action="store_true", help="Delete the store afterwards.")
    args = ap.parse_args(argv)

    store_uri = args.store_uri or f"s3://global-tessera-embeddings-dev/scoping/wedge-repro/{args.run_id}.icechunk"
    results = Path(args.results_dir or f"temp/wedge_repro/{args.run_id}")
    results.mkdir(parents=True, exist_ok=True)
    # The harness's own lines go to stdout; C-level stderr (icechunk's commit tracing, faulthandler)
    # goes to a file, so the terminal shows the run and the file keeps the evidence.
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _redirect_stderr(results / "main.stderr")
    region = args.region if store_uri.startswith("s3://") else None

    assignments, pool = _assign(args.coordinators, args.cells_per_coordinator)
    cfg: dict[str, Any] = {
        "run_id": args.run_id,
        "store_uri": store_uri,
        "region": region,
        "results_dir": str(results),
        "coordinators": args.coordinators,
        "n_workers": args.n_workers,
        "fork_seconds": args.fork_seconds,
        "marker_rate_s": args.marker_rate_s,
        "spacing": args.spacing,
        "wedge": args.wedge,
        "wedge_coordinator": args.wedge_coordinator,
        "fork_stall_timeout_s": args.fork_stall_timeout_s,
        "publish_step_timeout_s": args.publish_step_timeout_s,
        "assignments": assignments,
    }
    log.info(
        "Seeding %s with %d zone groups at production layout, then %d terminal marks",
        store_uri,
        len(ZONES),
        args.seed_snapshots,
    )
    seed = _seed(store_uri, region, args.seed_snapshots, pool)
    cfg["marker_cells"] = [c for c in pool if c not in set(map(tuple, seed["seed_used"]))]
    log.info(
        "Seeded in %.0fs (%d snapshots). Starting %d coordinators, spacing=%s, wedge=%s",
        seed["seed_groups_s"] + seed["seed_snapshots_s"],
        seed["seed_snapshots"],
        args.coordinators,
        args.spacing,
        args.wedge,
    )

    ctx = mp.get_context("spawn")
    lock = ctx.Lock() if args.spacing == "on" else None
    started = time.time()
    t0 = time.monotonic()
    procs = [
        ctx.Process(target=coordinator, args=(k, cfg, lock, k * args.stagger_seconds), name=f"coord-{k}")
        for k in range(args.coordinators)
    ]
    horizon = args.fork_seconds * args.cells_per_coordinator + args.coordinators * args.stagger_seconds + 120.0
    mk = ctx.Process(target=marker, args=(cfg, lock, horizon), name="marker") if args.marker_rate_s > 0 else None
    for p in procs:
        p.start()
    if mk:
        mk.start()
    # A wedged coordinator is the point of two arms, so the join is bounded: give the watchdog or
    # the catch-up recovery time to act, then move on and let the report say what did not return.
    stall = args.fork_stall_timeout_s or shard_writer.FORK_STALL_TIMEOUT_S
    deadline = time.monotonic() + horizon + 2 * stall + 2 * args.publish_step_timeout_s + 120.0
    for p in procs:
        p.join(timeout=max(1.0, deadline - time.monotonic()))
    stuck = [p.name for p in procs if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.terminate()
    if mk:
        mk.join(timeout=30)
        if mk.is_alive():
            mk.terminate()
    wall = time.monotonic() - t0

    report = _report(cfg, seed, started, wall)
    report["coordinators_still_running_at_deadline"] = stuck
    (results / "report.json").write_text(json.dumps(report, indent=2))
    log.info(
        "REPORT %s",
        json.dumps(
            {
                k: report[k]
                for k in ("arm", "cells", "catch_up_depth", "publication_gaps_from_store", "logs", "verdicts")
            },
            indent=1,
        ),
    )
    log.info("Written to %s", results / "report.json")
    if args.cleanup:
        from tessera_embeddings.storage.object_store import delete_prefix

        delete_prefix(store_uri, log=log)
        log.info("Deleted %s", store_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
