r"""Production-scale reproduction of the assembly wedge, on real S3, with native stack capture.

The small harness (`publication_density.py`) reproduces the *condition* — a coordinator falling
four snapshots behind the tip — but never the icechunk hang itself, and its wedges are injected.
This one aims at the real hang. It drives the REAL `write_year_shards` from many coordinator
processes into one icechunk store on dev S3, seeded to **production snapshot weight** (measured,
not assumed), with **production-sized shard objects** (incompressible random data across whole
shards, so nothing the codec elides), and fork phases stretched to tens of minutes so publications
from different coordinators land within one catch-up interval — the depth-4 precondition every
observed hang shared.

**It is branch-agnostic on purpose.** It imports only what exists on both `main` (pre-fix) and the
PR stack, and discovers the fix's spacing gate at runtime. So the same script runs two arms:

* installed at ``main`` → the write path with no watchdog, no killable child, no spacing. If a
  coordinator's rebase hangs, it hangs forever, exactly as on 2026-09-04.
* installed at the fix stack → spacing on, watchdog and killable-child recovery active.

**The capture that Fargate denied is available here.** This runs on a plain EC2 box, where
``py-spy`` has ``CAP_SYS_PTRACE``. When a coordinator makes no shard progress for
``--stall-seconds``, the monitor runs ``py-spy dump --native --subprocesses`` against it and its
fork workers, so a real hang yields its native Rust frame — the artefact the incident never had —
without anything armed in advance. The process is left alive and dumped again, to prove it is
wedged rather than slow, before the run ends.

Everything it writes goes under ``--results-dir`` and is mirrored to ``--s3-results`` so the run
survives the box being torn down. Run it on the box under the dev instance profile (credentials
resolve through the instance role), e.g.::

    python prod_scale_repro.py --arm reproduce --run-id repro-01 \
        --coordinators 10 --cells-per-coordinator 2 --live-shards 1500 \
        --seed-target-snapshot-kb 180 --s3-results s3://global-tessera-embeddings-dev/scoping/prod-repro/repro-01

A tiny local smoke (filesystem store, no py-spy, seconds) needs no credentials::

    python prod_scale_repro.py --arm fixed --run-id smoke --store-uri /tmp/pr.icechunk \
        --coordinators 3 --cells-per-coordinator 1 --live-shards 2 --seed-cells 2 \
        --seed-target-snapshot-kb 0 --fork-target-seconds 0 --no-pyspy
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import itertools
import json
import logging
import multiprocessing as mp
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from tessera_embeddings.config.store_layout import EMBEDDING_DIM, INNER_PX, SHARD_PX
from tessera_embeddings.storage import campaign, global_store, shard_writer
from tessera_embeddings.storage.time_axis import CAMPAIGN_YEARS
from tessera_embeddings.storage.zone_grid import ZONES

log = logging.getLogger("prod_scale_repro")

#: The fix stack installs a process-wide publication gate; `main` has no such symbol. Discovered,
#: not imported at module top, so the script loads under both.
try:
    from tessera_embeddings.storage.publication_spacing import install_publication_gate
except ImportError:  # pragma: no cover - main / pre-fix
    install_publication_gate = None  # type: ignore[assignment]

#: The store's time axis IS the snapshot's weight: the snapshot object lists every array node, and
#: the skeleton alone (120 zones x 9 years) is what puts production's object at ~250-375 KB. A
#: three-year axis capped the 2026-09-08 runs at 127 KB no matter how many cells were seeded.
YEARS: tuple[int, ...] = CAMPAIGN_YEARS


def _credentials(uri: str):  # noqa: ANN202 - icechunk credentials callable or None
    if not uri.startswith("s3://"):
        return None
    from tessera_embeddings.providers.aws.credentials import iam_icechunk_credentials

    return iam_icechunk_credentials


def _open(uri: str, region: str | None):  # noqa: ANN202
    return global_store.open_global_repo(uri, get_credentials=_credentials(uri), region=region)


# --------------------------------------------------------------------------------------------
# a source that writes PRODUCTION-SIZED objects: incompressible random across the whole shard
# --------------------------------------------------------------------------------------------


class RandomShardSource:
    """Every live shard is full of incompressible int8 noise, so the object lands at ~full size.

    The small harness wrote one inner chunk per shard, which the sharding codec compresses to a
    few MB; a hang that depends on manifest weight and object size needs shards at production
    scale. Random int8 does not compress, so each embeddings shard object is close to its raw
    ``SHARD_PX**2 * EMBEDDING_DIM`` bytes (~512 MB at 2048/128). ``live_shards`` sets how many, and
    therefore the fork phase's duration.

    Not a frozen dataclass because it carries a per-instance RNG seed only; it is picklable (the
    writer ships it to spawned workers) as a plain object with picklable fields.
    """

    def __init__(self, coordinator: int, n_live: int, *, compressible: bool = False) -> None:
        self.coordinator = coordinator
        self.n_live = n_live
        #: Seed cells set this: a constant non-fill value compresses to kilobytes per shard yet
        #: still commits every chunk, which is what grows the manifests — the snapshot weight the
        #: hang tracks — without the bytes. The subject cells stay incompressible.
        self.compressible = compressible

    def live_shards(self) -> list[tuple[int, int]]:
        """Column 0, the first ``n_live`` rows."""
        # Column 0, rows 0..n_live-1 — a tall strip, all real. The store is seeded wide enough
        # (the campaign layout) that these are valid shard positions for a dense zone.
        return [(r, 0) for r in range(self.n_live)]

    def load(self, shard: tuple[int, int]) -> dict[str, np.ndarray]:
        """Whole-shard int8 data stamped with the coordinator id; random (subject) or constant (seed)."""
        if self.compressible:
            emb = np.ones((1, SHARD_PX, SHARD_PX, EMBEDDING_DIM), dtype="int8")
        else:
            rng = np.random.default_rng((self.coordinator + 1) * 1_000_003 + shard[0])
            emb = rng.integers(-128, 128, size=(1, SHARD_PX, SHARD_PX, EMBEDDING_DIM), dtype="int8")
        # int8 can hold 1..127: subject coordinators are single digits, seed cells are numbered
        # from 900 and only need a non-fill stamp, so fold the id into range.
        emb[0, :INNER_PX, :INNER_PX, 0] = np.int8(self.coordinator % 126 + 1)
        return {"embeddings": emb}


# --------------------------------------------------------------------------------------------
# the fleet publication gate for a one-box fleet (fix stack only)
# --------------------------------------------------------------------------------------------


class _LockGate:
    def __init__(self, lock: Any) -> None:  # noqa: ANN401
        self._lock = lock

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def _install_spacing(lock: Any | None) -> bool:  # noqa: ANN401
    """Install the publication gate if this build has one and a lock was given. Returns whether."""
    if install_publication_gate is None or lock is None:
        return False
    gate = _LockGate(lock)
    install_publication_gate(lambda: gate)
    return True


# --------------------------------------------------------------------------------------------
# snapshot-weight measurement (branch-agnostic: reads the store's own objects)
# --------------------------------------------------------------------------------------------


def _snapshot_max_bytes(uri: str, region: str | None) -> int:
    """Largest object under the store's ``snapshots/`` prefix — the thing a rebase downloads."""
    if not uri.startswith("s3://"):
        root = Path(uri) / "snapshots"
        return max((p.stat().st_size for p in root.rglob("*") if p.is_file()), default=0)
    import boto3

    p = urlparse(uri)
    prefix = p.path.lstrip("/").rstrip("/") + "/snapshots/"
    s3 = boto3.client("s3", region_name=region)
    biggest = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=p.netloc, Prefix=prefix):
        for obj in page.get("Contents", []):
            biggest = max(biggest, obj["Size"])
    return biggest


# --------------------------------------------------------------------------------------------
# catch-up depth: how far behind the tip a coordinator is at every catch-up tick
# --------------------------------------------------------------------------------------------

#: Ancestry steps walked before giving up. Each step is one snapshot fetch, so the cap bounds the
#: instrumentation's own S3 load; the hang's precondition is 4, so anything at the cap is "deep".
DEPTH_CAP = 8


def _depth(repo: Any, base: str, tip: str, cap: int = DEPTH_CAP) -> int:  # noqa: ANN401
    """Snapshots between the session's base and the branch tip (0 = current), capped."""
    if base == tip:
        return 0
    for depth, snap in enumerate(repo.ancestry(snapshot_id=tip)):
        if snap.id == base:
            return depth
        if depth + 1 >= cap:
            return cap
    return cap


def _recording_catch_up(real: Any, sink: list[dict[str, Any]]) -> Any:  # noqa: ANN401
    """Wrap ``catch_up_best_effort`` so every tick records its depth, outcome and duration.

    The wrapper walks the ancestry from the tip BEFORE the real catch-up runs, so the depth is the
    one the catch-up then faces. That walk fetches up to ``DEPTH_CAP`` snapshot objects per tick on
    top of the subject's own traffic — the same perturbation the small harness carried. Both source
    trees import the function into ``shard_writer`` and call it through a module-level name, so the
    patch reaches the real call on ``main`` and on the fix stack alike.
    """

    def wrapper(repo: Any, session: Any, group: str, *, log: logging.Logger | None = None) -> str:  # noqa: ANN401
        depth = _depth(repo, session.snapshot_id, repo.lookup_branch("main"))
        started = time.monotonic()
        outcome = real(repo, session, group, log=log)
        sink.append({"depth": depth, "outcome": outcome, "s": round(time.monotonic() - started, 3)})
        return outcome

    return wrapper


def _depth_summary(ticks: list[dict[str, Any]]) -> dict[str, Any]:
    depths = [t["depth"] for t in ticks]
    return {
        "ticks": len(depths),
        "max": max(depths) if depths else None,
        "histogram": {str(d): depths.count(d) for d in sorted(set(depths))},
        "at_or_above_4": sum(d >= 4 for d in depths),
        "slowest_s": max((t["s"] for t in ticks), default=None),
    }


# --------------------------------------------------------------------------------------------
# coordinator and marker processes
# --------------------------------------------------------------------------------------------


def _log_to(path: Path) -> None:
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(logging.INFO)


def _coordinator(k: int, cfg: dict[str, Any], lock: Any | None, start_delay_s: float) -> None:  # noqa: ANN401
    results = Path(cfg["results_dir"])
    _log_to(results / f"coord-{k}.log")
    spacing = _install_spacing(lock if cfg["arm"] == "fixed" else None)
    logging.getLogger("prod_scale_repro").info("coordinator %d starting (spacing installed: %s)", k, spacing)
    repo = _open(cfg["store_uri"], cfg["region"])
    ticks: list[dict[str, Any]] = []
    shard_writer.catch_up_best_effort = _recording_catch_up(shard_writer.catch_up_best_effort, ticks)
    time.sleep(start_delay_s)
    out = results / f"coord-{k}.jsonl"
    for idx in range(cfg["cells_per_coordinator"]):
        ticks.clear()
        zone = cfg["assignments"][str(k)]
        year = YEARS[idx % len(YEARS)]
        source = RandomShardSource(k, cfg["live_shards"])
        telemetry: dict[str, Any] = {}
        rec: dict[str, Any] = {"coordinator": k, "zone": zone, "year": year, "started": time.time()}
        t0 = time.monotonic()
        try:
            snap = shard_writer.write_year_shards(
                repo,
                zone,
                YEARS.index(year),
                source,
                n_workers=cfg["n_workers"],
                run_id=f"{cfg['run_id']}-{k}-{idx}",
                telemetry=telemetry,
                log=logging.getLogger(f"coord.{k}"),
            )
            rec.update(outcome="published", snapshot=snap)
        except BaseException as exc:  # a hung arm is killed by the monitor; other failures recorded
            rec.update(outcome="failed", error=f"{type(exc).__name__}: {exc}"[:400])
        rec.update(
            wall_s=round(time.monotonic() - t0, 3),
            telemetry={k_: v for k_, v in telemetry.items() if k_ != "workers"},
            catch_up_depth=_depth_summary(ticks),
            depths=[t["depth"] for t in ticks],
        )
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def _marker(cfg: dict[str, Any], lock: Any | None, stop_after_s: float) -> None:  # noqa: ANN401
    """Terminal marks at the campaign's rate, so the run's publication density matches production."""
    results = Path(cfg["results_dir"])
    _log_to(results / "marker.log")
    _install_spacing(lock if cfg["arm"] == "fixed" else None)
    repo = _open(cfg["store_uri"], cfg["region"])
    deadline = time.monotonic() + stop_after_s
    out = results / "marker.jsonl"
    for zone, year in cfg["marker_cells"]:
        if time.monotonic() >= deadline:
            break
        t0 = time.monotonic()
        try:
            campaign.mark_zone_year_empty(repo, zone, year, run_id=f"{cfg['run_id']}-mark")
            ok, err = True, None
        except Exception as exc:
            ok, err = False, str(exc)[:200]
        with out.open("a") as f:
            f.write(json.dumps({"zone": zone, "year": year, "t": time.time(), "ok": ok, "error": err}) + "\n")
        time.sleep(max(0.0, cfg["marker_rate_s"] - (time.monotonic() - t0)))


# --------------------------------------------------------------------------------------------
# the native-stack monitor: the reason this runs on EC2 and not Fargate
# --------------------------------------------------------------------------------------------


def _shard_progress(logpath: Path) -> tuple[int, int]:
    """The coordinator's heartbeat: (number of progress lines, the latest 'shards written' value).

    A TUPLE, not the maximum: a coordinator's second cell counts up from zero and would sit below
    its first cell's total for its whole fork phase, so a maximum reads as twenty minutes of "no
    progress" on a healthy write. Any new progress line is progress; the value is for the report.
    """
    n, last = 0, 0
    with contextlib.suppress(OSError):
        for line in logpath.read_text(errors="replace").splitlines():
            if "shards written" in line:
                n += 1
                with contextlib.suppress(Exception):
                    last = int(line.split("shards written")[0].split()[-1].split("/")[0])
    return n, last


def _pyspy_dump(pid: int, dest: Path, *, native: bool = True) -> bool:
    cmd = ["py-spy", "dump", "--pid", str(pid), "--subprocesses"]
    if native:
        cmd.append("--native")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception as exc:
        dest.write_text(f"py-spy failed: {exc}\n")
        return False
    dest.write_text(f"$ {' '.join(cmd)}\nrc={out.returncode}\n\n{out.stdout}\n---STDERR---\n{out.stderr}\n")
    return out.returncode == 0


def _monitor(procs: dict[int, mp.Process], cfg: dict[str, Any], stop: Any) -> None:  # noqa: ANN401
    """Per coordinator: if its heartbeat has not moved for `stall_seconds`, py-spy dump it.

    Dumps up to `--stall-dumps` times per coordinator, spaced by the stall window, leaving the
    process ALIVE between them so a series of identical native stacks proves a wedge rather than a
    slow step. The dumps are the evidence; the process is only killed at teardown.
    """
    results = Path(cfg["results_dir"])
    last: dict[int, tuple[tuple[int, int], float]] = {k: ((0, 0), time.monotonic()) for k in procs}
    dumps: dict[int, int] = dict.fromkeys(procs, 0)
    announced: dict[int, bool] = dict.fromkeys(procs, False)
    while not stop.wait(cfg["monitor_period_s"]):
        for k, proc in procs.items():
            if not proc.is_alive() or proc.pid is None:
                continue
            beat = _shard_progress(results / f"coord-{k}.log")
            seen, since = last[k]
            if beat != seen:
                last[k] = (beat, time.monotonic())
                continue
            stalled_for = time.monotonic() - since
            if stalled_for < cfg["stall_seconds"] or dumps[k] >= cfg["stall_dumps"]:
                continue
            if not announced[k]:
                logging.getLogger("prod_scale_repro").critical(
                    "COORDINATOR %d STALLED: no shard progress for %.0f s (last %d shards written); "
                    "py-spy dumping pid %d",
                    k,
                    stalled_for,
                    beat[1],
                    proc.pid,
                )
                announced[k] = True
            dumps[k] += 1
            _pyspy_dump(proc.pid, results / f"coord-{k}.pyspy-{dumps[k]}.txt")
            last[k] = (beat, time.monotonic())  # space the dumps by the stall window


# --------------------------------------------------------------------------------------------
# seed / report / s3 mirror
# --------------------------------------------------------------------------------------------


def _assign(coordinators: int) -> tuple[dict[str, str], list[tuple[str, int]]]:
    names = sorted(ZONES)
    assignments = {str(k): names[k] for k in range(coordinators)}
    pool = [(z, y) for z in names[coordinators:] for y in YEARS]
    return assignments, pool


def _seed(cfg: dict[str, Any], marker_pool: list[tuple[str, int]]) -> dict[str, Any]:
    """Seed the groups, then pre-publish REAL fills until the snapshot object hits the target weight.

    Empty marks are cheap but light; the snapshot object grows by a roughly constant ~130 B per
    published FILL (the manifest reference and run attrs), independent of how many shards the fill
    wrote, so the weight comes from the NUMBER of fills: production's ~250 KB mean is the ~127 KB
    skeleton plus several hundred publications. Seeds therefore cycle over a small reserved slice of
    the pool (``seed_pool_cells``), re-filling the same cells with one compressible shard each —
    the cheapest publication there is — and leave the rest of the pool to the marker, whose marks
    are idempotent and cannot be repeated. Self-calibrating: publish, measure every ten, stop at
    ``seed_target_snapshot_kb`` (or ``seed_cells`` as a hard cap). A failure here is a broken
    environment and stops the run; it is never skipped.
    """
    t0 = time.monotonic()
    repo = global_store.create_global_repo(
        cfg["store_uri"], get_credentials=_credentials(cfg["store_uri"]), region=cfg["region"]
    )
    global_store.seed_zone_groups(repo, ZONES.values(), years=YEARS)
    groups_s = time.monotonic() - t0
    target_bytes = cfg["seed_target_snapshot_kb"] * 1024
    used: list[tuple[str, int]] = []
    t1 = time.monotonic()
    biggest = _snapshot_max_bytes(cfg["store_uri"], cfg["region"])
    for zone, year in itertools.cycle(marker_pool[: cfg["seed_pool_cells"]]):
        if len(used) >= cfg["seed_cells"]:
            break
        if len(used) % 10 == 0:
            biggest = _snapshot_max_bytes(cfg["store_uri"], cfg["region"])
        if target_bytes and biggest >= target_bytes:
            break
        shard_writer.write_year_shards(
            repo,
            zone,
            YEARS.index(year),
            RandomShardSource(900 + len(used), cfg["seed_shards"], compressible=True),
            n_workers=min(8, cfg["seed_shards"]),
            run_id=f"{cfg['run_id']}-seed-{len(used)}",
        )
        used.append((zone, year))
        log.info(
            "seed fill %d (%s-%d) published; snapshot_max %d B (as of the last measurement)",
            len(used),
            zone,
            year,
            biggest,
        )
    return {
        "seed_groups_s": round(groups_s, 1),
        "seed_cells_published": len(used),
        "seed_s": round(time.monotonic() - t1, 1),
        "snapshot_max_bytes": _snapshot_max_bytes(cfg["store_uri"], cfg["region"]),
        "seed_used": used,
    }


def _history(repo: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    out = [{"id": s.id, "at": s.written_at.isoformat(), "message": s.message} for s in repo.ancestry(branch="main")]
    out.reverse()
    return out


def _publication_gaps(history: list[dict[str, Any]], since: float) -> dict[str, Any]:
    starts: list[float] = []
    prev_fill: str | None = None
    for h in history:
        at = dt.datetime.fromisoformat(h["at"]).timestamp()
        if at < since:
            continue
        msg = h["message"]
        if msg.startswith("fill "):
            starts.append(at)
            prev_fill = msg.removeprefix("fill ").split(" year ")[0]
        elif msg.startswith("mark ") and prev_fill and msg.startswith(f"mark {prev_fill} year"):
            prev_fill = None
        else:
            starts.append(at)
            prev_fill = None
    gaps = sorted(round(b - a, 3) for a, b in itertools.pairwise(starts))
    return {"publications": len(starts), "min_gap_s": gaps[0] if gaps else None, "smallest_gaps": gaps[:10]}


def _verify(repo: Any, records: list[dict[str, Any]]) -> dict[str, Any]:  # noqa: ANN401
    import zarr

    root = zarr.open_group(repo.readonly_session(branch="main").store, mode="r")
    problems: list[str] = []
    checked = 0
    for r in records:
        if r["outcome"] != "published":
            continue
        arr = root[r["zone"]]["embeddings"]
        got = int(np.asarray(arr[YEARS.index(r["year"]), :INNER_PX, :INNER_PX, 0])[0, 0])
        want = r["coordinator"] % 126 + 1
        if got != want:
            problems.append(f"{r['zone']}-{r['year']}: expected {want}, saw {got}")
        checked += 1
    return {"chunks_checked": checked, "problems": problems, "ok": not problems}


def _mirror_to_s3(local: Path, s3_uri: str | None) -> None:
    if not s3_uri:
        return
    with contextlib.suppress(Exception):
        subprocess.run(["aws", "s3", "sync", str(local), s3_uri.rstrip("/") + "/", "--only-show-errors"], timeout=600)


def _report(cfg: dict[str, Any], seed: dict[str, Any], started: float, wall: float, stuck: list[int]) -> dict[str, Any]:
    results = Path(cfg["results_dir"])
    records: list[dict[str, Any]] = []
    for p in sorted(results.glob("coord-*.jsonl")):
        records += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    repo = _open(cfg["store_uri"], cfg["region"])
    history = _history(repo)
    pyspy = sorted(str(p.name) for p in results.glob("coord-*.pyspy-*.txt"))
    report = {
        "arm": cfg["arm"],
        "run_id": cfg["run_id"],
        "config": {
            k: cfg[k]
            for k in (
                "coordinators",
                "cells_per_coordinator",
                "live_shards",
                "n_workers",
                "marker_rate_s",
                "stagger_seconds",
                "stall_seconds",
                "seed_target_snapshot_kb",
            )
        },
        "seed": {k: v for k, v in seed.items() if k != "seed_used"},
        "wall_s": round(wall, 1),
        "store_snapshots_total": len(history),
        "cells": {
            "attempted": len(records),
            "published": sum(r["outcome"] == "published" for r in records),
            "failed": sum(r["outcome"] == "failed" for r in records),
            "failures": [
                {"coordinator": r["coordinator"], "cell": f"{r['zone']}-{r['year']}", "error": r.get("error")}
                for r in records
                if r["outcome"] == "failed"
            ],
            "rehomed": sum(bool(r.get("telemetry", {}).get("rehomed")) for r in records),
            "publish_retries": sum(int(r.get("telemetry", {}).get("publish_retries", 0) or 0) for r in records),
            "partitions_rerun": sum(len(r.get("telemetry", {}).get("partitions_rerun", []) or []) for r in records),
            "wall_s": [r.get("wall_s") for r in records],
        },
        "catch_up_depth": _depth_summary(
            [
                {"depth": d, "s": r.get("catch_up_depth", {}).get("slowest_s") or 0.0}
                for r in records
                for d in r.get("depths", [])
            ]
        ),
        "coordinators_stuck_at_teardown": stuck,
        "pyspy_dumps": pyspy,
        "publication_gaps_from_store": _publication_gaps(history, since=started),
        "integrity": _verify(repo, records),
        "hang_captured": bool(pyspy),
    }
    report["condition_reached"] = report["catch_up_depth"]["at_or_above_4"] > 0
    return report


def main(argv: list[str] | None = None) -> int:
    """Seed the store, run the arm, capture any hang, write and mirror the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["reproduce", "fixed"], required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--store-uri", default=None)
    ap.add_argument("--s3-results", default=None, help="Mirror the results dir here (survives teardown).")
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--coordinators", type=int, default=10)
    ap.add_argument("--cells-per-coordinator", type=int, default=2)
    ap.add_argument("--n-workers", type=int, default=16)
    ap.add_argument("--live-shards", type=int, default=1500, help="Real shards per cell; sets the fork-phase duration.")
    ap.add_argument("--stagger-seconds", type=float, default=3.0)
    ap.add_argument("--marker-rate-s", type=float, default=3.0)
    ap.add_argument("--seed-cells", type=int, default=40, help="Hard cap on pre-published seed fills.")
    ap.add_argument(
        "--seed-shards", type=int, default=1, help="Shards per seed fill; weight is per FILL, not per shard."
    )
    ap.add_argument(
        "--seed-pool-cells",
        type=int,
        default=200,
        help="Pool cells reserved for (repeated) seed fills; the rest of the pool feeds the marker.",
    )
    ap.add_argument(
        "--seed-target-snapshot-kb",
        type=float,
        default=180.0,
        help="Stop seeding once the snapshot object reaches this (0 = seed_cells only).",
    )
    ap.add_argument("--fork-target-seconds", type=float, default=0.0, help="Advisory only; recorded in the report.")
    ap.add_argument("--stall-seconds", type=float, default=900.0, help="No shard progress this long → py-spy dump.")
    ap.add_argument("--stall-dumps", type=int, default=5, help="Dumps per stalled coordinator before leaving it be.")
    ap.add_argument("--monitor-period-s", type=float, default=30.0)
    ap.add_argument("--no-pyspy", action="store_true", help="Skip native dumps (local smoke).")
    ap.add_argument("--max-run-seconds", type=float, default=6 * 3600, help="Hard ceiling on the run phase.")
    ap.add_argument("--cleanup", action="store_true")
    args = ap.parse_args(argv)

    store_uri = args.store_uri or f"s3://global-tessera-embeddings-dev/scoping/prod-repro/{args.run_id}.icechunk"
    results = Path(args.results_dir or f"temp/prod_repro/{args.run_id}")
    results.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    region = args.region if store_uri.startswith("s3://") else None
    assignments, pool = _assign(args.coordinators)
    ctx = mp.get_context("spawn")
    # ONE lock, shared by every publisher — coordinators AND the marker — or a fill and a terminal
    # mark would not exclude each other and the spacing measurement would lie.
    lock = ctx.Lock() if args.arm == "fixed" else None
    cfg: dict[str, Any] = {
        "arm": args.arm,
        "run_id": args.run_id,
        "store_uri": store_uri,
        "region": region,
        "results_dir": str(results),
        "coordinators": args.coordinators,
        "cells_per_coordinator": args.cells_per_coordinator,
        "n_workers": args.n_workers,
        "live_shards": args.live_shards,
        "stagger_seconds": args.stagger_seconds,
        "marker_rate_s": args.marker_rate_s,
        "seed_cells": args.seed_cells,
        "seed_shards": args.seed_shards,
        "seed_pool_cells": args.seed_pool_cells,
        "seed_target_snapshot_kb": args.seed_target_snapshot_kb,
        "stall_seconds": args.stall_seconds,
        "stall_dumps": args.stall_dumps,
        "monitor_period_s": args.monitor_period_s,
        "assignments": assignments,
    }
    log.info("[%s] seeding %s to a %.0f KB snapshot target", args.arm, store_uri, args.seed_target_snapshot_kb)
    seed = _seed(cfg, pool)
    log.info(
        "[%s] seeded: %d cells, snapshot_max %d B (%.0f KB)",
        args.arm,
        seed["seed_cells_published"],
        seed["snapshot_max_bytes"],
        seed["snapshot_max_bytes"] / 1024,
    )
    cfg["marker_cells"] = pool[args.seed_pool_cells :]

    if args.no_pyspy:
        cfg["stall_dumps"] = 0
    started = time.time()
    t0 = time.monotonic()
    procs = {
        k: ctx.Process(target=_coordinator, args=(k, cfg, lock, k * args.stagger_seconds), name=f"coord-{k}")
        for k in range(args.coordinators)
    }
    horizon = min(
        args.max_run_seconds,
        args.live_shards * 3.0 * args.cells_per_coordinator
        + args.coordinators * args.stagger_seconds
        + args.stall_seconds * 2
        + 600,
    )
    mk = ctx.Process(target=_marker, args=(cfg, lock, horizon), name="marker") if args.marker_rate_s > 0 else None
    stop = ctx.Event()
    mon = threading.Thread(target=_monitor, args=(procs, cfg, stop), name="native-monitor", daemon=True)
    for p in procs.values():
        p.start()
    if mk:
        mk.start()
    if not args.no_pyspy:
        mon.start()
    deadline = t0 + horizon
    for p in procs.values():
        p.join(timeout=max(1.0, deadline - time.monotonic()))
    stop.set()
    stuck = [k for k, p in procs.items() if p.is_alive()]
    # Mirror BEFORE killing, so a wedged coordinator's dumps are already safe off-box.
    _mirror_to_s3(results, args.s3_results)
    for p in procs.values():
        if p.is_alive():
            p.kill()
    if mk:
        mk.join(timeout=30)
        if mk.is_alive():
            mk.kill()
    wall = time.monotonic() - t0
    report = _report(cfg, seed, started, wall, stuck)
    (results / "report.json").write_text(json.dumps(report, indent=2))
    log.info(
        "REPORT %s",
        json.dumps(
            {
                k: report[k]
                for k in (
                    "arm",
                    "seed",
                    "cells",
                    "catch_up_depth",
                    "condition_reached",
                    "publication_gaps_from_store",
                    "coordinators_stuck_at_teardown",
                    "pyspy_dumps",
                    "hang_captured",
                    "integrity",
                )
            },
            indent=1,
        ),
    )
    _mirror_to_s3(results, args.s3_results)
    if args.cleanup:
        from tessera_embeddings.storage.object_store import delete_prefix

        with contextlib.suppress(Exception):
            delete_prefix(store_uri, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
