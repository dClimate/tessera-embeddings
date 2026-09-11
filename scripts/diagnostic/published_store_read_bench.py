"""Measure read performance against the published store, comparably to the scoping runs.

The chunk and shard geometry was chosen on a synthetic benchmark (``scripts/scoping/scale_tests/``,
recorded in ADR 008). This asks the same questions of the store that was built, so the workloads,
their extents and the concurrency sweep are copied from that harness: a different point count would
produce figures that look comparable and are not.

Three definitions, because the obvious version of each misleads:

* **Probes land only on pixels that hold data.** A read of an absent chunk is answered from the
  manifest without a request, so a sample mixing elided ocean with real reads reports the mixture.
* **Cold is a fresh process; warm is a second pass through the same open.**
* **Throughput is decompressed elements per second**, the scoping harness's definition — within a
  few percent of the wire rate here, because the quantized embeddings barely compress.
  ``band_subset`` counts only the bands asked for while all 128 are fetched, so it understates by
  sixteenfold; kept for comparability.

Run from the REPOSITORY ROOT::

    uv run python scripts/diagnostic/published_store_read_bench.py --zone 33N --year 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, cast

import numpy as np
import zarr

from tessera_embeddings.config.store_layout import INNER_PX, SHARD_PX
from tessera_embeddings.storage import published_store
from tessera_embeddings.storage.global_store import open_global_repo

DEFAULT_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGION = "us-west-2"

#: From the scoping harness (`t1_read_bench.CONCURRENCIES_BENCH`). Its published figures are medians
#: ACROSS this sweep, so reproducing the sweep is what makes a median comparable to a median.
CONCURRENCIES = (10, 64, 128)

#: ``(label, northing extent, easting extent, bands)``, extents from `t1_read_bench.WORKLOADS`.
#: `bulk` at 4096 px spans a 2x2 block of 2048-px shards, which is why the benchmark needs a
#: contiguous live block rather than any live shard.
WORKLOADS = (
    ("patch", 100, 100, 128),
    ("tile", 1000, 1000, 128),
    ("band_subset", 512, 512, 8),
    ("bulk", 4096, 4096, 128),
)


def _host_facts() -> dict[str, Any]:
    """Where this ran: EC2 identity when there is one, the machine's own name otherwise.

    Recorded rather than passed in, because an argument can be wrong and a benchmark that cannot
    say which region it ran in is not evidence about either region.
    """
    facts: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "ec2": False,
    }
    try:
        token_request = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(token_request, timeout=1) as response:
            token = response.read().decode()
        document_request = urllib.request.Request(
            "http://169.254.169.254/latest/dynamic/instance-identity/document",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(document_request, timeout=1) as response:
            identity = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError):
        return facts
    facts.update(
        ec2=True,
        region=identity.get("region"),
        availability_zone=identity.get("availabilityZone"),
        instance_type=identity.get("instanceType"),
        instance_id=identity.get("instanceId"),
    )
    return facts


def _percentiles(samples: list[float]) -> dict[str, float]:
    """p50/p95/p99 in milliseconds, plus the sample size the percentiles rest on."""
    array = np.asarray(samples, dtype="float64") * 1e3
    return {
        "n": int(array.size),
        "p50_ms": round(float(np.percentile(array, 50)), 2),
        "p95_ms": round(float(np.percentile(array, 95)), 2),
        "p99_ms": round(float(np.percentile(array, 99)), 2),
        "mean_ms": round(float(array.mean()), 2),
    }


# ── the worker: one phase, one process ───────────────────────────────────────


def _open_zone(payload: dict[str, Any]) -> tuple[zarr.Group, dict[str, float]]:
    """Open the store to one zone group at the payload's concurrency, timing each step."""
    zarr.config.set({"async.concurrency": payload["concurrency"]})
    timings: dict[str, float] = {}
    started = time.monotonic()
    # Opens the way a consumer does, inheriting whatever config the store holds.
    repo = open_global_repo(payload["uri"], region=payload["region"], anonymous=payload["anonymous"])
    timings["repository_open_s"] = round(time.monotonic() - started, 3)
    started = time.monotonic()
    session = repo.readonly_session(branch="main")
    timings["readonly_session_s"] = round(time.monotonic() - started, 3)
    started = time.monotonic()
    root = zarr.open_group(session.store, mode="r")
    timings["root_group_open_s"] = round(time.monotonic() - started, 3)
    started = time.monotonic()
    group = root[payload["zone"]]
    _ = group["embeddings"].shape  # force the array metadata, which is what a reader needs
    timings["zone_group_open_s"] = round(time.monotonic() - started, 3)
    return group, timings


def _net_bytes_received() -> int | None:
    """Interface bytes received so far, or None when psutil is unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.net_io_counters().bytes_recv)


def run_phase(payload: dict[str, Any], repeats: int = 1) -> list[dict[str, Any]]:
    """Open the store once, run one phase ``repeats`` times on that open, and return each result.

    **The repeats share one opened group, and that is the whole point.** Calling this twice for the
    warm arm re-opens everything, so the "warm" figure becomes a second cold reader and every cache
    the benchmark exists to observe is thrown away between the two.

    The open phases ignore ``repeats``: their measurement IS the open, paid once per handle.
    """
    group, timings = _open_zone(payload)
    # The sum, not the steps, is what a consumer waits for before their first read; the steps stay
    # beside it because they say WHERE the wait is.
    timings["total_open_s"] = round(sum(timings.values()), 3)
    base: dict[str, Any] = {"phase": payload["phase"], "concurrency": payload["concurrency"], **timings}
    if payload["phase"] == "open":
        return [base]
    return [{**base, "pass": n + 1, **_measure(group, payload)} for n in range(max(1, repeats))]


def _measure(group: zarr.Group, payload: dict[str, Any]) -> dict[str, Any]:
    """One pass of a read workload on an already-open group."""
    result: dict[str, Any] = {}
    embeddings = group["embeddings"]
    time_index = payload["time_index"]

    if payload["phase"] == "point":
        before = _net_bytes_received()
        latencies: list[float] = []
        for y, x in payload["points"]:
            started = time.monotonic()
            _ = embeddings[time_index, y, x, :]
            latencies.append(time.monotonic() - started)
        after = _net_bytes_received()
        result.update(_percentiles(latencies))
        if before is not None and after is not None:
            result["wire_bytes_per_point"] = round((after - before) / max(1, len(latencies)), 1)
        return result

    label, dy, dx, bands = next(w for w in WORKLOADS if w[0] == payload["phase"])
    y0, x0 = payload["region_origin"]
    before = _net_bytes_received()
    started = time.monotonic()
    block = embeddings[time_index, y0 : y0 + dy, x0 : x0 + dx, 0:bands]
    wall = time.monotonic() - started
    after = _net_bytes_received()
    if before is not None and after is not None:
        result["wire_bytes"] = after - before
    # A read of elided chunks issues no requests, so a partly-elided window reports as high
    # throughput for having done less work. This fraction is the guard; it goes into the table
    # rather than the JSON, and the driver refuses a window that is mostly fill.
    nonfill = float(np.count_nonzero(block) / block.size)
    result.update(
        workload=label,
        wall_s=round(wall, 3),
        elements=int(block.size),
        # The scoping harness's definition: decompressed elements per second, not a wire rate.
        throughput_mbps=round((block.size / 1e6) / wall, 1) if wall > 0 else 0.0,
        # NOT a coverage figure: int8 zero is both the fill value and a legitimate band value, so a
        # fully written block lands a little under 1.0 rather than exactly at it.
        nonfill_fraction=round(nonfill, 4),
    )
    return result


# ── the parent: choose the addresses, then drive cold and warm arms ──────────


#: A region window must be FULLY non-fill: a read of elided chunks issues no requests, and so
#: reports as throughput for having done less work. **One, not a threshold, because every workload
#: reads from the same origin** — a 90%-live window can have its north-west corner elided, and
#: `patch` reads 100 px of that corner while `tile` reads 1000, so a partial threshold leaves the
#: small workloads measuring fill. It costs nothing: of 40 candidate blocks in 33N/2025, all 40
#: qualified.
MIN_NONFILL = 1.0


def _contiguous_live_block(
    group: zarr.Group, time_index: int, shards: frozenset[tuple[int, int]], span: int
) -> tuple[tuple[int, int], float] | None:
    """Origin of a ``span x span`` block of live shards that is actually full, plus its fill fraction.

    **A live shard is not a full shard**: `live_shards` reports one initialized inner chunk, and
    along a coastline the rest can be elided ocean — so contiguous live coordinates are necessary
    and not sufficient. Each candidate is sampled first, one strided `scales` value per inner chunk.
    None means no candidate reached :data:`MIN_NONFILL`, a real answer for a zone with no solid
    block of that size.
    """
    needed = span // SHARD_PX + (1 if span % SHARD_PX else 0)
    scales = cast("zarr.Array", group["scales"])
    best: tuple[tuple[int, int], float] | None = None
    for shard_y, shard_x in sorted(shards):
        if not all((shard_y + dy, shard_x + dx) in shards for dy in range(needed) for dx in range(needed)):
            continue
        y0, x0 = shard_y * SHARD_PX, shard_x * SHARD_PX
        if y0 + span > scales.shape[1] or x0 + span > scales.shape[2]:
            continue
        probe = np.asarray(scales[time_index, y0 : y0 + span : INNER_PX, x0 : x0 + span : INNER_PX])
        fraction = float(np.isfinite(probe).mean())
        if fraction >= MIN_NONFILL:
            return (y0, x0), fraction
        if best is None or fraction > best[1]:
            best = ((y0, x0), fraction)
    if best is not None:
        logging.getLogger(__name__).warning(
            "no %d px block reached %.0f%% non-fill; the fullest found was %.0f%% at %s",
            span,
            MIN_NONFILL * 100,
            best[1] * 100,
            best[0],
        )
    return None


def _run_cold(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one phase in a fresh interpreter so no cache or connection pool is warm."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps({**payload, "repeats": 1}),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "phase": payload["phase"],
            "concurrency": payload["concurrency"],
            "error": (completed.stderr or "").strip()[-800:],
        }
    return json.loads(completed.stdout)[0]


def main(argv: list[str] | None = None) -> int:
    """Measure the store's read performance and print a table; 0 unless a phase failed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--region", default=DEFAULT_REGION, help="the bucket's region")
    parser.add_argument("--zone", default="33N", help="zone group to read")
    parser.add_argument("--year", type=int, default=2025, help="calendar year to read")
    parser.add_argument("--points", type=int, default=200, help="point-vector probes per arm")
    parser.add_argument("--seed", type=int, default=0, help="pixel-sample seed")
    parser.add_argument(
        "--workloads",
        default="all",
        help=(
            "comma-separated region workloads to run, or 'all', or 'none'. "
            f"Available: {','.join(w[0] for w in WORKLOADS)}. The open and point phases always run."
        ),
    )
    parser.add_argument(
        "--anonymous",
        action="store_true",
        help="read with no credentials (the published bucket grants public reads)",
    )
    parser.add_argument("--json", dest="json_out", help="write the full report here")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        request = json.loads(sys.stdin.read())
        print(json.dumps(run_phase(request, repeats=int(request.get("repeats", 1)))))
        return 0

    host = _host_facts()
    print(f"host:  {host}")
    print(f"store: {args.uri} ({args.region})  zone {args.zone} year {args.year}")

    repo = open_global_repo(args.uri, region=args.region, anonymous=args.anonymous)
    session = repo.readonly_session(branch="main")
    group = zarr.open_group(session.store, mode="r")[args.zone]
    years = published_store.calendar_years(group)
    if args.year not in years:
        parser.error(f"{args.zone} has no {args.year} slot; its time axis is {years}")
    time_index = years.index(args.year)

    started = time.monotonic()
    coverage = published_store.live_shards(session, args.zone)
    shards = coverage.get(time_index, frozenset())
    print(f"live shards in {args.zone}/{args.year}: {len(shards):,} (enumerated in {time.monotonic() - started:.1f}s)")
    if not shards:
        parser.error(f"{args.zone}/{args.year} has no live shards; nothing to benchmark")

    # ESCALATING oversample until the count is met. On a sparse coastal zone-year most candidates
    # hold no data, and raising `--points` does not help because the pool scales with it while the
    # success ratio does not — so percentiles from a handful would sit beside a scoping figure
    # taken over a thousand.
    points: list[tuple[int, int]] = []
    for factor in (4, 16, 64):
        points = published_store.sample_live_pixels(
            group, time_index, shards, args.points, seed=args.seed, oversample=factor
        )
        if len(points) >= args.points:
            break
        print(f"  {len(points)} of {args.points} probes held data at oversample {factor}; escalating")
    print(f"probe pixels confirmed to hold embeddings: {len(points)} of {args.points} requested")
    if len(points) < args.points:
        parser.error(
            f"only {len(points)} of {args.points} probes landed on data in {args.zone}/{args.year} even at "
            "64x oversampling — its live shards are mostly elided. Lower --points, or pick a denser "
            "zone-year; a short sample would not be comparable to the scoping workload."
        )

    if args.workloads == "all":
        wanted = [w[0] for w in WORKLOADS]
    elif args.workloads == "none":
        wanted = []
    else:
        wanted = args.workloads.split(",")
        if unknown := sorted(set(wanted) - {w[0] for w in WORKLOADS}):
            parser.error(f"unknown workload(s) {unknown}; available: {[w[0] for w in WORKLOADS]}")

    region_origin: tuple[int, int] | None = None
    block_nonfill: float | None = None
    if wanted:
        # Sized for the LARGEST requested workload, so a run that skips `bulk` is not refused for
        # want of a block only `bulk` needs — and over BOTH extents, so a workload wider than it is
        # tall is not handed a block big enough on one axis only.
        span = max(max(w[1], w[2]) for w in WORKLOADS if w[0] in wanted)
        found = _contiguous_live_block(group, time_index, shards, span)
        if found is None:
            print(f"no {span} px block in {args.zone}/{args.year} is {MIN_NONFILL:.0%} non-fill; skipping region reads")
            wanted = []
        else:
            region_origin, block_nonfill = found
            print(f"region window {region_origin}, {block_nonfill:.1%} of its inner chunks hold data")

    base = {
        "uri": args.uri,
        "region": args.region,
        "anonymous": args.anonymous,
        "zone": args.zone,
        "time_index": time_index,
        "points": points,
        "region_origin": region_origin,
    }
    phases = ["open", "point", *wanted]

    results: list[dict[str, Any]] = []
    for concurrency in CONCURRENCIES:
        for phase in phases:
            payload = {**base, "phase": phase, "concurrency": concurrency}
            cold = {**_run_cold(payload), "cache": "cold"}
            arms = [cold]
            # Warm: a SECOND pass through the same opened group, inside one call, so whatever the
            # first pass cached is still there. The open phases have no second pass to take.
            if phase != "open":
                try:
                    passes = run_phase(payload, repeats=2)
                    arms.append({**passes[-1], "cache": "warm"})
                except Exception as exc:
                    arms.append({"phase": phase, "concurrency": concurrency, "cache": "warm", "error": str(exc)})
            results += arms
            warm_note = f"  warm={_one_line(arms[-1])}" if len(arms) > 1 else ""
            print(f"  c={concurrency:<4} {phase:<17} cold={_one_line(cold)}{warm_note}")

    print()
    _print_table(results)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "host": host,
                    "store": {"uri": args.uri, "region": args.region, "zone": args.zone, "year": args.year},
                    "snapshot_id": session.snapshot_id,
                    "live_shards": len(shards),
                    "probe_pixels": len(points),
                    "region_origin": region_origin,
                    "region_window_nonfill": block_nonfill,
                    "concurrencies": list(CONCURRENCIES),
                    "results": results,
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}")

    return 1 if any("error" in r for r in results) else 0


def _one_line(result: dict[str, Any]) -> str:
    """One phase's headline figure, for the progress line."""
    if "error" in result:
        return f"FAILED {result['error'][:60]}"
    if result["phase"] == "open":
        return (
            f"{result['total_open_s'] * 1e3:.0f} ms to first read"
            f" (repo {result['repository_open_s'] * 1e3:.0f}"
            f" + root {result['root_group_open_s'] * 1e3:.0f}"
            f" + zone {result['zone_group_open_s'] * 1e3:.0f})"
        )
    if result["phase"] == "point":
        wire = result.get("wire_bytes_per_point")
        wire_note = f", {wire / 1e6:.2f} MB/point" if wire else ""
        return f"p50 {result['p50_ms']:.0f} / p95 {result['p95_ms']:.0f} ms{wire_note}"
    return f"{result['throughput_mbps']:.0f} MB/s in {result['wall_s']:.1f}s"


def _print_table(results: list[dict[str, Any]]) -> None:
    """The whole sweep as one table, cold and warm side by side."""
    # The `nonfill` column was printed per row with no header, so the region rows' guard fraction
    # landed under a blank heading.
    print(
        f"{'phase':<17} {'conc':>5} {'cache':<5} {'p50 ms':>8} {'p95 ms':>8} {'MB/s':>8} "
        f"{'MB/point':>9} {'open ms':>8} {'nonfill':>8}"
    )
    for result in results:
        if "error" in result:
            head = f"{result['phase']:<17} {result['concurrency']:>5} {result['cache']:<5}"
            print(f"{head}   FAILED: {result['error'][:60]}")
            continue
        wire = result.get("wire_bytes_per_point")
        print(
            f"{result['phase']:<17} {result['concurrency']:>5} {result['cache']:<5} "
            f"{result.get('p50_ms', ''):>8} {result.get('p95_ms', ''):>8} "
            f"{result.get('throughput_mbps', ''):>8} "
            f"{(round(wire / 1e6, 2) if wire else ''):>9} "
            f"{round(result['total_open_s'] * 1e3):>8} "
            f"{result.get('nonfill_fraction', ''):>8}"
        )


if __name__ == "__main__":
    sys.exit(main())
