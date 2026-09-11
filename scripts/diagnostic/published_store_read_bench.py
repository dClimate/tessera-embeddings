"""Measure read performance against the published global store, comparably to the scoping runs.

The sizing decisions behind this store were taken on a synthetic benchmark
(``scripts/scoping/scale_tests/t1_read_bench.py`` and ``t8_sharding.py``, recorded in
[ADR 008](../../context_docs/decisions/008-global-store-architecture.md)). This script asks the
same questions of the store that was actually built, so the architecture's read claims can be
checked rather than inherited. The workloads, their extents and the concurrency sweep are copied
from that harness deliberately — a different point count or a different region size would produce
numbers that look comparable and are not.

**Four things this script does because the naive version is wrong:**

* **Probes are placed in pixels that provably hold embeddings.** A read of an absent chunk is
  answered from the manifest without a request, so probes that land on fill are nearly free — and
  a sample mixing those with real reads reports the mixture as latency. Live shards come from the
  store's own chunk enumeration and each candidate pixel is then confirmed against ``scales``
  (:func:`~tessera_embeddings.storage.published_store.sample_live_pixels`).

* **Cold means a fresh process; warm means a second pass through the SAME open.** Icechunk caches
  manifests and pools connections, so every cold phase runs in a subprocess that exits afterwards.
  The warm arm is a second pass inside one `run_phase` call, on the group the first pass opened —
  an earlier version called `run_phase` twice, which re-opened everything and made "warm" a second
  cold reader, throwing away the caches this benchmark exists to observe. The parent chooses the
  pixels once and passes them down, so both arms and both regions read exactly the same addresses.

* **Throughput is reported the way the scoping harness reported it**, as decompressed elements per
  second (``elements / wall``). For the int8 ``embeddings`` one element is one byte, so the figure
  is a logical MB/s and can exceed the host's network bandwidth, because zstd means fewer bytes
  cross the wire than reach the array. It is a decompressed-delivery rate, not a wire rate. The
  band-subset workload counts only the bands asked for, while the reader must fetch and decode the
  whole 128-band inner chunk, so its number understates the work by design — kept for
  comparability, not as a bandwidth claim.

* **Bytes on the wire come from the interface counters**, as in the scoping run, which means any
  other traffic on the host is counted too. Trustworthy on a dedicated instance; noise on a
  laptop. The report says which host it ran on so a reader can judge.

Run from the REPOSITORY ROOT::

    uv run python scripts/diagnostic/published_store_read_bench.py --zone 33N --year 2025
    uv run python scripts/diagnostic/published_store_read_bench.py --points 200 --json bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import icechunk
import numpy as np
import zarr

from tessera_embeddings.config.store_layout import SHARD_PX
from tessera_embeddings.storage import published_store
from tessera_embeddings.storage.global_store import open_global_repo

DEFAULT_URI = "s3://tessera-embeddings/v1.1/dclimate.icechunk"
DEFAULT_REGION = "us-west-2"

#: Concurrency sweep, from the scoping harness (`t1_read_bench.CONCURRENCIES_BENCH`). The scoping
#: figures published in ADR 008 are medians ACROSS this sweep, so reproducing the sweep is what
#: makes a median comparable to a median.
CONCURRENCIES = (10, 64, 128)

#: Region workloads as ``(label, northing extent, easting extent, bands)`` — the extents from
#: `t1_read_bench.WORKLOADS`. `bulk` at 4096 px spans a 2x2 block of 2048-px shards, which is why
#: the benchmark needs a contiguous live block rather than any live shard.
WORKLOADS = (
    ("patch", 100, 100, 128),
    ("tile", 1000, 1000, 128),
    ("band_subset", 512, 512, 8),
    ("bulk", 4096, 4096, 128),
)


def _host_facts() -> dict[str, Any]:
    """Where this ran: EC2 identity when there is one, the machine's own name otherwise.

    Asked of the instance metadata service with a short timeout and IMDSv2's token handshake. A
    benchmark whose report cannot say which region it ran in is not evidence about either region,
    so this is recorded rather than passed in by the operator — an argument can be wrong.
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


def _calendar_years(group: zarr.Group) -> list[int]:
    """The calendar year of each time slot, from the group's own ``time`` coordinate.

    The coordinate is int64 nanoseconds since the epoch, so it has to be VIEWED as
    ``datetime64[ns]`` before being truncated to years — casting the raw integers straight to
    ``datetime64[Y]`` reads each nanosecond count as a year offset and yields years in the
    billions, which then silently fails to match any year a caller asks for. The range check
    turns a future change of units into a loud failure rather than a wrong time index.
    """
    stamps = np.asarray(group["time"][:]).astype("datetime64[ns]")
    years = [int(y) for y in stamps.astype("datetime64[Y]").astype(int) + 1970]
    if not all(1970 <= y <= 2200 for y in years):
        raise ValueError(f"time coordinate did not decode to plausible years: {years[:4]}")
    return years


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


#: Open-path phases, measured in one sweep so they are comparable to each other rather than across
#: runs. ``open`` is the documented recipe; ``open_zone_direct`` opens one zone group by path
#: instead of going through the root, which is worth measuring because the natural guess — that the
#: root open is slow because it enumerates 120 groups — is testable and wrong.
OPEN_PHASES = ("open", "open_zone_direct")


def _reader_config(payload: dict[str, Any]) -> icechunk.RepositoryConfig | None:
    """The store's SAVED config with only the payload's overrides applied, or None to change nothing.

    **Starts from what the store saved, never from a fresh config.** A `RepositoryConfig` passed to
    `Repository.open` replaces the saved one wholesale rather than layering on it, so asking for a
    chunk cache with a fresh config also silently reverts the manifest preload the writer chose —
    which was measured here at 2.4 s of the open path, and would have moved between two arms that
    were supposed to differ only in their cache. Fetching the saved config first and mutating one
    field keeps each arm a one-variable change.
    """
    if not (payload.get("no_preload") or payload.get("chunk_cache_mb")):
        return None
    config = icechunk.Repository.fetch_config(_storage_for(payload))
    if config is None:
        raise RuntimeError("the store saved no repository config; an override here would not be a one-variable change")
    if payload.get("no_preload"):
        config.manifest = icechunk.ManifestConfig(
            preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0),
            splitting=config.manifest.splitting if config.manifest else None,
        )
    if payload.get("chunk_cache_mb"):
        # The saved config sets no caching, so a reader gets icechunk's own default chunk cache. A
        # working set larger than that cache is evicted before it can be revisited, which is the
        # difference between a repeated read of one inner chunk costing a request and costing
        # nothing.
        config.caching = icechunk.CachingConfig(
            num_bytes_chunks=int(payload["chunk_cache_mb"]) * 1024 * 1024,
        )
    return config


def _open_zone(payload: dict[str, Any]) -> tuple[zarr.Group, dict[str, float]]:
    """Open the store to one zone group at the payload's concurrency, timing each step."""
    zarr.config.set({"async.concurrency": payload["concurrency"]})
    phase = payload["phase"]
    timings: dict[str, float] = {}
    started = time.monotonic()
    config = _reader_config(payload)
    if config is not None:
        repo = icechunk.Repository.open(_storage_for(payload), config=config)
    else:
        repo = open_global_repo(payload["uri"], region=payload["region"], anonymous=payload["anonymous"])
    timings["repository_open_s"] = round(time.monotonic() - started, 3)
    started = time.monotonic()
    session = repo.readonly_session(branch="main")
    timings["readonly_session_s"] = round(time.monotonic() - started, 3)
    started = time.monotonic()
    if phase == "open_zone_direct":
        # Straight to the group, never touching the root node.
        timings["root_group_open_s"] = 0.0
        group = zarr.open_group(session.store, path=payload["zone"], mode="r")
    else:
        root = zarr.open_group(session.store, mode="r")
        timings["root_group_open_s"] = round(time.monotonic() - started, 3)
        started = time.monotonic()
        group = root[payload["zone"]]
    _ = group["embeddings"].shape  # force the array metadata, which is what a reader needs
    timings["zone_group_open_s"] = round(time.monotonic() - started, 3)
    return group, timings


def _storage_for(payload: dict[str, Any]) -> icechunk.Storage:
    """Icechunk storage for the payload's URI, for the one phase that needs its own config."""
    bucket, _, prefix = payload["uri"].removeprefix("s3://").partition("/")
    return icechunk.s3_storage(
        bucket=bucket,
        prefix=prefix,
        region=payload["region"],
        anonymous=True if payload["anonymous"] else None,
        from_env=None if payload["anonymous"] else True,
    )


def _net_bytes_received() -> int | None:
    """Interface bytes received so far, or None when psutil is unavailable."""
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.net_io_counters().bytes_recv)


def run_phase(payload: dict[str, Any], repeats: int = 1) -> list[dict[str, Any]]:
    """Open the store once, run one phase ``repeats`` times on that open, and return each result.

    **The repeats share one opened group, and that is the whole point.** An earlier version called
    this function twice for the warm arm, which re-opened the storage, repository, session and
    group each time — so the "warm" figure was a second cold reader, and every cache this
    benchmark is meant to observe was thrown away between the two. A reader who keeps their handle
    is the case worth measuring, and it is the case the scoping harness measured.

    The open phases ignore ``repeats``: their measurement IS the open, and a second one on an
    already-open group has nothing to time. That the open is paid once per handle rather than once
    per read is the useful fact about it.
    """
    group, timings = _open_zone(payload)
    # The sum, not the steps, is what a consumer waits for before their first read. The steps are
    # kept beside it because they say WHERE the wait is — and on this store the zone-group step is
    # nearly free, because opening the root already fetched the snapshot that describes every group.
    timings["total_open_s"] = round(sum(timings.values()), 3)
    base: dict[str, Any] = {"phase": payload["phase"], "concurrency": payload["concurrency"], **timings}
    if payload["phase"] in OPEN_PHASES:
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
    result.update(
        workload=label,
        wall_s=round(wall, 3),
        elements=int(block.size),
        # The scoping harness's definition: decompressed elements per second. See the module
        # docstring on why this is not a wire rate.
        throughput_mbps=round((block.size / 1e6) / wall, 1) if wall > 0 else 0.0,
        # A sanity indicator that the region held data rather than fill, NOT a coverage figure:
        # int8 zero is both the fill value and a legitimate band value, so a fully written block
        # lands a little under 1.0 rather than exactly at it. It is here to make a read that
        # silently landed on ocean obvious, which would show as a fraction near zero.
        nonfill_fraction=round(float(np.count_nonzero(block) / block.size), 4),
    )
    return result


# ── the parent: choose the addresses, then drive cold and warm arms ──────────


def _contiguous_live_block(shards: frozenset[tuple[int, int]], span: int) -> tuple[int, int] | None:
    """Pixel origin of a ``span x span`` shard block whose every shard is live, or None.

    The region workloads must sit inside written data or they measure fill. `bulk` at 4096 px
    needs 2x2 shards, and a zone's live shards follow its coastline, so a block of the right size
    has to be searched for rather than assumed.
    """
    needed = span // SHARD_PX + (1 if span % SHARD_PX else 0)
    for shard_y, shard_x in sorted(shards):
        if all((shard_y + dy, shard_x + dx) in shards for dy in range(needed) for dx in range(needed)):
            return shard_y * SHARD_PX, shard_x * SHARD_PX
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
        "--chunk-cache-mb",
        type=int,
        default=0,
        help=(
            "override the chunk cache size in MiB. The store saves no caching setting, so a reader "
            "gets icechunk's own default; raising it above the working set is what turns a repeated "
            "read of one inner chunk from a request into a cache hit."
        ),
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help=(
            "open with manifest preloading switched off instead of inheriting the writer's saved "
            "setting. Applies to EVERY phase, so a run with and a run without it compare the whole "
            "read path rather than only its first step — which is what says whether the open time "
            "the preload buys back is paid for later in the reads."
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
    preload_note = "manifest preload DISABLED" if args.no_preload else "manifest preload as saved in the store"
    if args.chunk_cache_mb:
        preload_note += f", chunk cache {args.chunk_cache_mb} MiB"
    print(f"store: {args.uri} ({args.region})  zone {args.zone} year {args.year}  [{preload_note}]")

    repo = open_global_repo(args.uri, region=args.region, anonymous=args.anonymous)
    session = repo.readonly_session(branch="main")
    group = zarr.open_group(session.store, mode="r")[args.zone]
    years = _calendar_years(group)
    if args.year not in years:
        parser.error(f"{args.zone} has no {args.year} slot; its time axis is {years}")
    time_index = years.index(args.year)

    started = time.monotonic()
    coverage = published_store.live_shards(session, args.zone)
    shards = coverage.get(time_index, frozenset())
    print(f"live shards in {args.zone}/{args.year}: {len(shards):,} (enumerated in {time.monotonic() - started:.1f}s)")
    if not shards:
        parser.error(f"{args.zone}/{args.year} has no live shards; nothing to benchmark")

    points = published_store.sample_live_pixels(group, time_index, shards, args.points, seed=args.seed)
    print(f"probe pixels confirmed to hold embeddings: {len(points)} of {args.points} requested")
    if not points:
        parser.error("no candidate pixel held data; raise --points or pick another zone-year")

    if args.workloads == "all":
        wanted = [w[0] for w in WORKLOADS]
    elif args.workloads == "none":
        wanted = []
    else:
        wanted = args.workloads.split(",")
        if unknown := sorted(set(wanted) - {w[0] for w in WORKLOADS}):
            parser.error(f"unknown workload(s) {unknown}; available: {[w[0] for w in WORKLOADS]}")

    # Sized for the LARGEST requested workload, so a run that skips `bulk` is not refused for want
    # of a block only `bulk` needs.
    region_origin = None
    if wanted:
        # Both extents, not just the northing one: a workload that is wider than it is tall would
        # otherwise be handed a block big enough on one axis only.
        span = max(max(w[1], w[2]) for w in WORKLOADS if w[0] in wanted)
        region_origin = _contiguous_live_block(shards, span)
        if region_origin is None:
            print(f"no contiguous live block of {span} px in {args.zone}/{args.year}; skipping region workloads")
            wanted = []

    base = {
        "uri": args.uri,
        "region": args.region,
        "anonymous": args.anonymous,
        "no_preload": args.no_preload,
        "chunk_cache_mb": args.chunk_cache_mb,
        "zone": args.zone,
        "time_index": time_index,
        "points": points,
        "region_origin": region_origin,
    }
    phases = [*OPEN_PHASES, "point", *wanted]

    results: list[dict[str, Any]] = []
    for concurrency in CONCURRENCIES:
        for phase in phases:
            payload = {**base, "phase": phase, "concurrency": concurrency}
            cold = {**_run_cold(payload), "cache": "cold"}
            arms = [cold]
            # Warm: a SECOND pass through the same opened group, inside one call, so whatever the
            # first pass cached is still there. The open phases have no second pass to take —
            # their measurement IS the open, which a reader pays once per handle.
            if phase not in OPEN_PHASES:
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
                    "no_preload": args.no_preload,
                    "chunk_cache_mb": args.chunk_cache_mb,
                    "snapshot_id": session.snapshot_id,
                    "live_shards": len(shards),
                    "probe_pixels": len(points),
                    "region_origin": region_origin,
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
    if result["phase"] in OPEN_PHASES:
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
    print(
        f"{'phase':<17} {'conc':>5} {'cache':<5} {'p50 ms':>8} {'p95 ms':>8} {'MB/s':>8} {'MB/point':>9} {'open ms':>8}"
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
            f"{round(result['total_open_s'] * 1e3):>8}"
        )


if __name__ == "__main__":
    sys.exit(main())
