r"""Time the S2 mosaic write path with source reads taken out, to compare library versions and settings.

The S2 ingest's `write` phase includes reading the source imagery, whose variance swamps any
library-level difference between two dev runs. This drives the library's own
`zarr_store.write_days_windows` (one Dask compute and one commit per batch, as the ingest does) under
`manifest_split(INGEST_MANIFEST_SPLIT)`, on a `LocalCluster` shaped like an ingest worker (6 threads),
from real mosaic data that is read once and held in worker memory BEFORE timing starts. A batch's time
is then encode, upload, the merge of every fork and the commit, and nothing that reads S2.

Each run writes `--batches` batches of three dates into a fresh store at `out`, cycling the three
source dates, and appends one JSON line per batch to `results`: wall time, the driver's commit and
merge time (the icechunk `Session` methods, wrapped), driver CPU, and worker task time by task prefix.
To compare library versions, run it from one tree under venvs that differ only in the pinned package,
interleaving the arms. Method and results: context_docs/ingest/campaign-ingest-measurements.md §3.22.

Run from the REPOSITORY ROOT, in-region::

    python scripts/diagnostic/ingest_write_bench.py <label> s3://<bucket>/<prefix>/reflectance.zarr results.jsonl \\
        --source s3://<bucket>/<mosaic>/reflectance.zarr --dates 2025-07-10,2025-07-12,2025-07-14

`--fetches N` overrides icechunk's `max_concurrent_manifest_fetches_during_commit` (icechunk >= 2.3);
`--smoke` writes synthetic data into a local `out` on a small cluster, to check the path on a laptop.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import dask.array as da
import icechunk
import numpy as np
import xarray as xr
import zarr
from affine import Affine
from dask.utils import key_split
from distributed import Client, LocalCluster, futures_of, get_task_stream, wait
from odc.geo.geobox import GeoBox

from tessera_embeddings.config.ingest import INGEST_CHUNKS, INGEST_MANIFEST_SPLIT
from tessera_embeddings.storage import zarr_store
from tessera_embeddings.storage.manifest import IngestManifest

PER_BATCH = 3
BANDS = ("blue", "green", "red", "nir", "nir08", "rededge1", "rededge2", "rededge3", "swir16", "swir22")
TIMES: dict[str, list[float]] = defaultdict(list)


def _timed(cls: type, name: str) -> None:
    """Wrap a driver-side icechunk ``Session`` method so each call's wall time lands in ``TIMES``."""
    orig: Callable[..., Any] = getattr(cls, name)

    def wrapper(self: Any, *a: Any, **k: Any) -> Any:  # noqa: ANN401
        t = time.monotonic()
        try:
            return orig(self, *a, **k)
        finally:
            TIMES[name].append(time.monotonic() - t)

    setattr(cls, name, wrapper)


def _source(client: Client, args: argparse.Namespace) -> list[xr.Dataset]:
    """The source dates' square, read once and persisted in worker memory."""
    shape, ch = (1, args.side, args.side), (1, INGEST_CHUNKS["northing"], INGEST_CHUNKS["easting"])
    if args.smoke:
        days = [
            xr.Dataset(
                {
                    **{
                        b: (
                            ("time", "northing", "easting"),
                            da.random.default_rng(i).integers(0, 9000, shape, chunks=ch, dtype=np.uint16),
                        )
                        for b in BANDS
                    },
                    "scl": (
                        ("time", "northing", "easting"),
                        da.random.default_rng(i).integers(0, 11, shape, chunks=ch, dtype=np.uint8),
                    ),
                },
                coords={"time": [np.datetime64("2025-01-01", "ns")]},
            )
            for i in range(PER_BATCH)
        ]
    else:
        ds = zarr_store.open_store(args.source, chunks={}, region=args.region)
        times = [str(t)[:10] for t in ds.time.values]
        y0, x0 = (int(v) for v in args.origin.split(","))
        days = [
            ds.isel(time=[times.index(d)], northing=slice(y0, y0 + args.side), easting=slice(x0, x0 + args.side))
            for d in args.dates.split(",")
        ]
        days = [d.drop_vars([c for c in d.coords if c != "time"]).chunk(INGEST_CHUNKS) for d in days]
    persisted = [client.persist(d) for d in days]
    wait(futures_of(persisted))
    return persisted


def main() -> None:
    """Write ``--batches`` batches into a fresh store; one JSON line per batch."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", help="arm name, recorded on every line")
    ap.add_argument("out", help="the fresh mosaic store to write")
    ap.add_argument("results", help="JSONL file to append to")
    ap.add_argument("--source", help="an S2 mosaic store to read the dates from")
    ap.add_argument("--dates", help=f"{PER_BATCH} comma-separated dates the source holds")
    ap.add_argument("--origin", default="8192,16384", help="northing,easting of the square, chunk-aligned")
    ap.add_argument("--side", type=int, default=3 * INGEST_CHUNKS["northing"], help="square side in pixels")
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--fetches", type=int, default=None, help="manifest fetches per commit (icechunk >= 2.3)")
    ap.add_argument("--smoke", action="store_true", help="synthetic source, small cluster, two batches")
    args = ap.parse_args()
    if args.smoke:
        args.side, args.batches = INGEST_CHUNKS["northing"], 2
    if args.fetches:
        base = zarr_store._default_repo_config

        def with_fetches() -> icechunk.RepositoryConfig:
            cfg = base()
            splitting = cfg.manifest.splitting if cfg.manifest else None
            cfg.manifest = icechunk.ManifestConfig(
                splitting=splitting, max_concurrent_manifest_fetches_during_commit=args.fetches
            )
            return cfg

        zarr_store._default_repo_config = with_fetches
    _timed(icechunk.Session, "commit")
    _timed(icechunk.Session, "merge")

    size = {"n_workers": 2, "threads_per_worker": 2} if args.smoke else {"n_workers": 8, "threads_per_worker": 6}
    client = Client(LocalCluster(**size))
    lib = {"zarr": zarr.__version__, "icechunk": icechunk.__version__, "fetches": args.fetches}
    roi = SimpleNamespace(
        geobox=GeoBox((args.side, args.side), Affine(10.0, 0.0, 200_000.0, 0.0, -10.0, 2_200_000.0), "EPSG:5070"),
        height=args.side,
        width=args.side,
    )
    with zarr_store.manifest_split(INGEST_MANIFEST_SPLIT), Path(args.results).open("a") as fh:
        src = _source(client, args)
        day0 = np.datetime64("2026-01-01", "ns")
        for b in range(args.batches):
            whens = [day0 + np.timedelta64(b * PER_BATCH + i, "D") for i in range(PER_BATCH)]
            days = [(src[i].assign_coords(time=[w]), [(0, args.side, 0, args.side)]) for i, w in enumerate(whens)]
            TIMES.clear()
            c0, t0 = time.process_time(), time.monotonic()
            with get_task_stream(client) as ts:
                zarr_store.write_days_windows(
                    args.out,
                    days,
                    roi=roi,
                    manifest=IngestManifest(roi_manifest_hash="ingest-write-bench"),
                    baselines={str(w)[:10]: 511 for w in whens},
                    tile_id="ingest-write-bench",
                    crs="EPSG:5070",
                    chunks=INGEST_CHUNKS,
                    parallel_windows=True,
                    s3_region=None if args.smoke else args.region,
                )
            wall = time.monotonic() - t0
            by_prefix: dict[str, float] = defaultdict(float)
            for task in ts.data:
                for s in task["startstops"]:
                    if s["action"] == "compute":
                        by_prefix[key_split(task["key"])] += s["stop"] - s["start"]
            row = lib | {
                "label": args.label,
                "batch": b,
                "wall_s": round(wall, 3),
                "driver_cpu_s": round(time.process_time() - c0, 3),
                "commit_s": round(sum(TIMES["commit"]), 3),
                "merge_s": round(sum(TIMES["merge"]), 3),
                "task_s": {k: round(v, 2) for k, v in sorted(by_prefix.items(), key=lambda kv: -kv[1])[:6]},
                "tasks": len(ts.data),
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(json.dumps(row), flush=True)
    client.close()


if __name__ == "__main__":
    main()
