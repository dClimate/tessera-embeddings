"""Global Sentinel-2 usable-observation census on the campaign's own land grid.

The optical counterpart to ``census_s1_coverage.py``, and the instrument behind
``context_docs/inference/minimum-optical-depth.md``. For each point it asks
earth-search how many DISTINCT acquisition dates cover that point in each
campaign year, and how cloudy each of those dates was, giving

    usable optical observations in a year
        = sum over distinct dates of (1 - eo:cloud_cover)

which is the quantity ``s2_obs_count`` holds per pixel and ``OPTICAL_MIN_OBS``
is compared against.

**Distinct dates, not scenes.** Adjacent MGRS tiles overlap by ~10 km, so a point
inside the overlap sees one acquisition twice; counting scenes would double it.
Item geometries on earth-search are true granule footprints (swath-edge granules
carry clipped polygons), so a point query returns only acquisitions that really
cover the point.

Sampling is one point per 1 deg land bin, taken from the land-mask registry
rather than a regular lattice: binning keeps small islands, which a subsample
drops. 1 deg is about the scale an MGRS tile spans and therefore about the scale
on which acquisition dates are actually constant.

    python scripts/scoping/census_s2_coverage.py grid   --registry s3://.../registry.txt
    python scripts/scoping/census_s2_coverage.py census --workers 40 --rate 9

Unauthenticated - earth-search needs no credentials. About 19,000 requests, ~35
minutes at the default rate. RATE IS LOAD-BEARING: earth-search answers a burst
happily and then WAFs the client with a 403 that no per-request retry clears. The
first run of this census took 10,783 of them at ~90 req/s. The ceiling is global
across workers, not inferred from worker count.

Resumable: results append to the output JSONL and a re-run skips finished bins.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tessera_embeddings.storage.time_axis import CAMPAIGN_YEARS

URL = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"
#: One response holds a point's whole nine-year history; paging costs a round trip
#: per point and stac-server emits a `next` link even on a complete page.
PAGE = 5000
DEFAULT_REGISTRY = "s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/registry.txt"
CELL_NAME = re.compile(r"^grid_(-?\d+\.\d+)_(-?\d+\.\d+)\.tiff$")

_local = threading.local()
_lock = threading.Lock()


class Throttle:
    """Global request-rate ceiling shared by every worker."""

    def __init__(self, per_second: float) -> None:
        self._gap = 1.0 / per_second
        self._next = time.monotonic()
        self._lk = threading.Lock()

    def take(self) -> None:
        """Block until this request is allowed under the ceiling."""
        with self._lk:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self._gap
        if wait > 0:
            time.sleep(wait)


def session() -> requests.Session:
    """One retrying session per thread."""
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.mount(
            "https://",
            HTTPAdapter(
                max_retries=Retry(
                    total=8,
                    backoff_factor=4.0,
                    # 403 is earth-search's throttle response, not an auth failure:
                    # the API is unauthenticated, so it can only mean "slow down".
                    status_forcelist=(403, 429, 500, 502, 503, 504),
                    allowed_methods=frozenset({"GET", "POST"}),
                    respect_retry_after_header=True,
                ),
                pool_maxsize=4,
            ),
        )
        _local.s = s
    return s


def build_grid(registry_uri: str, bin_deg: float, out: Path) -> None:
    """Registry land cells -> one representative cell per ``bin_deg`` bin."""
    best: dict[tuple[int, int], tuple[float, float, float]] = {}
    counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    n = 0
    with fsspec.open(registry_uri, "rt") as fh:
        for line in fh:
            name = line.split(None, 1)[0]
            m = CELL_NAME.match(name)
            if m is None:
                raise ValueError(f"unparsed registry entry: {name!r}")
            lon, lat = float(m.group(1)), float(m.group(2))
            n += 1
            key = (int((lon + 180.0) // bin_deg), int((lat + 90.0) // bin_deg))
            counts[key] += 1
            cx = key[0] * bin_deg - 180.0 + bin_deg / 2
            cy = key[1] * bin_deg - 90.0 + bin_deg / 2
            d = (lon - cx) ** 2 + (lat - cy) ** 2
            cur = best.get(key)
            if cur is None or d < cur[2]:
                best[key] = (lon, lat, d)

    points = [
        {"lon": lon, "lat": lat, "bin": list(key), "n_cells": counts[key]}
        for key, (lon, lat, _) in sorted(best.items())
    ]
    out.write_text(json.dumps(points))
    print(f"{n:,} land cells -> {len(points):,} bins of {bin_deg} deg; wrote {out}")


def census_point(pt: dict, throttle: Throttle, years: tuple[int, ...]) -> dict:
    """Distinct-date count and usable-observation estimate per year at one point."""
    lon, lat = pt["lon"], pt["lat"]
    body: dict = {
        "collections": [COLLECTION],
        "bbox": [lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001],
        "datetime": f"{years[0]}-01-01T00:00:00Z/{years[-1]}-12-31T23:59:59Z",
        "limit": PAGE,
        "fields": {
            "include": ["properties.datetime", "properties.eo:cloud_cover"],
            "exclude": ["geometry", "assets", "links", "bbox", "collection", "id", "type"],
        },
    }
    by_date: defaultdict[str, list[float]] = defaultdict(list)
    pages = got = 0
    while True:
        throttle.take()
        r = session().post(URL, json=body, timeout=120)
        r.raise_for_status()
        d = r.json()
        pages += 1
        for f in d.get("features", []):
            p = f["properties"]
            cc = p.get("eo:cloud_cover")
            # A missing cloud figure is treated as fully cloudy rather than dropped:
            # dropping it would silently shorten the year.
            by_date[p["datetime"][:10]].append(100.0 if cc is None else float(cc))
        got += len(d.get("features", []))
        matched = d.get("numberMatched")
        if matched is not None and got >= matched:
            break
        nxt = [ln for ln in d.get("links", []) if ln.get("rel") == "next"]
        if not nxt or not d.get("features") or pages > 12:
            break
        body = nxt[0]["body"]

    out_years: dict[str, dict[str, float]] = {}
    for y in years:
        pre = str(y)
        dates = [ds for ds in by_date if ds.startswith(pre)]
        obs = sum(1.0 - (sum(by_date[ds]) / len(by_date[ds])) / 100.0 for ds in dates)
        out_years[pre] = {"d": len(dates), "obs": round(obs, 3)}

    rec = {"lon": lon, "lat": lat, "bin": pt["bin"], "n_cells": pt["n_cells"], "pages": pages, "years": out_years}

    # A point with NO acquisitions is either genuinely outside every granule footprint
    # (the land mask runs ~11 km into the sea, so some listed cells are offshore) or an
    # indexing artifact: near the antimeridian earth-search stores zone 01/60 granules
    # clipped, so a real location can fall outside its own granule's geometry. One wide
    # re-query tells the two apart, and the consumer must keep them apart rather than
    # calling both "no optical data".
    if not by_date:
        throttle.take()
        r = session().post(
            URL,
            json={
                "collections": [COLLECTION],
                "bbox": [lon - 0.25, lat - 0.25, lon + 0.25, lat + 0.25],
                "datetime": f"{years[-1]}-01-01T00:00:00Z/{years[-1]}-12-31T23:59:59Z",
                "limit": 1,
            },
            timeout=120,
        )
        r.raise_for_status()
        rec["wide_matched"] = r.json().get("numberMatched", 0)
    return rec


def run_census(points_path: Path, out_path: Path, workers: int, rate: float, years: tuple[int, ...]) -> None:
    """Census every unfinished bin, appending as it goes."""
    points = json.loads(points_path.read_text())
    seen: set[tuple[int, ...]] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                seen.add(tuple(json.loads(line)["bin"]))
            except (ValueError, KeyError):  # a torn final line from an interrupted run
                pass
    todo = [p for p in points if tuple(p["bin"]) not in seen]
    print(f"points {len(points):,}  done {len(seen):,}  todo {len(todo):,}  workers {workers}  rate {rate}/s")

    throttle = Throttle(rate)
    state = {"done": 0, "fail": 0}
    t0 = time.monotonic()
    with out_path.open("a") as fh:

        def one(pt: dict) -> None:
            try:
                rec = census_point(pt, throttle, years)
            except Exception as exc:  # record and carry on; a re-run retries the bin
                with _lock:
                    state["fail"] += 1
                    print(f"FAIL {pt['lon']},{pt['lat']}: {type(exc).__name__} {exc}", flush=True)
                return
            with _lock:
                fh.write(json.dumps(rec) + "\n")
                state["done"] += 1
                if state["done"] % 500 == 0:
                    rt = state["done"] / (time.monotonic() - t0)
                    fh.flush()
                    print(
                        f"{state['done']:,}/{len(todo):,}  {rt:.1f} pt/s  fails {state['fail']}  "
                        f"eta {(len(todo) - state['done']) / rt / 60:.0f} min",
                        flush=True,
                    )

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, todo))
    print(f"done {state['done']:,} fails {state['fail']} in {(time.monotonic() - t0) / 60:.1f} min")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the requested subcommand."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grid", help="build the sampling grid from the land-mask registry")
    g.add_argument("--registry", default=DEFAULT_REGISTRY)
    g.add_argument("--bin-deg", type=float, default=1.0)
    g.add_argument("--out", type=Path, default=Path("s2_census_points.json"))

    c = sub.add_parser("census", help="query earth-search for every grid point")
    c.add_argument("--points", type=Path, default=Path("s2_census_points.json"))
    c.add_argument("--out", type=Path, default=Path("s2_census_obs.jsonl"))
    c.add_argument("--workers", type=int, default=40)
    c.add_argument("--rate", type=float, default=9.0, help="global requests/second ceiling")

    a = ap.parse_args(argv)
    if a.cmd == "grid":
        build_grid(a.registry, a.bin_deg, a.out)
    else:
        run_census(a.points, a.out, a.workers, a.rate, CAMPAIGN_YEARS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
