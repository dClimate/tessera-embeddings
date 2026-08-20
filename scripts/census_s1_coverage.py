"""Global OPERA RTC-S1 coverage census on an equal-area-weighted land grid.

Produces the radar half of the observation-count model in
``context_docs/design/campaign-cost-model.md`` section 6: for each point on a
5 deg x 10 deg grid it asks CMR how many OPERA RTC-S1 granules exist per orbit
direction in a given year, which classifies the point as dual-orbit,
single-orbit or uncovered and gives a granule count. Dividing the count by
cos(lat) turns granules-per-box into observations-per-pixel.

Ocean needs no land mask: OPERA is a land product, so a point with zero
granules in both directions is either ocean or genuinely uncovered.

    python scripts/census_s1_coverage.py 2017 2019 2022 2024 2025

Unauthenticated - CMR granule search needs no Earthdata login. Roughly 1,000
requests per year queried; a few minutes each.

Re-run this after any change to the campaign year range, or to check whether
OPERA coverage has expanded again (it was withdrawn from about a fifth of the
land after Sentinel-1B failed in 2021 and largely restored with Sentinel-1C
in 2025).
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

U = "https://cmr.earthdata.nasa.gov/search/granules.json"
S = requests.Session()
S.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(total=5, backoff_factor=1, status_forcelist=(429, 500, 502, 503, 504)), pool_maxsize=16
    ),
)


def hits(lon: float, lat: float, year: int, direction: str | None = None, box: float = 1.0) -> int:
    """Granule count for a box/year, optionally filtered to one orbit direction."""
    p = {
        "short_name": "OPERA_L2_RTC-S1_V1",
        "provider": "ASF",
        "page_size": 0,
        "bounding_box": f"{lon - box},{max(-89, lat - box)},{lon + box},{min(89, lat + box)}",
        "temporal": f"{year}-01-01T00:00:00Z,{year}-12-31T23:59:59Z",
    }
    if direction:
        p["attribute[]"] = f"string,ASCENDING_DESCENDING,{direction}"
    r = S.get(U, params=p, timeout=60)
    r.raise_for_status()
    return int(r.headers.get("CMR-Hits", 0))


LATS = list(range(-56, 79, 5))
LONS = list(range(-175, 180, 10))
GRID = [(lo, la) for la in LATS for lo in LONS]


def classify(args: tuple[float, float, int]) -> tuple[float, float, str, int, int]:
    """Classify one grid point as dual / single / ocean-or-none, with its counts."""
    lon, lat, year = args
    if hits(lon, lat, year) == 0:
        return (lon, lat, "ocean-or-none", 0, 0)
    a = hits(lon, lat, year, "ASCENDING")
    d = hits(lon, lat, year, "DESCENDING")
    return (lon, lat, "dual" if (a and d) else ("single" if (a or d) else "none"), a, d)


def census(year: int) -> list[tuple[float, float, str, int, int]]:
    """Classify every grid point for one year."""
    with ThreadPoolExecutor(max_workers=10) as ex:
        return list(ex.map(classify, [(lo, la, year) for lo, la in GRID]))


out = {}
for year in (int(a) for a in sys.argv[1:]):
    rows = census(year)
    covered = [(lo, la, k, a, d) for lo, la, k, a, d in rows if k in ("dual", "single")]
    tw = sum(math.cos(math.radians(la)) for _, la, _, _, _ in covered)
    dual = sum(math.cos(math.radians(la)) for _, la, k, _, _ in covered if k == "dual")
    out[year] = {"points_covered": len(covered), "dual_fraction": dual / tw if tw else 0, "rows": rows}
    print(f"{year}: {len(covered)} covered grid points, area-weighted DUAL fraction = {dual / tw if tw else 0:.3f}")
payload = {str(k): {"dual_fraction": v["dual_fraction"], "rows": v["rows"]} for k, v in out.items()}
Path("s1_census.json").write_text(json.dumps(payload))
print("wrote s1_census.json")
