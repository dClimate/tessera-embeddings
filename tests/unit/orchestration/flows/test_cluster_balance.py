"""How the world's UTM zones actually divide across N Ray clusters.

The chained campaign's GPU economics rest on three properties of the split, and
all three are consequences of code that never states them outright — the LPT
assignment in ``_partition_by_live_tiles`` and the densest-first sort in the
chained fill. Asserted here against REAL coverage counts (``zone_density``)
rather than synthetic weights, because the thing being checked is how the split
behaves on the planet's actual land distribution: a handful of huge continental
zones and a long tail of islands.

1. **Every cluster opens on a dense zone.** A cluster asks for its fleet as soon
   as its FIRST zone has ingested, so that zone has to be big enough to keep the
   fleet busy while the rest of its window ingests behind it.
2. **Every cluster runs dense to sparse.** The autoscaled fleet then only ever
   shrinks, and the cheap island zones land at the tail.
3. **Clusters finish together.** Total work per cluster is near-even, so no
   cluster is still grinding after the others have released their fleets.

To SEE a split rather than assert on it::

    TE_CLUSTERS=12 uv run pytest tests/unit/orchestration/flows/test_cluster_balance.py -k report -s

``TE_CLUSTERS`` accepts a comma-separated list (``4,8,16``) and defaults to the
campaign's own ``max_parallel_clusters``.
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

import tessera_embeddings.orchestration.prefect.flows.run_global_campaign as campaign
import tessera_embeddings.orchestration.runners.zone_fill as zone_fill
import tessera_embeddings.storage.zone_grid as zone_grid
from tests.unit.zone_density import LIVE_ZONES, ZONE_TILES, plan

#: The shipped default, read from the flow so this file cannot drift from it.
DEFAULT_CLUSTERS: int = inspect.signature(campaign.run_global_campaign.fn).parameters["max_parallel_clusters"].default

#: Spanning the plausible operating range: one cluster for the year, the shipped
#: 8, and well past it. 120 is the degenerate cluster-per-zone end.
CLUSTER_COUNTS = [1, 2, 4, 6, 8, 12, 16, 24, 40, 120]


@pytest.mark.parametrize("n", CLUSTER_COUNTS)
def test_every_cluster_opens_on_one_of_the_densest_zones(n: int) -> None:
    """Property 1, on real counts: the N densest zones go one per cluster.

    This is what lets a cluster request GPUs after a single zone instead of
    waiting out its whole ingest. It falls out of LPT starting every cluster at
    zero rather than being an explicit step, so it is worth pinning.
    """
    clusters = plan(n)
    densest = sorted(ZONE_TILES.values(), reverse=True)[: len(clusters)]
    assert sorted((c.opener for c in clusters), reverse=True) == densest


@pytest.mark.parametrize("n", CLUSTER_COUNTS)
def test_every_cluster_runs_dense_to_sparse(n: int) -> None:
    """Property 2: within a cluster, tile counts never increase."""
    for c in plan(n):
        assert c.tiles == sorted(c.tiles, reverse=True), c.zones


@pytest.mark.parametrize("n", [1, 2, 4, 6, 8, 12, 16])
def test_clusters_finish_together(n: int) -> None:
    """Property 3: total work per cluster is within 5% of the heaviest.

    Only checked where clusters hold several zones. Past ~24 the split is
    dominated by the fact that the single largest zone cannot be divided, which
    is a fact about the planet rather than about the algorithm — see
    ``test_beyond_a_point_the_biggest_zone_sets_the_floor``.
    """
    totals = [c.total for c in plan(n)]
    assert max(totals) / min(totals) - 1 < 0.05, totals


def test_the_default_is_in_the_regime_where_balancing_works() -> None:
    """8 clusters is comfortably below where the largest zone starts to dominate."""
    totals = [c.total for c in plan(DEFAULT_CLUSTERS)]
    assert max(totals) / min(totals) - 1 < 0.01, totals


def test_beyond_a_point_the_biggest_zone_sets_the_floor() -> None:
    """The wall any zone-granular split hits, stated so it is not rediscovered.

    A cluster cannot be given a fraction of a zone, so once the perfectly even
    share drops below the largest zone, that zone alone is the critical path and
    adding clusters stops shortening the year. Real numbers: the biggest zone is
    ~2.5% of all land, so the knee is somewhere near 40 clusters.
    """
    biggest = max(ZONE_TILES.values())
    total = sum(ZONE_TILES.values())
    knee = total / biggest
    assert 30 < knee < 50, knee
    # Past the knee, the heaviest cluster IS the biggest zone and stops shrinking.
    assert max(c.total for c in plan(int(knee) + 20)) == biggest


def test_all_ocean_zones_are_never_assigned() -> None:
    """They produce no mosaic and need no GPU; carrying them would skew the split."""
    assigned = {z for c in plan(DEFAULT_CLUSTERS) for z in c.zones}
    assert assigned == set(LIVE_ZONES)
    assert len(ZONE_TILES) - len(LIVE_ZONES) == 8  # all-ocean zones in the snapshot


def test_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the split for each count in ``TE_CLUSTERS`` (see the module docstring).

    Runs as an ordinary test so it cannot rot: it asserts the same invariants it
    prints. Use ``-s`` to actually see the tables.
    """
    counts = [int(x) for x in os.environ.get("TE_CLUSTERS", str(DEFAULT_CLUSTERS)).split(",")]
    total = sum(ZONE_TILES.values())
    with capsys.disabled():
        print(f"\n{len(LIVE_ZONES)} live UTM zones, {total:,} land tiles\n")
        for n in counts:
            clusters = plan(n)
            print(f"=== {len(clusters)} cluster(s) " + "=" * 52)
            for i, c in enumerate(clusters):
                head = " ".join(f"{z}:{t}" for z, t in list(zip(c.zones, c.tiles, strict=True))[:5])
                tail = " …" if len(c.zones) > 5 else ""
                print(f"  {i:>3} | {len(c.zones):>3} zones | {c.total:>7,} tiles | {head}{tail}")
            totals = [c.total for c in clusters]
            print(
                f"      totals: min={min(totals):,} max={max(totals):,} "
                f"spread={max(totals) / min(totals) - 1:.1%} | "
                f"openers {min(c.opener for c in clusters):,} to {max(c.opener for c in clusters):,}\n"
            )
            assert (
                sorted((c.opener for c in clusters), reverse=True)
                == sorted(ZONE_TILES.values(), reverse=True)[: len(clusters)]
            )


# --- the work weight itself -----------------------------------------------------------
#
# `test_clusters_finish_together` above feeds the partitioner raw tile counts, because its
# subject is the LPT dealing. These tests cover the other half: that the weight the
# partitioner actually uses reflects WORK rather than AREA.


def _fake_coverage(tile_live: np.ndarray) -> object:
    """The one thing `zone_work_weight` reads: a group with a `tile_live_2048` array."""
    return {"tile_live_2048": tile_live}


def test_zone_work_weight_prices_a_boreal_tile_above_an_equatorial_one(monkeypatch) -> None:
    """The defect this fixes: identical tile COUNTS, materially different work.

    Two synthetic zones with the same number of live tiles, one packed at the top of a
    northern zone (boreal) and one at the bottom (equatorial). Balancing on counts calls
    these equal; they are not, because a boreal pixel carries about twice the
    observations and therefore about twice the tokens.
    """
    spec = zone_grid.zone("35N")
    rows = spec.height // 2048
    boreal, equatorial = np.zeros((rows, 4), bool), np.zeros((rows, 4), bool)
    boreal[:10] = True  # row 0 is the TOP = max northing = highest latitude
    equatorial[-10:] = True
    assert boreal.sum() == equatorial.sum()

    weights = {}
    for name, bitmap in (("boreal", boreal), ("equatorial", equatorial)):
        monkeypatch.setattr(zone_fill, "open_store_as_zarr_group", lambda *_a, _b=bitmap, **_k: _fake_coverage(_b))
        weights[name] = zone_fill.zone_work_weight("<mask>", "35N")

    assert weights["boreal"] > weights["equatorial"], weights
    # 208 tok/px at the top band against 120 at the equatorial one.
    assert weights["boreal"] / weights["equatorial"] == pytest.approx(208 / 120, rel=0.01)


def test_zone_work_weight_reduces_to_tokens_times_tiles(monkeypatch) -> None:
    """Sanity on the units: the weight is tiles x tokens-per-px, not a rescaled count."""
    spec = zone_grid.zone("35N")
    rows = spec.height // 2048
    bitmap = np.zeros((rows, 7), bool)
    bitmap[0] = True  # one whole row at the very top: 7 tiles, all in the 60-84 band
    monkeypatch.setattr(zone_fill, "open_store_as_zarr_group", lambda *_a, **_k: _fake_coverage(bitmap))
    assert zone_fill.zone_work_weight("<mask>", "35N") == pytest.approx(7 * 208)


def test_the_band_table_is_ordered_and_covers_every_zone_row() -> None:
    """No zone row may fall outside the table, in either hemisphere.

    A row that fell through would silently take the table's fallback value, which is the
    kind of default that goes unnoticed until a cluster finishes hours late.
    """
    lowers = [lo for lo, _ in zone_fill.TOKENS_PER_PX_BY_BAND]
    assert lowers == sorted(lowers, reverse=True), "table is written top-down; keep it that way"
    for name in ("35N", "20S"):
        spec = zone_grid.zone(name)
        lat = zone_grid.tile_row_latitudes(spec, spec.height // 2048)
        assert lat.min() >= min(lowers), (name, lat.min())
        assert lat.max() <= 84.0, (name, lat.max())


def test_southern_zones_get_southern_latitudes() -> None:
    """The 10,000,000 m false northing, which is the easy thing to get backwards.

    Without the shift a southern zone reads as +10 to +90 degrees and would be priced as
    boreal — the most expensive band — inverting the very imbalance this fixes.
    """
    south = zone_grid.zone("20S")
    lat = zone_grid.tile_row_latitudes(south, south.height // 2048)
    assert lat.max() <= 0.1, lat.max()
    assert -80.5 <= lat.min() <= -79.0, lat.min()
    north = zone_grid.zone("20N")
    lat_n = zone_grid.tile_row_latitudes(north, north.height // 2048)
    assert lat_n.min() >= -0.1 and 83.0 <= lat_n.max() <= 84.0
