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

    TE_CLUSTERS=12 uv run pytest tests/unit/test_cluster_balance.py -k report -s

``TE_CLUSTERS`` accepts a comma-separated list (``4,8,16``) and defaults to the
campaign's own ``max_parallel_clusters``.
"""

from __future__ import annotations

import inspect
import os

import pytest

import tessera_embeddings.orchestration.prefect.flows.run_global_campaign as campaign
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
