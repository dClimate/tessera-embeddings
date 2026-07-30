"""Real per-zone land-tile counts, and the campaign's own cluster planning.

The numbers below are a SNAPSHOT of the production coverage bitmaps — one
``tile_live_2048`` popcount per UTM zone, the exact quantity
:func:`zone_live_tile_count` returns and the campaign balances its Ray clusters
on. Snapshotted rather than read live so the diagnostics run offline and
deterministically; the coverage store is rebuilt rarely and the shape of the
distribution (a handful of huge continental zones, a long tail of islands) is a
property of the planet, not of a particular build.

Provenance: ``s3://global-tessera-inputs-dev/masks/global.icechunk``, built
2026-07-24 from ``s3://tessera-embeddings/v1.1/global_0.1_degree_tiff_all/``,
registry sha256 ``5ea80dd9…c794e``. To refresh after a mask rebuild::

    from concurrent.futures import ThreadPoolExecutor
    from tessera_embeddings.orchestration.runners.zone_fill import zone_live_tile_count
    from tessera_embeddings.storage.zone_grid import ZONES
    mask = "s3://<inputs-bucket>/masks/global.icechunk"
    with ThreadPoolExecutor(max_workers=16) as ex:
        counts = dict(ex.map(lambda z: (z, zone_live_tile_count(mask, z)), sorted(ZONES)))

:func:`plan` runs the REAL campaign mechanics over these counts — the same
``_partition_by_live_tiles`` the driver calls, then the same densest-first sort
the chained fill applies within a cluster — so what it reports is what a
campaign would actually do, not a re-implementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass

from tessera_embeddings.orchestration.prefect.flows.run_global_campaign import _partition_by_live_tiles

#: Live 2048-px tiles per UTM zone. 8 zones are all-ocean (0) and never ingest.
ZONE_TILES: dict[str, int] = {
    "01N": 334,
    "01S": 113,
    "02N": 245,
    "02S": 76,
    "03N": 614,
    "03S": 4,
    "04N": 1278,
    "04S": 33,
    "05N": 1150,
    "05S": 40,
    "06N": 856,
    "06S": 112,
    "07N": 850,
    "07S": 122,
    "08N": 1197,
    "08S": 25,
    "09N": 1743,
    "09S": 10,
    "10N": 3440,
    "10S": 0,
    "11N": 4827,
    "11S": 0,
    "12N": 5747,
    "12S": 8,
    "13N": 6575,
    "13S": 0,
    "14N": 6985,
    "14S": 0,
    "15N": 5748,
    "15S": 59,
    "16N": 5805,
    "16S": 17,
    "17N": 5322,
    "17S": 767,
    "18N": 6051,
    "18S": 3787,
    "19N": 5140,
    "19S": 7951,
    "20N": 3929,
    "20S": 7004,
    "21N": 2329,
    "21S": 6327,
    "22N": 1765,
    "22S": 5403,
    "23N": 1402,
    "23S": 3953,
    "24N": 1034,
    "24S": 2330,
    "25N": 854,
    "25S": 182,
    "26N": 830,
    "26S": 27,
    "27N": 724,
    "27S": 0,
    "28N": 2589,
    "28S": 10,
    "29N": 5514,
    "29S": 6,
    "30N": 7208,
    "30S": 4,
    "31N": 6704,
    "31S": 6,
    "32N": 8092,
    "32S": 386,
    "33N": 8593,
    "33S": 4267,
    "34N": 8734,
    "34S": 5913,
    "35N": 9132,
    "35S": 5757,
    "36N": 8738,
    "36S": 4715,
    "37N": 8731,
    "37S": 2364,
    "38N": 9100,
    "38S": 1266,
    "39N": 6694,
    "39S": 496,
    "40N": 6547,
    "40S": 58,
    "41N": 5583,
    "41S": 6,
    "42N": 6119,
    "42S": 54,
    "43N": 7808,
    "43S": 30,
    "44N": 7907,
    "44S": 0,
    "45N": 6609,
    "45S": 0,
    "46N": 7116,
    "46S": 0,
    "47N": 9030,
    "47S": 239,
    "48N": 8803,
    "48S": 774,
    "49N": 7317,
    "49S": 943,
    "50N": 6917,
    "50S": 2915,
    "51N": 5590,
    "51S": 3590,
    "52N": 4228,
    "52S": 3702,
    "53N": 3269,
    "53S": 4296,
    "54N": 2696,
    "54S": 4952,
    "55N": 1306,
    "55S": 4247,
    "56N": 1130,
    "56S": 1348,
    "57N": 1441,
    "57S": 267,
    "58N": 957,
    "58S": 354,
    "59N": 855,
    "59S": 577,
    "60N": 620,
    "60S": 610,
}

#: Zones with any land — the only ones a campaign dispatches work for.
LIVE_ZONES: list[str] = sorted(z for z, n in ZONE_TILES.items() if n > 0)


@dataclass(frozen=True)
class ClusterPlan:
    """One Ray cluster's assigned UTM zones, in the order it will work them."""

    zones: list[str]
    tiles: list[int]

    @property
    def total(self) -> int:
        return sum(self.tiles)

    @property
    def opener(self) -> int:
        """Tiles in the zone this cluster starts on — what its fleet waits for."""
        return self.tiles[0]


def plan(n_clusters: int, tiles: dict[str, int] | None = None) -> list[ClusterPlan]:
    """How ``n_clusters`` Ray clusters would divide a year, via the real mechanics.

    Uses the driver's own partitioner (so the balancing and the densest-zone
    dealing are the production ones) and then the chained fill's own ordering
    rule: descending by UNCLAMPED tile count. Anything this reports is a property
    of the shipped code, not of the diagnostic.
    """
    tiles = tiles if tiles is not None else ZONE_TILES
    live = sorted(z for z, n in tiles.items() if n > 0)
    # The partitioner reads the coverage store; feed it the snapshot instead.
    import tessera_embeddings.orchestration.prefect.flows.run_global_campaign as campaign

    # The partitioner now weights tiles by their latitude band's observation count
    # (`zone_work_weight`). This diagnostic feeds it RAW TILE COUNTS on purpose: its
    # subject is the LPT dealing and the density ordering, and holding the weights equal
    # to tiles keeps those properties comparable against the counts the snapshot records.
    # The weighting itself is covered by `test_zone_work_weight`.
    real = campaign.zone_work_weight
    campaign.zone_work_weight = lambda _mask, z, **_k: float(tiles[z])
    try:
        groups = _partition_by_live_tiles(live, n_clusters, land_mask_path="<snapshot>")
    finally:
        campaign.zone_work_weight = real
    return [
        ClusterPlan(zones=ordered, tiles=[tiles[z] for z in ordered])
        for ordered in (sorted(g, key=lambda z: tiles[z], reverse=True) for g in groups)
    ]
