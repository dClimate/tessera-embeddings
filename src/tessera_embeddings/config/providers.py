"""STAC provider and collection configuration.

This module defines the dataclasses and provider registry for querying
satellite data from various STAC catalogs.
"""

from dataclasses import dataclass, field

from .satellites import (
    LANDSAT_C2_BANDS,
    S1_BASELINE_OFFSET,
    S1_BASELINE_THRESHOLD,
    S1_GRD_BANDS,
    S1_OPERA_BANDS,
    S2_BASELINE_OFFSET,
    S2_BASELINE_THRESHOLD,
    S2_L1C_BANDS,
    S2_L2A_BANDS,
)


@dataclass(frozen=True)
class CollectionConfig:
    """Configuration for a specific STAC collection.

    Attributes:
        collection_id: STAC collection identifier (e.g., "sentinel-2-l2a")
        bands: List of band names to load
        resolution: Target resolution in meters
        baseline_threshold: Min baseline requiring correction (None = none)
        baseline_offset: Pixel offset for baseline correction (negative)
        tile_id_property: STAC property containing tile/grid ID
        tile_id_prefix: Prefix for tile ID in queries (e.g., "MGRS-")
        has_scl: whether this collection provides an SCL layer to use for cloudmask
        harmonisation_varies_by_item: whether items in this collection can disagree about
            whether the BOA offset has already been subtracted, so the decision has to be made
            per item from where its assets live rather than once for the collection
    """

    collection_id: str
    bands: list[str]
    resolution: int
    baseline_threshold: int | None = None
    baseline_offset: int = 0
    tile_id_property: str | None = "grid:code"
    tile_id_prefix: str = ""
    has_scl: bool = False
    #: Off by default, because a collection served by one producer has one answer. Turning it on
    #: makes the producer decision read each item's asset LOCATIONS, which only works where the
    #: assets are keyed by the names in `bands` — Planetary Computer serves the same imagery under
    #: native keys (`B02`, `SCL`), where that read finds nothing and could refuse every date.
    harmonisation_varies_by_item: bool = False

    @property
    def requires_baseline_correction(self) -> bool:
        """Whether this collection requires baseline correction."""
        return self.baseline_threshold is not None


@dataclass(frozen=True)
class STACProvider:
    """Configuration for a STAC catalog provider.

    Attributes:
        name: Human-readable provider name
        catalog_url: STAC API endpoint URL
        collections: Mapping of collection aliases to configurations
    """

    name: str
    catalog_url: str
    collections: dict[str, CollectionConfig] = field(default_factory=dict)
    max_page_size: int = 250
    """STAC search page size (the ``limit`` per page request).

    Used only by providers queried through ``client.search()`` (Earth Search,
    Planetary Computer). The OPERA ``cmr-asf`` path bypasses CMR-STAC search
    entirely and queries the native CMR granule API instead (see
    ``opera_query.make_s1_item_provider``), so this value does not apply there.
    Raising it for CMR-STAC made the 500s worse, not better — see
    context_docs/decisions/009-native-cmr-granule-query.md."""


# =============================================================================
# Pre-configured Providers
# =============================================================================

PROVIDERS: dict[str, STACProvider] = {
    "earth-search": STACProvider(
        name="Earth Search (Element 84)",
        catalog_url="https://earth-search.aws.element84.com/v1",
        collections={
            # Earth Search harmonises the BOA offset (subtracts 1000 from post-baseline
            # 04.00 data) in ITS OWN COGs, but this collection also indexes items whose
            # assets point at ESA's originals, which carry the offset. So the threshold is
            # set and `harmonisation_varies_by_item` makes the exemption per item, from where
            # the assets live. Leaving the threshold unset exempts the whole collection and is
            # only correct while every item is a harmonised COG.
            # The raster:bands offset=-0.1 and earthsearch:boa_offset_applied metadata are
            # unreliable (see sertit/eoreader#120), so the asset location is the signal.
            "sentinel-2-l2a": CollectionConfig(
                collection_id="sentinel-2-l2a",
                bands=S2_L2A_BANDS,
                resolution=10,
                baseline_threshold=S2_BASELINE_THRESHOLD,
                baseline_offset=S2_BASELINE_OFFSET,
                tile_id_property="grid:code",
                tile_id_prefix="MGRS-",
                has_scl=True,
                harmonisation_varies_by_item=True,
            ),
            "sentinel-2-l1c": CollectionConfig(
                collection_id="sentinel-2-l1c",
                bands=S2_L1C_BANDS,
                resolution=10,
                tile_id_property="grid:code",
                tile_id_prefix="MGRS-",
            ),
            "sentinel-1-grd": CollectionConfig(
                collection_id="sentinel-1-grd",
                bands=S1_GRD_BANDS,
                resolution=10,
                baseline_threshold=S1_BASELINE_THRESHOLD,
                baseline_offset=S1_BASELINE_OFFSET,
                tile_id_property="sat:relative_orbit",
                tile_id_prefix="",
            ),
            "landsat-c2-l2": CollectionConfig(
                collection_id="landsat-c2-l2",
                bands=LANDSAT_C2_BANDS,
                resolution=30,
                tile_id_property="landsat:wrs_path",
                tile_id_prefix="",
            ),
        },
        # BELOW the class default, and measured rather than guessed. This catalogue's search
        # returns 502 for some (area, date-window) pairs at 250 items per page while
        # answering the SAME query at 100 in under two seconds, and answering an adjacent
        # year at 250 without trouble — so it is a page-size sensitivity in the service, not
        # our load and not missing data. It also defeats the nine attempts of backoff
        # underneath us, so the retry ladder cannot absorb it and only a smaller page can.
        # Roughly twice the page requests for a query that answers at all is a good trade:
        # search calls are a rounding error against the per-scene COG reads that dominate an
        # ingest. Set per provider rather than on the class default, because nothing implicates
        # the other catalogues. Reproduction: `scripts/e84_search_502_probe.py` (yield-embeddings).
        max_page_size=100,
    ),
    "cmr-asf": STACProvider(
        name="NASA CMR-STAC (ASF)",
        catalog_url="https://cmr.earthdata.nasa.gov/stac/ASF",
        collections={
            "opera-rtc-s1": CollectionConfig(
                collection_id="OPERA_L2_RTC-S1_V1_1",
                bands=S1_OPERA_BANDS,
                resolution=30,
                tile_id_property=None,
                tile_id_prefix="",
            ),
        },
    ),
    "planetary-computer": STACProvider(
        name="Microsoft Planetary Computer",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collections={
            # Planetary Computer serves ESA's values unharmonised, and serves them ALL that
            # way — so the answer is the collection's, not each item's, and
            # `harmonisation_varies_by_item` stays off. It also keys its assets natively
            # (`B02`, `SCL`) and relies on the loader resolving the common names in `bands`,
            # so a per-item read of asset locations would find nothing here and refuse.
            "sentinel-2-l2a": CollectionConfig(
                collection_id="sentinel-2-l2a",
                bands=S2_L2A_BANDS,
                resolution=10,
                baseline_threshold=S2_BASELINE_THRESHOLD,
                baseline_offset=S2_BASELINE_OFFSET,
                tile_id_property="s2:mgrs_tile",
                tile_id_prefix="",
            ),
            "sentinel-1-grd": CollectionConfig(
                collection_id="sentinel-1-rtc",
                bands=S1_GRD_BANDS,
                resolution=10,
                baseline_threshold=S1_BASELINE_THRESHOLD,
                baseline_offset=S1_BASELINE_OFFSET,
                tile_id_property="sat:relative_orbit",
                tile_id_prefix="",
            ),
            "landsat-c2-l2": CollectionConfig(
                collection_id="landsat-c2-l2",
                bands=LANDSAT_C2_BANDS,
                resolution=30,
                tile_id_property="landsat:wrs_path",
                tile_id_prefix="",
            ),
        },
    ),
}
