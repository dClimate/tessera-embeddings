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
            per item from where its assets live
        band_names_are_asset_keys: whether the names in ``bands`` are the item's asset keys rather
            than common names resolved through the loader's alias table rather than once for the collection
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
    #: makes the producer decision read each item's asset LOCATIONS, which requires
    #: `band_names_are_asset_keys`.
    harmonisation_varies_by_item: bool = False
    #: Whether the names in `bands` are the item's actual asset KEYS, rather than common names the
    #: loader resolves through an alias table. Earth Search keys its assets by these names;
    #: Planetary Computer serves the same imagery under native keys (`B02`, `SCL`). Any check that
    #: looks an asset up BY NAME — producer classification, read-set completeness, locality — is
    #: uninformative where this is False, and reports every copy as incomplete and remote.
    band_names_are_asset_keys: bool = False

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
        refuses_oversized_pages: True where this catalogue answers a request whose response is
            too large with a 5xx rather than a smaller answer. The query layer then handles it
            by asking a DIFFERENT request -- a shorter date window, or fewer items -- and so
            skips retrying it in place, which buys nothing and costs the whole retry backoff.
            Only set it for a provider actually measured to behave this way: a catalogue with
            no such remedy wants its retries, and a 502 there is ordinarily transient.
    """

    name: str
    catalog_url: str
    collections: dict[str, CollectionConfig] = field(default_factory=dict)
    max_page_size: int = 250
    refuses_oversized_pages: bool = False
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
            # set, and `harmonisation_varies_by_item` sends the decision to the assets so it is
            # made per ASSET, from where that asset lives. Leaving the threshold unset exempts the
            # whole collection and is
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
                band_names_are_asset_keys=True,
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
        # BELOW the class default, because this catalogue refuses a request whose RESPONSE
        # would exceed about 6 MB — AWS Lambda's synchronous response limit — and 250 items
        # of `sentinel-2-l2a` is over it. Per provider rather than on the class default,
        # because nothing implicates the other catalogues.
        #
        # Item size is what the cap tracks, not the `limit` value, and what makes an item big is
        # the SHAPE OF THE GROUND it covers rather than the files attached to it. Most items give
        # a rectangle; some trace the edge of where the imagery actually is, which for Sentinel-2
        # is a twelve-detector sawtooth running to thousands of points and 98 KB against an
        # ordinary 0.2 KB. Those are confined to roughly Nov 2018 - Mar 2019, a stretch never
        # reprocessed. Outside it a 100-item page is ~2.2 MB; inside it, at or over the cap.
        #
        # 100 averages 4.6 MB but the largest page SERVED was 5.73 MB, 96% of the cap, so do not
        # reason from the average. Lowering this is deliberately NOT the answer: it taxes every
        # query in every year for six months of a ten-year archive, and `ingest/stac.py` already
        # re-asks a refused page at half the size. Measurements in the campaign record.
        #
        # The same cap is why one page deep in a walk is sometimes refused while every other
        # page is served: item sizes vary, so whether a given hundred clears 6 MB depends on
        # which hundred the cursor and date window select. `ingest/stac.py` answers that by
        # re-cutting the date window, which regroups the items into smaller responses.
        max_page_size=100,
        refuses_oversized_pages=True,
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
