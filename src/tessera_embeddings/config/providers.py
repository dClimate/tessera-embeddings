"""Dataclasses and the provider registry for querying satellite data from STAC catalogs."""

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
        harmonisation_varies_by_item: whether items here can disagree about whether the BOA
            offset was already subtracted, forcing a per-item decision from asset locations
        band_names_are_asset_keys: whether the names in ``bands`` are the item's asset keys
            rather than common names the loader resolves through an alias table
    """

    collection_id: str
    bands: list[str]
    resolution: int
    baseline_threshold: int | None = None
    baseline_offset: int = 0
    tile_id_property: str | None = "grid:code"
    tile_id_prefix: str = ""
    has_scl: bool = False
    #: Off by default: a collection served by one producer has one answer. On, the producer
    #: decision reads each item's asset LOCATIONS, which requires `band_names_are_asset_keys`.
    harmonisation_varies_by_item: bool = False
    #: Earth Search keys its assets by the names in `bands`; Planetary Computer serves the same
    #: imagery under native keys (`B02`, `SCL`). Any check that looks an asset up BY NAME —
    #: producer classification, read-set completeness, locality — is uninformative where this is
    #: False, and reports every copy as incomplete and remote.
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
        refuses_oversized_pages: True where an over-large response comes back as a 5xx rather
            than a smaller answer. The query layer then re-asks a DIFFERENT request (shorter
            date window, fewer items) instead of retrying in place, which buys nothing and
            costs the whole backoff. Set only where measured: elsewhere a 502 is transient and
            the retries are wanted.
        throttles_with_forbidden: True where a 403 means "not right now" rather than "never",
            so waiting is the remedy and the query layer applies its 429 backoff ladder. Set
            only for a public, unauthenticated catalogue measured to behave this way: elsewhere
            a 403 is a verdict about who is asking, which patience cannot change.
    """

    name: str
    catalog_url: str
    collections: dict[str, CollectionConfig] = field(default_factory=dict)
    #: STAC search page size (the ``limit`` per page request), used only by providers queried
    #: through ``client.search()`` (Earth Search, Planetary Computer). The OPERA ``cmr-asf`` path
    #: bypasses CMR-STAC search for the native CMR granule API
    #: (``opera_query.make_s1_item_provider``), so it does not apply there; raising it for
    #: CMR-STAC made the
    #: 500s worse — context_docs/decisions/009-native-cmr-granule-query.md.
    max_page_size: int = 250
    refuses_oversized_pages: bool = False
    throttles_with_forbidden: bool = False


# =============================================================================
# Pre-configured Providers
# =============================================================================

PROVIDERS: dict[str, STACProvider] = {
    "earth-search": STACProvider(
        name="Earth Search (Element 84)",
        catalog_url="https://earth-search.aws.element84.com/v1",
        collections={
            # Earth Search harmonises the BOA offset (subtracts 1000 post-baseline 04.00) in its
            # OWN COGs, but this collection also indexes items pointing at ESA originals, which
            # still carry it — hence the threshold plus `harmonisation_varies_by_item`, which
            # decides per ASSET from where that asset lives. Unsetting the threshold exempts the
            # whole collection and is correct only if every item is a harmonised COG. The
            # raster:bands offset=-0.1 and earthsearch:boa_offset_applied metadata are unreliable
            # (sertit/eoreader#120), so asset location is the signal. ADR 021.
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
        # BELOW the class default: this catalogue refuses a response over ~6 MB (AWS Lambda's
        # synchronous response limit) and 250 `sentinel-2-l2a` items exceeds it. Per provider,
        # since nothing implicates the other catalogues. The cap tracks RESPONSE BYTES, and item
        # size is driven by footprint shape — items tracing a twelve-detector sawtooth run to
        # 98 KB against 0.2 KB for a rectangle, confined to roughly Nov 2018 - Mar 2019. So a
        # given hundred may or may not clear 6 MB depending on which hundred the cursor and date
        # window select, which is why one page deep in a walk is refused while the rest serve;
        # `ingest/stac.py` re-cuts the date window to regroup them.
        #
        # Headroom at 100 is only ~4% — largest page SERVED was 5.73 MB against a 4.6 MB average,
        # so do not reason from the average. Lowering further is deliberately NOT the answer: it
        # taxes every query in every year for six months of a ten-year archive, and
        # `ingest/stac.py` already re-asks a refused page at half the size. Measurements:
        # context_docs/ingest/ingest-performance.md §7c.
        max_page_size=100,
        refuses_oversized_pages=True,
        # Public and unauthenticated — we send no credential — so a 403 here cannot be about who
        # is asking; it is the aggregate request rate, i.e. a 429, and is waited out as one. Per
        # provider, because a 403 from a catalogue that DOES authorize is permanent and must keep
        # failing the leg on the first refusal.
        throttles_with_forbidden=True,
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
            # Planetary Computer serves ESA's values unharmonised and serves them ALL that way,
            # so the answer is the collection's and `harmonisation_varies_by_item` stays off. It
            # also keys assets natively (`B02`, `SCL`), relying on the loader to resolve the
            # common names in `bands`, so a per-item read of asset locations would find nothing.
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
