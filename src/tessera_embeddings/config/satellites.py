"""Per-satellite bands, baseline corrections and mappings, independent of provider or store."""

# =============================================================================
# Sentinel-2 L2A Configuration
# =============================================================================

# Native Sentinel-2 names -> STAC common names (STAC catalogs index by common name).
S2_BAND_MAPPING = {
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B8A": "nir08",
    "B11": "swir16",
    "B12": "swir22",
}

# Bands to load from STAC (common names)
S2_L2A_BANDS = list(S2_BAND_MAPPING.values())
S2_L1C_BANDS = list(S2_BAND_MAPPING.values())


# Bands as stored in Zarr (native names)
S2_STORED_BANDS = list(S2_BAND_MAPPING.keys())

# Scene Classification Layer — loaded as an extra band for cloud masking.
S2_SCL_BAND = "scl"
# SCL classes considered invalid (nodata, saturated, cloud shadow, cloud, snow/ice)
S2_SCL_INVALID_CLASSES = frozenset({0, 1, 2, 3, 8, 9})

# ESA added a +1000 offset to reflectances from processing baseline 4.00 onward.
S2_BASELINE_THRESHOLD = 400
S2_BASELINE_OFFSET = -1000


# =============================================================================
# Sentinel-1 GRD Configuration
# =============================================================================

S1_GRD_BANDS = ["vv", "vh"]

# No baseline correction currently required for Sentinel-1
S1_BASELINE_THRESHOLD = None
S1_BASELINE_OFFSET = 0

# =============================================================================
# OPERA RTC-S1 Configuration
# =============================================================================

# OPERA RTC-S1 band names as they appear in STAC assets
S1_OPERA_BANDS = ["0_VV", "0_VH"]

# Amplitude-to-dB conversion constants
# Formula: (20 * log10(amplitude) + S1_DB_SHIFT) * S1_DB_SCALE, clipped to [0, S1_DB_CLIP_MAX]
# and stored as UINT16, with 0 reserved as the nodata marker. The clip ceiling is int16's
# maximum, which is where an earlier comment's "clipped to int16" came from — but the stored
# dtype is unsigned and the floor is 0, not -32768. See `ingest.transforms.amplitude_to_db`.
# Ported from tessera_preprocessing/s1_fast_processor.py:758-797
S1_DB_SHIFT = 50
S1_DB_SCALE = 200
S1_DB_CLIP_MAX = 32767


# =============================================================================
# Landsat Collection 2 Level 2 Configuration
# =============================================================================

LANDSAT_C2_BANDS = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# No baseline correction for Landsat
LANDSAT_BASELINE_THRESHOLD = None
LANDSAT_BASELINE_OFFSET = 0
