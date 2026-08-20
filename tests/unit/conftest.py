"""Shared fixtures for unit tests (ported from the reference repo's top-level conftest).

Provides moto-backed S3, sample data factories, STAC item mocks, and ROI helpers
used across the storage and ingest test suites. Cloudmask fixtures from the
reference are intentionally omitted — tessera-embeddings has no cloudmask module.
"""

from __future__ import annotations

import boto3
import icechunk
import numpy as np
import pytest
import xarray as xr
from moto.server import ThreadedMotoServer

# -----------------------------------------------------------------------------
# Warning Suppression
# -----------------------------------------------------------------------------

# Suppress Icechunk's Rust-level warning about local filesystem concurrency.
# Valid for production but irrelevant for tests (moto S3 or single-threaded local).
icechunk.set_logs_filter("icechunk::storage::object_store=error")

# -----------------------------------------------------------------------------
# AWS Mock Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moto_server():
    """Start a threaded moto server for S3 operations.

    Yields the endpoint URL (e.g., "http://127.0.0.1:5555").
    Server runs for the duration of the test module.
    """
    server = ThreadedMotoServer(port=0)  # Port 0 = random available port
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def s3_client(moto_server):
    """Create a boto3 S3 client connected to the moto server."""
    return boto3.client(
        "s3",
        endpoint_url=moto_server,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        region_name="us-east-1",
    )


@pytest.fixture
def test_bucket(s3_client):
    """Create a test bucket and return its name."""
    bucket_name = "test-tessera-embeddings"
    s3_client.create_bucket(Bucket=bucket_name)
    return bucket_name


# -----------------------------------------------------------------------------
# Sample Data Fixtures
# -----------------------------------------------------------------------------

SAMPLE_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

# Common names used internally after loading from Zarr
COMMON_BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]


@pytest.fixture
def sample_reflectance_data():
    """Generate sample reflectance data with common band names.

    Returns a factory that builds xarray Datasets with configurable dates and
    spatial dimensions (default 256x256). Data uses common band names
    (blue, red, nir, ...) matching what modules expect after loading from Zarr.
    """

    def _make_data(
        dates: list[str],
        height: int = 256,
        width: int = 256,
        seed: int = 42,
    ) -> xr.Dataset:
        rng = np.random.default_rng(seed)
        n_times = len(dates)

        data_vars = {}
        for band in COMMON_BANDS:
            # Realistic reflectance: 100-5000 for most pixels
            data = rng.integers(100, 5000, size=(n_times, height, width), dtype=np.uint16)
            data_vars[band] = (["time", "northing", "easting"], data)

        # Use nanosecond precision for xarray/zarr compatibility
        coords = {
            "time": [np.datetime64(d, "ns") for d in dates],
            "northing": np.arange(height),
            "easting": np.arange(width),
        }

        return xr.Dataset(data_vars, coords=coords)

    return _make_data


@pytest.fixture
def sample_sar_data():
    """Factory returning xarray Dataset with VV/VH bands (SAR-style)."""

    def _make(
        dates: list[str],
        height: int = 20,
        width: int = 20,
        seed: int = 42,
    ) -> xr.Dataset:
        rng = np.random.default_rng(seed)
        n_times = len(dates)
        data_vars = {}
        for band in ["VV", "VH"]:
            data = rng.uniform(-25.0, 5.0, size=(n_times, height, width)).astype(np.float32)
            data_vars[band] = (["time", "northing", "easting"], data)
        coords = {
            "time": [np.datetime64(d, "ns") for d in dates],
            "northing": np.arange(height),
            "easting": np.arange(width),
        }
        return xr.Dataset(data_vars, coords=coords)

    return _make


# -----------------------------------------------------------------------------
# STAC Mock Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_stac_item():
    """Create a mock STAC item with configurable properties.

    Returns a factory that generates mock items with an acquisition datetime,
    processing baseline, cloud cover, and MGRS tile code.
    """

    def _make_item(
        date: str,
        baseline: str = "04.00",
        cloud_cover: float = 15.0,
        tile_id: str = "33UUP",
        host_root: str = "s3://sentinel-s2-l2a/tiles/33/U/UP/2024/1",
    ):
        from datetime import datetime
        from unittest.mock import Mock

        from tessera_embeddings.ingest.asset_locations import READ_ASSET_KEYS

        item = Mock()
        item.datetime = datetime.fromisoformat(date)
        item.properties = {
            "s2:processing_baseline": baseline,
            "eo:cloud_cover": cloud_cover,
            "grid:code": f"MGRS-{tile_id}",
            # The ACQUISITION instant, which is where a real item keeps it and the only surviving
            # record of it once normalize_to_solar_day has stamped `.datetime` with noon. Duplicate
            # selection reads this to tell distinct same-day passes from reprocessings of one, so a
            # fixture without it makes every scene of a day look like one acquisition.
            "datetime": date,
        }
        # REAL asset hrefs, because whether the BOA offset is corrected is decided from where
        # the assets live. A bare Mock auto-creates `assets`, so an href read off it is a Mock
        # rather than a string, and the item then classifies as "producer unknown" — which means
        # "correct it", silently changing what a test measures without the test saying so.
        # Defaults to ESA's originals, the case that DOES need correcting, so a test about
        # baseline parsing sees its baseline flow through. Pass `host_root` pointing at
        # sentinel-cogs to model Element 84's harmonised COGs instead.
        item.assets = {key: {"href": f"{host_root}/{key}"} for key in READ_ASSET_KEYS}
        return item

    return _make_item


# -----------------------------------------------------------------------------
# Temporary Directory Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def local_zarr_path(tmp_path):
    """Provide a temporary path for local Zarr store tests."""
    zarr_dir = tmp_path / "zarr_stores"
    zarr_dir.mkdir()
    return zarr_dir


# -----------------------------------------------------------------------------
# Icechunk S3 Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def icechunk_s3_config(moto_server, test_bucket):
    """Provide Icechunk S3 configuration for moto-backed tests.

    Returns a dict with parameters for icechunk.s3_storage().
    """
    return {
        "bucket": test_bucket,
        "endpoint_url": moto_server,
        "allow_http": True,
        "access_key_id": "testing",
        "secret_access_key": "testing",
        "region": "us-east-1",
    }


@pytest.fixture
def icechunk_s3_store_path(test_bucket):
    """Return a factory that generates S3 store paths for Icechunk.

    Usage:
        path = icechunk_s3_store_path("my-store")
        # Returns: "s3://test-tessera-embeddings/my-store"
    """

    def _make_path(store_name: str) -> str:
        return f"s3://{test_bucket}/{store_name}"

    return _make_path


# -----------------------------------------------------------------------------
# ROI Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def mock_roi_metadata():
    """Factory returning ROIMetadata with configurable size/CRS.

    Builds ROIMetadata dataclass instances without touching the filesystem.
    """
    from unittest.mock import Mock

    from tessera_embeddings.ingest.roi import ROIMetadata

    def _make(
        height: int = 20,
        width: int = 20,
        crs: str = "EPSG:32615",
        bbox_wgs84: tuple[float, float, float, float] = (-90.5, 44.0, -90.0, 44.5),
    ) -> ROIMetadata:
        geobox = Mock()
        geobox.shape = Mock(y=height, x=width)
        return ROIMetadata(
            bbox_wgs84=bbox_wgs84,
            native_crs=crs,
            geobox=geobox,
            width=width,
            height=height,
        )

    return _make


@pytest.fixture
def roi_mask_array():
    """Factory returning boolean numpy array with configurable coverage.

    Args via factory call:
        height, width: spatial dimensions (default 20x20)
        coverage: fraction of True pixels (default 0.8)
        seed: RNG seed for reproducibility
    """

    def _make(
        height: int = 20,
        width: int = 20,
        coverage: float = 0.8,
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mask = rng.random((height, width)) < coverage
        return mask

    return _make


# -----------------------------------------------------------------------------
# Inference Fixtures (ported from the reference inference/conftest.py)
# -----------------------------------------------------------------------------


@pytest.fixture
def inference_config():
    """Minimal v1.1 inference config for tests (small model, tiny buckets)."""
    from tessera_embeddings.config.inference import InferenceConfig
    from tessera_embeddings.config.time_windows import parse_time_window

    return InferenceConfig(
        latent_dim=32,
        representation_dim=32,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        fusion_method="concat",
        # Keep buckets small so synthetic chunks fit and tests stay fast.
        num_obs_checkpoints=(4, 8),
        batch_size=16,
        num_workers=0,
        checkpoint_path="dummy",  # non-empty so __post_init__ doesn't hit resolve_buckets
        time_window=parse_time_window("June 2025"),
    )


@pytest.fixture
def sample_chunk_data():
    """Factory that creates synthetic ChunkData with configurable dimensions."""
    from tessera_embeddings.inference.data_loading import ChunkData

    def _make(
        height: int = 10,
        width: int = 10,
        t_s2: int = 10,
        t_s1a: int = 5,
        t_s1d: int = 5,
        seed: int = 42,
    ) -> ChunkData:
        rng = np.random.default_rng(seed)

        s2_bands = rng.integers(100, 5000, size=(t_s2, height, width, 10)).astype(np.uint16)
        s2_masks = rng.choice([0, 1], size=(t_s2, height, width), p=[0.2, 0.8]).astype(np.int32)
        s2_doys = np.linspace(30, 330, t_s2).astype(np.int32)

        s1_asc_bands = rng.integers(1000, 8000, size=(t_s1a, height, width, 2)).astype(np.uint16)
        s1_asc_doys = np.linspace(30, 330, t_s1a).astype(np.int32)

        s1_desc_bands = rng.integers(1000, 8000, size=(t_s1d, height, width, 2)).astype(np.uint16)
        s1_desc_doys = np.linspace(30, 330, t_s1d).astype(np.int32)

        s2_obs_count = s2_masks.sum(axis=0).astype(np.uint16)

        return ChunkData(
            s2_bands=s2_bands,
            s2_masks=s2_masks,
            s2_doys=s2_doys,
            s1_asc_bands=s1_asc_bands,
            s1_asc_doys=s1_asc_doys,
            s1_desc_bands=s1_desc_bands,
            s1_desc_doys=s1_desc_doys,
            height=height,
            width=width,
            s2_obs_count=s2_obs_count,
        )

    return _make


@pytest.fixture
def test_model(inference_config):
    """Randomly-initialized v1.1 inference model on CPU in eval mode."""
    import torch

    from tessera_embeddings.inference.models.builder import _build_inference_model

    model = _build_inference_model(inference_config, torch.device("cpu"))
    model.eval()
    return model
