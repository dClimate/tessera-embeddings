"""Sensor-agnostic ROI processing: mask application and coverage filtering.

These operations work on any xarray/Dask Dataset with spatial dimensions (northing, easting)
and a quality band. They are shared by S2 and S1 ROI ingestion flows.
"""

import logging
from collections.abc import Set

import dask.array as da
import numpy as np
import xarray as xr

from tessera_embeddings.ingest.roi import read_roi_mask

logger = logging.getLogger(__name__)

# Default minimum percentage of valid ROI pixels required to keep a date.
DEFAULT_MIN_VALID_COVERAGE = 5.0


def apply_roi_mask(
    data: xr.Dataset,
    roi_zarr_path: str,
    spatial_chunks: dict[str, int],
    fill_value: int = 0,
    roi_mask: da.Array | None = None,
) -> tuple[xr.Dataset, int]:
    """Apply a binary ROI mask to all variables in a dataset.

    Reads the Zarr ROI store, and sets pixels outside the ROI to
    ``fill_value`` for every data variable.

    Args:
        data: Dataset with (time, northing, easting) dimensions and an ``odc.geobox``.
        roi_zarr_path: Path to the Zarr ROI store (local or s3://).
        spatial_chunks: Dict with ``"northing"`` and ``"easting"`` chunk sizes for the
            broadcast dask array (should match the dataset's load chunks).
        fill_value: Value to assign outside the ROI. Default 0.
        roi_mask: Pre-computed boolean dask array (northing, easting), e.g. already
            persisted on workers. When provided, avoids re-reading from
            the Zarr store.

    Returns:
        Tuple of (masked dataset, roi_pixel_count) where roi_pixel_count is
        the number of True pixels in the mask.
    """
    mask_2d = roi_mask if roi_mask is not None else read_roi_mask(roi_zarr_path, spatial_chunks)
    mask_da = mask_2d[np.newaxis, :, :]
    roi_pixel_count = int(mask_2d.sum().compute())
    for var in data.data_vars:
        data[var] = data[var].where(mask_da, other=fill_value)
    return data, roi_pixel_count


def filter_low_coverage_dates(
    data: xr.Dataset,
    roi_pixel_count: int,
    quality_band: str,
    invalid_values: Set[int],
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
) -> xr.Dataset:
    """Drop time steps where valid coverage within the ROI is too low.

    Expects the ROI mask to have already been applied (out-of-ROI pixels
    set to a value included in ``invalid_values``) so that the quality band
    captures both no-data and outside-ROI.

    Only the per-date valid-pixel fractions (one scalar per time step) are
    computed to decide which dates to keep; all band arrays stay lazy.

    Args:
        data: Dataset with (time, northing, easting) dims and the ROI mask already applied.
        roi_pixel_count: Number of pixels inside the ROI mask, used as the
            denominator for coverage percentage.
        quality_band: Name of the quality/classification variable to check
            (e.g. ``"scl"`` for S2, ``"0_VV"`` for S1).
        invalid_values: Set of values in the quality band considered invalid.
        min_valid_coverage: Minimum percentage of valid pixels to keep a
            date. Dates below this are dropped.

    Returns:
        Dataset with low-coverage dates removed. If all dates pass,
        the original (lazy) dataset is returned unchanged.
    """
    if quality_band not in data.data_vars:
        logger.warning(f"No '{quality_band}' band in dataset — skipping coverage filter")
        return data

    qb = data[quality_band]  # lazy (time, northing, easting)

    # isin reads qb once in a single fused pass. The previous OR-chain over
    # individual equality checks created a fan-out (one qb read per invalid
    # value) that Dask cannot fuse, causing repeated S3 fetches of the SCL band.
    valid_counts = (~qb.isin(list(invalid_values))).sum(dim=("northing", "easting"))  # (time,), lazy

    # Compute only the tiny per-date counts — band arrays stay lazy.
    valid_counts_np = valid_counts.compute()
    pcts = 100.0 * valid_counts_np.values / roi_pixel_count
    dates = [str(t.values)[:10] for t in data.time]
    keep_mask = pcts >= min_valid_coverage

    dropped = [d for d, keep in zip(dates, keep_mask, strict=True) if not keep]
    if dropped:
        logger.info(f"Dropping {len(dropped)} dates with <{min_valid_coverage}% valid coverage: {dropped}")
        for d, pct in zip(dates, pcts, strict=True):
            if pct < min_valid_coverage:
                logger.debug(f"  {d}: {pct:.1f}% valid")

    if not keep_mask.any():
        return data.isel(time=[])  # empty along time

    if keep_mask.all():
        return data  # nothing to drop

    return data.isel(time=keep_mask)


def identify_low_coverage_ds(
    data: xr.Dataset,
    roi_pixel_count: int,
    quality_band: str,
    invalid_values: Set[int],
    min_valid_coverage: float = DEFAULT_MIN_VALID_COVERAGE,
) -> xr.Dataset:
    """Lazily zero out all bands when valid coverage is below the threshold.

    Unlike :func:`filter_low_coverage_dates` (which eagerly computes coverage
    to decide which timesteps to drop), this function keeps the entire Dask
    graph lazy by using ``xr.where`` to mask bands to ``fill_value`` when
    coverage is insufficient.  Best suited for single-timestep datasets where
    there is no need to shrink the time dimension.

    Args:
        data: Dataset with (time, northing, easting) dims and the ROI mask already applied.
        roi_pixel_count: Number of pixels inside the ROI mask, used as the
            denominator for coverage percentage.
        quality_band: Name of the quality/classification variable to check
            (e.g. ``"scl"`` for S2, ``"0_VV"`` for S1).
        invalid_values: Set of values in the quality band considered invalid.
        min_valid_coverage: Minimum percentage of valid pixels to keep a
            date. Dates below this are zeroed out.
        fill_value: Value to assign when coverage is too low. Default 0.

    Returns:
        Dataset with the same shape and a ``valid_coverage`` coordinate on
        the time dimension (lazy bool).  Downstream code can check
        ``ds["valid_coverage"].compute()`` — a single scalar — to decide
        whether to skip the timestep without reading any band data.
        If coverage is below the threshold, all bands are set to
        ``fill_value``; otherwise the data is unchanged.
        The entire operation remains lazy.
    """
    if quality_band not in data.data_vars:
        logger.warning(f"No '{quality_band}' band in dataset — skipping coverage filter")
        return data

    qb = data[quality_band]  # lazy (time, northing, easting)
    valid_count = (~qb.isin(list(invalid_values))).sum(dim=("northing", "easting"))  # lazy
    is_valid = (100.0 * valid_count / roi_pixel_count) >= min_valid_coverage  # lazy bool

    data = data.assign_coords(valid_coverage=("time", is_valid.data))

    return data
