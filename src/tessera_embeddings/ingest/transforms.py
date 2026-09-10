"""Post-load data transformations applied after odc.stac.load.

These are sensor-specific operations that modify pixel values.
They operate as lazy Dask operations — no data is materialized.
"""

from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr

from tessera_embeddings.config import S1_DB_CLIP_MAX, S1_DB_SCALE, S1_DB_SHIFT


def amplitude_to_db(ds: xr.Dataset) -> xr.Dataset:
    """Convert OPERA RTC-S1 linear amplitude to scaled dB as uint16.

    Ported from tessera_preprocessing/s1_fast_processor.py:758-797.
    Formula: (20 * log10(amplitude) + 50) * 200, clipped to [0, 32767]. Applied per
    data variable, lazily — no data is materialized.

    Args:
        ds: xarray Dataset with float32 amplitude variables

    Returns:
        Dataset with uint16 scaled dB variables; 0 marks nodata.
    """
    result = {}
    for var in ds.data_vars:
        amp = ds[var].data  # Dask array

        # Mask zero/negative amplitudes to avoid log10 domain errors
        safe_amp = da.where(amp > 0, amp, np.float32(1e-10))

        db = 20.0 * da.log10(safe_amp)
        db_shifted = db + S1_DB_SHIFT
        scaled = db_shifted * S1_DB_SCALE

        clipped = da.clip(scaled, 0, S1_DB_CLIP_MAX)
        converted = clipped.astype(np.uint16)

        # Set zero-amplitude pixels back to 0 (nodata marker)
        final = da.where(amp > 0, converted, np.uint16(0))

        result[var] = xr.DataArray(final, dims=ds[var].dims, coords=ds[var].coords)

    return xr.Dataset(result, attrs=ds.attrs)
