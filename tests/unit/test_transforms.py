"""Unit tests for transforms.py - Post-load data transformations."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from tessera_embeddings.ingest.transforms import amplitude_to_db

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_amplitude_dataset(values, bands=("0_VV",)):
    """Create a lazy xarray Dataset with given amplitude values.

    Args:
        values: Flat list of 4 floats (reshaped to 1x2x2).
        bands: Data variable names.
    """
    arr = da.from_array(
        np.array(values, dtype=np.float32).reshape(1, 2, 2),
        chunks=(1, 2, 2),
    )
    return xr.Dataset(
        {band: (["time", "northing", "easting"], arr) for band in bands},
        coords={"time": [np.datetime64("2024-06-01")], "northing": [0, 1], "easting": [0, 1]},
    )


def _convert_scalar(amplitude):
    """Run amplitude_to_db on a uniform 2x2 patch and return the [0,0,0] pixel."""
    ds = _make_amplitude_dataset([amplitude] * 4)
    return amplitude_to_db(ds)["0_VV"].compute().values[0, 0, 0]


# ---------------------------------------------------------------------------
# Known-value conversions — parameterized
# ---------------------------------------------------------------------------


class TestAmplitudeToDb:
    """Tests for amplitude_to_db()."""

    @pytest.mark.parametrize(
        "amplitude, expected",
        [
            (1.0, 10000),    # dB=0, shifted=50, scaled=10000
            (0.1, 6000),     # dB=-20, shifted=30, scaled=6000
            (1e10, 32767),   # clipped to int16 max
            (1e-30, 0),      # clipped to 0
        ],
        ids=["amp-1.0", "amp-0.1", "clip-max", "clip-min"],
    )
    def test_known_values(self, amplitude, expected):
        assert _convert_scalar(amplitude) == expected

    def test_zero_amplitude_maps_to_nodata(self):
        """Zero amplitude → 0 (nodata marker); nonzero pixels unaffected."""
        ds = _make_amplitude_dataset([0.0, 1.0, 0.0, 1.0])
        result = amplitude_to_db(ds)["0_VV"].compute().values
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 10000

    def test_nan_amplitude_maps_to_nodata(self):
        """NaN amplitude should map to 0 (treated like zero/nodata)."""
        ds = _make_amplitude_dataset([np.nan, 1.0, np.nan, 1.0])
        result = amplitude_to_db(ds)["0_VV"].compute().values
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 10000

    def test_negative_amplitude_maps_to_nodata(self):
        """Negative amplitudes are physically impossible; treat as nodata."""
        ds = _make_amplitude_dataset([-0.1, 1.0, -999.0, 1.0])
        result = amplitude_to_db(ds)["0_VV"].compute().values
        assert result[0, 0, 0] == 0
        assert result[0, 0, 1] == 10000

    def test_output_dtype_is_uint16(self):
        ds = _make_amplitude_dataset([0.5, 0.5, 0.5, 0.5])
        assert amplitude_to_db(ds)["0_VV"].dtype == np.uint16

    def test_multiple_bands(self):
        """Should convert all data variables in the dataset."""
        ds = _make_amplitude_dataset([1.0, 1.0, 1.0, 1.0], bands=("0_VV", "0_VH"))
        result = amplitude_to_db(ds)
        for band in ("0_VV", "0_VH"):
            assert band in result
            assert result[band].compute().values[0, 0, 0] == 10000

    def test_preserves_coordinates(self):
        ds = _make_amplitude_dataset([1.0, 1.0, 1.0, 1.0])
        result = amplitude_to_db(ds)
        for coord in ("time", "northing", "easting"):
            assert coord in result.coords

    def test_lazy_operation(self):
        """Result should remain a Dask array (not eagerly computed)."""
        ds = _make_amplitude_dataset([1.0, 1.0, 1.0, 1.0])
        result = amplitude_to_db(ds)
        assert isinstance(result["0_VV"].data, da.Array)

    def test_typical_rtc_mean_value(self):
        """A typical RTC amplitude (0.182) lands at the hand-computed scaled-dB value.

        (20*log10(0.182) + 50) * 200 ≈ 7040 (after float32 truncation to uint16).
        Hardcoded independently of the source constants so this is a real
        cross-check, not a re-derivation.
        """
        assert _convert_scalar(0.182) == 7040
