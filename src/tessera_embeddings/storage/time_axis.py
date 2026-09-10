"""How a store encodes its time axis, and the calendar the campaign runs on.

Gathered here because the answer used to be in three places: the encoding constant and its
two readers in the general Zarr store module, the calendar-year convention in the module
about the 120 UTM-zone *spatial* grid, and the daily-timestamp builder in the all-fill
seeder. Callers reached into two of them on adjacent lines.

Every symbol here is a pure function over ``numpy``/``zarr`` with no other in-package
dependency, so this is a leaf module: importing it never drags in a store implementation.
"""

from __future__ import annotations

import numpy as np
import zarr

# Standard time encoding for all stores
TIME_ENCODING = {"units": "nanoseconds since 1970-01-01", "calendar": "proleptic_gregorian"}


#: The campaign's annual timesteps (2025 filled first, then backwards).
CAMPAIGN_YEARS: tuple[int, ...] = tuple(range(2017, 2026))


def read_time_values(node: zarr.Group) -> np.ndarray:
    """Decode a group's ``time`` coordinate to ``datetime64[ns]`` values.

    Raw-zarr counterpart to xarray's CF decoding for the one convention every engine-written
    store uses (:data:`TIME_ENCODING`); anything else is a loud error, not a silent misread.
    """
    time_arr = node["time"]
    units = str(time_arr.attrs.get("units", ""))
    if not units.startswith("nanoseconds since 1970-01-01"):
        raise ValueError(
            f"Unsupported time units {units!r} on {node.store!r} — every engine-written "
            "store uses TIME_ENCODING (nanoseconds since 1970-01-01)."
        )
    return np.asarray(time_arr[:]).astype("int64").astype("datetime64[ns]")  # type: ignore[index]


def time_index_of(node: zarr.Group, value: np.datetime64) -> int | None:
    """Index of ``value`` on a group's time axis, or ``None`` if absent."""
    hits = np.flatnonzero(read_time_values(node) == value)
    return int(hits[0]) if hits.size else None


def compute_doy(timestamps: np.ndarray) -> np.ndarray:
    """Compute day-of-year from datetime64 timestamps: an (N,) int32 array of 1-366."""
    years = timestamps.astype("datetime64[Y]")
    return ((timestamps.astype("datetime64[D]") - years).astype(int) + 1).astype(np.int32)


def daily_times(start: str, end: str) -> np.ndarray:
    """Return one ``datetime64[ns]`` timestamp per day in ``[start, end]`` (inclusive).

    Both bounds are ``YYYY-MM-DD`` strings and ``end`` must not precede ``start``;
    ``2025-01-01`` → ``2025-12-31`` yields 365 timestamps.

    A daily axis pre-seeds every day a later infill might land on, so a region-overwrite of any
    single date aligns to an existing coordinate without an append. Seeding instead from the
    union of a collection of input stores' time axes belongs to the batch region-merge's
    date-union helper.
    """
    start_d = np.datetime64(start, "D")
    end_d = np.datetime64(end, "D")
    if end_d < start_d:
        msg = f"end {end!r} precedes start {start!r}"
        raise ValueError(msg)
    days = np.arange(start_d, end_d + np.timedelta64(1, "D"), dtype="datetime64[D]")
    return days.astype("datetime64[ns]")


def year_timestamp(year: int) -> np.datetime64:
    """The Q2 calendar-year convention, encoded once: ``year`` → ``YYYY-01-01`` ns."""
    return np.datetime64(f"{year}-01-01", "ns")


def year_of(value: np.datetime64 | np.ndarray) -> int:
    """Inverse of :func:`year_timestamp`: a (scalar) timestamp's calendar year."""
    return int(np.asarray(value).astype("datetime64[Y]").astype(int)) + 1970


def calendar_year_times(years: tuple[int, ...] = CAMPAIGN_YEARS) -> np.ndarray:
    """Return one ``datetime64[ns]`` per year at ``YYYY-01-01`` (Q2 convention)."""
    return np.array([year_timestamp(y) for y in years])
