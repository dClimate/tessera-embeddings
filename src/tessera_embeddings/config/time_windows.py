import calendar
from dataclasses import dataclass
from datetime import datetime


def _month_add(year: int, month: int, delta: int) -> tuple[int, int]:
    """Add *delta* months to a (year, month) pair. *delta* may be negative."""
    zero_based = (year * 12 + (month - 1)) + delta
    return zero_based // 12, zero_based % 12 + 1


@dataclass(frozen=True)
class TimeWindow:
    """A resolved 12-month observation window for inference.

    Always contains exactly 12 chronologically sorted ``(year, month)`` pairs.
    Created via :func:`parse_time_window` from a single end-month string.

    Attributes:
        window_start: First ``(year, month)`` of the window.
        window_end: Last ``(year, month)`` of the window.
        months: All 12 ``(year, month)`` tuples in chronological order.
        window_end_label: ISO date string (``"YYYY-MM-DD"``) using first-of-month
            convention (e.g. ``"2025-06-01"`` represents the full month of
            June 2025).  Used as the output Zarr time coordinate label.
            Data filtering is by ``(year, month)`` membership, so **all**
            observations within each month are included regardless of day.
    """

    window_start: tuple[int, int]
    window_end: tuple[int, int]
    months: tuple[tuple[int, int], ...]
    window_end_label: str

    def to_date_range(self) -> tuple[str, str]:
        """Return ``(start_date, end_date)`` as ``YYYY-MM-DD`` strings.

        ``start_date`` is the first day of :attr:`window_start` month;
        ``end_date`` is the last day of :attr:`window_end` month.
        """
        sy, sm = self.window_start
        ey, em = self.window_end
        start = f"{sy}-{sm:02d}-01"
        end = f"{ey}-{em:02d}-{calendar.monthrange(ey, em)[1]:02d}"
        return start, end


def _parse_month_year(s: str) -> tuple[int, int]:
    """Parse a single ``"Month Year"`` string into a ``(year, month)`` tuple.

    Args:
        s: A string like ``"June 2025"``.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        dt = datetime.strptime(s, "%B %Y")
    except ValueError:
        msg = f"Cannot parse '{s}' — expected 'Month Year' format (e.g. 'April 2025')"
        raise ValueError(msg) from None
    return (dt.year, dt.month)


def parse_time_window(date: str) -> TimeWindow:
    """Parse a single end-month into a 12-month rolling ``TimeWindow``.

    The window always contains exactly 12 months ending at *date*, in
    chronological order.

    Args:
        date: End month as ``"Month Year"`` (e.g. ``"June 2025"``).

    Returns:
        A ``TimeWindow`` spanning the 12 months ending at *date*.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    end_ym = _parse_month_year(date)
    start_ym = _month_add(end_ym[0], end_ym[1], -11)
    months = tuple(_month_add(start_ym[0], start_ym[1], i) for i in range(12))
    window_end_label = f"{end_ym[0]}-{end_ym[1]:02d}-01"
    return TimeWindow(
        window_start=start_ym,
        window_end=end_ym,
        months=months,
        window_end_label=window_end_label,
    )
