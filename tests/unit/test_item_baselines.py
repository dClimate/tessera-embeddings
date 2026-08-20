"""One reader of `s2:processing_baseline`, on one scale, with one notion of unreadable.

There used to be two — a float parser for ranking duplicates and a scaled-integer parser for the
correction threshold — and each numeric edge case had to be found and fixed in both. They had
already drifted when this was written: one still accepted `"1e308"` as a baseline while the other
rejected it. These tests pin the properties that made the split dangerous.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tessera_embeddings.config.satellites import S2_BASELINE_THRESHOLD
from tessera_embeddings.ingest.item_baselines import BASELINE_SCALE, processing_baseline


def _item(raw: object, *, with_properties: bool = True) -> SimpleNamespace:
    item = SimpleNamespace(id="S2A_33TWM_20220107_0_L2A")
    if with_properties:
        item.properties = {} if raw is None else {"s2:processing_baseline": raw}
    return item


class TestTheScale:
    """Reported as an integer hundredth, which is the space the threshold is expressed in."""

    @pytest.mark.parametrize(("raw", "expected"), [("04.00", 400), ("05.10", 510), ("00.01", 1), ("00.00", 0)])
    def test_a_baseline_is_scaled_by_a_hundred(self, raw: str, expected: int) -> None:
        assert processing_baseline(_item(raw)) == expected

    def test_the_scale_matches_the_threshold(self) -> None:
        """A conversion between scales is where a factor of a hundred goes missing, so there is
        none: the parser's output is directly comparable to the threshold.
        """
        assert BASELINE_SCALE == 100
        assert S2_BASELINE_THRESHOLD == 400
        assert processing_baseline(_item("04.00")) == S2_BASELINE_THRESHOLD
        assert processing_baseline(_item("03.99")) < S2_BASELINE_THRESHOLD

    def test_a_rounding_artefact_does_not_lose_a_step(self) -> None:
        """5.10 * 100 is 509.999... in binary floating point."""
        assert processing_baseline(_item("05.10")) == 510


class TestUnreadableIsOneThing:
    """`None` means the item does not tell us, whatever the reason."""

    @pytest.mark.parametrize(
        "raw", [None, "", "not-a-number", "NaN", "nan", "Infinity", "-Infinity", "inf", "1e308", "-1e308"]
    )
    def test_every_kind_of_unreadable_is_none(self, raw: object) -> None:
        assert processing_baseline(_item(raw)) is None

    def test_an_item_with_no_properties_at_all_is_unreadable(self) -> None:
        assert processing_baseline(_item(None, with_properties=False)) is None

    def test_a_declared_zero_is_not_unreadable(self) -> None:
        """The distinction the old integer parser could not make. A baseline of 0 is below every
        threshold, so conflating it with "unknown" hid a correctness question.
        """
        assert processing_baseline(_item("00.00")) == 0
        assert processing_baseline(_item(None)) is None

    def test_a_non_string_value_is_handled(self) -> None:
        assert processing_baseline(_item(5.1)) == 510
        assert processing_baseline(_item(object())) is None

    def test_an_overflowing_value_does_not_raise(self) -> None:
        """`"1e308"` is finite on its own and becomes infinity scaled, where `round()` raises
        OverflowError — which aborted a whole batch over one item's metadata.
        """
        assert processing_baseline(_item("1e308")) is None

    @pytest.mark.parametrize("raw", ["1e10", "1e308", "999.00", "100.01"])
    def test_a_value_beyond_any_real_version_is_unreadable(self, raw: str) -> None:
        """No processing version reaches these. Read as numbers they clear every threshold, so a
        nonsense value would force a correction rather than being refused as ambiguous.
        """
        assert processing_baseline(_item(raw)) is None

    def test_a_high_but_plausible_version_still_parses(self) -> None:
        """The bound must leave real headroom above ESA's 05.x rather than pinning today's value."""
        assert processing_baseline(_item("99.99")) == 9999


class TestANegativeBaselineIsNotAVersion:
    """A negative processing version is no evidence that pixels predate baseline 04.00.

    Read as a real value it sits below every threshold, so an unharmonised item carrying one was
    exempted from a correction it may well have been owed. Raised on PR #107.
    """

    @pytest.mark.parametrize("raw", ["-1.00", "-0.01", "-99", -4.0])
    def test_a_negative_baseline_is_unreadable(self, raw: object) -> None:
        assert processing_baseline(_item(raw)) is None

    def test_zero_is_still_readable(self) -> None:
        """The boundary: 0 is a real declared value, and distinguishable from "declared nothing"."""
        assert processing_baseline(_item("00.00")) == 0
        assert processing_baseline(_item(None)) is None


class TestABaselineMustLandOnAHundredth:
    """A processing baseline is a two-decimal version, so a value between hundredths is malformed.

    `"03.999"` scaled by a hundred is 399.9, which rounds to 400 and crosses the correction
    threshold — forcing a subtraction on metadata that should have been refused as ambiguous.
    Raised on PR #107.
    """

    @pytest.mark.parametrize("raw", ["03.999", "04.001", "3.9999", "0.001"])
    def test_a_value_between_hundredths_is_unreadable(self, raw: str) -> None:
        assert processing_baseline(_item(raw)) is None

    def test_the_specific_threshold_crossing_case(self) -> None:
        """`"03.999"` must not become 400 and trigger a correction."""
        assert processing_baseline(_item("03.999")) is None
        assert processing_baseline(_item("04.00")) == 400

    @pytest.mark.parametrize(("raw", "expected"), [("05.10", 510), ("00.01", 1), ("5.1", 510), ("4", 400)])
    def test_exact_hundredths_still_parse_however_written(self, raw: str, expected: int) -> None:
        """Trailing-zero and shorthand forms are the same value, and binary floating point is why
        this is checked with Decimal: 5.10 * 100 is 509.999... as a float.
        """
        assert processing_baseline(_item(raw)) == expected


class TestScalingCannotUnderflowToAConfidentZero:
    """A tiny positive exponent underflows rather than overflowing.

    `Decimal("1e-1000000000") * 100` becomes `0E-1000026`, which IS integral, so the parser
    reported a confident baseline of 0 — read downstream as a real pre-threshold version, which
    exempts the date instead of refusing ambiguous metadata. Raised on PR #107.
    """

    @pytest.mark.parametrize("raw", ["1e-1000000000", "1e-9", "0.001", "0.009"])
    def test_a_value_below_one_hundredth_is_unreadable(self, raw: str) -> None:
        assert processing_baseline(_item(raw)) is None

    def test_one_hundredth_itself_still_parses(self) -> None:
        """The boundary: 00.01 is a real version and the smallest representable one."""
        assert processing_baseline(_item("00.01")) == 1

    def test_a_declared_zero_stays_readable(self) -> None:
        """Zero is a real declared value; only positive values BELOW a hundredth are rejected."""
        assert processing_baseline(_item("00.00")) == 0
        assert processing_baseline(_item("0")) == 0
