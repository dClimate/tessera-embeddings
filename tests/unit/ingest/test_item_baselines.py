"""One reader of `s2:processing_baseline`, validated as a VERSION rather than as a number.

That distinction is the point of this module, and it was learned the hard way: seven separate review
findings, each naming a different value a numeric parser accepts and a version never contains —
not-a-number, infinity, negatives, excess decimal places, exponents that overflow when scaled,
exponents that underflow to zero, and long fractions that round onto a valid-looking value under the
ambient decimal context. Each was fixed as reported and the next arrived.

Matching the shape closes all of them at once, so these tests are organised by CATEGORY rather than
by reported value: anything outside the grammar is unreadable, whatever it is. A newly discovered
malformed input belongs in `NOT_A_VERSION` below, not in a new guard.
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


class _Opaque:
    """An arbitrary object with a STABLE repr.

    `object()`'s repr carries its memory address, which parallel test workers disagree about and
    pytest then rejects as a collection mismatch.
    """

    def __repr__(self) -> str:
        return "<opaque>"


#: Every shape the catalogue emits, plus the lenient forms of the same versions.
IS_A_VERSION = [
    ("05.10", 510),
    ("05.11", 511),
    ("02.06", 206),
    ("04.00", 400),
    ("00.01", 1),
    ("00.00", 0),
    ("5.10", 510),
    ("5.1", 510),
    ("4", 400),
    (4, 400),
    (4.0, 400),
    (5.1, 510),
    ("  05.10  ", 510),
]

#: Anything a numeric parser accepts and a version never contains, grouped by WHY, and labelled so
#: the parametrised ids are stable. Add to this list rather than adding a guard: the grammar already
#: rejects the whole class.
NOT_A_VERSION: list[tuple[str, object]] = [
    # special values that float() and Decimal() both accept
    ("nan-word", "NaN"),
    ("nan-lower", "nan"),
    ("inf-word", "Infinity"),
    ("neg-inf-word", "-Infinity"),
    ("inf-short", "inf"),
    ("nan-float", float("nan")),
    ("inf-float", float("inf")),
    # signed, which reads as evidence the pixels predate the threshold
    ("neg", "-1.00"),
    ("neg-small", "-0.01"),
    ("neg-int", "-4"),
    ("plus", "+4.00"),
    # more than two decimal places
    ("three-dp", "3.999"),
    ("milli", "0.001"),
    ("four-dp", "4.0001"),
    # long enough to round onto 400 under the ambient decimal precision
    ("precision-rounds", "3.99999999999999999999999999999"),
    # exponents: one overflows when scaled, one underflows to zero, none is a version
    ("exp-ten", "1e10"),
    ("exp-overflow", "1e308"),
    ("exp-underflow", "1e-1000000000"),
    ("exp-zero", "4e0"),
    ("exp-caps", "5E1"),
    # beyond any real version, which the digit limits bound without a range constant
    ("hundred", "100.00"),
    ("three-digits", "999"),
    # other integer literals
    ("hex", "0x10"),
    ("binary", "0b1"),
    # Unicode decimal digits, which a bare ``\\d`` and ``int()`` both accept
    ("arabic-indic", chr(0x664)),
    ("arabic-indic-dotted", chr(0x661) + "." + chr(0x660)),
    ("fullwidth", chr(0xFF11)),
    # malformed punctuation
    ("trailing-dot", "4."),
    ("leading-dot", ".4"),
    ("dots", ".."),
    ("three-parts", "4.0.0"),
    ("comma", "5,1"),
    # empty and whitespace
    ("empty", ""),
    ("spaces", "   "),
    ("tab", "\t"),
    # words and adornments
    ("word", "four"),
    ("v-prefix", "v4.00"),
    ("suffix", "04.00-beta"),
    # non-strings that are not numbers
    ("none", None),
    ("true", True),
    ("object", _Opaque()),
    ("list", []),
    ("dict", {}),
]


class TestARealVersionParses:
    """The forms the catalogue emits, and the lenient spellings of the same versions."""

    @pytest.mark.parametrize(("raw", "expected"), IS_A_VERSION)
    def test_every_known_form(self, raw: object, expected: int) -> None:
        assert processing_baseline(_item(raw)) == expected

    def test_the_scale_matches_the_threshold(self) -> None:
        """A conversion between scales is where a factor of a hundred goes missing, so there is
        none: the output is directly comparable to the threshold.
        """
        assert BASELINE_SCALE == 100
        assert processing_baseline(_item("04.00")) == S2_BASELINE_THRESHOLD
        assert processing_baseline(_item("03.99")) < S2_BASELINE_THRESHOLD
        assert processing_baseline(_item("04.01")) > S2_BASELINE_THRESHOLD

    def test_a_declared_zero_is_not_unreadable(self) -> None:
        """The distinction callers depend on: 0 is below every threshold, so conflating it with
        "declared nothing" hides a correctness question rather than raising one.
        """
        assert processing_baseline(_item("00.00")) == 0
        assert processing_baseline(_item(None)) is None

    def test_a_one_digit_fraction_means_tenths(self) -> None:
        """A one-digit fraction means TENTHS: 5.1 and 05.10 are the same version, so it is padded."""
        assert processing_baseline(_item("5.1")) == processing_baseline(_item("05.10")) == 510


class TestAnythingElseIsUnreadable:
    """The whole class, not the reported members of it."""

    @pytest.mark.parametrize(("label", "raw"), NOT_A_VERSION, ids=[c[0] for c in NOT_A_VERSION])
    def test_it_reads_as_nothing(self, label: str, raw: object) -> None:
        assert processing_baseline(_item(raw)) is None, label

    def test_an_item_with_no_properties_at_all(self) -> None:
        assert processing_baseline(_item(None, with_properties=False)) is None

    def test_nothing_in_the_list_raises(self) -> None:
        """The other half of the contract. Several of these used to raise — OverflowError from
        rounding an infinity, and from scaling a finite-but-enormous value — aborting a whole batch
        over one item's metadata rather than treating that item as unreadable.
        """
        for _label, raw in NOT_A_VERSION:
            processing_baseline(_item(raw))

    def test_the_grammar_bounds_the_value_without_a_range_check(self) -> None:
        """Two digits for the whole part is what caps this at 99.99, so there is no range constant
        to drift out of step with the pattern.
        """
        assert processing_baseline(_item("99.99")) == 9999
        assert processing_baseline(_item("100.00")) is None
