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


#: Every shape the catalogue emits, plus the lenient forms of the same versions.
IS_A_VERSION = [
    ("05.10", 510),
    ("00.01", 1),
    ("00.00", 0),  # declared zero is a version, distinct from declaring nothing
    ("5.1", 510),  # a one-digit fraction means tenths
    ("4", 400),
    (4.0, 400),
    ("  05.10  ", 510),
    ("99.99", 9999),  # two whole-part digits cap the value, with no range constant to drift
]

#: Anything a numeric parser accepts and a version never contains, one per category, labelled so
#: the parametrised ids are stable. Add to this list rather than adding a guard: the grammar already
#: rejects the whole class. A value that raised instead would fail its row.
NOT_A_VERSION: list[tuple[str, object]] = [
    ("nan-word", "NaN"),
    ("inf-float", float("inf")),
    ("neg", "-1.00"),
    ("plus", "+4.00"),
    ("three-dp", "3.999"),
    ("precision-rounds", "3.99999999999999999999999999999"),
    ("exp-overflow", "1e308"),
    ("exp-underflow", "1e-1000000000"),
    ("hundred", "100.00"),
    ("hex", "0x10"),
    ("arabic-indic", chr(0x664)),
    ("trailing-dot", "4."),
    ("empty", ""),
    ("v-prefix", "v4.00"),
    ("none", None),
    ("true", True),
    ("list", []),
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


class TestAnythingElseIsUnreadable:
    """The whole class, not the reported members of it."""

    @pytest.mark.parametrize(("label", "raw"), NOT_A_VERSION, ids=[c[0] for c in NOT_A_VERSION])
    def test_it_reads_as_nothing(self, label: str, raw: object) -> None:
        assert processing_baseline(_item(raw)) is None, label

    def test_an_item_with_no_properties_at_all(self) -> None:
        assert processing_baseline(_item(None, with_properties=False)) is None
