"""Reading a Sentinel-2 item's processing baseline.

The single reader for ``s2:processing_baseline``, so the scale and the notion of "unreadable" are
defined once.

**Scale.** Reported as an integer hundredth: ``"04.00"`` is ``400``, ``"05.10"`` is ``510``. This is
the space :data:`~tessera_embeddings.config.satellites.S2_BASELINE_THRESHOLD` is expressed in, so a
comparison against the threshold needs no conversion.

**Unreadable.** ``None`` means the item does not declare a usable baseline — absent, empty,
non-numeric, negative, not an exact hundredth, or beyond any real version. Callers substitute
their own default explicitly,
which keeps "declared 0" distinguishable from "declared nothing"; a baseline of 0 is below every
threshold, so conflating them hides a correctness question.
"""

from __future__ import annotations

import decimal
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: What one whole baseline step is worth on the reported scale, e.g. ``04.00`` -> ``400``.
BASELINE_SCALE = 100

#: Beyond this a declared value is not a processing version. ESA is on 05.x; a hundred whole
#: versions of headroom is generous, and the bound is what keeps a nonsense value from being read
#: as an enormous baseline that clears every threshold.
MAX_BASELINE = 100 * BASELINE_SCALE


def processing_baseline(item: Any) -> int | None:  # noqa: ANN401 — any STAC-like item
    """The baseline an item declares, as an integer hundredth, or ``None`` if it declares none.

    A baseline is a two-decimal version, so the declared value must land exactly on a hundredth.
    Parsed as a :class:`~decimal.Decimal` to check that: ``"03.999"`` scaled by a hundred is 399.9,
    which rounds to 400 and would cross the correction threshold on metadata that is malformed.

    Also rejected, each for the same reason — none of them is a baseline, and reading one as a
    number would let malformed metadata decide whether an offset is subtracted:

    * ``"NaN"`` and ``"Infinity"``, which :func:`float` and :class:`~decimal.Decimal` both accept.
    * negatives, which sit below every threshold and so read as positive evidence that the pixels
      predate baseline 04.00.
    * values beyond :data:`MAX_BASELINE`, which no processing version reaches.
    """
    properties = getattr(item, "properties", None)
    raw = properties.get("s2:processing_baseline") if isinstance(properties, dict) else None
    if raw is None or raw == "":
        return None
    try:
        value = decimal.Decimal(str(raw))
    except (decimal.InvalidOperation, TypeError, ValueError):
        logger.debug("Unparseable s2:processing_baseline %r on %s", raw, getattr(item, "id", "?"))
        return None
    # Bounded BEFORE scaling: `Decimal("1e999999")` is finite and constructs happily, but
    # multiplying it raises `decimal.Overflow`, which would abort a whole batch over one item.
    if not value.is_finite() or value < 0 or value > MAX_BASELINE:
        logger.debug("Not a baseline: s2:processing_baseline %r on %s", raw, getattr(item, "id", "?"))
        return None
    # Below one hundredth is not a version, and scaling such a value UNDERFLOWS rather than
    # overflowing — `Decimal("1e-1000000000") * 100` is `0E-1000026`, which passes the integral
    # check below and would read as a confident baseline of 0. A declared zero stays readable.
    if 0 < value < decimal.Decimal("0.01"):
        logger.debug("Below one hundredth: s2:processing_baseline %r on %s", raw, getattr(item, "id", "?"))
        return None
    scaled = value * BASELINE_SCALE
    if scaled != scaled.to_integral_value() or scaled > MAX_BASELINE:
        logger.debug("Not an integer hundredth: s2:processing_baseline %r on %s", raw, getattr(item, "id", "?"))
        return None
    return int(scaled)
