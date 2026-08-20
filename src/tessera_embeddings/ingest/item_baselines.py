"""Reading a Sentinel-2 item's processing baseline.

The single reader for ``s2:processing_baseline``, so the scale and the notion of "unreadable" are
defined once.

**Scale.** Reported as an integer hundredth: ``"04.00"`` is ``400``, ``"05.10"`` is ``510``. This is
the space :data:`~tessera_embeddings.config.satellites.S2_BASELINE_THRESHOLD` is expressed in, so a
comparison against the threshold needs no conversion.

**Unreadable.** ``None`` means the item does not declare a usable baseline — absent, empty,
non-numeric, or not representable on this scale. Callers substitute their own default explicitly,
which keeps "declared 0" distinguishable from "declared nothing"; a baseline of 0 is below every
threshold, so conflating them hides a correctness question.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

#: What one whole baseline step is worth on the reported scale, e.g. ``04.00`` -> ``400``.
BASELINE_SCALE = 100


def processing_baseline(item: Any) -> int | None:  # noqa: ANN401 — any STAC-like item
    """The baseline an item declares, as an integer hundredth, or ``None`` if it declares none.

    ``"NaN"`` and ``"Infinity"`` are rejected: :func:`float` accepts both, and neither is a
    baseline — every NaN comparison is false, and an infinity outranks every real value. A finite
    value that overflows when scaled is rejected for the same reason.
    """
    properties = getattr(item, "properties", None)
    raw = properties.get("s2:processing_baseline") if isinstance(properties, dict) else None
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.debug("Unparseable s2:processing_baseline %r on %s", raw, getattr(item, "id", "?"))
        return None
    scaled = value * BASELINE_SCALE if math.isfinite(value) else value
    if not math.isfinite(scaled):
        logger.debug("Unrepresentable s2:processing_baseline %r on %s", raw, getattr(item, "id", "?"))
        return None
    # round() rather than int(): 5.10 * 100 is 509.999... in binary floating point.
    return round(scaled)
