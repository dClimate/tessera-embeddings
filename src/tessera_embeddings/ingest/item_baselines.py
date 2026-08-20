"""The ONE reader of an item's Sentinel-2 processing baseline.

This module exists because there were two of these, and they disagreed. One parsed
``s2:processing_baseline`` into a float (``5.1``) for ranking duplicate copies; the other scaled it
to an integer (``510``) for comparing against the correction threshold. Same string, two answers,
two notions of "unreadable" — and every numeric edge case had to be fixed twice. It was found once
they had drifted: one still accepted ``"1e308"`` as a baseline while the other had learned to
reject it.

**One scale.** The value is reported as an integer hundredth — ``"04.00"`` is ``400``, ``"05.10"``
is ``510``. That is the space :data:`S2_BASELINE_THRESHOLD` is expressed in, so a comparison
against the threshold needs no conversion, and a conversion is where a factor of a hundred goes
missing.

**One notion of unreadable.** ``None`` means the item does not tell us its baseline: the property
is absent, or empty, or not a number, or a number that cannot be represented on this scale.
Callers that need a number substitute their own default explicitly, which keeps "declared 0" and
"declared nothing" distinguishable — they are different facts and one of them is a correctness
question, since a baseline of 0 is below every threshold.
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

    Rejects ``"NaN"`` and ``"Infinity"``, which :func:`float` accepts and which are not baselines:
    every NaN comparison is false, so it leaves an ordering dependent on catalogue response order,
    and an infinity outranks every real value. Rejects a finite value that overflows when scaled
    — ``"1e308"`` passes a finiteness test on its own and becomes infinity multiplied by a
    hundred, where :func:`round` raises :class:`OverflowError` and would abort a whole batch over
    one item's metadata.
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
    # round() rather than int(), because 5.10 * 100 is 509.999... in binary floating point.
    return round(scaled)
