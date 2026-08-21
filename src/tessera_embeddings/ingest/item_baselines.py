"""Reading a Sentinel-2 item's processing baseline.

The single reader for ``s2:processing_baseline``, so the scale and the notion of "unreadable" are
defined once.

**Scale.** Reported as an integer hundredth: ``"04.00"`` is ``400``, ``"05.10"`` is ``510``. This is
the space :data:`~tessera_embeddings.config.satellites.S2_BASELINE_THRESHOLD` is expressed in, so a
comparison against the threshold needs no conversion.

**Validated as a version, not as a number.** A baseline is a two-part version — the catalogue emits
``"05.10"``, ``"02.06"`` — so this matches that shape and converts with integer arithmetic. Parsing
it into a general numeric type instead accepts a long list of things no version contains, and each
one becomes a *confident* answer rather than an admission of ignorance: ``"NaN"`` and ``"Infinity"``
parse, so do negatives, so do exponents that overflow or underflow when scaled, and excess precision
rounds onto a valid-looking value under the ambient decimal context. Matching the shape first makes
every one of those unreadable by construction.

**Unreadable.** ``None`` means the item does not declare a usable baseline. Callers substitute their
own default explicitly, which keeps "declared 0" distinguishable from "declared nothing"; a baseline
of 0 is below every threshold, so conflating them hides a correctness question.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: What one whole baseline step is worth on the reported scale, e.g. ``04.00`` -> ``400``.
BASELINE_SCALE = 100

#: A processing baseline: one or two ASCII digits, optionally a fractional part of one or two.
#: Deliberately narrow. The digit limits are what bound the value — no version reaches 100 — so
#: there is no separate range check to keep in step, and no sign, exponent or special value can
#: reach the arithmetic below. ``[0-9]`` rather than ``\d``, which also matches Unicode decimal
#: digits: an Arabic-Indic or fullwidth numeral is not a version, and ``int()`` would accept it.
_BASELINE_RE = re.compile(r"([0-9]{1,2})(?:\.([0-9]{1,2}))?")


def processing_baseline(item: Any) -> int | None:  # noqa: ANN401 — any STAC-like item
    """The baseline an item declares, as an integer hundredth, or ``None`` if it declares none.

    Non-string values are accepted by their textual form, so an integer ``4`` or a float ``4.0``
    read as ``400`` while an infinity reads as nothing.
    """
    properties = getattr(item, "properties", None)
    raw = properties.get("s2:processing_baseline") if isinstance(properties, dict) else None
    if raw is None:
        return None
    match = _BASELINE_RE.fullmatch(str(raw).strip())
    if match is None:
        logger.debug("Not a processing baseline: %r on %s", raw, getattr(item, "id", "?"))
        return None
    whole, fraction = match.group(1), (match.group(2) or "")
    # Padded rather than scaled: "05.1" and "05.10" are the same version, and a one-digit fraction
    # means tenths. Integer arithmetic throughout, so nothing can round.
    return int(whole) * BASELINE_SCALE + int(fraction.ljust(2, "0") or "0")
