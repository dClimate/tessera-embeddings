"""Forwarding optional parameters to a child run without overriding its own defaults.

Prefect has one channel for two different statements — *the caller chose this value* and
*the caller chose nothing, so the child's registered default stands* — so "chose nothing"
must be expressed by leaving the key out of the parameters dict.

**An omitted parameter is not the same as one passed as its falsy value.** The hand-written
guard ``if value else {}`` is right only while no legitimate value is falsy, and
``actor_request_batch_size`` documents ``0`` as the all-at-once mode: that guard silently
dropped the mode at all six call sites across three flows, each of which looked correct in
isolation. So the rule lives here, tested once, keyed on ``None`` alone.
"""

from __future__ import annotations

from typing import Any


def set_overrides(**candidates: Any) -> dict[str, Any]:  # noqa: ANN401 - forwards arbitrary flow parameters
    """Keep the parameters a caller actually set; drop the ones it left unset.

    Args:
        **candidates: Parameter names mapped to the caller's value, where ``None``
            means the caller expressed no preference.

    Returns:
        A dict for ``**``-splatting into a child run's parameters, holding only the
        keys whose value is not ``None``. ``0``, ``False`` and ``""`` are values a
        caller chose and are kept.
    """
    return {name: value for name, value in candidates.items() if value is not None}
