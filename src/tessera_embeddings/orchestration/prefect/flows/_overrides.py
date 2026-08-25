"""Forwarding optional parameters to a child run without overriding its own defaults.

A flow that dispatches to another deployment has two different things to say about a
parameter, and the difference is load-bearing: *the caller chose this value* and *the
caller chose nothing, so whatever the child registered stands*. Prefect has one channel
for both — a key is either in the parameters dict or it is not — so "chose nothing" has
to be expressed by leaving the key out.

The rule that gives this module its reason to exist: **an omitted parameter is not the
same as one passed as its falsy value.** Written by hand the guard reads
``if value else {}``, which is right only while no legitimate value is falsy. It stops
being right the moment a parameter's domain includes ``0`` — and one here does:
``actor_request_batch_size`` documents ``0`` as the all-at-once mode. Applied at six
call sites across three flows, that guard silently dropped the mode at every one of
them, and each site looked correct in isolation.

So the rule lives here instead, tested once, keyed on ``None`` alone.
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
