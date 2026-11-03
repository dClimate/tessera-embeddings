"""A minimal stand-in for a STAC item, for tests that drive the date loop without a catalogue.

The loop touches three things on an item: its ``datetime``, its ``eo:cloud_cover`` and its
``id``, which the ingest sort uses to break cloud-cover ties. Tests that need asset hrefs, a
footprint or a real :class:`pystac.Item` build their own.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace


def fake_stac_item(when: str, *, cloud: float = 0.0, item_id: str | None = None) -> SimpleNamespace:
    """An item acquired at ``when``: a date (taken at 10:00) or a full ISO timestamp."""
    acquired = datetime.fromisoformat(when if "T" in when else f"{when}T10:00:00")
    return SimpleNamespace(
        id=item_id or f"S2A_T13TDE_{acquired:%Y%m%d}_0_L2A",
        datetime=acquired,
        properties={"eo:cloud_cover": cloud},
    )
