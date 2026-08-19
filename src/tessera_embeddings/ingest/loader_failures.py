"""Naming the source object a failed load could not read.

``odc.stac.load`` reports the object it gave up on in its OWN log record, and raises an
exception that does not carry it. So after a read failure the caller knows that *a* source
was unreadable but not WHICH — and the object is named only in a log line on whichever
reader process hit it, which in a fleet is one stream among hundreds.

Everything downstream of that gap has to guess. The duplicate ladder steps every duplicated
tile in the date down one copy, because it cannot tell which tile to step; a date carrying
hundreds of duplicated tiles therefore swaps hundreds of copies to recover from one bad
object, and a date whose bad object has no alternate at all still walks the whole ladder
before giving up.

This module closes the gap. A logging handler on each reader process records the hrefs the
loader aborted on, and the caller collects them after a failure and maps them back to the
tile-dates that produced them.

**Attribution is best effort, and no recovery may depend on it.** A worker that died with
the read, a cluster already gone when the caller asks, or a loader that words the message
differently all yield nothing. An empty answer must therefore mean "attribute nothing" and
leave the caller on its unattributed path — never "nothing was at fault".
"""

from __future__ import annotations

import logging
import re
import threading
from collections import deque
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

import dask.distributed
from dask.distributed import WorkerPlugin

from tessera_embeddings.ingest.duplicates import item_tile
from tessera_embeddings.ingest.solar_days import solar_day_of

logger = logging.getLogger(__name__)

#: Logger whose records name the aborted object. A handler here also sees records from its
#: children (``odc.loader._rio`` is the one that emits), so this survives odc moving the
#: message between its own modules.
_ODC_LOGGER = "odc"

#: The loader's own wording for "I gave up on this object". Kept as a cheap substring test
#: plus a capture, so the common case (no match) costs one ``in``.
_ABORT_MARKER = "failure while reading"
_ABORT_RE = re.compile(r"failure while reading:\s*(\S+)")

#: How many aborted hrefs one process retains. A failing date aborts once per affected load
#: task, so the same href repeats; a few hundred is far more than attribution needs and
#: bounds the memory a pathological run can hold.
_CAPACITY = 256

#: Bounded because it is written from the loader's threads and read from the caller's.
_aborted: deque[str] = deque(maxlen=_CAPACITY)
_lock = threading.Lock()
_handler: logging.Handler | None = None


def _strip_band_suffix(href: str) -> str:
    """Drop the ``:<band>`` the loader appends to the href it names.

    The message reads ``...B02.tif:1``. The trailing index identifies the band within the
    file, which is not part of the object's identity, and leaving it on defeats every
    comparison against a catalogue asset href.
    """
    return re.sub(r":\d+$", "", href)


def href_key(href: str) -> str:
    """A comparable identity for one source object, across URL spellings.

    A catalogue asset says ``s3://bucket/key`` while the loader logs
    ``https://bucket.s3.<region>.amazonaws.com/key``, and a path-style URL puts the bucket
    in the path instead. The last two path segments — the granule directory and the file —
    are the same in all three and are unique within a collection, so they are the key.
    """
    path = urlsplit(_strip_band_suffix(href)).path
    parts = [p for p in path.split("/") if p]
    return "/".join(parts[-2:])


class _AbortedReadHandler(logging.Handler):
    """Records the href from each of the loader's abort messages.

    A handler rather than a wrapper around the load call, because the loader reports the
    object from inside its own thread pool: nothing at the call site sees it.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover — a broken record must not break a read
            return
        if _ABORT_MARKER not in message:
            return
        match = _ABORT_RE.search(message)
        if match is None:
            return
        with _lock:
            _aborted.append(match.group(1))


def install_capture() -> None:
    """Start recording aborted hrefs in THIS process. Idempotent.

    Idempotence matters more than it looks: the worker plugin's ``setup`` runs again when a
    worker reconnects, and a second handler would double every entry.
    """
    global _handler
    if _handler is not None:
        return
    handler = _AbortedReadHandler(level=logging.ERROR)
    logging.getLogger(_ODC_LOGGER).addHandler(handler)
    _handler = handler


def drain_local() -> list[str]:
    """Take and clear this process's recorded hrefs.

    Draining rather than reading, so one failure's attribution cannot be inherited by the
    next one. A date that fails, steps down and fails again must attribute the second
    failure to the second attempt's objects only.
    """
    with _lock:
        found = list(_aborted)
        _aborted.clear()
    return found


class AbortedReadCapture(WorkerPlugin):
    """Installs the capture on every current and future worker.

    A plugin rather than ``client.run``, for the same reason the credential broadcast is
    one: workers that join later — which under adaptive scaling is most of them — would
    otherwise read without capturing, and those are exactly the reads a wide fleet fails.
    """

    name = "loader-abort-capture"

    def setup(self, worker: object) -> None:  # noqa: ARG002 — plugin interface
        """Install the capture on the worker this plugin has just been attached to."""
        install_capture()


def install_capture_everywhere(client: dask.distributed.Client | None) -> None:
    """Install the capture locally and across the cluster, tolerating a failure to do so.

    Never raises. Losing attribution costs precision in a recovery that still works
    without it, so a plugin registration that fails must not fail the ingest.
    """
    install_capture()
    if client is None:
        return
    try:
        client.register_plugin(AbortedReadCapture())
    except Exception:
        logger.warning("Could not install the aborted-read capture on workers", exc_info=True)


def collect_aborted_hrefs(client: dask.distributed.Client | None) -> list[str]:
    """Every aborted href recorded since the last collection, cluster-wide.

    Drains the local process and each worker. Worker errors are ignored rather than
    raised — this runs while handling a failure, and a second failure here would replace a
    recoverable read error with an unrecoverable one.
    """
    found = drain_local()
    if client is None:
        return found
    try:
        per_worker = client.run(drain_local, on_error="ignore")
    except Exception:
        logger.warning("Could not collect aborted hrefs from workers", exc_info=True)
        return found
    for result in per_worker.values():
        if isinstance(result, list):
            found.extend(h for h in result if isinstance(h, str))
    return found


#: How many objects a label names before it summarises the rest. One failing date aborts
#: once per affected load task, so the same handful of objects repeats; a label longer than
#: this is a symptom of many distinct objects failing at once, which the count conveys
#: better than the names do.
_OBJECT_LABEL_CAP = 8


def label_objects(hrefs: Iterable[str]) -> str:
    """A compact, deduplicated name for the objects a load aborted on.

    Granule directory and file only — the bucket and collection prefix are the same for
    every object in a run, so including them makes each entry four times longer without
    distinguishing anything.
    """
    keys = sorted({href_key(h) for h in hrefs})
    if not keys:
        return "unattributed"
    if len(keys) > _OBJECT_LABEL_CAP:
        return ",".join(keys[:_OBJECT_LABEL_CAP]) + f",+{len(keys) - _OBJECT_LABEL_CAP} more"
    return ",".join(keys)


def implicated_tile_dates(
    items: Iterable[Any],
    hrefs: Iterable[str],
) -> set[tuple[str, str]]:
    """The ``(tile, solar day)`` keys whose objects appear among ``hrefs``.

    Matched two ways because either can be the one that works: against the items' own asset
    hrefs, and against the granule id as a path segment. The second covers a catalogue whose
    asset hrefs are signed, aliased or otherwise not the URL the loader opened.

    An href that matches nothing is dropped silently. It means the object belongs to another
    date — plausible, since a fleet's workers read many dates — and attributing it here
    would step down a tile that read perfectly well.
    """
    return {(item_tile(it), solar_day_of(it)) for it in implicated_items(items, hrefs)}  # type: ignore[misc]


def implicated_items(items: Iterable[Any], hrefs: Iterable[str]) -> list[Any]:
    """The ITEMS whose objects appear among ``hrefs`` — what the tile-date keys reduce from.

    :func:`implicated_tile_dates` answers "which tile-dates failed", which is the granularity
    a caller reports at. This answers "which copies failed", which is the granularity a
    fallback should STEP at: a tile-date can hold several distinct acquisitions, and stepping
    the one whose alternate happens to rank highest downgrades a healthy acquisition to an
    older baseline while leaving the unreadable one selected.

    Same two-way matching and the same silent drop of an unmatched href, for the same
    reasons — see :func:`implicated_tile_dates`.
    """
    wanted = {href_key(h) for h in hrefs}
    if not wanted:
        return []
    segments = {seg for h in hrefs for seg in urlsplit(_strip_band_suffix(h)).path.split("/") if seg}

    matched: list[Any] = []
    for item in items:
        if item_tile(item) is None:
            continue
        if str(getattr(item, "id", "")) in segments:
            matched.append(item)
            continue
        assets = getattr(item, "assets", None) or {}
        hrefs_of_item = (getattr(a, "href", None) for a in assets.values())
        if any(h and href_key(h) in wanted for h in hrefs_of_item):
            matched.append(item)
    return matched
