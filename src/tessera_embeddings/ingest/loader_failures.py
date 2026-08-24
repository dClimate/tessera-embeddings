"""Keeping what a failed load knows: WHICH source object, and WHY.

Both are known only on the process that did the reading, and neither reaches the caller on
its own — one is never put in the exception, the other does not survive being serialised out
of the worker. Both are kept by code that has to be running BEFORE the read fails, on every
worker, so both are installed by :func:`install_capture_everywhere`: one call per ingest,
and the one place a third such rescue would go.

Naming the object
-----------------

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

Keeping the reason
------------------

A read failure's reason is its CAUSE. GDAL states it — a codec, a status, a refusal — and
rasterio wraps that in a generic ``Read failed. See previous exception for details.``, so the
wrapper alone says only that a read failed. Every verdict this pipeline reaches about a
failed read is therefore a verdict about the chain, never about the exception it arrives as.

The chain did not arrive. Dask serialises a failure with ``tblib``, which reconstructs an
exception class with a custom ``__init__`` by assigning to its ``args`` — and GDAL's classes
publish ``args`` as a read-only property, so the assignment throws and the whole chain fails
to unpickle. Dask notices before sending and substitutes a plain ``Exception`` holding the
outer exception's repr: one line of text, ``__cause__`` of ``None``, reason gone. So the
undecidable failure was not a gap in the classifier. It was the architecture discarding the
only evidence the classifier had to read.

:func:`keep_causes_picklable` removes the assignment tblib cannot make, by giving those
classes a ``__reduce__`` that rebuilds them from their own ``args`` through their own
constructor. Nothing else changes: the chain arrives, and the predicates that always wanted
to read it now can.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

import dask.distributed
from dask.distributed import WorkerPlugin

# Private because rasterio publishes no other name for it: `rasterio.errors` holds the
# wrappers, and every GDAL/CPL error that becomes a wrapper's cause lives here.
from rasterio._err import CPLE_BaseError

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


#: Exception classes whose cause cannot be serialised out of the worker that raised it, and
#: the one line to extend when another turns up. One entry suffices today because every
#: read-only ``args`` in the read stack belongs to a ``CPLE_BaseError`` subclass — a census
#: rather than a guess, recorded in the design doc. Patching the base covers the subclasses,
#: so a GDAL release adding error classes needs no change here.
_CAUSES_THAT_DO_NOT_TRAVEL: tuple[type[BaseException], ...] = (CPLE_BaseError,)


def _reduce_by_args(exc: BaseException) -> tuple[type[BaseException], tuple[Any, ...]]:
    """Rebuild ``exc`` by calling its own class on its own ``args``.

    Faithful for these classes rather than merely sufficient: GDAL's constructor takes
    exactly the triple its ``args`` reports, so the rebuilt exception carries the error
    class, the CPL error number and the message the codec wrote, not a flattened string.
    """
    return (type(exc), tuple(exc.args))


def keep_causes_picklable() -> None:
    """Let a read failure's cause survive serialisation out of THIS process. Idempotent.

    Installed on the process that will RAISE, which is the reading worker: Dask checks the
    round-trip before sending and substitutes the flattened form on the sending side, so a
    process that only receives cannot repair what it is handed.

    Defining ``__reduce__`` is what reaches tblib, which consults it and only falls back to
    assigning ``args`` for classes that define none. A ``copyreg`` reducer would not: tblib
    re-registers itself there for every class in the chain on every failure, and would
    overwrite ours each time.
    """
    for cls in _CAUSES_THAT_DO_NOT_TRAVEL:
        cls.__reduce__ = _reduce_by_args  # type: ignore[method-assign, assignment]


def rescues_are_installed() -> bool:
    """Whether THIS process can let a read failure's cause survive serialisation.

    The reducer only: the capture sharpens attribution, but the reducer decides whether a verdict
    is reachable at all.
    """
    return all(getattr(cls, "__reduce__", None) is _reduce_by_args for cls in _CAUSES_THAT_DO_NOT_TRAVEL)


class ReadRescuesNotInstalledError(RuntimeError):
    """A leg refused to start because its fleet cannot classify read failures.

    Left retryable on purpose: a scheduler that refused the plugin once usually accepts it on the
    next dispatch.
    """


class AbortedReadCapture(WorkerPlugin):
    """Installs both rescues on every current and future worker.

    A plugin rather than ``client.run``, for the same reason the credential broadcast is
    one: workers that join later — which under adaptive scaling is most of them — would
    otherwise read without capturing, and those are exactly the reads a wide fleet fails.
    """

    name = "loader-abort-capture"

    def setup(self, worker: object) -> None:  # noqa: ARG002 — plugin interface
        """Install both rescues on the worker this plugin has just been attached to."""
        install_capture()
        keep_causes_picklable()


#: Tries before refusing the leg — enough that a briefly unreachable scheduler costs seconds
#: rather than a cell.
_REGISTRATION_ATTEMPTS = 3

#: Seconds to wait after a failed attempt, doubling. Without a wait the three attempts complete
#: in the same instant, which is one attempt with extra steps: the case they exist for is a
#: scheduler that is restarting, and it cannot recover inside a microsecond. Bounded low because
#: this is paid at leg dispatch, before any fleet exists to sit idle.
_REGISTRATION_BACKOFF_S = 2.0


def install_capture_everywhere(client: dask.distributed.Client | None) -> None:
    """Install both rescues locally and across the cluster, and VERIFY they took.

    Raises :class:`ReadRescuesNotInstalledError` if any worker is reading without the reducer: an
    undecidable read failure strands a whole zone-year, so a fleet that cannot classify refuses to
    start. Verified rather than assumed, because registration returning is not the worker having
    run ``setup``, and the ask doubles as the barrier against reading before it has. See
    ``context_docs/design/ingest_read_failure_causes_2026_08.md``.
    """
    install_capture()
    keep_causes_picklable()
    if client is None:
        return

    last: Exception | None = None
    for attempt in range(1, _REGISTRATION_ATTEMPTS + 1):
        try:
            client.register_plugin(AbortedReadCapture())
            answers = client.run(rescues_are_installed)
        except Exception as exc:
            last = exc
            logger.warning(
                "Could not install the read-failure rescues on workers (attempt %d/%d)",
                attempt,
                _REGISTRATION_ATTEMPTS,
                exc_info=True,
            )
            if attempt < _REGISTRATION_ATTEMPTS:
                time.sleep(_REGISTRATION_BACKOFF_S * 2 ** (attempt - 1))
            continue
        unrescued = [w for w, ok in answers.items() if ok is not True]
        if not unrescued:
            return
        last = None
        logger.warning(
            "%d of %d worker(s) are reading without the read-failure rescues (attempt %d/%d)",
            len(unrescued),
            len(answers),
            attempt,
            _REGISTRATION_ATTEMPTS,
        )
        if attempt < _REGISTRATION_ATTEMPTS:
            time.sleep(_REGISTRATION_BACKOFF_S * 2 ** (attempt - 1))

    detail = f"registration kept failing ({last!r})" if last else "workers reported it missing"
    raise ReadRescuesNotInstalledError(
        "Refusing to read: this fleet cannot classify read failures because the cause-preserving "
        f"reducer is not installed on every worker after {_REGISTRATION_ATTEMPTS} attempts — "
        f"{detail}. Every read failure would arrive with its cause destroyed, so one corrupt "
        "object would strand this zone-year rather than cost a single date."
    )


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
