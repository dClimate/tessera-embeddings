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

Keeping the reason GDAL did not raise
-------------------------------------

An intact chain is still only what the reader chose to RAISE, and GDAL does not raise
everything it knows. A refused object comes back as an error document where the imagery
should be, GDAL states the refusal in its own log and hands the document to the codec, and
the codec raises the only thing it can see: a decode failure. So the chain says
``ZIPDecode`` and the classifier says these bytes are bad, while the words that say the
service refused sit one log line away, never consulted.

The verdicts that follow are opposites. Unreadable means give the date up; refused means wait
and give up nothing — and a date given up below a store's append-only maximum cannot be
back-filled by the re-run meant to recover it. A whole outage's worth of recoverable dates can
therefore be spent on which of two errors GDAL happened to raise last.

So a second handler records the refusals GDAL logs, and :func:`carry_logged_refusal` attaches
them to the failing exception, where ``duplicates.classify_read_failure`` reads them with the
rest of the chain. Both sensors reach it through ``roi_processing.read_failure_context``, so one
classifier decides both, over the same evidence.

**And most of what GDAL logs is not on a logger.** rasterio's handler is installed per THREAD, so
GDAL's own fetch threads report to GDAL's process-wide handler instead, which writes to stderr and
to nothing a ``logging.Handler`` can reach. :func:`hear_gdal_from_every_thread` gives that handler
somewhere to forward to, in rasterio's own wording, so the handler above is what decides on those
lines too.

**Only refusals are recorded, and that is what makes the direction safe.** A line is kept only
if the classifier reads THAT LINE ALONE as ``PROVIDER_REFUSED`` or ``OUR_CREDENTIAL``. Refusal
is tested before anything about the bytes, so attaching such a line can only move a verdict
INTO those two — never into "unreadable" or "absent", and so never into giving a date up. What
it can cost is patience: a date that borrows another's refusal waits out the write budget and
then fails its leg with the axis unmoved, and is judged again, alone, on the re-dispatch.
"""

from __future__ import annotations

import atexit
import ctypes
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlsplit

import dask.distributed
import rasterio._base
from dask.distributed import WorkerPlugin

# Private because rasterio publishes no other name for either, and imported rather than restated
# so the wording this module forwards in cannot drift from the wording rasterio's own handler
# already uses: `code_map` names the CPL error class, `level_map` maps GDAL's error class onto a
# logging level.
from rasterio._env import code_map, level_map

# Private for the same reason: `rasterio.errors` holds the wrappers, and every GDAL/CPL error
# that becomes a wrapper's cause lives here.
from rasterio._err import CPLE_BaseError

from tessera_embeddings.ingest.duplicates import (
    ReadFailure,
    classify_read_failure_in,
    is_provider_refusal,
    is_source_read_failure,
    item_tile,
)
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

#: Logger GDAL's OWN messages reach. rasterio installs a CPL error handler that forwards GDAL's
#: errors and warnings here, which is the only place a refusal GDAL declined to raise is stated.
#:
#: **It does not reach every thread, and that is why** :func:`hear_gdal_from_every_thread` exists.
#: rasterio installs that handler with ``CPLPushErrorHandler``, which GDAL keeps per THREAD, so a
#: message emitted on a thread that never entered a ``rasterio.Env`` reaches no logger at all.
_GDAL_LOGGER = "rasterio._env"

#: How many distinct refusal lines one process retains. Far smaller than the href capacity
#: because these repeat almost exactly — an outage says the same sentence about every object —
#: and because every one of them can end up quoted on an exception.
_REFUSAL_CAPACITY = 16

#: The verdicts a line has to reach ON ITS OWN to be worth keeping. Both are ordered above every
#: statement about the bytes, which is what bounds what this capture can do to a verdict: it can
#: move one into these two and nowhere else, and neither of them gives a date up.
_KEEPABLE = (ReadFailure.PROVIDER_REFUSED, ReadFailure.OUR_CREDENTIAL)

#: Bounded because they are written from the loader's threads and read from the caller's.
_aborted: deque[str] = deque(maxlen=_CAPACITY)
#: Each entry is ``(recorded_at, message)``. The time is this WORKER's ``monotonic`` clock and is
#: only ever compared inside the process that wrote it, so no clock is compared across machines.
_refused: deque[tuple[float, str]] = deque(maxlen=_REFUSAL_CAPACITY)
_lock = threading.Lock()
#: Serialises the install in :func:`hear_gdal_from_every_thread`. Separate from ``_lock``
#: because the handlers that take that one run INSIDE the forwarder, so a lock held across the
#: install must not be one a forwarded message can ask for.
_install_lock = threading.Lock()

_handlers: list[logging.Handler] = []


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


class _LoggedRefusalHandler(logging.Handler):
    """Records the refusals GDAL states in its own log and does not raise.

    Filtered by the CLASSIFIER rather than by a marker list of its own, so there is one
    vocabulary for what a refusal is and this cannot drift from it. A line is kept only if it
    reaches :data:`_KEEPABLE` read alone — which also settles corroboration, since GDAL's
    wording carries its own ``CPLE_`` class name and GDAL reads nothing here but source imagery.

    Everything else GDAL says is dropped, and that is the safety property rather than tidiness:
    a benign 404 probing for a sidecar, kept, would be evidence that an object is ABSENT.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
            if classify_read_failure_in(message) not in _KEEPABLE:
                return
        except Exception:  # pragma: no cover — a broken record must not break a read
            return
        with _lock:
            _refused.append((time.monotonic(), message))


#: GDAL error classes worth forwarding. ``CE_Warning`` and ``CE_Failure`` are the two a refusal is
#: ever stated at. ``CE_Debug`` is left alone because forwarding it would put GDAL's whole debug
#: stream on a logger, and ``CE_Fatal`` because the handler already installed answers it by
#: aborting the process — a response this must not take over or pre-empt.
_CE_WARNING = 2
_CE_FAILURE = 3

#: A CPL error handler: ``void (*)(CPLErr, CPLErrorNum, const char *)``.
_CPL_ERROR_HANDLER = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_char_p)

#: Every forwarder ever installed, held at module scope for as long as GDAL may call it — GDAL
#: keeps the function POINTER, so nothing else keeps the callback object alive. Doubles as the
#: idempotence flag, since a second install would chain the forwarder to itself.
#:
#: **A list that is only ever appended to, rather than one slot.** A slot makes the callback's
#: lifetime depend on nothing ever reassigning it: overwrite it while GDAL and a chained handler
#: still hold the old address and the next GDAL error calls into freed memory, which recurses
#: through the chain rather than failing cleanly. Appending makes that structurally impossible
#: instead of a rule to remember.
_forwarders: list[Any] = []


def hear_gdal_from_every_thread() -> None:
    """Forward the GDAL messages no rasterio handler is on the stack for. Idempotent.

    rasterio installs its logging handler with ``CPLPushErrorHandler``, and GDAL keeps that stack
    **per thread**. GDAL's ranged reader fetches on threads of its own, which never enter a
    ``rasterio.Env``, so their messages fall through to GDAL's process-wide handler — which writes
    to the process's stderr and to no logger at all. :class:`_LoggedRefusalHandler` is a
    ``logging.Handler``, so it cannot see them: not for want of a level or a wording, but because
    the message was never a log record.

    A refusal stated only there is therefore invisible to the capture, and the read it belongs to
    is judged on its exception alone — which is how a transient refusal comes to be given up as
    unreadable. See ``context_docs/design/ingest_read_failure_causes_2026_08.md``.

    **Chained, not replaced.** The handler already installed is called first, so GDAL's stderr line
    still appears where an operator greps for it and ``CE_Fatal`` still aborts through it. All this
    adds is a second copy, on the logger the capture listens on.

    **In rasterio's own wording, from rasterio's own** ``code_map``. The classifier's corroboration
    that a line is the source READER's is GDAL's vocabulary, so ``CPLE_AppDefined in <message>``
    carries the ``CPLE_`` that a bare GDAL sentence does not — which is what lets such a line be
    judged on its own, and is why this borrows the format instead of inventing one.

    **Not levelled the way rasterio levels it, deliberately.** rasterio downgrades ``CE_Failure``
    to ``INFO`` because one of its calls may emit several and still succeed. A message reaching the
    process-wide handler has no such call to judge it, and ``INFO`` is below both the logger's
    default level and the capture's, so honouring the downgrade would record it nowhere.

    **Installed once, under a lock.** GDAL keeps the function POINTER, and only the module-level
    reference keeps the callback object alive; two installs would leave the first collectable while
    GDAL and the chained handler still hold its address, so the next GDAL error would call into
    freed memory. The check and the install therefore have to be one critical section.

    Never raises. A process that cannot install this reads exactly as it did before it.
    """
    with _install_lock:
        if _forwarders:
            return
        gdal_log = logging.getLogger(_GDAL_LOGGER)
        # Reassigned below to whatever was installed; read by `forward` through the closure, so the
        # chain is live from the moment it exists and is simply empty for the instant before.
        already: Any = None

        def forward(err_class: int, err_no: int, message: bytes) -> None:
            try:
                if already is not None:
                    already(err_class, err_no, message)
                if err_class not in (_CE_WARNING, _CE_FAILURE):
                    return
                text = message.decode("utf-8", "replace")
                if err_no in code_map:
                    gdal_log.log(level_map[err_class], "%s in %s", code_map[err_no], text)
                else:
                    gdal_log.log(level_map[err_class], "%s", text)
            except Exception:  # pragma: no cover — GDAL is C, and cannot be handed an exception
                return

        try:
            # Through the rasterio extension that LINKS GDAL rather than by a library filename:
            # rasterio's wheels ship GDAL version-stamped inside a private directory whose name
            # differs by platform and release, while `dlsym` on a loaded extension searches what that
            # extension was linked against. So this asks the one thing whose path Python already knows.
            gdal = ctypes.CDLL(rasterio._base.__file__)
            gdal.CPLSetErrorHandler.restype = _CPL_ERROR_HANDLER
            gdal.CPLSetErrorHandler.argtypes = [_CPL_ERROR_HANDLER]
            forwarder = _CPL_ERROR_HANDLER(forward)
            already = gdal.CPLSetErrorHandler(forwarder)
            _forwarders.append(forwarder)
        except Exception:
            logger.warning(
                "Could not forward GDAL's process-wide messages to %s: a refusal stated on a thread "
                "rasterio's handler does not cover will not reach the capture, and the read it "
                "belongs to will be judged on its exception alone.",
                _GDAL_LOGGER,
                exc_info=True,
            )
            return

        # Put GDAL's own handler back while the interpreter is still there to be called into. A
        # callback GDAL invokes from one of its threads has to take the GIL, and taking it after
        # finalisation has begun is fatal to the process rather than to the read.
        atexit.register(gdal.CPLSetErrorHandler, already)


def install_capture() -> None:
    """Start recording aborted hrefs and logged refusals in THIS process. Idempotent.

    Idempotence matters more than it looks: the worker plugin's ``setup`` runs again when a
    worker reconnects, and a second handler would double every entry.

    A handler sees a record only if the logger it sits on is enabled for that level, and
    GDAL's refusals arrive as warnings — nothing in this package raises a logger above
    ``WARNING``, and lowering GDAL's own would emit its whole warning stream to the root
    handlers, so the level is read rather than set.

    The forwarder goes in alongside them because it is the same sensor: these handlers hear what
    GDAL says to a logger, and it is what makes GDAL say the rest of it to one.
    """
    hear_gdal_from_every_thread()
    if _handlers:
        return
    for name, handler in (
        (_ODC_LOGGER, _AbortedReadHandler(level=logging.ERROR)),
        (_GDAL_LOGGER, _LoggedRefusalHandler(level=logging.WARNING)),
    ):
        logging.getLogger(name).addHandler(handler)
        _handlers.append(handler)


def _drain(buffer: deque[str]) -> list[str]:
    """Take and clear one of this process's buffers.

    Draining rather than reading, so one failure's evidence cannot be inherited by the next
    one. A date that fails, steps down and fails again must be judged on the second attempt's
    objects only.
    """
    with _lock:
        found = list(buffer)
        buffer.clear()
    return found


def drain_local() -> list[str]:
    """Take and clear this process's recorded hrefs."""
    return _drain(_aborted)


def clear_local_refusals() -> None:
    """Empty this process's refusal buffer outright.

    For a caller that needs a known-empty starting point — a test between cases, never the read
    path. See :func:`read_local_refusals` for why.
    """
    with _lock:
        _refused.clear()


#: How long a recorded refusal is kept. It has to outlast the longest single read — the read
#: ladder is :data:`~tessera_embeddings.ingest.roi_processing.SOURCE_READ_ATTEMPTS` attempts and
#: the radar write may then wait out :data:`~tessera_embeddings.storage.zarr_store.WAIT_OUT_BACKOFF_S`
#: on top — while being far shorter than a leg, so a buffer cannot accumulate a whole day. The
#: ``maxlen`` on the buffer bounds it a second way, by count.
_REFUSAL_RETENTION_S = 3600.0


def read_local_refusals() -> list[tuple[float, str]]:
    """This process's recorded refusal lines with their AGES, without consuming them.

    Reading rather than draining, and that is the whole safety property. Two reads are in flight
    at once whenever the optical path pipelines a date: the look-ahead prepares date N+1, whose
    coverage gate reads, while date N's write is still reading. Both enter their own
    ``read_failure_context``, and a destructive collection let whichever failed FIRST take the
    other's evidence — the second failure then saw only the codec's complaint, classified as
    unreadable data, and gave its date up. Reading leaves the line for the other to find.

    **Ages, not timestamps, and the decision is not taken here.** This runs on a worker and the
    caller runs somewhere else, so their clocks are not comparable; an elapsed time is. Whether a
    line is recent enough to describe the read being judged is decided by the caller, against its
    own clock, AFTER this answer arrives — see :func:`collect_logged_refusals`. Filtering here
    against a duration the caller measured before the round trip discarded evidence whenever the
    trip took longer than the read, which classified a refusal as unreadable data and cost a date.

    Separate from the href buffer, which is still drained destructively: one buffer would mean the
    caller that classifies destroys the evidence the optical copy ladder attributes from.
    """
    now = time.monotonic()
    with _lock:
        # Eviction is the only thing that removes a line, and it happens on the way past rather
        # than on a timer: nothing in flight can be evicted, because the horizon is longer than
        # any read that could still be judging it.
        while _refused and now - _refused[0][0] > _REFUSAL_RETENTION_S:
            _refused.popleft()
        return [(now - at, message) for at, message in _refused]


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

    The reducer only. Both captures matter to a verdict — the href sharpens attribution, and the
    refusal decides whether a refusal GDAL only logged is reachable at all — but neither is gated
    here. A worker missing them reads as the fleet used to, which costs dates in an outage; a
    worker missing the reducer cannot classify anything, which strands a zone-year. Only the
    second is worth refusing a leg over, and installation is one call, so in practice they travel
    together.
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


def _collect[T](client: dask.distributed.Client | None, read: Callable[[], list[T]], what: str) -> list[T]:
    """Read one buffer on the local process and on each worker, never raising.

    This runs while handling a failure, so a second failure here would replace a recoverable read
    error with an unrecoverable one. A worker that could not answer is COUNTED rather than dropped
    — see the call below for why that distinction is the whole point.
    """
    found = read()
    if client is None:
        return found
    try:
        # ``"return"`` and not ``"ignore"``: both refuse to raise, which is the requirement here —
        # a second failure while handling a read failure would replace a recoverable error with an
        # unrecoverable one. But ``"ignore"`` REMOVES the failed worker from the result dict, so a
        # worker that died holding the only refusal line is indistinguishable from one that had
        # nothing to give. ``"return"`` puts the exception in its place, which is what lets the
        # count below exist at all.
        per_worker = client.run(read, on_error="return")
    except Exception:
        logger.warning("Could not collect %s from workers", what, exc_info=True)
        return found
    silent = 0
    for result in per_worker.values():
        if isinstance(result, list):
            found.extend(result)
        else:
            silent += 1
    if silent:
        # NAMED, because the buffer lives on the worker that read, and a worker that dies or is
        # retired between the failure and this call takes its evidence with it. Returning a short
        # answer in silence would let a refusal be judged as bad bytes with nothing recording that
        # the evidence was merely unreachable. This line is what makes that diagnosable.
        logger.warning(
            "Collected %s from %d of %d worker(s): %d did not answer, so any evidence they held is not in this verdict",
            what,
            len(per_worker) - silent,
            len(per_worker),
            silent,
        )
    return found


def collect_aborted_hrefs(client: dask.distributed.Client | None) -> list[str]:
    """Every aborted href recorded since the last collection, cluster-wide."""
    return _collect(client, drain_local, "aborted hrefs")


def collect_logged_refusals(client: dask.distributed.Client | None, since: float | None = None) -> list[str]:
    """Every refusal GDAL logged since ``since``, cluster-wide.

    ``since`` is a ``time.monotonic`` reading on THIS process, taken when the read began. Each
    worker reports how OLD its lines are by its own clock, and the cutoff is computed here, after
    the round trip, against this process's clock. So nothing depends on the fleet's clocks
    agreeing, and nothing depends on the round trip being quick.

    **The cutoff has to be taken after the call, not before it.** Measured before, it excluded the
    scheduling and RPC latency that the workers' ages already include, so a line logged during a
    short read looked older than the read itself and was thrown away — leaving the codec exception
    classified as unreadable data, which costs the date. The error that remains is the return leg
    alone, and it biases towards KEEPING a line, which costs a wait rather than a date.
    """
    aged = _collect(client, read_local_refusals, "logged refusals")
    if since is None:
        return [message for _age, message in aged]
    cutoff = time.monotonic() - since
    return [message for age, message in aged if age <= cutoff]


#: How the attached evidence introduces itself in a log and on an exception. A reader has to be
#: able to tell a line the reader COPIED from a line the reader was given.
_REFUSAL_NOTE = "The source reader logged, but did not raise:"

#: Distinct lines one note quotes. An outage repeats one sentence per object, so the first few
#: distinct ones carry the whole finding and the rest are volume on a store attribute and in a
#: traceback.
_LINES_PER_NOTE = 4


def carry_logged_refusal(
    exc: BaseException, client: dask.distributed.Client | None, since: float | None = None
) -> None:
    """Attach to ``exc`` any refusal GDAL logged for this read but did not raise.

    The single bridge between the evidence and the verdict, and the only writer of exception
    notes here. Every predicate in ``duplicates.py`` reads the chain's text, notes included, so
    attaching is what makes one classifier decide from the whole of what is known — rather than
    each caller re-deciding from whatever it happens to hold.

    **Attached, not merely read.** A caller that classified and discarded would leave the next
    reader of the same exception a different verdict from the same failure, and the radar path
    has two readers: the retry policy that spends patience, and the handler that decides whether
    a date is lost. Written onto the exception, the evidence travels with it.

    Only onto a SOURCE READ failure, per :func:`is_source_read_failure`. A store conflict or a
    duplicate date raised while some other read is being refused would otherwise be explained by
    that refusal, and answered with a wait that fixes nothing.

    ``since`` is when the read being judged began, by this process's ``time.monotonic``; what that
    buys and what it costs is documented on :func:`read_local_refusals`.

    The evidence is cluster-wide, so a date failing while ANOTHER date's read is being refused can
    take the other's lines into its note. That is the same race the href attribution runs, bounded
    the same way — by what a wrong answer costs. A borrowed refusal buys a write the refusal budget
    and then fails the leg with the time axis unmoved; nothing is skipped, and the date is judged
    alone on the re-dispatch.
    """
    if not is_source_read_failure(exc):
        return
    lines = collect_logged_refusals(client, since)
    if not lines:
        return
    note = f"{_REFUSAL_NOTE} {' | '.join(sorted(set(lines))[:_LINES_PER_NOTE])}"
    if note not in getattr(exc, "__notes__", ()):
        exc.add_note(note)


def refusal_wait_out(client: dask.distributed.Client | None) -> Callable[[BaseException], bool]:
    """The ``wait_out`` predicate for a write whose read failures may be refusals GDAL only logged.

    :func:`~tessera_embeddings.ingest.duplicates.is_provider_refusal` unchanged — the verdict is
    still one classifier's — over evidence gathered at the moment the question is asked. That
    timing is the whole of it: the retry policy asks PER ATTEMPT, so a predicate reading only the
    exception decides a refusal it cannot see, declines, and spends the ordinary three attempts
    on an outage the budget exists to outlast.
    """
    since = time.monotonic()

    def _refused(exc: BaseException) -> bool:
        # The gap between two calls IS the attempt that just failed, and on the first call it is
        # the time since the write began. So the evidence considered is always the evidence this
        # attempt produced, without the predicate having to be told when the attempt started.
        nonlocal since
        started, since = since, time.monotonic()
        carry_logged_refusal(exc, client, since=started)
        return is_provider_refusal(exc)

    return _refused


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
