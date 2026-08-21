"""Telling a catalogue that is BUSY apart from one that cannot serve a given request.

Both arrive as the same exception type from the same endpoint, and they need opposite
responses. A busy upstream is what the campaign's expansive retry exists for: waiting
is the whole remedy, and refusing the cell early throws away coverage the source would
have given us. An upstream that answers one particular request with an error every time
cannot be outwaited — every further attempt re-runs a query whose answer is already
known, and the only thing more patience buys is spending the cell's attempt budget more
slowly while a fleet idles against it.

**Our layer sits ABOVE a retry ladder, not next to one.** ``_http.make_logging_retry``
mounts a ``urllib3`` ``Retry`` on the catalogue session, so every individual page fetch
is already retried with exponential backoff before anything here is reached. The
exception that escapes is therefore not one refusal: it is the ladder reporting that it
exhausted itself, which is a much stronger statement than a single error response and
must not be treated as a first attempt.

**What distinguishes the two, and what does not.** The status the ladder exhausted on
separates a stated overload (the upstream naming itself as the constraint) from a
backend failure (the upstream failing to produce an answer). That is a necessary
signal, not a sufficient one: a gateway can fail for minutes and recover. What settles
it is a REPEAT — the identical request, refused the identical way, on a later attempt.
So the taxonomy here classifies the refusal, and the caller that holds the attempt
budget supplies the repeat. Neither half is a verdict alone.

**The signature is the load-bearing part.** A refusal only counts as repeated if the
request is the same request, so the signature covers exactly the fields that decide
what was asked — collection, temporal window, area, page — and nothing that varies
between attempts. Including a counter, a timestamp or a retry ordinal would make every
refusal unique and the repeat check dead code that always passes.

The signature also has to survive a boundary that carries no Python objects: the leg
that queries and the layer that owns the attempt budget are separate runs, and the only
thing that crosses is the failure text. It is therefore emitted as one whitespace-free
token under a stable name, matched by name rather than by position, so a message that
gains a field around it still parses.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, NoReturn, final

from urllib3.exceptions import MaxRetryError, ResponseError

logger = logging.getLogger(__name__)

#: Statuses on which the upstream names ITSELF as the constraint. Waiting is the
#: remedy, so these stay retryable however often they recur — that is what the
#: campaign's expansive retry is for, and refusing on them loses coverage the source
#: would have served.
LOAD_REFUSAL_STATUSES = frozenset({429, 503})

#: Statuses on which the upstream failed to PRODUCE an answer. Retryable once, because
#: a gateway can fail transiently; deterministic in the request once the identical
#: request has been refused this way again.
UPSTREAM_ERROR_STATUSES = frozenset({500, 502, 504})

#: Name under which the signature is emitted into the failure text. Matched by NAME —
#: never by position within the message — so the human-readable detail around it can
#: change without breaking the parse.
REFUSAL_TOKEN = "CATALOGUE_REFUSAL"

_TOKEN_RE = re.compile(rf"{REFUSAL_TOKEN}=(\S+)")

# urllib3 formats its exhaustion cause from this template, so the status is recovered
# by reading urllib3's own constant back rather than by hard-coding its wording. A
# template change upstream therefore breaks the parse loudly at the regex, instead of
# silently classifying every refusal as UNKNOWN.
_EXHAUSTED_STATUS_RE = re.compile(
    re.escape(ResponseError.SPECIFIC_ERROR).replace(re.escape("{status_code}"), r"(\d{3})")
)


class RefusalKind(Enum):
    """Which of the two refusals this is, by what the upstream said.

    ``LOAD`` and ``UPSTREAM_ERROR`` are not severities of one thing; they are different
    claims about where the constraint lies, and the correct response to each is the
    wrong response to the other. ``UNKNOWN`` is neither claim and must behave like
    today's default — retryable — because a refusal we cannot read is not evidence.
    """

    LOAD = "load"
    UPSTREAM_ERROR = "upstream-error"
    UNKNOWN = "unknown"


@final
@dataclass(frozen=True)
class CatalogueRequest:
    """The identity of one catalogue request, in the fields that decide its answer.

    Carries what makes a request reproducible by someone who has only the log — an
    archive's operator included — and deliberately not the request body, which is
    dominated by fields that do not vary and is large enough to matter at the volume a
    zone-year of paging produces.

    Attributes:
        collection: Catalogue collection id the search names.
        window: Temporal window as the catalogue was asked for it, ``start/end``.
        area: Spatial term — a tile id where the query is property-based, a rounded
            bbox where it is spatial. Rounded because an unrounded float repr differs
            between attempts that asked for the same box.
        page: 1-based ordinal of the page fetch, so a failure says whether the first
            request or a deep cursor is what breaks. Zero for a failure that precedes
            any search (opening the catalogue itself).
    """

    collection: str
    window: str
    area: str
    page: int

    @property
    def signature(self) -> str:
        """Whitespace-free identity, stable across attempts asking the same thing."""
        return f"{self.collection}@{self.window}@{self.area}@p{self.page}"

    @property
    def label(self) -> str:
        """Human-readable form for the log line."""
        page = f"page {self.page}" if self.page else "catalogue root (before any search)"
        return f"collection={self.collection} window={self.window} {self.area} {page}"


@final
@dataclass(frozen=True)
class CatalogueRefusal:
    """How a catalogue refused, and whether the refusal came from an exhausted ladder.

    Attributes:
        kind: See :class:`RefusalKind`.
        status: HTTP status the refusal rests on, or ``None`` when none could be read.
        exhausted: True when the refusal is a retry ladder reporting its own
            exhaustion, rather than a single unretried response. The distinction
            matters to a reader deciding whether patience has already been spent.
    """

    kind: RefusalKind
    status: int | None
    exhausted: bool

    @property
    def signature(self) -> str:
        """Whitespace-free identity of the refusal itself."""
        return f"{self.kind.value}:{self.status if self.status is not None else 'none'}"

    @property
    def label(self) -> str:
        """Human-readable form for the log line."""
        where = "after exhausting the retry ladder" if self.exhausted else "without being retried"
        status = f"HTTP {self.status}" if self.status is not None else "an unreadable status"
        return f"{status} {where}"


@final
class CatalogueQueryError(RuntimeError):
    """A catalogue refused an identified request, classified.

    Exists so the request survives to the layer that decides whether to try again.
    The client library raises its transport failures with the request discarded — the
    message names the host and the endpoint path but not the search, and a search is a
    request body — so without this the failure cannot be narrowed, reproduced, or
    reported to whoever runs the archive.

    The message leads with the machine-readable token, because the only thing that
    crosses the boundary to the attempt-budget layer is text and a long human sentence
    ahead of the token is a truncation risk.
    """

    def __init__(self, request: CatalogueRequest, refusal: CatalogueRefusal, cause: BaseException) -> None:
        self.request = request
        self.refusal = refusal
        self._cause_text = str(cause)
        super().__init__(
            f"{REFUSAL_TOKEN}={refusal.signature}|{request.signature} "
            f"catalogue refused {request.label} with {refusal.label}: {cause}"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild from picklable state, since pickle's default rebuilds as ``cls(*args)``."""
        return (self.__class__, (self.request, self.refusal, RuntimeError(self._cause_text)))


@final
@dataclass(frozen=True)
class RefusalSignature:
    """A refusal recovered from failure TEXT, for a caller comparing attempts.

    Attributes:
        kind: The refusal's class, as re-read from the token.
        token: The whole signature, refusal and request together. Two attempts whose
            tokens are equal asked the same thing and were refused the same way.
    """

    kind: RefusalKind
    token: str


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """The exception and everything it was raised from or during, innermost last.

    Follows ``__context__`` as well as ``__cause__``: the client library re-raises its
    transport failures without ``from``, so the cause chain alone stops at the wrapper
    and the evidence sits one link further in. Guards against a cycle, which a
    hand-built chain can have.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _status_from_exhaustion(exc: BaseException) -> int | None:
    """The status a retry ladder exhausted on, read from urllib3's own cause text."""
    reason = getattr(exc, "reason", None)
    if not isinstance(reason, ResponseError):
        return None
    match = _EXHAUSTED_STATUS_RE.search(str(reason))
    return int(match.group(1)) if match else None


def classify_refusal(exc: BaseException) -> CatalogueRefusal:
    """Classify a catalogue failure by the status it rests on.

    Reads the exception CHAIN rather than the top-level message, because the status is
    structured data one or two links down and the top-level text is a stringification
    of it. Falls back to the message only when the chain carries no evidence, so a
    client library that discards the chain degrades to a weaker read instead of to no
    read at all.

    Never raises and never guesses: anything it cannot ground in a status is
    ``UNKNOWN``, which callers must treat as today's default rather than as a finding.
    """
    for link in _chain(exc):
        if isinstance(link, MaxRetryError) and (status := _status_from_exhaustion(link)) is not None:
            return CatalogueRefusal(_kind_for(status), status, exhausted=True)
        code = getattr(link, "status_code", None)
        if isinstance(code, int):
            return CatalogueRefusal(_kind_for(code), code, exhausted=False)
    # The chain is gone (a re-raise across a boundary that carries only text, a client
    # that raises `from None`). urllib3's exhaustion cause is still quoted inside the
    # message it stringified, so the weaker read is the same read on worse evidence.
    match = _EXHAUSTED_STATUS_RE.search(str(exc))
    if match:
        status = int(match.group(1))
        return CatalogueRefusal(_kind_for(status), status, exhausted=True)
    return CatalogueRefusal(RefusalKind.UNKNOWN, None, exhausted=False)


def _kind_for(status: int) -> RefusalKind:
    if status in LOAD_REFUSAL_STATUSES:
        return RefusalKind.LOAD
    if status in UPSTREAM_ERROR_STATUSES:
        return RefusalKind.UPSTREAM_ERROR
    return RefusalKind.UNKNOWN


def repeat_is_deterministic(kind: RefusalKind) -> bool:
    """Whether an IDENTICAL repeat of this refusal proves the request cannot be served.

    The policy lives here, with the taxonomy, so the layer holding the attempt budget
    supplies only the observation that a refusal repeated and does not also decide what
    a repeat means.

    True only for an upstream that failed to produce an answer. A repeat of a stated
    overload proves the opposite of a defect — the upstream is still busy, which is
    precisely the condition patience is for — and a refusal we could not classify is
    not evidence of anything.
    """
    return kind is RefusalKind.UPSTREAM_ERROR


def refusal_in(detail: str) -> RefusalSignature | None:
    """Recover a refusal signature from failure text, or ``None`` if it holds none.

    The counterpart to the token :class:`CatalogueQueryError` emits. Matched by token
    NAME so the message may gain or lose text around it, and returns ``None`` rather
    than a default for text that never carried a token — a failure that is not a
    catalogue refusal must not be classified as one.
    """
    match = _TOKEN_RE.search(detail)
    if match is None:
        return None
    token = match.group(1)
    kind_name = token.split(":", 1)[0]
    try:
        kind = RefusalKind(kind_name)
    except ValueError:
        return None
    return RefusalSignature(kind, token)


def bbox_area_label(bbox: tuple[float, float, float, float]) -> str:
    """A bbox as a stable, whitespace-free area term.

    Rounded, because the same requested box can stringify differently between attempts
    and a signature that varies with float repr can never repeat.
    """
    return "bbox=" + ",".join(f"{v:.4f}" for v in bbox)


def raise_catalogue_query_error(
    request: CatalogueRequest,
    cause: BaseException,
    *,
    log: logging.Logger | logging.LoggerAdapter | None = None,
    items_so_far: int | None = None,
) -> NoReturn:
    """Log the failing request, then raise it classified.

    One function for both so a refusal cannot be raised without being named: the log
    line is the only place the volatile context (how far the query had got, the
    traceback holding the transport cause) can go, since none of it belongs in a
    signature that has to be stable across attempts.
    """
    log = log or logger
    refusal = classify_refusal(cause)
    log.error(
        "CATALOGUE REFUSED %s with %s%s — classified %s. A load refusal is waited out; an "
        "upstream error that repeats on the identical request is not.",
        request.label,
        refusal.label,
        f" after {items_so_far} item(s)" if items_so_far is not None else "",
        refusal.signature,
        exc_info=cause,
    )
    raise CatalogueQueryError(request, refusal, cause) from cause
