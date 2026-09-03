"""Telling a catalogue that is BUSY apart from one that cannot serve a given request.

Both arrive as the same exception type from the same endpoint and need opposite responses.
Waiting is the whole remedy for a busy upstream, and refusing the cell early throws away
coverage the source would have served. An upstream that answers one particular request with an
error every time cannot be outwaited: more patience only spends the cell's attempt budget more
slowly while a fleet idles against it. Incidents and full derivation:
``context_docs/design/ingest_read_failure_causes_2026_08.md`` (causes 3 and 6).

**Our layer sits ABOVE a retry ladder, and only partly behind it.** ``_http.make_logging_retry``
mounts a ``urllib3`` ``Retry`` on the catalogue session, so a page fetch failing on a status in
that ladder's list is already retried with exponential backoff before anything here is reached;
what escapes is the ladder reporting its own EXHAUSTION, a much stronger statement than a single
error response. A status kept OUT of that list arrives here on its first refusal instead — the
right shape when the remedy is to ask a different request rather than the same one again, and why
:mod:`~tessera_embeddings.ingest.stac` excludes the 502 it re-cuts date windows for.
:attr:`CatalogueRefusal.exhausted` records which of the two happened.

**Status is necessary, not sufficient.** It separates a stated overload from a backend failure,
but a gateway can fail for minutes and recover. What settles it is a REPEAT — the identical
request refused the identical way on a later attempt — so this module classifies and the caller
holding the attempt budget supplies the repeat.

**The signature is the load-bearing part.** It covers exactly the fields that decide what was
asked — collection, temporal window, area, page — and nothing that varies between attempts; a
counter, timestamp or retry ordinal would make every refusal unique and the repeat check dead
code. It also has to cross a boundary carrying no Python objects (the querying leg and the
attempt-budget layer are separate runs, and only the failure text crosses), so it is emitted as
one whitespace-free token under a stable name, matched by name rather than position.
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

#: Statuses on which the upstream names ITSELF as the constraint. Waiting is the remedy, so these
#: stay retryable however often they recur; refusing on them loses coverage the source would serve.
LOAD_REFUSAL_STATUSES = frozenset({429, 503})

#: Statuses on which the upstream failed to PRODUCE an answer. Retryable once, because
#: a gateway can fail transiently; deterministic in the request once the identical
#: request has been refused this way again.
UPSTREAM_ERROR_STATUSES = frozenset({500, 502, 504})

#: The subset meaning "the answer you asked for was too big", as opposed to "I broke". Only these
#: are worth re-asking as a SMALLER request, and the reason is cost: 500 and 504 are force-listed,
#: so one reaches us only after the ladder spent its whole backoff, and re-cutting it hands that
#: same ladder to every child — a persistent 500 would multiply minutes of backoff across a
#: recursion instead of failing the leg once and letting the attempt budget decide.
OVERSIZED_RESPONSE_STATUSES = frozenset({502})

#: Statuses that are a THROTTLE from some providers and an authorization verdict from others, so
#: they are named per provider and never globally. On a public unauthenticated catalogue a 403
#: cannot be about who is asking, and waiting is the remedy, as for 429; where it IS about who is
#: asking, a backoff ladder is spent to learn nothing. The default names neither.
THROTTLE_STATUSES = frozenset({403})

#: Name under which the signature is emitted into the failure text. Matched by NAME — never by
#: position — so detail around it can change without breaking the parse.
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

    ``LOAD`` and ``UPSTREAM_ERROR`` are not severities of one thing but different claims about
    where the constraint lies, and the right response to each is the wrong response to the other.
    ``UNKNOWN`` is neither claim and stays retryable: a refusal we cannot read is not evidence.
    """

    LOAD = "load"
    UPSTREAM_ERROR = "upstream-error"
    UNKNOWN = "unknown"


@final
@dataclass(frozen=True)
class CatalogueRequest:
    """The identity of one catalogue request, in the fields that decide its answer.

    Carries what makes a request reproducible by someone who has only the log — an archive's
    operator included — and deliberately not the request body, which is dominated by fields that
    do not vary and is large enough to matter at a zone-year's paging volume.

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

    Exists so the request survives to the layer that decides whether to try again. The client
    library raises transport failures with the request discarded — its message names the host and
    endpoint path but not the search, which is a request body — so without this the failure cannot
    be narrowed, reproduced, or reported to whoever runs the archive.

    The message LEADS with the machine-readable token: only text crosses the boundary to the
    attempt-budget layer, and a long human sentence ahead of the token is a truncation risk.
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

    Follows ``__context__`` as well as ``__cause__``: the client library re-raises transport
    failures without ``from``, so the cause chain alone stops at the wrapper and the evidence sits
    one link further in. Guards against a cycle, which a hand-built chain can have.
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


def classify_refusal(exc: BaseException, *, throttle_statuses: frozenset[int] = frozenset()) -> CatalogueRefusal:
    """Classify a catalogue failure by the status it rests on.

    ``throttle_statuses`` names the statuses THIS catalogue uses to mean "slow down" among those
    that mean something else elsewhere (see :data:`THROTTLE_STATUSES`); they classify ``LOAD``, as
    429 does and for the same reason. Empty by default, so a caller that does not know which
    catalogue it is talking to cannot put an authorization refusal on a backoff ladder.

    Reads the exception CHAIN rather than the top-level message, since the status is structured
    data one or two links down and the top-level text is a stringification of it. Falls back to the
    message only when the chain carries no evidence.

    Never raises and never guesses: anything it cannot ground in a status is ``UNKNOWN``, which
    callers must treat as today's default rather than as a finding.
    """
    for link in _chain(exc):
        if isinstance(link, MaxRetryError) and (status := _status_from_exhaustion(link)) is not None:
            return CatalogueRefusal(_kind_for(status, throttle_statuses), status, exhausted=True)
        code = getattr(link, "status_code", None)
        if isinstance(code, int):
            return CatalogueRefusal(_kind_for(code, throttle_statuses), code, exhausted=False)
    # The chain is gone (a re-raise across a boundary that carries only text, a client
    # that raises `from None`). urllib3's exhaustion cause is still quoted inside the
    # message it stringified, so the weaker read is the same read on worse evidence.
    match = _EXHAUSTED_STATUS_RE.search(str(exc))
    if match:
        status = int(match.group(1))
        return CatalogueRefusal(_kind_for(status, throttle_statuses), status, exhausted=True)
    return CatalogueRefusal(RefusalKind.UNKNOWN, None, exhausted=False)


def _kind_for(status: int, throttle_statuses: frozenset[int]) -> RefusalKind:
    if status in LOAD_REFUSAL_STATUSES or status in throttle_statuses:
        return RefusalKind.LOAD
    if status in UPSTREAM_ERROR_STATUSES:
        return RefusalKind.UPSTREAM_ERROR
    return RefusalKind.UNKNOWN


def is_oversized_response(refusal: CatalogueRefusal) -> bool:
    """Whether asking for a SMALLER answer is a sensible response to this refusal.

    True only for the status a catalogue uses to refuse an over-large response. A backend that
    merely failed is no likelier to succeed when asked for less, and has already cost a full retry
    ladder by the time we see it.
    """
    return refusal.status in OVERSIZED_RESPONSE_STATUSES


def repeat_is_deterministic(kind: RefusalKind) -> bool:
    """Whether an IDENTICAL repeat of this refusal proves the request cannot be served.

    The policy lives here, with the taxonomy, so the layer holding the attempt budget only
    OBSERVES that a refusal repeated and does not also decide what a repeat means.

    True only for an upstream that failed to produce an answer. A repeated stated overload proves
    the opposite of a defect — the upstream is still busy, the condition patience is for — and an
    unclassifiable refusal is not evidence of anything.
    """
    return kind is RefusalKind.UPSTREAM_ERROR


def refusal_in(detail: str) -> RefusalSignature | None:
    """Recover a refusal signature from failure text, or ``None`` if it holds none.

    The counterpart to the token :class:`CatalogueQueryError` emits. Matched by token NAME so the
    message may gain or lose text around it, and returns ``None`` rather than a default for text
    that never carried a token — a failure that is not a catalogue refusal must not read as one.
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
    throttle_statuses: frozenset[int] = frozenset(),
) -> NoReturn:
    """Log the failing request, then raise it classified.

    One function for both so a refusal cannot be raised without being named: the log line is the
    only place the volatile context (how far the query had got, the traceback holding the transport
    cause) can go, none of it belonging in a signature that must be stable across attempts.
    """
    log = log or logger
    refusal = classify_refusal(cause, throttle_statuses=throttle_statuses)
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
