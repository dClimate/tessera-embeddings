"""Shared retry and abandonment helpers for ingest catalogue/granule queries."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any, final

import requests
from urllib3.connectionpool import ConnectionPool
from urllib3.response import BaseHTTPResponse
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

#: How much of an unparseable body to quote. Enough to tell an HTML error page from a
#: truncated document from an empty one, and short enough to leave a log line readable.
_EXCERPT_CHARS = 300


@final
class NonJsonResponseError(RuntimeError):
    """A provider returned a SUCCESS status and a body that is not JSON.

    Its own class because it is invisible to every status-based defence in this package. A
    ``urllib3`` ``Retry`` mounted with a ``status_forcelist`` never sees it, and neither
    does ``raise_for_status``: the status IS a success and only the body is wrong. So a
    truncated or error-page-substituted answer escapes the retry ladder untouched and
    reaches the caller as a bare ``JSONDecodeError`` reading ``Expecting value: line 1
    column 1 (char 0)`` — naming neither the endpoint, nor the status, nor what actually
    arrived. That one line is the whole diagnostic content of the failure, and ingest
    parses JSON from several different endpoints, so it does not even say which one broke.

    Every attribute here exists to answer that question at the next occurrence.

    Attributes:
        url: Endpoint that answered, so the failure names its own source.
        status: The success status the body arrived under.
        content_type: What the provider CLAIMED to send. ``application/json`` beside an
            unparseable body means truncation; ``text/html`` means an error page.
        body_length: Bytes received. Zero is an empty body; a large value beside a parse
            failure at character 0 is a body that was never JSON to begin with.
        excerpt: Leading bytes of the body, or ``None`` where the caller suppressed them
            (see :func:`json_or_raise`). Deliberately NOT in the message — see below.

    The excerpt is an attribute and never part of the message, because the message is
    matched as a SUBSTRING to decide whether a failed leg may be retried
    (``ingest_zone_year._is_retryable_leg_failure``). The excerpt is text the PROVIDER
    chose, so putting it in the message lets an error page that happens to contain a
    non-retryable marker word — ``ValidationError`` is one — turn a retryable leg into a
    permanently dead one, which costs a zone-year of coverage. Callers log the excerpt
    beside the failure instead, where it is just as readable and steers nothing.
    """

    def __init__(
        self,
        *,
        url: str,
        status: int,
        content_type: str,
        body_length: int,
        excerpt: str | None,
    ) -> None:
        self.url = url
        self.status = status
        self.content_type = content_type
        self.body_length = body_length
        self.excerpt = excerpt
        super().__init__(
            f"{url} answered HTTP {status} with a body that is not JSON "
            f"(content-type {content_type!r}, {body_length} byte(s))."
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Rebuild through a module-level function, since pickle's default cannot.

        Dask serialises a worker's failure before sending it and rebuilds an exception as
        ``cls(*args)``. This constructor is keyword-only, so that call would raise and the
        caller would be handed a flattened substitute instead of the fields above — which
        are the entire reason the class exists.
        """
        return (
            _rebuild_non_json_response_error,
            (self.url, self.status, self.content_type, self.body_length, self.excerpt),
        )


def _rebuild_non_json_response_error(
    url: str, status: int, content_type: str, body_length: int, excerpt: str | None
) -> NonJsonResponseError:
    """Positional trampoline for :meth:`NonJsonResponseError.__reduce__`."""
    return NonJsonResponseError(
        url=url, status=status, content_type=content_type, body_length=body_length, excerpt=excerpt
    )


def json_or_raise(resp: requests.Response, *, quote_body: bool) -> Any:  # noqa: ANN401 — whatever the JSON holds
    """Parse ``resp`` as JSON, or raise :class:`NonJsonResponseError` naming what arrived.

    ``quote_body`` decides whether the body is CAPTURED at all, and is required rather than
    defaulted because the safe answer differs per endpoint and a default would silently pick
    one for a caller that never considered it. Capture a PUBLIC catalogue's body: it holds
    nothing secret, and it is what tells truncation apart from an error page. Never capture a
    credential endpoint's — a document truncated mid-write is precisely the case that fails to
    parse, and the leading bytes of a credential document are the credential.
    """
    try:
        return resp.json()
    except ValueError as exc:
        # ValueError rather than JSONDecodeError: `requests` raises its own subclass, and
        # which JSON library backs it varies with the install. Both are ValueErrors.
        excerpt = resp.content[:_EXCERPT_CHARS].decode("utf-8", errors="replace") if quote_body else None
        raise NonJsonResponseError(
            url=resp.url,
            status=resp.status_code,
            content_type=resp.headers.get("Content-Type", "<absent>"),
            # Measured off `content` so an oversized body is not decoded in full to be sized.
            body_length=len(resp.content),
            excerpt=excerpt,
        ) from exc


def make_logging_retry(label: str, **kwargs: object) -> Retry:
    """Build a urllib3 Retry that logs a warning on every retry attempt.

    Catalog/granule endpoints (CMR, STAC) intermittently time out or
    return 5xx under load. urllib3 retries these silently inside the
    ``HTTPAdapter``; without logging, a stalled query that is quietly
    retrying looks indistinguishable from a hang. This wrapper surfaces
    each attempt — including the final exhausted one — at WARNING.

    Args:
        label: Short prefix identifying the endpoint (e.g. ``"CMR"``, ``"STAC"``).
        **kwargs: Passed through to :class:`urllib3.util.retry.Retry`.

    Returns:
        A ``Retry`` instance whose ``increment`` logs before delegating.
        ``label`` is captured by closure, so it survives the internal
        ``Retry.new()`` copies urllib3 makes on each attempt.
    """

    class _LoggingRetry(Retry):
        def increment(
            self,
            method: str | None = None,
            url: str | None = None,
            response: BaseHTTPResponse | None = None,
            error: Exception | None = None,
            _pool: ConnectionPool | None = None,
            _stacktrace: TracebackType | None = None,
        ) -> _LoggingRetry:
            if error is not None:
                logger.warning("%s retry: %s %s — %s", label, method, url, error)
            elif response is not None:
                logger.warning("%s retry: %s %s — HTTP %s", label, method, url, response.status)
            return super().increment(
                method=method,
                url=url,
                response=response,
                error=error,
                _pool=_pool,
                _stacktrace=_stacktrace,
            )

    return _LoggingRetry(**kwargs)  # type: ignore[arg-type]


def spawn_abandonable[A, T](fn: Callable[[A], T], arg: A) -> Callable[..., T]:
    """Start ``fn(arg)`` on a DAEMON thread; return a waiter that may give up on it.

    For catalogue calls that must be bounded by a deadline the caller owns. Two things about
    ``ThreadPoolExecutor`` make it the wrong tool here, and both are silent:

    ``shutdown(wait=False)`` returns without abandoning anything. The executor's workers are
    NON-DAEMON and joined by an ``atexit`` hook, so a stalled HTTP read holds the interpreter — and on
    the Prefect path the whole ECS task — open for the rest of that call's timeout-and-retry budget,
    long after the code that wanted the answer stopped waiting. ``cancel_futures=True`` does not help
    either: it only drops futures that have not STARTED.

    A daemon thread does not hold the process open, so abandonment costs nothing beyond a socket the
    OS reclaims. There is no cancel — an in-flight ``requests`` call is not interruptible from
    outside — so abandonment is the only mechanism available, and it is enough.

    The waiter takes an optional ``timeout`` and raises :class:`TimeoutError` when it expires, leaving
    the thread running. Exceptions from ``fn`` are re-raised in the CALLER's thread, where a
    ``future.result()`` would have raised them.
    """
    out: list[T] = []
    err: list[BaseException] = []

    def target() -> None:
        try:
            out.append(fn(arg))
        except BaseException as exc:
            err.append(exc)

    thread = threading.Thread(target=target, name=f"abandonable-{getattr(fn, '__name__', 'call')}", daemon=True)
    thread.start()

    def result(timeout: float | None = None) -> T:
        thread.join(timeout)
        if thread.is_alive():
            raise TimeoutError(f"call did not finish within {timeout}s and was abandoned")
        if err:
            raise err[0]
        return out[0]

    return result
