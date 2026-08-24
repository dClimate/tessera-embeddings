"""Shared retry and abandonment helpers for ingest catalogue/granule queries."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from types import TracebackType
from typing import Any

import requests
from urllib3.connectionpool import ConnectionPool
from urllib3.response import BaseHTTPResponse
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

#: How much of an unparseable body to log. Enough to tell an error page from a truncated
#: document from an empty one.
_EXCERPT_CHARS = 300


class NonJsonResponseError(RuntimeError):
    """A provider answered with a success status and a body that is not JSON.

    Invisible to a status-based retry ladder and to ``raise_for_status`` alike: the status
    is fine and only the body is wrong.
    """


def json_or_raise(resp: requests.Response, *, log_body: bool) -> Any:  # noqa: ANN401 — whatever the JSON holds
    """Parse ``resp`` as JSON, or raise :class:`NonJsonResponseError` naming the endpoint.

    The body never enters the MESSAGE: a failed leg's retry decision substring-matches the
    failure text against ``ingest_zone_year._NON_RETRYABLE_LEG_MARKERS``, so provider-chosen
    bytes there could turn a retryable leg permanently dead. It is logged instead — except
    from a credential endpoint, whose truncated body IS the credential, so pass
    ``log_body=False`` there.
    """
    try:
        return resp.json()
    except ValueError as exc:
        detail = (
            f"{resp.url} answered HTTP {resp.status_code} with a body that is not JSON "
            f"(content-type {resp.headers.get('Content-Type', '<absent>')!r}, {len(resp.content)} bytes)"
        )
        if log_body:
            logger.error("%s. Body began: %r", detail, resp.content[:_EXCERPT_CHARS])
        raise NonJsonResponseError(detail) from exc


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
