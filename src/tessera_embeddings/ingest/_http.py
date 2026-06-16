"""Shared HTTP retry helpers for ingest catalog/granule queries."""

from __future__ import annotations

import logging
from types import TracebackType

from urllib3.connectionpool import ConnectionPool
from urllib3.response import BaseHTTPResponse
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


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
