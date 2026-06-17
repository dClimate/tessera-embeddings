"""Unit tests for the AWS Dask provider's cluster-start retry policy.

These cover the retry *decision* — which cluster-start failures are transient
and worth retrying vs. which must fail fast — without provisioning real
Fargate. The predicate is the load-bearing piece (a misclassification either
burns retries on a permanent misconfiguration or gives up on a recoverable
ENI/control-plane race), so it is tested directly, and the tenacity policy it
feeds is exercised against a stub callable to confirm the wiring.
"""

from __future__ import annotations

import pytest
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_fixed

from tessera_embeddings.providers.aws.dask import (
    _RETRYABLE_CLUSTER_START_ERRORS,
    _is_retryable_cluster_start_error,
)


class TestRetryablePredicate:
    """``_is_retryable_cluster_start_error`` classification."""

    @pytest.mark.parametrize("needle", _RETRYABLE_CLUSTER_START_ERRORS)
    def test_known_transient_errors_retry(self, needle: str) -> None:
        """Each documented transient substring is matched, wrapped as _start() wraps it."""
        exc = RuntimeError(f"Cluster failed to start: {needle} (xyz)")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_scheduler_failed_to_start_retries(self) -> None:
        """The ENI-exhaustion failure observed in the fan-out → merge handoff."""
        exc = RuntimeError("Cluster failed to start: Scheduler failed to start")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_eni_placement_failure_retries(self) -> None:
        """RunTask placement rejection: dask raises RuntimeError(response) with
        the whole run_task response, whose failure reason is ``RESOURCE:ENI``.
        """
        exc = RuntimeError("{'tasks': [], 'failures': [{'reason': 'RESOURCE:ENI'}]}")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_describe_tasks_race_retries(self) -> None:
        """The original describe_tasks read-after-write race still matches."""
        exc = RuntimeError("Cluster failed to start: not enough values to unpack (expected 1, got 0)")
        assert _is_retryable_cluster_start_error(exc) is True

    def test_unrelated_runtime_error_does_not_retry(self) -> None:
        """A RuntimeError that isn't a known transient must fail fast."""
        exc = RuntimeError("Cluster failed to start: image pull failed (manifest unknown)")
        assert _is_retryable_cluster_start_error(exc) is False

    def test_non_runtime_error_does_not_retry(self) -> None:
        """Even a matching substring on a non-RuntimeError is not retried — the
        isinstance guard keeps the policy scoped to dask-cloudprovider's wrapper.
        """
        exc = ValueError("Scheduler failed to start")
        assert _is_retryable_cluster_start_error(exc) is False


class TestRetryPolicyWiring:
    """The tenacity policy built from the predicate retries/aborts as intended.

    Uses ``wait_fixed(0)`` to keep the test instant — the wait *schedule* is a
    config value asserted by reading it, not something to sit through here.
    """

    def _run(self, fn, *, max_attempts: int = 4):
        retrying = Retrying(
            retry=retry_if_exception(_is_retryable_cluster_start_error),
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(0),
            reraise=True,
        )
        return retrying(fn)

    def test_retries_until_success(self) -> None:
        """A transient failure on early attempts is ridden out; the call succeeds."""
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Cluster failed to start: Scheduler failed to start")
            return "cluster"

        assert self._run(flaky) == "cluster"
        assert calls["n"] == 3  # failed twice, succeeded on the third

    def test_gives_up_after_max_attempts_and_reraises_original(self) -> None:
        """A persistently-transient failure exhausts attempts and re-raises the
        ORIGINAL RuntimeError (reraise=True), not a tenacity RetryError.
        """
        calls = {"n": 0}

        def always_transient():
            calls["n"] += 1
            raise RuntimeError("Cluster failed to start: Scheduler failed to start")

        with pytest.raises(RuntimeError, match="Scheduler failed to start"):
            self._run(always_transient, max_attempts=4)
        assert calls["n"] == 4  # 1 initial + 3 retries

    def test_non_transient_fails_immediately(self) -> None:
        """A non-matching error is not retried — it surfaces on the first attempt."""
        calls = {"n": 0}

        def misconfigured():
            calls["n"] += 1
            raise RuntimeError("Cluster failed to start: image pull failed")

        with pytest.raises(RuntimeError, match="image pull failed"):
            self._run(misconfigured)
        assert calls["n"] == 1  # no retries burned on a permanent failure
