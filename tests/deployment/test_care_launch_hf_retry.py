"""Transient Hub faults must be retried; permanent ones must not be."""
from __future__ import annotations

import random

import pytest

from deployment.care_launch.hf_retry import (
    RetryPolicy,
    is_retryable,
    with_hub_retry,
)


class _Response:
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.headers = headers or {}


class _HubError(Exception):
    def __init__(self, status: int, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"status {status}")
        self.response = _Response(status, headers)


# Named exactly as requests/httpx name it: matching is by class name.
class ReadTimeout(Exception):
    pass


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_rate_limits_and_server_faults_are_retryable(status: int) -> None:
    assert is_retryable(_HubError(status))


@pytest.mark.parametrize("status", [401, 403, 404])
def test_auth_and_missing_repositories_are_not_retryable(status: int) -> None:
    """Retrying a gated-model denial only buries the real cause."""
    assert not is_retryable(_HubError(status))


def test_transport_faults_are_retryable_by_exception_name() -> None:
    assert is_retryable(ReadTimeout("dropped"))


def test_a_retryable_cause_propagates_through_a_wrapper() -> None:
    outer = RuntimeError("upload failed")
    outer.__cause__ = _HubError(429)
    assert is_retryable(outer)


def test_call_succeeds_after_transient_failures() -> None:
    attempts = {"count": 0}
    slept: list[float] = []

    def call() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise _HubError(429)
        return "ok"

    result = with_hub_retry(
        call,
        policy=RetryPolicy(attempts=5, base_seconds=1.0),
        sleep=slept.append,
        log=None,
        rng=random.Random(0),
    )

    assert result == "ok"
    assert attempts["count"] == 3
    assert len(slept) == 2
    assert slept[1] > slept[0]  # backoff grows


def test_permanent_failure_raises_immediately() -> None:
    attempts = {"count": 0}

    def call() -> str:
        attempts["count"] += 1
        raise _HubError(403)

    with pytest.raises(_HubError):
        with_hub_retry(call, policy=RetryPolicy(attempts=5), sleep=lambda _s: None, log=None)
    assert attempts["count"] == 1


def test_exhausted_attempts_reraise_the_real_error() -> None:
    def call() -> str:
        raise _HubError(503)

    with pytest.raises(_HubError, match="status 503"):
        with_hub_retry(
            call,
            policy=RetryPolicy(attempts=2, base_seconds=0.1),
            sleep=lambda _s: None,
            log=None,
        )


def test_retry_after_header_overrides_the_backoff_schedule() -> None:
    slept: list[float] = []
    attempts = {"count": 0}

    def call() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise _HubError(429, {"Retry-After": "42"})
        return "ok"

    with_hub_retry(
        call,
        policy=RetryPolicy(attempts=3, base_seconds=1.0),
        sleep=slept.append,
        log=None,
    )
    assert slept == [42.0]


def test_backoff_is_bounded_and_jittered() -> None:
    policy = RetryPolicy(attempts=10, base_seconds=4.0, max_seconds=60.0, jitter=0.25)
    source = random.Random(7)
    delays = [policy.delay(attempt, rng=source) for attempt in range(1, 10)]

    assert all(delay <= 60.0 * 1.25 for delay in delays)
    assert len(set(delays)) == len(delays)  # jitter desynchronizes parallel shards


def test_interrupts_are_never_swallowed() -> None:
    def call() -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        with_hub_retry(call, sleep=lambda _s: None, log=None)
