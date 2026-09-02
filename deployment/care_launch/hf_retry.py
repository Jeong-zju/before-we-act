"""Retry Hugging Face Hub calls that fail for reasons that pass on their own.

Every downloader in this repository pins a revision and verifies hashes, but
none of them handle transient Hub failures: a 429 from the rate limiter, a 5xx,
or a dropped connection aborts the stage. On an unattended run that turns a
recoverable pause into a dead pipeline.

Retries are deliberately narrow. Authentication failures, missing repositories,
and gated-model denials are permanent: retrying them wastes the run and buries
the real cause. Only rate limiting, server errors, and transport faults are
retried, with exponential backoff and jitter so parallel shards do not
resynchronize onto the same retry instant.

``Retry-After`` is honoured when the Hub sends one, since the server knows
better than the backoff schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Any, Callable, Sequence, TypeVar


T = TypeVar("T")

# Rate limiting and server-side faults clear on their own; 408 is a server-side
# read timeout rather than a malformed request.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# Permanent: retrying only delays the real diagnosis.
FATAL_STATUS = frozenset({401, 403, 404, 416})
RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "ChunkedEncodingError",
        "ConnectionError",
        "ConnectTimeout",
        "IncompleteRead",
        "ProtocolError",
        "ReadTimeout",
        "ReadTimeoutError",
        "RemoteDisconnected",
        "SSLError",
        "Timeout",
    }
)


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 6
    base_seconds: float = 4.0
    max_seconds: float = 300.0
    jitter: float = 0.25

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("retry policy needs at least one attempt")
        if self.base_seconds <= 0 or self.max_seconds < self.base_seconds:
            raise ValueError("retry backoff bounds are inconsistent")
        if not 0.0 <= self.jitter < 1.0:
            raise ValueError("retry jitter must lie in [0, 1)")

    def delay(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Exponential backoff with jitter, for a one-based attempt number."""
        source = rng or random
        raw = min(self.base_seconds * (2 ** (attempt - 1)), self.max_seconds)
        return raw * (1.0 + source.uniform(-self.jitter, self.jitter))


def _status_code(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    if code is None:
        code = getattr(error, "status_code", None)
    if code is None:
        code = getattr(error, "server_message_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after(error: BaseException) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    for key in ("Retry-After", "retry-after"):
        if key in headers:
            try:
                return max(0.0, float(headers[key]))
            except (TypeError, ValueError):
                return None
    return None


def is_retryable(error: BaseException) -> bool:
    """Whether an exception describes a fault that may clear on its own."""

    status = _status_code(error)
    if status is not None:
        if status in FATAL_STATUS:
            return False
        if status in RETRYABLE_STATUS:
            return True
        # Any other 5xx is a server fault.
        return 500 <= status < 600
    names = {type(error).__name__} | {
        base.__name__ for base in type(error).__mro__
    }
    if names & RETRYABLE_EXCEPTION_NAMES:
        return True
    cause = error.__cause__ or error.__context__
    return bool(cause) and cause is not error and is_retryable(cause)


def with_hub_retry(
    call: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    description: str = "hugging face hub call",
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] | None = print,
    rng: random.Random | None = None,
) -> T:
    """Run ``call``, retrying only transient Hub failures.

    Raises the last error once the attempts are exhausted, so a genuinely broken
    run still fails with its real cause rather than a retry wrapper's.
    """

    active = policy or RetryPolicy()
    last: BaseException | None = None
    for attempt in range(1, active.attempts + 1):
        try:
            return call()
        except BaseException as error:  # noqa: BLE001 - re-raised below
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            if not is_retryable(error) or attempt == active.attempts:
                raise
            last = error
            pause = _retry_after(error)
            if pause is None:
                pause = active.delay(attempt, rng=rng)
            if log is not None:
                status = _status_code(error)
                log(
                    f"{description}: attempt {attempt}/{active.attempts} failed "
                    f"({type(error).__name__}"
                    + (f", status {status}" if status is not None else "")
                    + f"); retrying in {pause:.1f}s"
                )
            sleep(pause)
    raise AssertionError("unreachable") from last


def retrying_snapshot_download(
    *args: Any,
    policy: RetryPolicy | None = None,
    log: Callable[[str], None] | None = print,
    **kwargs: Any,
) -> Any:
    """``huggingface_hub.snapshot_download`` under the transient-fault policy.

    ``snapshot_download`` resumes from its own cache, so a retry re-uses the
    bytes already on disk instead of restarting the transfer.
    """

    from huggingface_hub import snapshot_download

    return with_hub_retry(
        lambda: snapshot_download(*args, **kwargs),
        policy=policy,
        description=f"snapshot_download({kwargs.get('repo_id', args[0] if args else '?')})",
        log=log,
    )


__all__ = [
    "FATAL_STATUS",
    "RETRYABLE_EXCEPTION_NAMES",
    "RETRYABLE_STATUS",
    "RetryPolicy",
    "is_retryable",
    "retrying_snapshot_download",
    "with_hub_retry",
]
