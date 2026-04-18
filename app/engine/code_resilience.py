"""Power Interpreter - Code Resilience

Retry and failure-classification utilities for code execution.

Design principles
-----------------
- Single-user: conservative retry counts, no aggressive backoff.
- Never swallow errors: after retries are exhausted, the final
  exception is re-raised unchanged so existing error-handling paths
  keep working.
- Classify failures: only *transient* errors are retried. Code-level
  errors (SyntaxError, NameError, etc.) are never retried - they will
  fail identically on the next attempt and only waste time.
- Stdlib only.
"""

import asyncio
import logging
import random
from enum import Enum
from functools import wraps
from typing import Any, Awaitable, Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


class FailureClass(Enum):
    TRANSIENT = "transient"   # Safe to retry (network, timeout, IO)
    PERMANENT = "permanent"   # Do not retry (code-level error)
    UNKNOWN = "unknown"       # Unclassified - retry conservatively


# Infrastructure / transient exception types - safe to retry.
_TRANSIENT_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    TimeoutError,
)

# Code-level exception types - retrying will produce the same result.
_PERMANENT_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    SyntaxError,
    IndentationError,
    NameError,
    TypeError,
    ValueError,
    AttributeError,
    ImportError,
    ModuleNotFoundError,
    KeyError,
    IndexError,
    ZeroDivisionError,
    ArithmeticError,
    AssertionError,
    RecursionError,
)


def classify_failure(exc: BaseException) -> FailureClass:
    """Classify an exception as transient, permanent, or unknown."""
    if isinstance(exc, _PERMANENT_EXCEPTIONS):
        return FailureClass.PERMANENT
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return FailureClass.TRANSIENT
    return FailureClass.UNKNOWN


def is_retryable(exc: BaseException) -> bool:
    """Return True if the exception should trigger a retry attempt."""
    cls = classify_failure(exc)
    # Retry transient failures. Unknown failures also retry (bounded
    # by max_attempts). Permanent failures never retry.
    return cls in (FailureClass.TRANSIENT, FailureClass.UNKNOWN)


class ResilienceStats:
    """Lightweight counters for retry observability. Never raises."""

    def __init__(self) -> None:
        self.attempts = 0
        self.retries = 0
        self.transient_failures = 0
        self.permanent_failures = 0
        self.final_failures = 0
        self.successes = 0

    def snapshot(self) -> dict:
        return {
            "attempts": self.attempts,
            "retries": self.retries,
            "transient_failures": self.transient_failures,
            "permanent_failures": self.permanent_failures,
            "final_failures": self.final_failures,
            "successes": self.successes,
        }

    def reset(self) -> None:
        self.attempts = 0
        self.retries = 0
        self.transient_failures = 0
        self.permanent_failures = 0
        self.final_failures = 0
        self.successes = 0


# Module-level singleton stats
stats = ResilienceStats()


def with_retry(
    max_attempts: int = 2,
    base_delay: float = 0.5,
    max_delay: float = 5.0,
    jitter: float = 0.1,
) -> Callable:
    """Async decorator: retry transient failures with exponential backoff.

    Parameters
    ----------
    max_attempts : total attempts (including the first). Default 2 =
        one retry after an initial failure.
    base_delay   : initial delay between attempts, seconds.
    max_delay    : cap on the delay between attempts, seconds.
    jitter       : uniform random jitter added to each delay, seconds.

    Behavior
    --------
    - Permanent failures re-raise immediately without retrying.
    - Transient/unknown failures sleep then retry, up to max_attempts.
    - After the last attempt, the final exception is re-raised unchanged.
    - Does not catch BaseException, so CancelledError / KeyboardInterrupt /
      SystemExit propagate cleanly without retry.
    """
    if max_attempts < 1:
        max_attempts = 1

    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[BaseException] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    stats.attempts += 1
                    result = await func(*args, **kwargs)
                    stats.successes += 1
                    return result
                except Exception as exc:  # noqa: BLE001 - intentional broad catch
                    last_exc = exc
                    cls = classify_failure(exc)
                    if cls == FailureClass.PERMANENT:
                        stats.permanent_failures += 1
                        raise
                    stats.transient_failures += 1
                    if attempt >= max_attempts:
                        stats.final_failures += 1
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter > 0:
                        delay += random.uniform(0, jitter)
                    stats.retries += 1
                    logger.debug(
                        "resilience: retry %d/%d after %s (delay=%.2fs)",
                        attempt, max_attempts, type(exc).__name__, delay,
                    )
                    await asyncio.sleep(delay)
            # Defensive: loop either returned or raised above.
            if last_exc is not None:
                raise last_exc
            return None
        return wrapper
    return decorator
