"""Power Interpreter - Sandbox Backpressure Queue

Small-depth concurrency limiter for the execution sandbox.

Design principles
-----------------
- Single-user: shallow limits suitable for one-operator workloads.
- Backpressure, not rejection: an acquire waits for a slot, up to
  acquire_timeout. Only timeouts / hard-full raise; routine waiting
  does not.
- Observable: exposes stats about concurrency, queue depth, waits.
- Stdlib only.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

logger = logging.getLogger(__name__)

# Conservative defaults for a single-user Personal MCP
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_MAX_QUEUED = 8
DEFAULT_ACQUIRE_TIMEOUT = 30.0


class SandboxQueueFull(Exception):
    """Raised when the queue is full and a new acquirer cannot be queued."""


class SandboxQueueTimeout(Exception):
    """Raised when an acquire waits longer than acquire_timeout seconds."""


class SandboxQueue:
    """Backpressure queue for the sandbox.

    Concurrency is bounded by an asyncio.Semaphore. A second soft cap
    limits how many tasks may be *waiting* to acquire at once so a
    runaway caller cannot build an unbounded backlog.
    """

    def __init__(
        self,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        max_queued: int = DEFAULT_MAX_QUEUED,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> None:
        if max_concurrent < 1:
            max_concurrent = 1
        if max_queued < 0:
            max_queued = 0
        if acquire_timeout <= 0:
            acquire_timeout = DEFAULT_ACQUIRE_TIMEOUT

        self._max_concurrent = max_concurrent
        self._max_queued = max_queued
        self._acquire_timeout = acquire_timeout

        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()
        self._waiters = 0
        self._in_flight = 0

        # Counters (telemetry; never raise)
        self._total_acquired = 0
        self._total_rejected = 0
        self._total_timed_out = 0
        self._max_wait_seen_ms = 0

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def max_queued(self) -> int:
        return self._max_queued

    async def _acquire(self) -> None:
        # Fast-fail when all slots are busy AND the waiter bench is full.
        async with self._lock:
            if (
                self._in_flight >= self._max_concurrent
                and self._waiters >= self._max_queued
            ):
                self._total_rejected += 1
                raise SandboxQueueFull(
                    f"sandbox queue full (in_flight={self._in_flight}, "
                    f"waiters={self._waiters}, max_queued={self._max_queued})"
                )
            self._waiters += 1

        start = time.monotonic()
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._acquire_timeout
            )
        except asyncio.TimeoutError:
            async with self._lock:
                self._waiters = max(0, self._waiters - 1)
                self._total_timed_out += 1
            raise SandboxQueueTimeout(
                f"sandbox queue acquire timed out after "
                f"{self._acquire_timeout:.1f}s"
            )

        wait_ms = int((time.monotonic() - start) * 1000)
        async with self._lock:
            self._waiters = max(0, self._waiters - 1)
            self._in_flight += 1
            self._total_acquired += 1
            if wait_ms > self._max_wait_seen_ms:
                self._max_wait_seen_ms = wait_ms

    async def _release(self) -> None:
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
        try:
            self._semaphore.release()
        except ValueError:
            # Defensive: releasing more than acquired is a logic bug,
            # but must not crash the engine. Log and move on.
            logger.debug("sandbox_queue: semaphore release beyond acquire")

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Async context manager: hold a slot for the duration of the block.

        Usage::

            async with sandbox_queue.slot():
                await executor.execute(...)
        """
        await self._acquire()
        try:
            yield
        finally:
            await self._release()

    def get_stats(self) -> Dict[str, Any]:
        """Snapshot of current queue state. Never raises."""
        try:
            return {
                "max_concurrent": self._max_concurrent,
                "max_queued": self._max_queued,
                "in_flight": self._in_flight,
                "waiters": self._waiters,
                "total_acquired": self._total_acquired,
                "total_rejected": self._total_rejected,
                "total_timed_out": self._total_timed_out,
                "max_wait_ms": self._max_wait_seen_ms,
                "acquire_timeout_s": self._acquire_timeout,
            }
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("sandbox_queue.get_stats failed: %s", e)
            return {"error": "stats_unavailable"}


# Module-level singleton
sandbox_queue = SandboxQueue()
