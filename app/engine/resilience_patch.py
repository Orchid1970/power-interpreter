"""Power Interpreter - Resilience Patch

Monkey-patches the executor singleton to layer resilience and
backpressure on top of the existing execute() coroutine *without*
modifying executor.py.

What it adds
------------
1. Sandbox backpressure queue: every execute() acquires a queue slot
   before running, bounding in-flight concurrency.
2. Transient-failure retry: wraps execute() with code_resilience's
   with_retry decorator so network / timeout style errors get one
   automatic retry. Code-level errors are never retried.

Safety
------
- Idempotent: apply() can be called multiple times safely.
- Reversible: unpatch() restores the original execute() coroutine.
- Opt-in: has no effect until apply() is called.
- Lazy import of executor avoids circular-import risk at module load.
- Never crashes the process - apply() / unpatch() log warnings and
  return False on failure.
"""

import logging
from typing import Any, Optional

from app.engine.code_resilience import with_retry
from app.engine.sandbox_queue import (
    sandbox_queue,
    SandboxQueueFull,
    SandboxQueueTimeout,
)

logger = logging.getLogger(__name__)

# Sentinel attribute to recognize our patched wrapper
_PATCH_MARKER = "_pi_resilience_patched"

# Cached original bound method so we can restore it on unpatch()
_original_execute: Optional[Any] = None


def is_patched() -> bool:
    """Return True if the executor singleton is currently patched."""
    try:
        from app.engine.executor import executor  # lazy import
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("resilience_patch.is_patched: executor import failed: %s", e)
        return False
    return getattr(executor.execute, _PATCH_MARKER, False)


def apply(
    max_attempts: int = 2,
    base_delay: float = 0.5,
) -> bool:
    """Install the resilience + backpressure wrapper on executor.execute.

    Returns True if the patch is active after the call (freshly applied
    or already active). Returns False on failure. Never raises.
    """
    global _original_execute

    try:
        from app.engine.executor import executor  # lazy import
    except Exception as e:
        logger.warning("resilience_patch.apply: cannot import executor: %s", e)
        return False

    if getattr(executor.execute, _PATCH_MARKER, False):
        logger.debug("resilience_patch.apply: already patched")
        return True

    try:
        original = executor.execute
        _original_execute = original

        @with_retry(max_attempts=max_attempts, base_delay=base_delay)
        async def _guarded_execute(*args: Any, **kwargs: Any) -> Any:
            # Backpressure: hold a slot for the duration of the run.
            async with sandbox_queue.slot():
                return await original(*args, **kwargs)

        async def _patched_execute(*args: Any, **kwargs: Any) -> Any:
            try:
                return await _guarded_execute(*args, **kwargs)
            except SandboxQueueFull as e:
                logger.warning("sandbox_queue full: %s", e)
                raise
            except SandboxQueueTimeout as e:
                logger.warning("sandbox_queue timeout: %s", e)
                raise

        setattr(_patched_execute, _PATCH_MARKER, True)
        executor.execute = _patched_execute  # type: ignore[assignment]
        logger.info(
            "resilience_patch: applied (max_attempts=%d, base_delay=%.2fs)",
            max_attempts, base_delay,
        )
        return True
    except Exception as e:
        logger.warning("resilience_patch.apply failed: %s", e)
        return False


def unpatch() -> bool:
    """Restore the original executor.execute. Returns True on success."""
    global _original_execute

    try:
        from app.engine.executor import executor  # lazy import
    except Exception as e:
        logger.warning("resilience_patch.unpatch: cannot import executor: %s", e)
        return False

    if not getattr(executor.execute, _PATCH_MARKER, False):
        logger.debug("resilience_patch.unpatch: not patched")
        return True

    if _original_execute is None:
        logger.warning("resilience_patch.unpatch: no original cached")
        return False

    try:
        executor.execute = _original_execute  # type: ignore[assignment]
        _original_execute = None
        logger.info("resilience_patch: reverted")
        return True
    except Exception as e:
        logger.warning("resilience_patch.unpatch failed: %s", e)
        return False
