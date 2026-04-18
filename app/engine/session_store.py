"""Power Interpreter - Session Store

Async-native session lifecycle coordination layer.

Responsibilities (complementary to kernel_manager, which owns execution
namespaces / sandbox_globals):
- Track session existence and last-activity timestamps
- Enforce TTL (default 1 hour) via a background asyncio sweeper task
- Provide async get/touch/delete APIs for executor wire-in

On TTL expiration, the sweeper also calls kernel_manager.reset_session()
so the persistent Python namespace is released. The kernel eviction is
deliberately performed AFTER releasing SessionStore's own lock to avoid
holding one lock across a foreign lock acquisition (kernel_manager uses
its own threading.Lock).

Thread-safety: async-only. All state transitions serialize on a single
asyncio.Lock. Not intended for use from non-async contexts.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 3600          # 1 hour
DEFAULT_SWEEP_INTERVAL_SECONDS = 300  # 5 minutes


@dataclass
class SessionRecord:
    """Lightweight session metadata. Namespaces live in kernel_manager."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    touch_count: int = 0

    def touch(self) -> None:
        self.last_activity = time.time()
        self.touch_count += 1

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "idle_seconds": round(self.idle_seconds, 1),
            "touch_count": self.touch_count,
        }


class SessionStore:
    """Async-native session lifecycle coordinator with background TTL sweeper."""

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        sweep_interval_seconds: int = DEFAULT_SWEEP_INTERVAL_SECONDS,
    ):
        self._records: Dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()
        self._ttl = ttl_seconds
        self._sweep_interval = sweep_interval_seconds
        self._sweeper_task: Optional[asyncio.Task] = None
        self._started = False
        logger.info(
            f"SessionStore initialized: ttl={ttl_seconds}s, "
            f"sweep_interval={sweep_interval_seconds}s"
        )

    async def _ensure_sweeper(self) -> None:
        """Lazily start the background sweeper on first async entry.

        Deferred to first async call so the task binds to the actually
        running event loop (rather than whatever loop existed at import).
        """
        if self._started:
            return
        async with self._lock:
            if self._started:
                return
            try:
                loop = asyncio.get_running_loop()
                self._sweeper_task = loop.create_task(
                    self._sweep_loop(), name="session-store-sweeper"
                )
                self._started = True
                logger.info("SessionStore sweeper task started")
            except RuntimeError:
                logger.warning(
                    "SessionStore._ensure_sweeper: no running loop; deferring"
                )

    async def touch(self, session_id: str) -> SessionRecord:
        """Register a new session or refresh last_activity on an existing one.

        Intended to be called at the top of every executor entry point.
        """
        await self._ensure_sweeper()
        async with self._lock:
            record = self._records.get(session_id)
            if record is None:
                record = SessionRecord(session_id=session_id)
                self._records[session_id] = record
                logger.info(f"SessionStore CREATED: session={session_id}")
            record.touch()
            return record

    async def get(self, session_id: str) -> Optional[SessionRecord]:
        async with self._lock:
            return self._records.get(session_id)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            if session_id in self._records:
                del self._records[session_id]
                logger.info(f"SessionStore DELETED: session={session_id}")
                return True
            return False

    async def list_sessions(self) -> List[Dict]:
        async with self._lock:
            return [r.to_dict() for r in self._records.values()]

    async def _sweep_once(self) -> int:
        """Evict records whose idle_seconds exceeds TTL. Returns eviction count.

        Kernel eviction is performed AFTER releasing this store's lock so we
        never hold our asyncio.Lock across kernel_manager's threading.Lock.
        """
        expired_ids: List[str] = []
        async with self._lock:
            for sid, record in self._records.items():
                if record.idle_seconds > self._ttl:
                    expired_ids.append(sid)
            for sid in expired_ids:
                del self._records[sid]

        if expired_ids:
            try:
                from app.engine.kernel_manager import kernel_manager
                for sid in expired_ids:
                    try:
                        kernel_manager.reset_session(sid)
                    except Exception as e:
                        logger.warning(
                            f"kernel_manager.reset_session({sid}) failed: {e}"
                        )
            except Exception as e:
                logger.warning(
                    f"SessionStore sweep: kernel_manager import failed: {e}"
                )

            logger.info(
                f"SessionStore sweep evicted {len(expired_ids)} session(s): "
                f"{expired_ids[:5]}"
            )
        return len(expired_ids)

    async def _sweep_loop(self) -> None:
        """Background TTL sweeper. Runs until cancelled."""
        logger.info(
            f"SessionStore sweep_loop running every {self._sweep_interval}s"
        )
        while True:
            try:
                await asyncio.sleep(self._sweep_interval)
                await self._sweep_once()
            except asyncio.CancelledError:
                logger.info("SessionStore sweep_loop cancelled")
                raise
            except Exception as e:
                logger.error(
                    f"SessionStore sweep_loop error (continuing): {e}",
                    exc_info=True,
                )

    async def shutdown(self) -> None:
        """Cancel the sweeper task. Call during application shutdown."""
        if self._sweeper_task and not self._sweeper_task.done():
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass
            logger.info("SessionStore shutdown complete")

    @property
    def active_count(self) -> int:
        return len(self._records)


# Singleton
session_store = SessionStore()
