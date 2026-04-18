"""Power Interpreter - User Tracker

Lightweight in-memory session and execution tracker for single-user
Personal MCP.

Design principles
-----------------
- Single-user: no multi-tenant isolation, no auth boundary.
- Never raises: telemetry failures must never break execution. Every
  public method swallows exceptions and logs them at DEBUG level.
- Bounded memory: all collections are capped to prevent unbounded growth.
- Stdlib only: no external dependencies.
- Async-safe: an asyncio.Lock serializes all state mutations so the
  tracker is safe to call from concurrent job coroutines.

Usage
-----
    from app.engine.user_tracker import user_tracker

    await user_tracker.record_job_submitted(job_id, session_id)
    await user_tracker.record_job_started(job_id)
    await user_tracker.record_job_completed(job_id, success=True, duration_ms=1234)

    snapshot = await user_tracker.get_snapshot()
    events = await user_tracker.get_recent_activity(limit=20)
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounded limits - protect against unbounded memory growth
# ---------------------------------------------------------------------------
MAX_ACTIVITY_LOG = 500          # Most recent activity events
MAX_TRACKED_SESSIONS = 100      # Most recent sessions
MAX_TRACKED_JOBS = 500          # Most recent job records

# Session idle threshold (informational only, not enforced)
SESSION_IDLE_MINUTES = 30


class UserTracker:
    """Single-user session and execution tracker.

    All public methods are non-raising: any internal error is logged at
    DEBUG and swallowed so telemetry cannot break the execution path.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        # Bounded activity log (oldest entries evict automatically)
        self._activity: Deque[Dict[str, Any]] = deque(maxlen=MAX_ACTIVITY_LOG)

        # Session state: session_id -> {first_seen, last_seen, job_count}
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # Job state: job_id -> {session_id, submitted_at, started_at,
        #                       completed_at, status, duration_ms, metadata}
        self._jobs: Dict[str, Dict[str, Any]] = {}

        # Aggregate counters
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_cancelled = 0

        self._started_at = datetime.utcnow()

    # ------------------------------------------------------------------
    # Job lifecycle recording (called from job_manager)
    # ------------------------------------------------------------------

    async def record_job_submitted(
        self,
        job_id: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a job submission. Never raises."""
        try:
            async with self._lock:
                now = datetime.utcnow()
                sid = session_id or "default"

                self._jobs[job_id] = {
                    "session_id": sid,
                    "submitted_at": now,
                    "started_at": None,
                    "completed_at": None,
                    "status": "pending",
                    "duration_ms": None,
                    "metadata": metadata or {},
                }
                self._trim_jobs()

                session = self._sessions.get(sid)
                if session is None:
                    session = {
                        "first_seen": now,
                        "last_seen": now,
                        "job_count": 0,
                    }
                    self._sessions[sid] = session
                    self._trim_sessions()
                session["last_seen"] = now
                session["job_count"] = session.get("job_count", 0) + 1

                self._total_submitted += 1
                self._log_activity("job_submitted", job_id=job_id, session_id=sid)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("user_tracker.record_job_submitted failed: %s", e)

    async def record_job_started(self, job_id: str) -> None:
        """Record that job execution has started. Never raises."""
        try:
            async with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["started_at"] = datetime.utcnow()
                    job["status"] = "running"
                self._log_activity("job_started", job_id=job_id)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("user_tracker.record_job_started failed: %s", e)

    async def record_job_completed(
        self,
        job_id: str,
        success: bool,
        duration_ms: Optional[int] = None,
    ) -> None:
        """Record job completion (success or failure). Never raises."""
        try:
            async with self._lock:
                now = datetime.utcnow()
                job = self._jobs.get(job_id)
                if job is not None:
                    job["completed_at"] = now
                    job["status"] = "completed" if success else "failed"
                    job["duration_ms"] = duration_ms

                if success:
                    self._total_completed += 1
                else:
                    self._total_failed += 1

                self._log_activity(
                    "job_completed" if success else "job_failed",
                    job_id=job_id,
                    duration_ms=duration_ms,
                )
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("user_tracker.record_job_completed failed: %s", e)

    async def record_job_cancelled(self, job_id: str) -> None:
        """Record a job cancellation. Never raises."""
        try:
            async with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["completed_at"] = datetime.utcnow()
                    job["status"] = "cancelled"
                self._total_cancelled += 1
                self._log_activity("job_cancelled", job_id=job_id)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("user_tracker.record_job_cancelled failed: %s", e)

    # ------------------------------------------------------------------
    # Read-only inspection API
    # ------------------------------------------------------------------

    async def get_snapshot(self) -> Dict[str, Any]:
        """Return a snapshot of current tracker state. Never raises."""
        try:
            async with self._lock:
                now = datetime.utcnow()
                uptime_s = int((now - self._started_at).total_seconds())

                idle_threshold = now - timedelta(minutes=SESSION_IDLE_MINUTES)
                active_sessions = sum(
                    1
                    for s in self._sessions.values()
                    if s.get("last_seen") and s["last_seen"] >= idle_threshold
                )

                in_flight = sum(
                    1 for j in self._jobs.values() if j.get("status") == "running"
                )

                return {
                    "uptime_seconds": uptime_s,
                    "started_at": self._started_at.isoformat(),
                    "totals": {
                        "submitted": self._total_submitted,
                        "completed": self._total_completed,
                        "failed": self._total_failed,
                        "cancelled": self._total_cancelled,
                    },
                    "sessions": {
                        "total_tracked": len(self._sessions),
                        "active": active_sessions,
                    },
                    "jobs": {
                        "total_tracked": len(self._jobs),
                        "in_flight": in_flight,
                    },
                    "activity_log_size": len(self._activity),
                }
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("user_tracker.get_snapshot failed: %s", e)
            return {"error": "snapshot_unavailable"}

    async def get_recent_activity(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent activity events, newest first. Never raises."""
        try:
            async with self._lock:
                if limit <= 0:
                    return []
                # Deque iterates oldest -> newest; slice the tail, reverse.
                tail = list(self._activity)[-limit:]
                tail.reverse()
                return tail
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("user_tracker.get_recent_activity failed: %s", e)
            return []

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return tracked info for a specific session, or None. Never raises."""
        try:
            async with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return None
                return {
                    "session_id": session_id,
                    "first_seen": session["first_seen"].isoformat(),
                    "last_seen": session["last_seen"].isoformat(),
                    "job_count": session.get("job_count", 0),
                }
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("user_tracker.get_session_info failed: %s", e)
            return None

    async def reset(self) -> None:
        """Reset all tracker state. Intended for tests. Never raises."""
        try:
            async with self._lock:
                self._activity.clear()
                self._sessions.clear()
                self._jobs.clear()
                self._total_submitted = 0
                self._total_completed = 0
                self._total_failed = 0
                self._total_cancelled = 0
                self._started_at = datetime.utcnow()
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("user_tracker.reset failed: %s", e)

    # ------------------------------------------------------------------
    # Internal helpers - callers must hold self._lock
    # ------------------------------------------------------------------

    def _log_activity(self, event_type: str, **fields: Any) -> None:
        """Append an event to the bounded activity log."""
        entry: Dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(),
            "event": event_type,
        }
        entry.update(fields)
        self._activity.append(entry)

    def _trim_sessions(self) -> None:
        """Cap session dict to MAX_TRACKED_SESSIONS, evicting oldest by last_seen."""
        if len(self._sessions) <= MAX_TRACKED_SESSIONS:
            return
        sorted_items = sorted(
            self._sessions.items(),
            key=lambda kv: kv[1].get("last_seen", datetime.min),
        )
        to_evict = len(self._sessions) - MAX_TRACKED_SESSIONS
        for key, _ in sorted_items[:to_evict]:
            self._sessions.pop(key, None)

    def _trim_jobs(self) -> None:
        """Cap job dict to MAX_TRACKED_JOBS, evicting oldest by submitted_at."""
        if len(self._jobs) <= MAX_TRACKED_JOBS:
            return
        sorted_items = sorted(
            self._jobs.items(),
            key=lambda kv: kv[1].get("submitted_at", datetime.min),
        )
        to_evict = len(self._jobs) - MAX_TRACKED_JOBS
        for key, _ in sorted_items[:to_evict]:
            self._jobs.pop(key, None)


# Module-level singleton
user_tracker = UserTracker()
