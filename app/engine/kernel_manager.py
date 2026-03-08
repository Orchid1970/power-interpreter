"""Power Interpreter - Kernel Manager

Manages persistent Python execution namespaces per session.
Instead of rebuilding sandbox_globals on every execute() call,
we store them keyed by session_id and reuse across calls.

This gives users "notebook-like" continuity:
  Call 1: df = pd.read_csv('data.csv')
  Call 2: df.describe()  # df still exists!

Features:
- One namespace per session_id
- Idle timeout cleanup (configurable, default 30 min)
- Max concurrent kernels cap
- Session metadata tracking (variable count, last activity, execution count)
- Manual reset capability
- has_session() check to avoid unnecessary globals rebuilds
- Per-session execution locks to serialize concurrent calls

Version: 2.2.0 - Add per-session exec locks for concurrent call safety
"""

import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KernelSession:
    """A persistent execution namespace for one session"""
    session_id: str
    sandbox_globals: Dict[str, Any]
    session_dir: Path
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    execution_count: int = 0

    def touch(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        self.execution_count += 1

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    @property
    def user_variables(self) -> Dict[str, str]:
        """Get user-defined variables (exclude system/library vars)"""
        skip_vars = {
            'pd', 'pandas', 'np', 'numpy', 'json', 'csv',
            'math', 'statistics', 'datetime', 'collections',
            'itertools', 'functools', 're', 'io', 'copy',
            'hashlib', 'base64', 'Path', 'dataclass', 'field',
            'asdict', 'Dict', 'List', 'Optional', 'Tuple',
            'Set', 'Any', 'SANDBOX_DIR', 'RESULT',
            'plt', 'matplotlib', 'sns', 'seaborn',
            'plotly', 'px', 'go', 'scipy', 'sklearn',
            'statsmodels', 'sm', 'openpyxl', 'pdfplumber',
            'tabulate', 'xlsxwriter', 'Decimal', 'Fraction',
            '__builtins__', '__name__',
            'textwrap', 'string', 'struct', 'decimal',
            'fractions', 'random', 'time', 'calendar',
            'pprint', 'dataclasses', 'typing', 'pathlib', 'os',
            'urllib', 'requests',
        }
        result = {}
        for key, value in self.sandbox_globals.items():
            if not key.startswith('_') and key not in skip_vars:
                try:
                    result[key] = type(value).__name__
                except Exception:
                    pass
        return result

    def to_info_dict(self) -> Dict:
        """Session metadata for API responses"""
        return {
            'session_id': self.session_id,
            'execution_count': self.execution_count,
            'idle_seconds': round(self.idle_seconds, 1),
            'created_at': self.created_at,
            'last_activity': self.last_activity,
            'variable_count': len(self.user_variables),
            'variables': self.user_variables,
        }


class KernelManager:
    """Manages persistent execution namespaces across sessions.

    Thread-safe. One namespace per session_id.
    Handles idle cleanup and max kernel limits.
    Per-session execution locks serialize concurrent calls to the same session.
    """

    def __init__(
        self,
        max_kernels: int = 6,
        idle_timeout_seconds: int = 1800,  # 30 minutes
    ):
        self._sessions: Dict[str, KernelSession] = {}
        self._lock = threading.Lock()
        self._exec_locks: Dict[str, threading.Lock] = {}
        self.max_kernels = max_kernels
        self.idle_timeout = idle_timeout_seconds
        logger.info(
            f"KernelManager initialized: max_kernels={max_kernels}, "
            f"idle_timeout={idle_timeout_seconds}s"
        )

    def get_exec_lock(self, session_id: str) -> threading.Lock:
        """Get a per-session execution lock for serializing concurrent calls.

        If SimTheory fires Steps 1-5 simultaneously for session="default",
        this ensures they queue up and run in order instead of racing.
        """
        with self._lock:
            if session_id not in self._exec_locks:
                self._exec_locks[session_id] = threading.Lock()
            return self._exec_locks[session_id]

    def has_session(self, session_id: str) -> bool:
        """Check if a session kernel exists (without modifying it).

        Used by executor to skip _build_safe_globals() when a kernel
        already exists — significant performance optimization since
        building globals imports pandas, numpy, etc.

        Thread-safe. Also cleans up expired sessions.
        """
        with self._lock:
            self._cleanup_expired()
            return session_id in self._sessions

    def get_or_create(
        self,
        session_id: str,
        sandbox_globals: Dict[str, Any],
        session_dir: Path,
    ) -> Dict[str, Any]:
        """Get existing session globals or store new ones.

        If session exists: returns the PERSISTED globals (ignores new sandbox_globals).
        If session is new: stores sandbox_globals and returns them.

        Args:
            session_id: Unique session identifier
            sandbox_globals: Freshly built globals (used only for new sessions)
            session_dir: Session sandbox directory

        Returns:
            The persistent sandbox_globals dict for this session
        """
        with self._lock:
            self._cleanup_expired()

            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                logger.info(
                    f"Kernel REUSED: session={session_id}, "
                    f"exec_count={session.execution_count}, "
                    f"user_vars={len(session.user_variables)}, "
                    f"idle={round(session.idle_seconds)}s"
                )
                return session.sandbox_globals

            if len(self._sessions) >= self.max_kernels:
                self._evict_oldest()

            session = KernelSession(
                session_id=session_id,
                sandbox_globals=sandbox_globals,
                session_dir=session_dir,
            )
            session.touch()
            self._sessions[session_id] = session
            logger.info(
                f"Kernel CREATED: session={session_id}, "
                f"active_kernels={len(self._sessions)}/{self.max_kernels}"
            )
            return session.sandbox_globals

    def get_existing(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get existing session globals WITHOUT creating a new session.

        Returns None if session doesn't exist.
        Used by executor to get persisted globals and skip _build_safe_globals.
        """
        with self._lock:
            self._cleanup_expired()
            session = self._sessions.get(session_id)
            if session:
                session.touch()
                logger.info(
                    f"Kernel REUSED: session={session_id}, "
                    f"exec_count={session.execution_count}, "
                    f"user_vars={len(session.user_variables)}, "
                    f"idle={round(session.idle_seconds)}s"
                )
                return session.sandbox_globals
            return None

    def reset_session(self, session_id: str) -> bool:
        """Destroy a session's kernel (next execute will start fresh).

        Returns True if session existed and was removed.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Kernel reset: session={session_id}")
                return True
            return False

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get metadata about a session's kernel"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.to_info_dict()
            return None

    def list_sessions(self) -> List[Dict]:
        """List all active kernel sessions"""
        with self._lock:
            return [s.to_info_dict() for s in self._sessions.values()]

    def _cleanup_expired(self):
        """Remove sessions that have been idle too long (called under lock)"""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.idle_seconds > self.idle_timeout
        ]
        for sid in expired:
            del self._sessions[sid]
            logger.info(f"Kernel expired: session={sid} (idle timeout)")
        if expired:
            logger.info(
                f"Cleaned up {len(expired)} expired kernels. "
                f"Active: {len(self._sessions)}"
            )

    def _evict_oldest(self):
        """Evict the oldest idle session to make room (called under lock)"""
        if not self._sessions:
            return
        oldest_sid = min(
            self._sessions.keys(),
            key=lambda sid: self._sessions[sid].last_activity
        )
        del self._sessions[oldest_sid]
        logger.info(f"Kernel evicted: session={oldest_sid} (capacity limit)")

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


# Singleton
kernel_manager = KernelManager()
