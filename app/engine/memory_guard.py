"""Memory-isolated subprocess execution for sandbox code.

Prevents a runaway pandas/numpy operation from eating all container memory
and crashing the uvicorn process. Each sandbox execution gets its own
subprocess with RLIMIT_AS set.

v1.9.2: Initial implementation after MemoryError crash at 03:56:54 UTC
         killed the entire container.

Usage:
    from app.engine.memory_guard import execute_with_memory_limit
    result = await execute_with_memory_limit(code, timeout=300, max_gb=4)
"""

import os
import sys
import asyncio
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default: 4 GB per sandbox execution
# Container has 32 GB, uvicorn needs ~1 GB, leave headroom
DEFAULT_MAX_GB = 4


def _set_memory_limit_bytes(max_bytes: int):
    """Called inside the subprocess via preexec_fn to set RLIMIT_AS."""
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except (ValueError, ImportError):
        pass  # Some environments don't support RLIMIT_AS


async def execute_with_memory_limit(
    code: str,
    timeout: int = 300,
    max_gb: float = DEFAULT_MAX_GB,
    env_extras: Optional[dict] = None,
) -> dict:
    """
    Execute Python code in a memory-isolated subprocess.

    The subprocess gets its own memory limit so a runaway operation
    gets MemoryError inside the subprocess without affecting uvicorn.

    Args:
        code: Python source code to execute
        timeout: Max execution time in seconds (default 300)
        max_gb: Max memory in GB for the subprocess (default 4)
        env_extras: Additional environment variables for the subprocess

    Returns:
        dict with keys: success, stdout, stderr, return_code, timed_out, oom_killed
    """
    max_bytes = int(max_gb * 1024 * 1024 * 1024)

    # Write code to a temp file
    fd, script_path = tempfile.mkstemp(suffix=".py", prefix="sandbox_", dir="/tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(code)

        # Build environment
        env = dict(os.environ)
        if env_extras:
            env.update(env_extras)

        # Launch subprocess with memory limit
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=lambda: _set_memory_limit_bytes(max_bytes),
            env=env,
        )

        timed_out = False
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            timed_out = True
            proc.kill()
            stdout_bytes, stderr_bytes = await proc.communicate()

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Detect OOM kill (return code -9 = SIGKILL, often from OOM)
        oom_killed = proc.returncode == -9 or "MemoryError" in stderr

        if timed_out:
            stderr += f"\n[TIMEOUT] Execution exceeded {timeout} seconds and was killed."
        if oom_killed and not timed_out:
            stderr += f"\n[OOM] Process exceeded {max_gb} GB memory limit."

        return {
            "success": proc.returncode == 0 and not timed_out,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": proc.returncode,
            "timed_out": timed_out,
            "oom_killed": oom_killed,
        }

    except Exception as e:
        logger.error(f"memory_guard: subprocess launch failed: {e}", exc_info=True)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Failed to launch sandbox subprocess: {e}",
            "return_code": -1,
            "timed_out": False,
            "oom_killed": False,
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
