"""Power Interpreter - Code Execution Routes

Sync execution for quick snippets (<30s).
For longer operations, use the Jobs API.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict

from app.engine.executor import executor
from app.config import settings
from app.response_budget import enforce_response_budget
from app.response_guard import smart_truncate

router = APIRouter()


class ExecuteRequest(BaseModel):
    """Request to execute Python code"""
    code: str = Field(..., description="Python code to execute")
    session_id: str = Field(default="default", description="Session ID for file isolation")
    timeout: Optional[int] = Field(default=30, description="Max execution time in seconds (max 60 for sync)")
    context: Optional[Dict] = Field(default=None, description="Variables to inject into sandbox")
    sequence: Optional[int] = Field(default=None, description="Step number for ordered execution (1, 2, 3...). When multiple calls arrive simultaneously, they execute in sequence order.")


class ExecuteResponse(BaseModel):
    """Response from code execution"""
    success: bool
    stdout: str
    stderr: str
    result: Optional[object] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    execution_time_ms: int
    memory_used_mb: float
    files_created: list
    variables: Dict[str, str]


@router.post("/execute")
async def execute_code(request: ExecuteRequest):
    """Execute Python code synchronously (for quick operations)

    Use this for:
    - Quick calculations
    - Small data transformations
    - File generation
    - Anything that completes in <60 seconds

    For longer operations, use POST /api/jobs/submit instead.

    Note: response_model intentionally omitted so the response_budget
    wrapper can reshape the payload on truncation without schema errors.
    """
    # Limit sync execution time
    timeout = min(request.timeout or 30, 60)

    if not request.code.strip():
        raise HTTPException(status_code=400, detail="No code provided")

    result = await executor.execute(
        code=request.code,
        session_id=request.session_id,
        timeout=timeout,
        context=request.context,
        sequence=request.sequence
    )

    # Guard stack composition at the route layer:
    # 1) Boundary-aware truncation on stdout/stderr via response_guard
    # 2) Overall payload bounded via response_budget (50K token ceiling)
    # context_guard + syntax_guard are already wired inside the executor.
    if result.stdout:
        result.stdout = smart_truncate(result.stdout)
    if result.stderr:
        result.stderr = smart_truncate(result.stderr)

    payload = result.to_dict()
    return enforce_response_budget("execute", payload)


@router.post("/execute/quick")
async def execute_quick(code: str):
    """Ultra-quick execution endpoint (10s max)

    Convenience endpoint for simple expressions and calculations.
    """
    if not code.strip():
        raise HTTPException(status_code=400, detail="No code provided")

    result = await executor.execute(
        code=code,
        session_id="quick",
        timeout=10
    )

    # Guard stack composition (see execute_code for rationale).
    output = result.stdout.strip() if result.success else result.error_message
    if output:
        output = smart_truncate(output)

    payload = {
        'success': result.success,
        'output': output,
        'result': result.result,
        'time_ms': result.execution_time_ms
    }
    return enforce_response_budget("execute_quick", payload)
