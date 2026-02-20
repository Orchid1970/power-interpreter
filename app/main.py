"""Power Interpreter MCP - Main Application

General-purpose sandboxed Python execution engine.
Designed for SimTheory.ai MCP integration.

Features:
- Execute Python code in sandboxed environment
- Async job queue for long-running operations (no timeouts)
- Large dataset support (1.5M+ rows via PostgreSQL)
- File upload/download management
- Pre-installed data science libraries
- Persistent session state (kernel architecture)
- Auto file storage in Postgres with public download URLs

Author: Kaffer AI for Timothy Escamilla
Version: 1.2.1
"""

import logging
import asyncio
import json
import inspect
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, Depends, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.auth import verify_api_key
from app.routes import execute, jobs, files, data, sessions, health
from app.routes.files import public_router as download_router
from app.mcp_server import mcp

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # --- STARTUP ---
    logger.info("="*60)
    logger.info("Power Interpreter MCP v1.2.1 starting...")
    logger.info("="*60)
    
    # Ensure directories exist
    settings.ensure_directories()
    
    # Initialize database (graceful - don't crash if DB not ready)
    db_ok = False
    if settings.DATABASE_URL:
        try:
            from app.database import init_database
            await init_database()
            db_ok = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            logger.warning("App will start without database. Some features disabled.")
    else:
        logger.warning("No DATABASE_URL configured. Running without database.")
        logger.warning("Set DATABASE_URL to enable: jobs, sessions, datasets, file tracking")
    
    # Start periodic cleanup (jobs + expired sandbox files)
    cleanup_task = None
    if db_ok:
        cleanup_task = asyncio.create_task(_periodic_cleanup())
    
    # Log public URL for download links
    public_url = settings.public_base_url
    
    logger.info("Power Interpreter ready!")
    logger.info(f"  Database: {'connected' if db_ok else 'NOT CONNECTED'}")
    logger.info(f"  Sandbox dir: {settings.SANDBOX_DIR}")
    logger.info(f"  Public URL: {public_url or '(auto-detect from RAILWAY_PUBLIC_DOMAIN)'}")
    logger.info(f"  Download endpoint: /dl/{{file_id}} (public, no auth)")
    logger.info(f"  Sandbox file max: {settings.SANDBOX_FILE_MAX_MB} MB")
    logger.info(f"  Sandbox file TTL: {settings.SANDBOX_FILE_TTL_HOURS} hours")
    logger.info(f"  Max execution time: {settings.MAX_EXECUTION_TIME}s")
    logger.info(f"  Max memory: {settings.MAX_MEMORY_MB} MB")
    logger.info(f"  Max concurrent jobs: {settings.MAX_CONCURRENT_JOBS}")
    logger.info(f"  Job timeout: {settings.JOB_TIMEOUT}s")
    logger.info(f"  MCP SSE transport: GET /mcp/sse (standard clients)")
    logger.info(f"  MCP JSON-RPC direct: POST /mcp/sse (SimTheory)")
    
    yield
    
    # --- SHUTDOWN ---
    logger.info("Power Interpreter shutting down...")
    if cleanup_task:
        cleanup_task.cancel()
    if db_ok:
        try:
            from app.database import shutdown_database
            await shutdown_database()
        except Exception:
            pass
    logger.info("Shutdown complete")


async def _periodic_cleanup():
    """Periodically clean up old jobs, temp files, and expired sandbox files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            
            # Clean up old jobs
            from app.engine.job_manager import job_manager
            count = await job_manager.cleanup_old_jobs()
            if count:
                logger.info(f"Periodic cleanup: removed {count} old jobs")
            
            # Clean up expired sandbox files
            try:
                await _cleanup_expired_sandbox_files()
            except Exception as e:
                logger.error(f"Sandbox file cleanup error: {e}")
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def _cleanup_expired_sandbox_files():
    """Delete sandbox files past their TTL from Postgres."""
    try:
        from app.database import get_session_factory
        from app.models import SandboxFile
        from sqlalchemy import delete
        
        factory = get_session_factory()
        async with factory() as session:
            result = await session.execute(
                delete(SandboxFile).where(
                    SandboxFile.expires_at != None,
                    SandboxFile.expires_at < datetime.utcnow()
                )
            )
            deleted = result.rowcount
            if deleted:
                await session.commit()
                logger.info(f"Cleaned up {deleted} expired sandbox files")
    except Exception as e:
        logger.error(f"Failed to clean expired sandbox files: {e}")


# Create FastAPI app
app = FastAPI(
    title="Power Interpreter MCP",
    description=(
        "General-purpose sandboxed Python execution engine. "
        "Execute code, manage files, query large datasets, "
        "and run long-running analysis jobs without timeouts. "
        "Generated files get persistent download URLs via /dl/{file_id}."
    ),
    version="1.2.1",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (allow SimTheory.ai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# DIRECT MCP JSON-RPC HANDLER (for SimTheory)
# =============================================================================
# SimTheory POSTs JSON-RPC to /mcp/sse without maintaining an SSE stream.
# The SSE transport sends responses via the stream, not the HTTP body.
# SimTheory expects the response IN the HTTP body.
#
# This handler bypasses the SSE transport entirely:
# 1. Parses the JSON-RPC request
# 2. Routes to the correct handler
# 3. Calls tool functions directly
# 4. Returns JSON-RPC response in the HTTP body
#
# MUST be defined BEFORE app.mount("/mcp", ...) so FastAPI matches it first.
# GET /mcp/sse still goes to the SSE transport for standard MCP clients.
# =============================================================================


def _build_tool_schema(tool) -> dict:
    """Build the JSON Schema for a tool's input parameters."""
    # Try to get schema from the tool object first
    if hasattr(tool, 'parameters') and tool.parameters:
        return tool.parameters

    # Fall back to building from function signature
    fn = tool.fn if hasattr(tool, 'fn') else tool
    if not callable(fn):
        return {"type": "object", "properties": {}}

    sig = inspect.signature(fn)
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name in ('self', 'cls'):
            continue

        prop = {"type": "string"}
        if param.annotation != inspect.Parameter.empty:
            type_map = {
                int: "integer",
                float: "number",
                bool: "boolean",
                str: "string",
            }
            ann = param.annotation
            # Handle Optional types
            origin = getattr(ann, '__origin__', None)
            if origin is not None:
                args = getattr(ann, '__args__', ())
                if args:
                    ann = args[0]
            prop["type"] = type_map.get(ann, "string")

        if param.default != inspect.Parameter.empty:
            if param.default is not None:
                prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _get_tool_registry() -> dict:
    """Get all registered MCP tools as {name: tool_object}."""
    tools = {}
    # FastMCP stores tools in _tool_manager._tools
    if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
        for name, tool in mcp._tool_manager._tools.items():
            tools[name] = tool
    return tools


def _get_tools_list() -> list:
    """Build the tools/list response array."""
    result = []
    registry = _get_tool_registry()
    for name, tool in registry.items():
        desc = ""
        if hasattr(tool, 'description'):
            desc = tool.description or ""
        elif hasattr(tool, 'fn') and tool.fn.__doc__:
            desc = tool.fn.__doc__.strip()

        result.append({
            "name": name,
            "description": desc,
            "inputSchema": _build_tool_schema(tool),
        })
    return result


def _validate_tool_args(fn, tool_args: dict, tool_name: str) -> str | None:
    """
    Validate that all required arguments are present before calling a tool.
    Returns an error message string if validation fails, None if OK.
    """
    try:
        sig = inspect.signature(fn)
        missing = []
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            # If no default value, it's required
            if param.default is inspect.Parameter.empty:
                if param_name not in tool_args:
                    missing.append(param_name)
        if missing:
            return (
                f"Missing required parameter(s) for '{tool_name}': {', '.join(missing)}. "
                f"Please provide: {', '.join(missing)}"
            )
    except Exception:
        pass  # If we can't inspect, let it through and fail naturally
    return None


@app.post("/mcp/sse")
async def handle_mcp_jsonrpc(request: Request):
    """
    Direct MCP JSON-RPC handler for SimTheory.

    Handles: initialize, notifications/initialized, tools/list,
    tools/call, ping, and unknown methods.
    """
    # --- Parse body ---
    try:
        body = await request.body()
        body_str = body.decode("utf-8", errors="replace")
        logger.info(f"MCP direct: received {len(body)} bytes")
        logger.info(f"MCP direct: {body_str[:500]}")
        data = json.loads(body_str)
    except Exception as e:
        logger.error(f"MCP direct: parse error: {e}")
        return JSONResponse(content={
            "jsonrpc": "2.0",
            "error": {"code": -32700, "message": f"Parse error: {e}"},
            "id": None,
        }, status_code=400)

    # --- Batch ---
    if isinstance(data, list):
        responses = []
        for item in data:
            r = await _handle_single_jsonrpc(item)
            if r is not None:
                responses.append(r)
        if responses:
            return JSONResponse(content=responses)
        return Response(status_code=204)

    # --- Single ---
    result = await _handle_single_jsonrpc(data)
    if result is None:
        return Response(status_code=204)
    return JSONResponse(content=result)


async def _handle_single_jsonrpc(data: dict):
    """Route a single JSON-RPC message. Returns dict or None for notifications."""
    method = data.get("method", "")
    msg_id = data.get("id")
    params = data.get("params", {})

    logger.info(f"MCP direct: method={method}  id={msg_id}")

    # ---- Notifications (no id or notification method) ----
    if msg_id is None or method.startswith("notifications/"):
        logger.info(f"MCP direct: notification '{method}' ack")
        return None

    # ---- initialize ----
    if method == "initialize":
        logger.info("MCP direct: -> initialize response")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                },
                "serverInfo": {
                    "name": "Power Interpreter",
                    "version": "1.2.1",
                },
            },
        }

    # ---- ping ----
    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    # ---- tools/list ----
    if method == "tools/list":
        tools = _get_tools_list()
        logger.info(f"MCP direct: -> {len(tools)} tools")
        for t in tools:
            logger.info(f"  tool: {t['name']}")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": tools},
        }

    # ---- tools/call ----
    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        logger.info(f"MCP direct: -> tools/call '{tool_name}' args={json.dumps(tool_args)[:300]}")

        registry = _get_tool_registry()
        if tool_name not in registry:
            logger.error(f"MCP direct: tool '{tool_name}' not found. Available: {list(registry.keys())}")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Tool not found: {tool_name}"},
            }

        try:
            tool = registry[tool_name]
            fn = tool.fn if hasattr(tool, 'fn') else tool

            # ============================================================
            # FIX: Validate required arguments BEFORE calling the function
            # Prevents TypeError crashes when SimTheory sends empty args
            # ============================================================
            validation_error = _validate_tool_args(fn, tool_args, tool_name)
            if validation_error:
                logger.warning(f"MCP direct: {tool_name} argument validation failed: {validation_error}")
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Error: {validation_error}"}],
                        "isError": True,
                    },
                }

            logger.info(f"MCP direct: invoking {tool_name}...")
            result = await fn(**tool_args)
            result_str = str(result)
            logger.info(f"MCP direct: {tool_name} returned {len(result_str)} chars")
            logger.info(f"MCP direct: result preview: {result_str[:300]}")

            # Format as MCP content blocks
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": json.dumps(result)}]
            elif isinstance(result, list):
                content = result
            else:
                content = [{"type": "text", "text": result_str}]

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"content": content, "isError": False},
            }

        except Exception as e:
            logger.error(f"MCP direct: {tool_name} error: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error executing {tool_name}: {e}"}],
                    "isError": True,
                },
            }

    # ---- Unknown method ----
    logger.warning(f"MCP direct: unknown method '{method}'")
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


# =============================================================================
# ROUTE MOUNTING
# =============================================================================

# --- PUBLIC ROUTES (no auth) ---
app.include_router(health.router, tags=["Health"])

# --- PUBLIC DOWNLOAD (no auth) ---
# Files stored in Postgres via execute_code are served here.
# URL format: /dl/{file_id} where file_id is a UUID
app.include_router(
    download_router,
    prefix="/dl",
    tags=["Downloads"],
)

# --- PROTECTED ROUTES (API key required) ---
app.include_router(
    execute.router,
    prefix="/api",
    tags=["Execute"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    jobs.router,
    prefix="/api",
    tags=["Jobs"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    files.router,
    prefix="/api",
    tags=["Files"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    data.router,
    prefix="/api",
    tags=["Data"],
    dependencies=[Depends(verify_api_key)],
)
app.include_router(
    sessions.router,
    prefix="/api",
    tags=["Sessions"],
    dependencies=[Depends(verify_api_key)],
)

# --- MCP SSE TRANSPORT (for standard MCP clients) ---
# Still available at GET /mcp/sse for clients that use SSE properly.
# POST /mcp/sse is intercepted by the direct handler above.
app.mount("/mcp", mcp.sse_app())
