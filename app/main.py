"""Power Interpreter MCP - Main Application
Version: see app/version.py
"""

# =============================================================================
# EARLY STDERR CAPTURE  (must run before ANY other imports)
# =============================================================================
# FastMCP installs Rich's RichHandler on a logger during its own module
# import. RichHandler (via Rich's Console) grabs sys.stderr at __init__
# time and holds a direct reference to it. When app/mcp_server.py's
# module-level code runs logger.info() during
# "from app.mcp_server import mcp", the banner ("MCP Server:
# API_BASE=...", "Microsoft integration: DISABLED", etc.) is written
# directly to that captured stderr reference -- BEFORE setup_logging()
# has a chance to run. Cloud log parsers like Railway then classify
# those records as "error" severity.
#
# We fix this by redirecting sys.stderr to an in-memory buffer BEFORE
# any imports run. Libraries that capture sys.stderr during import
# capture our buffer, not the real stderr. After imports finish and
# setup_logging() has neutralized all third-party handlers, we restore
# real sys.stderr and re-emit the captured text through the correctly-
# routed logger so the banner still appears in the log stream, but at
# the correct INFO severity (stdout).
import sys
import io

_original_stderr = sys.stderr
_stderr_buffer = io.StringIO()
sys.stderr = _stderr_buffer

try:
    import logging
    import asyncio
    import json
    import inspect
    from contextlib import asynccontextmanager
    from datetime import datetime, timezone
    from fastapi import FastAPI, Depends, Request
    from fastapi.responses import Response, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    from app.version import __version__
    from app.config import settings
    from app.auth import verify_api_key
    from app.routes import execute, jobs, files, data, sessions, health
    from app.routes.files import public_router as download_router
    from app.mcp_server import mcp
    from app.logging_config import setup_logging
finally:
    # Restore real stderr BEFORE setup_logging so any subsequent
    # ERROR/CRITICAL records correctly route to the real stderr.
    sys.stderr = _original_stderr

# Configure logging AFTER all imports so we can override any handlers
# installed by third-party libraries (e.g., FastMCP's RichHandler, which
# writes to stderr and causes every log line to be tagged as "error" in
# cloud log parsers like Railway).
setup_logging(settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# Re-emit any text captured from stderr during imports through the now-
# correctly-configured logger. This is typically FastMCP's startup
# banner. Routing via logger.info() sends it to stdout (INFO severity)
# instead of stderr (error severity).
_captured_banner = _stderr_buffer.getvalue()
if _captured_banner.strip():
    for line in _captured_banner.splitlines():
        cleaned = line.rstrip()
        if cleaned:
            logger.info(cleaned)
del _stderr_buffer


def _safe_log_preview(result, max_len: int = 300) -> str:
    """Log-safe preview: replaces base64 image data with size summary."""
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "image":
                parts.append(f'[image:{block.get("mimeType","")},{len(block.get("data",""))}ch]')
            elif isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text", "")
                parts.append(f'[text:{len(t)}ch]{t[:100]}')
            else:
                parts.append(str(block)[:100])
        return " | ".join(parts)
    return str(result)[:max_len]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Second logging setup pass, AFTER uvicorn has had a chance to
    # install its own handlers on named loggers (uvicorn.error, etc.).
    # setup_logging() is idempotent, so this just re-neutralizes any
    # handlers that appeared between module import and server startup.
    # Without this pass, "Application startup complete." and similar
    # uvicorn INFO messages are still classified as error severity.
    setup_logging(settings.LOG_LEVEL)

    logger.info(f"Power Interpreter MCP v{__version__} starting...")
    settings.ensure_directories()

    db_ok = False
    if settings.DATABASE_URL:
        try:
            from app.database import init_database
            await init_database()
            db_ok = True
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
    else:
        logger.warning("No DATABASE_URL configured")

    if db_ok:
        try:
            from app.mcp_server import _ms_auth
            if _ms_auth:
                await _ms_auth.ensure_db_table()
                logger.info("Microsoft token persistence: ENABLED")
            else:
                logger.info("Microsoft token persistence: SKIPPED (no auth manager)")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Microsoft token table setup failed: {e}")

    cleanup_task = asyncio.create_task(_periodic_cleanup()) if db_ok else None

    logger.info("Power Interpreter ready!")
    logger.info(f"  Version: {__version__}")
    logger.info(f"  Database: {'connected' if db_ok else 'NOT CONNECTED'}")
    logger.info(f"  Public URL: {settings.public_base_url or '(auto-detect)'}")
    logger.info(f"  Max memory: {settings.MAX_MEMORY_MB} MB")

    yield

    logger.info("Shutting down...")
    if cleanup_task:
        cleanup_task.cancel()
    if db_ok:
        try:
            from app.database import shutdown_database
            await shutdown_database()
        except Exception:
            pass


async def _periodic_cleanup():
    while True:
        try:
            await asyncio.sleep(3600)
            from app.engine.job_manager import job_manager
            count = await job_manager.cleanup_old_jobs()
            if count:
                logger.info(f"Cleanup: removed {count} old jobs")
            await _cleanup_expired_sandbox_files()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def _cleanup_expired_sandbox_files():
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
            if result.rowcount:
                await session.commit()
                logger.info(f"Cleaned {result.rowcount} expired sandbox files")
    except Exception as e:
        logger.error(f"Sandbox file cleanup failed: {e}")


app = FastAPI(
    title="Power Interpreter MCP",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/charts/{session_id}/{filename}")
async def serve_chart(session_id: str, filename: str):
    """Serve chart images from Postgres. Public, no auth."""
    try:
        from app.database import get_session_factory
        from app.models import SandboxFile
        from sqlalchemy import select

        factory = get_session_factory()
        async with factory() as db_session:
            result = await db_session.execute(
                select(SandboxFile)
                .where(SandboxFile.session_id == session_id)
                .where(SandboxFile.filename == filename)
                .order_by(SandboxFile.created_at.desc())
                .limit(1)
            )
            file_record = result.scalar_one_or_none()

            if not file_record:
                return JSONResponse(status_code=404, content={"error": f"Chart not found: {session_id}/{filename}"})

            ext_map = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                       '.svg': 'image/svg+xml', '.gif': 'image/gif', '.pdf': 'application/pdf'}
            fname_lower = filename.lower()
            ct = next((v for k, v in ext_map.items() if fname_lower.endswith(k)),
                      getattr(file_record, 'mime_type', None) or 'application/octet-stream')

            file_data = file_record.content
            if file_data is None:
                return JSONResponse(status_code=500, content={"error": "File record found but binary data missing"})

            return Response(
                content=file_data,
                media_type=ct,
                headers={
                    "Content-Disposition": f'inline; filename="{filename}"',
                    "Cache-Control": "public, max-age=3600",
                }
            )
    except ImportError:
        return JSONResponse(status_code=503, content={"error": "Database not available"})
    except Exception as e:
        logger.error(f"Chart serve error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})


# =============================================================================
# MCP JSON-RPC HANDLER  (used by /mcp/sse POST — the path SimTheory hits)
# =============================================================================

def _build_tool_schema(tool) -> dict:
    if hasattr(tool, 'parameters') and tool.parameters:
        return tool.parameters

    fn = tool.fn if hasattr(tool, 'fn') else tool
    if not callable(fn):
        return {"type": "object", "properties": {}}

    sig = inspect.signature(fn)
    properties = {}
    required = []
    type_map = {int: "integer", float: "number", bool: "boolean", str: "string"}

    for name, param in sig.parameters.items():
        if name in ('self', 'cls'):
            continue
        prop = {"type": "string"}
        if param.annotation != inspect.Parameter.empty:
            ann = param.annotation
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
            required.append(name)
        properties[name] = prop

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _get_tool_registry() -> dict:
    tools = {}
    if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
        for name, tool in mcp._tool_manager._tools.items():
            tools[name] = tool
    return tools


def _get_tools_list() -> list:
    result = []
    for name, tool in _get_tool_registry().items():
        desc = ""
        if hasattr(tool, 'description'):
            desc = tool.description or ""
        elif hasattr(tool, 'fn') and tool.fn.__doc__:
            desc = tool.fn.__doc__.strip()
        result.append({"name": name, "description": desc, "inputSchema": _build_tool_schema(tool)})
    return result


def _validate_tool_args(fn, tool_args: dict, tool_name: str):
    try:
        sig = inspect.signature(fn)
        missing = [p for p, v in sig.parameters.items()
                   if p not in ('self', 'cls') and v.default is inspect.Parameter.empty and p not in tool_args]
        if missing:
            return f"Missing required parameter(s) for '{tool_name}': {', '.join(missing)}"
    except Exception:
        pass
    return None


@app.post("/mcp/sse")
async def handle_mcp_jsonrpc(request: Request):
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8", errors="replace"))
    except Exception as e:
        return JSONResponse(content={"jsonrpc": "2.0", "error": {"code": -32700, "message": str(e)}, "id": None}, status_code=400)

    if isinstance(data, list):
        responses = [r for item in data if (r := await _handle_single_jsonrpc(item)) is not None]
        return JSONResponse(content=responses) if responses else Response(status_code=204)

    result = await _handle_single_jsonrpc(data)
    return JSONResponse(content=result) if result else Response(status_code=204)


async def _handle_single_jsonrpc(data: dict):
    method = data.get("method", "")
    msg_id = data.get("id")
    params = data.get("params", {})

    if msg_id is None or method.startswith("notifications/"):
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0", "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "Power Interpreter", "version": __version__},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if method == "tools/list":
        tools = _get_tools_list()
        logger.info(f"tools/list: {len(tools)} tools")
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        logger.info(f"tools/call: {tool_name}")

        registry = _get_tool_registry()
        if tool_name not in registry:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}}

        try:
            tool = registry[tool_name]
            fn = tool.fn if hasattr(tool, 'fn') else tool

            validation_error = _validate_tool_args(fn, tool_args, tool_name)
            if validation_error:
                if tool_name == "execute_code" and len(tool_args) == 0:
                    error_text = (
                        "ERROR: No code provided. The 'code' argument was empty. "
                        "Try breaking code into smaller steps or writing to a .py file first."
                    )
                else:
                    error_text = f"Error: {validation_error}"
                return {"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": error_text}], "isError": True}}

            result = await fn(**tool_args)
            logger.info(f"{tool_name} done: {_safe_log_preview(result)}")

            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": json.dumps(result)}]
            elif isinstance(result, list):
                content = result
            else:
                content = [{"type": "text", "text": str(result)}]

            return {"jsonrpc": "2.0", "id": msg_id, "result": {"content": content, "isError": False}}

        except Exception as e:
            logger.error(f"{tool_name} error: {e}", exc_info=True)
            return {"jsonrpc": "2.0", "id": msg_id, "result": {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}}

    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


# =============================================================================
# ROUTES
# =============================================================================

app.include_router(health.router, tags=["Health"])
app.include_router(download_router, prefix="/dl", tags=["Downloads"])
app.include_router(execute.router, prefix="/api", tags=["Execute"], dependencies=[Depends(verify_api_key)])
app.include_router(jobs.router, prefix="/api", tags=["Jobs"], dependencies=[Depends(verify_api_key)])
app.include_router(files.router, prefix="/api", tags=["Files"], dependencies=[Depends(verify_api_key)])
app.include_router(data.router, prefix="/api", tags=["Data"], dependencies=[Depends(verify_api_key)])
app.include_router(sessions.router, prefix="/api", tags=["Sessions"], dependencies=[Depends(verify_api_key)])
app.mount("/mcp", mcp.sse_app())
