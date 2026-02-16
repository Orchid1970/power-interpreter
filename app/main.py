"""Power Interpreter MCP - Main Application

General-purpose sandboxed Python execution engine.
Designed for SimTheory.ai MCP integration.

Features:
- Execute Python code in sandboxed environment
- Async job queue for long-running operations (no timeouts)
- Large dataset support (1.5M+ rows via PostgreSQL)
- File upload/download management
- Pre-installed data science libraries

Author: Kaffer AI for Timothy Escamilla
Version: 1.0.5
"""

import logging
import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.auth import verify_api_key
from app.routes import execute, jobs, files, data, sessions, health
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
    logger.info("Power Interpreter MCP v1.0.5 starting...")
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
    
    # Start periodic job cleanup only if DB is available
    cleanup_task = None
    if db_ok:
        cleanup_task = asyncio.create_task(_periodic_cleanup())
    
    logger.info("Power Interpreter ready!")
    logger.info(f"  Database: {'connected' if db_ok else 'NOT CONNECTED'}")
    logger.info(f"  Sandbox dir: {settings.SANDBOX_DIR}")
    logger.info(f"  Max execution time: {settings.MAX_EXECUTION_TIME}s")
    logger.info(f"  Max memory: {settings.MAX_MEMORY_MB} MB")
    logger.info(f"  Max concurrent jobs: {settings.MAX_CONCURRENT_JOBS}")
    logger.info(f"  Job timeout: {settings.JOB_TIMEOUT}s")
    logger.info(f"  MCP server: mounted at /mcp (SSE transport)")
    logger.info(f"  Direct JSON-RPC handler: POST /mcp/sse (bypasses SSE for SimTheory)")
    
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
    """Periodically clean up old jobs and temp files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Every hour
            from app.engine.job_manager import job_manager
            count = await job_manager.cleanup_old_jobs()
            if count:
                logger.info(f"Periodic cleanup: removed {count} old jobs")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# Create FastAPI app
app = FastAPI(
    title="Power Interpreter MCP",
    description=(
        "General-purpose sandboxed Python execution engine. "
        "Execute code, manage files, query large datasets, "
        "and run long-running analysis jobs without timeouts."
    ),
    version="1.0.5",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (allow SimTheory.ai)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Railway handles security at network level
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# DIRECT JSON-RPC HANDLER for SimTheory
# =============================================================================
# SimTheory's MCP client doesn't use the SSE transport properly. It sends
# JSON-RPC messages via POST to /mcp/sse without maintaining an SSE stream.
#
# Instead of trying to proxy through the SSE transport (which requires a
# persistent bidirectional connection), we handle JSON-RPC directly:
#
# 1. Parse the incoming JSON-RPC request
# 2. Route to the appropriate handler (initialize, tools/list, tools/call)
# 3. Call the MCP tool functions directly
# 4. Return the JSON-RPC response on the POST response body
#
# This MUST be defined before app.mount("/mcp", ...) so FastAPI matches
# this route before the mounted sub-application.
# =============================================================================

# Build a registry of MCP tools from the mcp server object
def _get_tool_registry():
    """Build a dict of tool_name -> tool_function from the MCP server."""
    tools = {}
    # FastMCP stores tools in _tool_manager
    if hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, '_tools'):
        for name, tool in mcp._tool_manager._tools.items():
            tools[name] = tool
    return tools


def _get_tools_list():
    """Get the list of tools in MCP format for tools/list response."""
    tools = []
    registry = _get_tool_registry()
    for name, tool in registry.items():
        tool_info = {
            "name": name,
            "description": tool.description if hasattr(tool, 'description') else "",
        }
        # Get input schema
        if hasattr(tool, 'parameters') and tool.parameters:
            tool_info["inputSchema"] = tool.parameters
        elif hasattr(tool, 'fn'):
            # Try to build schema from function signature
            import inspect
            sig = inspect.signature(tool.fn)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                prop = {"type": "string"}  # default
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        prop = {"type": "integer"}
                    elif param.annotation == float:
                        prop = {"type": "number"}
                    elif param.annotation == bool:
                        prop = {"type": "boolean"}
                    elif param.annotation == str:
                        prop = {"type": "string"}
                if param.default != inspect.Parameter.empty:
                    prop["default"] = param.default
                else:
                    required.append(param_name)
                # Use parameter description from docstring if available
                properties[param_name] = prop
            
            tool_info["inputSchema"] = {
                "type": "object",
                "properties": properties,
            }
            if required:
                tool_info["inputSchema"]["required"] = required
        
        tools.append(tool_info)
    return tools


@app.post("/mcp/sse")
async def handle_mcp_jsonrpc(request: Request):
    """
    Direct JSON-RPC handler for SimTheory MCP client.
    
    Handles MCP protocol messages without requiring SSE transport:
    - initialize: Return server capabilities
    - tools/list: Return available tools
    - tools/call: Execute a tool and return result
    - notifications/initialized: Acknowledge (no response needed)
    - ping: Respond with pong
    """
    try:
        body = await request.body()
        logger.info(f"MCP JSON-RPC: received {len(body)} bytes")
        logger.info(f"MCP JSON-RPC: body = {body.decode('utf-8', errors='replace')[:500]}")
        
        data = json.loads(body)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"MCP JSON-RPC: failed to parse body: {e}")
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error"},
                "id": None
            },
            status_code=400,
        )
    
    # Handle batch requests
    if isinstance(data, list):
        responses = []
        for item in data:
            resp = await _handle_single_jsonrpc(item)
            if resp is not None:  # Notifications don't get responses
                responses.append(resp)
        if responses:
            return JSONResponse(content=responses)
        return Response(status_code=204)
    
    # Handle single request
    result = await _handle_single_jsonrpc(data)
    if result is None:
        # Notification - no response
        return Response(status_code=204)
    
    return JSONResponse(content=result)


async def _handle_single_jsonrpc(data: dict) -> dict | None:
    """Handle a single JSON-RPC message and return the response dict."""
    method = data.get("method", "")
    msg_id = data.get("id")
    params = data.get("params", {})
    
    logger.info(f"MCP JSON-RPC: method={method}, id={msg_id}")
    
    # --- NOTIFICATIONS (no response expected) ---
    if msg_id is None or method.startswith("notifications/"):
        logger.info(f"MCP JSON-RPC: notification '{method}' acknowledged")
        return None
    
    # --- INITIALIZE ---
    if method == "initialize":
        logger.info("MCP JSON-RPC: handling initialize")
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
                    "version": "1.0.5",
                },
            }
        }
    
    # --- PING ---
    if method == "ping":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {}
        }
    
    # --- TOOLS/LIST ---
    if method == "tools/list":
        logger.info("MCP JSON-RPC: handling tools/list")
        tools = _get_tools_list()
        logger.info(f"MCP JSON-RPC: returning {len(tools)} tools")
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": tools
            }
        }
    
    # --- TOOLS/CALL ---
    if method == "tools/call":
        tool_name = params.get("name", "")
        tool_args = params.get("arguments", {})
        
        logger.info(f"MCP JSON-RPC: calling tool '{tool_name}' with args: {json.dumps(tool_args)[:200]}")
        
        registry = _get_tool_registry()
        
        if tool_name not in registry:
            logger.error(f"MCP JSON-RPC: tool '{tool_name}' not found")
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            }
        
        try:
            tool = registry[tool_name]
            # Call the tool function directly
            fn = tool.fn if hasattr(tool, 'fn') else tool
            result = await fn(**tool_args)
            
            logger.info(f"MCP JSON-RPC: tool '{tool_name}' returned {len(str(result))} chars")
            
            # Format as MCP tool result
            # MCP expects content as an array of content blocks
            if isinstance(result, str):
                content = [{"type": "text", "text": result}]
            elif isinstance(result, dict):
                content = [{"type": "text", "text": json.dumps(result)}]
            elif isinstance(result, list):
                content = result  # Assume already formatted
            else:
                content = [{"type": "text", "text": str(result)}]
            
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": content,
                    "isError": False
                }
            }
            
        except Exception as e:
            logger.error(f"MCP JSON-RPC: tool '{tool_name}' failed: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {str(e)}"}],
                    "isError": True
                }
            }
    
    # --- UNKNOWN METHOD ---
    logger.warning(f"MCP JSON-RPC: unknown method '{method}'")
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}"
        }
    }


# --- PUBLIC ROUTES (no auth) ---
app.include_router(health.router, tags=["Health"])

# --- PROTECTED ROUTES (API key required) ---
app.include_router(
    execute.router, 
    prefix="/api", 
    tags=["Execute"],
    dependencies=[Depends(verify_api_key)]
)
app.include_router(
    jobs.router, 
    prefix="/api", 
    tags=["Jobs"],
    dependencies=[Depends(verify_api_key)]
)
app.include_router(
    files.router, 
    prefix="/api", 
    tags=["Files"],
    dependencies=[Depends(verify_api_key)]
)
app.include_router(
    data.router, 
    prefix="/api", 
    tags=["Data"],
    dependencies=[Depends(verify_api_key)]
)
app.include_router(
    sessions.router, 
    prefix="/api", 
    tags=["Sessions"],
    dependencies=[Depends(verify_api_key)]
)

# --- MCP SERVER (SSE transport - kept for standard MCP clients) ---
# The SSE transport still works for clients that use it properly.
# SimTheory uses the direct JSON-RPC handler above instead.
app.mount("/mcp", mcp.sse_app())
