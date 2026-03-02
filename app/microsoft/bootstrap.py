"""Bootstrap Microsoft OneDrive + SharePoint tools into the Power Interpreter MCP.

Usage in mcp_server.py:
    from app.microsoft.bootstrap import init_microsoft_tools
    init_microsoft_tools(mcp)

Or call automatically at server startup.
This module is safe to import even if Azure env vars are not set —
it will log a warning and skip registration.

v1.9.2: Removed db_pool parameter. Auth manager now uses SQLAlchemy directly.
v1.9.4: Changed import from mcp_tools -> tools (consolidated duplicate files).
         mcp_tools.py is now a deprecated redirect.
"""

import os
import logging

logger = logging.getLogger(__name__)


def init_microsoft_tools(mcp):
    """
    Initialize and register Microsoft OneDrive + SharePoint MCP tools.

    Args:
        mcp: The FastMCP server instance

    Returns:
        tuple: (auth_manager, graph_client) or (None, None) if not configured
    """
    tenant_id = os.environ.get("AZURE_TENANT_ID", "")
    client_id = os.environ.get("AZURE_CLIENT_ID", "")

    if not tenant_id or not client_id:
        logger.warning(
            "Microsoft integration skipped: AZURE_TENANT_ID and/or "
            "AZURE_CLIENT_ID not set. OneDrive/SharePoint tools will "
            "not be available."
        )
        return None, None

    try:
        from app.microsoft.auth_manager import MSAuthManager
        from app.microsoft.graph_client import GraphClient
        # v1.9.4: Import from tools.py (the canonical file)
        # Previously imported from mcp_tools.py which was a stale duplicate
        from app.microsoft.tools import register_microsoft_tools

        auth_manager = MSAuthManager()
        graph_client = GraphClient(auth_manager)
        register_microsoft_tools(mcp, graph_client, auth_manager)

        logger.info(
            f"Microsoft OneDrive + SharePoint integration enabled "
            f"(tenant: {tenant_id[:8]}...)"
        )
        return auth_manager, graph_client

    except Exception as e:
        logger.error(f"Failed to initialize Microsoft integration: {e}")
        return None, None


async def ensure_microsoft_db(auth_manager):
    """
    Create the ms_tokens table if auth_manager is available.
    Call this after database pool is ready.
    """
    if auth_manager:
        await auth_manager.ensure_db_table()
