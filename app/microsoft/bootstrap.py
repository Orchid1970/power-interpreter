"""Microsoft integration bootstrap (disabled stub).

The Personal MCP does not integrate with Microsoft 365. This module
preserves the `init_microsoft_tools(mcp)` contract expected by
`app.mcp_server` so the MCP server can import and start cleanly.

`init_microsoft_tools` returns (None, None) to signal that no auth
manager and no Graph client are available. The existing try/except
ImportError around the call site in mcp_server.py means the "SKIPPED"
log path fires naturally without any changes to mcp_server.py.
"""

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def init_microsoft_tools(mcp: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """No-op initializer. Signals Microsoft integration is disabled.

    Parameters
    ----------
    mcp : Any
        The FastMCP server instance. Accepted for signature compatibility
        with the Work MCP implementation and then ignored.

    Returns
    -------
    (None, None)
        Indicates that no auth manager and no Graph client were
        constructed. The caller should treat this as "SKIPPED".
    """
    logger.info("Microsoft integration: DISABLED (Personal MCP stub)")
    return None, None
