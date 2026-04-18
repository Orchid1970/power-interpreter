"""Microsoft MCP tools (intentionally stubbed).

The Personal MCP does not register any Microsoft tools. This module is
retained as an empty placeholder so that any historical import such as
`from app.microsoft.mcp_tools import register_microsoft_tools` resolves
to a no-op rather than raising ImportError.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def register_microsoft_tools(mcp: Any) -> None:
    """No-op registration. Microsoft tools are disabled in Personal MCP.

    Accepts `mcp` for signature compatibility with the Work MCP and
    then does nothing.
    """
    logger.debug("register_microsoft_tools called but Microsoft is disabled")
    return None
