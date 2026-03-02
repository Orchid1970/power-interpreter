"""DEPRECATED — This file is a redirect to app/microsoft/tools.py

v1.9.4: Consolidated duplicate tool registration files.
The canonical file is now app/microsoft/tools.py which contains:
- 22 tools (was 21)
- Optional user_id on all Microsoft tools (auto-resolves from auth)
- New resolve_share_link tool for SharePoint/OneDrive sharing URLs

bootstrap.py now imports directly from tools.py.
This file exists only for backwards compatibility.
"""

import logging
logger = logging.getLogger(__name__)
logger.warning("mcp_tools.py is DEPRECATED — use app.microsoft.tools instead")

# Re-export for any stale imports
from app.microsoft.tools import register_microsoft_tools

__all__ = ["register_microsoft_tools"]
