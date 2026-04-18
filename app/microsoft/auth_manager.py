"""Microsoft auth manager (disabled stub).

The Personal MCP does not authenticate against Microsoft identity
providers. This stub preserves the `AuthManager` class symbol so that
legacy imports resolve, but any attribute access on an instance raises
NotImplementedError. That makes accidental use loud and obvious rather
than silently broken.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DISABLED_MESSAGE = (
    "Microsoft auth is disabled in the Personal MCP. "
    "This AuthManager stub exists only to preserve imports."
)


class AuthManager:
    """Disabled Microsoft auth manager.

    Constructing this class is cheap and does not raise. Any attribute
    access beyond construction raises NotImplementedError, so callers
    that accidentally rely on Microsoft auth in the Personal MCP get a
    clear, loud failure instead of a silent mis-execution.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        logger.debug("AuthManager stub instantiated (Microsoft disabled)")

    def __getattr__(self, name: str) -> Any:
        # __getattr__ fires only when normal attribute lookup fails.
        # Since the class defines no methods beyond __init__, every
        # call-site like `auth_manager.get_token()` lands here.
        raise NotImplementedError(_DISABLED_MESSAGE)


# Module-level singleton preserved for legacy import patterns such as
# `from app.microsoft.auth_manager import auth_manager`.
auth_manager: AuthManager = AuthManager()
