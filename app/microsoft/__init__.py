"""Microsoft integration package (disabled in Personal MCP).

The Personal MCP is single-user and does not carry the full Microsoft
365 / Graph integration that lives in the Work MCP. This package is
retained only so that historical imports continue to resolve without
raising ImportError.

The only external contract preserved is::

    from app.microsoft.bootstrap import init_microsoft_tools

Everything else in this package is a stub.
"""

from app.microsoft.bootstrap import init_microsoft_tools

__all__ = ["init_microsoft_tools"]
