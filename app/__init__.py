"""Power Interpreter MCP - General Purpose Code Execution Engine

A robust, sandboxed Python execution environment with:
- PostgreSQL storage for large datasets (1.5M+ rows)
- Async job queue for long-running operations
- File management with persistent storage
- Pre-installed data science libraries
- MCP protocol support for SimTheory.ai

Author: Kaffer AI for Timothy Escamilla
Version: see app/version.py (single source of truth)
"""

from app.version import __version__  # single source of truth
__author__ = "Kaffer AI"
