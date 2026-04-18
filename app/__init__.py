"""Power Interpreter MCP - General Purpose Code Execution Engine

A robust, sandboxed Python execution environment with:
- PostgreSQL storage for large datasets (1.5M+ rows)
- Async job queue for long-running operations
- File management with persistent storage
- Pre-installed data science libraries
- MCP protocol support for SimTheory.ai

Execution Guards:
- context_guard: Post-execution stdout truncation (bounded output)
- syntax_guard: Pre-execution syntax validation
- response_guard: Smart truncation for response fields
- response_budget: Route-layer payload budget enforcement

Resilience & Safety:
- code_resilience: Execution retry and recovery logic
- resilience_patch: Runtime resilience patching
- sandbox_queue: Backpressure queue for sandbox execution
- user_tracker: Multi-user session and token safety

Skills Framework:
- skills_integration: App-level skills registry wiring
- app/skills/: Modular skill definitions and handlers

Microsoft 365 Integration:
- Consolidated graph client with cursor pagination
- Batched downloads and save-to-sandbox support
- Admin auth tooling (auth_admin)
- Safer multi-user token management

Author: Kaffer AI for Timothy Escamilla
Version: 3.0.3
"""

__version__ = "3.0.3"
__author__ = "Kaffer AI"

# ---------------------------------------------------------------------------
# Execution Guards
# ---------------------------------------------------------------------------
from app.context_guard import truncate_stdout          # noqa: F401
from app.syntax_guard import check_syntax              # noqa: F401
from app.response_guard import smart_truncate          # noqa: F401
from app.response_budget import enforce_response_budget  # noqa: F401

# ---------------------------------------------------------------------------
# Resilience & Safety
# ---------------------------------------------------------------------------
from app.engine.code_resilience import *               # noqa: F401,F403
from app.engine.resilience_patch import *              # noqa: F401,F403
from app.engine.sandbox_queue import *                 # noqa: F401,F403
from app.engine.user_tracker import *                  # noqa: F401,F403

# ---------------------------------------------------------------------------
# Skills Framework
# ---------------------------------------------------------------------------
from app.skills_integration import *                   # noqa: F401,F403
