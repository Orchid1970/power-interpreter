"""Power Interpreter - Logging Configuration

Routes log records by severity level so cloud log parsers (Railway, GCP
Cloud Logging, AWS CloudWatch, etc.) classify them correctly.

Problem this solves
-------------------
Python's ``logging.basicConfig()`` defaults to stderr for ALL levels.
Most cloud log aggregators tag anything written to stderr as "error"
severity, regardless of the log level embedded in the message. This
caused every INFO and WARNING record in Railway to be rendered red in
the error dashboard, drowning real errors in noise.

Third-party libraries (uvicorn, FastMCP, Rich) attach handlers to
NAMED loggers (``uvicorn.error``, ``mcp``, ``fastmcp``), not the root
logger. Clearing root alone is insufficient because named loggers
bypass root entirely when they have their own handlers installed.
This is the fix that v3.0.4 missed.

Solution
--------
    - DEBUG / INFO / WARNING  -> stdout
    - ERROR / CRITICAL        -> stderr

Handler cleanup happens at three tiers:
    1. Root logger -- clear existing handlers and install stdout/stderr
       handlers with severity routing.
    2. All existing named loggers -- clear their handlers and force
       ``propagate=True`` so their records flow up to root.
    3. Known third-party logger names -- same treatment even if they
       haven't been created yet, so libraries that create their loggers
       lazily find them in the correct state on first use.

Usage
-----
Call ``setup_logging()`` AFTER all third-party imports so that any
handlers they install on import are cleared. For maximum robustness,
also call it inside the FastAPI lifespan startup hook so any handlers
installed by uvicorn AFTER module import are also neutralized:

    from app.config import settings
    from app.mcp_server import mcp          # may install its own handler
    from app.logging_config import setup_logging

    setup_logging(settings.LOG_LEVEL)       # first pass, at import
    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging(settings.LOG_LEVEL)   # second pass, after boot
        ...
"""

import logging
import sys


# Known third-party loggers that install their own handlers at import
# time (or first use) and bypass root-logger routing. These are cleared
# on every setup_logging() call. Pre-creating them ensures they exist
# with propagate=True and no handlers when libraries later try to use
# them.
THIRD_PARTY_LOGGER_NAMES = (
    # uvicorn -- "uvicorn.error" is misleadingly named; it handles all
    # non-access uvicorn logs, including INFO-level startup messages
    # such as "Application startup complete." and "Uvicorn running on".
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
    "uvicorn.asgi",
    # FastMCP / MCP / Rich -- FastMCP installs a RichHandler on the
    # "mcp" (and/or "fastmcp") logger at import time, which writes to
    # stderr and produces the "[04/18/26] INFO  MCP Server:" format
    # seen in the Railway logs.
    "mcp",
    "mcp.server",
    "fastmcp",
    "rich",
    # FastAPI / Starlette (defensive; usually propagate by default but
    # included for completeness in case future versions add handlers).
    "fastapi",
    "starlette",
    "starlette.routing",
)


class MaxLevelFilter(logging.Filter):
    """Allow only records strictly below the given level.

    Used on the stdout handler so that ERROR/CRITICAL records are not
    duplicated on both stdout and stderr.
    """

    def __init__(self, max_level: int):
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.max_level


def _neutralize_logger(name: str) -> None:
    """Clear handlers on a named logger and force it to propagate.

    After this call, the named logger has no handlers of its own, so
    records flow up to the root logger (where our stdout/stderr routing
    is installed).

    ``logging.getLogger(name)`` materializes the logger if it does not
    already exist, which means libraries that create their logger
    lazily will find it in the correct state on first use.
    """
    logger = logging.getLogger(name)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.propagate = True


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with severity-based stream routing.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to INFO. Case-insensitive.

    Side effects:
        - Removes all existing handlers from the root logger.
        - Installs a stdout handler for DEBUG/INFO/WARNING.
        - Installs a stderr handler for ERROR/CRITICAL.
        - Clears handlers from ALL existing named loggers and sets
          ``propagate=True`` so their records flow up to root.
        - Same treatment applied to a known list of third-party logger
          names (``THIRD_PARTY_LOGGER_NAMES``) whether they exist yet
          or not.

    Idempotent. Safe to call multiple times (e.g. once at module import
    and again inside the FastAPI lifespan startup) to catch handlers
    that were installed between calls.
    """
    normalized_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(normalized_level)

    # --- Tier 1: clear root handlers and install our own ---
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # DEBUG / INFO / WARNING -> stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(MaxLevelFilter(logging.ERROR))
    stdout_handler.setFormatter(formatter)
    root_logger.addHandler(stdout_handler)

    # ERROR / CRITICAL -> stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)

    # --- Tier 2: neutralize every existing named logger ---
    # Catches any library that installed a handler before setup_logging
    # was called (FastMCP RichHandler, uvicorn, etc.). Snapshot the
    # names first because _neutralize_logger may mutate loggerDict.
    existing_names = list(logging.Logger.manager.loggerDict.keys())
    for name in existing_names:
        _neutralize_logger(name)

    # --- Tier 3: pre-emptively neutralize known third-party loggers ---
    # Ensures libraries that create their logger lazily will find it
    # with propagate=True and no handlers when they eventually log.
    for name in THIRD_PARTY_LOGGER_NAMES:
        _neutralize_logger(name)
