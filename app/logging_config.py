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

Additionally, ``logging.basicConfig()`` is a no-op if the root logger
already has handlers installed, so it silently fails to override a
``RichHandler`` that FastMCP installs at import time.

Solution
--------
    - DEBUG / INFO / WARNING  -> stdout
    - ERROR / CRITICAL        -> stderr

All pre-existing handlers are cleared first so third-party handlers
(FastMCP, Rich, uvicorn, etc.) cannot interfere with the routing.

Usage
-----
Call ``setup_logging()`` AFTER all third-party imports so that any
handlers they install on import are cleared and replaced:

    from app.config import settings
    from app.mcp_server import mcp      # may install its own handler
    from app.logging_config import setup_logging

    setup_logging(settings.LOG_LEVEL)   # wipes and re-installs handlers
    logger = logging.getLogger(__name__)
"""

import logging
import sys


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


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with severity-based stream routing.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to INFO. Case-insensitive.

    Side effects:
        - Removes all existing handlers from the root logger.
        - Installs a stdout handler for DEBUG/INFO/WARNING.
        - Installs a stderr handler for ERROR/CRITICAL.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear any existing handlers (including those installed by
    # FastMCP's RichHandler, uvicorn, or anything imported before us).
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
