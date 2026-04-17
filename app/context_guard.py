"""Context Pressure Guard — Prevents token exhaustion cascades.

When execute_code returns a large stdout (e.g., 3 PDF extractions),
the model's output token budget can be nearly exhausted. The NEXT
tool call then arrives with empty arguments because the model ran
out of tokens mid-JSON-serialization.

This module provides:
  1. Per-tool response caps (tighter than the global 50K limit)
  2. Context pressure warnings injected into large responses
  3. Improved error recovery messages for empty-args failures
  4. Stdout truncation helper for the execute engine

Version: 3.0.0
Ref: Railway logs 2026-03-17T23:18–23:21Z (Calcium Chloride PDF incident)
"""

import logging

logger = logging.getLogger(__name__)

# ── Fix 1: Per-Tool Response Caps ────────────────────────────────
# The global MCP_RESPONSE_MAX_CHARS (50K) is too generous for execute_code.
# PDF extractions (3 files × ~15K chars each) slide right under 50K but
# leave no room for the model to generate its next tool call.
#
# Usage in app/main.py:
#   effective_cap = get_effective_cap(tool_name, MCP_RESPONSE_MAX_CHARS)
#   if original_len > effective_cap:

TOOL_RESPONSE_CAPS = {
    "execute_code": 25_000,    # ~6K tokens — leaves room for reasoning + next call
    "onedrive": 30_000,        # OneDrive listings can be massive (the 321K incident)
    "sharepoint": 30_000,      # Same concern as OneDrive
}


def get_effective_cap(tool_name: str, default_cap: int = 50_000) -> int:
    """Return the response character cap for a given tool.

    Falls back to the global default if no tool-specific cap is set.
    """
    return TOOL_RESPONSE_CAPS.get(tool_name, default_cap)


# ── Fix 3: Context Pressure Warning ─────────────────────────
# Even responses UNDER the cap can consume enough context to cause the
# model's next tool call to truncate. This injects a warning that models
# actually respond to — they'll write shorter follow-up calls.

CONTEXT_PRESSURE_THRESHOLD = 15_000  # ~3.75K tokens


def maybe_add_pressure_warning(tool_name: str, result_str: str) -> str:
    """Prepend a context-pressure warning if the result is large.

    Only applies to execute_code — other tools have structured output
    that models handle differently.
    """
    if tool_name == "execute_code" and len(result_str) > CONTEXT_PRESSURE_THRESHOLD:
        warning = (
            "\u26a0\ufe0f LARGE OUTPUT ({}  chars). To avoid token exhaustion on your next call:\n"
            "  \u2022 Keep your next code block under 20 lines\n"
            "  \u2022 Save intermediate results to files instead of printing\n"
            "  \u2022 Process one file at a time, not multiple\n\n"
        ).format(f"{len(result_str):,}")
        logger.info(
            f"Context pressure warning injected for {tool_name} "
            f"({len(result_str):,} chars > {CONTEXT_PRESSURE_THRESHOLD:,} threshold)"
        )
        return warning + result_str
    return result_str


# ── Fix 4: Improved Empty-Args Error Recovery ───────────────────
# The v2.9.1 error message is good but too generic. This version:
#   - Tells the model NOT to retry the same approach
#   - Gives concrete patterns (save to file, one-file-at-a-time)
#   - References the actual cause (output token exhaustion)

def get_empty_args_recovery_message(tool_name: str, tool_args: dict) -> str | None:
    """Return a recovery-oriented error message for empty tool arguments.

    Returns None if this isn't an empty-args situation, so the caller
    can fall through to the default validation error.
    """
    if tool_name == "execute_code" and len(tool_args) == 0:
        return (
            "ERROR: The 'code' argument was empty \u2014 your previous response likely "
            "consumed most of your output token budget.\n\n"
            "TO RECOVER (do NOT retry the same approach):\n"
            "1. Write a VERY short code snippet (under 10 lines)\n"
            "2. If processing multiple files, do ONE file at a time\n"
            "3. Save results to a file instead of printing:\n"
            "       with open('/tmp/results.txt', 'w') as f: f.write(text)\n"
            "4. Then read specific sections:\n"
            "       with open('/tmp/results.txt') as f: print(f.read()[:5000])\n"
            "5. If this keeps failing, tell the user to start a fresh conversation"
        )
    return None


# ── Fix 2: Stdout Truncation Helper ──────────────────────────
# Applied at the SOURCE (execute engine) before the response even reaches
# the MCP response budget guard. This prevents the chain reaction from
# starting in the first place.
#
# Usage in app/routes/execute.py:
#   from app.context_guard import truncate_stdout
#   stdout = truncate_stdout(stdout)

MAX_STDOUT_CHARS = 20_000  # ~5K tokens


def truncate_stdout(stdout: str) -> str:
    """Truncate execution stdout if it exceeds the safe limit.

    Preserves the beginning of the output (which usually contains
    the most relevant information) and appends a clear notice.
    """
    if not stdout or len(stdout) <= MAX_STDOUT_CHARS:
        return stdout

    original_len = len(stdout)
    truncated = stdout[:MAX_STDOUT_CHARS]

    # Try to cut at a clean line boundary
    last_newline = truncated.rfind('\n')
    if last_newline > MAX_STDOUT_CHARS * 0.8:  # Don't cut too much
        truncated = truncated[:last_newline]

    notice = (
        f"\n\n--- STDOUT TRUNCATED ---\n"
        f"Output was {original_len:,} chars. Showing first {len(truncated):,}.\n"
        f"TIP: Process fewer files per call, or write results to a file "
        f"and read specific sections with:\n"
        f"    with open('/tmp/results.txt') as f: print(f.read()[:5000])"
    )

    logger.warning(
        f"Stdout truncated: {original_len:,} -> {len(truncated):,} chars "
        f"(limit: {MAX_STDOUT_CHARS:,})"
    )

    return truncated + notice
