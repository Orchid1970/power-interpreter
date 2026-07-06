"""Smart Response Truncation for MCP tool responses.

Replaces hard character-cut truncation with boundary-aware truncation.
Preserves complete JSON objects, URLs, and line boundaries.

Strategies (tried in order):
  1. JSON array: truncate at complete object boundary
  2. JSON embedded in text: find and truncate the array portion
  3. Traceback/error tail: for text containing a Python traceback, keep
     the END of the text (exception type + message) instead of the
     head, since head-first truncation silently discards exactly the
     line the caller needs to see.
  4. Line boundary: truncate at last complete line (head-first fallback
     for all other plain text)

Version: 2.9.9
"""

import json
import re
import os
import logging

logger = logging.getLogger(__name__)

MCP_RESPONSE_MAX_CHARS = int(os.getenv("MCP_RESPONSE_MAX_CHARS", "50000"))


def smart_truncate(text: str, max_chars: int = None) -> str:
    """Truncate text at clean boundaries, preserving complete JSON objects/URLs.

    Returns the original text unchanged if under the limit.

    Args:
        text: The response text to potentially truncate.
        max_chars: Override the env-based limit. Defaults to MCP_RESPONSE_MAX_CHARS.
    """
    if max_chars is None:
        max_chars = MCP_RESPONSE_MAX_CHARS

    if not text or len(text) <= max_chars:
        return text

    original_len = len(text)
    budget = max_chars - 300  # reserve space for truncation notice

    if budget <= 0:
        return text[:max_chars]

    # Strategy 1: Pure JSON array or JSON object with array values
    stripped = text.strip()
    result = _try_truncate_json(stripped, budget)
    if result:
        items_kept, items_total, truncated_text = result
        logger.info(
            f"Smart truncate: JSON array {items_kept}/{items_total} items "
            f"({original_len:,} -> {len(truncated_text):,} chars)"
        )
        notice = (
            f"\n\n⚠️ TRUNCATED: Returned {items_kept} of {items_total} items "
            f"(response budget: {max_chars:,} chars). "
            f"Use pagination, smaller batches, or offset to retrieve "
            f"remaining {items_total - items_kept} items."
        )
        return truncated_text + notice

    # Strategy 2: JSON array embedded in larger text
    result = _try_truncate_embedded_json(stripped, budget)
    if result:
        items_kept, items_total, truncated_text = result
        logger.info(
            f"Smart truncate: embedded JSON {items_kept}/{items_total} items"
        )
        notice = (
            f"\n\n⚠️ TRUNCATED: Returned {items_kept} of {items_total} items "
            f"(response budget: {max_chars:,} chars). "
            f"Use pagination or smaller batches for remaining items."
        )
        return truncated_text + notice

    # Strategy 3: Tail-aware truncation for Python tracebacks / error blocks.
    # Tracebacks (and this app's "Execution Error: ...\n\nTraceback:\n..."
    # wrapper built in mcp_server.py) carry their most useful information —
    # the exception TYPE and MESSAGE — on the LAST lines. Head-first
    # truncation (Strategy 4 below) would keep the earliest stack frames
    # and silently cut off exactly the line the caller needs to see (e.g.
    # a deep-recursion traceback ending in "RecursionError: ..."). Detect
    # this case and keep the tail instead.
    if _looks_like_traceback(text):
        truncated = _truncate_traceback_tail(text, budget)
        total_lines = text.count('\n') + 1
        kept_lines = truncated.count('\n') + 1
        logger.info(
            f"Smart truncate: traceback tail ~{kept_lines}/{total_lines} lines "
            f"({original_len:,} -> {len(truncated):,} chars)"
        )
        notice = (
            f"\n\n⚠️ TRUNCATED: Showing END of traceback/error "
            f"(~{kept_lines} of ~{total_lines} lines) — earliest frames "
            f"omitted, exception type/message preserved "
            f"(response budget: {max_chars:,} chars, "
            f"original: {original_len:,} chars)."
        )
        return truncated + notice

    # Strategy 4: Line-boundary truncation (fallback, head-first)
    truncated = _truncate_at_line_boundary(text, budget)
    total_lines = text.count('\n') + 1
    kept_lines = truncated.count('\n') + 1
    logger.info(
        f"Smart truncate: line boundary ~{kept_lines}/{total_lines} lines "
        f"({original_len:,} -> {len(truncated):,} chars)"
    )
    notice = (
        f"\n\n⚠️ TRUNCATED: Showing ~{kept_lines} of ~{total_lines} lines "
        f"(response budget: {max_chars:,} chars, "
        f"original: {original_len:,} chars). "
        f"Request smaller result sets or use pagination."
    )
    return truncated + notice


# ============================================================
# INTERNAL STRATEGIES
# ============================================================

def _try_truncate_json(text: str, budget: int):
    """Parse text as JSON and truncate arrays at object boundaries.

    Handles:
      - Pure JSON arrays: [{"file": ...}, {"file": ...}, ...]
      - JSON objects with array values: {"files": [...], "count": 697}
    """
    if not (text.startswith('[') or text.startswith('{')):
        return None

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    # Case A: JSON object wrapping an array
    if isinstance(parsed, dict):
        # Common wrapper keys from Graph API, internal APIs, etc.
        _ARRAY_KEYS = (
            'files', 'items', 'results', 'data', 'value',
            'records', 'children', 'entries', 'messages',
        )
        for key in _ARRAY_KEYS:
            if key in parsed and isinstance(parsed[key], list):
                inner = parsed[key]
                if len(inner) <= 1:
                    continue
                # Budget for the array = total budget minus the wrapper overhead
                wrapper_copy = {k: v for k, v in parsed.items() if k != key}
                wrapper_overhead = len(json.dumps(wrapper_copy, ensure_ascii=False)) + 50
                array_budget = budget - wrapper_overhead
                if array_budget <= 0:
                    continue
                kept = _fit_items_in_budget(inner, array_budget)
                if kept is not None and len(kept) < len(inner):
                    result = dict(parsed)
                    result[key] = kept
                    return (
                        len(kept),
                        len(inner),
                        json.dumps(result, indent=2, ensure_ascii=False),
                    )
        return None

    # Case B: Pure JSON array
    if not isinstance(parsed, list) or len(parsed) <= 1:
        return None

    kept = _fit_items_in_budget(parsed, budget)
    if kept is None or len(kept) >= len(parsed):
        return None

    return (
        len(kept),
        len(parsed),
        json.dumps(kept, indent=2, ensure_ascii=False),
    )


def _try_truncate_embedded_json(text: str, budget: int):
    """Find a large JSON array embedded in text and truncate it.

    Looks for [...] blocks that are at least 1000 chars (to skip
    small inline arrays like [1, 2, 3]).
    """
    match = re.search(r'(\[[\s\S]{1000,}\])', text)
    if not match:
        return None

    prefix = text[:match.start()]
    json_str = match.group(1)

    try:
        parsed = json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, list) or len(parsed) <= 1:
        return None

    array_budget = budget - len(prefix) - 100
    if array_budget <= 0:
        return None

    kept = _fit_items_in_budget(parsed, array_budget)
    if kept is None or len(kept) >= len(parsed):
        return None

    truncated_array = json.dumps(kept, indent=2, ensure_ascii=False)
    return len(kept), len(parsed), prefix + truncated_array


def _fit_items_in_budget(items: list, budget: int):
    """Keep as many complete items as fit within the character budget.

    Each item is serialized individually to ensure clean boundaries.
    Never splits a JSON object mid-field or mid-URL.
    """
    kept = []
    running = 2  # account for [ and ]

    for item in items:
        item_str = json.dumps(item, ensure_ascii=False)
        separator = 2 if kept else 0  # ", " between items
        needed = len(item_str) + separator
        if running + needed > budget:
            break
        kept.append(item)
        running += needed

    return kept if kept else None


# Signatures that indicate text contains a Python traceback (or this app's
# "Execution Error: ...\n\nTraceback:\n..." wrapper) where the crucial
# exception TYPE and MESSAGE live at the END of the text, not the start.
_TRACEBACK_SIGNATURES = (
    "Traceback (most recent call last):",
    "\n\nTraceback:\n",
)


def _looks_like_traceback(text: str) -> bool:
    """Detect whether text contains a Python traceback / error block whose
    most useful content (the exception type + message) is on its last
    lines rather than its first lines.
    """
    return any(sig in text for sig in _TRACEBACK_SIGNATURES)


def _truncate_traceback_tail(text: str, budget: int) -> str:
    """Truncate traceback/error text by keeping the TAIL, not the head.

    Python tracebacks -- and this app's "Execution Error: ...\\n\\nTraceback:
    ..." wrapper built in mcp_server.py -- carry their most useful
    information (the exception TYPE and MESSAGE, e.g. "ValueError: invalid
    literal...") on the LAST lines. A plain head-first truncation would
    keep the earliest stack frames and silently cut off exactly the line
    the caller needs to see.

    Strategy:
      1. Keep a short "head anchor" (first line only) when it's small
         relative to the budget -- this is usually the app's
         "Execution Error: <message>" summary line, and is cheap, useful
         context for *what* failed.
      2. Fill the rest of the budget from the END of the text, walking
         backwards a whole line at a time so a line is never cut mid-way.
      3. If even the single last line doesn't fit the budget (e.g. one
         extremely long line), hard-cut the last `budget` characters as a
         last resort so the tail still survives.
    """
    if budget >= len(text):
        return text
    if budget <= 0:
        return ""

    lines = text.split('\n')

    # Reserve a small slice of the budget for a head anchor (first line)
    # so the reader still sees *what* failed, not just the raw traceback.
    head_anchor = ""
    tail_budget = budget
    if lines and len(lines[0]) <= max(budget * 0.2, 40):
        head_anchor = lines[0]
        tail_budget = budget - len(head_anchor) - len("\n...\n")
        if tail_budget <= 0:
            head_anchor = ""
            tail_budget = budget

    tail_lines = []
    running = 0
    for line in reversed(lines):
        separator = 1 if tail_lines else 0  # '\n' joining this line to kept tail
        needed = len(line) + separator
        if running + needed > tail_budget:
            break
        tail_lines.insert(0, line)
        running += needed

    if not tail_lines:
        # Not even the last line fits — hard character cut from the end
        # so we still surface *something* of the exception message.
        return text[-budget:]

    tail_text = '\n'.join(tail_lines)

    if head_anchor and len(tail_lines) < len(lines):
        return f"{head_anchor}\n...\n{tail_text}"
    return tail_text


def _truncate_at_line_boundary(text: str, budget: int) -> str:
    """Truncate at the last complete line within budget.

    Falls back to hard cut if no reasonable line boundary exists.
    """
    if budget >= len(text):
        return text

    cut = text[:budget]
    last_newline = cut.rfind('\n')

    # Only use line boundary if it keeps at least half the budget
    if last_newline > budget * 0.5:
        return cut[:last_newline]

    return cut
