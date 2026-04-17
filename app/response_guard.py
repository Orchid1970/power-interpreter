"""Smart Response Truncation for MCP tool responses.

Replaces hard character-cut truncation with boundary-aware truncation.
Preserves complete JSON objects, URLs, and line boundaries.

Strategies (tried in order):
  1. JSON array: truncate at complete object boundary
  2. JSON embedded in text: find and truncate the array portion
  3. Line boundary: truncate at last complete line

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

    # Strategy 3: Line-boundary truncation (fallback)
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
