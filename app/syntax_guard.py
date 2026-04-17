"""Power Interpreter - Pre-Execution Syntax Guard (Fix 5)

Catches truncated or malformed code BEFORE it reaches the sandbox,
saving execution time and giving the model actionable recovery guidance.

The most common failure pattern:
  Model runs out of output tokens mid-code-generation
  → sends truncated try: block with no body
  → SyntaxError wastes 200-500ms of sandbox time
  → error message isn't helpful for self-correction

This guard catches these patterns in <1ms and returns a message
that helps the model retry with shorter, working code.

Version: 3.0.1
Ref: Railway logs 2026-03-18T23:53Z (spc3_hydrocooler SyntaxError)
"""

import ast
import logging

logger = logging.getLogger(__name__)


def check_syntax(code: str) -> str | None:
    """Validate code before execution. Returns None if OK, or an error message.

    This runs in <1ms — much cheaper than a 200-500ms sandbox failure.
    """
    if not code or not code.strip():
        return None  # Empty code handled elsewhere

    stripped = code.strip()

    # ── Check 1: Python's own syntax check ────────────────
    try:
        ast.parse(stripped)
        return None  # Valid Python — let it through
    except SyntaxError:
        # Code has a syntax error. Now determine if it's truncation.
        pass

    lines = stripped.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]

    # ── Check 2: Truncated block detection ──────────────
    # If the last non-empty line ends with ':' and there's nothing after,
    # the code was almost certainly truncated mid-generation.
    if non_empty_lines:
        last_line = non_empty_lines[-1].strip()

        if last_line.endswith(':'):
            keyword = last_line.split()[0].rstrip(':') if last_line.split() else ''
            return _truncation_message(
                f"Code appears truncated — ends with '{last_line}' but has no body. "
                f"The code block after '{keyword}:' was cut off."
            )

    # ── Check 3: Unclosed brackets / parens / braces ─────────
    open_count = 0
    in_string = False
    string_char = None
    i = 0
    while i < len(stripped):
        ch = stripped[i]

        # Skip string contents to avoid counting brackets inside strings
        if not in_string:
            if ch in ('"', "'"):
                # Check for triple quotes
                triple = stripped[i:i+3]
                if triple in ('"""', "'''"):
                    # Find closing triple quote
                    end = stripped.find(triple, i + 3)
                    if end == -1:
                        # Unclosed triple-quoted string
                        return _truncation_message(
                            f"Code appears truncated — unclosed triple-quoted string ({triple}). "
                            "The string literal was cut off mid-content."
                        )
                    i = end + 3
                    continue
                else:
                    in_string = True
                    string_char = ch
            elif ch in '([{':
                open_count += 1
            elif ch in ')]}':
                open_count -= 1
        else:
            if ch == string_char and (i == 0 or stripped[i-1] != '\\'):
                in_string = False
                string_char = None

        i += 1

    if in_string:
        return _truncation_message(
            "Code appears truncated — unclosed string literal. "
            "The code was cut off inside a string."
        )

    if open_count > 0:
        return _truncation_message(
            f"Code appears truncated — {open_count} unclosed bracket(s)/parenthesis(es). "
            "The code was likely cut off mid-expression."
        )

    # ── Check 4: Very short code with syntax error ──────────
    # If the code is under 5 meaningful lines and has a syntax error,
    # it's likely a severely truncated snippet.
    if len(non_empty_lines) <= 5:
        return _truncation_message(
            f"Code has a syntax error and is only {len(non_empty_lines)} line(s) long — "
            "likely truncated during generation."
        )

    # ── Fallback: Real syntax error (not truncation) ────────────
    # Return None to let the sandbox handle it — the user may have
    # intentionally written code with a syntax error for debugging,
    # or it could be a legitimate but complex error.
    return None


def _truncation_message(detail: str) -> str:
    """Build a standardized truncation recovery message."""
    return (
        f"CODE TRUNCATION DETECTED: {detail}\n\n"
        "This happens when the code block exceeds the model's output token limit. "
        "To fix this, please:\n"
        "1. Break your code into 2-3 smaller sequential execute_code calls\n"
        "2. Each call should be under 30 lines\n"
        "3. Use variables/files to pass data between calls\n"
        "4. Do NOT retry the same long code block — it will truncate again\n\n"
        "Example pattern:\n"
        "  Call 1: Load/prepare data → save to variable or file\n"
        "  Call 2: Process/analyze → save results\n"
        "  Call 3: Format output/generate charts"
    )
