"""Tests for app.response_guard.smart_truncate.

Focused on the tail-aware traceback truncation path added to fix:
head-first truncation was silently discarding the exception type/message
(always the LAST lines of a Python traceback) for long error outputs.

Run with:
    python -m pytest tests/test_response_guard.py -v
or, if pytest isn't installed in this environment:
    python tests/test_response_guard.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.response_guard import smart_truncate


def _long_traceback(n_frames: int = 400, exc_line: str = "RecursionError: maximum recursion depth exceeded") -> str:
    frames = "\n".join(
        f'  File "sandbox.py", line {i}, in frame_{i}\n    do_something_{i}()'
        for i in range(1, n_frames)
    )
    tb = f"Traceback (most recent call last):\n{frames}\n{exc_line}"
    return f"Execution Error: {exc_line}\n\nTraceback:\n{tb}"


def test_traceback_tail_preserves_exception_message():
    error_text = _long_traceback()
    out = smart_truncate(error_text, max_chars=2000)
    assert "RecursionError: maximum recursion depth exceeded" in out
    assert "Showing END of traceback" in out


def test_traceback_tail_drops_earliest_frames_not_latest():
    error_text = _long_traceback()
    out = smart_truncate(error_text, max_chars=2000)
    # Earliest frame should be gone; a late frame (near the exception) should survive.
    assert "frame_1)" not in out
    assert "frame_399" in out or "frame_398" in out or "RecursionError" in out


def test_plain_text_truncation_is_unaffected_head_first():
    plain = "\n".join(f"plain text line number {i} with padding" for i in range(2000))
    out = smart_truncate(plain, max_chars=2000)
    assert "Showing END of traceback" not in out
    assert "Showing ~" in out and "lines" in out
    # Head-first: line 0 should be present, not the tail lines.
    assert "plain text line number 0 " in out


def test_json_array_truncation_still_works():
    arr = json.dumps([{"file": f"f{i}.txt", "size": i} for i in range(500)])
    out = smart_truncate(arr, max_chars=2000)
    assert "Returned" in out and "items" in out
    assert "Showing END of traceback" not in out


def test_short_text_under_budget_is_unchanged():
    short = "short text"
    assert smart_truncate(short, max_chars=2000) == short


def test_short_traceback_under_budget_is_unchanged():
    short_tb = (
        "Execution Error: boom\n\nTraceback:\n"
        'Traceback (most recent call last):\n  File "x.py", line 1\nValueError: boom'
    )
    assert smart_truncate(short_tb, max_chars=2000) == short_tb


def test_traceback_with_tiny_budget_still_returns_tail_chars():
    huge_last_line = "ValueError: " + ("x" * 5000)
    tb = f'Traceback (most recent call last):\n  File "a.py", line 1\n{huge_last_line}'
    out = smart_truncate(tb, max_chars=500)
    assert len(out) > 0
    assert "x" * 10 in out


def test_app_wrapper_signature_detected_without_stdlib_traceback_line():
    # mcp_server.py's own "Execution Error: ...\n\nTraceback:\n..." wrapper
    # should be detected even in the rare case the inner traceback text
    # itself doesn't start with "Traceback (most recent call last):".
    wrapper_only = "Execution Error: custom failure\n\nTraceback:\n" + ("z" * 3000) + "\nEND_MARKER_XYZ"
    out = smart_truncate(wrapper_only, max_chars=1000)
    assert "END_MARKER_XYZ" in out


# ── v3.1.2: source-level caps in executor.py ────────────────────────

def test_cap_error_message_preserves_head_and_tail():
    from app.engine.executor import _cap_error_message
    msg = "MARKER_HEAD_" + "x" * 99000 + "_MARKER_TAIL"
    out = _cap_error_message(msg)
    assert len(out) < 5000
    assert "MARKER_HEAD_" in out
    assert "_MARKER_TAIL" in out
    assert "chars omitted" in out


def test_cap_error_message_short_unchanged():
    from app.engine.executor import _cap_error_message
    assert _cap_error_message("boom") == "boom"


def test_cap_traceback_keeps_tail():
    from app.engine.executor import _cap_traceback
    tb = "Traceback (most recent call last):\n" + "\n".join(f"  frame {i}" for i in range(5000)) + "\nValueError: THE_ANSWER"
    out = _cap_traceback(tb)
    assert len(out) <= 21000
    assert "ValueError: THE_ANSWER" in out
    assert "Traceback (most recent call last):" in out


def test_app_version_single_source():
    import app
    from app.version import __version__ as v
    assert app.__version__ == v


def _run_as_script():
    """Fallback runner for environments without pytest installed."""
    tests = [
        test_traceback_tail_preserves_exception_message,
        test_traceback_tail_drops_earliest_frames_not_latest,
        test_plain_text_truncation_is_unaffected_head_first,
        test_json_array_truncation_still_works,
        test_short_text_under_budget_is_unchanged,
        test_short_traceback_under_budget_is_unchanged,
        test_traceback_with_tiny_budget_still_returns_tail_chars,
        test_app_wrapper_signature_detected_without_stdlib_traceback_line,
        test_cap_error_message_preserves_head_and_tail,
        test_cap_error_message_short_unchanged,
        test_cap_traceback_keeps_tail,
        test_app_version_single_source,
    ]
    failures = []
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failures.append(t.__name__)
    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run_as_script()
