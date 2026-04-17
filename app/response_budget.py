import json
import math
from typing import Any


CHARS_PER_TOKEN = 4
MAX_TOOL_RESPONSE_TOKENS = 50_000
MAX_TOOL_RESPONSE_CHARS = MAX_TOOL_RESPONSE_TOKENS * CHARS_PER_TOKEN


def estimate_tokens(value: str) -> int:
    return math.ceil(len(value) / CHARS_PER_TOKEN)


def serialize_result(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(result, default=str)
    if isinstance(result, list):
        return json.dumps(result, default=str)
    return str(result)


def enforce_response_budget(tool_name: str, result: Any) -> Any:
    serialized = serialize_result(result)
    estimated_tokens = estimate_tokens(serialized)

    if estimated_tokens <= MAX_TOOL_RESPONSE_TOKENS:
        return result

    truncated = serialized[:MAX_TOOL_RESPONSE_CHARS]

    return {
        "status": "truncated",
        "tool": tool_name,
        "warning": (
            f"Response truncated: estimated {estimated_tokens:,} tokens exceeded "
            f"{MAX_TOOL_RESPONSE_TOKENS:,} token budget"
        ),
        "original_size_tokens": estimated_tokens,
        "truncated_to_tokens": MAX_TOOL_RESPONSE_TOKENS,
        "data": truncated,
        "message": "Use pagination or narrower filters to retrieve remaining results.",
    }
