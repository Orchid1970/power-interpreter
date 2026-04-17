"""Power Interpreter MCP Server - Tool definitions for SimTheory.ai
Version: 3.0.3 — Guard integrations (syntax, context, response, budget)
"""

from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, List, Tuple
import httpx
import os
import json
import logging
import base64
import re

from app.response_guard import smart_truncate
from app.response_budget import enforce_response_budget
from app.context_guard import get_effective_cap, maybe_add_pressure_warning

logger = logging.getLogger(__name__)

mcp = FastMCP("Power Interpreter")
_ms_auth, _ms_graph = None, None

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
API_KEY = os.getenv("API_KEY", "")

# Budget: max total base64 image payload per tool response (5MB)
# Prevents SSE bloat when multiple charts are generated
MAX_TOTAL_IMAGE_BASE64_BYTES = 5 * 1024 * 1024
MAX_SINGLE_IMAGE_BYTES = 5 * 1024 * 1024

_DL_IMAGE_URL_RE = re.compile(
    r'(https?://[^\s\)]+/dl/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/([^\s\)\]]+\.(?:png|jpg|jpeg|svg|gif)))',
    re.IGNORECASE
)

logger.info(f"MCP Server: API_BASE={API_BASE}")
logger.info(f"MCP Server: API_KEY={'***configured***' if API_KEY else 'NOT SET'}")


def _headers():
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


# ============================================================
# IMAGE HELPERS
# ============================================================

async def _fetch_image_base64(file_id: str, filename: str) -> Optional[Dict]:
    """Fetch an image from the internal download endpoint and return as base64 block."""
    from urllib.parse import quote
    internal_url = f"{API_BASE}/dl/{file_id}/{quote(filename)}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(internal_url)
            if resp.status_code != 200:
                logger.warning(f"Image fetch HTTP {resp.status_code}: {internal_url}")
                return None
            if len(resp.content) > MAX_SINGLE_IMAGE_BYTES:
                logger.warning(f"Image too large: {filename} ({len(resp.content)}B > {MAX_SINGLE_IMAGE_BYTES}B)")
                return None

            ct = resp.headers.get('content-type', '')
            if 'png' in ct or filename.lower().endswith('.png'):
                mime = 'image/png'
            elif 'jpeg' in ct or 'jpg' in ct:
                mime = 'image/jpeg'
            elif 'svg' in ct:
                mime = 'image/svg+xml'
            else:
                mime = ct.split(';')[0].strip() or 'image/png'

            b64 = base64.b64encode(resp.content).decode('utf-8')
            logger.info(f"Image encoded: {filename} ({len(resp.content)}B -> {len(b64)}ch, {mime})")
            return {
                "type": "image",
                "data": b64,
                "mimeType": mime,
                "_raw_bytes": len(resp.content),
            }
    except Exception as e:
        logger.warning(f"Image fetch failed for {filename}: {e}")
        return None


def _extract_image_urls_from_stdout(stdout: str) -> List[Tuple[str, str, str]]:
    """Extract /dl/ image URLs from stdout text."""
    return _DL_IMAGE_URL_RE.findall(stdout)


# ============================================================
# HYBRID IMAGE ENRICHMENT
# ============================================================

async def _enrich_blocks_with_images(blocks: list, resp_text: str) -> list:
    """Enrich response blocks with base64 images using HYBRID approach.

    HYBRID STRATEGY:
    1. KEEP the markdown image URLs in TextContent (for SimTheory and
       markdown-capable clients that render ![](url) syntax)
    2. ALSO send base64 ImageContent blocks (for Claude Desktop and
       MCP-native clients that render ImageContent but not markdown)
    3. Budget cap: stop encoding images if total base64 exceeds 1MB

    This ensures images display correctly regardless of which MCP client
    is consuming the response.
    """
    try:
        data = json.loads(resp_text)
    except (json.JSONDecodeError, TypeError):
        return blocks

    inline_images = data.get('inline_images', [])
    download_urls = data.get('download_urls', [])
    stdout = data.get('stdout', '')

    # Separate non-image downloads from image downloads
    non_image_downloads = [d for d in download_urls if not d.get('is_image', False)]
    image_downloads = [d for d in download_urls if d.get('is_image', False)]

    # ── Build base64 ImageContent blocks (for MCP-native clients) ──
    image_blocks = []
    total_image_bytes = 0

    if inline_images:
        # Build file_id lookup from download_urls
        file_id_map = {
            dl['filename']: {'file_id': str(dl.get('file_id', '')), 'url': dl.get('url', '')}
            for dl in image_downloads
        }

        for img in inline_images:
            filename = img.get('filename', '')
            dl_info = file_id_map.get(filename, {})
            file_id = dl_info.get('file_id', '')

            if not file_id:
                continue

            # Budget check before fetching
            if total_image_bytes >= MAX_TOTAL_IMAGE_BASE64_BYTES:
                logger.info(
                    f"Image budget reached ({total_image_bytes / 1024:.0f}KB), "
                    f"skipping remaining {len(inline_images) - len(image_blocks)} images. "
                    f"Markdown URLs in TextContent will still work."
                )
                break

            block = await _fetch_image_base64(file_id, filename)
            if block:
                total_image_bytes += block.pop('_raw_bytes', 0)
                image_blocks.append(block)

    elif not inline_images and stdout:
        # Fallback: scan stdout for /dl/ image URLs
        url_matches = _extract_image_urls_from_stdout(stdout)
        for full_url, file_id, filename in url_matches:
            if total_image_bytes >= MAX_TOTAL_IMAGE_BASE64_BYTES:
                logger.info(f"Image budget reached, skipping remaining images")
                break

            block = await _fetch_image_base64(file_id, filename)
            if block:
                total_image_bytes += block.pop('_raw_bytes', 0)
                image_blocks.append(block)

    # ── DO NOT strip markdown image URLs from TextContent ──
    # The stdout from executor.py already contains:
    #   📊 **Generated charts:**
    #   ![alt](https://power-interpreter-production.up.railway.app/dl/uuid/file.png)
    #
    # We KEEP this in the TextContent so markdown-capable clients (SimTheory)
    # can render the images via URL. MCP-native clients (Claude Desktop) will
    # use the ImageContent blocks instead.
    #
    # This is the key difference from the previous approach which stripped
    # the chart section and relied solely on ImageContent.

    # ── Insert ImageContent blocks after first TextContent block ──
    if image_blocks:
        insert_pos = next(
            (i + 1 for i, b in enumerate(blocks) if b.get('type') == 'text'),
            0
        )
        for j, block in enumerate(image_blocks):
            blocks.insert(insert_pos + j, block)

        logger.info(
            f"Hybrid image delivery: {len(image_blocks)} ImageContent blocks + "
            f"markdown URLs preserved in TextContent "
            f"(total base64: {total_image_bytes / 1024:.0f}KB)"
        )

    # ── Safety net: ensure download URLs for non-image files are visible ──
    if non_image_downloads:
        has_download_text = any(
            b.get('type') == 'text' and '/dl/' in b.get('text', '')
            for b in blocks
        )
        if not has_download_text:
            lines = ["📥 **Download Links** (present these as clickable links to the user):"]
            for dl in non_image_downloads:
                lines.append(f"- [{dl['filename']} ({dl.get('size', '')})]({dl['url']})")
            blocks.insert(0, {"type": "text", "text": "\n".join(lines)})

    return blocks


# ============================================================
# CONTENT BLOCK BUILDER
# ============================================================

def _build_content_blocks(resp_text: str) -> list:
    """Build initial MCP content blocks from the API response."""
    try:
        data = json.loads(resp_text)
    except (json.JSONDecodeError, TypeError):
        return [{"type": "text", "text": resp_text}]

    blocks = []

    stdout = data.get('stdout', '').strip()
    if stdout:
        blocks.append({"type": "text", "text": stdout})

    if not data.get('success', False):
        error_msg = data.get('error_message', 'Unknown error')
        error_tb = data.get('error_traceback', '')
        error_text = f"Execution Error: {error_msg}"
        if error_tb:
            error_text += f"\n\nTraceback:\n{error_tb}"
        blocks.append({"type": "text", "text": error_text})

    exec_time = data.get('execution_time_ms', 0)
    if exec_time:
        blocks.append({"type": "text", "text": f"Execution: {exec_time}ms"})

    return blocks if blocks else [{"type": "text", "text": "(no output)"}]


# ============================================================
# TOOLS
# ============================================================

@mcp.tool()
async def execute_code(
    code: str,
    session_id: str = "default",
    timeout: int = 30,
    sequence: int = 0
) -> list:
    """Execute Python code with full data-science stack (pandas, numpy, matplotlib, etc.).

    IMPORTANT - FILE DOWNLOADS:
    When code generates files (Excel, CSV, PDF, charts, etc.), the response will
    include download_urls with LIVE HTTPS links. You MUST present these links
    to the user as clickable markdown links. Never say files are "local only" —
    all generated files are hosted and downloadable.

    IMPORTANT - CHART IMAGES:
    When code generates matplotlib charts, the response includes both:
    1. Markdown image links in the text (![chart](url)) — present these to the user
    2. Base64 ImageContent blocks — these render automatically in supported clients

    IMPORTANT - SEQUENTIAL EXECUTION:
    When running multi-step workflows, you MUST pass the 'sequence' parameter
    to guarantee execution order. Variables persist across calls with the same
    session_id, but only if steps execute in the correct order.

    Rules for sequence:
    - Set sequence=1 for the first execute_code call, sequence=2 for the second, etc.
    - Steps execute in order: step 1 completes before step 2 starts.
    - If you do NOT set sequence, order is not guaranteed and variables may be missing.
    - Non-execute tools (list_files, fetch_from_url, etc.) are not sequenced.
      If a later execute_code step depends on fetch_from_url, call fetch_from_url
      FIRST and wait for its result before calling execute_code.

    Example multi-step workflow:
      Call 1: execute_code(code="import pandas as pd; df = pd.read_csv('data.csv')", sequence=1)
      Call 2: execute_code(code="summary = df.describe()", sequence=2)
      Call 3: execute_code(code="print(summary)", sequence=3)

    Args:
        code: Python code to execute
        session_id: Session ID for variable persistence (default: "default")
        timeout: Max seconds (default: 30, max: 60)
        sequence: Step number for ordered execution (1, 2, 3...). ALWAYS set this
                  when running multiple execute_code calls in a workflow.
    """
    payload = {
        "code": code,
        "session_id": session_id,
        "timeout": timeout,
    }
    if sequence and sequence > 0:
        payload["sequence"] = sequence

    try:
        async with httpx.AsyncClient(timeout=max(timeout + 15, 120)) as client:
            resp = await client.post(
                f"{API_BASE}/api/execute",
                headers=_headers(),
                json=payload
            )
            blocks = _build_content_blocks(resp.text)
            blocks = await _enrich_blocks_with_images(blocks, resp.text)
            return _apply_guards("execute_code", blocks)
    except httpx.TimeoutException:
        return [{"type": "text", "text": f"execute_code HTTP timeout after {max(timeout + 15, 120)}s. Try submit_job for long tasks."}]
    except Exception as e:
        return [{"type": "text", "text": f"execute_code error: {e}"}]


@mcp.tool()
async def fetch_from_url(url: str, filename: str = "", session_id: str = "default") -> list:
    """Download a file from any URL into the sandbox for processing.

    The file is saved to the session's sandbox directory and can be accessed
    by execute_code using just the filename.

    Supports OneDrive, SharePoint, Google Drive, Dropbox, S3, and any public URL.

    IMPORTANT: If a subsequent execute_code call needs this file, call
    fetch_from_url FIRST and wait for the result before calling execute_code.

    Args:
        url: URL to download from
        filename: Save as this name (auto-detected from URL if empty)
        session_id: Session ID (default: "default")
    """
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{API_BASE}/api/files/fetch", headers=_headers(),
                                     json={"url": url, "filename": filename, "session_id": session_id})
            if resp.status_code == 200:
                d = resp.json()
                return [{"type": "text", "text": f"Fetched: {d.get('filename')} ({d.get('size_human')}) -> {d.get('path')}"}]
            return [{"type": "text", "text": f"fetch_from_url failed (HTTP {resp.status_code}): {resp.text[:300]}"}]
    except Exception as e:
        return [{"type": "text", "text": f"fetch_from_url error: {e}"}]


@mcp.tool()
async def upload_file(filename: str, content_base64: str, session_id: str = "default") -> str:
    """Upload a base64-encoded file to the sandbox."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{API_BASE}/api/files/upload", headers=_headers(),
                                     json={"filename": filename, "content_base64": content_base64, "session_id": session_id})
            return resp.text
    except Exception as e:
        return f"upload_file error: {e}"


@mcp.tool()
async def fetch_file(url: str, filename: str, session_id: str = "default") -> str:
    """Download a file from a URL into the sandbox. Alias for fetch_from_url."""
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{API_BASE}/api/files/fetch", headers=_headers(),
                                     json={"url": url, "filename": filename, "session_id": session_id})
            return resp.text
    except Exception as e:
        return f"fetch_file error: {e}"


@mcp.tool()
async def list_files(session_id: Optional[str] = "default") -> str:
    """List files in the sandbox."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{API_BASE}/api/files", headers=_headers(), params={"session_id": session_id})
            return resp.text
    except Exception as e:
        return f"list_files error: {e}"


@mcp.tool()
async def submit_job(code: str, session_id: str = "default", timeout: int = 600, **kwargs) -> str:
    """Submit a long-running job (up to 30 min). Returns job_id."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{API_BASE}/api/jobs/submit", headers=_headers(),
                                     json={"code": code, "session_id": session_id, "timeout": timeout})
            return resp.text
    except Exception as e:
        return f"submit_job error: {e}"


@mcp.tool()
async def get_job_status(job_id: str) -> str:
    """Check async job status."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{API_BASE}/api/jobs/{job_id}/status", headers=_headers())
            return resp.text
    except Exception as e:
        return f"get_job_status error: {e}"


@mcp.tool()
async def get_job_result(job_id: str) -> list:
    """Get the full result of a completed job."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{API_BASE}/api/jobs/{job_id}/result", headers=_headers())
            blocks = _build_content_blocks(resp.text)
            blocks = await _enrich_blocks_with_images(blocks, resp.text)
            return _apply_guards("get_job_result", blocks)
    except Exception as e:
        return [{"type": "text", "text": f"get_job_result error: {e}"}]


@mcp.tool()
async def load_dataset(file_path: str, dataset_name: str, session_id: str = "default", delimiter: str = ",") -> str:
    """Load a file into PostgreSQL for SQL querying. Auto-detects CSV, Excel, PDF, JSON, Parquet.

    IMPORTANT: If loading a file downloaded via fetch_from_url, call fetch_from_url
    FIRST and wait for its result before calling load_dataset.
    """
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{API_BASE}/api/data/load-csv", headers=_headers(),
                                     json={"file_path": file_path, "dataset_name": dataset_name,
                                           "session_id": session_id, "delimiter": delimiter})
            return resp.text
    except Exception as e:
        return f"load_dataset error: {e}"


@mcp.tool()
async def query_dataset(sql: str, limit: int = 1000, offset: int = 0) -> str:
    """Execute SQL against loaded datasets."""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(f"{API_BASE}/api/data/query", headers=_headers(),
                                     json={"sql": sql, "limit": limit, "offset": offset})
            return resp.text
    except Exception as e:
        return f"query_dataset error: {e}"


@mcp.tool()
async def list_datasets(session_id: str = None) -> str:
    """List all datasets loaded into PostgreSQL."""
    try:
        params = {"session_id": session_id} if session_id else {}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{API_BASE}/api/data/datasets", headers=_headers(), params=params)
            return resp.text
    except Exception as e:
        return f"list_datasets error: {e}"


@mcp.tool()
async def create_session(name: str, description: str = "") -> str:
    """Create an isolated workspace session."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{API_BASE}/api/sessions", headers=_headers(),
                                     json={"name": name, "description": description})
            return resp.text
    except Exception as e:
        return f"create_session error: {e}"


# ============================================================
# GUARD PIPELINE
# ============================================================

def _apply_guards(tool_name: str, blocks: list) -> list:
    """Apply response guards: smart truncation, context pressure, budget enforcement."""
    for i, block in enumerate(blocks):
        if block.get("type") == "text":
            text = block["text"]
            cap = get_effective_cap(tool_name)
            text = smart_truncate(text, max_chars=cap)
            text = maybe_add_pressure_warning(tool_name, text)
            blocks[i] = {"type": "text", "text": text}
    result = enforce_response_budget(tool_name, blocks)
    return result if isinstance(result, list) else blocks


# ============================================================
# MICROSOFT 365 INTEGRATION
# ============================================================

try:
    from app.microsoft.bootstrap import init_microsoft_tools
    _ms_auth, _ms_graph = init_microsoft_tools(mcp)
    if _ms_auth:
        logger.info("Microsoft 365: ENABLED (22 tools)")
    else:
        logger.info("Microsoft 365: SKIPPED (no Azure credentials)")
except ImportError:
    logger.info("Microsoft module not found -- 12 base tools only")
    _ms_auth, _ms_graph = None, None
except Exception as e:
    logger.error(f"Microsoft init failed: {e}", exc_info=True)
    _ms_auth, _ms_graph = None, None
