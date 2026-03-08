"""Power Interpreter MCP Server - Tool definitions for SimTheory.ai
Version: 2.9.3
"""

from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict, List, Tuple
import httpx
import os
import json
import logging
import base64
import re

logger = logging.getLogger(__name__)

mcp = FastMCP("Power Interpreter")
_ms_auth, _ms_graph = None, None

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
API_KEY = os.getenv("API_KEY", "")
MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024

_DL_IMAGE_URL_RE = re.compile(
    r'(https?://[^\s\)]+/dl/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/([^\s\)\]]+\.(?:png|jpg|jpeg|svg|gif)))',
    re.IGNORECASE
)
_MARKDOWN_IMAGE_RE = re.compile(r'!\[[^\]]*\]\([^\)]*\.(?:png|jpg|jpeg|svg|gif)\)', re.IGNORECASE)
_GENERATED_CHARTS_RE = re.compile(r'Generated charts?:\s*\n*', re.IGNORECASE)

logger.info(f"MCP Server: API_BASE={API_BASE}")
logger.info(f"MCP Server: API_KEY={'***configured***' if API_KEY else 'NOT SET'}")


def _headers():
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


# ============================================================
# IMAGE HELPERS
# ============================================================

async def _fetch_image_base64(file_id: str, filename: str) -> Optional[Dict]:
    from urllib.parse import quote
    internal_url = f"{API_BASE}/dl/{file_id}/{quote(filename)}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(internal_url)
            if resp.status_code != 200 or len(resp.content) > MAX_IMAGE_BASE64_BYTES:
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
            return {"type": "image", "data": b64, "mimeType": mime}
    except Exception as e:
        logger.warning(f"Image fetch failed for {filename}: {e}")
        return None


def _extract_image_urls_from_stdout(stdout: str) -> List[Tuple[str, str, str]]:
    return _DL_IMAGE_URL_RE.findall(stdout)


def _strip_image_markdown_from_text(text: str) -> str:
    cleaned = _MARKDOWN_IMAGE_RE.sub('', text)
    cleaned = _GENERATED_CHARTS_RE.sub('', cleaned)
    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()


async def _enrich_blocks_with_images(blocks: list, resp_text: str) -> list:
    try:
        data = json.loads(resp_text)
    except (json.JSONDecodeError, TypeError):
        return blocks

    inline_images = data.get('inline_images', [])
    download_urls = data.get('download_urls', [])
    stdout = data.get('stdout', '')

    image_blocks = []
    fallback_blocks = []
    images_found = False

    # Path A: inline_images from JSON
    if inline_images:
        images_found = True
        file_id_map = {dl['filename']: {'file_id': dl.get('file_id', ''), 'url': dl.get('url', '')}
                       for dl in download_urls if dl.get('is_image')}

        for img in inline_images:
            filename = img.get('filename', '')
            dl_info = file_id_map.get(filename, {})
            file_id = dl_info.get('file_id', '')
            public_url = dl_info.get('url', '') or img.get('url', '')

            if file_id:
                block = await _fetch_image_base64(file_id, filename)
                if block:
                    image_blocks.append(block)
                    continue
            if public_url:
                fallback_blocks.append({"type": "text", "text": f"Chart: {img.get('alt_text', filename)}\nURL: {public_url}"})

    # Path B: scan stdout for /dl/ URLs
    if not images_found and stdout:
        url_matches = _extract_image_urls_from_stdout(stdout)
        if url_matches:
            images_found = True
            for full_url, file_id, filename in url_matches:
                block = await _fetch_image_base64(file_id, filename)
                if block:
                    image_blocks.append(block)
                else:
                    fallback_blocks.append({"type": "text", "text": f"Chart: {filename}\nURL: {full_url}"})

    # Strip markdown image syntax from text blocks
    if image_blocks and blocks:
        for i, block in enumerate(blocks):
            if block.get('type') == 'text':
                cleaned = _strip_image_markdown_from_text(block['text'])
                if cleaned != block['text']:
                    blocks[i] = {"type": "text", "text": cleaned} if cleaned else None
                break
        blocks = [b for b in blocks if b is not None]

    # Insert image blocks after first text block
    if image_blocks or fallback_blocks:
        insert_pos = next((i + 1 for i, b in enumerate(blocks) if b.get('type') == 'text'), 0)
        for j, block in enumerate(image_blocks + fallback_blocks):
            blocks.insert(insert_pos + j, block)

    return blocks


# ============================================================
# CONTENT BLOCK BUILDER
# ============================================================

def _build_content_blocks(resp_text: str) -> list:
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
            error_text += f"\n\nTraceback:\n{'...' + error_tb[-500:] if len(error_tb) > 500 else error_tb}"
        blocks.append({"type": "text", "text": error_text})

    for info in data.get('download_urls', []):
        if not info.get('is_image', False) and info.get('url'):
            blocks.append({"type": "text", "text": f"File: {info.get('filename', 'file')} ({info.get('size', '')})\nDownload: {info['url']}"})

    meta_parts = []
    if data.get('execution_time_ms'):
        meta_parts.append(f"Execution: {data['execution_time_ms']}ms")
    ki = data.get('kernel_info', {})
    if ki.get('session_persisted'):
        meta_parts.append(f"Session: {ki.get('variable_count', 0)} vars (call #{ki.get('execution_count', 0)})")
    if meta_parts:
        blocks.append({"type": "text", "text": " | ".join(meta_parts)})

    return blocks or [{"type": "text", "text": "Code executed successfully (no output)."}]


# ============================================================
# TOOLS
# ============================================================

@mcp.tool()
async def execute_code(code: str, session_id: str = "default", timeout: int = 55, sequence: int = 0) -> list:
    """Execute Python in a persistent sandbox. Variables, imports, and files persist. Charts auto-captured.

    Args:
        code: Python code to execute
        session_id: Session ID for kernel persistence (default: "default")
        timeout: Max execution time in seconds (default: 55)
        sequence: Step number for ordered execution (1, 2, 3...).
                  When multiple calls arrive simultaneously, they execute in
                  sequence order. Use 0 or omit to skip ordering.
    """
    try:
        payload = {"code": code, "session_id": session_id, "timeout": timeout}
        if sequence and sequence > 0:
            payload["sequence"] = sequence
        async with httpx.AsyncClient(timeout=timeout + 15) as client:
            resp = await client.post(f"{API_BASE}/api/execute", headers=_headers(), json=payload)
            blocks = _build_content_blocks(resp.text)
            return await _enrich_blocks_with_images(blocks, resp.text)
    except Exception as e:
        return [{"type": "text", "text": f"execute_code error: {e}"}]


@mcp.tool()
async def fetch_from_url(url: str, filename: Optional[str] = None, session_id: str = "default") -> list:
    """Download a file from any URL into the sandbox."""
    if not filename:
        from urllib.parse import urlparse
        filename = urlparse(url).path.split('/')[-1].split('?')[0] or 'downloaded_file'
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
async def submit_job(code: str, session_id: str = "default", timeout: int = 600) -> str:
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
            return await _enrich_blocks_with_images(blocks, resp.text)
    except Exception as e:
        return [{"type": "text", "text": f"get_job_result error: {e}"}]


@mcp.tool()
async def load_dataset(file_path: str, dataset_name: str, session_id: str = "default", delimiter: str = ",") -> str:
    """Load a file into PostgreSQL for SQL querying. Auto-detects CSV, Excel, PDF, JSON, Parquet."""
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
