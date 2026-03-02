"""Power Interpreter MCP - MCP Server Definition

Defines the MCP tools that SimTheory.ai can call.
This maps MCP tool calls to the FastAPI endpoints.

MCP Tools (33):
- execute_code: Run Python code (sync, <60s)
- submit_job: Submit long-running job (async)
- get_job_status: Check job progress
- get_job_result: Get completed job output
- upload_file: Upload a file (base64) to sandbox
- fetch_file: Download a file from URL to sandbox
- fetch_from_url: Load file from CDN/URL directly into sandbox
- list_files: List sandbox files
- load_dataset: Load data file into PostgreSQL (CSV, Excel, PDF, JSON, Parquet)
- query_dataset: SQL query against datasets
- list_datasets: List loaded datasets
- create_session: Create workspace session
- ms_auth_status: Check Microsoft 365 auth status (OneDrive/SharePoint)
- ms_auth_start: Start Microsoft device login flow
- ms_auth_poll: Complete Microsoft device login after code entry
- resolve_share_link: Resolve SharePoint/OneDrive sharing URL
- onedrive_list_files: List files/folders in OneDrive
- onedrive_search: Search OneDrive by name or content
- onedrive_download_file: Download file from OneDrive
- onedrive_upload_file: Upload file to OneDrive
- onedrive_create_folder: Create folder in OneDrive
- onedrive_delete_item: Delete file/folder from OneDrive
- onedrive_move_item: Move item in OneDrive
- onedrive_copy_item: Copy item in OneDrive
- onedrive_share_item: Create sharing link
- sharepoint_list_sites: List/search SharePoint sites
- sharepoint_get_site: Get SharePoint site details
- sharepoint_list_drives: List document libraries in a site
- sharepoint_list_files: List files in SharePoint library
- sharepoint_download_file: Download from SharePoint
- sharepoint_upload_file: Upload to SharePoint
- sharepoint_search: Search within SharePoint site
- sharepoint_list_lists: List SharePoint lists
- sharepoint_list_items: List items in a SharePoint list

Version: 1.9.2 - fix: token persistence, auth poll, memory guard
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

# MCP Server
mcp = FastMCP("Power Interpreter")

# Microsoft integration (initialized AFTER base tools — see bottom of file)
_ms_auth, _ms_graph = None, None

# Internal API base URL
_default_base = "http://127.0.0.1:8080"
API_BASE = os.getenv("API_BASE_URL", _default_base)
API_KEY = os.getenv("API_KEY", "")

# Max image size to base64 encode (5MB)
MAX_IMAGE_BASE64_BYTES = 5 * 1024 * 1024

# Regex to find /dl/{uuid}/{filename} image URLs in stdout
_DL_IMAGE_URL_RE = re.compile(
    r'(https?://[^\s\)]+/dl/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/([^\s\)\]]+\.(?:png|jpg|jpeg|svg|gif)))',
    re.IGNORECASE
)

# Regex to strip markdown image syntax from stdout
_MARKDOWN_IMAGE_RE = re.compile(
    r'!\[[^\]]*\]\([^\)]*\.(?:png|jpg|jpeg|svg|gif)\)',
    re.IGNORECASE
)
_GENERATED_CHARTS_RE = re.compile(
    r'Generated charts?:\s*\n*',
    re.IGNORECASE
)

logger.info(f"MCP Server: API_BASE={API_BASE}")
logger.info(f"MCP Server: API_KEY={'***configured***' if API_KEY else 'NOT SET'}")


def _headers():
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


# ============================================================
# IMAGE HELPERS (v1.8.0 + v1.8.1)
# ============================================================

async def _fetch_image_base64(file_id: str, filename: str) -> Optional[Dict]:
    """Fetch image bytes from internal /dl/ route, return MCP ImageContent block."""
    from urllib.parse import quote
    encoded_filename = quote(filename)
    internal_url = f"{API_BASE}/dl/{file_id}/{encoded_filename}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(internal_url)

            if resp.status_code != 200:
                logger.warning(f"Image fetch failed: {internal_url} -> HTTP {resp.status_code}")
                return None

            if len(resp.content) > MAX_IMAGE_BASE64_BYTES:
                logger.warning(f"Image too large for base64: {filename} ({len(resp.content)} bytes)")
                return None

            content_type = resp.headers.get('content-type', '')
            if 'png' in content_type or filename.lower().endswith('.png'):
                mime = 'image/png'
            elif 'jpeg' in content_type or 'jpg' in content_type:
                mime = 'image/jpeg'
            elif 'svg' in content_type:
                mime = 'image/svg+xml'
            else:
                mime = content_type.split(';')[0].strip() or 'image/png'

            b64 = base64.b64encode(resp.content).decode('utf-8')

            logger.info(f"Image base64 encoded: {filename} ({len(resp.content)} bytes -> {len(b64)} chars, {mime})")

            return {
                "type": "image",
                "data": b64,
                "mimeType": mime,
            }

    except Exception as e:
        logger.warning(f"Image base64 fetch failed for {filename}: {e}")
        return None


def _extract_image_urls_from_stdout(stdout: str) -> List[Tuple[str, str, str]]:
    """Extract /dl/{uuid}/{filename} image URLs from stdout text."""
    matches = _DL_IMAGE_URL_RE.findall(stdout)
    if matches:
        logger.info(f"Found {len(matches)} image URL(s) in stdout via regex")
        for full_url, file_id, filename in matches:
            logger.info(f"  -> file_id={file_id}, filename={filename}")
    return matches


def _strip_image_markdown_from_text(text: str) -> str:
    """Remove markdown image syntax and 'Generated charts:' prefix from text."""
    cleaned = _MARKDOWN_IMAGE_RE.sub('', text)
    cleaned = _GENERATED_CHARTS_RE.sub('', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


async def _enrich_blocks_with_images(blocks: list, resp_text: str) -> list:
    """Replace text-based image URLs with native MCP ImageContent blocks."""
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

    # PATH A: Use inline_images[] + download_urls[]
    if inline_images:
        logger.info(f"Path A: {len(inline_images)} inline_images in JSON")
        images_found = True

        file_id_map = {}
        for dl in download_urls:
            if dl.get('is_image'):
                file_id_map[dl.get('filename', '')] = {
                    'file_id': dl.get('file_id', ''),
                    'url': dl.get('url', ''),
                }

        for img in inline_images:
            filename = img.get('filename', '')
            alt_text = img.get('alt_text', 'Generated chart')
            dl_info = file_id_map.get(filename, {})
            file_id = dl_info.get('file_id', '')
            public_url = dl_info.get('url', '') or img.get('url', '')

            if file_id:
                block = await _fetch_image_base64(file_id, filename)
                if block:
                    image_blocks.append(block)
                    logger.info(f"Path A: image block created for {filename}")
                    continue

            if public_url:
                fallback_blocks.append({
                    "type": "text",
                    "text": f"Chart: {alt_text}\nImage URL: {public_url}"
                })

    # PATH B: Scan stdout for /dl/ URLs (RELIABLE path)
    if not images_found and stdout:
        url_matches = _extract_image_urls_from_stdout(stdout)

        if url_matches:
            images_found = True
            logger.info(f"Path B: found {len(url_matches)} image URL(s) in stdout")

            for full_url, file_id, filename in url_matches:
                block = await _fetch_image_base64(file_id, filename)
                if block:
                    image_blocks.append(block)
                    logger.info(f"Path B: image block created for {filename}")
                else:
                    fallback_blocks.append({
                        "type": "text",
                        "text": f"Chart: {filename}\nImage URL: {full_url}"
                    })

    # Strip markdown image syntax from stdout text block
    if image_blocks and blocks:
        for i, block in enumerate(blocks):
            if block.get('type') == 'text':
                original_text = block['text']
                cleaned_text = _strip_image_markdown_from_text(original_text)
                if cleaned_text != original_text:
                    if cleaned_text:
                        blocks[i] = {"type": "text", "text": cleaned_text}
                    else:
                        blocks[i] = None
                break
        blocks = [b for b in blocks if b is not None]

    # Insert image blocks
    if image_blocks or fallback_blocks:
        insert_pos = 0
        for i, block in enumerate(blocks):
            if block.get('type') == 'text':
                insert_pos = i + 1
                break

        for j, block in enumerate(image_blocks + fallback_blocks):
            blocks.insert(insert_pos + j, block)

        logger.info(f"Enriched response: {len(image_blocks)} image blocks, {len(fallback_blocks)} fallback blocks")

    return blocks


# ============================================================
# CONTENT BLOCK BUILDER
# ============================================================

def _build_content_blocks(resp_text: str) -> list:
    """Build MCP content blocks from execute_code API response."""
    try:
        data = json.loads(resp_text)
    except (json.JSONDecodeError, TypeError):
        return [{"type": "text", "text": resp_text}]

    blocks = []

    # Block 1: stdout (passed through unmodified)
    stdout = data.get('stdout', '').strip()
    if stdout:
        blocks.append({"type": "text", "text": stdout})

    # Block 2: Error information
    if not data.get('success', False):
        error_msg = data.get('error_message', 'Unknown error')
        error_tb = data.get('error_traceback', '')
        error_text = f"Execution Error: {error_msg}"
        if error_tb:
            if len(error_tb) > 500:
                error_tb = "..." + error_tb[-500:]
            error_text += f"\n\nTraceback:\n{error_tb}"
        blocks.append({"type": "text", "text": error_text})

    # Block 3: Non-image download URLs
    download_urls = data.get('download_urls', [])
    non_image_downloads = [d for d in download_urls if not d.get('is_image', False)]
    for info in non_image_downloads:
        filename = info.get('filename', 'file')
        url = info.get('url', '')
        size = info.get('size', '')
        if url:
            blocks.append({"type": "text", "text": f"File: {filename} ({size})\nDownload URL: {url}"})

    # Block 4: Metadata
    meta_parts = []
    exec_time = data.get('execution_time_ms', 0)
    if exec_time:
        meta_parts.append(f"Execution: {exec_time}ms")
    kernel_info = data.get('kernel_info', {})
    if kernel_info.get('session_persisted'):
        var_count = kernel_info.get('variable_count', 0)
        exec_count = kernel_info.get('execution_count', 0)
        meta_parts.append(f"Session: {var_count} variables persisted (call #{exec_count})")
    if meta_parts:
        blocks.append({"type": "text", "text": " | ".join(meta_parts)})

    if not blocks:
        blocks.append({"type": "text", "text": "Code executed successfully (no output)."})

    logger.info(f"Built {len(blocks)} content blocks for MCP response")
    return blocks


# ============================================================
# CODE EXECUTION TOOLS
# ============================================================

@mcp.tool()
async def execute_code(
    code: str,
    session_id: str = "default",
    timeout: int = 55
) -> list:
    """Execute Python code in a persistent sandbox kernel.

    The kernel persists between calls — variables, imports, and loaded
    files are all available in subsequent execute_code calls.

    WORKFLOW — always follow this pattern:
      1. fetch_from_url(url, filename) — load a file from URL into sandbox
      2. execute_code("import pandas as pd; df = pd.read_excel('filename.xlsx')")
      3. execute_code("print(df.head())")  — variables persist!

    OUTPUT — stdout is returned as-is. Charts (matplotlib/seaborn/plotly)
    are auto-captured and returned as inline images.

    Args:
        code: Python code to execute. Multi-line strings work fine.
        session_id: Session for state persistence (default: 'default').
        timeout: Max seconds before timeout (default 55, max 59).
    """
    url = f"{API_BASE}/api/execute"
    logger.info(f"execute_code: POST {url} session={session_id}")
    try:
        async with httpx.AsyncClient(timeout=timeout + 5) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={"code": code, "session_id": session_id, "timeout": timeout}
            )
            blocks = _build_content_blocks(resp.text)
            blocks = await _enrich_blocks_with_images(blocks, resp.text)
            return blocks

    except Exception as e:
        logger.error(f"execute_code: error: {e}", exc_info=True)
        return [{"type": "text", "text": f"Error calling execute_code API: {e}"}]


# ============================================================
# FILE TOOLS
# ============================================================

@mcp.tool()
async def fetch_from_url(
    url: str,
    filename: Optional[str] = None,
    session_id: str = "default",
) -> list:
    """Fetch a file from any HTTPS URL directly into the sandbox.

    USE THIS to load files before running execute_code on them.

    Supports:
    - Cloudinary CDN URLs (SimTheory file attachments)
    - Google Sheets export URLs
    - S3 pre-signed URLs
    - Any public HTTPS download link

    WORKFLOW:
      1. fetch_from_url(url="https://...", filename="data.xlsx")
      2. execute_code("import pandas as pd; df = pd.read_excel('data.xlsx')")

    Args:
        url: HTTPS URL to download from.
        filename: Name to save as in sandbox. If omitted, derived from URL.
        session_id: Session for file isolation (default: 'default').
    """
    if not filename:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        filename = parsed.path.split('/')[-1].split('?')[0] or 'downloaded_file'

    api_url = f"{API_BASE}/api/files/fetch"
    logger.info(f"fetch_from_url: POST {api_url} url={url[:80]} filename={filename}")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                api_url,
                headers=_headers(),
                json={"url": url, "filename": filename, "session_id": session_id}
            )
            logger.info(f"fetch_from_url: response status={resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                return [{
                    "type": "text",
                    "text": (
                        f"File fetched successfully!\n"
                        f"  Filename : {data.get('filename')}\n"
                        f"  Size     : {data.get('size_human')}\n"
                        f"  Path     : {data.get('path')}\n"
                        f"  Session  : {data.get('session_id')}\n"
                        f"  Preview  : {data.get('preview', 'N/A')}\n\n"
                        f"Now call execute_code to work with this file."
                    )
                }]
            else:
                return [{"type": "text", "text": f"fetch_from_url failed (HTTP {resp.status_code}):\n  {resp.text[:300]}"}]
    except Exception as e:
        logger.error(f"fetch_from_url: error: {e}", exc_info=True)
        return [{"type": "text", "text": f"fetch_from_url error: {e}"}]


@mcp.tool()
async def upload_file(
    filename: str,
    content_base64: str,
    session_id: str = "default"
) -> str:
    """Upload a file to the sandbox via base64 encoding.

    Use for files under 10MB. For larger files or URL-accessible files,
    use fetch_from_url instead.

    Args:
        filename: Name to save as (e.g., 'data.csv')
        content_base64: Base64-encoded file content
        session_id: Session for isolation (default: 'default')
    """
    url = f"{API_BASE}/api/files/upload"
    logger.info(f"upload_file: POST {url} filename={filename}")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={"filename": filename, "content_base64": content_base64,
                      "session_id": session_id}
            )
            return resp.text
    except Exception as e:
        logger.error(f"upload_file: error: {e}", exc_info=True)
        return f"Error calling upload_file API: {e}"


@mcp.tool()
async def fetch_file(
    url: str,
    filename: str,
    session_id: str = "default"
) -> str:
    """Download a file from a URL into the sandbox.

    Alternative to fetch_from_url. Both call the same backend route.

    Args:
        url: URL to download from
        filename: Name to save as in sandbox
        session_id: Session for isolation (default: 'default')
    """
    api_url = f"{API_BASE}/api/files/fetch"
    logger.info(f"fetch_file: POST {api_url}")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                api_url,
                headers=_headers(),
                json={"url": url, "filename": filename, "session_id": session_id}
            )
            return resp.text
    except Exception as e:
        logger.error(f"fetch_file: error: {e}", exc_info=True)
        return f"Error calling fetch_file API: {e}"


@mcp.tool()
async def list_files(session_id: Optional[str] = "default") -> str:
    """List files currently in the sandbox.

    Args:
        session_id: Session to list files for (default: 'default')
    """
    url = f"{API_BASE}/api/files"
    logger.info(f"list_files: GET {url}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                url,
                headers=_headers(),
                params={"session_id": session_id}
            )
            return resp.text
    except Exception as e:
        logger.error(f"list_files: error: {e}", exc_info=True)
        return f"Error calling list_files API: {e}"


# ============================================================
# ASYNC JOB TOOLS
# ============================================================

@mcp.tool()
async def submit_job(
    code: str,
    session_id: str = "default",
    timeout: int = 600
) -> str:
    """Submit a long-running job for async execution.

    Use for jobs that exceed 60 seconds. Poll with get_job_status,
    then retrieve output with get_job_result.

    Args:
        code: Python code to execute.
        session_id: Session for state persistence.
        timeout: Max seconds (default 600 = 10 min)
    """
    url = f"{API_BASE}/api/jobs/submit"
    logger.info(f"submit_job: POST {url}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={"code": code, "session_id": session_id, "timeout": timeout}
            )
            return resp.text
    except Exception as e:
        logger.error(f"submit_job: error: {e}", exc_info=True)
        return f"Error calling submit_job API: {e}"


@mcp.tool()
async def get_job_status(job_id: str) -> str:
    """Check the status of a submitted job.

    Args:
        job_id: The job ID from submit_job
    """
    url = f"{API_BASE}/api/jobs/{job_id}/status"
    logger.info(f"get_job_status: GET {url}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=_headers())
            return resp.text
    except Exception as e:
        logger.error(f"get_job_status: error: {e}", exc_info=True)
        return f"Error calling get_job_status API: {e}"


@mcp.tool()
async def get_job_result(job_id: str) -> list:
    """Get the full result of a completed job.

    Args:
        job_id: The job ID from submit_job
    """
    url = f"{API_BASE}/api/jobs/{job_id}/result"
    logger.info(f"get_job_result: GET {url}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=_headers())
            blocks = _build_content_blocks(resp.text)
            blocks = await _enrich_blocks_with_images(blocks, resp.text)
            return blocks

    except Exception as e:
        logger.error(f"get_job_result: error: {e}", exc_info=True)
        return f"Error calling get_job_result API: {e}"


# ============================================================
# DATASET TOOLS
# ============================================================

@mcp.tool()
async def load_dataset(
    file_path: str,
    dataset_name: str,
    session_id: str = "default",
    delimiter: str = ","
) -> str:
    """Load a data file from the sandbox into PostgreSQL for fast SQL querying.

    Supports multiple file formats — auto-detected from file extension:
      - CSV / TSV (.csv, .tsv, .txt)
      - Excel (.xlsx, .xls, .xlsm, .xlsb)
      - PDF with tables (.pdf)
      - JSON (.json)
      - Parquet (.parquet, .pq)

    WORKFLOW:
      1. fetch_from_url(url, filename="invoices.xlsx")
      2. load_dataset(file_path="invoices.xlsx", dataset_name="invoices")
      3. query_dataset(sql="SELECT * FROM data_xxx WHERE amount > 1000")

    Args:
        file_path: Filename in sandbox (format auto-detected from extension)
        dataset_name: Logical name for SQL queries
        session_id: Session for file isolation (default: 'default')
        delimiter: CSV delimiter (default comma, CSV only)
    """
    url = f"{API_BASE}/api/data/load-csv"
    logger.info(f"load_dataset: POST {url}")
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={"file_path": file_path, "dataset_name": dataset_name,
                      "session_id": session_id, "delimiter": delimiter}
            )
            return resp.text
    except Exception as e:
        logger.error(f"load_dataset: error: {e}", exc_info=True)
        return f"Error calling load_dataset API: {e}"


@mcp.tool()
async def query_dataset(
    sql: str,
    limit: int = 1000,
    offset: int = 0
) -> str:
    """Execute a SQL query against datasets loaded into PostgreSQL.

    Args:
        sql: SQL SELECT query
        limit: Max rows returned (default 1000)
        offset: Row offset for pagination
    """
    url = f"{API_BASE}/api/data/query"
    logger.info(f"query_dataset: POST {url}")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={"sql": sql, "limit": limit, "offset": offset}
            )
            return resp.text
    except Exception as e:
        logger.error(f"query_dataset: error: {e}", exc_info=True)
        return f"Error calling query_dataset API: {e}"


@mcp.tool()
async def list_datasets(session_id: str = None) -> str:
    """List all datasets loaded into PostgreSQL.

    Args:
        session_id: Optional session filter
    """
    params = {}
    if session_id:
        params["session_id"] = session_id
    url = f"{API_BASE}/api/data/datasets"
    logger.info(f"list_datasets: GET {url}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=_headers(), params=params)
            return resp.text
    except Exception as e:
        logger.error(f"list_datasets: error: {e}", exc_info=True)
        return f"Error calling list_datasets API: {e}"


# ============================================================
# SESSION TOOLS
# ============================================================

@mcp.tool()
async def create_session(
    name: str,
    description: str = ""
) -> str:
    """Create a new workspace session for file/data isolation.

    Args:
        name: Session name (e.g., 'vestis-audit', 'financial-model')
        description: Optional description
    """
    url = f"{API_BASE}/api/sessions"
    logger.info(f"create_session: POST {url}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={"name": name, "description": description}
            )
            return resp.text
    except Exception as e:
        logger.error(f"create_session: error: {e}", exc_info=True)
        return f"Error calling create_session API: {e}"


# ============================================================
# MICROSOFT 365 INTEGRATION (v1.9.2)
# Registered AFTER all 12 base tools (v1.9.1 safety fix)
# ============================================================

try:
    from app.microsoft.bootstrap import init_microsoft_tools
    _ms_auth, _ms_graph = init_microsoft_tools(mcp)
    if _ms_auth:
        logger.info("Microsoft OneDrive + SharePoint integration: ENABLED (22 tools)")
    else:
        logger.info("Microsoft OneDrive + SharePoint integration: SKIPPED (no Azure credentials)")
except Exception as e:
    logger.error(f"Microsoft integration failed to initialize: {e}", exc_info=True)
    logger.info("Continuing with 12 base tools — Microsoft tools unavailable")
    _ms_auth, _ms_graph = None, None
