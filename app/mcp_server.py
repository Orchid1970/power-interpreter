"""Power Interpreter - MCP Server Definition

Defines the MCP tools that SimTheory.ai can call.
This maps MCP tool calls to the FastAPI endpoints.

MCP Tools (12):
- execute_code: Run Python code (sync, <60s)
- submit_job: Submit long-running job (async)
- get_job_status: Check job progress
- get_job_result: Get completed job output
- upload_file: Upload a file (base64) to sandbox
- fetch_file: Download a file from URL to sandbox
- list_files: List sandbox files
- load_dataset: Load CSV into PostgreSQL
- query_dataset: SQL query against datasets
- list_datasets: List loaded datasets
- create_session: Create workspace session

Version: 1.4.0 - Inline chart rendering support
                 Images returned as markdown ![alt](url) in text response
                 AND as MCP Image content blocks for maximum compatibility
                 Non-image files still use plain text download links
"""

from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict
import httpx
import os
import json
import logging

logger = logging.getLogger(__name__)

# MCP Server
mcp = FastMCP(
    "Power Interpreter",
    description="General-purpose sandboxed Python execution engine with large dataset support"
)

# Internal API base URL
_default_base = "http://127.0.0.1:8080"
API_BASE = os.getenv("API_BASE_URL", _default_base)
API_KEY = os.getenv("API_KEY", "")

logger.info(f"MCP Server: API_BASE={API_BASE}")
logger.info(f"MCP Server: API_KEY={'***configured***' if API_KEY else 'NOT SET'}")


def _headers():
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def _reformat_execution_response(resp_text: str) -> str:
    """Reformat execute_code response for SimTheory consumption.
    
    Handles three types of generated content:
    
    1. INLINE IMAGES (charts, plots):
       - Formatted as markdown: ![alt text](url)
       - Placed prominently so the AI agent renders them in chat
       - Also strips from download_urls to avoid duplicate links
    
    2. DOWNLOAD FILES (xlsx, csv, pdf):
       - Formatted as plain text URLs (not structured JSON)
       - Prevents SimTheory's broken download widget
    
    3. MIXED (charts + files):
       - Images rendered inline, files as download links
    """
    try:
        data = json.loads(resp_text)
    except (json.JSONDecodeError, TypeError):
        return resp_text
    
    # ================================================================
    # Handle inline images (charts, plots)
    # ================================================================
    inline_images = data.get('inline_images', [])
    download_urls = data.get('download_urls', [])
    
    # Build the response text sections
    current_stdout = data.get('stdout', '')
    
    # Remove any previously appended sections from executor
    # (we'll rebuild them here with better formatting)
    for marker in [
        '\n\nGenerated files ready for download:',
        '\n\nGenerated charts:',
    ]:
        idx = current_stdout.find(marker)
        if idx >= 0:
            current_stdout = current_stdout[:idx]
    
    # ================================================================
    # Section 1: Inline images as markdown
    # ================================================================
    if inline_images:
        image_section = "\n\n"
        for img in inline_images:
            alt = img.get('alt_text', 'chart')
            url = img.get('url', '')
            image_section += f"![{alt}]({url})\n\n"
        
        current_stdout = current_stdout + image_section
        
        logger.info(f"Formatted {len(inline_images)} inline images as markdown")
    
    # ================================================================
    # Section 2: Non-image download links as plain text
    # ================================================================
    # Get non-image downloads (exclude files that are already inline images)
    image_filenames = {img.get('filename', '') for img in inline_images}
    non_image_downloads = [
        d for d in download_urls
        if d.get('filename', '') not in image_filenames
        and not d.get('is_image', False)
    ]
    
    if non_image_downloads:
        download_text = "\n" + "=" * 50 + "\n"
        download_text += "DOWNLOAD LINKS (copy URL or click):\n"
        download_text += "=" * 50 + "\n"
        for info in non_image_downloads:
            filename = info.get('filename', 'file')
            url = info.get('url', '')
            size = info.get('size', '')
            download_text += f"\n  {filename} ({size})\n"
            download_text += f"  {url}\n"
        download_text += "\n" + "=" * 50
        
        current_stdout = current_stdout + download_text
    
    # Update stdout
    data['stdout'] = current_stdout
    
    # Clear structured fields to prevent SimTheory widget issues
    data['download_urls'] = []
    data['inline_images'] = []
    
    # Set result hint for the AI agent
    parts = []
    if inline_images:
        parts.append(
            f"{len(inline_images)} chart(s) generated. "
            f"The chart images are in the output above as markdown images. "
            f"Display them to the user — they should render inline."
        )
    if non_image_downloads:
        parts.append(
            f"{len(non_image_downloads)} file(s) generated. "
            f"Download links are in the output above. "
            f"Present the URLs to the user as clickable links."
        )
    
    if parts:
        data['result'] = ' '.join(parts)
    
    return json.dumps(data)


# ============================================================
# CODE EXECUTION TOOLS
# ============================================================

@mcp.tool()
async def execute_code(
    code: str,
    session_id: str = "default",
    timeout: int = 30
) -> str:
    """Execute Python code in a persistent sandboxed environment on Railway.
    
    SESSION PERSISTENCE - IMPORTANT:
    This environment works like a Jupyter notebook. Variables, DataFrames,
    imports, and objects persist across calls WITHIN THE SAME session_id.
    
    ALWAYS use session_id="default" unless you have a specific reason to
    isolate work (e.g., separate projects). DO NOT create new session IDs
    for each call — that defeats persistence and forces a fresh environment.
    
    Example of persistence:
      Call 1: execute_code("import pandas as pd; df = pd.read_csv('data.csv')", session_id="default")
      Call 2: execute_code("print(df.shape)", session_id="default")  # df still exists!
      Call 3: execute_code("summary = df.describe(); print(summary)", session_id="default")  # works!
    
    WRONG (breaks persistence):
      Call 1: execute_code("df = pd.read_csv('data.csv')", session_id="analysis_1")
      Call 2: execute_code("print(df.shape)", session_id="analysis_2")  # ERROR: df doesn't exist!
    
    CHARTS AND VISUALIZATIONS:
    When code calls plt.show(), the chart is automatically captured as a PNG
    image and a URL is returned. Display the chart inline using the markdown
    image URL from the output. The chart will render directly in the chat.
    
    Example:
      execute_code("import matplotlib.pyplot as plt; plt.bar(['A','B','C'], [1,2,3]); plt.show()")
      → Returns markdown image: ![chart](https://...chart_001.png)
      → Display this to the user - it renders inline!
    
    REMOTE EXECUTION:
    This runs on a REMOTE server, NOT locally. You CANNOT reference
    local file paths like /home/ubuntu/... or /tmp/uploads/...
    
    To work with files:
    1. First use upload_file (for small files <10MB, base64 encoded)
    2. Or use fetch_file (for files available at a URL)
    3. Then reference files by their sandbox path: just the filename like 'data.csv'
    
    The sandbox directory is the working directory. Use relative paths only.
    
    Pre-installed libraries: pandas, numpy, matplotlib, plotly, seaborn,
    scipy, scikit-learn, statsmodels, openpyxl, pdfplumber, requests, urllib.
    
    GENERATED FILES: When code creates files (xlsx, csv, png, etc.), download
    URLs will appear in the stdout output. Present these URLs to the user
    as clickable links so they can download the files.
    
    Use for quick operations (<60s). For longer tasks, use submit_job.
    
    Args:
        code: Python code to execute. Do NOT use absolute file paths.
        session_id: Session ID for state persistence. Use "default" to maintain
                    variables across calls. Only change this if you need isolated
                    workspaces for separate projects.
        timeout: Max seconds (max 60 for sync)
    
    Returns:
        Execution result with stdout, result, errors, files created.
        If charts were generated, markdown image URLs appear in stdout.
        If files were generated, download URLs appear in stdout.
    """
    url = f"{API_BASE}/api/execute"
    logger.info(f"execute_code: POST {url}")
    try:
        async with httpx.AsyncClient(timeout=70) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={"code": code, "session_id": session_id, "timeout": timeout}
            )
            logger.info(f"execute_code: response status={resp.status_code}")
            
            # Reformat response: inline images as markdown, downloads as text
            return _reformat_execution_response(resp.text)
    except Exception as e:
        logger.error(f"execute_code: error: {e}", exc_info=True)
        return f"Error calling execute API: {e}"


@mcp.tool()
async def submit_job(
    code: str,
    session_id: str = "default",
    timeout: int = 600
) -> str:
    """Submit a long-running job for async execution on the remote server.
    
    IMPORTANT: This runs on a REMOTE server. You CANNOT reference local file paths.
    Files must be uploaded first using upload_file or fetch_file.
    
    SESSION PERSISTENCE: Use session_id="default" to share state with
    execute_code calls. Variables created in execute_code will be available
    in the job, and vice versa.
    
    CHARTS: Jobs that generate charts will also auto-capture them.
    Use get_job_result to retrieve chart URLs after completion.
    
    Returns immediately with a job_id. Use get_job_status to check progress.
    Use get_job_result to get output when complete.
    
    Use for:
    - Large data processing (1.5M+ rows)
    - Complex analysis (>60 seconds)
    - Report generation
    
    Args:
        code: Python code to execute. Use relative paths for sandbox files.
        session_id: Session ID for state persistence. Use "default" to share
                    state with execute_code calls.
        timeout: Max seconds (default 600 = 10 min)
    
    Returns:
        Job ID and status
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
    
    Status values: pending, running, completed, failed, cancelled, timeout
    
    Args:
        job_id: The job ID from submit_job
    
    Returns:
        Job status with timing info
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
async def get_job_result(job_id: str) -> str:
    """Get the full result of a completed job.
    
    Includes stdout, stderr, result data, files created, execution time.
    If the job generated charts, inline image URLs will be in the output.
    
    Args:
        job_id: The job ID from submit_job
    
    Returns:
        Full job result with inline images and download links
    """
    url = f"{API_BASE}/api/jobs/{job_id}/result"
    logger.info(f"get_job_result: GET {url}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=_headers())
            return _reformat_execution_response(resp.text)
    except Exception as e:
        logger.error(f"get_job_result: error: {e}", exc_info=True)
        return f"Error calling get_job_result API: {e}"


# ============================================================
# FILE MANAGEMENT TOOLS
# ============================================================

@mcp.tool()
async def upload_file(
    filename: str,
    content_base64: str,
    session_id: str = "default"
) -> str:
    """Upload a file to the remote sandbox using base64-encoded content.
    
    USE THIS when the user provides or attaches a file in the conversation.
    This is the PRIMARY way to get files into the sandbox for analysis.
    
    Best for files under 10MB. For larger files, the user should host
    the file at a URL and use fetch_file instead.
    
    After uploading, the file is available in the sandbox by its filename.
    
    Complete workflow example:
    1. upload_file("invoices.csv", "<base64 content>", session_id="default")
    2. execute_code("import pandas as pd; df = pd.read_csv('invoices.csv'); print(df.head())", session_id="default")
    
    Or load into PostgreSQL for SQL queries:
    1. upload_file("data.csv", "<base64 content>", session_id="default")
    2. load_dataset("data.csv", "my_dataset", session_id="default")
    3. query_dataset("SELECT * FROM my_dataset LIMIT 10")
    
    Args:
        filename: Name for the file (e.g., 'invoices.csv', 'report.xlsx')
        content_base64: Base64-encoded file content
        session_id: Session for file isolation (default: 'default')
    
    Returns:
        Confirmation with file path, size, and preview info
    """
    url = f"{API_BASE}/api/files/upload"
    logger.info(f"upload_file: POST {url} filename={filename}")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={
                    "filename": filename,
                    "content_base64": content_base64,
                    "session_id": session_id
                }
            )
            logger.info(f"upload_file: response status={resp.status_code}")
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
    """Download a file from a URL into the remote sandbox.
    
    USE THIS for large files or files hosted online. Supports up to 500MB.
    
    Supports any publicly accessible URL:
    - Google Drive: https://drive.google.com/uc?export=download&id=FILE_ID
    - Dropbox: Change dl=0 to dl=1 in the sharing URL
    - S3 pre-signed URLs
    - Any direct download link
    
    After fetching, the file is available in the sandbox by its filename.
    Use session_id="default" to keep files accessible to execute_code calls.
    
    Example workflow:
    1. fetch_file(url="https://example.com/big_data.csv", filename="data.csv", session_id="default")
    2. execute_code("import pandas as pd; df = pd.read_csv('data.csv'); print(df.shape)", session_id="default")
    
    Args:
        url: Public URL to download from
        filename: What to name the file in the sandbox (e.g., 'invoices.csv')
        session_id: Session for file isolation (default: 'default')
    
    Returns:
        Confirmation with file path, size, type detection, and preview
    """
    api_url = f"{API_BASE}/api/files/fetch"
    logger.info(f"fetch_file: POST {api_url} url={url[:80]}... filename={filename}")
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                api_url,
                headers=_headers(),
                json={
                    "url": url,
                    "filename": filename,
                    "session_id": session_id
                }
            )
            logger.info(f"fetch_file: response status={resp.status_code}")
            return resp.text
    except Exception as e:
        logger.error(f"fetch_file: error: {e}", exc_info=True)
        return f"Error calling fetch_file API: {e}"


@mcp.tool()
async def list_files(session_id: str = None) -> str:
    """List files in the remote sandbox.
    
    Use this to see what files are available before running code.
    
    Args:
        session_id: Optional session filter (use "default" to see main workspace files)
    
    Returns:
        List of files with metadata
    """
    params = {}
    if session_id:
        params["session_id"] = session_id
    
    url = f"{API_BASE}/api/files"
    logger.info(f"list_files: GET {url}")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=_headers(), params=params)
            return resp.text
    except Exception as e:
        logger.error(f"list_files: error: {e}", exc_info=True)
        return f"Error calling list_files API: {e}"


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
    """Load a CSV file from the sandbox into PostgreSQL for fast SQL querying.
    
    IMPORTANT: The file must already be in the sandbox. Use upload_file or
    fetch_file first to get the file into the sandbox.
    
    Handles 1.5M+ rows by loading in chunks.
    After loading, use query_dataset with SQL to analyze.
    
    Args:
        file_path: Filename in sandbox (e.g., 'invoices.csv' - NOT a local path)
        dataset_name: Logical name for SQL queries (e.g., 'vestis_invoices')
        session_id: Session for file isolation (default: 'default')
        delimiter: CSV delimiter (default comma)
    
    Returns:
        Dataset info with row count, columns, preview
    """
    url = f"{API_BASE}/api/data/load-csv"
    logger.info(f"load_dataset: POST {url}")
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                url,
                headers=_headers(),
                json={
                    "file_path": file_path,
                    "dataset_name": dataset_name,
                    "session_id": session_id,
                    "delimiter": delimiter
                }
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
    """Execute a SQL query against loaded datasets in PostgreSQL.
    
    Only SELECT queries allowed. Results paginated.
    
    Use list_datasets first to find table names.
    
    Args:
        sql: SQL SELECT query
        limit: Max rows (default 1000)
        offset: Row offset for pagination
    
    Returns:
        Query results with columns and data
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
    
    Shows dataset names, table names, row counts, and sizes.
    Use the table_name in SQL queries with query_dataset.
    
    Args:
        session_id: Optional session filter
    
    Returns:
        List of datasets
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
    
    Most of the time you should use session_id="default" which is
    automatically available. Only create new sessions when you need
    to isolate work between separate projects.
    
    Use session_id in other tools to scope operations.
    Each session has its own sandbox directory and file space.
    
    Args:
        name: Session name (e.g., 'vestis-audit', 'financial-model')
        description: Optional description
    
    Returns:
        Session ID and details
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
