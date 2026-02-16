"""Power Interpreter - MCP Server Definition

Defines the MCP tools that SimTheory.ai can call.
This maps MCP tool calls to the FastAPI endpoints.

MCP Tools:
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

Version: 1.1.0 - Added upload_file and fetch_file tools
"""

from mcp.server.fastmcp import FastMCP
from typing import Optional, Dict
import httpx
import os
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


# ============================================================
# CODE EXECUTION TOOLS
# ============================================================

@mcp.tool()
async def execute_code(
    code: str,
    session_id: str = "default",
    timeout: int = 30
) -> str:
    """Execute Python code in a sandboxed environment.
    
    Pre-installed libraries: pandas, numpy, matplotlib, plotly, seaborn,
    scipy, scikit-learn, statsmodels, openpyxl, pdfplumber.
    
    Use for quick operations (<60s). For longer tasks, use submit_job.
    
    Args:
        code: Python code to execute
        session_id: Session ID for file isolation
        timeout: Max seconds (max 60 for sync)
    
    Returns:
        Execution result with stdout, result, errors, files created
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
            return resp.text
    except Exception as e:
        logger.error(f"execute_code: error: {e}", exc_info=True)
        return f"Error calling execute API: {e}"


@mcp.tool()
async def submit_job(
    code: str,
    session_id: str = None,
    timeout: int = 600
) -> str:
    """Submit a long-running job for async execution.
    
    Returns immediately with a job_id. Use get_job_status to check progress.
    Use get_job_result to get output when complete.
    
    Use for:
    - Large data processing (1.5M+ rows)
    - Complex analysis (>60 seconds)
    - Report generation
    
    Args:
        code: Python code to execute
        session_id: Session ID for file isolation
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
    
    Args:
        job_id: The job ID from submit_job
    
    Returns:
        Full job result
    """
    url = f"{API_BASE}/api/jobs/{job_id}/result"
    logger.info(f"get_job_result: GET {url}")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=_headers())
            return resp.text
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
    """Upload a file to the sandbox using base64-encoded content.
    
    Best for small-to-medium files (<10MB). The file will be saved
    to the sandbox and can be used by execute_code, load_dataset, etc.
    
    To upload a CSV for analysis:
    1. upload_file("data.csv", base64_content)
    2. load_dataset("data.csv", "my_dataset")
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
    """Download a file from a URL into the sandbox.
    
    Supports any publicly accessible URL: Google Drive sharing links,
    Dropbox links, S3 pre-signed URLs, direct download links, etc.
    
    For Google Drive, use the format:
    https://drive.google.com/uc?export=download&id=FILE_ID
    
    After fetching, use load_dataset to load CSVs into PostgreSQL,
    or execute_code to process with pandas.
    
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
    """List files in the sandbox.
    
    Args:
        session_id: Optional session filter
    
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
    session_id: str = None,
    delimiter: str = ","
) -> str:
    """Load a CSV file into PostgreSQL for fast SQL querying.
    
    Handles 1.5M+ rows by loading in chunks.
    After loading, use query_dataset with SQL to analyze.
    
    Args:
        file_path: Path to CSV in sandbox (just the filename if in session root)
        dataset_name: Logical name (e.g., 'vestis_invoices')
        session_id: Optional session
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
    """Execute a SQL query against loaded datasets.
    
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
    Use the table_name in SQL queries.
    
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
    
    Use session_id in other tools to scope operations.
    
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
